import warnings

warnings.filterwarnings("ignore")

from copy import deepcopy

import torch
from torch import nn, multiprocessing
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

# Local noisy layers (optional)
from .noisy import NoisyLazyLinear


def get_device():
    """Determine the appropriate device for training"""
    is_fork = multiprocessing.get_start_method() == "fork"
    if torch.cuda.is_available() and not is_fork:
        return torch.device(0)
    return torch.device("cpu")


class SACPolicy:
    """Soft Actor-Critic policy + twin critics built from nn modules.

    Uses a single shared CNN encoder for actor and both critics.
    Optionally fuses a telemetry vector with CNN image features.

    Attributes:
        actor: ProbabilisticActor
        q1: CriticNet (Q1)
        q2: CriticNet (Q2)
        q1_target: CriticNet (Q1 target, frozen)
        q2_target: CriticNet (Q2 target, frozen)
        shared_cnn: shared CNN encoder (actor + critics)
        target_cnn: polyak-updated copy of shared_cnn (used by targets)
    """

    def __init__(
        self,
        env: GymEnv,
        num_cells: int = 256,
        device=None,
        use_noisy: bool = False,
        noise_sigma: float = 0.5,
        actor_dropout: float = 0.0,
        vae_checkpoint_path: str = None,
        obs_dim: int = 0,
    ):
        if device is None:
            device = get_device()
        self.device = device
        self.use_noisy = use_noisy
        self.noise_sigma = noise_sigma
        self.actor_dropout = float(actor_dropout)
        self.obs_dim = obs_dim

        action_dim = int(env.action_spec.shape[-1])
        in_channels = 3

        # ── Shared CNN encoder ────────────────────────────────────────────
        if vae_checkpoint_path:
            from .vae import load_vae_encoder

            print(f"Loading VAE encoder from {vae_checkpoint_path}...")
            shared_cnn, cnn_output_size = load_vae_encoder(
                vae_checkpoint_path, device, in_channels, trainable=True, verbose=True
            )
        else:
            cnn_output_size = 3136
            shared_cnn = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0, device=device),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, device=device),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=device),
                nn.ReLU(),
                nn.Flatten(start_dim=1),
            )
            print(f"Using default CNN encoder, output size: {cnn_output_size}")

        target_cnn = deepcopy(shared_cnn)
        for p in target_cnn.parameters():
            p.requires_grad = False

        self.shared_cnn = shared_cnn
        self.target_cnn = target_cnn

        # ── Helpers ───────────────────────────────────────────────────────
        class BoundedNormalParams(nn.Module):
            def __init__(self, min_scale, max_scale):
                super().__init__()
                self.register_buffer("min_scale", min_scale)
                self.register_buffer("max_scale", max_scale)
                self.register_buffer("scale_range", max_scale - min_scale)

            def forward(self, x):
                loc, scale_raw = x.chunk(2, dim=-1)
                scale = self.min_scale + torch.sigmoid(scale_raw) * self.scale_range
                return {"loc": loc, "scale": scale}

        def _make_linear(in_f: int, out_f: int):
            if self.use_noisy:
                return NoisyLazyLinear(out_f, sigma=self.noise_sigma, device=device)
            return nn.Linear(in_f, out_f, device=device)

        min_scale = torch.tensor([0.1, 0.1, 0.1], device=device)
        max_scale = torch.tensor([1.0, 1.0, 1.0], device=device)

        # ── Actor ─────────────────────────────────────────────────────────
        # Fuses CNN features with optional telemetry vector
        fusion_input_size = cnn_output_size + obs_dim

        class ActorNet(nn.Module):
            def __init__(
                self,
                cnn,
                fusion_size,
                num_cells,
                action_dim,
                dropout,
                use_noisy,
                noise_sigma,
                device,
            ):
                super().__init__()
                self.cnn = cnn
                self.obs_dim = obs_dim

                def make_lin(i, o):
                    if use_noisy:
                        return NoisyLazyLinear(o, sigma=noise_sigma, device=device)
                    return nn.Linear(i, o, device=device)

                self.mlp = nn.Sequential(
                    make_lin(fusion_size, num_cells),
                    nn.Tanh(),
                    nn.Dropout(p=dropout),
                    make_lin(num_cells, num_cells),
                    nn.Tanh(),
                    nn.Dropout(p=dropout),
                    make_lin(num_cells, 2 * action_dim),
                    BoundedNormalParams(min_scale=min_scale, max_scale=max_scale),
                )

            def forward(self, pixels, vector=None):
                img_feat = self.cnn(pixels)
                if vector is not None and self.obs_dim > 0:
                    x = torch.cat([img_feat, vector], dim=-1)
                else:
                    x = img_feat
                return self.mlp(x)

        actor_net = ActorNet(
            shared_cnn,
            fusion_input_size,
            num_cells,
            action_dim,
            self.actor_dropout,
            self.use_noisy,
            self.noise_sigma,
            device,
        )

        if obs_dim > 0:
            policy_module = TensorDictModule(
                actor_net,
                in_keys=["pixels", "vector"],
                out_keys=["loc", "scale"],
            )
        else:
            policy_module = TensorDictModule(
                actor_net,
                in_keys=["pixels"],
                out_keys=["loc", "scale"],
            )

        low = [-1.0, 0.0, 0.0]
        high = [1.0, 1.0, 1.0]

        try:
            low_t = torch.as_tensor(low, dtype=torch.float32)
            high_t = torch.as_tensor(high, dtype=torch.float32)
            if not torch.all(high_t > low_t):
                print(
                    f"Warning: invalid action bounds detected (low={low_t}, high={high_t}). "
                    "Defaulting to [-1, 1]."
                )
                low_t = -torch.ones_like(low_t)
                high_t = torch.ones_like(high_t)
            dist_kwargs = {"low": low_t, "high": high_t}
        except Exception as e:
            print(f"Warning: could not validate action bounds ({e}); using raw spec values.")
            dist_kwargs = {"low": low, "high": high}

        self.actor = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs=dist_kwargs,
            return_log_prob=True,
        )

        if self.use_noisy:
            noisy_count = sum(1 for m in self.actor.modules() if hasattr(m, "sample_noise"))
            print(f"Using noisy actor: found {noisy_count} noisy layer(s)")

        # ── Critics ───────────────────────────────────────────────────────
        # Both critics share the same CNN encoder as the actor.
        # Targets use a frozen deepcopy updated via polyak in the trainer.
        critic_input_size = cnn_output_size + action_dim + obs_dim

        class CriticNet(nn.Module):
            def __init__(self, encoder, hidden, device, obs_dim):
                super().__init__()
                self.cnn = encoder
                self.obs_dim = obs_dim
                self.fc = nn.Sequential(
                    nn.Linear(critic_input_size, hidden, device=device),
                    nn.Tanh(),
                    nn.Linear(hidden, hidden, device=device),
                    nn.Tanh(),
                    nn.Linear(hidden, 1, device=device),
                )

            def forward(self, pixels, action, vector=None):
                img_features = self.cnn(pixels)
                act = action.flatten(start_dim=1)
                parts = [img_features, act]
                if vector is not None and self.obs_dim > 0:
                    parts.append(vector)
                x = torch.cat(parts, dim=-1)
                return self.fc(x)

        q1_net = CriticNet(shared_cnn, num_cells, device, obs_dim)
        q2_net = CriticNet(shared_cnn, num_cells, device, obs_dim)
        q1_net_target = CriticNet(target_cnn, num_cells, device, obs_dim)
        q2_net_target = CriticNet(target_cnn, num_cells, device, obs_dim)

        # Wrap in ValueOperator for state dict / parameter access,
        # but call via .module() directly in trainer to pass raw tensors
        self.q1 = ValueOperator(module=q1_net, in_keys=["pixels", "action"])
        self.q2 = ValueOperator(module=q2_net, in_keys=["pixels", "action"])
        self.q1_target = ValueOperator(module=q1_net_target, in_keys=["pixels", "action"])
        self.q2_target = ValueOperator(module=q2_net_target, in_keys=["pixels", "action"])

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        print(
            f"SACPolicy initialized | obs_dim={obs_dim} | "
            f"fusion_input={fusion_input_size} | critic_input={critic_input_size}"
        )

    def sample_noise(self):
        """Resample noise for all noisy layers in the actor."""
        for m in self.actor.modules():
            if hasattr(m, "sample_noise"):
                m.sample_noise()

    def modules(self):
        return {
            "actor": self.actor,
            "q1": self.q1,
            "q2": self.q2,
            "q1_target": self.q1_target,
            "q2_target": self.q2_target,
        }
