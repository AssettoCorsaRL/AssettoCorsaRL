import warnings

warnings.filterwarnings("ignore")

from copy import deepcopy

import torch
from torch import nn, multiprocessing
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
import torch.nn.functional as F

# Local noisy layers (optional)
from .noisy import NoisyLazyLinear


def _init_orthogonal(module, gain=1.0):
    """Apply orthogonal init to a Linear or Conv2d layer."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class BoundedNormalParams(nn.Module):
    def __init__(self, min_scale, max_scale):
        super().__init__()
        self.register_buffer("min_scale", min_scale)
        self.register_buffer("max_scale", max_scale)
        self.register_buffer("scale_range", max_scale - min_scale)

        self.register_buffer("log_scale_min", torch.log(min_scale))
        self.register_buffer("log_scale_max", torch.log(max_scale))

    def forward(self, x):
        loc, log_scale = x.chunk(2, dim=-1)
        # soft clamp - gradients flow everywhere
        log_scale = self.log_scale_max - F.softplus(self.log_scale_max - log_scale)
        log_scale = self.log_scale_min + F.softplus(log_scale - self.log_scale_min)
        scale = torch.exp(log_scale)
        return {"loc": loc, "scale": scale}


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
        obs_dim,
        min_scale,
        max_scale,
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
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            make_lin(num_cells, num_cells),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            make_lin(num_cells, 2 * action_dim),
            BoundedNormalParams(min_scale=min_scale, max_scale=max_scale),
        )

        self._init_weights()

    def _init_weights(self):
        lrelu_gain = nn.init.calculate_gain("leaky_relu", 0.01)

        _init_orthogonal(self.mlp[0], gain=lrelu_gain)
        _init_orthogonal(self.mlp[3], gain=lrelu_gain)
        _init_orthogonal(self.mlp[6], gain=0.01)

        # loc biases at 0 is fine.
        # log_scale biases must start WITHIN [log(min), log(max)] so clamp doesn't kill gradients.
        # log(0.01) = -4.6,  log(0.5) = -0.69,  midpoint = -2.65
        action_dim = self.mlp[6].out_features // 2
        with torch.no_grad():
            # Access min/max from BoundedNormalParams (mlp[7])
            log_min = self.mlp[7].log_scale_min  # [-4.6, -4.6, -4.6]
            log_max = self.mlp[7].log_scale_max  # [-0.69, -0.69, -0.69]
            mid = (log_min + log_max) / 2  # [-2.65, -2.65, -2.65]
            self.mlp[6].bias[action_dim:] = mid

    def forward(self, pixels, vector=None):
        img_feat = self.cnn(pixels)
        if vector is not None and self.obs_dim > 0:
            x = torch.cat([img_feat, vector], dim=-1)
        else:
            x = img_feat
        return self.mlp(x)


class CriticNet(nn.Module):
    def __init__(self, encoder, cnn_output_size, action_dim, hidden, device, obs_dim):
        super().__init__()
        self.cnn = encoder
        self.obs_dim = obs_dim

        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, 128, device=device),
            nn.LeakyReLU(),
        )

        fusion_size = cnn_output_size + 128 + obs_dim
        self.fc = nn.Sequential(
            nn.Linear(fusion_size, hidden, device=device),
            nn.LayerNorm(hidden, device=device),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden, device=device),
            nn.LayerNorm(hidden, device=device),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1, device=device),
        )
        self._init_weights()

    def forward(self, pixels, action, vector=None):
        img_features = self.cnn(pixels)
        act = self.action_embed(action.flatten(start_dim=1))
        parts = [img_features, act]
        if vector is not None and self.obs_dim > 0:
            parts.append(vector)
        x = torch.cat(parts, dim=-1)
        return self.fc(x)


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

        # Auto-detect obs_dim from env if not explicitly provided
        if obs_dim == 0:
            try:
                spec = env.observation_spec
                if "vector" in spec.keys():
                    obs_dim = int(spec["vector"].shape[-1])
            except Exception:
                pass
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

        min_scale = torch.tensor([0.01, 0.01, 0.01], device=device)
        max_scale = torch.tensor([0.5, 0.5, 0.5], device=device)

        # ── Actor ─────────────────────────────────────────────────────────
        # Fuses CNN features with optional telemetry vector
        fusion_input_size = cnn_output_size + obs_dim

        actor_net = ActorNet(
            shared_cnn,
            fusion_input_size,
            num_cells,
            action_dim,
            self.actor_dropout,
            self.use_noisy,
            self.noise_sigma,
            device,
            obs_dim,
            min_scale,
            max_scale,
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

        q1_net = CriticNet(shared_cnn, cnn_output_size, action_dim, num_cells, device, obs_dim)
        q2_net = CriticNet(shared_cnn, cnn_output_size, action_dim, num_cells, device, obs_dim)
        q1_net_target = CriticNet(
            target_cnn, cnn_output_size, action_dim, num_cells, device, obs_dim
        )
        q2_net_target = CriticNet(
            target_cnn, cnn_output_size, action_dim, num_cells, device, obs_dim
        )

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
