import warnings

warnings.filterwarnings("ignore")

from copy import deepcopy

import torch
from torch import nn, multiprocessing
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

from .noisy import NoisyLazyLinear


def _module_device(module: nn.Module, fallback: torch.device | None = None) -> torch.device:
    """Return the device for a module's parameters (or fallback if empty)."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return fallback if fallback is not None else torch.device("cpu")


def _init_orthogonal(module, gain=1.0):
    """Apply orthogonal init to a Linear or Conv2d layer."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


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
        self.use_lstm = False
        self.stateful_inference = False
        self._context_state = None
        self.context_lstm = None
        self.register_buffer("_min_scale", min_scale)
        self.register_buffer("_max_scale", max_scale)
        mlp_input_size = fusion_size

        def make_lin(i, o):
            if use_noisy:
                return NoisyLazyLinear(o, sigma=noise_sigma, device=device)
            return nn.Linear(i, o, device=device)

        self.mlp = nn.Sequential(
            make_lin(mlp_input_size, num_cells),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            make_lin(num_cells, num_cells),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            make_lin(num_cells, 2 * action_dim),
        )
        self.param_extractor = NormalParamExtractor(
            scale_mapping="biased_softplus_1.0",
            scale_lb=float(min_scale.min().item()),
        )
        self._init_weights()

    def _init_weights(self):
        lrelu_gain = nn.init.calculate_gain("leaky_relu", 0.01)
        _init_orthogonal(self.mlp[0], gain=lrelu_gain)
        _init_orthogonal(self.mlp[3], gain=lrelu_gain)
        _init_orthogonal(self.mlp[6], gain=0.01)

        action_dim = self.mlp[6].out_features // 2  # = 2

        with torch.no_grad():
            # loc biases: keep steer centered; bias accel/throttle action positive.
            loc_bias = torch.zeros(
                action_dim, device=self.mlp[6].bias.device, dtype=self.mlp[6].bias.dtype
            )
            if action_dim >= 2:
                # pre-tanh mean; tanh(1.0) ~= 0.76 so sampled accel starts clearly > 0.
                loc_bias[1] = 1.0
            elif action_dim == 1:
                loc_bias[0] = 1.0
            self.mlp[6].bias[:action_dim] = loc_bias

            # log_scale biases: allow wider steer exploration, keep accel tighter.
            log_scale_bias = torch.zeros(
                action_dim, device=self.mlp[6].bias.device, dtype=self.mlp[6].bias.dtype
            )
            if action_dim >= 1:
                log_scale_bias[0] = 0.8
            if action_dim >= 2:
                # Smaller accel std so sampled throttle reflects loc bias at init.
                log_scale_bias[1] = -1.5
            self.mlp[6].bias[action_dim:] = log_scale_bias

    def reset_context(self):
        self._context_state = None

    def forward_features(self, img_feat, vector=None):
        if vector is not None and self.obs_dim > 0:
            if img_feat.ndim == 2 and vector.ndim == 1:
                vector = vector.unsqueeze(0)
            if img_feat.ndim == 3 and vector.ndim == 2:
                vector = vector.unsqueeze(1)
            x = torch.cat([img_feat, vector], dim=-1)
        else:
            x = img_feat

        if x.ndim == 3:
            x = x[:, -1, :]

        loc, scale = self.param_extractor(self.mlp(x))
        scale = torch.clamp(scale, min=self._min_scale, max=self._max_scale)
        return {"loc": loc, "scale": scale}

    def forward(self, pixels, vector=None):
        target_device = _module_device(self.cnn, fallback=self._min_scale.device)
        if isinstance(pixels, torch.Tensor) and pixels.device != target_device:
            pixels = pixels.to(target_device, non_blocking=True)
        if isinstance(vector, torch.Tensor) and vector.device != target_device:
            vector = vector.to(target_device, non_blocking=True)

        squeeze_batch = False
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
            squeeze_batch = True
            if vector is not None and vector.ndim == 1:
                vector = vector.unsqueeze(0)

        if pixels.ndim == 5:
            b, t, c, h, w = pixels.shape
            img_feat = self.cnn(pixels.view(b * t, c, h, w)).view(b, t, -1)
        else:
            img_feat = self.cnn(pixels)

        out = self.forward_features(img_feat, vector=vector)
        if squeeze_batch:
            out = {k: v.squeeze(0) for k, v in out.items()}
        return out


class CriticNet(nn.Module):
    """
    LSTM processes observation features ONLY (no action).
    Actions are injected after the LSTM into the MLP head.
    This lets online/target LSTMs share the same obs pass.
    """

    def __init__(
        self,
        encoder,
        cnn_output_size,
        action_dim,
        hidden,
        device,
        obs_dim,
    ):
        super().__init__()
        self.cnn = encoder
        self.obs_dim = obs_dim
        self.use_lstm = False
        self.stateful_inference = False
        self._context_state = None
        self.action_dim = action_dim
        self.register_buffer("_dummy_device", torch.empty(0, device=device))

        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, 128, device=device),
            nn.LeakyReLU(),
        )

        obs_fusion_size = cnn_output_size + obs_dim
        fc_input_size = obs_fusion_size + 128

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden, device=device),
            nn.LayerNorm(hidden, device=device),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden, device=device),
            nn.LayerNorm(hidden, device=device),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1, device=device),
        )
        self._init_weights()

    def _init_weights(self):
        lrelu_gain = nn.init.calculate_gain("leaky_relu", 0.01)
        _init_orthogonal(self.fc[0], gain=lrelu_gain)
        _init_orthogonal(self.fc[3], gain=lrelu_gain)
        _init_orthogonal(self.fc[6], gain=0.01)

    def reset_context(self):
        self._context_state = None

    def forward(self, pixels, action, vector=None, img_features=None):
        target_device = _module_device(self.cnn, fallback=self._dummy_device.device)
        if isinstance(pixels, torch.Tensor) and pixels.device != target_device:
            pixels = pixels.to(target_device, non_blocking=True)
        if isinstance(action, torch.Tensor) and action.device != target_device:
            action = action.to(target_device, non_blocking=True)
        if isinstance(vector, torch.Tensor) and vector.device != target_device:
            vector = vector.to(target_device, non_blocking=True)

        if img_features is None:
            img_features = self.cnn(pixels)

        parts = [img_features]
        if vector is not None and self.obs_dim > 0:
            parts.append(vector)
        obs_feat = torch.cat(parts, dim=-1)

        obs_context = obs_feat
        if obs_context.ndim == 3:
            obs_context = obs_context[:, -1, :]

        act_emb = self.action_embed(action.flatten(start_dim=1))
        x = torch.cat([obs_context, act_emb], dim=-1)
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

        min_scale = torch.full((action_dim,), 0.01, device=device)
        max_scale = torch.full((action_dim,), 2.0, device=device)

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
                actor_net, in_keys=["pixels", "vector"], out_keys=["loc", "scale"]
            )
        else:
            policy_module = TensorDictModule(
                actor_net, in_keys=["pixels"], out_keys=["loc", "scale"]
            )

        low_t = torch.full((action_dim,), -1.0, dtype=torch.float32, device=device)
        high_t = torch.full((action_dim,), 1.0, dtype=torch.float32, device=device)
        dist_kwargs = {"low": low_t, "high": high_t}

        self.actor = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs=dist_kwargs,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
        )

        if self.use_noisy:
            noisy_count = sum(1 for m in self.actor.modules() if hasattr(m, "sample_noise"))
            print(f"Using noisy actor: found {noisy_count} noisy layer(s)")

        critic_input_size = cnn_output_size + action_dim + obs_dim

        q1_net = CriticNet(
            shared_cnn,
            cnn_output_size,
            action_dim,
            num_cells,
            device,
            obs_dim,
        )
        q2_net = CriticNet(
            shared_cnn,
            cnn_output_size,
            action_dim,
            num_cells,
            device,
            obs_dim,
        )
        q1_net_target = CriticNet(
            target_cnn,
            cnn_output_size,
            action_dim,
            num_cells,
            device,
            obs_dim,
        )
        q2_net_target = CriticNet(
            target_cnn,
            cnn_output_size,
            action_dim,
            num_cells,
            device,
            obs_dim,
        )

        q_in_keys = ["pixels", "action"]
        if obs_dim > 0:
            q_in_keys.append("vector")

        # wrap in ValueOperator for state dict / parameter access,
        # but call via .module() directly in trainer to pass raw tensors
        self.q1 = ValueOperator(module=q1_net, in_keys=q_in_keys)
        self.q2 = ValueOperator(module=q2_net, in_keys=q_in_keys)
        self.q1_target = ValueOperator(module=q1_net_target, in_keys=q_in_keys)
        self.q2_target = ValueOperator(module=q2_net_target, in_keys=q_in_keys)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        print(
            f"SACPolicy initialized | obs_dim={obs_dim} | "
            f"fusion_input={fusion_input_size} | critic_input={critic_input_size}"
        )

    def reset_context(self):
        for m in [self.actor, self.q1, self.q2, self.q1_target, self.q2_target]:
            if m is None:
                continue
            for sub in m.modules():
                if hasattr(sub, "reset_context"):
                    sub.reset_context()

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
