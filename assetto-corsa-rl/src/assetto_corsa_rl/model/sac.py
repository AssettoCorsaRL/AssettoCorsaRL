import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn, multiprocessing
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
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
    """Soft Actor-Critic policy + value + twin critics built from nn modules.

    Attributes:
        actor: ProbabilisticActor
        value: ValueOperator (state value)
        q1: ValueOperator (Q1)
        q2: ValueOperator (Q2)
    """

    def __init__(
        self,
        env: GymEnv,
        num_cells: int = 256,
        device=None,
        use_noisy: bool = False,
        noise_sigma: float = 0.5,
    ):
        if device is None:
            device = get_device()
        self.device = device
        self.use_noisy = use_noisy
        self.noise_sigma = noise_sigma

        action_dim = int(env.action_spec.shape[-1])

        in_channels = 4  # TODO: remove this hardcoding later

        # CNN output size for 84x84 input: 64 * 7 * 7 = 3136
        cnn_output_size = 3136

        # CNN feature extractor for image inputs
        # Input: [B, C, H, W] - e.g., [B, in_channels, 84, 84] (CarRacing frames stacked)
        # For 84x84 inputs the conv outputs will be: 20x20 → 9x9 → 7x7 → flattened (64*7*7 = 3136)
        cnn_features = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=8, stride=4, padding=0, device=device
            ),  # → [B, 32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, device=device),
            # → [B, 64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=device),
            # → [B, 64, 7, 7]
            nn.ReLU(),
            nn.Flatten(start_dim=1),  # → [B, 64*7*7] = [B, 3136]
        )

        class BoundedNormalParams(nn.Module):
            def __init__(self, min_scale=0.1, max_scale=2.0):
                super().__init__()
                self.min_scale = min_scale
                self.max_scale = max_scale
                self.scale_range = max_scale - min_scale

            def forward(self, x):
                # x shape: [batch, 2*action_dim]
                loc, scale_raw = x.chunk(2, dim=-1)
                # Use sigmoid to bound between [min_scale, max_scale]
                scale = self.min_scale + torch.sigmoid(scale_raw) * self.scale_range
                return {"loc": loc, "scale": scale}

        # Helper to choose between noisy and standard linear layers
        def _make_linear(in_f: int, out_f: int):
            if self.use_noisy:
                # Lazy noisy linear is used so we don't need to specify in_features explicitly
                return NoisyLazyLinear(out_f, sigma=self.noise_sigma, device=device)
            return nn.Linear(in_f, out_f, device=device)

        actor_net = nn.Sequential(
            cnn_features,
            _make_linear(cnn_output_size, num_cells),
            nn.Tanh(),
            _make_linear(num_cells, num_cells),
            nn.Tanh(),
            _make_linear(num_cells, 2 * action_dim),
            BoundedNormalParams(min_scale=0.3, max_scale=3.0),
        )

        policy_module = TensorDictModule(
            actor_net, in_keys=["pixels"], out_keys=["loc", "scale"]
        )

        # NOTE: To get the hardcoded action bounds, we would need to create the env here:
        # import gymnasium as gym

        # g = gym.make("CarRacing-v3")
        # print("gym action_bounds:", g.action_space.low, g.action_space.high)

        # TODO: remove hardcoded bounds and use env specs directly
        low = [-1.0, 0.0, 0.0]  # env.action_spec_unbatched.space.low
        high = [1.0, 1.0, 1.0]  # env.action_spec_unbatched.space.high

        try:
            low_t = torch.as_tensor(low, dtype=torch.float32)
            high_t = torch.as_tensor(high, dtype=torch.float32)
            if not torch.all(high_t > low_t):
                print(
                    f"Warning: invalid action bounds detected (low={low_t}, high={high_t}). "
                    "Defaulting to [-1, 1] for each action dimension to satisfy TanhNormal requirements."
                )
                low_t = -torch.ones_like(low_t)
                high_t = torch.ones_like(high_t)
            dist_kwargs = {
                "low": low_t,
                "high": high_t,
            }
        except Exception as _e:
            print(
                f"Warning: could not validate action bounds ({_e}); using raw spec values."
            )
            dist_kwargs = {
                "low": low,
                "high": high,
            }

        self.actor = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs=dist_kwargs,
            return_log_prob=True,
        )

        # Informational log when noisy nets are enabled
        if self.use_noisy:
            noisy_count = sum(
                1 for m in self.actor.modules() if hasattr(m, "sample_noise")
            )
            print(f"Using noisy actor: found {noisy_count} noisy layer(s)")

        value_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=8, stride=4, padding=0, device=device
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, device=device),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=device),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        value_net = nn.Sequential(
            value_cnn,
            nn.Linear(cnn_output_size, num_cells, device=device),
            nn.Tanh(),
            nn.Linear(num_cells, num_cells, device=device),
            nn.Tanh(),
            nn.Linear(num_cells, 1, device=device),
        )

        self.value = ValueOperator(module=value_net, in_keys=["pixels"])

        # CNN feature extractor for target value network
        value_cnn_target = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=8, stride=4, padding=0, device=device
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, device=device),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, device=device),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
        )

        value_net_target = nn.Sequential(
            value_cnn_target,
            nn.Linear(cnn_output_size, num_cells, device=device),
            nn.Tanh(),
            nn.Linear(num_cells, num_cells, device=device),
            nn.Tanh(),
            nn.Linear(num_cells, 1, device=device),
        )
        self.value_target = ValueOperator(module=value_net_target, in_keys=["pixels"])

        class CriticNet(nn.Module):
            def __init__(self, hidden: int, device):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        32,
                        kernel_size=8,
                        stride=4,
                        padding=0,
                        device=device,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        32, 64, kernel_size=4, stride=2, padding=0, device=device
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        64, 64, kernel_size=3, stride=1, padding=0, device=device
                    ),
                    nn.ReLU(),
                    nn.Flatten(start_dim=1),
                )

                # Critic input: CNN features (3136) + action (3) = 3139
                critic_input_size = cnn_output_size + action_dim
                self.fc = nn.Sequential(
                    nn.Linear(critic_input_size, hidden, device=device),
                    nn.Tanh(),
                    nn.Linear(hidden, hidden, device=device),
                    nn.Tanh(),
                    nn.Linear(hidden, 1, device=device),
                )

            def forward(self, pixels, action):
                img_features = self.cnn(pixels)
                act = action.flatten(start_dim=1)
                x = torch.cat([img_features, act], dim=-1)
                return self.fc(x)

        q1_net = CriticNet(num_cells, device)
        q2_net = CriticNet(num_cells, device)

        q1_net_target = CriticNet(num_cells, device)
        q2_net_target = CriticNet(num_cells, device)

        self.q1 = ValueOperator(module=q1_net, in_keys=["pixels", "action"])
        self.q2 = ValueOperator(module=q2_net, in_keys=["pixels", "action"])

        self.q1_target = ValueOperator(
            module=q1_net_target, in_keys=["pixels", "action"]
        )
        self.q2_target = ValueOperator(
            module=q2_net_target, in_keys=["pixels", "action"]
        )

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def sample_noise(self):
        """Resample noise for all noisy layers in the actor (and others if present)."""
        for name, module in self.modules().items():
            for m in module.modules():
                if hasattr(m, "sample_noise"):
                    m.sample_noise()

    def modules(self):
        return {
            "actor": self.actor,
            "value": self.value,
            "value_target": self.value_target,
            "q1": self.q1,
            "q2": self.q2,
            "q1_target": self.q1_target,
            "q2_target": self.q2_target,
        }
