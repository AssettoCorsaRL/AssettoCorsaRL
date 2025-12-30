import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn, multiprocessing
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator


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

    def __init__(self, env: GymEnv, num_cells: int = 256, device=None):
        if device is None:
            device = get_device()
        self.device = device

        action_dim = int(env.action_spec.shape[-1])

        actor_net = nn.Sequential(
            nn.Flatten(
                start_dim=1
            ),  # ← ADD THIS LINE to flatten [B, C, H, W] → [B, C*H*W]
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(2 * action_dim, device=device),
            NormalParamExtractor(),
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
            dist_kwargs = {"low": low_t, "high": high_t}
        except Exception as _e:
            print(
                f"Warning: could not validate action bounds ({_e}); using raw spec values."
            )
            dist_kwargs = {"low": low, "high": high}

        self.actor = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs=dist_kwargs,
            return_log_prob=True,
        )

        value_net = nn.Sequential(
            nn.Flatten(start_dim=1),  # ← ADD THIS LINE
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(1, device=device),
        )

        self.value = ValueOperator(module=value_net, in_keys=["pixels"])

        class CriticNet(nn.Module):
            def __init__(self, hidden: int, device):
                super().__init__()
                self.net = nn.Sequential(
                    nn.LazyLinear(hidden, device=device),
                    nn.Tanh(),
                    nn.LazyLinear(hidden, device=device),
                    nn.Tanh(),
                    nn.LazyLinear(hidden, device=device),
                    nn.Tanh(),
                    nn.LazyLinear(1, device=device),
                )

            def forward(self, pixels, action):
                obs = pixels.flatten(start_dim=1)
                act = action.flatten(start_dim=1)
                x = torch.cat([obs, act], dim=-1)
                return self.net(x)

        q1_net = CriticNet(num_cells, device)
        q2_net = CriticNet(num_cells, device)

        self.q1 = ValueOperator(module=q1_net, in_keys=["pixels", "action"])
        self.q2 = ValueOperator(module=q2_net, in_keys=["pixels", "action"])

    def modules(self):
        return {"actor": self.actor, "value": self.value, "q1": self.q1, "q2": self.q2}


class SACConfig:
    """Configuration for SAC training"""

    # Network / optimizer
    num_cells = 256
    lr = 3e-4
    max_grad_norm = 1.0

    # Replay / training
    batch_size = 256
    replay_size = 100_000
    start_steps = 1_000

    # SAC specific
    gamma = 0.99
    tau = 0.005
    alpha = 0.2

    frames_per_batch = 1000
    # Use a single env by default to avoid a shared-TensorDict shape mismatch
    # observed when using ParallelEnv + SyncDataCollector in this workspace.
    # Set to >1 if you confirm your torchrl version/ParallelEnv behavior is compatible.
    num_envs = 1  # Number of parallel environments (set to 1 as a safe default)
