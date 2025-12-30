import warnings

warnings.filterwarnings("ignore")

import torch
from torch import nn, multiprocessing
from collections import defaultdict

import matplotlib.pyplot as plt
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    DoubleToFloat,
    StepCounter,
    TransformedEnv,
    ToTensorImage,
    GrayScale,
    Resize,
    CatFrames,
    RewardSum,
    VecNorm,
    ParallelEnv,
    SqueezeTransform,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import copy

try:
    import wandb
except Exception:
    wandb = None


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
            # Convert to torch tensors and ensure float dtype
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
    start_steps = 1000

    # SAC specific
    gamma = 0.99
    tau = 0.005
    alpha = 0.2

    frames_per_batch = 1000
    # Use a single env by default to avoid a shared-TensorDict shape mismatch
    # observed when using ParallelEnv + SyncDataCollector in this workspace.
    # Set to >1 if you confirm your torchrl version/ParallelEnv behavior is compatible.
    num_envs = 1  # Number of parallel environments (set to 1 as a safe default)


def setup_multiprocessing():
    """Initialize multiprocessing settings"""
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass


def get_device():
    """Determine the appropriate device for training"""
    is_fork = multiprocessing.get_start_method() == "fork"
    if torch.cuda.is_available() and not is_fork:
        return torch.device(0)
    return torch.device("cpu")


from torchrl.envs.libs.gym import GymWrapper

import numpy as np
import torch


class SqueezedGymWrapper(GymWrapper):
    """GymWrapper that ensures the action is a 1D numpy array before
    forwarding it to the underlying gym env."""

    def step(self, tensordict):
        if "action" in tensordict:
            action = tensordict["action"]
            action_np = np.asarray(action).squeeze()
            if action_np.ndim == 0:
                action_np = action_np.reshape(1)
            tensordict["action"] = action_np
        return super().step(tensordict)


class SACEnv:
    """Environment factory and wrapper for SAC training with parallel environments."""

    def __init__(self, env_name: str = "CarRacing-v3", num_envs: int = 4, device=None):
        if device is None:
            device = get_device()
        self.device = device
        self.num_envs = num_envs

        def make_env():
            import gymnasium as gym  # TODO: stop doing stupid stuff

            base_env = SqueezedGymWrapper(
                gym.make(env_name),
                device=device,
            )

            env = TransformedEnv(base_env)
            env.append_transform(ToTensorImage(in_keys=["pixels"]))
            env.append_transform(GrayScale())
            env.append_transform(Resize(84, 84))
            env.append_transform(CatFrames(N=3, dim=-3))
            env.append_transform(RewardSum())
            env.append_transform(StepCounter())
            env.append_transform(DoubleToFloat())

            env.append_transform(SqueezeTransform(dim=-2, in_keys=["action"]))

            return env

        self.make_env = make_env

        env = ParallelEnv(
            num_workers=num_envs,
            create_env_fn=make_env,
            serial_for_single=True,
        )
        env.append_transform(VecNorm(in_keys=["pixels"]))
        self.env = env

    def get_env(self) -> TransformedEnv:
        return self.env

    def print_specs(self):
        # Skip check_env_specs for parallel envs as it can be overly strict
        print("Environment specs:")
        print(f"  Observation spec: {self.env.observation_spec}")
        print(f"  Action spec: {self.env.action_spec}")
        print(f"  Reward spec: {self.env.reward_spec}")
        print(f"  Batch size: {self.env.batch_size}")
        print("=" * 60)

    def rollout(self, *args, **kwargs):
        return self.env.rollout(*args, **kwargs)


def print_env_specs(env: TransformedEnv):
    """Print environment specifications"""
    check_env_specs(env)
    print("=" * 60)


def main():
    """Main training function"""
    setup_multiprocessing()
    config = SACConfig()
    device = get_device()

    print(f"Using device: {device}")
    print(f"Using {config.num_envs} parallel environments")

    env_obj = SACEnv(num_envs=config.num_envs, device=device)
    env = env_obj.get_env()

    env_obj.print_specs()
    print("Running test rollout...")
    try:
        env_obj.rollout(3)
        print("Test rollout successful!")
    except Exception as e:
        print(f"Test rollout failed (this is okay): {e}")

    # --- Runtime diagnostics to help find shape mismatch ---
    try:
        print("Diagnostic: calling env.reset() and printing 'pixels' shapes if present")
        td = env.reset()
        print("Diagnostic: reset batch_size:", td.batch_size)
        if "pixels" in td:
            p = td["pixels"]
            try:
                print("Diagnostic: reset 'pixels' shape:", tuple(p.shape))
            except Exception as _e:
                print("Diagnostic: couldn't read 'pixels' shape:", _e)
    except Exception as _e:
        print("Diagnostic: env.reset() failed:", _e)

    # Try worker-level reset/shapes (if ParallelEnv exposes workers)
    # TODO: remove this garbage code
    for attr in ("_envs", "envs", "workers"):
        ws = getattr(env, attr, None)
        if ws:
            try:
                w = ws[0]
                print(f"Diagnostic: worker container '{attr}' found, type: {type(w)}")
                try:
                    w_td = w.reset()
                    print("Diagnostic: worker reset batch_size:", w_td.batch_size)
                    if "pixels" in w_td:
                        try:
                            print(
                                "Diagnostic: worker 'pixels' shape:",
                                tuple(w_td["pixels"].shape),
                            )
                        except Exception as _e:
                            print(
                                "Diagnostic: couldn't read worker 'pixels' shape:", _e
                            )
                except Exception as _e:
                    print("Diagnostic: worker reset failed:", _e)
            except Exception:
                pass
            break
    # ------------------------------------------------------

    sac_policy = SACPolicy(env, num_cells=config.num_cells, device=device)

    target_q1 = copy.deepcopy(sac_policy.q1)
    target_q2 = copy.deepcopy(sac_policy.q2)

    optimizers = {
        "actor": torch.optim.Adam(sac_policy.actor.parameters(), lr=config.lr),
        "value": torch.optim.Adam(sac_policy.value.parameters(), lr=config.lr),
        "q1": torch.optim.Adam(sac_policy.q1.parameters(), lr=config.lr),
        "q2": torch.optim.Adam(sac_policy.q2.parameters(), lr=config.lr),
    }

    # ======== SAC training inits ========
    log_alpha = torch.tensor(
        [float(torch.log(torch.tensor(config.alpha)))],
        requires_grad=True,
        device=device,
    )
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.lr)

    def get_alpha():
        return log_alpha.exp()

    # -- losses
    critic_loss_fn = nn.SmoothL1Loss()
    value_loss_fn = nn.MSELoss()

    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizers["actor"],
        max(1, config.replay_size // max(1, config.batch_size)),
        eta_min=0.0,
    )

    # -- soft update helper
    def soft_update(target_module, source_module, tau=config.tau):
        for t_param, s_param in zip(
            target_module.parameters(), source_module.parameters()
        ):
            t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)

    training_inits = {
        "log_alpha": log_alpha,
        "get_alpha": get_alpha,
        "alpha_optimizer": alpha_optimizer,
        "critic_loss_fn": critic_loss_fn,
        "value_loss_fn": value_loss_fn,
        "actor_scheduler": actor_scheduler,
        "soft_update": soft_update,
    }

    replay = ReplayBuffer(
        storage=LazyTensorStorage(max_size=config.replay_size),
        sampler=SamplerWithoutReplacement(),
    )

    return {
        "env": env_obj,
        "policy": sac_policy,
        "targets": (target_q1, target_q2),
        "optimizers": optimizers,
        "replay": replay,
        "config": config,
        "training_inits": training_inits,
    }


def train_sac(components, total_frames: int = 100_000, updates_per_batch: int = 1):
    """Simple SAC training loop skeleton.

    This follows the structure of the PPO example: collect batches from the
    `SyncDataCollector`, fill the replay buffer, and perform update steps.

    The update step here is a focused, minimal implementation that relies on
    the provided `components` dict returned from `main()`.
    """
    env_obj = components["env"]
    sac_policy = components["policy"]
    target_q1, target_q2 = components["targets"]
    optimizers = components["optimizers"]
    replay = components["replay"]
    config = components["config"]

    device = sac_policy.device

    collector = SyncDataCollector(
        create_env_fn=env_obj.make_env,
        policy=sac_policy.actor,
        frames_per_batch=config.frames_per_batch,
        total_frames=total_frames,
        # split_trajs=False,
        storing_device=device,
        device=device,
        reset_at_each_iter=True,
    )

    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""
    use_wandb = False
    if wandb is not None:
        try:
            wandb.init(project="AssetoCorsaRL", reinit=True)
            use_wandb = True
        except Exception:
            use_wandb = False

    try:
        for i, tensordict_data in enumerate(collector, start=1):
            data_view = tensordict_data.reshape(-1)
            try:
                replay.extend(data_view.cpu())
            except Exception as e:
                print(f"Warning: Could not extend replay buffer: {e}")
                pass

            if len(replay) >= config.batch_size:
                for _ in range(updates_per_batch):
                    batch = replay.sample(config.batch_size)
                    # TODO: Add actual SAC update logic here

            try:
                reward = tensordict_data[("next", "reward")].mean().item()
            except Exception:
                reward = tensordict_data["next", "reward"].mean().item()

            logs["reward"].append(reward)
            pbar.update(tensordict_data.numel())
            if "step_count" in tensordict_data:
                step_count = tensordict_data["step_count"].max().item()
            else:
                step_count = 0
            logs["step_count"].append(step_count)

            if i % 10 == 0:
                with set_exploration_type(
                    ExplorationType.DETERMINISTIC
                ), torch.no_grad():
                    try:
                        eval_rollout = env_obj.get_env().rollout(1000, sac_policy.actor)
                        eval_reward = eval_rollout["next", "reward"].mean().item()
                        logs.setdefault("eval reward", []).append(eval_reward)
                        if use_wandb:
                            wandb.log({"eval/reward": eval_reward, "step": pbar.n})
                    except Exception as e:
                        print(f"Warning: Eval rollout failed: {e}")
                        pass

            pbar.set_description(f"avg_reward={logs['reward'][-1]:4.4f}")
            if use_wandb:
                wandb.log({"train/reward": reward, "train/step": pbar.n})

            if pbar.n >= total_frames:
                break
    except Exception as e:
        print(f"Collector iteration failed: {e}")
        # Diagnostic: inspect env.reset() output shapes
        try:
            td_reset = env_obj.get_env().reset()
            print("Diagnostic: env.reset() batch_size:", td_reset.batch_size)
            for k, v in td_reset.items():
                # Print type and shape; if nested TensorDict, recurse one level
                if isinstance(v, TensorDict):
                    print(
                        f"  reset key {k}: TensorDict, batch_size={v.batch_size}, keys={list(v.keys())}"
                    )
                    for nk, nv in v.items():
                        try:
                            print(f"    nested {nk}: shape={tuple(nv.shape)}")
                        except Exception:
                            print(f"    nested {nk}: type={type(nv)}")
                else:
                    try:
                        print(
                            f"  reset key {k}: type={type(v)}, shape={tuple(v.shape)}"
                        )
                    except Exception:
                        print(f"  reset key {k}: type={type(v)} (no shape)")
        except Exception as e2:
            print("Diagnostic: env.reset() failed:", e2)
            td_reset = None

        # Try a manual step with a midpoint action to see shapes
        try:
            if td_reset is not None:
                action_spec = env_obj.get_env().action_spec
                # Build a mid-range action compatible with the spec
                try:
                    low = action_spec.space.low.to(device)
                    high = action_spec.space.high.to(device)
                    sample_action = ((low + high) / 2).to(device)
                except Exception:
                    # Fallback: zeros with correct shape
                    sample_action = torch.zeros(action_spec.shape, device=device)

                td_in = td_reset.clone()
                td_in["action"] = sample_action
                td_out = env_obj.get_env().step(td_in)
                print("Diagnostic: manual step output keys and shapes:")
                for k, v in td_out.items():
                    # If TensorDict, print nested keys and shapes
                    if isinstance(v, TensorDict):
                        print(
                            f"  step key {k}: TensorDict, batch_size={v.batch_size}, keys={list(v.keys())}"
                        )
                        for nk, nv in v.items():
                            try:
                                print(f"    nested {nk}: shape={tuple(nv.shape)}")
                            except Exception:
                                print(f"    nested {nk}: type={type(nv)}")
                    else:
                        try:
                            print(
                                f"  step key {k}: type={type(v)}, shape={tuple(v.shape)}"
                            )
                        except Exception:
                            print(f"  step key {k}: type={type(v)} (no shape)")
        except Exception as e3:
            print("Diagnostic: manual step failed:", e3)

        # Inspect each worker individually to find the one returning unexpected shapes
        try:
            for attr in ("_envs", "envs", "workers"):
                ws = getattr(env_obj.get_env(), attr, None)
                if not ws:
                    continue
                print(
                    f"Diagnostic: found worker container '{attr}' with {len(ws)} workers; testing each worker step:"
                )
                for idx, w in enumerate(ws):
                    try:
                        w_td = w.reset()
                        print(f"  worker {idx} reset batch_size: {w_td.batch_size}")
                        for k, v in w_td.items():
                            try:
                                print(f"    reset {k}: shape={tuple(v.shape)}")
                            except Exception:
                                print(f"    reset {k}: type={type(v)}")
                        # build a sample action for this worker
                        try:
                            a_spec = w.action_spec
                        except Exception:
                            a_spec = env_obj.get_env().action_spec
                        try:
                            low = a_spec.space.low
                            high = a_spec.space.high
                            sample_act = ((low + high) / 2).to(device)
                        except Exception:
                            sample_act = torch.zeros(a_spec.shape, device=device)

                        td_w_in = w_td.clone()
                        td_w_in["action"] = sample_act
                        w_out = w.step(td_w_in)
                        print(f"  worker {idx} step output:")
                        for k, v in w_out.items():
                            try:
                                print(f"    {k}: shape={tuple(v.shape)}")
                            except Exception:
                                print(f"    {k}: type={type(v)}")
                    except Exception as we:
                        print(f"    worker {idx} test failed: {we}")
                break
        except Exception as e4:
            print("Diagnostic: per-worker inspection failed:", e4)

        # Re-raise so the original failure is still visible to the caller
        raise

    pbar.close()
    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass
    return logs


if __name__ == "__main__":
    env = main()
    train_sac(env)
