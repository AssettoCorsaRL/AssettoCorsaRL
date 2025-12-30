import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
from torch import multiprocessing
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import (
    TransformedEnv,
    ToTensorImage,
    GrayScale,
    Resize,
    CatFrames,
    RewardSum,
    StepCounter,
    DoubleToFloat,
    VecNorm,
)


def setup_multiprocessing():
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass


class RandomPolicy:
    """Return a random action compatible with the env.action_spec.

    This lightweight policy returns a TensorDict with the sampled action.
    It works for batch size=(1,) environments used here.
    """

    def __init__(self, action_spec):
        self.action_spec = action_spec

    def __call__(self, tensordict: TensorDict):
        # sample using the wrapped gym space if available
        try:
            sample = self.action_spec.space.sample()
            a = torch.as_tensor(sample, dtype=torch.float32)
        except Exception:
            # fallback: zeros
            a = torch.zeros(self.action_spec.shape, dtype=torch.float32)

        # expand to match batch if needed
        batch_size = tensordict.batch_size
        if isinstance(batch_size, torch.Size):
            batch_shape = tuple(batch_size)
        else:
            batch_shape = (1,)
        a = a.reshape((1,) + tuple(a.shape)).expand(batch_shape + tuple(a.shape))
        out = tensordict.clone()
        out["action"] = a
        return out


def make_env(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create GymEnv with a single-batch worker
    gname = "CarRacing-v3"
    raw = GymEnv(
        gname,
        device=device,
        batch_size=(1,),
        from_pixels=True,
    )

    print("Gym raw observation_space:", raw._env.observation_space)
    print("Gym raw action_space:", raw._env.action_space)

    env = TransformedEnv(raw)
    env.append_transform(ToTensorImage(in_keys=["pixels"]))
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=3, dim=-3))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(DoubleToFloat())
    env.append_transform(VecNorm(in_keys=["pixels"]))

    return env


def run_rollout_test():
    setup_multiprocessing()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(device)

    print("TorchRL observation_spec:", env.observation_spec)
    print("TorchRL action_spec:", env.action_spec)

    rp = RandomPolicy(env.action_spec)
    try:
        print("Running short deterministic rollout with random policy...")
        td_roll = env.rollout(5, policy=rp)
        print("Rollout keys:", list(td_roll.keys()))
        if ("next", "reward") in td_roll:
            print("Mean reward:", td_roll[("next", "reward")].mean().item())
        if "pixels" in td_roll:
            print("pixels shape:", tuple(td_roll["pixels"].shape))
    except Exception as e:
        print("rollout failed:", e)

    # Now exercise SyncDataCollector for one iteration
    print("Creating SyncDataCollector to collect a small batch...")
    collector = SyncDataCollector(
        create_env_fn=make_env,
        policy=rp,
        frames_per_batch=128,
        total_frames=128,
        storing_device=device,
        device=device,
        reset_at_each_iter=True,
    )

    try:
        for i, tensordict_data in enumerate(collector, start=1):
            print(
                f"Collected batch #{i}, batch_size: {tensordict_data.batch_size}, keys: {list(tensordict_data.keys())}"
            )
            if "pixels" in tensordict_data:
                try:
                    print(
                        "  pixels sample shape:", tuple(tensordict_data["pixels"].shape)
                    )
                except Exception as _e:
                    print("  couldn't read pixels shape:", _e)
            # print a small summary and break
            print("  reward mean:", tensordict_data[("next", "reward")].mean().item())
            break
    except Exception as e:
        print("Collector iteration failed:", e)


if __name__ == "__main__":
    run_rollout_test()
