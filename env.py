import torch
import numpy as np
import matplotlib.pyplot as plt
from torchrl.envs.transforms import (
    Compose,
    Resize,
    ToTensorImage,
    GrayScale,
    VecNorm,
    CatFrames,
    ObservationNorm,
)
from torchrl.envs import ParallelEnv, TransformedEnv, GymEnv


def create_gym_env(
    env_name: str = "CarRacing-v3",
    height: int = 84,
    width: int = 84,
    device="cuda",
    num_envs: int = 1,
    render_mode: str = None,
) -> ParallelEnv:
    def _make_env():
        return GymEnv(
            env_name,
            from_pixels=True,
            pixels_only=True,
            device=device,
            render_mode=render_mode,
        )

    base_env = ParallelEnv(num_workers=num_envs, create_env_fn=_make_env, device=device)

    transform = Compose(
        ToTensorImage(from_int=True),
        Resize(h=height, w=width),
        GrayScale(),
        # catframes BEFORE normalization to stack grayscale frames properly
        CatFrames(N=4, in_keys=["pixels"], dim=-3),
        # VecNorm(in_keys=["pixels"]), # * env already normalized [0,1]
    )

    transformed_env = TransformedEnv(base_env, transform)
    return transformed_env


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    num_envs = 4
    env = create_gym_env(device=DEVICE, num_envs=num_envs)
    td = env.reset()
    print("Initial time step:", td)

    pixels = td["pixels"]
    print("Pixels shape (batch, channels, height, width):", pixels.shape)
    print("Pixels dtype:", pixels.dtype)
    print("Pixels min/max (across batch):", pixels.min().item(), pixels.max().item())

    per_env_min = pixels.view(num_envs, -1).min(dim=1).values
    per_env_max = pixels.view(num_envs, -1).max(dim=1).values
    print("Per-env min:", per_env_min)
    print("Per-env max:", per_env_max)
