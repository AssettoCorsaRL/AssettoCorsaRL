import torch
from typing import Optional
from torchrl.envs.transforms import (
    Compose,
    Resize,
    ToTensorImage,
    GrayScale,
    CatFrames,
)
from torchrl.envs import ParallelEnv, TransformedEnv, GymEnv


def create_gym_env(
    env_name: str = "CarRacing-v3",
    height: int = 84,
    width: int = 84,
    device="cuda",
    num_envs: int = 1,
    render_mode: str = None,
    domain_randomize=False,
    fixed_track_seed=6741,
) -> ParallelEnv:
    class FixedSeedGymEnv(GymEnv):
        def __init__(self, *args, fixed_seed: Optional[int] = None, **kwargs):
            super().__init__(*args, **kwargs)
            self._fixed_seed = fixed_seed

        def reset(self, *args, **kwargs):
            if self._fixed_seed is not None:
                kwargs.setdefault("seed", self._fixed_seed)
                try:
                    return super().reset(*args, **kwargs)
                except TypeError:
                    if hasattr(self, "seed"):
                        self.seed(self._fixed_seed)
                    return super().reset(*args, **kwargs)
            return super().reset(*args, **kwargs)

    def _make_env():
        return FixedSeedGymEnv(
            env_name,
            from_pixels=True,
            pixels_only=True,
            device=device,
            render_mode=render_mode,
            domain_randomize=domain_randomize,
            fixed_seed=fixed_track_seed,
        )

    base_env = ParallelEnv(num_workers=num_envs, create_env_fn=_make_env, device=device)

    transform = Compose(
        ToTensorImage(from_int=True),
        Resize(h=height, w=width),
        GrayScale(),
        CatFrames(N=4, in_keys=["pixels"], dim=-3),
    )

    transformed_env = TransformedEnv(base_env, transform)
    return transformed_env


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    num_envs = 4
    env = create_gym_env(device=DEVICE, num_envs=num_envs, fixed_track_seed=42)
    td = env.reset()
    print("Initial time step:", td)

    # multiple resets will now use the same track (seed=42)
    td2 = env.reset()
    print("Reset again; track should be identical:", td2)
