import time
import torch
from tensordict import TensorDict

from env import create_gym_env
from .train.train_utils import (
    sample_random_action,
    expand_actions_for_envs,
    get_inner,
    extract_reward_and_done,
)


def simulate(
    num_envs: int = 4,
    episodes_per_env: int = 3,
    device=None,
    max_steps: int = 1_000_000,
):
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("Using device:", device)

    env = create_gym_env(device=device, num_envs=num_envs)
    td = env.reset()
    current_td = get_inner(td)

    episode_counts = [0 for _ in range(num_envs)]
    current_returns = torch.zeros(num_envs, device=device)

    steps = 0
    start = time.time()

    print(f"Starting simulation: {num_envs} envs, {episodes_per_env} episodes each")

    while any(c < episodes_per_env for c in episode_counts) and steps < max_steps:
        actions = sample_random_action(num_envs, dev=device)
        target_batch = current_td.batch_size
        actions_step = expand_actions_for_envs(actions, target_batch)
        action_td = TensorDict({"action": actions_step}, batch_size=target_batch)

        next_td = env.step(action_td)
        td_next = get_inner(next_td)

        rewards, dones = extract_reward_and_done(td_next, num_envs, device)

        current_returns = current_returns.to(rewards.device) + rewards.to(
            current_returns.device
        )

        if dones.any():
            next_pixels = td_next["pixels"]
            for i in range(num_envs):
                if dones[i].item() and episode_counts[i] < episodes_per_env:
                    episode_counts[i] += 1
                    px = next_pixels[i]
                    try:
                        pmin = float(px.min().item())
                        pmax = float(px.max().item())
                    except Exception:
                        pmin = None
                        pmax = None

                    print(
                        f"Env {i} finished episode {episode_counts[i]}: return={current_returns[i].item():.2f}, pixels min={pmin}, max={pmax}"
                    )
                    # reset return for that env
                    current_returns[i] = 0.0

            try:
                reset_td = env.reset()
                current_td = get_inner(reset_td)
            except Exception:
                current_td = env.reset()
        else:
            current_td = td_next

        steps += num_envs

    elapsed = time.time() - start
    print(f"Simulation completed in {elapsed:.1f}s after {steps} env steps")
    print("Episode counts per env:", episode_counts)

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    simulate()
