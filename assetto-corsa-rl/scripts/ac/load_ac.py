"""
Usage:
    acrl ac test --checkpoint models/bc_sac_pretrained.pt --vae-checkpoint loss=0.1050.ckpt
"""

import warnings

warnings.filterwarnings("ignore")

import sys
import time
import json
import socket
from pathlib import Path
import torch
from torch import nn, multiprocessing
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
import click
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from assetto_corsa_rl.model.vae import load_vae_encoder
from assetto_corsa_rl.ac_env import create_transformed_env
from assetto_corsa_rl.model.sac import SACPolicy

try:
    from assetto_corsa_rl.cli_registry import cli_command, cli_option
except Exception:
    from ...src.assetto_corsa_rl.cli_registry import cli_command, cli_option


def get_device():
    """Determine the appropriate device for training"""
    is_fork = multiprocessing.get_start_method() == "fork"
    if torch.cuda.is_available() and not is_fork:
        return torch.device(0)
    return torch.device("cpu")


@cli_command(group="ac", name="test", help="Test trained BC-SAC policy in Assetto Corsa")
@cli_option(
    "--checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to BC-SAC checkpoint",
)
@cli_option(
    "--vae-checkpoint",
    type=click.Path(exists=True),
    required=True,
    help="Path to VAE checkpoint",
)
@cli_option("--max-steps", default=10000, help="Maximum steps per episode")
@cli_option("--episodes", default=5, help="Number of episodes to run")
@cli_option("--render", is_flag=True, help="Render the environment")
def test(checkpoint, vae_checkpoint, max_steps, episodes, render):
    """Load and test a trained BC-SAC policy in Assetto Corsa."""
    checkpoint = Path(checkpoint)
    vae_checkpoint = Path(vae_checkpoint)

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {checkpoint}...")
    ckpt = torch.load(checkpoint, map_location=device)
    config = ckpt.get("config", {})
    num_cells = config.get("num_cells", 256)

    print(f"\nCheckpoint info:")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Val MSE: {ckpt.get('val_mse', 'N/A')}")
    print(f"  Num cells: {num_cells}")
    print(f"  VAE checkpoint: {config.get('vae_checkpoint_path', 'N/A')}")

    print("\nCreating Assetto Corsa environment...")
    env = create_transformed_env(
        racing_line_path="racing_lines.json",
        device=device,
        image_shape=(84, 84),
        frame_stack=4,
    )

    print("\nInitializing policy...")
    agent = SACPolicy(
        env=env,
        num_cells=num_cells,
        device=device,
        use_noisy=False,
        actor_dropout=0.0,
        vae_checkpoint_path=str(vae_checkpoint),
    )

    modules = agent.modules()
    actor = modules["actor"]
    print("\nLoading actor weights...")
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()

    print(f"\n{'='*60}")
    print("Starting evaluation...")
    print(f"{'='*60}")
    print("\nWaiting for Assetto Corsa connection...")
    print("Make sure the AC_RL app is running in Assetto Corsa!")
    input("Press Enter when ready...")

    episode_rewards = []
    episode_lengths = []

    td = env.reset()

    input("Press Enter when in position...")

    for episode in range(episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{episodes}")
        print(f"{'='*60}")

        if episode > 0:
            td = env.reset()

        episode_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            with torch.no_grad():
                if td.batch_size == torch.Size([]):
                    td = td.unsqueeze(0)

                actor_out = actor(td)
                action_batched = actor_out["loc"]  # mean action (deterministic)

                action_unbatched = (
                    action_batched.squeeze(0) if action_batched.shape[0] == 1 else action_batched
                )

            action_np = action_unbatched.cpu().numpy()

            gym_env = env.base_env._env
            obs, reward, done, truncated, info = gym_env.step(action_np)

            pixels = torch.from_numpy(obs["image"]).to(device).permute(2, 0, 1).float() / 255.0
            current_pixels = td["pixels"].squeeze(0)  # Remove batch dim
            if current_pixels.shape[0] == 4:
                new_pixels = torch.cat([current_pixels[1:], pixels], dim=0)
            else:
                new_pixels = pixels

            td["pixels"] = new_pixels.unsqueeze(0)

            episode_reward += reward
            steps += 1

            if steps % 100 == 0:
                print(f"  Step {steps}: reward={episode_reward:.2f} action={action_np}")

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        print(f"\nEpisode {episode + 1} finished:")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Steps: {steps}")

    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    test()
