"""
Load a trained SAC model and watch it play CarRacing.

Usage:
    python load_sac.py --actor models/sac_actor_carracer_best.pt --encoder models/sac_encoder_carracer_best.pt
    python load_sac.py --actor models/sac_actor_carracer_bc_best.pt --encoder models/sac_encoder_carracer_bc_best.pt --episodes 5
"""

import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    FrameStackObservation,
    TransformObservation,
)

from sac.sac import SAC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(render_mode: str = "human"):
    """Create the CarRacing environment with same preprocessing as training."""
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    env = GrayscaleObservation(env, keep_dim=False)
    obs_space = env.observation_space
    normalized_obs_space = gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=obs_space.shape,
        dtype=np.float32,
    )
    env = TransformObservation(
        env,
        lambda obs: (obs / 255.0).astype(np.float32),
        normalized_obs_space,
    )
    env = FrameStackObservation(env, stack_size=3)
    return env


def load_agent(
    path_to_actor: str,
    path_to_encoder: str,
    path_to_critic: str = None,
):
    """Load a trained SAC agent."""
    # Create dummy env to get action space
    env = gym.make("CarRacing-v3")
    action_space = env.action_space
    env.close()

    # Create SAC agent with same architecture as training
    agent = SAC(
        action_space,
        policy="Gaussian",
        gamma=0.99,
        lr=0.0001,
        alpha=0.2,
        automatic_temperature_tuning=True,
        batch_size=256,
        hidden_size=512,
        target_update_interval=1,
        input_dim=32,
        in_channels=3,
    )

    # Load model weights
    agent.load_model(path_to_actor, path_to_critic, path_to_encoder)
    print(f"Loaded model from {path_to_actor} and {path_to_encoder}")

    return agent


def play_episodes(
    agent: SAC,
    num_episodes: int = 3,
    render_mode: str = "human",
    deterministic: bool = True,
):
    """Watch the agent play."""
    env = make_env(render_mode=render_mode)

    total_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")

        while not done:
            # Select action (eval=True for deterministic/mean action)
            action = agent.select_action(state, eval=deterministic)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            step_count += 1

            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"  Steps: {step_count}, Reward: {episode_reward:.2f}")

        total_rewards.append(episode_reward)
        print(
            f"Episode {ep + 1} finished: Reward = {episode_reward:.2f}, Steps = {step_count}"
        )

    env.close()

    # Summary
    print(f"\n{'='*50}")
    print(f"Results over {num_episodes} episodes:")
    print(f"  Mean reward: {np.mean(total_rewards):.2f}")
    print(f"  Std reward:  {np.std(total_rewards):.2f}")
    print(f"  Min reward:  {np.min(total_rewards):.2f}")
    print(f"  Max reward:  {np.max(total_rewards):.2f}")
    print(f"{'='*50}")

    return total_rewards


def main():
    parser = argparse.ArgumentParser(
        description="Load and watch a trained SAC agent play CarRacing"
    )
    parser.add_argument(
        "--actor",
        type=str,
        default="models/sac_actor_carracer_best.pt",
        help="Path to actor model",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="models/sac_encoder_carracer_best.pt",
        help="Path to encoder model",
    )
    parser.add_argument(
        "--critic",
        type=str,
        default=None,
        help="Path to critic model (optional)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to play",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (for benchmarking)",
    )

    args = parser.parse_args()

    # Load agent
    agent = load_agent(args.actor, args.encoder, args.critic)

    # Play episodes
    render_mode = None if args.no_render else "human"
    play_episodes(
        agent,
        num_episodes=args.episodes,
        render_mode=render_mode,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
