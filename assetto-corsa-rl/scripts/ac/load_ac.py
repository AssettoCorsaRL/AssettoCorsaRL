"""Load a trained SAC agent and test it in Assetto Corsa environment.

Usage:
    python assetto-corsa-rl/scripts/ac/load_ac.py --ckpt models/bc_pretrained.pt --episodes 5
"""

from __future__ import annotations
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from tensordict import TensorDict

repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from assetto_corsa_rl.ac_env import AssettoCorsa, create_transformed_env, get_device
from assetto_corsa_rl.model.sac import SACPolicy

try:
    from assetto_corsa_rl.cli_registry import cli_command, cli_option
except Exception:
    from ...src.assetto_corsa_rl.cli_registry import cli_command, cli_option


def parse_args():
    parser = argparse.ArgumentParser(description="Load and test trained SAC agent")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., models/sac_checkpoint.pt)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of test episodes to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--racing-line",
        type=str,
        default="racing_lines.json",
        help="Path to racing line JSON file",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (mean of distribution)",
    )
    return parser.parse_args()


def create_env(racing_line_path: str, device):
    """Create the Assetto Corsa environment with transforms matching training."""
    return create_transformed_env(
        racing_line_path=racing_line_path,
        device=device,
        image_shape=(84, 84),
        frame_stack=4,
    )


def load_checkpoint(ckpt_path: str, device: str):
    """Load the SAC actor from checkpoint."""
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f"✓ Loaded checkpoint from {ckpt_path}")

    actor_state = None
    config = checkpoint.get("config", {})

    if "actor_state" in checkpoint:
        actor_state = checkpoint["actor_state"]
    elif "actor_state_dict" in checkpoint:
        actor_state = checkpoint["actor_state_dict"]
    elif "modules" in checkpoint and "actor" in checkpoint["modules"]:
        actor_state = checkpoint["modules"]["actor"]
    else:
        raise KeyError(
            f"Could not find actor state dict in checkpoint. Available keys: {list(checkpoint.keys())}"
        )

    return checkpoint, actor_state, config


def main():
    args = parse_args()
    device = get_device(args.device)

    print("=" * 60)
    print("Loading Trained SAC Agent for Assetto Corsa")
    print("=" * 60)

    print(f"Creating environment with racing line: {args.racing_line}")
    env = create_env(args.racing_line, device)

    checkpoint, actor_state, config = load_checkpoint(args.ckpt, args.device)

    num_cells = config.get("num_cells", 256)
    vae_checkpoint_path = config.get("vae_checkpoint_path", None)
    use_noisy = config.get("use_noisy", False)

    print(
        f"Model config: num_cells={num_cells}, vae={vae_checkpoint_path is not None}, noisy={use_noisy}"
    )

    print("Building SAC agent...")
    agent = SACPolicy(
        env=env,
        num_cells=num_cells,
        device=device,
        use_noisy=use_noisy,
        vae_checkpoint_path=vae_checkpoint_path,
    )

    print("Initializing lazy layers...")
    with torch.no_grad():
        dummy_td = env.reset()
        dummy_pixels = dummy_td.get("pixels").unsqueeze(0).to(device)
        init_td = TensorDict({"pixels": dummy_pixels}, batch_size=[1])
        agent.actor(init_td.clone())

    actor = agent.actor
    actor.load_state_dict(actor_state)
    actor.eval()
    print(f"✓ Actor loaded on {device}")

    if "step" in checkpoint:
        print(f"  Checkpoint step: {checkpoint['step']}")
    if "steps" in checkpoint:
        print(f"  Checkpoint steps: {checkpoint['steps']}")
    if "epoch" in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if "val_mse" in checkpoint:
        print(f"  Validation MSE: {checkpoint['val_mse']:.6f}")

    input("\nPress Enter when Assetto Corsa is running and ready...")

    print("\n" + "=" * 60)
    print(f"Running {args.episodes} test episodes...")
    print("=" * 60)

    episode_rewards = []
    episode_lengths = []

    try:
        for ep in range(args.episodes):
            td = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            print(f"\nEpisode {ep + 1}/{args.episodes}")

            while not done:
                pixels = td.get("pixels")
                if pixels.dim() == 3:
                    pixels = pixels.unsqueeze(0)

                actor_input = TensorDict(
                    {"pixels": pixels.to(device)}, batch_size=[pixels.shape[0]]
                )

                with torch.no_grad():
                    actor_output = actor(actor_input)

                    if args.deterministic:
                        action = actor_output.get("loc")
                    else:
                        action = actor_output.get("action")

                action_td = TensorDict(
                    {"action": action.squeeze(0).cpu()}, batch_size=[]
                )

                td = env.step(action_td)

                reward = td.get("reward", td.get(("next", "reward"), 0.0))
                done = td.get("done", td.get(("next", "done"), False))

                if isinstance(reward, torch.Tensor):
                    reward = reward.item()
                if isinstance(done, torch.Tensor):
                    done = done.item()

                episode_reward += reward
                episode_length += 1

                if episode_length % 100 == 0:
                    action_vals = action.squeeze().cpu().numpy()
                    print(
                        f"  Step {episode_length}: reward={reward:.3f}, "
                        f"action=[{action_vals[0]:.2f}, {action_vals[1]:.2f}, {action_vals[2]:.2f}]"
                    )

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            print(f"  Episode {ep + 1} complete:")
            print(f"    Total reward: {episode_reward:.2f}")
            print(f"    Episode length: {episode_length} steps")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        env.close()

    if episode_rewards:
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Episodes completed: {len(episode_rewards)}")
        print(
            f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
        )
        print(
            f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
        )
        print(f"Best episode reward: {np.max(episode_rewards):.2f}")
        print(f"Worst episode reward: {np.min(episode_rewards):.2f}")


if __name__ == "__main__":
    main()
