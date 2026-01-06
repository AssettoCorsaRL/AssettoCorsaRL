"""Load a trained VAE and visualize reconstructions in real-time with user control.

Usage:
    python assetto-corsa-rl/scripts/load_vae.py --model vae_pretrained.ckpt --frames 4

Controls:
    Arrow Keys: Steer, accelerate, brake
    Q: Quit
    R: Reset episode
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

matplotlib.use("TkAgg")  # Interactive backend

try:
    from assetto_corsa_rl.model.vae import ConvVAE
except Exception:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from assetto_corsa_rl.model.vae import ConvVAE

import gymnasium as gym
from torchvision import transforms as T


class KeyboardController:
    """Simple keyboard controller for CarRacing using matplotlib key events."""

    def __init__(self):
        self.action = np.array([0.0, 0.0, 0.0])  # steering, gas, brake
        self.quit = False
        self.reset = False

    def on_key_press(self, event):
        if event.key == "q":
            self.quit = True
        elif event.key == "r":
            self.reset = True
        elif event.key == "up":
            self.action[1] = 1.0  # gas
        elif event.key == "down":
            self.action[2] = 0.8  # brake
        elif event.key == "left":
            self.action[0] = -1.0  # steer left
        elif event.key == "right":
            self.action[0] = 1.0  # steer right

    def on_key_release(self, event):
        if event.key == "up":
            self.action[1] = 0.0
        elif event.key == "down":
            self.action[2] = 0.0
        elif event.key in ("left", "right"):
            self.action[0] = 0.0

    def get_action(self):
        """Return current action."""
        return self.action.copy()


def load_vae_model(
    checkpoint_path: str, z_dim: int, in_channels: int, device: torch.device
):
    """Load VAE model from checkpoint."""
    model = ConvVAE(z_dim=z_dim, in_channels=in_channels)

    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print(f"✓ Loaded VAE from {checkpoint_path}")
    return model


def run_visualization(
    model_path: str,
    z_dim: int = 32,
    frames: int = 4,
    device: str = None,
    max_steps: int = 10000,
):
    """Run interactive visualization with VAE reconstruction."""
    device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Load VAE model
    in_channels = 3 * frames
    vae = load_vae_model(model_path, z_dim, in_channels, device)

    # Create environment
    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    # Setup transforms
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
        ]
    )

    # Setup keyboard controller
    controller = KeyboardController()

    # Setup matplotlib figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.canvas.mpl_connect("key_press_event", controller.on_key_press)
    fig.canvas.mpl_connect("key_release_event", controller.on_key_release)

    ax1.set_title("Environment (96x96)", fontsize=12, fontweight="bold")
    ax2.set_title("Input to VAE (64x64)", fontsize=12, fontweight="bold")
    ax3.set_title("VAE Reconstruction (64x64)", fontsize=12, fontweight="bold")

    for ax in (ax1, ax2, ax3):
        ax.axis("off")

    # Add control instructions
    fig.text(
        0.5,
        0.02,
        "Controls: Arrow Keys = Drive | R = Reset | Q = Quit",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Initialize display images
    img1 = ax1.imshow(np.zeros((96, 96, 3), dtype=np.uint8))
    img2 = ax2.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
    img3 = ax3.imshow(np.zeros((64, 64, 3), dtype=np.uint8))

    # Stats text
    stats_text = fig.text(0.5, 0.95, "", ha="center", fontsize=11, fontweight="bold")

    plt.ion()
    plt.show()

    # Game loop
    obs, _ = env.reset()
    frame_buffer = deque([transform(obs).clone() for _ in range(frames)], maxlen=frames)

    total_reward = 0.0
    steps = 0
    episode = 1

    print("\n🎮 Starting interactive session...")
    print("Use arrow keys to control the car!")

    while not controller.quit and steps < max_steps:
        # Get action from keyboard
        action = controller.get_action()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Process frame
        frame_tensor = transform(obs)
        frame_buffer.append(frame_tensor)

        # Create stacked input for VAE
        stacked = torch.cat(list(frame_buffer), dim=0).unsqueeze(0).to(device)

        # Get VAE reconstruction
        with torch.no_grad():
            recon, _, _ = vae(stacked)

        # Convert to numpy for display
        env_img = obs  # 96x96x3
        input_img = frame_tensor.permute(1, 2, 0).cpu().numpy()  # 64x64x3
        recon_img = recon[0].permute(1, 2, 0).cpu().numpy()  # 64x64x3

        # Update displays
        img1.set_data(env_img)
        img2.set_data(input_img)
        img3.set_data(recon_img)

        # Update stats
        stats_text.set_text(
            f"Episode: {episode} | Step: {steps} | Reward: {total_reward:.1f} | "
            f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]"
        )

        # Refresh display
        plt.pause(0.001)

        # Handle reset
        if terminated or truncated or controller.reset:
            print(
                f"Episode {episode} finished: {steps} steps, reward={total_reward:.2f}"
            )
            obs, _ = env.reset()
            frame_buffer = deque(
                [transform(obs).clone() for _ in range(frames)], maxlen=frames
            )
            total_reward = 0.0
            steps = 0
            episode += 1
            controller.reset = False

    print(f"\n✓ Session ended after {episode-1} episodes")
    env.close()
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize VAE reconstructions with interactive control"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained VAE checkpoint (.pth)"
    )
    parser.add_argument("--z-dim", type=int, default=32, help="Latent dimension of VAE")
    parser.add_argument(
        "--frames", type=int, default=4, help="Number of stacked frames"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--max-steps", type=int, default=10000, help="Maximum total steps"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_visualization(
        model_path=args.model,
        z_dim=args.z_dim,
        frames=args.frames,
        device=args.device,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
