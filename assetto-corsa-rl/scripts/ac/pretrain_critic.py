"""
cd scripts/ac
python pretrain_critic.py --demo-dir ../../datasets/demonstrations4  --epochs 50  --batch-size 256 --save-path ../../models/critic_pretrained.pt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Optional, Dict
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from assetto_corsa_rl.model.sac import CriticNet
from assetto_corsa_rl.ac_env import create_mock_env, get_device
from assetto_corsa_rl.model.vae import load_vae_encoder


def load_demonstrations(
    demo_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load demonstration batches and concatenate them.

    Returns:
        frames: [N, C, H, W] uint8
        actions: [N, action_dim] float32
        rewards: [N] float32
        next_frames: [N, C, H, W] uint8
        observations: [N, obs_dim] float32 (optional, may be None)
    """
    demo_dir = Path(demo_dir)
    batch_files = sorted(demo_dir.glob("demo_batch_*.npz"))

    if not batch_files:
        raise RuntimeError(f"No demonstration files found in {demo_dir}")

    print(f"Loading {len(batch_files)} demonstration batches...")

    all_frames = []
    all_actions = []
    all_rewards = []
    all_next_frames = []
    all_observations = []

    for batch_file in batch_files:
        try:
            data = np.load(batch_file, allow_pickle=True)
            all_frames.append(data["frames"])
            all_actions.append(data["actions"])
            all_rewards.append(data["rewards"])

            # Compute next_frames by rolling frames forward by 1
            # frames[i] -> next_frames should be frames[i+1]
            # We'll handle this in trajectory splitting

            if "observations" in data:
                all_observations.append(data["observations"])
        except Exception as e:
            print(f"Warning: Failed to load {batch_file}: {e}")
            continue

    frames = np.concatenate(all_frames, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    observations = np.concatenate(all_observations, axis=0) if all_observations else None

    # Compute next_frames by shifting frames forward
    # For last frame in each batch, we don't have a valid next frame
    # For now, we'll create next_frames by rolling, and we'll filter invalid ones later
    next_frames = np.roll(frames, -1, axis=0)
    # Mark the last frame of each batch as invalid (we'll filter these out)

    print(f"Loaded demonstrations:")
    print(f"  Frames shape: {frames.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}")
    if observations is not None:
        print(f"  Observations shape: {observations.shape}")

    return frames, actions, rewards, next_frames, observations


def compute_discounted_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Compute discounted returns for each timestep.

    Returns G_t = sum_{k=0}^{inf} gamma^k * r_{t+k}
    """
    n = len(rewards)
    returns = np.zeros_like(rewards, dtype=np.float32)

    # We'll use a forward pass to compute returns
    # But we need to know episode boundaries to reset the accumulator
    # For now, assume all transitions are in one long trajectory
    # (This is typically OK if demos are from the same track/session)

    returns[-1] = rewards[-1]
    for t in range(n - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]

    return returns


def create_critic_networks(
    device: torch.device,
    cnn: nn.Module,
    fusion_size: int,
    hidden_dim: int,
    obs_dim: int = 0,
) -> Tuple[CriticNet, CriticNet]:
    """Create Q1 and Q2 networks."""
    q1 = CriticNet(
        encoder=cnn,
        critic_input_size=fusion_size + 3,  # +3 for action_dim (steer, gas, brake)
        hidden=hidden_dim,
        device=device,
        obs_dim=obs_dim,
    )
    q2 = CriticNet(
        encoder=cnn,
        critic_input_size=fusion_size + 3,
        hidden=hidden_dim,
        device=device,
        obs_dim=obs_dim,
    )
    return q1, q2


def pretrain_critic(
    q1: CriticNet,
    q2: CriticNet,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    clip_grad_norm: float = 10.0,
) -> Dict[str, float]:
    """Pretrain critics on demonstration data."""

    q1.to(device)
    q2.to(device)

    optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=learning_rate)

    mse_loss = nn.MSELoss()

    all_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (pixels, actions, returns, next_pixels, observations) in enumerate(
            dataloader
        ):
            pixels = pixels.to(device)
            actions = actions.to(device).float()
            returns = returns.to(device).float().unsqueeze(-1)  # [B, 1]
            next_pixels = next_pixels.to(device)

            if observations is not None:
                observations = observations.to(device).float()

            # Unpack pixel values from uint8 to float32
            pixels = pixels.float() / 255.0
            next_pixels = next_pixels.float() / 255.0

            optimizer.zero_grad()

            # Forward pass
            q1_pred = q1(pixels, actions.unsqueeze(0).expand(pixels.shape[0], -1), observations)
            q2_pred = q2(pixels, actions.unsqueeze(0).expand(pixels.shape[0], -1), observations)

            # Compute loss
            loss_q1 = mse_loss(q1_pred, returns)
            loss_q2 = mse_loss(q2_pred, returns)
            loss = loss_q1 + loss_q2

            # Backward pass
            loss.backward()

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(q1.parameters()) + list(q2.parameters()), clip_grad_norm
                )

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        all_losses.append(avg_loss)

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

    return {
        "final_loss": all_losses[-1],
        "min_loss": min(all_losses),
        "avg_loss": np.mean(all_losses),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain critic networks on expert demonstrations"
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default="datasets/demonstrations4",
        help="Path to demonstration directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/assetto_corsa_rl/configs/ac/model_config.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default=None,
        help="Path to VAE checkpoint (overrides config if provided)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for critic optimizer")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for computing returns"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/critic_pretrained.pt",
        help="Path to save pretrained critic checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda:0, cpu, etc). If None, auto-select.",
    )

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = repo_root / args.config

    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")
        model_cfg = {}

    num_cells = model_cfg.get("num_cells", 256)
    gamma = model_cfg.get("gamma", 0.99)
    lr = model_cfg.get("critic_lr", 3e-4)
    vae_path = args.vae_checkpoint or model_cfg.get("vae_checkpoint_path", None)
    obs_dim = model_cfg.get("obs_dim", 0)

    # Override with CLI args
    gamma = args.gamma
    lr = args.lr

    print(f"Model config: num_cells={num_cells}, gamma={gamma}, critic_lr={lr}")

    # Create mock environment to get specs
    print("Creating mock environment...")
    env = create_mock_env(device=device)

    # Load or create CNN encoder
    print("Loading VAE encoder...")
    try:
        from assetto_corsa_rl.model.vae import get_encoder_from_checkpoint

        if vae_path:
            cnn = get_encoder_from_checkpoint(
                vae_path, device=device, in_channels=3, trainable=False, verbose=True
            )
            fusion_size = cnn.output_size
        else:
            print("No VAE checkpoint provided, using random CNN")
            from assetto_corsa_rl.model.vae import SimpleEncoder

            cnn = SimpleEncoder(out_dim=num_cells)
            fusion_size = num_cells
    except Exception as e:
        print(f"Warning: Failed to load VAE: {e}, using simple CNN")
        from assetto_corsa_rl.model.vae import SimpleEncoder

        cnn = SimpleEncoder(out_dim=num_cells)
        fusion_size = num_cells

    cnn.to(device)

    # Load demonstrations
    print(f"\nLoading demonstrations from {args.demo_dir}...")
    frames, actions, rewards, next_frames, observations = load_demonstrations(args.demo_dir)

    # Compute discounted returns
    print("Computing discounted returns...")
    returns = compute_discounted_returns(rewards, gamma=gamma)

    print(f"Returns: min={returns.min():.4f}, max={returns.max():.4f}, mean={returns.mean():.4f}")

    # Filter out invalid transitions (keep only those with valid next states)
    # For simplicity, we'll use all but the last transition
    valid_indices = np.arange(len(frames) - 1)

    frames = frames[valid_indices]
    next_frames = next_frames[valid_indices]
    actions = actions[valid_indices]
    returns = returns[valid_indices]
    if observations is not None:
        observations = observations[valid_indices]

    print(f"Valid transitions: {len(frames)}")

    # Create tensors
    frames_tensor = torch.from_numpy(frames).to(torch.uint8)
    next_frames_tensor = torch.from_numpy(next_frames).to(torch.uint8)
    actions_tensor = torch.from_numpy(actions).to(torch.float32)
    returns_tensor = torch.from_numpy(returns).to(torch.float32)

    if observations is not None:
        observations_tensor = torch.from_numpy(observations).to(torch.float32)
    else:
        observations_tensor = torch.zeros((len(frames), 0), dtype=torch.float32)

    # Create dataset and dataloader
    dataset = TensorDataset(
        frames_tensor,
        actions_tensor,
        returns_tensor,
        next_frames_tensor,
        observations_tensor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Create critic networks
    print("Creating critic networks...")
    q1, q2 = create_critic_networks(
        device=device,
        cnn=cnn,
        fusion_size=fusion_size,
        hidden_dim=num_cells,
        obs_dim=observations_tensor.shape[1],
    )

    # Pretrain critics
    print(f"\nPretraining critics for {args.epochs} epochs...")
    stats = pretrain_critic(
        q1=q1,
        q2=q2,
        dataloader=dataloader,
        epochs=args.epochs,
        learning_rate=lr,
        device=device,
        clip_grad_norm=10.0,
    )

    print(f"\nTraining complete!")
    print(f"  Final loss: {stats['final_loss']:.6f}")
    print(f"  Min loss:   {stats['min_loss']:.6f}")
    print(f"  Avg loss:   {stats['avg_loss']:.6f}")

    # Save checkpoint
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "q1_state": q1.state_dict(),
        "q2_state": q2.state_dict(),
        "config": {
            "num_cells": num_cells,
            "gamma": gamma,
            "vae_checkpoint_path": vae_path,
            "obs_dim": observations_tensor.shape[1],
        },
        "training_stats": stats,
    }

    torch.save(checkpoint, save_path)
    print(f"\nSaved pretrained critics to {save_path}")


if __name__ == "__main__":
    main()
