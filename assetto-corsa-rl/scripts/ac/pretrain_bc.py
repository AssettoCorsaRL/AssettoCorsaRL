"""Pretrain SAC actor using behavioral cloning on recorded demonstrations.

This script loads recorded human demonstrations and pretrains the SAC actor
to imitate the human policy before starting RL training.

Usage:
    python scripts/ac/pretrain_bc.py --data-dir datasets/demonstrations --epochs 50 --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import click
import wandb

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from assetto_corsa_rl.model.sac import SACPolicy
from assetto_corsa_rl.ac_env import create_mock_env, get_device

try:
    from assetto_corsa_rl.cli_registry import cli_command, cli_option
except Exception:
    from ...src.assetto_corsa_rl.cli_registry import cli_command, cli_option


class DemonstrationDataset(Dataset):
    """Dataset for loading recorded demonstrations."""

    def __init__(
        self,
        data_dir: Path,
        image_shape: Tuple[int, int] = (84, 84),
        frame_stack: int = 4,
    ):
        self.data_dir = Path(data_dir)
        self.image_shape = image_shape
        self.frame_stack = frame_stack

        # Find all batch files
        self.batch_files = sorted(self.data_dir.glob("demo_batch_*.npz"))
        if len(self.batch_files) == 0:
            raise RuntimeError(f"No demonstration files found in {data_dir}")

        # Load all data into memory (or use lazy loading for large datasets)
        self.frames = []
        self.actions = []

        print(f"Loading {len(self.batch_files)} demonstration batches...")
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            self.frames.append(data["frames"])
            self.actions.append(data["actions"])

        self.frames = np.concatenate(self.frames, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)

        print(f"✓ Loaded {len(self.frames)} demonstration samples")
        print(f"  Frames shape: {self.frames.shape}")
        print(f"  Actions shape: {self.actions.shape}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # frames: (N, H, W) uint8 -> (N, H, W) float32 [0, 1]
        frames = self.frames[idx].astype(np.float32) / 255.0
        actions = self.actions[idx].astype(np.float32)

        return torch.from_numpy(frames), torch.from_numpy(actions)


class BehavioralCloningTrainer:
    """Trainer for behavioral cloning pretraining."""

    def __init__(
        self,
        actor: nn.Module,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.actor = actor
        self.device = device

        # Only optimize actor parameters
        self.optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """Train for one epoch."""
        self.actor.train()

        total_loss = 0.0
        total_mse = 0.0
        total_samples = 0

        for batch_idx, (frames, actions) in enumerate(dataloader):
            frames = frames.to(self.device)
            actions = actions.to(self.device)

            batch_size = frames.shape[0]

            # Forward pass through actor
            from tensordict import TensorDict

            td_in = TensorDict({"pixels": frames}, batch_size=[batch_size])
            td_out = self.actor(td_in)

            # Get predicted action (mean of the distribution)
            pred_loc = td_out["loc"]  # (B, action_dim)

            # Behavioral cloning loss: MSE between predicted and demonstration actions
            # We use the mean (loc) directly for BC, not sampling
            mse_loss = F.mse_loss(pred_loc, actions)

            # Additionally, we can add a regularization on the scale to encourage
            # low-variance predictions during BC
            pred_scale = td_out["scale"]
            scale_reg = pred_scale.mean() * 0.01  # Small regularization

            loss = mse_loss + scale_reg

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_mse += mse_loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_mse = total_mse / total_samples

        return {
            "loss": avg_loss,
            "mse": avg_mse,
        }

    def validate(self, dataloader: DataLoader) -> dict:
        """Validate on held-out data."""
        self.actor.eval()

        total_mse = 0.0
        total_samples = 0

        with torch.no_grad():
            for frames, actions in dataloader:
                frames = frames.to(self.device)
                actions = actions.to(self.device)

                batch_size = frames.shape[0]

                from tensordict import TensorDict

                td_in = TensorDict({"pixels": frames}, batch_size=[batch_size])
                td_out = self.actor(td_in)

                pred_loc = td_out["loc"]
                mse = F.mse_loss(pred_loc, actions)

                total_mse += mse.item() * batch_size
                total_samples += batch_size

        avg_mse = total_mse / total_samples

        return {
            "val_mse": avg_mse,
        }

    def step_scheduler(self, val_loss: float):
        """Step the learning rate scheduler."""
        self.scheduler.step(val_loss)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain SAC actor with behavioral cloning")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing demonstration data",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/bc_pretrained.pt"),
        help="Path to save pretrained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--num-cells",
        type=int,
        default=256,
        help="Hidden layer size",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=None,
        help="Path to VAE checkpoint for encoder",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="AssetoCorsaRL-BC",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run WandB in offline mode",
    )
    return parser.parse_args()


@cli_command(group="ac", name="train-bc", help="Pretrain SAC actor using behavioral cloning")
@cli_option(
    "--data-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory with demonstrations",
)
@cli_option("--output-path", default="models/bc_pretrained.pt", help="Output model path")
@cli_option("--epochs", default=50, help="Number of training epochs")
@cli_option("--batch-size", default=64, help="Batch size")
@cli_option("--lr", default=1e-4, type=float, help="Learning rate")
@cli_option("--val-split", default=0.1, type=float, help="Validation split ratio")
@cli_option("--num-workers", default=4, help="Data loader workers")
@cli_option("--num-cells", default=256, help="Hidden layer size")
@cli_option(
    "--vae-checkpoint",
    type=click.Path(exists=True),
    default=None,
    help="VAE checkpoint path",
)
@cli_option("--wandb-project", default="AssetoCorsaRL-BC", help="WandB project name")
@cli_option("--wandb-offline", is_flag=True, help="Run WandB offline")
def main(
    data_dir,
    output_path,
    epochs,
    batch_size,
    lr,
    val_split,
    num_workers,
    num_cells,
    vae_checkpoint,
    wandb_project,
    wandb_offline,
):
    args = parse_args()

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Initialize WandB
    wandb_mode = "offline" if args.wandb_offline else "online"
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        mode=wandb_mode,
    )

    # Load dataset
    dataset = DemonstrationDataset(args.data_dir)

    # Split into train/val
    n_samples = len(dataset)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    print(f"Train samples: {n_train}, Val samples: {n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create actor
    mock_env = create_mock_env(device)

    vae_path = str(args.vae_checkpoint) if args.vae_checkpoint else None
    agent = SACPolicy(
        env=mock_env,
        num_cells=args.num_cells,
        device=device,
        use_noisy=False,
        vae_checkpoint_path=vae_path,
    )

    # Initialize lazy layers with a dummy forward pass
    dummy_frames = torch.zeros(1, 4, 84, 84, device=device)
    from tensordict import TensorDict

    with torch.no_grad():
        init_td = TensorDict({"pixels": dummy_frames}, batch_size=[1])
        agent.actor(init_td)

    actor = agent.actor
    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")

    # Create trainer
    trainer = BehavioralCloningTrainer(
        actor=actor,
        device=device,
        lr=args.lr,
    )

    # Training loop
    best_val_mse = float("inf")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("Starting Behavioral Cloning Training")
    print("=" * 50)

    for epoch in range(args.epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_metrics = trainer.validate(val_loader)

        # Update scheduler
        trainer.step_scheduler(val_metrics["val_mse"])

        # Logging
        metrics = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/mse": train_metrics["mse"],
            "val/mse": val_metrics["val_mse"],
            "lr": trainer.optimizer.param_groups[0]["lr"],
        }
        wandb.log(metrics)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Train MSE: {train_metrics['mse']:.6f} | "
            f"Val MSE: {val_metrics['val_mse']:.6f}"
        )

        # Save best model
        if val_metrics["val_mse"] < best_val_mse:
            best_val_mse = val_metrics["val_mse"]
            torch.save(
                {
                    "actor_state": actor.state_dict(),
                    "epoch": epoch,
                    "val_mse": best_val_mse,
                    "config": {
                        "num_cells": args.num_cells,
                        "vae_checkpoint_path": vae_path,
                    },
                },
                args.output_path,
            )
            print(f"  ✓ Saved best model (val_mse: {best_val_mse:.6f})")

    # Final save
    final_path = args.output_path.with_stem(args.output_path.stem + "_final")
    torch.save(
        {
            "actor_state": actor.state_dict(),
            "epoch": args.epochs - 1,
            "val_mse": val_metrics["val_mse"],
            "config": {
                "num_cells": args.num_cells,
                "vae_checkpoint_path": vae_path,
            },
        },
        final_path,
    )

    wandb.finish()

    print("\n" + "=" * 50)
    print("Behavioral Cloning Training Complete!")
    print(f"Best Val MSE: {best_val_mse:.6f}")
    print(f"Best model saved to: {args.output_path}")
    print(f"Final model saved to: {final_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
