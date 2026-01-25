"""Pretrain SAC actor using behavioral cloning on recorded demonstrations.

This script loads recorded human demonstrations and pretrains the SAC actor
to imitate the human policy before starting RL training.

Usage:
    acrl ac train-bc --data-dir datasets/demonstrations --epochs 250 --batch-size 64 --vae-checkpoint loss=0.1031.ckpt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import click
import wandb
from tensordict import TensorDict
from typing import Tuple

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from assetto_corsa_rl.model.sac import SACPolicy  # type: ignore
from assetto_corsa_rl.ac_env import create_mock_env, get_device  # type: ignore

try:
    from assetto_corsa_rl.cli_registry import cli_command, cli_option  # type: ignore
except Exception:
    from ...src.assetto_corsa_rl.cli_registry import cli_command, cli_option


class DemonstrationDataset(Dataset):
    """Dataset for loading recorded demonstrations."""

    def __init__(
        self,
        data_dir: Path,
        image_shape: Tuple[int, int] = (84, 84),
        frame_stack: int = 4,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.image_shape = image_shape
        self.frame_stack = frame_stack
        self.augment = augment

        if self.augment:
            self.transform = T.Compose(
                [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)]
            )
        else:
            self.transform = None

        self.batch_files = sorted(self.data_dir.glob("demo_batch_*.npz"))
        if len(self.batch_files) == 0:
            raise RuntimeError(f"No demonstration files found in {data_dir}")

        self.frames = []
        self.actions = []
        self.observations = []  # store observations if available
        self.has_observations = False
        self.observation_keys = None

        print(f"Loading {len(self.batch_files)} demonstration batches...")
        for batch_file in self.batch_files:
            try:
                data = np.load(batch_file, allow_pickle=True)
                if "frames" not in data or "actions" not in data:
                    raise ValueError(f"Missing required keys in {batch_file}")
                self.frames.append(data["frames"])
                self.actions.append(data["actions"])

                if "observations" in data:
                    self.observations.append(data["observations"])
                    if not self.has_observations and "observation_keys" in data:
                        self.observation_keys = data["observation_keys"].tolist()
                    self.has_observations = True
            except Exception as e:
                raise RuntimeError(f"Failed to load {batch_file}: {e}")

        self.frames = np.concatenate(self.frames, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)

        if self.has_observations and self.observations:
            self.observations = np.concatenate(self.observations, axis=0)
            print(f"✓ Loaded {len(self.frames)} demonstration samples with observations")
            print(f"  Observations shape: {self.observations.shape}")
            if self.observation_keys:
                print(
                    f"  Observation keys ({len(self.observation_keys)}): {', '.join(self.observation_keys[:5])}..."
                )
        else:
            self.observations = None
            print(f"✓ Loaded {len(self.frames)} demonstration samples (no observations)")

        print(f"  Frames shape: {self.frames.shape}")
        print(f"  Actions shape: {self.actions.shape}")

        assert self.frames.ndim == 4, f"Expected 4D frames, got {self.frames.ndim}D"
        assert self.actions.ndim == 2, f"Expected 2D actions, got {self.actions.ndim}D"
        print(f"  Action dim: {self.actions.shape[1]}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # frames: (N, H, W) uint8 -> (N, H, W) float32 [0, 1]
        frames = self.frames[idx].astype(np.float32) / 255.0
        actions = self.actions[idx].astype(np.float32)

        frames_tensor = torch.from_numpy(frames)

        # Apply augmentation if enabled (apply to all frames in stack)
        if self.augment and self.transform is not None:
            # Stack frames temporarily for augmentation
            # frames_tensor is (frame_stack, H, W), need (1, frame_stack, H, W) for transforms
            frames_stacked = frames_tensor.unsqueeze(0)  # (1, frame_stack, H, W)
            frames_stacked = self.transform(frames_stacked)
            frames_tensor = frames_stacked.squeeze(0)  # (frame_stack, H, W)

        result = {"frames": frames_tensor, "actions": torch.from_numpy(actions)}

        if self.has_observations and self.observations is not None:
            observations = self.observations[idx].astype(np.float32)
            result["observations"] = torch.from_numpy(observations)

        return result


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
        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            # Handle both old tuple format and new dict format
            if isinstance(batch, dict):
                frames = batch["frames"].to(self.device)
                actions = batch["actions"].to(self.device)
                # observations available but not used in BC training (vision-based)
            else:
                frames, actions = batch
                frames = frames.to(self.device)
                actions = actions.to(self.device)

            batch_size = frames.shape[0]

            # Forward pass through actor
            td_in = TensorDict({"pixels": frames}, batch_size=[batch_size])
            td_out = self.actor(td_in)

            # Get predicted action distribution
            pred_loc = td_out["loc"]  # (B, action_dim)

            # Use MSE loss between predicted mean and target actions
            loss = F.mse_loss(pred_loc, actions)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples

        return {
            "loss": avg_loss,
        }

    def validate(self, dataloader: DataLoader) -> dict:
        """Validate on held-out data."""
        self.actor.eval()

        total_mse = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                # Handle both old tuple format and new dict format
                if isinstance(batch, dict):
                    frames = batch["frames"].to(self.device)
                    actions = batch["actions"].to(self.device)
                else:
                    frames, actions = batch
                    frames = frames.to(self.device)
                    actions = actions.to(self.device)

                batch_size = frames.shape[0]

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
@cli_option("--augment", is_flag=True, help="Enable data augmentation")
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
    augment,
    vae_checkpoint,
    wandb_project,
    wandb_offline,
):
    # Convert paths
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    vae_checkpoint = Path(vae_checkpoint) if vae_checkpoint else None

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Initialize WandB
    wandb_mode = "offline" if wandb_offline else "online"
    try:
        wandb.init(
            project=wandb_project,
            config={
                "data_dir": str(data_dir),
                "output_path": str(output_path),
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "val_split": val_split,
                "num_workers": num_workers,
                "num_cells": num_cells,
                "augment": augment,
                "vae_checkpoint": str(vae_checkpoint) if vae_checkpoint else None,
                "wandb_project": wandb_project,
            },
            mode=wandb_mode,
        )
    except Exception as e:
        print(f"Warning: WandB initialization failed: {e}")
        print("Continuing without WandB logging...")

    # Image configuration
    IMAGE_HEIGHT = 84
    IMAGE_WIDTH = 84
    FRAME_STACK = 4

    # Load dataset
    dataset = DemonstrationDataset(
        data_dir,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
        frame_stack=FRAME_STACK,
        augment=augment,
    )

    # Split into train/val
    n_samples = len(dataset)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    print(f"Train samples: {n_train}, Val samples: {n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create actor
    mock_env = create_mock_env(device)

    vae_path = str(vae_checkpoint) if vae_checkpoint else None
    agent = SACPolicy(
        env=mock_env,
        num_cells=num_cells,
        device=device,
        use_noisy=False,
        vae_checkpoint_path=vae_path,
    )

    # Initialize lazy layers with a dummy forward pass
    dummy_frames = torch.zeros(1, FRAME_STACK, IMAGE_HEIGHT, IMAGE_WIDTH, device=device)

    with torch.no_grad():
        init_td = TensorDict({"pixels": dummy_frames}, batch_size=[1])
        agent.actor(init_td)

    actor = agent.actor
    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")

    # Create trainer
    trainer = BehavioralCloningTrainer(
        actor=actor,
        device=device,
        lr=lr,
    )

    # Training loop
    best_val_mse = float("inf")
    patience_counter = 0
    EARLY_STOP_PATIENCE = 10
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("Starting Behavioral Cloning Training")
    print("=" * 50)

    for epoch in range(epochs):
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
            "val/mse": val_metrics["val_mse"],
            "lr": trainer.optimizer.param_groups[0]["lr"],
        }
        try:
            wandb.log(metrics)
        except:
            pass  # Continue if wandb logging fails

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val MSE: {val_metrics['val_mse']:.6f}"
        )

        # Save best model
        if val_metrics["val_mse"] < best_val_mse:
            best_val_mse = val_metrics["val_mse"]
            patience_counter = 0
            try:
                torch.save(
                    {
                        "actor_state": actor.state_dict(),
                        "epoch": epoch,
                        "val_mse": best_val_mse,
                        "config": {
                            "num_cells": num_cells,
                            "vae_checkpoint_path": vae_path,
                        },
                    },
                    output_path,
                )
                print(f"  ✓ Saved best model (val_mse: {best_val_mse:.6f})")
            except Exception as e:
                print(f"  ✗ Failed to save model: {e}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

    # Final save
    final_path = output_path.with_stem(output_path.stem + "_final")
    try:
        torch.save(
            {
                "actor_state": actor.state_dict(),
                "epoch": epoch,
                "val_mse": val_metrics["val_mse"],
                "config": {
                    "num_cells": num_cells,
                    "vae_checkpoint_path": vae_path,
                },
            },
            final_path,
        )
    except Exception as e:
        print(f"Failed to save final model: {e}")

    try:
        wandb.finish()
    except:
        pass

    print("\n" + "=" * 50)
    print("Behavioral Cloning Training Complete!")
    print(f"Best Val MSE: {best_val_mse:.6f}")
    print(f"Best model saved to: {output_path}")
    print(f"Final model saved to: {final_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
