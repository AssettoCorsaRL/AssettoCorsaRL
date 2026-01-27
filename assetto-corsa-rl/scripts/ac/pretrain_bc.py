"""Pretrain SAC actor using behavioral cloning on recorded demonstrations.

This script loads recorded human demonstrations and pretrains the SAC actor
to imitate the human policy before starting RL training. CRSfD (Conservative
Reward Shaping from Demonstrations) is always enabled.

Usage:
acrl ac train-bc --data-dir datasets/demonstrations --epochs 250 --batch-size 64 --vae-checkpoint loss=0.1031.ckpt --demo-gamma 0.99 --target-gamma 0.995 --ood-weight 0.5
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
                [
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                    T.RandomAdjustSharpness(sharpness_factor=2),
                    T.ElasticTransform(),
                    T.RandomPosterize(),
                ]
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
        # (N, H, W) uint8 -> (N, H, W) float32 [0, 1]
        frames = self.frames[idx].astype(np.float32) / 255.0
        actions = self.actions[idx].astype(np.float32)

        frames_tensor = torch.from_numpy(frames)

        if self.augment and self.transform is not None:
            frames_stacked = frames_tensor.unsqueeze(0)  # (1, frame_stack, H, W)
            frames_stacked = self.transform(frames_stacked)
            frames_tensor = frames_stacked.squeeze(0)  # (frame_stack, H, W)

        result = {"frames": frames_tensor, "actions": torch.from_numpy(actions)}

        if self.has_observations and self.observations is not None:
            observations = self.observations[idx].astype(np.float32)
            result["observations"] = torch.from_numpy(observations)

        return result


class ValueFunction(nn.Module):
    """Value function network for estimating V_M0 from demonstrations.

    Used in Conservative Reward Shaping from Demonstrations (CRSfD).
    """

    def __init__(
        self, obs_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.3
    ):
        super().__init__()

        layers = []
        input_dim = obs_dim

        for i in range(num_layers):
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    (
                        nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity()
                    ),  # Dropout between layers
                ]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))  # single value

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate value for given observations.

        Args:
            obs: Observations (B, obs_dim) or frames (B, C, H, W)

        Returns:
            values: Estimated values (B, 1)
        """
        if obs.dim() > 2:
            obs = obs.flatten(start_dim=1)

        return self.network(obs)


class ConservativeValueTrainer:
    """Trainer for conservative value function estimation from demonstrations.

    Implements Algorithm 1 from CRSfD paper:
    - Monte-Carlo policy evaluation on demonstrations
    - Regress OOD states to 0 for conservativeness
    """

    def __init__(
        self,
        value_fn: nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        demo_gamma: float = 0.99,
        ood_weight: float = 0.5,
    ):
        self.value_fn = value_fn
        self.device = device
        self.demo_gamma = demo_gamma
        self.ood_weight = ood_weight

        self.optimizer = torch.optim.AdamW(value_fn.parameters(), lr=lr, weight_decay=1e-4)
        self.mc_returns = {}  # cache for Monte-Carlo returns

    def compute_mc_returns(self, dataset: DemonstrationDataset, gamma: float) -> dict:
        """Compute Monte-Carlo returns for all demonstration states.

        Args:
            dataset: Demonstration dataset
            gamma: Discount factor for computing returns

        Returns:
            Dictionary mapping indices to MC returns
        """
        print("Computing Monte-Carlo returns from demonstrations...")

        mc_returns = {}

        idx = 0
        for batch_file in dataset.batch_files:
            data = np.load(batch_file, allow_pickle=True)
            frames_batch = data["frames"]

            rewards = data["rewards"]

            returns = np.zeros(len(rewards))
            running_return = 0.0

            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + gamma * running_return
                returns[t] = running_return

            for i, ret in enumerate(returns):
                mc_returns[idx + i] = ret

            idx += len(frames_batch)

        print(f"  Computed {len(mc_returns)} MC returns")
        print(f"  Mean return: {np.mean(list(mc_returns.values())):.4f}")
        print(f"  Max return: {np.max(list(mc_returns.values())):.4f}")

        self.mc_returns = mc_returns
        return mc_returns

    def sample_ood_states(self, batch_size: int, obs_shape: tuple) -> torch.Tensor:
        """Sample out-of-distribution states from observation space.

        For image observations, sample random pixel values.

        Args:
            batch_size: Number of OOD samples
            obs_shape: Shape of observations (C, H, W)

        Returns:
            Random observations
        """
        ood_states = torch.rand(batch_size, *obs_shape, device=self.device)
        return ood_states

    def train_step(
        self,
        demo_batch: dict,
        demo_indices: list,
        obs_shape: tuple,
    ) -> dict:
        """Perform one training step of conservative value function.

        Args:
            demo_batch: Batch from demonstration dataset
            demo_indices: Original indices in dataset
            obs_shape: Shape of observations

        Returns:
            Training metrics
        """
        demo_obs = demo_batch["frames"].to(self.device)
        batch_size = demo_obs.shape[0]

        demo_returns = torch.tensor(
            [self.mc_returns.get(idx.item(), 0.0) for idx in demo_indices],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        demo_values = self.value_fn(demo_obs)

        # Loss 1: regress demo states to MC returns
        demo_loss = F.mse_loss(demo_values, demo_returns)

        # Loss 2: regress OOD states to 0 (conservative)
        ood_obs = self.sample_ood_states(batch_size, obs_shape)
        ood_values = self.value_fn(ood_obs)
        ood_targets = torch.zeros_like(ood_values)
        ood_loss = F.mse_loss(ood_values, ood_targets)

        total_loss = demo_loss + self.ood_weight * ood_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_fn.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "value_loss": total_loss.item(),
            "demo_loss": demo_loss.item(),
            "ood_loss": ood_loss.item(),
            "mean_demo_value": demo_values.mean().item(),
            "mean_ood_value": ood_values.mean().item(),
        }


class BehavioralCloningTrainer:
    """Trainer for behavioral cloning pretraining."""

    def __init__(
        self,
        actor: nn.Module,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,  # Increased from 1e-5 to reduce overfitting
    ):
        self.actor = actor
        self.device = device

        # only optimize actor parameters
        self.optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

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

            pred_loc = td_out["loc"]  # (B, action_dim)

            loss = F.mse_loss(pred_loc, actions)

            self.optimizer.zero_grad()
            loss.backward()

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
@cli_option("--dropout", default=0.0, type=float, help="Dropout probability for actor MLP")
@cli_option(
    "--vae-checkpoint",
    type=click.Path(exists=True),
    default=None,
    help="VAE checkpoint path",
)
@cli_option("--wandb-project", default="AssetoCorsaRL-BC", help="WandB project name")
@cli_option("--wandb-offline", is_flag=True, help="Run WandB offline")
@cli_option(
    "--demo-gamma",
    default=0.99,
    type=float,
    help="Discount factor for demo value function (gamma_0)",
)
@cli_option(
    "--target-gamma",
    default=0.995,
    type=float,
    help="Discount factor for target task (gamma_k > gamma_0)",
)
@cli_option("--ood-weight", default=0.5, type=float, help="Weight for OOD regression loss (lambda)")
@cli_option("--value-hidden-dim", default=256, help="Hidden dimension for value function network")
@cli_option("--value-lr", default=3e-4, type=float, help="Learning rate for value function")
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
    dropout,
    vae_checkpoint,
    wandb_project,
    wandb_offline,
    demo_gamma,
    target_gamma,
    ood_weight,
    value_hidden_dim,
    value_lr,
):
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    vae_checkpoint = Path(vae_checkpoint) if vae_checkpoint else None

    device = get_device()
    print(f"Using device: {device}")

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
                "dropout": dropout,
                "vae_checkpoint": str(vae_checkpoint) if vae_checkpoint else None,
                "wandb_project": wandb_project,
                "demo_gamma": demo_gamma,
                "target_gamma": target_gamma,
                "ood_weight": ood_weight,
                "value_hidden_dim": value_hidden_dim,
                "value_lr": value_lr,
            },
            mode=wandb_mode,
        )
    except Exception as e:
        print(f"Warning: WandB initialization failed: {e}")
        print("Continuing without WandB logging...")

    IMAGE_HEIGHT = 84
    IMAGE_WIDTH = 84
    FRAME_STACK = 4

    dataset = DemonstrationDataset(
        data_dir,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
        frame_stack=FRAME_STACK,
        augment=augment,
    )

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

    mock_env = create_mock_env(device)

    vae_path = str(vae_checkpoint) if vae_checkpoint else None
    agent = SACPolicy(
        env=mock_env,
        num_cells=num_cells,
        device=device,
        use_noisy=False,
        actor_dropout=dropout,
        vae_checkpoint_path=vae_path,
    )

    dummy_frames = torch.zeros(1, FRAME_STACK, IMAGE_HEIGHT, IMAGE_WIDTH, device=device)

    with torch.no_grad():
        init_td = TensorDict({"pixels": dummy_frames}, batch_size=[1])
        agent.actor(init_td)

    actor = agent.actor
    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")

    trainer = BehavioralCloningTrainer(
        actor=actor,
        device=device,
        lr=lr,
    )

    # CRSfD is always enabled
    print("\n" + "=" * 50)
    print("Initializing Conservative Reward Shaping from Demonstrations (CRSfD)")
    print("=" * 50)

    obs_dim = FRAME_STACK * IMAGE_HEIGHT * IMAGE_WIDTH
    value_fn = ValueFunction(
        obs_dim=obs_dim,
        hidden_dim=value_hidden_dim,
        num_layers=3,
        dropout=0.3,
    ).to(device)

    print(f"Value function parameters: {sum(p.numel() for p in value_fn.parameters()):,}")

    value_trainer = ConservativeValueTrainer(
        value_fn=value_fn,
        device=device,
        lr=value_lr,
        demo_gamma=demo_gamma,
        ood_weight=ood_weight,
    )

    # Monte-Carlo returns from demonstrations
    value_trainer.compute_mc_returns(dataset, gamma=demo_gamma)

    print(f"Demo discount factor (γ₀): {demo_gamma}")
    print(f"Target discount factor (γₖ): {target_gamma}")
    print(f"OOD regression weight (λ): {ood_weight}")
    print("=" * 50 + "\n")

    best_val_mse = float("inf")
    patience_counter = 0
    EARLY_STOP_PATIENCE = 10
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("Starting Behavioral Cloning Training")
    print("=" * 50)

    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # (CRSfD)
        value_metrics = {
            "value_loss": 0.0,
            "demo_loss": 0.0,
            "ood_loss": 0.0,
            "mean_demo_value": 0.0,
            "mean_ood_value": 0.0,
        }

        for batch_idx, batch in enumerate(train_loader):
            indices = torch.arange(
                batch_idx * batch_size, min((batch_idx + 1) * batch_size, n_train)
            )

            obs_shape = (FRAME_STACK, IMAGE_HEIGHT, IMAGE_WIDTH)
            step_metrics = value_trainer.train_step(
                batch,
                indices,
                obs_shape,
            )

            for k, v in step_metrics.items():
                value_metrics[k] += v

        num_batches = len(train_loader)
        for k in value_metrics:
            value_metrics[k] /= num_batches

        val_metrics = trainer.validate(val_loader)

        trainer.step_scheduler(val_metrics["val_mse"])

        metrics = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "val/mse": val_metrics["val_mse"],
            "lr": trainer.optimizer.param_groups[0]["lr"],
            "train/value_loss": value_metrics["value_loss"],
            "train/demo_loss": value_metrics["demo_loss"],
            "train/ood_loss": value_metrics["ood_loss"],
        }

        try:
            wandb.log(metrics)
        except:
            pass

        log_str = (
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val MSE: {val_metrics['val_mse']:.6f} | "
            f"Value Loss: {value_metrics['value_loss']:.6f}"
        )

        print(log_str)

        if val_metrics["val_mse"] < best_val_mse:
            best_val_mse = val_metrics["val_mse"]
            patience_counter = 0
            try:
                checkpoint = {
                    "actor_state": actor.state_dict(),
                    "value_fn_state": value_fn.state_dict(),
                    "epoch": epoch,
                    "val_mse": best_val_mse,
                    "config": {
                        "num_cells": num_cells,
                        "vae_checkpoint_path": vae_path,
                        "demo_gamma": demo_gamma,
                        "target_gamma": target_gamma,
                        "value_hidden_dim": value_hidden_dim,
                    },
                }

                torch.save(checkpoint, output_path)
                print(f"  ✓ Saved best model (val_mse: {best_val_mse:.6f})")
            except Exception as e:
                print(f"  ✗ Failed to save model: {e}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

    final_path = output_path.with_stem(output_path.stem + "_final")
    try:
        final_checkpoint = {
            "actor_state": actor.state_dict(),
            "value_fn_state": value_fn.state_dict(),
            "epoch": epoch,
            "val_mse": val_metrics["val_mse"],
            "config": {
                "num_cells": num_cells,
                "vae_checkpoint_path": vae_path,
                "demo_gamma": demo_gamma,
                "target_gamma": target_gamma,
                "value_hidden_dim": value_hidden_dim,
            },
        }

        torch.save(final_checkpoint, final_path)
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
