"""Train SAC policy using BC-SAC: Behavioral Cloning + Soft Actor-Critic.

BC-SAC from "Imitation Is Not Enough: Robustifying Imitation with Reinforcement Learning for Challenging Driving Scenarios" (Waymo, 2023).

- Actor objective: E[Q(s,a) + H(π(·|s))] + λ * E[log π(a|s)]
- Critic objective: Standard SAC Bellman error

Usage:
acrl ac train-bc --data-dir datasets/demonstrations2 --epochs 250 --batch-size 64 --vae-checkpoint loss=0.1050.ckpt --bc-weight 1.0 --dropout 0.3 --augment
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tensordict import TensorDict

import click
import wandb

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
                    T.RandomApply(
                        [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)],
                        p=0.5,
                    ),
                    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.3),
                    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
                ]
            )
        else:
            self.transform = None

        self.batch_files = sorted(self.data_dir.glob("demo_batch_*.npz"))
        if len(self.batch_files) == 0:
            raise RuntimeError(f"No demonstration files found in {data_dir}")

        self.frames = []
        self.actions = []
        self.rewards = []
        self.observations = []
        self.observation_keys = None

        print(f"Loading {len(self.batch_files)} demonstration batches...")
        for batch_file in self.batch_files:
            try:
                data = np.load(batch_file, allow_pickle=True)
                self.frames.append(data["frames"])
                self.actions.append(data["actions"])
                self.rewards.append(data["rewards"])
                self.observations.append(data["observations"])
                if self.observation_keys is None and "observation_keys" in data:
                    self.observation_keys = data["observation_keys"].tolist()
            except Exception as e:
                raise RuntimeError(f"Failed to load {batch_file}: {e}")

        self.frames = np.concatenate(self.frames, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        self.observations = np.concatenate(self.observations, axis=0)

        print(
            f"  Rewards: min={self.rewards.min():.4f}, max={self.rewards.max():.4f}, mean={self.rewards.mean():.4f}"
        )

        self.obs_min = self.observations.min(axis=0, keepdims=True).astype(np.float32)
        self.obs_max = self.observations.max(axis=0, keepdims=True).astype(np.float32)
        self.obs_mean = self.observations.mean(axis=0, keepdims=True).astype(np.float32)
        self.obs_range = self.obs_max - self.obs_min
        self.obs_range = np.where(self.obs_range < 1e-6, 1.0, self.obs_range)

        self.obs_normalizations_values = {}
        print(f"  {'Key':<25} {'Min':>12} {'Max':>12} {'Mean':>12} {'Range':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for i, key in enumerate(self.observation_keys):
            min_val = float(self.obs_min[0, i])
            max_val = float(self.obs_max[0, i])
            mean_val = float(self.obs_mean[0, i])
            range_val = float(self.obs_range[0, i])
            key_short = key[:24] if len(key) > 24 else key
            self.obs_normalizations_values[key] = (min_val, range_val)
            print(
                f"  {key_short:<25} {min_val:>12.4f} {max_val:>12.4f} {mean_val:>12.4f} {range_val:>12.4f}"
            )

        print(f"  Frames shape: {self.frames.shape}")
        print(f"  Actions shape: {self.actions.shape}")

        assert self.frames.ndim == 4, f"Expected 4D frames, got {self.frames.ndim}D"
        assert self.actions.ndim == 2, f"Expected 2D actions, got {self.actions.ndim}D"
        print(f"  Action dim: {self.actions.shape[1]}")

        steering = self.actions[:, 0]
        throttle = self.actions[:, 1]
        brake = self.actions[:, 2]
        print(f"\n  Action Statistics:")
        print(
            f"    Steering: mean={steering.mean():.4f}, std={steering.std():.4f}, min={steering.min():.4f}, max={steering.max():.4f}"
        )
        print(
            f"    Throttle: mean={throttle.mean():.4f}, std={throttle.std():.4f}, min={throttle.min():.4f}, max={throttle.max():.4f}"
        )
        print(
            f"    Brake:    mean={brake.mean():.4f}, std={brake.std():.4f}, min={brake.min():.4f}, max={brake.max():.4f}"
        )

        left_turns = (steering < -0.03).sum()
        right_turns = (steering > 0.03).sum()
        straight = ((steering >= -0.03) & (steering <= 0.03)).sum()
        print(f"    Left turns: {left_turns} ({100*left_turns/len(steering):.1f}%)")
        print(f"    Right turns: {right_turns} ({100*right_turns/len(steering):.1f}%)")
        print(f"    Straight: {straight} ({100*straight/len(steering):.1f}%)")

        # build valid indices for consecutive frame pairs (s, a, s')
        # exclude last frame of each batch to avoid cross-batch transitions
        self.valid_indices = list(range(len(self.frames) - 1))

    def __len__(self):
        return len(self.valid_indices)

    def set_augment(self, augment: bool):
        """Enable or disable augmentation."""
        self.augment = augment

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        next_idx = actual_idx + 1

        # (N, H, W) uint8 -> (N, H, W) float32 [0, 1]
        frames = self.frames[actual_idx].astype(np.float32) / 255.0
        next_frames = self.frames[next_idx].astype(np.float32) / 255.0
        actions = self.actions[actual_idx].astype(np.float32).copy()

        frames_tensor = torch.from_numpy(frames)
        next_frames_tensor = torch.from_numpy(next_frames)

        # apply color augmentation to each frame individually (before stacking)
        if self.augment and self.transform is not None:
            augmented_frames = []
            for i in range(frames_tensor.shape[0]):
                frame = frames_tensor[i : i + 1]  # (1, H, W)
                frame = self.transform(frame)
                augmented_frames.append(frame)
            frames_tensor = torch.cat(augmented_frames, dim=0)  # (frame_stack, H, W)

            augmented_next_frames = []
            for i in range(next_frames_tensor.shape[0]):
                frame = next_frames_tensor[i : i + 1]  # (1, H, W)
                frame = self.transform(frame)
                augmented_next_frames.append(frame)
            next_frames_tensor = torch.cat(augmented_next_frames, dim=0)

        # horizontal flip (50% chance) to balance left/right turns
        #! HIGHLY EXPIREMENTAL
        if self.augment and np.random.rand() < 0.5:
            frames_tensor = torch.flip(frames_tensor, dims=[-1])  # flip width
            next_frames_tensor = torch.flip(next_frames_tensor, dims=[-1])
            actions[0] = -actions[0]  # invert steering

        result = {
            "frames": frames_tensor,
            "next_frames": next_frames_tensor,
            "actions": torch.from_numpy(actions),
            "rewards": torch.tensor(self.rewards[actual_idx].astype(np.float32)).reshape(1),
        }

        # norm observations
        obs = self.observations[actual_idx].astype(np.float32)
        normalized_obs = np.zeros_like(obs)
        for i, key in enumerate(self.observation_keys):
            min_val, range_val = self.obs_normalizations_values[key]
            normalized_obs[i] = (obs[i] - min_val) / range_val
        result["observations"] = torch.from_numpy(normalized_obs)

        return result


class BCSACTrainer:
    """BC-SAC: Combine Behavioral Cloning with Soft Actor-Critic.

    Implements the approach from Waymo's "Imitation Is Not Enough" paper.
    Trains both critics (with Bellman updates) and actor (with Q-values + BC loss).
    """

    def __init__(
        self,
        actor: nn.Module,
        q1: nn.Module,
        q2: nn.Module,
        q1_target: nn.Module,
        q2_target: nn.Module,
        device: torch.device,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        bc_weight: float = 1.0,
        gamma: float = 0.92,
        tau: float = 0.005,
        alpha: float = 0.2,
        weight_decay: float = 1e-5,
    ):
        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.q1_target = q1_target
        self.q2_target = q2_target
        self.device = device
        self.bc_weight = bc_weight
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor_optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=actor_lr,
            weight_decay=weight_decay,
        )
        self.critic_optimizer = torch.optim.AdamW(
            list(q1.parameters()) + list(q2.parameters()),
            lr=critic_lr,
            weight_decay=weight_decay,
        )

        self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self._update_target_network(tau=1.0)

    def _update_target_network(self, tau: Optional[float] = None):
        """Soft update of target network parameters."""
        if tau is None:
            tau = self.tau

        with torch.no_grad():
            for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """Train for one epoch with BC-SAC."""
        self.actor.train()
        self.q1.train()
        self.q2.train()

        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_bc_loss = 0.0
        total_q_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            frames = batch["frames"].to(self.device)
            actions = batch["actions"].to(self.device)
            next_frames = batch["next_frames"].to(self.device)
            rewards = batch["rewards"].to(self.device)
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(1)

            # dones = 0 is fine here for now cuz we dont crash into any walls when recording (hopefully)
            # TODO: save dones in record_demonstrations
            dones = torch.zeros(frames.shape[0], 1, device=self.device)

            batch_size = frames.shape[0]

            # ========== Train Critic ==========
            # get target Q-values using target network
            with torch.no_grad():
                next_td = TensorDict({"pixels": next_frames}, batch_size=[batch_size])
                next_actor_out = self.actor(next_td)
                next_actions = next_actor_out["loc"]
                next_log_probs = next_actor_out.get(
                    "log_prob", torch.zeros(batch_size, 1, device=self.device)
                )

                next_target_td = TensorDict(
                    {"pixels": next_frames, "action": next_actions}, batch_size=[batch_size]
                )
                next_q1_target = self.q1_target(next_target_td)["state_action_value"]
                next_q2_target = self.q2_target(next_target_td)["state_action_value"]
                next_q_target = torch.min(next_q1_target, next_q2_target)

                target_q = rewards + (1.0 - dones) * self.gamma * (
                    next_q_target - self.alpha * next_log_probs
                )

            current_td = TensorDict({"pixels": frames, "action": actions}, batch_size=[batch_size])
            current_q1 = self.q1(current_td)["state_action_value"]
            current_q2 = self.q2(current_td)["state_action_value"]

            # twin Q-learning
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # critic updates
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=1.0
            )
            self.critic_optimizer.step()

            # ========== Train Actor ==========
            td_in = TensorDict({"pixels": frames}, batch_size=[batch_size])
            actor_out = self.actor(td_in)
            pred_actions = actor_out["loc"]
            log_probs = actor_out.get("log_prob", torch.zeros(batch_size, 1, device=self.device))

            # BC loss: match expert actions
            bc_loss = F.mse_loss(pred_actions, actions)

            actor_td = TensorDict(
                {"pixels": frames, "action": pred_actions}, batch_size=[batch_size]
            )
            actor_q1 = self.q1(actor_td)["state_action_value"]
            actor_q2 = self.q2(actor_td)["state_action_value"]
            actor_q = torch.min(actor_q1, actor_q2)

            # RL loss: maximize Q-value + entropy
            q_loss = -(actor_q - self.alpha * log_probs).mean()

            actor_loss = self.bc_weight * bc_loss + q_loss

            # actor updates
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # ========== Soft update target network ==========
            self._update_target_network()

            total_critic_loss += critic_loss.item() * batch_size
            total_actor_loss += actor_loss.item() * batch_size
            total_bc_loss += bc_loss.item() * batch_size
            total_q_loss += q_loss.item() * batch_size
            total_samples += batch_size

        return {
            "critic_loss": total_critic_loss / total_samples,
            "actor_loss": total_actor_loss / total_samples,
            "bc_loss": total_bc_loss / total_samples,
            "q_loss": total_q_loss / total_samples,
        }

    def validate(self, dataloader: DataLoader) -> dict:
        """Validate on held-out data."""
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()

        total_mse = 0.0
        total_q_value = 0.0
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

                q_td = TensorDict({"pixels": frames, "action": pred_loc}, batch_size=[batch_size])
                q1_val = self.q1(q_td)["state_action_value"]
                q2_val = self.q2(q_td)["state_action_value"]
                q_value = torch.min(q1_val, q2_val).mean()

                total_mse += mse.item() * batch_size
                total_q_value += q_value.item() * batch_size
                total_samples += batch_size

        return {
            "val_mse": total_mse / total_samples,
            "val_q_value": total_q_value / total_samples,
        }

    def step_scheduler(self, val_loss: float):
        """Step the learning rate schedulers."""
        self.actor_scheduler.step(val_loss)
        self.critic_scheduler.step(val_loss)


@cli_command(group="ac", name="train-bc", help="Train SAC policy using BC-SAC")
@cli_option(
    "--data-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory with demonstrations",
)
@cli_option("--output-path", default="models/bc_sac_pretrained.pt", help="Output model path")
@cli_option("--epochs", default=50, help="Number of training epochs")
@cli_option("--batch-size", default=64, help="Batch size")
@cli_option("--actor-lr", default=1e-4, type=float, help="Actor learning rate")
@cli_option("--critic-lr", default=1e-4, type=float, help="Critic learning rate")
@cli_option("--bc-weight", default=1.0, type=float, help="Weight for BC loss (λ)")
@cli_option("--gamma", default=0.92, type=float, help="Discount factor")
@cli_option("--tau", default=0.005, type=float, help="Target network update rate")
@cli_option("--alpha", default=0.2, type=float, help="SAC entropy coefficient")
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
@cli_option("--wandb-project", default="AssetoCorsaRL-BCSAC", help="WandB project name")
@cli_option("--wandb-offline", is_flag=True, help="Run WandB offline")
def main(
    data_dir,
    output_path,
    epochs,
    batch_size,
    actor_lr,
    critic_lr,
    bc_weight,
    gamma,
    tau,
    alpha,
    val_split,
    num_workers,
    num_cells,
    augment,
    dropout,
    vae_checkpoint,
    wandb_project,
    wandb_offline,
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
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "bc_weight": bc_weight,
                "gamma": gamma,
                "tau": tau,
                "alpha": alpha,
                "val_split": val_split,
                "num_workers": num_workers,
                "num_cells": num_cells,
                "augment": augment,
                "dropout": dropout,
                "vae_checkpoint": str(vae_checkpoint) if vae_checkpoint else None,
                "wandb_project": wandb_project,
            },
            mode=wandb_mode,
        )
    except Exception as e:
        print(f"Warning: WandB initialization failed: {e}")
        print("Continuing without WandB logging...")

    IMAGE_HEIGHT = 84
    IMAGE_WIDTH = 84
    FRAME_STACK = 4

    train_dataset_full = DemonstrationDataset(
        data_dir,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
        frame_stack=FRAME_STACK,
        augment=augment,
    )

    val_dataset_full = DemonstrationDataset(
        data_dir,
        image_shape=(IMAGE_HEIGHT, IMAGE_WIDTH),
        frame_stack=FRAME_STACK,
        augment=False,  # no augmentation for validation
    )

    n_samples = len(train_dataset_full)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    indices = torch.randperm(n_samples).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

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

    modules = agent.modules()
    actor = modules["actor"]
    q1 = modules["q1"]
    q2 = modules["q2"]
    q1_target = modules["q1_target"]
    q2_target = modules["q2_target"]

    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Q1 parameters: {sum(p.numel() for p in q1.parameters()):,}")
    print(f"Q2 parameters: {sum(p.numel() for p in q2.parameters()):,}")

    print("\n" + "=" * 50)
    print("Initializing BC-SAC Trainer")
    print("=" * 50)
    print(f"BC weight (λ): {bc_weight}")
    print(f"Discount factor (γ): {gamma}")
    print(f"Target network tau (τ): {tau}")
    print(f"SAC entropy alpha (α): {alpha}")
    print("=" * 50 + "\n")

    trainer = BCSACTrainer(
        actor=actor,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        device=device,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        bc_weight=bc_weight,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
    )

    best_val_mse = float("inf")
    patience_counter = 0
    EARLY_STOP_PATIENCE = 50
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("Starting BC-SAC Training")
    print("=" * 50)

    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader)

        trainer.step_scheduler(val_metrics["val_mse"])

        metrics = {
            "epoch": epoch,
            "train/critic_loss": train_metrics["critic_loss"],
            "train/actor_loss": train_metrics["actor_loss"],
            "train/bc_loss": train_metrics["bc_loss"],
            "train/q_loss": train_metrics["q_loss"],
            "val/mse": val_metrics["val_mse"],
            "val/q_value": val_metrics["val_q_value"],
            "lr/actor": trainer.actor_optimizer.param_groups[0]["lr"],
            "lr/critic": trainer.critic_optimizer.param_groups[0]["lr"],
        }

        try:
            wandb.log(metrics)
        except:
            pass

        log_str = (
            f"Epoch {epoch + 1}/{epochs} | "
            f"Critic Loss: {train_metrics['critic_loss']:.6f} | "
            f"Actor Loss: {train_metrics['actor_loss']:.6f} | "
            f"BC Loss: {train_metrics['bc_loss']:.6f} | "
            f"Val MSE: {val_metrics['val_mse']:.6f}"
        )

        print(log_str)

        if val_metrics["val_mse"] < best_val_mse:
            best_val_mse = val_metrics["val_mse"]
            patience_counter = 0
            try:
                checkpoint = {
                    "actor_state": actor.state_dict(),
                    "q1_state": q1.state_dict(),
                    "q2_state": q2.state_dict(),
                    "q1_target_state": q1_target.state_dict(),
                    "q2_target_state": q2_target.state_dict(),
                    "epoch": epoch,
                    "val_mse": best_val_mse,
                    "config": {
                        "num_cells": num_cells,
                        "vae_checkpoint_path": vae_path,
                        "bc_weight": bc_weight,
                        "gamma": gamma,
                        "tau": tau,
                        "alpha": alpha,
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
            "q1_state": q1.state_dict(),
            "q2_state": q2.state_dict(),
            "q1_target_state": q1_target.state_dict(),
            "q2_target_state": q2_target.state_dict(),
            "epoch": epoch,
            "val_mse": val_metrics["val_mse"],
            "config": {
                "num_cells": num_cells,
                "vae_checkpoint_path": vae_path,
                "bc_weight": bc_weight,
                "gamma": gamma,
                "tau": tau,
                "alpha": alpha,
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
    print("BC-SAC Training Complete!")
    print(f"Best Val MSE: {best_val_mse:.6f}")
    print(f"Best model saved to: {output_path}")
    print(f"Final model saved to: {final_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
