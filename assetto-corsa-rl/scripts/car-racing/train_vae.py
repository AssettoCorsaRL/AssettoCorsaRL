"""
Usage:
    acrl car-racing train-vae --epochs 10 --batch-size 128 --steps-per-epoch 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms as T

repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from assetto_corsa_rl.model.vae import ConvVAE  # type: ignore
except Exception:
    from assetto_corsa_rl.model.vae import ConvVAE  # type: ignore

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import os


# ===== action generation =====


def generate_action(prev_action: np.ndarray) -> np.ndarray:
    """Generate random actions biased toward acceleration and steering."""
    if np.random.randint(3) % 3:
        return prev_action

    index = np.random.randn(3)
    index[1] = np.abs(index[1])
    index = np.argmax(index)

    mask = np.zeros(3, dtype=float)
    mask[index] = 1.0

    action = np.random.randn(3)
    action = np.tanh(action)
    action[1] = (action[1] + 1.0) / 2.0
    action[2] = (action[2] + 1.0) / 2.0

    return action * mask


# ===== dataset =====


class CarRacingIterableDataset(IterableDataset):
    """Streams frames from CarRacing env with frame stacking."""

    def __init__(
        self,
        num_samples: int,
        time_steps: int = 150,
        frames: int = 4,
        seed: int = 0,
        render: bool = False,
        warmup_steps: int = 30,
    ):
        self.num_samples = int(num_samples)
        self.time_steps = int(time_steps)
        self.frames = int(frames)
        self.seed = int(seed)
        self.render = bool(render)
        self.warmup_steps = int(warmup_steps)
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((64, 64)),
                T.ToTensor(),
            ]
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        import gymnasium as gym
        from collections import deque

        env = gym.make("CarRacing-v3")

        samples_yielded = 0
        action = np.array([0.0, 0.0, 0.0])

        try:
            while samples_yielded < self.num_samples:
                obs = env.reset()
                if isinstance(obs, (tuple, list)):
                    obs = obs[0]

                frame_buffer = deque(maxlen=self.frames)

                for step in range(self.time_steps):
                    if self.render:
                        try:
                            env.render()
                        except Exception:
                            pass

                    action = generate_action(action)
                    step_res = env.step(action)

                    if isinstance(step_res, tuple) and len(step_res) == 5:
                        obs, _, terminated, truncated, _ = step_res
                        done = bool(terminated or truncated)
                    elif isinstance(step_res, tuple) and len(step_res) == 4:
                        obs, _, done, _ = step_res
                    else:
                        raise ValueError(f"Unexpected env.step() return: {step_res}")

                    new_img = self.transform(obs)
                    frame_buffer.append(new_img)

                    if step < self.warmup_steps or len(frame_buffer) < self.frames:
                        continue

                    stacked = torch.cat(list(frame_buffer), dim=0)
                    yield stacked

                    samples_yielded += 1
                    if samples_yielded >= self.num_samples:
                        break

                    if done:
                        break
        finally:
            try:
                env.close()
            except Exception:
                pass


# ===== training helper =====


def make_dataloaders(
    batch_size: int,
    steps_per_epoch: int,
    val_steps: int,
    num_workers: int,
    seed: int,
    frames: int = 4,
    warmup_steps: int = 30,
):
    train_samples = batch_size * steps_per_epoch
    val_samples = batch_size * val_steps

    train_ds = CarRacingIterableDataset(
        num_samples=train_samples,
        time_steps=150,
        frames=frames,
        seed=seed,
        render=False,
        warmup_steps=warmup_steps,
    )
    val_ds = CarRacingIterableDataset(
        num_samples=val_samples,
        time_steps=150,
        frames=frames,
        seed=seed + 999,
        render=False,
        warmup_steps=warmup_steps,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--val-steps", type=int, default=200)
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=500, help="Linear LR warmup steps")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument(
        "--frames",
        type=int,
        default=4,
        help="Number of consecutive frames to stack as input",
    )
    p.add_argument("--wandb-project", type=str, default="assetto_corsa_rl_vae")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-offline", action="store_true", help="Run wandb in offline mode")
    return p.parse_args()


@cli_command(group="car-racing", name="train-vae", help="Train VAE on CarRacing environment")
@cli_option("--epochs", default=10, help="Number of epochs")
@cli_option("--batch-size", default=32, help="Batch size")
@cli_option("--steps-per-epoch", default=1000, help="Steps per epoch")
@cli_option("--val-steps", default=200, help="Validation steps")
@cli_option("--z-dim", default=32, help="Latent dimension")
@cli_option("--lr", default=1e-3, type=float, help="Learning rate")
@cli_option("--beta", default=1.0, type=float, help="KL weight")
@cli_option("--warmup-steps", default=500, help="Linear LR warmup steps")
@cli_option("--grad-clip", default=1.0, type=float, help="Gradient clipping value")
@cli_option("--num-workers", default=4, help="Data loader workers")
@cli_option("--seed", default=42, help="Random seed")
@cli_option("--gpus", default=0, help="Number of GPUs")
@cli_option("--ckpt-dir", default="checkpoints", help="Checkpoint directory")
@cli_option("--frames", default=4, help="Number of consecutive frames to stack")
@cli_option("--wandb-project", default="assetto_corsa_rl_vae", help="WandB project")
@cli_option("--wandb-entity", default=None, help="WandB entity")
@cli_option("--wandb-name", default=None, help="WandB run name")
@cli_option("--wandb-offline", is_flag=True, help="Run wandb in offline mode")
def main(
    epochs,
    batch_size,
    steps_per_epoch,
    val_steps,
    z_dim,
    lr,
    beta,
    warmup_steps,
    grad_clip,
    num_workers,
    seed,
    gpus,
    ckpt_dir,
    frames,
    wandb_project,
    wandb_entity,
    wandb_name,
    wandb_offline,
):
    from types import SimpleNamespace

    args = SimpleNamespace(
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        z_dim=z_dim,
        lr=lr,
        beta=beta,
        warmup_steps=warmup_steps,
        grad_clip=grad_clip,
        num_workers=num_workers,
        seed=seed,
        gpus=gpus,
        ckpt_dir=ckpt_dir,
        frames=frames,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_name=wandb_name,
        wandb_offline=wandb_offline,
    )

    pl.seed_everything(args.seed, workers=True)

    # Create dataloaders
    train_loader, val_loader = make_dataloaders(
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        num_workers=args.num_workers,
        seed=args.seed,
        frames=args.frames,
    )

    xb = next(iter(train_loader))
    in_channels = 3 * args.frames
    if xb.ndim != 4 or xb.size(1) != in_channels:
        raise RuntimeError(
            f"Unexpected batch shape {xb.shape}, expected (B, {in_channels}, 64, 64)"
        )
    print(f"✓ Batch shape verified: {xb.shape}")

    model = ConvVAE(
        z_dim=args.z_dim,
        lr=args.lr,
        beta=args.beta,
        in_channels=in_channels,
        warmup_steps=args.warmup_steps,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="vae-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        entity=args.wandb_entity,
        offline=args.wandb_offline,
    )
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    logger.watch(model, log="all", log_freq=100)
    if torch.cuda.is_available():
        accelerator = "cuda"
        devices = (
            torch.cuda.device_count()
            if args.gpus <= 0
            else min(args.gpus, torch.cuda.device_count())
        )
        print(f"✓ Using CUDA with {devices} device(s)")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        print("✓ Using MPS")
    else:
        accelerator = None
        devices = None
        print("✓ Using CPU")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.ckpt_dir,
        callbacks=[ckpt_cb],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=50,
        limit_train_batches=args.steps_per_epoch,
        limit_val_batches=args.val_steps,
        gradient_clip_val=args.grad_clip,
    )

    print(f"\n🚀 Starting training: {args.epochs} epochs, {args.steps_per_epoch} steps/epoch\n")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_ckpt = Path(args.ckpt_dir) / "vae-final.pth"
    model.cpu()
    torch.save(model.state_dict(), final_ckpt)
    print(f"Saved final model to {final_ckpt}")


if __name__ == "__main__":
    main()
