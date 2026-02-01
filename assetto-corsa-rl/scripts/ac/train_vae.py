"""
Example:
    acrl ac train-vae --input-dir datasets/ac_images --frames 1 --epochs 100 --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from assetto_corsa_rl.model.vae import ConvVAE  # type: ignore
from assetto_corsa_rl.ac_env import parse_image_shape  # type: ignore

try:
    from assetto_corsa_rl.cli_registry import cli_command, cli_option  # type: ignore
except Exception:
    from ...src.assetto_corsa_rl.cli_registry import cli_command, cli_option

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import os


class ACImageStackDataset(Dataset):
    """Loads .npz stacks saved by save_image_dataset.py.

    Handles two formats:
    - Single stack: (F, H, W) - one stack per file
    - Batch format: (N, F, H, W) - multiple stacks per file
    """

    def __init__(
        self,
        input_dir: Path,
        image_shape: Tuple[int, int] = (84, 84),
        files: List[Path] = None,
        frames: int | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.image_shape = tuple(map(int, image_shape))
        self.frames = frames
        self.files = files or sorted([p for p in self.input_dir.glob("*.npz") if p.is_file()])
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {self.input_dir}")

        self.index = []
        for file_idx, file_path in enumerate(self.files):
            d = np.load(str(file_path))
            key = "stack" if "stack" in d else d.files[0]
            stack = np.asarray(d[key])
            d.close()

            if stack.ndim == 3:
                # single stack (v1.0.2 or below): (F, H, W)
                self.index.append((file_idx, None))
            elif stack.ndim == 4:
                # batch format (v1.0.3 or above): (N, F, H, W)
                for i in range(stack.shape[0]):
                    self.index.append((file_idx, i))
            else:
                raise ValueError(f"Unexpected shape in {file_path}: {stack.shape}")

        print(f"Loaded {len(self.index)} stacks from {len(self.files)} files")

    def __len__(self):
        return len(self.index)

    def _load_stack(self, p: Path, sample_idx: int | None = None) -> np.ndarray:
        d = np.load(str(p))
        if "stack" in d:
            stack = d["stack"]
        else:
            keys = [k for k in d.files]
            stack = d[keys[0]]
        stack = np.asarray(stack).astype(np.uint8)

        if stack.ndim == 3:
            # single stack (v1.0.2 or below): (F, H, W)
            return stack
        elif stack.ndim == 4:
            # batch format (v1.0.3 or above): (N, F, H, W)
            if sample_idx is None:
                raise ValueError(f"sample_idx required for batch format in {p}")
            return stack[sample_idx]
        else:
            raise ValueError(f"Expected (F,H,W) or (N,F,H,W) in {p}, got {stack.shape}")

    def _resize_frame(self, fr: np.ndarray) -> np.ndarray:
        h, w = self.image_shape
        if fr.shape != (h, w):
            try:
                return cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            except Exception:
                return cv2.resize(fr, (w, h)).astype(np.uint8)
        return fr

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index[idx]
        p = self.files[file_idx]
        stack = self._load_stack(p, sample_idx)  # (F, H, W)

        if self.frames is not None and stack.shape[0] > self.frames:
            stack = stack[-self.frames :]  # keep most recent frames

        rgb_frames = []
        for fr in stack:
            fr_resized = self._resize_frame(fr)
            fr_rgb = np.stack([fr_resized, fr_resized, fr_resized], axis=0)
            rgb_frames.append(fr_rgb)

        sample = np.concatenate(rgb_frames, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(sample)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True, help="Directory with .npz stacks")
    p.add_argument("--image-shape", type=str, default="84x84", help="HxW target size (e.g. 84x84)")
    p.add_argument("--frames", type=int, default=4, help="Number of grayscale frames per stack")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="KL weight (lower = less collapse, try 0.001-0.1)",
    )
    p.add_argument(
        "--mse-weight",
        type=float,
        default=1.0,
        help="MSE loss weight (add pixel-level penalty)",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument(
        "--encoder-ckpt",
        type=Path,
        default=None,
        help="Path to pretrained encoder checkpoint to load",
    )
    p.add_argument("--wandb-project", type=str, default="assetto_corsa_rl_vae")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-offline", action="store_true")
    return p.parse_args()


@cli_command(group="ac", name="train-vae", help="Train VAE on Assetto Corsa image dataset")
@cli_option("--input-dir", required=True, help="Directory with .npz stacks")
@cli_option("--image-shape", default="84x84", help="Target image shape HxW")
@cli_option("--frames", default=4, help="Number of grayscale frames per stack")
@cli_option("--epochs", default=10, help="Number of epochs")
@cli_option("--batch-size", default=32, help="Batch size")
@cli_option("--lr", default=3e-4, type=float, help="Learning rate")
@cli_option("--beta", default=0.01, type=float, help="KL weight")
@cli_option("--mse-weight", default=1.0, type=float, help="MSE loss weight")
@cli_option("--num-workers", default=4, help="Number of data loader workers")
@cli_option("--gpus", default=1, help="Number of GPUs")
@cli_option("--ckpt-dir", default="checkpoints", help="Checkpoint directory")
@cli_option("--encoder-ckpt", default=None, help="Pretrained encoder checkpoint")
@cli_option("--wandb-project", default="assetto_corsa_rl_vae", help="WandB project")
@cli_option("--wandb-entity", default=None, help="WandB entity")
@cli_option("--wandb-name", default=None, help="WandB run name")
@cli_option("--wandb-offline", is_flag=True, help="Run WandB offline")
def main(
    input_dir,
    image_shape,
    frames,
    epochs,
    batch_size,
    lr,
    beta,
    mse_weight,
    num_workers,
    gpus,
    ckpt_dir,
    encoder_ckpt,
    wandb_project,
    wandb_entity,
    wandb_name,
    wandb_offline,
):
    img_h, img_w = parse_image_shape(image_shape)

    files = sorted([p for p in Path(input_dir).glob("*.npz") if p.is_file()])
    if len(files) == 0:
        raise RuntimeError(f"No .npz files found in {input_dir}")

    # split train/val 90/10
    split = max(1, int(len(files) * 0.9))
    train_files = files[:split]
    val_files = files[split:]

    train_ds = ACImageStackDataset(
        input_dir,
        image_shape=(img_h, img_w),
        files=train_files,
        frames=frames,
    )
    val_ds = ACImageStackDataset(
        input_dir, image_shape=(img_h, img_w), files=val_files, frames=frames
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
    )

    xb = next(iter(train_loader))
    in_channels = 3 * frames
    if xb.ndim != 4 or xb.size(1) != in_channels:
        raise RuntimeError(f"Unexpected batch shape {xb.shape}, expected (B, {in_channels}, H, W)")
    print(f"✓ Batch shape verified: {xb.shape}")

    model = ConvVAE(
        z_dim=1024,
        lr=lr,
        beta=beta,
        in_channels=in_channels,
        warmup_steps=500,
        im_shape=(img_h, img_w),
        mse_weight=mse_weight,
    )

    if encoder_ckpt:
        print(f"Loading encoder from {encoder_ckpt}...")
        try:
            ckpt = torch.load(str(encoder_ckpt), map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt

            encoder_keys = ["encoder_in", "encoder_blocks", "middle", "to_latent"]
            encoder_state = {
                k: v
                for k, v in state_dict.items()
                if any(k.startswith(prefix) for prefix in encoder_keys)
            }

            if encoder_state:
                model.load_state_dict(encoder_state, strict=False)
                print(f"✓ Loaded {len(encoder_state)} encoder parameters")
            else:
                print("Warning: No encoder parameters found in checkpoint")
        except Exception as e:
            print(f"Warning: Failed to load encoder checkpoint: {e}")

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="vae-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    logger = WandbLogger(
        project=wandb_project,
        name=wandb_name,
        entity=wandb_entity,
        offline=wandb_offline,
    )
    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    logger.watch(model, log="all", log_freq=100)

    if torch.cuda.is_available() and gpus != 0:
        accelerator = "cuda"
        devices = gpus if gpus > 0 else torch.cuda.device_count()
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
        max_epochs=epochs,
        default_root_dir=ckpt_dir,
        callbacks=[ckpt_cb],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=50,
        gradient_clip_val=1.0,  # clip gradients to prevent NaN
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_ckpt = Path(ckpt_dir) / "vae-final.pth"
    model.cpu()
    torch.save(model.state_dict(), final_ckpt)
    print(f"Saved final model to {final_ckpt}")


if __name__ == "__main__":
    main()
