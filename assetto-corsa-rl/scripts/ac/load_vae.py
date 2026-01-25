"""Load a trained VAE and inspect encoder/decoder outputs on environment frames.

Example:
    acrl ac load-vae --ckpt loss=0.1031.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple
import torch
import torchvision
import numpy as np
from PIL import Image
import cv2
import time
import click

repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from assetto_corsa_rl.model.vae import ConvVAE  # type: ignore
from assetto_corsa_rl.env_helper import create_gym_env  # type: ignore
from assetto_corsa_rl.ac_env import parse_image_shape, get_device  # type: ignore
from collections import deque

try:
    from assetto_corsa_rl.cli_registry import cli_command, cli_option  # type: ignore
except Exception:
    from ...src.assetto_corsa_rl.cli_registry import cli_command, cli_option


# optional AC env
try:
    from assetto_corsa_rl.ac_env import make_env as make_ac_env  # type: ignore
except Exception:
    make_ac_env = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True, help="VAE checkpoint (state_dict) to load")
    p.add_argument(
        "--env",
        type=str,
        choices=["gym", "ac"],
        default="ac",
        help="Which environment to use: 'gym' (CarRacing) or 'ac' (Assetto Corsa)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Number of grayscale frames stacked by the env",
    )
    p.add_argument("--image-shape", type=str, default="84x84", help="HxW image shape (e.g. 84x84)")
    p.add_argument("--device", type=str, default=None, help="Device to run on (cpu/cuda)")
    return p.parse_args()


def graystack_to_rgb_input(pixels: torch.Tensor, frames: int):
    """Convert grayscale stacked frames (C=frames) to VAE input with 3*frames channels.

    Accepts pixels shaped (C,H,W) or (B,C,H,W) and returns (B, 3*frames, H, W).
    """
    if pixels.dim() == 3:
        pixels = pixels.unsqueeze(0)
    B, C, H, W = pixels.shape
    assert C == frames, f"Expected {frames} frames in pixels, got {C}"
    per_frame = []
    for i in range(frames):
        fr = pixels[:, i : i + 1, :, :].repeat(1, 3, 1, 1)
        per_frame.append(fr)
    x = torch.cat(per_frame, dim=1)
    return x


def _recon_display(
    original: torch.Tensor, recon_rgb: torch.Tensor, title: str = "VAE Comparison"
) -> int:
    """Display original and reconstruction side-by-side in a popout window. Returns pressed key code."""
    if original.dim() == 4:
        orig_t = original[0]
    else:
        orig_t = original
    if orig_t.size(0) > 3:
        orig_t = orig_t[-3:]
    elif orig_t.size(0) == 1:
        orig_t = orig_t.repeat(3, 1, 1)

    if recon_rgb.dim() == 4:
        recon_t = recon_rgb[0]
    else:
        recon_t = recon_rgb
    if recon_t.size(0) > 3:
        recon_t = recon_t[-3:]

    orig_arr = (orig_t.detach().cpu().numpy() * 255.0).astype(np.uint8)
    orig_arr = np.transpose(orig_arr, (1, 2, 0))  # H,W,C

    recon_arr = (recon_t.detach().cpu().numpy() * 255.0).astype(np.uint8)
    recon_arr = np.transpose(recon_arr, (1, 2, 0))  # H,W,C

    combined = np.hstack([orig_arr, recon_arr])

    combined = combined[..., ::-1]

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, combined.shape[1] * 4, combined.shape[0] * 4)
    cv2.imshow(title, combined)
    key = cv2.waitKey(1) & 0xFF
    return key


@cli_command(
    group="ac", name="load-vae", help="Load and inspect a trained VAE on environment frames"
)
@cli_option("--ckpt", required=True, help="VAE checkpoint path")
@cli_option("--env", default="ac", type=click.Choice(["gym", "ac"]), help="Environment to use")
@cli_option("--frames", default=1, help="Number of grayscale frames stacked")
@cli_option("--image-shape", default="84x84", help="Image shape HxW")
@cli_option("--device", default=None, help="Device to run on")
def main(ckpt, env, frames, image_shape, device):
    img_h, img_w = parse_image_shape(image_shape)
    device = get_device(device)

    vae = ConvVAE(
        z_dim=1024,
        in_channels=3,
        warmup_steps=500,
        im_shape=(img_h, img_w),
    )
    ckpt_data = torch.load(str(ckpt), map_location="cpu")
    try:
        vae.load_state_dict(ckpt_data)
    except Exception:
        if isinstance(ckpt_data, dict) and "state_dict" in ckpt_data:
            vae.load_state_dict(ckpt_data["state_dict"])  # type: ignore
        else:
            raise

    vae.to(device)
    vae.eval()
    print(f"✓ Loaded VAE from {ckpt}")

    if env == "gym":
        env = create_gym_env(height=img_h, width=img_w, device=device, num_envs=1)
        td = env.reset()

        try:
            while True:
                pixels = td["pixels"]  # shape (B, C, H, W) or (C, H, W)
                pixels_sample = pixels[0] if pixels.dim() == 4 else pixels

                x = graystack_to_rgb_input(pixels_sample.cpu(), frames).to(device)
                with torch.no_grad():
                    recon = vae(x)

                key = _recon_display(x, recon)
                if key in (ord("q"), 27):
                    break

                action_shape = env.action_spec.shape
                action = torch.zeros((env.num_workers, *action_shape), device=device)
                td = env.step(action)

                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    else:
        if make_ac_env is None:
            raise RuntimeError("Assetto Corsa env not available (failed to import).")

        # create AC environment: returns gym-style obs dicts
        env = make_ac_env(include_image=True, observation_image_shape=(img_h, img_w))
        obs, info = env.reset()

        # initialize frame buffer with the initial frame repeated
        initial_img = obs["image"]  # H, W, 1 uint8
        frame_buf = deque(maxlen=frames)
        fr = torch.from_numpy(initial_img[..., 0].astype(np.float32) / 255.0)
        for _ in range(frames):
            frame_buf.append(fr.clone())

        try:
            while True:
                stacked = torch.stack(list(frame_buf), dim=0)
                x = graystack_to_rgb_input(stacked, frames).to(device)

                with torch.no_grad():
                    recon = vae(x)

                key = _recon_display(x, recon)
                if key in (ord("q"), 27):
                    break

                # step env with zero action to advance (steer, throttle, brake)
                action = np.zeros(3, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)

                next_img = obs["image"]  # H,W,1
                next_fr = torch.from_numpy(next_img[..., 0].astype(np.float32) / 255.0)
                frame_buf.append(next_fr)

                if terminated or truncated:
                    obs, info = env.reset()
                    initial_img = obs["image"]
                    frame_buf = deque(
                        [torch.from_numpy(initial_img[..., 0].astype(np.float32) / 255.0)] * frames,
                        maxlen=frames,
                    )

                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
