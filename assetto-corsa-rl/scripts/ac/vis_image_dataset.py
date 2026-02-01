"""Visualize saved frame-stack datasets (.npz) from Assetto Corsa.

Usage:
    acrl ac vis-dataset --input-dir datasets/demonstrations2

Controls (when window active):
 - n : next sample
 - b : previous sample
 - t : toggle transformations
 - space : toggle play/pause (anim mode)
 - m : toggle view mode (anim / montage)
 - t : toggle transform visualization
 - > : speed up (decrease delay)
 - < : slow down (increase delay)
 - s : save current visualization (PNG)
 - q or ESC : quit

Displays either an animated single-frame playback (`anim`) or a montage of the stacked frames (`montage`).
"""

from pathlib import Path
import argparse
import sys
import time
import math
import click

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

try:
    from assetto_corsa_rl.cli_registry import cli_command, cli_option  # type: ignore
except Exception:
    import importlib.util

    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    spec = importlib.util.spec_from_file_location(
        "cli_registry", src_path / "assetto_corsa_rl" / "cli_registry.py"
    )
    if spec and spec.loader:
        cli_registry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli_registry)
        cli_command = cli_registry.cli_command
        cli_option = cli_registry.cli_option


def parse_args():
    p = argparse.ArgumentParser(description="Visualize frame-stack dataset (.npz)")
    p.add_argument("--input-dir", type=Path, required=True, help="Directory with .npz stacks")
    p.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern to find stacks")
    p.add_argument("--delay", type=float, default=0.08, help="Frame playback delay in seconds")
    p.add_argument("--scale", type=float, default=1.0, help="Display scale factor")
    p.add_argument("--start", type=int, default=0, help="Starting sample index")
    p.add_argument(
        "--view-mode",
        choices=["anim", "montage"],
        default="anim",
        help="Initial view mode",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, will save visualizations when pressing 's'",
    )
    return p.parse_args()


def list_files(input_dir: Path, pattern: str):
    files = sorted([p for p in input_dir.glob(pattern) if p.is_file()])
    return files


def load_stack(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a frame stack and any associated action from a .npz sample.

    Returns (stack, action) where action may be None if not present.
    """
    try:
        d = np.load(str(path))

        if "stack" in d:
            stack = d["stack"]
        elif "frames" in d:
            frames = d["frames"]
            if frames.ndim == 4:
                stack = frames[0]
            else:
                stack = frames
        else:
            keys = [k for k in d.files]
            stack = d[keys[0]]

        action = None
        if "action" in d:
            action = np.asarray(d["action"])
        elif "actions" in d:
            actions = d["actions"]
            # If actions is a batch, pick the first entry to match stack[0]
            if actions.ndim == 2:
                action = np.asarray(actions[0])
            else:
                action = np.asarray(actions)

        stack = np.asarray(stack)

        if np.issubdtype(stack.dtype, np.floating):
            if stack.ndim == 3:  # (F, H, W)
                normalized_frames = []
                for frame in stack:
                    frame_min = frame.min()
                    frame_max = frame.max()
                    if frame_max > frame_min:
                        # normalize to 0-255
                        normalized = ((frame - frame_min) / (frame_max - frame_min)) * 255.0
                    else:
                        normalized = frame * 0
                    normalized_frames.append(normalized)
                stack = np.array(normalized_frames)
            else:
                max_val = float(stack.max() if stack.size else 1.0)
                if max_val <= 1.0:
                    stack = stack * 255.0
            stack = stack.clip(0, 255)

        if stack.ndim == 4:
            stack = stack[0]

        if stack.ndim != 3:
            raise ValueError(f"Expected stack with shape (F, H, W), got {stack.shape}")
        return stack.astype(np.uint8), action
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")


def make_montage(stack: np.ndarray) -> np.ndarray:
    F, H, W = stack.shape
    cols = math.ceil(math.sqrt(F))
    rows = math.ceil(F / cols)
    pad = cols * rows - F
    if pad > 0:
        pad_frames = np.zeros((pad, H, W), dtype=np.uint8)
        stack = np.concatenate([stack, pad_frames], axis=0)
    tiles = []
    for r in range(rows):
        row_frames = [stack[r * cols + c] for c in range(cols)]
        row_img = cv2.hconcat(row_frames)
        tiles.append(row_img)
    montage = cv2.vconcat(tiles)
    return montage


def to_bgr(img: np.ndarray) -> np.ndarray:
    # img is grayscale HxW
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_action_img(action: np.ndarray | None, size=(200, 300)) -> np.ndarray:
    """Create a small visualization image for an action vector.

    Expects action = [steer, gas, brake] or None.
    """
    h, w = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if action is None:
        cv2.putText(canvas, "No action", (8, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        return canvas

    labels = ["steer", "gas", "brake"]
    colors = [(200, 200, 0), (0, 200, 0), (0, 0, 200)]

    bar_w = w - 40
    y = 20
    for i, lbl in enumerate(labels):
        val = float(action[i]) if i < len(action) else 0.0
        # steer is in [-1,1], map to center line
        if lbl == "steer":
            cx = 20 + bar_w // 2
            half = bar_w // 2
            fill = int((val + 1.0) / 2.0 * half)
            # background bar
            cv2.rectangle(canvas, (20, y), (20 + bar_w, y + 24), (50, 50, 50), -1)
            # center line
            cv2.line(canvas, (cx, y), (cx, y + 24), (100, 100, 100), 1)
            if val >= 0:
                cv2.rectangle(canvas, (cx, y), (cx + fill, y + 24), colors[i], -1)
            else:
                cv2.rectangle(canvas, (cx + fill, y), (cx, y + 24), colors[i], -1)
        else:
            # gas/brake in [0,1]
            fill = int(max(0.0, min(1.0, val)) * bar_w)
            cv2.rectangle(canvas, (20, y), (20 + bar_w, y + 24), (50, 50, 50), -1)
            cv2.rectangle(canvas, (20, y), (20 + fill, y + 24), colors[i], -1)
        cv2.putText(
            canvas,
            f"{lbl}: {val:.2f}",
            (24 + bar_w + 4, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        y += 34

    return canvas


def overlay_text(img: np.ndarray, text: str, y_pos: int = 20) -> None:
    cv2.putText(img, text, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def apply_transforms(stack: np.ndarray) -> np.ndarray:
    """Apply augmentation transforms to a stack of frames."""
    transforms = T.Compose(
        [
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomAdjustSharpness(sharpness_factor=2),
            T.ElasticTransform(),
            T.RandomPosterize(bits=4, p=1.0),
        ]
    )

    transformed_stack = []
    for frame in stack:
        pil_img = Image.fromarray(frame)
        pil_img = pil_img.convert("RGB")
        transformed = transforms(pil_img)
        transformed_np = np.array(transformed.convert("L"))
        transformed_stack.append(transformed_np)

    return np.array(transformed_stack)


@cli_command(
    group="ac", name="vis-dataset", help="Visualize saved frame-stack datasets from Assetto Corsa"
)
@cli_option("--input-dir", required=True, help="Directory with .npz stacks")
@cli_option("--pattern", default="*.npz", help="Glob pattern to find stacks")
@cli_option("--delay", default=0.08, type=float, help="Frame playback delay in seconds")
@cli_option("--scale", default=1.0, type=float, help="Display scale factor")
@cli_option("--start", default=0, help="Starting sample index")
@cli_option(
    "--view-mode", default="anim", type=click.Choice(["anim", "montage"]), help="Initial view mode"
)
@cli_option("--save-dir", default=None, help="Directory to save visualizations")
def main(input_dir, pattern, delay, scale, start, view_mode, save_dir):
    input_dir = Path(input_dir)
    files = list_files(input_dir, pattern)
    if len(files) == 0:
        print(f"No files found in {input_dir} matching {pattern}")
        sys.exit(1)

    idx = max(0, min(start, len(files) - 1))

    window_name = "AC Dataset Viewer"
    action_window = "AC Action Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(action_window, cv2.WINDOW_NORMAL)

    playing = False
    frame_idx = 0
    last_time = time.time()
    show_transforms = False
    transformed_stack = None
    current_action = None

    while True:
        path = files[idx]
        try:
            stack, action = load_stack(path)
            current_action = action
        except Exception as e:
            print(e)
            idx = (idx + 1) % len(files)
            continue

        F, H, W = stack.shape

        display_stack = stack
        if show_transforms:
            if transformed_stack is None:
                transformed_stack = apply_transforms(stack)
            display_stack = transformed_stack

        action_img = make_action_img(current_action)
        cv2.imshow(action_window, action_img)

        if view_mode == "montage":
            montage = make_montage(display_stack)
            display_img = to_bgr(montage)
            title = f"{idx+1}/{len(files)} {path.name} [montage]"
        else:
            # anim mode: show single frame at frame_idx
            frame_idx = frame_idx % F
            frame = display_stack[frame_idx]
            display_img = to_bgr(frame)
            title = f"{idx+1}/{len(files)} {path.name} [frame {frame_idx+1}/{F}]"

        overlay = display_img.copy()
        status = " [TRANSFORMS ON]" if show_transforms else ""
        overlay_text(overlay, title + status, 20)
        overlay_text(
            overlay, "n:next  b:prev  space:play/pause  m:view  t:transform  s:save  q:quit", 40
        )

        if scale != 1.0:
            h = int(overlay.shape[0] * scale)
            w = int(overlay.shape[1] * scale)
            overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)

        cv2.imshow(window_name, overlay)

        key = cv2.waitKey(int(max(1, delay * 1000))) & 0xFF
        if key != 0xFF:
            if key == ord("q") or key == 27:  # esc
                break
            elif key == ord("n"):
                idx = (idx + 1) % len(files)
                frame_idx = 0
                playing = False
                transformed_stack = None
            elif key == ord("b"):
                idx = (idx - 1) % len(files)
                frame_idx = 0
                playing = False
                transformed_stack = None
            elif key == ord(" "):
                playing = not playing
            elif key == ord("m"):
                view_mode = "montage" if view_mode == "anim" else "anim"
                frame_idx = 0
            elif key == ord("t"):
                show_transforms = not show_transforms
                transformed_stack = None  # regen transforms
            elif key == ord(">"):
                delay = max(0.001, delay * 0.5)
            elif key == ord("<"):
                delay = delay * 1.5
            elif key == ord("s"):
                save_path = Path(save_dir) if save_dir else input_dir
                save_path.mkdir(parents=True, exist_ok=True)
                if view_mode == "montage":
                    out = make_montage(display_stack)
                else:
                    out = display_stack[frame_idx]
                suffix = "_transformed" if show_transforms else ""
                out_path = save_path / f"viz_{idx+1:06d}_{path.stem}{suffix}.png"
                cv2.imwrite(str(out_path), out)
                print(f"Saved visualization: {out_path}")

                action_out_path = save_path / f"viz_{idx+1:06d}_{path.stem}{suffix}_action.png"
                try:
                    action_img = make_action_img(current_action)
                    cv2.imwrite(str(action_out_path), action_img)
                    print(f"Saved action visualization: {action_out_path}")
                except Exception as e:
                    print(f"Failed to save action visualization: {e}")

        # advance frame if playing and in anim mode
        if playing and view_mode == "anim":
            now = time.time()
            if now - last_time >= delay:
                frame_idx = (frame_idx + 1) % F
                last_time = now

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
