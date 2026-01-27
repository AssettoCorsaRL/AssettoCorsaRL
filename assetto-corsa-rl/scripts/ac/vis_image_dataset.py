"""Visualize saved frame-stack datasets (.npz) from Assetto Corsa.

Usage:
    acrl ac vis-dataset --input-dir datasets/demonstrations

Controls (when window active):
 - n : next sample
 - b : previous sample
 - space : toggle play/pause (anim mode)
 - m : toggle view mode (anim / montage)
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


def load_stack(path: Path) -> np.ndarray:
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

        stack = np.asarray(stack)

        if np.issubdtype(stack.dtype, np.floating):
            max_val = float(stack.max() if stack.size else 1.0)
            if max_val <= 1.0:
                stack = stack * 255.0
            stack = stack.clip(0, 255)

        if stack.ndim == 4:
            stack = stack[0]

        if stack.ndim != 3:
            raise ValueError(f"Expected stack with shape (F, H, W), got {stack.shape}")
        return stack.astype(np.uint8)
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


def overlay_text(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


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
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    playing = False
    # view_mode is already a parameter
    frame_idx = 0
    last_time = time.time()

    while True:
        path = files[idx]
        try:
            stack = load_stack(path)
        except Exception as e:
            print(e)
            idx = (idx + 1) % len(files)
            continue

        F, H, W = stack.shape

        if view_mode == "montage":
            montage = make_montage(stack)
            display_img = to_bgr(montage)
            title = f"{idx+1}/{len(files)} {path.name} [montage]"
        else:
            # anim mode: show single frame at frame_idx
            frame_idx = frame_idx % F
            frame = stack[frame_idx]
            display_img = to_bgr(frame)
            title = f"{idx+1}/{len(files)} {path.name} [frame {frame_idx+1}/{F}]"

        # overlay filename and instructions
        overlay = display_img.copy()
        overlay_text(overlay, title)
        overlay_text(overlay, "n:next  b:prev  space:play/pause  m:toggle view  s:save  q:quit")

        # apply scaling
        if scale != 1.0:
            h = int(overlay.shape[0] * scale)
            w = int(overlay.shape[1] * scale)
            overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LINEAR)

        cv2.imshow(window_name, overlay)

        # manage playback timing
        key = cv2.waitKey(int(max(1, delay * 1000))) & 0xFF
        if key != 0xFF:
            # handle key
            if key == ord("q") or key == 27:  # esc
                break
            elif key == ord("n"):
                idx = (idx + 1) % len(files)
                frame_idx = 0
                playing = False
            elif key == ord("b"):
                idx = (idx - 1) % len(files)
                frame_idx = 0
                playing = False
            elif key == ord(" "):
                playing = not playing
            elif key == ord("m"):
                view_mode = "montage" if view_mode == "anim" else "anim"
                frame_idx = 0
            elif key == ord(">"):
                delay = max(0.001, delay * 0.5)
            elif key == ord("<"):
                delay = delay * 1.5
            elif key == ord("s"):
                save_path = Path(save_dir) if save_dir else input_dir
                save_path.mkdir(parents=True, exist_ok=True)
                if view_mode == "montage":
                    out = make_montage(stack)
                else:
                    out = stack[frame_idx]
                out_path = save_path / f"viz_{idx+1:06d}_{path.stem}.png"
                cv2.imwrite(str(out_path), out)
                print(f"Saved visualization: {out_path}")

        # advance frame if playing and in anim mode
        if playing and view_mode == "anim":
            now = time.time()
            if now - last_time >= delay:
                frame_idx = (frame_idx + 1) % F
                last_time = now

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
