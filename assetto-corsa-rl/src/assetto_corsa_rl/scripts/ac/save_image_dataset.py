"""Save grayscale screenshots from Assetto Corsa to a dataset directory.

Controls (console):
- s : start/resume saving
- p : pause saving
- q : quit

Images are captured from the AC window via Telemetry (capture_images=True).
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np

try:
    from assetto_corsa_rl.ac_telemetry_helper import Telemetry  # type: ignore
    from assetto_corsa_rl.ac_env import parse_image_shape
    from assetto_corsa_rl.cli_registry import cli_command, cli_option
except Exception:
    repo_root = Path(__file__).resolve().parents[2]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from assetto_corsa_rl.ac_telemetry_helper import Telemetry  # type: ignore
    from assetto_corsa_rl.cli_registry import cli_command, cli_option


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture screenshots from Assetto Corsa and save to disk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/ac_images"),
        help="Directory where image stacks will be saved",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Telemetry host")
    parser.add_argument("--recv-port", type=int, default=9876, help="Telemetry receive port")
    parser.add_argument(
        "--send-port", type=int, default=9877, help="Action send port (unused here)"
    )
    parser.add_argument(
        "--capture-rate",
        type=float,
        default=0.1,
        help="Seconds between capture attempts",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Number of frames to stack per sample",
    )
    parser.add_argument(
        "--image-shape",
        type=str,
        default="84x84",
        help="Target image shape HxW (e.g. 84x84)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of image stacks to save",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Filename prefix for saved image stacks",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save a PNG visualization concatenating the stack horizontally",
    )
    return parser.parse_args()


def save_stack_npz(stack: np.ndarray, out_dir: Path, prefix: str, counter: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = out_dir / f"{prefix}_{counter:06d}_{ts}.npz"
    # stack shape: (frames, H, W)
    np.savez_compressed(str(fname), stack=stack)
    return fname


def save_stack_png(stack: np.ndarray, out_dir: Path, prefix: str, counter: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = out_dir / f"{prefix}_{counter:06d}_{ts}.png"
    # stack: (F, H, W) -> list of (H, W) uint8
    imgs = [frame for frame in stack]
    concat = cv2.hconcat(imgs)
    cv2.imwrite(str(fname), concat)
    return fname


from collections import deque
from typing import Tuple


def key_loop(telemetry: Telemetry, args) -> None:
    try:
        import msvcrt  # windows-only non-blocking keypress
    except ImportError:
        msvcrt = None

    saving = False
    saved = 0
    frame_stack = args.frame_stack
    img_h, img_w = parse_image_shape(args.image_shape)

    buffer = deque(maxlen=frame_stack)

    print("Controls: s=start/resume, p=pause, q=quit")

    while True:
        if msvcrt and msvcrt.kbhit():
            ch = msvcrt.getch().decode("utf-8", errors="ignore").lower()
            if ch == "s":
                saving = True
                print("▶️  Saving started")
            elif ch == "p":
                saving = False
                print("⏸️  Saving paused")
            elif ch == "q":
                print("👋 Quitting...")
                break

        latest = telemetry.get_latest_image()
        if latest is not None:
            try:
                resized = cv2.resize(latest, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                resized = cv2.resize(latest, (img_w, img_h))
            # keep as uint8 (H, W)
            fr = resized.astype(np.uint8)
            if len(buffer) == 0:
                # initialize buffer with the first frame repeated so stack is full quickly
                for _ in range(frame_stack):
                    buffer.append(fr)
            else:
                buffer.append(fr)

        if saving and len(buffer) == frame_stack:
            stack = np.stack(list(buffer), axis=0)  # (F, H, W)
            save_stack_npz(stack, args.output_dir, args.prefix, saved)
            if args.save_png:
                save_stack_png(stack, args.output_dir, args.prefix + "_viz", saved)
            saved += 1
            if args.max_images is not None and saved >= args.max_images:
                print(f"Reached max_images={args.max_images}, stopping.")
                break

        time.sleep(args.capture_rate)

    print(f"Saved {saved} stacks to {args.output_dir}")


@cli_command(group="ac", name="save-images", help="Save grayscale screenshots from Assetto Corsa")
@cli_option("--output-dir", default="datasets/ac_images", help="Output directory")
@cli_option("--host", default="127.0.0.1", help="Telemetry host")
@cli_option("--recv-port", default=9876, help="Telemetry receive port")
@cli_option("--send-port", default=9877, help="Action send port")
@cli_option("--capture-rate", default=0.1, type=float, help="Seconds between captures")
@cli_option("--frame-stack", default=4, help="Number of frames to stack per sample")
@cli_option("--image-shape", default="84x84", help="Target image shape HxW")
@cli_option("--max-images", default=None, type=int, help="Maximum number of image stacks")
@cli_option("--prefix", default="frame", help="Filename prefix")
@cli_option("--save-png", is_flag=True, help="Also save PNG visualization")
def main(
    output_dir,
    host,
    recv_port,
    send_port,
    capture_rate,
    frame_stack,
    image_shape,
    max_images,
    prefix,
    save_png,
):
    from types import SimpleNamespace

    args = SimpleNamespace(
        output_dir=Path(output_dir),
        host=host,
        recv_port=recv_port,
        send_port=send_port,
        capture_rate=capture_rate,
        frame_stack=frame_stack,
        image_shape=image_shape,
        max_images=max_images,
        prefix=prefix,
        save_png=save_png,
    )

    with Telemetry(
        host=args.host,
        send_port=args.send_port,
        recv_port=args.recv_port,
        timeout=0.1,
        auto_start_receiver=True,
        capture_images=True,
        image_capture_rate=args.capture_rate,
    ) as telem:
        key_loop(telem, args)


if __name__ == "__main__":
    main()
