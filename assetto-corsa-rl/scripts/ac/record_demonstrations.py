"""Record human demonstrations for behavioral cloning.

This script records image observations and user inputs (steering, throttle, brake)
from Assetto Corsa to create a dataset for pretraining the SAC actor.

Usage:
    python scripts/ac/record_demonstrations.py --output-dir datasets/demonstrations --duration 300

Controls:
    - Press Ctrl+C to stop recording early
    - The script automatically saves data periodically
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

import numpy as np
import cv2

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from assetto_corsa_rl.ac_telemetry_helper import Telemetry
from assetto_corsa_rl.ac_env import parse_image_shape

try:
    from assetto_corsa_rl.cli_registry import cli_command, cli_option
except Exception:
    from ...src.assetto_corsa_rl.cli_registry import cli_command, cli_option


class DemonstrationRecorder:
    """Records human demonstrations from Assetto Corsa."""

    def __init__(
        self,
        output_dir: Path,
        image_shape: tuple = (84, 84),
        frame_stack: int = 4,
        save_interval: int = 100,
        min_speed_mph: float = 5.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_shape = image_shape
        self.frame_stack = frame_stack
        self.save_interval = save_interval
        self.min_speed_mph = min_speed_mph

        # Storage
        self.frames: list = []
        self.actions: list = []
        self.metadata: list = []

        # Frame buffer for stacking
        self.frame_buffer: list = []

        # Statistics
        self.total_frames = 0
        self.saved_samples = 0
        self.session_start = None

        # Telemetry
        self.telemetry: Optional[Telemetry] = None

    def start(self):
        """Initialize telemetry connection."""
        self.telemetry = Telemetry(
            host="127.0.0.1",
            send_port=9877,
            recv_port=9876,
            timeout=0.1,
            auto_start_receiver=True,
            capture_images=True,
            image_capture_rate=0.02,  # 50 FPS capture
        )
        self.session_start = datetime.now()
        print(f"✓ Telemetry started, recording to {self.output_dir}")

    def stop(self):
        """Stop recording and cleanup."""
        if self.telemetry:
            self.telemetry.close()
            self.telemetry = None

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Resize and normalize image."""
        if img is None:
            return np.zeros(self.image_shape, dtype=np.uint8)

        # Resize to target shape
        h, w = self.image_shape
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        # Ensure grayscale
        if resized.ndim == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        return resized.astype(np.uint8)

    def _get_stacked_frames(self, new_frame: np.ndarray) -> np.ndarray:
        """Maintain frame buffer and return stacked frames."""
        self.frame_buffer.append(new_frame)

        # Keep only last N frames
        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer = self.frame_buffer[-self.frame_stack :]

        # Pad with zeros if not enough frames yet
        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.insert(0, np.zeros_like(new_frame))

        # Stack frames: (N, H, W)
        return np.stack(self.frame_buffer, axis=0)

    def _extract_action(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract user inputs as action array [steering, throttle, brake]."""
        if data is None or "inputs" not in data:
            return None

        inputs = data["inputs"]
        steer = inputs.get("steer", 0.0)
        gas = inputs.get("gas", 0.0)
        brake = inputs.get("brake", 0.0)

        if steer is None or gas is None or brake is None:
            return None

        # Normalize steering to [-1, 1] (AC gives [-1, 1] already)
        # Gas and brake are already [0, 1]
        action = np.array([float(steer), float(gas), float(brake)], dtype=np.float32)
        return action

    def _should_record(self, data: Dict[str, Any]) -> bool:
        """Check if current frame should be recorded (e.g., car moving)."""
        if data is None:
            return False

        car = data.get("car", {})
        speed = car.get("speed_mph", 0)

        if speed is None or speed < self.min_speed_mph:
            return False

        # Don't record if in pit lane
        if car.get("in_pit_lane", False):
            return False

        # Don't record if damaged
        damage = car.get("damage", [0, 0, 0, 0, 0])
        if damage and sum(damage) > 0:
            return False

        return True

    def record_frame(self) -> bool:
        """Record a single frame. Returns True if successful."""
        if self.telemetry is None:
            return False

        # Get telemetry data
        data = self.telemetry.get_latest()
        if not self._should_record(data):
            return False

        # Get image
        img = self.telemetry.get_latest_image()
        if img is None:
            return False

        # Preprocess and stack
        processed = self._preprocess_image(img)
        stacked = self._get_stacked_frames(processed)

        # Extract action
        action = self._extract_action(data)
        if action is None:
            return False

        # Store
        self.frames.append(stacked.copy())
        self.actions.append(action.copy())
        self.metadata.append(
            {
                "timestamp": time.time(),
                "speed_mph": data.get("car", {}).get("speed_mph", 0),
                "lap": data.get("lap", {}).get("get_lap_count", 0),
            }
        )

        self.total_frames += 1

        # Save periodically
        if len(self.frames) >= self.save_interval:
            self._save_batch()

        return True

    def _save_batch(self):
        """Save current batch to disk."""
        if len(self.frames) == 0:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        batch_idx = self.saved_samples // self.save_interval

        # Save as .npz
        filename = f"demo_batch_{batch_idx:05d}_{timestamp}.npz"
        filepath = self.output_dir / filename

        np.savez_compressed(
            filepath,
            frames=np.array(self.frames, dtype=np.uint8),
            actions=np.array(self.actions, dtype=np.float32),
            # Don't save full metadata to save space, just essential stats
        )

        self.saved_samples += len(self.frames)
        print(
            f"✓ Saved {len(self.frames)} samples to {filename} (total: {self.saved_samples})"
        )

        # Clear buffers
        self.frames = []
        self.actions = []
        self.metadata = []

    def finalize(self):
        """Save any remaining data and session info."""
        # Save remaining frames
        if len(self.frames) > 0:
            self._save_batch()

        # Save session metadata
        session_info = {
            "session_start": (
                self.session_start.isoformat() if self.session_start else None
            ),
            "session_end": datetime.now().isoformat(),
            "total_samples": self.saved_samples,
            "image_shape": list(self.image_shape),
            "frame_stack": self.frame_stack,
            "min_speed_mph": self.min_speed_mph,
        }

        info_path = self.output_dir / "session_info.json"
        with open(info_path, "w") as f:
            json.dump(session_info, f, indent=2)

        print(f"\n{'=' * 50}")
        print(f"Recording complete!")
        print(f"Total samples: {self.saved_samples}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 50}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record human demonstrations from Assetto Corsa"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/demonstrations"),
        help="Directory to save recorded data",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Recording duration in seconds (0 = until Ctrl+C)",
    )
    parser.add_argument(
        "--image-shape",
        type=str,
        default="84x84",
        help="Image shape HxW (e.g., 84x84)",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Number of frames to stack",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Save batch every N frames",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=5.0,
        help="Minimum speed (mph) to record",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=20.0,
        help="Target recording FPS",
    )
    return parser.parse_args()


@cli_command(
    group="ac",
    name="record-demonstrations",
    help="Record human demonstrations for behavioral cloning",
)
@cli_option("--output-dir", default="datasets/demonstrations", help="Output directory")
@cli_option(
    "--duration", default=0, help="Recording duration in seconds (0 = until Ctrl+C)"
)
@cli_option("--image-shape", default="84x84", help="Image shape HxW")
@cli_option("--frame-stack", default=4, help="Number of frames to stack")
@cli_option("--save-interval", default=500, help="Save batch every N frames")
@cli_option(
    "--min-speed", default=5.0, type=float, help="Minimum speed (mph) to record"
)
@cli_option("--target-fps", default=20.0, type=float, help="Target recording FPS")
def main(
    output_dir, duration, image_shape, frame_stack, save_interval, min_speed, target_fps
):
    args = parse_args()

    # Parse image shape
    try:
        image_shape = parse_image_shape(args.image_shape)
    except ValueError as e:
        print(f"Error: {e}")
        return

    recorder = DemonstrationRecorder(
        output_dir=args.output_dir,
        image_shape=image_shape,
        frame_stack=args.frame_stack,
        save_interval=args.save_interval,
        min_speed_mph=args.min_speed,
    )

    print("=" * 50)
    print("Assetto Corsa Demonstration Recorder")
    print("=" * 50)
    print(f"Output: {args.output_dir}")
    print(f"Image shape: {image_shape}")
    print(f"Frame stack: {args.frame_stack}")
    print(f"Min speed: {args.min_speed} mph")
    print(f"Target FPS: {args.target_fps}")
    if args.duration > 0:
        print(f"Duration: {args.duration} seconds")
    else:
        print("Duration: Until Ctrl+C")
    print("=" * 50)

    input("\nPress Enter when Assetto Corsa is running and you're ready to record...")

    recorder.start()
    frame_interval = 1.0 / args.target_fps
    start_time = time.time()
    last_frame_time = start_time

    try:
        while True:
            current_time = time.time()

            # Check duration
            if args.duration > 0 and (current_time - start_time) >= args.duration:
                print("\nDuration reached, stopping...")
                break

            # Rate limiting
            elapsed = current_time - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue

            last_frame_time = current_time

            # Record frame
            success = recorder.record_frame()

            # Progress update
            if recorder.total_frames % 100 == 0 and recorder.total_frames > 0:
                elapsed_total = current_time - start_time
                fps = recorder.total_frames / elapsed_total if elapsed_total > 0 else 0
                status = "recording" if success else "waiting"
                print(
                    f"\rFrames: {recorder.total_frames} | "
                    f"Saved: {recorder.saved_samples} | "
                    f"FPS: {fps:.1f} | "
                    f"Status: {status}    ",
                    end="",
                )

    except KeyboardInterrupt:
        print("\n\nStopping recording...")

    finally:
        recorder.finalize()
        recorder.stop()


if __name__ == "__main__":
    main()
