"""Record human demonstrations for behavioral cloning.

This script records image observations and user inputs (steering, throttle, brake)
from Assetto Corsa to create a dataset for pretraining the SAC actor.

Usage:
    acrl ac record-demonstrations --config-path assetto-corsa-rl/configs/ac/env_config.yaml  --output-dir datasets/demonstrations2 --duration 999999999999 --display --display-scale 5.0

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

repo_root = Path(__file__).resolve().parents[2]
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from assetto_corsa_rl.ac_telemetry_helper import Telemetry  # type: ignore
from assetto_corsa_rl.ac_env import parse_image_shape  # type: ignore

try:
    from assetto_corsa_rl.cli_registry import cli_command, cli_option  # type: ignore
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
        input_config: Optional[Dict[str, bool]] = None,
        racing_line_path: Optional[str] = "racing_lines.json",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_shape = image_shape
        self.frame_stack = frame_stack
        self.save_interval = save_interval
        self.min_speed_mph = min_speed_mph
        self.input_config = input_config or {}
        self.observation_keys = [k for k, v in self.input_config.items() if v]
        self.racing_line_path = racing_line_path

        self.frames: list = []
        self.actions: list = []
        self.observations: list = []
        self.rewards: list = []
        self.metadata: list = []

        self.frame_buffer: list = []

        self.total_frames = 0
        self.saved_samples = 0
        self.session_start = None

        self.telemetry: Optional[Telemetry] = None

        from assetto_corsa_rl.ac_env import AssettoCorsa

        self._env_helper = AssettoCorsa.__new__(AssettoCorsa)

        try:
            self._env_helper._load_racing_line(racing_line_path)
            self._env_helper.constant_reward_per_ms = 0.01
            self._env_helper.reward_per_m_advanced_along_centerline = 1.0
            self._env_helper.final_speed_reward_per_m_per_s = 0.1
            self._env_helper.ms_per_action = 20.0
            self._env_helper._meters_advanced = 0.0
            self._env_helper._last_speed = 0.0
            self._env_helper._current_racing_line_index = 0
            # Track last reset time to avoid repeated resets
            self._last_reset_time = 0.0
            self._reset_cooldown_seconds = 2.0
            print(f"✓ Using environment reward function with racing line")
        except Exception as e:
            print(f"Warning: Could not load racing line: {e}")

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

    def _handle_lap_reset(self):
        """Handle environment reset when a lap threshold is reached.

        Saves any pending batch, clears buffers, and resets environment helper state
        so recording can continue fresh on the next lap.
        """
        print("Lap count reached 2 — saving batch and resetting environment state")
        try:
            self._save_batch()
        except Exception as e:
            print(f"Warning: Failed to save batch during lap reset: {e}")

        self.frame_buffer = []

        try:
            self._env_helper._meters_advanced = 0.0
            self._env_helper._last_speed = 0.0
            self._env_helper._current_racing_line_index = 0
        except Exception:
            pass

        try:
            if self.telemetry:
                self.telemetry.send_reset()
                try:
                    self.telemetry.clear_queue()
                except Exception:
                    pass
                time.sleep(0.5)
        except Exception as e:
            print(f"Warning: failed to send telemetry reset: {e}")

        self._last_reset_time = time.time()

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Resize and normalize image."""
        if img is None:
            return np.zeros(self.image_shape, dtype=np.uint8)

        h, w = self.image_shape
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        if resized.ndim == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        return resized.astype(np.uint8)

    def _get_stacked_frames(self, new_frame: np.ndarray) -> np.ndarray:
        """Maintain frame buffer and return stacked frames."""
        self.frame_buffer.append(new_frame)

        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer = self.frame_buffer[-self.frame_stack :]

        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.insert(0, np.zeros_like(new_frame))

        return np.stack(self.frame_buffer, axis=0)

    def _extract_observation(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract observation values from telemetry data based on input_config."""
        if data is None or not self.observation_keys:
            return None

        obs_values = [
            self._env_helper._extract_value_from_data(key, data) for key in self.observation_keys
        ]
        return np.array(obs_values, dtype=np.float32)

    def _calculate_reward(self, data: Dict[str, Any]) -> float:
        """Calculate reward from telemetry data.

        Uses environment's reward function if racing line is available,
        otherwise uses simple speed-based reward.
        """
        obs = np.array([], dtype=np.float32)
        return self._env_helper._calculate_reward(obs, data)

    def _extract_action(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract user inputs as action array [steering, throttle, brake]."""
        if data is None or "inputs" not in data:
            return None

        inputs = data["inputs"]
        steer = inputs.get("steer", 0.0)
        steer = (
            max(-1.0, min(1.0, float(steer) / 260.0))
            if abs(float(steer)) <= 260
            else float(steer) / abs(float(steer))
        )
        gas = inputs.get("gas", 0.0)
        brake = inputs.get("brake", 0.0)

        if steer is None or gas is None or brake is None:
            return None

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

        if car.get("in_pit_lane", False):
            return False

        return True

    def record_frame(self, return_frame: bool = False):
        """Record a single frame. Optionally returns (success, stacked_frame, action)."""
        if self.telemetry is None:
            return (False, None, None) if return_frame else False

        data = self.telemetry.get_latest()
        if not self._should_record(data):
            return (False, None, None) if return_frame else False

        img = self.telemetry.get_latest_image()
        if img is None:
            return (False, None, None) if return_frame else False

        processed = self._preprocess_image(img)
        stacked = self._get_stacked_frames(processed)

        # avoids saving samples that contain padding frames at start or
        # frames where the capture failed and returned an empty image.
        if np.any(np.all(stacked == 0, axis=(1, 2))):
            return (False, None, None) if return_frame else False

        action = self._extract_action(data)
        if action is None:
            return (False, None, None) if return_frame else False

        # save and reset
        lap_count = None
        lap_info = data.get("lap") if isinstance(data, dict) else None
        if isinstance(lap_info, dict):
            lap_count = (
                lap_info.get("get_lap_count") or lap_info.get("lap_count") or lap_info.get("count")
            )
        if lap_count is None:
            lap_count = data.get("lap_count") if isinstance(data, dict) else None

        try:
            if lap_count is not None and int(lap_count) == 2:
                # Avoid repeated resets by enforcing a cooldown
                if time.time() - getattr(self, "_last_reset_time", 0.0) > getattr(
                    self, "_reset_cooldown_seconds", 2.0
                ):
                    self._handle_lap_reset()
                    return (False, None, None) if return_frame else False
                else:
                    # recently reset; skip recording this frame
                    return (False, None, None) if return_frame else False
        except Exception:
            pass

        observation = self._extract_observation(data)

        reward = self._calculate_reward(data)

        self.frames.append(stacked.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        if observation is not None:
            self.observations.append(observation.copy())
        self.metadata.append(
            {
                "timestamp": time.time(),
                "speed_mph": data.get("car", {}).get("speed_mph", 0),
                "lap": data.get("lap", {}).get("get_lap_count", 0),
                "reward": reward,
            }
        )

        self.total_frames += 1

        if len(self.frames) >= self.save_interval:
            self._save_batch()

        if return_frame:
            return True, stacked, action
        return True

    def _save_batch(self):
        """Save current batch to disk."""
        if len(self.frames) == 0:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        batch_idx = self.saved_samples // self.save_interval

        filename = f"demo_batch_{batch_idx:05d}_{timestamp}.npz"
        filepath = self.output_dir / filename

        save_dict = {
            "frames": np.array(self.frames, dtype=np.uint8),
            "actions": np.array(self.actions, dtype=np.float32),
            "rewards": np.array(self.rewards, dtype=np.float32),
        }

        if self.observations:
            save_dict["observations"] = np.array(self.observations, dtype=np.float32)
            save_dict["observation_keys"] = np.array(self.observation_keys, dtype=object)

        np.savez_compressed(filepath, **save_dict)

        self.saved_samples += len(self.frames)
        obs_info = f" with {len(self.observation_keys)} obs dims" if self.observations else ""
        print(
            f"✓ Saved {len(self.frames)} samples{obs_info} to {filename} (total: {self.saved_samples})"
        )

        self.frames = []
        self.actions = []
        self.rewards = []
        self.observations = []
        self.metadata = []

    def finalize(self):
        """Save any remaining data and session info."""
        print(f"\n{'=' * 50}")
        print(f"Recording complete!")
        print(f"Total samples: {self.saved_samples}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 50}")


def parse_args():
    parser = argparse.ArgumentParser(description="Record human demonstrations from Assetto Corsa")
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
@cli_option("--duration", default=0, help="Recording duration in seconds (0 = until Ctrl+C)")
@cli_option("--image-shape", default="84x84", help="Image shape HxW")
@cli_option("--frame-stack", default=4, help="Number of frames to stack")
@cli_option("--save-interval", default=500, help="Save batch every N frames")
@cli_option("--min-speed", default=5.0, type=float, help="Minimum speed (mph) to record")
@cli_option("--target-fps", default=20.0, type=float, help="Target recording FPS")
@cli_option("--display", is_flag=True, help="Show live frames and input values in a window")
@cli_option("--display-scale", default=2.0, type=float, help="Scale factor for display window")
@cli_option(
    "--config-path", default=None, help="Path to env config YAML (loads input configuration)"
)
@cli_option(
    "--racing-line-path",
    default="racing_lines.json",
    help="Path to racing line JSON (enables env reward function)",
)
def main(
    output_dir,
    duration,
    image_shape,
    frame_stack,
    save_interval,
    min_speed,
    target_fps,
    display,
    display_scale,
    config_path,
    racing_line_path,
):
    try:
        parsed_image_shape = parse_image_shape(image_shape)
    except ValueError as e:
        print(f"Error: {e}")
        return

    input_config = None
    if config_path:
        try:
            import yaml
            from pathlib import Path

            cfg_path = Path(config_path)
            with open(cfg_path, "r") as f:
                cfg_data = yaml.safe_load(f)
                input_config = cfg_data.get("environment", {}).get("inputs", {})
                if input_config:
                    enabled_count = sum(1 for v in input_config.values() if v)
                    print(
                        f"✓ Loaded input config: {enabled_count}/{len(input_config)} inputs enabled"
                    )
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            input_config = None

    recorder = DemonstrationRecorder(
        output_dir=output_dir,
        image_shape=parsed_image_shape,
        frame_stack=frame_stack,
        save_interval=save_interval,
        min_speed_mph=min_speed,
        input_config=input_config,
        racing_line_path=racing_line_path,
    )

    print("=" * 50)
    print("Assetto Corsa Demonstration Recorder")
    print("=" * 50)
    print(f"Output: {output_dir}")
    print(f"Image shape: {parsed_image_shape}")
    print(f"Frame stack: {frame_stack}")
    print(f"Min speed: {min_speed} mph")
    print(f"Target FPS: {target_fps}")
    if duration > 0:
        print(f"Duration: {duration} seconds")
    else:
        print("Duration: Until Ctrl+C")
    print("=" * 50)

    input("\nPress Enter when Assetto Corsa is running and you're ready to record...")

    recorder.start()
    frame_interval = 1.0 / target_fps
    start_time = time.time()
    last_frame_time = start_time

    window_name = "AC Demo Recorder" if display else None
    if display and window_name:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = False

    try:
        while True:
            current_time = time.time()

            if duration > 0 and (current_time - start_time) >= duration:
                print("\nDuration reached, stopping...")
                break

            # raet limiting
            elapsed = current_time - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue

            last_frame_time = current_time

            if paused:
                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("\nQuit requested (q pressed)")
                        break
                    if key == ord("p"):
                        paused = not paused
                        print("\nResumed recording")
                    if key == ord("d"):
                        recorder.frames = []
                        recorder.actions = []
                        recorder.rewards = []
                        recorder.observations = []
                        recorder.metadata = []
                        recorder.frame_buffer = []
                        print("\nCleared unsaved batch")
                continue

            # record frame
            result = recorder.record_frame(return_frame=display)
            if display:
                success, stacked_frame, action = (
                    result if isinstance(result, tuple) else (False, None, None)
                )
            else:
                success = bool(result)

            if display and success and stacked_frame is not None and action is not None:
                frame_to_show = stacked_frame[-1]
                vis = cv2.cvtColor(frame_to_show, cv2.COLOR_GRAY2BGR)

                # Resize first so text scales appropriately and remains legible
                if display_scale and display_scale != 1.0:
                    h, w = vis.shape[:2]
                    new_w = max(1, int(w * display_scale))
                    new_h = max(1, int(h * display_scale))
                    vis = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # Render action text with font size proportional to display scale
                text = f"{action[0]:.2f} {action[1]:.2f} {action[2]:.2f}"
                base_font = 0.5
                font_scale = base_font * max(1.0, display_scale)
                thickness = max(1, int(round(font_scale * 2)))

                (text_w, text_h), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                pad = 6
                rect_x1, rect_y1 = 8, 8
                rect_x2, rect_y2 = rect_x1 + text_w + pad, rect_y1 + text_h + pad

                # draw semi-opaque background for readability
                cv2.rectangle(vis, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), cv2.FILLED)

                text_org = (rect_x1 + pad // 2, rect_y1 + text_h + pad // 2 - baseline)
                cv2.putText(
                    vis,
                    text,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    thickness,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nQuit requested (q pressed)")
                    break
                if key == ord("p"):
                    paused = not paused
                    print("\nPaused recording" if paused else "\nResumed recording")
                if key == ord("d"):
                    recorder.frames = []
                    recorder.actions = []
                    recorder.rewards = []
                    recorder.observations = []
                    recorder.metadata = []
                    recorder.frame_buffer = []
                    print("\nCleared unsaved batch")

            if recorder.total_frames % 100 == 0 and recorder.total_frames > 0:
                elapsed_total = current_time - start_time
                fps = recorder.total_frames / elapsed_total if elapsed_total > 0 else 0
                status = "paused" if paused else ("recording" if success else "waiting")
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
        if display:
            cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
