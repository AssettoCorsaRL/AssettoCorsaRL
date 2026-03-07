import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
import time
import json
from pathlib import Path
import cv2
import torch

try:
    from .ac_send_actions import XboxController
    from .ac_telemetry_helper import Telemetry
except:
    from ac_send_actions import XboxController
    from ac_telemetry_helper import Telemetry


class AssettoCorsa(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        host: str = "127.0.0.1",
        send_port: int = 9877,
        recv_port: int = 9876,
        max_episode_steps: int = 1000,
        timeout: float = 1.0,
        observation_keys: Optional[list] = None,
        input_config: Optional[Dict[str, bool]] = None,
        racing_line_path: str = "racing_lines.json",
        constant_reward_per_ms: float = 0,
        reward_per_m_advanced_along_centerline: float = 1.0,
        final_speed_reward_per_m_per_s: float = 0.05,
        include_image: bool = False,
        use_ac_ai_racer: bool = True,
        observation_image_shape: Tuple[int, int] = (84, 84),
        normalize_observations: bool = True,
        normalization_bounds: Optional[Dict[str, list]] = None,
    ):
        super().__init__()

        self.host = host
        self.send_port = send_port
        self.recv_port = recv_port
        self.max_episode_steps = max_episode_steps
        self.timeout = timeout
        self.ai_racer = use_ac_ai_racer

        self.constant_reward_per_ms = constant_reward_per_ms
        self.reward_per_m_advanced_along_centerline = reward_per_m_advanced_along_centerline
        self.final_speed_reward_per_m_per_s = final_speed_reward_per_m_per_s

        self.normalize_observations = normalize_observations
        self.normalization_bounds = normalization_bounds or {}

        self.racing_line = None
        self.racing_line_positions = None
        self._load_racing_line(racing_line_path)

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

        if input_config is not None:
            self.input_config = input_config
            self.observation_keys = self._build_observation_keys_from_config(input_config)
        elif observation_keys is not None:
            self.observation_keys = observation_keys
            self.input_config = None
        else:
            self.observation_keys = []
            self.input_config = None

        self.include_image = include_image
        self.observation_image_shape = observation_image_shape  # (H, W)

        vector_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.observation_keys),),
            dtype=np.float32,
        )

        if self.include_image:
            img_h, img_w = self.observation_image_shape
            image_space = spaces.Box(
                low=0,
                high=255,
                shape=(img_h, img_w, 1),
                dtype=np.uint8,
            )
            self.observation_space = spaces.Dict({"vector": vector_space, "image": image_space})
        else:
            self.observation_space = vector_space

        self.controller = XboxController()
        self.telemetry = Telemetry(
            host=host,
            send_port=send_port,
            recv_port=recv_port,
            timeout=0.1,
            auto_start_receiver=True,
            capture_images=self.include_image,
        )

        self._episode_step = 0
        self._last_obs = None

        self._meters_advanced = 0.0
        self._last_speed = 0.0
        self._current_racing_line_index = 0

    def _build_observation_keys_from_config(self, input_config: Dict[str, bool]) -> list:
        """Build list of observation keys from input configuration dict.

        Args:
            input_config: Dict mapping input names to boolean (True = include in observation)

        Returns:
            List of observation key strings in consistent order
        """
        keys = []

        for key, enabled in input_config.items():
            if enabled:
                keys.append(key)

        return keys

    def _get_observation(self):
        """Extract observation from telemetry data and optional screenshot image.

        Returns:
            If `include_image` is True: dict with keys `vector` (np.ndarray float32) and `image` (uint8 HxWx1)
            Otherwise: np.ndarray float32 vector
        """
        data = self.telemetry.get_latest()

        img = None
        if self.include_image:
            latest_img = self.telemetry.get_latest_image()
            img_h, img_w = self.observation_image_shape
            if latest_img is None:
                img = np.zeros((img_h, img_w, 1), dtype=np.uint8)
            else:
                try:
                    resized = cv2.resize(latest_img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    resized = cv2.resize(latest_img, (img_w, img_h))
                img = resized[..., np.newaxis].astype(np.uint8)

        if data is None:
            vector = np.zeros(len(self.observation_keys), dtype=np.float32)
            self._last_obs = None
        else:
            self._last_obs = data
            vector = np.array(
                [self._extract_value_from_data(key, data) for key in self.observation_keys],
                dtype=np.float32,
            )
            # Apply normalization if enabled
            if self.normalize_observations:
                vector = self._normalize_vector(vector)

        if self.include_image:
            return {"vector": vector, "image": img}
        return vector

    def _extract_value_from_data(self, key: str, data: Dict[str, Any]) -> float:
        car = data["car"]
        inputs = data["inputs"]
        lap = data["lap"]
        session = data["session"]
        stats = data["stats"]
        tyres = data["tyres"]

        # TODO: verify all these work

        match key:
            case "speed_kmh":
                return float(car["speed_kmh"])
            case "speed_mph":
                return float(car["speed_mph"])
            case "speed_ms":
                return float(car["speed_ms"])
            case "rpm":
                return float(car["rpm"])
            case "fuel":
                return float(car["fuel"])
            case "tyres_off_track":
                return float(car["tyres_off_track"])
            case "in_pit_lane":
                return float(car["in_pit_lane"])
            case "cg_height":
                return float(car["cg_height"])
            case "drive_train_speed":
                return float(car["drive_train_speed"])
            case "drs_available":
                return float(car["drs_available"])
            case "drs_enabled":
                return float(car["drs_enabled"])

            case "gear":
                gear = car["gear"]
                if gear == "R":
                    return -1.0
                if gear == "N":
                    return 0.0
                return float(gear)

            case "damage":
                return float(sum(car["damage"]))

            case "velocity_x":
                return float(car["velocity"][0])
            case "velocity_y":
                return float(car["velocity"][1])
            case "velocity_z":
                return float(car["velocity"][2])

            case "acceleration_x":
                return float(car["acceleration"][0])
            case "acceleration_y":
                return float(car["acceleration"][1])
            case "acceleration_z":
                return float(car["acceleration"][2])

            case "gas":
                return float(inputs["gas"])
            case "brake":
                return float(inputs["brake"])
            case "clutch":
                return float(inputs["clutch"])
            case "steer":
                return float(inputs["steer"])

            case "current_lap_time":
                return float(lap["get_current_lap_time"])
            case "last_lap_time":
                return float(lap["get_last_lap_time"])
            case "best_lap_time":
                return float(lap["get_best_lap_time"])
            case "lap_count":
                return float(lap["get_lap_count"])
            case "lap_delta":
                return float(lap["get_lap_delta"])
            case "current_sector":
                return float(lap["get_current_sector"])
            case "invalid_lap":
                return float(lap["get_invalid"])

            case "track_length":
                return float(session["track_length"])
            case "air_temp":
                return float(session["air_temp"])
            case "road_temp":
                return float(session["road_temp"])
            case "session_status":
                return float(session["session_status"])

            case "has_drs":
                return float(stats["has_drs"])
            case "has_ers":
                return float(stats["has_ers"])
            case "has_kers":
                return float(stats["has_kers"])

            case "tyre_wear_avg":
                return float(np.mean([t["wear"] for t in tyres]))
            case "tyre_pressure_avg":
                return float(np.mean([t["pressure"] for t in tyres]))
            case "tyre_temp_avg":
                return float(np.mean([t["temp_m"] for t in tyres]))
            case "tyre_dirty_avg":
                return float(np.mean([t["dirty"] for t in tyres]))

            case "tyre_slip_ratio_avg":
                # Handle mixed array/scalar values and -1 sentinel values
                slip_values = []
                for t in tyres:
                    val = t.get("slip_ratio", 0.0)
                    # Convert arrays to first element, handle -1 sentinel
                    if isinstance(val, (list, np.ndarray)):
                        scalar_val = float(val[0]) if len(val) > 0 else 0.0
                    else:
                        scalar_val = float(val) if val != -1 else 0.0
                    slip_values.append(scalar_val)
                return float(np.mean(slip_values))
            case "tyre_slip_angle_avg":
                angle_values = []
                for t in tyres:
                    val = t.get("slip_angle", 0.0)
                    if isinstance(val, (list, np.ndarray)):
                        scalar_val = float(val[0]) if len(val) > 0 else 0.0
                    else:
                        scalar_val = float(val) if val != -1 else 0.0
                    angle_values.append(scalar_val)
                return float(np.mean(angle_values))

            case _ if key.startswith("tyre_") and "_temp_" in key:
                _, tyre_idx, _, temp_pos = key.split("_")
                return float(tyres[int(tyre_idx)][f"temp_{temp_pos}"])

            case _:
                return float(data[key])

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize observation vector to [-1, 1] range using normalization bounds.

        Args:
            vector: Observation vector of shape (obs_dim,)

        Returns:
            Normalized observation vector clipped to [-1, 1]
        """
        if not self.normalize_observations or not self.normalization_bounds:
            return vector

        normalized = np.copy(vector)
        for i, key in enumerate(self.observation_keys):
            if key not in self.normalization_bounds:
                continue

            bounds = self.normalization_bounds[key]
            if len(bounds) != 2:
                continue

            min_val, max_val = bounds
            if max_val <= min_val:
                continue

            # Normalize to [-1, 1]: 2 * (x - min) / (max - min) - 1
            normalized[i] = 2.0 * (vector[i] - min_val) / (max_val - min_val) - 1.0

        return normalized

    def _load_racing_line(self, filepath: str) -> None:
        """Load racing line from JSON file or URL.

        The `filepath` may be a local path or an HTTP/HTTPS URL.  When a URL is
        provided we download the JSON and parse it in-memory; local files are
        read normally.
        """
        # support remote files so users can point at our hosted lines
        if filepath.startswith("http://") or filepath.startswith("https://"):
            try:
                import requests

                resp = requests.get(filepath, timeout=10)
                resp.raise_for_status()
                self.racing_line = resp.json()
            except Exception as e:  # pylint: disable=broad-except
                raise IOError(f"Failed to download racing line from {filepath}: {e}")
        else:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"Racing line file not found: {filepath}")

            with open(path, "r") as f:
                self.racing_line = json.load(f)

        if self.racing_line.get("num_laps", 0) == 0:
            raise ValueError("Racing line file contains no laps")

        lap = self.racing_line["laps"][0]
        positions = np.array([[p["x"], p["y"], p["z"]] for p in lap["positions"]])
        self.racing_line_positions = positions

        # avoid unicode characters in logs to prevent encoding issues on some consoles
        print(f"Loaded racing line with {len(positions)} points")

    def _find_closest_point_on_racing_line(
        self, position: np.ndarray, search_window: int = 100
    ) -> Tuple[int, float]:
        if self.racing_line_positions is None:
            return 0, 0.0

        n = len(self.racing_line_positions)
        if search_window == -1 or search_window >= n:
            indices = np.arange(n)
        else:
            center = self._current_racing_line_index
            indices = np.arange(center - search_window, center + search_window + 1) % n

        candidates = self.racing_line_positions[indices]
        distances = np.linalg.norm(candidates - position, axis=1)
        local_idx = np.argmin(distances)
        closest_idx = int(indices[local_idx])
        return closest_idx, float(distances[local_idx])

    def _calculate_meters_advanced(self, position: np.ndarray) -> float:
        if self.racing_line_positions is None:
            return 0.0

        closest_idx, _ = self._find_closest_point_on_racing_line(position, search_window=100)

        # if self._current_racing_line_index == 0:
        # self._current_racing_line_index = closest_idx

        idx_diff = closest_idx - self._current_racing_line_index
        n = len(self.racing_line_positions)
        if idx_diff < -(n // 2):
            idx_diff += n
        elif idx_diff > n // 2:
            idx_diff -= n

        if idx_diff < 0:
            return self._meters_advanced

        # Project position onto the segment between closest and next point
        next_idx = (closest_idx + 1) % n
        seg_start = self.racing_line_positions[closest_idx]
        seg_end = self.racing_line_positions[next_idx]
        seg = seg_end - seg_start
        seg_len = np.linalg.norm(seg)

        if seg_len > 0:
            t = np.clip(np.dot(position - seg_start, seg) / (seg_len**2), 0.0, 1.0)
        else:
            t = 0.0

        # Arc length up to closest_idx
        segments = (
            self.racing_line_positions[1 : closest_idx + 1]
            - self.racing_line_positions[0:closest_idx]
        )
        arc = float(np.sum(np.linalg.norm(segments, axis=1)))

        # Add fractional progress along current segment
        total = arc + t * seg_len

        if total < self._meters_advanced:
            return self._meters_advanced

        self._current_racing_line_index = closest_idx
        return total

    # inspired by linesight-rl: https://github.com/Linesight-RL/linesight/tree/main
    #! UNTESTED WITH PRETRAINED AGENTS
    def _calculate_reward(self, obs, data):
        position = np.array(data["car"]["world_location"][:3])
        current_meters = self._calculate_meters_advanced(position)
        meters_progress = current_meters - self._meters_advanced
        self._meters_advanced = current_meters

        velocity = data.get("car", {}).get("velocity", [0, 0, 0])
        speed = np.linalg.norm(velocity)  # m/s

        off_track = float(data.get("car", {}).get("tyres_off_track", 0))
        damage = sum(data["car"].get("damage", [0]))

        reward = (
            meters_progress * 1.0  # progress along racing line
            + speed * 0.01  # bonus for going fast (always positive signal)
            - off_track * 0.5  # penalty for being off track
            - (1.0 if damage > 0 else 0.0)  # terminal penalty for damage
        )
        self._last_speed = speed
        return reward

    def _check_done(self, obs: np.ndarray, data: Optional[Dict]) -> bool:
        if self._episode_step >= self.max_episode_steps:
            return True

        if data is None or data.get("car") is None or data.get("lap") is None:
            return False

        if data["lap"]["get_lap_count"] == 2:
            return True

        if sum(data["car"]["damage"]) > 0:
            return True

        return False

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Sends reset command to AC and waits for telemetry.
        """
        super().reset(seed=seed)

        self._current_racing_line_index = 0
        self._meters_advanced = 0.0
        self._last_speed = 0.0

        self.controller.reset()
        self.controller.update()

        self.telemetry.send_reset()
        self.telemetry.clear_queue()

        time.sleep(0.1)

        self._episode_step = 0
        obs = self._get_observation()

        if self.racing_line_positions is not None and self._last_obs:
            position = np.array(
                [
                    self._last_obs.get("car", {}).get("world_location", [0, 0, 0])[0],
                    self._last_obs.get("car", {}).get("world_location", [0, 0, 0])[1],
                    self._last_obs.get("car", {}).get("world_location", [0, 0, 0])[2],
                ]
            )
            self._meters_advanced = self._calculate_meters_advanced(position)
        else:
            self._meters_advanced = 0.0

        if self._last_obs:
            velocity = self._last_obs.get("car", {}).get("velocity", [0, 0, 0])
            self._last_speed = np.linalg.norm(velocity)
        else:
            self._last_speed = 0.0

        info = {"episode_step": self._episode_step}

        if self.ai_racer:
            success = self.telemetry.send_ctrl_c()
            if not success:
                print("Warning: send_ctrl_c failed during env.reset()")

        time.sleep(1.5)

        obs = self._get_observation()

        if self.racing_line_positions is not None and self._last_obs:
            position = np.array(self._last_obs.get("car", {}).get("world_location", [0, 0, 0]))
            # full search to find true spawn index
            closest_idx, _ = self._find_closest_point_on_racing_line(position, search_window=-1)
            self._current_racing_line_index = closest_idx
            # compute arc to spawn so first step reward starts from 0 delta
            segments = (
                self.racing_line_positions[1 : closest_idx + 1]
                - self.racing_line_positions[0:closest_idx]
            )
            self._meters_advanced = float(np.sum(np.linalg.norm(segments, axis=1)))

        if self._last_obs:
            velocity = self._last_obs.get("car", {}).get("velocity", [0, 0, 0])
            self._last_speed = np.linalg.norm(velocity)

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: [steering, throttle, brake] in normalized ranges

        Returns:
            observation, reward, terminated, truncated, info
        """
        action = np.asarray(action).flatten()

        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        self.controller.left_joystick_float(x_value_float=steering, y_value_float=0.0)
        self.controller.right_trigger_float(value_float=throttle)
        self.controller.left_trigger_float(value_float=brake)
        self.controller.update()

        time.sleep(0.02)

        obs = self._get_observation()
        data = self._last_obs

        reward = self._calculate_reward(obs, data)

        terminated = self._check_done(obs, data)
        truncated = False

        self._episode_step += 1

        info = {
            "episode_step": self._episode_step,
            "steering": steering,
            "throttle": throttle,
            "brake": brake,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up resources."""
        self.controller.reset()
        self.controller.update()
        self.telemetry.close()


def make_env(
    host: str = "127.0.0.1",
    send_port: int = 9877,
    recv_port: int = 9876,
    racing_line_path: str = "racing_lines.json",
    input_config: Optional[Dict[str, bool]] = None,
    **kwargs,
) -> AssettoCorsa:
    """Create an AssettoCorsa environment with default settings.

    Args:
        host: Telemetry host address
        send_port: Port for sending actions
        recv_port: Port for receiving telemetry
        racing_line_path: Path to racing line JSON file (required)
        input_config: Dict mapping input names to bool (True = include in observation)
        **kwargs: Additional arguments passed to AssettoCorsa
    """
    return AssettoCorsa(
        host=host,
        send_port=send_port,
        recv_port=recv_port,
        racing_line_path=racing_line_path,
        input_config=input_config,
        **kwargs,
    )


def parse_image_shape(shape_str: str) -> Tuple[int, int]:
    """Parse image shape string like '84x84' into (height, width) tuple.

    Args:
        shape_str: String in format 'HxW' (e.g., '84x84')

    Returns:
        Tuple of (height, width)

    Raises:
        ValueError: If format is invalid
    """
    try:
        parts = shape_str.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Expected format 'HxW', got '{shape_str}'")
        height, width = int(parts[0]), int(parts[1])
        return height, width
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid image shape format '{shape_str}'. Expected format like '84x84'"
        ) from e


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get torch device, defaulting to CUDA if available.

    Args:
        device_str: Optional device string ('cuda', 'cpu', 'mps', etc.)

    Returns:
        torch.device instance
    """
    if device_str is not None:
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_transformed_env(
    racing_line_path: str = "racing_lines.json",
    device: Optional[torch.device] = None,
    host: str = "127.0.0.1",
    send_port: int = 9877,
    recv_port: int = 9876,
    image_shape: Tuple[int, int] = (84, 84),
    frame_stack: int = 3,
    input_config: Optional[Dict[str, bool]] = None,
    **env_kwargs,
):
    """Create an AssettoCorsa environment with standard transforms for vision-based RL.

    Args:
        racing_line_path: Path to racing line JSON file
        device: Torch device (defaults to CUDA if available)
        host: Telemetry host address
        send_port: Port for sending actions
        recv_port: Port for receiving telemetry
        image_shape: Target image dimensions (H, W)
        frame_stack: Number of frames to stack
        input_config: Dict mapping input names to bool (True = include in observation)
        **env_kwargs: Additional arguments passed to AssettoCorsa

    Returns:
        TransformedEnv instance ready for training/testing
    """
    from torchrl.envs import GymWrapper, TransformedEnv
    from torchrl.envs.transforms import (
        Compose,
        ToTensorImage,
        Resize,
        CatFrames,
        RenameTransform,
    )

    if device is None:
        device = get_device()

    h, w = image_shape

    env = AssettoCorsa(
        host=host,
        send_port=send_port,
        recv_port=recv_port,
        racing_line_path=racing_line_path,
        include_image=True,
        observation_image_shape=(h, w),
        input_config=input_config,
        **env_kwargs,
    )

    base_env = GymWrapper(env, device=device)
    transform = Compose(
        RenameTransform(in_keys=["image"], out_keys=["pixels"]),
        ToTensorImage(from_int=True, in_keys=["pixels"], out_keys=["pixels"]),
        Resize(h=h, w=w, in_keys=["pixels"], out_keys=["pixels"]),
        CatFrames(N=frame_stack, in_keys=["pixels"], out_keys=["pixels"], dim=-3),
    )

    return TransformedEnv(base_env, transform)


def create_mock_env(device: Optional[torch.device] = None):
    """Create a minimal mock environment for initializing models without AC running.

    Useful for loading pretrained models or initializing architectures.

    Args:
        device: Torch device (defaults to CUDA if available)

    Returns:
        Mock environment object with action_spec and observation_spec
    """
    from torchrl.data import (
        Bounded,  # replaces BoundedTensorSpec
        Composite,  # still valid in 0.11, but will move to Composite later
        UnboundedContinuous,
    )

    if device is None:
        device = get_device()

    class MockEnv:
        def __init__(self, device):
            self.device = device
            self.action_spec = Bounded(
                low=-1.0, high=1.0, shape=(3,), dtype=torch.float32, device=device
            )
            self.observation_spec = Composite(
                pixels=UnboundedContinuous(shape=(4, 84, 84), dtype=torch.float32, device=device)
            )

    return MockEnv(device)


def main() -> None:
    """Test observation normalization with actual AC environment."""
    import argparse

    parser = argparse.ArgumentParser(description="Test AC environment observation normalization.")
    parser.add_argument("--racing-line", type=str, default="racing_lines.json")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--send-port", type=int, default=9877)
    parser.add_argument("--recv-port", type=int, default=9876)
    parser.add_argument("--normalize", action="store_true", help="Enable normalization")
    args = parser.parse_args()

    # Include all observations from env_config
    input_config = {
        # car dynamics
        "speed_kmh": True,
        "speed_mph": False,
        "speed_ms": False,
        "rpm": True,
        "gear": True,
        "velocity_x": True,
        "velocity_y": True,
        "velocity_z": True,
        "acceleration_x": True,
        "acceleration_y": True,
        "acceleration_z": True,
        # car state
        "fuel": False,
        "tyres_off_track": True,
        "in_pit_lane": False,
        "damage": False,
        "cg_height": False,
        "drive_train_speed": False,
        # DRS/ERS/KERS
        "drs_available": False,
        "drs_enabled": False,
        "has_drs": False,
        "has_ers": False,
        "has_kers": False,
        # inputs (current)
        "gas": True,
        "brake": True,
        "clutch": True,
        "steer": True,
        # lap info
        "current_lap_time": False,
        "last_lap_time": False,
        "best_lap_time": False,
        "lap_count": False,
        "lap_delta": False,
        "current_sector": False,
        "invalid_lap": False,
        # Track/Session info
        "track_length": False,
        "air_temp": False,
        "road_temp": False,
        "session_status": False,
        # Tyre info (per-tyre averages)
        "tyre_wear_avg": True,
        "tyre_pressure_avg": True,
        "tyre_temp_avg": True,
        "tyre_dirty_avg": False,
        "tyre_slip_ratio_avg": True,
        "tyre_slip_angle_avg": True,
        # Individual tyre temps (front-left, front-right, rear-left, rear-right)
        "tyre_0_temp_i": True,
        "tyre_0_temp_m": True,
        "tyre_0_temp_o": True,
        "tyre_1_temp_i": True,
        "tyre_1_temp_m": True,
        "tyre_1_temp_o": True,
        "tyre_2_temp_i": True,
        "tyre_2_temp_m": True,
        "tyre_2_temp_o": True,
        "tyre_3_temp_i": True,
        "tyre_3_temp_m": True,
        "tyre_3_temp_o": True,
    }

    normalization_bounds = {
        "speed_kmh": [0, 300],
        "speed_mph": [0, 186],
        "speed_ms": [0, 83],
        "rpm": [0, 8000],
        "gear": [0, 8],
        "velocity_x": [-50, 50],
        "velocity_y": [-50, 50],
        "velocity_z": [-50, 50],
        "acceleration_x": [-30, 30],
        "acceleration_y": [-30, 30],
        "acceleration_z": [-30, 30],
        "fuel": [0, 100],
        "tyres_off_track": [0, 4],
        "cg_height": [-2, 2],
        "drive_train_speed": [0, 400],
        "damage": [0, 100],
        "gas": [0, 1],
        "brake": [0, 1],
        "clutch": [0, 1],
        "steer": [-1, 1],
        "current_lap_time": [0, 300],
        "last_lap_time": [0, 300],
        "best_lap_time": [0, 300],
        "lap_count": [0, 100],
        "lap_delta": [-10, 10],
        "current_sector": [0, 3],
        "track_length": [0, 50000],
        "air_temp": [-50, 50],
        "road_temp": [-50, 150],
        "tyre_wear_avg": [0, 100],
        "tyre_pressure_avg": [20, 30],
        "tyre_temp_avg": [0, 150],
        "tyre_dirty_avg": [0, 100],
        "tyre_slip_ratio_avg": [0, 2],
        "tyre_slip_angle_avg": [0, 50],
        "tyre_0_temp_i": [0, 150],
        "tyre_0_temp_m": [0, 150],
        "tyre_0_temp_o": [0, 150],
        "tyre_1_temp_i": [0, 150],
        "tyre_1_temp_m": [0, 150],
        "tyre_1_temp_o": [0, 150],
        "tyre_2_temp_i": [0, 150],
        "tyre_2_temp_m": [0, 150],
        "tyre_2_temp_o": [0, 150],
        "tyre_3_temp_i": [0, 150],
        "tyre_3_temp_m": [0, 150],
        "tyre_3_temp_o": [0, 150],
    }

    env = AssettoCorsa(
        host=args.host,
        send_port=args.send_port,
        recv_port=args.recv_port,
        racing_line_path=args.racing_line,
        include_image=False,
        use_ac_ai_racer=False,
        normalize_observations=args.normalize,
        normalization_bounds=normalization_bounds,
        input_config=input_config,
    )

    print("=" * 150)
    print(f"AC ENVIRONMENT NORMALIZATION TEST (normalize={args.normalize})")
    print("=" * 150)
    print(f"Total observation keys: {len(env.observation_keys)}")
    print(f"Observation keys: {env.observation_keys}\n")

    try:
        obs, info = env.reset()
        print("Environment reset. Running steps...\n")

        while True:
            if env._last_obs is None:
                time.sleep(0.05)
                obs = env._get_observation()
                continue

            data = env._last_obs

            # Extract raw values before normalization
            raw_values = np.array(
                [env._extract_value_from_data(key, data) for key in env.observation_keys],
                dtype=np.float32,
            )

            # Get normalized observation (if enabled in env)
            obs = env._get_observation()

            print("-" * 150)
            print(f"{'Key':<25} {'Raw Value':<15} {'Bounds':<25} {'Normalized':<15}")
            print("-" * 150)

            for i, key in enumerate(env.observation_keys):
                raw_val = raw_values[i]
                norm_val = obs[i] if args.normalize else raw_val

                bounds = env.normalization_bounds.get(key, None)
                bounds_str = str(bounds) if bounds else "N/A"

                print(f"{key:<25} {raw_val:<15.4f} {bounds_str:<25} {norm_val:<15.4f}")

            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("\nEpisode done, resetting...")
                obs, info = env.reset()

            time.sleep(0.05)
            print()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()
