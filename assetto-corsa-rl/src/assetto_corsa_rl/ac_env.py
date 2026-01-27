import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
import time
import json
from pathlib import Path
import cv2
import torch

from .ac_send_actions import XboxController
from .ac_telemetry_helper import Telemetry


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
        constant_reward_per_ms: float = 0.01,
        reward_per_m_advanced_along_centerline: float = 1.0,
        final_speed_reward_per_m_per_s: float = 0.1,
        ms_per_action: float = 20.0,
        include_image: bool = False,
        observation_image_shape: Tuple[int, int] = (84, 84),
    ):
        super().__init__()

        self.host = host
        self.send_port = send_port
        self.recv_port = recv_port
        self.max_episode_steps = max_episode_steps
        self.timeout = timeout

        self.constant_reward_per_ms = constant_reward_per_ms
        self.reward_per_m_advanced_along_centerline = reward_per_m_advanced_along_centerline
        self.final_speed_reward_per_m_per_s = final_speed_reward_per_m_per_s
        self.ms_per_action = ms_per_action

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

        if self.include_image:
            return {"vector": vector, "image": img}
        return vector

    def _extract_value_from_data(self, key: str, data: Dict[str, Any]) -> float:
        """Extract a single observation value from telemetry data.

        Args:
            key: The observation key name
            data: Full telemetry data dict

        Returns:
            Float value for the observation
        """

        # TODO: remove this ass code

        if key == "speed_kmh":
            return float(data.get("car", {}).get("speed_kmh", 0.0))
        elif key == "speed_mph":
            return float(data.get("car", {}).get("speed_mph", 0.0))
        elif key == "speed_ms":
            return float(data.get("car", {}).get("speed_ms", 0.0))
        elif key == "rpm":
            return float(data.get("car", {}).get("rpm", 0.0))
        elif key == "gear":
            return float(data.get("car", {}).get("gear", 0.0))
        elif key == "velocity_x":
            vel = data.get("car", {}).get("velocity", [0.0, 0.0, 0.0])
            return float(vel[0] if len(vel) > 0 else 0.0)
        elif key == "velocity_y":
            vel = data.get("car", {}).get("velocity", [0.0, 0.0, 0.0])
            return float(vel[1] if len(vel) > 1 else 0.0)
        elif key == "velocity_z":
            vel = data.get("car", {}).get("velocity", [0.0, 0.0, 0.0])
            return float(vel[2] if len(vel) > 2 else 0.0)
        elif key == "acceleration_x":
            acc = data.get("car", {}).get("acceleration", [0.0, 0.0, 0.0])
            return float(acc[0] if len(acc) > 0 else 0.0)
        elif key == "acceleration_y":
            acc = data.get("car", {}).get("acceleration", [0.0, 0.0, 0.0])
            return float(acc[1] if len(acc) > 1 else 0.0)
        elif key == "acceleration_z":
            acc = data.get("car", {}).get("acceleration", [0.0, 0.0, 0.0])
            return float(acc[2] if len(acc) > 2 else 0.0)

        elif key == "fuel":
            return float(data.get("car", {}).get("fuel", 0.0))
        elif key == "tyres_off_track":
            return float(data.get("car", {}).get("tyres_off_track", 0.0))
        elif key == "in_pit_lane":
            return float(data.get("car", {}).get("in_pit_lane", 0.0))
        elif key == "damage":
            damage = data.get("car", {}).get("damage", [0.0, 0.0, 0.0, 0.0, 0.0])
            return float(sum(damage) if isinstance(damage, list) else damage)
        elif key == "cg_height":
            return float(data.get("car", {}).get("cg_height", 0.0))
        elif key == "drive_train_speed":
            return float(data.get("car", {}).get("drive_train_speed", 0.0))

        elif key == "drs_available":
            return float(data.get("car", {}).get("drs_available", 0.0))
        elif key == "drs_enabled":
            return float(data.get("car", {}).get("drs_enabled", 0.0))
        elif key == "has_drs":
            return float(data.get("stats", {}).get("has_drs", 0.0))
        elif key == "has_ers":
            return float(data.get("stats", {}).get("has_ers", 0.0))
        elif key == "has_kers":
            return float(data.get("stats", {}).get("has_kers", 0.0))

        elif key == "gas":
            return float(data.get("inputs", {}).get("gas", 0.0))
        elif key == "brake":
            return float(data.get("inputs", {}).get("brake", 0.0))
        elif key == "clutch":
            return float(data.get("inputs", {}).get("clutch", 0.0))
        elif key == "steer":
            return float(data.get("inputs", {}).get("steer", 0.0))

        elif key == "current_lap_time":
            return float(data.get("lap", {}).get("get_current_lap_time", 0.0))
        elif key == "last_lap_time":
            return float(data.get("lap", {}).get("get_last_lap_time", 0.0))
        elif key == "best_lap_time":
            return float(data.get("lap", {}).get("get_best_lap_time", 0.0))
        elif key == "lap_count":
            return float(data.get("lap", {}).get("get_lap_count", 0.0))
        elif key == "lap_delta":
            return float(data.get("lap", {}).get("get_lap_delta", 0.0))
        elif key == "current_sector":
            return float(data.get("lap", {}).get("get_current_sector", 0.0))
        elif key == "invalid_lap":
            return float(data.get("lap", {}).get("get_invalid", 0.0))

        elif key == "track_length":
            return float(data.get("session", {}).get("track_length", 0.0))
        elif key == "air_temp":
            return float(data.get("session", {}).get("air_temp", 0.0))
        elif key == "road_temp":
            return float(data.get("session", {}).get("road_temp", 0.0))
        elif key == "session_status":
            return float(data.get("session", {}).get("session_status", 0.0))

        elif key == "tyre_wear_avg":
            tyres = data.get("tyres", [])
            if tyres:
                wear_vals = [t.get("wear", 0.0) for t in tyres if t.get("wear") is not None]
                return float(np.mean(wear_vals) if wear_vals else 0.0)
            return 0.0
        elif key == "tyre_pressure_avg":
            tyres = data.get("tyres", [])
            if tyres:
                pressure_vals = [
                    t.get("pressure", 0.0) for t in tyres if t.get("pressure") is not None
                ]
                return float(np.mean(pressure_vals) if pressure_vals else 0.0)
            return 0.0
        elif key == "tyre_temp_avg":
            tyres = data.get("tyres", [])
            if tyres:
                temp_vals = []
                for t in tyres:
                    if t.get("temp_m") is not None:
                        temp_vals.append(t["temp_m"])
                return float(np.mean(temp_vals) if temp_vals else 0.0)
            return 0.0
        elif key == "tyre_dirty_avg":
            tyres = data.get("tyres", [])
            if tyres:
                dirty_vals = [t.get("dirty", 0.0) for t in tyres if t.get("dirty") is not None]
                return float(np.mean(dirty_vals) if dirty_vals else 0.0)
            return 0.0
        elif key == "tyre_slip_ratio_avg":
            tyres = data.get("tyres", [])
            if tyres:
                slip_vals = [
                    t.get("slip_ratio", 0.0) for t in tyres if t.get("slip_ratio") is not None
                ]
                return float(np.mean(slip_vals) if slip_vals else 0.0)
            return 0.0
        elif key == "tyre_slip_angle_avg":
            tyres = data.get("tyres", [])
            if tyres:
                slip_vals = [
                    t.get("slip_angle", 0.0) for t in tyres if t.get("slip_angle") is not None
                ]
                return float(np.mean(slip_vals) if slip_vals else 0.0)
            return 0.0

        # Individual tyre temps (tyre_N_temp_X where N=0-3, X=i/m/o)
        elif key.startswith("tyre_") and "_temp_" in key:
            parts = key.split("_")
            tyre_idx = int(parts[1])
            temp_pos = parts[3]  # i, m, or o
            tyres = data.get("tyres", [])
            if tyre_idx < len(tyres):
                return float(tyres[tyre_idx].get(f"temp_{temp_pos}", 0.0) or 0.0)
            return 0.0

        else:
            return float(data.get(key, 0.0))

    def _load_racing_line(self, filepath: str) -> None:
        """Load racing line from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Racing line file not found: {filepath}")

        with open(path, "r") as f:
            self.racing_line = json.load(f)

        if self.racing_line["num_laps"] == 0:
            raise ValueError("Racing line file contains no laps")

        lap = self.racing_line["laps"][0]
        positions = np.array([[p["x"], p["y"], p["z"]] for p in lap["positions"]])
        self.racing_line_positions = positions

        print(f"✓ Loaded racing line with {len(positions)} points")

    def _find_closest_point_on_racing_line(self, position: np.ndarray) -> Tuple[int, float]:
        if self.racing_line_positions is None:
            return 0, 0.0

        distances = np.linalg.norm(self.racing_line_positions - position, axis=1)
        closest_idx = np.argmin(distances)

        return int(closest_idx), float(distances[closest_idx])

    def _calculate_meters_advanced(self, position: np.ndarray) -> float:
        """Calculate meters advanced along racing line.

        Args:
            position: Current car position [x, y, z]

        Returns:
            Total meters advanced along the racing line
        """
        if self.racing_line_positions is None:
            return 0.0

        closest_idx, _ = self._find_closest_point_on_racing_line(position)

        max_reasonable_idx_jump = 50

        if self._current_racing_line_index == 0:
            self._current_racing_line_index = closest_idx

        idx_diff = closest_idx - self._current_racing_line_index

        if idx_diff < -len(self.racing_line_positions) // 2:
            idx_diff += len(self.racing_line_positions)
        elif idx_diff > len(self.racing_line_positions) // 2:
            idx_diff -= len(self.racing_line_positions)

        if abs(idx_diff) > max_reasonable_idx_jump:
            return self._meters_advanced

        if idx_diff > 0:
            self._current_racing_line_index = closest_idx
        else:
            return self._meters_advanced

        if closest_idx == 0:
            return 0.0

        segments = (
            self.racing_line_positions[1 : closest_idx + 1]
            - self.racing_line_positions[0:closest_idx]
        )
        distances = np.linalg.norm(segments, axis=1)
        total_distance = np.sum(distances)

        return float(total_distance)

    # inspired by linesight-rl: https://github.com/Linesight-RL/linesight/tree/main
    #! UNTESTED WITH PRETRAINED AGENTS
    def _calculate_reward(self, obs: np.ndarray, data: Optional[Dict]) -> float:
        """Calculate reward based on observation and telemetry.

        reward = constant_reward_per_ms * ms_per_action
               + (meters_advanced[i] - meters_advanced[i-1]) * reward_per_m_advanced_along_centerline
               + final_speed_reward_per_m_per_s * (|v_i| - |v_{i-1}|) if moving forward
        """

        reward = 0.0

        reward += self.constant_reward_per_ms * self.ms_per_action

        position = np.array(
            [
                data["car"]["world_location"][0],
                data["car"]["world_location"][1],
                data["car"]["world_location"][2],
            ]
        )

        current_meters = self._calculate_meters_advanced(position)
        meters_progress = current_meters - self._meters_advanced

        reward += meters_progress * self.reward_per_m_advanced_along_centerline

        self._meters_advanced = current_meters

        current_speed = data["car"]["speed_mph"]

        speed_change = current_speed - self._last_speed
        reward += self.final_speed_reward_per_m_per_s * speed_change

        self._last_speed = current_speed

        return reward

    def _check_done(self, obs: np.ndarray, data: Optional[Dict]) -> bool:
        if self._episode_step >= self.max_episode_steps:
            return True

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

        time.sleep(1)

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

    def render(self):
        """Rendering is handled by Assetto Corsa itself."""
        pass

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
    frame_stack: int = 4,
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
        BoundedTensorSpec,
        CompositeSpec,
        UnboundedContinuousTensorSpec,
    )

    if device is None:
        device = get_device()

    class MockEnv:
        def __init__(self, device):
            self.device = device
            # Match AssettoCorsa action space: [steering, throttle, brake]
            self.action_spec = BoundedTensorSpec(
                low=-1.0, high=1.0, shape=(3,), dtype=torch.float32, device=device
            )
            # Vision-based observation: 4 stacked 84x84 grayscale frames
            self.observation_spec = CompositeSpec(
                pixels=UnboundedContinuousTensorSpec(
                    shape=(4, 84, 84), dtype=torch.float32, device=device
                )
            )

    return MockEnv(device)
