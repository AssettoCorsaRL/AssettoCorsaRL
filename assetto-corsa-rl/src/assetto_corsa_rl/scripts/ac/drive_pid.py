import argparse
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from assetto_corsa_rl.ac_env import make_env
from assetto_corsa_rl.cli_registry import cli_command, cli_option


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, output_limits: tuple = (-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limits = output_limits

        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def step(self, error: float, dt: float = None) -> float:
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time
        if dt <= 0:
            dt = 1e-4

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        output = max(self.limits[0], min(self.limits[1], output))

        self.prev_error = error
        self.last_time = current_time
        return output


def load_racing_line(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r") as f:
        data = json.load(f)

    return data["laps"][0]["positions"]  # js use the first lap


# quick calc
# TODO: use a utils file to share this with env and this
def get_closest_point_idx(car_pos: np.ndarray, racing_line: List[Dict[str, Any]]) -> int:
    min_dist = float("inf")
    best_idx = 0
    for i, point in enumerate(racing_line):
        px, pz = point["x"], point["z"]
        dist = math.hypot(car_pos[0] - px, car_pos[2] - pz)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx


@cli_command(
    group="ac",
    name="drive-pid",
    help="Imitate a recorded racing line using PID and randomness",
)
@cli_option("--line-file", "-l", default="racing_lines.json", help="Path to racing lines JSON")
@cli_option("--noise", "-n", default=0.05, type=float, help="Amount of random noise to apply")
def main(line_file: str, noise: float):
    print(f"Loading racing line from {line_file}...")

    if line_file.startswith("http://") or line_file.startswith("https://"):
        import urllib.request

        local_filename = "downloaded_racing_lines.json"
        print(f"Downloading from {line_file} to {local_filename}...")
        urllib.request.urlretrieve(line_file, local_filename)
        line_file = local_filename

    try:
        racing_line = load_racing_line(line_file)
        print(f"Loaded a lap with {len(racing_line)} points.")
    except Exception as e:
        print(f"Failed to load racing line: {e}")
        return

    input_config = {
        "steer": True,
        "gas": True,
        "brake": True,
    }

    env = make_env(
        racing_line_path=line_file,
        input_config=input_config,
        include_image=False,
        max_episode_steps=10000,
        use_ac_ai_racer=False,
    )

    # TODO: tune this better
    steer_pid = PIDController(kp=3, ki=0.01, kd=0.5, output_limits=(-1.0, 1.0))
    speed_pid = PIDController(kp=0.5, ki=0.001, kd=0.01, output_limits=(-1.0, 1.0))

    _, _ = env.reset()

    try:
        import win32gui

        hwnd = env.unwrapped.telemetry._find_ac_window()
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
            print("Activated Assetto Corsa window.")
    except Exception as e:
        print(f"Could not activate Assetto Corsa window: {e}")

    start_time = time.time()

    print("Driving... Press 'q' or Ctrl+C to quit.")
    try:
        while True:
            latest_data = env.unwrapped.telemetry.get_latest()
            if not latest_data:
                time.sleep(0.01)
                continue

            car_pos = np.array(latest_data["car"]["world_location"])
            current_speed = latest_data["car"]["speed_kmh"]

            closest_idx = get_closest_point_idx(car_pos, racing_line)
            lookahead = 5
            target_idx = (closest_idx + lookahead) % len(racing_line)
            target_point = racing_line[target_idx]

            car_vel = np.array(latest_data["car"]["velocity"])
            current_yaw = math.atan2(car_vel[0], car_vel[2])

            dx = target_point["x"] - car_pos[0]
            dz = target_point["z"] - car_pos[2]
            target_yaw = math.atan2(dx, dz)

            angle_error = target_yaw - current_yaw
            # norm angle error to [-pi, pi]
            angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

            base_steer = steer_pid.step(angle_error)

            target_speed = target_point.get("speed_kmh", 80.0)
            speed_error = target_speed - current_speed

            speed_out = speed_pid.step(speed_error)

            # PID output -> continuous acceleration (-1.0 to 1.0)
            # env expects a 2-dim action space: [steering, acceleration]
            # acceleration >= 0 is gas, acceleration < 0 is brake
            if time.time() - start_time < 1.0:
                accel = 1.0
            else:
                accel = np.clip(speed_out + np.random.uniform(-noise, noise), -1.0, 1.0)

            steer = np.clip(base_steer + np.random.uniform(-noise, noise), -1.0, 1.0)

            action = np.array([-steer, accel], dtype=np.float32)
            _, _, done, truncated, info = env.step(action)

            if done or truncated:
                print("Episode finished. Resetting...")
                _, _ = env.reset()
                start_time = time.time()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--line-file", "-l", default="racing_lines.json", help="Path to racing lines JSON"
    )
    parser.add_argument("--noise", "-n", default=0.05, type=float, help="Random noise factor")
    args = parser.parse_args()
    main(line_file=args.line_file, noise=args.noise)
