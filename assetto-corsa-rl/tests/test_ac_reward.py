"""Test script to print out rewards from current AC state.

Connects to running Assetto Corsa instance and continuously prints
reward values based on current telemetry.

Usage:
    python tests/test_ac_reward.py
    python tests/test_ac_reward.py --racing-line monza_20260109_194248.json
"""

import sys
import time
import argparse
from pathlib import Path

# Add src to path
_repo_root = Path(__file__).resolve().parents[1]
_src_path = str((_repo_root / "src").resolve())
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from assetto_corsa_rl.ac_env import AssettoCorsa
import numpy as np

import matplotlib.pyplot as plt


def test_reward_monitoring(
    racing_line_path: str = None,
    constant_reward_per_ms: float = -0.005,
    reward_per_m_advanced: float = 1.0,
    speed_reward: float = 0.1,
    ms_per_action: float = 20.0,
    show_image: bool = False,
):
    """Monitor and print rewards from current AC state."""

    print("=" * 60)
    print("AC Reward Monitor")
    print("=" * 60)

    # Create environment
    print("\nInitializing environment...")
    env = AssettoCorsa(
        use_dummy_controller=True,  # Don't send actions, just monitor
        constant_reward_per_ms=constant_reward_per_ms,
        reward_per_m_advanced_along_centerline=reward_per_m_advanced,
        final_speed_reward_per_m_per_s=speed_reward,
        ms_per_action=ms_per_action,
    )

    if racing_line_path:
        print(f"✓ Loaded racing line from: {racing_line_path}")
    else:
        print("⚠ No racing line loaded - progress reward will be 0")

    print("\nReward Parameters:")
    print(f"  Constant reward per ms: {constant_reward_per_ms}")
    print(f"  Reward per meter advanced: {reward_per_m_advanced}")
    print(f"  Speed change reward: {speed_reward}")
    print(f"  MS per action: {ms_per_action}")
    print("=" * 60)

    fig, ax = None, None

    print("\n✓ Image display enabled")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Assetto Corsa View")
    ax.axis("off")

    try:
        print("\n🔴 Monitoring rewards... (Press Ctrl+C to stop)\n")

        step_count = 0
        total_reward = 0.0

        while True:
            obs = env._get_observation()
            data = env._last_obs

            if data is None:
                print("\r⚠ Waiting for telemetry data...", end="", flush=True)
                time.sleep(0.1)
                continue

            if env._check_done(obs, data):
                print("\n\n✓ Episode completed!")

                env.reset()

            reward = env._calculate_reward(obs, data)
            total_reward += reward
            step_count += 1

            position = data.get("car", {}).get("world_location", [0, 0, 0])
            velocity = data.get("car", {}).get("velocity", [0, 0, 0])
            speed = np.linalg.norm(velocity)

            if show_image:
                img = env.telemetry.get_latest_image()
                if img is not None:
                    ax.clear()
                    ax.imshow(img)
                    ax.set_title(f"Assetto Corsa View - Step {step_count}")
                    ax.axis("off")
                    plt.pause(0.00001)

            print(f"\r[Step {step_count:4d}] ", end="")
            print(f"Reward: {reward:8.3f} | ", end="")
            print(f"Total: {total_reward:8.2f} | ", end="")
            print(f"Speed: {speed:6.2f} m/s | ", end="")

            if env.racing_line_positions is not None:
                meters = env._meters_advanced
                closest_idx, distance = env._find_closest_point_on_racing_line(
                    np.array(position)
                )
                print(f"Progress: {meters:7.1f}m | ", end="")
                print(f"Dist to line: {distance:5.2f}m", end="")

            print("", flush=True)

            time.sleep(0.05)  # 20 Hz update rate

    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")
        print(f"\nFinal Statistics:")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {total_reward:.2f}")
        if step_count > 0:
            print(f"  Average reward per step: {total_reward/step_count:.3f}")
        if env.racing_line_positions is not None:
            print(f"  Final progress: {env._meters_advanced:.1f} meters")

    finally:
        if fig is not None:
            plt.close(fig)
        env.close()
        print("\n✓ Environment closed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor and print AC reward values")
    parser.add_argument(
        "--racing-line",
        "-r",
        type=str,
        default=None,
        help="Path to racing line JSON file",
    )
    parser.add_argument(
        "--constant-reward",
        type=float,
        default=-0.005,
        help="Constant reward per millisecond (default: -0.005)",
    )
    parser.add_argument(
        "--progress-reward",
        type=float,
        default=1.0,
        help="Reward per meter advanced along centerline (default: 1.0)",
    )
    parser.add_argument(
        "--speed-reward",
        type=float,
        default=0.1,
        help="Reward per m/s speed change (default: 0.1)",
    )
    parser.add_argument(
        "--ms-per-action",
        type=float,
        default=20.0,
        help="Milliseconds per action step (default: 20.0)",
    )

    args = parser.parse_args()

    # Resolve racing line path if provided
    racing_line_path = None
    if args.racing_line:
        path = Path(args.racing_line)
        if not path.is_absolute():
            # Try relative to repo root
            path = _repo_root.parent / args.racing_line
        if not path.exists():
            print(f"❌ Error: Racing line file not found: {path}")
            return
        racing_line_path = str(path)

    test_reward_monitoring(
        racing_line_path=racing_line_path,
        constant_reward_per_ms=args.constant_reward,
        reward_per_m_advanced=args.progress_reward,
        speed_reward=args.speed_reward,
        ms_per_action=args.ms_per_action,
        show_image=True,
    )


if __name__ == "__main__":
    main()
