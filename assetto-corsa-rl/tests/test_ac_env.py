"""Tests for AssettoCorsa Gymnasium environment."""

import pytest
import numpy as np
import time
import os
import sys
from pathlib import Path

# Add src to path for imports
_repo_root = Path(__file__).resolve().parents[1]
_src_path = str((_repo_root / "src").resolve())
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from assetto_corsa_rl.ac_env import make_env, AssettoCorsa  # type: ignore


class TestAssettoCorsa:
    """Test suite for AssettoCorsa environment."""

    def test_env_creation(self):
        """Test that environment can be created with dummy controller."""
        env = make_env(use_dummy_controller=True)
        assert env is not None
        assert isinstance(env, AssettoCorsa)
        env.close()

    def test_action_space(self):
        """Test action space shape and bounds."""
        env = make_env(use_dummy_controller=True)

        assert env.action_space.shape == (3,)
        assert np.allclose(env.action_space.low, [-1.0, 0.0, 0.0])
        assert np.allclose(env.action_space.high, [1.0, 1.0, 1.0])

        # Sample actions should be within bounds
        for _ in range(10):
            action = env.action_space.sample()
            assert env.action_space.contains(action)

        env.close()

    def test_observation_space(self):
        """Test observation space configuration."""
        obs_keys = ["speed", "pos_x", "pos_y"]
        env = make_env(use_dummy_controller=True, observation_keys=obs_keys)

        assert env.observation_space.shape == (len(obs_keys),)
        assert len(env.observation_keys) == len(obs_keys)

        env.close()

    def test_reset(self):
        """Test environment reset."""
        env = make_env(use_dummy_controller=True)

        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
        assert "episode_step" in info
        assert info["episode_step"] == 0

        env.close()

    def test_step(self):
        """Test environment step with valid actions."""
        env = make_env(use_dummy_controller=True, max_episode_steps=10)

        obs, info = env.reset()

        # Test with various actions
        test_actions = [
            np.array([0.0, 0.5, 0.0], dtype=np.float32),  # Straight + throttle
            np.array([-0.5, 0.3, 0.0], dtype=np.float32),  # Left turn
            np.array([0.5, 0.0, 0.8], dtype=np.float32),  # Right + brake
        ]

        for action in test_actions:
            obs, reward, terminated, truncated, info = env.step(action)

            assert isinstance(obs, np.ndarray)
            assert obs.shape == env.observation_space.shape
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            assert "episode_step" in info
            assert "steering" in info
            assert "throttle" in info
            assert "brake" in info

        env.close()

    def test_episode_termination(self):
        """Test that episode terminates after max steps."""
        max_steps = 5
        env = make_env(use_dummy_controller=True, max_episode_steps=max_steps)

        obs, info = env.reset()

        for i in range(max_steps + 2):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if i < max_steps - 1:
                assert not terminated, f"Episode terminated too early at step {i}"
            else:
                # Should terminate at or after max_steps
                if terminated:
                    break

        assert terminated, "Episode did not terminate after max_steps"

        env.close()

    def test_context_manager(self):
        """Test that environment works as context manager."""
        with make_env(use_dummy_controller=True) as env:
            obs, info = env.reset()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None

    def test_action_clipping(self):
        """Test that actions outside bounds are clipped."""
        env = make_env(use_dummy_controller=True)
        obs, info = env.reset()

        # Actions outside bounds
        invalid_action = np.array([2.0, -0.5, 1.5], dtype=np.float32)

        # Should not raise error and should clip values
        obs, reward, terminated, truncated, info = env.step(invalid_action)

        # Check that clipped values are used
        assert -1.0 <= info["steering"] <= 1.0
        assert 0.0 <= info["throttle"] <= 1.0
        assert 0.0 <= info["brake"] <= 1.0

        env.close()


def test_manual_control_interactive():
    """
    Interactive test: manually control the car using keyboard.


    Controls:
        Arrow Keys / WASD: Steer left/right, gas, brake
        Q: Quit
        R: Reset episode

    Note: Requires Assetto Corsa running with telemetry enabled.
    """
    print("\n" + "=" * 60)
    print("Manual Control Test - Assetto Corsa Environment")
    print("=" * 60)
    print("\nControls:")
    print("  Arrow Keys / WASD: Control car")
    print("  Q: Quit")
    print("  R: Reset episode")
    print("\nMake sure Assetto Corsa is running with telemetry enabled!")
    print("=" * 60)

    input("\nPress Enter to start...")

    # Try with real controller (not dummy)
    try:
        env = make_env(use_dummy_controller=False, max_episode_steps=10000)
        print("✓ Real Xbox controller initialized")
    except Exception as e:
        print(f"⚠ Could not initialize real controller: {e}")
        print("Falling back to dummy controller (actions won't affect game)")
        env = make_env(use_dummy_controller=True, max_episode_steps=10000)

    try:
        obs, info = env.reset()
        print(f"\n✓ Environment reset. Initial observation shape: {obs.shape}")

        step = 0
        episode = 1

        # Try to import input library
        try:
            import msvcrt

            use_msvcrt = True
            print("\nUsing keyboard input (WASD keys). Press Q to quit, R to reset.")
        except ImportError:
            use_msvcrt = False
            print("\n⚠ Direct keyboard input not available. Using random actions.")

        while step < 1000:  # Max 1000 steps for test
            # Get keyboard input
            steering = 0.0
            throttle = 0.0
            brake = 0.0
            quit_flag = False
            reset_flag = False

            if use_msvcrt:
                while msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b"q" or key == b"Q":
                        quit_flag = True
                    elif key == b"r" or key == b"R":
                        reset_flag = True
                    elif key == b"w" or key == b"W":
                        throttle = 1.0
                    elif key == b"s" or key == b"S":
                        brake = 1.0
                    elif key == b"a" or key == b"A":
                        steering = -1.0
                    elif key == b"d" or key == b"D":
                        steering = 1.0
            else:
                # Use random actions if no keyboard input
                action = env.action_space.sample()
                steering, throttle, brake = action

            if quit_flag:
                print("\n\nQuitting...")
                break

            if reset_flag:
                print(f"\n\nResetting episode {episode}...")
                obs, info = env.reset()
                step = 0
                episode += 1
                continue

            action = np.array([steering, throttle, brake], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 20 == 0:
                print(
                    f"\rEpisode {episode}, Step {step}: "
                    f"Reward={reward:.2f}, "
                    f"Steer={steering:+.2f}, Gas={throttle:.2f}, Brake={brake:.2f}",
                    end="",
                    flush=True,
                )

            if terminated:
                print(f"\n\nEpisode {episode} terminated at step {step}")
                print("Press R to reset, Q to quit")
                obs, info = env.reset()
                step = 0
                episode += 1
            else:
                step += 1

            time.sleep(0.033)  # ~30 FPS

    finally:
        env.close()
        print("\n\n✓ Environment closed")


def test_basic_integration():
    """Basic integration test that runs a few steps."""
    env = make_env(use_dummy_controller=True, max_episode_steps=100)

    obs, info = env.reset()

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    # Run basic tests
    print("Running basic environment tests...\n")
    test_suite = TestAssettoCorsa()

    test_suite.test_env_creation()
    print("✓ Environment creation test passed")

    test_suite.test_action_space()
    print("✓ Action space test passed")

    test_suite.test_observation_space()
    print("✓ Observation space test passed")

    test_suite.test_reset()
    print("✓ Reset test passed")

    test_suite.test_step()
    print("✓ Step test passed")

    test_suite.test_episode_termination()
    print("✓ Episode termination test passed")

    test_suite.test_context_manager()
    print("✓ Context manager test passed")

    test_suite.test_action_clipping()
    print("✓ Action clipping test passed")

    test_basic_integration()
    print("✓ Basic integration test passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    test_manual_control_interactive()
