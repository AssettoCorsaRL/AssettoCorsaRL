import os
import sys
import time
import pytest
import torch
from pathlib import Path

# Ensure the package in src/ is importable when running tests from the repo root
_repo_root = Path(__file__).resolve().parents[1]
_src_path = str((_repo_root / "src").resolve())
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

try:
    import pygame
except Exception:
    pygame = None

from tensordict import TensorDict
from assetto_corsa_rl.env import create_gym_env  # type: ignore


@pytest.mark.skipif(
    os.environ.get("INTERACTIVE") is None,
    reason="Interactive test - set INTERACTIVE=1 to run",
)
def test_manual_play_interactive():
    """
    Manual interactive test: play one episode using arrow keys or WASD.

    Usage (Windows CMD):
      set INTERACTIVE=1 && pytest -k manual -s tests/test_env.py

    Requires: pygame (for window and key capture).
    """
    if pygame is None:
        pytest.skip(
            "pygame is not installed; install pygame to run this interactive test"
        )

    # Initialize pygame so we can capture key events
    pygame.init()

    try:
        env = create_gym_env(device="cpu", num_envs=1, render_mode="human")
    except Exception as e:
        pytest.skip(f"Could not create env: {e}")

    print(
        "Interactive controls: Arrow keys or WASD to steer/gas/brake. ESC to quit. R to restart."
    )
    td = env.reset()
    inner = td["next"] if "next" in td.keys() else td

    max_steps = 2000
    step = 0
    quit_flag = False
    restart_flag = False

    # Give the env time to create the window and tell the user to focus it
    try:
        env.render()
    except Exception:
        pass
    print(
        "Click the game window and press any key to start (ESC to quit). Waiting up to 10s..."
    )
    started = False
    start_time = time.time()
    while not started and time.time() - start_time < 10:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                started = True
            elif event.type == pygame.QUIT:
                quit_flag = True
                started = True
        time.sleep(0.05)
    if not started:
        print(
            "No keypress detected; make sure the game window has focus — proceeding anyway."
        )

    # Detect joystick if present (useful for controllers)
    joystick = None
    if pygame is not None:
        try:
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                joystick = pygame.joystick.Joystick(0)
                joystick.init()
                try:
                    print(f"Joystick detected: {joystick.get_name()}")
                except Exception:
                    pass
        except Exception:
            pass

    def poll_input(joystick=None):
        """Return (steering, gas, brake), quit_requested, reset_requested"""
        steering = 0.0
        gas = 0.0
        brake = 0.0
        quit_req = False
        reset_req = False

        if pygame is not None:
            # Pump events to keep window responsive and update key state
            pygame.event.pump()

            # Handle queued events for quit/reset and joystick buttons
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_req = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        quit_req = True
                    elif event.key == pygame.K_SPACE:
                        reset_req = True

            # Joystick (if available)
            if joystick is not None and getattr(joystick, "get_init", lambda: False)():
                try:
                    axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
                    if len(axes) > 0:
                        # steering usually on axis 0
                        steer = float(axes[0])
                        if abs(steer) > 0.05:
                            steering = float(max(-1.0, min(1.0, steer)))

                    # PS4-like triggers (map [-1,1] to [0,1])
                    def axis_to_trigger(v):
                        return (
                            max(0.0, min(1.0, (v + 1.0) / 2.0))
                            if v is not None
                            else 0.0
                        )

                    if len(axes) > 5:
                        gas = float(axis_to_trigger(axes[5]))
                    if len(axes) > 4:
                        brake = float(axis_to_trigger(axes[4]))

                    if joystick.get_numbuttons() > 0:
                        try:
                            if joystick.get_button(3):
                                reset_req = True
                            if joystick.get_button(1):
                                quit_req = True
                        except Exception:
                            pass
                except Exception:
                    pass
            else:
                # Keyboard polling (works when window has focus)
                keys = pygame.key.get_pressed()
                if keys[pygame.K_q]:
                    quit_req = True
                if keys[pygame.K_SPACE]:
                    reset_req = True
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    steering = -1.0
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    steering = 1.0
                if keys[pygame.K_UP] or keys[pygame.K_w]:
                    gas = 1.0
                if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    brake = 0.8
        else:
            # Fallback to console keyboard polling (Windows)
            try:
                import msvcrt

                while msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b"q":
                        quit_req = True
                    elif key == b" ":
                        reset_req = True
                    elif key == b"w":
                        gas = 1.0
                    elif key == b"s":
                        brake = 0.8
                    elif key == b"a":
                        steering = -1.0
                    elif key == b"d":
                        steering = 1.0
            except Exception:
                pass

        return (steering, gas, brake), quit_req, reset_req

    try:
        while step < max_steps and not quit_flag:
            (steering, gas, brake), q, restart = poll_input(joystick=joystick)
            if q:
                quit_flag = True
            if restart:
                restart_flag = True

            action = torch.tensor(
                [steering, gas, brake], dtype=torch.float32
            ).unsqueeze(0)
            action_td = TensorDict({"action": action}, batch_size=[1])
            next_td = env.step(action_td)
            inner_next = next_td["next"] if "next" in next_td.keys() else next_td

            # print occasional status and debug inputs
            if step % 10 == 0:
                reward = float(
                    inner_next.get(
                        "reward", inner_next.get("rewards", torch.tensor([0.0]))
                    ).item()
                )
                print(
                    f"Step {step}, reward {reward:.2f}, action [{steering:.2f}, {gas:.2f}, {brake:.2f}]"
                )

            step += 1
            inner = inner_next

            # render and small delay
            try:
                env.render()
            except Exception:
                pass
            time.sleep(0.033)

            if restart_flag:
                print("Restarting episode...")
                restart_flag = False
                td = env.reset()
                inner = td["next"] if "next" in td.keys() else td
                step = 0

    finally:
        try:
            env.close()
        except Exception:
            pass
        pygame.quit()


def main() -> None:
    test_manual_play_interactive()


if __name__ == "__main__":
    main()
