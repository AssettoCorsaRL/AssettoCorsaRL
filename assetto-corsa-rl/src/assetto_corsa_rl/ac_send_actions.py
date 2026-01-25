from __future__ import annotations

from typing import Optional


class XboxController:
    def __init__(self):
        self._vg = None
        self._gamepad = None
        self._init_gamepad()

    def _init_gamepad(self):
        from pyvjoystick import vigem as vg  # type: ignore

        self._vg = vg
        self._gamepad = vg.VX360Gamepad()

    def press_button(self, button):
        self._gamepad.press_button(button=button)

    def release_button(self, button):
        self._gamepad.release_button(button=button)

    def press_a(self):
        self._gamepad.press_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

    def release_a(self):
        self._gamepad.release_button(button=self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

    def left_trigger(self, value: int):
        # int [0, 255]
        self._gamepad.left_trigger(value=value)

    def right_trigger(self, value: int):
        # int [0, 255]
        self._gamepad.right_trigger(value=value)

    def left_joystick(self, x_value: int, y_value: int):
        # int [-32768, 32767]

        self._gamepad.left_joystick(x_value=x_value, y_value=y_value)

    def left_joystick_float(self, x_value_float: float, y_value_float: float):
        # float [-1.0, 1.0]

        max_int = 32767
        xi = int(max(-1.0, min(1.0, x_value_float)) * max_int)
        yi = int(max(-1.0, min(1.0, y_value_float)) * max_int)
        if hasattr(self._gamepad, "left_joystick_float"):
            try:
                self._gamepad.left_joystick_float(
                    x_value_float=x_value_float, y_value_float=y_value_float
                )
                return
            except Exception:
                pass
        self._gamepad.left_joystick(x_value=xi, y_value=yi)

    def right_trigger_float(self, value_float: float):
        # float [0.0, 1.0]

        vi = int(max(0.0, min(1.0, value_float)) * 255.0)
        if hasattr(self._gamepad, "right_trigger_float"):
            try:
                self._gamepad.right_trigger_float(value_float=value_float)
                return
            except Exception:
                pass
        self._gamepad.right_trigger(value=vi)

    def left_trigger_float(self, value_float: float):
        # float [0.0, 1.0]

        vi = int(max(0.0, min(1.0, value_float)) * 255.0)
        self._gamepad.left_trigger(value=vi)

    def update(self):
        self._gamepad.update()

    def reset(self):
        self._gamepad.reset()

    def __enter__(self) -> "XboxController":
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.reset()
            self.update()
        except Exception:
            pass


def create_controller() -> XboxController:
    return XboxController()
