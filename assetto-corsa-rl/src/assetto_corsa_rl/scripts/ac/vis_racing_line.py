"""
Usage:
    acrl ac vis-racing-line --input racing_lines.json
    acrl ac vis-racing-line --input monza_20260109_194248.json --lap 0
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from assetto_corsa_rl.cli_registry import cli_command, cli_option  # type: ignore


class RacingLineVisualizer:
    def __init__(self, racing_data: Dict[str, Any], lap_index: int = 0):
        self.racing_data = racing_data
        self.lap_index = lap_index

        if lap_index < 0 or lap_index >= len(racing_data["laps"]):
            raise ValueError(
                f"Invalid lap index {lap_index}. Available: 0-{len(racing_data['laps'])-1}"
            )

        self.lap = racing_data["laps"][lap_index]
        self.positions = np.array([[p["x"], p["y"], p["z"]] for p in self.lap["positions"]])

        self.azim = -60
        self.elev = 30
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.mouse_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        self.car_index = 0
        self.animation_speed = 1
        self.paused = False

        self.setup_plot()

    def setup_plot(self):
        """Setup matplotlib 3D plot."""
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.line = self.ax.plot(
            self.positions[:, 0],
            self.positions[:, 1],
            self.positions[:, 2],
            "b-",
            linewidth=2,
            alpha=0.6,
            label="Racing Line",
        )[0]

        self.ax.scatter(
            self.positions[0, 0],
            self.positions[0, 1],
            self.positions[0, 2],
            c="green",
            s=100,
            marker="o",
            label="Start",
        )
        self.ax.scatter(
            self.positions[-1, 0],
            self.positions[-1, 1],
            self.positions[-1, 2],
            c="red",
            s=100,
            marker="X",
            label="End",
        )

        self.car_marker = self.ax.scatter(
            [self.positions[0, 0]],
            [self.positions[0, 1]],
            [self.positions[0, 2]],
            c="yellow",
            s=200,
            marker="o",
            edgecolors="black",
            linewidths=2,
            label="Car",
            zorder=10,
        )

        self.trail = self.ax.plot([], [], [], "r-", linewidth=3, alpha=0.8)[0]

        self.ax.set_xlabel("X Position (m)", fontsize=10)
        self.ax.set_ylabel("Y Position (m)", fontsize=10)
        self.ax.set_zlabel("Z Position (m)", fontsize=10)

        metadata = self.lap.get("metadata", {})
        title = f"Racing Line - Lap {self.lap['lap_number']}"
        if metadata:
            title += f"\nTrack: {metadata.get('track', 'Unknown')} | Car: {metadata.get('car', 'Unknown')}"
        title += f"\nPoints: {len(self.positions)} | Lap Time: {self.lap.get('lap_time', 0):.2f}s"

        self.ax.set_title(title, fontsize=12, pad=20)
        self.ax.legend(loc="upper right")

        self.set_equal_aspect()

        controls_text = (
            "Controls:\n"
            "  Mouse Drag: Rotate view\n"
            "  Scroll: Zoom\n"
            "  W/S: Tilt up/down\n"
            "  A/D: Rotate left/right\n"
            "  Q/E: Zoom in/out\n"
            "  Space: Play/Pause\n"
            "  R: Reset view\n"
            "  +/-: Speed up/down\n"
        )
        self.fig.text(
            0.02,
            0.98,
            controls_text,
            fontsize=9,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

    def set_equal_aspect(self):
        """Set equal aspect ratio for 3D plot."""

        x_range = self.positions[:, 0].max() - self.positions[:, 0].min()
        y_range = self.positions[:, 1].max() - self.positions[:, 1].min()
        z_range = self.positions[:, 2].max() - self.positions[:, 2].min()

        max_range = max(x_range, y_range, z_range)

        x_middle = (self.positions[:, 0].max() + self.positions[:, 0].min()) / 2
        y_middle = (self.positions[:, 1].max() + self.positions[:, 1].min()) / 2
        z_middle = (self.positions[:, 2].max() + self.positions[:, 2].min()) / 2

        self.ax.set_xlim(x_middle - max_range / 2, x_middle + max_range / 2)
        self.ax.set_ylim(y_middle - max_range / 2, y_middle + max_range / 2)
        self.ax.set_zlim(z_middle - max_range / 2, z_middle + max_range / 2)

    def update_view(self):
        """Update the 3D view based on current parameters."""
        self.ax.view_init(elev=self.elev, azim=self.azim)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zlim = self.ax.get_zlim()

        x_range = (xlim[1] - xlim[0]) / self.zoom
        y_range = (ylim[1] - ylim[0]) / self.zoom
        z_range = (zlim[1] - zlim[0]) / self.zoom

        x_center = (xlim[0] + xlim[1]) / 2 + self.pan_x
        y_center = (ylim[0] + ylim[1]) / 2 + self.pan_y
        z_center = (zlim[0] + zlim[1]) / 2

        self.ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
        self.ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
        self.ax.set_zlim(z_center - z_range / 2, z_center + z_range / 2)

        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == "w":
            self.elev += 5
        elif event.key == "s":
            self.elev -= 5
        elif event.key == "a":
            self.azim -= 5
        elif event.key == "d":
            self.azim += 5
        elif event.key == "q":
            self.zoom *= 1.1
        elif event.key == "e":
            self.zoom /= 1.1
        elif event.key == "r":

            self.azim = -60
            self.elev = 30
            self.zoom = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.set_equal_aspect()
        elif event.key == " ":

            self.paused = not self.paused
        elif event.key == "+" or event.key == "=":
            self.animation_speed = min(self.animation_speed + 1, 10)
            print(f"Animation speed: {self.animation_speed}x")
        elif event.key == "-":
            self.animation_speed = max(self.animation_speed - 1, 1)
            print(f"Animation speed: {self.animation_speed}x")
        else:
            return

        self.update_view()

    def on_mouse_press(self, event):
        """Handle mouse press events."""
        if event.button == 1:
            self.mouse_pressed = True
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y

    def on_mouse_release(self, event):
        """Handle mouse release events."""
        if event.button == 1:
            self.mouse_pressed = False

    def on_mouse_move(self, event):
        """Handle mouse move events."""
        if self.mouse_pressed and event.x is not None and event.y is not None:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y

            self.azim += dx * 0.5
            self.elev -= dy * 0.5

            self.last_mouse_x = event.x
            self.last_mouse_y = event.y

            self.update_view()

    def on_scroll(self, event):
        """Handle mouse scroll events."""
        if event.button == "up":
            self.zoom *= 1.1
        elif event.button == "down":
            self.zoom /= 1.1

        self.update_view()

    def animate(self, frame):
        """Animation function to move car along racing line."""
        if self.paused:
            return self.car_marker, self.trail

        self.car_index = (self.car_index + self.animation_speed) % len(self.positions)

        pos = self.positions[self.car_index]
        self.car_marker._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        trail_start = max(0, self.car_index - 50)
        trail_positions = self.positions[trail_start : self.car_index + 1]

        if len(trail_positions) > 1:
            self.trail.set_data(trail_positions[:, 0], trail_positions[:, 1])
            self.trail.set_3d_properties(trail_positions[:, 2])

        return self.car_marker, self.trail

    def show(self, animate: bool = True):
        """Show the visualization.

        Args:
            animate: Whether to animate the car along the racing line
        """
        if animate:

            self.anim = FuncAnimation(
                self.fig, self.animate, interval=50, blit=False, cache_frame_data=False
            )

        plt.tight_layout()
        plt.show()


def load_racing_data(filepath: str) -> Dict[str, Any]:
    """Load racing line data from JSON file.

    Args:
        filepath: Path to racing line JSON file

    Returns:
        Racing line data dictionary
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


@cli_command(
    group="ac", name="vis-racing-line", help="Visualize racing line from recorded telemetry"
)
@cli_option("--input", "-i", required=True, help="Input racing line JSON file")
@cli_option("--lap", "-l", default=0, help="Lap index to visualize")
@cli_option("--no-animate", is_flag=True, help="Disable car animation")
def main(input, lap, no_animate):
    parser = argparse.ArgumentParser(description="Visualize racing line from recorded telemetry")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input racing line JSON file",
    )
    parser.add_argument(
        "--lap",
        "-l",
        type=int,
        default=0,
        help="Lap index to visualize (default: 0)",
    )
    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="Disable car animation",
    )

    # args = parser.parse_args()

    input_path = Path(input)
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        return

    print(f"📂 Loading racing line from: {input_path}")
    racing_data = load_racing_data(str(input_path))

    print(f"✓ Loaded {racing_data['num_laps']} lap(s)")

    try:
        visualizer = RacingLineVisualizer(racing_data, lap_index=lap)
        print(f"\n🏁 Visualizing lap {lap}")
        print(f"   Points: {len(visualizer.positions)}")
        print(f"   Lap time: {visualizer.lap.get('lap_time', 0):.2f}s\n")

        visualizer.show(animate=not no_animate)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return


if __name__ == "__main__":
    main()
