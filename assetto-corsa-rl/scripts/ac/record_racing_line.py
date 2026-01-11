"""Record racing line from Assetto Corsa telemetry.

Continuously records car position during laps and saves the racing line
to a file when each lap completes. The script waits for a new lap to start
before beginning recording.

Usage:
    python assetto-corsa-rl/scripts/ac/record_racing_line.py --output racing_lines.json --track monza
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

_repo_root = Path(__file__).resolve().parents[2]
_src_path = str((_repo_root / "src").resolve())
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from assetto_corsa_rl.ac_telemetry_helper import Telemetry  # type: ignore


class RacingLineRecorder:
    """Records racing lines from Assetto Corsa telemetry."""

    def __init__(
        self,
        output_file: str = "racing_lines.json",
        host: str = "127.0.0.1",
        recv_port: int = 9876,
        sample_rate: float = 0.01,  # 100 Hz (higher sample rate)
    ):
        self.output_file = output_file
        self.sample_rate = sample_rate

        self.telemetry = Telemetry(
            host=host,
            recv_port=recv_port,
            timeout=0.1,
            auto_start_receiver=True,
        )

        self.current_lap_positions: List[Dict[str, float]] = []
        self.completed_laps: List[Dict[str, Any]] = []
        self.last_lap_count = 0
        self.recording = False
        self.lap_start_time = None

    def record_position(self, data: Dict[str, Any]) -> None:
        if not data:
            return

        position = {
            "timestamp": time.time(),
            "x": data["car"]["world_location"][0],
            "y": data["car"]["world_location"][1],
            "z": data["car"]["world_location"][2],
        }

        self.current_lap_positions.append(position)

    def check_lap_complete(self, data: Dict[str, Any]) -> bool:
        if not data:
            return False

        if data["lap"]["get_lap_count"] > self.last_lap_count:
            return True

        return False

    def save_lap(self, data: Optional[Dict[str, Any]] = None) -> None:
        if not self.current_lap_positions:
            print("⚠ No positions recorded for this lap")
            return

        lap_time = time.time() - self.lap_start_time if self.lap_start_time else 0.0

        lap_data = {
            "lap_number": self.last_lap_count,
            "num_points": len(self.current_lap_positions),
            "lap_time": lap_time,
            "timestamp": datetime.now().isoformat(),
            "positions": self.current_lap_positions,
        }

        if data:
            lap_data["metadata"] = {
                "car": data.get("car", data.get("car_name", "unknown")),
                "track": data.get("track", data.get("track_name", "unknown")),
                "track_config": data.get("track_config", ""),
            }

        self.completed_laps.append(lap_data)

        print(f"\n✓ Lap {self.last_lap_count} saved:")
        print(f"   Points recorded: {len(self.current_lap_positions)}")
        print(f"   Lap time: {lap_time:.2f}s")

        self.current_lap_positions = []

    def save_to_file(self) -> None:
        output_data = {
            "version": "1.0",
            "recorded_at": datetime.now().isoformat(),
            "num_laps": len(self.completed_laps),
            "laps": self.completed_laps,
        }

        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Saved {len(self.completed_laps)} laps to: {output_path}")

    def run(self, max_laps: Optional[int] = None) -> None:
        print("=" * 60)
        print("Racing Line Recorder")
        print("=" * 60)
        print(f"Output file: {self.output_file}")
        print(f"Sample rate: {1.0/self.sample_rate:.1f} Hz")
        if max_laps:
            print(f"Max laps: {max_laps}")
        print("=" * 60)

        laps_recorded = 0

        try:
            print("\n🔴 Recording started! Drive the racing line.")
            print("   Press Ctrl+C to stop and save\n")

            last_sample_time = time.time()

            while True:
                if max_laps and laps_recorded >= max_laps:
                    print(f"\n✓ Reached max laps ({max_laps})")
                    break
                data = self.telemetry.get_latest()

                if data is None:
                    time.sleep(0.01)
                    continue

                if self.check_lap_complete(data):
                    self.save_lap(data)
                    laps_recorded += 1

                    self.last_lap_count = data["lap"]["get_lap_count"]
                    self.lap_start_time = time.time()
                    self.recording = True

                    if max_laps and laps_recorded >= max_laps:
                        print(f"\n✓ Completed recording {max_laps} laps")
                        break

                current_time = time.time()
                if (
                    self.recording
                    and (current_time - last_sample_time) >= self.sample_rate
                ):
                    self.record_position(data)
                    last_sample_time = current_time

                    if len(self.current_lap_positions) % 50 == 0:
                        print(
                            f"\rLap {self.last_lap_count}: {len(self.current_lap_positions)} points recorded",
                            end="",
                            flush=True,
                        )

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\n⚠ Recording interrupted by user")

            if self.current_lap_positions:
                print("\nSaving current incomplete lap...")
                self.save_lap(data)

        finally:
            if self.completed_laps:
                self.save_to_file()
            else:
                print("\n⚠ No complete laps recorded")

            self.telemetry.close()
            print("\n✓ Recorder closed")


def main():
    """Main entry point for the racing line recorder."""
    parser = argparse.ArgumentParser(
        description="Record racing line from Assetto Corsa telemetry"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="racing_lines.json",
        help="Output file path (default: racing_lines.json)",
    )
    parser.add_argument(
        "--track",
        "-t",
        type=str,
        help="Track name (used for output filename)",
    )
    parser.add_argument(
        "--laps",
        "-l",
        type=int,
        default=2,
        help="Maximum number of laps to record (default: 1)",
    )
    parser.add_argument(
        "--sample-rate",
        "-s",
        type=float,
        default=0.05,
        help="Sample rate in seconds (default: 0.05 = 20Hz)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Telemetry host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=9876,
        help="Telemetry receive port (default: 9876)",
    )

    args = parser.parse_args()

    output_file = args.output
    if args.track:
        output_path = Path(output_file)
        output_file = str(
            output_path.parent
            / f"{args.track}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    recorder = RacingLineRecorder(
        output_file=output_file,
        host=args.host,
        recv_port=args.port,
        sample_rate=args.sample_rate,
    )

    recorder.run(max_laps=args.laps)


if __name__ == "__main__":
    main()
