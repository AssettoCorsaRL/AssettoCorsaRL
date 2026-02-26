"""CLI command that reports the resolved paths of all AC config files."""

import sys
from pathlib import Path

try:
    from assetto_corsa_rl.cli_registry import cli_command, _resolve_config_root
except Exception:
    from ...src.assetto_corsa_rl.cli_registry import cli_command, _resolve_config_root  # type: ignore


@cli_command(
    group="ac", name="config-locations", help="Print the resolved paths of all AC config files"
)
def give_config_locations():
    root = _resolve_config_root()

    configs = {
        "env_config": root / "configs" / "ac" / "env_config.yaml",
        "model_config": root / "configs" / "ac" / "model_config.yaml",
        "train_config": root / "configs" / "ac" / "train_config.yaml",
    }

    print(f"\nConfig root: {root}\n")
    print(f"{'Config':<20} {'Exists':<8} Path")
    print("-" * 80)

    all_found = True
    for name, path in configs.items():
        exists = path.exists()
        status = "✓" if exists else "✗ MISSING (dm @ved patel on slack)"
        print(f"{name:<20} {status:<8} {path}")
        if not exists:
            all_found = False

    print()
    if all_found:
        print("All config files found.")
    else:
        print("Warning: one or more config files are missing.")
        print("Expected layout:")
        print("  <repo_root>/configs/ac/env_config.yaml")
        print("  <repo_root>/configs/ac/model_config.yaml")
        print("  <repo_root>/configs/ac/train_config.yaml")
        print(f"\nResolved repo root: {root}")
        print("If this looks wrong, the script may not be at the expected depth in the repo.")
        print("You can override by calling _resolve_config_root(Path('/your/repo/root')) manually.")
    print()


if __name__ == "__main__":
    give_config_locations()
