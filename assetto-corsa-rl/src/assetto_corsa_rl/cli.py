"""Command-line interface for Assetto Corsa RL toolkit.

This module provides a unified CLI for all training, testing, and utility scripts.
Commands are automatically discovered from scripts using the @cli_command decorator.
"""

import sys
import os
from pathlib import Path
import click
import importlib
import importlib.util

from assetto_corsa_rl.cli_registry import get_registered_commands


def _find_scripts_dir():
    # 1. dev mode - scripts at repo root
    dev_scripts = Path(__file__).parent.parent.parent / "scripts"
    if dev_scripts.exists():
        return dev_scripts

    # 2. via pip - check sys.prefix/share
    installed_scripts = Path(sys.prefix) / "share" / "assetto_corsa_rl" / "scripts"
    if installed_scripts.exists():
        return installed_scripts

    # 3. check in package directory (if copied during install)
    package_scripts = Path(__file__).parent / "scripts"
    if package_scripts.exists():
        return package_scripts

    return None


_scripts_dir = _find_scripts_dir()


def discover_commands():
    """Discover and import all scripts with CLI decorators."""
    if _scripts_dir is None or not _scripts_dir.exists():
        print(f"Warning: Scripts directory not found. Searched locations:", file=sys.stderr)
        print(
            f"  - Development: {Path(__file__).parent.parent.parent / 'scripts'}", file=sys.stderr
        )
        print(
            f"  - Installed: {Path(sys.prefix) / 'share' / 'assetto_corsa_rl' / 'scripts'}",
            file=sys.stderr,
        )
        return

    # Add scripts directory to path for imports
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))

    # Import all AC scripts
    ac_scripts_dir = _scripts_dir / "ac"
    if ac_scripts_dir.exists():
        for script_file in ac_scripts_dir.glob("*.py"):
            if script_file.name.startswith("_"):
                continue
            try:
                module_name = f"ac.{script_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, script_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
            except Exception as e:
                print(
                    f"Warning: Could not import {script_file.name}: {e}",
                    file=sys.stderr,
                )

    car_racing_scripts_dir = _scripts_dir / "car-racing"
    if car_racing_scripts_dir.exists():
        for script_file in car_racing_scripts_dir.glob("*.py"):
            if script_file.name.startswith("_"):
                continue
            try:
                module_name = f"car_racing.{script_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, script_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
            except Exception as e:
                print(
                    f"Warning: Could not import {script_file.name}: {e}",
                    file=sys.stderr,
                )


@click.group()
@click.version_option(version="0.1.3", prog_name="acrl")
def cli():
    """Assetto Corsa Reinforcement Learning Toolkit.

    A comprehensive toolkit for training RL agents in Assetto Corsa and Car Racing environments.
    """
    pass


def build_cli():
    """Build the CLI by registering all decorated commands."""
    discover_commands()

    all_commands = get_registered_commands()

    for group_name, commands in all_commands.items():

        def make_group_factory(gname):
            @cli.group(name=gname)
            def group():
                pass

            if gname == "ac":
                group.__doc__ = "Assetto Corsa commands"
            elif gname == "car-racing":
                group.__doc__ = "Car Racing (Gym) commands"
            else:
                group.__doc__ = f"{gname} commands"

            return group

        current_group = make_group_factory(group_name)
        for cmd_def in commands:
            original_func = cmd_def["func"]
            options = cmd_def.get("options", [])

            def make_cmd_func(func):
                def wrapper(**kwargs):
                    return func(**kwargs)

                return wrapper

            cmd_func = make_cmd_func(original_func)
            cmd_func.__name__ = cmd_def["name"].replace("-", "_")
            cmd_func.__doc__ = cmd_def.get("help") or original_func.__doc__

            cmd_func = current_group.command(
                name=cmd_def["name"],
                help=cmd_def.get("help"),
                short_help=cmd_def.get("short_help"),
            )(cmd_func)

            for opt in reversed(options):
                cmd_func = click.option(*opt.param_decls, **opt.attrs)(cmd_func)


def main():
    """Entry point for the CLI."""
    build_cli()
    cli()


if __name__ == "__main__":
    main()
