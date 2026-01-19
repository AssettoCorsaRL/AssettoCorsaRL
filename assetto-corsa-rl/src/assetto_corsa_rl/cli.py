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

# Add scripts directory to path
_scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from assetto_corsa_rl.cli_registry import get_registered_commands


def discover_commands():
    """Discover and import all scripts with CLI decorators."""
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

    # Import all car-racing scripts
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
@click.version_option(version="0.1.1", prog_name="acrl")
def cli():
    """Assetto Corsa Reinforcement Learning Toolkit.

    A comprehensive toolkit for training RL agents in Assetto Corsa and Car Racing environments.
    """
    pass


def build_cli():
    """Build the CLI by registering all decorated commands."""
    # Discover and import all scripts
    discover_commands()

    # Get registered commands
    all_commands = get_registered_commands()

    # Create groups and commands dynamically
    for group_name, commands in all_commands.items():
        # Create the group
        @cli.group(name=group_name)
        def make_group():
            pass

        if group_name == "ac":
            make_group.__doc__ = "Assetto Corsa commands"
        elif group_name == "car-racing":
            make_group.__doc__ = "Car Racing (Gym) commands"
        else:
            make_group.__doc__ = f"{group_name} commands"

        # Add each command to the group
        for cmd_def in commands:
            original_func = cmd_def["func"]
            options = cmd_def.get("options", [])

            # Create a wrapper function for each command
            def make_cmd_func(func):
                def wrapper(**kwargs):
                    return func(**kwargs)

                return wrapper

            cmd_func = make_cmd_func(original_func)
            cmd_func.__name__ = cmd_def["name"].replace("-", "_")
            cmd_func.__doc__ = cmd_def.get("help") or original_func.__doc__

            # Apply click.command decorator
            cmd_func = make_group.command(
                name=cmd_def["name"],
                help=cmd_def.get("help"),
                short_help=cmd_def.get("short_help"),
            )(cmd_func)

            # Apply options in reverse order (click stacks them)
            for opt in reversed(options):
                cmd_func = click.option(*opt.param_decls, **opt.attrs)(cmd_func)


def main():
    """Entry point for the CLI."""
    build_cli()
    cli()


if __name__ == "__main__":
    main()
