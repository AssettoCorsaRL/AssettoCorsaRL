"""CLI command registration system.

This module provides decorators that allow scripts to self-register as CLI commands
without requiring manual updates to cli.py.

Usage in a script:
    from assetto_corsa_rl.cli_registry import cli_command, cli_option

    @cli_command(group="ac", name="my-command", help="My command description")
    @cli_option("--epochs", default=50, help="Number of epochs")
    @cli_option("--batch-size", default=64, help="Batch size")
    def main(epochs, batch_size):
        print(f"Training with {epochs} epochs and batch size {batch_size}")
"""

from __future__ import annotations

from typing import Callable, Any, Dict, List, Optional
import functools

# Global registry of commands
_COMMAND_REGISTRY: Dict[str, List[Dict[str, Any]]] = {}


class CLIOption:
    """Represents a CLI option to be added to a command."""

    def __init__(self, *param_decls, **attrs):
        self.param_decls = param_decls
        self.attrs = attrs


def cli_option(*param_decls, **attrs):
    """Decorator to add a CLI option to a command.

    Args:
        *param_decls: Option names (e.g., "--epochs", "-e")
        **attrs: Click option attributes (default, type, help, is_flag, etc.)

    Example:
        @cli_option("--epochs", default=50, help="Number of epochs")
        @cli_option("--batch-size", "-b", default=64, type=int)
        def main(epochs, batch_size):
            pass
    """

    def decorator(func):
        if not hasattr(func, "_cli_options"):
            func._cli_options = []
        func._cli_options.insert(0, CLIOption(*param_decls, **attrs))
        return func

    return decorator


def cli_command(
    group: str,
    name: Optional[str] = None,
    help: Optional[str] = None,
    short_help: Optional[str] = None,
):
    """Decorator to register a function as a CLI command.

    Args:
        group: Command group (e.g., "ac", "car-racing")
        name: Command name (defaults to function name with underscores->hyphens)
        help: Command help text
        short_help: Short help text for command listing

    Example:
        @cli_command(group="ac", name="train", help="Train the model")
        @cli_option("--epochs", default=50)
        def main(epochs):
            print(f"Training for {epochs} epochs")
    """

    def decorator(func: Callable) -> Callable:
        # Get command name from function name if not provided
        cmd_name = name or func.__name__.replace("_", "-")
        if cmd_name == "main":
            # If function is called "main", try to infer from module
            import inspect

            module = inspect.getmodule(func)
            if module and module.__name__ != "__main__":
                # Use module name as command name
                parts = module.__name__.split(".")
                cmd_name = parts[-1].replace("_", "-")

        # Get help text from docstring if not provided
        cmd_help = help
        if cmd_help is None and func.__doc__:
            # Extract first line of docstring
            cmd_help = func.__doc__.strip().split("\n")[0]

        # Extract options from decorated function
        options = getattr(func, "_cli_options", [])

        # Register the command
        if group not in _COMMAND_REGISTRY:
            _COMMAND_REGISTRY[group] = []

        _COMMAND_REGISTRY[group].append(
            {
                "name": cmd_name,
                "func": func,
                "help": cmd_help,
                "short_help": short_help,
                "options": options,
            }
        )

        return func

    return decorator


def get_registered_commands(
    group: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Get all registered commands, optionally filtered by group.

    Args:
        group: Optional group name to filter by

    Returns:
        Dictionary mapping group names to lists of command definitions
    """
    if group:
        return {group: _COMMAND_REGISTRY.get(group, [])}
    return _COMMAND_REGISTRY.copy()


def clear_registry():
    """Clear the command registry (mainly for testing)."""
    _COMMAND_REGISTRY.clear()
