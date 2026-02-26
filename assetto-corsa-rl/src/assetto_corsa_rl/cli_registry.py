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
from importlib.resources import files
from types import SimpleNamespace
from pathlib import Path
import functools
import yaml


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
        cmd_name = name or func.__name__.replace("_", "-")
        if cmd_name == "main":
            import inspect

            module = inspect.getmodule(func)
            if module and module.__name__ != "__main__":
                parts = module.__name__.split(".")
                cmd_name = parts[-1].replace("_", "-")

        cmd_help = help
        if cmd_help is None and func.__doc__:
            cmd_help = func.__doc__.strip().split("\n")[0]

        options = getattr(func, "_cli_options", [])

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


def _resolve_config_root():
    try:
        return Path(files("assetto_corsa_rl"))
    except Exception:
        return Path(__file__).resolve().parents[4]


def load_cfg_from_yaml(root: Path = None):
    """Load configs/ac/env_config.yaml, model_config.yaml, train_config.yaml and merge."""
    if root is None:
        root = _resolve_config_root()

    env_p = root / "configs" / "ac" / "env_config.yaml"
    model_p = root / "configs" / "ac" / "model_config.yaml"
    train_p = root / "configs" / "ac" / "train_config.yaml"

    def _read(p):
        try:
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: could not read config {p}: {e}")
            return {}

    env = _read(env_p).get("environment", {})
    model = _read(model_p).get("model", {})
    train_raw = _read(train_p)
    train = {}
    if isinstance(train_raw, dict):
        train.update(train_raw.get("train", {}))
        train.update(train_raw.get("training", {}))

    cfg_dict = {}
    cfg_dict.update(model)
    cfg_dict.update(env)
    cfg_dict.update(train)

    input_config = env.get("inputs", None)

    def _try_convert(x):
        if x is None or isinstance(x, bool):
            return x
        if isinstance(x, dict):
            return {k: _try_convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_try_convert(v) for v in x]
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return x
        if isinstance(x, str):
            s = x.strip().replace(",", "").replace("_", "")
            try:
                if "." not in s and "e" not in s.lower():
                    return int(s)
                return float(s)
            except Exception:
                return x
        return x

    converted = {k: _try_convert(v) for k, v in cfg_dict.items()}

    if isinstance(converted.get("wandb"), dict):
        wandb_dict = converted.pop("wandb")
        for k, v in wandb_dict.items():
            converted[f"wandb_{k}"] = v

    for k in ("wandb_project", "wandb_entity", "wandb_name", "wandb_enabled"):
        converted.setdefault(k, None)

    converted["num_envs"] = 1
    converted["input_config"] = input_config

    cfg = SimpleNamespace(**converted)
    print(f"Loaded config from: {env_p}, {model_p}, {train_p}")
    if input_config:
        enabled_count = sum(1 for v in input_config.values() if v)
        print(f"  Input config: {enabled_count}/{len(input_config)} inputs enabled")
    return cfg
