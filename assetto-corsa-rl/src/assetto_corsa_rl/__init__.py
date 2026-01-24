"""Assetto Corsa Reinforcement Learning Toolkit.

A comprehensive toolkit for training RL agents in Assetto Corsa and Car Racing environments.
"""

__version__ = "0.1.3"
__author__ = "Assetto Corsa RL Contributors"
__license__ = "MIT"

# Import key components for easier access
from .ac_env import (
    AssettoCorsa,
    make_env,
    create_transformed_env,
    create_mock_env,
    parse_image_shape,
    get_device,
)

__all__ = [
    "__version__",
    "AssettoCorsa",
    "make_env",
    "create_transformed_env",
    "create_mock_env",
    "parse_image_shape",
    "get_device",
]
