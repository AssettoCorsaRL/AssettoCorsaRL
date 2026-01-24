"""Setup script for assetto-corsa-rl package.

For modern Python packaging, see pyproject.toml instead.
This file exists for backward compatibility and to handle data files.
"""

from setuptools import setup
import glob
from pathlib import Path


# Collect scripts and configs as data files to install alongside the package
def get_data_files():
    data_files = []

    # Collect all script files
    script_files = []
    for pattern in ["scripts/**/*.py"]:
        for file in glob.glob(pattern, recursive=True):
            if "__pycache__" not in file and not file.endswith(".pyc"):
                script_files.append(file)

    # Collect all config files
    config_files = []
    for pattern in ["configs/**/*.yaml", "configs/**/*.json"]:
        for file in glob.glob(pattern, recursive=True):
            config_files.append(file)

    # Install to share/assetto_corsa_rl/
    if script_files:
        data_files.append(
            (
                "share/assetto_corsa_rl/scripts/ac",
                [f for f in script_files if "scripts/ac" in f and f.endswith(".py")],
            )
        )
        data_files.append(
            (
                "share/assetto_corsa_rl/scripts/car-racing",
                [f for f in script_files if "scripts/car-racing" in f and f.endswith(".py")],
            )
        )

    if config_files:
        data_files.append(
            ("share/assetto_corsa_rl/configs/ac", [f for f in config_files if "configs/ac" in f])
        )
        data_files.append(
            (
                "share/assetto_corsa_rl/configs/car-racing",
                [f for f in config_files if "configs/car-racing" in f],
            )
        )

    return data_files


# Configuration is in pyproject.toml and setup.cfg
setup(data_files=get_data_files())
