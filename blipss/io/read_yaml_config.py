"""Routines for reading YAML config files"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """
    Read a YAML config file and return its contents as a dictionary.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary containing the parsed YAML config

    Raises:
        FileNotFoundError: When config_path does not point to an existing file
        TypeError: When the file is empty or its top-level structure is not a mapping
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping at the top level, got {type(config).__name__}")

    return config
