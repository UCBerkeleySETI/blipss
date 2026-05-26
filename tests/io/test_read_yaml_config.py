"""Unit tests for modules in blipss.io.read_yaml_config"""

from pathlib import Path
from typing import Any

import pytest

from blipss.io.read_yaml_config import load_yaml_config


def test_load_yaml_config_returns_dict(tmp_path: Path) -> None:
    """load_yaml_config returns a dict for a valid single-level YAML mapping."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("key: value\n")
    result: dict[str, Any] = load_yaml_config(config_file)
    assert result == {"key": "value"}


def test_load_yaml_config_nested_mapping(tmp_path: Path) -> None:
    """load_yaml_config correctly parses nested YAML mappings."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("outer:\n  inner: 42\nflag: true\n")
    result: dict[str, Any] = load_yaml_config(config_file)
    assert result == {"outer": {"inner": 42}, "flag": True}


def test_load_yaml_config_file_not_found(tmp_path: Path) -> None:
    """load_yaml_config raises FileNotFoundError when the path does not exist."""
    missing: Path = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_yaml_config(missing)


@pytest.mark.parametrize(
    ("content", "top_level_type"),
    [
        ("", "NoneType"),
        ("- item1\n- item2\n", "list"),
        ("42\n", "int"),
    ],
    ids=["empty_file", "list_top_level", "scalar_top_level"],
)
def test_load_yaml_config_raises_type_error(tmp_path: Path, content: str, top_level_type: str) -> None:
    """load_yaml_config raises TypeError when the top-level YAML structure is not a mapping."""
    config_file: Path = tmp_path / "config.yaml"
    config_file.write_text(content)
    with pytest.raises(TypeError, match=f"Expected a YAML mapping at the top level, got {top_level_type}"):
        load_yaml_config(config_file)
