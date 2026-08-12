"""Unit tests for modules in blipss.utils.general_utils"""

from pathlib import Path

import pytest

from blipss.utils.general_utils import check_file_exists, ensure_path_exists


def test_ensure_path_exists_creates_new_directory(tmp_path: Path) -> None:
    """ensure_path_exists creates a single-level directory that did not previously exist."""
    new_dir: Path = tmp_path / "new_dir"
    assert not new_dir.exists()
    ensure_path_exists(new_dir)
    assert new_dir.is_dir()


@pytest.mark.parametrize(
    "subpath",
    [
        "a/b",
        "x/y/z",
        "p/q/r/s",
    ],
    ids=["two_levels", "three_levels", "four_levels"],
)
def test_ensure_path_exists_creates_nested_directories(tmp_path: Path, subpath: str) -> None:
    """ensure_path_exists creates the full directory tree including missing intermediate parents."""
    nested: Path = tmp_path / subpath
    assert not nested.exists()
    ensure_path_exists(nested)
    assert nested.is_dir()


def test_ensure_path_exists_noop_when_dir_already_exists(tmp_path: Path) -> None:
    """ensure_path_exists returns immediately without error when the path is an existing directory."""
    existing_dir: Path = tmp_path / "existing"
    existing_dir.mkdir()
    ensure_path_exists(existing_dir)
    assert existing_dir.is_dir()


def test_ensure_path_exists_creates_parent_when_path_is_file(tmp_path: Path) -> None:
    """ensure_path_exists creates the parent directory tree when the path points to an existing file."""
    file_path: Path = tmp_path / "subdir" / "data.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("content")
    ensure_path_exists(file_path)
    assert file_path.parent.is_dir()


def test_ensure_path_exists_file_branch_leaves_file_intact(tmp_path: Path) -> None:
    """ensure_path_exists does not delete or modify an existing file when taking the file branch."""
    file_path: Path = tmp_path / "data.txt"
    file_path.write_text("hello")
    ensure_path_exists(file_path)
    assert file_path.read_text() == "hello"


def test_check_file_exists_true_for_existing_file(tmp_path: Path) -> None:
    """check_file_exists returns True for a path pointing at an existing regular file."""
    file_path: Path = tmp_path / "data.fil"
    file_path.write_text("content")
    assert check_file_exists(file_path) is True


def test_check_file_exists_false_for_missing_path(tmp_path: Path) -> None:
    """check_file_exists returns False for a path that does not exist."""
    assert check_file_exists(tmp_path / "absent.fil") is False


def test_check_file_exists_false_for_directory(tmp_path: Path) -> None:
    """check_file_exists returns False for an existing directory, which is not a regular file."""
    dir_path: Path = tmp_path / "subdir"
    dir_path.mkdir()
    assert check_file_exists(dir_path) is False
