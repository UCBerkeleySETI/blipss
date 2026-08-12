"""General filesystem utility functions"""

from pathlib import Path


def ensure_path_exists(path: Path) -> None:
    """
    Create a directory at the given path, handling file and missing-parent cases.

    If the path already exists as a directory, returns immediately. If the path
    points to an existing file, creates the file's parent directory tree instead.
    Otherwise creates the directory along with any missing intermediate parents.

    Args:
        path: Filesystem path at which to create the directory
    """
    if path.is_dir():
        return
    if path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        return
    path.mkdir(parents=True, exist_ok=True)


def check_file_exists(filepath: Path) -> bool:
    """
    Return True if the given path points to an existing file, False otherwise.

    Args:
        filepath: Filesystem path to check

    Returns:
        True if ``filepath`` exists and is a regular file, False otherwise
    """
    return filepath.is_file()
