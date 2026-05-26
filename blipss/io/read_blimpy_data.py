"""Utilities for loading blimpy-compatible filterbank and HDF5 data files"""

from pathlib import Path

from blimpy import Waterfall


def read_waterfall_file(file_path: Path | str, max_memory_gb: float) -> Waterfall:
    """
    Load a .h5 or .fil file into a blimpy Waterfall object.

    Args:
        file_path: Path to the .h5 or .fil data file to load
        max_memory_gb: Maximum data size in GB permitted in memory

    Returns:
        Blimpy Waterfall object containing the data file contents
    """
    return Waterfall(str(file_path), max_load=max_memory_gb)
