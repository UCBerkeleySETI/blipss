"""Unit tests for modules in blipss.io.read_blimpy_data"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from blipss.io.read_blimpy_data import read_waterfall_file


@patch("blipss.io.read_blimpy_data.Waterfall")
def test_read_waterfall_file_returns_waterfall_object(mock_waterfall: MagicMock) -> None:
    """read_waterfall_file returns the Waterfall instance created by blimpy."""
    mock_instance: MagicMock = MagicMock()
    mock_waterfall.return_value = mock_instance
    result = read_waterfall_file(Path("/data/obs.h5"), max_memory_gb=1.0)
    assert result is mock_instance


@pytest.mark.parametrize(
    ("file_path", "max_memory_gb"),
    [
        (Path("/data/obs.h5"), 1.0),
        (Path("/data/obs.fil"), 4.0),
        ("/data/flat.h5", 0.25),
    ],
    ids=["path_h5", "path_fil", "str_h5"],
)
@patch("blipss.io.read_blimpy_data.Waterfall")
def test_read_waterfall_file_call_args(
    mock_waterfall: MagicMock,
    file_path: Path | str,
    max_memory_gb: float,
) -> None:
    """read_waterfall_file calls Waterfall with str(file_path) and max_load for every input type."""
    read_waterfall_file(file_path, max_memory_gb=max_memory_gb)
    mock_waterfall.assert_called_once_with(str(file_path), max_load=max_memory_gb)
