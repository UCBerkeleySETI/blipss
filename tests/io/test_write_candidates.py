"""Unit tests for modules in blipss.io.write_candidates"""

import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt

from blipss.constants import FFA_CANDIDATE_CSV_COLUMNS
from blipss.io.write_candidates import write_candidates_csv

_FREQS_MHZ: npt.NDArray[np.floating] = np.array([1000.0, 1001.0, 1002.0, 1003.0])


def _read_csv(path: Path) -> list[list[str]]:
    """Return all rows of a CSV file as lists of raw strings."""
    with path.open(newline="") as f:
        return list(csv.reader(f))


def test_write_candidates_csv_writes_header_row(tmp_path: Path) -> None:
    """write_candidates_csv writes FFA_CANDIDATE_CSV_COLUMNS as the first row."""
    output_path: Path = tmp_path / "cands.csv"
    write_candidates_csv(
        output_path,
        _FREQS_MHZ,
        np.array([0], dtype=np.intp),
        np.array([10.0]),
        np.array([8.0]),
        np.array([10], dtype=np.uint),
        np.array([2], dtype=np.uint),
        np.array(["F"]),
        0,
    )
    assert _read_csv(output_path)[0] == FFA_CANDIDATE_CSV_COLUMNS


def test_write_candidates_csv_sorts_rows_by_descending_snr(tmp_path: Path) -> None:
    """write_candidates_csv orders candidate rows from highest to lowest S/N."""
    output_path: Path = tmp_path / "cands.csv"
    write_candidates_csv(
        output_path,
        _FREQS_MHZ,
        np.array([0, 1, 2], dtype=np.intp),
        np.array([10.0, 20.0, 30.0]),
        np.array([8.0, 15.0, 11.0]),
        np.array([10, 10, 10], dtype=np.uint),
        np.array([2, 2, 2], dtype=np.uint),
        np.array(["F", "F", "H"]),
        0,
    )
    rows = _read_csv(output_path)[1:]
    np.testing.assert_array_equal([float(row[5]) for row in rows], [15.0, 11.0, 8.0])
    np.testing.assert_array_equal([int(row[0]) for row in rows], [1, 2, 0])


def test_write_candidates_csv_column_order_and_values(tmp_path: Path) -> None:
    """write_candidates_csv writes channel, frequency, bins, width, period, S/N, and flag in that order."""
    output_path: Path = tmp_path / "cands.csv"
    write_candidates_csv(
        output_path,
        _FREQS_MHZ,
        np.array([2], dtype=np.intp),
        np.array([12.5]),
        np.array([9.25]),
        np.array([16], dtype=np.uint),
        np.array([3], dtype=np.uint),
        np.array(["S"]),
        0,
    )
    assert _read_csv(output_path)[1] == ["2", "1002.0", "16", "3", "12.5", "9.25", "S"]


def test_write_candidates_csv_maps_frequencies_using_start_ch_offset(tmp_path: Path) -> None:
    """write_candidates_csv indexes freqs_MHz by candidate channel minus start_ch."""
    output_path: Path = tmp_path / "cands.csv"
    write_candidates_csv(
        output_path,
        _FREQS_MHZ,
        np.array([100, 103], dtype=np.intp),
        np.array([10.0, 20.0]),
        np.array([12.0, 8.0]),
        np.array([10, 10], dtype=np.uint),
        np.array([2, 2], dtype=np.uint),
        np.array(["F", "F"]),
        100,
    )
    rows = _read_csv(output_path)[1:]
    np.testing.assert_allclose([float(row[1]) for row in rows], [1000.0, 1003.0])


def test_write_candidates_csv_empty_candidates_writes_header_only(tmp_path: Path) -> None:
    """write_candidates_csv writes only the header row when there are no candidates."""
    output_path: Path = tmp_path / "cands.csv"
    write_candidates_csv(
        output_path,
        _FREQS_MHZ,
        np.array([], dtype=np.intp),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=np.uint),
        np.array([], dtype=np.uint),
        np.array([], dtype=np.str_),
        0,
    )
    assert _read_csv(output_path) == [FFA_CANDIDATE_CSV_COLUMNS]
