"""Unit tests for modules in blipss.io.write_compared_candidates"""

import csv
from pathlib import Path

import numpy as np

from blipss.constants import COMPARE_CANDS_CSV_COLUMNS
from blipss.io.write_compared_candidates import write_compared_candidates_csv


def _read_csv(path: Path) -> list[list[str]]:
    """Return all rows of a CSV file as lists of raw strings."""
    with path.open(newline="") as f:
        return list(csv.reader(f))


def test_write_compared_candidates_csv_writes_header_row(tmp_path: Path) -> None:
    """write_compared_candidates_csv writes COMPARE_CANDS_CSV_COLUMNS as the first row."""
    output_path = tmp_path / "compared.csv"
    write_compared_candidates_csv(
        output_path,
        np.array([0], dtype=np.intp),
        np.array([1000.0]),
        np.array([10], dtype=np.uint),
        np.array([2], dtype=np.uint),
        np.array([1.5]),
        np.array([8.0]),
        np.array(["10"]),
    )
    assert _read_csv(output_path)[0] == COMPARE_CANDS_CSV_COLUMNS


def test_write_compared_candidates_csv_preserves_given_row_order(tmp_path: Path) -> None:
    """write_compared_candidates_csv writes rows in the order given, without re-sorting by S/N."""
    output_path = tmp_path / "compared.csv"
    write_compared_candidates_csv(
        output_path,
        np.array([2, 0, 1], dtype=np.intp),
        np.array([1002.0, 1000.0, 1001.0]),
        np.array([10, 10, 10], dtype=np.uint),
        np.array([2, 2, 2], dtype=np.uint),
        np.array([3.0, 1.0, 2.0]),
        np.array([5.0, 9.0, 7.0]),
        np.array(["100", "010", "001"]),
    )
    rows = _read_csv(output_path)[1:]
    assert [row[0] for row in rows] == ["2", "0", "1"]
    assert [row[6] for row in rows] == ["100", "010", "001"]


def test_write_compared_candidates_csv_column_order_and_values(tmp_path: Path) -> None:
    """write_compared_candidates_csv writes channel, frequency, bins, width, period, S/N, and code in that order."""
    output_path = tmp_path / "compared.csv"
    write_compared_candidates_csv(
        output_path,
        np.array([2], dtype=np.intp),
        np.array([12.5]),
        np.array([16], dtype=np.uint),
        np.array([3], dtype=np.uint),
        np.array([9.25]),
        np.array([8.0]),
        np.array(["1101"]),
    )
    assert _read_csv(output_path)[1] == ["2", "12.5", "16", "3", "9.25", "8.0", "1101"]


def test_write_compared_candidates_csv_empty_candidates_writes_header_only(tmp_path: Path) -> None:
    """write_compared_candidates_csv writes only the header row when there are no candidates."""
    output_path = tmp_path / "compared.csv"
    write_compared_candidates_csv(
        output_path,
        np.array([], dtype=np.intp),
        np.array([], dtype=float),
        np.array([], dtype=np.uint),
        np.array([], dtype=np.uint),
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([], dtype=np.str_),
    )
    assert _read_csv(output_path) == [COMPARE_CANDS_CSV_COLUMNS]
