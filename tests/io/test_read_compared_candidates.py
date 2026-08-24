"""Unit tests for modules in blipss.io.read_compared_candidates"""

import csv
from pathlib import Path

import pytest

from blipss.constants import COMPARE_CANDS_CSV_COLUMNS
from blipss.io.read_compared_candidates import read_compared_candidates_csv


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write a compared-candidates CSV file with the given header and rows."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_read_compared_candidates_csv_returns_columns_in_declared_order(tmp_path: Path) -> None:
    """read_compared_candidates_csv returns arrays matching COMPARE_CANDS_CSV_COLUMNS values, in file row order."""
    csv_path = tmp_path / "compared.csv"
    _write_csv(
        csv_path,
        COMPARE_CANDS_CSV_COLUMNS,
        [["0", "1000.0", "10", "2", "1.5", "8.0", "10"], ["1", "1001.0", "20", "3", "2.5", "9.0", "01"]],
    )

    channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, codes = read_compared_candidates_csv(csv_path)

    assert list(channels) == [0, 1]
    assert list(radiofreqs) == [1000.0, 1001.0]
    assert list(phase_bins) == [10, 20]
    assert list(boxcar_widths) == [2, 3]
    assert list(periods) == [1.5, 2.5]
    assert list(snrs) == [8.0, 9.0]
    assert list(codes) == ["10", "01"]


def test_read_compared_candidates_csv_preserves_leading_zeros_in_code(tmp_path: Path) -> None:
    """read_compared_candidates_csv reads the Code column as a string, keeping leading zeros intact."""
    csv_path = tmp_path / "compared.csv"
    _write_csv(csv_path, COMPARE_CANDS_CSV_COLUMNS, [["0", "1000.0", "10", "2", "1.5", "8.0", "001"]])

    _, _, _, _, _, _, codes = read_compared_candidates_csv(csv_path)

    assert codes[0] == "001"


def test_read_compared_candidates_csv_missing_column_raises_value_error(tmp_path: Path) -> None:
    """read_compared_candidates_csv raises ValueError when the header is missing a required column."""
    csv_path = tmp_path / "compared.csv"
    incomplete_header = [c for c in COMPARE_CANDS_CSV_COLUMNS if c != "Code"]
    _write_csv(csv_path, incomplete_header, [])

    with pytest.raises(ValueError, match="columns expected but not found"):
        read_compared_candidates_csv(csv_path)
