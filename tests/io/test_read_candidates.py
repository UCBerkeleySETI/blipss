"""Unit tests for modules in blipss.io.read_candidates"""

import csv
from pathlib import Path

import pytest

from blipss.constants import FFA_CANDIDATE_CSV_COLUMNS
from blipss.io.read_candidates import read_candidates_csv


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Write a candidate CSV file with the given header and rows."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_read_candidates_csv_returns_columns_in_declared_order(tmp_path: Path) -> None:
    """read_candidates_csv returns arrays matching FFA_CANDIDATE_CSV_COLUMNS values, in file row order."""
    csv_path = tmp_path / "cands.csv"
    _write_csv(
        csv_path,
        FFA_CANDIDATE_CSV_COLUMNS,
        [["0", "1000.0", "10", "2", "1.5", "8.0", "F"], ["1", "1001.0", "20", "3", "2.5", "9.0", "H"]],
    )

    channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, flags = read_candidates_csv(csv_path)

    assert list(channels) == [0, 1]
    assert list(radiofreqs) == [1000.0, 1001.0]
    assert list(phase_bins) == [10, 20]
    assert list(boxcar_widths) == [2, 3]
    assert list(periods) == [1.5, 2.5]
    assert list(snrs) == [8.0, 9.0]
    assert list(flags) == ["F", "H"]


def test_read_candidates_csv_single_row_returns_length_one_arrays(tmp_path: Path) -> None:
    """read_candidates_csv returns 1-D arrays of length 1, not scalars, for a single-row file."""
    csv_path = tmp_path / "cands.csv"
    _write_csv(csv_path, FFA_CANDIDATE_CSV_COLUMNS, [["3", "1003.0", "10", "2", "4.0", "7.5", "S"]])

    channels, _, _, _, periods, _, _ = read_candidates_csv(csv_path)

    assert channels.shape == (1,)
    assert periods.shape == (1,)
    assert channels[0] == 3


def test_read_candidates_csv_column_order_in_file_is_independent_of_header_order(tmp_path: Path) -> None:
    """read_candidates_csv maps columns by header name, independent of their physical order in the file."""
    csv_path = tmp_path / "cands.csv"
    shuffled_header = list(reversed(FFA_CANDIDATE_CSV_COLUMNS))
    shuffled_row = list(reversed(["0", "1000.0", "10", "2", "1.5", "8.0", "F"]))
    _write_csv(csv_path, shuffled_header, [shuffled_row])

    channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, flags = read_candidates_csv(csv_path)

    assert channels[0] == 0
    assert radiofreqs[0] == 1000.0
    assert phase_bins[0] == 10
    assert boxcar_widths[0] == 2
    assert periods[0] == 1.5
    assert snrs[0] == 8.0
    assert flags[0] == "F"


def test_read_candidates_csv_missing_column_raises_value_error(tmp_path: Path) -> None:
    """read_candidates_csv raises ValueError when the header is missing a required column."""
    csv_path = tmp_path / "cands.csv"
    incomplete_header = [c for c in FFA_CANDIDATE_CSV_COLUMNS if c != "S/N"]
    _write_csv(csv_path, incomplete_header, [])

    with pytest.raises(ValueError, match="not in list"):
        read_candidates_csv(csv_path)
