"""Read utilities for FFA candidate detection CSV files."""

import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt

from blipss.constants import FFA_CANDIDATE_CSV_COLUMNS

# Structured dtype matching FFA_CANDIDATE_CSV_COLUMNS order. Harmonic flags are always the
# single-character FUNDAMENTAL_FLAG/HARMONIC_FLAG/SUBHARMONIC_FLAG values from blipss.constants.
_CANDIDATE_DTYPE = [
    ("channels", np.intp),
    ("radiofreqs", np.float64),
    ("phase_bins", np.uint),
    ("boxcar_widths", np.uint),
    ("periods", np.float64),
    ("snrs", np.float64),
    ("flags", "U1"),
]


def read_candidates_csv(
    csv_path: Path,
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.floating],
    npt.NDArray[np.uint],
    npt.NDArray[np.uint],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.str_],
]:
    """
    Read FFA candidate detections previously written by ``write_candidates_csv``.

    Args:
        csv_path: Path to a candidate CSV file with ``FFA_CANDIDATE_CSV_COLUMNS`` columns.

    Returns:
        Tuple of (channels, radiofreqs_MHz, phase_bins, boxcar_widths, periods, snrs, flags)
        arrays, one entry per candidate row in file order.

    Raises:
        ValueError: When the file header is missing one of ``FFA_CANDIDATE_CSV_COLUMNS``.
    """
    with csv_path.open(newline="") as f:
        header = next(csv.reader(f))
    col = [header.index(name) for name in FFA_CANDIDATE_CSV_COLUMNS]
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=col, dtype=_CANDIDATE_DTYPE, ndmin=1)
    return (
        data["channels"],
        data["radiofreqs"],
        data["phase_bins"],
        data["boxcar_widths"],
        data["periods"],
        data["snrs"],
        data["flags"],
    )
