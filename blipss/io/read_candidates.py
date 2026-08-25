"""Read utilities for FFA candidate detection CSV files."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

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
    df = pd.read_csv(csv_path)
    missing = [name for name in FFA_CANDIDATE_CSV_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in {csv_path}: {missing}")
    channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, flags = (
        df[name].to_numpy(dtype=dtype)
        for name, (_, dtype) in zip(FFA_CANDIDATE_CSV_COLUMNS, _CANDIDATE_DTYPE, strict=True)
    )
    return channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, flags
