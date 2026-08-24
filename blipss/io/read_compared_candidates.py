"""Read utilities for cross-file candidate comparison output files."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from blipss.constants import COMPARE_CANDS_CSV_COLUMNS


def read_compared_candidates_csv(
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
    Read cross-file candidate comparison results previously written by ``write_compared_candidates_csv``.

    Args:
        csv_path: Path to a comparison CSV file with ``COMPARE_CANDS_CSV_COLUMNS`` columns.

    Returns:
        Tuple of (channels, radiofreqs_MHz, phase_bins, boxcar_widths, periods, snrs, codes)
        arrays, one entry per candidate row in file order.

    Raises:
        ValueError: When the file is missing one of ``COMPARE_CANDS_CSV_COLUMNS``.
    """
    df = pd.read_csv(csv_path, usecols=COMPARE_CANDS_CSV_COLUMNS, dtype={"Code": "string"})
    return (
        df["Channel"].to_numpy(dtype=np.intp),
        df["Radio frequency (MHz)"].to_numpy(dtype=np.float64),
        df["Bins"].to_numpy(dtype=np.uint),
        df["Best width"].to_numpy(dtype=np.uint),
        df["Period (s)"].to_numpy(dtype=np.float64),
        df["S/N"].to_numpy(dtype=np.float64),
        df["Code"].to_numpy(dtype=str),
    )
