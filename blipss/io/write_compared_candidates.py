"""Write utilities for cross-file candidate comparison output files."""

import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt

from blipss.constants import COMPARE_CANDS_CSV_COLUMNS


def write_compared_candidates_csv(
    output_path: Path,
    channels: npt.NDArray[np.intp],
    radiofreqs: npt.NDArray[np.floating],
    phase_bins: npt.NDArray[np.uint],
    boxcar_widths: npt.NDArray[np.uint],
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    codes: npt.NDArray[np.str_],
) -> None:
    """
    Write cross-file candidate comparison results to a CSV file.

    Rows are written in the order given. Column layout follows ``COMPARE_CANDS_CSV_COLUMNS``.

    Args:
        output_path: Destination path for the output CSV file.
        channels: Spectral channel index of each candidate.
        radiofreqs: Radio frequency (MHz) of each candidate.
        phase_bins: Number of phase bins in the folded profile for each candidate.
        boxcar_widths: Best-fit boxcar widths in phase bins for each candidate.
        periods: Best-fit periods in seconds for each candidate.
        snrs: Peak signal-to-noise ratios for each candidate.
        codes: Per-file binary detection code string for each candidate.
    """
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COMPARE_CANDS_CSV_COLUMNS)
        for row in zip(channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, codes, strict=False):
            writer.writerow(row)
