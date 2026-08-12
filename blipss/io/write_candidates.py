"""Write utilities for FFA candidate detection output files."""

import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt

from blipss.constants import FFA_CANDIDATE_CSV_COLUMNS


def write_candidates_csv(
    output_path: Path,
    freqs_MHz: npt.NDArray[np.floating],
    cand_channels: npt.NDArray[np.intp],
    cand_periods: npt.NDArray[np.floating],
    cand_snrs: npt.NDArray[np.floating],
    cand_phase_bins: npt.NDArray[np.uint],
    cand_boxcar_widths: npt.NDArray[np.uint],
    cand_flags: npt.NDArray[np.str_],
    start_ch: int,
) -> None:
    """
    Write FFA candidate detections to a CSV file sorted by descending S/N.

    Rows are ordered from highest to lowest S/N. Column layout follows
    ``FFA_CANDIDATE_CSV_COLUMNS``.

    Args:
        output_path: Destination path for the output CSV file.
        freqs_MHz: Radio frequencies in MHz for all channels in the processed band.
        cand_channels: Absolute channel indices of each candidate detection.
        cand_periods: Best-fit periods in seconds for each candidate.
        cand_snrs: Peak signal-to-noise ratios for each candidate.
        cand_phase_bins: Number of phase bins in the folded profile for each candidate.
        cand_boxcar_widths: Best-fit boxcar widths in phase bins for each candidate.
        cand_flags: Harmonic classification label for each candidate.
        start_ch: Absolute channel index of the first channel in the processed sub-band,
            used to map ``cand_channels`` to entries in ``freqs_MHz``.
    """
    cand_radiofreqs = freqs_MHz[cand_channels - start_ch]
    sort_desc_idx = np.argsort(cand_snrs)[::-1]
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FFA_CANDIDATE_CSV_COLUMNS)
        for row in zip(
            cand_channels[sort_desc_idx],
            cand_radiofreqs[sort_desc_idx],
            cand_phase_bins[sort_desc_idx],
            cand_boxcar_widths[sort_desc_idx],
            cand_periods[sort_desc_idx],
            cand_snrs[sort_desc_idx],
            cand_flags[sort_desc_idx],
            strict=False,
        ):
            writer.writerow(row)
