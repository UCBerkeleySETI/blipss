"""Core logic for selecting and folding periodicity candidates for verification plots."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from riptide import Periodogram, TimeSeries, ffa_search


def select_candidates_by_code(
    cand_channels: npt.NDArray[np.intp],
    cand_periods: npt.NDArray[np.floating],
    cand_bins: npt.NDArray[np.uint],
    cand_codes: npt.NDArray[np.str_],
    codes_plot: Sequence[str],
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.floating],
    npt.NDArray[np.uint],
    npt.NDArray[np.str_],
]:
    """
    Retain candidates whose binary detection code is one of the codes selected for plotting.

    Args:
        cand_channels: Spectral channel index of each candidate
        cand_periods: Best-fit period (s) of each candidate
        cand_bins: Number of phase bins in the folded profile for each candidate
        cand_codes: Per-file binary detection code string for each candidate
        codes_plot: Binary codes selected for plotting

    Returns:
        Tuple of (channels, periods, bins, codes) for candidates matching codes_plot
    """
    mask = np.isin(cand_codes, codes_plot)
    return cand_channels[mask], cand_periods[mask], cand_bins[mask], cand_codes[mask]


def run_ffa_and_fold_channel(
    channel_data: npt.NDArray[np.floating],
    tsamp: float,
    min_period: float,
    max_period: float,
    fpmin: int,
    bins_min: int,
    bins_max: int,
    ducy_max: float,
    do_deredden: bool,
    rmed_width: float,
) -> tuple[TimeSeries, Periodogram]:
    """
    Run an FFA search on a single-channel time series, keeping both the detrended series and periodogram.

    Args:
        channel_data: 1D array of flux density samples for one spectral channel
        tsamp: Sampling time (s)
        min_period: Minimum trial period (s)
        max_period: Maximum trial period (s)
        fpmin: Minimum number of signal periods that must fit in the data duration
        bins_min: Minimum number of phase bins across the full [0, 1] phase range
        bins_max: Maximum number of phase bins across the full [0, 1] phase range
        ducy_max: Maximum duty cycle searched
        do_deredden: Whether to detrend the time series with a running median filter
        rmed_width: Running median window width (s)

    Returns:
        Tuple of (detrended_ts, periodogram) from the FFA search
    """
    raw_ts = TimeSeries.from_numpy_array(channel_data, tsamp=tsamp)
    detrended_ts, periodogram = ffa_search(
        raw_ts,
        period_min=min_period,
        period_max=max_period,
        fpmin=fpmin,
        bins_min=bins_min,
        bins_max=bins_max,
        ducy_max=ducy_max,  # riptide abbreviates "duty cycle" as "ducy"
        deredden=do_deredden,
        rmed_width=rmed_width,
        already_normalised=False,
    )
    return detrended_ts, periodogram
