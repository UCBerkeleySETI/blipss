"""
Period search across spectral channels using the Fast Folding Algorithm (FFA).

Pipeline
--------
For each spectral channel the following steps are applied in sequence:

1. FFA search: Dold the time series at trial periods and compute matched-filter S/N.
2. Threshold: Retain only periods whose peak S/N exceeds a detection threshold.
3. Cluster: Group nearby periods via Friends-of-Friends and keep the highest-S/N
   representative per cluster.
4. Label harmonics: Classify each surviving candidate as a fundamental, harmonic,
   or sub-harmonic.

Results from all channels are merged and returned by `search_all_channels`.

"""

import os
from functools import partial

import numpy as np
import numpy.typing as npt
from riptide import Periodogram, TimeSeries, ffa_search
from riptide.clustering import cluster1d
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ..constants import CANDIDATE_DECIMAL_PRECISION
from .harmonic_detection import label_harmonics


def _run_ffa_on_channel(
    channel_data: npt.NDArray[np.floating],
    sampling_time_in_seconds: float,
    minimum_period_in_seconds: float,
    maximum_period_in_seconds: float,
    minimum_fold_periods: int,
    minimum_bins: int,
    maximum_bins: int,
    max_duty_cycle: float,
    do_deredden: bool,
    running_median_width_in_seconds: float,
) -> Periodogram:
    """
    Run an FFA search on a single-channel time series.

    Args:
        channel_data: 1D array of flux density samples for one spectral channel
        sampling_time_in_seconds: Sampling time (s)
        minimum_period_in_seconds: Minimum trial period (s)
        maximum_period_in_seconds: Maximum trial period (s)
        minimum_fold_periods: Minimum number of signal periods that must fit in the data duration
        minimum_bins: Minimum number of phase bins across the full [0, 1] phase range;
            a folded profile may cover only a fraction of this range
        maximum_bins: Maximum number of phase bins across the full [0, 1] phase range;
            a folded profile may cover only a fraction of this range
        max_duty_cycle: Maximum duty cycle searched
        do_deredden: Whether to detrend the time series with a running median filter
        running_median_width_in_seconds: Running median window width (s)

    Returns:
        Riptide Periodogram for the channel
    """
    ts = TimeSeries.from_numpy_array(channel_data, tsamp=sampling_time_in_seconds)
    _, periodogram = ffa_search(
        ts,
        period_min=minimum_period_in_seconds,
        period_max=maximum_period_in_seconds,
        fpmin=minimum_fold_periods,
        bins_min=minimum_bins,
        bins_max=maximum_bins,
        ducy_max=max_duty_cycle,  # riptide abbreviates "duty cycle" as "ducy"
        deredden=do_deredden,
        rmed_width=running_median_width_in_seconds,
        already_normalised=False,
    )
    return periodogram


def _extract_candidates_above_threshold(
    periodogram: Periodogram,
    snr_threshold: float,
) -> (
    tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.uint],
        npt.NDArray[np.uint],
    ]
    | None
):
    """
    Extract period candidates whose peak S/N exceeds a detection threshold.

    Args:
        periodogram: Riptide Periodogram from an FFA search
        snr_threshold: Minimum S/N for a candidate to be retained

    Returns:
        Tuple of (periods, snrs, phase_bins, boxcar_matched_filter_widths_in_bins) for
        candidates above the threshold, or None if no candidate meets the threshold.
        - periods: Trial periods (s) at which the peak S/N across all boxcar widths
          exceeded snr_threshold
        - snrs: Peak S/N for each candidate, maximised over all trial boxcar widths
        - phase_bins: Number of phase bins across the full [0, 1] phase range used
          to generate the folded profile for each candidate
        - boxcar_matched_filter_widths_in_bins: Width (in phase bins) of the boxcar
          matched filter that produced the peak S/N for each candidate; the implied
          duty cycle is boxcar_matched_filter_width_in_bins / phase_bins
    """
    # periodogram.snrs.shape = (No. of trial periods, No. of trial widths)
    snr_max = periodogram.snrs.max(axis=1)
    mask = snr_max >= snr_threshold
    if not np.any(mask):
        return None
    periods = np.array(periodogram.periods[mask])
    snrs = snr_max[mask]
    phase_bins = np.array(periodogram.foldbins[mask], dtype=int)
    boxcar_matched_filter_widths_in_bins = periodogram.widths[np.argmax(periodogram.snrs, axis=1)[mask]]
    return periods, snrs, phase_bins, boxcar_matched_filter_widths_in_bins


def _best_candidate_per_cluster(
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    phase_bins: npt.NDArray[np.uint],
    boxcar_matched_filter_widths_in_bins: npt.NDArray[np.uint],
    epsilon_fof: float,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.uint],
    npt.NDArray[np.uint],
]:
    """
    Group candidate periods into clusters and retain only the highest-S/N detection per cluster.

    Candidates are first sorted by period for clustering, then the output is sorted by
    descending S/N ready for harmonic identification.

    Args:
        periods: Candidate periods (s)
        snrs: S/N values corresponding to each period
        phase_bins: Number of phase bins across the full [0, 1] phase range used
          to generate the folded profile for each candidate
        boxcar_matched_filter_widths_in_bins: Width (in phase bins) of the boxcar
          matched filter that produced the peak S/N for each candidate; the implied
          duty cycle is boxcar_matched_filter_width_in_bins / phase_bins
        epsilon_fof: Period tolerance used by the Friends-of-Friends clustering algorithm

    Returns:
        Tuple of (periods, snrs, phase_bins, boxcar_matched_filter_widths_in_bins) for one
        best candidate per cluster, sorted in descending order of S/N
    """
    sort_idx = np.argsort(periods)
    periods = periods[sort_idx]
    snrs = snrs[sort_idx]
    phase_bins = phase_bins[sort_idx]
    boxcar_matched_filter_widths_in_bins = boxcar_matched_filter_widths_in_bins[sort_idx]

    cluster_indices = cluster1d(periods, epsilon_fof, already_sorted=True)

    best_idxs = np.array([indices[np.argmax(snrs[indices])] for indices in cluster_indices], dtype=int)
    best_periods_arr = periods[best_idxs]
    best_snrs_arr = snrs[best_idxs]
    best_phase_bins_arr = phase_bins[best_idxs]
    best_boxcar_widths_arr = boxcar_matched_filter_widths_in_bins[best_idxs]

    snr_sort_idx = np.argsort(best_snrs_arr)[::-1]
    return (
        best_periods_arr[snr_sort_idx],
        best_snrs_arr[snr_sort_idx],
        best_phase_bins_arr[snr_sort_idx],
        best_boxcar_widths_arr[snr_sort_idx],
    )


def _search_single_channel(
    channel_data: npt.NDArray[np.floating],
    sampling_time_in_seconds: float,
    minimum_period_in_seconds: float,
    maximum_period_in_seconds: float,
    minimum_fold_periods: int,
    minimum_bins: int,
    maximum_bins: int,
    max_duty_cycle: float,
    do_deredden: bool,
    running_median_width_in_seconds: float,
    snr_threshold: float,
    epsilon_fof: float,
    epsilon_harmonic: float,
) -> (
    tuple[
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.uint],
        npt.NDArray[np.uint],
        npt.NDArray[np.str_],
    ]
    | None
):
    """
    Execute a complete FFA search pipeline on a single spectral channel.

    Runs FFA, applies an S/N threshold, deduplicates via period clustering, and labels harmonics.

    Args:
        channel_data: 1D intensity samples for one spectral channel
        sampling_time_in_seconds: Sampling time (s)
        minimum_period_in_seconds: Minimum trial period (s)
        maximum_period_in_seconds: Maximum trial period (s)
        minimum_fold_periods: Minimum number of signal periods that must fit in the data
        minimum_bins: Minimum number of phase bins across the full [0, 1] phase range;
            a folded profile may cover only a fraction of this range
        maximum_bins: Maximum number of phase bins across the full [0, 1] phase range;
            a folded profile may cover only a fraction of this range
        max_duty_cycle: Maximum duty cycle searched
        do_deredden: Whether to detrend the time series with a running median filter
        running_median_width_in_seconds: Running median window width (s)
        snr_threshold: Minimum S/N for a candidate to be retained
        epsilon_fof: Period tolerance for Friends-of-Friends clustering
        epsilon_harmonic: Period tolerance for harmonic matching

    Returns:
        Tuple of (periods, snrs, phase_bins, boxcar_matched_filter_widths_in_bins, flags) for
        detected candidates sorted by descending S/N, or None if no candidate exceeds the threshold.
        - periods: Trial periods (s) at which the peak S/N across all boxcar widths
          exceeded snr_threshold
        - snrs: Peak S/N for each candidate, maximised over all trial boxcar widths
        - phase_bins: Number of phase bins across the full [0, 1] phase range to be
          used to generate the folded profile for each candidate
        - boxcar_matched_filter_widths_in_bins: Width (in phase bins) of the boxcar
          matched filter that produced the peak S/N for each candidate; the implied
          duty cycle is boxcar_matched_filter_width_in_bins / phase_bins
        - flags: Harmonic classification for each candidate ('F': fundamental,
          'H': harmonic, 'S': sub-harmonic)
    """
    periodogram = _run_ffa_on_channel(
        channel_data,
        sampling_time_in_seconds,
        minimum_period_in_seconds,
        maximum_period_in_seconds,
        minimum_fold_periods,
        minimum_bins,
        maximum_bins,
        max_duty_cycle,
        do_deredden,
        running_median_width_in_seconds,
    )
    candidates = _extract_candidates_above_threshold(periodogram, snr_threshold)
    if candidates is None:
        return None
    periods, snrs, phase_bins, boxcar_matched_filter_widths_in_bins = candidates
    periods, snrs, phase_bins, boxcar_matched_filter_widths_in_bins = _best_candidate_per_cluster(
        periods, snrs, phase_bins, boxcar_matched_filter_widths_in_bins, epsilon_fof
    )
    flags = label_harmonics(periods, snrs, epsilon_harmonic, presorted=True)
    return periods, snrs, phase_bins, boxcar_matched_filter_widths_in_bins, flags


def _finalize_candidate_arrays(
    channels: npt.NDArray[np.intp],
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    phase_bins: npt.NDArray[np.intp],
    boxcar_matched_filter_widths_in_bins: npt.NDArray[np.intp],
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.uint],
    npt.NDArray[np.uint],
]:
    """
    Cast accumulated candidate arrays to their final types and rounding precision.

    Args:
        channels: Channel indices (cast to int)
        periods: Trial periods (s) (rounded and cast to float64)
        snrs: S/N values (rounded and cast to float64)
        phase_bins: Number of phase bins in the folded profile (cast to int)
        boxcar_matched_filter_widths_in_bins: Best boxcar filter widths in phase bins (cast to int)

    Returns:
        Tuple of (channels, periods, snrs, phase_bins, boxcar_matched_filter_widths_in_bins)
        with finalized dtypes and precision
    """
    return (
        channels.astype(int),
        np.round(periods.astype(np.float64, copy=False), CANDIDATE_DECIMAL_PRECISION),
        np.round(snrs.astype(np.float64, copy=False), CANDIDATE_DECIMAL_PRECISION),
        phase_bins.astype(int),
        boxcar_matched_filter_widths_in_bins.astype(int),
    )


def search_all_channels(
    data: npt.NDArray[np.floating],
    start_channel: int,
    sampling_time_in_seconds: float,
    minimum_period_in_seconds: float,
    maximum_period_in_seconds: float,
    minimum_fold_periods: int,
    minimum_bins: int,
    maximum_bins: int,
    max_duty_cycle: float,
    do_deredden: bool,
    running_median_width_in_seconds: float,
    snr_threshold: float,
    epsilon_fof: float,
    epsilon_harmonic: float,
    n_workers: int | None = None,
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.uint],
    npt.NDArray[np.uint],
    npt.NDArray[np.str_],
]:
    """
    Search every spectral channel for periodic signals using the Fast Folding Algorithm.

    Runs an FFA search on each channel of data, clusters and deduplicates period candidates,
    labels harmonics, and returns merged results across all channels.

    Args:
        data: 2D array of shape (n_channels, n_samples); each row is one spectral channel
        start_channel: Global channel index of data[0]; offsets channel numbers in the output
        sampling_time_in_seconds: Sampling time (s)
        minimum_period_in_seconds: Minimum trial period (s) for the FFA search
        maximum_period_in_seconds: Maximum trial period (s) for the FFA search
        minimum_fold_periods: Minimum number of signal periods that must fit in the data
        minimum_bins: Minimum number of phase bins across the full [0, 1] phase range;
            a folded profile may cover only a fraction of this range
        maximum_bins: Maximum number of phase bins across the full [0, 1] phase range;
            a folded profile may cover only a fraction of this range
        max_duty_cycle: Maximum duty cycle searched
        do_deredden: Whether to detrend each time series with a running median filter
        running_median_width_in_seconds: Running median window width (s)
        snr_threshold: Minimum matched-filtering S/N for a detection
        epsilon_fof: Period tolerance for Friends-of-Friends clustering
        epsilon_harmonic: Period tolerance for harmonic matching
        n_workers: Number of parallel worker processes; 1 runs the serial tqdm loop,
            None or values >1 dispatch via process_map using all available CPUs (None)
            or the specified count

    Returns:
        Tuple of (cand_channels, cand_periods, cand_snrs, cand_phase_bins,
        cand_boxcar_widths, cand_flags) across all channels.
        - cand_channels: Spectral channel index of each candidate
        - cand_periods: Trial periods (s) at which the peak S/N across all boxcar widths
            exceeded snr_threshold
        - cand_snrs: Peak S/N for each candidate, maximised over all trial boxcar widths
        - cand_phase_bins: Number of phase bins across the full [0, 1] phase range to be
            used to generate the folded profile for each candidate
        - cand_boxcar_widths: Width (in phase bins) of the boxcar matched filter that
            produced the peak S/N for each candidate; the implied duty cycle is
            cand_boxcar_widths / cand_phase_bins
        - cand_flags: Harmonic classification for each candidate ('F': fundamental,
            'H': harmonic, 'S': sub-harmonic)
    """
    # np.flip returns negatively-strided views; contiguous layout avoids a per-worker copy on each row access.
    data = np.ascontiguousarray(data)
    n_channels = len(data)

    cand_channels: list[int] = []
    cand_periods: list[float] = []
    cand_snrs: list[float] = []
    cand_phase_bins: list[int] = []
    cand_boxcar_widths: list[int] = []
    cand_flags: list[str] = []

    if n_workers == 1:
        for ch_idx in tqdm(range(n_channels)):
            result = _search_single_channel(
                data[ch_idx],
                sampling_time_in_seconds,
                minimum_period_in_seconds,
                maximum_period_in_seconds,
                minimum_fold_periods,
                minimum_bins,
                maximum_bins,
                max_duty_cycle,
                do_deredden,
                running_median_width_in_seconds,
                snr_threshold,
                epsilon_fof,
                epsilon_harmonic,
            )
            if result is None:
                continue
            periods, snrs, phase_bins, boxcar_widths, flags = result
            n = len(periods)
            cand_channels.extend([start_channel + ch_idx] * n)
            cand_periods.extend(periods.tolist())
            cand_snrs.extend(snrs.tolist())
            cand_phase_bins.extend(phase_bins.tolist())
            cand_boxcar_widths.extend(boxcar_widths.tolist())
            cand_flags.extend(flags.tolist())
    else:
        search_fn = partial(
            _search_single_channel,
            sampling_time_in_seconds=sampling_time_in_seconds,
            minimum_period_in_seconds=minimum_period_in_seconds,
            maximum_period_in_seconds=maximum_period_in_seconds,
            minimum_fold_periods=minimum_fold_periods,
            minimum_bins=minimum_bins,
            maximum_bins=maximum_bins,
            max_duty_cycle=max_duty_cycle,
            do_deredden=do_deredden,
            running_median_width_in_seconds=running_median_width_in_seconds,
            snr_threshold=snr_threshold,
            epsilon_fof=epsilon_fof,
            epsilon_harmonic=epsilon_harmonic,
        )
        # Batching amortizes IPC overhead; chunksize=1 would mean one round-trip per channel.
        chunksize = max(1, n_channels // ((n_workers or os.cpu_count() or 1) * 4))
        results = process_map(search_fn, data, max_workers=n_workers, chunksize=chunksize, total=n_channels)
        for ch_idx, result in enumerate(results):
            if result is None:
                continue
            periods, snrs, phase_bins, boxcar_widths, flags = result
            n = len(periods)
            cand_channels.extend([start_channel + ch_idx] * n)
            cand_periods.extend(periods.tolist())
            cand_snrs.extend(snrs.tolist())
            cand_phase_bins.extend(phase_bins.tolist())
            cand_boxcar_widths.extend(boxcar_widths.tolist())
            cand_flags.extend(flags.tolist())

    channels_arr, periods_arr, snrs_arr, phase_bins_arr, boxcar_widths_arr = _finalize_candidate_arrays(
        np.array(cand_channels),
        np.array(cand_periods),
        np.array(cand_snrs),
        np.array(cand_phase_bins),
        np.array(cand_boxcar_widths),
    )
    return channels_arr, periods_arr, snrs_arr, phase_bins_arr, boxcar_widths_arr, np.array(cand_flags)
