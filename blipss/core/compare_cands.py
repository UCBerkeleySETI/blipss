"""
Cross-file comparison and clustering of FFA candidate periods.

Pipeline
--------
1. Filter: For each input candidate file, retain only fundamental-flagged candidates
   whose S/N exceeds a pointing-specific threshold.
2. Merge: Concatenate filtered candidates across all files, tagged with a source file index.
3. Group: For each spectral channel, cluster candidate periods via Friends-of-Friends and
   keep the highest-S/N representative per cluster.
4. Encode: Build an N-file binary detection code for each surviving cluster, where the
   i-th digit is '1' if file i contributed a candidate to that cluster and '0' otherwise.

Results from all channels are merged and returned by `group_candidates_by_channel`.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import numpy.typing as npt
from riptide.clustering import cluster1d
from tqdm import tqdm

from blipss.constants import FUNDAMENTAL_FLAG


def filter_fundamental_candidates(
    channels: npt.NDArray[np.intp],
    radiofreqs: npt.NDArray[np.floating],
    phase_bins: npt.NDArray[np.uint],
    boxcar_widths: npt.NDArray[np.uint],
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    flags: npt.NDArray[np.str_],
    snr_threshold: float,
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.floating],
    npt.NDArray[np.uint],
    npt.NDArray[np.uint],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """
    Retain fundamental-flagged candidates whose S/N exceeds a detection threshold.

    Args:
        channels: Spectral channel index of each candidate.
        radiofreqs: Radio frequency (MHz) of each candidate.
        phase_bins: Number of phase bins in the folded profile for each candidate.
        boxcar_widths: Best-fit boxcar widths in phase bins for each candidate.
        periods: Best-fit periods in seconds for each candidate.
        snrs: Peak signal-to-noise ratios for each candidate.
        flags: Harmonic classification label for each candidate.
        snr_threshold: Minimum S/N for a candidate to be retained.

    Returns:
        Tuple of (channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs) for
        candidates flagged as fundamental with S/N at or above ``snr_threshold``.
    """
    mask = (snrs >= snr_threshold) & (flags == FUNDAMENTAL_FLAG)
    return channels[mask], radiofreqs[mask], phase_bins[mask], boxcar_widths[mask], periods[mask], snrs[mask]


def _detection_code(file_indices: npt.NDArray[np.intp], n_files: int) -> str:
    """
    Build an N-file binary detection code from the file indices contributing to one cluster.

    Args:
        file_indices: Source file index of each candidate in the cluster.
        n_files: Total number of input files being compared.

    Returns:
        String of length ``n_files`` with '1' at each position whose file contributed a
        candidate to the cluster, and '0' elsewhere.
    """
    code = np.array(["0"] * n_files)
    code[np.unique(file_indices)] = "1"
    return "".join(code)


def _best_candidate_per_channel_cluster(
    file_index: npt.NDArray[np.intp],
    radiofreqs: npt.NDArray[np.floating],
    phase_bins: npt.NDArray[np.uint],
    boxcar_widths: npt.NDArray[np.uint],
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    n_files: int,
    cluster_radius: float,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.uint],
    npt.NDArray[np.uint],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.str_],
]:
    """
    Cluster one channel's candidates by period and pick the highest-S/N candidate per cluster.

    Args:
        file_index: Source file index of each candidate.
        radiofreqs: Radio frequency (MHz) of each candidate.
        phase_bins: Number of phase bins in the folded profile for each candidate.
        boxcar_widths: Best-fit boxcar widths in phase bins for each candidate.
        periods: Best-fit periods in seconds for each candidate.
        snrs: Peak signal-to-noise ratios for each candidate.
        n_files: Total number of input files being compared.
        cluster_radius: Friends-of-Friends clustering radius (s) applied to periods.

    Returns:
        Tuple of (radiofreqs, phase_bins, boxcar_widths, periods, snrs, codes) for one
        best-S/N candidate per period cluster.
    """
    sort_idx = np.argsort(periods)
    file_index = file_index[sort_idx]
    radiofreqs = radiofreqs[sort_idx]
    phase_bins = phase_bins[sort_idx]
    boxcar_widths = boxcar_widths[sort_idx]
    periods = periods[sort_idx]
    snrs = snrs[sort_idx]

    cluster_indices = cluster1d(periods, cluster_radius, already_sorted=True)
    n_clusters = len(cluster_indices)

    best_radiofreqs = np.empty(n_clusters)
    best_phase_bins = np.empty(n_clusters, dtype=np.uint)
    best_boxcar_widths = np.empty(n_clusters, dtype=np.uint)
    best_periods = np.empty(n_clusters)
    best_snrs = np.empty(n_clusters)
    codes = np.empty(n_clusters, dtype=object)

    for n, indices in enumerate(cluster_indices):
        best_idx = indices[np.argmax(snrs[indices])]
        best_radiofreqs[n] = radiofreqs[best_idx]
        best_phase_bins[n] = phase_bins[best_idx]
        best_boxcar_widths[n] = boxcar_widths[best_idx]
        best_periods[n] = periods[best_idx]
        best_snrs[n] = snrs[best_idx]
        codes[n] = _detection_code(file_index[indices], n_files)

    return best_radiofreqs, best_phase_bins, best_boxcar_widths, best_periods, best_snrs, codes.astype(np.str_)


_ChunkResult = tuple[list[int], list[float], list[int], list[int], list[float], list[float], list[str]]


def _process_channel_range(
    chunk_channels: npt.NDArray[np.intp],
    chunk_bounds: npt.NDArray[np.intp],
    file_index: npt.NDArray[np.intp],
    radiofreqs: npt.NDArray[np.floating],
    phase_bins: npt.NDArray[np.uint],
    boxcar_widths: npt.NDArray[np.uint],
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    n_files: int,
    cluster_radius: float,
    show_progress: bool = False,
) -> _ChunkResult:
    """
    Cluster every channel in one contiguous range of channel-sorted candidates.

    Args:
        chunk_channels: Unique channel values covered by this chunk, in ascending order.
        chunk_bounds: Row offsets into the per-array arguments delimiting each channel's
            candidates; ``chunk_bounds[i]:chunk_bounds[i + 1]`` selects channel ``chunk_channels[i]``.
        file_index: Source file index of each candidate in this chunk (channel-sorted).
        radiofreqs: Radio frequency (MHz) of each candidate in this chunk (channel-sorted).
        phase_bins: Number of phase bins in the folded profile for each candidate in this chunk.
        boxcar_widths: Best-fit boxcar widths in phase bins for each candidate in this chunk.
        periods: Best-fit periods in seconds for each candidate in this chunk (channel-sorted).
        snrs: Peak signal-to-noise ratios for each candidate in this chunk (channel-sorted).
        n_files: Total number of input files being compared.
        cluster_radius: Friends-of-Friends clustering radius (s) applied to periods.
        show_progress: Whether to display a per-channel tqdm progress bar.

    Returns:
        Tuple of (channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, codes) lists,
        one entry per surviving period cluster across all channels in this chunk.
    """
    out_channels: list[int] = []
    out_radiofreqs: list[float] = []
    out_phase_bins: list[int] = []
    out_boxcar_widths: list[int] = []
    out_periods: list[float] = []
    out_snrs: list[float] = []
    out_codes: list[str] = []

    channel_iter = tqdm(chunk_channels) if show_progress else chunk_channels
    for i, ch in enumerate(channel_iter):
        start, end = chunk_bounds[i], chunk_bounds[i + 1]
        ch_radiofreqs, ch_phase_bins, ch_boxcar_widths, ch_periods, ch_snrs, ch_codes = (
            _best_candidate_per_channel_cluster(
                file_index[start:end],
                radiofreqs[start:end],
                phase_bins[start:end],
                boxcar_widths[start:end],
                periods[start:end],
                snrs[start:end],
                n_files,
                cluster_radius,
            )
        )
        n = len(ch_periods)
        out_channels.extend([int(ch)] * n)
        out_radiofreqs.extend(ch_radiofreqs.tolist())
        out_phase_bins.extend(ch_phase_bins.tolist())
        out_boxcar_widths.extend(ch_boxcar_widths.tolist())
        out_periods.extend(ch_periods.tolist())
        out_snrs.extend(ch_snrs.tolist())
        out_codes.extend(ch_codes.tolist())

    return out_channels, out_radiofreqs, out_phase_bins, out_boxcar_widths, out_periods, out_snrs, out_codes


def group_candidates_by_channel(
    file_index: npt.NDArray[np.intp],
    channels: npt.NDArray[np.intp],
    radiofreqs: npt.NDArray[np.floating],
    phase_bins: npt.NDArray[np.uint],
    boxcar_widths: npt.NDArray[np.uint],
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    n_files: int,
    cluster_radius: float,
    n_jobs: int = 1,
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
    Cluster candidate periods within each spectral channel and assign a detection code to each.

    Candidates are sorted by channel once up front, so each channel's rows are a contiguous slice
    rather than being re-selected with a fresh ``channels == ch`` scan of the full array per channel.

    Args:
        file_index: Source file index of each merged candidate.
        channels: Spectral channel index of each merged candidate.
        radiofreqs: Radio frequency (MHz) of each merged candidate.
        phase_bins: Number of phase bins in the folded profile for each merged candidate.
        boxcar_widths: Best-fit boxcar widths in phase bins for each merged candidate.
        periods: Best-fit periods in seconds for each merged candidate.
        snrs: Peak signal-to-noise ratios for each merged candidate.
        n_files: Total number of input files being compared.
        cluster_radius: Friends-of-Friends clustering radius (s) applied to periods.
        n_jobs: Number of worker processes for channel clustering. 1 (default) runs sequentially
            in-process; -1 uses all available CPU cores; any other positive value is used as-is.

    Returns:
        Tuple of (channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, codes)
        merged across all channels, one entry per surviving period cluster.
    """
    order = np.argsort(channels, kind="stable")
    s_file_index = file_index[order]
    s_radiofreqs = radiofreqs[order]
    s_phase_bins = phase_bins[order]
    s_boxcar_widths = boxcar_widths[order]
    s_periods = periods[order]
    s_snrs = snrs[order]
    s_channels = channels[order]

    unique_channels, starts = np.unique(s_channels, return_index=True)
    bounds = np.append(starts, len(s_channels))

    n_jobs = (os.cpu_count() or 1) if n_jobs == -1 else max(n_jobs, 1)

    if n_jobs == 1:
        chunk_results = [
            _process_channel_range(
                unique_channels,
                bounds,
                s_file_index,
                s_radiofreqs,
                s_phase_bins,
                s_boxcar_widths,
                s_periods,
                s_snrs,
                n_files,
                cluster_radius,
                show_progress=True,
            )
        ]
    else:
        chunk_indices = [c for c in np.array_split(np.arange(len(unique_channels)), n_jobs) if len(c)]
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _process_channel_range,
                    unique_channels[idx],
                    bounds[idx[0] : idx[-1] + 2] - bounds[idx[0]],
                    s_file_index[bounds[idx[0]] : bounds[idx[-1] + 1]],
                    s_radiofreqs[bounds[idx[0]] : bounds[idx[-1] + 1]],
                    s_phase_bins[bounds[idx[0]] : bounds[idx[-1] + 1]],
                    s_boxcar_widths[bounds[idx[0]] : bounds[idx[-1] + 1]],
                    s_periods[bounds[idx[0]] : bounds[idx[-1] + 1]],
                    s_snrs[bounds[idx[0]] : bounds[idx[-1] + 1]],
                    n_files,
                    cluster_radius,
                )
                for idx in chunk_indices
            ]
            chunk_results = [f.result() for f in tqdm(futures)]

    out_channels: list[int] = []
    out_radiofreqs: list[float] = []
    out_phase_bins: list[int] = []
    out_boxcar_widths: list[int] = []
    out_periods: list[float] = []
    out_snrs: list[float] = []
    out_codes: list[str] = []
    for (
        res_channels,
        res_radiofreqs,
        res_phase_bins,
        res_boxcar_widths,
        res_periods,
        res_snrs,
        res_codes,
    ) in chunk_results:
        out_channels.extend(res_channels)
        out_radiofreqs.extend(res_radiofreqs)
        out_phase_bins.extend(res_phase_bins)
        out_boxcar_widths.extend(res_boxcar_widths)
        out_periods.extend(res_periods)
        out_snrs.extend(res_snrs)
        out_codes.extend(res_codes)

    return (
        np.array(out_channels, dtype=np.intp),
        np.array(out_radiofreqs, dtype=np.float64),
        np.array(out_phase_bins, dtype=np.uint),
        np.array(out_boxcar_widths, dtype=np.uint),
        np.array(out_periods, dtype=np.float64),
        np.array(out_snrs, dtype=np.float64),
        np.array(out_codes, dtype=np.str_),
    )
