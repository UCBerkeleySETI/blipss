"""
Atomic data-manipulation routines for the phase-resolved dynamic spectrum pipeline.

These functions implement the internal ``(n_channels, n_samples)`` array representation:

1. ``extract_waterfall_metadata``: Pull start MJD, tsamp, and frequency axis from a Waterfall header.
2. ``align_band_orientation``: Flip data and frequency axis so frequencies increase monotonically.
3. ``clip_channels``: Select a contiguous channel sub-band by index.
4. ``fold_all_channels``: Fold each channel time series in parallel with riptide and return the 2-D phase-resolved DS.
"""

import os
from functools import partial

import numpy as np
import numpy.typing as npt
from blimpy import Waterfall
from riptide import TimeSeries
from tqdm.contrib.concurrent import process_map


def extract_waterfall_metadata(
    wat: Waterfall,
) -> tuple[npt.NDArray[np.floating], float, float]:
    """
    Extract the radio frequency axis, observation start MJD, and sampling interval from a Waterfall header.

    Args:
        wat: Blimpy Waterfall object containing the loaded filterbank data.

    Returns:
        Tuple of
        - freqs_MHz: Radio frequency axis in MHz,
        - start_mjd: Observation start as MJD (UTC),
        - tsamp: sampling interval in seconds.
    """
    # Standard filterbank convention: fch1 is the centre frequency of the first channel; foff is the channel spacing.
    freqs_MHz: npt.NDArray[np.floating] = wat.header["fch1"] + np.arange(wat.header["nchans"]) * wat.header["foff"]
    start_mjd: float = wat.header["tstart"]
    tsamp: float = wat.header["tsamp"]
    return freqs_MHz, start_mjd, tsamp


def align_band_orientation(
    data: npt.NDArray[np.floating],
    freqs_MHz: npt.NDArray[np.floating],
    foff: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Flip data and frequency arrays so that channel index 0 corresponds to the lowest frequency.

    When foff is negative the Waterfall stores channels in descending frequency order.
    This function normalises to ascending order so that downstream code can treat
    index 0 as the lowest frequency unconditionally.

    Args:
        data: 2-D data array of shape (n_channels, n_samples).
        freqs_MHz: 1-D frequency array of shape (n_channels,).
        foff: Channel bandwidth in MHz; negative indicates descending frequency order.

    Returns:
        Tuple of (data, freqs_MHz) flipped along the channel axis when foff < 0, otherwise unchanged.
    """
    if foff < 0:
        return np.flip(data, axis=0), np.flip(freqs_MHz)
    return data, freqs_MHz


def clip_channels(
    data: npt.NDArray[np.floating],
    freqs_MHz: npt.NDArray[np.floating],
    start_ch: int,
    stop_ch: int | None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Restrict data and frequency arrays to a contiguous range of channel indices.

    Args:
        data: 2-D data array of shape (n_channels, n_samples).
        freqs_MHz: 1-D frequency array of shape (n_channels,).
        start_ch: First channel index to include (inclusive).
        stop_ch: Last channel index to exclude (exclusive); None retains all remaining channels.

    Returns:
        Tuple of (data[start_ch:stop_ch], freqs_MHz[start_ch:stop_ch]).

    Raises:
        ValueError: If start_ch is outside [0, n_channels) or stop_ch does not satisfy start_ch < stop_ch <= n_channels.
    """
    n_channels = len(data)
    if start_ch < 0 or start_ch >= n_channels:
        raise ValueError(f"start_ch {start_ch} is out of range [0, {n_channels})")
    if stop_ch is not None and (stop_ch <= start_ch or stop_ch > n_channels):
        raise ValueError(f"stop_ch {stop_ch} must satisfy {start_ch} < stop_ch <= {n_channels}")
    return data[start_ch:stop_ch], freqs_MHz[start_ch:stop_ch]


def _fold_single_channel(
    channel_data: npt.NDArray[np.floating],
    tsamp: float,
    period: float,
    bins: int,
    do_deredden: bool,
    rmed_width: float,
) -> npt.NDArray[np.floating]:
    """
    Fold one channel's time series at a given period into a phase profile.

    Args:
        channel_data: 1-D time series for a single spectral channel.
        tsamp: Sampling interval in seconds.
        period: Folding period in seconds.
        bins: Number of phase bins in the folded profile.
        do_deredden: Apply running-median detrending before folding when True.
        rmed_width: Running median window width in seconds; used only when do_deredden is True.

    Returns:
        1-D folded phase profile of length bins.
    """
    ts = TimeSeries.from_numpy_array(channel_data, tsamp=tsamp)
    if do_deredden:
        ts = ts.deredden(rmed_width)
    ts = ts.normalise()
    return ts.fold(period, bins, subints=1)


def fold_all_channels(
    data: npt.NDArray[np.floating],
    tsamp: float,
    period: float,
    bins: int,
    do_deredden: bool,
    rmed_width: float,
    n_workers: int | None = None,
) -> npt.NDArray[np.floating]:
    """
    Fold each spectral channel's time series in parallel to produce a phase-resolved dynamic spectrum.

    Ensures a C-contiguous memory layout before dispatching channel rows to worker processes.
    Each channel is optionally detrended with a running-median filter, normalised to
    zero median and unit standard deviation, then folded using riptide.

    Args:
        data: 2-D data array of shape (n_channels, n_samples).
        tsamp: Sampling interval in seconds.
        period: Folding period in seconds.
        bins: Number of phase bins in the folded profile.
        do_deredden: Apply running-median detrending before folding when True.
        rmed_width: Running median window width in seconds; used only when do_deredden is True.
        n_workers: Number of parallel worker processes; None uses all available CPUs.

    Returns:
        2-D phase-resolved dynamic spectrum of shape (n_channels, bins).
    """
    # np.flip returns negatively-strided views; contiguous layout avoids a per-worker copy on each row access.
    data = np.ascontiguousarray(data)
    n_channels = len(data)
    fold_fn = partial(
        _fold_single_channel,
        tsamp=tsamp,
        period=period,
        bins=bins,
        do_deredden=do_deredden,
        rmed_width=rmed_width,
    )
    # Batching amortizes IPC overhead; chunksize=1 would mean one round-trip per channel.
    chunksize = max(1, n_channels // ((n_workers or os.cpu_count() or 1) * 4))
    profiles = process_map(
        fold_fn,
        data,
        max_workers=n_workers,
        chunksize=chunksize,
        total=n_channels,
    )
    return np.array(profiles, dtype=np.float32)
