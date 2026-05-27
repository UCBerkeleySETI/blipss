"""
Atomic data-manipulation routines for the real-data signal-injection pipeline.

These functions operate on the internal ``(n_channels, n_samples)`` array
representation used throughout the inject_signal pipeline:

1. ``extract_data_array`` — load a ``Waterfall`` object into the internal layout.
2. ``compute_median_bandpass`` / ``compute_per_channel_std`` — characterise the
   per-channel noise statistics of the real data; used to calibrate injected
   pulse amplitudes relative to the local bandpass.
3. ``pack_data_into_waterfall`` — convert the modified array back to the sigproc
   ``(n_samples, n_ifs, n_channels)`` layout and store it in the ``Waterfall``
   object prior to writing.

The actual disk write is handled by ``blipss.io.write_filterbank.write_waterfall``.
Signal injection itself is handled by ``blipss.core.simulate_data.inject_periodic_signal``,
which is shared with the *simulate_data* pipeline.
"""

import numpy as np
import numpy.typing as npt
from blimpy import Waterfall


def extract_data_array(
    wat: Waterfall,
    if_channel: int = 0,
) -> tuple[npt.NDArray[np.floating], int, float]:
    """
    Extract the 2-D data array, sample count, and sampling interval from a Waterfall object.

    Converts from blimpy's on-disk ``(n_samples, n_ifs, n_channels)`` layout to the
    internal ``(n_channels, n_samples)`` representation used by the rest of the
    inject_signal pipeline.

    Args:
        wat: Blimpy ``Waterfall`` object containing the loaded filterbank data.
        if_channel: Index of the polarisation (IF) channel to extract. Defaults to 0.

    Returns:
        Tuple of:
        - data: 2-D array of shape ``(n_channels, n_samples)``
        - n_samples: Number of time samples in the file
        - tsamp: Sampling interval in seconds
    """
    n_samples: int = wat.n_ints_in_file
    tsamp: float = wat.header["tsamp"]
    # wat.data has shape (n_samples, n_ifs, n_channels)
    data: npt.NDArray[np.floating] = wat.data[:, if_channel, :].T
    return data, n_samples, tsamp


def compute_median_bandpass(
    data: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Compute the per-channel median value across all time samples.

    Used to estimate the baseline level of each spectral channel so that injected
    pulse amplitudes can be set relative to the local bandpass rather than an
    absolute scale.

    Args:
        data: Array of shape ``(n_channels, n_samples)``.

    Returns:
        1-D array of per-channel median values, shape ``(n_channels,)``.
    """
    return np.median(data, axis=1)


def compute_per_channel_std(
    data: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Compute the per-channel standard deviation across all time samples.

    Used alongside ``compute_median_bandpass`` to express the injected pulse
    amplitude in units of the local noise standard deviation (i.e., as an SNR).

    Args:
        data: Array of shape ``(n_channels, n_samples)``.

    Returns:
        1-D array of per-channel standard deviations, shape ``(n_channels,)``.
    """
    return np.std(data, axis=1)


def pack_data_into_waterfall(
    data: npt.NDArray[np.floating],
    wat: Waterfall,
    n_samples: int,
) -> Waterfall:
    """
    Reshape the modified data array back into sigproc layout and store it in the Waterfall object.

    This is the inverse of ``extract_data_array``: it converts from the internal
    ``(n_channels, n_samples)`` representation back to the ``(n_samples, n_ifs, n_channels)``
    layout expected by blimpy's serialisers. ``n_ifs`` is read from the original
    Waterfall header so that multi-polarisation files are handled correctly.

    Call this immediately before ``blipss.io.write_filterbank.write_waterfall``.

    Args:
        data: Array of shape ``(n_channels, n_samples)`` after signal injection.
        wat: Blimpy ``Waterfall`` object whose ``.data`` attribute will be overwritten.
            The header is not modified.
        n_samples: Number of time samples (as returned by ``extract_data_array``).

    Returns:
        The same ``Waterfall`` object with its ``.data`` attribute updated.
    """
    n_ifs: int = wat.header["nifs"]
    n_channels: int = wat.header["nchans"]
    wat.data = data.T.reshape((n_samples, n_ifs, n_channels))
    return wat
