"""Atomic data-generation and signal-injection routines for synthetic filterbank simulation"""

import numpy as np
import numpy.typing as npt


def generate_white_noise_background(
    n_channels: int,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """
    Generate a 2-D array of Gaussian white noise with shape (n_channels, n_samples).

    Args:
        n_channels: Number of spectral channels.
        n_samples: Number of time samples.
        rng: Random number generator. Defaults to a fresh unseeded Generator.

    Returns:
        Array of shape (n_channels, n_samples) drawn from N(0, 1).
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_normal((n_channels, n_samples))


def inject_periodic_signal(
    data: npt.NDArray[np.floating],
    sample_times: npt.NDArray[np.floating],
    channel: int,
    period: float,
    duty_cycle: float,
    pulse_snr: float,
    initial_phase: float,
) -> None:
    """
    Add a boxcar pulse train in-place to a single channel of the data array.

    Args:
        data: Array of shape (n_channels, n_samples) to modify in place.
        sample_times: 1-D array of sample timestamps (s).
        channel: Index of the spectral channel to inject the signal into.
        period: Pulse repetition period (s).
        duty_cycle: Fraction of the period during which the pulse is on; in (0, 1].
        pulse_snr: Peak signal-to-noise ratio added to on-pulse samples.
        initial_phase: Phase offset of the pulse centre (fraction of a period); in [0, 1).
    """
    pulse_phase = sample_times / period % 1.0
    on_pulse_mask = (pulse_phase >= initial_phase - 0.5 * duty_cycle) & (pulse_phase < initial_phase + 0.5 * duty_cycle)
    data[channel, on_pulse_mask] += pulse_snr


def reshape_for_sigproc(data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Transpose and add an IF axis to match the sigproc (n_samples, n_ifs, n_channels) layout.

    Args:
        data: Array of shape (n_channels, n_samples).

    Returns:
        Array of shape (n_samples, 1, n_channels).
    """
    return np.expand_dims(data.T, axis=1)
