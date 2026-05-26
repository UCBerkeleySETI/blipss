"""Unit tests for modules in blipss.core.simulate_data"""

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from blipss.core.simulate_data import (
    generate_white_noise_background,
    inject_periodic_signal,
    reshape_for_sigproc,
)

# ---------------------------------------------------------------------------
# generate_white_noise_background
# ---------------------------------------------------------------------------


def test_generate_white_noise_background_dtype() -> None:
    """generate_white_noise_background returns a float64 array."""
    result = generate_white_noise_background(3, 5)
    assert result.dtype == np.float64


def test_generate_white_noise_background_seeded_reproducible() -> None:
    """generate_white_noise_background with the same seeded rng produces identical arrays."""
    result1 = generate_white_noise_background(4, 8, rng=np.random.default_rng(42))
    result2 = generate_white_noise_background(4, 8, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(result1, result2)


def test_generate_white_noise_background_different_seeds_differ() -> None:
    """generate_white_noise_background with different seeds produces different arrays."""
    result1 = generate_white_noise_background(4, 100, rng=np.random.default_rng(0))
    result2 = generate_white_noise_background(4, 100, rng=np.random.default_rng(1))
    assert not np.array_equal(result1, result2)


@given(
    n_channels=st.integers(min_value=1, max_value=32),
    n_samples=st.integers(min_value=1, max_value=64),
)
def test_generate_white_noise_background_shape_invariant(n_channels: int, n_samples: int) -> None:
    """generate_white_noise_background output shape equals (n_channels, n_samples) for any valid inputs."""
    result = generate_white_noise_background(n_channels, n_samples)
    assert result.shape == (n_channels, n_samples)


# ---------------------------------------------------------------------------
# inject_periodic_signal
# ---------------------------------------------------------------------------


def test_inject_periodic_signal_only_target_channel_modified() -> None:
    """inject_periodic_signal leaves channels other than the target channel unchanged."""
    n_channels, n_samples = 4, 20
    data = np.zeros((n_channels, n_samples))
    times = np.linspace(0, 1, n_samples, endpoint=False)
    inject_periodic_signal(data, times, channel=2, period=1.0, duty_cycle=1.0, pulse_snr=3.0, initial_phase=0.5)
    for ch in [0, 1, 3]:
        np.testing.assert_array_equal(data[ch], np.zeros(n_samples))


def test_inject_periodic_signal_partial_duty_cycle_on_off_split() -> None:
    """inject_periodic_signal boosts only on-pulse samples; off-pulse samples remain at zero."""
    # 100-sample array treated as one period; duty_cycle=0.2, initial_phase=0.5
    # On-pulse phase window: [0.4, 0.6)
    n_samples = 100
    data = np.zeros((1, n_samples))
    times = np.arange(n_samples, dtype=float)
    period = float(n_samples)
    pulse_snr = 5.0
    inject_periodic_signal(
        data, times, channel=0, period=period, duty_cycle=0.2, pulse_snr=pulse_snr, initial_phase=0.5
    )
    pulse_phase = times / period % 1.0
    on_mask = (pulse_phase >= 0.4) & (pulse_phase < 0.6)
    np.testing.assert_allclose(data[0, on_mask], pulse_snr)
    np.testing.assert_array_equal(data[0, ~on_mask], 0.0)


def test_inject_periodic_signal_accumulates_on_existing_data() -> None:
    """inject_periodic_signal adds pulse_snr on top of pre-existing non-zero values."""
    data = np.ones((1, 10))
    times = np.linspace(0, 1, 10, endpoint=False)
    inject_periodic_signal(data, times, channel=0, period=1.0, duty_cycle=1.0, pulse_snr=4.0, initial_phase=0.5)
    np.testing.assert_allclose(data[0], 5.0)


# ---------------------------------------------------------------------------
# reshape_for_sigproc
# ---------------------------------------------------------------------------


def test_reshape_for_sigproc_preserves_values() -> None:
    """reshape_for_sigproc preserves all data values under the axis permutation."""
    rng = np.random.default_rng(0)
    data: npt.NDArray[np.float64] = rng.standard_normal((5, 12))
    result = reshape_for_sigproc(data)
    for ch in range(5):
        for t in range(12):
            assert result[t, 0, ch] == pytest.approx(data[ch, t])


@given(
    n_channels=st.integers(min_value=1, max_value=16),
    n_samples=st.integers(min_value=1, max_value=32),
)
def test_reshape_for_sigproc_shape_invariant(n_channels: int, n_samples: int) -> None:
    """reshape_for_sigproc output shape is (n_samples, 1, n_channels) for any valid input dimensions."""
    data = np.zeros((n_channels, n_samples))
    result = reshape_for_sigproc(data)
    assert result.shape == (n_samples, 1, n_channels)
