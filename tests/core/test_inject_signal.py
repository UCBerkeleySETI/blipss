"""Unit tests for blipss.core.inject_signal — data extraction, bandpass statistics, and packing."""

from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from blipss.core.inject_signal import (
    compute_median_bandpass,
    compute_per_channel_std,
    extract_data_array,
    pack_data_into_waterfall,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_waterfall(
    n_channels: int = 4,
    n_samples: int = 8,
    n_ifs: int = 1,
    tsamp: float = 0.001,
    rng: np.random.Generator | None = None,
) -> MagicMock:
    """Return a minimal Waterfall mock populated with random data."""
    if rng is None:
        rng = np.random.default_rng(0)
    wat = MagicMock()
    wat.n_ints_in_file = n_samples
    wat.header = {"tsamp": tsamp, "nifs": n_ifs, "nchans": n_channels}
    wat.data = rng.standard_normal((n_samples, n_ifs, n_channels))
    return wat


# ---------------------------------------------------------------------------
# extract_data_array
# ---------------------------------------------------------------------------


def test_extract_data_array_output_shape() -> None:
    """extract_data_array returns a (n_channels, n_samples) array."""
    n_channels, n_samples = 4, 8
    wat = _make_waterfall(n_channels=n_channels, n_samples=n_samples)
    data, _, _ = extract_data_array(wat)
    assert data.shape == (n_channels, n_samples)


def test_extract_data_array_n_samples_and_tsamp() -> None:
    """extract_data_array returns n_samples and tsamp from the Waterfall header and attribute."""
    wat = _make_waterfall(n_samples=12, tsamp=0.004)
    _, n_samples, tsamp = extract_data_array(wat)
    assert n_samples == 12
    assert tsamp == pytest.approx(0.004)


def test_extract_data_array_default_if_channel_zero() -> None:
    """extract_data_array selects if_channel=0 by default and transposes the result."""
    n_channels, n_samples, n_ifs = 3, 5, 2
    wat = _make_waterfall(n_channels=n_channels, n_samples=n_samples, n_ifs=n_ifs)
    data, _, _ = extract_data_array(wat)
    expected: npt.NDArray[np.floating] = wat.data[:, 0, :].T
    np.testing.assert_array_equal(data, expected)


def test_extract_data_array_non_default_if_channel() -> None:
    """extract_data_array selects the requested if_channel index."""
    n_channels, n_samples, n_ifs = 3, 5, 2
    wat = _make_waterfall(
        n_channels=n_channels,
        n_samples=n_samples,
        n_ifs=n_ifs,
        rng=np.random.default_rng(7),
    )
    data, _, _ = extract_data_array(wat, if_channel=1)
    expected: npt.NDArray[np.floating] = wat.data[:, 1, :].T
    np.testing.assert_array_equal(data, expected)


# ---------------------------------------------------------------------------
# compute_median_bandpass
# ---------------------------------------------------------------------------


def test_compute_median_bandpass_shape() -> None:
    """compute_median_bandpass returns a 1-D array of length n_channels."""
    data = np.ones((6, 10))
    result = compute_median_bandpass(data)
    assert result.shape == (6,)


def test_compute_median_bandpass_known_values() -> None:
    """compute_median_bandpass returns the per-channel median across time samples."""
    data = np.array(
        [
            [1.0, 3.0, 5.0],  # channel 0: median = 3.0
            [2.0, 2.0, 2.0],  # channel 1: median = 2.0
        ]
    )
    result = compute_median_bandpass(data)
    np.testing.assert_allclose(result, [3.0, 2.0])


# ---------------------------------------------------------------------------
# compute_per_channel_std
# ---------------------------------------------------------------------------


def test_compute_per_channel_std_shape() -> None:
    """compute_per_channel_std returns a 1-D array of length n_channels."""
    data = np.ones((5, 12))
    result = compute_per_channel_std(data)
    assert result.shape == (5,)


@pytest.mark.parametrize(
    ("values", "expected_std"),
    [
        ([[7.0, 7.0, 7.0, 7.0], [7.0, 7.0, 7.0, 7.0]], [0.0, 0.0]),
        ([[1.0, 3.0], [0.0, 0.0]], [1.0, 0.0]),
    ],
    ids=["constant-channels", "mixed-channels"],
)
def test_compute_per_channel_std_known_values(
    values: list[list[float]],
    expected_std: list[float],
) -> None:
    """compute_per_channel_std returns the correct population std for each channel."""
    data = np.array(values)
    result = compute_per_channel_std(data)
    np.testing.assert_allclose(result, expected_std)


# ---------------------------------------------------------------------------
# pack_data_into_waterfall
# ---------------------------------------------------------------------------


def test_pack_data_into_waterfall_returns_same_waterfall_object() -> None:
    """pack_data_into_waterfall returns the exact Waterfall object that was passed in."""
    wat = _make_waterfall(n_channels=4, n_samples=8)
    data: npt.NDArray[np.float64] = np.zeros((4, 8))
    result = pack_data_into_waterfall(data, wat, n_samples=8)
    assert result is wat


def test_pack_data_into_waterfall_output_shape() -> None:
    """pack_data_into_waterfall stores data as (n_samples, 1, n_channels) in sigproc layout."""
    n_channels, n_samples = 4, 8
    wat = _make_waterfall(n_channels=n_channels, n_samples=n_samples, n_ifs=1)
    data: npt.NDArray[np.float64] = np.zeros((n_channels, n_samples))
    pack_data_into_waterfall(data, wat, n_samples=n_samples)
    assert wat.data.shape == (n_samples, 1, n_channels)


def test_pack_data_into_waterfall_preserves_values() -> None:
    """pack_data_into_waterfall correctly inverts the (n_channels, n_samples) transpose."""
    n_channels, n_samples = 3, 5
    rng = np.random.default_rng(42)
    data: npt.NDArray[np.float64] = rng.standard_normal((n_channels, n_samples))
    wat = _make_waterfall(n_channels=n_channels, n_samples=n_samples, n_ifs=1)
    pack_data_into_waterfall(data, wat, n_samples=n_samples)
    # wat.data[:, 0, :] has shape (n_samples, n_channels); data.T is the same layout.
    np.testing.assert_array_equal(wat.data[:, 0, :], data.T)
