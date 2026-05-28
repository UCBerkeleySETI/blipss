"""Unit tests for blipss.core.compute_phase_resolved_ds"""

from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from blipss.core.compute_phase_resolved_ds import (
    _fold_single_channel,
    align_band_orientation,
    clip_channels,
    extract_waterfall_metadata,
    fold_all_channels,
)

_MODULE = "blipss.core.compute_phase_resolved_ds"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_waterfall(
    fch1: float = 1000.0,
    foff: float = 0.5,
    nchans: int = 4,
    tstart: float = 59000.0,
    tsamp: float = 0.001,
) -> MagicMock:
    """Return a minimal Waterfall mock with a realistic header."""
    wat = MagicMock()
    wat.header = {
        "fch1": fch1,
        "foff": foff,
        "nchans": nchans,
        "tstart": tstart,
        "tsamp": tsamp,
    }
    return wat


# ---------------------------------------------------------------------------
# extract_waterfall_metadata
# ---------------------------------------------------------------------------


def test_extract_waterfall_metadata_freqs_shape() -> None:
    """extract_waterfall_metadata returns a freqs_MHz array of length nchans."""
    nchans = 6
    wat = _make_waterfall(nchans=nchans)
    freqs_MHz, _, _ = extract_waterfall_metadata(wat)
    assert freqs_MHz.shape == (nchans,)


def test_extract_waterfall_metadata_freqs_values() -> None:
    """extract_waterfall_metadata builds freqs_MHz as fch1 + arange(nchans) * foff."""
    wat = _make_waterfall(fch1=1000.0, foff=0.5, nchans=4)
    freqs_MHz, _, _ = extract_waterfall_metadata(wat)
    expected = np.array([1000.0, 1000.5, 1001.0, 1001.5])
    np.testing.assert_allclose(freqs_MHz, expected)


def test_extract_waterfall_metadata_start_mjd_and_tsamp() -> None:
    """extract_waterfall_metadata returns tstart as start_mjd and tsamp directly from the header."""
    wat = _make_waterfall(tstart=59123.456, tsamp=0.004)
    _, start_mjd, tsamp = extract_waterfall_metadata(wat)
    assert start_mjd == pytest.approx(59123.456)
    assert tsamp == pytest.approx(0.004)


# ---------------------------------------------------------------------------
# align_band_orientation
# ---------------------------------------------------------------------------


def test_align_band_orientation_positive_foff_unchanged() -> None:
    """align_band_orientation returns data and freqs unchanged when foff > 0."""
    data = np.arange(12, dtype=float).reshape(3, 4)
    freqs = np.array([1000.0, 1000.5, 1001.0])
    out_data, out_freqs = align_band_orientation(data, freqs, foff=0.5)
    np.testing.assert_array_equal(out_data, data)
    np.testing.assert_array_equal(out_freqs, freqs)


def test_align_band_orientation_negative_foff_flips_both() -> None:
    """align_band_orientation flips data and freqs along the channel axis when foff < 0."""
    data = np.arange(12, dtype=float).reshape(3, 4)
    freqs = np.array([1001.0, 1000.5, 1000.0])
    out_data, out_freqs = align_band_orientation(data, freqs, foff=-0.5)
    np.testing.assert_array_equal(out_data, np.flip(data, axis=0))
    np.testing.assert_array_equal(out_freqs, np.flip(freqs))


def test_align_band_orientation_negative_foff_produces_ascending_freqs() -> None:
    """align_band_orientation produces a monotonically increasing freqs_MHz array when foff < 0."""
    freqs = np.array([1002.0, 1001.0, 1000.0])
    data = np.zeros((3, 4))
    _, out_freqs = align_band_orientation(data, freqs, foff=-1.0)
    assert np.all(np.diff(out_freqs) > 0)


# ---------------------------------------------------------------------------
# clip_channels
# ---------------------------------------------------------------------------


def test_clip_channels_stop_none_returns_full_tail() -> None:
    """clip_channels with stop_ch=None returns all channels from start_ch onward."""
    data = np.ones((6, 10))
    freqs = np.arange(6, dtype=float)
    out_data, out_freqs = clip_channels(data, freqs, start_ch=2, stop_ch=None)
    assert out_data.shape == (4, 10)
    np.testing.assert_array_equal(out_freqs, freqs[2:])


def test_clip_channels_explicit_stop_slices_correctly() -> None:
    """clip_channels returns data[start_ch:stop_ch] and freqs[start_ch:stop_ch]."""
    data = np.arange(20, dtype=float).reshape(5, 4)
    freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out_data, out_freqs = clip_channels(data, freqs, start_ch=1, stop_ch=4)
    np.testing.assert_array_equal(out_data, data[1:4])
    np.testing.assert_array_equal(out_freqs, freqs[1:4])


@pytest.mark.parametrize(
    ("start_ch", "stop_ch", "match"),
    [
        (-1, None, "out of range"),
        (5, None, "out of range"),
        (2, 2, "must satisfy"),
        (2, 1, "must satisfy"),
        (0, 6, "must satisfy"),
    ],
    ids=["start-negative", "start-at-nchans", "stop-equal-start", "stop-below-start", "stop-past-nchans"],
)
def test_clip_channels_invalid_indices_raise_value_error(
    start_ch: int,
    stop_ch: int | None,
    match: str,
) -> None:
    """clip_channels raises ValueError for out-of-range or inconsistent start_ch/stop_ch."""
    data = np.ones((5, 8))
    freqs = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match=match):
        clip_channels(data, freqs, start_ch=start_ch, stop_ch=stop_ch)


# ---------------------------------------------------------------------------
# fold_all_channels
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.process_map")
def test_fold_all_channels_output_shape_and_dtype(mock_process_map: MagicMock) -> None:
    """fold_all_channels returns a float32 array of shape (n_channels, bins)."""
    n_channels, bins = 3, 16
    mock_process_map.return_value = [np.zeros(bins) for _ in range(n_channels)]
    data = np.random.default_rng(0).standard_normal((n_channels, 64))
    result = fold_all_channels(data, tsamp=0.001, period=1.0, bins=bins, do_deredden=False, rmed_width=12.0)
    assert result.shape == (n_channels, bins)
    assert result.dtype == np.float32
    mock_process_map.assert_called_once()


@patch(f"{_MODULE}.process_map")
def test_fold_all_channels_passes_n_workers_to_process_map(mock_process_map: MagicMock) -> None:
    """fold_all_channels passes n_workers as max_workers to process_map."""
    n_channels, bins = 2, 8
    mock_process_map.return_value = [np.zeros(bins) for _ in range(n_channels)]
    data = np.zeros((n_channels, 32))
    fold_all_channels(data, tsamp=0.001, period=1.0, bins=bins, do_deredden=False, rmed_width=12.0, n_workers=2)
    mock_process_map.assert_called_once()
    assert mock_process_map.call_args.kwargs["max_workers"] == 2


@patch(f"{_MODULE}.process_map")
def test_fold_all_channels_dispatches_contiguous_array(mock_process_map: MagicMock) -> None:
    """fold_all_channels converts a non-contiguous view to C-contiguous before dispatching."""
    n_channels, bins = 4, 8
    mock_process_map.return_value = [np.zeros(bins) for _ in range(n_channels)]
    # np.flip produces a negatively-strided, non-contiguous view
    data = np.flip(np.arange(n_channels * 16, dtype=float).reshape(n_channels, 16), axis=0)
    assert not data.flags["C_CONTIGUOUS"]
    fold_all_channels(data, tsamp=0.001, period=1.0, bins=bins, do_deredden=False, rmed_width=12.0)
    mock_process_map.assert_called_once()
    dispatched: npt.NDArray[np.floating] = mock_process_map.call_args.args[1]
    assert dispatched.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# _fold_single_channel
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.TimeSeries")
def test_fold_single_channel_builds_timeseries_from_input(mock_ts_class: MagicMock) -> None:
    """_fold_single_channel calls TimeSeries.from_numpy_array with channel_data and tsamp."""
    channel_data = np.zeros(128)
    _fold_single_channel(channel_data, tsamp=0.001, period=1.0, bins=16, do_deredden=False, rmed_width=12.0)
    mock_ts_class.from_numpy_array.assert_called_once_with(channel_data, tsamp=0.001)


@patch(f"{_MODULE}.TimeSeries")
def test_fold_single_channel_skips_deredden_when_false(mock_ts_class: MagicMock) -> None:
    """_fold_single_channel does not call deredden when do_deredden is False."""
    channel_data = np.zeros(128)
    _fold_single_channel(channel_data, tsamp=0.001, period=1.0, bins=16, do_deredden=False, rmed_width=12.0)
    ts_instance = mock_ts_class.from_numpy_array.return_value
    ts_instance.deredden.assert_not_called()


@patch(f"{_MODULE}.TimeSeries")
def test_fold_single_channel_applies_deredden_with_rmed_width(mock_ts_class: MagicMock) -> None:
    """_fold_single_channel calls deredden(rmed_width) on the TimeSeries when do_deredden is True."""
    channel_data = np.zeros(128)
    _fold_single_channel(channel_data, tsamp=0.001, period=1.0, bins=16, do_deredden=True, rmed_width=12.0)
    ts_instance = mock_ts_class.from_numpy_array.return_value
    ts_instance.deredden.assert_called_once_with(12.0)


@patch(f"{_MODULE}.TimeSeries")
def test_fold_single_channel_normalises_before_folding_no_deredden(mock_ts_class: MagicMock) -> None:
    """_fold_single_channel calls normalise on the TimeSeries before folding when do_deredden is False."""
    channel_data = np.zeros(128)
    _fold_single_channel(channel_data, tsamp=0.001, period=1.0, bins=16, do_deredden=False, rmed_width=12.0)
    ts_instance = mock_ts_class.from_numpy_array.return_value
    ts_instance.normalise.assert_called_once()


@patch(f"{_MODULE}.TimeSeries")
def test_fold_single_channel_normalises_before_folding_with_deredden(mock_ts_class: MagicMock) -> None:
    """_fold_single_channel calls normalise on the deredden output when do_deredden is True."""
    channel_data = np.zeros(128)
    _fold_single_channel(channel_data, tsamp=0.001, period=1.0, bins=16, do_deredden=True, rmed_width=12.0)
    ts_after_deredden = mock_ts_class.from_numpy_array.return_value.deredden.return_value
    ts_after_deredden.normalise.assert_called_once()


@patch(f"{_MODULE}.TimeSeries")
def test_fold_single_channel_folds_with_correct_args_no_deredden(mock_ts_class: MagicMock) -> None:
    """_fold_single_channel calls fold(period, bins, subints=1) on the normalised TimeSeries."""
    channel_data = np.zeros(128)
    _fold_single_channel(channel_data, tsamp=0.001, period=2.5, bins=32, do_deredden=False, rmed_width=12.0)
    ts_after_normalise = mock_ts_class.from_numpy_array.return_value.normalise.return_value
    ts_after_normalise.fold.assert_called_once_with(2.5, 32, subints=1)


@patch(f"{_MODULE}.TimeSeries")
def test_fold_single_channel_folds_with_correct_args_with_deredden(mock_ts_class: MagicMock) -> None:
    """_fold_single_channel calls fold(period, bins, subints=1) after deredden and normalise."""
    channel_data = np.zeros(128)
    _fold_single_channel(channel_data, tsamp=0.001, period=2.5, bins=32, do_deredden=True, rmed_width=12.0)
    ts_after_normalise = mock_ts_class.from_numpy_array.return_value.deredden.return_value.normalise.return_value
    ts_after_normalise.fold.assert_called_once_with(2.5, 32, subints=1)


@patch(f"{_MODULE}.TimeSeries")
def test_fold_single_channel_returns_fold_result(mock_ts_class: MagicMock) -> None:
    """_fold_single_channel returns exactly the array produced by TimeSeries.fold."""
    channel_data = np.zeros(128)
    expected = np.linspace(0.0, 1.0, 16)
    mock_ts_class.from_numpy_array.return_value.normalise.return_value.fold.return_value = expected
    result = _fold_single_channel(channel_data, tsamp=0.001, period=1.0, bins=16, do_deredden=False, rmed_width=12.0)
    assert result is expected
