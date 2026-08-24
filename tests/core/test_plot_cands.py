"""Unit tests for modules in blipss.core.plot_cands"""

from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from blipss.core.plot_cands import run_ffa_and_fold_channel, select_candidates_by_code

_MODULE = "blipss.core.plot_cands"

# ---------------------------------------------------------------------------
# select_candidates_by_code
# ---------------------------------------------------------------------------


def test_select_candidates_by_code_keeps_only_matching_codes() -> None:
    """select_candidates_by_code retains candidates whose code is one of codes_plot, in original order."""
    channels = np.array([0, 1, 2], dtype=np.intp)
    periods = np.array([1.0, 2.0, 3.0])
    bins = np.array([10, 20, 30], dtype=np.uint)
    codes = np.array(["100", "010", "001"])

    out_channels, out_periods, out_bins, out_codes = select_candidates_by_code(
        channels, periods, bins, codes, ["100", "001"]
    )

    np.testing.assert_array_equal(out_channels, [0, 2])
    np.testing.assert_allclose(out_periods, [1.0, 3.0])
    np.testing.assert_array_equal(out_bins, [10, 30])
    np.testing.assert_array_equal(out_codes, ["100", "001"])


def test_select_candidates_by_code_no_match_returns_empty_arrays() -> None:
    """select_candidates_by_code returns empty arrays when no candidate code is in codes_plot."""
    channels = np.array([0, 1], dtype=np.intp)
    periods = np.array([1.0, 2.0])
    bins = np.array([10, 20], dtype=np.uint)
    codes = np.array(["100", "010"])

    out_channels, out_periods, out_bins, out_codes = select_candidates_by_code(channels, periods, bins, codes, ["111"])

    assert len(out_channels) == 0
    assert len(out_periods) == 0
    assert len(out_bins) == 0
    assert len(out_codes) == 0


def test_select_candidates_by_code_empty_codes_plot_returns_empty_arrays() -> None:
    """select_candidates_by_code returns empty arrays when codes_plot is empty."""
    channels = np.array([0], dtype=np.intp)
    periods = np.array([1.0])
    bins = np.array([10], dtype=np.uint)
    codes = np.array(["100"])

    out_channels, *_rest = select_candidates_by_code(channels, periods, bins, codes, [])
    assert len(out_channels) == 0


# ---------------------------------------------------------------------------
# run_ffa_and_fold_channel
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.ffa_search")
@patch(f"{_MODULE}.TimeSeries")
def test_run_ffa_and_fold_channel_passes_search_params_to_ffa_search(
    mock_ts_cls: MagicMock, mock_ffa: MagicMock
) -> None:
    """run_ffa_and_fold_channel forwards all search parameters as keyword arguments to ffa_search."""
    channel_data: npt.NDArray[np.floating] = np.zeros(200)
    mock_ts_cls.from_numpy_array.return_value = MagicMock()
    mock_ffa.return_value = (MagicMock(), MagicMock())

    run_ffa_and_fold_channel(channel_data, 0.001, 0.5, 60.0, 5, 32, 128, 0.2, True, 2.0)

    mock_ts_cls.from_numpy_array.assert_called_once_with(channel_data, tsamp=0.001)
    kw = mock_ffa.call_args.kwargs
    assert kw["period_min"] == pytest.approx(0.5)
    assert kw["period_max"] == pytest.approx(60.0)
    assert kw["fpmin"] == 5
    assert kw["bins_min"] == 32
    assert kw["bins_max"] == 128
    assert kw["ducy_max"] == pytest.approx(0.2)
    assert kw["deredden"] is True
    assert kw["rmed_width"] == pytest.approx(2.0)
    assert kw["already_normalised"] is False


@patch(f"{_MODULE}.ffa_search")
@patch(f"{_MODULE}.TimeSeries")
def test_run_ffa_and_fold_channel_returns_detrended_series_and_periodogram(
    mock_ts_cls: MagicMock, mock_ffa: MagicMock
) -> None:
    """run_ffa_and_fold_channel returns the (detrended_ts, periodogram) tuple produced by ffa_search."""
    mock_ts_cls.from_numpy_array.return_value = MagicMock()
    detrended_ts, periodogram = MagicMock(), MagicMock()
    mock_ffa.return_value = (detrended_ts, periodogram)

    result = run_ffa_and_fold_channel(np.zeros(50), 0.001, 0.5, 60.0, 5, 32, 128, 0.2, False, 2.0)

    assert result == (detrended_ts, periodogram)
