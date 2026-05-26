"""Unit tests for modules in blipss.core.period_finding"""

from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from blipss.constants import CANDIDATE_DECIMAL_PRECISION
from blipss.core.period_finding import (
    _best_candidate_per_cluster,
    _extract_candidates_above_threshold,
    _finalize_candidate_arrays,
    _run_ffa_on_channel,
    _search_single_channel,
    search_all_channels,
)


def _make_mock_periodogram(
    snrs_2d: npt.NDArray[np.floating],
    periods: npt.NDArray[np.floating],
    foldbins: npt.NDArray[np.intp],
    widths: npt.NDArray[np.intp],
) -> MagicMock:
    pg: MagicMock = MagicMock()
    pg.snrs = np.array(snrs_2d)
    pg.periods = np.array(periods)
    pg.foldbins = np.array(foldbins)
    pg.widths = np.array(widths)
    return pg


# ---------------------------------------------------------------------------
# _finalize_candidate_arrays
# ---------------------------------------------------------------------------


def test_finalize_candidate_arrays_empty_inputs_return_empty_arrays() -> None:
    """_finalize_candidate_arrays returns empty arrays of correct types for empty inputs."""
    channels = np.array([], dtype=float)
    periods = np.array([], dtype=float)
    snrs = np.array([], dtype=float)
    phase_bins = np.array([], dtype=int)
    bw = np.array([], dtype=int)
    out = _finalize_candidate_arrays(channels, periods, snrs, phase_bins, bw)
    for arr in out:
        assert len(arr) == 0


def test_finalize_candidate_arrays_channels_and_bins_cast_to_int() -> None:
    """_finalize_candidate_arrays casts channels, phase_bins, and boxcar widths to int dtype."""
    channels = np.array([0.0, 1.0, 2.0])
    periods = np.zeros(3)
    snrs = np.zeros(3)
    phase_bins = np.array([10.0, 20.0, 30.0])
    bw = np.array([1.0, 2.0, 3.0])
    out_ch, _, _, out_pb, out_bw = _finalize_candidate_arrays(channels, periods, snrs, phase_bins, bw)
    assert out_ch.dtype == np.dtype(int)
    assert out_pb.dtype == np.dtype(int)
    assert out_bw.dtype == np.dtype(int)


@pytest.mark.parametrize(
    "input_dtype",
    [np.float32, np.float16],
    ids=["float32", "float16"],
)
def test_finalize_candidate_arrays_periods_and_snrs_upcast_to_float64(
    input_dtype: type,
) -> None:
    """_finalize_candidate_arrays upcasts float32/float16 period and S/N arrays to float64."""
    n = 3
    channels = np.zeros(n, dtype=int)
    periods = np.ones(n, dtype=input_dtype)
    snrs = np.ones(n, dtype=input_dtype)
    phase_bins = np.zeros(n, dtype=int)
    bw = np.zeros(n, dtype=int)
    _, out_p, out_s, _, _ = _finalize_candidate_arrays(channels, periods, snrs, phase_bins, bw)
    assert out_p.dtype == np.float64
    assert out_s.dtype == np.float64


def test_finalize_candidate_arrays_rounds_periods_and_snrs_to_precision() -> None:
    """_finalize_candidate_arrays rounds periods and S/N values to CANDIDATE_DECIMAL_PRECISION."""
    channels = np.array([0])
    periods = np.array([1.123456789])
    snrs = np.array([7.987654321])
    phase_bins = np.array([10])
    bw = np.array([2])
    _, out_p, out_s, _, _ = _finalize_candidate_arrays(channels, periods, snrs, phase_bins, bw)
    assert out_p[0] == pytest.approx(round(1.123456789, CANDIDATE_DECIMAL_PRECISION))
    assert out_s[0] == pytest.approx(round(7.987654321, CANDIDATE_DECIMAL_PRECISION))


# Largest float where x * 10^CANDIDATE_DECIMAL_PRECISION stays within float64 range,
# preventing overflow inside np.round.
_ROUND_SAFE_MAX = np.finfo(np.float64).max / 10**CANDIDATE_DECIMAL_PRECISION


@given(
    st.lists(
        st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-_ROUND_SAFE_MAX,
            max_value=_ROUND_SAFE_MAX,
        ),
        min_size=0,
        max_size=20,
    )
)
def test_finalize_candidate_arrays_rounding_is_idempotent(
    period_values: list[float],
) -> None:
    """Re-rounding output periods does not change them (idempotency of rounding to precision)."""
    n = len(period_values)
    channels = np.zeros(n, dtype=int)
    periods = np.array(period_values)
    snrs = np.zeros(n)
    phase_bins = np.zeros(n, dtype=int)
    bw = np.zeros(n, dtype=int)
    _, out_p, _, _, _ = _finalize_candidate_arrays(channels, periods, snrs, phase_bins, bw)
    np.testing.assert_array_equal(out_p, np.round(out_p, CANDIDATE_DECIMAL_PRECISION))


# ---------------------------------------------------------------------------
# _extract_candidates_above_threshold
# ---------------------------------------------------------------------------


def test_extract_candidates_above_threshold_all_below_returns_none() -> None:
    """_extract_candidates_above_threshold returns None when no period exceeds the S/N threshold."""
    pg = _make_mock_periodogram(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([0.5, 1.0]),
        np.array([10, 20]),
        np.array([1, 2]),
    )
    result = _extract_candidates_above_threshold(pg, snr_threshold=5.0)
    assert result is None


def test_extract_candidates_above_threshold_at_threshold_is_included() -> None:
    """_extract_candidates_above_threshold includes a period whose peak S/N equals the threshold."""
    pg = _make_mock_periodogram(
        np.array([[3.0, 5.0]]),
        np.array([1.0]),
        np.array([10]),
        np.array([1, 2]),
    )
    result = _extract_candidates_above_threshold(pg, snr_threshold=5.0)
    assert result is not None
    periods, snrs, _, _ = result
    assert len(periods) == 1
    np.testing.assert_allclose(snrs, [5.0])


def test_extract_candidates_above_threshold_returns_only_above_threshold_entries() -> None:
    """_extract_candidates_above_threshold filters out periods whose peak S/N falls below threshold."""
    # period 0: max S/N = 3.0 (below 5.0), period 1: max S/N = 6.0 (above 5.0)
    pg = _make_mock_periodogram(
        np.array([[1.0, 3.0], [5.0, 6.0]]),
        np.array([0.5, 1.0]),
        np.array([10, 20]),
        np.array([1, 2]),
    )
    result = _extract_candidates_above_threshold(pg, snr_threshold=5.0)
    assert result is not None
    periods, snrs, phase_bins, _ = result
    assert len(periods) == 1
    np.testing.assert_allclose(periods, [1.0])
    np.testing.assert_allclose(snrs, [6.0])
    np.testing.assert_array_equal(phase_bins, [20])


def test_extract_candidates_above_threshold_selects_argmax_width() -> None:
    """_extract_candidates_above_threshold returns the boxcar width index with maximum S/N."""
    # snrs=[[2.0, 8.0]], widths=[3, 5]; argmax at width index 1 → width value 5
    pg = _make_mock_periodogram(
        np.array([[2.0, 8.0]]),
        np.array([1.0]),
        np.array([10]),
        np.array([3, 5]),
    )
    result = _extract_candidates_above_threshold(pg, snr_threshold=5.0)
    assert result is not None
    _, _, _, boxcar_widths = result
    np.testing.assert_array_equal(boxcar_widths, [5])


# ---------------------------------------------------------------------------
# _best_candidate_per_cluster
# ---------------------------------------------------------------------------


@patch("blipss.core.period_finding.cluster1d")
def test_best_candidate_per_cluster_single_cluster_keeps_highest_snr(
    mock_cluster1d: MagicMock,
) -> None:
    """_best_candidate_per_cluster returns the single highest-S/N candidate from a one-cluster input."""
    mock_cluster1d.return_value = [np.array([0, 1, 2])]
    periods = np.array([1.0, 2.0, 3.0])
    snrs = np.array([5.0, 10.0, 3.0])
    phase_bins = np.array([10, 20, 30])
    boxcar_widths = np.array([1, 2, 3])
    out_p, out_s, out_pb, out_bw = _best_candidate_per_cluster(
        periods, snrs, phase_bins, boxcar_widths, epsilon_fof=0.1
    )
    assert len(out_p) == 1
    np.testing.assert_allclose(out_p, [2.0])
    np.testing.assert_allclose(out_s, [10.0])
    np.testing.assert_array_equal(out_pb, [20])
    np.testing.assert_array_equal(out_bw, [2])


@patch("blipss.core.period_finding.cluster1d")
def test_best_candidate_per_cluster_independent_clusters_all_retained_sorted_by_snr(
    mock_cluster1d: MagicMock,
) -> None:
    """_best_candidate_per_cluster returns one best entry per cluster, sorted by descending S/N."""
    mock_cluster1d.return_value = [np.array([0]), np.array([1]), np.array([2])]
    periods = np.array([1.0, 2.0, 3.0])
    snrs = np.array([5.0, 10.0, 3.0])
    phase_bins = np.array([10, 20, 30])
    boxcar_widths = np.array([1, 2, 3])
    out_p, out_s, _, _ = _best_candidate_per_cluster(periods, snrs, phase_bins, boxcar_widths, epsilon_fof=0.1)
    assert len(out_p) == 3
    np.testing.assert_allclose(out_s, [10.0, 5.0, 3.0])
    np.testing.assert_allclose(out_p, [2.0, 1.0, 3.0])


@patch("blipss.core.period_finding.cluster1d")
def test_best_candidate_per_cluster_sorts_by_period_before_clustering(
    mock_cluster1d: MagicMock,
) -> None:
    """_best_candidate_per_cluster passes period-sorted arrays to cluster1d."""
    mock_cluster1d.return_value = [np.array([0, 1, 2])]
    periods = np.array([3.0, 1.0, 2.0])  # unsorted
    snrs = np.array([3.0, 5.0, 10.0])
    phase_bins = np.array([30, 10, 20])
    boxcar_widths = np.array([3, 1, 2])
    _best_candidate_per_cluster(periods, snrs, phase_bins, boxcar_widths, epsilon_fof=0.1)
    sorted_periods_arg: npt.NDArray[np.floating] = mock_cluster1d.call_args.args[0]
    np.testing.assert_array_equal(sorted_periods_arg, np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# _run_ffa_on_channel
# ---------------------------------------------------------------------------


@patch("blipss.core.period_finding.ffa_search")
@patch("blipss.core.period_finding.TimeSeries")
def test_run_ffa_on_channel_passes_search_params_to_ffa_search(mock_ts_cls: MagicMock, mock_ffa: MagicMock) -> None:
    """_run_ffa_on_channel forwards all search parameters as keyword arguments to ffa_search."""
    channel_data = np.zeros(200)
    mock_ts_cls.from_numpy_array.return_value = MagicMock()
    mock_ffa.return_value = (MagicMock(), MagicMock())
    _run_ffa_on_channel(channel_data, 0.001, 0.5, 60.0, 5, 32, 128, 0.2, True, 2.0)
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


@patch("blipss.core.period_finding.ffa_search")
@patch("blipss.core.period_finding.TimeSeries")
def test_run_ffa_on_channel_returns_periodogram_from_ffa_search(mock_ts_cls: MagicMock, mock_ffa: MagicMock) -> None:
    """_run_ffa_on_channel returns the second element (Periodogram) of the ffa_search result."""
    channel_data = np.zeros(200)
    mock_ts_cls.from_numpy_array.return_value = MagicMock()
    mock_pg = MagicMock()
    mock_ffa.return_value = (MagicMock(), mock_pg)
    result = _run_ffa_on_channel(channel_data, 0.001, 0.5, 60.0, 5, 32, 128, 0.2, False, 2.0)
    assert result is mock_pg
    mock_ts_cls.from_numpy_array.assert_called_once()
    mock_ffa.assert_called_once()


# ---------------------------------------------------------------------------
# _search_single_channel
# ---------------------------------------------------------------------------


@patch("blipss.core.period_finding.label_harmonics")
@patch("blipss.core.period_finding._best_candidate_per_cluster")
@patch("blipss.core.period_finding._extract_candidates_above_threshold")
@patch("blipss.core.period_finding._run_ffa_on_channel")
def test_search_single_channel_no_candidates_returns_none(
    mock_ffa: MagicMock,
    mock_extract: MagicMock,
    mock_cluster: MagicMock,
    mock_label: MagicMock,
) -> None:
    """_search_single_channel returns None when no period passes the S/N threshold."""
    mock_extract.return_value = None
    result = _search_single_channel(np.zeros(100), 0.001, 0.1, 10.0, 5, 32, 128, 0.5, False, 1.0, 7.0, 0.1, 0.01)
    assert result is None
    mock_ffa.assert_called_once()
    mock_extract.assert_called_once()
    mock_cluster.assert_not_called()
    mock_label.assert_not_called()


@patch("blipss.core.period_finding.label_harmonics")
@patch("blipss.core.period_finding._best_candidate_per_cluster")
@patch("blipss.core.period_finding._extract_candidates_above_threshold")
@patch("blipss.core.period_finding._run_ffa_on_channel")
def test_search_single_channel_with_candidates_returns_five_tuple(
    mock_ffa: MagicMock,
    mock_extract: MagicMock,
    mock_cluster: MagicMock,
    mock_label: MagicMock,
) -> None:
    """_search_single_channel returns a 5-tuple of arrays when candidates exceed the threshold."""
    periods = np.array([1.0])
    snrs = np.array([8.0])
    phase_bins = np.array([10])
    bw = np.array([2])
    flags = np.array(["F"])
    mock_extract.return_value = (periods, snrs, phase_bins, bw)
    mock_cluster.return_value = (periods, snrs, phase_bins, bw)
    mock_label.return_value = flags
    result = _search_single_channel(np.zeros(100), 0.001, 0.1, 10.0, 5, 32, 128, 0.5, False, 1.0, 7.0, 0.1, 0.01)
    assert result is not None
    assert len(result) == 5
    np.testing.assert_array_equal(result[0], periods)
    np.testing.assert_array_equal(result[4], flags)
    mock_ffa.assert_called_once()
    mock_extract.assert_called_once()
    mock_cluster.assert_called_once()
    mock_label.assert_called_once()


@patch("blipss.core.period_finding.label_harmonics")
@patch("blipss.core.period_finding._best_candidate_per_cluster")
@patch("blipss.core.period_finding._extract_candidates_above_threshold")
@patch("blipss.core.period_finding._run_ffa_on_channel")
def test_search_single_channel_calls_label_harmonics_with_presorted_true(
    mock_ffa: MagicMock,
    mock_extract: MagicMock,
    mock_cluster: MagicMock,
    mock_label: MagicMock,
) -> None:
    """_search_single_channel always invokes label_harmonics with presorted=True."""
    periods = np.array([1.0])
    snrs = np.array([8.0])
    phase_bins = np.array([10])
    bw = np.array([2])
    mock_extract.return_value = (periods, snrs, phase_bins, bw)
    mock_cluster.return_value = (periods, snrs, phase_bins, bw)
    mock_label.return_value = np.array(["F"])
    _search_single_channel(np.zeros(100), 0.001, 0.1, 10.0, 5, 32, 128, 0.5, False, 1.0, 7.0, 0.1, 0.01)
    assert mock_label.call_args.kwargs.get("presorted") is True
    mock_ffa.assert_called_once()
    mock_extract.assert_called_once()
    mock_cluster.assert_called_once()


# ---------------------------------------------------------------------------
# search_all_channels
# ---------------------------------------------------------------------------


@patch("blipss.core.period_finding.tqdm", side_effect=lambda x, **kw: x)
@patch("blipss.core.period_finding._search_single_channel")
def test_search_all_channels_no_detections_returns_empty_arrays(mock_search: MagicMock, mock_tqdm: MagicMock) -> None:
    """search_all_channels returns six empty arrays when every channel yields no candidates."""
    mock_search.return_value = None
    data = np.zeros((3, 100))
    channels, periods, snrs, phase_bins, bw, flags = search_all_channels(
        data, 0, 0.001, 0.1, 10.0, 5, 32, 128, 0.5, False, 1.0, 7.0, 0.1, 0.01
    )
    assert mock_search.call_count == 3
    mock_tqdm.assert_called_once()
    for arr in (channels, periods, snrs, phase_bins, bw, flags):
        assert len(arr) == 0


@patch("blipss.core.period_finding.tqdm", side_effect=lambda x, **kw: x)
@patch("blipss.core.period_finding._search_single_channel")
def test_search_all_channels_merges_candidates_from_every_channel(mock_search: MagicMock, mock_tqdm: MagicMock) -> None:
    """search_all_channels concatenates candidates from all channels into a single flat array."""
    p = np.array([1.0])
    s = np.array([8.0])
    pb = np.array([10])
    bw = np.array([2])
    flags = np.array(["F"])
    mock_search.return_value = (p, s, pb, bw, flags)
    data = np.zeros((3, 100))
    channels, periods, _, _, _, out_flags = search_all_channels(
        data, 0, 0.001, 0.1, 10.0, 5, 32, 128, 0.5, False, 1.0, 7.0, 0.1, 0.01
    )
    # 3 channels x 1 candidate each
    assert mock_search.call_count == 3
    mock_tqdm.assert_called_once()
    assert len(channels) == 3
    assert len(periods) == 3
    assert len(out_flags) == 3


@patch("blipss.core.period_finding.tqdm", side_effect=lambda x, **kw: x)
@patch("blipss.core.period_finding._search_single_channel")
def test_search_all_channels_applies_start_channel_offset(mock_search: MagicMock, mock_tqdm: MagicMock) -> None:
    """search_all_channels adds start_channel to per-channel indices in the output."""
    p = np.array([1.0])
    mock_search.return_value = (p, np.array([8.0]), np.array([10]), np.array([2]), np.array(["F"]))
    data = np.zeros((2, 100))
    channels, *_ = search_all_channels(data, 10, 0.001, 0.1, 10.0, 5, 32, 128, 0.5, False, 1.0, 7.0, 0.1, 0.01)
    assert mock_search.call_count == 2
    mock_tqdm.assert_called_once()
    np.testing.assert_array_equal(np.sort(channels), [10, 11])


@patch("blipss.core.period_finding.tqdm", side_effect=lambda x, **kw: x)
@patch("blipss.core.period_finding._search_single_channel")
def test_search_all_channels_skips_channels_with_no_candidates(mock_search: MagicMock, mock_tqdm: MagicMock) -> None:
    """search_all_channels omits channels for which _search_single_channel returns None."""
    p = np.array([1.0])
    hit = (p, np.array([8.0]), np.array([10]), np.array([2]), np.array(["F"]))
    mock_search.side_effect = [None, hit, None]
    data = np.zeros((3, 100))
    channels, *_ = search_all_channels(data, 0, 0.001, 0.1, 10.0, 5, 32, 128, 0.5, False, 1.0, 7.0, 0.1, 0.01)
    assert mock_search.call_count == 3
    mock_tqdm.assert_called_once()
    assert len(channels) == 1
    assert channels[0] == 1
