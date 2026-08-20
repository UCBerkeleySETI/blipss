"""Unit tests for modules in blipss.core.compare_cands"""

from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from blipss.core.compare_cands import (
    _best_candidate_per_channel_cluster,
    _detection_code,
    _process_channel_range,
    filter_fundamental_candidates,
    group_candidates_by_channel,
)

_MODULE = "blipss.core.compare_cands"

# ---------------------------------------------------------------------------
# filter_fundamental_candidates
# ---------------------------------------------------------------------------


def test_filter_fundamental_candidates_keeps_fundamental_above_threshold() -> None:
    """filter_fundamental_candidates retains only fundamental-flagged candidates at or above the S/N cutoff."""
    channels = np.array([0, 1, 2], dtype=np.intp)
    radiofreqs = np.array([1000.0, 1001.0, 1002.0])
    phase_bins = np.array([10, 10, 10], dtype=np.uint)
    boxcar_widths = np.array([2, 2, 2], dtype=np.uint)
    periods = np.array([1.0, 2.0, 3.0])
    snrs = np.array([8.0, 6.0, 9.0])
    flags = np.array(["F", "F", "H"])

    out = filter_fundamental_candidates(channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, flags, 7.0)

    np.testing.assert_array_equal(out[0], [0])
    np.testing.assert_allclose(out[1], [1000.0])
    np.testing.assert_allclose(out[4], [1.0])
    np.testing.assert_allclose(out[5], [8.0])


def test_filter_fundamental_candidates_threshold_is_inclusive() -> None:
    """filter_fundamental_candidates retains a fundamental candidate whose S/N equals the threshold exactly."""
    channels = np.array([0], dtype=np.intp)
    radiofreqs = np.array([1000.0])
    phase_bins = np.array([10], dtype=np.uint)
    boxcar_widths = np.array([2], dtype=np.uint)
    periods = np.array([1.0])
    snrs = np.array([7.0])
    flags = np.array(["F"])

    out = filter_fundamental_candidates(channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, flags, 7.0)
    assert len(out[4]) == 1


def test_filter_fundamental_candidates_empty_input_returns_empty_arrays() -> None:
    """filter_fundamental_candidates returns empty arrays when given empty input candidate arrays."""
    empty_intp: npt.NDArray[np.intp] = np.array([], dtype=np.intp)
    empty_f: npt.NDArray[np.floating] = np.array([], dtype=float)
    empty_u: npt.NDArray[np.uint] = np.array([], dtype=np.uint)
    empty_flags: npt.NDArray[np.str_] = np.array([], dtype=np.str_)

    out = filter_fundamental_candidates(empty_intp, empty_f, empty_u, empty_u, empty_f, empty_f, empty_flags, 7.0)
    for arr in out:
        assert len(arr) == 0


# ---------------------------------------------------------------------------
# _detection_code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file_indices", "n_files", "expected"),
    [
        (np.array([0]), 3, "100"),
        (np.array([0, 2]), 3, "101"),
        (np.array([0, 0, 1, 1]), 2, "11"),
        (np.array([2]), 3, "001"),
    ],
    ids=["single_file", "two_files_sorted", "duplicate_indices_dedup", "highest_index_only"],
)
def test_detection_code_marks_contributing_files(
    file_indices: npt.NDArray[np.intp], n_files: int, expected: str
) -> None:
    """_detection_code sets '1' at each contributing file's position and '0' elsewhere."""
    assert _detection_code(file_indices, n_files) == expected


# ---------------------------------------------------------------------------
# _best_candidate_per_channel_cluster
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.cluster1d")
def test_best_candidate_per_channel_cluster_single_cluster_keeps_highest_snr(
    mock_cluster1d: MagicMock,
) -> None:
    """Returns the highest-S/N candidate and a code covering all contributing files."""
    mock_cluster1d.return_value = [np.array([0, 1, 2, 3])]

    file_index = np.array([0, 2, 2, 3], dtype=np.intp)
    radiofreqs = np.array([1001.0, 1002.0, 1003.0, 1004.0])
    phase_bins = np.array([10, 10, 10, 10], dtype=np.uint)
    boxcar_widths = np.array([2, 2, 2, 2], dtype=np.uint)
    periods = np.array([1.0, 2.0, 3.0, 4.0])
    snrs = np.array([5.0, 10.0, 3.0, 7.0])

    out_freqs, _, _, out_p, out_s, out_codes = _best_candidate_per_channel_cluster(
        file_index,
        radiofreqs,
        phase_bins,
        boxcar_widths,
        periods,
        snrs,
        n_files=4,
        cluster_radius=1.0,
    )

    assert len(out_p) == 1
    np.testing.assert_allclose(out_p, [2.0])
    np.testing.assert_allclose(out_s, [10.0])
    np.testing.assert_allclose(out_freqs, [1002.0])
    np.testing.assert_array_equal(out_codes, ["1011"])


@patch(f"{_MODULE}.cluster1d")
def test_best_candidate_per_channel_cluster_independent_clusters_each_get_own_code(
    mock_cluster1d: MagicMock,
) -> None:
    """_best_candidate_per_channel_cluster assigns one best candidate and detection code per cluster."""
    mock_cluster1d.return_value = [np.array([0]), np.array([1]), np.array([2])]
    file_index = np.array([0, 1, 2], dtype=np.intp)
    radiofreqs = np.array([1000.0, 1000.0, 1000.0])
    phase_bins = np.array([10, 20, 30], dtype=np.uint)
    boxcar_widths = np.array([1, 2, 3], dtype=np.uint)
    periods = np.array([1.0, 2.0, 3.0])
    snrs = np.array([5.0, 10.0, 3.0])

    _, _, _, out_p, out_s, out_codes = _best_candidate_per_channel_cluster(
        file_index, radiofreqs, phase_bins, boxcar_widths, periods, snrs, n_files=3, cluster_radius=0.1
    )

    assert len(out_p) == 3
    np.testing.assert_allclose(out_s, [5.0, 10.0, 3.0])
    np.testing.assert_array_equal(out_codes, ["100", "010", "001"])


@patch(f"{_MODULE}.cluster1d")
def test_best_candidate_per_channel_cluster_sorts_by_period_before_clustering(mock_cluster1d: MagicMock) -> None:
    """_best_candidate_per_channel_cluster passes period-sorted arrays to cluster1d."""
    mock_cluster1d.return_value = [np.array([0, 1, 2])]
    file_index = np.array([0, 1, 2], dtype=np.intp)
    radiofreqs = np.array([1000.0, 1001.0, 1002.0])
    phase_bins = np.array([30, 10, 20], dtype=np.uint)
    boxcar_widths = np.array([3, 1, 2], dtype=np.uint)
    periods = np.array([3.0, 1.0, 2.0])  # unsorted
    snrs = np.array([3.0, 5.0, 10.0])

    _best_candidate_per_channel_cluster(
        file_index, radiofreqs, phase_bins, boxcar_widths, periods, snrs, n_files=3, cluster_radius=0.1
    )

    sorted_periods_arg: npt.NDArray[np.floating] = mock_cluster1d.call_args.args[0]
    np.testing.assert_array_equal(sorted_periods_arg, np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# _process_channel_range
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}._best_candidate_per_channel_cluster")
def test_process_channel_range_merges_all_channels_in_order(mock_best: MagicMock) -> None:
    """_process_channel_range concatenates one cluster's worth of output per channel, in channel order."""
    mock_best.side_effect = [
        (
            np.array([1000.0]),
            np.array([10], dtype=np.uint),
            np.array([2], dtype=np.uint),
            np.array([1.0]),
            np.array([8.0]),
            np.array(["10"]),
        ),
        (
            np.array([1001.0, 1002.0]),
            np.array([10, 10], dtype=np.uint),
            np.array([2, 2], dtype=np.uint),
            np.array([2.0, 3.0]),
            np.array([9.0, 7.0]),
            np.array(["01", "11"]),
        ),
    ]
    chunk_channels = np.array([5, 6], dtype=np.intp)
    chunk_bounds = np.array([0, 1, 3], dtype=np.intp)
    file_index = np.zeros(3, dtype=np.intp)
    radiofreqs = np.zeros(3)
    phase_bins = np.zeros(3, dtype=np.uint)
    boxcar_widths = np.zeros(3, dtype=np.uint)
    periods = np.zeros(3)
    snrs = np.zeros(3)

    out_channels, out_freqs, _, _, out_periods, _, out_codes = _process_channel_range(
        chunk_channels, chunk_bounds, file_index, radiofreqs, phase_bins, boxcar_widths, periods, snrs, 2, 0.1
    )

    assert out_channels == [5, 6, 6]
    assert out_freqs == [1000.0, 1001.0, 1002.0]
    assert out_periods == [1.0, 2.0, 3.0]
    assert out_codes == ["10", "01", "11"]
    assert mock_best.call_count == 2


@patch(f"{_MODULE}._best_candidate_per_channel_cluster")
def test_process_channel_range_no_channels_returns_empty_lists(mock_best: MagicMock) -> None:
    """_process_channel_range returns empty output lists when given no channels to process."""
    empty_intp: npt.NDArray[np.intp] = np.array([], dtype=np.intp)
    out = _process_channel_range(
        empty_intp,
        np.array([0], dtype=np.intp),
        empty_intp,
        np.array([]),
        np.array([], dtype=np.uint),
        np.array([], dtype=np.uint),
        np.array([]),
        np.array([]),
        2,
        0.1,
    )
    for lst in out:
        assert lst == []
    mock_best.assert_not_called()


@pytest.mark.parametrize("show_progress", [True, False], ids=["with_progress_bar", "without_progress_bar"])
@patch(f"{_MODULE}.tqdm", side_effect=lambda x, **kw: x)
@patch(f"{_MODULE}._best_candidate_per_channel_cluster")
def test_process_channel_range_uses_tqdm_only_when_progress_requested(
    mock_best: MagicMock, mock_tqdm: MagicMock, show_progress: bool
) -> None:
    """_process_channel_range wraps the channel iterator in tqdm only when show_progress is True."""
    mock_best.return_value = (
        np.array([]),
        np.array([], dtype=np.uint),
        np.array([], dtype=np.uint),
        np.array([]),
        np.array([]),
        np.array([], dtype=np.str_),
    )
    chunk_channels = np.array([1], dtype=np.intp)
    chunk_bounds = np.array([0, 0], dtype=np.intp)
    empty_f: npt.NDArray[np.floating] = np.array([])
    empty_u: npt.NDArray[np.uint] = np.array([], dtype=np.uint)
    _process_channel_range(
        chunk_channels,
        chunk_bounds,
        np.array([], dtype=np.intp),
        empty_f,
        empty_u,
        empty_u,
        empty_f,
        empty_f,
        2,
        0.1,
        show_progress=show_progress,
    )
    assert mock_tqdm.called == show_progress


# ---------------------------------------------------------------------------
# group_candidates_by_channel
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}._process_channel_range")
def test_group_candidates_by_channel_sequential_sorts_by_channel_before_delegating(mock_process: MagicMock) -> None:
    """group_candidates_by_channel (n_jobs=1) sorts all inputs by channel and delegates to one sequential chunk."""
    mock_process.return_value = ([1], [1000.0], [10], [2], [1.5], [8.0], ["10"])
    channels = np.array([2, 0, 1], dtype=np.intp)
    file_index = np.array([0, 1, 2], dtype=np.intp)
    radiofreqs = np.array([1002.0, 1000.0, 1001.0])
    phase_bins = np.array([10, 10, 10], dtype=np.uint)
    boxcar_widths = np.array([2, 2, 2], dtype=np.uint)
    periods = np.array([3.0, 1.0, 2.0])
    snrs = np.array([9.0, 7.0, 8.0])

    out_channels, out_freqs, *_rest = group_candidates_by_channel(
        file_index,
        channels,
        radiofreqs,
        phase_bins,
        boxcar_widths,
        periods,
        snrs,
        n_files=3,
        cluster_radius=0.1,
        n_jobs=1,
    )

    mock_process.assert_called_once()
    call_args = mock_process.call_args.args
    np.testing.assert_array_equal(call_args[0], [0, 1, 2])  # unique channels, ascending
    np.testing.assert_array_equal(call_args[3], [1000.0, 1001.0, 1002.0])  # channel-sorted radiofreqs
    assert mock_process.call_args.kwargs == {"show_progress": True}
    np.testing.assert_array_equal(out_channels, [1])
    np.testing.assert_allclose(out_freqs, [1000.0])


@patch(f"{_MODULE}.ProcessPoolExecutor")
def test_group_candidates_by_channel_parallel_dispatch_merges_chunk_results(mock_executor_cls: MagicMock) -> None:
    """group_candidates_by_channel (n_jobs>1) dispatches chunks via ProcessPoolExecutor and merges their results."""
    future_a = MagicMock()
    future_a.result.return_value = ([0], [1000.0], [10], [2], [1.0], [8.0], ["10"])
    future_b = MagicMock()
    future_b.result.return_value = ([1], [1001.0], [10], [2], [2.0], [9.0], ["01"])
    mock_executor = MagicMock()
    mock_executor.submit.side_effect = [future_a, future_b]
    mock_executor_cls.return_value.__enter__.return_value = mock_executor

    channels = np.array([0, 1], dtype=np.intp)
    file_index = np.array([0, 1], dtype=np.intp)
    radiofreqs = np.array([1000.0, 1001.0])
    phase_bins = np.array([10, 10], dtype=np.uint)
    boxcar_widths = np.array([2, 2], dtype=np.uint)
    periods = np.array([1.0, 2.0])
    snrs = np.array([8.0, 9.0])

    out_channels, out_freqs, _, _, _, _, out_codes = group_candidates_by_channel(
        file_index,
        channels,
        radiofreqs,
        phase_bins,
        boxcar_widths,
        periods,
        snrs,
        n_files=2,
        cluster_radius=0.1,
        n_jobs=2,
    )

    assert mock_executor.submit.call_count == 2
    np.testing.assert_array_equal(out_channels, [0, 1])
    np.testing.assert_allclose(out_freqs, [1000.0, 1001.0])
    np.testing.assert_array_equal(out_codes, ["10", "01"])


@patch(f"{_MODULE}.ProcessPoolExecutor")
@patch(f"{_MODULE}._process_channel_range")
@patch(f"{_MODULE}.os.cpu_count", return_value=1)
def test_group_candidates_by_channel_n_jobs_negative_one_resolves_via_cpu_count(
    mock_cpu_count: MagicMock, mock_process: MagicMock, mock_executor_cls: MagicMock
) -> None:
    """group_candidates_by_channel resolves n_jobs=-1 to os.cpu_count(), taking the sequential path when it is 1."""
    mock_process.return_value = ([], [], [], [], [], [], [])
    channels = np.array([0], dtype=np.intp)
    file_index = np.array([0], dtype=np.intp)
    radiofreqs = np.array([1000.0])
    phase_bins = np.array([10], dtype=np.uint)
    boxcar_widths = np.array([2], dtype=np.uint)
    periods = np.array([1.0])
    snrs = np.array([8.0])

    group_candidates_by_channel(
        file_index,
        channels,
        radiofreqs,
        phase_bins,
        boxcar_widths,
        periods,
        snrs,
        n_files=1,
        cluster_radius=0.1,
        n_jobs=-1,
    )

    mock_cpu_count.assert_called_once()
    mock_process.assert_called_once()
    mock_executor_cls.assert_not_called()


def test_group_candidates_by_channel_end_to_end_clusters_and_codes_real_data() -> None:
    """group_candidates_by_channel clusters same-channel periods and builds correct per-cluster detection codes."""
    # Channel 0: two candidates from different files within cluster_radius -> merge, keep higher S/N, code "11".
    # Channel 1: one candidate from file 0 only -> code "10".
    file_index = np.array([0, 1, 0], dtype=np.intp)
    channels = np.array([0, 0, 1], dtype=np.intp)
    radiofreqs = np.array([1000.0, 1000.0, 1001.0])
    phase_bins = np.array([10, 10, 10], dtype=np.uint)
    boxcar_widths = np.array([2, 2, 2], dtype=np.uint)
    periods = np.array([1.0, 1.0002, 5.0])
    snrs = np.array([8.0, 12.0, 6.0])

    out_channels, _out_freqs, _out_pb, _out_bw, out_periods, out_snrs, out_codes = group_candidates_by_channel(
        file_index,
        channels,
        radiofreqs,
        phase_bins,
        boxcar_widths,
        periods,
        snrs,
        n_files=2,
        cluster_radius=1.0e-3,
        n_jobs=1,
    )

    np.testing.assert_array_equal(out_channels, [0, 1])
    np.testing.assert_allclose(out_periods, [1.0002, 5.0])
    np.testing.assert_allclose(out_snrs, [12.0, 6.0])
    np.testing.assert_array_equal(out_codes, ["11", "10"])
