"""Unit tests for functions in blipss.plotting.plots"""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from blipss.plotting.plots import (
    _add_candidate_legend,
    _add_snr_colorbar,
    _annotate_period_bins,
    _apply_row_x_labels,
    _configure_periodogram_ax,
    _configure_scatter_axes,
    _fold_and_normalize,
    _log_period_tick_formatter,
    _make_snr_norm,
    _phasebin_centers,
    _plot_avg_profile_ax,
    _plot_band_averaged_spectrum,
    _plot_freq_averaged_profile,
    _plot_phase_resolved_spectrum,
    _plot_phase_time_ax,
    _require_latex,
    _save_and_close,
    _scatter_candidate_group,
    candverf_plot,
    plot_phase_resolved_dynamic_spectrum,
    scatterplot_period_radiofreq,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _agg_backend() -> Generator[None, None, None]:
    """Switch to the non-interactive Agg backend and close all figures after each test."""
    plt.switch_backend("Agg")
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# _require_latex
# ---------------------------------------------------------------------------


@patch("blipss.plotting.plots.shutil.which", return_value=None)
def test_require_latex_raises_when_pdflatex_missing(mock_which: MagicMock) -> None:
    """_require_latex raises RuntimeError when pdflatex is absent from PATH."""
    with pytest.raises(RuntimeError, match="use_latex=True requires pdflatex"):
        _require_latex()
    mock_which.assert_called_once_with("pdflatex")


@patch("blipss.plotting.plots.shutil.which", return_value="/usr/bin/pdflatex")
def test_require_latex_passes_when_pdflatex_available(mock_which: MagicMock) -> None:
    """_require_latex returns without error when pdflatex is found on PATH."""
    _require_latex()  # must not raise
    mock_which.assert_called_once_with("pdflatex")


# ---------------------------------------------------------------------------
# _phasebin_centers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nbins", "expected"),
    [
        (1, np.array([0.5])),
        (2, np.array([0.25, 0.75])),
        (4, np.array([0.125, 0.375, 0.625, 0.875])),
    ],
    ids=["single_bin", "two_bins", "four_bins"],
)
def test_phasebin_centers_returns_correct_values(nbins: int, expected: npt.NDArray[np.floating]) -> None:
    """_phasebin_centers returns the central phase of each bin for known bin counts."""
    np.testing.assert_allclose(_phasebin_centers(nbins), expected)


@pytest.mark.parametrize("nbins", [1, 3, 8, 16, 100], ids=["1", "3", "8", "16", "100"])
def test_phasebin_centers_length_equals_nbins(nbins: int) -> None:
    """_phasebin_centers returns an array whose length equals nbins."""
    assert len(_phasebin_centers(nbins)) == nbins


@given(st.integers(min_value=1, max_value=1024))
def test_phasebin_centers_all_values_in_unit_interval(nbins: int) -> None:
    """_phasebin_centers values always lie in [0, 1) for any positive bin count."""
    centers = _phasebin_centers(nbins)
    assert np.all(centers >= 0.0)
    assert np.all(centers < 1.0)


# ---------------------------------------------------------------------------
# _save_and_close
# ---------------------------------------------------------------------------


@patch("blipss.plotting.plots.plt.close")
@patch("blipss.plotting.plots.plt.savefig")
def test_save_and_close_calls_savefig_once_per_format(mock_savefig: MagicMock, mock_close: MagicMock) -> None:
    """_save_and_close calls plt.savefig exactly once per format entry."""
    _save_and_close("output/plot", [".png", ".pdf"])
    assert mock_savefig.call_count == 2
    mock_savefig.assert_any_call("output/plot.png")
    mock_savefig.assert_any_call("output/plot.pdf")
    mock_close.assert_called_once()


@patch("blipss.plotting.plots.plt.close")
@patch("blipss.plotting.plots.plt.savefig")
def test_save_and_close_empty_formats_skips_savefig(mock_savefig: MagicMock, mock_close: MagicMock) -> None:
    """_save_and_close skips plt.savefig entirely when the formats list is empty."""
    _save_and_close("plot", [])
    mock_savefig.assert_not_called()
    mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# _make_snr_norm
# ---------------------------------------------------------------------------


def test_make_snr_norm_returns_cividis_colormap() -> None:
    """_make_snr_norm returns the cividis colormap as the first element of the tuple."""
    cmap, _ = _make_snr_norm(np.array([3.0, 7.0, 5.0]))
    assert cmap.name == "cividis"


def test_make_snr_norm_norm_bounds_match_snr_range() -> None:
    """_make_snr_norm normalizer spans exactly [min(snrs), max(snrs)]."""
    snrs = np.array([2.0, 5.0, 3.5])
    _, norm = _make_snr_norm(snrs)
    assert norm.vmin == pytest.approx(2.0)
    assert norm.vmax == pytest.approx(5.0)


def test_make_snr_norm_single_element_has_equal_bounds() -> None:
    """_make_snr_norm sets vmin == vmax when the S/N array contains a single value."""
    _, norm = _make_snr_norm(np.array([7.5]))
    assert norm.vmin == pytest.approx(7.5)
    assert norm.vmax == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# _scatter_candidate_group
# ---------------------------------------------------------------------------


def test_scatter_candidate_group_no_match_returns_false() -> None:
    """_scatter_candidate_group returns False when no candidate carries the requested flag."""
    plt.figure()
    cmap, norm = _make_snr_norm(np.array([5.0]))
    result = _scatter_candidate_group(
        np.array([1.0]),
        np.array([1400.0]),
        np.array([5.0]),
        np.array(["F"]),
        "H",
        "x",
        "Harmonic",
        cmap,
        norm,
    )
    assert result is False


def test_scatter_candidate_group_matching_flag_returns_true() -> None:
    """_scatter_candidate_group returns True when at least one candidate matches the flag."""
    plt.figure()
    cmap, norm = _make_snr_norm(np.array([5.0, 8.0]))
    result = _scatter_candidate_group(
        np.array([1.0, 2.0]),
        np.array([1400.0, 1500.0]),
        np.array([5.0, 8.0]),
        np.array(["F", "H"]),
        "H",
        "x",
        "Harmonic",
        cmap,
        norm,
    )
    assert result is True


def test_scatter_candidate_group_plots_only_matching_subset() -> None:
    """_scatter_candidate_group scatter-plots only the candidates whose flag matches."""
    plt.figure()
    cmap, norm = _make_snr_norm(np.array([5.0, 8.0, 6.0]))
    _scatter_candidate_group(
        np.array([1.0, 2.0, 3.0]),
        np.array([1400.0, 1500.0, 1600.0]),
        np.array([5.0, 8.0, 6.0]),
        np.array(["F", "H", "F"]),
        "F",
        "o",
        "Fundamental",
        cmap,
        norm,
    )
    ax = plt.gca()
    assert len(ax.collections) == 1
    assert len(np.asarray(ax.collections[0].get_offsets())) == 2  # two 'F' candidates


# ---------------------------------------------------------------------------
# _configure_scatter_axes
# ---------------------------------------------------------------------------


def test_configure_scatter_axes_xlim_and_ylim() -> None:
    """_configure_scatter_axes sets the x- and y-axis limits to the supplied period/freq bounds."""
    plt.figure()
    _configure_scatter_axes(0.1, 10.0, 1200.0, 1600.0)
    ax = plt.gca()
    assert ax.get_xlim() == pytest.approx((0.1, 10.0))
    assert ax.get_ylim() == pytest.approx((1200.0, 1600.0))


# ---------------------------------------------------------------------------
# _add_snr_colorbar
# ---------------------------------------------------------------------------


def test_add_snr_colorbar_adds_axes_to_figure() -> None:
    """_add_snr_colorbar appends a colorbar axes to the current figure."""
    fig, ax = plt.subplots()
    cmap, norm = _make_snr_norm(np.array([3.0, 8.0]))
    _add_snr_colorbar(cmap, norm, ax)
    assert len(fig.axes) == 2  # original axes + colorbar axes


# ---------------------------------------------------------------------------
# _add_candidate_legend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker_count", [0, 1], ids=["zero_markers", "one_marker"])
def test_add_candidate_legend_no_legend_for_single_or_zero_markers(marker_count: int) -> None:
    """_add_candidate_legend does not create a legend when marker_count is 0 or 1."""
    plt.figure()
    _add_candidate_legend(marker_count)
    assert plt.gca().get_legend() is None


@pytest.mark.parametrize("marker_count", [2, 3], ids=["two_markers", "three_markers"])
def test_add_candidate_legend_creates_legend_for_multiple_markers(marker_count: int) -> None:
    """_add_candidate_legend creates a legend when more than one marker type was plotted."""
    plt.figure()
    plt.scatter([1.0], [1400.0], label="Fundamental")
    _add_candidate_legend(marker_count)
    assert plt.gca().get_legend() is not None


# ---------------------------------------------------------------------------
# _log_period_tick_formatter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (1.0, "$1$"),
        (10.0, "$10$"),
        (100.0, "$100$"),
        (0.1, "$0.1$"),
        (0.01, "$0.01$"),
        (0.001, "$0.001$"),
    ],
    ids=["1s", "10s", "100s", "0.1s", "0.01s", "0.001s"],
)
def test_log_period_tick_formatter_positive_values(x: float, expected: str) -> None:
    """_log_period_tick_formatter produces the correct decimal label for positive period values."""
    assert _log_period_tick_formatter(x, 0) == expected


def test_log_period_tick_formatter_zero_returns_zero_decimals() -> None:
    """_log_period_tick_formatter returns '$0$' (zero decimal places) when x is zero."""
    assert _log_period_tick_formatter(0.0, 0) == "$0$"


# ---------------------------------------------------------------------------
# _fold_and_normalize
# ---------------------------------------------------------------------------


def test_fold_and_normalize_returns_two_element_tuple() -> None:
    """_fold_and_normalize returns a 2-tuple of (phase_time, profile) arrays."""
    mock_ts = MagicMock()
    mock_ts.fold.side_effect = [np.ones((10, 16)), np.array([[2.0, 4.0, 3.0, 1.0]])]
    result = _fold_and_normalize(mock_ts, period=1.0, bins=4)
    assert len(result) == 2


def test_fold_and_normalize_profile_normalized_to_unit_max() -> None:
    """_fold_and_normalize divides the profile by its maximum so the peak value equals 1."""
    mock_ts = MagicMock()
    profile_raw = np.array([[2.0, 4.0, 3.0, 1.0]])
    mock_ts.fold.side_effect = [np.ones((5, 4)), profile_raw]
    _, profile = _fold_and_normalize(mock_ts, period=0.5, bins=4)
    assert float(np.max(profile)) == pytest.approx(1.0)


def test_fold_and_normalize_all_zero_profile_returned_unchanged() -> None:
    """_fold_and_normalize returns a zero profile without modification when max_val is zero."""
    mock_ts = MagicMock()
    zero_profile = np.zeros((1, 4))
    mock_ts.fold.side_effect = [np.ones((3, 4)), zero_profile]
    _, profile = _fold_and_normalize(mock_ts, period=1.0, bins=4)
    np.testing.assert_array_equal(profile, zero_profile)


# ---------------------------------------------------------------------------
# _apply_row_x_labels
# ---------------------------------------------------------------------------


def test_apply_row_x_labels_non_last_row_suppresses_tick_labels() -> None:
    """_apply_row_x_labels hides x tick labels on ax_prd and ax_prof for non-last rows."""
    ax_prd, ax_prof, ax_pt = MagicMock(), MagicMock(), MagicMock()
    _apply_row_x_labels(ax_prd, ax_prof, ax_pt, is_last_row=False)
    ax_prd.set_xticklabels.assert_called_once_with([])
    ax_prof.set_xticklabels.assert_called_once_with([])


def test_apply_row_x_labels_last_row_sets_xlabel_and_hides_profile_xaxis() -> None:
    """_apply_row_x_labels sets x-axis labels on ax_prd/ax_pt and hides ax_prof xaxis for last row."""
    ax_prd, ax_prof, ax_pt = MagicMock(), MagicMock(), MagicMock()
    _apply_row_x_labels(ax_prd, ax_prof, ax_pt, is_last_row=True)
    ax_prd.set_xlabel.assert_called_once()
    ax_pt.set_xlabel.assert_called_once()
    ax_prof.xaxis.set_visible.assert_called_once_with(False)


# ---------------------------------------------------------------------------
# _configure_periodogram_ax
# ---------------------------------------------------------------------------


def test_configure_periodogram_ax_linear_scale_when_log_disabled() -> None:
    """_configure_periodogram_ax keeps a linear x-axis when periodaxis_log is False."""
    _, ax = plt.subplots()
    periods = np.linspace(0.1, 10.0, 50)
    snrs = np.ones(50) * 6.0
    _configure_periodogram_ax(ax, periods, snrs, "ON", 59000.0, 1.0, 12.0, False)
    assert ax.get_xscale() == "linear"


def test_configure_periodogram_ax_log_scale_when_log_enabled() -> None:
    """_configure_periodogram_ax applies a logarithmic x-axis when periodaxis_log is True."""
    _, ax = plt.subplots()
    periods = np.linspace(0.1, 10.0, 50)
    snrs = np.ones(50) * 6.0
    _configure_periodogram_ax(ax, periods, snrs, "ON", 59000.0, 1.0, 12.0, True)
    assert ax.get_xscale() == "log"


def test_configure_periodogram_ax_ylim_upper_equals_max_snr() -> None:
    """_configure_periodogram_ax sets the y-axis upper limit to max_snr."""
    _, ax = plt.subplots()
    periods = np.linspace(0.5, 5.0, 20)
    snrs = np.ones(20) * 5.0
    _configure_periodogram_ax(ax, periods, snrs, "OFF", 58000.0, 1.0, 9.0, False)
    assert ax.get_ylim()[1] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# _plot_avg_profile_ax
# ---------------------------------------------------------------------------


def test_plot_avg_profile_ax_xlim_is_unit_interval() -> None:
    """_plot_avg_profile_ax sets x-axis limits to [0.0, 1.0]."""
    _, ax = plt.subplots()
    centers = _phasebin_centers(8)
    _plot_avg_profile_ax(ax, centers, np.ones(8))
    assert ax.get_xlim() == pytest.approx((0.0, 1.0))


# ---------------------------------------------------------------------------
# _plot_phase_time_ax
# ---------------------------------------------------------------------------


def test_plot_phase_time_ax_xlim_is_unit_interval() -> None:
    """_plot_phase_time_ax sets x-axis limits to [0.0, 1.0]."""
    _, ax = plt.subplots()
    centers = _phasebin_centers(8)
    phase_time = np.random.default_rng(1).uniform(0, 1, (5, 8))
    _plot_phase_time_ax(ax, phase_time, centers, period=1.0)
    assert ax.get_xlim() == pytest.approx((0.0, 1.0))


# ---------------------------------------------------------------------------
# _plot_freq_averaged_profile
# ---------------------------------------------------------------------------


def test_plot_freq_averaged_profile_xlim_is_unit_interval() -> None:
    """_plot_freq_averaged_profile sets x-axis limits to [0.0, 1.0]."""
    _, ax = plt.subplots()
    ds = np.random.default_rng(2).uniform(0, 1, (16, 8))
    centers = _phasebin_centers(8)
    _plot_freq_averaged_profile(ax, centers, ds)
    assert ax.get_xlim() == pytest.approx((0.0, 1.0))


def test_plot_freq_averaged_profile_plots_channel_mean() -> None:
    """_plot_freq_averaged_profile plots the channel-averaged (axis=0) dynamic spectrum."""
    _, ax = plt.subplots()
    ds = np.arange(16, dtype=float).reshape(4, 4)
    centers = _phasebin_centers(4)
    _plot_freq_averaged_profile(ax, centers, ds)
    plotted_y: npt.NDArray[np.floating] = np.asarray(ax.lines[0].get_ydata())
    np.testing.assert_allclose(plotted_y, np.mean(ds, axis=0))


# ---------------------------------------------------------------------------
# _plot_phase_resolved_spectrum
# ---------------------------------------------------------------------------


def test_plot_phase_resolved_spectrum_xlim_and_ylim() -> None:
    """_plot_phase_resolved_spectrum sets x-axis to [0, 1] and y-axis to the frequency range."""
    _, ax = plt.subplots()
    ds = np.ones((4, 8))
    centers = _phasebin_centers(8)
    freqs = np.linspace(1200.0, 1500.0, 4)
    _plot_phase_resolved_spectrum(ax, ds, centers, freqs, start_mjd=59000.0)
    assert ax.get_xlim() == pytest.approx((0.0, 1.0))
    assert ax.get_ylim() == pytest.approx((freqs[0], freqs[-1]))


# ---------------------------------------------------------------------------
# _plot_band_averaged_spectrum
# ---------------------------------------------------------------------------


def test_plot_band_averaged_spectrum_ylim_matches_freq_range() -> None:
    """_plot_band_averaged_spectrum sets y-axis limits to the first and last frequency values."""
    _, ax = plt.subplots()
    freqs = np.linspace(1200.0, 1600.0, 8)
    ds = np.ones((8, 4))
    _plot_band_averaged_spectrum(ax, ds, freqs)
    assert ax.get_ylim() == pytest.approx((freqs[0], freqs[-1]))


def test_plot_band_averaged_spectrum_plots_phase_bin_mean() -> None:
    """_plot_band_averaged_spectrum plots the time-averaged spectrum (mean over phase bins)."""
    _, ax = plt.subplots()
    ds = np.arange(24, dtype=float).reshape(6, 4)
    freqs = np.linspace(1200.0, 1600.0, 6)
    _plot_band_averaged_spectrum(ax, ds, freqs)
    plotted_x: npt.NDArray[np.floating] = np.asarray(ax.lines[0].get_xdata())
    np.testing.assert_allclose(plotted_x, np.mean(ds, axis=1))


# ---------------------------------------------------------------------------
# _annotate_period_bins
# ---------------------------------------------------------------------------


def test_annotate_period_bins_adds_two_text_elements() -> None:
    """_annotate_period_bins places exactly two text artists on the blank panel axes."""
    _, axes = plt.subplots(2, 2)
    ax = axes[0, 1]
    initial_count = len(ax.texts)
    _annotate_period_bins(ax, period=1.23456, bins=16)
    assert len(ax.texts) == initial_count + 2


# ---------------------------------------------------------------------------
# scatterplot_period_radiofreq
# ---------------------------------------------------------------------------


@patch("blipss.plotting.plots._save_and_close")
@patch("blipss.plotting.plots._require_latex")
def test_scatterplot_calls_require_latex_when_use_latex_true(mock_require: MagicMock, mock_save: MagicMock) -> None:
    """scatterplot_period_radiofreq calls _require_latex when use_latex is True."""
    scatterplot_period_radiofreq(
        np.array([1.0, 2.0]),
        np.array([1400.0, 1500.0]),
        np.array([5.0, 7.0]),
        np.array(["F", "H"]),
        "out/plot",
        0.1,
        5.0,
        1200.0,
        1600.0,
        use_latex=True,
    )
    mock_require.assert_called_once()
    mock_save.assert_called_once_with("out/plot", [".png"])


@patch("blipss.plotting.plots._save_and_close")
@patch("blipss.plotting.plots._require_latex")
def test_scatterplot_does_not_call_require_latex_when_use_latex_false(
    mock_require: MagicMock, mock_save: MagicMock
) -> None:
    """scatterplot_period_radiofreq skips _require_latex when use_latex is False."""
    scatterplot_period_radiofreq(
        np.array([1.0]),
        np.array([1400.0]),
        np.array([5.0]),
        np.array(["F"]),
        "out/plot",
        0.1,
        5.0,
        1200.0,
        1600.0,
        use_latex=False,
    )
    mock_require.assert_not_called()
    mock_save.assert_called_once_with("out/plot", [".png"])


@patch("blipss.plotting.plots._save_and_close")
def test_scatterplot_forwards_custom_formats_to_save(mock_save: MagicMock) -> None:
    """scatterplot_period_radiofreq passes user-supplied plot_formats to _save_and_close."""
    formats = [".pdf", ".svg"]
    scatterplot_period_radiofreq(
        np.array([1.0]),
        np.array([1400.0]),
        np.array([5.0]),
        np.array(["F"]),
        "out/plot",
        0.1,
        5.0,
        1200.0,
        1600.0,
        plot_formats=formats,
    )
    mock_save.assert_called_once_with("out/plot", formats)


# ---------------------------------------------------------------------------
# candverf_plot
# ---------------------------------------------------------------------------


def _make_mock_periodogram_cv(n_periods: int, n_widths: int) -> MagicMock:
    pg: MagicMock = MagicMock()
    pg.periods = np.linspace(0.1, 10.0, n_periods)
    pg.snrs = np.random.default_rng(0).uniform(3, 10, (n_periods, n_widths))
    return pg


@patch("blipss.plotting.plots._save_and_close")
@patch("blipss.plotting.plots._fold_and_normalize")
@patch("blipss.plotting.plots._configure_periodogram_ax")
@patch("blipss.plotting.plots._require_latex")
def test_candverf_plot_calls_require_latex_when_use_latex_true(
    mock_require: MagicMock,
    mock_config_ax: MagicMock,
    mock_fold: MagicMock,
    mock_save: MagicMock,
) -> None:
    """candverf_plot calls _require_latex when use_latex is True."""
    mock_fold.return_value = (np.ones((3, 8)), np.ones(8))
    candverf_plot(
        period=1.0,
        bins=8,
        detrended_ts=[MagicMock()],
        periodograms=[_make_mock_periodogram_cv(20, 5)],
        annotations=["ON"],
        start_mjds=[59000.0],
        max_snr=12.0,
        periodaxis_log=False,
        plot_name="out/cand",
        output_formats=[".png"],
        use_latex=True,
    )
    mock_require.assert_called_once()
    mock_fold.assert_called_once()
    mock_config_ax.assert_called_once()
    mock_save.assert_called_once_with("out/cand", [".png"])


@patch("blipss.plotting.plots._save_and_close")
@patch("blipss.plotting.plots._fold_and_normalize")
@patch("blipss.plotting.plots._configure_periodogram_ax")
@patch("blipss.plotting.plots._require_latex")
def test_candverf_plot_does_not_call_require_latex_when_use_latex_false(
    mock_require: MagicMock,
    mock_config_ax: MagicMock,
    mock_fold: MagicMock,
    mock_save: MagicMock,
) -> None:
    """candverf_plot skips _require_latex when use_latex is False."""
    mock_fold.return_value = (np.ones((3, 8)), np.ones(8))
    candverf_plot(
        period=1.0,
        bins=8,
        detrended_ts=[MagicMock()],
        periodograms=[_make_mock_periodogram_cv(20, 5)],
        annotations=["ON"],
        start_mjds=[59000.0],
        max_snr=12.0,
        periodaxis_log=False,
        plot_name="out/cand",
        output_formats=[".png"],
        use_latex=False,
    )
    mock_require.assert_not_called()
    mock_fold.assert_called_once()
    mock_config_ax.assert_called_once()
    mock_save.assert_called_once_with("out/cand", [".png"])


@patch("blipss.plotting.plots._save_and_close")
@patch("blipss.plotting.plots._fold_and_normalize")
@patch("blipss.plotting.plots._configure_periodogram_ax")
def test_candverf_plot_calls_save_and_close_with_correct_args(
    mock_config_ax: MagicMock,
    mock_fold: MagicMock,
    mock_save: MagicMock,
) -> None:
    """candverf_plot passes plot_name and output_formats to _save_and_close."""
    mock_fold.return_value = (np.ones((3, 8)), np.ones(8))
    candverf_plot(
        period=1.0,
        bins=8,
        detrended_ts=[MagicMock()],
        periodograms=[_make_mock_periodogram_cv(20, 5)],
        annotations=["ON"],
        start_mjds=[59000.0],
        max_snr=12.0,
        periodaxis_log=False,
        plot_name="out/cand",
        output_formats=[".pdf", ".png"],
    )
    mock_save.assert_called_once_with("out/cand", [".pdf", ".png"])
    mock_fold.assert_called_once()
    mock_config_ax.assert_called_once()


# ---------------------------------------------------------------------------
# plot_phase_resolved_dynamic_spectrum
# ---------------------------------------------------------------------------


@patch("blipss.plotting.plots._save_and_close")
@patch("blipss.plotting.plots._require_latex")
def test_plot_phase_resolved_dynamic_spectrum_calls_require_latex_when_use_latex_true(
    mock_require: MagicMock, mock_save: MagicMock
) -> None:
    """plot_phase_resolved_dynamic_spectrum calls _require_latex when use_latex is True."""
    ds = np.random.default_rng(0).uniform(0, 1, (8, 16))
    freqs = np.linspace(1200.0, 1500.0, 8)
    plot_phase_resolved_dynamic_spectrum(
        ds, freqs, period=1.0, start_mjd=59000.0, plot_name="out/ds", plot_formats=[".png"], use_latex=True
    )
    mock_require.assert_called_once()
    mock_save.assert_called_once_with("out/ds", [".png"])


@patch("blipss.plotting.plots._save_and_close")
@patch("blipss.plotting.plots._require_latex")
def test_plot_phase_resolved_dynamic_spectrum_does_not_call_require_latex_when_use_latex_false(
    mock_require: MagicMock, mock_save: MagicMock
) -> None:
    """plot_phase_resolved_dynamic_spectrum skips _require_latex when use_latex is False."""
    ds = np.random.default_rng(0).uniform(0, 1, (8, 16))
    freqs = np.linspace(1200.0, 1500.0, 8)
    plot_phase_resolved_dynamic_spectrum(
        ds, freqs, period=1.0, start_mjd=59000.0, plot_name="out/ds", plot_formats=[".png"], use_latex=False
    )
    mock_require.assert_not_called()
    mock_save.assert_called_once_with("out/ds", [".png"])


@patch("blipss.plotting.plots._save_and_close")
def test_plot_phase_resolved_dynamic_spectrum_calls_save_and_close_with_correct_args(mock_save: MagicMock) -> None:
    """plot_phase_resolved_dynamic_spectrum passes plot_name and plot_formats to _save_and_close."""
    ds = np.random.default_rng(0).uniform(0, 1, (8, 16))
    freqs = np.linspace(1200.0, 1500.0, 8)
    plot_phase_resolved_dynamic_spectrum(
        ds, freqs, period=1.0, start_mjd=59000.0, plot_name="out/ds", plot_formats=[".png", ".pdf"]
    )
    mock_save.assert_called_once_with("out/ds", [".png", ".pdf"])
