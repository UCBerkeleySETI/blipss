"""
Plotting utilities for BLIPSS candidate visualization and diagnostic plots.
"""

import shutil
from collections.abc import Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from ..constants import LABEL_FONTSIZE, SCATTER_MARKER_SIZE, TICK_LABELSIZE, TICK_LENGTH

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _require_latex() -> None:
    if shutil.which("pdflatex") is None:
        raise RuntimeError(
            "use_latex=True requires pdflatex on the system PATH. "
            "Install a LaTeX distribution (e.g. 'brew install --cask mactex' on macOS, "
            "'apt install texlive-full' on Debian/Ubuntu) and ensure pdflatex is accessible."
        )


def _phasebin_centers(nbins: int) -> npt.NDArray[np.floating]:
    """
    Compute the central phase value of each phase bin.

    Args:
        nbins: Number of phase bins

    Returns:
        Array of length nbins with each bin's central phase in [0, 1)
    """
    return (0.5 + np.arange(nbins)) / nbins


def _save_and_close(plot_name: str, formats: Sequence[str]) -> None:
    """
    Save the current matplotlib figure in each requested format, then close it.

    Args:
        plot_name: Output file path without extension
        formats: File extensions (including leading dot) for each output format
    """
    for fmt in formats:
        plt.savefig(plot_name + fmt)
    plt.close()


# ---------------------------------------------------------------------------
# scatterplot_period_radiofreq helpers
# ---------------------------------------------------------------------------


def _make_snr_norm(
    snrs: npt.NDArray[np.floating],
) -> tuple[mpl.colors.Colormap, mpl.colors.Normalize]:
    """
    Build the cividis colormap and a normalizer scaled to the given S/N range.

    Args:
        snrs: Array of matched-filtering S/N values

    Returns:
        Tuple of (colormap, normalizer) ready for scatter or ScalarMappable
    """
    cmap = mpl.colormaps["cividis"]
    norm = mpl.colors.Normalize(vmin=float(np.min(snrs)), vmax=float(np.max(snrs)))
    return cmap, norm


def _scatter_candidate_group(
    periods: npt.NDArray[np.floating],
    freqs: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    flags: npt.NDArray[np.str_],
    flag: str,
    marker: str,
    label: str,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> bool:
    """
    Scatter-plot the subset of candidates whose flag matches a given value.

    Args:
        periods: Trial periods (s) for all candidates
        freqs: Radio frequencies (MHz) for all candidates
        snrs: S/N values for all candidates
        flags: Harmonic flag string for each candidate ('F', 'S', or 'H')
        flag: The flag value to select and plot
        marker: Matplotlib marker style string
        label: Legend label for this group
        cmap: Colormap for coloring markers by S/N
        norm: Normalizer that maps S/N values to the [0, 1] color range

    Returns:
        True if at least one candidate matched the flag; False otherwise
    """
    idx = np.where(flags == flag)[0]
    if len(idx) == 0:
        return False
    plt.scatter(
        x=periods[idx],
        y=freqs[idx],
        c=snrs[idx],
        s=SCATTER_MARKER_SIZE,
        marker=marker,
        cmap=cmap,
        norm=norm,
        label=label,
    )
    return True


def _configure_scatter_axes(
    min_period: float,
    max_period: float,
    min_freq: float,
    max_freq: float,
) -> None:
    """
    Apply axis labels, font sizes, and limits to the current scatter axes.

    Args:
        min_period: Minimum trial period (s) for the x-axis
        max_period: Maximum trial period (s) for the x-axis
        min_freq: Minimum radio frequency (MHz) for the y-axis
        max_freq: Maximum radio frequency (MHz) for the y-axis
    """
    plt.xlabel("Trial folding period (s)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Radio frequency (MHz)", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_LABELSIZE)
    plt.yticks(fontsize=TICK_LABELSIZE)
    plt.gca().set_xlim((min_period, max_period))
    plt.gca().set_ylim((min_freq, max_freq))


def _add_snr_colorbar(
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    ax: Axes,
) -> None:
    """
    Attach a labeled S/N colorbar to the current figure.

    Args:
        cmap: Colormap used for the scatter plot
        norm: Normalizer that maps S/N values to the color range
        ax: Axes from which the colorbar steals space
    """
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    cbar.set_label("Matched filtering S/N", fontsize=LABEL_FONTSIZE)


def _add_candidate_legend(marker_count: int) -> None:
    """
    Add a multi-column legend below the axes when more than one marker type is present.

    Args:
        marker_count: Number of distinct marker groups that were plotted
    """
    if marker_count <= 1:
        return
    leg = plt.legend(
        bbox_to_anchor=(0.0, -0.22, 1.1, 0.1),
        loc="center",
        ncol=3,
        fancybox=True,
        frameon=True,
        borderpad=0.3,
        fontsize=TICK_LABELSIZE,
        handletextpad=0.2,
        columnspacing=2.0,
    )
    leg.get_frame().set_edgecolor("silver")


def scatterplot_period_radiofreq(
    cand_periods: npt.NDArray[np.floating],
    cand_freqs: npt.NDArray[np.floating],
    cand_snrs: npt.NDArray[np.floating],
    cand_flags: npt.NDArray[np.str_],
    basename: str,
    min_period: float,
    max_period: float,
    min_freq: float,
    max_freq: float,
    plot_formats: Sequence[str] | None = None,
    use_latex: bool = False,
) -> None:
    """
    Produce a scatter plot of periodicity detections in the radio-frequency vs. trial-period plane.

    Candidate S/N values are shown on a color scale. Fundamental frequencies use a circular
    marker, sub-harmonics a plus symbol, and harmonics a cross symbol.

    Args:
        cand_periods: Trial periods (s) of detected signals
        cand_freqs: Radio frequencies (MHz) corresponding to candidate detections
        cand_snrs: Matched-filtering S/N values of detected candidates
        cand_flags: Harmonic flags ('F', 'S', or 'H') assigned to each candidate
        basename: Output plot basename including output path (no extension)
        min_period: Minimum trial period (s) shown on the x-axis
        max_period: Maximum trial period (s) shown on the x-axis
        min_freq: Minimum radio frequency (MHz) on the y-axis
        max_freq: Maximum radio frequency (MHz) on the y-axis
        plot_formats: File extensions (with leading dot) for saving the plot; defaults to ['.png']
        use_latex: Render text with LaTeX (requires a system LaTeX installation); defaults to False
    """
    if use_latex:
        _require_latex()
    if plot_formats is None:
        plot_formats = [".png"]
    rc = {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"} if use_latex else {}
    with mpl.rc_context(rc):
        cmap, norm = _make_snr_norm(cand_snrs)
        plt.figure(figsize=(7, 6))
        marker_count = 0
        for flag, marker, label in [
            ("F", "o", "Fundamental"),
            ("S", "+", "Subharmonic"),
            ("H", "x", "Harmonic"),
        ]:
            if _scatter_candidate_group(
                cand_periods, cand_freqs, cand_snrs, cand_flags, flag, marker, label, cmap, norm
            ):
                marker_count += 1
        _add_snr_colorbar(cmap, norm, plt.gca())
        _configure_scatter_axes(min_period, max_period, min_freq, max_freq)
        _add_candidate_legend(marker_count)
        plt.tight_layout()
        _save_and_close(basename, plot_formats)


# ---------------------------------------------------------------------------
# candverf_plot helpers
# ---------------------------------------------------------------------------


def _log_period_tick_formatter(x: float, _pos: Any) -> str:
    """
    Format a period value as a decimal label for a log-scale axis.

    Args:
        x: Axis tick value (period in seconds)
        _pos: Tick position index required by matplotlib's FuncFormatter interface

    Returns:
        LaTeX-formatted decimal string with enough significant decimal places for x
    """
    n_decimals = int(max(-np.log10(x), 0)) if x > 0 else 0
    return f"${x:.{n_decimals}f}$"


def _configure_periodogram_ax(
    ax: Axes,
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    annotation: str,
    start_mjd: float,
    cand_period: float,
    max_snr: float,
    periodaxis_log: bool,
) -> None:
    """
    Plot the periodogram SNR curve and configure annotations, limits, and scale.

    Args:
        ax: Axes on which to draw the periodogram
        periods: Trial period grid (s)
        snrs: Maximum S/N at each trial period (pre-reduced over DM/width axes)
        annotation: Custom text placed in the upper-right of the panel
        start_mjd: Start MJD (UTC) of the observation, shown in the upper-left
        cand_period: Candidate folding period (s), marked with a dashed vertical line
        max_snr: Upper S/N limit for the y-axis
        periodaxis_log: Whether to use a logarithmic period axis
    """
    ax.plot(periods, snrs, "-k")
    ax.annotate(annotation, xy=(0.8, 0.82), xycoords="axes fraction", fontsize=TICK_LABELSIZE)
    ax.annotate(
        f"Start MJD (UTC) = {start_mjd:.4f}",
        xy=(0.03, 0.82),
        xycoords="axes fraction",
        fontsize=TICK_LABELSIZE,
    )
    ax.axvline(x=cand_period, ymin=0.0, ymax=0.8, color="salmon", linestyle="--")
    ax.set_ylim((-0.1, max_snr))
    if periodaxis_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_log_period_tick_formatter))
        ax.tick_params(axis="x", which="minor", length=TICK_LENGTH)
    ax.tick_params(axis="x", which="major", length=TICK_LENGTH, labelsize=TICK_LABELSIZE)
    ax.tick_params(axis="y", which="major", length=TICK_LENGTH, labelsize=TICK_LABELSIZE)


def _fold_and_normalize(
    ts: Any,
    period: float,
    bins: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Fold a riptide TimeSeries and return the phase-time array and normalized average profile.

    Args:
        ts: Riptide TimeSeries object with a fold() method
        period: Folding period (s)
        bins: Number of phase bins

    Returns:
        Tuple of (phase_time, profile) where phase_time has shape (N_periods, bins)
        and profile is normalized to unit maximum
    """
    phase_time: npt.NDArray[np.floating] = ts.fold(period, bins, subints=None)
    profile: npt.NDArray[np.floating] = ts.fold(period, bins, subints=1)
    max_val = np.max(profile)
    profile = profile / max_val if max_val != 0 else profile
    return phase_time, profile


def _plot_avg_profile_ax(
    ax: Axes,
    phasebin_centers: npt.NDArray[np.floating],
    profile: npt.NDArray[np.floating],
) -> None:
    """
    Plot a normalized average pulse profile on the given axes.

    Args:
        ax: Axes on which to draw the profile
        phasebin_centers: Central phase value of each bin
        profile: Normalized flux density at each phase bin
    """
    ax.plot(phasebin_centers, profile, "-k")
    ax.set_ylabel(r"$\overline{S}$ (a.u.)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="y", length=TICK_LENGTH, labelsize=TICK_LABELSIZE)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim((0.0, 1.0))


def _plot_phase_time_ax(
    ax: Axes,
    phase_time: npt.NDArray[np.floating],
    phasebin_centers: npt.NDArray[np.floating],
    period: float,
) -> None:
    """
    Plot a phase-time pulse stack diagram on the given axes.

    Args:
        ax: Axes on which to draw the diagram
        phase_time: 2D array of shape (N_periods, bins) with flux values
        phasebin_centers: Central phase value of each bin
        period: Folding period (s), used to compute the time extent on the y-axis
    """
    ax.imshow(
        phase_time,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        cmap="Greys",
        extent=(phasebin_centers[0], phasebin_centers[-1], 0.0, len(phase_time) * period),
    )
    ax.set_ylabel(r"$t$ (s)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", length=TICK_LENGTH, labelsize=TICK_LABELSIZE)
    ax.tick_params(axis="y", length=TICK_LENGTH, labelsize=TICK_LABELSIZE)
    ax.set_xlim((0.0, 1.0))


def _apply_row_x_labels(
    ax_prd: Axes,
    ax_prof: Axes,
    ax_pt: Axes,
    is_last_row: bool,
) -> None:
    """
    Show or hide x-axis labels on one row of candidate verification subplots.

    Labels are suppressed on all rows except the last to avoid clutter in multi-row figures.

    Args:
        ax_prd: Periodogram axes (left column)
        ax_prof: Average profile axes (top-right)
        ax_pt: Phase-time diagram axes (bottom-right)
        is_last_row: True only for the bottommost data-file row
    """
    if not is_last_row:
        ax_prd.set_xticklabels([])
        ax_prof.set_xticklabels([])
        ax_pt.tick_params(axis="x", length=TICK_LENGTH)
    else:
        ax_prd.set_xlabel("Trial period (s)", fontsize=LABEL_FONTSIZE)
        ax_prof.xaxis.set_visible(False)
        ax_pt.set_xlabel("Phase relative to start MJD", fontsize=LABEL_FONTSIZE)


def candverf_plot(
    period: float,
    bins: int,
    detrended_ts: Sequence[Any],
    periodograms: Sequence[Any],
    annotations: Sequence[str],
    start_mjds: Sequence[float],
    max_snr: float,
    periodaxis_log: bool,
    plot_name: str,
    output_formats: Sequence[str],
    use_latex: bool = False,
) -> None:
    """
    Multi-row candidate verification plot combining periodogram, pulse profile, and phase-time diagram.

    Each row corresponds to one data file. Left panel: periodogram with a dashed vertical
    line at the candidate period. Top-right panel: normalized average pulse profile.
    Bottom-right panel: phase-time diagram with flux density on a grayscale.

    Args:
        period: Candidate folding period (s)
        bins: Number of phase bins in the folded profile
        detrended_ts: Riptide TimeSeries objects, one per data file
        periodograms: Riptide Periodogram objects, one per data file
        annotations: Custom text annotations for each row (e.g., beam or ON/OFF labels)
        start_mjds: Start MJDs (UTC) for each data file
        max_snr: Maximum S/N value shown on the periodogram y-axis
        periodaxis_log: Whether to use a logarithmic period axis
        plot_name: Output file path without extension
        output_formats: File extensions (with leading dot) for each output format
        use_latex: Render text with LaTeX (requires a system LaTeX installation); defaults to False
    """
    if use_latex:
        _require_latex()
    rc = {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"} if use_latex else {}
    with mpl.rc_context(rc):
        n_datafiles = len(detrended_ts)
        fig = plt.figure(figsize=(12, 14))
        outer = gridspec.GridSpec(n_datafiles, 2, figure=fig, height_ratios=list(np.ones(n_datafiles)))
        phasebin_centers = _phasebin_centers(bins)
        for j in range(n_datafiles):
            gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[j, 0])
            ax1 = plt.subplot(gs1[0])
            _configure_periodogram_ax(
                ax1,
                periodograms[j].periods,
                periodograms[j].snrs.max(axis=1),
                annotations[j],
                start_mjds[j],
                period,
                max_snr,
                periodaxis_log,
            )

            phase_time, profile = _fold_and_normalize(detrended_ts[j], period, bins)

            gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[j, 1], height_ratios=[1, 1], hspace=0)
            ax20 = plt.subplot(gs2[0])
            _plot_avg_profile_ax(ax20, phasebin_centers, profile)

            ax21 = plt.subplot(gs2[1], sharex=ax20)
            _plot_phase_time_ax(ax21, phase_time, phasebin_centers, period)

            _apply_row_x_labels(ax1, ax20, ax21, is_last_row=(j == n_datafiles - 1))

        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.98, hspace=0.1, wspace=0.2)
        fig.text(0.013, 0.55, "Matched filtering S/N", va="center", rotation="vertical", fontsize=LABEL_FONTSIZE)
        _save_and_close(plot_name, output_formats)


# ---------------------------------------------------------------------------
# plot_phase_resolved_dynamic_spectrum helpers
# ---------------------------------------------------------------------------


def _plot_freq_averaged_profile(
    ax: Axes,
    phasebin_centers: npt.NDArray[np.floating],
    ds: npt.NDArray[np.floating],
) -> None:
    """
    Plot the frequency-averaged pulse profile in the top-left panel.

    Args:
        ax: Axes on which to draw the profile
        phasebin_centers: Central phase value of each bin
        ds: Phase-resolved dynamic spectrum of shape (Nchans, Nbins)
    """
    ax.plot(phasebin_centers, np.mean(ds, axis=0), "-k")
    ax.set_ylabel(r"$\overline{S}$ (arb. units)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)
    ax.set_xticklabels([])
    ax.set_xlim(0.0, 1.0)


def _plot_phase_resolved_spectrum(
    ax: Axes,
    ds: npt.NDArray[np.floating],
    phasebin_centers: npt.NDArray[np.floating],
    freqs_MHz: npt.NDArray[np.floating],
    start_mjd: float,
) -> None:
    """
    Plot the phase-resolved dynamic spectrum in the bottom-left panel.

    Args:
        ax: Axes on which to draw the spectrum
        ds: Phase-resolved dynamic spectrum of shape (Nchans, Nbins)
        phasebin_centers: Central phase value of each bin
        freqs_MHz: Radio frequencies (MHz) for each channel
        start_mjd: Start MJD (UTC) of the observation, used in the x-axis label
    """
    ax.imshow(
        ds,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="Greys",
        extent=(phasebin_centers[0], phasebin_centers[-1], freqs_MHz[0], freqs_MHz[-1]),
    )
    ax.set_xlabel(f"Phase relative to MJD {start_mjd:.4f} UTC", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Radio frequency (MHz)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABELSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((freqs_MHz[0], freqs_MHz[-1]))


def _plot_band_averaged_spectrum(
    ax: Axes,
    ds: npt.NDArray[np.floating],
    freqs_MHz: npt.NDArray[np.floating],
) -> None:
    """
    Plot the time-averaged band spectrum in the bottom-right panel.

    Args:
        ax: Axes on which to draw the spectrum
        ds: Phase-resolved dynamic spectrum of shape (Nchans, Nbins)
        freqs_MHz: Radio frequencies (MHz) for each channel
    """
    ax.plot(np.mean(ds, axis=1), freqs_MHz, "-k")
    ax.set_xlabel(r"$\langle S_{\nu} \rangle$ (arb. units)", fontsize=LABEL_FONTSIZE)
    ax.set_yticklabels([])
    ax.set_ylim((freqs_MHz[0], freqs_MHz[-1]))
    ax.tick_params(axis="x", labelsize=TICK_LABELSIZE)


def _annotate_period_bins(
    ref_ax: Axes,
    period: float,
    bins: int,
) -> None:
    """
    Add folding period and phase-bin-count annotations to the blank top-right panel.

    Args:
        ref_ax: The blank top-right axes on which to place the text
        period: Folding period (s)
        bins: Number of phase bins
    """
    ref_ax.text(0.05, 0.65, f"$P = {period:.5f}$ s", fontsize=LABEL_FONTSIZE, transform=ref_ax.transAxes)
    ref_ax.text(0.05, 0.35, f"$N_{{\\mathrm{{bins}}}} = {bins:d}$", fontsize=LABEL_FONTSIZE, transform=ref_ax.transAxes)


def plot_phase_resolved_dynamic_spectrum(
    phaseresolved_ds: npt.NDArray[np.floating],
    freqs_MHz: npt.NDArray[np.floating],
    period: float,
    start_mjd: float,
    plot_name: str,
    plot_formats: Sequence[str],
    use_latex: bool = False,
) -> None:
    """
    Produce a grayscale imshow plot of a phase-resolved dynamic spectrum.

    The figure has four panels: frequency-averaged profile (top-left), blank annotation
    panel (top-right), phase-resolved spectrum (bottom-left), and time-averaged band
    spectrum (bottom-right).

    Args:
        phaseresolved_ds: 2D data array of shape (n_chans, n_bins)
        freqs_MHz: Radio frequencies (MHz) corresponding to each channel
        period: Folding period (s)
        start_mjd: Start MJD (UTC) of the observation
        plot_name: Output file path without extension
        plot_formats: File extensions (with leading dot) for each output format
        use_latex: Render text with LaTeX (requires a system LaTeX installation); defaults to False
    """
    if use_latex:
        _require_latex()
    rc = {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"} if use_latex else {}
    with mpl.rc_context(rc):
        bins = len(phaseresolved_ds[0])
        phasebin_centers = _phasebin_centers(bins)
        _, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(8, 7),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [1, 3], "width_ratios": [3, 1]},
        )
        axes[0, 1].axis("off")
        _plot_freq_averaged_profile(axes[0, 0], phasebin_centers, phaseresolved_ds)
        _plot_phase_resolved_spectrum(axes[1, 0], phaseresolved_ds, phasebin_centers, freqs_MHz, start_mjd)
        _plot_band_averaged_spectrum(axes[1, 1], phaseresolved_ds, freqs_MHz)
        _annotate_period_bins(axes[0, 1], period, bins)
        _save_and_close(plot_name, plot_formats)
