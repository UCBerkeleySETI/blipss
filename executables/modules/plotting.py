# Plotting tools

from riptide import TimeSeries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
# Use non-interactive backend for plotting.
mpl.use('Agg')
# Enable use of LaTeX labels in plots.
plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"})
##########################################################################
# Scatter plot of periodicity candidates in the radio frequency vs. period diagram
def scatterplot_period_radiofreq(cand_periods, cand_freqs, cand_snrs, cand_flags, basename, min_period, max_period, min_freq, max_freq, plot_formats=['.png']):
    """
    Produce a scatter plot of periodicity detections in the radio frequency vs. trial period diagram.
    Candidate S/N values are shown on a color scale.
    Fundamental frequencies are displayed with a circular marker, whereas harmonics are indicated by a plus symbol.

    Parameters
    ----------
    cand_periods: 1D Numpy array
        Trial periods (s) of detected signals

    cand_freqs: 1D Numpy array
        Radio frequencies (MHz) corresponding to candidate detections

    cand_snrs: 1D Numpy array
        Matched filtering S/N values of detected candidates

    cand_flags: 1D Numpy array
        Harmonic flags ('F', 'S', or 'H') assigned to each candidate

    basename: string
        Output plot basename, including output path

    min_period: float
        Minimum trial period (s) shown on x-axis of plot

    max_period: float
        Maximum trial period (s) on x-axiss of plot

    min_freq: float
        Minimum radio frequency (MHz) limit on y-axis of plot

    max_freq: float
        Maximum radio frequency (MHz) shown on y-axis of plot

    plot_formats: list
        List of file formats for saving plot to disk
    """
    # Set up color map.
    cmap = plt.cm.cividis
    norm = mpl.colors.Normalize(vmin = np.min(cand_snrs), vmax = np.max(cand_snrs))
    # Begin plotting.
    fig1 = plt.figure(1, figsize=(7,6))
    marker_count = 0 # Keep track of the number of distinct marker shapes to plot.
    # Plot fundamental frequencies.
    f_idx = np.where(cand_flags=='F')[0]
    if len(f_idx) > 0:
        plt.scatter(x=cand_periods[f_idx], y=cand_freqs[f_idx], c=cand_snrs[f_idx], s=38, marker='o', cmap=cmap, norm=norm, label='Fundamental')
        marker_count += 1
    # Plot sub-harmonics.
    subharm_idx = np.where(cand_flags=='S')[0]
    if len(subharm_idx) > 0:
        plt.scatter(x=cand_periods[subharm_idx], y=cand_freqs[subharm_idx], c=cand_snrs[subharm_idx], s=38, marker='+', cmap=cmap, norm=norm, label='Subharmonic')
        marker_count += 1
    # Plot harmonics.
    harm_idx = np.where(cand_flags=='H')[0]
    if len(harm_idx) > 0:
        plt.scatter(x=cand_periods[harm_idx], y=cand_freqs[harm_idx], c=cand_snrs[harm_idx], s=38, marker='x', cmap=cmap, norm=norm, label='Harmonic')
        marker_count += 1
    # Set up colorbar.
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = norm))
    cbar.set_label('Matched filtering S/N', fontsize=16)
    # Set axes labels.
    plt.xlabel('Trial folding period (s)', fontsize=16)
    plt.ylabel('Radio frequency (MHz)', fontsize=16)
    # Set tick properties.
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Set axes limits.
    plt.gca().set_xlim( (min_period, max_period) )
    plt.gca().set_ylim( (min_freq, max_freq) )
    # Show legend.
    if marker_count>1:
        leg = plt.legend(bbox_to_anchor=(0.0, -0.22, 1.1, 0.1), loc='center', ncol=3, fancybox=True, frameon=True,
                         borderpad=0.3, fontsize=14, handletextpad=0.2, columnspacing=2.0)
        leg.get_frame().set_edgecolor('silver')
    # Set tight layout for figure.
    plt.tight_layout()
    # Save plot to disk.
    for format in plot_formats:
        plt.savefig(basename + format)
    plt.close()
##########################################################################
# Candidate verification plot including periodogram, average pulse profile, and pulse stacks in phase-time plane.
def candverf_plot(period, bins, detrended_ts, periodograms, annotations, start_mjds, max_snr, periodaxis_log, plot_name, output_formats):
    '''
    Each row corresponds to a different data file.
    Left subplot in each row = Periodogram. Vertical dashed red line denotes folding period.
    Top right subplot in each row = Average pulse profile
    Bottom right subplot in each row = Phase-time diagram with flux density on the grayscale

    Parameters
    ----------
    period: float
        Candidate folding period (s)

    bins: integer
       No. of bins desired across folded profile

    detrended_ts: list
        List of riptide TimeSeries objects containing detrended, normalized time series from different data files

    periodograms: list
        List of riptide Periodogram objects precomputed for different data files

    annotations: list
        List of custom plot annotations. For example, indicate beam and ON/OFF labels

    start_mjds: list
        List of start MJDs (UTC) of data sets

    max_snr: float
        Maximum S/N limit to be shown on y-scale of periodogram plot

    periodaxis_log: Boolean
        Do you want to use a log scale for the period axis of the periodogram? (True/False)

    plot_name: string
        Name (including path but excluding extension) of output plot

    output_formats: list
        List of file extensions that specify output plot formats
    '''
    N_datafiles = len(detrended_ts) # No. of data files
    fig = plt.figure(figsize=(12,14))
    # Make outer gridspec.
    outer = gridspec.GridSpec(N_datafiles, 2, figure=fig, height_ratios=list(np.ones(N_datafiles)))
    # 1 row per data file
    for j in range(N_datafiles):
        # Left column = Periodogram
        gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[j,0])
        ax1 = plt.subplot(gs1[0])
        ax1.plot(periodograms[j].periods, periodograms[j].snrs.max(axis=1), '-k')
        ax1.annotate(annotations[j], xy=(0.8,0.82), xycoords='axes fraction', fontsize=14)
        ax1.annotate('Start MJD (UTC) = %.4f'%  (start_mjds[j]), xy=(0.03,0.82), xycoords='axes fraction', fontsize=14)
        # Indicate candidate period with a light red vertical line.
        ax1.axvline(x=period, ymin=0.0, ymax=0.8, color='salmon', linestyle='--')
        # Set common S/N axes limits for all plots in the left column.
        ax1.set_ylim((-0.1, max_snr))
        # Use log-spaced period axis if specified.
        if periodaxis_log:
            ax1.set_xscale('log')
            # Show x-axis labels in decimal form.
            ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: ('${{:.{:1d}f}}$'.format(int(np.maximum(-np.log10(x),0)))).format(x)))
        # Set font size of axes labels.
        ax1.tick_params(axis='x', labelsize=14)
        ax1.tick_params(axis='y', labelsize=14)

        # Right column = Phase-time diagram and average profile
        phase_time = detrended_ts[j].fold(period, bins, subints=None) # Shape = (No. of full periods, No. of bins)
        profile = detrended_ts[j].fold(period, bins, subints=1)
        # Normalize profile to unit maximum.
        profile /= np.max(profile)
        phasebin_centers = (0.5 + np.arange(0, bins))/bins
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[j,1], height_ratios = [1, 1], hspace = 0)
        # Average pulse profile
        ax20 = plt.subplot(gs2[0])
        ax20.plot(phasebin_centers, profile, '-k')
        ax20.set_ylabel(r'$\overline{S}$ (a.u.)', fontsize=16)
        ax20.tick_params(axis='y', labelsize=14)
        # Phase-time diagram
        ax21 = plt.subplot(gs2[1],sharex=ax20)
        ax21.imshow(phase_time, origin='lower', interpolation='nearest', aspect='auto',cmap='Greys',
                    extent=[phasebin_centers[0], phasebin_centers[-1], 0.0, len(phase_time)*period])
        ax21.set_ylabel(r'$t$ (s)', fontsize=16)
        ax21.tick_params(axis='x', labelsize=14)
        ax21.tick_params(axis='y', labelsize=14)
        ax21.set_xlim((0., 1.))
        # Hide xticks for all but the bottom row.
        if j!=(N_datafiles-1):
            ax1.set_xticks([])
            ax20.set_xticks([])
            ax21.set_xticks([])
        else:
            ax1.set_xlabel('Trial period (s)', fontsize=16)
            ax20.xaxis.set_visible(False)
            ax21.set_xlabel('Phase relative to start MJD', fontsize=16)
    #  Figure boundary formatting
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.98, hspace=0.1, wspace=0.2)
    fig.text(0.013, 0.55, 'Matched filtering S/N', va='center', rotation='vertical', fontsize=16)
    for format in output_formats:
        plt.savefig(plot_name+format)
    plt.close()
##########################################################################
# Plot phase-resolved dynamic spectrum.
def plot_phaseds(phaseresolved_ds, freqs_MHz, period, start_MJD, plot_name, plot_formats):
    """
    Outputs a grayscale imshow plot of a phase-resolved spectrum.

    Parameters
    ----------
    phaseresolved_ds: 2D Numpy array
        Data array of shape (Nchans, Nbins)

    freqs_MHz: 1D Numpy array
        Radio frequencies (MHz) corresponding to different channels

    period: float
        Folding period (s)

    start_MJD: float
        Start MJD (UTC) of observation

    plot_name: string
        Output plot basename, including output path

    plot_formats: list
        List of file formats for saving plot to disk
    """
    bins  = len(phaseresolved_ds[0]) # No. of phase bins
    phasebin_centers = (0.5 + np.arange(0, bins))/bins # Central phase of different bins
    # Set up figure
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,7), constrained_layout=True,
                             gridspec_kw={'height_ratios':[1,3], 'width_ratios':[3,1]})
    # Hide axes boundaries of top right plot.
    axes[0,1].axis('off')
    # Top left plot = Frequency-averaged profile
    axes[0,0].plot(phasebin_centers, np.mean(phaseresolved_ds, axis=0), '-k')
    axes[0,0].set_ylabel(r'$\overline{S}$ (arb. units)', fontsize=16)
    axes[0,0].tick_params(axis='y', labelsize=14)
    axes[0,0].set_xticklabels([])
    axes[0,0].set_xlim(0.0, 1.0)
    # Bottom left plot = Phase-resolved spectrum
    axes[1,0].imshow(phaseresolved_ds, aspect='auto', origin='lower', interpolation='nearest', cmap='Greys',
                     extent=[phasebin_centers[0], phasebin_centers[-1], freqs_MHz[0], freqs_MHz[-1]])
    axes[1,0].set_xlabel('Phase relative to MJD %.4f UTC'% (start_MJD), fontsize=16)
    axes[1,0].set_ylabel('Radio frequency (MHz)', fontsize=16)
    axes[1,0].tick_params(axis='x', labelsize=14)
    axes[1,0].tick_params(axis='y', labelsize=14)
    axes[1,0].set_xlim((0.0, 1.0))
    axes[1,0].set_ylim((freqs_MHz[0], freqs_MHz[-1]))
    # Bottom right plot = Profile spectrum
    axes[1,1].plot(np.mean(phaseresolved_ds,axis=1), freqs_MHz, '-k')
    axes[1,1].set_xlabel(r'$\langle S_{\nu} \rangle$ (arb. units)', fontsize=16)
    axes[1,1].set_yticklabels([])
    axes[1,1].set_ylim((freqs_MHz[0], freqs_MHz[-1]))
    axes[1,1].tick_params(axis='x', labelsize=14)
    # Overall figure settings
    fig.text(axes[0,1].get_position().x0+axes[0,1].margins()[0]+0.01, axes[0,1].get_position().y0+axes[0,1].margins()[1]+0.06,
             '$P = %.5f$ s'% (period), fontsize=16)
    fig.text(axes[0,1].get_position().x0+axes[0,1].margins()[0]+0.01, axes[0,1].get_position().y0+axes[0,1].margins()[1]+0.01,
             '$N_{\mathrm{bins}} = %d$'% (bins), fontsize=16)
    # Save plot to disk.
    for format in plot_formats:
        plt.savefig(plot_name+format)
    plt.close()
