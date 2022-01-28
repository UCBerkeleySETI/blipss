# Functions to search for unknown periodicities in data.
from .harmonic_detection import label_harmonics
from riptide import TimeSeries, ffa_search
from tqdm import tqdm
import numpy as np
#########################################################################
def periodic_helper(data, start_ch, tsamp, min_period, max_period, fpmin, bins_min, bins_max, ducy_max, deredden_flag, rmed_width, SNR_threshold, epsilon, mem_load):
    """
    Read in a blimpy data file, execute an FFA search on a per-channel basis, flag harmonics (or sub-harmonics), and output results.

    Parameters
    ----------
    data: 2D Numpy array
        2D array with shape (No. of spectral channels, No. of time samples)

    start_ch: integer
        Channels with index i >= start_ch are searched via FFA.

    tsamp: float
        Sampling time (s)

    min_period: float
        Minimum period (s) of FFA search

    max_period: float
        Maximum period (s) of FFA search

    fpmin: integer
        Minimum number of signal periods that must fit in the data. In other words, place a cap on period_max equal to DATA_LENGTH / fpmin.

    bins_min: integer
        Minimum number of bins across folded profile

    bins_max: integer
        Maximum number of bins across folded profile. Set bins_max 10% larger than bins_min to maintain roughly uniform duty cycle resolution across search.

    ducy_max: float
        Maximum duty cycle searched

    deredden_flag: boolean
        Do you want to detrend input time series with a running median filter?  (True/False)

    rmed_width: float
        Running median window width (s)

    SNR_threshold: float
        S/N threshold of FFA + matched filtering search

    epsilon: float
        Floating point threshold value for harmonic detection

    mem_load: float
        Maximum data size in GB allowed in memory (default: 1 GB)

    Returns
    -------
    cand_chans: 1D Numpy array
        Channel indices at which significant periodic signals were detected

    cand_periods: 1D Numpy array
        Trial periods (s) of detected signals

    cand_snrs: 1D Numpy array
        Maximum matched filtering S/N values returned at detected periods.

    cand_bins: 1D Numpy array
        No. of bins used to produce folded profile of the candidate

    cand_best_widths: 1D Numpy array
        Best trial Boxcar filter widths (no. of phase bins) that yield the greatest matched filtering S/N of folded profiles at respective detected periods

    cand_flags: 1D Numpy array
        Harmonic flags for all detected candidate periods. Harmonics are labeled as 'H', whereas fundamental periods are marked as 'F' in this array.
    """

    # Loop over channels and run FFA search on a per-channel basis.
    cand_chans = np.array([])
    cand_periods = np.array([])
    cand_snrs = np.array([])
    cand_bins = np.array([])
    cand_best_widths = np.array([]) # Best trial width of boxcar filter that matches the FWHM of the folded profile for a given trial period
    cand_flags = np.array([]) # Harmonic flags for every detected candidate period.
    # Use tqddm to track completion progress within "for" loop.
    for ch in tqdm(range(len(data))):
        orig_ts = TimeSeries.from_numpy_array(data[ch], tsamp=tsamp)
        detrended_ts, pgram = ffa_search(orig_ts, period_min=min_period, period_max=max_period, fpmin=fpmin, bins_min=bins_min,
                                         bins_max=bins_max, ducy_max=ducy_max, deredden=deredden_flag, rmed_width=rmed_width, already_normalised=False)
        # pgram.snrs.shape = (No. of trial periods, No. of trial widths)
        mask = pgram.snrs.max(axis=1) >= SNR_threshold
        if True in mask:
            ch_periods = np.array(pgram.periods[mask])
            ch_snrs = np.array(pgram.snrs.max(axis=1)[mask])
            ch_bins = np.array(pgram.foldbins[mask], dtype=int)
            ch_widths = np.array([pgram.widths[i] for i in np.argmax(pgram.snrs, axis = 1)[mask]], dtype=int)

            # Sort arrays in increasing order of folding period for harmonic identification.
            sort_index_order = np.argsort(ch_periods)
            ch_periods = ch_periods[sort_index_order]
            ch_snrs = ch_snrs[sort_index_order]
            ch_bins = ch_bins[sort_index_order]
            ch_widths = ch_widths[sort_index_order]

            # Label harmonics.
            harmonic_flags = label_harmonics(ch_periods, ch_snrs, epsilon, sorted=True)

            # Update grand arrays before execution moves to the next channel.
            cand_chans = np.append(cand_chans, [start_ch+ch]*len(ch_periods))
            cand_periods = np.append(cand_periods, ch_periods)
            cand_snrs = np.append(cand_snrs, ch_snrs)
            cand_bins = np.append(cand_bins, ch_bins)
            cand_best_widths = np.append(cand_best_widths, ch_widths)
            cand_flags = np.append(cand_flags, harmonic_flags)

    # Explicitly, set array types and precision.
    cand_chans = np.array(cand_chans, dtype=int)
    cand_periods = np.round(np.array(cand_periods, dtype=np.float64), 5)
    cand_snrs = np.round(np.array(cand_snrs, dtype=np.float64), 5)
    cand_bins = np.array(cand_bins, dtype=int)
    cand_best_widths = np.array(cand_best_widths, dtype=int)

    return cand_chans, cand_periods, cand_snrs, cand_bins, cand_best_widths, cand_flags
#########################################################################
