# Helper functions to perform various tasks.
from .read_data import read_watfile
from riptide import TimeSeries, ffa_search
from tqdm import tqdm
import numpy as np
#########################################################################
def periodic_helper(datafile, start_ch, stop_ch, min_period, max_period, fpmin, bins_min, bins_max, ducy_max, deredden_flag, rmed_width, SNR_threshold, mem_load, return_radiofreq_limits=True):
    """
    Read in a blimpy data file, execute an FFA search on a per-channel basis, and output results.

    Parameters
    ----------
    datafile : string
         Name of data file to load

    start_ch: integer
         Channels with index i >= start_ch are searched via FFA.

    stop_ch: integer
         Channels with index i < stop_ch are searched via FFA.

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

    mem_load: float
         Maximum data size in GB allowed in memory (default: 1 GB)

    return_radiofreq_limits: boolean
         Do you want to return the radio frequency limits (MHz) of the searched data?

    Returns
    -------
    select_chans: 1D Numpy array
         Channel indices at which significant periodic signals were detected

    select_radiofreqs: 1D Numpy array
         Radio frequencies (MHz) at which significant periodic signals were detected

    periods: 1D Numpy array
         Trial periods (s) of detected signals

    snrs: 1D Numpy array
         Maximum matched filtering S/N values returned at detected periods.

    best_widths: 1D Numpy array
         Best trial Boxcar filter widths (no. of phase bins) that yield the greatest matched filtering S/N of folded profiles at respective detected periods

    min_radiofreq: float
         Low radio frequency limit (MHz) of FFA-searched data. Value returned only if return_radiofreq_limits=True.

    max_radiofreq: float
         High radio frequency limit (MHz) of FFA-searched data. Value returned only if return_radiofreq_limits=True.
    """
    # Read in datafile contents.
    wat = read_watfile(datafile, mem_load)
    # Gather relevant metadata.
    nchans = wat.header['nchans'] # No. of spectral channels
    tsamp = wat.header['tsamp'] # Sampling time (s)
    # 1D array of radio frequencies (MHz)
    freqs_MHz = wat.header['fch1'] + np.arange(nchans)*wat.header['foff']

    # Clip off edge channels.
    if stop_ch is None:
        stop_ch = len(data)
    # Start channel included, stop channel excluded.
    data = wat.data[:,0,start_ch:stop_ch].T # data.shape = (nchans, nsamples)
    # Revise radio frequency coverage and channel count to reflect properties of the clipped data.
    nchans = len(data)
    freqs_MHz = freqs_MHz[start_ch:stop_ch]
    min_radiofreq = np.min(freqs_MHz) # Low radio frequency (MHz) limit of FFA-searched data
    max_radiofreq = np.max(freqs_MHz) # High radio frequency (MHz) limit of FFA-searched data

    # Loop over channels and run FFA search on a per-channel basis.
    select_chans = np.array([])
    periods = np.array([])
    snrs = np.array([])
    best_widths = np.array([]) # Best trial width of boxcar filter that matches the FWHM of the folded profile for a given trial period
    # Use tqddm to track completion progress within "for" loop.
    for ch in tqdm(range(nchans)):
        orig_ts = TimeSeries.from_numpy_array(data[ch], tsamp=tsamp)
        detrended_ts, pgram = ffa_search(orig_ts, period_min=min_period, period_max=max_period, fpmin=fpmin, bins_min=bins_min,
                                         bins_max=bins_max, ducy_max=ducy_max, deredden=deredden_flag, rmed_width=rmed_width, already_normalised=False)
        # pgram.shape = (No. of trial periods, No. of trial widths)
        mask = pgram.snrs.max(axis=1) >= SNR_threshold
        if True in mask:
            ch_periods = list(pgram.periods[mask])
            ch_snrs = list(pgram.snrs.max(axis=1)[mask])
            ch_widths = [pgram.widths[i] for i in np.argmax(pgram.snrs, axis = 1)[mask]]

            select_chans = np.append(select_chans, [start_ch+ch]*len(ch_periods))
            periods = np.append(periods, ch_periods)
            snrs = np.append(snrs, ch_snrs)
            best_widths = np.append(best_widths, ch_widths)

    # Set array types.
    select_chans = np.array(select_chans, dtype=int)
    periods = np.array(periods, dtype=np.float64)
    snrs = np.array(snrs, dtype=np.float64)
    best_widths = np.array(best_widths, dtype=int)
    select_radiofreqs = freqs_MHz[select_chans-start_ch]

    if return_radiofreq_limits:
        return select_chans, select_radiofreqs, periods, snrs, best_widths, min_radiofreq, max_radiofreq
    else:
        return select_chans, select_radiofreqs, periods, snrs, best_widths
#########################################################################
