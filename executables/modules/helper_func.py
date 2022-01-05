# Helper functions to perform various tasks.
from read_data import read_watfile
from riptide import TimeSeries, ffa_search
import numpy as np
#########################################################################
def periodic_helper(datafile, start_ch, stop_ch, min_period, max_period, fpmin, bins_min, bins_max, ducy_max, deredden_flag, rmed_width, SNR_threshold, mem_load):
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

    Returns
    -------
    select_chans: List
         List of channel indices in which significant periodic signals were detected

    select_radiofreqs: List
         List of radio frequencies (GHz) at which significant periodic signals were detected

    periods: List
         List of periods (s) detected in above channels

    width: List
         List of Boxcar filter widths (no. of phase bins) yielding greatest S/N in matched filtering detection of folded profile at detected periods

    snrs: List
         List of matched filtering S/N values returned at detected periods.
    """
    # Read in datafile contents.
    wat = read_watfile(datafile, mem_load)
    # Gather relevant metadata.
    nchans = wat.header['nchans'] # No. of spectral channels
    tsamp = wat.header['tsamp'] # Sampling time (s)
    # 1D array of radio frequencies (GHz)
    freqs_GHz = (wat.header['fch1'] + np.arange(nchans)*wat.header['foff'])*1.e-3

    # Clip off edge channels.
    if stop_ch is None:
        stop_ch = len(data)
    # Start channel included, stop channel excluded.
    data = wat.data[:,0,start_ch:stop_ch].T # data.shape = (nchans, nsamples)
    # Revise radio frequency coverage and channel count to reflect properties of the clipped data.
    nchans = len(data)
    freqs_GHz = freqs_GHz[start_ch:stop_ch]
    min_radiofreq = freqs_GHz[0]
    max_radiofreq = freqs_GHz[-1]

    # Loop over channels and run FFA search on a per-channel basis.
    select_chans = []
    select_radiofreqs = []
    periods = []
    snrs = []
    widths = []
    for ch in range(nchans):
        print(ch)
        orig_ts = TimeSeries.from_numpy_array(data[ch], tsamp=tsamp)
        detrended_ts, pgram = ffa_search(orig_ts, period_min=min_period, period_max=max_period, fpmin=fpmin, bins_min=bins_min,
                                         bins_max=bins_max, ducy_max=ducy_max, deredden=deredden_flag, rmed_width=rmed_width, already_normalised=False)
        # pgram.shape = (No. of trial periods, No. of trial widths)
        mask = pgram.snrs.max(axis=1) >= SNR_threshold
        if True in mask:
            ch_periods = list(pgram.periods[mask])
            ch_snrs = list(pgram.snrs.max(axis=1)[mask])
            ch_widths = [pgram.widths[i] for i in np.argmax(pgram.snrs, axis = 1)[mask]]

            select_chans.append(start_ch+ch)
            select_radiofreqs.append(freqs_GHz[ch])
            periods.append(ch_periods)
            widths.append(ch_widths)
            snrs.append(ch_snrs)

    return select_chans, select_radiofreqs, periods, widths, snrs
#########################################################################
