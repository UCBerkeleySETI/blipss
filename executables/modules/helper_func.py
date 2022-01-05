# Helper functions to perform various tasks.
from read_data import read_watfile
from riptide import TimeSeries, ffa_search
import numpy as np
#########################################################################

def periodic_helper(datafile, start_ch, stop_ch, min_period, max_period, bins_min, bins_max, ducy_max, deredden_flag, rmed_width, mem_load):
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

    mem_load: float
         Maximum data size in GB allowed in memory (default: 1 GB)

    Returns
    -------
    ts: class object
        Riptide TimeSeries object containing detrended, normalized time series data
    """
    # Read in datafile contents.
    wat = read_watfile(datafile, mem_loads)
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


    #





#########################################################################
# Run FFA on data from a blimpy Waterfall object.
'''
Inputs:
data = 2D data array of shape (No.of channels, No. of time samples)
frequency = 1D array of radio frequencies (GHz)
tsamp = Sampling time (s)
SNR_threshold = S/N threshold applied for matched filtering via riptide
'''
def periodic_helper(data, frequency, tsamp, SNR_threshold):

    periods = []
    frequencies = []
    snrs = []
    best_periods = []

    time_series = TimeSeries.from_numpy_array(data, tsamp = tsamp)
    if on and simulate:
        time_series = time_series.normalise()
        if frequency in injection:
            fts = TimeSeries.generate(length=len(data)*tsamp, tsamp=tsamp, period=5.0, ducy=0.02, amplitude=100.0).normalise()
            time_series = TimeSeries.from_numpy_array(time_series.data + fts.data, tsamp = tsamp).normalise()
    ts, pgram = ffa_search(time_series, rmed_width=4.0, period_min=period_range[0], period_max=period_range[1], bins_min=2, bins_max=260)
    mask = pgram.snrs.max(axis = 1) >= cutoff
    periods = pgram.periods[mask]

    if not on:
        return periods
    else:
        frequencies = np.ones(len(pgram.periods[mask])) * frequency
        snrs = pgram.snrs.max(axis = 1)[mask]
        best_period = pgram.periods[np.argmax(pgram.snrs.max(axis=1))]
        best_widths = [pgram.widths[i] for i in np.argmax(pgram.snrs, axis = 1)[mask]]
        min_widths = [min(pgram.widths) for w in best_widths]
        max_widths = [max(pgram.widths) for w in best_widths]
        return periods, frequencies, snrs, best_period, best_widths, min_widths, max_widths
#########################################################################
