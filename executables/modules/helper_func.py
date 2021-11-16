# Helper functions to perform various tasks.
from read_data import read_blimpy_file
from riptide import TimeSeries, ffa_search
import numpy as np
#########################################################################
#
'''
Inputs:
datafile =
start_ch =
stop_ch =

'''
def execute_blipss(datafile, start_ch, stop_ch, ):
    # Read in data.
    data, header = read_blimpy_file(datafile)
    # Gather relevant metadata.
    nchans, tsamp = header['nchans'], header['tsamp']
    radio_freqs = header['fch1'] + np.arange(nchans)*header['foff']

    # Clip off edge channels.
    if stop_ch==0:
        stop_ch = len(data) - 1
    data = data[start_ch:stop_ch+1]  # Start channel and stop channel included.
    # Revise radio frequency coverage and channel count to reflect properties of the clipped data.
    radio_freqs = radio_freqs[start_ch:stop_ch+1]
    nchans = len(data)

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
