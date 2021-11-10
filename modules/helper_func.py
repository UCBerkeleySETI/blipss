import numpy as np


# Run FFA on data from a blimpy Waterfall object.
def periodic_helper(data, frequency, tsamp, cutoff):

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

        
