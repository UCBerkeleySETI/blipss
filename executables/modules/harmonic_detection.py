# Identify and flag harmonics (or sub-harmonics) in data.

import numpy as np
#########################################################################
def label_harmonics(periods, snrs, epsilon=1.0e-3, sorted=False):
    """
    Search through an array of periods for harmonics and sub-harmonics.
    Allowed a tolerance threshold epsilon, a period P is considered to be the (N-1)th harmonic of P0 if
    |P - P0/N| <= epsilon.

    Similarly, a period P is taken to be the (N-1)th sub-harmonic of P0 if
    |P - N*P0| <= epsilon

    Given a set of harmonically related periods, the period with the highest S/N is labeled as the fundamental.

    Parameters
    ----------
    periods: 1D Numpy array
        Periods (s)
    snrs: 1D Numpy array
        S/N values associated with the above periods
    epsilon: float
        Floating-point tolerance limit
    sorted: boolean
        Are the input periods pre-sorted in descending order of S/N? If True, skip redundant sorting during execution.
    Returns
    ----------
    flags: 1D Numpy array
        Flags assigned to different periods. 'F' for the fundamental, 'H' for harmonics, and 'S' for sub-harmonics.
    sorted_periods: 1D Numpy array
        Periods (s) arranged in descending order of S/N. Returned only if sorted = False.
    sorted_snrs: 1D Numpy array
        S/N values corresponding to array of sorted periods. Returned only if sorted = False.
    """
    if not sorted:
        sort_idx =  np.argsort(snrs)[::-1]
        periods = periods[sort_idx]
        snrs = snrs[sort_idx]
    # Temporary arrays to store periods and snrs after classification
    temp_periods = np.array([])
    temp_snrs = np.array([])
    # Flags for periods in temp_periods
    harm_flag = np.array([])
    while len(periods) > 0:
        # Start off at the period with highest S/N.
        P0 = periods[0]
        # Max no. of sub-harmonics to check = ROUND(Maximum period / P0)
        N_subharm = int(np.round(np.max(periods)/ P0))
        # Max no. of harmonics to check = ROUND(P0/Minimum period)
        N_harm = int(np.round(P0/np.min(periods)))
        # Store indices of periods identified as sub-harmonics of P0. Also, include index of P0.
        idx_subharm = np.array([0], dtype=int)
        for n in range(2,N_subharm+1):
            idx_subharm = np.append(idx_subharm, np.where(np.abs(periods - n*P0) <= epsilon)[0])
        # Store indices of periods identified as harmonics of P0.
        idx_harm = np.array([], dtype=int)
        for n in range(2,N_harm+1):
            idx_harm = np.append(idx_harm, np.where(np.abs(periods - P0/n) <= epsilon)[0])
        # Gather all indices.
        idx = np.append(idx_subharm, idx_harm)
        # Update temporary arrays.
        temp_periods = np.append(temp_periods, periods[idx])
        temp_snrs = np.append(temp_snrs, snrs[idx])
        # Assign harmonic flags.
        flags = ['F'] + ['S']*(len(idx_subharm)-1) + ['H']*len(idx_harm)
        harm_flag = np.append(harm_flag, flags)
        # Delete P0, its harmonics, and its sub-harmonics from original arrays.
        periods = np.delete(periods, idx)
        snrs = np.delete(snrs, idx)
    # Resort arrays in order of increasing period.
    sort_idx = np.argsort(temp_snrs)[::-1]
    sorted_periods = temp_periods[sort_idx]
    sorted_snrs = temp_snrs[sort_idx]
    harm_flag = harm_flag[sort_idx]
    # Return arrays.
    if sorted:
        return harm_flag
    else:
        return harm_flag, sorted_periods, sorted_snrs
#########################################################################
