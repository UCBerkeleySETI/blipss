# Identify and flag harmonics (or sub-harmonics) in data.

import numpy as np
#########################################################################
def label_harmonics(periods, snrs, epsilon=1.0e-5, sorted=False):
    """
    Search through an array of periods for harmonics.

    Allowed a tolerance threshold epsilon, a period P_N is considered to be the Nth harmonic of P_0 if
    |P_N - N*P_0| <= epsilon.

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
        Are the input periods pre-sorted in ascending order? If True, skip redundant sorting during execution.

    Returns
    ----------
    flags: 1D Numpy array
        Flags assigned to different periods. 'H' for harmonics, and 'F' for the fundamental

    sorted_periods: 1D Numpy array
        Periods (s) arranged in ascending order. Returned only if sorted is False.

    sorted_snrs: 1D Numpy array
        S/N values corresponding to array of sorted periods. Returned only if sorted is False.

    """
    if not sorted:
        sort_idx =  np.argsort(periods)
        periods = periods[sort_idx]
        snrs = snrs[sort_idx]
    # Temporary arrays to store periods and snrs after classification
    temp_periods = np.array([])
    temp_snrs = np.array([])
    # Flags for periods in temp_periods
    harm_flag = np.array([])
    while len(periods) > 0:
        # Max no. of harmonics to check = ROUND(Maximum period / Minimum period)
        Nharm = int(np.round( periods[-1]/ periods[0]))
        idx = np.array([],dtype=int) # Store indices of periods identified as multiples of the minimum period in the array
        for n in range(1,Nharm+1):
            idx = np.append(idx, np.where(np.abs(periods - n*periods[0]) < epsilon)[0])

        # Update temporary arrays.
        temp_periods = np.append(temp_periods, periods[idx])
        temp_snrs = np.append(temp_snrs, snrs[idx])
        flags = ['H']*len(idx)
        # Label the preriod with the highest S/N as the fundamental.
        flags[np.argmax(snrs[idx])] = 'F'
        harm_flag = np.append(harm_flag, flags)

        # Delete harmomics of the minimum period from original arrays.
        periods = np.delete(periods, idx)
        snrs = np.delete(snrs, idx)
    # Resort arrays in order of increasing period.
    sort_idx = np.argsort(temp_periods)
    sorted_periods = temp_periods[sort_idx]
    sorted_snrs = temp_snrs[sort_idx]
    harm_flag = harm_flag[sort_idx]
    if sorted:
        return harm_flag
    else:
        return harm_flag, sorted_periods, sorted_snrs
#########################################################################
