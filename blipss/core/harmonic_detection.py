"""
Utilities for identifying and labeling harmonic and sub-harmonic relationships
among a set of candidate periods.

Terminology
-----------
Given a fundamental period p0:

- The (N-1)th **harmonic** of p0 is a period P such that ``P ≈ p0 / N``  (N >= 2).
- The (N-1)th **sub-harmonic** of p0 is a period P such that ``P ≈ N * p0``  (N >= 2).

Algorithm
---------
:func:`label_harmonics` uses a greedy approach: the unlabeled period with the
highest S/N is selected as a fundamental ('F'), all of its harmonics ('H') and
sub-harmonics ('S') present in the input are labeled and removed from
consideration, then the process repeats on the remaining periods until every
period has been assigned a label.
"""

from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from blipss.constants import DEFAULT_EPSILON_HARMONIC


def _sort_by_snr(
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Sort periods and S/N values in descending order of S/N.

    Args:
        periods: Array of periods (s)
        snrs: Array of S/N values corresponding to the periods

    Returns:
        Tuple of (sorted_periods, sorted_snrs) in descending S/N order
    """
    sort_idx = np.argsort(-snrs)
    return periods[sort_idx], snrs[sort_idx]


def _find_subharmonic_indices(
    periods: npt.NDArray[np.floating],
    p0: float,
    epsilon_harmonic: float,
) -> npt.NDArray[np.intp]:
    """
    Find indices of periods that are sub-harmonics of p0.

    A period P is the (N-1)th sub-harmonic of p0 if ``|P - N*p0| <= epsilon_harmonic``
    for integer N >= 2.

    Index 0 (p0 itself) is always prepended to the result. This is a contract
    relied upon by :func:`label_harmonics`, which constructs the 'F' flag by
    assuming ``idx_subharm[0]`` always corresponds to p0.

    Args:
        periods: Candidate periods (s); must be non-empty, with p0 at index 0
        p0: Fundamental period (s)
        epsilon_harmonic: Tolerance for harmonic period matching

    Returns:
        Indices into periods of all sub-harmonics of p0, with p0's own index (0) first
    """
    # Upper bound on sub-harmonic order N: the largest period in the array
    # cannot exceed N*p0, so N <= max(periods) / p0.
    n_subharm = int(np.round(np.max(periods) / p0))
    ns = np.arange(2, n_subharm + 1)
    if len(ns) == 0:
        return np.array([0], dtype=int)
    targets = ns[:, None] * p0
    matches = np.abs(periods[None, :] - targets) <= epsilon_harmonic
    extra = np.where(np.any(matches, axis=0))[0]
    return np.concatenate([[0], extra])


def _find_harmonic_indices(
    periods: npt.NDArray[np.floating],
    p0: float,
    epsilon_harmonic: float,
) -> npt.NDArray[np.intp]:
    """
    Find indices of periods that are harmonics of p0.

    A period P is the (N-1)th harmonic of p0 if ``|P - p0/N| <= epsilon_harmonic``
    for integer N >= 2.

    Args:
        periods: Candidate periods (s); must be non-empty
        p0: Fundamental period (s)
        epsilon_harmonic: Tolerance for harmonic period matching

    Returns:
        Indices into periods of all harmonics of p0 (p0 itself is not included)
    """
    # Upper bound on harmonic order N: the smallest period in the array
    # cannot be shorter than p0/N, so N <= p0 / min(periods).
    n_harm = int(np.round(p0 / np.min(periods)))
    ns = np.arange(2, n_harm + 1)
    if len(ns) == 0:
        return np.array([], dtype=int)
    targets = p0 / ns[:, None]
    matches = np.abs(periods[None, :] - targets) <= epsilon_harmonic
    return np.where(np.any(matches, axis=0))[0]


@overload
def label_harmonics(
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    epsilon_harmonic: float = ...,
    presorted: Literal[False] = ...,
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...


@overload
def label_harmonics(
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    epsilon_harmonic: float = ...,
    *,
    presorted: Literal[True],
) -> npt.NDArray[np.str_]: ...


def label_harmonics(
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    epsilon_harmonic: float = DEFAULT_EPSILON_HARMONIC,
    presorted: bool = False,
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.floating], npt.NDArray[np.floating]] | npt.NDArray[np.str_]:
    """
    Assign harmonic labels to an array of candidate periods.

    Uses a greedy, highest-S/N-first algorithm (see module docstring). Each
    iteration selects the unlabeled period with the highest S/N as a
    fundamental, labels its harmonics and sub-harmonics, removes all of them
    from the active set, then repeats until no unlabeled periods remain.

    A period P is the (N-1)th harmonic of p0 if ``|P - p0/N| <= epsilon_harmonic``.
    A period P is the (N-1)th sub-harmonic of p0 if ``|P - N*p0| <= epsilon_harmonic``.

    Args:
        periods: Periods (s)
        snrs: S/N values associated with the above periods
        epsilon_harmonic: Floating-point tolerance for harmonic period matching
        presorted: Set to True if periods are already sorted in descending S/N
            order. Skips the sort and returns only the flags array, which is
            useful when calling in a tight loop where the sort has already been
            done.

    Returns:
        If presorted is False: tuple of (flags, sorted_periods, sorted_snrs)
        where periods and S/N arrays are in descending S/N order.

        If presorted is True: the flags array only (no copy of the sorted
        input arrays is returned).

        Flags are single characters: 'F' (fundamental), 'H' (harmonic), or
        'S' (sub-harmonic).
    """
    if not presorted:
        periods, snrs = _sort_by_snr(periods, snrs)

    temp_periods: list[npt.NDArray[np.floating]] = []
    temp_snrs: list[npt.NDArray[np.floating]] = []
    harm_flag: list[str] = []

    active = np.ones(len(periods), dtype=bool)

    while np.any(active):
        active_idx = np.where(active)[0]
        rem_p = periods[active_idx]

        p0 = float(rem_p[0])
        idx_subharm = _find_subharmonic_indices(rem_p, p0, epsilon_harmonic)
        idx_harm = _find_harmonic_indices(rem_p, p0, epsilon_harmonic)

        # idx_subharm and idx_harm are indices into rem_p (the active subset);
        # map them back to the original periods array via active_idx.
        local_idx = np.concatenate([idx_subharm, idx_harm])
        global_idx = active_idx[local_idx]

        temp_periods.append(periods[global_idx])
        temp_snrs.append(snrs[global_idx])

        # _find_subharmonic_indices guarantees idx_subharm[0] is p0 itself,
        # so the flag list always starts with exactly one 'F'.
        flags: list[str] = ["F"] + ["S"] * (len(idx_subharm) - 1) + ["H"] * len(idx_harm)
        harm_flag.extend(flags)

        active[global_idx] = False

    if not temp_periods:
        empty: npt.NDArray[np.str_] = np.array([], dtype="<U1")
        if presorted:
            return empty
        return empty, np.array([]), np.array([])

    all_periods = np.concatenate(temp_periods)
    all_snrs = np.concatenate(temp_snrs)
    harm_flag_arr: npt.NDArray[np.str_] = np.array(harm_flag)

    sort_idx = np.argsort(-all_snrs)
    sorted_periods = all_periods[sort_idx]
    sorted_snrs = all_snrs[sort_idx]
    harm_flag_arr = harm_flag_arr[sort_idx]

    if presorted:
        return harm_flag_arr
    return harm_flag_arr, sorted_periods, sorted_snrs
