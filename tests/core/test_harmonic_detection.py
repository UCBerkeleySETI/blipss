"""Unit tests for modules in blipss.core.harmonic_detection"""

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from blipss.constants import DEFAULT_EPSILON_HARMONIC
from blipss.core.harmonic_detection import (
    _find_harmonic_indices,
    _find_subharmonic_indices,
    _sort_by_snr,
    label_harmonics,
)

# ---------------------------------------------------------------------------
# _sort_by_snr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("periods", "snrs", "expected_periods", "expected_snrs"),
    [
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([5.0, 10.0, 3.0]),
            np.array([2.0, 1.0, 3.0]),
            np.array([10.0, 5.0, 3.0]),
        ),
        (
            np.array([0.5, 1.0, 2.0]),
            np.array([1.0, 3.0, 2.0]),
            np.array([1.0, 2.0, 0.5]),
            np.array([3.0, 2.0, 1.0]),
        ),
    ],
    ids=["highest_snr_in_middle", "highest_snr_first"],
)
def test_sort_by_snr_descending_order(
    periods: npt.NDArray[np.floating],
    snrs: npt.NDArray[np.floating],
    expected_periods: npt.NDArray[np.floating],
    expected_snrs: npt.NDArray[np.floating],
) -> None:
    """_sort_by_snr returns periods and S/N values arranged in descending S/N order."""
    out_periods, out_snrs = _sort_by_snr(periods, snrs)
    np.testing.assert_array_equal(out_periods, expected_periods)
    np.testing.assert_array_equal(out_snrs, expected_snrs)


def test_sort_by_snr_single_element() -> None:
    """_sort_by_snr returns the single-element arrays unchanged."""
    periods = np.array([1.5])
    snrs = np.array([7.0])
    out_periods, out_snrs = _sort_by_snr(periods, snrs)
    np.testing.assert_array_equal(out_periods, periods)
    np.testing.assert_array_equal(out_snrs, snrs)


def test_sort_by_snr_already_sorted_is_unchanged() -> None:
    """_sort_by_snr leaves an already-descending array unchanged."""
    periods = np.array([3.0, 2.0, 1.0])
    snrs = np.array([9.0, 6.0, 3.0])
    out_periods, out_snrs = _sort_by_snr(periods, snrs)
    np.testing.assert_array_equal(out_periods, periods)
    np.testing.assert_array_equal(out_snrs, snrs)


# ---------------------------------------------------------------------------
# _find_subharmonic_indices
# ---------------------------------------------------------------------------


def test_find_subharmonic_indices_only_fundamental_when_p0_is_max() -> None:
    """_find_subharmonic_indices returns only index 0 when no periods exceed p0."""
    periods = np.array([3.0, 1.0, 0.5])
    indices = _find_subharmonic_indices(periods, p0=3.0, epsilon_harmonic=0.01)
    np.testing.assert_array_equal(indices, np.array([0]))


def test_find_subharmonic_indices_finds_exact_multiples() -> None:
    """_find_subharmonic_indices identifies all exact integer multiples of p0."""
    # p0=1.0: 2*p0=2.0 (index 1), 3*p0=3.0 (index 2)
    periods = np.array([1.0, 2.0, 3.0])
    indices = _find_subharmonic_indices(periods, p0=1.0, epsilon_harmonic=0.01)
    np.testing.assert_array_equal(np.sort(indices), np.array([0, 1, 2]))


def test_find_subharmonic_indices_partial_multiples() -> None:
    """_find_subharmonic_indices detects only the period that is an integer multiple of p0."""
    # p0=1.5: 2*1.5=3.0 matches index 2; 1.0 is not a multiple of 1.5
    periods = np.array([1.5, 1.0, 3.0])
    indices = _find_subharmonic_indices(periods, p0=1.5, epsilon_harmonic=0.01)
    assert 0 in indices
    assert 2 in indices
    assert 1 not in indices


@pytest.mark.parametrize(
    ("near_multiple", "epsilon", "should_match"),
    [
        (2.005, 0.01, True),  # |2.005 - 2.0| = 0.005 <= 0.01
        (2.02, 0.01, False),  # |2.02  - 2.0| = 0.02  >  0.01
    ],
    ids=["within_tolerance", "outside_tolerance"],
)
def test_find_subharmonic_indices_epsilon_boundary(near_multiple: float, epsilon: float, should_match: bool) -> None:
    """_find_subharmonic_indices respects the epsilon tolerance boundary for N*p0 matching."""
    periods = np.array([1.0, near_multiple])
    indices = _find_subharmonic_indices(periods, p0=1.0, epsilon_harmonic=epsilon)
    if should_match:
        assert 1 in indices
    else:
        np.testing.assert_array_equal(indices, np.array([0]))


# ---------------------------------------------------------------------------
# _find_harmonic_indices
# ---------------------------------------------------------------------------


def test_find_harmonic_indices_no_harmonics_when_p0_equals_min() -> None:
    """_find_harmonic_indices returns an empty array when p0 is the minimum period."""
    periods = np.array([1.0])
    indices = _find_harmonic_indices(periods, p0=1.0, epsilon_harmonic=0.01)
    np.testing.assert_array_equal(indices, np.array([], dtype=int))


def test_find_harmonic_indices_finds_multiple_fractions() -> None:
    """_find_harmonic_indices detects p0/2 and p0/4 when both are present."""
    periods = np.array([0.25, 0.5, 1.0])
    indices = _find_harmonic_indices(periods, p0=1.0, epsilon_harmonic=0.01)
    assert 0 in indices  # p0/4 = 0.25
    assert 1 in indices  # p0/2 = 0.5


@pytest.mark.parametrize(
    ("near_fraction", "epsilon", "should_match"),
    [
        (0.505, 0.01, True),  # |0.505 - 0.5| = 0.005 <= 0.01
        (0.52, 0.01, False),  # |0.52  - 0.5| = 0.02  >  0.01
    ],
    ids=["within_tolerance", "outside_tolerance"],
)
def test_find_harmonic_indices_epsilon_boundary(near_fraction: float, epsilon: float, should_match: bool) -> None:
    """_find_harmonic_indices respects the epsilon tolerance boundary for p0/N matching."""
    periods = np.array([near_fraction, 1.0])
    indices = _find_harmonic_indices(periods, p0=1.0, epsilon_harmonic=epsilon)
    if should_match:
        assert 0 in indices
    else:
        np.testing.assert_array_equal(indices, np.array([], dtype=int))


# ---------------------------------------------------------------------------
# label_harmonics
# ---------------------------------------------------------------------------


def test_label_harmonics_no_relationships_all_fundamentals() -> None:
    """label_harmonics marks every period as 'F' when none are harmonically related."""
    # 2.3, 1.0, 0.7 share no harmonic relationships within DEFAULT_EPSILON_HARMONIC
    periods = np.array([2.3, 1.0, 0.7])
    snrs = np.array([8.0, 5.0, 3.0])
    flags, _, _ = label_harmonics(periods, snrs)
    assert all(f == "F" for f in flags)


def test_label_harmonics_labels_subharmonic_correctly() -> None:
    """label_harmonics assigns 'S' to a period that is an integer multiple of the fundamental."""
    # 2.0 = 2*1.0; fundamental is 1.0 with the higher S/N
    periods = np.array([1.0, 2.0])
    snrs = np.array([10.0, 3.0])
    flags, sorted_p, _ = label_harmonics(periods, snrs)
    flag_map = dict(zip(sorted_p.tolist(), flags.tolist(), strict=True))
    assert flag_map[1.0] == "F"
    assert flag_map[2.0] == "S"


def test_label_harmonics_labels_harmonic_correctly() -> None:
    """label_harmonics assigns 'H' to a period that is an integer fraction of the fundamental."""
    # 0.5 = 1.0/2; fundamental is 1.0 with the higher S/N
    periods = np.array([1.0, 0.5])
    snrs = np.array([10.0, 3.0])
    flags, sorted_p, _ = label_harmonics(periods, snrs)
    flag_map = dict(zip(sorted_p.tolist(), flags.tolist(), strict=True))
    assert flag_map[1.0] == "F"
    assert flag_map[0.5] == "H"


def test_label_harmonics_mixed_family_contains_all_labels() -> None:
    """label_harmonics produces 'F', 'S', and 'H' labels when both types appear in one family."""
    # 1.0 (highest S/N) is fundamental; 2.0 is its sub-harmonic; 0.5 is its harmonic
    periods = np.array([1.0, 2.0, 0.5])
    snrs = np.array([10.0, 5.0, 3.0])
    flags, sorted_p, _ = label_harmonics(periods, snrs)
    flag_map = dict(zip(sorted_p.tolist(), flags.tolist(), strict=True))
    assert flag_map[1.0] == "F"
    assert flag_map[2.0] == "S"
    assert flag_map[0.5] == "H"


def test_label_harmonics_highest_snr_in_family_is_fundamental() -> None:
    """label_harmonics designates the highest-S/N member of a harmonic family as 'F'."""
    # 0.5 has the highest S/N; 1.0 and 2.0 are its sub-harmonics
    periods = np.array([1.0, 2.0, 0.5])
    snrs = np.array([3.0, 1.0, 10.0])
    flags, sorted_p, sorted_s = label_harmonics(periods, snrs)
    assert sorted_p[0] == pytest.approx(0.5)
    assert sorted_s[0] == pytest.approx(10.0)
    assert flags[0] == "F"
    flag_map = dict(zip(sorted_p.tolist(), flags.tolist(), strict=True))
    assert flag_map[1.0] == "S"
    assert flag_map[2.0] == "S"


def test_label_harmonics_output_sorted_by_descending_snr() -> None:
    """label_harmonics output S/N array is in non-increasing order."""
    periods = np.array([0.5, 1.0, 2.0, 3.7])
    snrs = np.array([2.0, 8.0, 4.0, 6.0])
    _, _, sorted_s = label_harmonics(periods, snrs)
    assert np.all(sorted_s[:-1] >= sorted_s[1:])


def test_label_harmonics_two_independent_families() -> None:
    """label_harmonics correctly classifies members of two unrelated harmonic families."""
    # Family A: p=1.0 (F, snr=10), p=0.5 (H, snr=3)
    # Family B: p=0.3 (F, snr=7), p=0.6 (S, snr=2)
    periods = np.array([1.0, 0.5, 0.3, 0.6])
    snrs = np.array([10.0, 3.0, 7.0, 2.0])
    flags, sorted_p, _ = label_harmonics(periods, snrs)
    flag_map = dict(zip(sorted_p.tolist(), flags.tolist(), strict=True))
    assert flag_map[1.0] == "F"
    assert flag_map[0.3] == "F"
    assert flag_map[0.5] == "H"
    assert flag_map[0.6] == "S"


def test_label_harmonics_presorted_returns_flags_array_only() -> None:
    """label_harmonics with presorted=True returns only the flags ndarray, not a tuple."""
    # Input is already in descending S/N order
    periods = np.array([1.0, 2.0, 0.5])
    snrs = np.array([10.0, 5.0, 3.0])
    result = label_harmonics(periods, snrs, presorted=True)
    assert isinstance(result, np.ndarray)
    assert result.dtype.kind == "U"


def test_label_harmonics_presorted_flags_consistent_with_unsorted() -> None:
    """label_harmonics produces identical flags whether or not the caller pre-sorts."""
    periods = np.array([1.0, 2.0, 0.5])
    snrs = np.array([10.0, 5.0, 3.0])
    flags_unsorted, sorted_p, sorted_s = label_harmonics(periods, snrs, presorted=False)
    flags_presorted = label_harmonics(sorted_p, sorted_s, presorted=True)
    np.testing.assert_array_equal(flags_unsorted, flags_presorted)


def test_label_harmonics_custom_epsilon_detects_near_match() -> None:
    """label_harmonics with a wider epsilon identifies a period slightly off the exact harmonic."""
    # 0.51 is 0.01 away from p0/2=0.5; within epsilon=0.02 but outside DEFAULT_EPSILON_HARMONIC=0.001
    periods = np.array([1.0, 0.51])
    snrs = np.array([10.0, 3.0])
    flags_narrow, _, _ = label_harmonics(periods, snrs, epsilon_harmonic=DEFAULT_EPSILON_HARMONIC)
    flags_wide, _, _ = label_harmonics(periods, snrs, epsilon_harmonic=0.02)
    assert "H" not in flags_narrow
    assert "H" in flags_wide


def test_label_harmonics_empty_input_returns_empty_arrays() -> None:
    """label_harmonics returns empty arrays when given empty period and S/N inputs."""
    periods: npt.NDArray[np.floating] = np.array([], dtype=float)
    snrs: npt.NDArray[np.floating] = np.array([], dtype=float)
    flags, sorted_p, sorted_s = label_harmonics(periods, snrs)
    assert len(flags) == 0
    assert len(sorted_p) == 0
    assert len(sorted_s) == 0


def test_label_harmonics_empty_presorted_returns_empty_flags() -> None:
    """label_harmonics with presorted=True and empty input returns an empty flags array."""
    periods: npt.NDArray[np.floating] = np.array([], dtype=float)
    snrs: npt.NDArray[np.floating] = np.array([], dtype=float)
    result = label_harmonics(periods, snrs, presorted=True)
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


@given(st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False))
def test_label_harmonics_single_period_always_fundamental(period: float) -> None:
    """label_harmonics always assigns 'F' to a single-period input, for any period value."""
    periods = np.array([period])
    snrs = np.array([1.0])
    flags, _, _ = label_harmonics(periods, snrs)
    assert flags[0] == "F"
