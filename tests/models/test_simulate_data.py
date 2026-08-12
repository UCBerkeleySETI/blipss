"""Unit tests for config models in blipss.models.simulate_data"""

import pytest
from pydantic import ValidationError

from blipss.models.simulate_data import PeriodicSignalInjection


def test_injection_lists_of_unequal_length_rejected() -> None:
    """PeriodicSignalInjection rejects injection lists whose lengths do not all match."""
    with pytest.raises(ValidationError, match="must have the same length"):
        PeriodicSignalInjection(
            inject_channels=[1, 2],
            periods=[10.0],
            duty_cycles=[0.1, 0.1],
            pulse_snr=[8.0, 8.0],
            initial_phase=[0.0, 0.0],
        )


def test_injection_lists_of_equal_length_accepted() -> None:
    """PeriodicSignalInjection accepts injection lists that all share the same length."""
    cfg = PeriodicSignalInjection(
        inject_channels=[1, 2],
        periods=[10.0, 20.0],
        duty_cycles=[0.1, 1.0],
        pulse_snr=[8.0, 9.0],
        initial_phase=[0.0, 0.999],
    )
    assert cfg.inject_channels == [1, 2]


def test_empty_injection_lists_accepted() -> None:
    """PeriodicSignalInjection defaults to empty lists, representing no injected signals."""
    cfg = PeriodicSignalInjection()
    assert cfg.inject_channels == []
    assert cfg.periods == []


@pytest.mark.parametrize("duty_cycle", [0.0, -0.1, 1.5], ids=["zero", "negative", "above_one"])
def test_duty_cycle_out_of_range_rejected(duty_cycle: float) -> None:
    """PeriodicSignalInjection rejects a duty cycle outside the half-open interval (0, 1]."""
    with pytest.raises(ValidationError, match=r"must be in \(0, 1\]"):
        PeriodicSignalInjection(
            inject_channels=[1],
            periods=[10.0],
            duty_cycles=[duty_cycle],
            pulse_snr=[8.0],
            initial_phase=[0.0],
        )


@pytest.mark.parametrize("initial_phase", [1.0, -0.1, 2.5], ids=["one", "negative", "above_one"])
def test_initial_phase_out_of_range_rejected(initial_phase: float) -> None:
    """PeriodicSignalInjection rejects an initial phase outside the half-open interval [0, 1)."""
    with pytest.raises(ValidationError, match=r"must be in \[0, 1\)"):
        PeriodicSignalInjection(
            inject_channels=[1],
            periods=[10.0],
            duty_cycles=[0.1],
            pulse_snr=[8.0],
            initial_phase=[initial_phase],
        )
