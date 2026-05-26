"""Unit tests for the simulate orchestrator in blipss.cli.simulate_data"""

import tempfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import numpy.testing
import pytest

from blipss.cli.simulate_data import simulate
from blipss.models.simulate_data import (
    OptionalHeaderParameters,
    OutputConfig,
    PeriodicSignalInjection,
    SimulateDataConfig,
    SimulationProperties,
)

_MODULE = "blipss.cli.simulate_data"


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _make_config(
    *,
    n_samples: int = 16,
    n_channels: int = 4,
    t_samp: float = 0.25,
    fch1: float = 1500.0,
    foff: float = -0.001,
    inject_channels: list[int] | None = None,
    periods: list[float] | None = None,
    duty_cycles: list[float] | None = None,
    pulse_snr: list[float] | None = None,
    initial_phase: list[float] | None = None,
    output_dir: Path | None = None,
    basename: str = "test_output",
    source_name: str = "Unknown",
    tstart: float = 0.0,
) -> SimulateDataConfig:
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "blipss_tests"
    inject_channels = inject_channels or []
    periods = periods or []
    duty_cycles = duty_cycles or []
    pulse_snr = pulse_snr or []
    initial_phase = initial_phase or []
    return SimulateDataConfig(
        output=OutputConfig(basename=basename, output_dir=output_dir),
        simulation_properties=SimulationProperties(
            n_samples=n_samples,
            n_channels=n_channels,
            t_samp=t_samp,
            foff=foff,
            fch1=fch1,
        ),
        periodic_signal_injection=PeriodicSignalInjection(
            inject_channels=inject_channels,
            periods=periods,
            duty_cycles=duty_cycles,
            pulse_snr=pulse_snr,
            initial_phase=initial_phase,
        ),
        optional_header_parameters=OptionalHeaderParameters(source_name=source_name, tstart=tstart),
    )


# ---------------------------------------------------------------------------
# generate_white_noise_background wiring
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.write_filterbank")
@patch(f"{_MODULE}.build_sigproc_header")
@patch(f"{_MODULE}.reshape_for_sigproc")
@patch(f"{_MODULE}.inject_periodic_signal")
@patch(f"{_MODULE}.generate_white_noise_background")
def test_simulate_noise_generation_uses_n_channels_and_n_samples(
    mock_gwn: MagicMock,
    mock_inject: MagicMock,
    mock_reshape: MagicMock,
    mock_build_header: MagicMock,
    mock_write: MagicMock,
) -> None:
    """simulate calls generate_white_noise_background with (n_channels, n_samples) from config."""
    cfg = _make_config(n_channels=8, n_samples=32)
    simulate(cfg)
    mock_gwn.assert_called_once_with(8, 32, rng=ANY)
    mock_inject.assert_not_called()
    mock_reshape.assert_called_once()
    mock_build_header.assert_called_once()
    mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# inject_periodic_signal call-count — all three loop-path branches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inject_channels,periods,duty_cycles,pulse_snr,initial_phase",
    [
        ([], [], [], [], []),
        ([2], [1.0], [0.5], [5.0], [0.0]),
        ([0, 1, 3], [0.5, 1.0, 2.0], [0.1, 0.2, 0.3], [3.0, 4.0, 5.0], [0.0, 0.1, 0.2]),
    ],
    ids=["no-injection", "single-injection", "multi-injection"],
)
@patch(f"{_MODULE}.write_filterbank")
@patch(f"{_MODULE}.build_sigproc_header")
@patch(f"{_MODULE}.reshape_for_sigproc")
@patch(f"{_MODULE}.inject_periodic_signal")
@patch(f"{_MODULE}.generate_white_noise_background")
def test_simulate_inject_call_count_matches_channel_count(
    mock_gwn: MagicMock,
    mock_inject: MagicMock,
    mock_reshape: MagicMock,
    mock_build_header: MagicMock,
    mock_write: MagicMock,
    inject_channels: list[int],
    periods: list[float],
    duty_cycles: list[float],
    pulse_snr: list[float],
    initial_phase: list[float],
) -> None:
    """simulate calls inject_periodic_signal exactly once per injection channel (including zero)."""
    cfg = _make_config(
        inject_channels=inject_channels,
        periods=periods,
        duty_cycles=duty_cycles,
        pulse_snr=pulse_snr,
        initial_phase=initial_phase,
    )
    simulate(cfg)
    assert mock_inject.call_count == len(inject_channels)
    mock_gwn.assert_called_once()
    mock_reshape.assert_called_once()
    mock_build_header.assert_called_once()
    mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# inject_periodic_signal argument correctness
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.write_filterbank")
@patch(f"{_MODULE}.build_sigproc_header")
@patch(f"{_MODULE}.reshape_for_sigproc")
@patch(f"{_MODULE}.inject_periodic_signal")
@patch(f"{_MODULE}.generate_white_noise_background")
def test_simulate_inject_receives_noise_data_array(
    mock_gwn: MagicMock,
    mock_inject: MagicMock,
    mock_reshape: MagicMock,
    mock_build_header: MagicMock,
    mock_write: MagicMock,
) -> None:
    """simulate passes the array returned by generate_white_noise_background to inject_periodic_signal."""
    sentinel_data = np.zeros((4, 16))
    mock_gwn.return_value = sentinel_data
    cfg = _make_config(
        inject_channels=[1],
        periods=[1.0],
        duty_cycles=[0.5],
        pulse_snr=[3.0],
        initial_phase=[0.0],
    )
    simulate(cfg)
    assert mock_inject.call_args.kwargs["data"] is sentinel_data
    mock_reshape.assert_called_once()
    mock_build_header.assert_called_once()
    mock_write.assert_called_once()


@patch(f"{_MODULE}.write_filterbank")
@patch(f"{_MODULE}.build_sigproc_header")
@patch(f"{_MODULE}.reshape_for_sigproc")
@patch(f"{_MODULE}.inject_periodic_signal")
@patch(f"{_MODULE}.generate_white_noise_background")
def test_simulate_inject_receives_correct_sample_times(
    mock_gwn: MagicMock,
    mock_inject: MagicMock,
    mock_reshape: MagicMock,
    mock_build_header: MagicMock,
    mock_write: MagicMock,
) -> None:
    """simulate passes sample_times = arange(n_samples) * t_samp to inject_periodic_signal."""
    n_samples, t_samp = 20, 0.25
    cfg = _make_config(
        n_samples=n_samples,
        t_samp=t_samp,
        inject_channels=[0],
        periods=[1.0],
        duty_cycles=[0.5],
        pulse_snr=[3.0],
        initial_phase=[0.0],
    )
    simulate(cfg)
    actual_times = mock_inject.call_args.kwargs["sample_times"]
    expected_times = np.arange(n_samples) * t_samp
    numpy.testing.assert_array_equal(actual_times, expected_times)
    mock_gwn.assert_called_once()
    mock_reshape.assert_called_once()
    mock_build_header.assert_called_once()
    mock_write.assert_called_once()


@patch(f"{_MODULE}.write_filterbank")
@patch(f"{_MODULE}.build_sigproc_header")
@patch(f"{_MODULE}.reshape_for_sigproc")
@patch(f"{_MODULE}.inject_periodic_signal")
@patch(f"{_MODULE}.generate_white_noise_background")
def test_simulate_inject_passes_correct_per_signal_kwargs(
    mock_gwn: MagicMock,
    mock_inject: MagicMock,
    mock_reshape: MagicMock,
    mock_build_header: MagicMock,
    mock_write: MagicMock,
) -> None:
    """simulate forwards channel, period, duty_cycle, pulse_snr, and initial_phase per injection."""
    cfg = _make_config(
        inject_channels=[0, 3],
        periods=[1.5, 3.0],
        duty_cycles=[0.2, 0.4],
        pulse_snr=[6.0, 8.0],
        initial_phase=[0.0, 0.25],
    )
    simulate(cfg)
    calls = mock_inject.call_args_list

    kwargs0 = calls[0].kwargs
    assert kwargs0["channel"] == 0
    assert kwargs0["period"] == pytest.approx(1.5)
    assert kwargs0["duty_cycle"] == pytest.approx(0.2)
    assert kwargs0["pulse_snr"] == pytest.approx(6.0)
    assert kwargs0["initial_phase"] == pytest.approx(0.0)

    kwargs1 = calls[1].kwargs
    assert kwargs1["channel"] == 3
    assert kwargs1["period"] == pytest.approx(3.0)
    assert kwargs1["duty_cycle"] == pytest.approx(0.4)
    assert kwargs1["pulse_snr"] == pytest.approx(8.0)
    assert kwargs1["initial_phase"] == pytest.approx(0.25)
    mock_gwn.assert_called_once()
    mock_reshape.assert_called_once()
    mock_build_header.assert_called_once()
    mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# Data pipeline: noise → reshape
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.write_filterbank")
@patch(f"{_MODULE}.build_sigproc_header")
@patch(f"{_MODULE}.reshape_for_sigproc")
@patch(f"{_MODULE}.inject_periodic_signal")
@patch(f"{_MODULE}.generate_white_noise_background")
def test_simulate_pipes_noise_data_to_reshape(
    mock_gwn: MagicMock,
    mock_inject: MagicMock,
    mock_reshape: MagicMock,
    mock_build_header: MagicMock,
    mock_write: MagicMock,
) -> None:
    """simulate passes the noise array (post-injection) to reshape_for_sigproc."""
    sentinel_data = np.zeros((4, 16))
    mock_gwn.return_value = sentinel_data
    cfg = _make_config()
    simulate(cfg)
    mock_reshape.assert_called_once_with(sentinel_data)
    mock_inject.assert_not_called()
    mock_build_header.assert_called_once()
    mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# build_sigproc_header wiring
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.write_filterbank")
@patch(f"{_MODULE}.build_sigproc_header")
@patch(f"{_MODULE}.reshape_for_sigproc")
@patch(f"{_MODULE}.inject_periodic_signal")
@patch(f"{_MODULE}.generate_white_noise_background")
def test_simulate_builds_header_with_sim_and_header_params(
    mock_gwn: MagicMock,
    mock_inject: MagicMock,
    mock_reshape: MagicMock,
    mock_build_header: MagicMock,
    mock_write: MagicMock,
) -> None:
    """simulate calls build_sigproc_header with simulation_properties and optional_header_parameters."""
    cfg = _make_config(source_name="Cygnus", tstart=59000.5)
    simulate(cfg)
    mock_build_header.assert_called_once_with(cfg.simulation_properties, cfg.optional_header_parameters)
    mock_gwn.assert_called_once()
    mock_inject.assert_not_called()
    mock_reshape.assert_called_once()
    mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# write_filterbank wiring
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.write_filterbank")
@patch(f"{_MODULE}.build_sigproc_header")
@patch(f"{_MODULE}.reshape_for_sigproc")
@patch(f"{_MODULE}.inject_periodic_signal")
@patch(f"{_MODULE}.generate_white_noise_background")
def test_simulate_writes_filterbank_with_reshaped_data_and_header(
    mock_gwn: MagicMock,
    mock_inject: MagicMock,
    mock_reshape: MagicMock,
    mock_build_header: MagicMock,
    mock_write: MagicMock,
    tmp_path: Path,
) -> None:
    """simulate calls write_filterbank with reshaped data, built header, output_dir, and basename."""
    sentinel_reshaped = np.zeros((16, 1, 4))
    sentinel_header: dict[str, object] = {"nchans": 4}
    mock_reshape.return_value = sentinel_reshaped
    mock_build_header.return_value = sentinel_header
    cfg = _make_config(output_dir=tmp_path, basename="my_output")
    simulate(cfg)
    mock_write.assert_called_once_with(
        sentinel_reshaped,
        sentinel_header,
        tmp_path,
        "my_output",
    )
    mock_gwn.assert_called_once()
    mock_inject.assert_not_called()
