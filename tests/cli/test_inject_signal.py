"""Unit tests for blipss.cli.inject_signal — orchestrator for real-data signal injection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from blipss.cli.inject_signal import run_inject_signal
from blipss.models.inject_signal import (
    InjectSignalConfig,
    InputDataConfig,
    OutputConfig,
    ResourceLimits,
)
from blipss.models.simulate_data import PeriodicSignalInjection

_MODULE = "blipss.cli.inject_signal"


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _make_config(
    *,
    datafile: str = "obs.fil",
    data_dir: Path = Path("/fake/data"),
    basename: str = "out",
    output_ext: str = ".fil",
    output_dir: Path = Path("/fake/output"),
    inject_channels: list[int] | None = None,
    periods: list[float] | None = None,
    duty_cycles: list[float] | None = None,
    pulse_snr: list[float] | None = None,
    initial_phase: list[float] | None = None,
    mem_load: float = 1.0,
) -> InjectSignalConfig:
    """Return a valid InjectSignalConfig with sensible defaults, overridable per-test."""
    inject_channels = inject_channels or []
    periods = periods or []
    duty_cycles = duty_cycles or []
    pulse_snr = pulse_snr or []
    initial_phase = initial_phase or []
    return InjectSignalConfig(
        input_data=InputDataConfig(datafile=datafile, data_dir=data_dir),
        output=OutputConfig(basename=basename, output_ext=output_ext, output_dir=output_dir),
        periodic_signal_injection=PeriodicSignalInjection(
            inject_channels=inject_channels,
            periods=periods,
            duty_cycles=duty_cycles,
            pulse_snr=pulse_snr,
            initial_phase=initial_phase,
        ),
        resource_limits=ResourceLimits(mem_load=mem_load),
    )


# ---------------------------------------------------------------------------
# Error branch
# ---------------------------------------------------------------------------


def test_inject_raises_when_output_dir_is_none() -> None:
    """inject raises ValueError when output_dir is None (guard for model_validator bypass)."""
    cfg = MagicMock()
    cfg.output.output_dir = None
    with pytest.raises(ValueError, match="Output directory was not resolved"):
        run_inject_signal(cfg)


# ---------------------------------------------------------------------------
# inject_periodic_signal call-count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inject_channels", "periods", "duty_cycles", "pulse_snr", "initial_phase"),
    [
        ([], [], [], [], []),
        ([2], [1.0], [0.5], [5.0], [0.0]),
        ([0, 1, 3], [0.5, 1.0, 2.0], [0.1, 0.2, 0.3], [3.0, 4.0, 5.0], [0.0, 0.1, 0.2]),
    ],
    ids=["no-injection", "single-injection", "multi-injection"],
)
@patch(f"{_MODULE}.write_waterfall")
@patch(f"{_MODULE}.ensure_path_exists")
@patch(f"{_MODULE}.pack_data_into_waterfall")
@patch(f"{_MODULE}.inject_periodic_signal")
@patch(f"{_MODULE}.compute_per_channel_std")
@patch(f"{_MODULE}.compute_median_bandpass")
@patch(f"{_MODULE}.extract_data_array")
@patch(f"{_MODULE}.read_waterfall_file")
def test_inject_call_count_matches_channel_count(
    mock_read: MagicMock,
    mock_extract: MagicMock,
    mock_median: MagicMock,
    mock_std: MagicMock,
    mock_inject_sig: MagicMock,
    mock_pack: MagicMock,
    mock_ensure: MagicMock,
    mock_write: MagicMock,
    inject_channels: list[int],
    periods: list[float],
    duty_cycles: list[float],
    pulse_snr: list[float],
    initial_phase: list[float],
) -> None:
    """inject calls inject_periodic_signal exactly once per injection channel (including zero)."""
    mock_extract.return_value = (np.zeros((4, 8)), 8, 0.001)
    mock_median.return_value = np.zeros(4)
    mock_std.return_value = np.ones(4)

    cfg = _make_config(
        inject_channels=inject_channels,
        periods=periods,
        duty_cycles=duty_cycles,
        pulse_snr=pulse_snr,
        initial_phase=initial_phase,
    )
    run_inject_signal(cfg)
    assert mock_inject_sig.call_count == len(inject_channels)
    mock_read.assert_called_once()
    mock_pack.assert_called_once()
    mock_ensure.assert_called_once()
    mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# Amplitude calibration
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.write_waterfall")
@patch(f"{_MODULE}.ensure_path_exists")
@patch(f"{_MODULE}.pack_data_into_waterfall")
@patch(f"{_MODULE}.inject_periodic_signal")
@patch(f"{_MODULE}.compute_per_channel_std")
@patch(f"{_MODULE}.compute_median_bandpass")
@patch(f"{_MODULE}.extract_data_array")
@patch(f"{_MODULE}.read_waterfall_file")
def test_inject_amplitude_calibrated_from_bandpass_and_std(
    mock_read: MagicMock,
    mock_extract: MagicMock,
    mock_median: MagicMock,
    mock_std: MagicMock,
    mock_inject_sig: MagicMock,
    mock_pack: MagicMock,
    mock_ensure: MagicMock,
    mock_write: MagicMock,
) -> None:
    """inject passes pulse_snr = median_bp[ch] + cfg_snr * std[ch] to inject_periodic_signal."""
    channel = 2
    cfg_snr = 5.0
    mock_extract.return_value = (np.zeros((4, 8)), 8, 0.001)
    mock_median.return_value = np.array([0.0, 0.0, 100.0, 0.0])
    mock_std.return_value = np.array([1.0, 1.0, 3.0, 1.0])

    cfg = _make_config(
        inject_channels=[channel],
        periods=[1.0],
        duty_cycles=[0.5],
        pulse_snr=[cfg_snr],
        initial_phase=[0.0],
    )
    run_inject_signal(cfg)
    # Expected amplitude: median_bp[2] + pulse_snr * std[2] = 100.0 + 5.0 * 3.0 = 115.0
    actual_amplitude: float = mock_inject_sig.call_args.kwargs["pulse_snr"]
    assert actual_amplitude == pytest.approx(115.0)
    mock_read.assert_called_once()
    mock_pack.assert_called_once()
    mock_ensure.assert_called_once()
    mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# Output path construction
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.write_waterfall")
@patch(f"{_MODULE}.ensure_path_exists")
@patch(f"{_MODULE}.pack_data_into_waterfall")
@patch(f"{_MODULE}.inject_periodic_signal")
@patch(f"{_MODULE}.compute_per_channel_std")
@patch(f"{_MODULE}.compute_median_bandpass")
@patch(f"{_MODULE}.extract_data_array")
@patch(f"{_MODULE}.read_waterfall_file")
def test_inject_calls_write_waterfall_with_correct_output_path(
    mock_read: MagicMock,
    mock_extract: MagicMock,
    mock_median: MagicMock,
    mock_std: MagicMock,
    mock_inject_sig: MagicMock,
    mock_pack: MagicMock,
    mock_ensure: MagicMock,
    mock_write: MagicMock,
) -> None:
    """inject calls write_waterfall with the packed Waterfall and output_dir/basename+ext."""
    mock_extract.return_value = (np.zeros((4, 8)), 8, 0.001)
    mock_median.return_value = np.zeros(4)
    mock_std.return_value = np.ones(4)
    sentinel_wat = MagicMock()
    mock_pack.return_value = sentinel_wat

    output_dir = Path("/some/output")
    cfg = _make_config(basename="my_obs", output_ext=".fil", output_dir=output_dir)
    run_inject_signal(cfg)
    mock_write.assert_called_once_with(sentinel_wat, output_dir / "my_obs.fil")
    mock_read.assert_called_once()
    mock_inject_sig.assert_not_called()
    mock_ensure.assert_called_once()


@patch(f"{_MODULE}.write_waterfall")
@patch(f"{_MODULE}.ensure_path_exists")
@patch(f"{_MODULE}.pack_data_into_waterfall")
@patch(f"{_MODULE}.inject_periodic_signal")
@patch(f"{_MODULE}.compute_per_channel_std")
@patch(f"{_MODULE}.compute_median_bandpass")
@patch(f"{_MODULE}.extract_data_array")
@patch(f"{_MODULE}.read_waterfall_file")
def test_inject_calls_ensure_path_exists_with_output_dir(
    mock_read: MagicMock,
    mock_extract: MagicMock,
    mock_median: MagicMock,
    mock_std: MagicMock,
    mock_inject_sig: MagicMock,
    mock_pack: MagicMock,
    mock_ensure: MagicMock,
    mock_write: MagicMock,
) -> None:
    """inject calls ensure_path_exists with the resolved output directory."""
    mock_extract.return_value = (np.zeros((4, 8)), 8, 0.001)
    mock_median.return_value = np.zeros(4)
    mock_std.return_value = np.ones(4)

    output_dir = Path("/expected/output_dir")
    cfg = _make_config(output_dir=output_dir)
    run_inject_signal(cfg)
    mock_ensure.assert_called_once_with(output_dir)
    mock_read.assert_called_once()
    mock_inject_sig.assert_not_called()
    mock_pack.assert_called_once()
    mock_write.assert_called_once()
