"""Unit tests for blipss.cli.compute_phase_resolved_ds"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from blipss.cli.compute_phase_resolved_ds import run_compute_phase_resolved_ds
from blipss.models.compute_phase_resolved_ds import (
    ChannelSelectionConfig,
    InputDataConfig,
    OutputConfig,
    PhaseFoldingConfig,
    PhaseResolvedDsConfig,
    ResourceLimits,
)

_MODULE = "blipss.cli.compute_phase_resolved_ds"


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


_N_CHANNELS = 4
_N_SAMPLES = 100
_TSAMP = 0.002
_FREQS_MHZ = np.linspace(1200.0, 1500.0, _N_CHANNELS)
_DATA = np.random.randn(_N_CHANNELS, _N_SAMPLES).astype(np.float32)
_START_MJD = 59000.0


def _configure_pipeline_mocks(
    mock_extract_data: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    *,
    tsamp: float = _TSAMP,
) -> None:
    """Configure pipeline mocks to return properly-typed tuples matching their real signatures."""
    mock_extract_data.return_value = (_DATA, _N_SAMPLES, tsamp)
    mock_extract_meta.return_value = (_FREQS_MHZ, _START_MJD, tsamp)
    mock_align.return_value = (_DATA, _FREQS_MHZ)
    mock_clip.return_value = (_DATA, _FREQS_MHZ)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _make_config(
    *,
    datafile: str = "obs.fil",
    data_dir: Path = Path("/fake/data"),
    basename: str = "out",
    plot_formats: list[str] | None = None,
    plot_dir: Path = Path("/fake/plots"),
    use_latex: bool = False,
    start_ch: int = 0,
    stop_ch: int | None = None,
    period: float = 1.0,
    bins: int = 10,
    do_deredden: bool = False,
    rmed_width: float = 12.0,
    mem_load: float = 1.0,
    n_workers: int | None = None,
) -> PhaseResolvedDsConfig:
    """Return a valid PhaseResolvedDsConfig with sensible defaults, overridable per-test."""
    return PhaseResolvedDsConfig(
        input_data=InputDataConfig(datafile=datafile, data_dir=data_dir),
        output=OutputConfig(
            basename=basename,
            plot_formats=plot_formats or [".png"],
            plot_dir=plot_dir,
            use_latex=use_latex,
        ),
        channel_selection=ChannelSelectionConfig(start_ch=start_ch, stop_ch=stop_ch),
        phase_folding_parameters=PhaseFoldingConfig(
            period=period,
            bins=bins,
            do_deredden=do_deredden,
            rmed_width=rmed_width,
        ),
        resource_limits=ResourceLimits(mem_load=mem_load, n_workers=n_workers),
    )


# ---------------------------------------------------------------------------
# Error branch
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.fold_all_channels")
@patch(f"{_MODULE}.clip_channels")
@patch(f"{_MODULE}.align_band_orientation")
@patch(f"{_MODULE}.extract_waterfall_metadata")
@patch(f"{_MODULE}.extract_data_array")
@patch(f"{_MODULE}.read_waterfall_file")
def test_run_raises_when_plot_dir_is_none(
    mock_read: MagicMock,
    mock_extract_data: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    mock_fold: MagicMock,
) -> None:
    """run_compute_phase_resolved_ds raises ValueError when plot_dir is None (guard for model_validator bypass)."""

    cfg = MagicMock()
    cfg.output.plot_dir = None
    _configure_pipeline_mocks(mock_extract_data, mock_extract_meta, mock_align, mock_clip)

    with pytest.raises(ValueError, match="Output plot directory was not resolved"):
        run_compute_phase_resolved_ds(cfg)

    mock_read.assert_called_once()
    mock_extract_data.assert_called_once()
    mock_extract_meta.assert_called_once()
    mock_align.assert_called_once()
    mock_clip.assert_called_once()
    mock_fold.assert_called_once()


# ---------------------------------------------------------------------------
# Input path construction
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.plot_phase_resolved_dynamic_spectrum")
@patch(f"{_MODULE}.ensure_path_exists")
@patch(f"{_MODULE}.fold_all_channels")
@patch(f"{_MODULE}.clip_channels")
@patch(f"{_MODULE}.align_band_orientation")
@patch(f"{_MODULE}.extract_waterfall_metadata")
@patch(f"{_MODULE}.extract_data_array")
@patch(f"{_MODULE}.read_waterfall_file")
def test_run_calls_read_waterfall_with_correct_path(
    mock_read: MagicMock,
    mock_extract_data: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    mock_fold: MagicMock,
    mock_ensure: MagicMock,
    mock_plot: MagicMock,
) -> None:
    """run_compute_phase_resolved_ds calls read_waterfall_file with data_dir / datafile."""

    cfg = _make_config(datafile="target.fil", data_dir=Path("/data/observations"))
    _configure_pipeline_mocks(mock_extract_data, mock_extract_meta, mock_align, mock_clip)
    run_compute_phase_resolved_ds(cfg)

    mock_read.assert_called_once_with(Path("/data/observations/target.fil"), cfg.resource_limits.mem_load)
    mock_extract_data.assert_called_once()
    mock_extract_meta.assert_called_once()
    mock_align.assert_called_once()
    mock_clip.assert_called_once()
    mock_fold.assert_called_once()
    mock_ensure.assert_called_once()
    mock_plot.assert_called_once()


# ---------------------------------------------------------------------------
# Folding parameters forwarded from config
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.plot_phase_resolved_dynamic_spectrum")
@patch(f"{_MODULE}.ensure_path_exists")
@patch(f"{_MODULE}.fold_all_channels")
@patch(f"{_MODULE}.clip_channels")
@patch(f"{_MODULE}.align_band_orientation")
@patch(f"{_MODULE}.extract_waterfall_metadata")
@patch(f"{_MODULE}.extract_data_array")
@patch(f"{_MODULE}.read_waterfall_file")
def test_run_calls_fold_all_channels_with_config_params(
    mock_read: MagicMock,
    mock_extract_data: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    mock_fold: MagicMock,
    mock_ensure: MagicMock,
    mock_plot: MagicMock,
) -> None:
    """run_compute_phase_resolved_ds forwards period, bins, do_deredden, rmed_width, n_workers to fold_all_channels."""

    cfg = _make_config(period=0.15637, bins=128, do_deredden=True, rmed_width=1.0, n_workers=None)
    _configure_pipeline_mocks(mock_extract_data, mock_extract_meta, mock_align, mock_clip)
    run_compute_phase_resolved_ds(cfg)

    kwargs = mock_fold.call_args.kwargs
    assert kwargs["tsamp"] == pytest.approx(0.002)
    assert kwargs["period"] == pytest.approx(0.15637)
    assert kwargs["bins"] == 128
    assert kwargs["do_deredden"] is True
    assert kwargs["rmed_width"] == pytest.approx(1.0)
    assert kwargs["n_workers"] is None
    mock_read.assert_called_once()
    mock_extract_data.assert_called_once()
    mock_extract_meta.assert_called_once()
    mock_align.assert_called_once()
    mock_clip.assert_called_once()
    mock_ensure.assert_called_once()
    mock_plot.assert_called_once()


# ---------------------------------------------------------------------------
# Output path construction
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.plot_phase_resolved_dynamic_spectrum")
@patch(f"{_MODULE}.ensure_path_exists")
@patch(f"{_MODULE}.fold_all_channels")
@patch(f"{_MODULE}.clip_channels")
@patch(f"{_MODULE}.align_band_orientation")
@patch(f"{_MODULE}.extract_waterfall_metadata")
@patch(f"{_MODULE}.extract_data_array")
@patch(f"{_MODULE}.read_waterfall_file")
def test_run_calls_plot_with_correct_plot_name(
    mock_read: MagicMock,
    mock_extract_data: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    mock_fold: MagicMock,
    mock_ensure: MagicMock,
    mock_plot: MagicMock,
) -> None:
    """run_compute_phase_resolved_ds passes basename_period<period:.5f> as the plot_name argument."""

    cfg = _make_config(basename="my_obs", period=0.15637, plot_dir=Path("/output"))
    _configure_pipeline_mocks(mock_extract_data, mock_extract_meta, mock_align, mock_clip)
    run_compute_phase_resolved_ds(cfg)

    expected_plot_name = "/output/my_obs_period0.15637"
    actual_plot_name: str = mock_plot.call_args.args[4]
    assert actual_plot_name == expected_plot_name
    mock_read.assert_called_once()
    mock_extract_data.assert_called_once()
    mock_extract_meta.assert_called_once()
    mock_align.assert_called_once()
    mock_clip.assert_called_once()
    mock_fold.assert_called_once()
    mock_ensure.assert_called_once()


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.plot_phase_resolved_dynamic_spectrum")
@patch(f"{_MODULE}.ensure_path_exists")
@patch(f"{_MODULE}.fold_all_channels")
@patch(f"{_MODULE}.clip_channels")
@patch(f"{_MODULE}.align_band_orientation")
@patch(f"{_MODULE}.extract_waterfall_metadata")
@patch(f"{_MODULE}.extract_data_array")
@patch(f"{_MODULE}.read_waterfall_file")
def test_run_calls_ensure_path_exists_with_plot_dir(
    mock_read: MagicMock,
    mock_extract_data: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    mock_fold: MagicMock,
    mock_ensure: MagicMock,
    mock_plot: MagicMock,
) -> None:
    """run_compute_phase_resolved_ds calls ensure_path_exists with the resolved plot_dir."""

    plot_dir = Path("/expected/output_dir")
    cfg = _make_config(plot_dir=plot_dir)
    _configure_pipeline_mocks(mock_extract_data, mock_extract_meta, mock_align, mock_clip)
    run_compute_phase_resolved_ds(cfg)

    mock_ensure.assert_called_once_with(plot_dir)
    mock_read.assert_called_once()
    mock_extract_data.assert_called_once()
    mock_extract_meta.assert_called_once()
    mock_align.assert_called_once()
    mock_clip.assert_called_once()
    mock_fold.assert_called_once()
    mock_plot.assert_called_once()
