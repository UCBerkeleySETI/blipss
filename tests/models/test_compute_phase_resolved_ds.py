"""Unit tests for config models in blipss.models.compute_phase_resolved_ds"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from blipss.models.compute_phase_resolved_ds import (
    InputDataConfig,
    OutputConfig,
    PhaseFoldingConfig,
    PhaseResolvedDsConfig,
    ResourceLimits,
)

_MODULE = "blipss.models.compute_phase_resolved_ds"


def test_output_config_null_plot_formats_resolves_to_png() -> None:
    """OutputConfig falls back to ['.png'] when plot_formats is supplied as null."""
    cfg = OutputConfig.model_validate({"basename": "out", "plot_formats": None})
    assert cfg.plot_formats == [".png"]


def test_phase_folding_config_null_rmed_width_resolves_to_default() -> None:
    """PhaseFoldingConfig falls back to 12.0 s when rmed_width is supplied as null."""
    cfg = PhaseFoldingConfig.model_validate({"rmed_width": None})
    assert cfg.rmed_width == 12.0


def test_plot_dir_defaults_to_data_dir_when_null() -> None:
    """PhaseResolvedDsConfig resolves a null plot_dir to the input data directory."""
    cfg = PhaseResolvedDsConfig(
        input_data=InputDataConfig(datafile="obs.fil", data_dir=Path("/fake/data")),
        output=OutputConfig(basename="out", plot_dir=None),
    )
    assert cfg.output.plot_dir == Path("/fake/data")


def test_explicit_plot_dir_is_preserved() -> None:
    """PhaseResolvedDsConfig leaves an explicitly supplied plot_dir unchanged."""
    cfg = PhaseResolvedDsConfig(
        input_data=InputDataConfig(datafile="obs.fil", data_dir=Path("/fake/data")),
        output=OutputConfig(basename="out", plot_dir=Path("/fake/plots")),
    )
    assert cfg.output.plot_dir == Path("/fake/plots")


def test_resource_limits_n_workers_below_one_rejected() -> None:
    """ResourceLimits rejects an n_workers value below 1."""
    with pytest.raises(ValidationError, match="n_workers must be >= 1"):
        ResourceLimits(n_workers=0)


@patch(f"{_MODULE}.os.cpu_count", return_value=4)
def test_resource_limits_n_workers_above_cpu_count_rejected(mock_cpu_count: MagicMock) -> None:
    """ResourceLimits rejects an n_workers value exceeding the available CPU count."""
    with pytest.raises(ValidationError, match=r"exceeds available CPU count \(4\)"):
        ResourceLimits(n_workers=5)
    mock_cpu_count.assert_called_once()


@patch(f"{_MODULE}.os.cpu_count", return_value=None)
def test_resource_limits_n_workers_accepted_when_cpu_count_unknown(mock_cpu_count: MagicMock) -> None:
    """ResourceLimits accepts any n_workers >= 1 when the CPU count cannot be determined."""
    assert ResourceLimits(n_workers=999).n_workers == 999
    mock_cpu_count.assert_called_once()
