"""Unit tests for config models in blipss.models.plot_cands"""

from pathlib import Path
from typing import Any

from blipss.models.plot_cands import (
    CandidateFileConfig,
    InputDataConfig,
    PlotCandsConfig,
    PlottingParametersConfig,
)
from blipss.models.run_ffa_search import FfaSearchConfig, ResourceConfig

_CANDIDATE_FILE_CFG = CandidateFileConfig(csvfile=Path("/fake/compared.csv"))

# ---------------------------------------------------------------------------
# InputDataConfig
# ---------------------------------------------------------------------------


def test_input_data_config_beam_labels_default_to_blank_per_datafile() -> None:
    """InputDataConfig defaults beam_labels to one blank label per data file when omitted."""
    cfg = InputDataConfig(data_dir=Path("/fake/data"), datafile_list=[Path("a.fil"), Path("b.fil")])
    assert cfg.beam_labels == ["", ""]


def test_input_data_config_explicit_beam_labels_preserved() -> None:
    """InputDataConfig leaves explicitly supplied beam_labels untouched."""
    cfg = InputDataConfig(
        data_dir=Path("/fake/data"),
        datafile_list=[Path("a.fil"), Path("b.fil")],
        beam_labels=["Source X", "Source Y"],
    )
    assert cfg.beam_labels == ["Source X", "Source Y"]


# ---------------------------------------------------------------------------
# PlottingParametersConfig
# ---------------------------------------------------------------------------


def test_plotting_parameters_config_defaults() -> None:
    """PlottingParametersConfig applies documented defaults for optional fields."""
    cfg = PlottingParametersConfig(codes_plot=["101010"], basename="cands")
    assert cfg.plot_formats == [".png"]
    assert cfg.plot_dir is None
    assert cfg.periodaxis_log is True
    assert cfg.use_latex is False


# ---------------------------------------------------------------------------
# PlotCandsConfig
# ---------------------------------------------------------------------------


def _make_input_data_cfg(data_dir: Path = Path("/fake/data")) -> InputDataConfig:
    """Return a minimal valid InputDataConfig rooted at the given data_dir."""
    return InputDataConfig(data_dir=data_dir, datafile_list=[Path("a.fil")])


def test_plot_cands_config_plot_dir_defaults_to_data_dir() -> None:
    """PlotCandsConfig resolves an omitted plot_dir to the input data directory."""
    cfg = PlotCandsConfig(
        input_data=_make_input_data_cfg(Path("/fake/data")),
        candidate_file=_CANDIDATE_FILE_CFG,
        plotting_parameters=PlottingParametersConfig(codes_plot=["10"], basename="cands"),
    )
    assert cfg.plotting_parameters.plot_dir == Path("/fake/data")


def test_plot_cands_config_explicit_plot_dir_preserved() -> None:
    """PlotCandsConfig leaves an explicitly supplied plot_dir unchanged."""
    cfg = PlotCandsConfig(
        input_data=_make_input_data_cfg(Path("/fake/data")),
        candidate_file=_CANDIDATE_FILE_CFG,
        plotting_parameters=PlottingParametersConfig(codes_plot=["10"], basename="cands", plot_dir=Path("/fake/plots")),
    )
    assert cfg.plotting_parameters.plot_dir == Path("/fake/plots")


def test_plot_cands_config_section_defaults() -> None:
    """PlotCandsConfig populates omitted folding_search_parameters and resource_limits with default sub-models."""
    cfg = PlotCandsConfig(
        input_data=_make_input_data_cfg(),
        candidate_file=_CANDIDATE_FILE_CFG,
        plotting_parameters=PlottingParametersConfig(codes_plot=["10"], basename="cands"),
    )
    assert cfg.folding_search_parameters == FfaSearchConfig()
    assert cfg.resource_limits == ResourceConfig()


def test_plot_cands_config_from_raw_yaml_style_dict_with_nulls() -> None:
    """PlotCandsConfig resolves null YAML entries in nested sections to their declared defaults."""
    raw: dict[str, Any] = {
        "input_data": {"data_dir": "/fake/data", "datafile_list": ["a.fil"], "beam_labels": None},
        "candidate_file": {"csvfile": "/fake/compared.csv"},
        "plotting_parameters": {
            "codes_plot": ["10"],
            "basename": "cands",
            "plot_formats": None,
            "plot_dir": None,
            "use_latex": None,
        },
    }
    cfg = PlotCandsConfig(**raw)
    assert cfg.input_data.beam_labels == [""]
    assert cfg.plotting_parameters.plot_formats == [".png"]
    assert cfg.plotting_parameters.plot_dir == Path("/fake/data")
    assert cfg.plotting_parameters.use_latex is False
