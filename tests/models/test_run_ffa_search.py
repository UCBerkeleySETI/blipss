"""Unit tests for config models in blipss.models.run_ffa_search"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from blipss.constants import DEFAULT_EPSILON_HARMONIC
from blipss.models.run_ffa_search import (
    BlipssConfig,
    FfaSearchConfig,
    InputConfig,
    OutputConfig,
    PlottingConfig,
    ResourceConfig,
)

_MODULE = "blipss.models.run_ffa_search"


# ---------------------------------------------------------------------------
# InputConfig
# ---------------------------------------------------------------------------


def test_input_config_negative_start_ch_rejected() -> None:
    """InputConfig rejects a negative start_ch."""
    with pytest.raises(ValidationError, match="start_ch must be >= 0"):
        InputConfig(data_dir=Path("/fake/data"), start_ch=-1)


def test_input_config_glob_input_populates_sorted_file_list(tmp_path: Path) -> None:
    """InputConfig expands glob_input against data_dir into a sorted input_file_list."""
    for name in ("b.fil", "a.fil", "skip.h5"):
        (tmp_path / name).touch()
    cfg = InputConfig(data_dir=tmp_path, glob_input="*.fil")
    assert cfg.input_file_list == [tmp_path / "a.fil", tmp_path / "b.fil"]


def test_input_config_explicit_file_list_takes_precedence_over_glob(tmp_path: Path) -> None:
    """InputConfig leaves an explicitly supplied input_file_list untouched even when glob_input is set."""
    (tmp_path / "a.fil").touch()
    explicit = [tmp_path / "chosen.fil"]
    cfg = InputConfig(data_dir=tmp_path, glob_input="*.fil", input_file_list=explicit)
    assert cfg.input_file_list == explicit


def test_input_config_no_glob_leaves_file_list_empty(tmp_path: Path) -> None:
    """InputConfig leaves input_file_list empty when neither glob_input nor input_file_list is given."""
    cfg = InputConfig(data_dir=tmp_path)
    assert cfg.input_file_list == []


@pytest.mark.parametrize("stop_ch", [5, 3], ids=["equal_to_start", "below_start"])
def test_input_config_stop_ch_must_exceed_start_ch(stop_ch: int) -> None:
    """InputConfig rejects a stop_ch that is not strictly greater than start_ch."""
    with pytest.raises(ValidationError, match="must be greater than start_ch"):
        InputConfig(data_dir=Path("/fake/data"), start_ch=5, stop_ch=stop_ch)


def test_input_config_null_channel_bounds_resolve_to_defaults() -> None:
    """InputConfig maps null start_ch and stop_ch onto their defaults of 0 and None."""
    cfg = InputConfig.model_validate({"data_dir": Path("/fake/data"), "start_ch": None, "stop_ch": None})
    assert cfg.start_ch == 0
    assert cfg.stop_ch is None


# ---------------------------------------------------------------------------
# FfaSearchConfig
# ---------------------------------------------------------------------------


def test_ffa_search_config_defaults() -> None:
    """FfaSearchConfig applies documented defaults, including the shared harmonic tolerance constant."""
    cfg = FfaSearchConfig()
    assert cfg.min_period == 10.0
    assert cfg.max_period == 100.0
    assert cfg.bins_min == 10
    assert cfg.bins_max == 11
    assert cfg.do_deredden is False
    assert cfg.epsilon_harmonic == DEFAULT_EPSILON_HARMONIC


@pytest.mark.parametrize(
    ("min_period", "max_period"),
    [(50.0, 50.0), (60.0, 50.0)],
    ids=["equal_periods", "min_above_max"],
)
def test_ffa_search_config_period_range_must_be_increasing(min_period: float, max_period: float) -> None:
    """FfaSearchConfig rejects a min_period that is not strictly below max_period."""
    with pytest.raises(ValidationError, match="must be less than max_period"):
        FfaSearchConfig(min_period=min_period, max_period=max_period)


def test_ffa_search_config_bins_min_above_bins_max_rejected() -> None:
    """FfaSearchConfig rejects bins_min greater than bins_max."""
    with pytest.raises(ValidationError, match="must be <= bins_max"):
        FfaSearchConfig(bins_min=12, bins_max=11)


def test_ffa_search_config_equal_bins_bounds_accepted() -> None:
    """FfaSearchConfig accepts bins_min equal to bins_max."""
    cfg = FfaSearchConfig(bins_min=10, bins_max=10)
    assert cfg.bins_min == cfg.bins_max == 10


@pytest.mark.parametrize("ducy_max", [0.0, -0.1, 1.5], ids=["zero", "negative", "above_one"])
def test_ffa_search_config_ducy_max_out_of_range_rejected(ducy_max: float) -> None:
    """FfaSearchConfig rejects a ducy_max outside the half-open interval (0, 1]."""
    with pytest.raises(ValidationError, match=r"must be in \(0, 1\]"):
        FfaSearchConfig(ducy_max=ducy_max)


def test_ffa_search_config_ducy_max_upper_bound_accepted() -> None:
    """FfaSearchConfig accepts the boundary value ducy_max = 1.0."""
    assert FfaSearchConfig(ducy_max=1.0).ducy_max == 1.0


@pytest.mark.parametrize(
    "field_name",
    ["snr_threshold", "epsilon_fof", "epsilon_harmonic", "rmed_width", "min_period"],
    ids=["snr_threshold", "epsilon_fof", "epsilon_harmonic", "rmed_width", "min_period"],
)
@pytest.mark.parametrize("value", [0.0, -1.0], ids=["zero", "negative"])
def test_ffa_search_config_float_fields_must_be_positive(field_name: str, value: float) -> None:
    """FfaSearchConfig rejects non-positive values for each positively-constrained float field."""
    with pytest.raises(ValidationError, match="value must be positive"):
        FfaSearchConfig.model_validate({field_name: value})


@pytest.mark.parametrize("field_name", ["fpmin", "bins_min", "bins_max"], ids=["fpmin", "bins_min", "bins_max"])
def test_ffa_search_config_int_fields_must_be_positive(field_name: str) -> None:
    """FfaSearchConfig rejects non-positive values for each positively-constrained integer field."""
    with pytest.raises(ValidationError, match="value must be positive"):
        FfaSearchConfig.model_validate({field_name: 0})


# ---------------------------------------------------------------------------
# ResourceConfig
# ---------------------------------------------------------------------------


def test_resource_config_defaults() -> None:
    """ResourceConfig defaults to a 1 GB memory budget and an unset worker count."""
    cfg = ResourceConfig()
    assert cfg.mem_load == 1.0
    assert cfg.n_workers is None


@pytest.mark.parametrize("mem_load", [0.0, -2.0], ids=["zero", "negative"])
def test_resource_config_mem_load_must_be_positive(mem_load: float) -> None:
    """ResourceConfig rejects a non-positive mem_load."""
    with pytest.raises(ValidationError, match="mem_load must be positive"):
        ResourceConfig(mem_load=mem_load)


def test_resource_config_n_workers_below_one_rejected() -> None:
    """ResourceConfig rejects an n_workers value below 1."""
    with pytest.raises(ValidationError, match="n_workers must be >= 1"):
        ResourceConfig(n_workers=0)


@patch(f"{_MODULE}.os.cpu_count", return_value=4)
def test_resource_config_n_workers_above_cpu_count_rejected(mock_cpu_count: MagicMock) -> None:
    """ResourceConfig rejects an n_workers value exceeding the available CPU count."""
    with pytest.raises(ValidationError, match=r"exceeds available CPU count \(4\)"):
        ResourceConfig(n_workers=5)
    mock_cpu_count.assert_called_once()


@patch(f"{_MODULE}.os.cpu_count", return_value=None)
def test_resource_config_n_workers_accepted_when_cpu_count_unknown(mock_cpu_count: MagicMock) -> None:
    """ResourceConfig accepts any n_workers >= 1 when the CPU count cannot be determined."""
    cfg = ResourceConfig(n_workers=999)
    assert cfg.n_workers == 999
    mock_cpu_count.assert_called_once()


# ---------------------------------------------------------------------------
# BlipssConfig
# ---------------------------------------------------------------------------


def test_blipss_config_output_dir_defaults_to_data_dir() -> None:
    """BlipssConfig resolves an omitted output_dir to the input data directory."""
    cfg = BlipssConfig(input=InputConfig(data_dir=Path("/fake/data")))
    assert cfg.output.output_dir == Path("/fake/data")


def test_blipss_config_explicit_output_dir_preserved() -> None:
    """BlipssConfig leaves an explicitly supplied output_dir unchanged."""
    cfg = BlipssConfig(
        input=InputConfig(data_dir=Path("/fake/data")),
        output=OutputConfig(output_dir=Path("/fake/out")),
    )
    assert cfg.output.output_dir == Path("/fake/out")


def test_blipss_config_section_defaults() -> None:
    """BlipssConfig populates optional sections with their default sub-models."""
    cfg = BlipssConfig(input=InputConfig(data_dir=Path("/fake/data")))
    assert cfg.plotting == PlottingConfig()
    assert cfg.ffa_search == FfaSearchConfig()
    assert cfg.resources == ResourceConfig()


def test_blipss_config_from_raw_yaml_style_dict_with_nulls() -> None:
    """BlipssConfig resolves null YAML entries in nested sections to their declared defaults."""
    raw: dict[str, Any] = {
        "input": {"data_dir": "/fake/data", "glob_input": None, "start_ch": None, "stop_ch": None},
        "output": {"output_dir": None},
        "plotting": {"do_plot": True, "plot_formats": None, "use_latex": None},
        "resources": {"mem_load": 15.0, "n_workers": None},
    }
    cfg = BlipssConfig(**raw)
    assert cfg.input.start_ch == 0
    assert cfg.output.output_dir == Path("/fake/data")
    assert cfg.plotting.do_plot is True
    assert cfg.plotting.plot_formats == [".png"]
    assert cfg.plotting.use_latex is False
    assert cfg.resources.n_workers is None
