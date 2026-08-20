"""Unit tests for config models in blipss.models.compare_cands"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from blipss.constants import DEFAULT_CLUSTER_RADIUS_SECONDS, DEFAULT_OFF_SNR_CUTOFF, DEFAULT_ON_SNR_CUTOFF
from blipss.models.compare_cands import (
    CandidateFilesConfig,
    CandidateGroupingConfig,
    CompareCandsConfig,
    OnOffClassificationConfig,
    OutputConfig,
)

_MODULE = "blipss.models.compare_cands"

_FILES_CFG = CandidateFilesConfig(csv_dir=Path("/fake/dir"), csv_list=[Path("a.csv"), Path("b.csv")])
_OUTPUT_CFG = OutputConfig(basename="out", output_dir=Path("/fake/out"))


# ---------------------------------------------------------------------------
# OnOffClassificationConfig
# ---------------------------------------------------------------------------


def test_on_off_classification_config_defaults() -> None:
    """OnOffClassificationConfig applies the shared default S/N cutoffs from blipss.constants."""
    cfg = OnOffClassificationConfig(labels=["ON", "OFF"])
    assert cfg.on_cutoff == DEFAULT_ON_SNR_CUTOFF
    assert cfg.off_cutoff == DEFAULT_OFF_SNR_CUTOFF


def test_on_off_classification_config_labels_normalised_to_uppercase() -> None:
    """OnOffClassificationConfig uppercases lowercase or mixed-case pointing labels."""
    cfg = OnOffClassificationConfig(labels=["on", "Off"])
    assert cfg.labels == ["ON", "OFF"]


def test_on_off_classification_config_invalid_label_rejected() -> None:
    """OnOffClassificationConfig rejects a label that is neither 'ON' nor 'OFF'."""
    with pytest.raises(ValidationError, match="labels must be 'ON' or 'OFF'"):
        OnOffClassificationConfig(labels=["ON", "MAYBE"])


@pytest.mark.parametrize("field_name", ["on_cutoff", "off_cutoff"], ids=["on_cutoff", "off_cutoff"])
@pytest.mark.parametrize("value", [0.0, -1.0], ids=["zero", "negative"])
def test_on_off_classification_config_cutoffs_must_be_positive(field_name: str, value: float) -> None:
    """OnOffClassificationConfig rejects a non-positive S/N cutoff for either pointing type."""
    with pytest.raises(ValidationError, match="S/N cutoff must be > 0"):
        OnOffClassificationConfig.model_validate({"labels": ["ON"], field_name: value})


# ---------------------------------------------------------------------------
# CandidateGroupingConfig
# ---------------------------------------------------------------------------


def test_candidate_grouping_config_defaults() -> None:
    """CandidateGroupingConfig defaults cluster_radius to the shared constant."""
    cfg = CandidateGroupingConfig()
    assert cfg.cluster_radius == DEFAULT_CLUSTER_RADIUS_SECONDS


@patch(f"{_MODULE}.os.cpu_count", return_value=4)
def test_candidate_grouping_config_n_jobs_defaults_to_cpu_count(mock_cpu_count: MagicMock) -> None:
    """CandidateGroupingConfig defaults n_jobs to the detected CPU count when omitted."""
    cfg = CandidateGroupingConfig()
    assert cfg.n_jobs == 4
    mock_cpu_count.assert_called_once()


@patch(f"{_MODULE}.os.cpu_count", return_value=None)
def test_candidate_grouping_config_n_jobs_defaults_to_one_when_cpu_count_unknown(mock_cpu_count: MagicMock) -> None:
    """CandidateGroupingConfig falls back to n_jobs=1 when the CPU count cannot be determined."""
    cfg = CandidateGroupingConfig()
    assert cfg.n_jobs == 1


def test_candidate_grouping_config_cluster_radius_must_be_positive() -> None:
    """CandidateGroupingConfig rejects a non-positive cluster_radius."""
    with pytest.raises(ValidationError, match="cluster_radius must be positive"):
        CandidateGroupingConfig(cluster_radius=0.0)


def test_candidate_grouping_config_n_jobs_must_be_at_least_one() -> None:
    """CandidateGroupingConfig rejects an n_jobs value below 1."""
    with pytest.raises(ValidationError, match="n_jobs must be >= 1"):
        CandidateGroupingConfig(n_jobs=0)


# ---------------------------------------------------------------------------
# CompareCandsConfig
# ---------------------------------------------------------------------------


def test_compare_cands_config_defaults_candidate_grouping() -> None:
    """CompareCandsConfig populates an omitted candidate_grouping section with its default sub-model."""
    cfg = CompareCandsConfig(
        candidate_files=_FILES_CFG,
        on_off_classification=OnOffClassificationConfig(labels=["ON", "OFF"]),
        output=_OUTPUT_CFG,
    )
    assert cfg.candidate_grouping.cluster_radius == DEFAULT_CLUSTER_RADIUS_SECONDS


def test_compare_cands_config_labels_length_must_match_csv_list() -> None:
    """CompareCandsConfig rejects a labels list whose length differs from csv_list."""
    with pytest.raises(ValidationError, match="csv_list has 2 entries but labels has 1"):
        CompareCandsConfig(
            candidate_files=_FILES_CFG,
            on_off_classification=OnOffClassificationConfig(labels=["ON"]),
            output=_OUTPUT_CFG,
        )


def test_compare_cands_config_matching_label_and_csv_list_lengths_accepted() -> None:
    """CompareCandsConfig accepts a labels list with one entry per csv_list file."""
    cfg = CompareCandsConfig(
        candidate_files=_FILES_CFG,
        on_off_classification=OnOffClassificationConfig(labels=["ON", "OFF"]),
        output=_OUTPUT_CFG,
    )
    assert len(cfg.candidate_files.csv_list) == len(cfg.on_off_classification.labels)


def test_compare_cands_config_from_raw_yaml_style_dict_with_nulls() -> None:
    """CompareCandsConfig resolves null YAML entries in nested sections to their declared defaults."""
    raw = {
        "candidate_files": {"csv_dir": "/fake/dir", "csv_list": ["a.csv", "b.csv"]},
        "on_off_classification": {"labels": ["on", "off"], "on_cutoff": None, "off_cutoff": None},
        "output": {"basename": "out", "output_dir": "/fake/out"},
        "candidate_grouping": {"cluster_radius": None, "n_jobs": None},
    }
    cfg = CompareCandsConfig(**raw)
    assert cfg.on_off_classification.labels == ["ON", "OFF"]
    assert cfg.on_off_classification.on_cutoff == DEFAULT_ON_SNR_CUTOFF
    assert cfg.candidate_grouping.cluster_radius == DEFAULT_CLUSTER_RADIUS_SECONDS
