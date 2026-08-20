"""Unit tests for the candidate comparison orchestrator in blipss.cli.compare_cands"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt

from blipss.cli.compare_cands import _read_and_filter_file, main, run_compare_cands
from blipss.models.compare_cands import (
    CandidateFilesConfig,
    CandidateGroupingConfig,
    CompareCandsConfig,
    OnOffClassificationConfig,
    OutputConfig,
)

_MODULE = "blipss.cli.compare_cands"


def _make_config(
    *,
    csv_dir: Path = Path("/fake/csvs"),
    csv_list: list[Path] | None = None,
    labels: list[str] | None = None,
    on_cutoff: float = 7.0,
    off_cutoff: float = 6.0,
    output_dir: Path = Path("/fake/out"),
    basename: str = "compared",
    n_jobs: int = 1,
) -> CompareCandsConfig:
    """Return a valid CompareCandsConfig with sensible defaults, overridable per-test."""
    return CompareCandsConfig(
        candidate_files=CandidateFilesConfig(csv_dir=csv_dir, csv_list=csv_list or [Path("a.csv"), Path("b.csv")]),
        on_off_classification=OnOffClassificationConfig(
            labels=labels or ["ON", "OFF"], on_cutoff=on_cutoff, off_cutoff=off_cutoff
        ),
        output=OutputConfig(basename=basename, output_dir=output_dir),
        candidate_grouping=CandidateGroupingConfig(n_jobs=n_jobs),
    )


# ---------------------------------------------------------------------------
# _read_and_filter_file
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.filter_fundamental_candidates")
@patch(f"{_MODULE}.read_candidates_csv")
def test_read_and_filter_file_reads_then_filters_by_given_threshold(
    mock_read: MagicMock, mock_filter: MagicMock
) -> None:
    """_read_and_filter_file forwards the read candidate arrays and threshold to filter_fundamental_candidates."""
    channels = np.array([0], dtype=np.intp)
    radiofreqs = np.array([1000.0])
    phase_bins = np.array([10], dtype=np.uint)
    boxcar_widths = np.array([2], dtype=np.uint)
    periods = np.array([1.5])
    snrs = np.array([8.0])
    flags = np.array(["F"])
    mock_read.return_value = (channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, flags)
    filtered = (channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs)
    mock_filter.return_value = filtered

    result = _read_and_filter_file(Path("/fake/csvs/a.csv"), "ON", 7.0)

    mock_read.assert_called_once_with(Path("/fake/csvs/a.csv"))
    mock_filter.assert_called_once_with(channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, flags, 7.0)
    assert result == filtered


# ---------------------------------------------------------------------------
# run_compare_cands
# ---------------------------------------------------------------------------


def _candidate_tuple(
    n: int, value: float
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.floating],
    npt.NDArray[np.uint],
    npt.NDArray[np.uint],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Build a filtered-candidate tuple of length n, filled with a constant value, for mock return values."""
    return (
        np.full(n, 0, dtype=np.intp),
        np.full(n, value),
        np.full(n, 10, dtype=np.uint),
        np.full(n, 2, dtype=np.uint),
        np.full(n, value),
        np.full(n, value),
    )


@patch(f"{_MODULE}.write_compared_candidates_csv")
@patch(f"{_MODULE}.group_candidates_by_channel")
@patch(f"{_MODULE}._read_and_filter_file")
def test_run_compare_cands_applies_on_and_off_snr_cutoffs_per_file(
    mock_read_filter: MagicMock, mock_group: MagicMock, mock_write: MagicMock, tmp_path: Path
) -> None:
    """run_compare_cands passes each file's pointing-specific S/N cutoff to _read_and_filter_file."""
    mock_read_filter.side_effect = [_candidate_tuple(1, 1.0), _candidate_tuple(1, 2.0)]
    mock_group.return_value = (
        np.array([0], dtype=np.intp),
        np.array([1000.0]),
        np.array([10], dtype=np.uint),
        np.array([2], dtype=np.uint),
        np.array([1.0]),
        np.array([8.0]),
        np.array(["11"]),
    )
    cfg = _make_config(csv_dir=tmp_path, output_dir=tmp_path / "out", on_cutoff=7.0, off_cutoff=6.0)

    run_compare_cands(cfg)

    assert mock_read_filter.call_args_list[0].args == (tmp_path / "a.csv", "ON", 7.0)
    assert mock_read_filter.call_args_list[1].args == (tmp_path / "b.csv", "OFF", 6.0)
    mock_write.assert_called_once()


@patch(f"{_MODULE}.write_compared_candidates_csv")
@patch(f"{_MODULE}.group_candidates_by_channel")
@patch(f"{_MODULE}._read_and_filter_file")
def test_run_compare_cands_tags_candidates_with_source_file_index(
    mock_read_filter: MagicMock, mock_group: MagicMock, mock_write: MagicMock, tmp_path: Path
) -> None:
    """run_compare_cands concatenates candidates with a file_index array matching each file's position."""
    mock_read_filter.side_effect = [_candidate_tuple(2, 1.0), _candidate_tuple(1, 2.0)]
    mock_group.return_value = (
        np.array([], dtype=np.intp),
        np.array([]),
        np.array([], dtype=np.uint),
        np.array([], dtype=np.uint),
        np.array([]),
        np.array([]),
        np.array([], dtype=np.str_),
    )
    cfg = _make_config(csv_dir=tmp_path, output_dir=tmp_path / "out")

    run_compare_cands(cfg)

    file_index_arg: npt.NDArray[np.intp] = mock_group.call_args.args[0]
    np.testing.assert_array_equal(file_index_arg, [0, 0, 1])
    assert mock_group.call_args.args[7] == 2  # n_files


@patch(f"{_MODULE}.write_compared_candidates_csv")
@patch(f"{_MODULE}.group_candidates_by_channel")
@patch(f"{_MODULE}._read_and_filter_file")
def test_run_compare_cands_creates_output_dir_and_writes_named_csv(
    mock_read_filter: MagicMock, mock_group: MagicMock, mock_write: MagicMock, tmp_path: Path
) -> None:
    """run_compare_cands creates output_dir and writes to <output_dir>/<basename>_comparecands.csv."""
    mock_read_filter.side_effect = [_candidate_tuple(1, 1.0), _candidate_tuple(1, 2.0)]
    grouped = (
        np.array([0], dtype=np.intp),
        np.array([1000.0]),
        np.array([10], dtype=np.uint),
        np.array([2], dtype=np.uint),
        np.array([1.0]),
        np.array([8.0]),
        np.array(["11"]),
    )
    mock_group.return_value = grouped
    output_dir = tmp_path / "nested" / "out"
    cfg = _make_config(csv_dir=tmp_path, output_dir=output_dir, basename="B06on_B03off")

    run_compare_cands(cfg)

    assert output_dir.is_dir()
    mock_write.assert_called_once_with(output_dir / "B06on_B03off_comparecands.csv", *grouped)


@patch(f"{_MODULE}.write_compared_candidates_csv")
@patch(f"{_MODULE}.group_candidates_by_channel")
@patch(f"{_MODULE}._read_and_filter_file")
def test_run_compare_cands_forwards_grouping_parameters(
    mock_read_filter: MagicMock, mock_group: MagicMock, mock_write: MagicMock, tmp_path: Path
) -> None:
    """run_compare_cands forwards cluster_radius and n_jobs from candidate_grouping to group_candidates_by_channel."""
    mock_read_filter.side_effect = [_candidate_tuple(1, 1.0), _candidate_tuple(1, 2.0)]
    mock_group.return_value = (
        np.array([], dtype=np.intp),
        np.array([]),
        np.array([], dtype=np.uint),
        np.array([], dtype=np.uint),
        np.array([]),
        np.array([]),
        np.array([], dtype=np.str_),
    )
    cfg = _make_config(csv_dir=tmp_path, output_dir=tmp_path / "out", n_jobs=3)
    cfg = cfg.model_copy(update={"candidate_grouping": CandidateGroupingConfig(cluster_radius=2.0e-3, n_jobs=3)})

    run_compare_cands(cfg)

    assert mock_group.call_args.args[8] == 2.0e-3
    assert mock_group.call_args.kwargs["n_jobs"] == 3


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.run_compare_cands")
@patch(f"{_MODULE}.load_yaml_config")
def test_main_validates_config_and_runs_comparison(mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path) -> None:
    """main loads the YAML config, validates it into a CompareCandsConfig, and hands it to run_compare_cands."""
    mock_load.return_value = {
        "candidate_files": {"csv_dir": str(tmp_path), "csv_list": ["a.csv", "b.csv"]},
        "on_off_classification": {"labels": ["ON", "OFF"]},
        "output": {"basename": "out", "output_dir": str(tmp_path / "out")},
    }

    main(config=Path("config/compare_cands.yaml"))

    mock_load.assert_called_once_with(Path("config/compare_cands.yaml"))
    cfg: CompareCandsConfig = mock_run.call_args.args[0]
    assert isinstance(cfg, CompareCandsConfig)
    assert cfg.candidate_files.csv_dir == tmp_path
    assert cfg.on_off_classification.labels == ["ON", "OFF"]
