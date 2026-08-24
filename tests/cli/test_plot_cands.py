"""Unit tests for the candidate verification plot orchestrator in blipss.cli.plot_cands"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from blipss.cli.plot_cands import _load_datafiles, _produce_candidate_plot, main, run_plot_cands
from blipss.constants import SNR_PLOT_HEADROOM_FACTOR
from blipss.models.plot_cands import CandidateFileConfig, InputDataConfig, PlotCandsConfig, PlottingParametersConfig
from blipss.models.run_ffa_search import FfaSearchConfig

_MODULE = "blipss.cli.plot_cands"


def _make_config(
    *,
    data_dir: Path = Path("/fake/data"),
    datafile_list: list[Path] | None = None,
    csvfile: Path = Path("/fake/compared.csv"),
    codes_plot: list[str] | None = None,
    basename: str = "cands",
    plot_dir: Path = Path("/fake/plots"),
) -> PlotCandsConfig:
    """Return a valid PlotCandsConfig with sensible defaults, overridable per-test."""
    return PlotCandsConfig(
        input_data=InputDataConfig(data_dir=data_dir, datafile_list=datafile_list or [Path("a.fil")]),
        candidate_file=CandidateFileConfig(csvfile=csvfile),
        plotting_parameters=PlottingParametersConfig(
            codes_plot=codes_plot or ["10"], basename=basename, plot_dir=plot_dir
        ),
    )


# ---------------------------------------------------------------------------
# _load_datafiles
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.align_band_orientation")
@patch(f"{_MODULE}.extract_waterfall_metadata")
@patch(f"{_MODULE}.extract_data_array")
@patch(f"{_MODULE}.read_waterfall_file")
def test_load_datafiles_reads_each_file_and_returns_results_in_order(
    mock_read: MagicMock, mock_extract_data: MagicMock, mock_extract_meta: MagicMock, mock_align: MagicMock
) -> None:
    """_load_datafiles reads every data file under data_dir and returns per-file results in datafile_list order."""
    data_dir = Path("/fake/data")
    datafile_list = [Path("a.fil"), Path("b.fil")]
    mock_read.side_effect = [MagicMock(), MagicMock()]
    data_a: npt.NDArray[np.floating] = np.zeros((2, 10))
    data_b: npt.NDArray[np.floating] = np.ones((2, 10))
    mock_extract_data.side_effect = [(data_a, 10, 0.1), (data_b, 10, 0.2)]
    freqs = np.array([1000.0, 1001.0])
    mock_extract_meta.side_effect = [(freqs, 59000.0, 0.1), (freqs, 59001.0, 0.2)]
    mock_align.side_effect = [(data_a, freqs), (data_b, freqs)]

    all_data, start_mjds, tsamps = _load_datafiles(data_dir, datafile_list, 5.0)

    assert mock_read.call_args_list[0].args == (data_dir / "a.fil", 5.0)
    assert mock_read.call_args_list[1].args == (data_dir / "b.fil", 5.0)
    np.testing.assert_array_equal(all_data[0], data_a)
    np.testing.assert_array_equal(all_data[1], data_b)
    assert start_mjds == [59000.0, 59001.0]
    assert tsamps == [0.1, 0.2]


def test_load_datafiles_empty_list_returns_empty_results() -> None:
    """_load_datafiles returns empty lists when given an empty datafile_list, without reading anything."""
    all_data, start_mjds, tsamps = _load_datafiles(Path("/fake/data"), [], 5.0)
    assert all_data == []
    assert start_mjds == []
    assert tsamps == []


# ---------------------------------------------------------------------------
# _produce_candidate_plot
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.candverf_plot")
@patch(f"{_MODULE}.run_ffa_and_fold_channel")
def test_produce_candidate_plot_snr_max_is_headroom_times_largest_periodogram_peak(
    mock_fold: MagicMock, mock_plot: MagicMock
) -> None:
    """_produce_candidate_plot sets candverf_plot's max_snr to SNR_PLOT_HEADROOM_FACTOR times the largest per-file peak S/N."""
    pgram_a = MagicMock()
    pgram_a.snrs = np.array([1.0, 5.0])
    pgram_b = MagicMock()
    pgram_b.snrs = np.array([2.0, 9.0])
    mock_fold.side_effect = [(MagicMock(), pgram_a), (MagicMock(), pgram_b)]
    all_data = [np.zeros((3, 10)), np.zeros((3, 10))]

    _produce_candidate_plot(
        1,
        2.5,
        10,
        "10",
        all_data,
        [0.1, 0.2],
        [59000.0, 59001.0],
        ["A", "B"],
        Path("/fake/plots"),
        "cands",
        FfaSearchConfig(),
        PlottingParametersConfig(codes_plot=["10"], basename="cands", plot_dir=Path("/fake/plots")),
    )

    assert mock_plot.call_args.args[6] == pytest.approx(SNR_PLOT_HEADROOM_FACTOR * 9.0)


@patch(f"{_MODULE}.candverf_plot")
@patch(f"{_MODULE}.run_ffa_and_fold_channel")
def test_produce_candidate_plot_builds_plot_name_and_forwards_plot_options(
    mock_fold: MagicMock, mock_plot: MagicMock
) -> None:
    """_produce_candidate_plot names the plot from plot_dir/basename/channel/code/period and forwards plot options."""
    pgram = MagicMock()
    pgram.snrs = np.array([5.0])
    mock_fold.return_value = (MagicMock(), pgram)

    _produce_candidate_plot(
        3,
        2.5,
        10,
        "10",
        [np.zeros((5, 10))],
        [0.1],
        [59000.0],
        ["A"],
        Path("/fake/plots"),
        "cands",
        FfaSearchConfig(),
        PlottingParametersConfig(
            codes_plot=["10"],
            basename="cands",
            plot_dir=Path("/fake/plots"),
            plot_formats=[".pdf"],
            periodaxis_log=False,
            use_latex=True,
        ),
    )

    args = mock_plot.call_args.args
    assert args[0] == pytest.approx(2.5)
    assert args[1] == 10
    assert args[4] == ["A"]
    assert args[5] == [59000.0]
    assert args[7] is False
    assert args[8] == "/fake/plots/cands_ch3_code10_period2.50000"
    assert args[9] == [".pdf"]
    assert args[10] is True


@patch(f"{_MODULE}.candverf_plot")
@patch(f"{_MODULE}.run_ffa_and_fold_channel")
def test_produce_candidate_plot_forwards_fold_params_and_channel_data_per_file(
    mock_fold: MagicMock, mock_plot: MagicMock
) -> None:
    """_produce_candidate_plot folds each file's selected channel using the fold_cfg parameters and that file's tsamp."""
    pgram = MagicMock()
    pgram.snrs = np.array([5.0])
    mock_fold.return_value = (MagicMock(), pgram)
    data0 = np.arange(30, dtype=np.float64).reshape(3, 10)
    fold_cfg = FfaSearchConfig(
        min_period=1.0, max_period=50.0, fpmin=4, bins_min=8, bins_max=9, ducy_max=0.3, do_deredden=True, rmed_width=6.0
    )

    _produce_candidate_plot(
        2,
        2.5,
        10,
        "10",
        [data0],
        [0.05],
        [59000.0],
        ["A"],
        Path("/fake/plots"),
        "cands",
        fold_cfg,
        PlottingParametersConfig(codes_plot=["10"], basename="cands", plot_dir=Path("/fake/plots")),
    )

    call_args = mock_fold.call_args.args
    np.testing.assert_array_equal(call_args[0], data0[2])
    assert call_args[1] == pytest.approx(0.05)
    assert call_args[2] == pytest.approx(1.0)
    assert call_args[3] == pytest.approx(50.0)
    assert call_args[4] == 4
    assert call_args[5] == 8
    assert call_args[6] == 9
    assert call_args[7] == pytest.approx(0.3)
    assert call_args[8] is True
    assert call_args[9] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# run_plot_cands
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}._produce_candidate_plot")
@patch(f"{_MODULE}.ensure_path_exists")
@patch(f"{_MODULE}._load_datafiles")
@patch(f"{_MODULE}.select_candidates_by_code")
@patch(f"{_MODULE}.read_compared_candidates_csv")
def test_run_plot_cands_raises_when_plot_dir_is_none(
    mock_read_csv: MagicMock,
    mock_select: MagicMock,
    mock_load: MagicMock,
    mock_ensure: MagicMock,
    mock_produce: MagicMock,
) -> None:
    """run_plot_cands raises ValueError when plot_dir is None (guard for model_validator bypass)."""
    mock_read_csv.return_value = (
        np.array([0], dtype=np.intp),
        None,
        np.array([10], dtype=np.uint),
        None,
        np.array([1.0]),
        None,
        np.array(["10"]),
    )
    mock_select.return_value = (
        np.array([0], dtype=np.intp),
        np.array([1.0]),
        np.array([10], dtype=np.uint),
        np.array(["10"]),
    )
    mock_load.return_value = ([], [], [])
    cfg = MagicMock()
    cfg.plotting_parameters.plot_dir = None

    with pytest.raises(ValueError, match="plot directory was not resolved"):
        run_plot_cands(cfg)

    mock_ensure.assert_not_called()
    mock_produce.assert_not_called()


@patch(f"{_MODULE}._produce_candidate_plot")
@patch(f"{_MODULE}.ensure_path_exists")
@patch(f"{_MODULE}._load_datafiles")
@patch(f"{_MODULE}.select_candidates_by_code")
@patch(f"{_MODULE}.read_compared_candidates_csv")
def test_run_plot_cands_produces_one_plot_per_selected_candidate(
    mock_read_csv: MagicMock,
    mock_select: MagicMock,
    mock_load: MagicMock,
    mock_ensure: MagicMock,
    mock_produce: MagicMock,
    tmp_path: Path,
) -> None:
    """run_plot_cands creates the plot directory once and calls _produce_candidate_plot once per selected candidate."""
    mock_read_csv.return_value = (
        np.array([0, 1, 2], dtype=np.intp),
        None,
        np.array([10, 10, 10], dtype=np.uint),
        None,
        np.array([1.0, 2.0, 3.0]),
        None,
        np.array(["100", "010", "001"]),
    )
    mock_select.return_value = (
        np.array([0, 2], dtype=np.intp),
        np.array([1.0, 3.0]),
        np.array([10, 10], dtype=np.uint),
        np.array(["100", "001"]),
    )
    mock_load.return_value = ([np.zeros((3, 5))], [59000.0], [0.1])
    plot_dir = tmp_path / "plots"
    cfg = _make_config(plot_dir=plot_dir, codes_plot=["100", "001"])

    run_plot_cands(cfg)

    mock_ensure.assert_called_once_with(plot_dir)
    assert mock_produce.call_count == 2
    first_args = mock_produce.call_args_list[0].args
    assert (first_args[0], first_args[1], first_args[2], first_args[3]) == (0, pytest.approx(1.0), 10, "100")
    second_args = mock_produce.call_args_list[1].args
    assert (second_args[0], second_args[1], second_args[2], second_args[3]) == (2, pytest.approx(3.0), 10, "001")


@patch(f"{_MODULE}._produce_candidate_plot")
@patch(f"{_MODULE}.ensure_path_exists")
@patch(f"{_MODULE}._load_datafiles")
@patch(f"{_MODULE}.select_candidates_by_code")
@patch(f"{_MODULE}.read_compared_candidates_csv")
def test_run_plot_cands_no_selected_candidates_loads_data_but_plots_nothing(
    mock_read_csv: MagicMock,
    mock_select: MagicMock,
    mock_load: MagicMock,
    mock_ensure: MagicMock,
    mock_produce: MagicMock,
    tmp_path: Path,
) -> None:
    """run_plot_cands still loads the data files and creates the plot directory when no candidate codes match."""
    mock_read_csv.return_value = (
        np.array([0], dtype=np.intp),
        None,
        np.array([10], dtype=np.uint),
        None,
        np.array([1.0]),
        None,
        np.array(["100"]),
    )
    mock_select.return_value = (
        np.array([], dtype=np.intp),
        np.array([]),
        np.array([], dtype=np.uint),
        np.array([], dtype=str),
    )
    mock_load.return_value = ([np.zeros((1, 5))], [59000.0], [0.1])
    plot_dir = tmp_path / "plots"
    cfg = _make_config(plot_dir=plot_dir, codes_plot=["999"])

    run_plot_cands(cfg)

    mock_load.assert_called_once()
    mock_ensure.assert_called_once_with(plot_dir)
    mock_produce.assert_not_called()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.run_plot_cands")
@patch(f"{_MODULE}.load_yaml_config")
def test_main_validates_config_and_runs_plot_cands(
    mock_load_yaml: MagicMock, mock_run: MagicMock, tmp_path: Path
) -> None:
    """main loads the YAML config, validates it into a PlotCandsConfig, and hands it to run_plot_cands."""
    mock_load_yaml.return_value = {
        "input_data": {"data_dir": str(tmp_path), "datafile_list": ["a.fil"]},
        "candidate_file": {"csvfile": str(tmp_path / "compared.csv")},
        "plotting_parameters": {"codes_plot": ["10"], "basename": "cands"},
    }
    main(config=Path("config/plot_cands.yaml"))

    mock_load_yaml.assert_called_once_with(Path("config/plot_cands.yaml"))
    cfg: PlotCandsConfig = mock_run.call_args.args[0]
    assert isinstance(cfg, PlotCandsConfig)
    assert cfg.input_data.data_dir == tmp_path
    assert cfg.plotting_parameters.plot_dir == tmp_path
