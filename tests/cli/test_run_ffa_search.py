"""Unit tests for the FFA search orchestrator in blipss.cli.run_ffa_search"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from blipss.cli.run_ffa_search import (
    _make_scatter_plot,
    _process_single_file,
    _resolve_stop_ch,
    main,
    run_blipss,
)
from blipss.models.run_ffa_search import (
    BlipssConfig,
    FfaSearchConfig,
    InputConfig,
    OutputConfig,
    PlottingConfig,
    ResourceConfig,
)

_MODULE = "blipss.cli.run_ffa_search"

_N_CHANNELS = 4
_N_SAMPLES = 32
_FREQS_MHZ: npt.NDArray[np.floating] = np.array([1000.0, 1001.0, 1002.0, 1003.0])
_TSAMP = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    data_dir: Path = Path("/fake/data"),
    input_file_list: list[Path] | None = None,
    start_ch: int = 0,
    stop_ch: int | None = None,
    output_dir: Path = Path("/fake/out"),
    do_plot: bool = False,
    plot_formats: list[str] | None = None,
    use_latex: bool = False,
    n_workers: int | None = 1,
) -> BlipssConfig:
    """Return a valid BlipssConfig with sensible defaults, overridable per-test."""
    return BlipssConfig(
        input=InputConfig(
            data_dir=data_dir,
            input_file_list=input_file_list if input_file_list is not None else [data_dir / "obs.fil"],
            start_ch=start_ch,
            stop_ch=stop_ch,
        ),
        output=OutputConfig(output_dir=output_dir),
        plotting=PlottingConfig(
            do_plot=do_plot,
            plot_formats=plot_formats or [".png"],
            use_latex=use_latex,
        ),
        ffa_search=FfaSearchConfig(),
        resources=ResourceConfig(n_workers=n_workers),
    )


def _configure_pipeline_mocks(
    mock_read: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    mock_search: MagicMock,
    *,
    n_candidates: int = 1,
) -> npt.NDArray[np.floating]:
    """Configure pipeline mocks to return properly-typed values matching their real signatures."""
    data: npt.NDArray[np.floating] = np.zeros((_N_CHANNELS, _N_SAMPLES))
    wat = MagicMock()
    wat.data = np.zeros((_N_SAMPLES, 1, _N_CHANNELS))
    wat.header = {"foff": -1.0}
    mock_read.return_value = wat
    mock_extract_meta.return_value = (_FREQS_MHZ, 59000.0, _TSAMP)
    mock_align.return_value = (data, _FREQS_MHZ)
    mock_clip.return_value = (data, _FREQS_MHZ)
    mock_search.return_value = (
        np.arange(n_candidates, dtype=np.intp),
        np.full(n_candidates, 20.0),
        np.full(n_candidates, 9.0),
        np.full(n_candidates, 10, dtype=np.uint),
        np.full(n_candidates, 2, dtype=np.uint),
        np.array(["F"] * n_candidates),
    )
    return data


# ---------------------------------------------------------------------------
# _resolve_stop_ch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stop_ch", "n_channels", "expected"),
    [(None, 64, 64), (16, 64, 16)],
    ids=["none_defaults_to_n_channels", "explicit_value_preserved"],
)
def test_resolve_stop_ch(stop_ch: int | None, n_channels: int, expected: int) -> None:
    """_resolve_stop_ch returns n_channels when stop_ch is None and the given value otherwise."""
    assert _resolve_stop_ch(stop_ch, n_channels) == expected


# ---------------------------------------------------------------------------
# _make_scatter_plot
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.scatterplot_period_radiofreq")
def test_make_scatter_plot_sorts_candidates_by_ascending_snr(mock_scatter: MagicMock) -> None:
    """_make_scatter_plot passes candidate arrays reordered by ascending S/N."""
    _make_scatter_plot(
        Path("/fake/out"),
        "obs",
        _FREQS_MHZ,
        np.array([0, 1, 2], dtype=np.intp),
        np.array([10.0, 20.0, 30.0]),
        np.array([12.0, 8.0, 9.0]),
        np.array(["F", "H", "S"]),
        10.0,
        100.0,
        [".png"],
        0,
        False,
    )
    args = mock_scatter.call_args.args
    np.testing.assert_allclose(args[0], [20.0, 30.0, 10.0])
    np.testing.assert_allclose(args[1], [1001.0, 1002.0, 1000.0])
    np.testing.assert_allclose(args[2], [8.0, 9.0, 12.0])
    np.testing.assert_array_equal(args[3], ["H", "S", "F"])


@patch(f"{_MODULE}.scatterplot_period_radiofreq")
def test_make_scatter_plot_forwards_plot_name_and_axis_limits(mock_scatter: MagicMock) -> None:
    """_make_scatter_plot builds the plot basename from output_dir/stem and forwards period and frequency limits."""
    _make_scatter_plot(
        Path("/fake/out"),
        "obs",
        _FREQS_MHZ,
        np.array([102], dtype=np.intp),
        np.array([20.0]),
        np.array([9.0]),
        np.array(["F"]),
        10.0,
        70.0,
        [".png", ".pdf"],
        100,
        True,
    )
    args = mock_scatter.call_args.args
    np.testing.assert_allclose(args[1], [1002.0])
    assert args[4] == "/fake/out/obs"
    assert args[5] == 10.0
    assert args[6] == 70.0
    assert args[7] == 1000.0
    assert args[8] == 1003.0
    assert args[9] == [".png", ".pdf"]
    assert args[10] is True


# ---------------------------------------------------------------------------
# _process_single_file
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.write_candidates_csv")
@patch(f"{_MODULE}.search_all_channels")
@patch(f"{_MODULE}.clip_channels")
@patch(f"{_MODULE}.align_band_orientation")
@patch(f"{_MODULE}.extract_waterfall_metadata")
@patch(f"{_MODULE}.read_waterfall_file")
def test_process_single_file_raises_when_output_dir_is_none(
    mock_read: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    mock_search: MagicMock,
    mock_write: MagicMock,
) -> None:
    """_process_single_file raises ValueError when output_dir is None (guard for model_validator bypass)."""
    cfg = MagicMock()
    cfg.output.output_dir = None

    with pytest.raises(ValueError, match=r"output\.output_dir must be resolved"):
        _process_single_file(Path("/fake/data/obs.fil"), cfg)

    mock_read.assert_not_called()
    mock_extract_meta.assert_not_called()
    mock_align.assert_not_called()
    mock_clip.assert_not_called()
    mock_search.assert_not_called()
    mock_write.assert_not_called()


@patch(f"{_MODULE}.write_candidates_csv")
@patch(f"{_MODULE}.search_all_channels")
@patch(f"{_MODULE}.clip_channels")
@patch(f"{_MODULE}.align_band_orientation")
@patch(f"{_MODULE}.extract_waterfall_metadata")
@patch(f"{_MODULE}.read_waterfall_file")
def test_process_single_file_forwards_search_parameters_from_config(
    mock_read: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    mock_search: MagicMock,
    mock_write: MagicMock,
) -> None:
    """_process_single_file forwards the sampling time, channel offset, and FFA settings to search_all_channels."""
    cfg = _make_config(start_ch=2, n_workers=1)
    data = _configure_pipeline_mocks(mock_read, mock_extract_meta, mock_align, mock_clip, mock_search)
    _process_single_file(Path("/fake/data/obs.fil"), cfg)

    args = mock_search.call_args.args
    np.testing.assert_array_equal(args[0], data)
    assert args[1] == 2
    assert args[2] == pytest.approx(_TSAMP)
    assert args[3] == cfg.ffa_search.min_period
    assert args[4] == cfg.ffa_search.max_period
    assert mock_search.call_args.kwargs["n_workers"] == 1
    mock_read.assert_called_once_with(Path("/fake/data/obs.fil"), cfg.resources.mem_load)
    mock_write.assert_called_once()


@patch(f"{_MODULE}.write_candidates_csv")
@patch(f"{_MODULE}.search_all_channels")
@patch(f"{_MODULE}.clip_channels")
@patch(f"{_MODULE}.align_band_orientation")
@patch(f"{_MODULE}.extract_waterfall_metadata")
@patch(f"{_MODULE}.read_waterfall_file")
def test_process_single_file_clips_channels_with_resolved_stop_ch(
    mock_read: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    mock_search: MagicMock,
    mock_write: MagicMock,
) -> None:
    """_process_single_file resolves a null stop_ch to the channel count before clipping."""
    cfg = _make_config(start_ch=1, stop_ch=None)
    _configure_pipeline_mocks(mock_read, mock_extract_meta, mock_align, mock_clip, mock_search)
    _process_single_file(Path("/fake/data/obs.fil"), cfg)

    assert mock_clip.call_args.kwargs == {"start_ch": 1, "stop_ch": _N_CHANNELS}
    mock_search.assert_called_once()
    mock_write.assert_called_once()


@patch(f"{_MODULE}.write_candidates_csv")
@patch(f"{_MODULE}.search_all_channels")
@patch(f"{_MODULE}.clip_channels")
@patch(f"{_MODULE}.align_band_orientation")
@patch(f"{_MODULE}.extract_waterfall_metadata")
@patch(f"{_MODULE}.read_waterfall_file")
def test_process_single_file_writes_csv_named_after_input_stem(
    mock_read: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    mock_search: MagicMock,
    mock_write: MagicMock,
) -> None:
    """_process_single_file writes candidates to <output_dir>/<stem>_cands.csv."""
    cfg = _make_config(output_dir=Path("/fake/out"), start_ch=3)
    _configure_pipeline_mocks(mock_read, mock_extract_meta, mock_align, mock_clip, mock_search, n_candidates=2)
    _process_single_file(Path("/fake/data/target.fil"), cfg)

    args = mock_write.call_args.args
    assert args[0] == Path("/fake/out/target_cands.csv")
    np.testing.assert_allclose(args[1], _FREQS_MHZ)
    assert args[8] == 3


@patch(f"{_MODULE}._make_scatter_plot")
@patch(f"{_MODULE}.write_candidates_csv")
@patch(f"{_MODULE}.search_all_channels")
@patch(f"{_MODULE}.clip_channels")
@patch(f"{_MODULE}.align_band_orientation")
@patch(f"{_MODULE}.extract_waterfall_metadata")
@patch(f"{_MODULE}.read_waterfall_file")
def test_process_single_file_no_candidates_skips_csv_and_plot(
    mock_read: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    mock_search: MagicMock,
    mock_write: MagicMock,
    mock_plot: MagicMock,
) -> None:
    """_process_single_file returns early without writing a CSV or plot when no candidates are found."""
    cfg = _make_config(do_plot=True)
    _configure_pipeline_mocks(mock_read, mock_extract_meta, mock_align, mock_clip, mock_search, n_candidates=0)
    _process_single_file(Path("/fake/data/obs.fil"), cfg)

    mock_search.assert_called_once()
    mock_write.assert_not_called()
    mock_plot.assert_not_called()


@pytest.mark.parametrize("do_plot", [True, False], ids=["plotting_enabled", "plotting_disabled"])
@patch(f"{_MODULE}._make_scatter_plot")
@patch(f"{_MODULE}.write_candidates_csv")
@patch(f"{_MODULE}.search_all_channels")
@patch(f"{_MODULE}.clip_channels")
@patch(f"{_MODULE}.align_band_orientation")
@patch(f"{_MODULE}.extract_waterfall_metadata")
@patch(f"{_MODULE}.read_waterfall_file")
def test_process_single_file_plots_only_when_do_plot_is_true(
    mock_read: MagicMock,
    mock_extract_meta: MagicMock,
    mock_align: MagicMock,
    mock_clip: MagicMock,
    mock_search: MagicMock,
    mock_write: MagicMock,
    mock_plot: MagicMock,
    do_plot: bool,
) -> None:
    """_process_single_file produces a scatter plot only when plotting.do_plot is enabled."""
    cfg = _make_config(do_plot=do_plot, plot_formats=[".pdf"], use_latex=True, start_ch=5)
    _configure_pipeline_mocks(mock_read, mock_extract_meta, mock_align, mock_clip, mock_search)
    _process_single_file(Path("/fake/data/obs.fil"), cfg)

    assert mock_plot.call_count == int(do_plot)
    if do_plot:
        args = mock_plot.call_args.args
        assert args[0] == Path("/fake/out")
        assert args[1] == "obs"
        assert args[9] == [".pdf"]
        assert args[10] == 5
        assert args[11] is True
    mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# run_blipss
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}._process_single_file")
def test_run_blipss_raises_when_output_dir_is_none(mock_process: MagicMock) -> None:
    """run_blipss raises ValueError when output_dir is None (guard for model_validator bypass)."""
    cfg = MagicMock()
    cfg.output.output_dir = None

    with pytest.raises(ValueError, match=r"output\.output_dir must be resolved"):
        run_blipss(cfg)

    mock_process.assert_not_called()


@patch(f"{_MODULE}._process_single_file")
def test_run_blipss_creates_output_dir_and_processes_every_file(mock_process: MagicMock, tmp_path: Path) -> None:
    """run_blipss creates the output directory and processes each input file exactly once."""
    output_dir: Path = tmp_path / "nested" / "outputs"
    files: list[Path] = [Path("/fake/data/a.fil"), Path("/fake/data/b.fil")]
    cfg = _make_config(input_file_list=files, output_dir=output_dir)
    run_blipss(cfg)

    assert output_dir.is_dir()
    assert [call.args[0] for call in mock_process.call_args_list] == files


@patch(f"{_MODULE}._process_single_file")
def test_run_blipss_empty_file_list_processes_nothing(mock_process: MagicMock, tmp_path: Path) -> None:
    """run_blipss creates the output directory but processes no files when the input list is empty."""
    cfg = _make_config(input_file_list=[], output_dir=tmp_path / "outputs")
    run_blipss(cfg)

    assert (tmp_path / "outputs").is_dir()
    mock_process.assert_not_called()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


@patch(f"{_MODULE}.run_blipss")
@patch(f"{_MODULE}.load_yaml_config")
def test_main_validates_config_and_runs_search(mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path) -> None:
    """main loads the YAML config, validates it into a BlipssConfig, and hands it to run_blipss."""
    mock_load.return_value = {
        "input": {"data_dir": str(tmp_path), "start_ch": None},
        "output": {"output_dir": None},
    }
    main(config=Path("config/run_ffa_search.yaml"))

    mock_load.assert_called_once_with(Path("config/run_ffa_search.yaml"))
    cfg: BlipssConfig = mock_run.call_args.args[0]
    assert isinstance(cfg, BlipssConfig)
    assert cfg.input.data_dir == tmp_path
    assert cfg.output.output_dir == tmp_path
