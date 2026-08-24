#!/usr/bin/env python
"""
Produce candidate verification plots combining periodograms, pulse profiles, and phase-time diagrams.

Example usage
-------------
    python -m blipss.cli.plot_cands --config config/plot_cands.yaml
"""

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import typer

from blipss.constants import SNR_PLOT_HEADROOM_FACTOR
from blipss.core.compute_phase_resolved_ds import align_band_orientation, extract_waterfall_metadata
from blipss.core.inject_signal import extract_data_array
from blipss.core.plot_cands import run_ffa_and_fold_channel, select_candidates_by_code
from blipss.io.read_blimpy_data import read_waterfall_file
from blipss.io.read_compared_candidates import read_compared_candidates_csv
from blipss.io.read_yaml_config import load_yaml_config
from blipss.models.plot_cands import PlotCandsConfig, PlottingParametersConfig
from blipss.models.run_ffa_search import FfaSearchConfig
from blipss.plotting.plots import candverf_plot
from blipss.utils.general_utils import ensure_path_exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d-%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

app = typer.Typer()


def _load_datafiles(
    data_dir: Path,
    datafile_list: Sequence[Path],
    mem_load: float,
) -> tuple[list[npt.NDArray[np.floating]], list[float], list[float]]:
    """
    Load each candidate data file into memory, aligned to ascending channel frequency order.

    Args:
        data_dir: Directory containing the input data files
        datafile_list: Data file names to load, in order
        mem_load: Maximum data volume (GB) to load into memory per file

    Returns:
        Tuple of (all_data, start_mjds, tsamps): per-file 2D dynamic spectra arrays of shape
        (n_channels, n_samples), start MJDs (UTC), and sampling intervals (s)
    """
    all_data: list[npt.NDArray[np.floating]] = []
    start_mjds: list[float] = []
    tsamps: list[float] = []
    for datafile in datafile_list:
        file_path = data_dir / datafile
        logger.info("Reading data from %s", file_path)
        wat = read_waterfall_file(file_path, mem_load)
        data, _, tsamp = extract_data_array(wat)
        freqs_MHz, start_mjd, _ = extract_waterfall_metadata(wat)
        data, _ = align_band_orientation(data, freqs_MHz, wat.header["foff"])
        all_data.append(data)
        start_mjds.append(start_mjd)
        tsamps.append(tsamp)
    return all_data, start_mjds, tsamps


def _produce_candidate_plot(
    chan: int,
    period: float,
    bins: int,
    code: str,
    all_data: Sequence[npt.NDArray[np.floating]],
    tsamps: Sequence[float],
    start_mjds: Sequence[float],
    beam_labels: Sequence[str],
    plot_dir: Path,
    basename: str,
    fold_cfg: FfaSearchConfig,
    plotting_cfg: PlottingParametersConfig,
) -> None:
    """
    Fold one candidate across all data files and save its verification plot.

    Args:
        chan: Spectral channel index of the candidate
        period: Candidate folding period (s)
        bins: Number of phase bins in the folded profile
        code: Binary detection code for the candidate
        all_data: Per-file 2D dynamic spectra arrays of shape (n_channels, n_samples)
        tsamps: Per-file sampling intervals (s)
        start_mjds: Per-file start MJDs (UTC)
        beam_labels: Per-file annotation labels
        plot_dir: Output directory for the plot
        basename: Basename of the output plot files
        fold_cfg: Fast folding algorithm parameters used to fold each file's time series
        plotting_cfg: Candidate selection and plot output options
    """
    detrended_ts = []
    periodograms = []
    max_snrs = []
    for data, tsamp in zip(all_data, tsamps, strict=True):
        dts, pgram = run_ffa_and_fold_channel(
            data[chan],
            tsamp,
            fold_cfg.min_period,
            fold_cfg.max_period,
            fold_cfg.fpmin,
            fold_cfg.bins_min,
            fold_cfg.bins_max,
            fold_cfg.ducy_max,
            fold_cfg.do_deredden,
            fold_cfg.rmed_width,
        )
        detrended_ts.append(dts)
        periodograms.append(pgram)
        max_snrs.append(pgram.snrs.max())

    snr_max = SNR_PLOT_HEADROOM_FACTOR * max(max_snrs)
    plot_name = str(plot_dir / f"{basename}_ch{chan}_code{code}_period{period:.5f}")
    candverf_plot(
        period,
        bins,
        detrended_ts,
        periodograms,
        beam_labels,
        start_mjds,
        snr_max,
        plotting_cfg.periodaxis_log,
        plot_name,
        plotting_cfg.plot_formats,
        plotting_cfg.use_latex,
    )


def run_plot_cands(cfg: PlotCandsConfig) -> None:
    """
    Select candidates by code, fold each across all data files, and save verification plots.

    Args:
        cfg: Validated plot_cands configuration.
    """
    input_cfg = cfg.input_data
    plotting_cfg = cfg.plotting_parameters
    fold_cfg = cfg.folding_search_parameters

    logger.info("Reading file: %s", cfg.candidate_file.csvfile)
    channels, _, phase_bins, _, periods, _, codes = read_compared_candidates_csv(cfg.candidate_file.csvfile)
    sel_channels, sel_periods, sel_bins, sel_codes = select_candidates_by_code(
        channels, periods, phase_bins, codes, plotting_cfg.codes_plot
    )
    n_cands = len(sel_channels)
    logger.info("No. of candidates selected for plotting = %d", n_cands)

    all_data, start_mjds, tsamps = _load_datafiles(
        input_cfg.data_dir, input_cfg.datafile_list, cfg.resource_limits.mem_load
    )

    if plotting_cfg.plot_dir is None:
        raise ValueError("Output plot directory was not resolved by model_validator.")
    ensure_path_exists(plotting_cfg.plot_dir)

    for i in range(n_cands):
        chan = int(sel_channels[i])
        period = float(sel_periods[i])
        bins = int(sel_bins[i])
        code = str(sel_codes[i])
        logger.info(
            "Working with candidate %d/%d: channel=%d, period=%.5f s, bins=%d, code=%s",
            i + 1,
            n_cands,
            chan,
            period,
            bins,
            code,
        )
        _produce_candidate_plot(
            chan,
            period,
            bins,
            code,
            all_data,
            tsamps,
            start_mjds,
            input_cfg.beam_labels,
            plotting_cfg.plot_dir,
            plotting_cfg.basename,
            fold_cfg,
            plotting_cfg,
        )


@app.command()
def main(config: Annotated[Path, typer.Option("--config", help="Path to YAML config file")]) -> None:
    """Produce candidate verification plots combining periodograms, pulse profiles, and phase-time diagrams."""
    t_start = time.time()

    logger.info("Loading config from YAML file: %s", config)
    raw_config = load_yaml_config(config)
    logger.info("Raw config loaded. Now validating config entries...")

    validated_config = PlotCandsConfig(**raw_config)
    logger.info("Config validation completed.")

    run_plot_cands(validated_config)

    elapsed_minutes = (time.time() - t_start) / 60.0
    logger.info("Code run time = %.3f minutes", elapsed_minutes)


if __name__ == "__main__":
    app()
