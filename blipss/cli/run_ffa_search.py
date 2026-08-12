#!/usr/bin/env python
"""
Run channel-wise FFA on a set of input filterbank files, flag harmonics, and write one CSV of candidates per file.

Example usage
-------------
    python -m blipss.cli.run_ffa_search --config config/run_ffa_search.yaml
"""

import logging
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import typer

from blipss.constants import SIGPROC_N_IFS
from blipss.core.compute_phase_resolved_ds import align_band_orientation, clip_channels, extract_waterfall_metadata
from blipss.core.period_finding import search_all_channels
from blipss.io.read_blimpy_data import read_waterfall_file
from blipss.io.read_yaml_config import load_yaml_config
from blipss.io.write_candidates import write_candidates_csv
from blipss.models.run_ffa_search import BlipssConfig
from blipss.plotting.plots import scatterplot_period_radiofreq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d-%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

app = typer.Typer()


def _resolve_stop_ch(stop_ch: int | None, n_channels: int) -> int:
    """Return a concrete stop_ch, defaulting to n_channels when None."""
    return n_channels if stop_ch is None else stop_ch


def _make_scatter_plot(
    output_dir: Path,
    stem: str,
    freqs_MHz: npt.NDArray[np.floating],
    cand_channels: npt.NDArray[np.intp],
    cand_periods: npt.NDArray[np.floating],
    cand_snrs: npt.NDArray[np.floating],
    cand_flags: npt.NDArray[np.str_],
    min_period: float,
    max_period: float,
    plot_formats: list[str],
    start_ch: int,
    use_latex: bool,
) -> None:
    """Produce a scatter plot of candidates in the period vs. radio frequency plane."""
    cand_radiofreqs = freqs_MHz[cand_channels - start_ch]
    sort_asc_idx = np.argsort(cand_snrs)
    scatterplot_period_radiofreq(
        cand_periods[sort_asc_idx],
        cand_radiofreqs[sort_asc_idx],
        cand_snrs[sort_asc_idx],
        cand_flags[sort_asc_idx],
        str(output_dir / stem),
        min_period,
        max_period,
        float(freqs_MHz[0]),
        float(freqs_MHz[-1]),
        plot_formats,
        use_latex,
    )


def _process_single_file(datafile: Path, cfg: BlipssConfig) -> None:
    """Load one filterbank file, run channel-wise FFA, and write outputs to disk."""
    stem = datafile.stem
    output_dir = cfg.output.output_dir
    if output_dir is None:
        raise ValueError("output.output_dir must be resolved before use")
    logger.info("Processing %s", datafile.name)

    wat = read_waterfall_file(datafile, cfg.resources.mem_load)
    freqs_MHz, _, tsamp = extract_waterfall_metadata(wat)
    data = wat.data[:, SIGPROC_N_IFS - 1, :].T
    # data.shape = (No. of channels, No. of time samples)
    data, freqs_MHz = align_band_orientation(data, freqs_MHz, wat.header["foff"])
    start_ch = cfg.input.start_ch
    stop_ch = _resolve_stop_ch(cfg.input.stop_ch, data.shape[0])
    data, freqs_MHz = clip_channels(data, freqs_MHz, start_ch=start_ch, stop_ch=stop_ch)

    ffa = cfg.ffa_search
    cand_channels, cand_periods, cand_snrs, cand_phase_bins, cand_boxcar_widths, cand_flags = search_all_channels(
        data,
        start_ch,
        tsamp,
        ffa.min_period,
        ffa.max_period,
        ffa.fpmin,
        ffa.bins_min,
        ffa.bins_max,
        ffa.ducy_max,
        ffa.do_deredden,
        ffa.rmed_width,
        ffa.snr_threshold,
        ffa.epsilon_fof,
        ffa.epsilon_harmonic,
        n_workers=cfg.resources.n_workers,
    )

    n_cands = len(cand_periods)
    logger.info("%d candidates found in %s", n_cands, datafile.name)

    if n_cands == 0:
        return

    output_csv = output_dir / f"{stem}_cands.csv"
    logger.info("Writing CSV: %s", output_csv)
    write_candidates_csv(
        output_csv,
        freqs_MHz,
        cand_channels,
        cand_periods,
        cand_snrs,
        cand_phase_bins,
        cand_boxcar_widths,
        cand_flags,
        start_ch,
    )

    if cfg.plotting.do_plot:
        logger.info("Producing scatter plot for %s", stem)
        _make_scatter_plot(
            output_dir,
            stem,
            freqs_MHz,
            cand_channels,
            cand_periods,
            cand_snrs,
            cand_flags,
            ffa.min_period,
            ffa.max_period,
            cfg.plotting.plot_formats,
            start_ch,
            cfg.plotting.use_latex,
        )


def run_blipss(cfg: BlipssConfig) -> None:
    """Orchestrate per-file FFA search and output writing for all files in the config."""
    output_dir = cfg.output.output_dir
    if output_dir is None:
        raise ValueError("output.output_dir must be resolved before use")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Total input files: %d", len(cfg.input.input_file_list))
    for datafile in cfg.input.input_file_list:
        _process_single_file(datafile, cfg)


@app.command()
def main(config: Annotated[Path, typer.Option("--config", help="Path to YAML config file")]) -> None:
    """Run channel-wise FFA period search on a set of filterbank files."""
    t_start = time.time()

    logger.info("Loading config from YAML file: %s", config)
    raw_config = load_yaml_config(config)
    logger.info("Raw config loaded. Now validating config entries...")

    validated_config = BlipssConfig(**raw_config)
    logger.info("Config validation completed.")

    logger.info("Starting FFA period search.")
    run_blipss(validated_config)
    logger.info("FFA period search completed.")

    elapsed_minutes = (time.time() - t_start) / 60.0
    logger.info("Code run time = %.3f minutes", elapsed_minutes)


if __name__ == "__main__":
    app()
