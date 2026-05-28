#!/usr/bin/env python
"""
Produce a phase-resolved dynamic spectrum plot for a given folding period.

Reads folding parameters from a YAML config file, loads the target filterbank,
folds each spectral channel time series using the riptide FFA library, and
saves a grayscale phase-resolved spectrum plot.

Example Usage
-------
    python -m blipss.cli.compute_phase_resolved_ds --config config/compute_phase_resolved_ds.yaml
"""

import logging
import time
from pathlib import Path
from typing import Annotated

import typer

from blipss.core.compute_phase_resolved_ds import (
    align_band_orientation,
    clip_channels,
    extract_waterfall_metadata,
    fold_all_channels,
)
from blipss.core.inject_signal import extract_data_array
from blipss.io.read_blimpy_data import read_waterfall_file
from blipss.io.read_yaml_config import load_yaml_config
from blipss.models.compute_phase_resolved_ds import PhaseResolvedDsConfig
from blipss.plotting.plots import plot_phase_resolved_dynamic_spectrum
from blipss.utils.general_utils import ensure_path_exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d-%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

app = typer.Typer()


def run_compute_phase_resolved_ds(cfg: PhaseResolvedDsConfig) -> None:
    """
    Orchestrate data loading, band alignment, channel clipping, phase folding, and plot writing.

    Args:
        cfg: Validated phase-resolved dynamic spectrum configuration.
    """
    input_cfg = cfg.input_data
    output_cfg = cfg.output
    ch_cfg = cfg.channel_selection
    fold_cfg = cfg.phase_folding_parameters
    limits = cfg.resource_limits

    file_path = input_cfg.data_dir / input_cfg.datafile
    logger.info("Reading in: %s", file_path)
    wat = read_waterfall_file(file_path, limits.mem_load)
    data, _, tsamp = extract_data_array(wat)
    logger.info("Waterfall data successfully loaded into memory.")

    freqs_MHz, start_mjd, _ = extract_waterfall_metadata(wat)
    data, freqs_MHz = align_band_orientation(data, freqs_MHz, wat.header["foff"])
    data, freqs_MHz = clip_channels(data, freqs_MHz, ch_cfg.start_ch, ch_cfg.stop_ch)

    logger.info("Computing phase-resolved spectrum over %d channels.", len(data))
    phaseresolved_ds = fold_all_channels(
        data=data,
        tsamp=tsamp,
        period=fold_cfg.period,
        bins=fold_cfg.bins,
        do_deredden=fold_cfg.do_deredden,
        rmed_width=fold_cfg.rmed_width,
        n_workers=limits.n_workers,
    )
    logger.info("Done")

    if output_cfg.plot_dir is None:
        raise ValueError("Output plot directory was not resolved by model_validator.")

    ensure_path_exists(output_cfg.plot_dir)
    plot_name = str(output_cfg.plot_dir / f"{output_cfg.basename}_period{fold_cfg.period:.5f}")
    logger.info("Saving plot to disk.")
    plot_phase_resolved_dynamic_spectrum(
        phaseresolved_ds,
        freqs_MHz,
        fold_cfg.period,
        start_mjd,
        plot_name,
        output_cfg.plot_formats,
        output_cfg.use_latex,
    )
    logger.info(f"Phase-resolved dynamic spectrum saved with basepath (excluding file extension): {plot_name}")


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", help="Path to YAML config file")],
) -> None:
    """Produce a phase-resolved dynamic spectrum plot for a given folding period."""
    t_start = time.time()

    logger.info("Loading config from YAML file: %s", config)
    raw_config = load_yaml_config(config)
    logger.info("Raw config loaded. Now, validating config entries...")

    validated_config = PhaseResolvedDsConfig(**raw_config)
    logger.info("Config validation completed.")

    run_compute_phase_resolved_ds(validated_config)

    elapsed_minutes = (time.time() - t_start) / 60.0
    logger.info("Code run time = %.3f minutes", elapsed_minutes)


if __name__ == "__main__":
    app()
