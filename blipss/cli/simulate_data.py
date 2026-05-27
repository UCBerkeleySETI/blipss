#!/usr/bin/env python
"""
Generate a synthetic filterbank file containing periodic signals on a Gaussian white noise background.

Reads simulation parameters from a YAML config file, injects one or more channel-wide
boxcar pulse trains at specified periods and duty cycles, and writes the result as a
32-bit sigproc filterbank (.fil) file.

Example Usage
-------
    python -m blipss.cli.simulate_data --config config/simulate_data.yaml
"""

import logging
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from blipss.core.simulate_data import (
    generate_white_noise_background,
    inject_periodic_signal,
    reshape_for_sigproc,
)
from blipss.io.read_yaml_config import load_yaml_config
from blipss.io.write_filterbank import build_sigproc_header, write_filterbank
from blipss.models.simulate_data import SimulateDataConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d-%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

app = typer.Typer()


def run_simulate_data(cfg: SimulateDataConfig) -> None:
    """Orchestrate noise generation, signal injection, and filterbank file writing."""
    sim = cfg.simulation_properties
    signal_injection = cfg.periodic_signal_injection
    header_params = cfg.optional_header_parameters
    output_cfg = cfg.output

    sample_times = np.arange(sim.n_samples) * sim.t_samp
    rng = np.random.default_rng(sim.seed)
    data = generate_white_noise_background(sim.n_channels, sim.n_samples, rng=rng)
    logger.info("Background Gaussian white noise data generated.")

    for i, channel in enumerate(signal_injection.inject_channels):
        inject_periodic_signal(
            data=data,
            sample_times=sample_times,
            channel=channel,
            period=signal_injection.periods[i],
            duty_cycle=signal_injection.duty_cycles[i],
            pulse_snr=signal_injection.pulse_snr[i],
            initial_phase=signal_injection.initial_phase[i],
        )
        logger.info("Injected P = %.2f s signal into channel %d.", signal_injection.periods[i], channel)

    data = reshape_for_sigproc(data)
    header = build_sigproc_header(sim, header_params)
    logger.info("Writing filterbank file to %s/%s.fil", output_cfg.output_dir, output_cfg.basename)
    write_filterbank(data, header, output_cfg.output_dir, output_cfg.basename)
    logger.info("Write operation successfully completed.")


@app.command()
def main(config: Annotated[Path, typer.Option("--config", help="Path to YAML config file")]) -> None:
    """Generate an artificial filterbank data set."""
    t_start = time.time()

    logger.info("Loading config from YAML file: %s", config)
    raw_config = load_yaml_config(config)
    logger.info("Raw config loaded. Now, validating config entries...")

    validated_config = SimulateDataConfig(**raw_config)
    logger.info("Config validation completed.")

    logger.info("Initiating simulation of artificial dataset")
    run_simulate_data(validated_config)
    logger.info("Dataset simulation completed.")

    elapsed_minutes = (time.time() - t_start) / 60.0
    logger.info("Code run time = %.3f minutes", elapsed_minutes)


if __name__ == "__main__":
    app()
