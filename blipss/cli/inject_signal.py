#!/usr/bin/env python
"""
Inject a periodic signal of known properties into a real-world filterbank data file.

Reads injection parameters from a YAML config file, loads the target filterbank,
injects one or more channel-wide boxcar pulse trains calibrated to the local
bandpass statistics, and writes the result as a .fil or .h5 file.

Example Usage
-------
    python -m blipss.cli.inject_signal --config config/inject_signal.yaml
"""

import logging
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from blipss.core.inject_signal import (
    compute_median_bandpass,
    compute_per_channel_std,
    extract_data_array,
    pack_data_into_waterfall,
)
from blipss.core.simulate_data import inject_periodic_signal
from blipss.io.read_blimpy_data import read_waterfall_file
from blipss.io.read_yaml_config import load_yaml_config
from blipss.io.write_filterbank import write_waterfall
from blipss.models.inject_signal import InjectSignalConfig
from blipss.utils.general_utils import ensure_path_exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d-%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

app = typer.Typer()


def run_inject_signal(cfg: InjectSignalConfig) -> None:
    """
    Orchestrate data loading, bandpass estimation, signal injection, and output writing.

    Args:
        cfg: Validated injection configuration.
    """
    input_cfg = cfg.input_data
    output_cfg = cfg.output
    injection_cfg = cfg.periodic_signal_injection
    limits = cfg.resource_limits

    if output_cfg.output_dir is None:
        raise ValueError("Output directory was not resolved by model_validator.")

    file_path = input_cfg.data_dir / input_cfg.datafile
    logger.info(f"Reading in: {file_path}")
    wat = read_waterfall_file(file_path, limits.mem_load)
    data, n_samples, tsamp = extract_data_array(wat)
    sample_times = np.arange(n_samples, dtype=np.float64) * tsamp
    logger.info("Waterfall data successfully loaded into memory.")

    logger.info("Computing median bandpass and per-channel standard deviation.")
    median_bp = compute_median_bandpass(data)
    std_perchan = compute_per_channel_std(data)
    logger.info("Per-channel statistics computed.")

    logger.info("Beginning periodic signal injection...")
    for i, channel in enumerate(injection_cfg.inject_channels):
        # Set the pulse peak pulse_snr sigma above the local bandpass level.
        amplitude = median_bp[channel] + injection_cfg.pulse_snr[i] * std_perchan[channel]
        # inject_periodic_signal adds its pulse_snr argument directly to on-pulse samples,
        # which equals the desired SNR only when the background has unit variance. For real
        # data, the noise variance is channel-dependent, so the noise-calibrated amplitude is
        # passed instead.
        inject_periodic_signal(
            data=data,
            sample_times=sample_times,
            channel=channel,
            period=injection_cfg.periods[i],
            duty_cycle=injection_cfg.duty_cycles[i],
            pulse_snr=amplitude,
            initial_phase=injection_cfg.initial_phase[i],
        )
        logger.info("Injected P = %.2f s signal into channel %d.", injection_cfg.periods[i], channel)
    logger.info("Signal injections completed.")

    wat = pack_data_into_waterfall(data, wat, n_samples)
    ensure_path_exists(output_cfg.output_dir)
    output_path = output_cfg.output_dir / f"{output_cfg.basename}{output_cfg.output_ext}"
    write_waterfall(wat, output_path)
    logger.info("%s written to disk.", output_path)


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", help="Path to YAML config file")],
) -> None:
    """Inject a fake periodic signal into a real-world filterbank data file."""
    t_start = time.time()

    logger.info("Loading config from YAML file: %s", config)
    raw_config = load_yaml_config(config)
    logger.info("Raw config loaded. Now, validating config entries...")

    validated_config = InjectSignalConfig(**raw_config)
    logger.info("Config validation completed.")

    run_inject_signal(validated_config)

    elapsed_minutes = (time.time() - t_start) / 60.0
    logger.info("Code run time = %.3f minutes", elapsed_minutes)


if __name__ == "__main__":
    app()
