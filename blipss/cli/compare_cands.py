#!/usr/bin/env python
"""
Compare candidate periods across N candidate CSV files and output one N-digit binary code per candidate.

In the binary code, "1" denotes detection and "0" denotes non-detection. Candidate detection in
file i is denoted by "1" in the i-th position of the code (read left to right).

Example usage
-------------
    python -m blipss.cli.compare_cands --config config/compare_cands.yaml
"""

import logging
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import typer

from blipss.core.compare_cands import filter_fundamental_candidates, group_candidates_by_channel
from blipss.io.read_candidates import read_candidates_csv
from blipss.io.read_yaml_config import load_yaml_config
from blipss.io.write_compared_candidates import write_compared_candidates_csv
from blipss.models.compare_cands import CompareCandsConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d-%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

app = typer.Typer()


def _read_and_filter_file(
    csv_path: Path,
    label: str,
    snr_threshold: float,
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.floating],
    npt.NDArray[np.uint],
    npt.NDArray[np.uint],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Read one candidate CSV file and retain fundamental candidates above its S/N threshold."""
    logger.info("Reading file: %s", csv_path.name)
    logger.info("Label = %s", label)
    logger.info("S/N threshold applied = %.2f", snr_threshold)
    channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, flags = read_candidates_csv(csv_path)
    channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs = filter_fundamental_candidates(
        channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs, flags, snr_threshold
    )
    logger.info("No. of candidates = %d", len(periods))
    return channels, radiofreqs, phase_bins, boxcar_widths, periods, snrs


def run_compare_cands(cfg: CompareCandsConfig) -> None:
    """Compare candidates across all configured files and write the merged output CSV."""
    files_cfg = cfg.candidate_files
    labels_cfg = cfg.on_off_classification
    snr_cutoffs = {"ON": labels_cfg.on_cutoff, "OFF": labels_cfg.off_cutoff}
    n_files = len(files_cfg.csv_list)
    logger.info("Total no. of input .csv files = %d", n_files)

    file_index: list[npt.NDArray[np.intp]] = []
    channels: list[npt.NDArray[np.intp]] = []
    radiofreqs: list[npt.NDArray[np.floating]] = []
    phase_bins: list[npt.NDArray[np.uint]] = []
    boxcar_widths: list[npt.NDArray[np.uint]] = []
    periods: list[npt.NDArray[np.floating]] = []
    snrs: list[npt.NDArray[np.floating]] = []

    for i, (csv_name, label) in enumerate(zip(files_cfg.csv_list, labels_cfg.labels, strict=True)):
        file_channels, file_radiofreqs, file_phase_bins, file_boxcar_widths, file_periods, file_snrs = (
            _read_and_filter_file(files_cfg.csv_dir / csv_name, label, snr_cutoffs[label])
        )
        file_index.append(np.full(len(file_channels), i, dtype=np.intp))
        channels.append(file_channels)
        radiofreqs.append(file_radiofreqs)
        phase_bins.append(file_phase_bins)
        boxcar_widths.append(file_boxcar_widths)
        periods.append(file_periods)
        snrs.append(file_snrs)

    logger.info("Final no. of candidates across all input files = %d", sum(len(p) for p in periods))
    logger.info(
        "Channel-wise grouping of candidate periods into clusters of radius %.2f ms using %d worker process(es)",
        cfg.candidate_grouping.cluster_radius * 1.0e3,
        cfg.candidate_grouping.n_jobs,
    )
    out_channels, out_radiofreqs, out_phase_bins, out_boxcar_widths, out_periods, out_snrs, out_codes = (
        group_candidates_by_channel(
            np.concatenate(file_index),
            np.concatenate(channels),
            np.concatenate(radiofreqs),
            np.concatenate(phase_bins),
            np.concatenate(boxcar_widths),
            np.concatenate(periods),
            np.concatenate(snrs),
            n_files,
            cfg.candidate_grouping.cluster_radius,
            n_jobs=cfg.candidate_grouping.n_jobs,
        )
    )

    output_dir = cfg.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{cfg.output.basename}_comparecands.csv"
    logger.info("Writing CSV: %s", output_csv)
    write_compared_candidates_csv(
        output_csv, out_channels, out_radiofreqs, out_phase_bins, out_boxcar_widths, out_periods, out_snrs, out_codes
    )


@app.command()
def main(config: Annotated[Path, typer.Option("--config", help="Path to YAML config file")]) -> None:
    """Compare candidate periods across N files and output one N-digit binary code per candidate."""
    t_start = time.time()

    logger.info("Loading config from YAML file: %s", config)
    raw_config = load_yaml_config(config)
    logger.info("Raw config loaded. Now validating config entries...")

    validated_config = CompareCandsConfig(**raw_config)
    logger.info("Config validation completed.")

    logger.info("Comparing candidates across files.")
    run_compare_cands(validated_config)
    logger.info("Candidate comparison completed.")

    elapsed_minutes = (time.time() - t_start) / 60.0
    logger.info("Code run time = %.3f minutes", elapsed_minutes)


if __name__ == "__main__":
    app()
