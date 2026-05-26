"""Write synthetic data to a sigproc filterbank (.fil) file"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from blimpy.io.sigproc import generate_sigproc_header

from blipss.constants import (
    SIGPROC_DATA_TYPE,
    SIGPROC_MACHINE_ID,
    SIGPROC_N_BITS,
    SIGPROC_N_IFS,
    SIGPROC_TELESCOPE_ID,
)
from blipss.models.simulate_data import OptionalHeaderParameters, SimulationProperties
from blipss.utils.general_utils import ensure_path_exists


def build_sigproc_header(
    sim: SimulationProperties,
    header_params: OptionalHeaderParameters,
) -> dict[str, Any]:
    """
    Construct the sigproc header dictionary from simulation and metadata parameters.

    Args:
        sim: Filterbank dimension and frequency/time axis parameters.
        header_params: Optional observational metadata (source name, start MJD).

    Returns:
        Dictionary of sigproc header key-value pairs.
    """
    return {
        "machine_id": SIGPROC_MACHINE_ID,
        "telescope_id": SIGPROC_TELESCOPE_ID,
        "data_type": SIGPROC_DATA_TYPE,
        "nbits": SIGPROC_N_BITS,
        "nifs": SIGPROC_N_IFS,
        "fch1": sim.fch1,
        "foff": sim.foff,
        "tsamp": sim.t_samp,
        "nchans": sim.n_channels,
        "nsamples": sim.n_samples,
        "source_name": header_params.source_name,
        "tstart": header_params.tstart,
    }


class _HeaderCarrier:
    """Minimal stand-in accepted by generate_sigproc_header."""

    header: dict[str, Any]
    __slots__ = ("header",)


def write_filterbank(
    data: npt.NDArray[np.floating],
    header: dict[str, Any],
    output_dir: Path,
    basename: str,
) -> None:
    """
    Write data and a sigproc header to a .fil filterbank file on disk.

    Args:
        data: Array of shape (n_samples, 1, n_channels) ready for serialisation.
        header: Sigproc header dictionary.
        output_dir: Directory in which the output file is created.
        basename: Filename stem; the .fil extension is appended automatically.
    """
    ensure_path_exists(output_dir)
    carrier = _HeaderCarrier()
    carrier.header = header
    output_path = output_dir / f"{basename}.fil"
    with open(output_path, "wb") as file_handle:
        file_handle.write(generate_sigproc_header(carrier))
        data.ravel().astype(np.float32, copy=False).tofile(file_handle)
