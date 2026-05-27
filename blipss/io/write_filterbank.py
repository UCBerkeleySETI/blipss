"""
Filterbank write utilities for both synthetic and real-data pipelines.

Two write paths are provided:

- ``write_filterbank``: writes a raw numpy array produced by the *simulate_data*
  pipeline directly to a ``.fil`` file by serialising a sigproc header followed
  by the raw float32 samples.

- ``write_waterfall``: writes a blimpy ``Waterfall`` object produced by the
  *inject_signal* pipeline to either a ``.fil`` or ``.h5`` file, delegating to
  the appropriate blimpy serialiser based on the output file extension.

Helper utilities
----------------
``build_sigproc_header``
    Constructs the sigproc header dict required by ``write_filterbank``.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from blimpy import Waterfall
from blimpy.io.sigproc import generate_sigproc_header

from blipss.constants import (
    FILTERBANK_EXTENSIONS,
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

    This is a pre-write step for the *simulate_data* pipeline. The resulting
    dict is passed directly to ``write_filterbank``.

    Args:
        sim: Filterbank dimension and frequency/time axis parameters.
        header_params: Optional observational metadata (source name, start MJD).

    Returns:
        Dictionary of sigproc header key-value pairs compatible with
        ``blimpy.io.sigproc.generate_sigproc_header``.
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
    Write a numpy data array and a sigproc header to a ``.fil`` filterbank file.

    Used by the *simulate_data* pipeline, where the full data array is generated
    in memory. The array is serialised as contiguous float32 samples immediately
    after the binary sigproc header.

    Args:
        data: Array of shape ``(n_samples, 1, n_channels)`` in sigproc layout,
            as returned by ``blipss.core.simulate_data.reshape_for_sigproc``.
        header: Sigproc header dictionary, as returned by ``build_sigproc_header``.
        output_dir: Directory in which the output file is created; created
            automatically if it does not exist.
        basename: Filename stem; the ``.fil`` extension is appended automatically.
    """
    ensure_path_exists(output_dir)
    carrier = _HeaderCarrier()
    carrier.header = header
    output_path = output_dir / f"{basename}.fil"
    with open(output_path, "wb") as file_handle:
        file_handle.write(generate_sigproc_header(carrier))
        data.ravel().astype(np.float32, copy=False).tofile(file_handle)


def write_waterfall(wat: Waterfall, output_path: Path) -> None:
    """
    Write a blimpy ``Waterfall`` object to disk, dispatching on file extension.

    Used by the *inject_signal* pipeline, where signal injection modifies an
    existing ``Waterfall`` object loaded from a real-data filterbank. The
    original header is preserved intact; only ``wat.data`` is expected to have
    been updated before calling this function.

    Args:
        wat: Blimpy ``Waterfall`` object to serialise.
        output_path: Full output file path including a supported extension
            (``.fil`` or ``.h5`` / ``.hdf5``).

    Raises:
        ValueError: When the file extension is not in ``FILTERBANK_EXTENSIONS``.
    """
    suffix = output_path.suffix
    if suffix not in FILTERBANK_EXTENSIONS:
        raise ValueError(f"Unsupported output extension {suffix!r}. Expected one of {sorted(FILTERBANK_EXTENSIONS)}.")
    if suffix == ".fil":
        wat.write_to_fil(str(output_path))
    else:
        wat.write_to_hdf5(str(output_path))
