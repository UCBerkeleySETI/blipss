"""Unit tests for modules in blipss.io.write_filterbank"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from blipss.constants import (
    SIGPROC_DATA_TYPE,
    SIGPROC_MACHINE_ID,
    SIGPROC_N_BITS,
    SIGPROC_N_IFS,
    SIGPROC_TELESCOPE_ID,
)
from blipss.io.write_filterbank import _HeaderCarrier, build_sigproc_header, write_filterbank, write_waterfall
from blipss.models.simulate_data import OptionalHeaderParameters, SimulationProperties

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim(**overrides: Any) -> SimulationProperties:
    """Return a SimulationProperties object with sensible defaults, overridable per-test."""
    defaults: dict[str, Any] = {"n_samples": 16, "n_channels": 4, "t_samp": 0.001, "foff": -0.1, "fch1": 1500.0}
    defaults.update(overrides)
    return SimulationProperties(**defaults)


def _make_header_params(**overrides: Any) -> OptionalHeaderParameters:
    """Return an OptionalHeaderParameters object with sensible defaults, overridable per-test."""
    defaults: dict[str, Any] = {"source_name": "TestSrc", "tstart": 59000.0}
    defaults.update(overrides)
    return OptionalHeaderParameters(**defaults)


# ---------------------------------------------------------------------------
# build_sigproc_header
# ---------------------------------------------------------------------------


def test_build_sigproc_header_fixed_constants() -> None:
    """build_sigproc_header always populates the fixed sigproc constant fields from package constants."""
    result: dict[str, Any] = build_sigproc_header(_make_sim(), _make_header_params())
    assert result["machine_id"] == SIGPROC_MACHINE_ID
    assert result["telescope_id"] == SIGPROC_TELESCOPE_ID
    assert result["data_type"] == SIGPROC_DATA_TYPE
    assert result["nbits"] == SIGPROC_N_BITS
    assert result["nifs"] == SIGPROC_N_IFS


@pytest.mark.parametrize(
    ("fch1", "foff", "t_samp", "n_channels", "n_samples"),
    [
        (1500.0, -0.1, 0.001, 1024, 512),
        (2000.0, 0.05, 0.01, 256, 128),
    ],
    ids=["config_a", "config_b"],
)
def test_build_sigproc_header_sim_fields(
    fch1: float,
    foff: float,
    t_samp: float,
    n_channels: int,
    n_samples: int,
) -> None:
    """build_sigproc_header maps each SimulationProperties field to the correct header key."""
    sim = SimulationProperties(fch1=fch1, foff=foff, t_samp=t_samp, n_channels=n_channels, n_samples=n_samples)
    result: dict[str, Any] = build_sigproc_header(sim, OptionalHeaderParameters())
    assert result["fch1"] == fch1
    assert result["foff"] == foff
    assert result["tsamp"] == t_samp
    assert result["nchans"] == n_channels
    assert result["nsamples"] == n_samples


@pytest.mark.parametrize(
    ("source_name", "tstart"),
    [
        ("Vela", 59000.0),
        ("Unknown", 0.0),
    ],
    ids=["named_source", "default_params"],
)
def test_build_sigproc_header_optional_params(source_name: str, tstart: float) -> None:
    """build_sigproc_header maps OptionalHeaderParameters to source_name and tstart header keys."""
    params = OptionalHeaderParameters(source_name=source_name, tstart=tstart)
    result: dict[str, Any] = build_sigproc_header(_make_sim(), params)
    assert result["source_name"] == source_name
    assert result["tstart"] == tstart


# ---------------------------------------------------------------------------
# write_filterbank — ensure_path_exists and generate_sigproc_header coupling
# ---------------------------------------------------------------------------


@patch("blipss.io.write_filterbank.ensure_path_exists")
@patch("blipss.io.write_filterbank.generate_sigproc_header")
def test_write_filterbank_passes_carrier_with_header(
    mock_gen_header: MagicMock,
    mock_ensure: MagicMock,
    tmp_path: Path,
) -> None:
    """write_filterbank passes a _HeaderCarrier whose .header matches the given dict to generate_sigproc_header."""
    mock_gen_header.return_value = b""
    header: dict[str, Any] = {"nchans": 4, "tsamp": 0.001}
    data: npt.NDArray[np.float32] = np.zeros((2, 1, 4), dtype=np.float32)
    write_filterbank(data, header, tmp_path, "obs")
    mock_ensure.assert_called_once_with(tmp_path)
    carrier: Any = mock_gen_header.call_args[0][0]
    assert isinstance(carrier, _HeaderCarrier)
    assert carrier.header is header


# ---------------------------------------------------------------------------
# write_filterbank — output path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "basename",
    ["obs", "test_obs_001", "my-observation"],
    ids=["simple", "underscored", "hyphenated"],
)
@patch("blipss.io.write_filterbank.ensure_path_exists")
@patch("blipss.io.write_filterbank.generate_sigproc_header")
def test_write_filterbank_appends_fil_extension(
    mock_gen_header: MagicMock,
    mock_ensure: MagicMock,
    tmp_path: Path,
    basename: str,
) -> None:
    """write_filterbank always creates output_dir/basename.fil regardless of basename content."""
    mock_gen_header.return_value = b""
    data: npt.NDArray[np.float32] = np.zeros((2, 1, 2), dtype=np.float32)
    write_filterbank(data, {}, tmp_path, basename)
    mock_ensure.assert_called_once_with(tmp_path)
    assert (tmp_path / f"{basename}.fil").exists()


# ---------------------------------------------------------------------------
# write_filterbank — binary content
# ---------------------------------------------------------------------------


@patch("blipss.io.write_filterbank.ensure_path_exists")
@patch("blipss.io.write_filterbank.generate_sigproc_header")
def test_write_filterbank_header_bytes_precede_data(
    mock_gen_header: MagicMock,
    mock_ensure: MagicMock,
    tmp_path: Path,
) -> None:
    """write_filterbank writes sigproc header bytes immediately before the float32 data bytes."""
    header_bytes = b"\xde\xad\xbe\xef"
    mock_gen_header.return_value = header_bytes
    data: npt.NDArray[np.float64] = np.arange(8, dtype=np.float64).reshape(4, 1, 2)
    write_filterbank(data, {}, tmp_path, "obs")
    mock_ensure.assert_called_once_with(tmp_path)
    raw: bytes = (tmp_path / "obs.fil").read_bytes()
    expected_data_bytes: bytes = data.ravel().astype(np.float32).tobytes()
    assert raw == header_bytes + expected_data_bytes


@patch("blipss.io.write_filterbank.ensure_path_exists")
@patch("blipss.io.write_filterbank.generate_sigproc_header")
def test_write_filterbank_casts_data_to_float32(
    mock_gen_header: MagicMock,
    mock_ensure: MagicMock,
    tmp_path: Path,
) -> None:
    """write_filterbank serialises data as float32 even when the input array dtype is float64."""
    mock_gen_header.return_value = b""
    values = [1.0, 2.0, 3.0, 4.0]
    data_f64: npt.NDArray[np.float64] = np.array(values, dtype=np.float64).reshape(2, 1, 2)
    data_f32: npt.NDArray[np.float32] = np.array(values, dtype=np.float32).reshape(2, 1, 2)
    write_filterbank(data_f64, {}, tmp_path, "obs_f64")
    write_filterbank(data_f32, {}, tmp_path, "obs_f32")
    mock_ensure.assert_called_with(tmp_path)
    bytes_f64: bytes = (tmp_path / "obs_f64.fil").read_bytes()
    bytes_f32: bytes = (tmp_path / "obs_f32.fil").read_bytes()
    assert bytes_f64 == bytes_f32


# ---------------------------------------------------------------------------
# write_waterfall
# ---------------------------------------------------------------------------


def test_write_waterfall_fil_calls_write_to_fil() -> None:
    """write_waterfall calls wat.write_to_fil with the string output path for .fil files."""
    wat = MagicMock()
    output_path = Path("/some/output.fil")
    write_waterfall(wat, output_path)
    wat.write_to_fil.assert_called_once_with(str(output_path))
    wat.write_to_hdf5.assert_not_called()


def test_write_waterfall_h5_calls_write_to_hdf5() -> None:
    """write_waterfall calls wat.write_to_hdf5 with the string output path for .h5 files."""
    wat = MagicMock()
    output_path = Path("/some/output.h5")
    write_waterfall(wat, output_path)
    wat.write_to_hdf5.assert_called_once_with(str(output_path))
    wat.write_to_fil.assert_not_called()


def test_write_waterfall_unsupported_extension_raises_value_error() -> None:
    """write_waterfall raises ValueError for an extension not in FILTERBANK_EXTENSIONS."""
    wat = MagicMock()
    with pytest.raises(ValueError, match="Unsupported output extension"):
        write_waterfall(wat, Path("/some/output.fits"))
