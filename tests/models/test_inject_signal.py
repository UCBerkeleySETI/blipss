"""Unit tests for config models in blipss.models.inject_signal"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from blipss.models.inject_signal import InjectSignalConfig, InputDataConfig, OutputConfig
from blipss.models.simulate_data import PeriodicSignalInjection


def _make_config(
    *, datafile: str = "obs.fil", output_ext: str = "", output_dir: Path | None = None
) -> InjectSignalConfig:
    """Return a valid InjectSignalConfig with sensible defaults, overridable per-test."""
    return InjectSignalConfig(
        input_data=InputDataConfig(datafile=datafile, data_dir=Path("/fake/data")),
        output=OutputConfig(basename="out", output_ext=output_ext, output_dir=output_dir),
        periodic_signal_injection=PeriodicSignalInjection(),
    )


def test_output_ext_rejects_unsupported_extension() -> None:
    """OutputConfig rejects an output_ext that is neither .fil nor .h5."""
    with pytest.raises(ValidationError, match=r"output_ext must be \.fil or \.h5"):
        OutputConfig(basename="out", output_ext=".txt")


@pytest.mark.parametrize("output_ext", [".fil", ".h5"], ids=["fil", "h5"])
def test_output_ext_accepts_supported_extensions(output_ext: str) -> None:
    """OutputConfig accepts the supported filterbank extensions."""
    assert OutputConfig(basename="out", output_ext=output_ext).output_ext == output_ext


@pytest.mark.parametrize(
    ("datafile", "expected_ext"),
    [("obs.h5", ".h5"), ("obs.fil", ".fil")],
    ids=["hdf5_input", "filterbank_input"],
)
def test_output_ext_defaults_to_input_file_extension(datafile: str, expected_ext: str) -> None:
    """InjectSignalConfig resolves an empty output_ext to match the input file extension."""
    cfg = _make_config(datafile=datafile)
    assert cfg.output.output_ext == expected_ext


def test_explicit_output_ext_is_preserved() -> None:
    """InjectSignalConfig leaves an explicitly supplied output_ext unchanged."""
    cfg = _make_config(datafile="obs.h5", output_ext=".fil")
    assert cfg.output.output_ext == ".fil"


def test_output_dir_defaults_to_data_dir() -> None:
    """InjectSignalConfig resolves an omitted output_dir to the input data directory."""
    assert _make_config().output.output_dir == Path("/fake/data")


def test_explicit_output_dir_is_preserved() -> None:
    """InjectSignalConfig leaves an explicitly supplied output_dir unchanged."""
    assert _make_config(output_dir=Path("/fake/out")).output.output_dir == Path("/fake/out")
