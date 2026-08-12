"""Pydantic models for validating input configs read from config/inject_signal.yaml"""

from pathlib import Path

from pydantic import Field, field_validator, model_validator

from blipss.constants import FILTERBANK_EXTENSIONS
from blipss.models.base import BlipssConfigModel
from blipss.models.simulate_data import PeriodicSignalInjection


class InputDataConfig(BlipssConfigModel):
    datafile: str = Field(description="Name of the filterbank data file to load (.fil or .h5)")
    data_dir: Path = Field(description="Directory containing the input data file")


class OutputConfig(BlipssConfigModel):
    basename: str = Field(description="Basename of the output file (without extension)")
    output_ext: str = Field(default="", description="Output file extension (.fil or .h5); defaults to match input")
    output_dir: Path | None = Field(default=None, description="Output directory; defaults to data_dir")

    @field_validator("output_ext")
    @classmethod
    def validate_output_ext(cls, v: str) -> str:
        """
        Validate that the output extension is a recognised filterbank format.

        Args:
            v: The output extension string from the config.

        Returns:
            The validated extension string.

        Raises:
            ValueError: When the extension is non-empty but not .fil or .h5.
        """
        if v and v not in FILTERBANK_EXTENSIONS:
            raise ValueError(f"output_ext must be .fil or .h5, got {v!r}")
        return v


class ResourceLimits(BlipssConfigModel):
    mem_load: float = Field(default=1.0, description="Maximum data volume (GB) to load into memory")


class InjectSignalConfig(BlipssConfigModel):
    input_data: InputDataConfig = Field(description="Input data file configuration")
    output: OutputConfig = Field(description="Output file configuration")
    periodic_signal_injection: PeriodicSignalInjection = Field(
        default_factory=PeriodicSignalInjection,
        description="Periodic boxcar pulse trains to inject into specified channels",
    )
    resource_limits: ResourceLimits = Field(
        default_factory=ResourceLimits,
        description="Memory resource limits for data loading",
    )

    @model_validator(mode="after")
    def resolve_defaults(self) -> "InjectSignalConfig":
        """
        Resolve output_ext and output_dir defaults from input data configuration.

        Returns:
            The model instance with resolved defaults.
        """
        if not self.output.output_ext:
            self.output.output_ext = ".h5" if self.input_data.datafile.endswith(".h5") else ".fil"
        if self.output.output_dir is None:
            self.output.output_dir = self.input_data.data_dir
        return self
