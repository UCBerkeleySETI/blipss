"""Pydantic models for validating input configs read from config/phaseresolved_ds.yaml"""

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class InputDataConfig(BaseModel):
    datafile: str = Field(description="Name of the Blimpy-readable data file (.fil or .h5)")
    data_dir: Path = Field(description="Directory containing the input data file")


class OutputConfig(BaseModel):
    basename: str = Field(description="Basename of the output plot file (no extension)")
    plot_formats: list[str] = Field(
        default_factory=lambda: [".png"],
        description="File extensions for saved plots (e.g. ['.png', '.pdf']); defaults to ['.png']",
    )

    @field_validator("plot_formats", mode="before")
    @classmethod
    def resolve_plot_formats(cls, v: list[str] | None) -> list[str]:
        return [".png"] if v is None else v

    plot_dir: Path | None = Field(
        default=None,
        description="Output directory for plots; defaults to input data_dir",
    )
    use_latex: bool = Field(
        default=False,
        description="Render plot text with LaTeX (requires a system LaTeX installation)",
    )


class ChannelSelectionConfig(BaseModel):
    start_ch: int = Field(default=0, description="First channel index to include (inclusive)")
    stop_ch: int | None = Field(
        default=None,
        description="Last channel index to exclude (exclusive); None includes all remaining channels",
    )


class PhaseFoldingConfig(BaseModel):
    period: float = Field(default=1.0, description="Folding period (s)")
    bins: int = Field(default=10, description="Number of phase bins in the folded profile")
    do_deredden: bool = Field(
        default=False,
        description="Apply running-median detrending to each channel before folding",
    )
    rmed_width: float = Field(
        default=12.0,
        description="Running median window width (s) used for detrending; resolved from null to 12.0",
    )

    @field_validator("rmed_width", mode="before")
    @classmethod
    def resolve_rmed_width(cls, v: float | None) -> float:
        """
        Resolve rmed_width to its default when null is supplied in the config.

        Args:
            v: The raw rmed_width value from the config, or None.

        Returns:
            The validated width value; defaults to 12.0 when None.
        """
        return 12.0 if v is None else v


class ResourceLimits(BaseModel):
    mem_load: float = Field(default=1.0, description="Maximum data volume (GB) to load into memory")
    n_workers: int | None = Field(
        default=None,
        description="Number of parallel worker processes for folding; None uses all available CPUs",
    )

    @field_validator("n_workers")
    @classmethod
    def validate_n_workers(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 1:
            raise ValueError(f"n_workers must be >= 1, got {v}")
        cpu_count = os.cpu_count()
        if cpu_count is not None and v > cpu_count:
            raise ValueError(f"n_workers {v} exceeds available CPU count ({cpu_count})")
        return v


class PhaseResolvedDsConfig(BaseModel):
    input_data: InputDataConfig = Field(description="Input data file configuration")
    output: OutputConfig = Field(description="Output plot configuration")
    channel_selection: ChannelSelectionConfig = Field(
        default_factory=ChannelSelectionConfig,
        description="Channel index range to include in the phase-resolved spectrum",
    )
    phase_folding_parameters: PhaseFoldingConfig = Field(
        default_factory=PhaseFoldingConfig,
        description="Period, phase bins, and detrending parameters for folding",
    )
    resource_limits: ResourceLimits = Field(
        default_factory=ResourceLimits,
        description="Memory resource limits for data loading",
    )

    @model_validator(mode="after")
    def resolve_output_defaults(self) -> "PhaseResolvedDsConfig":
        """
        Resolve plot_dir and plot_formats defaults from input data configuration.

        Returns:
            The model instance with plot_dir and plot_formats resolved.
        """
        if self.output.plot_dir is None:
            self.output.plot_dir = self.input_data.data_dir
        return self
