"""Pydantic models for validating input configs read from a blipss YAML config file."""

import os
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from blipss.constants import DEFAULT_EPSILON_HARMONIC
from blipss.models.base import BlipssConfigModel


class InputConfig(BlipssConfigModel):
    data_dir: Path = Field(
        description="Root directory for input .fil or .h5 files; also used as the default output directory"
    )
    glob_input: str | None = Field(
        default=None,
        description="Glob pattern applied to data_dir to discover input files; used when input_file_list is not provided directly",
    )
    input_file_list: list[Path] = Field(
        default_factory=list, description="Paths to the .fil or .h5 filterbank files to process"
    )
    start_ch: int = Field(default=0, description="First channel index to include in the FFA search (inclusive)")
    stop_ch: int | None = Field(
        default=None, description="Last channel index to exclude; None defaults to the total number of channels"
    )

    @field_validator("start_ch")
    @classmethod
    def start_ch_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"start_ch must be >= 0; got {v}")
        return v

    @model_validator(mode="after")
    def resolve_glob_input(self) -> "InputConfig":
        if self.glob_input and not self.input_file_list:
            self.input_file_list = sorted(self.data_dir.glob(self.glob_input))
        return self

    @model_validator(mode="after")
    def stop_ch_after_start_ch(self) -> "InputConfig":
        if self.stop_ch is not None and self.stop_ch <= self.start_ch:
            raise ValueError(f"stop_ch {self.stop_ch} must be greater than start_ch {self.start_ch}")
        return self


class OutputConfig(BlipssConfigModel):
    output_dir: Path | None = Field(
        default=None,
        description="Output directory for CSV and plot files; defaults to data_dir when omitted",
    )


class PlottingConfig(BlipssConfigModel):
    do_plot: bool = Field(default=False, description="Produce a scatter plot of period vs. radio frequency when True")
    plot_formats: list[str] = Field(
        default=[".png"],
        description="File extensions (with leading dot) for saving plots; e.g. ['.png', '.pdf']",
    )
    use_latex: bool = Field(
        default=False,
        description="Render plot text with LaTeX (requires a system LaTeX installation)",
    )


class FfaSearchConfig(BlipssConfigModel):
    min_period: float = Field(default=10.0, description="Minimum trial period (s)")
    max_period: float = Field(default=100.0, description="Maximum trial period (s)")
    fpmin: int = Field(default=3, description="Minimum number of signal periods that must fit in the data duration")
    snr_threshold: float = Field(default=8.0, description="Minimum matched-filtering S/N for a detection")
    bins_min: int = Field(default=10, description="Minimum number of phase bins in the folded profile")
    bins_max: int = Field(default=11, description="Maximum number of phase bins in the folded profile")
    ducy_max: float = Field(default=0.5, description="Maximum duty cycle searched; must be in (0, 1]")
    do_deredden: bool = Field(default=False, description="Apply running-median detrending before folding when True")
    rmed_width: float = Field(
        default=12.0, description="Running median window width (s); used only when do_deredden is True"
    )
    epsilon_fof: float = Field(default=1e-3, description="Period tolerance for Friends-of-Friends clustering")
    epsilon_harmonic: float = Field(
        default=DEFAULT_EPSILON_HARMONIC, description="Period tolerance for harmonic matching"
    )

    @model_validator(mode="after")
    def period_range_valid(self) -> "FfaSearchConfig":
        if self.min_period >= self.max_period:
            raise ValueError(f"min_period {self.min_period} must be less than max_period {self.max_period}")
        return self

    @model_validator(mode="after")
    def bins_range_valid(self) -> "FfaSearchConfig":
        if self.bins_min > self.bins_max:
            raise ValueError(f"bins_min {self.bins_min} must be <= bins_max {self.bins_max}")
        return self

    @field_validator("ducy_max")
    @classmethod
    def ducy_max_in_range(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"ducy_max {v} must be in (0, 1]")
        return v

    @field_validator("snr_threshold", "epsilon_fof", "epsilon_harmonic", "rmed_width", "min_period", "max_period")
    @classmethod
    def positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"value must be positive; got {v}")
        return v

    @field_validator("fpmin", "bins_min", "bins_max")
    @classmethod
    def positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"value must be positive; got {v}")
        return v


class ResourceConfig(BlipssConfigModel):
    mem_load: float = Field(default=1.0, description="Maximum data size (GB) to load into memory per file")
    n_workers: int | None = Field(
        default=None,
        description="Number of parallel worker processes for the channel-wise FFA search; None uses all available CPUs",
    )

    @field_validator("mem_load")
    @classmethod
    def mem_load_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"mem_load must be positive; got {v}")
        return v

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


class BlipssConfig(BlipssConfigModel):
    input: InputConfig = Field(description="Input data location and channel selection")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output directory configuration")
    plotting: PlottingConfig = Field(default_factory=PlottingConfig, description="Plot generation options")
    ffa_search: FfaSearchConfig = Field(default_factory=FfaSearchConfig, description="FFA search parameters")
    resources: ResourceConfig = Field(default_factory=ResourceConfig, description="Computational resource limits")

    @model_validator(mode="after")
    def default_output_dir(self) -> "BlipssConfig":
        if self.output.output_dir is None:
            self.output.output_dir = self.input.data_dir
        return self
