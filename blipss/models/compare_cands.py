"""Pydantic models for validating input configs read from config/compare_cands.yaml"""

import os
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from blipss.constants import DEFAULT_CLUSTER_RADIUS_SECONDS, DEFAULT_OFF_SNR_CUTOFF, DEFAULT_ON_SNR_CUTOFF
from blipss.models.base import BlipssConfigModel


class CandidateFilesConfig(BlipssConfigModel):
    csv_dir: Path = Field(description="Directory containing the input candidate CSV files")
    csv_list: list[Path] = Field(description="Candidate CSV filenames to compare, in order")


class OnOffClassificationConfig(BlipssConfigModel):
    labels: list[str] = Field(description="Pointing classification ('ON' or 'OFF') for each file in csv_list")
    on_cutoff: float = Field(
        default=DEFAULT_ON_SNR_CUTOFF, description="S/N threshold for candidates detected in 'ON' pointings"
    )
    off_cutoff: float = Field(
        default=DEFAULT_OFF_SNR_CUTOFF, description="S/N threshold for candidates detected in 'OFF' pointings"
    )

    @field_validator("labels", mode="before")
    @classmethod
    def uppercase_labels(cls, v: list[str]) -> list[str]:
        """
        Normalise pointing labels to uppercase.

        Args:
            v: Raw label strings from the config.

        Returns:
            Labels converted to uppercase.
        """
        return [label.upper() for label in v]

    @field_validator("labels")
    @classmethod
    def labels_are_on_or_off(cls, v: list[str]) -> list[str]:
        """
        Validate that every label is either 'ON' or 'OFF'.

        Args:
            v: Uppercased label strings.

        Returns:
            The validated label strings.

        Raises:
            ValueError: When any label is not 'ON' or 'OFF'.
        """
        invalid = sorted(set(v) - {"ON", "OFF"})
        if invalid:
            raise ValueError(f"labels must be 'ON' or 'OFF'; got invalid values: {invalid}")
        return v

    @field_validator("on_cutoff", "off_cutoff")
    @classmethod
    def cutoff_positive(cls, v: float) -> float:
        """
        Validate that an S/N cutoff is positive.

        Args:
            v: Cutoff value from the config.

        Returns:
            The validated cutoff value.

        Raises:
            ValueError: When the cutoff is non-negative.
        """
        if v <= 0:
            raise ValueError(f"S/N cutoff must be > 0; got {v}")
        return v


class OutputConfig(BlipssConfigModel):
    basename: str = Field(description="Basename of the output comparison CSV file")
    output_dir: Path = Field(description="Output directory for the comparison CSV file (created if non-existent)")


class CandidateGroupingConfig(BlipssConfigModel):
    cluster_radius: float = Field(
        default=DEFAULT_CLUSTER_RADIUS_SECONDS,
        description="Clustering radius (s) for grouping candidate periods within a channel",
    )
    n_jobs: int = Field(
        default_factory=lambda: os.cpu_count() or 1,
        description="Worker processes for channel clustering (default: all available CPU cores; null in YAML "
        "falls back to this default). Set to 1 to run sequentially in-process.",
    )

    @field_validator("cluster_radius")
    @classmethod
    def cluster_radius_positive(cls, v: float) -> float:
        """
        Validate that the clustering radius is positive.

        Args:
            v: Clustering radius from the config.

        Returns:
            The validated clustering radius.

        Raises:
            ValueError: When the clustering radius is not positive.
        """
        if v <= 0:
            raise ValueError(f"cluster_radius must be positive; got {v}")
        return v

    @field_validator("n_jobs")
    @classmethod
    def n_jobs_valid(cls, v: int) -> int:
        """
        Validate that n_jobs is a positive worker count.

        Args:
            v: Worker process count from the config.

        Returns:
            The validated worker process count.

        Raises:
            ValueError: When n_jobs is not positive.
        """
        if v < 1:
            raise ValueError(f"n_jobs must be >= 1; got {v}")
        return v


class CompareCandsConfig(BlipssConfigModel):
    candidate_files: CandidateFilesConfig = Field(description="Input candidate CSV file locations")
    on_off_classification: OnOffClassificationConfig = Field(
        description="Pointing labels and S/N thresholds for candidate filtering"
    )
    output: OutputConfig = Field(description="Output file configuration")
    candidate_grouping: CandidateGroupingConfig = Field(
        default_factory=CandidateGroupingConfig, description="Period clustering parameters"
    )

    @model_validator(mode="after")
    def labels_match_csv_list(self) -> "CompareCandsConfig":
        """
        Validate that one pointing label is provided per candidate CSV file.

        Returns:
            The model instance, unchanged.

        Raises:
            ValueError: When the number of labels does not match the number of CSV files.
        """
        n_files = len(self.candidate_files.csv_list)
        n_labels = len(self.on_off_classification.labels)
        if n_files != n_labels:
            raise ValueError(f"csv_list has {n_files} entries but labels has {n_labels}; lengths must match")
        return self
