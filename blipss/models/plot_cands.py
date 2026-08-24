"""Pydantic models for validating input configs read from config/plot_cands.yaml"""

from pathlib import Path

from pydantic import Field, model_validator

from blipss.models.base import BlipssConfigModel
from blipss.models.run_ffa_search import FfaSearchConfig, ResourceConfig


class InputDataConfig(BlipssConfigModel):
    data_dir: Path = Field(description="Directory containing the input data files")
    datafile_list: list[Path] = Field(description="Blimpy-readable data files (.fil or .h5) to plot, in order")
    beam_labels: list[str] = Field(
        default_factory=list,
        description="Custom annotation labels for each data file; defaults to blank labels",
    )

    @model_validator(mode="after")
    def resolve_beam_labels(self) -> "InputDataConfig":
        """
        Default beam_labels to one blank label per data file.

        Returns:
            The model instance with beam_labels resolved.
        """
        if not self.beam_labels:
            self.beam_labels = [""] * len(self.datafile_list)
        return self


class CandidateFileConfig(BlipssConfigModel):
    csvfile: Path = Field(description="Compared-candidates CSV file produced by compare_cands.py")


class PlottingParametersConfig(BlipssConfigModel):
    codes_plot: list[str] = Field(description="Candidate binary codes selected for plotting")
    basename: str = Field(description="Basename of the output plot files")
    plot_formats: list[str] = Field(
        default_factory=lambda: [".png"],
        description="File extensions for saved plots (e.g. ['.png', '.pdf']); defaults to ['.png']",
    )
    plot_dir: Path | None = Field(
        default=None,
        description="Output directory for plots; defaults to input data_dir",
    )
    periodaxis_log: bool = Field(default=True, description="Plot the periodogram period axis on a log scale")
    use_latex: bool = Field(
        default=False,
        description="Render plot text with LaTeX (requires a system LaTeX installation)",
    )


class PlotCandsConfig(BlipssConfigModel):
    input_data: InputDataConfig = Field(description="Input data file configuration")
    candidate_file: CandidateFileConfig = Field(description="Compared-candidates CSV file location")
    plotting_parameters: PlottingParametersConfig = Field(description="Candidate selection and plot output options")
    folding_search_parameters: FfaSearchConfig = Field(
        default_factory=FfaSearchConfig, description="Fast folding algorithm parameters used to fold each candidate"
    )
    resource_limits: ResourceConfig = Field(
        default_factory=ResourceConfig, description="Memory resource limits for data loading"
    )

    @model_validator(mode="after")
    def resolve_plot_dir(self) -> "PlotCandsConfig":
        """
        Resolve plot_dir default from input data configuration.

        Returns:
            The model instance with plot_dir resolved.
        """
        if self.plotting_parameters.plot_dir is None:
            self.plotting_parameters.plot_dir = self.input_data.data_dir
        return self
