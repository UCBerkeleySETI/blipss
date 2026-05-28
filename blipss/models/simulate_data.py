"""Pydantic models for validating input configs read from config/simulate_data.yaml"""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class OutputConfig(BaseModel):
    basename: str = Field(description="Basename of the output filterbank file")
    output_dir: Path = Field(description="Output directory path (created if non-existent)")


class SimulationProperties(BaseModel):
    n_samples: int = Field(description="Number of time samples")
    n_channels: int = Field(description="Number of spectral channels")
    t_samp: float = Field(description="Sampling time (s)")
    foff: float = Field(description="Channel bandwidth (MHz)")
    fch1: float = Field(description="Radio frequency (MHz) of the first spectral channel")
    seed: int | None = Field(
        default=None,
        description="Random seed for the Gaussian white noise background; None gives a non-reproducible draw",
    )


class PeriodicSignalInjection(BaseModel):
    inject_channels: list[int] = Field(
        default=[], description="Channel indices into which a periodic signal is injected"
    )
    periods: list[float] = Field(default=[], description="Signal periods (s) for each injection channel")
    duty_cycles: list[float] = Field(
        default=[], description="Pulse duty cycles for each injection channel; must be in (0, 1]"
    )
    pulse_snr: list[float] = Field(
        default=[], description="Pulse peak signal-to-noise ratios for each injection channel"
    )
    initial_phase: list[float] = Field(
        default=[],
        description="Initial pulse emission phases (fraction of a period) for each injection channel; must be in [0, 1)",
    )

    @model_validator(mode="after")
    def lists_same_length(self) -> "PeriodicSignalInjection":
        lengths = {
            "inject_channels": len(self.inject_channels),
            "periods": len(self.periods),
            "duty_cycles": len(self.duty_cycles),
            "pulse_snr": len(self.pulse_snr),
            "initial_phase": len(self.initial_phase),
        }
        unique = set(lengths.values())
        if len(unique) > 1:
            raise ValueError(f"All periodic_signal_injection lists must have the same length; got {lengths}")
        return self

    @field_validator("duty_cycles", mode="before")
    @classmethod
    def duty_cycles_in_range(cls, v: list[float]) -> list[float]:
        for dc in v:
            if not (0.0 < dc <= 1.0):
                raise ValueError(f"duty_cycle {dc} must be in (0, 1]")
        return v

    @field_validator("initial_phase", mode="before")
    @classmethod
    def initial_phase_in_range(cls, v: list[float]) -> list[float]:
        for ip in v:
            if not (0.0 <= ip < 1.0):
                raise ValueError(f"initial_phase {ip} must be in [0, 1)")
        return v


class OptionalHeaderParameters(BaseModel):
    source_name: str = Field(default="Unknown", description="Source name written to the filterbank header")
    tstart: float = Field(default=0.0, description="Observation start time as a Modified Julian Date (d)")


class SimulateDataConfig(BaseModel):
    output: OutputConfig = Field(description="Output file configuration")
    simulation_properties: SimulationProperties = Field(
        description="Filterbank dimensions and frequency/time axis parameters"
    )
    periodic_signal_injection: PeriodicSignalInjection = Field(
        default_factory=PeriodicSignalInjection,
        description="Periodic boxcar pulse trains to inject into specified channels",
    )
    optional_header_parameters: OptionalHeaderParameters = Field(
        default_factory=OptionalHeaderParameters,
        description="Optional filterbank header fields that mimic a real-world observation",
    )
