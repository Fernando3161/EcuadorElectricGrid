from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal

# =============================================================================
# Basic literals
# =============================================================================

DemandMode = Literal["historical", "projected"]
DemandFamily = Literal["trend", "base", "productive_mix"]
ProjectionMethod = Literal["lin", "exp"]

HydrologyCase = Literal["normal", "dry", "severe_dry"]
RESLevel = Literal["none", "low", "medium", "high", "xl"]
NuclearWave = Literal["none", "W1", "W2", "W3"]
StressTag = Literal["", "DRY", "OUT", "CR", "FUEL"]


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class DemandConfig:
    """
    Demand is derived from ONE template profile.

    For projected cases, scaling info comes from a CSV like:

        year,trend_lin,trend_exp,base_lin,base_exp,productive_mix_lin,productive_mix_exp
        ...

    - mode = "projected": we use (family, projection) to pick the column name.
      e.g. "base" + "lin" -> "base_lin"
    - mode = "historical": return factor = 1.0 and bypass scaling if desired.
    """
    mode: DemandMode                          # "historical" or "projected"

    # Only needed if mode == "projected"
    family: Optional[DemandFamily] = None     # trend / base / productive_mix
    projection: Optional[ProjectionMethod] = None  # lin / exp

    scaling_csv: str = "demand_scaling.csv"
    profile_template_name: str = "load_template.csv"
    total_year_demand: float = 0.0


@dataclass
class HydroConfig:
    case: HydrologyCase
    inflow_profile_name: Optional[str] = None
    inflow_scaling_factor: float = 1.0


@dataclass
class NuclearConfig:
    wave: NuclearWave                         # none/W1/W2/W3
    total_capacity_mw: float                  # total nuclear capacity in MW
    layout_csv: Optional[str] = None          # maps units to buses etc.
    n_units: Optional[int] = None             # e.g. 3 for W1
    unit_capacity_mw: Optional[float] = None  # e.g. 300


@dataclass
class REConfig:
    """
    Renewables handled via % factors of some reference
    (technical potential, MP targets, etc.).

    Interpretation of onwind_factor/solar_factor is done in the RE pipeline.
    """
    level: RESLevel

    onwind_factor: float = 0.0   # 0.10 = 10% reference capacity
    solar_factor: float = 0.0
    other_re_factor: float = 0.0

    wind_cutout_nc: Optional[str] = None
    solar_cutout_nc: Optional[str] = None


@dataclass
class NetworkConfig:
    name: str
    year: int
    result_tag: Optional[str] = None


@dataclass
class Scenario:
    id: str
    year: int

    demand: DemandConfig
    hydro: HydroConfig
    network: NetworkConfig
    nuclear: NuclearConfig
    re: REConfig

    description: str = ""
    stress_tag: StressTag = ""

    validate_1day: bool = True
    run_full_year: bool = True

    extra: Dict[str, object] = field(default_factory=dict)
