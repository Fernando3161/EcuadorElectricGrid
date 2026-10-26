"""Caudal-first hydro-shortage scenario utilities for Phase 08A."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pypsa

from src.availability import build_hydro_availability, load_hydro_assignments, load_monthly_flows


ROOT = Path(__file__).resolve().parents[1]
FLOW_PATH = ROOT / "data/raw/generation/caudals.csv"
GENERATOR_PATH = ROOT / "data/processed/generation/generators_2022.csv"
ASSIGNMENT_PATH = ROOT / "data/interim/generation/hydro_profile_assignments_2022.csv"
REFERENCE_NETWORK = ROOT / "results/phase_06d/baseline/network_solved_2022.nc"
DEMAND_2025_GWH = 32650.66
DEMAND_2022_GWH = 27561.62
STRESS_MONTHS = (9, 10, 11, 12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_monthly_flow_shock(
    flows: pd.DataFrame,
    *,
    caudal_ids: list[str],
    months: tuple[int, ...] = STRESS_MONTHS,
    reduction: float,
) -> pd.DataFrame:
    """Reduce selected monthly caudals before hydro availability is rebuilt."""
    if not 0 <= reduction < 1:
        raise ValueError("reduction must be in [0, 1)")
    if not set(caudal_ids).issubset(flows.index):
        raise ValueError(f"Unknown caudal IDs: {sorted(set(caudal_ids) - set(flows.index))}")
    if not set(months).issubset(range(1, 13)):
        raise ValueError("months must contain integers from 1 to 12")
    result = flows.copy().astype(float)
    for month in months:
        result.loc[caudal_ids, str(month)] *= 1.0 - reduction
    if (result < 0).any().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError("shocked caudals must remain finite and non-negative")
    return result


def select_caudals(generators: pd.DataFrame, assignments: pd.DataFrame, count: int = 4) -> pd.DataFrame:
    """Select highest aggregate hydro capacity by assigned caudal series."""
    hydro = generators.loc[generators.carrier.eq("hydro"), ["plant_id", "name", "p_nom_mw", "effective_capacity_mw"]]
    joined = hydro.merge(assignments[["plant_id", "caudal_id", "confidence"]], on="plant_id", validate="one_to_one")
    selected = (joined.groupby("caudal_id", as_index=False)
                .agg(aggregate_p_nom_mw=("p_nom_mw", "sum"), aggregate_effective_capacity_mw=("effective_capacity_mw", "sum"),
                     plants=("plant_id", "size"), minimum_confidence=("confidence", "min"))
                .sort_values(["aggregate_p_nom_mw", "caudal_id"], ascending=[False, True], kind="stable")
                .head(count))
    return selected


def build_2026_demand(network: pypsa.Network, annual_gwh: float = DEMAND_2025_GWH) -> pd.DataFrame:
    """Scale the solved 2022 hourly demand shape to the 2025 CENACE target for 2026."""
    target = pd.date_range("2026-01-01", periods=8760, freq="h")
    base = network.loads_t.p_set.copy()
    base.index = target
    return base * ((annual_gwh * 1000.0) / base.to_numpy().sum())


def rebuild_hydro_profiles(generators: pd.DataFrame, flows: pd.DataFrame, assignments: pd.DataFrame, year: int = 2026):
    """Call the canonical project hydro-profile builder after a caudal shock."""
    return build_hydro_availability(generators, flows, assignments, year=year)


def build_scenario_network(
    reference: pypsa.Network,
    hydro_profiles: pd.DataFrame,
    demand: pd.DataFrame,
    *,
    year: int = 2026,
) -> pypsa.Network:
    """Create a scenario copy with only demand and rebuilt hydro profiles changed."""
    n = reference.copy()
    snapshots = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
    n.set_snapshots(snapshots)
    n.loads_t.p_set = demand.reindex(snapshots)
    hydro_ids = n.generators.index[n.generators.carrier.eq("hydro")]
    missing = sorted(set(hydro_ids) - set(hydro_profiles.columns))
    if missing:
        raise ValueError(f"Rebuilt hydro profiles missing network generators: {missing}")
    n.generators_t.p_max_pu.loc[:, hydro_ids] = hydro_profiles.loc[snapshots, hydro_ids]
    n.meta = dict(n.meta or {})
    n.meta.update({"phase": "08A", "year": year, "hydro_profile_method": "caudal_first_build_hydro_availability", "reference_network": str(REFERENCE_NETWORK.relative_to(ROOT))})
    return n


def summarize_solved(network: pypsa.Network, scenario: str, reduction: float) -> dict:
    supply = network.generators_t.p.mul(network.generators.sign, axis=1).sum()
    by_carrier = supply.groupby(network.generators.carrier).sum() / 1000.0
    demand = float(network.loads_t.p_set.sum().sum() / 1000.0)
    shed = float(by_carrier.get("load_shedding", 0.0))
    return {"scenario": scenario, "caudal_reduction": reduction, "demand_gwh": demand,
            "load_shedding_gwh": shed, "load_shedding_pct": 100 * shed / demand,
            "solver_status": str(network.meta.get("solver_status", "unknown")),
            "termination_condition": str(network.meta.get("termination_condition", "unknown")),
            **{f"{carrier}_gwh": float(value) for carrier, value in by_carrier.items()}}


__all__ = ["apply_monthly_flow_shock", "select_caudals", "build_2026_demand", "rebuild_hydro_profiles", "build_scenario_network", "summarize_solved"]
