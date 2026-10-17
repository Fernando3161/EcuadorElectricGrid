"""Reusable Phase 06B calibration controls and diagnostic aggregations."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


BIOMASS_MONTHLY_GWH = {
    1: 5.34, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 14.03,
    7: 52.76, 8: 52.05, 9: 50.66, 10: 49.13, 11: 47.37, 12: 17.15,
}


def load_monthly_biomass_shape(path: Path | str) -> pd.DataFrame:
    """Read the ARCONEL monthly shape and return a normalized source table."""
    frame = pd.read_csv(path)
    required = {"month", "biomass_gwh", "source_document", "source_locator", "boundary"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"biomass source table missing columns: {sorted(missing)}")
    frame = frame.sort_values("month").reset_index(drop=True)
    if frame.month.tolist() != list(range(1, 13)):
        raise ValueError("biomass source table must contain months 1..12 exactly once")
    frame["biomass_gwh"] = pd.to_numeric(frame["biomass_gwh"], errors="raise")
    if (frame.biomass_gwh < 0).any() or frame.biomass_gwh.sum() <= 0:
        raise ValueError("biomass monthly energy must be non-negative and non-zero")
    return frame


def monthly_energy_profile(index: pd.DatetimeIndex, monthly_gwh: pd.Series) -> pd.Series:
    """Convert monthly energy budgets to an hourly p.u. profile summing to 1 GWh/MW."""
    hours = pd.Series(index.month, index=index).map(monthly_gwh).astype(float)
    counts = hours.groupby(index.month).transform("size")
    return hours / (counts * hours.groupby(index.month).first().replace(0, np.nan)).reindex(index.month).to_numpy()


def apply_monthly_biomass_limits(network, source_table: pd.DataFrame, plants: pd.DataFrame) -> pd.DataFrame:
    """Apply a shared seasonal shape to biomass generators, preserving biogas."""
    shape = source_table.set_index("month")["biomass_gwh"]
    biomass = plants.loc[plants.carrier.eq("biomass"), "plant_id"].astype(str)
    rows = []
    for name in biomass:
        if name not in network.generators_t.p_max_pu:
            continue
        p_nom = float(network.generators.at[name, "p_nom"])
        for month, energy in shape.items():
            mask = network.snapshots.month == int(month)
            # Allocate the official aggregate net-shape proportionally by capacity.
            target_mwh = float(energy) * 1000.0 * p_nom / float(network.generators.loc[biomass, "p_nom"].sum())
            hourly_limit = target_mwh / max(int(mask.sum()), 1) / p_nom
            network.generators_t.p_max_pu.loc[mask, name] = min(1.0, hourly_limit)
        rows.append({"plant_id": name, "carrier": "biomass", "control": "ARCONEL_monthly_shared_shape"})
    return pd.DataFrame(rows)


def aggregate_dispatch_by_carrier(network) -> pd.DataFrame:
    """Return dispatch, potential, curtailment, and capacity factors by carrier."""
    weights = network.snapshot_weightings.generators.reindex(network.snapshots).to_numpy()
    dispatch = network.generators_t.p.reindex(columns=network.generators.index).fillna(0.0)
    potential = network.generators_t.p_max_pu.reindex(columns=network.generators.index).fillna(0.0) * network.generators.p_nom
    rows = []
    for carrier, names in network.generators.groupby("carrier").groups.items():
        d = float((dispatch[names].sum(axis=1).to_numpy() * weights).sum() / 1000.0)
        p = float((potential[names].sum(axis=1).to_numpy() * weights).sum() / 1000.0)
        rows.append({"carrier": carrier, "dispatch_gwh": d, "potential_gwh": p, "curtailment_gwh": max(p-d, 0.0), "capacity_factor": d / (float(network.generators.loc[names, "p_nom"].sum()) * weights.sum() / 1000.0) if weights.sum() else np.nan})
    return pd.DataFrame(rows).sort_values("carrier").reset_index(drop=True)


__all__ = ["aggregate_dispatch_by_carrier", "apply_monthly_biomass_limits", "load_monthly_biomass_shape", "monthly_energy_profile"]
