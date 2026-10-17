"""Phase 06D bounded calibration and candidate bookkeeping helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_SETTINGS = {
    "baseline": {"wind_factor": 1.0, "biomass_factor": 1.0, "thermal_derating": False},
    "thermal-availability": {"wind_factor": 1.0, "biomass_factor": 1.0, "thermal_derating": True},
    # The renewable notebook replaces this provisional factor with
    # reference / actual selected NLR profile energy after loading all sites.
    "renewable-calibration": {"wind_factor": 1.0, "biomass_factor": 195.68 / 195.68, "thermal_derating": False},
    "joint-final": {"wind_factor": 1.0, "biomass_factor": 195.68 / 195.68, "thermal_derating": True},
}


def candidate_name() -> str:
    import os
    name = os.environ.get("PHASE06D_CANDIDATE", "baseline").strip()
    if name not in CANDIDATE_SETTINGS:
        raise ValueError(f"Unknown PHASE06D_CANDIDATE={name!r}; choose {sorted(CANDIDATE_SETTINGS)}")
    return name


def candidate_settings(name: str | None = None) -> dict:
    return dict(CANDIDATE_SETTINGS[name or candidate_name()])


def apply_bounded_profile_factor(profile: pd.Series, factor: float) -> tuple[pd.Series, float, float]:
    """Scale availability before dispatch, clip to [0, 1], and report energy."""
    before = float(profile.sum())
    scaled = profile.astype(float).mul(float(factor)).clip(lower=0.0, upper=1.0)
    return scaled, before, float(scaled.sum())


def apply_monthly_thermal_derating(network, evidence_path: Path) -> pd.DataFrame:
    """Apply aggregate CENACE monthly unavailable MW proportionally to thermal p_max_pu.

    This is explicitly a fleet-level-to-plant allocation assumption, not plant-level
    outage evidence. It only affects thermal generators and is bounded at zero.
    """
    # The legacy evidence extract has unquoted commas in its free-text notes.
    # Keep the structured prefix and do not discard the authoritative row.
    evidence = pd.read_csv(evidence_path, skipinitialspace=True, engine="python", on_bad_lines=lambda fields: fields[:11])
    evidence.columns = evidence.columns.astype(str).str.strip()
    for column in evidence.select_dtypes(include="object"):
        evidence[column] = evidence[column].astype(str).str.strip()
    row = evidence.loc[evidence["plant_id"].astype(str).eq("ALL_THERMAL")].iloc[0]
    # Figure 1.21 provides an aggregate annual average, not a complete monthly
    # table in this extract. Repeat that average transparently for the initial
    # fleet-level scenario; this is a modeling assumption, not plant evidence.
    values = {m: float(row["unavailable_mw"]) for m in range(1, 13)}
    thermal = network.generators.index[network.generators.carrier.isin(["oil", "gas", "thermal", "oil_and_gas"])]
    thermal = thermal.intersection(network.generators_t.p_max_pu.columns)
    if len(thermal) == 0:
        return pd.DataFrame(columns=["month", "unavailable_mw", "eligible_capacity_mw"])
    capacity = network.generators.loc[thermal, "p_nom"].astype(float)
    rows = []
    for month, unavailable in values.items():
        mask = network.snapshots.month == month
        eligible = float(capacity.sum())
        fleet_factor = max(0.0, 1.0 - unavailable / eligible) if eligible else 0.0
        network.generators_t.p_max_pu.loc[mask, thermal] = network.generators_t.p_max_pu.loc[mask, thermal].mul(fleet_factor).clip(lower=0.0, upper=1.0)
        rows.append({"month": month, "unavailable_mw": unavailable, "eligible_capacity_mw": eligible, "allocation": "proportional_capacity_assumption", "fleet_factor": fleet_factor})
    return pd.DataFrame(rows)


def write_candidate_manifest(root: Path, name: str, summary: dict, solved_path: Path) -> Path:
    out = root / "results" / "phase_06d" / name
    out.mkdir(parents=True, exist_ok=True)
    record = dict(summary)
    record.update({"candidate": name, "network": str(solved_path.relative_to(root)), "sha256": hashlib.sha256(solved_path.read_bytes()).hexdigest()})
    path = out / "candidate_summary.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path
