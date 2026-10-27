"""Reusable Phase 06B calibration controls and diagnostic aggregations."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import os
from urllib.parse import urlencode, quote
from urllib.request import urlopen


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
    monthly = pd.Series(monthly_gwh).reindex(range(1, 13)).fillna(0.0)
    return monthly.reindex(index.month).to_numpy(dtype=float) / counts.to_numpy(dtype=float)


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


def build_biomass_availability(index: pd.DatetimeIndex, plants: pd.DataFrame, source_table: pd.DataFrame, *, annual_net_target_gwh: float = 195.68) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a shared monthly biomass p.u. cap using ARCONEL shape and CENACE net total.

    ARCONEL Table 54 supplies the gross seasonal shape. CENACE Table 2.4 supplies
    the independent net annual validation target; the two are kept explicit.
    """
    biomass = plants.loc[plants.carrier.eq("biomass"), ["plant_id", "p_nom_mw"]].copy()
    source = source_table.set_index("month")["biomass_gwh"].astype(float)
    source_share = source / source.sum()
    total_capacity = float(biomass.p_nom_mw.sum())
    profiles = pd.DataFrame(index=index)
    for row in biomass.itertuples(index=False):
        values = pd.Series(index.month, index=index).map(source_share).astype(float)
        hours = values.groupby(index.month).transform("size")
        monthly_mwh = values * float(annual_net_target_gwh) * 1000.0
        profiles[row.plant_id] = (monthly_mwh / hours / total_capacity).clip(upper=1.0)
    diagnostics = biomass.assign(carrier="biomass", source_boundary="ARCONEL gross shape / CENACE net annual control", annual_net_target_gwh=annual_net_target_gwh)
    return profiles, diagnostics


def aggregate_dispatch_by_carrier(network) -> pd.DataFrame:
    """Return dispatch, potential, curtailment, and capacity factors by carrier."""
    weights = network.snapshot_weightings.generators.reindex(network.snapshots).to_numpy()
    dispatch = network.generators_t.p.reindex(columns=network.generators.index).fillna(0.0)
    dynamic = network.generators_t.p_max_pu.reindex(index=network.snapshots, columns=network.generators.index)
    static = network.generators.p_max_pu.reindex(network.generators.index).fillna(1.0)
    potential_pu = dynamic.fillna(pd.DataFrame(np.broadcast_to(static.to_numpy(), dynamic.shape), index=dynamic.index, columns=dynamic.columns))
    potential = potential_pu * network.generators.p_nom
    rows = []
    for carrier, names in network.generators.groupby("carrier").groups.items():
        if (dispatch[names] < -1e-9).any().any():
            raise ValueError(f"negative dispatch found for carrier {carrier}")
        d = float((dispatch[names].sum(axis=1).to_numpy() * weights).sum() / 1000.0)
        p = float((potential[names].sum(axis=1).to_numpy() * weights).sum() / 1000.0)
        rows.append({"carrier": carrier, "dispatch_gwh": d, "potential_gwh": p, "curtailment_gwh": max(p-d, 0.0), "capacity_factor": d / (float(network.generators.loc[names, "p_nom"].sum()) * weights.sum() / 1000.0) if weights.sum() else np.nan})
    return pd.DataFrame(rows).sort_values("carrier").reset_index(drop=True)


def nlr_sup3rwind_csv_url(longitude: float, latitude: float, year: int = 2022, *, api_key: str | None = None, email: str | None = None) -> str:
    """Build the documented NLR Sup3rWind South America 60-minute CSV URL."""
    params = {"wkt": f"POINT({longitude} {latitude})", "names": str(year), "interval": "60"}
    params["api_key"] = api_key or os.environ.get("NLR_API_KEY", "<NLR_API_KEY>")
    params["email"] = email or os.environ.get("NLR_EMAIL", "<NLR_EMAIL>")
    base = "https://developer.nlr.gov/api/wind-toolkit/v2/wind/wtk-sup3rwind-south-america-v1-0-0-60min-download.csv"
    return base + "?" + urlencode(params, quote_via=quote)


def read_nlr_wind_csv(path: Path | str, *, year: int = 2022, hub_height_m: int = 100) -> pd.DataFrame:
    """Parse an NLR single-site CSV into an hourly wind-speed table."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Year,"))
    frame = pd.read_csv(path, skiprows=header)
    scalar_speed = f"wind speed at {hub_height_m}m (m/s)"
    speed = f"zonal wind component at {hub_height_m}m (m/s)"
    meridional = f"meridional wind component at {hub_height_m}m (m/s)"
    if scalar_speed in frame:
        values = frame[scalar_speed].to_numpy()
    elif speed in frame and meridional in frame:
        values = np.hypot(frame[speed], frame[meridional]).to_numpy()
    else:
        raise ValueError(f"NLR CSV lacks {hub_height_m} m wind speed data")
    index = pd.to_datetime(frame[["Year", "Month", "Day", "Hour", "Minute"]].rename(columns={"Year":"year", "Month":"month", "Day":"day", "Hour":"hour", "Minute":"minute"}))
    if set(index.dt.year) != {year}:
        raise ValueError(f"NLR CSV year is not {year}")
    return pd.DataFrame({"wind_speed_m_s": values}, index=index)


def wind_speed_to_pu(speed_m_s: pd.Series, *, cut_in: float = 3.0, rated: float = 12.0, cut_out: float = 25.0) -> pd.Series:
    """Apply an explicit simple turbine power curve for diagnostic comparison."""
    speed = pd.to_numeric(speed_m_s, errors="raise").clip(lower=0)
    pu = ((speed**3 - cut_in**3) / (rated**3 - cut_in**3)).clip(lower=0, upper=1)
    return pu.where(speed < cut_out, 0.0)


def scale_profile_to_annual_energy(profile: pd.Series, capacity_mw: float, target_gwh: float) -> tuple[pd.Series, float]:
    """Return a bounded p.u. profile scaled to a transparent annual target."""
    raw_gwh = float(profile.sum() * capacity_mw / 1000.0)
    if raw_gwh <= 0:
        raise ValueError("cannot scale a zero-energy wind profile")
    factor = float(target_gwh) / raw_gwh
    scaled = (profile * factor).clip(upper=1.0)
    achieved = float(scaled.sum() * capacity_mw / 1000.0)
    if not np.isclose(achieved, target_gwh, rtol=1e-9, atol=1e-9):
        raise ValueError(f"target {target_gwh} GWh is infeasible under p.u. <= 1; achievable {achieved} GWh")
    return scaled, factor


__all__ = ["aggregate_dispatch_by_carrier", "apply_monthly_biomass_limits", "build_biomass_availability", "load_monthly_biomass_shape", "monthly_energy_profile", "nlr_sup3rwind_csv_url", "read_nlr_wind_csv", "scale_profile_to_annual_energy", "wind_speed_to_pu"]
