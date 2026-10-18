"""Hydro and variable-renewable availability preparation for the 2022 SNI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.demand import expected_hourly_index, validate_hourly_index
from src.generation import comparison_key


ASSIGNMENT_COLUMNS = {
    "plant_id",
    "source_row",
    "plant_name",
    "caudal_id",
    "assignment_basis",
    "confidence",
    "evidence_note",
}


def load_monthly_flows(path: Path | str) -> pd.DataFrame:
    """Load eight source flow series with twelve non-negative monthly values."""

    table = pd.read_csv(path)
    expected = ["central"] + [str(month) for month in range(1, 13)]
    if table.columns.tolist() != expected:
        raise ValueError(f"monthly flow columns must be {expected}")
    if table["central"].duplicated().any():
        raise ValueError("monthly flow identifiers are duplicated")
    values = table.set_index("central").apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
        raise ValueError("monthly flows must be finite and non-negative")
    return values


def load_hydro_assignments(path: Path | str) -> pd.DataFrame:
    """Load reviewed plant-to-flow assignments and their confidence evidence."""

    table = pd.read_csv(path)
    missing = ASSIGNMENT_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"hydro assignments are missing columns: {sorted(missing)}")
    if table["plant_id"].duplicated().any():
        raise ValueError("hydro assignments contain duplicate plant_id values")
    allowed_confidence = {"high", "medium", "low"}
    if not set(table["confidence"]).issubset(allowed_confidence):
        raise ValueError("hydro assignment confidence must be high, medium, or low")
    return table


def _monthly_to_hourly(values: pd.Series, *, year: int) -> pd.Series:
    anchors = []
    for month in range(1, 13):
        start = pd.Timestamp(year, month, 1)
        anchors.append(start + pd.Timedelta(hours=12) + pd.Timedelta(days=start.days_in_month / 2))
    monthly = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(anchors))
    hourly_index = expected_hourly_index(year)
    expanded = monthly.reindex(monthly.index.union(hourly_index)).sort_index()
    return expanded.interpolate(method="time").reindex(hourly_index).bfill().ffill()


def build_hydro_availability(
    generators: pd.DataFrame,
    flows: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    year: int = 2022,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build hourly hydro p_max_pu using the preserved legacy flow method."""

    hydro = generators[generators["carrier"] == "hydro"].copy()
    expected = set(hydro["plant_id"])
    supplied = set(assignments["plant_id"])
    if supplied != expected:
        raise ValueError(
            "hydro assignments do not exactly cover the fleet: "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
        )
    unknown_flows = sorted(set(assignments["caudal_id"]) - set(flows.index))
    if unknown_flows:
        raise ValueError(f"hydro assignments reference unknown flow series: {unknown_flows}")

    joined = hydro.merge(
        assignments,
        on="plant_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_assignment"),
    )
    output = pd.DataFrame(index=expected_hourly_index(year))
    diagnostics: list[dict[str, object]] = []
    for row in joined.itertuples(index=False):
        monthly_flow = flows.loc[row.caudal_id].astype(float)
        is_reservoir = comparison_key(row.subtechnology_original) == "embalse"
        effective_flow = (
            monthly_flow.rolling(3, min_periods=1).mean()
            if is_reservoir
            else monthly_flow
        )
        # Phase 06H: normalize run-of-river flow by its empirical monthly
        # 0.90 quantile; reservoir profiles retain the original 0.95 rule.
        rated_flow = float(effective_flow.quantile(0.90 if not is_reservoir else 0.95))
        if not np.isfinite(rated_flow) or rated_flow <= 0:
            raise ValueError(f"invalid rated flow for {row.plant_id}")
        monthly_pu = (effective_flow / rated_flow).clip(0.0, 1.0)
        output[row.plant_id] = _monthly_to_hourly(monthly_pu, year=year)
        operating_start = pd.Timestamp(row.operating_start_date_2022)
        output.loc[output.index < operating_start, row.plant_id] = 0.0
        diagnostics.append(
            {
                "plant_id": row.plant_id,
                "plant_name": row.name,
                "caudal_id": row.caudal_id,
                "hydro_model": "reservoir_3_month_rolling" if is_reservoir else "run_of_river",
                "assignment_basis": row.assignment_basis,
                "confidence": row.confidence,
                "effective_capacity_mw": row.effective_capacity_mw,
                "operating_start_date_2022": operating_start.date().isoformat(),
                "minimum_p_max_pu": float(output[row.plant_id].min()),
                "maximum_p_max_pu": float(output[row.plant_id].max()),
            }
        )
    assert_profile_acceptance(output, expected_columns=expected, year=year)
    return output, pd.DataFrame(diagnostics)


def load_province_profile(path: Path | str) -> tuple[pd.DataFrame, int]:
    """Load a bus/province NetCDF profile and return its source weather year."""

    with xr.open_dataset(path) as dataset:
        if "profile" not in dataset or set(dataset["profile"].dims) != {"time", "bus"}:
            raise ValueError("renewable NetCDF must provide profile(time, bus)")
        time = pd.DatetimeIndex(dataset["time"].values, name="timestamp")
        years = time.year.unique()
        if len(years) != 1:
            raise ValueError("renewable profile must contain exactly one weather year")
        source_year = int(years[0])
        validate_hourly_index(time, year=source_year)
        provinces = [comparison_key(value) for value in dataset["bus"].values]
        if len(set(provinces)) != len(provinces):
            raise ValueError("renewable profile province keys are not unique")
        profile = pd.DataFrame(
            dataset["profile"].values,
            index=time,
            columns=provinces,
        )
    assert_profile_acceptance(profile, expected_columns=set(provinces), year=source_year)
    return profile, source_year


def attach_renewable_availability(
    generators: pd.DataFrame,
    *,
    solar_profile_path: Path | str,
    wind_profile_path: Path | str,
    target_year: int = 2022,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach 2013 provincial weather shapes to 2022 solar and wind plants."""

    solar, solar_year = load_province_profile(solar_profile_path)
    wind, wind_year = load_province_profile(wind_profile_path)
    if solar_year != wind_year:
        raise ValueError("solar and wind profiles use different weather years")
    target_index = expected_hourly_index(target_year)
    if len(solar) != len(target_index):
        raise ValueError("weather and target years have different hour counts")

    renewable = generators[generators["carrier"].isin(["solar", "onwind"])].copy()
    output = pd.DataFrame(index=target_index)
    diagnostics: list[dict[str, object]] = []
    for row in renewable.itertuples(index=False):
        province_key = comparison_key(row.province)
        source = solar if row.carrier == "solar" else wind
        if province_key not in source.columns:
            raise ValueError(
                f"no {row.carrier} province profile for {row.plant_id}: {row.province}"
            )
        values = source[province_key].to_numpy(dtype=float)
        output[row.plant_id] = values
        operating_start = pd.Timestamp(row.operating_start_date_2022)
        output.loc[output.index < operating_start, row.plant_id] = 0.0
        diagnostics.append(
            {
                "plant_id": row.plant_id,
                "plant_name": row.name,
                "carrier": row.carrier,
                "province": row.province,
                "weather_year": solar_year,
                "simulation_year": target_year,
                "operating_start_date_2022": operating_start.date().isoformat(),
                "mean_p_max_pu": float(output[row.plant_id].mean()),
                "maximum_p_max_pu": float(output[row.plant_id].max()),
            }
        )
    assert_profile_acceptance(
        output, expected_columns=set(renewable["plant_id"]), year=target_year
    )
    return output, pd.DataFrame(diagnostics)


def assert_profile_acceptance(
    profile: pd.DataFrame, *, expected_columns: set[str], year: int
) -> None:
    """Require exact plant coverage, complete time, and finite [0, 1] values."""

    validate_hourly_index(profile.index, year=year)
    if set(profile.columns) != expected_columns or profile.columns.duplicated().any():
        raise AssertionError("availability profile plant coverage is not exact")
    values = profile.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("availability profile contains non-finite values")
    if (values < -1e-12).any() or (values > 1.0 + 1e-12).any():
        raise AssertionError("availability profile values fall outside [0, 1]")


__all__ = [
    "assert_profile_acceptance",
    "attach_renewable_availability",
    "build_hydro_availability",
    "load_hydro_assignments",
    "load_monthly_flows",
    "load_province_profile",
]
