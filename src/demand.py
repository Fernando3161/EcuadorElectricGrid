"""Deterministic preparation and validation of the 2022 SNI demand profile."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_TARGET_COLUMNS = {
    "month_number",
    "month",
    "energy_gwh",
    "peak_mw",
    "source_id",
}


@dataclass(frozen=True)
class DemandPreparationResult:
    """Prepared hourly demand and the checks needed by the stage report."""

    profile_mw: pd.DataFrame
    monthly_diagnostics: pd.DataFrame
    summary: dict[str, float | int | str]


def expected_hourly_index(year: int) -> pd.DatetimeIndex:
    """Return the complete timezone-naive hourly index for a calendar year."""

    return pd.date_range(
        start=f"{year}-01-01 00:00:00",
        end=f"{year}-12-31 23:00:00",
        freq="h",
        name="timestamp",
    )


def validate_hourly_index(index: pd.DatetimeIndex, *, year: int) -> None:
    """Require a complete, ordered, duplicate-free local-standard-time year."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("demand profile index must be a DatetimeIndex")
    if index.tz is not None:
        raise ValueError("timestamps must be timezone-naive Ecuador local standard time")
    expected = expected_hourly_index(year)
    if not index.equals(expected):
        missing = expected.difference(index)
        extra = index.difference(expected)
        raise ValueError(
            f"hourly index is not the complete {year} calendar year: "
            f"expected {len(expected)} rows, found {len(index)}, "
            f"missing {len(missing)}, extra {len(extra)}"
        )


def load_monthly_targets(path: Path | str) -> pd.DataFrame:
    """Load and validate the official monthly energy and peak controls."""

    table = pd.read_csv(path)
    missing = REQUIRED_TARGET_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"monthly demand table is missing columns: {sorted(missing)}")
    table = table.copy()
    table["month_number"] = pd.to_numeric(table["month_number"], errors="raise").astype(int)
    table["energy_gwh"] = pd.to_numeric(table["energy_gwh"], errors="raise")
    table["peak_mw"] = pd.to_numeric(table["peak_mw"], errors="raise")
    if table["month_number"].tolist() != list(range(1, 13)):
        raise ValueError("monthly demand table must contain months 1 through 12 in order")
    expected_names = [calendar.month_name[month] for month in range(1, 13)]
    if table["month"].tolist() != expected_names:
        raise ValueError("monthly demand month labels do not match month_number")
    if table["source_id"].isna().any() or (table["source_id"].astype(str).str.len() == 0).any():
        raise ValueError("every monthly demand row must retain a source_id")
    if (table[["energy_gwh", "peak_mw"]] <= 0).any().any():
        raise ValueError("monthly energy and peak controls must be positive")
    return table


def load_hourly_template(path: Path | str) -> pd.DataFrame:
    """Load the historical hourly shape used only as a modeling template."""

    table = pd.read_csv(path)
    if table.shape[1] < 2:
        raise ValueError("hourly demand template must contain time and at least one bus")
    timestamp_column = table.columns[0]
    timestamps = pd.to_datetime(table.pop(timestamp_column), errors="raise")
    profile = table.apply(pd.to_numeric, errors="raise")
    profile.columns = profile.columns.astype(str)
    profile.index = pd.DatetimeIndex(timestamps, name="timestamp")
    year_values = profile.index.year.unique()
    if len(year_values) != 1:
        raise ValueError("hourly demand template must contain exactly one calendar year")
    validate_hourly_index(profile.index, year=int(year_values[0]))
    if profile.columns.duplicated().any():
        raise ValueError("hourly demand template contains duplicate bus columns")
    if not np.isfinite(profile.to_numpy()).all():
        raise ValueError("hourly demand template contains non-finite values")
    if (profile < 0).any().any():
        raise ValueError("hourly demand template contains negative values")
    if (profile.sum(axis=1) <= 0).any():
        raise ValueError("hourly demand template has a zero national-total hour")
    return profile


def reconcile_monthly_energy(
    monthly_targets: pd.DataFrame, *, annual_energy_gwh: float
) -> pd.DataFrame:
    """Reconcile published rounded months to the published annual control."""

    if annual_energy_gwh <= 0:
        raise ValueError("annual demand energy must be positive")
    table = monthly_targets.copy()
    published_sum = float(table["energy_gwh"].sum())
    if published_sum <= 0:
        raise ValueError("published monthly demand sum must be positive")
    factor = float(annual_energy_gwh) / published_sum
    table["model_energy_gwh"] = table["energy_gwh"] * factor
    table["annual_reconciliation_factor"] = factor
    return table


def prepare_hourly_demand(
    template: pd.DataFrame,
    monthly_targets: pd.DataFrame,
    *,
    target_year: int,
    annual_energy_gwh: float,
) -> DemandPreparationResult:
    """Reshape a historical profile to official monthly energy and peak controls.

    The method applies one affine transformation to the national load shape within
    each month. The transformed national series matches that month's target energy
    and peak. Every hourly bus share is then copied from the historical template,
    so the spatial distribution remains explicit and auditable.
    """

    source_years = template.index.year.unique()
    if len(source_years) != 1:
        raise ValueError("template must contain exactly one source year")
    source_year = int(source_years[0])
    validate_hourly_index(template.index, year=source_year)
    target_index = expected_hourly_index(target_year)
    if len(template) != len(target_index):
        raise ValueError(
            "source and target years have different hour counts; leap-day handling "
            "must be specified rather than silently interpolated"
        )

    targets = reconcile_monthly_energy(
        monthly_targets, annual_energy_gwh=annual_energy_gwh
    ).set_index("month_number")
    national_source = template.sum(axis=1)
    source_shares = template.div(national_source, axis=0)
    prepared = pd.DataFrame(index=target_index, columns=template.columns, dtype=float)
    diagnostics: list[dict[str, float | int | str]] = []

    for month in range(1, 13):
        source_mask = template.index.month == month
        target_mask = target_index.month == month
        source_month = national_source.loc[source_mask]
        hour_count = int(target_mask.sum())
        if len(source_month) != hour_count:
            raise ValueError(
                f"month {month} has different source and target hour counts"
            )

        target_energy_gwh = float(targets.loc[month, "model_energy_gwh"])
        published_energy_gwh = float(targets.loc[month, "energy_gwh"])
        target_peak_mw = float(targets.loc[month, "peak_mw"])
        target_mean_mw = target_energy_gwh * 1000.0 / hour_count
        source_range = float(source_month.max() - source_month.mean())
        if source_range <= 0:
            raise ValueError(f"month {month} has no usable variation in its source shape")
        if target_peak_mw <= target_mean_mw:
            raise ValueError(
                f"month {month} peak must be greater than its mean demand"
            )

        slope = (target_peak_mw - target_mean_mw) / source_range
        offset = target_mean_mw - slope * float(source_month.mean())
        transformed_national = slope * source_month.to_numpy() + offset
        if transformed_national.min() < -1e-9:
            raise ValueError(
                f"month {month} affine calibration produces negative demand"
            )
        transformed_national = np.maximum(transformed_national, 0.0)
        monthly_shares = source_shares.loc[source_mask].to_numpy()
        prepared.loc[target_mask, :] = monthly_shares * transformed_national[:, None]

        modeled_national = prepared.loc[target_mask].sum(axis=1)
        modeled_energy_gwh = float(modeled_national.sum() / 1000.0)
        modeled_peak_mw = float(modeled_national.max())
        diagnostics.append(
            {
                "month_number": month,
                "month": calendar.month_name[month],
                "published_energy_gwh": published_energy_gwh,
                "target_energy_gwh": target_energy_gwh,
                "modeled_energy_gwh": modeled_energy_gwh,
                "energy_error_gwh": modeled_energy_gwh - target_energy_gwh,
                "target_peak_mw": target_peak_mw,
                "modeled_peak_mw": modeled_peak_mw,
                "peak_error_mw": modeled_peak_mw - target_peak_mw,
                "shape_slope": slope,
                "shape_offset_mw": offset,
            }
        )

    prepared.index.name = "timestamp"
    monthly_diagnostics = pd.DataFrame(diagnostics)
    output_national = prepared.sum(axis=1)
    output_shares = prepared.div(output_national, axis=0)
    output_shares.index = template.index
    maximum_share_change = float(
        np.abs(output_shares.to_numpy() - source_shares.to_numpy()).max()
    )
    summary: dict[str, float | int | str] = {
        "target_year": target_year,
        "source_template_year": source_year,
        "snapshots": len(prepared),
        "buses": len(prepared.columns),
        "annual_energy_gwh": float(output_national.sum() / 1000.0),
        "annual_target_gwh": float(annual_energy_gwh),
        "annual_energy_error_gwh": float(output_national.sum() / 1000.0 - annual_energy_gwh),
        "annual_peak_mw": float(output_national.max()),
        "negative_values": int((prepared < 0).sum().sum()),
        "missing_values": int(prepared.isna().sum().sum()),
        "maximum_hourly_bus_share_change": maximum_share_change,
        "time_convention": "Ecuador local standard time (UTC-05:00), timezone-naive, no DST",
    }
    return DemandPreparationResult(prepared, monthly_diagnostics, summary)


def assert_demand_acceptance(result: DemandPreparationResult) -> None:
    """Raise when any deterministic Phase 3 demand invariant fails."""

    profile = result.profile_mw
    validate_hourly_index(profile.index, year=int(result.summary["target_year"]))
    if result.summary["negative_values"] != 0:
        raise AssertionError("prepared demand contains negative values")
    if result.summary["missing_values"] != 0:
        raise AssertionError("prepared demand contains missing values")
    if abs(float(result.summary["annual_energy_error_gwh"])) > 1e-8:
        raise AssertionError("prepared annual demand does not match the annual target")
    if result.monthly_diagnostics["energy_error_gwh"].abs().max() > 1e-8:
        raise AssertionError("prepared monthly energy does not match its targets")
    if result.monthly_diagnostics["peak_error_mw"].abs().max() > 1e-8:
        raise AssertionError("prepared monthly peaks do not match their targets")
    if float(result.summary["maximum_hourly_bus_share_change"]) > 1e-12:
        raise AssertionError("prepared demand changed the hourly spatial allocation")


__all__ = [
    "DemandPreparationResult",
    "assert_demand_acceptance",
    "expected_hourly_index",
    "load_hourly_template",
    "load_monthly_targets",
    "prepare_hourly_demand",
    "reconcile_monthly_energy",
    "validate_hourly_index",
]
