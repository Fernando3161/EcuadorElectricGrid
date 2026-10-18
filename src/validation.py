"""Carrier taxonomy and reproducible multi-criteria 2022 validation."""

from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "config" / "validation_2022.yaml"
REQUIRED_RAW_REFERENCE_COLUMNS = {
    "source_carrier",
    "reference_gwh",
    "source_class",
    "source_id",
    "source_page",
    "source_table",
    "evidence_class",
}


def load_validation_contract(path: Path | str = DEFAULT_CONTRACT) -> dict:
    """Load the versioned 2022 boundary, taxonomy, and tolerance contract."""

    with Path(path).open(encoding="utf-8") as handle:
        contract = yaml.safe_load(handle)
    required = {
        "contract_version",
        "year",
        "boundary",
        "sources",
        "demand",
        "supply_reference",
        "model_carrier_mapping",
        "acceptance",
    }
    missing = required.difference(contract)
    if missing:
        raise ValueError(f"validation contract is missing sections: {sorted(missing)}")
    return contract


def normalize_carrier_label(value: str) -> str:
    """Normalize accents and separators without guessing a carrier mapping."""

    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    return "_".join(ascii_value.strip().lower().replace("-", " ").split())


def load_raw_supply_reference(path: Path | str) -> pd.DataFrame:
    """Load the source-faithful CENACE Table 2.4 transcription."""

    table = pd.read_csv(path)
    missing = REQUIRED_RAW_REFERENCE_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"raw supply reference is missing columns: {sorted(missing)}")
    table = table.copy()
    table["source_carrier"] = table["source_carrier"].map(normalize_carrier_label)
    if table["source_carrier"].duplicated().any():
        raise ValueError("raw supply reference contains duplicate source carriers")
    table["reference_gwh"] = pd.to_numeric(table["reference_gwh"], errors="raise")
    if (table["reference_gwh"] < 0).any():
        raise ValueError("raw supply reference contains negative energy")
    if (table["evidence_class"] != "official_fact").any():
        raise ValueError("raw supply reference must contain only official facts")
    if table[["source_id", "source_page", "source_table"]].isna().any().any():
        raise ValueError("every raw supply row must retain a source locator")
    return table


def build_supply_reference(raw: pd.DataFrame, contract: Mapping) -> pd.DataFrame:
    """Aggregate official source rows to the model-comparable carrier taxonomy."""

    section = contract["supply_reference"]
    mapping = {
        normalize_carrier_label(source): target
        for source, target in section["raw_to_validation_carrier"].items()
    }
    unknown = sorted(set(raw["source_carrier"]).difference(mapping))
    if unknown:
        raise ValueError(f"unmapped official supply carriers: {unknown}")

    working = raw.copy()
    working["carrier"] = working["source_carrier"].map(mapping)
    grouped = (
        working.groupby("carrier", sort=False, as_index=False)
        .agg(
            reference_gwh=("reference_gwh", "sum"),
            source_components=("source_carrier", lambda values: ";".join(values)),
            source_ids=("source_id", lambda values: ";".join(dict.fromkeys(values))),
        )
    )
    expected = section["validation_carriers"]
    missing = sorted(set(expected).difference(grouped["carrier"]))
    if missing:
        raise ValueError(f"normalized supply reference is missing carriers: {missing}")
    grouped["in_scope"] = True
    grouped["evidence_class"] = "official_fact"
    grouped["notes"] = ""

    reported_total = float(section["reported_total_gwh"])
    residual = round(reported_total - float(grouped["reference_gwh"].sum()), 2)
    if abs(residual) > 0.05:
        raise ValueError(
            f"source rows differ from the reported supply total by {residual:.6f} GWh"
        )
    if not math.isclose(residual, 0.0, abs_tol=1e-12):
        grouped.loc[len(grouped)] = {
            "carrier": "source_rounding_residual",
            "reference_gwh": residual,
            "source_components": "reported_total_minus_displayed_rows",
            "source_ids": "CENACE_ANNUAL_2022",
            "in_scope": False,
            "evidence_class": "derived_reconciliation",
            "notes": "Displayed source rows are rounded to 0.01 GWh.",
        }

    order = {carrier: position for position, carrier in enumerate(expected)}
    order["source_rounding_residual"] = len(order)
    grouped["_order"] = grouped["carrier"].map(order)
    grouped = grouped.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return grouped


def map_model_carriers(
    carriers: pd.Index | pd.Series | list[str], contract: Mapping
) -> pd.Series:
    """Map model carrier labels to the frozen validation taxonomy."""

    values = pd.Series(list(carriers), dtype="object")
    mapping = {
        normalize_carrier_label(source): target
        for source, target in contract["model_carrier_mapping"].items()
    }
    normalized = values.map(normalize_carrier_label)
    mapped = normalized.map(mapping)
    unknown = sorted(values[mapped.isna()].astype(str).unique())
    if unknown:
        raise ValueError(f"unmapped model carriers: {unknown}")
    mapped.index = values.index
    return mapped


def aggregate_model_supply(
    supply_by_model_carrier_gwh: pd.Series, contract: Mapping
) -> pd.Series:
    """Aggregate simulated positive supply to validation carriers."""

    values = pd.to_numeric(supply_by_model_carrier_gwh, errors="raise")
    if (values < 0).any():
        raise ValueError("supply-by-carrier values must be non-negative")
    mapped = map_model_carriers(list(values.index), contract)
    mapped.index = values.index
    return values.groupby(mapped).sum().sort_index()


def compare_supply_to_reference(
    simulated_gwh: pd.Series,
    reference: pd.DataFrame,
    contract: Mapping,
) -> pd.DataFrame:
    """Return carrier diagnostics; acceptance is handled by TVD and pp screens."""

    required_columns = {"carrier", "reference_gwh", "in_scope"}
    missing_columns = required_columns.difference(reference.columns)
    if missing_columns:
        raise ValueError(f"normalized reference is missing columns: {sorted(missing_columns)}")
    simulated = pd.to_numeric(simulated_gwh, errors="raise")
    if simulated.index.duplicated().any():
        raise ValueError("simulated validation carriers must be unique")
    expected_carriers = set(reference.loc[reference["in_scope"], "carrier"])
    unknown = sorted(set(simulated.index).difference(expected_carriers))
    if unknown:
        raise ValueError(f"simulated series contains out-of-contract carriers: {unknown}")

    rows: list[dict[str, float | bool | str]] = []
    for row in reference.loc[reference["in_scope"]].itertuples(index=False):
        carrier = str(row.carrier)
        target = float(row.reference_gwh)
        actual = float(simulated.get(carrier, 0.0))
        absolute_error = abs(actual - target)
        if target > 0:
            relative_error = 100.0 * absolute_error / target
            passed = None
            tolerance_type = "diagnostic_only"
            tolerance_value = math.nan
        else:
            relative_error = math.nan
            passed = None
            tolerance_type = "diagnostic_only"
            tolerance_value = math.nan
        rows.append(
            {
                "carrier": carrier,
                "reference_gwh": target,
                "simulated_gwh": actual,
                "absolute_error_gwh": absolute_error,
                "relative_error_percent": relative_error,
                "tolerance_type": tolerance_type,
                "tolerance_value": tolerance_value,
                "passed": passed,
            }
        )
    return pd.DataFrame(rows)


def validate_annual_coverage(snapshots, year: int = 2022) -> tuple[bool, list[str]]:
    """Return comparability and explicit reasons for a complete hourly year."""
    idx = pd.DatetimeIndex(snapshots)
    reasons = []
    if len(idx) != (366 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 365) * 24:
        reasons.append(f"expected {8760 if not pd.Timestamp(year=year, month=12, day=31).is_leap_year else 8784} hourly snapshots, found {len(idx)}")
    if len(idx) and set(idx.year) != {year}:
        reasons.append(f"snapshots include years {sorted(set(idx.year))}, expected {year}")
    if len(idx) and idx.duplicated().any():
        reasons.append("snapshot index contains duplicates")
    return not reasons, reasons


def calculate_validation_metrics(simulated_gwh: pd.Series, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Calculate metrics independently of any screening thresholds."""
    required = {"carrier", "reference_gwh", "in_scope"}
    if not required.issubset(reference.columns):
        raise ValueError(f"reference is missing columns: {sorted(required - set(reference.columns))}")
    ref = reference[reference["in_scope"]].set_index("carrier")["reference_gwh"].astype(float)
    sim = simulated_gwh.astype(float).reindex(ref.index, fill_value=0.0)
    if (sim < 0).any() or (ref < 0).any():
        raise ValueError("energy values must be non-negative")
    ref_total, sim_total = float(ref.sum()), float(sim.sum())
    ref_share, sim_share = ref / ref_total, sim / sim_total if sim_total else ref * 0
    signed_pp = 100 * (sim_share - ref_share)
    relative = (sim - ref) / ref.replace(0, np.nan)
    rows = pd.DataFrame({
        "carrier": ref.index, "reference_gwh": ref.values, "simulated_gwh": sim.values,
        "reference_share": ref_share.values, "simulated_share": sim_share.values,
        "signed_error_gwh": (sim - ref).values, "absolute_error_gwh": abs(sim - ref).values,
        "signed_error_pp": signed_pp.values, "absolute_error_pp": abs(signed_pp).values,
        "relative_error": relative.values, "absolute_relative_error": abs(relative).values,
    })
    metrics = {
        "reference_total_gwh": ref_total, "simulated_total_gwh": sim_total,
        "total_energy_relative_error": abs(sim_total - ref_total) / ref_total if ref_total else np.nan,
        "tvd": float(0.5 * abs(sim_share - ref_share).sum()),
    }
    return rows, metrics


def screen_validation(metrics: dict, rows: pd.DataFrame, criteria: Mapping) -> dict:
    """Apply configurable exploratory criteria after metric calculation."""
    checks = {}
    for key, value in criteria.items():
        if value is None or key.startswith("enabled_"):
            continue
        if key == "max_tvd": checks[key] = metrics["tvd"] <= float(value)
        elif key.startswith("max_") and key.endswith("_error_pp"):
            carrier = key[len("max_") : -len("_error_pp")]
            match = rows.loc[rows.carrier == carrier, "absolute_error_pp"]
            checks[key] = bool(not match.empty and match.iloc[0] <= float(value))
    return {"checks": checks, "passed": all(checks.values()) if checks else True,
            "status": "PASS" if all(checks.values()) else "FAIL"}


def enforce_historical_import_energy(network, snapshots, contract: Mapping) -> None:
    """PyPSA optimization callback for the configurable 2022 import-energy band."""
    section = contract["interconnection_colombia"]
    if not section.get("enforce_historical_energy_band", False):
        return
    import xarray as xr

    lower_mwh = float(section["historical_energy_min_gwh"]) * 1000.0
    upper_mwh = float(section["historical_energy_max_gwh"]) * 1000.0
    dispatch = network.model.variables["Generator-p"].sel(
        Generator=["colombia_import_supply"]
    )
    weights = xr.DataArray(
        network.snapshot_weightings.generators.reindex(snapshots).to_numpy(),
        coords={"snapshot": snapshots}, dims=("snapshot",),
    )
    annual = (dispatch * weights).sum()
    network.model.add_constraints(
        annual >= lower_mwh, name="historical-import-energy-lower"
    )
    network.model.add_constraints(
        annual <= upper_mwh, name="historical-import-energy-upper"
    )


__all__ = [
    "DEFAULT_CONTRACT",
    "aggregate_model_supply",
    "build_supply_reference",
    "calculate_validation_metrics",
    "compare_supply_to_reference",
    "enforce_historical_import_energy",
    "load_raw_supply_reference",
    "load_validation_contract",
    "map_model_carriers",
    "normalize_carrier_label",
    "screen_validation",
    "validate_annual_coverage",
]
