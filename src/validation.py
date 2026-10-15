"""Carrier taxonomy and the fixed 2022 statistical acceptance calculation."""

from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml


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
    """Apply the exact 5 percent gate, including missing and zero carriers."""

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

    relative_limit = float(contract["acceptance"]["maximum_relative_error_percent"])
    zero_limit = float(contract["acceptance"]["zero_reference_absolute_tolerance_gwh"])
    rows: list[dict[str, float | bool | str]] = []
    for row in reference.loc[reference["in_scope"]].itertuples(index=False):
        carrier = str(row.carrier)
        target = float(row.reference_gwh)
        actual = float(simulated.get(carrier, 0.0))
        absolute_error = abs(actual - target)
        if target > 0:
            relative_error = 100.0 * absolute_error / target
            passed = relative_error <= relative_limit + 1e-12
            tolerance_type = "relative_percent"
            tolerance_value = relative_limit
        else:
            relative_error = math.nan
            passed = absolute_error <= zero_limit + 1e-12
            tolerance_type = "absolute_gwh"
            tolerance_value = zero_limit
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


__all__ = [
    "DEFAULT_CONTRACT",
    "aggregate_model_supply",
    "build_supply_reference",
    "compare_supply_to_reference",
    "load_raw_supply_reference",
    "load_validation_contract",
    "map_model_carriers",
    "normalize_carrier_label",
]
