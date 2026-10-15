"""Normalization and validation of the 2022 Ecuador SNI generator fleet."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_COLUMNS = {
    "source_row",
    "operator_original",
    "plant_original",
    "province_original",
    "system_original",
    "technology_original",
    "subtechnology_original",
    "nominal_capacity_mw_original",
    "effective_capacity_mw_original",
    "source_document_id",
    "source_locator",
    "source_workbook",
    "transcription_status",
}

LOCATION_COLUMNS = {
    "source_row",
    "plant_original",
    "province_original",
    "canton",
    "latitude",
    "longitude",
    "location_method",
    "location_source",
    "evidence_note",
}

COMMISSIONING_COLUMNS = {
    "source_row",
    "plant_original",
    "event_date",
    "event_capacity_mw",
    "event_description",
    "source_document_id",
    "source_locator",
    "source_url",
    "evidence_note",
}

CARRIER_BY_TECHNOLOGY = {
    "hidraulica": "hydro",
    "hidrulica": "hydro",
    "termica": "oil",
    "trmica": "oil",
    "biomasa": "biomass",
    "biogas": "biogas",
    "biogs": "biogas",
    "solar": "solar",
    "eolica": "onwind",
    "elica": "onwind",
}


@dataclass(frozen=True)
class GenerationPreparationResult:
    """Normalized SNI fleet and auditable reconciliation summaries."""

    generators: pd.DataFrame
    carrier_summary: pd.DataFrame
    location_summary: pd.DataFrame


def ascii_token(value: object) -> str:
    """Return a stable ASCII identifier token without changing source fields."""

    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_")
    return re.sub(r"_+", "_", text)


def comparison_key(value: object) -> str:
    """Normalize display text for matching while tolerating legacy mojibake."""

    text = str(value).replace("�", "")
    return ascii_token(text).casefold()


def load_generation_source(path: Path | str) -> pd.DataFrame:
    """Load the source-faithful existing-generation transcription."""

    table = pd.read_csv(path)
    missing = SOURCE_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"generation source is missing columns: {sorted(missing)}")
    table = table.copy()
    table["source_row"] = pd.to_numeric(table["source_row"], errors="raise").astype(int)
    for column in ("nominal_capacity_mw_original", "effective_capacity_mw_original"):
        table[column] = pd.to_numeric(table[column], errors="raise")
    if table["source_row"].duplicated().any():
        raise ValueError("generation source contains duplicate source_row values")
    if (table[["nominal_capacity_mw_original", "effective_capacity_mw_original"]] < 0).any().any():
        raise ValueError("generation source contains negative capacities")
    if table[list(SOURCE_COLUMNS - {"source_row"})].isna().any().any():
        raise ValueError("generation source contains missing required provenance values")
    return table


def load_location_evidence(path: Path | str) -> pd.DataFrame:
    """Load one-to-one plant location evidence."""

    table = pd.read_csv(path)
    missing = LOCATION_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"location evidence is missing columns: {sorted(missing)}")
    table = table.copy()
    table["source_row"] = pd.to_numeric(table["source_row"], errors="raise").astype(int)
    table["latitude"] = pd.to_numeric(table["latitude"], errors="raise")
    table["longitude"] = pd.to_numeric(table["longitude"], errors="raise")
    if table["source_row"].duplicated().any():
        raise ValueError("location evidence must contain exactly one row per source_row")
    if not np.isfinite(table[["latitude", "longitude"]].to_numpy()).all():
        raise ValueError("location evidence contains non-finite coordinates")
    if not table["latitude"].between(-5.5, 2.0).all():
        raise ValueError("location latitude falls outside Ecuador bounds")
    if not table["longitude"].between(-81.5, -74.5).all():
        raise ValueError("location longitude falls outside Ecuador bounds")
    return table


def load_commissioning_events(path: Path | str) -> pd.DataFrame:
    """Load official in-year commissioning events without reconciling capacities."""

    table = pd.read_csv(path)
    missing = COMMISSIONING_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"commissioning events are missing columns: {sorted(missing)}")
    table = table.copy()
    table["source_row"] = pd.to_numeric(table["source_row"], errors="raise").astype(int)
    table["event_capacity_mw"] = pd.to_numeric(
        table["event_capacity_mw"], errors="raise"
    )
    table["event_date"] = pd.to_datetime(table["event_date"], errors="raise")
    if table.duplicated(["source_row", "event_date", "event_description"]).any():
        raise ValueError("commissioning events contain duplicate source/date/event rows")
    if (table["event_capacity_mw"] <= 0).any():
        raise ValueError("commissioning event capacities must be positive")
    if not (table["event_date"].dt.year == 2022).all():
        raise ValueError("Phase 4 commissioning evidence must contain 2022 events")
    return table


def _map_carrier(technology: object) -> str:
    key = comparison_key(technology)
    try:
        return CARRIER_BY_TECHNOLOGY[key]
    except KeyError as exc:
        raise ValueError(f"unmapped generation technology: {technology!r}") from exc


def prepare_existing_generation(
    source: pd.DataFrame,
    locations: pd.DataFrame,
    commissioning_events: pd.DataFrame | None = None,
) -> GenerationPreparationResult:
    """Filter the mainland SNI fleet and normalize it without adding capacity."""

    source = source.copy()
    sni = source[source["system_original"].map(comparison_key) == "s_n_i"].copy()
    if sni.empty:
        raise ValueError("no SNI rows were found in the generation source")

    expected_rows = set(sni["source_row"])
    supplied_rows = set(locations["source_row"])
    if supplied_rows != expected_rows:
        missing = sorted(expected_rows - supplied_rows)
        extra = sorted(supplied_rows - expected_rows)
        raise ValueError(
            f"location evidence does not exactly cover the SNI fleet: missing={missing}, extra={extra}"
        )

    joined = sni.merge(
        locations,
        on="source_row",
        how="left",
        validate="one_to_one",
        suffixes=("", "_location"),
    )
    if not (
        joined["plant_original"] == joined["plant_original_location"]
    ).all() or not (
        joined["province_original"] == joined["province_original_location"]
    ).all():
        raise ValueError("location evidence plant/province labels do not match source rows")

    first_operation = {}
    if commissioning_events is not None:
        unknown_rows = sorted(set(commissioning_events["source_row"]) - expected_rows)
        if unknown_rows:
            raise ValueError(
                f"commissioning events reference non-SNI source rows: {unknown_rows}"
            )
        labels = sni.set_index("source_row")["plant_original"]
        for row in commissioning_events.itertuples(index=False):
            if row.plant_original != labels.loc[row.source_row]:
                raise ValueError(
                    f"commissioning event plant label does not match source row {row.source_row}"
                )
        first_operation = (
            commissioning_events.groupby("source_row")["event_date"].min().to_dict()
        )

    result = pd.DataFrame(
        {
            "plant_id": joined.apply(
                lambda row: f"ec2022_{int(row.source_row):03d}_{ascii_token(row.plant_original)}",
                axis=1,
            ),
            "source_row": joined["source_row"].astype(int),
            "name": joined["plant_original"],
            "operator": joined["operator_original"],
            "province": joined["province_original"],
            "canton": joined["canton"],
            "carrier": joined["technology_original"].map(_map_carrier),
            "technology_original": joined["technology_original"],
            "subtechnology_original": joined["subtechnology_original"],
            "p_nom_mw": joined["nominal_capacity_mw_original"].astype(float),
            "effective_capacity_mw": joined["effective_capacity_mw_original"].astype(float),
            "latitude": joined["latitude"].astype(float),
            "longitude": joined["longitude"].astype(float),
            "location_method": joined["location_method"],
            "location_source": joined["location_source"],
            "source_document_id": joined["source_document_id"],
            "source_locator": joined["source_locator"],
            "operating_start_date_2022": joined["source_row"].map(first_operation).fillna(
                pd.Timestamp("2022-01-01")
            ),
            "operating_start_basis": joined["source_row"].map(
                {row: "cenace_2022_commissioning_event" for row in first_operation}
            ).fillna("existing_inventory_full_year_assumption"),
            "commissioning_year": joined["source_row"].map(
                {row: date.year for row, date in first_operation.items()}
            ).astype("Int64"),
            "retirement_year": pd.Series(pd.NA, index=joined.index, dtype="Int64"),
            "baseline_status": "existing_2022",
        }
    )
    if result["plant_id"].duplicated().any():
        raise AssertionError("stable plant identifiers are not unique")

    carrier_summary = (
        result.groupby("carrier", as_index=False)
        .agg(
            plants=("plant_id", "size"),
            nominal_capacity_mw=("p_nom_mw", "sum"),
            effective_capacity_mw=("effective_capacity_mw", "sum"),
        )
        .sort_values("carrier")
        .reset_index(drop=True)
    )
    location_summary = (
        result.groupby("location_method", as_index=False)
        .agg(
            plants=("plant_id", "size"),
            effective_capacity_mw=("effective_capacity_mw", "sum"),
        )
        .sort_values("location_method")
        .reset_index(drop=True)
    )
    return GenerationPreparationResult(result, carrier_summary, location_summary)


def assert_generation_acceptance(
    result: GenerationPreparationResult,
    *,
    expected_rows: int = 133,
    expected_nominal_mw: float = 7489.65,
    expected_effective_mw: float = 7175.11,
) -> None:
    """Raise when a Phase 4 fleet invariant is violated."""

    generators = result.generators
    if len(generators) != expected_rows:
        raise AssertionError(f"expected {expected_rows} SNI plants, found {len(generators)}")
    if not generators["plant_id"].is_unique:
        raise AssertionError("plant_id is not unique")
    if abs(float(generators["p_nom_mw"].sum()) - expected_nominal_mw) > 1e-6:
        raise AssertionError("nominal capacity does not reconcile to the source")
    if abs(float(generators["effective_capacity_mw"].sum()) - expected_effective_mw) > 1e-6:
        raise AssertionError("effective capacity does not reconcile to the source")
    if generators[["latitude", "longitude", "carrier"]].isna().any().any():
        raise AssertionError("normalized fleet contains missing model fields")
    if set(generators["baseline_status"]) != {"existing_2022"}:
        raise AssertionError("future or hypothetical capacity leaked into the baseline")


__all__ = [
    "GenerationPreparationResult",
    "ascii_token",
    "assert_generation_acceptance",
    "comparison_key",
    "load_generation_source",
    "load_commissioning_events",
    "load_location_evidence",
    "prepare_existing_generation",
]
