"""One-time, deterministic migration of legacy Phase 4 generation workbooks.

The supported notebooks consume the resulting open CSV tables. This script is
retained so the workbook-to-CSV transcription and legacy location evidence can
be independently audited; it is not part of the normal student workflow.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.generation import (
    ascii_token,
    comparison_key,
    load_generation_source,
    load_location_evidence,
    prepare_existing_generation,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "generation"
INTERIM = ROOT / "data" / "interim" / "generation"

MANUAL_CANTON = {
    "Huascachaca": "Saraguro",
    "Sarapullo": "Mejia",
    "Sabanilla": "Zamora",
    "Rio_Verde_Chico": "Banos_de_Agua_Santa",
    "Chalpi": "Quijos",
    "San_Jose_de_Tambo": "Chillanes",
    "San_Jose_de_Minas": "Quito",
    "Ulba": "Banos_de_Agua_Santa",
    "Brineforcorp": "San_Vicente",
    "El_Laurel": "Mira",
}


def clean_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table.columns = table.columns.str.strip().str.replace("\n", " ", regex=False)
    table = table.rename(
        columns={
            "Potencia Nominal MW": "Potencia Nominal (MW)",
            "Potencia Efectiva MW": "Potencia Efectiva (MW)",
        }
    )
    canton_columns = [column for column in table.columns if str(column).startswith("Cant")]
    if len(canton_columns) == 1 and canton_columns[0] != "Canton":
        table = table.rename(columns={canton_columns[0]: "Canton"})
    return table


def source_table(raw_2022: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row": raw_2022.index + 2,
            "operator_original": raw_2022["Empresa"],
            "plant_original": raw_2022["Central"],
            "province_original": raw_2022["Provincia"],
            "system_original": raw_2022["Sistema"],
            "technology_original": raw_2022["Tipo de Central"],
            "subtechnology_original": raw_2022["Subtipo de Central"],
            "nominal_capacity_mw_original": raw_2022["Potencia Nominal (MW)"],
            "effective_capacity_mw_original": raw_2022["Potencia Efectiva (MW)"],
            "source_document_id": "PME_2022_CH4",
            "source_locator": [
                f"Chapter 4, section 4.11.1, Annex 1; workbook row {row}"
                for row in raw_2022.index + 2
            ],
            "source_workbook": "data/raw/generation/generation_existing_2022.xlsx",
            "transcription_status": "owner_transcribed_from_official_pdf",
        }
    )


def _key_frame(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["plant_key"] = result["Central"].map(comparison_key)
    result["province_key"] = result["Provincia"].map(comparison_key)
    return result


def location_table(raw_2022: pd.DataFrame, raw_2017: pd.DataFrame) -> pd.DataFrame:
    current = _key_frame(raw_2022)
    historical = _key_frame(raw_2017)
    current["source_row"] = current.index + 2
    current = current[current["Sistema"].map(comparison_key) == "s_n_i"].copy()

    exact = historical.dropna(subset=["Canton"]).copy()
    exact_groups = exact.groupby(["plant_key", "province_key"])["Canton"].nunique()
    ambiguous = exact_groups[exact_groups > 1]
    if not ambiguous.empty:
        raise ValueError(f"ambiguous exact 2017 canton matches: {ambiguous.index.tolist()}")
    exact = exact.drop_duplicates(["plant_key", "province_key"], keep="first")
    current = current.merge(
        exact[["plant_key", "province_key", "Canton"]],
        on=["plant_key", "province_key"],
        how="left",
        validate="many_to_one",
    )
    current["location_method"] = current["Canton"].notna().map(
        {True: "exact_2017", False: "missing"}
    )

    for index, row in current[current["Canton"].isna()].iterrows():
        manual_canton = MANUAL_CANTON.get(ascii_token(row["Central"]))
        if manual_canton:
            current.at[index, "Canton"] = manual_canton
            current.at[index, "location_method"] = "manual_legacy"

    numeric_columns = ["Potencia Nominal (MW)", "Potencia Efectiva (MW)"]
    for table in (current, historical):
        for column in numeric_columns:
            table[column] = pd.to_numeric(table[column], errors="raise")
    historical_candidates = historical.dropna(subset=["Canton"] + numeric_columns)
    for index, row in current[current["Canton"].isna()].iterrows():
        candidates = historical_candidates[
            historical_candidates["province_key"] == row["province_key"]
        ].copy()
        if candidates.empty:
            continue
        nominal_denominator = row["Potencia Nominal (MW)"] or 1.0
        effective_denominator = row["Potencia Efectiva (MW)"] or 1.0
        nominal_error = (
            candidates["Potencia Nominal (MW)"] - row["Potencia Nominal (MW)"]
        ).abs() / nominal_denominator
        effective_error = (
            candidates["Potencia Efectiva (MW)"] - row["Potencia Efectiva (MW)"]
        ).abs() / effective_denominator
        eligible = (nominal_error <= 0.05) & (effective_error <= 0.05)
        if eligible.any():
            best = (nominal_error + effective_error)[eligible].idxmin()
            current.at[index, "Canton"] = historical.at[best, "Canton"]
            current.at[index, "location_method"] = "capacity_proxy_2017"

    missing_canton = current[current["Canton"].isna()]
    if not missing_canton.empty:
        raise ValueError(
            "unresolved SNI plant cantons: " + ", ".join(missing_canton["Central"])
        )

    cantons = clean_columns(pd.read_excel(RAW / "Cantons.xlsx", sheet_name="Canton_Info"))
    cantons["province_key"] = cantons["Provincia"].map(comparison_key)
    cantons["canton_key"] = cantons["Canton"].map(comparison_key)
    coordinate_spread = cantons.groupby(["province_key", "canton_key"]).agg(
        latitude_spread=("Latitud", lambda values: values.max() - values.min()),
        longitude_spread=("Longitud", lambda values: values.max() - values.min()),
    )
    if (coordinate_spread[["latitude_spread", "longitude_spread"]] > 0.01).any().any():
        raise ValueError("duplicate canton coordinates differ by more than 0.01 degrees")
    coordinates = (
        cantons.groupby(["province_key", "canton_key"], as_index=False)
        .agg(latitude=("Latitud", "mean"), longitude=("Longitud", "mean"))
    )
    current["canton_key"] = current["Canton"].map(comparison_key)
    current = current.merge(
        coordinates,
        on=["province_key", "canton_key"],
        how="left",
        validate="many_to_one",
    )
    if current[["latitude", "longitude"]].isna().any().any():
        failed = current.loc[
            current[["latitude", "longitude"]].isna().any(axis=1),
            ["Central", "Provincia", "Canton"],
        ]
        raise ValueError(f"missing canton coordinates:\n{failed.to_string(index=False)}")

    notes = {
        "exact_2017": "Plant and province exact match to the 2017 fleet; canton coordinate lookup deduplicated by key.",
        "manual_legacy": "Legacy notebook manual canton assignment retained as evidence; geographic source requires later source audit.",
        "capacity_proxy_2017": "Legacy same-province capacity proxy within 5 percent retained; low-confidence location evidence.",
    }
    result = pd.DataFrame(
        {
            "source_row": current["source_row"].astype(int),
            "plant_original": current["Central"],
            "province_original": current["Provincia"],
            "canton": current["Canton"].map(ascii_token),
            "latitude": current["latitude"],
            "longitude": current["longitude"],
            "location_method": current["location_method"],
            "location_source": "generation_existing_2017.xlsx and Cantons.xlsx",
            "evidence_note": current["location_method"].map(notes),
        }
    ).sort_values("source_row")
    override_path = RAW / "plant_location_overrides_2022.csv"
    if override_path.is_file():
        overrides = pd.read_csv(override_path).set_index("source_row")
        required = {
            "plant_original", "latitude", "longitude", "location_method",
            "location_source", "evidence_note",
        }
        missing = required.difference(overrides.columns)
        if missing:
            raise ValueError(f"location overrides are missing columns: {sorted(missing)}")
        result = result.set_index("source_row")
        unknown = sorted(set(overrides.index) - set(result.index))
        if unknown:
            raise ValueError(f"location overrides reference unknown source rows: {unknown}")
        for source_row, override in overrides.iterrows():
            if result.at[source_row, "plant_original"] != override.plant_original:
                raise ValueError(f"location override plant mismatch at source row {source_row}")
            for column in required - {"plant_original"}:
                result.at[source_row, column] = override[column]
        result = result.reset_index()
    return result.sort_values("source_row")


def hydro_assignment_table(generators: pd.DataFrame) -> pd.DataFrame:
    legacy_path = (
        ROOT
        / "data"
        / "processed"
        / "generation"
        / "hidro_max_profiles"
        / "aliases_central_to_caudal.csv"
    )
    legacy = pd.read_csv(legacy_path)
    legacy["plant_key"] = legacy["central_id"].map(comparison_key)
    hydro = generators[generators["carrier"] == "hydro"].copy()
    hydro["plant_key"] = hydro["name"].map(comparison_key)
    joined = hydro.merge(
        legacy[["plant_key", "caudal_id"]],
        on="plant_key",
        how="left",
        validate="one_to_one",
    )
    if joined["caudal_id"].isna().any():
        raise ValueError(
            "hydro plants without a preserved flow assignment: "
            + ", ".join(joined.loc[joined["caudal_id"].isna(), "name"])
        )

    direct = {"Coca_Codo_Sinclair", "Delsitanisagua", "Mazar", "Agoyan"}
    system = {"Molino", "Sopladora", "Marcel_Laniado", "Baba", "San_Francisco"}

    def classification(name: object) -> tuple[str, str, str]:
        token = ascii_token(name)
        if token in direct:
            return (
                "direct_plant_series",
                "high",
                "Flow-series name directly identifies this plant.",
            )
        if token in system:
            return (
                "same_hydrological_system",
                "medium",
                "Legacy assignment uses a flow series from the same named river or generating complex.",
            )
        return (
            "legacy_regional_proxy",
            "low",
            "Legacy manual regional proxy retained to preserve prior work; not an observed plant-specific inflow.",
        )

    classes = joined["name"].map(classification)
    joined["assignment_basis"] = classes.map(lambda item: item[0])
    joined["confidence"] = classes.map(lambda item: item[1])
    joined["evidence_note"] = classes.map(lambda item: item[2])
    return joined[
        [
            "plant_id",
            "source_row",
            "name",
            "caudal_id",
            "assignment_basis",
            "confidence",
            "evidence_note",
        ]
    ].rename(columns={"name": "plant_name"}).sort_values("source_row")


def main() -> None:
    raw_2022 = clean_columns(pd.read_excel(RAW / "generation_existing_2022.xlsx"))
    raw_2017 = clean_columns(pd.read_excel(RAW / "generation_existing_2017.xlsx"))
    source = source_table(raw_2022)
    locations = location_table(raw_2022, raw_2017)
    INTERIM.mkdir(parents=True, exist_ok=True)
    source.to_csv(RAW / "generation_existing_2022.csv", index=False, float_format="%.10g")
    locations.to_csv(
        INTERIM / "plant_location_evidence_2022.csv", index=False, float_format="%.10g"
    )
    generators = prepare_existing_generation(
        load_generation_source(RAW / "generation_existing_2022.csv"),
        load_location_evidence(INTERIM / "plant_location_evidence_2022.csv"),
    ).generators
    assignments = hydro_assignment_table(generators)
    assignments.to_csv(
        INTERIM / "hydro_profile_assignments_2022.csv", index=False
    )
    print(
        f"Wrote {len(source)} source rows, {len(locations)} SNI location rows, "
        f"and {len(assignments)} hydro assignments."
    )


if __name__ == "__main__":
    main()
