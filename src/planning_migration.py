"""Preserve legacy planning evidence without promoting uncited project claims.

Run with ``python -m src.planning_migration`` in pypsa-earth-ec.
This migration performs no PDF retrieval, network mutation, or optimization.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.artifacts import ROOT, run_preflight
from src.environment import check_environment, format_environment_report


SOURCES = (
    ("generation", "data/raw/generation/generation_future_renewable_2022.csv", "Proyecto / Central", "Ano de entrada en operacion", "Power"),
    ("generation", "data/processed/generation/powerplants_future.csv", "Name", "DateIn", "Capacity"),
    ("substations", "data/raw/networks/EC_bus_expansion_masterplan_2022.csv", "Name", "Year", "v_nom"),
    ("lines", "data/raw/networks/EC_line_expansion_masterplan_2022.csv", "project", "Year", "length"),
    ("transformers", "data/raw/networks/EC_transformer_expansion_masterplan_2022.csv", "Transformer", "Year", "s_nom"),
)
ASSET_TYPES = {"generation": "generator", "substations": "substation", "lines": "line", "transformers": "transformer"}


def _number(value: str) -> float | None:
    return float(value) if value else None


def _bus_ref(value: str, known: set[str]) -> str | None:
    if not value:
        return None
    key = str(int(float(value)))
    return f"MP_BUS_{key}" if key in known else f"LEGACY_BUS_{key}"


def migrate(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """Write four candidate tables plus a separate nuclear quarantine.

    Original CSV fields are retained verbatim in original_payload_json. Null
    normalized quantities are intentional: no published units are established.
    source_row counts data records, excluding the header (multiline CSV-safe).
    """
    environment = check_environment()
    if not environment.ok:
        raise RuntimeError(format_environment_report(environment))
    run_preflight("07_planning", root=root).raise_for_errors()
    bus_path = root / SOURCES[2][1]
    with bus_path.open(encoding="utf-8-sig", newline="") as handle:
        known = {row["Bus"] for row in csv.DictReader(handle)}
    tables: dict[str, list[dict]] = {name: [] for name in ASSET_TYPES}
    quarantine = []
    fingerprints = []
    for kind, relative, name_key, year_key, value_key in SOURCES:
        path = root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        fingerprints.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
        with path.open(encoding="utf-8-sig", newline="") as handle:
            original_rows = list(csv.DictReader(handle))
        for row_number, raw in enumerate(original_rows, 1):
            nuclear = "nuclear" in json.dumps(raw, ensure_ascii=False).casefold()
            name = raw[name_key]
            year = int(float(raw[year_key])) if raw[year_key] else None
            key = raw.get("Line") or raw.get("Transformer") or name
            stable = hashlib.sha256((relative + "|" + key).encode()).hexdigest()[:16].upper()
            project_id = f"MP_{ASSET_TYPES[kind].upper()}_{stable}"
            if kind == "substations":
                project_id = f"MP_BUS_{raw['Bus']}"
            blockers = ["source_edition_unresolved", "source_locator_missing", "units_unverified"]
            if year is None:
                blockers.append("commissioning_year_unknown")
            elif year <= 2022:
                blockers.append("baseline_overlap_requires_reconciliation")
            if kind == "generation":
                blockers.extend(["generation_lists_overlap_unreconciled", "connection_unverified"])
            if kind == "transformers":
                blockers.append("legacy_2000_MVA_and_impedance_are_unverified_model_parameters")
            if nuclear:
                blockers = ["hypothetical_nuclear_excluded"]
            row = dict(
                project_id=project_id, asset_type=ASSET_TYPES[kind], official_name=name,
                commissioning_year=year, official_timing=raw[year_key],
                evidence_class="modeling_assumption" if nuclear else "legacy_unverified",
                source_document="Unverified legacy extract; official document not established",
                source_edition=None, source_chapter=None, source_page_or_table=None,
                original_name=name, original_value=raw[value_key] or None, original_unit=None,
                normalized_value=None, normalized_unit=None,
                transformation="Lossless legacy CSV migration; timing and joins are unverified legacy fields; no official units or capacities inferred",
                confidence="unresolved", source_artifact=relative, source_sha256=digest,
                source_row=row_number, review_status="quarantined" if nuclear else "unverified",
                blockers=";".join(blockers), dependencies="",
                original_payload_json=json.dumps(raw, ensure_ascii=False, sort_keys=True),
                carrier=raw.get("Tipo") or raw.get("Fueltype") or raw.get("carrier"),
                voltage_kv=_number(raw.get("v_nom", "")),
                latitude=_number(raw.get("lat", "")), longitude=_number(raw.get("lon", "")),
                endpoint0_id=None, endpoint1_id=None, voltage0_kv=None, voltage1_kv=None,
                legacy_connection=raw.get("Bus") if kind == "generation" else None,
            )
            if kind in {"lines", "transformers"}:
                refs = [_bus_ref(raw.get(key, ""), known) for key in ("bus0", "bus1")]
                row.update(endpoint0_id=refs[0], endpoint1_id=refs[1])
                row["dependencies"] = ";".join(sorted(set(ref for ref in refs if ref)))
                if any(ref is None or ref.startswith("LEGACY_") for ref in refs):
                    row["blockers"] += ";endpoint_mapping_unresolved"
            (quarantine if nuclear else tables[kind]).append(row)
    destination = root / "data/interim/planning/legacy_candidates"
    destination.mkdir(parents=True, exist_ok=True)
    prepared = {}
    for kind, rows in tables.items():
        frame = pd.DataFrame(rows).sort_values("project_id").reset_index(drop=True)
        frame.to_csv(destination / f"{kind}.csv", index=False, lineterminator="\n")
        prepared[kind] = frame
    evidence = root / "data/interim/planning"
    evidence.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(quarantine).sort_values("project_id").to_csv(evidence / "nuclear_quarantine.csv", index=False, lineterminator="\n")
    outputs = []
    for path in [*sorted(destination.glob("*.csv")), evidence / "nuclear_quarantine.csv"]:
        outputs.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (evidence / "migration_manifest.json").write_text(json.dumps({
        "status": "BLOCKED_OFFICIAL_PUBLICATION", "source_artifacts": fingerprints,
        "output_artifacts": outputs, "candidate_rows": {k: len(v) for k, v in prepared.items()},
        "quarantined_rows": len(quarantine), "official_rows": 0,
        "writer": "scripts/migrate_phase07_planning.py",
        "writer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validation_module_sha256": hashlib.sha256((root / "src/planning.py").read_bytes()).hexdigest(),
        "schema_sha256": hashlib.sha256((root / "config/schemas/planning_project.schema.json").read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    return prepared


if __name__ == "__main__":
    result = migrate()
    print({name: len(frame) for name, frame in result.items()})
    print("Evidence migration complete. Official portfolio publication remains blocked.")
