"""Source-traceable adjustment portfolio and separate teaching assumptions.

No network is mutated. Source tables retain all annex rows, including works
outside the four model asset types. Missing physical details remain explicit.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import math
from pathlib import Path

import pandas as pd
import yaml
from jsonschema import Draft202012Validator

from .artifacts import ROOT, run_preflight
from .environment import check_environment, format_environment_report
from .planning import TABLE_TYPES, ASSET_UNITS, DEFAULT_SCHEMA, PlanningValidationError, validate_published_portfolio

INVENTORY_TYPES = [*TABLE_TYPES.values(), "compound_grid_system", "reactive", "mobile_equipment", "plant_modernization", "storage"]


def query_inventory(projects, year, delay_years=0):
    """Query ALL published rows, without claiming electrical model readiness."""
    if isinstance(year, bool) or not isinstance(year, int) or year < 1900:
        raise ValueError("year must be an integer >= 1900")
    if isinstance(delay_years, bool) or not isinstance(delay_years, int) or delay_years < 0:
        raise ValueError("delay_years must be a nonnegative integer")
    schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
    schema["properties"]["asset_type"]["enum"] = INVENTORY_TYPES
    validator = Draft202012Validator(schema)
    if projects.project_id.duplicated().any():
        raise PlanningValidationError("Duplicate project IDs in full source inventory")
    for raw in projects.to_dict("records"):
        row = {key: (None if pd.isna(value) else value) for key, value in raw.items()}
        errors = list(validator.iter_errors(row))
        if errors or row["commissioning_year"] is None:
            raise PlanningValidationError(f"Unresolved source record {row.get('project_id')}: " + "; ".join(error.message for error in errors))
        if row.get("evidence_class") != "official_fact" or row.get("review_status") != "verified":
            raise PlanningValidationError("The full published inventory requires verified source facts")
        if "nuclear" in str(row["official_name"]).lower():
            raise PlanningValidationError("Nuclear assumptions cannot enter the official inventory")
        quantity = row.get("normalized_value")
        if quantity is not None and (not isinstance(quantity, (float, int)) or not math.isfinite(quantity)):
            raise PlanningValidationError("Reported quantities must be finite numbers")
        if quantity is not None and row["asset_type"] in ASSET_UNITS and row.get("normalized_unit") not in ASSET_UNITS[row["asset_type"]]:
            raise PlanningValidationError("Reported units do not match the asset type")
    result = projects.copy(deep=True)
    result["published_year"] = result.commissioning_year.astype(int)
    result["exercise_year"] = result.published_year + delay_years
    return result.loc[result.exercise_year <= year].sort_values("project_id").reset_index(drop=True)


def token(text):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower())


def decimal_es(text):
    """Parse the source's comma-decimal quantities without altering raw text."""
    return float(str(text).replace(".", "").replace(",", "."))


def load_config(root=ROOT):
    config = yaml.safe_load((root / "config/planning.yaml").read_text(encoding="utf-8"))
    delay = config["exercise"]["delay_years"]
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
        raise ValueError("exercise.delay_years must be a nonnegative integer")
    return config


def grid_kind(name):
    """Classify only explicitly described equipment; compound systems stay whole."""
    original_name = name
    name = token(name)
    if "almacenamiento" in name:
        return "storage"
    if "capacitor" in name or re.search(r"\bMVAR\b", str(original_name), re.I):
        return "reactive"
    if "movil" in name or "moviles" in name or "adquisicion" in name:
        return "mobile_equipment"
    if "innovacion" in name:
        return "plant_modernization"
    if "transformador" in name:
        return "transformer"
    if name.startswith("linea") or name.startswith("lt"):
        return "line"
    if name.startswith("subestacion") or name.startswith("ampliacionsubestacion") or name.startswith("modernizacion"):
        return "substation"
    return "compound_grid_system"


def _common(raw, config, source, source_hash, kind, value=None, unit=None, original=None):
    page, number = int(raw["source_page"]), int(str(raw["source_row"]).split("-")[-1])
    pid = f"PME26_{raw['annex']}_{number:03d}"
    timing = raw.get("official_timing") or str(raw["published_year"])
    years = re.findall(r"\b20\d{2}\b", str(timing))
    year = int(years[0]) if len(years) == 1 else None
    return dict(
        project_id=pid, asset_type=kind, official_name=raw["official_name"],
        commissioning_year=year, official_timing=timing, evidence_class="official_fact",
        source_document=config["source_document"], source_edition=config["source_edition"],
        source_chapter=f"Annex {raw['annex']}",
        source_page_or_table=f"PDF page {page}, Annex {raw['annex']}, data row {number}",
        original_name=raw["official_name"], original_value=original,
        original_unit=unit, normalized_value=value, normalized_unit=unit,
        quantity_status="reported" if value is not None else "not_specified",
        transformation="Source-faithful transcription; comma-decimal normalization only; absent quantities remain null",
        confidence="high", source_artifact=source, source_sha256=source_hash,
        source_row=number, review_status="verified", blockers=None, dependencies=None,
        source_pdf_sha256=config["_source_pdf_sha256"],
        original_payload_json=json.dumps(raw, ensure_ascii=False, sort_keys=True),
        financing=raw.get("financing"), source_status=config["source_status"],
        latitude=None, longitude=None, voltage_kv=None,
        endpoint0_id=None, endpoint1_id=None, voltage0_kv=None, voltage1_kv=None,
        substation_id=None,
    )


def _coordinates(root, config):
    """Exact names and unique canton points; no province/nearest-point fallback."""
    names, cantons = {}, {}
    for relative, name_col, lat_col, lon_col in [
        ("data/processed/generation/powerplants_future.csv", "Name", "lat", "lon"),
        ("data/raw/networks/EC_bus_expansion_masterplan_2022.csv", "Name", "lat", "lon"),
    ]:
        frame = pd.read_csv(root / relative, keep_default_na=False)
        for row in frame.to_dict("records"):
            if "nuclear" in str(row[name_col]).lower():
                continue
            try:
                lat, lon = float(row[lat_col]), float(row[lon_col])
            except (ValueError, TypeError):
                continue
            key = token(row[name_col])
            if name_col == "Name" and "Bus" in row:
                key = re.sub(r"(?:500|230|138|69)$", "", key)
            names.setdefault(key, []).append((lat, lon, relative, str(row[name_col])))
    relative = "data/interim/generation/plant_location_evidence_2022.csv"
    for row in pd.read_csv(root / relative).to_dict("records"):
        cantons.setdefault((token(row["province_original"]), token(row["canton"])), []).append((float(row["latitude"]), float(row["longitude"]), relative, str(row["canton"])))
    return names, cantons


def location_for(name, canton, names, cantons, aliases, province=None, methods=None):
    name_key = token(name)
    if re.search(r"subestaci[oó]n", str(name), re.I):
        site = re.split(r"subestaci[oó]n\s*", str(name), flags=re.I)[-1]
        name_key = token(re.split(r"\d+\s*(?:/|kV)|,", site, flags=re.I)[0])
    methods = methods if methods is not None else ["same_name_legacy", "unique_canton"]
    for method, key, lookup in [("same_name_legacy", aliases.get(name_key, name_key), names),
                                ("unique_canton", (token(province or ""), token(canton or "")), cantons)]:
        if method not in methods:
            continue
        points = lookup.get(key, [])
        distinct = {(point[0], point[1]) for point in points}
        if len(distinct) == 1:
            point = points[0]
            return dict(latitude=point[0], longitude=point[1], coordinate_method=method,
                        coordinate_source=point[2], coordinate_source_key=point[3],
                        coordinate_confidence="low", coordinate_evidence_class="modeling_assumption")
    return dict(latitude=None, longitude=None, coordinate_method="unresolved",
                coordinate_source=None, coordinate_source_key=None,
                coordinate_confidence="unresolved", coordinate_evidence_class="modeling_assumption")


def build_products(root=ROOT, *, delay_years=None):
    """Build published interfaces and a complete delayed exercise inventory."""
    root = Path(root)
    environment = check_environment()
    if not environment.ok:
        raise RuntimeError(format_environment_report(environment))
    preflight = run_preflight("07_planning", root=root)
    preflight.raise_for_errors()
    config = load_config(root)
    if delay_years is not None:
        if isinstance(delay_years, bool) or not isinstance(delay_years, int) or delay_years < 0:
            raise ValueError("delay_years must be a nonnegative integer")
        config["exercise"]["delay_years"] = delay_years
    config["_source_pdf_sha256"] = hashlib.sha256((root / config["source_document"]).read_bytes()).hexdigest()
    records = []
    source = "data/raw/planning/adjustment_2026_generation.csv"
    digest = hashlib.sha256((root / source).read_bytes()).hexdigest()
    for raw in pd.read_csv(root / source, dtype=str, keep_default_na=False).to_dict("records"):
        row = _common(raw, config, source, digest, "generator", decimal_es(raw["power_mw_original"]), "MW", raw["power_mw_original"])
        row.update(carrier=raw["technology"], province=raw["province"], canton=raw["canton"],
                   energy_gwh_year=decimal_es(raw["energy_gwh_year_original"]), operator=raw["operator"],
                   action="replacement" if "reposicion" in token(raw["official_name"]) else "planned_generation")
        records.append(row)
    source = "data/raw/planning/adjustment_2026_grid_projects.csv"
    digest = hashlib.sha256((root / source).read_bytes()).hexdigest()
    for raw in pd.read_csv(root / source, dtype=str, keep_default_na=False).to_dict("records"):
        name = raw["official_name"]
        kind = grid_kind(name)
        value, unit, original = None, None, None
        if kind == "line":
            match = re.search(r"(\d+(?:[.,]\d+)?)\s*km", name, re.I)
            if match:
                original = match[1]
                value, unit = float(original.replace(",", ".")), "km"
        if kind in {"transformer", "substation"} and not re.search(r"\d\s*[xX]\s*\d", name):
            match = re.search(r"(?<![\dxX])(\d+(?:[.,]\d+)?)\s*MVA\b", name, re.I)
            if match:
                original = match[1]
                value, unit = float(original.replace(",", ".")), "MVA"
        row = _common(raw, config, source, digest, kind, value, unit, original)
        if value is None and re.search(r"\d.*?\b(?:MVA|MVAR|MW|km|horas)\b", name, re.I):
            row["quantity_status"] = "not_normalized"
            row["original_value"] = name
        voltage = re.search(r"(\d+(?:\s*/\s*\d+)*)\s*kV\b", name, re.I)
        row["voltage_levels_original"] = voltage[1] if voltage else None
        row["action"] = next((action for word, action in [
            ("repotenciacion", "uprating"), ("reposicion", "replacement"),
            ("reemplazo", "replacement"), ("ampliacion", "extension"),
            ("modernizacion", "modernization")
        ] if word in token(name)), "planned_grid_work")
        records.append(row)
    projects = pd.DataFrame(records).sort_values("project_id").reset_index(drop=True)
    if len(projects) != 123 or projects.project_id.nunique() != 123:
        raise ValueError("Adjustment inventory must reconcile exactly to 50 generation + 73 grid source rows")
    query_inventory(projects, 2200)
    tables = {name: projects[projects.asset_type == kind].copy() for name, kind in TABLE_TYPES.items()}
    evidence_issues = validate_published_portfolio(tables)
    if not evidence_issues.empty:
        raise ValueError("Published evidence validation failed:\n" + evidence_issues.to_string(index=False))
    destination = root / "data/processed/planning"
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False, lineterminator="\n")
    projects.to_csv(destination / "projects.csv", index=False, lineterminator="\n")
    names, cantons = _coordinates(root, config)
    aliases = {token(k): token(v) for k, v in config["exercise"].get("name_aliases", {}).items()}
    existing = pd.read_csv(root / "data/processed/generation/generators_2022.csv")
    existing_names = {token(row["name"]): row["plant_id"] for row in existing.to_dict("records")}
    overlays = []
    delay = config["exercise"]["delay_years"]
    for row in projects.to_dict("records"):
        coords = location_for(row["official_name"], row.get("canton"), names, cantons, aliases,
                              row.get("province"), config["exercise"]["coordinate_methods"])
        baseline_id = existing_names.get(aliases.get(token(row["official_name"]), token(row["official_name"])))
        overlap = True if baseline_id else None
        issues = []
        if coords["latitude"] is None:
            issues.append("location_unresolved")
        if row["asset_type"] == "generator":
            issues.append("electrical_connection_not_defined")
        elif row["asset_type"] in {"line", "transformer", "compound_grid_system"}:
            issues.append("electrical_endpoints_and_parameters_not_defined")
        else:
            issues.append("equipment_model_not_defined")
        overlays.append(dict(project_id=row["project_id"],
            scenario_asset_id=("EX_REPEAT_" if config["exercise"]["repeat_mode"] else "EX_") + row["project_id"],
            official_name=row["official_name"], asset_type=row["asset_type"],
            published_year=row["commissioning_year"], published_timing=row["official_timing"],
            delay_years=delay, exercise_year=row["commissioning_year"] + delay,
            exercise_timing=re.sub(str(row["commissioning_year"]), str(row["commissioning_year"] + delay), str(row["official_timing"])),
            baseline_overlap=overlap, overlap_status="known_same_name" if overlap else "not_established",
            baseline_source="data/processed/generation/generators_2022.csv" if baseline_id else None,
            baseline_asset_id=baseline_id,
            repeat_mode=config["exercise"]["repeat_mode"], evidence_class="modeling_assumption",
            assumption_source="Owner instruction 2026-10-20 and config/planning.yaml",
            model_ready=False, model_blockers=";".join(issues), **coords))
    exercise = pd.DataFrame(overlays)
    exercise.to_csv(destination / "exercise_schedule.csv", index=False, lineterminator="\n")
    physical_issues = []
    for row in overlays:
        for issue in row["model_blockers"].split(";"):
            physical_issues.append(dict(project_id=row["project_id"], code=issue, severity="unresolved"))
    outputs = root / "results/phase_07"
    outputs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(physical_issues).to_csv(outputs / "exercise_readiness_issues.csv", index=False)
    pd.DataFrame([
        {"project_id": "PME26_B_039", "related_project_id": "PME26_B_045", "relationship": "potential_overlap", "resolution": "unresolved", "reason": "Ibarra-Pimampiro descriptions share corridor/year; one reports 25.88 km"},
        {"project_id": "PME26_B_052", "related_project_id": "PME26_B_055", "relationship": "potential_overlap", "resolution": "unresolved", "reason": "Shushufindi-Orellana descriptions may overlap; one includes Loreto sectionalization"},
        {"project_id": "PME26_B_026", "related_project_id": "PME26_B_031", "relationship": "potential_overlap", "resolution": "unresolved", "reason": "Portoviejo works report different bank capacities for the same half-year"},
    ]).to_csv(destination / "project_relationships.csv", index=False, lineterminator="\n")
    printed = pd.read_csv(root / "data/raw/planning/adjustment_2026_totals.csv", dtype=str).iloc[-1]
    pd.DataFrame([
        {"comparison": "generation_capacity", "row_sum": float(tables["generation"].normalized_value.sum()), "printed_total": decimal_es(printed.power_original), "unit": "MW"},
        {"comparison": "generation_mean_energy", "row_sum": float(tables["generation"].energy_gwh_year.sum()), "printed_total": decimal_es(printed.energy_original), "unit": "GWh/year"},
    ]).assign(difference=lambda frame: frame.row_sum - frame.printed_total).to_csv(outputs / "source_total_reconciliation.csv", index=False)
    generation_checks = tables["generation"][["project_id", "official_name", "normalized_value", "energy_gwh_year"]].copy()
    generation_checks["nameplate_8760h_gwh"] = generation_checks.normalized_value * 8.76
    generation_checks["implied_capacity_factor"] = generation_checks.energy_gwh_year / generation_checks.nameplate_8760h_gwh
    generation_checks["energy_above_nameplate_8760h"] = generation_checks.implied_capacity_factor > 1 + 1e-9
    generation_checks.to_csv(outputs / "generation_source_checks.csv", index=False)
    fingerprints = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in sorted(destination.glob("*.csv"))]
    manifest = dict(source_pdf=config["source_document"], source_pdf_sha256=hashlib.sha256((root / config["source_document"]).read_bytes()).hexdigest(),
                    source_edition=config["source_edition"], source_scope=config["scope"],
                    source_rows=123, generation_rows=50, grid_rows=73,
                    published_evidence_status="PASS", model_readiness_status="UNRESOLVED",
                    source_numeric_consistency="WARN" if generation_checks.energy_above_nameplate_8760h.any() or abs(float(tables["generation"].energy_gwh_year.sum()) - decimal_es(printed.energy_original)) > 1e-9 else "PASS",
                    energy_above_nameplate_rows=int(generation_checks.energy_above_nameplate_8760h.sum()),
                    exercise_delay_years=delay, repeat_mode=config["exercise"]["repeat_mode"],
                    generator_mw=float(tables["generation"].normalized_value.sum()),
                    input_artifacts=[result.__dict__ for result in preflight.results if result.artifact_id in preflight.required_ids],
                    output_artifacts=fingerprints,
                    writer="src/planning_products.py", writer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (destination / "portfolio_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return projects, tables, exercise, manifest
