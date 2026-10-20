"""Evidence-preserving Master Plan tables and conservative year selection.

This interface audits planning records; it never adds assets to a PyPSA network.
Only verified, source-cited official rows can pass the year-selection gate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Mapping

import pandas as pd
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "config/schemas/planning_project.schema.json"
TABLE_TYPES = {
    "generation": "generator", "lines": "line",
    "transformers": "transformer", "substations": "substation",
}
ASSET_UNITS = {
    "generator": {"MW", "kW"}, "line": {"km", "m"},
    "transformer": {"MVA", "kVA"}, "substation": {"count", "MVA", "kVA"},
}
ISSUE_COLUMNS = ["project_id", "severity", "code", "detail"]


class PlanningValidationError(ValueError):
    """The portfolio cannot safely define an official expansion selection."""

    def __init__(self, message: str, issues: pd.DataFrame | None = None):
        super().__init__(message)
        self.issues = issues


def _present(value):
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _number(value):
    try:
        return float(value) if _present(value) else None
    except (ValueError, TypeError):
        return None


def _dependencies(row):
    references = {part.strip() for part in str(row.get("dependencies") or "").split(";") if part.strip()}
    fields = ("substation_id",) if row.get("asset_type") == "generator" else ("endpoint0_id", "endpoint1_id")
    references.update(str(row[field]) for field in fields if _present(row.get(field)))
    return references


def _record(row):
    result = {}
    for key, value in row.items():
        if value is None or pd.isna(value):
            result[key] = None
        elif hasattr(value, "item"):
            result[key] = value.item()
        else:
            result[key] = value
    # CSV integer columns with empty cells arrive as floats. Preserve integer
    # meaning only for the two explicitly integral schema fields.
    for key in ("commissioning_year", "source_row"):
        value = result.get(key)
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            result[key] = int(value)
    return result


def load_portfolio(directory=None, filenames: Mapping[str, str] | None = None):
    """Read four CSVs, retaining raw strings and explicit nulls.

    Run the common artifact preflight before calling this scientific interface.
    Files default to generation.csv, lines.csv, transformers.csv, substations.csv.
    Missing tables fail together; no fallback to notebook-directory CSVs occurs.
    """
    directory = Path(directory) if directory is not None else ROOT / "data/processed/planning"
    names = {key: f"{key}.csv" for key in TABLE_TYPES}
    if filenames:
        if set(filenames) != set(TABLE_TYPES):
            raise PlanningValidationError("filenames must identify all four planning tables")
        names.update(filenames)
    missing = [str(directory / names[key]) for key in TABLE_TYPES if not (directory / names[key]).is_file()]
    if missing:
        raise PlanningValidationError("Missing planning tables: " + "; ".join(missing))
    tables = {}
    for key in TABLE_TYPES:
        frame = pd.read_csv(directory / names[key], dtype=str, keep_default_na=False)
        frame = frame.replace("", None)
        for column in ("commissioning_year", "source_row", "normalized_value"):
            if column in frame:
                # Invalid strings remain visible to the auditor.
                frame[column] = frame[column].map(
                    lambda value: _number(value) if _number(value) is not None else value
                )
        tables[key] = frame
    return tables


def validate_portfolio(tables, schema_path=DEFAULT_SCHEMA):
    """Return deterministic issues without suppressing unverified legacy rows.

    Payload conventions: geographic assets use latitude/longitude; lines use
    endpoint0_id/endpoint1_id and voltage_kv; transformers use endpoint refs and
    voltage0_kv/voltage1_kv. All endpoint references resolve to substation IDs.
    Generators require substation_id resolving to a commissioned substation.
    Dependencies are semicolon-separated project IDs, commissioned no later
    than the dependent project. Quantity units are retained, never converted.
    """
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    issues, records = [], []

    def issue(project_id, code, detail, severity="error"):
        issues.append({"project_id": project_id, "severity": severity, "code": code, "detail": detail})

    for key in sorted(set(TABLE_TYPES) - set(tables)):
        issue("", "missing_table", f"Required table {key} is absent")
    for key in sorted(set(tables) - set(TABLE_TYPES)):
        issue("", "unknown_table", f"Unknown table {key}")
    for key in TABLE_TYPES:
        if key not in tables:
            continue
        missing_columns = sorted(set(schema.get("required", [])) - set(tables[key].columns))
        if missing_columns:
            issue("", "missing_columns", f"Table {key} lacks columns: {', '.join(missing_columns)}")
        for raw in tables[key].to_dict("records"):
            row = _record(raw)
            pid = str(row.get("project_id") or "")
            records.append(row)
            for error in sorted(validator.iter_errors(row), key=lambda e: (str(list(e.path)), e.message)):
                issue(pid, "schema", f"{'.'.join(map(str, error.path)) or 'row'}: {error.message}")
            if row.get("asset_type") != TABLE_TYPES[key]:
                issue(pid, "asset_type", f"Table {key} requires {TABLE_TYPES[key]}")
    lookup = {}
    for row in records:
        pid = str(row.get("project_id") or "")
        if pid in lookup:
            issue(pid, "duplicate_id", "Project IDs must be unique across all four tables")
        lookup[pid] = row

    for row in records:
        pid = str(row.get("project_id") or "")
        official = row.get("evidence_class") == "official_fact"
        quarantined = row.get("review_status") == "quarantined"
        if not official or row.get("review_status") != "verified" or _present(row.get("blockers")):
            issue(pid, "unverified", "Record is unavailable for official expansion until evidence review clears all blockers", "blocked" if not quarantined else "quarantined")
        year = _number(row.get("commissioning_year"))
        if year is None:
            issue(pid, "unknown_year", "Commissioning year is unresolved", "blocked")
        elif not math.isfinite(year) or year < 2022:
            issue(pid, "baseline_overlap", "Pre-2022 records require baseline reconciliation before expansion selection", "blocked")
        if official:
            for field in ("source_document", "source_edition", "source_chapter", "source_page_or_table", "original_name", "original_value", "original_unit", "normalized_unit", "source_artifact", "transformation"):
                if not _present(row.get(field)):
                    issue(pid, "missing_evidence", f"Official row requires {field}")
            if not re.fullmatch(r"[0-9a-fA-F]{64}", str(row.get("source_sha256") or "")):
                issue(pid, "missing_evidence", "Official source_sha256 must be a SHA-256 hex digest")
            source_row = _number(row.get("source_row"))
            if source_row is None or not math.isfinite(source_row) or source_row < 1 or not source_row.is_integer():
                issue(pid, "missing_evidence", "Official source_row must be a positive 1-based integer")
            names = " ".join(str(row.get(field) or "") for field in ("official_name", "original_name", "carrier", "technology"))
            if re.search(r"nucle|\bnpp\b", names, re.IGNORECASE):
                issue(pid, "nuclear_official", "Nuclear assets cannot be part of the official Master Plan portfolio")
        quantity = _number(row.get("normalized_value"))
        if _present(row.get("normalized_value")) or official:
            if quantity is None or not math.isfinite(quantity) or quantity < 0:
                issue(pid, "invalid_quantity", "Normalized quantity must be finite and nonnegative")
            if not _present(row.get("normalized_unit")):
                issue(pid, "missing_unit", "Normalized quantities require an explicit unit")
        if _present(row.get("normalized_unit")) and row.get("normalized_unit") not in ASSET_UNITS.get(row.get("asset_type"), set()):
            issue(pid, "invalid_unit", f"Unit {row.get('normalized_unit')} is unsupported for {row.get('asset_type')}")
        for dependency in str(row.get("dependencies") or "").split(";"):
            dependency = dependency.strip()
            if not dependency:
                continue
            other = lookup.get(dependency)
            if other is None:
                issue(pid, "missing_dependency", f"Dependency {dependency} is absent")
                continue
            other_year = _number(other.get("commissioning_year"))
            if dependency == pid or year is None or other_year is None or other_year > year:
                issue(pid, "dependency_chronology", f"Dependency {dependency} must be available by commissioning year")
        kind = row.get("asset_type")
        if kind == "generator":
            connection = row.get("substation_id")
            other = lookup.get(connection)
            if other is None or other.get("asset_type") != "substation":
                issue(pid, "missing_connection", "Generator substation_id must identify a substation")
            else:
                other_year = _number(other.get("commissioning_year"))
                if year is None or other_year is None or other_year > year:
                    issue(pid, "dependency_chronology", f"Connection {connection} must exist by commissioning year")
        if kind in ("substation", "generator"):
            for field, lower, upper in (("latitude", -90, 90), ("longitude", -180, 180)):
                coordinate = _number(row.get(field))
                if coordinate is None or not math.isfinite(coordinate) or not lower <= coordinate <= upper:
                    issue(pid, "coordinates", f"{field} must be finite and within [{lower}, {upper}]")
        if kind in ("line", "transformer"):
            endpoints = [row.get("endpoint0_id"), row.get("endpoint1_id")]
            if endpoints[0] and endpoints[0] == endpoints[1]:
                issue(pid, "self_loop", "Branch endpoints must differ")
            for side, endpoint in enumerate(endpoints):
                other = lookup.get(endpoint)
                if other is None or other.get("asset_type") != "substation":
                    issue(pid, "missing_endpoint", f"Endpoint {side} must identify a substation")
                    continue
                other_year = _number(other.get("commissioning_year"))
                if year is None or other_year is None or other_year > year:
                    issue(pid, "dependency_chronology", f"Endpoint {endpoint} must exist by commissioning year")
                field = "voltage_kv" if kind == "line" else f"voltage{side}_kv"
                voltage, endpoint_voltage = _number(row.get(field)), _number(other.get("voltage_kv"))
                if voltage is None or endpoint_voltage is None or not math.isfinite(voltage) or voltage <= 0 or not math.isclose(voltage, endpoint_voltage, rel_tol=1e-9):
                    issue(pid, "voltage", f"{field} must match the positive voltage_kv at endpoint {endpoint}")
        elif kind == "substation":
            voltage = _number(row.get("voltage_kv"))
            if voltage is None or not math.isfinite(voltage) or voltage <= 0:
                issue(pid, "voltage", "Substation voltage_kv must be finite and positive")
    # Same-year circular dependencies cannot establish an executable sequence.
    visiting, visited = set(), set()

    def visit(pid):
        if pid in visiting:
            issue(pid, "dependency_cycle", "Dependencies contain a cycle")
            return
        if pid in visited:
            return
        visiting.add(pid)
        for dependency in sorted(_dependencies(lookup[pid])):
            if dependency in lookup:
                visit(dependency)
        visiting.remove(pid)
        visited.add(pid)

    for pid in sorted(lookup):
        visit(pid)
    return pd.DataFrame(issues, columns=ISSUE_COLUMNS).sort_values(
        ["project_id", "severity", "code", "detail"], kind="stable"
    ).reset_index(drop=True)


def filter_official_by_year(tables, year, schema_path=DEFAULT_SCHEMA):
    """Select commissioned official facts; unresolved eligible evidence fails closed.

    Unknown commissioning years cannot safely be placed outside the requested
    horizon and therefore block selection unless explicitly quarantined.
    """
    if isinstance(year, bool) or not isinstance(year, int) or year < 2022:
        raise ValueError("year must be an integer >= 2022")
    issues = validate_portfolio(tables, schema_path)
    candidates, selected = set(), {}
    for key in TABLE_TYPES:
        if key not in tables:
            continue
        frame = tables[key]
        selected_rows = []
        for raw in frame.to_dict("records"):
            row = _record(raw)
            if row.get("review_status") == "quarantined":
                continue
            commissioning = _number(row.get("commissioning_year"))
            if commissioning is None or not math.isfinite(commissioning) or commissioning <= year:
                candidates.add(str(row.get("project_id") or ""))
                selected_rows.append(raw)
        result = pd.DataFrame(selected_rows, columns=frame.columns)
        selected[key] = result.sort_values("project_id", kind="stable").reset_index(drop=True) if "project_id" in result else result
    relevant = issues[issues.project_id.isin(candidates) | (issues.project_id == "")]
    if not relevant.empty:
        raise PlanningValidationError("Official portfolio selection is blocked; inspect the attached issues", relevant)
    available = {str(row["project_id"]) for frame in selected.values() for row in frame.to_dict("records")}
    for frame in selected.values():
        for row in frame.to_dict("records"):
            dependencies = _dependencies(row)
            if not dependencies <= available:
                raise PlanningValidationError(f"Unavailable dependencies for {row['project_id']}: {sorted(dependencies - available)}")
    return selected


def aggregate_by_year(tables, year, schema_path=DEFAULT_SCHEMA):
    """Return counts and quantities grouped by asset type AND original unit."""
    selected = filter_official_by_year(tables, year, schema_path)
    combined = pd.concat(selected.values(), ignore_index=True)
    columns = ["asset_type", "normalized_unit", "normalized_value", "project_count"]
    if combined.empty:
        return pd.DataFrame(columns=columns)
    combined["normalized_value"] = pd.to_numeric(combined.normalized_value, errors="raise")
    return combined.groupby(["asset_type", "normalized_unit"], sort=True, as_index=False).agg(
        normalized_value=("normalized_value", "sum"), project_count=("project_id", "size")
    )[columns]


# Published evidence and exercise readiness are separate questions. These codes
# concern physical interpretation or the existing baseline, never source facts.
PHYSICAL_CODES = {
    "baseline_overlap", "coordinates", "missing_connection", "missing_endpoint",
    "self_loop", "voltage", "dependency_chronology", "missing_dependency",
    "dependency_cycle",
}


def validate_published_portfolio(tables, schema_path=DEFAULT_SCHEMA):
    """Audit source evidence without requiring a network-ready interpretation.

    Schema, provenance, IDs, official status, dates, quantity units, and the
    nuclear exclusion still apply. Coordinates, connections, dependency order,
    and overlap with the 2022 baseline belong to the exercise readiness audit.
    """
    issues = validate_portfolio(tables, schema_path)
    unquantified = {
        str(row.get("project_id") or "")
        for frame in tables.values() for row in frame.to_dict("records")
        if row.get("quantity_status") in ("not_specified", "not_applicable", "not_normalized")
        and not _present(row.get("normalized_value"))
    }
    absent_quantity = issues.project_id.isin(unquantified) & (
        issues.code.isin({"invalid_quantity", "missing_unit"})
        | ((issues.code == "missing_evidence") & issues.detail.isin({
            "Official row requires original_value", "Official row requires original_unit",
            "Official row requires normalized_unit",
        }))
    )
    return issues.loc[~issues.code.isin(PHYSICAL_CODES) & ~absent_quantity].reset_index(drop=True)


def _validate_delay(delay_years):
    if isinstance(delay_years, bool) or not isinstance(delay_years, int) or delay_years < 0:
        raise ValueError("delay_years must be a nonnegative integer")


def _validate_horizon(year):
    if isinstance(year, bool) or not isinstance(year, int) or year < 1900:
        raise ValueError("year must be an integer >= 1900")


def scheduled_portfolio(tables, delay_years=7, schema_path=DEFAULT_SCHEMA):
    """Copy official source rows and add published_year and exercise_year.

    The delay is a teaching assumption. Original commissioning_year,
    official_timing/raw_timing and all source quantities remain unchanged.
    Quarantined rows, legacy candidates, and modeling assumptions are excluded.
    Unknown official dates fail before scheduling, even outside a query horizon.
    """
    _validate_delay(delay_years)
    official, project_ids = {}, set()
    for key in TABLE_TYPES:
        if key not in tables:
            continue
        frame = tables[key]
        if not {"evidence_class", "review_status"} <= set(frame.columns):
            official[key] = frame.iloc[:0].copy()
            continue
        mask = (frame.evidence_class == "official_fact") & (frame.review_status != "quarantined")
        selected = frame.loc[mask].copy(deep=True)
        official[key] = selected
        if "project_id" in selected:
            project_ids.update(selected.project_id.astype(str))
    issues = validate_published_portfolio(tables, schema_path)
    relevant = issues[issues.project_id.isin(project_ids) | (issues.project_id == "")]
    if not relevant.empty:
        raise PlanningValidationError("Published evidence is unresolved; inspect the attached issues", relevant)
    for key, frame in official.items():
        # Evidence validation establishes integer source years before conversion.
        frame["published_year"] = pd.to_numeric(frame.commissioning_year).astype("int64")
        frame["exercise_year"] = frame.published_year + delay_years
        official[key] = frame.sort_values("project_id", kind="stable").reset_index(drop=True)
    return official


def filter_published_by_year(tables, year, delay_years=0, schema_path=DEFAULT_SCHEMA):
    """Query inclusive source dates, optionally shifted by an explicit delay.

    Default delay zero queries the published schedule. Returned copies carry
    both schedule columns and retain the unmodified source commissioning year.
    """
    _validate_horizon(year)
    scheduled = scheduled_portfolio(tables, delay_years, schema_path)
    return {key: frame.loc[frame.exercise_year <= year].copy().reset_index(drop=True)
            for key, frame in scheduled.items()}


def _overlap_flag(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def select_exercise_by_year(
    tables, year, delay_years=7, require_model_ready=True, repeat_mode=False,
    schema_path=DEFAULT_SCHEMA,
):
    """Select the teaching schedule and fail on unresolved physical inputs.

    Published dates do not establish whether an asset already exists in the
    baseline network. An explicit baseline_overlap flag is required for source
    dates through 2022; true flags require repeat_mode. Repeated assets receive
    scenario_asset_id = EX_REPEAT_<project_id>; their source IDs stay intact.
    Readiness flags may be disabled to inspect an incomplete exercise selection.
    """
    if not isinstance(require_model_ready, bool) or not isinstance(repeat_mode, bool):
        raise ValueError("require_model_ready and repeat_mode must be booleans")
    selected = filter_published_by_year(tables, year, delay_years, schema_path)
    selected_ids = {str(pid) for frame in selected.values() for pid in frame.project_id}
    issues = []
    for key, frame in selected.items():
        flags, scenario_ids = [], []
        for row in frame.to_dict("records"):
            pid = str(row["project_id"])
            flag = _overlap_flag(row.get("baseline_overlap"))
            if row["published_year"] <= 2022 and flag is None:
                issues.append(dict(project_id=pid, severity="blocked", code="baseline_overlap_review", detail="Source date is within the baseline period; baseline_overlap requires explicit review"))
            if _present(row.get("baseline_overlap")) and flag is None:
                issues.append(dict(project_id=pid, severity="blocked", code="baseline_overlap_review", detail="baseline_overlap must be true or false"))
            if flag is True and not repeat_mode:
                issues.append(dict(project_id=pid, severity="blocked", code="repeat_mode_required", detail="Baseline-overlapping assets require explicit repeat_mode"))
            flags.append(flag)
            scenario_ids.append(("EX_REPEAT_" if repeat_mode else "EX_") + pid)
        frame["baseline_overlap"] = flags
        frame["scenario_asset_id"] = scenario_ids
    if require_model_ready:
        physical = validate_portfolio(tables, schema_path)
        physical = physical.loc[(physical.code != "baseline_overlap") & physical.project_id.isin(selected_ids)]
        issues.extend(physical.to_dict("records"))
        for frame in selected.values():
            for row in frame.to_dict("records"):
                missing = _dependencies(row) - selected_ids
                if missing:
                    issues.append(dict(project_id=str(row["project_id"]), severity="blocked", code="unavailable_dependency", detail=f"Exercise dependencies unavailable in selected horizon: {sorted(missing)}"))
    if issues:
        issue_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS).sort_values(ISSUE_COLUMNS, kind="stable").reset_index(drop=True)
        raise PlanningValidationError("Exercise selection is blocked; inspect the attached issues", issue_frame)
    return selected
