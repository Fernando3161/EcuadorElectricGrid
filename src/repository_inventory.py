"""Load and validate the Phase 1 repository classification contract.

The inventory is intentionally about provenance and repository disposition.  It
does not decide whether a scientific artifact is valid for the 2022 baseline.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "config" / "repository_inventory.json"


def load_inventory(path: Path | None = None) -> dict[str, Any]:
    """Return the repository inventory document."""

    inventory_path = path or DEFAULT_INVENTORY
    with inventory_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def matching_rules(path: str, inventory: dict[str, Any]) -> list[dict[str, Any]]:
    """Return tracked-file rules matching a repository-relative POSIX path."""

    normalized = Path(path).as_posix()
    candidate = PurePosixPath(normalized)
    return [
        rule
        for rule in inventory["tracked_file_rules"]
        if any(candidate.match(pattern) for pattern in rule["patterns"])
    ]


def validate_inventory(
    inventory: dict[str, Any], tracked_paths: Iterable[str]
) -> list[str]:
    """Return validation errors for the inventory and tracked-file coverage."""

    errors: list[str] = []
    required_top_level = {
        "schema_version",
        "classification_terms",
        "tracked_file_rules",
        "ignored_artifact_rules",
        "notebook_register",
        "module_register",
        "source_document_register",
        "external_artifacts",
        "critical_evidence",
    }
    missing = sorted(required_top_level.difference(inventory))
    if missing:
        errors.append(f"missing top-level keys: {', '.join(missing)}")
        return errors

    rule_ids = [rule.get("id") for rule in inventory["tracked_file_rules"]]
    if len(rule_ids) != len(set(rule_ids)):
        errors.append("tracked-file rule IDs must be unique")

    allowed = set(inventory["classification_terms"])
    for rule in inventory["tracked_file_rules"] + inventory["ignored_artifact_rules"]:
        if rule.get("classification") not in allowed:
            errors.append(
                f"rule {rule.get('id', '<missing>')} has unknown classification "
                f"{rule.get('classification')!r}"
            )

    for path in tracked_paths:
        matches = matching_rules(path, inventory)
        if not matches:
            errors.append(f"unclassified tracked path: {Path(path).as_posix()}")
        elif len(matches) > 1:
            ids = ", ".join(rule["id"] for rule in matches)
            errors.append(f"ambiguous tracked path: {path} ({ids})")

    notebook_paths = [row["path"] for row in inventory["notebook_register"]]
    if len(notebook_paths) != len(set(notebook_paths)):
        errors.append("notebook register paths must be unique")

    module_paths = [row["path"] for row in inventory["module_register"]]
    if len(module_paths) != len(set(module_paths)):
        errors.append("module register paths must be unique")

    source_paths = [row["path"] for row in inventory["source_document_register"]]
    if len(source_paths) != len(set(source_paths)):
        errors.append("source-document register paths must be unique")

    artifact_ids = [row["id"] for row in inventory["external_artifacts"]]
    if len(artifact_ids) != len(set(artifact_ids)):
        errors.append("external artifact IDs must be unique")

    return errors


def matching_ignored_rules(
    path: str, inventory: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return ignored-artifact rules matching a relative POSIX path."""

    candidate = PurePosixPath(Path(path).as_posix())
    return [
        rule
        for rule in inventory["ignored_artifact_rules"]
        if any(candidate.match(pattern) for pattern in rule["patterns"])
    ]


__all__ = [
    "DEFAULT_INVENTORY",
    "ROOT",
    "load_inventory",
    "matching_rules",
    "matching_ignored_rules",
    "validate_inventory",
]
