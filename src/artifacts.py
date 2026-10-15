"""Manifest-driven external-artifact preflight and safe retrieval."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import h5netcdf
import yaml
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "config" / "artifacts.yaml"
DEFAULT_SCHEMA = ROOT / "config" / "schemas" / "artifact_manifest.schema.json"
SUCCESS_STATUSES = {"valid", "downloaded", "not-required"}


class ManifestError(ValueError):
    """Raised when the manifest is syntactically or semantically invalid."""


class ArtifactValidationError(ValueError):
    """Raised when an artifact fails integrity or structural validation."""


class PreflightError(RuntimeError):
    """Raised when one or more required artifacts remain unresolved."""

    def __init__(self, report: "PreflightReport") -> None:
        super().__init__(report.failure_summary())
        self.report = report


@dataclass(frozen=True)
class ArtifactResult:
    artifact_id: str
    destination: str
    status: str
    message: str
    bytes: int | None = None
    sha256: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in SUCCESS_STATUSES


@dataclass(frozen=True)
class PreflightReport:
    stage: str
    required_ids: tuple[str, ...]
    results: tuple[ArtifactResult, ...]

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.results)

    def unresolved(self) -> tuple[ArtifactResult, ...]:
        return tuple(result for result in self.results if not result.ok)

    def failure_summary(self) -> str:
        failures = self.unresolved()
        if not failures:
            return "artifact preflight passed"
        detail = "; ".join(
            f"{row.artifact_id}: {row.status} ({row.message})" for row in failures
        )
        return f"artifact preflight failed for stage {self.stage}: {detail}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "required_ids": list(self.required_ids),
            "ok": self.ok,
            "results": [asdict(result) | {"ok": result.ok} for result in self.results],
        }

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise PreflightError(self)


DownloadFunction = Callable[[str, Path, int | None], None]


def load_manifest(
    path: Path | str = DEFAULT_MANIFEST,
    schema_path: Path | str = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Load and validate the external-artifact manifest."""

    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    with Path(schema_path).open(encoding="utf-8") as handle:
        import json

        schema = json.load(handle)

    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        messages = [
            f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: "
            f"{error.message}"
            for error in errors
        ]
        raise ManifestError("manifest schema validation failed:\n" + "\n".join(messages))

    _validate_manifest_semantics(manifest)
    return manifest


def _validate_manifest_semantics(manifest: dict[str, Any]) -> None:
    stages = manifest["stages"]
    artifacts = manifest["artifacts"]
    artifact_ids = [row["id"] for row in artifacts]
    destinations = [Path(row["destination"]).as_posix() for row in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ManifestError("artifact IDs must be unique")
    if len(destinations) != len(set(destinations)):
        raise ManifestError("artifact destinations must be unique")

    for stage, definition in stages.items():
        unknown = sorted(set(definition["depends_on"]).difference(stages))
        if unknown:
            raise ManifestError(f"stage {stage} depends on unknown stages: {unknown}")
    for artifact in artifacts:
        unknown = sorted(set(artifact["stages"]).difference(stages))
        if unknown:
            raise ManifestError(
                f"artifact {artifact['id']} references unknown stages: {unknown}"
            )
        destination = Path(artifact["destination"])
        if destination.is_absolute() or ".." in destination.parts:
            raise ManifestError(
                f"artifact {artifact['id']} has unsafe destination {destination}"
            )

    for stage in stages:
        resolve_stage_dependencies(manifest, stage)


def resolve_stage_dependencies(manifest: dict[str, Any], stage: str) -> tuple[str, ...]:
    """Return a deterministic dependency closure ending with ``stage``."""

    stages = manifest["stages"]
    if stage not in stages:
        choices = ", ".join(stages)
        raise ManifestError(f"unknown stage {stage!r}; choose one of: {choices}")

    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(current: str) -> None:
        if current in visiting:
            raise ManifestError(f"stage dependency cycle detected at {current}")
        if current in visited:
            return
        visiting.add(current)
        for dependency in stages[current]["depends_on"]:
            visit(dependency)
        visiting.remove(current)
        visited.add(current)
        ordered.append(current)

    visit(stage)
    return tuple(ordered)


def required_artifact_ids(manifest: dict[str, Any], stage: str) -> tuple[str, ...]:
    """Return artifact IDs required by a stage and all its dependencies."""

    closure = set(resolve_stage_dependencies(manifest, stage))
    return tuple(
        artifact["id"]
        for artifact in manifest["artifacts"]
        if closure.intersection(artifact["stages"])
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(path: Path, artifact: dict[str, Any]) -> tuple[int, str]:
    """Validate existence, size, SHA-256, and configured structure."""

    if not path.is_file():
        raise ArtifactValidationError("file is missing")
    actual_size = path.stat().st_size
    expected_size = artifact["bytes"]
    if actual_size != expected_size:
        raise ArtifactValidationError(
            f"size mismatch: expected {expected_size}, found {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != artifact["sha256"]:
        raise ArtifactValidationError(
            f"SHA-256 mismatch: expected {artifact['sha256']}, found {actual_sha256}"
        )

    validation = artifact["validation"]
    if validation["kind"] == "netcdf":
        _validate_netcdf(path, validation)
    return actual_size, actual_sha256


def _validate_netcdf(path: Path, validation: dict[str, Any]) -> None:
    try:
        with h5netcdf.File(path, "r") as dataset:
            for dimension, expected in validation.get("dimensions", {}).items():
                if dimension not in dataset.dimensions:
                    raise ArtifactValidationError(f"missing NetCDF dimension {dimension}")
                actual = len(dataset.dimensions[dimension])
                if actual != expected:
                    raise ArtifactValidationError(
                        f"NetCDF dimension {dimension}: expected {expected}, found {actual}"
                    )
            missing_variables = sorted(
                set(validation.get("required_variables", [])).difference(dataset.variables)
            )
            if missing_variables:
                raise ArtifactValidationError(
                    f"missing NetCDF variables: {', '.join(missing_variables)}"
                )
            for attribute, expected in validation.get("attributes", {}).items():
                actual = dataset.attrs.get(attribute)
                if actual != expected:
                    raise ArtifactValidationError(
                        f"NetCDF attribute {attribute}: expected {expected!r}, "
                        f"found {actual!r}"
                    )
    except ArtifactValidationError:
        raise
    except Exception as exc:
        raise ArtifactValidationError(f"cannot inspect NetCDF structure: {exc}") from exc


def http_download(
    url: str,
    partial_path: Path,
    expected_bytes: int | None = None,
    *,
    timeout: tuple[int, int] = (15, 120),
    chunk_size: int = 1024 * 1024,
) -> None:
    """Download to a partial file, resuming with HTTP Range when supported."""

    import requests

    partial_path.parent.mkdir(parents=True, exist_ok=True)
    existing = partial_path.stat().st_size if partial_path.exists() else 0
    if expected_bytes is not None and existing > expected_bytes:
        existing = 0

    headers = {"User-Agent": "EcuadorElectricGrid-preflight/1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    response = requests.get(url, stream=True, timeout=timeout, headers=headers)
    if response.status_code == 416 and existing:
        response.close()
        existing = 0
        headers.pop("Range", None)
        response = requests.get(url, stream=True, timeout=timeout, headers=headers)
    response.raise_for_status()

    append = existing > 0 and response.status_code == 206
    mode = "ab" if append else "wb"
    with response, partial_path.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())


def _destination(root: Path, relative: str) -> Path:
    root = root.resolve()
    destination = (root / relative).resolve()
    if destination != root and root not in destination.parents:
        raise ManifestError(f"artifact destination escapes repository root: {relative}")
    return destination


def _backup_invalid(path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.name}.invalid-{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.invalid-{timestamp}-{suffix}")
        suffix += 1
    return candidate


def _install_valid_partial(partial: Path, destination: Path) -> Path | None:
    backup: Path | None = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        backup = _backup_invalid(destination)
        destination.replace(backup)
    try:
        os.replace(partial, destination)
    except Exception:
        if backup is not None and not destination.exists():
            backup.replace(destination)
        raise
    return backup


def run_preflight(
    stage: str = "00_preflight",
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    root: Path | str = ROOT,
    offline: bool = False,
    downloader: DownloadFunction = http_download,
) -> PreflightReport:
    """Validate/download the complete external-artifact closure for ``stage``."""

    manifest = load_manifest(manifest_path)
    required_ids = required_artifact_ids(manifest, stage)
    required = set(required_ids)
    root_path = Path(root)
    results: list[ArtifactResult] = []

    for artifact in manifest["artifacts"]:
        destination = _destination(root_path, artifact["destination"])
        if artifact["id"] not in required:
            results.append(
                ArtifactResult(
                    artifact["id"], artifact["destination"], "not-required", "stage does not require this artifact"
                )
            )
            continue

        try:
            size, digest = validate_artifact(destination, artifact)
        except ArtifactValidationError as initial_error:
            if offline:
                status = "invalid" if destination.exists() else "missing"
                results.append(
                    ArtifactResult(
                        artifact["id"],
                        artifact["destination"],
                        status,
                        f"{initial_error}; offline mode prevents download",
                    )
                )
                continue

            partial = destination.with_name(f"{destination.name}.part")
            try:
                downloader(artifact["download_url"], partial, artifact["bytes"])
                size, digest = validate_artifact(partial, artifact)
                backup = _install_valid_partial(partial, destination)
                message = "downloaded and verified"
                if backup is not None:
                    message += f"; invalid prior file preserved as {backup.name}"
                results.append(
                    ArtifactResult(
                        artifact["id"], artifact["destination"], "downloaded", message, size, digest
                    )
                )
            except Exception as download_error:
                status = "invalid" if destination.exists() else "missing"
                results.append(
                    ArtifactResult(
                        artifact["id"],
                        artifact["destination"],
                        status,
                        f"{initial_error}; retrieval failed: {download_error}",
                    )
                )
        else:
            results.append(
                ArtifactResult(
                    artifact["id"], artifact["destination"], "valid", "cached file verified", size, digest
                )
            )

    return PreflightReport(stage, required_ids, tuple(results))


def format_preflight_report(report: PreflightReport) -> str:
    """Return a compact table suitable for terminals and notebooks."""

    lines = [f"Artifact preflight: {'PASS' if report.ok else 'FAIL'} (stage={report.stage})"]
    for row in report.results:
        lines.append(f"  {row.status:12} {row.artifact_id}: {row.message}")
    return "\n".join(lines)


def copy_downloader(source: Path) -> DownloadFunction:
    """Return a deterministic downloader used by tests and offline demonstrations."""

    def download(_url: str, destination: Path, _expected_bytes: int | None) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    return download


__all__ = [
    "ArtifactResult",
    "ArtifactValidationError",
    "DEFAULT_MANIFEST",
    "DEFAULT_SCHEMA",
    "ManifestError",
    "PreflightError",
    "PreflightReport",
    "copy_downloader",
    "format_preflight_report",
    "http_download",
    "load_manifest",
    "required_artifact_ids",
    "resolve_stage_dependencies",
    "run_preflight",
    "sha256_file",
    "validate_artifact",
]
