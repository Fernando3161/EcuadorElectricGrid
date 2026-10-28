"""Runtime checks for the approved EcuadorElectricGrid Conda environment."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import Version


EXPECTED_ENVIRONMENT = "pypsa-earth-ec"
REQUIRED_PACKAGES = {
    "pypsa": ">=0.25,<=0.30.3",
    "numpy": "<2",
    "pandas": ">=2",
    "xarray": ">=2023.11.0,<=2025.1.2",
    "h5netcdf": ">=1",
    "PyYAML": ">=6",
    "requests": ">=2",
    "jsonschema": ">=4",
    "nbformat": ">=5",
    "nbclient": ">=0.10",
}


@dataclass(frozen=True)
class PackageCheck:
    package: str
    requirement: str
    installed: str | None
    ok: bool
    message: str


@dataclass(frozen=True)
class EnvironmentReport:
    expected_name: str
    detected_name: str
    python_version: str
    name_ok: bool
    python_ok: bool
    packages: tuple[PackageCheck, ...]

    @property
    def ok(self) -> bool:
        return self.name_ok and self.python_ok and all(row.ok for row in self.packages)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ok"] = self.ok
        return result


def detected_environment_name() -> str:
    """Return the active Conda name, falling back to the interpreter prefix."""

    prefix_name = Path(sys.prefix).name
    if prefix_name.lower() == EXPECTED_ENVIRONMENT.lower():
        return prefix_name
    return os.environ.get("CONDA_DEFAULT_ENV") or prefix_name


def check_environment(
    expected_name: str = EXPECTED_ENVIRONMENT,
    requirements: dict[str, str] | None = None,
) -> EnvironmentReport:
    """Check environment identity and the packages needed by the preflight."""

    detected = detected_environment_name()
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    python_ok = Version(python_version) in SpecifierSet(">=3.10,<3.12")
    packages = []
    for package, requirement in (requirements or REQUIRED_PACKAGES).items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            packages.append(
                PackageCheck(package, requirement, None, False, "package is not installed")
            )
            continue
        ok = Version(installed) in SpecifierSet(requirement)
        message = "ok" if ok else f"installed {installed} does not satisfy {requirement}"
        packages.append(PackageCheck(package, requirement, installed, ok, message))

    return EnvironmentReport(
        expected_name=expected_name,
        detected_name=detected,
        python_version=python_version,
        name_ok=detected.lower() == expected_name.lower(),
        python_ok=python_ok,
        packages=tuple(packages),
    )


def format_environment_report(report: EnvironmentReport) -> str:
    """Return a concise human-readable environment report."""

    state = "PASS" if report.ok else "FAIL"
    lines = [
        f"Environment preflight: {state}",
        f"  environment: {report.detected_name} (expected {report.expected_name})",
        f"  python: {report.python_version} (required >=3.10,<3.12)",
    ]
    for row in report.packages:
        installed = row.installed or "missing"
        marker = "ok" if row.ok else "FAIL"
        lines.append(f"  {row.package}: {installed} ({row.requirement}) [{marker}]")
    return "\n".join(lines)


__all__ = [
    "EXPECTED_ENVIRONMENT",
    "EnvironmentReport",
    "PackageCheck",
    "REQUIRED_PACKAGES",
    "check_environment",
    "detected_environment_name",
    "format_environment_report",
]
