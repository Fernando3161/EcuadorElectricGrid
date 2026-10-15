"""Command-line entry point for environment and external-data preflight."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "config" / "artifacts.yaml"
EXPECTED_ENVIRONMENT = "pypsa-earth-ec"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check the approved environment and obtain required artifacts."
    )
    parser.add_argument("--stage", default="00_preflight", help="Manifest stage ID")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Validate cached files without attempting downloads.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prefix_name = Path(sys.prefix).name
    detected = (
        prefix_name
        if prefix_name.lower() == EXPECTED_ENVIRONMENT.lower()
        else os.environ.get("CONDA_DEFAULT_ENV") or prefix_name
    )
    if detected.lower() != EXPECTED_ENVIRONMENT.lower():
        message = (
            f"Environment preflight: FAIL\n  environment: {detected} "
            f"(expected {EXPECTED_ENVIRONMENT})\n"
            "Activate the existing approved environment and rerun this command."
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "environment": {
                            "ok": False,
                            "detected_name": detected,
                            "expected_name": EXPECTED_ENVIRONMENT,
                        }
                    },
                    indent=2,
                )
            )
        else:
            print(message)
        return 2

    from .artifacts import format_preflight_report, run_preflight
    from .environment import check_environment, format_environment_report

    environment = check_environment()
    if not environment.ok:
        if args.json:
            print(json.dumps({"environment": environment.as_dict()}, indent=2))
        else:
            print(format_environment_report(environment))
        return 2

    report = run_preflight(
        args.stage,
        manifest_path=args.manifest,
        root=args.root,
        offline=args.offline,
    )
    if args.json:
        print(
            json.dumps(
                {"environment": environment.as_dict(), "artifacts": report.as_dict()},
                indent=2,
            )
        )
    else:
        print(format_environment_report(environment))
        print(format_preflight_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
