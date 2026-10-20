"""Run unchanged Phase 06H notebooks in a separately prepared checkout.

Invoke with the named Conda environment. This is environment acceptance tooling,
not a replacement for the student-facing notebooks. A full-year run requires an
explicit flag; no notebook cells, source modules or scientific settings are edited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

NOTEBOOKS = (
    "00_preflight_and_data_inventory",
    "01_prepare_2022_demand",
    "02_prepare_2022_generation",
    "03_prepare_renewable_and_hydro_profiles",
    "04_build_and_validate_2022_topology",
    "05_assemble_and_smoke_test_2022_network",
    "06_run_and_validate_2022_full_year",
    "06a_diagnose_2022_validation",
)


def fingerprints(root: Path) -> dict[str, str]:
    paths = [*root.joinpath("src").rglob("*.py")]
    paths += [root / "notebooks" / f"{name}.ipynb" for name in NOTEBOOKS]
    paths += [root / "config" / name for name in ("artifacts.yaml", "validation_2022.yaml")]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def restore_pinned_inputs(root: Path) -> list[dict[str, str]]:
    """Recover manifest bytes from Git where checkout newline conversion changed them.

    Only exact expected SHA-256 matches are installed. Generated products are
    left to their writer notebooks. An unmatched input remains a preflight error.
    """
    import yaml

    manifest = yaml.safe_load((root / "config/artifacts.yaml").read_text(encoding="utf-8"))
    restored = []
    for artifact in manifest["artifacts"]:
        if artifact.get("availability") != "repository" or artifact.get("role") == "generated_product":
            continue
        relative = artifact["destination"]
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"Input destination escapes checkout: {relative}")
        expected = artifact["sha256"]
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        blob = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=root,
                              capture_output=True, check=False)
        if blob.returncode or b"\0" in blob.stdout:
            continue
        lf = blob.stdout.replace(b"\r\n", b"\n")
        candidates = [("git blob", blob.stdout), ("LF", lf),
                      ("CRLF", lf.replace(b"\n", b"\r\n"))]
        if path.suffix.lower() == ".csv":
            # Original CSV writers use CRLF record terminators but keep LF
            # inside quoted provenance cells. Global conversion changes both.
            records = csv.reader(io.StringIO(lf.decode("utf-8"), newline=""))
            buffer = io.StringIO(newline="")
            csv.writer(buffer, lineterminator="\r\n").writerows(records)
            candidates.append(("CSV record CRLF; embedded LF", buffer.getvalue().encode("utf-8")))
        for convention, candidate in candidates:
            if (len(candidate) == artifact["bytes"]
                    and hashlib.sha256(candidate).hexdigest() == expected):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(candidate)
                restored.append({"path": relative, "representation": convention, "sha256": expected})
                break
    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--allow-full-year", action="store_true")
    parser.add_argument("--restore-pinned-inputs", action="store_true",
                        help="In the isolated checkout only, restore exact manifest text bytes from Git.")
    parser.add_argument("--start", choices=NOTEBOOKS, default=NOTEBOOKS[0],
                        help="Resume only after inspecting the retained prior failure.")
    args = parser.parse_args()
    if not args.allow_full_year:
        parser.error("This unchanged pipeline solves 8,760 hours; --allow-full-year is required.")
    if Path(sys.prefix).name.casefold() != "pypsa-earth-ec":
        parser.error("Run through conda run -n pypsa-earth-ec python ...")
    root = args.checkout.resolve()
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
    evidence = args.evidence.resolve() / run_id
    if not (root / "src" / "notebook_runner.py").is_file():
        parser.error("--checkout must contain the Phase 06H pipeline, not the old main tree.")
    evidence.mkdir(parents=True, exist_ok=True)
    restorations = restore_pinned_inputs(root) if args.restore_pinned_inputs else []
    before = fingerprints(root)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    report = {"executable": sys.executable, "checkout": str(root), "revision": revision,
              "start": args.start,
              "repository_input_restorations": restorations,
              "input_hashes": before, "full_year_authorized": True, "stages": []}
    report_path = evidence / f"pipeline-{time.strftime('%Y%m%d-%H%M%S')}.json"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"

    def run(label: str, arguments: list[str]) -> None:
        print(f"START {label}", flush=True)
        started = time.monotonic()
        log = evidence / f"{label}.log"
        with log.open("w", encoding="utf-8") as output:
            completed = subprocess.run([sys.executable, *arguments], cwd=root, env=env,
                                       stdout=output, stderr=subprocess.STDOUT, check=False)
        row = {"stage": label, "exit_code": completed.returncode,
               "seconds": round(time.monotonic() - started, 2), "log": str(log)}
        report["stages"].append(row)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"END {label}: exit={completed.returncode}, seconds={row['seconds']}", flush=True)
        if completed.returncode:
            print(log.read_text(encoding="utf-8", errors="replace")[-12000:], flush=True)
            raise RuntimeError(f"{label} failed; retained log: {log}")

    try:
        # Stage 04 itself is offline. Obtain its manifest-pinned GADM input before
        # execution using the existing common preflight, without changing a cell.
        run("topology-artifact-preflight", ["-m", "src.preflight", "--stage", "04_topology"])
        for name in NOTEBOOKS[NOTEBOOKS.index(args.start):]:
            run(name, ["-m", "src.notebook_runner", f"notebooks/{name}.ipynb",
                       "--output", str(evidence / f"{name}.executed.ipynb"),
                       "--timeout", "21600"])
        summary = json.loads((root / "results/phase_06/validation_summary.json").read_text())
        if (summary["solver_status"], summary["termination_condition"]) != ("ok", "optimal"):
            raise RuntimeError("Notebook 06 did not record a successful optimal solve.")
        metadata = json.loads((root / "results/phase_06a/diagnostic_metadata.json").read_text())
        manifest = json.loads((root / "results/phase_06a/figure_manifest.json").read_text())
        if metadata["snapshots"] != 8760 or metadata["solver_invoked"]:
            raise RuntimeError("Notebook 06a did not record read-only annual diagnostics.")
        if not manifest or any(row["status"] != "generated" for row in manifest):
            raise RuntimeError("Notebook 06a skipped a diagnostic figure; inspect its figure manifest.")
        artifact = json.loads((root / "results/phase_06/artifacts.json").read_text())
        with (root / "results/phase_06/network_solved_2022.nc").open("rb") as handle:
            network_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if (artifact["sha256"] != network_sha256
                or metadata["network_sha256"] != network_sha256
                or any(row["network_sha256"] != network_sha256 for row in manifest)):
            raise RuntimeError("Solved-network and diagnostic provenance checksums disagree.")
        report["network_sha256"] = network_sha256
        report["solver_status"] = summary["solver_status"]
        report["termination_condition"] = summary["termination_condition"]
        report["scientific_status"] = summary["status"]
        report["diagnostic_figures"] = len(manifest)
        report["status"] = "PASS"
    except Exception as exc:
        report["status"], report["error"] = "FAIL", str(exc)
        raise
    finally:
        after = fingerprints(root)
        report["pipeline_files_unchanged"] = before == after
        report["output_hashes"] = after
        if before != after:
            report["status"] = "FAIL"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if before != after:
            raise RuntimeError("Pipeline file fingerprints changed; inspect the evidence report.")


if __name__ == "__main__":
    main()
