"""Run the bounded four-case copperplate diagnostic (maximum 168 hours)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diagnostics.copperplate import FULL_YEAR_HOURS, MAX_HOURS, prepare_case, solve_period


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/copperplate_smoke"))
    parser.add_argument("--full-year", action="store_true", help="Explicitly allow the 8,760-hour diagnostic solve")
    parser.add_argument("--enforce-import-cap", action="store_true",
                        help="Enforce the configured historical annual import-energy band")
    args = parser.parse_args()
    maximum = FULL_YEAR_HOURS if args.full_year else MAX_HOURS
    if not 1 <= args.hours <= maximum:
        parser.error(f"hours must be between 1 and {maximum}")
    source = pypsa.Network(args.network)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, copperplate, restricted in (
        ("original-unrestricted", False, False),
        ("original-thermal-restricted", False, True),
        ("copperplate-unrestricted", True, False),
        ("copperplate-thermal-restricted", True, True),
    ):
        case, mapping = prepare_case(source, args.start, args.hours, copperplate=copperplate, thermal_restricted=restricted, allow_full_year=args.full_year)
        case_result = solve_period(
            case, log_path=args.output / f"{name}.log",
            allow_full_year=args.full_year, time_limit=None,
            enforce_import_cap=args.enforce_import_cap or args.full_year,
        )
        case_result.update({"case": name, "copperplate": copperplate, "thermal_restricted": restricted,
                            "line_count": len(case.lines), "transformer_count": len(case.transformers),
                            "bus_count": len(case.buses)})
        rows.append(case_result)
        mapping.to_csv(args.output / f"{name}_bus_mapping.csv")
    frame = pd.DataFrame(rows)
    frame.to_json(args.output / "comparison.json", orient="records", indent=2)
    (args.output / "README.txt").write_text(
        "Bounded copperplate diagnostic; full-year runs may enforce the configured historical import-energy band.\n"
        f"Period: {args.start}, {args.hours} hours.\n", encoding="utf-8"
    )
    print(frame[["case", "solver_status", "termination_condition", "balance_residual_mwh", "link_losses_mwh"]].to_string(index=False))
    return 0 if frame.termination_condition.eq("optimal").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
