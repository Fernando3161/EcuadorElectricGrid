"""Run one Phase 08A case in an isolated Python process."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/phase_08a"

from src.availability import load_hydro_assignments, load_monthly_flows
from src.phase08a import (ASSIGNMENT_PATH, FLOW_PATH, GENERATOR_PATH, REFERENCE_NETWORK,
                          apply_monthly_flow_shock, build_2026_demand, build_scenario_network,
                          rebuild_hydro_profiles, select_caudals)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduction", type=float, required=True)
    parser.add_argument("--method", choices=("simplex", "ipm"), default="simplex")
    args = parser.parse_args()
    reduction = float(args.reduction)
    scenario = f"H{int(round(reduction * 100)):02d}"
    flows = load_monthly_flows(FLOW_PATH)
    generators = pd.read_csv(GENERATOR_PATH)
    assignments = load_hydro_assignments(ASSIGNMENT_PATH)
    reference = pypsa.Network(REFERENCE_NETWORK)
    selected = select_caudals(generators, assignments, count=4)
    shocked = apply_monthly_flow_shock(flows, caudal_ids=selected.caudal_id.tolist(), months=(9, 10, 11, 12), reduction=reduction)
    hydro, diagnostics = rebuild_hydro_profiles(generators, shocked, assignments, year=2026)
    demand = build_2026_demand(reference)
    network = build_scenario_network(reference, hydro, demand, year=2026)
    status, termination = network.optimize(
        solver_name="highs",
        solver_options={"threads": 1, "solver": args.method},
        log_fn=str(OUT / f"{scenario}_highs_{args.method}.log"),
    )
    network.meta.update({"scenario": scenario, "caudal_reduction": reduction, "stress_months": [9, 10, 11, 12], "solver_status": str(status), "termination_condition": str(termination)})
    network.export_to_netcdf(OUT / f"{scenario}.solved.nc")
    supply = network.generators_t.p.mul(network.generators.sign, axis=1).sum() / 1000
    by_carrier = supply.groupby(network.generators.carrier).sum()
    demand_gwh = float(network.loads_t.p_set.sum().sum() / 1000)
    shed = float(by_carrier.get("load_shedding", 0.0))
    result = {"scenario": scenario, "caudal_reduction": reduction, "demand_gwh": demand_gwh,
              "solver_status": str(status), "termination_condition": str(termination),
              "load_shedding_gwh": shed, "load_shedding_pct": 100 * shed / demand_gwh,
              **{f"{carrier}_gwh": float(value) for carrier, value in by_carrier.items()}}
    (OUT / f"{scenario}_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    diagnostics.to_csv(OUT / f"{scenario}_hydro_profile_diagnostics.csv", index=False)
    shocked.reset_index(names="caudal_id").to_csv(OUT / f"{scenario}_caudals.csv", index=False)
    print(json.dumps(result, indent=2))
    return 0 if (status, termination) == ("ok", "optimal") else 1


if __name__ == "__main__":
    raise SystemExit(main())
