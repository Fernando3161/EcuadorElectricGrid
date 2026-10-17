"""Build reproducible CENACE/TVD comparison tables and figures for the diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_TO_VALIDATED = {
    "hydro": "hydro",
    "oil": "thermal",
    "biogas": "nonconventional",
    "biomass": "nonconventional",
    "onwind": "nonconventional",
    "solar": "nonconventional",
    "interconnection_colombia": "interconnection",
}


def main() -> None:
    out = ROOT / "results/diagnostics/copperplate_full_year_capped"
    cases = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    reference = pd.read_csv(ROOT / "data/processed/statistics/ecuador_sni_2022_supply_reference.csv")
    reference = reference[reference["in_scope"]].set_index("carrier")["reference_gwh"]
    rows = []
    for case in cases:
        raw = case.get("generator_supply_mwh_by_raw_carrier", {})
        sim = pd.Series(0.0, index=reference.index)
        for carrier, value in raw.items():
            target = RAW_TO_VALIDATED.get(carrier)
            if target in sim.index:
                sim[target] += float(value) / 1000.0
        ref_share = reference / reference.sum()
        sim_share = sim / sim.sum()
        errors = 100.0 * (sim_share - ref_share)
        rows.append({
            "case": case["case"],
            "solver_status": case["solver_status"],
            "termination_condition": case["termination_condition"],
            "hydro_gwh": sim["hydro"], "thermal_gwh": sim["thermal"],
            "nonconventional_gwh": sim["nonconventional"],
            "interconnection_gwh": sim["interconnection"],
            "load_shedding_gwh": raw.get("load_shedding", 0.0) / 1000.0,
            "losses_gwh": case["link_losses_mwh"] / 1000.0,
            "balance_residual_mwh": case["balance_residual_mwh"],
            "total_gwh": sim.sum(), "tvd": 0.5 * abs(sim_share - ref_share).sum(),
            "max_abs_error_pp": abs(errors).max(),
        })
        for carrier in reference.index:
            rows[-1][f"{carrier}_error_pp"] = errors[carrier]
    phase06d = pd.read_csv(ROOT / "results/phase_06d/renewable-calibration/validation_2022.csv")
    base = {"case": "original-phase06d-renewable-calibration", "solver_status": "ok",
            "termination_condition": "optimal", "load_shedding_gwh": None,
            "losses_gwh": None, "balance_residual_mwh": None}
    base.update({"hydro_gwh": phase06d.loc[phase06d.carrier.eq("hydro"), "simulated_gwh"].iloc[0],
                 "thermal_gwh": phase06d.loc[phase06d.carrier.eq("thermal"), "simulated_gwh"].iloc[0],
                 "nonconventional_gwh": phase06d.loc[phase06d.carrier.eq("nonconventional"), "simulated_gwh"].iloc[0],
                 "interconnection_gwh": phase06d.loc[phase06d.carrier.eq("interconnection"), "simulated_gwh"].iloc[0]})
    vals = pd.Series({k: base[f"{k}_gwh"] for k in reference.index})
    shares = vals / vals.sum()
    errors = 100.0 * (shares - reference / reference.sum())
    base.update({"total_gwh": vals.sum(), "tvd": 0.5 * abs(shares - reference / reference.sum()).sum(),
                 "max_abs_error_pp": abs(errors).max()})
    base.update({f"{k}_error_pp": errors[k] for k in reference.index})
    metrics = pd.DataFrame([base] + rows)
    metrics.to_csv(out / "validation_metrics.csv", index=False)
    comparison = metrics.melt(id_vars=["case"], value_vars=[f"{k}_gwh" for k in reference.index],
                              var_name="carrier", value_name="simulated_gwh")
    comparison["carrier"] = comparison["carrier"].str.replace("_gwh", "", regex=False)
    comparison["reference_gwh"] = comparison["carrier"].map(reference)
    comparison.to_csv(out / "validation_comparison.csv", index=False)

    plotdir = ROOT / "reports/figures"
    plotdir.mkdir(parents=True, exist_ok=True)
    labels = metrics["case"].str.replace("original-phase06d-renewable-calibration", "Phase 06D", regex=False)
    labels = labels.str.replace("-", "\n", regex=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    comparison.pivot(index="case", columns="carrier", values="simulated_gwh").plot.bar(ax=ax)
    ax.axhline(0, color="black", linewidth=.5); ax.set_ylabel("GWh"); ax.set_title("Annual supply vs CENACE 2022 reference")
    ax.legend(title="Carrier"); fig.tight_layout(); fig.savefig(plotdir / "phase_06d_copperplate_capped_carrier_gwh.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    errcols = [f"{k}_error_pp" for k in reference.index]
    metrics.set_index(labels)[errcols].plot.bar(ax=ax)
    ax.axhline(0, color="black", linewidth=.5); ax.set_ylabel("Signed error (percentage points)"); ax.set_title("Carrier-share error vs CENACE 2022")
    ax.legend([k.replace("_error_pp", "") for k in errcols], title="Carrier"); fig.tight_layout(); fig.savefig(plotdir / "phase_06d_copperplate_capped_errors_pp.png", dpi=160); plt.close(fig)
    fig, ax1 = plt.subplots(figsize=(10, 5)); x = range(len(metrics)); ax1.bar(x, metrics.tvd, label="TVD", color="steelblue"); ax1.axhline(.03, color="red", linestyle="--", label="TVD screen 0.03"); ax1.set_ylabel("TVD"); ax1.set_xticks(list(x), labels, rotation=20, ha="right"); ax1.legend(loc="upper left"); ax2=ax1.twinx(); ax2.plot(list(x), metrics.max_abs_error_pp, "o-", color="darkorange", label="Max abs error (pp)"); ax2.set_ylabel("Max absolute carrier error (pp)"); ax2.legend(loc="upper right"); ax1.set_title("Validation indicators"); fig.tight_layout(); fig.savefig(plotdir / "phase_06d_copperplate_capped_metrics.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
