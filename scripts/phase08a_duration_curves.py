"""Create Phase 08A annual power-duration curves from existing solved networks."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/phase_08a"
FIG = OUT / "figures"
REPORT = ROOT / "reports/PHASE_08A_EL_NINO_DROUGHT_REPORT.md"
SCENARIOS = ["H00", "H10", "H20", "H30", "H40", "H50"]
COLORS = plt.get_cmap("viridis")(np.linspace(.08, .92, len(SCENARIOS)))
plt.rcParams.update({"font.family": "Arial", "font.size": 10})


def series(network: pypsa.Network, carrier: str) -> pd.Series:
    cols = network.generators.index[network.generators.carrier.eq(carrier)]
    if len(cols) == 0:
        return pd.Series(0.0, index=network.snapshots)
    return network.generators_t.p.loc[:, cols].sum(axis=1).abs()


def duration(values: pd.Series) -> pd.DataFrame:
    ordered = np.sort(values.to_numpy(dtype=float))[::-1]
    return pd.DataFrame({"hours": np.arange(1, len(ordered) + 1), "power_mw": ordered})


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    records = []
    risk = pd.read_csv(OUT / "monthly_risk_indicators.csv")
    report = REPORT.read_text(encoding="utf-8")
    for scenario, color in zip(SCENARIOS, COLORS):
        network = pypsa.Network(OUT / f"{scenario}.solved.nc")
        shed = series(network, "load_shedding")
        imports = series(network, "interconnection_colombia")
        combined = shed + imports
        for metric, values in [("Shed power", shed), ("Import power", imports), ("Shed + import", combined)]:
            curve = duration(values)
            curve.insert(0, "scenario", scenario)
            curve.insert(1, "metric", metric)
            records.append(curve)

    curves = pd.concat(records, ignore_index=True)
    curves.to_csv(OUT / "duration_curve_data.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(9.4, 8), sharex=True)
    for i, scenario in enumerate(SCENARIOS):
        for ax, metric, style in [(axes[0], "Shed power", "-"), (axes[1], "Import power", "--")]:
            values = curves[(curves.scenario.eq(scenario)) & (curves.metric.eq(metric))]
            ax.plot(values.hours, values.power_mw, color=COLORS[i], lw=1.7, ls=style, label=scenario)
    axes[0].set_ylabel("Shed power (MW)"); axes[0].set_title("Annual duration curves: shedding and imports")
    axes[1].set_ylabel("Import power (MW)"); axes[1].set_xlabel("Hours per year at or above threshold")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, title="Scenario")
    fig.tight_layout(); fig.savefig(FIG / "duration_curves_shedding_and_imports.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for i, scenario in enumerate(SCENARIOS):
        values = curves[(curves.scenario.eq(scenario)) & (curves.metric.eq("Shed + import"))]
        ax.plot(values.hours, values.power_mw, color=COLORS[i], lw=2, label=scenario)
    ax.set(xlabel="Hours per year at or above threshold", ylabel="Shed + import power (MW)", title="Annual duration curve: combined stress")
    ax.grid(alpha=.2); ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, title="Scenario")
    fig.tight_layout(); fig.savefig(FIG / "duration_curve_combined_stress.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    table_start = report.find("| Scenario | Month | Imports (GWh)")
    table_end = report.find("\n\n![Dry-season imports and shedding]", table_start)
    if table_start >= 0 and table_end >= 0:
        dry = risk[risk.month.isin([9, 10, 11, 12])]
        table = "| Scenario | Month | Imports (GWh) | Shedding (GWh) | Imports + shedding (GWh) | Peak shedding (MW) | Peak (shedding + import) MW | Shedding hours | Model import-cap utilisation |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        for _, row in dry.iterrows():
            table += f"| {row.scenario} | {int(row.month)} | {row.imports_gwh:.1f} | {row.shedding_gwh:.1f} | {row.imports_plus_shedding_gwh:.1f} | {row.peak_shedding_mw:.1f} | {row.peak_imports_plus_shedding_mw:.1f} | {int(row.shedding_hours)} | {row.model_import_capacity_utilisation_pct:.1f}% |\n"
        report = report[:table_start] + table.rstrip() + report[table_end:]

    section = """## 10. Annual power-duration curves

The following duration curves count how many hours per year the system operates at or above each power threshold. Each curve is formed by sorting the 8,760 hourly values from highest to lowest; therefore, moving right means that the threshold is sustained for more hours. These are annual screening curves, not probability distributions.

![Annual shedding and import duration curves](../results/phase_08a/figures/duration_curves_shedding_and_imports.png)

The first panel shows the persistence of load-shedding power, while the second shows the persistence of Colombia import power. A curve extending far to the right indicates frequent exposure, whereas a high value near the left edge indicates a short-duration but severe peak event.

![Annual combined-stress duration curve](../results/phase_08a/figures/duration_curve_combined_stress.png)

The combined curve adds imports and shedding hour by hour. It should be interpreted as a domestic-independence stress indicator: it estimates the power that would need to be supplied domestically to replace imports and eliminate residual shedding at each hour. It is not itself a measure of unmet demand because imports are successfully served energy.

The duration-curve data are saved in `results/phase_08a/duration_curve_data.csv` for further threshold analysis.

"""
    report = report.replace(section, "")
    insert_at = report.find("\n> **Import-cap caveat.")
    if insert_at < 0:
        insert_at = len(report)
    report = report[:insert_at] + "\n" + section + report[insert_at:]
    REPORT.write_text(report, encoding="utf-8")
    print("Created duration curves, updated item 9, and expanded the report without rerunning models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
