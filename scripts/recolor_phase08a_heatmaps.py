"""Redraw only the Phase 08A heatmaps for improved annotation contrast."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/phase_08a"
FIG = OUT / "figures"
SCENARIOS = ["H00", "H10", "H20", "H30", "H40", "H50"]
DRY_MONTHS = [9, 10, 11, 12]
plt.rcParams.update({"font.family": "Arial", "font.size": 10})


def heatmap(data: pd.DataFrame, title: str, colorbar: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    values = data.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=max(1.0, float(np.nanmax(values))))
    ax.set(xticks=np.arange(4), xticklabels=["Sep", "Oct", "Nov", "Dec"],
           yticks=np.arange(len(data)), yticklabels=data.index,
           xlabel="Month", ylabel="Scenario", title=title)
    threshold = float(np.nanmax(values)) * 0.55
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.0f}", ha="center", va="center",
                    color="black", fontsize=9, fontweight="bold",
                    bbox={"facecolor": "white", "alpha": 0.42, "edgecolor": "none", "pad": 1.5})
    fig.colorbar(image, ax=ax, label=colorbar)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    risk = pd.read_csv(OUT / "monthly_risk_indicators.csv")
    dry = risk[risk.month.isin(DRY_MONTHS)]
    peak_shed = dry.pivot(index="scenario", columns="month", values="peak_shedding_mw").reindex(SCENARIOS)
    peak_combined = dry.pivot(index="scenario", columns="month", values="peak_imports_plus_shedding_mw").reindex(SCENARIOS)
    heatmap(peak_shed, "Worst hourly load shedding by dry-season month (MW)", "Peak shedding (MW)", "peak_shedding_by_dry_month.png")
    heatmap(peak_combined, "Worst combined import-plus-shedding stress (MW)", "Peak imports + shedding (MW)", "peak_combined_stress_by_dry_month.png")
    print("Redrew only the two Phase 08A heatmaps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
