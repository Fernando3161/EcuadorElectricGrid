"""Build the local figures used by docs/linkedin_posts.md.

The Phase 11 result bundle is the data source. No simulation is rerun here.
"""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/docker_runs/phase11-20261027-174819"
FIGURES = ROOT / "docs/linkedin_figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def copy_run_figure(source, name):
    shutil.copy2(RUN / "reports/figures" / source, FIGURES / name)


copy_run_figure("phase_05/candidate_topology_by_voltage_2022.png", "01_voltage_network.png")
copy_run_figure("phase_06a/a02_annual_generation_map.png", "03_generation_bubbles.png")
copy_run_figure("phase_11_reporting/08a_monthly_shedding_and_imports.png", "05_monthly_shedding_and_imports.png")
copy_run_figure("phase_08a/shedding_vs_caudal_reduction.png", "06_stress_response.png")

# Compare shares, not raw GWh, to make the nonconventional miss visible.
mix = pd.read_csv(RUN / "results/phase_06/validation_2022.csv").set_index("carrier")
labels = ["Hydro", "Thermal", "Nonconventional", "Interconnection"]
keys = ["hydro", "thermal", "nonconventional", "interconnection"]
ref = 100 * mix.loc[keys, "reference_share"].to_numpy()
model = 100 * mix.loc[keys, "simulated_share"].to_numpy()
fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), layout="constrained")
for ax, label, a, b in zip(axes.flat, labels, ref, model):
    ax.barh([1, 0], [a, b], height=0.52, color=["#64748b", "#2563eb"])
    ax.set_yticks([1, 0], ["CENACE", "Model"])
    ax.set_xlim(0, max(a, b) * 1.3 + 0.2)
    ax.set_title(label, loc="left", fontsize=15, fontweight="bold")
    for y, value in [(1, a), (0, b)]:
        ax.text(value + max(a, b) * 0.025 + 0.02, y, f"{value:.2f}%", va="center", fontsize=12)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", length=0)
fig.suptitle("2022 annual supply shares | CENACE vs model", fontsize=20, fontweight="bold")
fig.text(0.05, 0.005, "Nonconventional is 0.64 percentage points low; this screen fails.",
         color="#b91c1c", fontsize=11)
save(fig, "02_generation_shares.png")

monthly = pd.read_csv(RUN / "analysis/phase08a_monthly_kpis.csv")
order = [f"H{i:02d}" for i in range(0, 51, 10)]
heat = monthly.pivot(index="scenario", columns="month", values="peak_combined_stress_mw").loc[order]
fig, ax = plt.subplots(figsize=(14, 6.8), layout="constrained")
cmap = LinearSegmentedColormap.from_list("stress", ["#eff6ff", "#60a5fa", "#1e40af", "#7f1d1d"])
im = ax.imshow(heat.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1700)
ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_yticks(range(len(order)), order)
ax.set_xlabel("2026 month")
ax.set_ylabel("Sep–Dec hydro-flow reduction: H00–H50")
ax.set_title("Peak hour: imports + unserved demand (MW)", fontsize=19, fontweight="bold", pad=16)
for row in range(heat.shape[0]):
    for col in range(heat.shape[1]):
        value = heat.iat[row, col]
        ax.text(col, row, f"{value:,.0f}", ha="center", va="center",
                fontsize=9.5, color="white" if value > 950 else "#0f172a")
fig.colorbar(im, ax=ax, shrink=0.85, label="MW in the peak hour")
ax.text(0, -0.16, "Coincident sum. Imports are served load; only the shedding component is unmet demand.",
        transform=ax.transAxes, fontsize=10, color="#475569")
save(fig, "04_peak_stress_heatmap.png")


def box(ax, xy, width, height, title, detail, face="#eff6ff", edge="#2563eb"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.025",
                                facecolor=face, edgecolor=edge, linewidth=2))
    ax.text(x + 0.03, y + height - 0.06, title, fontsize=15, fontweight="bold", va="top", color="#0f172a")
    ax.text(x + 0.03, y + height - 0.16, detail, fontsize=11, va="top", color="#334155", linespacing=1.5)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                                 linewidth=2.3, color="#475569"))


fig, ax = plt.subplots(figsize=(15, 7.5))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
ax.text(0.02, 0.95, "A public SNI data pipeline | proposed", fontsize=23, fontweight="bold", color="#0f172a")
box(ax, (0.02, 0.58), 0.27, 0.26, "1. Publish source data", "Plant output and status\nSubstations, lines, transformers\nDemand and interchanges")
box(ax, (0.365, 0.58), 0.27, 0.26, "2. Harmonize", "Stable asset IDs + coordinates\nUnits, timestamps, versions\nMetering boundary + provenance", "#f0fdf4", "#16a34a")
box(ax, (0.71, 0.58), 0.27, 0.26, "3. Serve openly", "Documented API + downloads\nHistorical and live snapshots\nLicensing and quality flags", "#fefce8", "#ca8a04")
arrow(ax, (0.31, 0.71), (0.35, 0.71))
arrow(ax, (0.655, 0.71), (0.695, 0.71))
box(ax, (0.20, 0.15), 0.27, 0.23, "Reproducible models", "PyPSA / PyPSA-Earth\nScenario and stress studies", "#faf5ff", "#9333ea")
box(ax, (0.56, 0.15), 0.27, 0.23, "Public flow view", "Map, mix, imports, flows\nTime slider + data download", "#fff7ed", "#ea580c")
arrow(ax, (0.83, 0.56), (0.72, 0.40))
arrow(ax, (0.76, 0.56), (0.36, 0.40))
save(fig, "07_open_data_pipeline.png")

fig, ax = plt.subplots(figsize=(15, 7.5))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
ax.text(0.02, 0.95, "What a public Ecuador SNI site could show | concept", fontsize=21, fontweight="bold", color="#0f172a")
box(ax, (0.02, 0.13), 0.59, 0.70, "LIVE GRID MAP", "", "#f8fafc", "#94a3b8")
points = [(0.16, 0.65), (0.25, 0.56), (0.21, 0.40), (0.34, 0.44), (0.43, 0.29), (0.51, 0.51)]
links = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5)]
for a, b in links:
    ax.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]], color="#2563eb", linewidth=4, zorder=1)
for x, y in points:
    ax.scatter(x, y, s=520, color="#0ea5e9", edgecolors="white", linewidths=3, zorder=2)
ax.text(0.08, 0.22, "Voltage • plant output • line flow • imports", fontsize=12, color="#334155")
box(ax, (0.65, 0.57), 0.32, 0.26, "SYSTEM NOW", "Generation by plant and fuel\nDemand and cross-border exchange", "#eff6ff", "#2563eb")
box(ax, (0.65, 0.27), 0.32, 0.23, "EXPLORE HISTORY", "Time slider • quality flags\nDownload CSV / API", "#f0fdf4", "#16a34a")
ax.text(0.67, 0.16, "Illustrative interface; no live feed is connected.", fontsize=11, color="#64748b")
save(fig, "08_public_sni_site_concept.png")

print(f"Wrote 8 figures to {FIGURES}")
