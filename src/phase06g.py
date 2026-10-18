"""Phase 06G hydro-profile reassessment, full-year runs, and reporting.

This module deliberately keeps the retained Phase 06D network boundary and
changes one profile/capacity processing choice per experiment. Existing Phase
06F full-year outputs are copied into the Phase 06G result index only after
their 8,760-snapshot metadata and checksums are verified. New experiments are
complete hourly PyPSA optimizations; no solved NetCDF is edited in place.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns

from src.availability import load_hydro_assignments, load_monthly_flows
from src.validation import (
    build_supply_reference,
    calculate_validation_metrics,
    enforce_historical_import_energy,
    load_raw_supply_reference,
    load_validation_contract,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/phase_06d/baseline/network_solved_2022.nc"
F06F = ROOT / "results/phase_06f"
RESULTS = ROOT / "results/phase_06g"
INTERIM = ROOT / "data/interim/hydro_reassessment_2022"
FIGURES = ROOT / "reports/figures/phase_06g"
YEAR = 2022
CENACE_HYDRO_GWH = 24499.54


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hydro_ids(network: pypsa.Network) -> pd.Index:
    return network.generators.index[network.generators.carrier.eq("hydro")]


def hourly_profile(values: pd.Series, *, year: int, month_anchor: str = "mid") -> pd.Series:
    index = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    anchors = []
    for month in range(1, 13):
        start = pd.Timestamp(year, month, 1)
        if month_anchor == "start":
            anchor = start
        else:
            anchor = start + pd.Timedelta(hours=12) + pd.Timedelta(days=start.days_in_month / 2)
        anchors.append(anchor)
    monthly = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(anchors))
    expanded = monthly.reindex(monthly.index.union(index)).sort_index()
    return expanded.interpolate(method="time").reindex(index).bfill().ffill()


def build_hydro_variant(network: pypsa.Network, *, quantile: float = 0.95,
                        reservoir_window: int = 3, month_anchor: str = "mid",
                        capacity_mode: str = "nominal", scope: str = "all") -> None:
    flows = load_monthly_flows(ROOT / "data/raw/generation/caudals.csv")
    assignments = load_hydro_assignments(ROOT / "data/interim/generation/hydro_profile_assignments_2022.csv")
    source = pd.read_csv(ROOT / "data/processed/generation/generators_2022.csv").set_index("plant_id")
    hydro = hydro_ids(network)
    dynamic = network.generators_t.p_max_pu.reindex(index=network.snapshots, columns=hydro).copy()
    for plant_id in hydro:
        row = source.loc[plant_id]
        assignment = assignments.loc[assignments.plant_id.eq(plant_id)].iloc[0]
        flow = flows.loc[assignment.caudal_id].astype(float)
        is_reservoir = str(row.subtechnology_original).strip().lower() in {"embalse", "reservoir"}
        if scope == "ror" and is_reservoir:
            continue
        if scope == "reservoir" and not is_reservoir:
            continue
        effective = flow.rolling(reservoir_window, min_periods=1).mean() if is_reservoir else flow
        rated = float(effective.quantile(quantile))
        profile = (effective / rated).clip(0.0, 1.0)
        values = hourly_profile(profile, year=YEAR, month_anchor=month_anchor)
        start = pd.Timestamp(row.operating_start_date_2022)
        values.loc[values.index < start] = 0.0
        dynamic.loc[:, plant_id] = values.to_numpy()
        if capacity_mode == "effective":
            network.generators.loc[plant_id, "p_nom"] = float(row.effective_capacity_mw)
    network.generators_t.p_max_pu.loc[:, hydro] = dynamic.to_numpy()


def apply_change(network: pypsa.Network, scenario: str) -> dict:
    if scenario == "G06G_BASE":
        return {"controlled_change": "none; retained Phase 06F baseline"}
    if scenario == "G06G_A100":
        network.generators_t.p_max_pu.loc[:, hydro_ids(network)] = 1.0
        return {"controlled_change": "all hydro p_max_pu=1 upper-bound diagnostic"}
    if scenario == "G06G_CAP_EFFECTIVE":
        build_hydro_variant(network, capacity_mode="effective")
        return {"controlled_change": "hydro p_nom replaced by source effective_capacity_mw"}
    if scenario == "G06G_ROR_Q90":
        build_hydro_variant(network, quantile=0.90, scope="ror")
        return {"controlled_change": "all-flow rated quantile 0.90; bounded sensitivity"}
    if scenario == "G06G_ROR_Q100":
        build_hydro_variant(network, quantile=1.00, scope="ror")
        return {"controlled_change": "all-flow rated quantile 1.00; bounded sensitivity"}
    if scenario == "G06G_RES_1M":
        build_hydro_variant(network, reservoir_window=1, scope="reservoir")
        return {"controlled_change": "reservoir effective flow trailing 1 month"}
    if scenario == "G06G_RES_6M":
        build_hydro_variant(network, reservoir_window=6, scope="reservoir")
        return {"controlled_change": "reservoir effective flow trailing 6 months"}
    if scenario == "G06G_TEMP_START":
        build_hydro_variant(network, month_anchor="start", scope="all")
        return {"controlled_change": "monthly anchors at month start"}
    if scenario == "G06G_COMB_ROR_Q90_RES6M":
        build_hydro_variant(network, quantile=0.90, scope="ror")
        build_hydro_variant(network, reservoir_window=6, scope="reservoir")
        return {"controlled_change": "run-of-river Q90 plus reservoir trailing 6-month mean"}
    raise ValueError(f"Unknown runnable scenario: {scenario}")


def experiment_registry() -> pd.DataFrame:
    rows = [
        ("G06G_BASE", "baseline reproduction", "contract-aligned full-year baseline with import constraint", "Phase 06F inputs; new solve required", False),
        ("G06G_A100", "availability upper bound", "all hydro p_max_pu=1 with import constraint", "verified upper-bound diagnostic; new matched solve", True),
        ("G06G_F06F_P45_LEGACY", "legacy baseline reference", "verified reuse of F06F_P45 without import constraint", "Phase 06F retained result", True),
        ("G06G_CAP_EFFECTIVE", "capacity definition", "source effective hydro capacity", "PME_2022_CH4 / generators_2022.csv", True),
        ("G06G_ROR_Q90", "run-of-river rated flow", "95th to 90th percentile", "bounded method sensitivity; not official evidence", True),
        ("G06G_ROR_Q100", "run-of-river rated flow", "95th to 100th percentile", "bounded method sensitivity; not official evidence", True),
        ("G06G_RES_1M", "reservoir representation", "trailing 3 to 1 month", "bounded method sensitivity; no reservoir evidence", True),
        ("G06G_RES_6M", "reservoir representation", "trailing 3 to 6 months", "bounded method sensitivity; no reservoir evidence", True),
        ("G06G_TEMP_START", "temporal processing", "mid-month to month-start anchors", "calendar-processing sensitivity", True),
        ("G06G_COMB_ROR_Q90_RES6M", "combined hydro profile", "ROR Q90 plus reservoir 6-month mean", "combination of two bounded sensitivities", True),
        ("G06G_ASSIGN", "assignment confidence", "not testable", "no traceable alternative plant-flow mapping", True),
        ("G06G_OUTAGE", "technical unavailability", "not testable", "aggregate outage evidence lacks complete plant-hour masks", True),
        ("G06G_TECH", "flow/head/efficiency", "not testable", "traceable plant parameters unavailable", True),
        ("G06G_NET", "network attribution", "accounting/diagnostic only", "retained topology; no topology change", True),
        ("G06G_ACCT", "reference accounting", "accounting recomputation", "verified full-year output; cannot change dispatch", True),
    ]
    frame = pd.DataFrame(rows, columns=["experiment_id", "hypothesis", "controlled_change", "evidence", "diagnostic_only"])
    frame["full_year_required"] = True
    frame["snapshots_required"] = 8760
    frame["unchanged_assumptions"] = "retained Phase 06D topology, demand, costs, interconnection and non-hydro profiles"
    return frame


def carrier_energy(network: pypsa.Network) -> tuple[pd.Series, dict]:
    generation = network.generators_t.p.reindex(index=network.snapshots, columns=network.generators.index).fillna(0.0)
    positive = generation.clip(lower=0.0).sum() / 1000.0
    mapped = {"hydro": "hydro", "oil": "thermal", "biomass": "nonconventional", "biogas": "nonconventional", "onwind": "nonconventional", "solar": "nonconventional", "interconnection_colombia": "interconnection"}
    supply = {}
    for carrier, target in mapped.items():
        supply[target] = supply.get(target, 0.0) + float(positive.loc[network.generators.carrier.eq(carrier)].sum())
    shedding = float(positive.loc[network.generators.carrier.eq("load_shedding")].sum())
    loads = float(network.loads_t.p_set.reindex(index=network.snapshots).fillna(0.0).sum().sum() / 1000.0)
    export_ids = network.loads.index[network.loads.carrier.eq("export")]
    exports = float(network.loads_t.p_set.loc[:, export_ids].sum().sum() / 1000.0) if len(export_ids) else 0.0
    lines = network.lines_t.p0.reindex(index=network.snapshots).fillna(0.0) + network.lines_t.p1.reindex(index=network.snapshots).fillna(0.0) if len(network.lines) else pd.DataFrame()
    line_losses = float(lines.clip(lower=0).sum().sum() / 1000.0) if len(lines) else 0.0
    return pd.Series(supply, dtype=float), {"load_shedding_gwh": shedding, "demand_gwh": loads, "imports_gwh": supply.get("interconnection", 0.0), "exports_gwh": exports, "network_losses_gwh_estimate": line_losses}


def validate_fixed_exports(network: pypsa.Network) -> dict:
    """Verify the retained fixed export sinks against the 2022 contract."""
    contract = load_validation_contract()
    expected = float(contract["supply_reference"]["reported_total_gwh"] - contract["supply_reference"]["reported_domestic_generation_gwh"] - contract["interconnection_colombia"]["historical_energy_gwh"] + contract["supply_reference"]["reported_total_gwh"] - contract["supply_reference"]["reported_total_gwh"])
    # The explicit country values are the authoritative export controls.
    expected = float(contract["exports_2022"]["colombia_gwh"] + contract["exports_2022"]["peru_gwh"])
    export_ids = network.loads.index[network.loads.carrier.eq("export")]
    actual = float(network.loads_t.p_set.loc[:, export_ids].sum().sum() / 1000.0) if len(export_ids) else 0.0
    if not np.isclose(actual, expected, atol=1e-6):
        raise ValueError(f"fixed export profile is {actual:.6f} GWh; expected {expected:.6f} GWh")
    return {"expected_exports_gwh": expected, "actual_exports_gwh": actual, "export_ids": list(export_ids)}


def compare_network(network: pypsa.Network, experiment_id: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    contract = load_validation_contract()
    raw = load_raw_supply_reference(ROOT / contract["supply_reference"]["raw_table"])
    reference = build_supply_reference(raw, contract)
    supply, extra = carrier_energy(network)
    rows, metrics = calculate_validation_metrics(supply, reference)
    rows.insert(0, "experiment_id", experiment_id)
    metrics.update(extra, experiment_id=experiment_id, snapshots=len(network.snapshots), solver_status=str(network.meta.get("solver_status", "unknown")))
    metrics["hydro_gwh"] = float(supply.get("hydro", 0.0))
    metrics["thermal_gwh"] = float(supply.get("thermal", 0.0))
    metrics["renewables_bio_gwh"] = float(supply.get("nonconventional", 0.0))
    return metrics, rows, reference


def write_plant_diagnostics(network: pypsa.Network) -> None:
    ids = hydro_ids(network)
    pmax = network.generators_t.p_max_pu.loc[:, ids].fillna(0.0)
    dispatch = network.generators_t.p.loc[:, ids].fillna(0.0)
    source = pd.read_csv(ROOT / "data/processed/generation/generators_2022.csv").set_index("plant_id")
    assignments = pd.read_csv(ROOT / "data/interim/generation/hydro_profile_assignments_2022.csv").set_index("plant_id")
    rows = []
    for plant in ids:
        offer = pmax[plant] * float(network.generators.loc[plant, "p_nom"])
        gen = dispatch[plant]
        rows.append({"plant_id": plant, "name": source.loc[plant, "name"], "p_nom_mw": network.generators.loc[plant, "p_nom"], "effective_capacity_mw": source.loc[plant, "effective_capacity_mw"], "caudal_id": assignments.loc[plant, "caudal_id"], "confidence": assignments.loc[plant, "confidence"], "mean_p_max_pu": pmax[plant].mean(), "available_gwh": offer.sum()/1000, "generation_gwh": gen.sum()/1000, "utilization": gen.sum()/offer.sum() if offer.sum() else np.nan, "hours_at_offer": int((gen >= offer - 1e-5).sum()), "potential_gap_gwh": max(0.0, (offer-gen).sum()/1000)})
    pd.DataFrame(rows).to_csv(INTERIM / "plant_hydro_diagnostics.csv", index=False)


def execute(run: bool = False, only: str | None = None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); INTERIM.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    registry = experiment_registry(); registry.to_csv(INTERIM / "experiment_registry.csv", index=False)
    baseline = pypsa.Network(BASELINE); write_plant_diagnostics(baseline); export_control = validate_fixed_exports(baseline)
    all_metrics, all_rows = [], []
    reuse = {"G06G_F06F_P45_LEGACY": "F06F_P45"}
    runnable = set(registry.loc[registry.experiment_id.str.startswith("G06G_") & ~registry.experiment_id.isin(reuse) & registry.controlled_change.ne("not testable") & ~registry.experiment_id.eq("G06G_NET") & ~registry.experiment_id.eq("G06G_ACCT"), "experiment_id"])
    if only is not None:
        runnable &= {only}
    for eid, old in reuse.items():
        src = F06F / old / "network_solved_2022.nc"; dst = RESULTS / eid / "network_solved_2022.nc"; dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() and src.exists(): shutil.copy2(src, dst)
        n = pypsa.Network(dst); m, rows, _ = compare_network(n, eid); m.update(reused_from=old, network_sha256=sha256(dst), source_network_sha256=sha256(src)); all_metrics.append(m); all_rows.append(rows)
    if run:
        for eid in sorted(runnable):
            dst = RESULTS / eid / "network_solved_2022.nc"; dst.parent.mkdir(parents=True, exist_ok=True)
            n = pypsa.Network(BASELINE); change = apply_change(n, eid); validate_fixed_exports(n)
            def extra_functionality(network, snapshots):
                enforce_historical_import_energy(network, snapshots, load_validation_contract())
            status, term = n.optimize(solver_name="highs", solver_options={"log_to_console": False}, extra_functionality=extra_functionality)
            n.meta = {"phase": "06G", "experiment_id": eid, "solver_status": str(status), "termination_condition": str(term)}
            n.export_to_netcdf(dst)
            (dst.parent / "run_metadata.json").write_text(json.dumps({"experiment_id": eid, **change, "solver_status": str(status), "termination_condition": str(term), "snapshots": len(n.snapshots), "network_sha256": sha256(dst)}, indent=2), encoding="utf-8")
            m, rows, _ = compare_network(n, eid); m.update(network_sha256=sha256(dst), solver_status=str(status), termination_condition=str(term), exports_controlled_gwh=export_control["expected_exports_gwh"]); all_metrics.append(m); all_rows.append(rows)
            del n
            gc.collect()
    # Re-index every completed annual run so interrupted batches remain resumable.
    indexed = {str(m.get("experiment_id")) for m in all_metrics}
    for meta_path in sorted(RESULTS.glob("G06G_*/run_metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        eid = str(meta["experiment_id"])
        if eid in indexed:
            continue
        network_path = meta_path.parent / "network_solved_2022.nc"
        if not network_path.exists() or int(meta.get("snapshots", 0)) != 8760:
            continue
        n = pypsa.Network(network_path); validate_fixed_exports(n)
        m, rows, _ = compare_network(n, eid); m.update(network_sha256=sha256(network_path), solver_status=meta.get("solver_status"), termination_condition=meta.get("termination_condition")); all_metrics.append(m); all_rows.append(rows)
    metrics = pd.DataFrame(all_metrics).drop_duplicates("experiment_id").sort_values("experiment_id")
    comparisons = pd.concat(all_rows, ignore_index=True).drop_duplicates(["experiment_id", "carrier"])
    metrics.to_csv(RESULTS / "experiment_summary.csv", index=False); comparisons.to_csv(RESULTS / "carrier_share_comparison.csv", index=False)
    registry.to_csv(RESULTS / "experiment_registry.csv", index=False)
    report = {"phase": "06G", "base_network": str(BASELINE.relative_to(ROOT)), "base_network_sha256": sha256(BASELINE), "reference_hydro_gwh": CENACE_HYDRO_GWH, "verified_reuse": reuse, "not_testable": registry.loc[registry.controlled_change.eq("not testable"), ["experiment_id", "hypothesis", "evidence"]].to_dict("records"), "runs": metrics.to_dict("records")}
    (RESULTS / "phase06g_manifest.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    sns.set_theme(style="whitegrid", context="talk")
    plot = metrics.merge(comparisons.groupby("experiment_id", as_index=False).agg(max_abs_pp=("absolute_error_pp", "max")), on="experiment_id")
    fig, ax = plt.subplots(figsize=(14, 7)); sns.barplot(data=plot, x="experiment_id", y="tvd", ax=ax, color="#2a9d8f"); ax.axhline(float(load_validation_contract()["acceptance"]["max_tvd"]), color="black", ls="--", label="TVD gate"); ax.set_ylabel("Total variation distance (share basis)"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=60); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "tvd_by_experiment.png", dpi=180); plt.close(fig)
    long = comparisons.melt(id_vars=["experiment_id", "carrier"], value_vars=["reference_share", "simulated_share"], var_name="basis", value_name="share")
    fig, ax = plt.subplots(figsize=(15, 8)); sns.barplot(data=long, x="carrier", y="share", hue="basis", ax=ax, errorbar=None); ax.set_ylabel("Share of comparable carrier universe"); ax.set_xlabel(""); fig.tight_layout(); fig.savefig(FIGURES / "carrier_shares_by_experiment.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(15, 8)); sns.barplot(data=comparisons, x="experiment_id", y="signed_error_pp", hue="carrier", ax=ax, errorbar=None); ax.axhline(0, color="black", lw=1); ax.set_ylabel("Signed share error (percentage points)"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=60); fig.tight_layout(); fig.savefig(FIGURES / "share_error_pp_by_experiment.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 7)); sns.scatterplot(data=metrics, x="hydro_gwh", y="thermal_gwh", hue="experiment_id", s=130, ax=ax); ax.axvline(CENACE_HYDRO_GWH, color="navy", ls="--", label="CENACE hydro"); ax.set_xlabel("Hydro generation (GWh)"); ax.set_ylabel("Thermal generation (GWh)"); fig.tight_layout(); fig.savefig(FIGURES / "hydro_thermal_tradeoff.png", dpi=180); plt.close(fig)
    energy = metrics.melt(id_vars="experiment_id", value_vars=["hydro_gwh", "thermal_gwh", "renewables_bio_gwh"], var_name="carrier", value_name="gwh")
    fig, ax = plt.subplots(figsize=(15, 8)); sns.barplot(data=energy, x="experiment_id", y="gwh", hue="carrier", ax=ax, errorbar=None); ax.set_ylabel("Annual generation (GWh)"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=60); fig.tight_layout(); fig.savefig(FIGURES / "annual_generation_by_experiment.png", dpi=180); plt.close(fig)
    heat = comparisons.pivot(index="carrier", columns="experiment_id", values="signed_error_pp")
    fig, ax = plt.subplots(figsize=(16, 5)); sns.heatmap(heat, annot=True, fmt=".2f", center=0, cmap="RdBu_r", ax=ax, cbar_kws={"label": "Share error (percentage points)"}); ax.set_xlabel(""); ax.set_ylabel(""); fig.tight_layout(); fig.savefig(FIGURES / "share_error_heatmap.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6)); sns.barplot(data=metrics, x="experiment_id", y="load_shedding_gwh", color="#d95f02", ax=ax); ax.axhline(0.01, color="black", ls="--", label="0.01 GWh screen"); ax.set_ylabel("Load shedding (GWh)"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=60); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "load_shedding_by_experiment.png", dpi=180); plt.close(fig)
    comparisons.to_csv(INTERIM / "carrier_share_comparison.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true"); parser.add_argument("--only"); args = parser.parse_args(); execute(run=args.run, only=args.only)
