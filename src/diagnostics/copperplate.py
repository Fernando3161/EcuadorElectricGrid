"""Bounded transmission-relaxation diagnostic; never an annual validation run.

Only domestic AC buses are merged. Colombia's reviewed external bus and Links
retain their capacities/efficiencies. Disconnected domestic islands are joined
deliberately: this is an optimistic transmission relaxation, not a grid repair.
Generator/load IDs, carriers, signs, costs and hourly profiles are retained.
Storage, investment and global constraints are refused pending period semantics.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.phase06d import apply_monthly_thermal_derating

MAX_HOURS = 168
FULL_YEAR_HOURS = 8760
EXTERNAL_BUS = "interconnection_colombia"


def period_snapshots(network, start, hours: int = 24, *, allow_full_year: bool = False) -> pd.DatetimeIndex:
    """Select consecutive actual hours, without extrapolation or reweighting."""
    maximum = FULL_YEAR_HOURS if allow_full_year else MAX_HOURS
    if not 1 <= hours <= maximum:
        raise ValueError(f"hours must be between 1 and {maximum}")
    selected = pd.date_range(pd.Timestamp(start), periods=hours, freq="h")
    if not selected.isin(network.snapshots).all():
        raise ValueError("Requested consecutive hourly snapshots are unavailable")
    weights = network.snapshot_weightings.loc[selected]
    if not np.isfinite(weights.to_numpy()).all() or not weights.eq(1.0).all().all():
        raise ValueError("Diagnostic requires unit hourly snapshot weights")
    return selected


def build_copperplate(network, start, hours: int = 24, *, allow_full_year: bool = False):
    """Return an independent period network and original-to-diagnostic bus map.

    This dispatch-only diagnostic uses capacity-only imports: the annual
    historical-import callback is deliberately NOT applied or prorated.
    Existing global/energy constraints are rejected, never silently removed.
    """
    selected = period_snapshots(network, start, hours, allow_full_year=allow_full_year)
    for name in ("stores", "storage_units", "shunt_impedances", "global_constraints"):
        if not getattr(network, name).empty:
            raise ValueError(f"Unsupported {name}: explicit period semantics required")
    for name in ("generators", "links", "lines", "transformers"):
        table = getattr(network, name)
        for column in ("p_nom_extendable", "s_nom_extendable", "committable"):
            if column in table and table[column].any():
                raise ValueError(f"Unsupported {name}.{column}")
    for column in ("e_sum_min", "e_sum_max"):
        if column in network.generators:
            values = network.generators[column]
            active = values.ne(0) if column.endswith("min") else np.isfinite(values)
            if active.any():
                raise ValueError(f"Unsupported generator {column}: period budget required")
    if network.buses.empty or not network.buses.carrier.eq("AC").all():
        raise ValueError("Expected a nonempty AC-only network")
    domestic = network.buses.index.difference([EXTERNAL_BUS], sort=False)
    if domestic.empty:
        raise ValueError("No domestic AC buses")
    # Only the reviewed external import links have defined retained semantics.
    links = network.links
    if not links.empty:
        valid = (links.bus0.eq(EXTERNAL_BUS) & links.bus1.isin(domestic)
                 & links.carrier.eq("interconnection_colombia"))
        extra_ports = [c for c in links if c.startswith("bus") and c not in ("bus0", "bus1")]
        if not valid.all() or any(links[c].ne("").any() for c in extra_ports):
            raise ValueError("Only reviewed two-port Colombia import Links are supported")
    for component in ("lines", "transformers"):
        table = getattr(network, component)
        if not (table.bus0.isin(domestic) & table.bus1.isin(domestic)).all():
            raise ValueError("Passive branches must have domestic endpoints")
    anchor = domestic[0]
    mapping = pd.Series(network.buses.index, index=network.buses.index, name="diagnostic_bus")
    mapping.loc[domestic] = anchor
    mapping.index.name = "original_bus"
    result = network.copy(snapshots=selected)
    result.meta = copy.deepcopy(network.meta)
    for table in (result.generators, result.loads):
        table.loc[:, "bus"] = table.bus.map(mapping)
    for column in ("bus0", "bus1"):
        result.links.loc[:, column] = result.links[column].map(mapping)
    result.mremove("Line", result.lines.index)
    result.mremove("Transformer", result.transformers.index)
    result.mremove("Bus", domestic[1:])
    result.meta["copperplate_diagnostic"] = {
        "status": "diagnostic_only_not_annual_validation",
        "hours": hours,
        "import_policy": "capacity-only; annual historical energy band not applied",
        "relaxation": "all domestic AC buses including islands merged; passive branches removed",
    }
    return result, mapping


def solve_period(network, *, log_path=None, allow_full_year: bool = False,
                 time_limit: float | None = None,
                 enforce_import_cap: bool = False,
                 validation_contract=None) -> dict:
    """Solve a bounded prepared copy; return signed physical MWh accounting.

    Positive generator supply and positive load withdrawal follow PyPSA signs.
    Link loss = p0 + p1. Generator/load signs are included explicitly (including
    the legacy load-shedding convention). Raw carriers are never remapped.
    """
    if "copperplate_diagnostic" not in network.meta:
        raise ValueError("Prepare a diagnostic copy first")
    period_snapshots(network, network.snapshots[0], len(network.snapshots), allow_full_year=allow_full_year)
    options = {"solver_name": "highs", "solver_options": {"threads": 1}}
    if time_limit is not None:
        options["solver_options"]["time_limit"] = float(time_limit)
    if enforce_import_cap:
        from src.validation import enforce_historical_import_energy, load_validation_contract
        contract = validation_contract or load_validation_contract(Path("config/validation_2022.yaml"))
        options["extra_functionality"] = lambda n, snapshots: enforce_historical_import_energy(
            n, snapshots, contract
        )
    if log_path is not None:
        options["log_fn"] = str(log_path)
    try:
        status, termination = network.optimize(**options)
    finally:
        # HiGHS keeps its log handle open on Windows while solver_model lives.
        model = getattr(network, "model", None)
        solver = getattr(model, "solver_model", None)
        if log_path is not None and solver is not None:
            solver.setOptionValue("log_file", "")
    report = {"solver_status": status, "termination_condition": termination,
              "annual_comparable": False, "hours": len(network.snapshots),
              "import_policy": ("500 MW link capacity plus historical 2022 annual energy band"
                                 if enforce_import_cap else
                                 "capacity-only; annual historical energy band not applied"),
              "import_cap_enforced": bool(enforce_import_cap)}
    if (status, termination) != ("ok", "optimal"):
        return report
    supply = network.generators_t.p.mul(network.generators.sign, axis=1).sum()
    demand = network.loads_t.p.mul(-network.loads.sign, axis=1).sum()
    losses = float((network.links_t.p0 + network.links_t.p1).sum().sum())
    by_carrier = supply.groupby(network.generators.carrier).sum()
    report.update({"generator_supply_mwh_by_raw_carrier": by_carrier.to_dict(),
                   "load_withdrawal_mwh_by_raw_carrier": demand.groupby(network.loads.carrier).sum().to_dict(),
                   "link_losses_mwh": losses,
                   "balance_residual_mwh": float(supply.sum() - demand.sum() - losses)})
    return report


def prepare_case(network, start, hours=24, *, copperplate=False, thermal_restricted=False, allow_full_year=False):
    """Prepare one matched short-period reference or copperplate case."""
    if copperplate:
        result, mapping = build_copperplate(network, start, hours, allow_full_year=allow_full_year)
    else:
        selected = period_snapshots(network, start, hours, allow_full_year=allow_full_year)
        result = network.copy(snapshots=selected)
        mapping = pd.Series(network.buses.index, index=network.buses.index, name="diagnostic_bus")
        mapping.index.name = "original_bus"
        result.meta = copy.deepcopy(network.meta)
        result.meta["copperplate_diagnostic"] = {
            "status": "diagnostic_only_not_annual_validation",
            "hours": hours,
            "relaxation": "none; original internal lines and transformers retained",
        }
    if thermal_restricted:
        apply_monthly_thermal_derating(result, Path("data/interim/availability/thermal_availability_evidence_2022.csv"))
    result.meta["copperplate_diagnostic"]["thermal_policy"] = "aggregate proportional derating" if thermal_restricted else "unrestricted thermal availability"
    return result, mapping


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--period-import-policy", choices=["capacity-only"], required=True,
                        help="Explicit diagnostic assumption; no annual energy band")
    args = parser.parse_args(argv)
    if not 1 <= args.hours <= MAX_HOURS:
        parser.error(f"hours must be between 1 and {MAX_HOURS}")
    from src.preflight import ROOT, main as preflight
    from src.artifacts import sha256_file

    # Offline only: never download, replace, or repair any authoritative input.
    code = preflight(["--stage", "06_validation", "--offline"])
    if code:
        return code
    import pypsa

    source = args.network.resolve()
    digest = sha256_file(source)
    original = pypsa.Network(source)
    diagnostic, mapping = build_copperplate(original, args.start, args.hours)
    # Unique directory: no overwriting inputs or previous diagnostic evidence.
    from tempfile import mkdtemp
    parent = ROOT / "results" / "diagnostics"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="copperplate-", dir=parent))
    mapping.to_csv(output / "bus_mapping.csv")
    report = {"source": str(source), "source_sha256": digest,
              "source_metadata": original.meta,
              "manifest_sha256": sha256_file(ROOT / "config" / "artifacts.yaml"),
              "diagnostic_code_sha256": sha256_file(Path(__file__)),
              "pypsa_version": pypsa.__version__,
              "diagnostic": diagnostic.meta["copperplate_diagnostic"],
              "start": str(diagnostic.snapshots[0]), "end": str(diagnostic.snapshots[-1])}
    try:
        report.update(solve_period(diagnostic, log_path=output / "highs.log"))
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        report["source_unchanged"] = sha256_file(source) == digest
        (output / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(output)
    return 0 if report.get("termination_condition") == "optimal" and report["source_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
