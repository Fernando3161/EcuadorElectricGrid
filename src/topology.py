"""Deterministic Ecuador 2022 transmission-topology preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


SUPPORTED_VOLTAGES_KV = (138.0, 230.0, 500.0)
ECUADOR_VOLTAGES_KV = (48.0, 69.0, 138.0, 230.0, 500.0)


@dataclass(frozen=True)
class TopologyBuildResult:
    network: object
    mutation_audit: pd.DataFrame
    component_summary: pd.DataFrame
    fingerprint: dict[str, object]


def _nearest(values: pd.Series, choices: Iterable[float]) -> pd.Series:
    allowed = np.asarray(tuple(choices), dtype=float)
    raw = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    return pd.Series(allowed[np.abs(raw[:, None] - allowed).argmin(axis=1)], index=values.index)


def snap_nominal_voltages(network) -> pd.DataFrame:
    """Snap bus and line voltages to Ecuador levels and return an audit."""
    rows: list[dict[str, object]] = []
    for component, table in (("Bus", network.buses), ("Line", network.lines)):
        before = table["v_nom"].astype(float).copy()
        after = _nearest(before, ECUADOR_VOLTAGES_KV)
        table.loc[:, "v_nom"] = after
        for name in before.index[~np.isclose(before, after)]:
            rows.append(
                {
                    "mutation": "snap_voltage",
                    "component": component,
                    "name": str(name),
                    "before": float(before.loc[name]),
                    "after": float(after.loc[name]),
                    "rationale": "nearest declared Ecuador nominal voltage",
                    "evidence": "legacy topology method; reviewed Phase 5 boundary",
                }
            )
    return pd.DataFrame(rows)


def referenced_buses(network) -> set[str]:
    refs: set[str] = set()
    for list_name in (
        "lines", "transformers", "links", "loads", "generators",
        "storage_units", "stores", "shunt_impedances",
    ):
        frame = getattr(network, list_name, None)
        if frame is None or frame.empty:
            continue
        for column in (c for c in frame.columns if str(c).startswith("bus")):
            refs.update(frame[column].dropna().astype(str))
    return refs


def remove_unreferenced_buses(network) -> pd.DataFrame:
    orphans = sorted(set(network.buses.index.astype(str)) - referenced_buses(network))
    rows = []
    for bus in orphans:
        network.remove("Bus", bus)
        rows.append(
            {
                "mutation": "remove_orphan_bus",
                "component": "Bus",
                "name": bus,
                "before": "present",
                "after": "removed",
                "rationale": "no component references the bus in the upstream network",
                "evidence": "computed all-component reference scan",
            }
        )
    return pd.DataFrame(rows)


def prune_to_supported_voltage(network) -> pd.DataFrame:
    """Remove buses outside 138/230/500 kV through the PyPSA API."""
    keep = network.buses.index[network.buses.v_nom.isin(SUPPORTED_VOLTAGES_KV)]
    drop = network.buses.index.difference(keep)
    rows = []
    drop_set = set(map(str, drop))
    component_specs = (
        ("Line", network.lines, ("bus0", "bus1")),
        ("Transformer", network.transformers, ("bus0", "bus1")),
        ("Link", network.links, ("bus0", "bus1")),
        ("Load", network.loads, ("bus",)),
        ("Generator", network.generators, ("bus",)),
        ("StorageUnit", network.storage_units, ("bus",)),
        ("Store", network.stores, ("bus",)),
        ("ShuntImpedance", network.shunt_impedances, ("bus",)),
    )
    for component, table, columns in component_specs:
        if table.empty:
            continue
        mask = pd.Series(False, index=table.index)
        for column in columns:
            if column in table.columns:
                mask |= table[column].astype(str).isin(drop_set)
        for name in table.index[mask]:
            network.remove(component, name)
    for bus in drop:
        rows.append(
            {
                "mutation": "remove_out_of_scope_bus",
                "component": "Bus",
                "name": str(bus),
                "before": float(network.buses.at[bus, "v_nom"]),
                "after": "removed",
                "rationale": "supported transmission boundary is 138/230/500 kV",
                "evidence": "Phase 5 scope decision; legacy threshold was 137 kV",
            }
        )
        network.remove("Bus", bus)
    return pd.DataFrame(rows)


def load_repair_table(path: Path | str) -> pd.DataFrame:
    repairs = pd.read_csv(path, skipinitialspace=True)
    required = {
        "repair_id", "component", "action", "rationale", "evidence_source",
        "evidence_locator", "confidence", "official_fact",
    }
    missing = required - set(repairs.columns)
    if missing:
        raise ValueError(f"Topology repair table missing columns: {sorted(missing)}")
    if repairs.repair_id.duplicated().any():
        raise ValueError("Topology repair IDs must be unique")
    return repairs


def apply_reviewed_repairs(network, repairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    line_repairs = repairs.query("component == 'Line' and action == 'add'")
    for repair in line_repairs.itertuples(index=False):
        def bus_id(value) -> str:
            number = float(value)
            return str(int(number)) if number.is_integer() else str(value)

        bus0, bus1 = bus_id(repair.bus0), bus_id(repair.bus1)
        if bus0 not in network.buses.index or bus1 not in network.buses.index:
            raise ValueError(f"Repair {repair.repair_id} references a missing bus")
        v0, v1 = network.buses.loc[[bus0, bus1], "v_nom"].astype(float)
        if not np.isclose(v0, v1) or not np.isclose(v0, float(repair.v_nom_kv)):
            raise ValueError(f"Repair {repair.repair_id} has inconsistent endpoint voltage")
        if repair.repair_id in network.lines.index:
            raise ValueError(f"Repair line already exists: {repair.repair_id}")
        lon0, lat0 = network.buses.loc[bus0, ["x", "y"]].astype(float)
        lon1, lat1 = network.buses.loc[bus1, ["x", "y"]].astype(float)
        length = haversine_km(lat0, lon0, lat1, lon1)
        network.add(
            "Line",
            repair.repair_id,
            bus0=bus0,
            bus1=bus1,
            num_parallel=float(repair.num_parallel),
            type=str(repair.line_type),
            s_nom=float(repair.s_nom_mva),
            length=length,
            carrier="AC",
        )
        network.lines.loc[repair.repair_id, "v_nom"] = float(repair.v_nom_kv)
        rows.append(
            {
                "mutation": "add_reviewed_line",
                "component": "Line",
                "name": repair.repair_id,
                "before": "absent",
                "after": f"{bus0}-{bus1}",
                "rationale": repair.rationale,
                "evidence": f"{repair.evidence_source}; {repair.evidence_locator}",
            }
        )

    transformer_repairs = repairs.query(
        "component == 'Transformer' and action == 'patch_zero_resistance'"
    )
    if len(transformer_repairs) != 1:
        raise ValueError("Exactly one transformer resistance rule is required")
    zero_r = network.transformers.index[network.transformers.r.fillna(0).eq(0)]
    network.transformers.loc[zero_r, "r"] = 0.01
    rule = transformer_repairs.iloc[0]
    for name in zero_r:
        rows.append(
            {
                "mutation": "patch_transformer_resistance",
                "component": "Transformer",
                "name": str(name),
                "before": 0.0,
                "after": 0.01,
                "rationale": rule.rationale,
                "evidence": f"{rule.evidence_source}; {rule.evidence_locator}",
            }
        )
    network.calculate_dependent_values()
    return pd.DataFrame(rows)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1, phi2 = np.radians([lat1, lat2])
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def connected_component_summary(network) -> pd.DataFrame:
    graph = network.graph()
    rows = []
    for component_id, buses in enumerate(sorted(nx.connected_components(graph), key=lambda x: (-len(x), sorted(x)))):
        voltages = network.buses.loc[list(buses), "v_nom"]
        rows.append(
            {
                "component_id": component_id,
                "n_buses": len(buses),
                "min_voltage_kv": float(voltages.min()),
                "max_voltage_kv": float(voltages.max()),
                "buses": "|".join(sorted(map(str, buses))),
            }
        )
    return pd.DataFrame(rows)


def structural_fingerprint(network) -> dict[str, object]:
    return {
        "snapshots": int(len(network.snapshots)),
        "buses": int(len(network.buses)),
        "lines": int(len(network.lines)),
        "transformers": int(len(network.transformers)),
        "loads": int(len(network.loads)),
        "generators": int(len(network.generators)),
        "bus_voltage_counts": {
            str(float(k)): int(v) for k, v in network.buses.v_nom.value_counts().sort_index().items()
        },
        "connected_components": int(len(list(nx.connected_components(network.graph())))),
    }


def assert_topology_invariants(network) -> None:
    if not network.buses.index.is_unique or not network.lines.index.is_unique:
        raise AssertionError("Bus and line identifiers must be unique")
    if not set(network.buses.v_nom.astype(float)).issubset(SUPPORTED_VOLTAGES_KV):
        raise AssertionError("Network includes buses outside 138/230/500 kV")
    if network.buses[["x", "y"]].isna().any().any():
        raise AssertionError("Every retained bus needs geographic coordinates")
    for label, table in (("lines", network.lines), ("transformers", network.transformers)):
        missing = (set(table.bus0.astype(str)) | set(table.bus1.astype(str))) - set(network.buses.index.astype(str))
        if missing:
            raise AssertionError(f"{label} reference missing buses: {sorted(missing)}")
    line_v0 = network.lines.bus0.map(network.buses.v_nom)
    line_v1 = network.lines.bus1.map(network.buses.v_nom)
    if not np.allclose(line_v0, line_v1):
        raise AssertionError("A line joins buses at different nominal voltages")
    if (network.lines.length <= 0).any() or (network.lines.s_nom <= 0).any():
        raise AssertionError("Lines require positive length and rating")
    if (network.transformers.s_nom <= 0).any() or (network.transformers.r <= 0).any():
        raise AssertionError("Transformers require positive rating and resistance")
    if referenced_buses(network) != set(network.buses.index.astype(str)):
        raise AssertionError("Retained topology contains unreferenced buses")


def build_2022_topology(base_network, repairs: pd.DataFrame) -> TopologyBuildResult:
    network = base_network.copy()
    audits = [snap_nominal_voltages(network), remove_unreferenced_buses(network)]
    audits.append(prune_to_supported_voltage(network))
    audits.append(apply_reviewed_repairs(network, repairs))
    network.buses.loc[:, "carrier"] = "AC"
    network.lines.loc[:, "carrier"] = "AC"
    if "AC" not in network.carriers.index:
        network.add("Carrier", "AC", nice_name="Alternating current")
    assert_topology_invariants(network)
    return TopologyBuildResult(
        network=network,
        mutation_audit=pd.concat(audits, ignore_index=True),
        component_summary=connected_component_summary(network),
        fingerprint=structural_fingerprint(network),
    )
