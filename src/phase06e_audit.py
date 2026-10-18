"""Phase 06E documentary transmission audit and reporting utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "interim" / "topology_audit_2022"
FIG = ROOT / "reports" / "figures" / "phase_06e"
BASELINE = ROOT / "results" / "phase_06d" / "baseline" / "network_solved_2022.nc"
AFTER = ROOT / "results" / "phase_06d" / "renewable-calibration" / "network_solved_2022.nc"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def haversine_km(lon1, lat1, lon2, lat2):
    radius = 6371.0088
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def build_model_inventory(network_path: Path, component: str) -> pd.DataFrame:
    n = pypsa.Network(network_path)
    table = n.lines if component == "Line" else n.transformers
    rows = []
    for cid, row in table.iterrows():
        s_nom = float(row.get("s_nom", float("nan")))
        s_max = row.get("s_max_pu", float("nan"))
        s_max = float(s_max) if pd.notna(s_max) else 1.0
        rows.append(
            {
                "pypsa_component_id": str(cid),
                "bus0": _text(row.get("bus0")),
                "bus1": _text(row.get("bus1")),
                "bus0_voltage_kv": float(n.buses.at[row["bus0"], "v_nom"]),
                "bus1_voltage_kv": float(n.buses.at[row["bus1"], "v_nom"]),
                "bus0_longitude": float(n.buses.at[row["bus0"], "x"]),
                "bus0_latitude": float(n.buses.at[row["bus0"], "y"]),
                "bus1_longitude": float(n.buses.at[row["bus1"], "x"]),
                "bus1_latitude": float(n.buses.at[row["bus1"], "y"]),
                "s_nom_mva": s_nom,
                "s_max_pu": s_max,
                "effective_rating_mva": s_nom * s_max,
                "num_parallel": row.get("num_parallel", "NOT_DOCUMENTED"),
                "length_km": row.get("length", "NOT_DOCUMENTED"),
                "source_network": str(network_path.relative_to(ROOT)),
                "network_checksum": sha256(network_path),
                "inventory_status": "REVIEWED",
                "match_status": "PENDING_GEOGRAPHIC_RECONCILIATION",
            }
        )
    return pd.DataFrame(rows)


def geographic_reconciliation(network_path: Path) -> pd.DataFrame:
    """Create transparent geographic/name candidates from existing project evidence.

    Exact endpoint evidence is promoted to an official geographic match only when
    both modeled endpoint IDs and component type agree with the registered evidence.
    Master-plan substations are reported as name/coordinate candidates only; they
    are not treated as 2022 official assets.
    """
    n = pypsa.Network(network_path)
    evidence_path = ROOT / "data/interim/network/transmission_evidence_2022.csv"
    evidence = pd.read_csv(evidence_path) if evidence_path.exists() else pd.DataFrame()
    master_path = ROOT / "data/raw/networks/EC_bus_expansion_masterplan_2022.csv"
    master = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame()
    rows = []
    for component, table in [("Line", n.lines), ("Transformer", n.transformers)]:
        for cid, item in table.iterrows():
            b0, b1 = str(item.bus0), str(item.bus1)
            p0 = n.buses.loc[item.bus0]; p1 = n.buses.loc[item.bus1]
            endpoint_distance = np.nan
            exact = evidence[
                (evidence.component == component)
                & evidence.bus0.astype(str).isin([b0, b1])
                & evidence.bus1.astype(str).isin([b0, b1])
            ] if not evidence.empty else pd.DataFrame()
            if not exact.empty:
                ev = exact.iloc[0]
                status, method = "MATCHED_OFFICIAL_GEOGRAPHIC", "exact_endpoint_evidence"
                official_name = str(ev.evidence_id)
                source_id, locator = ev.source_document, ev.source_locator
                rating, source_length = ev.rating_mva, ev.length_km
                rating_delta = float(item.s_nom) - float(rating)
                length_delta = float(item.length) - float(source_length)
                confidence = ev.confidence
            else:
                status, method = "GEOGRAPHIC_CANDIDATE_ONLY", "nearest_named_substations"
                source_id, locator = "MASTERPLAN_2022", "EC_bus_expansion_masterplan_2022.csv"
                rating, source_length, rating_delta, length_delta = np.nan, np.nan, np.nan, np.nan
                confidence, official_name = "low", ""
                if not master.empty:
                    allowed = master[master.v_nom.isin([float(n.buses.loc[item.bus0].v_nom), float(n.buses.loc[item.bus1].v_nom)])].copy()
                    if not allowed.empty:
                        allowed["d0"] = allowed.apply(lambda r: haversine_km(p0.x, p0.y, r.lon, r.lat), axis=1)
                        allowed["d1"] = allowed.apply(lambda r: haversine_km(p1.x, p1.y, r.lon, r.lat), axis=1)
                        a, b = allowed.loc[allowed.d0.idxmin()], allowed.loc[allowed.d1.idxmin()]
                        official_name = f"{a.Name} -> {b.Name}"
                        endpoint_distance = float(a.d0 + b.d1)
                    else:
                        endpoint_distance = np.nan
                else:
                    endpoint_distance = np.nan
            rows.append({
                "component_type": component, "pypsa_component_id": str(cid),
                "bus0": b0, "bus1": b1, "bus0_voltage_kv": n.buses.loc[item.bus0].v_nom,
                "bus1_voltage_kv": n.buses.loc[item.bus1].v_nom,
                "bus0_longitude": p0.x, "bus0_latitude": p0.y,
                "bus1_longitude": p1.x, "bus1_latitude": p1.y,
                "modeled_length_km": item.get("length", np.nan), "modeled_s_nom_mva": item.get("s_nom", np.nan),
                "candidate_official_name": official_name, "match_method": method,
                "match_status": status, "confidence": confidence,
                "source_id": source_id, "source_locator": locator,
                "source_rating_mva": rating, "rating_delta_mva": rating_delta,
                "source_length_km": source_length, "length_delta_km": length_delta,
                "endpoint_distance_km": endpoint_distance,
                "interpretation": "Verified endpoint match from project evidence" if status.startswith("MATCHED") else "Candidate only; master-plan name/coordinates are not proof of 2022 asset identity",
            })
    return pd.DataFrame(rows)


def line_capacity_audit(network_path: Path) -> pd.DataFrame:
    """Evaluate every modeled line and record an explicit capacity decision."""
    n = pypsa.Network(network_path)
    evidence = pd.read_csv(ROOT / "data/interim/network/transmission_evidence_2022.csv")
    repairs = pd.read_csv(ROOT / "data/interim/network/topology_repairs_2022.csv")
    rows = []
    for cid, item in n.lines.iterrows():
        cid = str(cid); b0, b1 = str(item.bus0), str(item.bus1)
        endpoint = evidence[
            evidence.component.eq("Line")
            & evidence.bus0.astype(str).isin([b0, b1])
            & evidence.bus1.astype(str).isin([b0, b1])
        ]
        repair = repairs[repairs.source_component_id.apply(lambda x: str(int(float(x))) if pd.notna(x) else "").eq(cid)] if "source_component_id" in repairs else pd.DataFrame()
        if repair.empty:
            repair = repairs[
                repairs.component.eq("Line")
                & repairs.bus0.astype(str).isin([b0, b1])
                & repairs.bus1.astype(str).isin([b0, b1])
                & repairs.action.isin(["update", "add"])
            ]
        record = endpoint.iloc[0] if not endpoint.empty else (repair.iloc[0] if not repair.empty else None)
        if record is not None:
            source_rating = float(record.rating_mva if "rating_mva" in record.index and pd.notna(record.rating_mva) else record.s_nom_mva)
            delta = float(item.s_nom) - source_rating
            official = str(record.official_fact).lower() == "true" if "official_fact" in record.index else False
            source_id = record.source_document if "source_document" in record.index else record.evidence_source
            locator = record.source_locator if "source_locator" in record.index else record.evidence_locator
            basis = "exact endpoint documentary match"
            decision = "RETAIN" if abs(delta) < 1e-9 else ("CORRECT" if official else "REVIEW_REQUIRED")
        else:
            source_rating, delta, source_id, locator = np.nan, np.nan, "GEOGRAPHIC_CANDIDATE_ONLY", "endpoint coordinates and named-substation proximity"
            official, basis, decision = False, "geographic/name candidate; no component rating found", "RETAIN_PENDING_OFFICIAL_RATING"
        rows.append({
            "pypsa_component_id": cid, "bus0": b0, "bus1": b1,
            "voltage_kv": item.v_nom, "modeled_length_km": item.length,
            "modeled_capacity_mva": item.s_nom, "documented_capacity_mva": source_rating,
            "capacity_delta_mva": delta, "documented_source": source_id,
            "source_locator": locator, "official_fact": official,
            "evaluation_basis": basis, "capacity_decision": decision,
            "correction_required": decision == "CORRECT",
        })
    return pd.DataFrame(rows)


def official_inventory(kind: str) -> pd.DataFrame:
    source = "CENACE_ANNUAL_2022" if kind == "line" else "CENACE_ANNUAL_2022"
    filename = "cenace_informe_anual_2022_parte_1.pdf"
    page = "73-74" if kind == "line" else "73-74"
    rows = [
        {
            "official_asset_id": "OFFICIAL_DOCUMENTED_ASSET_INVENTORY_NOT_FOUND",
            "official_name": "NOT_DOCUMENTED_IN_SOURCE_TABLE",
            "from_substation": "NOT_DOCUMENTED",
            "to_substation": "NOT_DOCUMENTED",
            "voltage_kv": "NOT_DOCUMENTED",
            "capacity_value": "NOT_DOCUMENTED",
            "capacity_unit": "NOT_DOCUMENTED",
            "circuit_count": "NOT_DOCUMENTED",
            "status_2022": "NOT_DOCUMENTED",
            "asset_scope": "UNMATCHED_DOCUMENTARY_SCOPE",
            "source_id": source,
            "source_filename": filename,
            "source_page": page,
            "source_table": "Table 2.4 reports generation, not a line/transformer inventory",
            "original_value_text": "No individual transmission rating is provided",
            "conversion_formula": "NOT_APPLICABLE",
            "evidence_quality": "INSUFFICIENT_FOR_COMPONENT_RATING",
            "notes": "Do not infer an asset rating from generation statistics.",
        },
    ]
    return pd.DataFrame(rows)


def source_register() -> pd.DataFrame:
    sources = [
        ("CENACE_ANNUAL_2022", "literature/CENACE/cenace_informe_anual_2022_parte_1.pdf", "88f13f0106a190464c8954babc7529f785783d4d4017633a8d5ed8ff554fbd88", "CENACE Informe Anual 2022 Parte 1", "Official generation statistics; Table 2.4 pp. 73-74; no component rating inventory."),
        ("CELEC_SNT_MAP_2022", "literature/CELEC_TRANSELECTRIC/mapa_snt_diciembre_2022.pdf", "755e8811f37686e2eee39373d3069d06217c9c58bf7fc8da9ef6779764e8603b", "Mapa del Sistema Nacional de Transmisión SNT December 2022", "Official system map; page 1 gives aggregate circuit lengths and 66 substations, not individual MVA ratings."),
        ("CENACE_STATISTICS_CSV", "data/raw/statistics/cenace_2022_net_supply.csv", "1b9b3dbe3edb9df9100deaec80303a8b4dc080363284f81731ffd7839afc7ee8", "Source-faithful CENACE Table 2.4 transcription", "Carrier reference used for validation; not transmission technical evidence."),
        ("PHASE_06D_BASELINE", "results/phase_06d/baseline/network_solved_2022.nc", sha256(BASELINE), "Phase 06D unrestricted transmission-constrained baseline", "Before network and scientific comparison baseline."),
        ("PHASE_06D_RENEWABLE_CANDIDATE", "results/phase_06d/renewable-calibration/network_solved_2022.nc", sha256(AFTER), "Phase 06D renewable-calibration candidate", "Diagnostic after artifact; not a Phase 06E topology correction."),
    ]
    return pd.DataFrame(sources, columns=["source_id", "source_filename", "sha256", "title", "coverage_and_limitations"])


def validation_comparison() -> pd.DataFrame:
    refs = pd.read_csv(ROOT / "data/raw/statistics/cenace_2022_net_supply.csv")
    mapping = {"hydroelectric": "hydro", "diesel": "thermal", "fuel_oil": "thermal", "natural_gas": "thermal", "naphtha": "thermal", "residue": "thermal", "wind": "nonconventional", "biogas": "nonconventional", "biomass": "nonconventional", "solar_photovoltaic": "nonconventional", "imports_colombia": "interconnection", "imports_peru": "interconnection"}
    ref = refs.assign(carrier=refs.source_carrier.map(mapping)).groupby("carrier", as_index=False).reference_gwh.sum()
    rows = []
    for label, path in [("before_phase06d_baseline", ROOT / "results/phase_06d/baseline/validation_2022.csv"), ("after_phase06d_renewable_candidate", ROOT / "results/phase_06d/renewable-calibration/validation_2022.csv")]:
        sim = pd.read_csv(path).rename(columns={"simulated_gwh": "simulated_gwh"})[["carrier", "simulated_gwh"]]
        merged = ref.merge(sim, on="carrier", how="outer").fillna(0)
        merged["stage"] = label
        merged["signed_error_gwh"] = merged.simulated_gwh - merged.reference_gwh
        merged["absolute_error_gwh"] = merged.signed_error_gwh.abs()
        merged["relative_error"] = merged.signed_error_gwh / merged.reference_gwh.replace(0, pd.NA)
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def write_audit() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    lines = build_model_inventory(AFTER, "Line")
    trafos = build_model_inventory(AFTER, "Transformer")
    geographic = geographic_reconciliation(AFTER)
    line_audit = line_capacity_audit(AFTER)
    lines = lines.merge(geographic[geographic.component_type.eq("Line")][["pypsa_component_id", "match_status"]], on=["pypsa_component_id"], how="left", suffixes=("", "_geographic"))
    trafos = trafos.merge(geographic[geographic.component_type.eq("Transformer")][["pypsa_component_id", "match_status"]], on=["pypsa_component_id"], how="left", suffixes=("", "_geographic"))
    lines["match_status"] = lines.pop("match_status_geographic")
    trafos["match_status"] = trafos.pop("match_status_geographic")
    official_lines = official_inventory("line")
    official_trafos = official_inventory("transformer")
    lines.to_csv(OUT / "lines_audit.csv", index=False)
    trafos.to_csv(OUT / "transformers_audit.csv", index=False)
    geographic.to_csv(OUT / "geographic_match_candidates.csv", index=False)
    line_audit.to_csv(OUT / "line_capacity_audit.csv", index=False)
    official_lines.to_csv(OUT / "official_lines_inventory.csv", index=False)
    official_trafos.to_csv(OUT / "official_transformers_inventory.csv", index=False)
    corrections = line_audit[line_audit.correction_required].copy()
    corrections = corrections.assign(
        correction_id=lambda x: "capacity_" + x.pypsa_component_id,
        component_type="Line", original_attribute="s_nom",
        original_value=corrections.modeled_capacity_mva,
        proposed_value=corrections.documented_capacity_mva, unit="MVA",
        correction_type="documented_capacity_replacement",
        source_id=corrections.documented_source, source_page=corrections.source_locator,
        source_table="transmission_evidence_2022", conversion_formula="direct documented MVA",
        match_status="MATCHED_OFFICIAL_GEOGRAPHIC", confidence="high",
        rationale="Documented line capacity differs from modeled capacity",
        owner_review_status="pending", implementation_note="Apply only after owner review",
    )
    correction_columns = ["correction_id", "pypsa_component_id", "component_type", "original_attribute", "original_value", "proposed_value", "unit", "correction_type", "source_id", "source_page", "source_table", "conversion_formula", "match_status", "confidence", "rationale", "owner_review_status", "implementation_note"]
    corrections[correction_columns].to_csv(OUT / "capacity_corrections.csv", index=False)
    source_register().to_csv(OUT / "source_register.csv", index=False)
    comparison = validation_comparison()
    comparison.to_csv(OUT / "generation_before_after_cenace.csv", index=False)
    pivot = comparison.pivot(index="carrier", columns="stage", values="simulated_gwh").join(ref:=comparison[comparison.stage.eq("before_phase06d_baseline")].set_index("carrier")["reference_gwh"])
    ax = pivot[["reference_gwh", "before_phase06d_baseline", "after_phase06d_renewable_candidate"]].plot.bar(figsize=(11, 6), color=["#555555", "#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Annual energy (GWh)")
    ax.set_title("Phase 06D generation by carrier versus CENACE")
    ax.legend(["CENACE reference", "06D baseline", "06D renewable candidate"])
    ax.figure.tight_layout(); ax.figure.savefig(FIG / "generation_before_after_cenace.png", dpi=180); plt.close(ax.figure)
    err = comparison.copy(); err["absolute_percent_error"] = 100 * err.absolute_error_gwh / err.reference_gwh.replace(0, pd.NA)
    ax = err.pivot(index="carrier", columns="stage", values="absolute_percent_error").plot.bar(figsize=(11, 6), color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Absolute error (%) — diagnostic only"); ax.set_title("Absolute carrier error before and after (diagnostic)"); ax.legend(); ax.figure.tight_layout(); ax.figure.savefig(FIG / "carrier_error_before_after.png", dpi=180); plt.close(ax.figure)
    manifest = {"lines": len(lines), "transformers": len(trafos), "line_audit_unique": int(lines.pypsa_component_id.nunique()), "transformer_audit_unique": int(trafos.pypsa_component_id.nunique()), "after_network": str(AFTER.relative_to(ROOT)), "after_network_sha256": sha256(AFTER)}
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(write_audit(), indent=2))
