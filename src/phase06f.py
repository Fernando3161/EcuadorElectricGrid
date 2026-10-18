"""Reproducible Phase 06F hydro diagnostics and controlled experiments.

The module reads the retained Phase 06D solved network as evidence, never as a
write target.  Diagnostics are independent of optimization; experiments clone
the network into an experiment-specific directory before changing one offer
assumption and solving the complete snapshot set.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/phase_06d/baseline/network_solved_2022.nc"
OUT = ROOT / "data/interim/hydro_evaluation_2022"
RESULTS = ROOT / "results/phase_06f"
CENACE = 24499.54

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def hydro_ids(n):
    return n.generators.index[n.generators.carrier.eq("hydro")]

def availability(n, ids):
    p = n.generators_t.p_max_pu.reindex(index=n.snapshots, columns=ids).fillna(
        n.generators.loc[ids, "p_max_pu"].fillna(1.0)
    ).clip(0, 1)
    return p.mul(n.generators.loc[ids, "p_nom"], axis=1)

def write_lineage(n):
    ids = hydro_ids(n); gen = n.generators.loc[ids]
    assign_path = ROOT / "data/interim/generation/hydro_profile_assignments_2022.csv"
    assign = pd.read_csv(assign_path) if assign_path.exists() else pd.DataFrame()
    src = pd.read_csv(ROOT / "data/processed/generation/generators_2022.csv")
    src = src.set_index("plant_id") if "plant_id" in src else pd.DataFrame()
    rows=[]
    for gid, r in gen.iterrows():
        a = assign[assign.plant_id.eq(gid)].iloc[0].to_dict() if not assign.empty and (assign.plant_id == gid).any() else {}
        s = src.loc[gid].to_dict() if gid in src.index else {}
        rows.append({"generator_id":gid,"original_source_name":s.get("name","NOT_DOCUMENTED"),"technology":"hydro",
          "p_nom_mw":r.p_nom,"bus":r.bus,"bus_voltage_kv":n.buses.loc[r.bus,"v_nom"],"latitude":n.buses.loc[r.bus,"y"],"longitude":n.buses.loc[r.bus,"x"],
          "commissioning_year":s.get("commissioning_year","NOT_DOCUMENTED"),"commissioning_mask":"hourly offer is the retained Phase 06D offer",
          "flow_source":a.get("caudal_id","NOT_DOCUMENTED"),"plant_factor_source":"data/processed/generation/hidro_max_profiles/p_max_pu_hydro_2022_hourly.csv",
          "profile_transformation":"Phase 04 hydro profile pipeline; no Phase 06F transformation","annual_mean_p_max_pu":np.nan,
          "marginal_cost_usd_per_mwh":r.marginal_cost,"source_document":s.get("source_document_id","NOT_DOCUMENTED"),"source_locator":s.get("source_locator","NOT_DOCUMENTED"),
          "modeling_assumption":"yes","assignment_basis":a.get("assignment_basis","NOT_DOCUMENTED"),"confidence":a.get("confidence","NOT_DOCUMENTED")})
    out=pd.DataFrame(rows); OUT.mkdir(parents=True,exist_ok=True); out.to_csv(OUT/"hydro_input_lineage.csv",index=False); return out

def write_offer(n):
    ids=hydro_ids(n); av=availability(n,ids); gen=n.generators.loc[ids]
    rows=[]
    for gid in ids:
        x=av[gid]; pu=x/gen.loc[gid,"p_nom"]
        rows.append({"generator_id":gid,"p_nom_mw":gen.loc[gid,"p_nom"],"min_p_max_pu":pu.min(),"mean_p_max_pu":pu.mean(),"max_p_max_pu":pu.max(),
          "annual_available_gwh":x.sum()/1000,"hours_zero":int((pu==0).sum()),"hours_one":int((pu==1).sum()),"hours_close_to_one":int((pu>=.99).sum()),
          "missing_values":int(pu.isna().sum()),"clipping_events":int((pu>=1).sum()),"profile_hash":hashlib.sha256(np.asarray(np.round(pu,8)).tobytes()).hexdigest()})
    out=pd.DataFrame(rows); out.to_csv(OUT/"hydro_offer_diagnostics.csv",index=False); return out, av

def write_attribution(n,av):
    ids=hydro_ids(n); dispatch=n.generators_t.p.reindex(index=n.snapshots,columns=ids).fillna(0); rows=[]
    for gid in ids:
        a=av[gid]; d=dispatch[gid]; near=d.ge(a-1e-5); util=d.sum()/a.sum() if a.sum() else np.nan
        rows.append({"generator_id":gid,"annual_generation_gwh":d.sum()/1000,"annual_available_gwh":a.sum()/1000,"available_utilization":util,
          "hours_at_available_limit":int(near.sum()),"hours_above_99pct_available":int(d.ge(a*.99-1e-8).sum()),"hours_below_available":int((~near).sum()),
          "potential_curtailment_gwh":(a-d.clip(lower=0)).sum()/1000,"max_dispatch_mw":d.max(),"dispatch_capacity_factor":d.sum()/(len(d)*n.generators.loc[gid,"p_nom"]),
          "classification":"resource-limited" if util<.9 else "power-saturated"})
    out=pd.DataFrame(rows); out["gap_contribution_gwh"]=(out.annual_available_gwh-out.annual_generation_gwh).clip(lower=0); out.to_csv(OUT/"hydro_dispatch_attribution.csv",index=False); return out

def registry():
    rows=[]
    for eid, hyp, change, value in [("F06F_E00_BASELINE","baseline","none","retained Phase 06D network"),("F06F_P20","merit order","hydro marginal cost",20),("F06F_P25","merit order","hydro marginal cost",25),("F06F_P30","merit order","hydro marginal cost",30),("F06F_P35","merit order","hydro marginal cost",35),("F06F_P40","merit order","hydro marginal cost",40),("F06F_P45","merit order","hydro marginal cost",45),("F06F_P00","merit order diagnostic","hydro marginal cost",0),("F06F_A105","availability","p_max_pu multiplier",1.05),("F06F_A110","availability","p_max_pu multiplier",1.10),("F06F_A120","availability","p_max_pu multiplier",1.20),("F06F_A130","availability","p_max_pu multiplier",1.30),("F06F_A100","availability relaxation","p_max_pu",1.0)]:
        rows.append({"experiment_id":eid,"hypothesis":hyp,"changed_parameter":change,"experimental_value":value,"network_mode":"baseline topology","full_year":True,"diagnostic_only":eid not in {"F06F_E00_BASELINE","F06F_P45"},"expected_interpretation":"separate resource/economic effect; not a correction"})
    out=pd.DataFrame(rows); out.to_csv(OUT/"experiment_registry.csv",index=False); return out

def write_evidence_and_corrections():
    pd.DataFrame([
        {"evidence_id":"E06F_01","topic":"annual hydro reference","value":CENACE,"unit":"GWh","source_document":"literature/CENACE/cenace_informe_anual_2022_parte_1.pdf","source_locator":"Table 2.4, pp. 73-74","status":"official fact","limitation":"system boundary may differ from model"},
        {"evidence_id":"E06F_02","topic":"hydro offers","value":"65 plants; hourly p_max_pu","unit":"count / pu","source_document":"data/processed/generation/hidro_max_profiles/p_max_pu_hydro_2022_hourly.csv","source_locator":"hourly profile table","status":"model input","limitation":"profile lineage includes regional proxies"},
        {"evidence_id":"E06F_03","topic":"flow proxies","value":"8 monthly legacy series","unit":"series","source_document":"data/raw/generation/caudals.csv","source_locator":"header and rows","status":"model input","limitation":"primary provenance unresolved"},
        {"evidence_id":"E06F_04","topic":"hydro price","value":45,"unit":"USD/MWh","source_document":"Phase 06D baseline network","source_locator":"generators.marginal_cost for carrier hydro","status":"model assumption","limitation":"uniform price"},
    ]).to_csv(OUT/"hydro_reference_evidence.csv",index=False)
    pd.DataFrame([{"correction_id":"C06F_001","scope":"fleet","original_value":"Phase 06D p_max_pu","proposed_value":"none","unit":"pu","transformation":"diagnostic scaling is not promoted","source_locator":"F06F_A105-A130 results","confidence":"not a correction","owner_review_status":"pending owner decision","reason":"A120 matches annual reference but is an unsupported calibration multiplier"}]).to_csv(OUT/"hydro_corrections.csv",index=False)

def write_summary_and_figures():
    rows=[json.loads(p.read_text()) for p in sorted(RESULTS.glob("F06F_*/summary.json"))]
    summary=pd.DataFrame(rows); summary.to_csv(RESULTS/"experiment_summary.csv",index=False)
    if summary.empty: return
    fig=ROOT/"reports/figures/phase_06f"; fig.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(8,4.5)); plt.bar(summary.experiment_id,summary.hydro_generation_gwh); plt.axhline(CENACE,color="black",ls="--",label="CENACE 2022"); plt.xticks(rotation=60,ha="right"); plt.ylabel("Hydro generation (GWh)"); plt.legend(); plt.tight_layout(); plt.savefig(fig/"hydro_experiment_generation.png",dpi=160); plt.close()
    offer=pd.read_csv(OUT/"hydro_offer_diagnostics.csv"); plt.figure(figsize=(7,4)); plt.hist(offer.mean_p_max_pu,bins=12); plt.xlabel("Annual mean p_max_pu"); plt.ylabel("Hydro plants"); plt.tight_layout(); plt.savefig(fig/"hydro_offer_mean_capacity_factor.png",dpi=160); plt.close()

def solve_experiment(base, row):
    eid=row.experiment_id; d=RESULTS/eid; d.mkdir(parents=True,exist_ok=True); n=pypsa.Network(BASELINE); ids=hydro_ids(n)
    if row.changed_parameter == "hydro marginal cost": n.generators.loc[ids,"marginal_cost"]=float(row.experimental_value)
    elif row.changed_parameter == "p_max_pu": n.generators_t.p_max_pu.loc[:,ids]=1.0
    elif row.changed_parameter == "p_max_pu multiplier": n.generators_t.p_max_pu.loc[:,ids]=n.generators_t.p_max_pu.loc[:,ids].mul(float(row.experimental_value)).clip(upper=1)
    status, term=n.optimize(solver_name="highs", solver_options={"log_to_console":False})
    solved=d/"network_solved_2022.nc"; n.export_to_netcdf(solved)
    p=n.generators_t.p; hydro=p.loc[:,ids].sum().sum()/1000
    record={"experiment_id":eid,"solver_status":str(status),"termination_condition":str(term),"network_sha256":sha256(solved),"hydro_generation_gwh":hydro,"hydro_shortfall_gwh":CENACE-hydro,"thermal_generation_gwh":p.loc[:,n.generators.carrier.eq("oil")|n.generators.carrier.eq("biomass")|n.generators.carrier.eq("biogas")].sum().sum()/1000,"snapshots":len(n.snapshots)}
    (d/"summary.json").write_text(json.dumps(record,indent=2)); return record

def main(run=False):
    n=pypsa.Network(BASELINE); lineage=write_lineage(n); offer,av=write_offer(n); write_attribution(n,av); reg=registry()
    (OUT/"baseline_metadata.json").write_text(json.dumps({"network":str(BASELINE.relative_to(ROOT)),"network_sha256":sha256(BASELINE),"snapshots":len(n.snapshots),"hydro_plants":len(hydro_ids(n)),"hydro_nominal_mw":float(n.generators.loc[hydro_ids(n),"p_nom"].sum())},indent=2))
    write_evidence_and_corrections(); write_summary_and_figures()
    if run:
        records=[solve_experiment(n,row) for _,row in reg.iterrows() if row.experiment_id != "F06F_E00_BASELINE"]
        pd.DataFrame(records).to_csv(RESULTS/"experiment_summary.csv",index=False)

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--run-experiments",action="store_true"); main(ap.parse_args().run_experiments)
