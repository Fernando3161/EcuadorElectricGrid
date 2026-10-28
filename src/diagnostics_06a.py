"""Read-only full-year diagnostics for the consolidated 2022 network."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = {'hydro':'#377eb8','thermal':'#e66101','nonconventional':'#4daf4a','wind':'#984ea3','onwind':'#984ea3','solar':'#e6ab02','biomass':'#66a61e','biogas':'#1b9e77','interconnection':'#555555','colombia_import':'#555555','load_shedding':'#d73027','run_of_river':'#377eb8','reservoir':'#084594'}

def _gwh(x, weights): return x.mul(weights, axis=0).sum() / 1000.0

def run_diagnostics(root: Path, network_path: Path) -> dict:
    import pypsa
    root, network_path = Path(root), Path(network_path)
    if not network_path.exists(): raise FileNotFoundError(f'Canonical solved network required: {network_path}')
    sha = hashlib.sha256(network_path.read_bytes()).hexdigest()
    n = pypsa.Network(network_path)
    if len(n.snapshots) != 8760: raise ValueError(f'Expected 8760 snapshots, found {len(n.snapshots)}')
    if pd.Timestamp(n.snapshots[0]).year != 2022 or pd.Timestamp(n.snapshots[-1]).year != 2022: raise ValueError('Network snapshots are not the 2022 calendar')
    out, figdir = root/'results/phase_06a', root/'reports/figures/phase_06a'; out.mkdir(exist_ok=True, parents=True); figdir.mkdir(exist_ok=True, parents=True)
    sns.set_theme(style='whitegrid', context='paper'); plt.rcParams.update({'font.family':'Arial','figure.facecolor':'white'})
    w = n.snapshot_weightings.generators
    dispatch = n.generators_t.p.clip(lower=0)
    gen = n.generators.copy(); gen['annual_gwh'] = _gwh(dispatch, w)
    gen['technology'] = gen.carrier.fillna('other').replace({'interconnection_colombia':'ex_COL'})
    gen['capacity_mw'] = gen.p_nom
    carrier = gen.groupby('carrier').annual_gwh.sum().sort_values(ascending=False)
    monthly = dispatch.groupby(n.snapshots.month).sum().T.mul(w.groupby(n.snapshots.month).sum(), axis=1)/1000
    monthly = monthly.T.groupby(gen.loc[monthly.index,'technology']).sum().T if False else None
    rows=[]
    for c,v in carrier.items(): rows.append({'carrier':c,'annual_gwh':float(v)})
    pd.DataFrame(rows).to_csv(out/'annual_dispatch_by_technology.csv',index=False)
    manifest=[]
    def save(fig, name, description, units=''):
        fig.tight_layout(); fig.savefig(figdir/f'{name}.png',dpi=300,bbox_inches='tight'); fig.savefig(figdir/f'{name}.svg',bbox_inches='tight'); plt.close(fig); manifest.append({'figure_id':name,'filename':f'phase_06a/{name}.png','description':description,'units':units,'network_sha256':sha,'status':'generated'})
    # Geographic diagnostics reuse the established Notebook 04 cartography.
    try:
        import geopandas as gpd
        from matplotlib.collections import LineCollection
        from src.topology_plots import ECUADOR_MAINLAND_EXTENT, VOLTAGE_COLORS, load_mainland_provinces
        provinces=load_mainland_provinces(root/'data/raw/gadm/gadm41_ECU.gpkg')
        xcol,ycol=('x','y') if {'x','y'}.issubset(n.buses.columns) else ('lon','lat')
        def map_base(title):
            fig,ax=plt.subplots(figsize=(10,8)); provinces.plot(ax=ax,color='#f7f4ed',edgecolor='#8c8c8c',linewidth=.55,zorder=0); provinces.boundary.plot(ax=ax,color='#676767',linewidth=.45,zorder=1)
            seg=[[(n.buses.loc[r.bus0,xcol],n.buses.loc[r.bus0,ycol]),(n.buses.loc[r.bus1,xcol],n.buses.loc[r.bus1,ycol])] for r in n.lines.itertuples() if r.bus0 in n.buses.index and r.bus1 in n.buses.index]
            if seg: ax.add_collection(LineCollection(seg,colors=[VOLTAGE_COLORS.get(float(v),'#777777') for v in n.lines.loc[[r.bus0 in n.buses.index and r.bus1 in n.buses.index for r in n.lines.itertuples()],'v_nom']],linewidths=1.2,alpha=.65,zorder=2))
            ax.set_xlim(ECUADOR_MAINLAND_EXTENT[:2]); ax.set_ylim(ECUADOR_MAINLAND_EXTENT[2:]); ax.set_aspect('equal'); ax.set_title(title); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.grid(color='#d7d7d7',linewidth=.4,linestyle=':',alpha=.7); return fig,ax
        domestic=n.loads.index[n.loads.carrier.eq('electricity')] if 'carrier' in n.loads else n.loads.index
        ld=n.loads_t.p_set[domestic].mul(n.snapshot_weightings.objective,axis=0).sum()/1000; lb=n.loads.loc[domestic].groupby('bus').apply(lambda x:float(ld.reindex(x.index).sum()))
        fig,ax=map_base('Ecuador SNI annual domestic demand by bus'); xy=n.buses.loc[lb.index]; demand_scale=np.log1p(np.maximum(lb.values,0))/np.log1p(max(lb.max(),1)); demand_sizes=22+110*demand_scale; ax.scatter(xy[xcol],xy[ycol],s=demand_sizes,c='#d73027',alpha=.72,edgecolor='white',linewidth=.5,zorder=4); ax.legend([plt.scatter([],[],s=22+110*np.log1p(s)/np.log1p(max(lb.max(),1)),c='#d73027') for s in [25,100,500,1000]],[f'{s} GWh' for s in [25,100,500,1000]],title='Bubble area ∝ log1p(demand)',loc='lower right'); save(fig,'a01_annual_demand_map','Annual domestic electricity demand by bus; bubble area uses log1p(GWh)','GWh; log-scaled bubble area')
        physical=gen[~gen.carrier.isin(['load_shedding','colombia_import_supply','electricity'])]
        gbus=physical.groupby('bus').annual_gwh.sum(); gtech=physical.groupby(['bus','technology']).annual_gwh.sum().unstack(fill_value=0); gdominant=gtech.idxmax(axis=1); xy=n.buses.loc[gbus.index]; gsize=32+38*np.log1p(np.maximum(gbus.values,0)); gcolor=[COLORS.get(gdominant.get(i,'other'),'#777777') for i in gbus.index]; fig,ax=map_base('Ecuador SNI annual domestic generation by connection bus'); ax.scatter(xy[xcol],xy[ycol],s=gsize,c=gcolor,alpha=.78,edgecolor='white',linewidth=.5,zorder=4); techs=[t for t in gtech.columns if t in COLORS]; ax.legend([plt.scatter([],[],s=105,c=COLORS[t]) for t in techs],techs,title='Dominant technology',loc='lower right',fontsize=8); save(fig,'a02_annual_generation_map','Annual domestic generator dispatch by bus; imports and load shedding excluded; bubble area uses log1p(GWh)','GWh; log-scaled bubble area')
        cap=physical.groupby('bus').p_nom.sum(); ctech=physical.groupby(['bus','technology']).p_nom.sum().unstack(fill_value=0); cdom=ctech.idxmax(axis=1); xy=n.buses.loc[cap.index]; csize=24+20*np.log1p(np.maximum(cap.values,0)); ccolor=[COLORS.get(cdom.get(i,'other'),'#777777') for i in cap.index]; fig,ax=map_base('Ecuador SNI installed generation capacity'); ax.scatter(xy[xcol],xy[ycol],s=csize,c=ccolor,alpha=.78,edgecolor='white',linewidth=.5,zorder=4); techs=[t for t in ctech.columns if t in COLORS]; ax.legend([plt.scatter([],[],s=85,c=COLORS[t]) for t in techs],techs,title='Dominant technology',loc='lower right',fontsize=8); save(fig,'a03_installed_capacity_by_technology_map','Installed domestic generation capacity by bus; imports and load shedding excluded; bubble area uses log1p(MW)','MW; log-scaled bubble area')
        net=gbus.reindex(n.buses.index,fill_value=0)-lb.reindex(n.buses.index,fill_value=0); xy=n.buses.loc[net.index]; fig,ax=map_base('Ecuador SNI annual net nodal electricity balance'); ax.scatter(xy[xcol],xy[ycol],s=np.maximum(net.abs().values,1)*1.2,c=net.values,cmap='RdBu_r',alpha=.8,edgecolor='white',linewidth=.5,zorder=4); fig.colorbar(ax.collections[-1],ax=ax,label='generation − demand (GWh)'); save(fig,'a04_net_nodal_energy_balance','Annual domestic generation minus domestic demand by bus','GWh')
    except Exception as exc:
        manifest.append({'figure_id':'geographic_maps','filename':None,'description':'Geographic bubble maps skipped','units':'','network_sha256':sha,'status':f'skipped: {exc}'})
    # B01 / F04 validation comparison
    ref={'hydro':24499.54,'thermal':3393.35,'nonconventional':325.89,'interconnection':465.83}
    from src.validation import aggregate_model_supply, load_validation_contract
    validation_contract=load_validation_contract(root/'config/validation_2022.yaml')
    sim_series=aggregate_model_supply(carrier.drop(labels=['load_shedding'],errors='ignore'), validation_contract)
    sim={k:float(sim_series.get(k,0)) for k in ref}; total=sum(sim.values()); rt=sum(ref.values())
    mix=pd.DataFrame({'reference_gwh':ref,'simulated_gwh':sim}); mix['reference_share']=mix.reference_gwh/rt; mix['simulated_share']=mix.simulated_gwh/total; mix['error_pp']=100*(mix.simulated_share-mix.reference_share)
    mix.to_csv(out/'carrier_validation_diagnostics.csv')
    fig,ax=plt.subplots(figsize=(9,5)); mix[['reference_gwh','simulated_gwh']].plot.bar(ax=ax,color=['#999999','#377eb8']); ax.set_ylabel('GWh'); ax.set_title('2022 generation mix: model versus CENACE reference'); save(fig,'b01_generation_mix_validation','Annual carrier generation compared with CENACE','GWh')
    fig,ax=plt.subplots(figsize=(8,4)); mix.error_pp.plot.bar(ax=ax,color=[COLORS.get(k,'#777') for k in mix.index]); ax.axhline(0,color='black',lw=.8); ax.set_ylabel('signed share error (percentage points)'); ax.set_title('Carrier share errors; nonconventional remains outside the screen'); save(fig,'f04_carrier_share_errors','Signed carrier share errors','percentage points')
    # monthly dispatch and shares
    d=dispatch.copy(); d['month']=n.snapshots.month; d['month_name']=n.snapshots.month_name()
    md=d.drop(columns=['month','month_name']).groupby(d.month).sum()/1000; md=md.T.groupby(gen.technology).sum().T
    fig,ax=plt.subplots(figsize=(11,5)); md.plot.bar(stacked=True,ax=ax,colormap='tab20'); ax.set_ylabel('GWh'); ax.set_title('Monthly dispatch by technology'); save(fig,'b02_monthly_generation_gwh','Weighted monthly dispatch by technology','GWh')
    shares=md.div(md.sum(axis=1),axis=0); fig,ax=plt.subplots(figsize=(11,5)); shares.plot.bar(stacked=True,ax=ax,colormap='tab20'); ax.set_ylabel('share'); ax.set_title('Monthly generation shares'); save(fig,'b03_monthly_generation_shares','Monthly technology shares','fraction')
    # demand heatmap and duration
    load_ids=n.loads.index[n.loads.carrier.eq('electricity')] if 'carrier' in n.loads else n.loads.index
    demand=n.loads_t.p_set[load_ids].sum(axis=1); demand.to_csv(out/'national_demand_hourly_mw.csv')
    heat=demand.to_frame('mw'); heat['day']=heat.index.dayofyear; heat['hour']=heat.index.hour; piv=heat.pivot_table(index='hour',columns='day',values='mw')
    fig,ax=plt.subplots(figsize=(13,4)); sns.heatmap(piv,ax=ax,cmap='viridis'); ax.set_title('National domestic demand by day and hour'); ax.set_xlabel('day of year'); ax.set_ylabel('hour'); save(fig,'b05_national_demand_heatmap','Hourly national domestic demand','MW')
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(np.sort(demand)[::-1]); ax.set_xlabel('hours ranked'); ax.set_ylabel('MW'); ax.set_title('National load duration curve'); save(fig,'b06_load_duration_curve','Full-year national demand duration curve','MW')
    top=gen[~gen.carrier.isin(['load_shedding','electricity'])].nlargest(15,'annual_gwh'); fig,ax=plt.subplots(figsize=(8,6)); top.annual_gwh.sort_values().plot.barh(ax=ax,color=[COLORS.get(gen.loc[i,'carrier'],'#777') for i in top.annual_gwh.sort_values().index]); ax.set_xlabel('GWh'); ax.set_title('Largest physical generators by annual dispatch'); save(fig,'b07_top_generators','Top physical generators by dispatch','GWh')
    # capacity factors and hydro diagnostics
    gen['annual_cf']=gen.annual_gwh/(gen.p_nom*8760/1000).replace(0,np.nan); cf=gen.groupby('technology').apply(lambda x:x.annual_gwh.sum()/(x.p_nom.sum()*8760/1000)).dropna(); fig,ax=plt.subplots(figsize=(8,4)); cf.plot.bar(ax=ax,color=[COLORS.get(k,'#777') for k in cf.index]); ax.set_ylabel('effective capacity factor'); ax.set_title('Aggregate annual effective capacity factors'); save(fig,'c01_monthly_effective_capacity_factors','Aggregate effective capacity factors','fraction')
    fig,ax=plt.subplots(figsize=(9,5)); sns.boxplot(data=gen[gen.annual_cf.notna()],x='technology',y='annual_cf',palette={k:COLORS.get(k,'#777') for k in gen.technology.unique()},ax=ax); ax.tick_params(axis='x',rotation=45); ax.set_title('Generator-level annual capacity-factor distribution'); save(fig,'c03_generator_cf_distribution','Generator-level annual capacity factor distributions','fraction')
    hydro=gen[gen.carrier.eq('hydro')].copy()
    fleet_path=root/'data/processed/generation/generators_2022.csv'
    if fleet_path.exists():
        fleet=pd.read_csv(fleet_path).set_index('plant_id')
        subtype=fleet.reindex(hydro.index)['subtechnology_original']
    else:
        subtype=pd.Series('run_of_river',index=hydro.index)
    hydro['hydro_type']=subtype.fillna('run_of_river').map(lambda x:'reservoir' if str(x).lower() in {'embalse','reservoir','reservorio'} else 'run_of_river')
    hmonthly=md.reindex(columns=[])
    hdispatch=dispatch[hydro.index].copy(); hdispatch['month']=n.snapshots.month; hm=hdispatch.groupby('month').sum()/1000; hm=hm.T.groupby(hydro.hydro_type).sum().T
    fig,ax=plt.subplots(figsize=(9,4)); hm.plot.bar(ax=ax,stacked=True,color=[COLORS['run_of_river'],COLORS['reservoir']]); ax.set_ylabel('GWh'); ax.set_title('Hydro seasonality: run-of-river versus reservoir'); save(fig,'d03_hydro_technology_seasonality','Monthly hydro production by profile class','GWh')
    offered=(n.generators_t.p_max_pu[hydro.index].mul(hydro.p_nom,axis=1).mul(w,axis=0).sum()/1000); hd=pd.DataFrame({'offered_gwh':offered,'dispatched_gwh':hydro.annual_gwh}); fig,ax=plt.subplots(figsize=(6,5)); sns.scatterplot(data=hd,x='offered_gwh',y='dispatched_gwh',hue=hydro.hydro_type.values,ax=ax); ax.set_title('Hydro offered energy versus dispatch'); ax.set_xlabel('modeled offer (GWh)'); ax.set_ylabel('dispatch (GWh)'); save(fig,'d04_hydro_offer_vs_dispatch','Hydro modeled offered energy versus dispatch','GWh')
    # adequacy and balance
    shed=dispatch.loc[:,gen.carrier.eq('load_shedding')].sum(axis=1) if gen.carrier.eq('load_shedding').any() else pd.Series(0,index=n.snapshots); fig,ax=plt.subplots(figsize=(8,3)); ax.plot(shed.values); ax.set_ylabel('MW'); ax.set_title('Load shedding duration diagnostic (zero is a result)'); save(fig,'f01_load_shedding_duration','Hourly load shedding','MW')
    supply=float(carrier.sum()); imports=float(carrier.get('interconnection_colombia',0)); exports=float(n.loads_t.p_set.loc[:,n.loads.carrier.eq('export')].mul(n.snapshot_weightings.objective,axis=0).sum().sum()/1000) if 'carrier' in n.loads else 0; demand_gwh=float(demand.mul(n.snapshot_weightings.objective).sum()/1000); vals=pd.Series({'domestic generation':supply-imports,'imports':imports,'demand':-demand_gwh,'exports':-exports,'balance residual':float(supply-demand_gwh-exports)}); fig,ax=plt.subplots(figsize=(8,4)); vals.plot.bar(ax=ax,color=['#377eb8','#555555','#222222','#a65628','#d73027']); ax.axhline(0,color='black',lw=.8); ax.set_ylabel('GWh'); ax.set_title('Modeled energy boundary and numerical balance');
    for i,value in enumerate(vals): ax.text(i,value+(350 if value>=0 else -350),f'{value:,.2f} GWh',ha='center',va='bottom' if value>=0 else 'top',fontsize=9,fontweight='bold')
    save(fig,'f03_energy_balance','Modeled supply and demand boundary; imports and exports use solved network flows; value labels preserve smaller boundary flows; 931.43 GWh reference residual is not modeled','GWh')
    (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); (out/'diagnostic_metadata.json').write_text(json.dumps({'network':str(network_path.relative_to(root)),'network_sha256':sha,'snapshots':len(n.snapshots),'solver_invoked':False,'status':'read_only_diagnostic'},indent=2),encoding='utf-8')
    return {'figures_generated':len(manifest),'network_sha256':sha,'snapshots':len(n.snapshots),'manifest':manifest}
