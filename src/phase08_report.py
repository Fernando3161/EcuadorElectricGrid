from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

plt.rcParams.update({'font.family': 'Arial', 'font.sans-serif': ['Arial', 'DejaVu Sans']})
CASE_LABELS = {'2022 baseline (unsolved reference)': '2022 Base', '2030 Tendencial': '2030 Tend.', '2030 Expansion': '2030 Exp.'}
CARRIER_LABELS = {'interconnection_colombia': 'Inter COL', 'load_shedding': 'Shedding', 'biogas': 'Biogas', 'biomass': 'Biomass', 'hydro': 'Hydro', 'oil': 'Oil', 'onwind': 'Wind', 'solar': 'Solar'}
PALETTE = plt.get_cmap('viridis')
LINE_STYLES = {'hydro': '-', 'oil': '--', 'interconnection_colombia': '-.', 'solar': ':', 'onwind': (0, (5, 1)), 'load_shedding': (0, (3, 1, 1, 1))}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/phase_08'
FIG = ROOT / 'reports/figures/phase_08'
FIG.mkdir(parents=True, exist_ok=True)

def carrier_mix(n):
    supply = n.generators_t.p.mul(n.generators.sign, axis=1).sum() / 1000
    return supply.groupby(n.generators.carrier).sum()

def add_case(rows, name, path):
    n = pypsa.Network(path)
    mix = carrier_mix(n)
    total = float(n.loads_t.p_set.sum().sum() / 1000)
    shed = float(mix.get('load_shedding', 0))
    rows.append({'case': name, 'demand_gwh': total, 'load_shedding_gwh': shed,
                 'load_shedding_pct': 100 * shed / total, **{f'carrier_{k}_gwh': float(v) for k, v in mix.items()}})
    return n, mix

base = pypsa.Network(ROOT / 'results/phase_06d/baseline/network_solved_2022.nc')
rows = []
networks = {'2022 baseline (unsolved reference)': base}
for name, filename in [('2030 Tendencial', 'tendencial_2030.solved.nc'), ('2030 Expansion', 'expansion_2030.solved.nc')]:
    n, mix = add_case(rows, name, OUT / filename); networks[name] = n
base_mix = carrier_mix(base)
base_total = float(base.loads_t.p_set.sum().sum() / 1000)
rows.insert(0, {'case': '2022 baseline (unsolved reference)', 'demand_gwh': base_total,
                'load_shedding_gwh': float(base_mix.get('load_shedding', 0)),
                'load_shedding_pct': 100 * float(base_mix.get('load_shedding', 0)) / base_total,
                **{f'carrier_{k}_gwh': float(v) for k, v in base_mix.items()}})
metrics = pd.DataFrame(rows).fillna(0)
metrics.to_csv(OUT / 'base_vs_2030_metrics.csv', index=False)

carriers = sorted({c.removeprefix('carrier_').removesuffix('_gwh') for c in metrics if c.startswith('carrier_')})
energy = pd.DataFrame({CARRIER_LABELS.get(c, c): metrics.get(f'carrier_{c}_gwh', pd.Series(0, index=metrics.index)).to_numpy() for c in carriers}, index=metrics.case.map(CASE_LABELS))
energy.to_csv(OUT / 'base_vs_2030_carrier_energy.csv')
shares = energy.div(energy.sum(axis=1), axis=0).mul(100)
shares.to_csv(OUT / 'base_vs_2030_carrier_shares.csv')

colors = [PALETTE(x) for x in __import__('numpy').linspace(0.05, 0.95, len(energy.columns))]
fig, ax = plt.subplots(figsize=(11, 5)); energy.plot.bar(stacked=True, ax=ax, color=colors); ax.set_ylabel('GWh'); ax.set_title('Annual energy by carrier'); ax.tick_params(axis='x', labelrotation=0); ax.legend(title='Carrier', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False); fig.tight_layout(rect=[0, 0, .82, 1]); fig.savefig(FIG/'phase08_annual_energy_by_carrier.png', dpi=180); plt.close(fig)
fig, ax = plt.subplots(figsize=(11, 5)); shares.plot.bar(stacked=True, ax=ax, color=colors); ax.set_ylabel('Share (%)'); ax.set_title('Relative supply shares'); ax.tick_params(axis='x', labelrotation=0); ax.legend(title='Carrier', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False); fig.tight_layout(rect=[0, 0, .82, 1]); fig.savefig(FIG/'phase08_relative_carrier_shares.png', dpi=180); plt.close(fig)

week = next(iter(networks.values())).snapshots[:7*24]
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
for ax, (name, n) in zip(axes, networks.items()):
    snap = week.intersection(n.snapshots)
    dispatch = n.generators_t.p.loc[snap].mul(n.generators.sign, axis=1).groupby(n.generators.carrier, axis=1).sum()
    for carrier in dispatch.columns:
        nice = CARRIER_LABELS.get(carrier, carrier)
        ax.plot(dispatch.index, dispatch[carrier], label=nice, linewidth=1.5, linestyle=LINE_STYLES.get(carrier, '-'), color=PALETTE({'hydro':0.12, 'oil':0.28, 'interconnection_colombia':0.44, 'solar':0.60, 'onwind':0.76, 'load_shedding':0.92}.get(carrier, 0.5)))
    ax.set_title(CASE_LABELS.get(name, name)); ax.set_ylabel('MW'); ax.grid(alpha=.2); ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=8)
axes[-1].set_xlabel('Representative week (network timestamps)'); fig.tight_layout(rect=[0, 0, .82, 1]); fig.savefig(FIG/'phase08_representative_week_dispatch.png', dpi=180); plt.close(fig)

factors = pd.read_csv(ROOT / 'data/raw/generation/carriers.csv')
factor_col = next(c for c in factors.columns if 'co2' in c.lower())
factor_map = factors.set_index('Carrier')[factor_col].to_dict() if 'Carrier' in factors else {}
co2 = []
for name, n in networks.items():
    supply = n.generators_t.p.mul(n.generators.sign, axis=1).sum()
    emissions = sum(float(supply.get(g, 0)) * float(factor_map.get(c, 0)) for g, c in n.generators.carrier.items()) / 1e6
    co2.append({'case': name, 'operational_co2_tonnes': emissions, 'factor_source': 'data/raw/generation/carriers.csv', 'unmapped_carriers_assumed_zero': True})
pd.DataFrame(co2).to_csv(OUT / 'operational_co2_screening.csv', index=False)
metrics[['case','demand_gwh','load_shedding_gwh','load_shedding_pct']].to_csv(OUT / 'adequacy_indicators.csv', index=False)
print(metrics[['case','demand_gwh','load_shedding_gwh','load_shedding_pct']].to_string(index=False))
