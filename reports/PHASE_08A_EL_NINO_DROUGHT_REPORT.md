# Phase 08A — El Niño hydro-shortage screening

## Executive result

This standalone screening tests whether Ecuador's 2026 system can meet a 2025-scaled demand trajectory when selected hydro inflows are reduced in September–December. The experiment is intentionally transparent: it modifies monthly river caudals first, rebuilds the hydro availability profiles, and only then solves the full-year PyPSA network.

All six annual cases terminated **optimal**. However, the modeled system already produces non-zero shedding at a 10% reduction in the selected caudals: **10.6 GWh (0.032%)**. At 50%, shedding reaches **328.8 GWh (1.007%)**. This is an adequacy warning, not a forecast of actual 2026 rationing.

![Adequacy response](../results/phase_08a/figures/shedding_vs_caudal_reduction.png)

This response curve shows that annual unmet demand increases nonlinearly as the selected caudals are reduced. The first cases produce relatively small energy deficits, but the slope increases substantially beyond H20–H30, indicating a shrinking operational margin during severe hydro stress.

## 1. Scope and scientific question

The question is: **under a 2026 demand proxy, how sensitive is annual adequacy to a drought-like reduction in the caudal series supporting the dominant hydro plants?** January–August retain the source monthly flows. September–December are stress months. Six cases are tested: H00, H10, H20, H30, H40 and H50, where the label is the percentage reduction applied to the selected river series during the stress months.

This is a screening experiment. It is not a probabilistic El Niño forecast, a hydrological hindcast, or a reservoir-operation study.

## 2. Inputs and reproducibility

- **Reference network:** solved Ecuador 2022 baseline from `results/phase_06d/baseline/network_solved_2022.nc`.
- **Hydrology:** `data/raw/generation/caudals.csv`, processed with the same `build_hydro_availability` pathway documented in Notebook 03.
- **Hydro mapping:** `data/interim/generation/hydro_profile_assignments_2022.csv`.
- **Demand:** 2022 hourly shape scaled to the CENACE-reported 2025 annual demand proxy of 32,650.66 GWh; the same annual target is used for 2026 because a complete 2026 demand forecast is not available.
- **Interconnection:** the network's existing import representation and constraints are retained; no additional import capacity is invented in this experiment.

The selected caudal series are the four with the largest aggregate assigned hydro capacity: `coca_codo_sinclair`, `ingreso_amaluza`, `agoyan`, and `mazar`. Their scope is shown below.

![Caudal scope](../results/phase_08a/figures/caudal_monthly_original_vs_shocked.png)

The figure confirms that the perturbation is applied to the river-flow inputs rather than directly to generator availability. January–August remain unchanged, while the selected caudals are reduced only during September–December.

The monthly shock is applied before profile construction:

`Q_shocked(r,m) = Q_source(r,m) × (1 − reduction)` for selected rivers `r` and September–December `m`.

![Rebuilt availability](../results/phase_08a/figures/hydro_availability_original_vs_shocked.png)

The rebuilt profiles translate the monthly caudal shock into hourly hydro availability. The seasonal reduction is visible after September, while the profile shape and the non-stressed months remain governed by the canonical hydro-profile procedure.

## 3. Full-year results

| Case | Selected-caudal reduction | Shedding (GWh) | Shedding (% demand) | Hydro (GWh) | Inter COL (GWh) | Oil (GWh) | Operational CO₂ (kt) | Solver |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H00 | 0% | 3.9 | 0.012% | 25,439.6 | 2,171.8 | 3,939.2 | 976.9 | ok/optimal |
| H10 | 10% | 10.6 | 0.032% | 24,963.6 | 2,251.6 | 4,288.3 | 1,063.5 | ok/optimal |
| H20 | 20% | 26.0 | 0.080% | 24,446.0 | 2,328.3 | 4,670.3 | 1,158.2 | ok/optimal |
| H30 | 30% | 77.1 | 0.236% | 23,887.4 | 2,404.5 | 5,063.0 | 1,255.6 | ok/optimal |
| H40 | 40% | 180.9 | 0.554% | 23,303.3 | 2,471.6 | 5,444.6 | 1,350.3 | ok/optimal |
| H50 | 50% | 328.8 | 1.007% | 22,624.5 | 2,519.5 | 5,822.6 | 1,444.0 | ok/optimal |

![Annual mix](../results/phase_08a/figures/annual_generation_mix_by_scenario.png)

As hydro output falls, the system compensates primarily through higher oil generation, increased Colombia imports, and a small increase in other thermal or renewable contributions. Load shedding appears as an explicit residual adequacy category rather than being hidden in the energy balance.

![Generation response](../results/phase_08a/figures/generation_response_by_scenario.png)

This plot isolates the main substitution mechanism: hydro generation declines with the caudal shock, while oil generation and imports rise. The response demonstrates that imports provide partial relief but do not fully remove the adequacy exposure.

![Monthly shedding](../results/phase_08a/figures/shedding_by_month_and_scenario.png)

The monthly traces show that the annual shedding totals are concentrated in the dry-season transition and subsequent months. This is important operationally because the same annual shedding energy can represent very different risks depending on whether it is dispersed or concentrated in a few high-stress months.

The dispatch response is technically coherent: hydro energy declines as the caudal shock increases; imports and oil generation rise; shedding becomes material during the stress season. The solved network represents Colombia imports with a 500 MW model limit. The externally documented 450 MW transfer figure is contextual here and is not imposed in these existing solves.

![Representative traces](../results/phase_08a/figures/representative_week_dispatch_by_scenario.png)

The representative-week traces illustrate the dispatch response at hourly resolution. Increasing drought severity changes the balance between hydro, imports, thermal generation, and residual shedding; the figure should be read as an operational example rather than a statistical representative of every week in the year.

## 4. Environmental interpretation

The project carrier table assigns 0.248 tCO₂/MWh to oil and zero operational emissions to hydro, wind, solar, biomass and biogas. Under that accounting convention, the increased oil dispatch raises modeled operational CO₂ from approximately **1,064 kt in H00** to **1,444 kt in H50**. This is a dispatch-factor result, not a full life-cycle assessment: upstream fuel, construction, reservoir methane, land-use change, and imported-electricity accounting are outside this screening boundary.

![Operational CO2](../results/phase_08a/figures/operational_co2_by_scenario.png)

Operational CO₂ increases with drought severity because oil generation replaces part of the lost hydro energy. These values use the project’s operational carrier factors and therefore exclude life-cycle emissions, reservoir methane, construction emissions, and upstream fuel effects.

## 5. Technical conclusions

1. **The modeled 2026 proxy is hydro-sensitive.** Even a 10% stress applied only in September–December creates a small but non-zero annual shedding result.
2. **The response accelerates.** Shedding rises from 10.6 GWh at H10 to 328.8 GWh at H50; the system is not protected by imports alone.
3. **Thermal and import substitution is visible but bounded.** Oil generation and modeled Colombia imports increase as hydro falls, while the interconnection remains constrained.
4. **The stress season matters.** Monthly plots show that the adequacy impact is concentrated after the August-to-September transition, rather than being a uniform annual capacity derating.
5. **Implementation feasibility requires flexibility before the dry season.** A future operational scenario should test reserve procurement, demand response, maintenance scheduling, import availability, thermal fuel limits, and additional non-hydro capacity—not only hydro derating.

## 6. Limitations and next steps

- The four-river selection is a transparent capacity-based proxy, not a basin-scale hydrological model.
- The stress magnitudes are scenario assumptions; they are not assigned probabilities.
- The experiment rebuilds hydro availability but does not model explicit reservoir mass balance, evaporation, inflow correlation, or multi-year storage carry-over.
- The 2025 demand total is used as the 2026 proxy and the 2022 hourly shape is retained.
- Results are sensitive to network topology, generator costs, import offers, and the treatment of load shedding.

The next scientifically useful extension is a calibrated hindcast: replace the synthetic September–December reductions with observed basin-level inflow anomalies, add reservoir-state constraints, and test a matrix of demand, import ceiling, thermal availability, and demand-response assumptions.

## 7. Source context

The seasonal framing follows CENACE reporting that distinguishes a wetter April–September period from a drier October–March period, while this experiment uses a conservative September start to capture the transition. See the [CENACE 2025 annual report](https://www.cenace.gob.ec/wp-content/uploads/downloads/2026/03/informe_anual_2025_parte_1.pdf), [CENACE/INAMHI climate information](https://servicios.inamhi.gob.ec/clima/), and the [CENACE-reported 450 MW interconnection support](https://www.cenace.gob.ec/incremento-de-la-generacion-electrica/). These sources provide context; they do not validate the synthetic reduction percentages.

## 8. Reproduction

The scenario networks and logs are saved in `results/phase_08a/`. The executed, result-backed notebook is `results/phase_08a/notebooks/08a_hydro_shortage_2026.solved.ipynb`. The isolated-case command is:

```powershell
conda run -n pypsa-earth-ec python scripts/run_phase08a_case.py --reduction 0.30
```

The final aggregation and figure command is:

```powershell
conda run -n pypsa-earth-ec python scripts/finalize_phase08a.py
```

## 9. Dry-season risk indicators

The annual shedding total is complemented here with frequency and peak-stress indicators. A shedding hour is counted when modeled load shedding exceeds 1 kW. `Imports + shedding` is not interpreted as unmet demand: imports are served demand, while shedding is unmet demand. Their sum is reported as a combined indicator of external dependence plus residual adequacy risk.

The monthly table below covers September–December, the stressed period in this experiment.

| Scenario | Month | Imports (GWh) | Shedding (GWh) | Imports + shedding (GWh) | Peak shedding (MW) | Peak (shedding + import) MW | Shedding hours | Model import-cap utilisation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H00 | 9 | 205.1 | 0.1 | 205.3 | 9.0 | 509.0 | 26 | 100.0% |
| H00 | 10 | 254.3 | 0.1 | 254.4 | 8.5 | 508.5 | 24 | 100.0% |
| H00 | 11 | 262.3 | 0.6 | 263.0 | 33.4 | 533.4 | 68 | 100.0% |
| H00 | 12 | 325.7 | 2.4 | 328.1 | 157.5 | 657.5 | 84 | 100.0% |
| H10 | 9 | 220.2 | 0.2 | 220.4 | 10.9 | 510.9 | 27 | 100.0% |
| H10 | 10 | 268.1 | 0.2 | 268.3 | 18.1 | 518.1 | 25 | 100.0% |
| H10 | 11 | 280.0 | 1.6 | 281.6 | 61.0 | 561.0 | 89 | 100.0% |
| H10 | 12 | 356.0 | 8.0 | 364.0 | 353.6 | 853.6 | 251 | 100.0% |
| H20 | 9 | 232.4 | 0.2 | 232.7 | 13.7 | 513.7 | 27 | 100.0% |
| H20 | 10 | 284.3 | 0.6 | 284.9 | 33.6 | 533.6 | 55 | 100.0% |
| H20 | 11 | 310.3 | 4.5 | 314.8 | 221.6 | 721.6 | 118 | 100.0% |
| H20 | 12 | 370.6 | 20.1 | 390.8 | 549.6 | 1049.6 | 361 | 100.0% |
| H30 | 9 | 244.2 | 0.3 | 244.6 | 24.7 | 524.7 | 28 | 100.0% |
| H30 | 10 | 311.1 | 1.8 | 312.9 | 83.3 | 583.3 | 102 | 100.0% |
| H30 | 11 | 342.8 | 15.2 | 358.0 | 446.6 | 946.6 | 136 | 100.0% |
| H30 | 12 | 372.0 | 59.2 | 431.2 | 745.7 | 1245.7 | 409 | 100.0% |
| H40 | 9 | 259.0 | 0.5 | 259.6 | 39.4 | 539.4 | 30 | 100.0% |
| H40 | 10 | 343.2 | 6.3 | 349.5 | 308.4 | 808.4 | 120 | 100.0% |
| H40 | 11 | 358.8 | 39.9 | 398.7 | 671.7 | 1171.7 | 266 | 100.0% |
| H40 | 12 | 372.0 | 133.6 | 505.6 | 941.8 | 1441.8 | 424 | 100.0% |
| H50 | 9 | 280.2 | 1.1 | 281.2 | 123.0 | 623.0 | 33 | 100.0% |
| H50 | 10 | 364.4 | 21.2 | 385.7 | 533.6 | 1033.6 | 196 | 100.0% |
| H50 | 11 | 360.0 | 86.2 | 446.2 | 896.8 | 1396.8 | 366 | 100.0% |
| H50 | 12 | 372.0 | 219.8 | 591.8 | 1137.9 | 1637.9 | 450 | 100.0% |

![Dry-season imports and shedding](../results/phase_08a/figures/dry_season_imports_and_shedding.png)

This comparison separates successful external support from actual unmet demand. Imports rise as hydro availability declines, but the lower panel shows that combined external dependence and residual adequacy stress also increase across the scenarios.

![Peak shedding by month](../results/phase_08a/figures/peak_shedding_by_dry_month.png)

The heatmap identifies the worst hourly unmet-demand event in each dry-season month. December becomes the most critical month in the severe scenarios, suggesting that risk is not only seasonal but also concentrated within the late dry-season progression.

![Peak combined stress](../results/phase_08a/figures/peak_combined_stress_by_dry_month.png)

This figure measures the maximum hourly combination of imports and shedding. It highlights hours when the system is simultaneously highly dependent on external support and still unable to serve all demand; it is a stress indicator, not a direct measure of unmet demand.

![Shedding hours](../results/phase_08a/figures/shedding_hours_by_period.png)

The frequency measure complements the annual GWh results. Higher-severity cases produce both more shedding energy and more hours affected, with most of the affected hours occurring during September–December.

The risk profile should be read in three dimensions:

- **Magnitude:** monthly GWh of shedding and the worst hourly shedding in MW.
- **Frequency:** the number of hours with any unmet demand over the full year, during September–December, and by month.
- **Exposure:** import energy and the maximum utilisation of the 450 MW interconnection ceiling.

This avoids treating a single annual GWh value as a complete adequacy assessment. A scenario with modest annual shedding but concentrated high-power events may be more operationally disruptive than a scenario with the same energy spread over many hours.

The complete machine-readable indicators are saved in `results/phase_08a/monthly_risk_indicators.csv` and `results/phase_08a/shedding_hours_by_period.csv`.



## 10. Annual power-duration curves

The following duration curves count how many hours per year the system operates at or above each power threshold. Each curve is formed by sorting the 8,760 hourly values from highest to lowest; therefore, moving right means that the threshold is sustained for more hours. These are annual screening curves, not probability distributions.

![Annual shedding and import duration curves](../results/phase_08a/figures/duration_curves_shedding_and_imports.png)

The first panel shows the persistence of load-shedding power, while the second shows the persistence of Colombia import power. A curve extending far to the right indicates frequent exposure, whereas a high value near the left edge indicates a short-duration but severe peak event.

![Annual combined-stress duration curve](../results/phase_08a/figures/duration_curve_combined_stress.png)

The combined curve adds imports and shedding hour by hour. It should be interpreted as a domestic-independence stress indicator: it estimates the power that would need to be supplied domestically to replace imports and eliminate residual shedding at each hour. It is not itself a measure of unmet demand because imports are successfully served energy.

The duration-curve data are saved in `results/phase_08a/duration_curve_data.csv` for further threshold analysis.


> **Import-cap caveat.** The solved networks use a 500 MW model limit for Colombia imports. The externally documented 450 MW transfer figure is contextual here and was not imposed in these existing solves; a strict 450 MW sensitivity requires a separate rerun.
