# Phase 08A — El Niño hydro-shortage resilience experiment

## Status and purpose

**Status:** design plan only; no notebook execution or annual solve has been run.

This standalone experiment will test the resilience of the Ecuador SNI model to
reduced river flows during the 2026 dry season associated with the El Niño
scenario. It is intentionally a simple sensitivity experiment, not a rainfall
forecast, reservoir model, or climate-impact attribution study.

The scientific question is:

> How much reduction in the caudales of the most important hydroelectric rivers
> can the 2026 system absorb before load shedding becomes material?

The experiment will preserve the existing network structure and hydro-profile
method. The perturbation will be applied to `caudals.csv` before calling the
existing `build_hydro_availability()` function.

## Climate and hydrological basis

CENACE describes Ecuador's main eastern-basin hydrological regime as generally
rainy from April to September and dry from October to March. Its 2025 annual
report identifies eleven major hydroelectric plants and states that they provide
approximately 85% of conventional hydro generation and about 60% of total SNI
generation. See the hydrology section of the
[CENACE 2025 Annual Report](https://www.cenace.gob.ec/wp-content/uploads/downloads/2026/03/informe_anual_2025_parte_1.pdf).

For 2026, public operator and government reporting described an earlier dry
season beginning in September and extending approximately to March 2027. The
[CENACE/sector report on the 2026 estiaje](https://www.primicias.ec/economia/estiaje-septiembre-sequia-deficit-generacion-contratacion-retrasos-130720/)
reports that September–March was the expected stress period, while INAMHI
emphasizes that El Niño rainfall effects differ by region. The
[INAMHI climate outlook](https://servicios.inamhi.gob.ec/clima/) describes
regional variation rather than a uniform national drought.

The model must therefore not claim that El Niño reduces rainfall everywhere.
The scenario is a basin-specific hydro-availability sensitivity applied to
selected river series. Coastal rainfall and flooding impacts are outside this
experiment.

## Model boundary and inputs

The experiment will use:

| Input | Role |
|---|---|
| `data/raw/generation/caudals.csv` | Monthly river-flow proxy series |
| `data/processed/generation/generators_2022.csv` | Hydro fleet and plant metadata |
| `data/interim/generation/hydro_profile_assignments_2022.csv` | Plant-to-river assignment and confidence |
| `src.availability.load_monthly_flows()` | Flow-table reader and validation |
| `src.availability.build_hydro_availability()` | Existing flow-to-hourly-availability method |
| `results/phase_06d/baseline/network_solved_2022.nc` | Solved structural reference network |
| CENACE 2025 demand table | 2025 annual demand and monthly demand basis |

No existing Phase 03, Phase 06 or Phase 08 notebook will be modified.

## Correct caudal-first transformation

The experiment must not directly multiply the final `p_max_pu` profiles. It
will transform the monthly caudal table first:

```text
original caudals.csv
        ↓
selected river/month flow shock
        ↓
scenario caudal table
        ↓
build_hydro_availability(...)
        ↓
hourly hydro p_max_pu for 2026
        ↓
scenario network and solve
```

For selected river (r), month (m), and shortage severity (s):

\[
Q^{scenario}_{r,m} = Q^{base}_{r,m}(1-s)
\]

All non-selected rivers and all non-stress months remain unchanged.

`build_hydro_availability()` then applies the project’s existing rules:

- run-of-river plants use their assigned monthly flow series;
- reservoir plants use the existing three-month rolling-flow average;
- run-of-river profiles use the existing empirical 0.90 quantile rule;
- reservoir profiles use the existing empirical 0.95 quantile rule;
- monthly availability is interpolated to hourly values;
- operating-start dates and profile bounds remain enforced.

This rolling-average treatment means that a September caudal shock can affect
reservoir availability beyond September. That persistence is part of the
existing hydro-profile methodology and must be reported, not removed.

Every scenario caudal row must retain:

- original caudal value and unit;
- shocked caudal value and unit;
- river/caudal ID;
- month;
- shortage severity;
- scenario ID;
- source file and source checksum;
- transformation description;
- whether the row is an official input or a modeling assumption.

## 2026 chronology and demand

The first experiment will simulate calendar year 2026:

- January–August: normal caudals;
- September–December: stressed caudals;
- demand: 2025 CENACE SNI demand shape scaled to the 2025 annual total.

CENACE’s 2025 annual report reports 32,650.66 GWh of SNI demand and a maximum
demand of 5,110.08 MW. The 2026 base demand will therefore be formed from the
2022 hourly model shape:

\[
D_{2026,t} = D_{2022,t} \times
\frac{32{,}650.66}{27{,}561.62}
\]

This is a data-availability assumption, not a claim that 2026 demand equals
2025 demand. It isolates hydrological sensitivity while avoiding an additional
uncontrolled forecast. A separate 3% demand-growth sensitivity may be added
after the core result is understood.

The January–March 2027 continuation is outside a single calendar-year solve.
It should be a later multi-year extension, not silently folded into the 2026
annual result.

## Hydro scope selection

The notebook will rank hydro plants by:

1. nominal capacity;
2. 2022 simulated annual generation;
3. contribution to total hydro generation;
4. assigned caudal series and assignment confidence.

The selected set will contain the three or four most influential plants or
river series, subject to a valid plant-to-caudal assignment. Candidate assets
include Coca Codo Sinclair, Paute/Molino, Sopladora, Mazar-related assets,
San Francisco and other major plants present in the reviewed assignment table.

The final selection will not be inferred from public plant names alone. It will
be determined from the repository’s assignment table and recorded in
`results/phase_08a/hydro_scope.csv`.

## Scenario matrix

The first matrix will contain six caudal-shortage scenarios:

| Scenario | Caudal multiplier in September–December |
|---|---:|
| `H00` | 1.00 |
| `H10` | 0.90 |
| `H20` | 0.80 |
| `H30` | 0.70 |
| `H40` | 0.60 |
| `H50` | 0.50 |

The same severity applies to each selected caudal series in the core matrix.
Later extensions may test basin-specific shocks, for example a stronger Paute
shock than Coca shock, but that should not be mixed into the initial threshold
experiment.

## Network and solver strategy

Each scenario will copy the solved Phase 06d reference network, replace the
2026 demand profile, and replace only the hydro `p_max_pu` columns generated by
the caudal-first workflow.

The following must remain fixed:

- network buses, lines and transformers;
- generator capacities and carriers;
- non-hydro availability profiles;
- Colombia interconnection ceiling and policy;
- load-shedding generator definition;
- solver and solver options.

The solved NetCDF is a valuable structural and dispatch reference, but a saved
solution is not assumed to provide a reusable HiGHS basis after changing demand
and hydro constraints. The first implementation will use clean scenario solves.
A separate warm-start benchmark may be attempted only if the installed PyPSA,
Linopy and HiGHS versions support it reproducibly. Warm-start results must be
compared against a clean solve for status, objective, shedding and dispatch.

## Planned implementation

```text
notebooks/08a_hydro_shortage_2026.ipynb
src/phase08a_hydro_shortage.py
results/phase_08a/
reports/figures/phase_08a/
reports/PHASE_08A_EL_NINO_DROUGHT_REPORT.md
```

Notebook stages:

1. common preflight and environment check;
2. source and checksum inventory;
3. 2025 demand basis and 2026 profile construction;
4. hydro fleet and caudal assignment audit;
5. baseline caudal and hydro-availability diagnostics;
6. monthly caudal perturbation;
7. rebuild with `build_hydro_availability()`;
8. scenario network construction and invariants;
9. full-year scenario solves;
10. energy balance, shedding, imports, thermal dispatch and CO₂ diagnostics;
11. scientific plots and technical conclusions.

## Required results and figures

The executed notebook will be saved at:

```text
results/phase_08a/notebooks/08a_hydro_shortage_2026.executed.ipynb
```

Scenario outputs will include:

```text
results/phase_08a/
├── hydro_scope.csv
├── caudal_scenarios.csv
├── hydro_profile_diagnostics.csv
├── scenario_registry.csv
├── scenario_summary.csv
├── shedding_by_month.csv
├── hydro_generation_by_scenario.csv
├── thermal_generation_by_scenario.csv
├── imports_by_scenario.csv
├── operational_co2_by_scenario.csv
├── H00.solved.nc
├── H10.solved.nc
├── H20.solved.nc
├── H30.solved.nc
├── H40.solved.nc
└── H50.solved.nc
```

Publication-quality figures will use Arial with a viridis-based palette,
external legends, short labels, and line-style redundancy for accessibility:

```text
reports/figures/phase_08a/
├── caudal_monthly_original_vs_shocked.png
├── hydro_availability_original_vs_shocked.png
├── shedding_vs_caudal_reduction.png
├── shedding_by_month_and_scenario.png
├── hydro_generation_vs_caudal_reduction.png
├── thermal_generation_vs_caudal_reduction.png
├── imports_vs_caudal_reduction.png
├── operational_co2_vs_caudal_reduction.png
├── representative_week_dispatch_by_scenario.png
└── hydro_scope_and_assignment_confidence.png
```

The primary figure will show caudal reduction on the x-axis and load shedding
on the y-axis, with a zero-shedding reference line. The report will include
both annual and monthly views so that a small annual shedding value does not
hide a concentrated September–December operational crisis.

## Scientific acceptance criteria

The experiment will not be accepted until:

- all scenario inputs have passed preflight;
- original and transformed caudal tables are retained;
- only selected caudal IDs and stress months differ from H00;
- the transformed flows are finite and non-negative;
- rebuilt hydro `p_max_pu` values are finite and within [0, 1];
- non-hydro profiles are unchanged;
- network topology and generator capacities are unchanged;
- the H00 case reproduces the selected reference within documented tolerances;
- every scenario records solver status and termination condition;
- annual and monthly energy balances reconcile;
- imports, thermal generation and shedding are separately reported;
- operational CO₂ uses versioned model emission factors;
- no shedding is counted as normal supply;
- scenario conclusions distinguish model evidence from drought assumptions;
- source checksums and transformations are recorded;
- the executed notebook and all report figures are saved.

## Interpretation framework

The main result is the onset and slope of shedding as the caudal shock grows.
The report will identify:

- the first scenario with non-zero shedding;
- the first scenario with material shedding;
- the months in which shedding occurs;
- the amount of hydro generation displaced;
- the thermal and import response;
- the associated operational CO₂ increase;
- whether the result is driven by annual energy shortage, hourly availability,
  imports, or the retained transmission model.

An optimal solver status will not be interpreted as adequate supply. Because
load-shedding generators preserve mathematical feasibility, adequacy will be
judged using shedding metrics and energy-balance diagnostics.

## Limitations

This is not a hydrological forecast. The caudals are preserved model input
series with documented provenance limitations. A uniform reduction across
selected river series does not capture basin-specific precipitation, reservoir
levels, evaporation, sediment constraints, turbine head, dispatch policy or
water-use competition. The existing three-month reservoir smoothing is retained
because the goal is to test the project’s existing profile methodology, not to
replace it with an unreviewed reservoir model.

The experiment also does not model the effects of rainfall on transmission,
landslides, flooding, thermal-plant cooling, fuel availability or Colombia’s
own hydrological constraints. Those are separate scenario dimensions.

## Decision gate before execution

Before running 08A, the owner should confirm:

1. whether the selected hydro plants are the intended three or four assets;
2. whether the core stress window is September–December 2026;
3. whether 2026 demand should equal 2025 demand or include a growth factor;
4. whether the 450 MW Colombia cap remains enabled for the drought experiment;
5. whether the scenario matrix should stop at 50% or include 60%.

No annual scenario solve should begin until these choices are recorded in the
notebook configuration and scenario manifest.
