# Ecuador Power System Scenarios 2022 Baseline to 2050
Scenario Registry Specification v1.0

## Project overview
This work defines a consistent and reproducible scenario registry for Ecuador power system studies in PyPSA.

Main goals
- Validate the consistency of an as-is reference network using 2022 demand and the existing PyPSA Ecuador network dataset.
- Define a scenario space for future years that reflects the 2022 master plan structure while acknowledging that, by 2026, most planned projects have not been realized.
- Run all declared scenarios in batch, store results in a structured way, and later expose results through a dashboard or web API.

Core idea
- Use 2022 as the calibration and sanity-check baseline.
- Re-base the 2022 master plan expansion window to start in 2030 to reflect observed delays.
- Explore futures through a small set of orthogonal scenario axes: demand, generation, transmission, nuclear, and stress tests.

## Data inputs and scope
This registry is designed to be backed by these sources
- Demand expansion chapter of the 2022 master plan, including demand hypotheses and singular loads.
- Generation expansion chapter of the 2022 master plan, including PEG additions and firm capacity needs.
- Transmission expansion chapter of the 2022 master plan, including PET reinforcements and methodology.
- Additional nuclear SMR pathway assumptions as a separate explicit axis.

Up to 2032, the plan is documented in the master plan documents.
Beyond that, the registry defines scenario assumptions, but keeps them consistent with the master plan logic.

## Anchor years
Anchor years are the reference points at which scenarios are defined, built, and solved.

Recommended anchors
- 2022: Baseline sanity check on the as-is grid and demand
- 2030: First wave of delayed master plan projects becomes visible
- 2035: Midpoint plus first SMR deployment wave
- 2040: Completion of the re-based master plan window plus second SMR wave
- 2050: Long-run endpoint, full deep decarbonization and final SMR wave

Note
- The original master plan horizon is 2023 to 2032. In this registry, that window is re-based to approximately 2030 to 2039.

## Delay model for master plan projects
Because the plan was published in 2022 and by 2026 most projects are delayed, the registry uses a transparent time-shift rule.

Re-base rule
- Define a plan re-base start year of 2030
- Original plan start year is 2023
- delay_years = 2030 minus 2023 = 7
- Any project year in PEG or PET is shifted as
  shifted_year = original_year + delay_years

Timing sensitivities
- Delay_M: delay_years = 5  optimistic catch-up
- Delay_B: delay_years = 7  baseline re-base
- Delay_S: delay_years = 9  pessimistic

Coupling rule for feasibility
- Default: transmission_delay_years equals generation_delay_years
- Optional: allow transmission to lead by up to 2 years in high-renewable cases
- Avoid cases where generation leads transmission unless explicitly intended as a stress test

## Scenario axes
The registry is designed as a cartesian product of small, interpretable axes.
Each axis is encoded by a short token.

### Axis A: Demand
Demand cases follow the demand chapter structure and add an explicit upside stress case.

- D1: Hypothesis 1 tendential demand
- D2: Hypothesis 2 case base, includes singular loads and planning assumptions used by PET
- D3: Hypothesis 2 plus Industrias Basicas upside, treated as a stress or high-growth case

Optional demand modifiers, only if parameterized
- D2_EMOB_H: higher e-mobility uptake than the base trajectory
- D2_EE_H: stronger energy efficiency than the base trajectory

### Axis B: Generation
Generation cases are designed to be implementable in PyPSA while remaining aligned with the master plan logic.

Base building blocks
- G0: As-is generation fleet for the baseline year 2022
- G_MP: Apply master plan PEG additions as written, but shifted by delay_years
- G_FIRM: Add master plan firm capacity additions and repowering requirements, shifted by delay_years
- G_THERM_STRESS: Reduce thermal availability or retire a share to represent aging risk
- G_ALT_HYDRO: Replace delayed large hydro pipeline share with alternative hydro candidates if needed
- G_REX: Extra renewables beyond PEG after the re-based window, for deep decarb pathways

Typical combinations
- 2030 to 2040 realistic baseline: G_MP plus G_FIRM
- Thermal risk stress: G_MP plus G_THERM_STRESS
- Deep decarb: G_MP plus G_REX, optionally combined with nuclear

### Axis C: Transmission
Network cases are built from the as-is grid and the master plan PET reinforcements.

- N0: As-is network from the PyPSA Ecuador dataset, baseline year topology
- N_PET: Apply PET reinforcements, shifted by delay_years
- N_PET_PART: Apply only critical reinforcements and replacements, delay the rest
- N_NUC: N_PET plus additional reinforcements for nuclear siting and integration

Notes on N_NUC
- Nuclear reinforcements are not defined in the master plan documents and must be declared explicitly as assumptions.
- Keep N_NUC separate so nuclear assumptions do not contaminate master plan baselines.

### Axis D: Nuclear SMR pathway
Nuclear is introduced as an explicit axis to provide firm low-carbon capacity beyond the re-based master plan window.

Capacity waves
- NU0: No nuclear
- NU1: 2035 plus 0.9 GW SMR, cumulative 0.9 GW
- NU2: 2040 plus 1.2 GW SMR, cumulative 2.1 GW
- NU3: 2050 plus 0.9 GW SMR, cumulative 3.0 GW

Integration mode
- Mode A: Additive, nuclear is additional firm capacity on top of the firm plan
- Mode R: Replacement, nuclear replaces part of thermal firm additions and repowering after 2035

Nuclear siting placeholder
- Each nuclear wave must specify a nuclear bus or region and a connection voltage level.
- Track this in registry fields nuclear_bus, nuclear_zone, and nuclear_grid_assumptions.

### Axis E: Stress tests and suffixes
Stress tests are applied as suffixes so the base scenario definition remains unchanged.

- DRY: Severe drought hydrology or reduced hydro availability
- OUT: Major outage of a plant or transmission corridor
- CR: Compound crisis, multiple simultaneous stresses
- FUEL: Fuel price shock or fuel supply constraint affecting thermal dispatch
- N1: N minus 1 style derating of key elements, optional

Example
- NUC_2040_D3_G_MP_G_FIRM_N_NUC_NU2_A_DelayS_DRY

## Recommended minimal scenario set
To keep early work manageable, start with a small set that spans the main axes.

Baseline and validation
- REF_2022_D2_G0_N0_NU0_DelayB

Delayed master plan baseline
- MP_2030_D2_G_MP_G_FIRM_N_PET_NU0_DelayB
- MP_2035_D2_G_MP_G_FIRM_N_PET_NU0_DelayB
- MP_2040_D2_G_MP_G_FIRM_N_PET_NU0_DelayB

Nuclear introductions
- NUC_2035_D2_G_MP_G_FIRM_N_NUC_NU1_R_DelayB
- NUC_2040_D2_G_MP_G_FIRM_N_NUC_NU2_R_DelayB
- NUC_2050_D2_G_MP_G_REX_N_NUC_NU3_R_DelayB

Demand growth sensitivity
- MP_2040_D1_G_MP_G_FIRM_N_PET_NU0_DelayB
- MP_2040_D3_G_MP_G_FIRM_N_PET_NU0_DelayB

Hydrology stress
- MP_2035_D2_G_MP_G_FIRM_N_PET_NU0_DelayB_DRY
- NUC_2040_D2_G_MP_G_FIRM_N_NUC_NU2_R_DelayB_DRY

## Scenario ID convention
Scenario IDs should be unique, sortable, and parseable.

Recommended pattern
- FAMILY_YEAR_Dx_G..._N..._NUx_MODE_DELAY_SUFFIXES

Where
- FAMILY is one of REF, MP, NUC, RE, MIX
- YEAR is the anchor year
- Dx is demand case token
- G tokens list generation building blocks
- N token is network case
- NU token is nuclear wave
- MODE is A or R when NU is not NU0
- DELAY is DelayM, DelayB, or DelayS
- SUFFIXES are optional stress tokens

Examples
- REF_2022_D2_G0_N0_NU0_DelayB
- MP_2030_D2_G_MP_G_FIRM_N_PET_NU0_DelayB
- NUC_2035_D2_G_MP_G_FIRM_N_NUC_NU1_R_DelayB
- NUC_2040_D3_G_MP_G_FIRM_N_NUC_NU2_A_DelayS_DRY

## Registry schema
Store the registry as a machine-readable table, for example CSV, Parquet, or YAML.
Each row defines one scenario and all information needed to build PyPSA inputs.

Required fields
- scenario_id
- anchor_year
- delay_case  DelayM, DelayB, DelayS
- delay_years
- demand_case  D1, D2, D3
- generation_case  list of tokens, eg G_MP plus G_FIRM
- network_case  N0, N_PET, N_PET_PART, N_NUC
- nuclear_case  NU0, NU1, NU2, NU3
- nuclear_mode  blank if NU0, else A or R
- stress_cases  list, can be empty
- notes

Recommended additional fields for traceability
- created_by
- created_on
- source_version
- base_dataset  dataset name or commit hash
- nuclear_bus
- nuclear_zone
- nuclear_grid_assumptions
- tags  freeform, eg baseline, realism, stress, deep_decarb

Example registry rows
| scenario_id | anchor_year | delay_case | demand_case | generation_case | network_case | nuclear_case | nuclear_mode | stress_cases |
|---|---:|---|---|---|---|---|---|---|
| REF_2022_D2_G0_N0_NU0_DelayB | 2022 | DelayB | D2 | G0 | N0 | NU0 |  |  |
| MP_2030_D2_G_MP_G_FIRM_N_PET_NU0_DelayB | 2030 | DelayB | D2 | G_MP plus G_FIRM | N_PET | NU0 |  |  |
| NUC_2035_D2_G_MP_G_FIRM_N_NUC_NU1_R_DelayB | 2035 | DelayB | D2 | G_MP plus G_FIRM | N_NUC | NU1 | R |  |
| NUC_2040_D3_G_MP_G_FIRM_N_NUC_NU2_A_DelayS_DRY | 2040 | DelayS | D3 | G_MP plus G_FIRM | N_NUC | NU2 | A | DRY |

## Output structure for batch runs
Even though implementation comes later, define a consistent path convention now.

Recommended folder structure
- results/
  - scenario_id/
    - inputs/
      - scenario_registry_row.json
      - built_network.nc
      - assumptions.yaml
    - outputs/
      - dispatch_timeseries.parquet
      - nodal_prices.parquet
      - line_loading.parquet
      - summary_metrics.json
    - logs/
      - solve.log
      - warnings.log

This structure makes it straightforward to build a dashboard or API layer later.

## Next steps
1) Implement the scenario registry table in a repository-managed file.
2) Implement builders that map demand_case, generation_case, network_case, nuclear_case into PyPSA modifications.
3) Run a first batch on the minimal scenario set, validate outputs, then expand the registry.
