# Phase 06B interim report: NLR wind and biomass validation run

Date: 2026-10-17  
Branch: `v1-phase-06b-2022-validation-calibration`  
Environment: `pypsa-earth-ec`

## Result

Generated notebooks 00, 01, 02, 03, 04, 05, 06, and 06a completed successfully.
The annual 2022 solve used HiGHS and returned `ok` / `optimal` after 918,004
simplex iterations. The validation gate remains `FAIL`; comparability remains
`PARTIAL`; thresholds were not changed.

| Item | Result |
|---|---:|
| Snapshots | 8,760 hourly |
| Reference total | 28,684.61 GWh |
| Simulated total | 27,753.18 GWh |
| Total relative difference | 3.25% |
| TVD | 0.03980 |
| Load shedding | 0.0078 GWh |
| Energy-balance residual | 2.54e-10 GWh |
| Solved network SHA-256 | `7796ee0d3f145d754b5859bc94d67f2d0d88f3213d52bb302ed180828c029e23` |

## Carrier comparison

| Carrier | Reference GWh | Simulated GWh | Relative error | Share error (pp) |
|---|---:|---:|---:|---:|
| Hydro | 24,499.54 | 22,756.12 | -7.12% | -3.415 |
| Thermal | 3,393.35 | 4,349.36 | +28.17% | +3.842 |
| Nonconventional | 325.89 | 158.58 | -51.34% | -0.565 |
| Interconnection | 465.83 | 489.12 | +5.00% | +0.138 |

The detailed outputs are `results/phase_06/validation_2022.csv` and
`results/phase_06/validation_summary.json`. The unresolved 931.43 GWh source
boundary residual remains report-only, so total-energy screening remains
disabled.

## 06a diagnostic

| Carrier | Dispatch GWh | Potential GWh | Curtailment GWh |
|---|---:|---:|---:|
| Hydro | 22,756.12 | 26,894.11 | 4,137.99 |
| Oil | 4,349.36 | 18,062.07 | 13,712.71 |
| Biomass | 83.48 | 195.68 | 112.20 |
| Biogas | 37.00 | 72.88 | 35.89 |
| Onshore wind | 4.22 | 4.93 | 0.70 |
| Solar | 33.88 | 33.88 | 0.00 |

Biomass potential equals the imposed 195.68 GWh CENACE calibration control,
but dispatch is only 83.48 GWh. The monthly profile is therefore an availability
ceiling, not a forced production target, and biomass is not independent
validation evidence in this run.

The NLR comparison produced 37.96 GWh for Huascachaca and 3.58 GWh for
Villonaco before the model's commissioning treatment. The direct NLR profiles
are materially better resource evidence than the legacy 2.036 GWh combined
profile, but the current turbine curve and commissioning/capacity scope still
require review against CENACE/CNEL plant-level data.

## Recommended actions

1. Review hub height, turbine curve, commissioning dates, and plant capacity
   before scaling NLR profiles toward the 57.00 GWh CENACE reference. Do not
   force the annual total by arbitrary scaling.
2. Keep biomass, biogas, wind, and solar separate in the next validation review;
   aggregate nonconventional supply is affected by the biomass calibration
   control.
3. Investigate the 1,743.42 GWh hydro deficit using monthly hydro and 06a
   congestion diagnostics. Reservoir/cascade refinement remains deferred.
4. Investigate the 956.01 GWh thermal overproduction through plant-level
   availability, marginal costs, outages, and source-boundary definitions.
5. Keep the 5% thresholds unchanged and retain the 931.43 GWh residual as
   report-only until the CENACE/ARCONEL boundary bridge is closed.

## Reproducible outputs

- Annual validation: `results/phase_06/validation_summary.json`
- Solved network metadata: `results/phase_06/artifacts.json`
- 06a diagnostics: `results/phase_06a/dispatch_by_carrier.csv`
- Wind comparison: `results/phase_06b/wind_profile_source_comparison_2022.csv`
- Biomass diagnostics: `results/phase_06b/biomass_profile_diagnostics_2022.csv`

No thresholds were changed and no reservoir/cascade physics were added.
