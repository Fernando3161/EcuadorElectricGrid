# Phase 06H — Final calibration report

## Executive result

The canonical Stage 00→06a workflow was executed on branch `v1-pahe-06h-consolidation` from 06G commit `f1a9e871d4df04f08124d763e41f7a3d90fdbd2d`. The full 8,760-hour HiGHS solve completed with status `ok` and termination `optimal`. The result is a **provisionally calibrated 2022 baseline**, not an independently validated historical physical network.

The direct upstream hydro change reproduces the Phase 06G ROR-Q90 composition result: TVD **0.00830452**, versus **0.03980131** for the matched pre-Q90 formulation. Q90 passes the configured TVD, hydro, thermal and interconnection screens, but fails the nonconventional share screen: **−0.637 pp** versus the **0.5 pp** limit. Therefore the scientific acceptance status remains **FAIL/PARTIAL**, while Q90 is the preferred research baseline for scenario exploration.

## What was retained

The 06A/06B source-supported La Concordia–Pedernales, Minas San Francisco–La Unión–San Ildefonso and Chongón–Santa Elena connectivity repairs remain in the canonical topology. The 06E capacity-audit registry remains unused because it did not establish source-defensible ratings. The 06D aggregate thermal derating is not applied. Imports remain constrained to 465.83 GWh and Colombia/Peru exports remain fixed sinks totaling 191.57 GWh.

Stage 03 now uses the exact 06G ROR method directly: each assigned monthly flow series is divided by its empirical 0.90 quantile and clipped to [0,1]. Reservoirs retain trailing-three-month smoothing and the original 0.95 quantile. This is a modeling normalization of monthly magnitudes, not observed rated flow and not hydrological Q90 exceedance. NLR Sup3rWind 2022 site CSVs are required for both wind plants; the legacy `onwind.nc` fallback and renewable candidate switch are removed from the active path. ARCONEL monthly biomass shape and the 195.68 GWh annual control are retained, explicitly as calibration input rather than independent validation.

## Comparable carrier results

The machine-readable carrier table is [comparison_carriers.csv](../results/phase_06h/comparison_carriers.csv). C is the matched historical Phase 06G BASE control; B is the freshly rebuilt canonical Q90 baseline. A is shown as the retained original-06 evidence identity; it is numerically represented by the same pre-Q90 historical control, not claimed as a fresh rerun of the old branch.

| Case | Hydro GWh | Thermal GWh | Nonconventional GWh | Imports GWh | Exports GWh | TVD | Shedding GWh | Solver |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A original Phase 06 evidence / pre-Q90 | 22,756.12 | 4,372.65 | 158.58 | 465.83 | 191.57 | 0.03980 | 0.00782 | historical optimal |
| C matched G06G_BASE | 22,756.12 | 4,372.65 | 158.58 | 465.83 | 191.57 | 0.03980 | 0.00782 | optimal |
| B fresh 06H ROR-Q90 | 23,919.36 | 3,229.49 | 138.51 | 465.83 | 191.57 | **0.00830** | **0.00000** | optimal |

For B, total comparable supply is 27,753.19 GWh versus 28,684.61 GWh in the reference table, a 3.247% boundary difference. The numerical network balance residual is 2.5×10⁻¹⁰ GWh. The 931.43 GWh CENACE statistical boundary residual is not modeled as losses. No physical network-loss series is represented in this result.

Relative to the reference mix, B has hydro −580.18 GWh / +0.776 pp, thermal −163.86 GWh / −0.193 pp, nonconventional −187.38 GWh / −0.637 pp, and interconnection approximately 0 GWh / +0.055 pp. The sum of absolute share errors is 1.662 pp; TVD is half of that, 0.830 percentage points.

## Figures

![Carrier comparison](figures/phase_06h/carrier_gwh_comparison.png)

![Share errors](figures/phase_06h/share_error_comparison.png)

The plots show the key scientific trade-off: Q90 moves hydro toward the reference and reduces thermal overproduction, but it does not solve the independently unresolved nonconventional deficit. The improvement is therefore a composition-fit sensitivity, not proof that the underlying proxy flows equal plant hydrology.

## Registries, provenance and executed evidence

The canonical execution copies are in `results/phase_06h/notebooks/`, covering notebooks 00, 01, 02, 03, 04, 05, 06 and 06a. The solved network is `results/phase_06/network_solved_2022.nc`; its artifact metadata is in `results/phase_06/artifacts.json`. Stage 03 product fingerprints were updated in `config/artifacts.yaml` after deterministic regeneration. The active workflow no longer selects `baseline`, `renewable-calibration`, or `thermal-availability` candidates.

The retained limitations are material: eight legacy monthly hydro series have unverified primary origin; 65 plants use 4 direct, 5 same-system and 56 low-confidence regional proxy assignments; Q90 is model-tuned; wind uses simplified curves and may contain calibration leakage in its historical source selection; biomass potential and dispatch differ; plant-hour thermal outage data are unavailable; historical line/transformer ratings remain unresolved; and the 931.43 GWh accounting boundary remains open.

## Recommendation and Stage 07 handoff

Adopt **ROR-Q90** as the single provisional research baseline for exploratory Stage 07 scenarios, with the qualification that it is source-constrained and composition-improved, not independently validated. Do not present the baseline as passing the historical acceptance gate. Future work requiring a validated historical claim needs primary plant-level hydro observations/rated flows, plant-hour thermal availability, and a resolved CENACE metering boundary; further percentile tuning alone should not be treated as evidence.

