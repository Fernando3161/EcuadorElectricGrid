# Phase 06C — 2022 validation interim report

Branch: `v1-phase-06c-2022-validation-accounting-availability`  
Execution date: 2026-10-17  
Scope: availability, accounting-boundary, and provenance investigation. Reservoir and cascade refinement remains deferred; validation thresholds were not changed.

## Execution result

The generated notebooks 00–06a were executed in `pypsa-earth-ec`, with executed copies retained under `results/phase_06c/notebooks/`. The annual optimization completed successfully:

- solver status: `ok`
- termination: `optimal`
- HiGHS iterations: 918,004
- internal energy-balance residual: `2.54e-10 GWh`
- load shedding: `0.0078 GWh`

The scientific validation gate remains **FAIL** and comparability remains **PARTIAL**. The run is therefore not an acceptance baseline.

## Validation summary

| Carrier / metric | Reference GWh | Simulated GWh | Relative error |
|---|---:|---:|---:|
| Hydro | 24,499.54 | 22,756.12 | -7.12% |
| Thermal | 3,393.35 | 4,349.36 | +28.17% |
| Nonconventional | 325.89 | 158.58 | -51.34% |
| Interconnection | 465.83 | 489.12 | +5.00% |
| Total supply | 28,684.61 | 27,753.18 | +3.25% absolute relative error |

The total-variation distance is `0.03980`; the unchanged screening checks fail TVD and interconnection. The 931.43 GWh source-boundary residual remains report-only and is not modeled as losses.

## 06C findings

### Wind and biomass

Notebook 03 continues to select the direct NLR Sup3rWind site profiles after turbine conversion and commissioning masks. The solved dispatch is 4.22 GWh onwind, while the NLR-derived potential is 4.93 GWh. The profile is therefore passed into assembly as a physical availability product, not scaled to force the CENACE/CNEL annual observation. Biomass retains the ARCONEL monthly shape with the CENACE annual control explicitly labeled as a calibration input; dispatch is 83.48 GWh against 195.68 GWh potential because the optimization curtails the remainder.

### Thermal availability

The new evidence table is `data/interim/availability/thermal_availability_evidence_2022.csv`. CENACE reports aggregate unavailable thermal capacity averaging 627.79 MW in 2022, with a 944.6 MW peak at 16:00 on 7 October; CELEC notices document dated restoration or maintenance events at Jaramijó, Aníbal Santos, Termogas Machala, and Esmeraldas. These sources do not provide a complete, plant-resolved hourly 2022 series. Consequently, no unsupported outage derating was encoded in the baseline.

### Accounting boundary

`results/phase_06c/source_boundary_bridge_2022.csv` records the model terms and the published CENACE terms separately. The PyPSA balance closes independently, while the 931.43 GWh source residual remains unresolved. It must not be represented as a loss load without a source-backed equivalent metering boundary.

### PDF provenance

The retroactive registry is `data/interim/provenance/pdf_source_registry.csv`. The ARCONEL *Panorama Eléctrico 2023, Edición 14* PDF used for the unavailable-capacity methodology was downloaded to `literature/ARCONEL/panorama_electrico_2023_ed14.pdf` and SHA-256 registered. Existing CENACE and Plan Maestro PDFs used by the workflow are also registered with local paths, sizes, hashes, and locators.

## Recommended next actions

1. Obtain the CENACE SAMWEB unit-level maintenance export for all of 2022 and map unit IDs to the aggregated fleet rows before implementing an evidence-based thermal availability variant.
2. Keep the no-outage baseline as the comparison anchor; test the dated-event and aggregate-availability variants separately and report plant/month/GWh impacts.
3. Reconcile CENACE net supply, ARCONEL gross production, imports, exports, distribution availability, final consumption, station service, and losses at common metering points before enabling total-energy screening.
4. Decompose nonconventional production by wind, biomass, biogas, and solar against boundary-compatible monthly observations; do not force dispatch to annual targets.
5. Investigate the hydro shortfall with plant-level potential/dispatch/curtailment and copperplate or transmission-relaxed diagnostics only. Reservoir and cascade refinement remains outside this phase.
6. Preserve the executed notebook copies and current FAIL/PARTIAL result as the reproducible 06C handoff; do not relax thresholds or claim scientific acceptance.

## Reproducibility artifacts

- Executed notebooks: `results/phase_06c/notebooks/`
- Validation result: `results/phase_06/validation_summary.json`
- Carrier comparison: `results/phase_06/validation_2022.csv`
- Dispatch diagnostics: `results/phase_06a/dispatch_by_carrier.csv`
- Thermal evidence: `data/interim/availability/thermal_availability_evidence_2022.csv`
- Accounting bridge: `results/phase_06c/source_boundary_bridge_2022.csv`
- PDF registry: `data/interim/provenance/pdf_source_registry.csv`
