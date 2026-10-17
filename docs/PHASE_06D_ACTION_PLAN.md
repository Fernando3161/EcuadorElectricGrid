# Phase 06D action plan — final iterative 2022 calibration

Branch: `v1-phase-06d-2022-final-iterative-calibration`  
Parent: `v1-phase-06c-2022-validation-accounting-availability`  
Status: planned

## Purpose and operating rule

Phase 06D is the final authorized calibration iteration for the Ecuador 2022
baseline. It will use the existing canonical notebooks under
`notebooks/00_*.ipynb` through `notebooks/06a_*.ipynb`. No separate 06D
notebooks will be created or executed. Reproducibility copies may be saved
under `results/phase_06d/notebooks/`, but the source workflow remains the
existing notebook series.

The objective is to reduce the total-variation error and bring wind, biomass,
and thermal generation into an acceptable calibration range while preserving a
solvable, internally balanced network. Hydro remains diagnostic and is not
refined through reservoir or cascade changes in this phase. The established
validation definitions and reported thresholds remain visible; the final report
must distinguish strict screen results from the owner-approved practical
calibration assessment.

## Starting point

The Phase 06C annual solve is optimal and internally balanced, but its carrier
comparison is not acceptable: wind is far below the official annual observation,
biomass is below its target, thermal generation is high, hydro is low, and the
931.43 GWh source-boundary residual remains unresolved. The residual must not be
encoded as losses merely to improve TVD.

## Iteration loop

Each iteration performs the following sequence using the same notebooks:

1. Update only the configuration/data tables or reusable source functions needed
   for the current candidate.
2. Run notebooks 00–05 to regenerate the processed inputs and assembled network.
3. Run the full-year notebook 06 and notebook 06a.
4. Record solver status, balance residual, load shedding, carrier totals,
   carrier percentage-point errors, TVD, curtailment, and the checksum of the
   solved network.
5. Compare the candidate against the previous candidate and retain the change
   only when it improves the declared objective without creating a material
   balance or feasibility regression.

At most four named candidates will be evaluated unless an iteration is clearly
making progress: `baseline`, `thermal-availability`, `renewable-calibration`,
and `joint-final`. Every candidate is retained as a result table, not silently
overwritten.

## Work package 1 — wind calibration

Notebook 03 will retain the NLR Sup3rWind profile comparison and the direct
turbine-converted profile as the physical basis. The annual wind output will be
compared with CENACE/CNEL. If the direct profile cannot meet the official annual
wind energy within the practical calibration tolerance, apply a documented
bounded annual scaling factor to the selected availability profile before
assembly. Scaling is a calibration assumption, not independent validation.

Requirements:

- preserve the raw NLR profiles and turbine parameters;
- report pre-scaling, post-scaling, and solved dispatch energy;
- clip availability to `[0, 1]` and preserve all 8,760 timestamps;
- do not scale the profile using solved dispatch in a circular way;
- record the factor and its provenance in the existing notebook 03 outputs.

## Work package 2 — biomass calibration

Notebook 03 will retain the ARCONEL monthly shape and apply an explicit annual
calibration factor to the biomass availability budget only if the unscaled
candidate does not reach the agreed biomass target. Biomass and biogas remain
separate carriers. The annual target, monthly shape, factor, potential,
dispatch, and curtailment must all be reported. No annual target will be forced
through a generator dispatch override.

## Work package 3 — thermal unavailability

Implement the CENACE 2022 monthly average thermal unavailable-capacity series as
a separate, source-backed availability scenario. The supplied dataset must be
checked against the official CENACE Figure 1.21 source and registered in the PDF
provenance table.

Because the public evidence is fleet-level rather than unit-level, the initial
allocation will be explicitly labeled a modeling assumption:

- distribute monthly unavailable MW proportionally across eligible thermal
  plants, bounded so no plant availability is negative;
- preserve the no-outage baseline for comparison;
- report monthly unavailable MW, fleet available MW, thermal dispatch, and
  generation change;
- do not call the allocation plant-level official evidence;
- keep dated CELEC events as supporting evidence, not as a fabricated complete
  outage history.

If proportional derating does not reduce thermal error sufficiently, test one
alternative allocation based on dispatch priority/capacity, document it, and
retain the better-behaved candidate. Do not alter marginal costs or remove
plants solely to pass validation.

## Work package 4 — total generation and losses

After wind, biomass, and thermal calibration, compare total simulated generation
with CENACE/CNEL production and the published loss/accounting terms. Use the
existing source-boundary bridge to separate demand, imports, exports,
curtailment, load shedding, and losses.

No explicit loss component will be added unless the source bridge identifies a
comparable quantity, metering point, and inclusion scope. If the difference is
still unresolved, retain it as report-only and state why it cannot be modeled.

## Acceptance and stopping criteria

The iteration stops when one of the following occurs:

1. Wind and biomass are within their practical annual calibration tolerances,
   thermal error is materially reduced, TVD is within the agreed acceptable
   range, the solve is optimal, and the internal balance closes; or
2. four documented candidates have been tested and no further evidence-backed
   change improves the objective without worsening another required metric.

Hydro is reported as an unresolved diagnostic in the final phase and is not
subject to reservoir/cascade refinement. The final report will include a table
for every candidate, unchanged definitions, calibration assumptions, strict
screen outcomes, practical acceptance interpretation, and remaining actions.

## Required outputs

- Updated existing notebooks 00–06a, with outputs retained where practical.
- Candidate comparison under `results/phase_06d/`.
- Monthly thermal availability and allocation tables.
- Wind and biomass calibration diagnostics from notebook 03.
- Updated source-boundary/loss comparison.
- PDF provenance entries for every consulted source.
- `reports/PHASE_06D_2022_VALIDATION_INTERIM_REPORT.md`.

The phase does not authorize reservoir/cascade redesign, undocumented plant
outages, arbitrary loss loads, threshold weakening, or creation of parallel
phase-specific notebooks.
