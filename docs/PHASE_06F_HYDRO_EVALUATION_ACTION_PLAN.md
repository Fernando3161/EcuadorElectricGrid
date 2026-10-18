# Phase 06F — Hydro resource and dispatch evaluation

## Purpose

Phase 06F will determine why the 2022 model dispatches less hydroelectric
energy than the CENACE reference despite many hydro plants operating at their
hourly offered maximum. The phase will audit the chain from river-flow and
plant-factor assumptions through `p_max_pu`, plant commissioning, bus
attachment, marginal costs, network congestion, and annual dispatch.

This is an evaluation and evidence-repair phase. It must not force hydro
dispatch to the CENACE total, multiply profiles until the comparison passes,
or treat a copperplate result as the validated network. Any correction must
remain source-traceable and reproducible through the notebook workflow.

## Starting evidence

The current Phase 06D/06E evidence establishes:

- CENACE hydro reference: **24,499.54 GWh**;
- Phase 06D network hydro dispatch: **22,755.84 GWh**;
- hydro shortfall: **1,743.70 GWh**;
- 65 modeled hydro generators;
- all 65 plants reach their modeled available limit at least once;
- 61 of 65 plants are at that available limit for more than half of the year;
- hydro dispatch is approximately **22,755.8 GWh** against approximately
  **26,894.1 GWh** of modeled available potential;
- 84.6% of available potential is dispatched;
- all hydro generators currently use a marginal cost of 45 USD/MWh;
- copperplate dispatch increases hydro only to **23,116.08 GWh**, leaving a
  **1,383.46 GWh** shortfall;
- copperplating reduces thermal generation from 4,348.17 to 3,981.43 GWh,
  proving that topology contributes but does not explain the full hydro gap;
- thermal generation remains above CENACE by approximately 588 GWh even in
  the copperplate case;
- the total modeled supply remains approximately 931.4 GWh below the CENACE
  reported supply boundary.

These results mean that Phase 06F must distinguish resource underestimation,
dispatch economics, plant availability, and network effects before changing
inputs.

## Questions to answer

1. Are the river-flow series and plant factors plant-specific and appropriate
   for 2022, or are simplified CENACE/CNEL factors reused across plants?
2. Does each `p_max_pu` series represent physical water/resource availability,
   a plant factor, a generic seasonal proxy, or a commissioning mask?
3. Which plants are energy-limited, and which are power-limited?
4. Are plants dispatched at the available limit because of genuine resource
   scarcity, because their marginal costs are too low, or because the model
   lacks reservoir/water-value constraints?
5. How much hydro energy is lost to transmission congestion, curtailment,
   bus attachment, or infeasible temporal profiles?
6. Can the missing hydro be supported by CENACE/CNEL monthly or plant-level
   evidence without inventing an annual calibration multiplier?
7. Are hydro prices, opportunity costs, or water values causing the optimizer
   to dispatch the fleet incorrectly relative to thermal generation?

## Competing hypotheses

Each hypothesis must be tested independently before combinations are tried.
The test matrix must preserve the Phase 06D baseline and change only the
hypothesis variable.

| ID | Hypothesis | Controlled test | Evidence that would support it |
|---|---|---|---|
| H1 | River-flow or plant-factor inputs understate hydro energy | Replace only the affected evidence-backed factors/series | Available hydro GWh rises and dispatch gap closes without forced dispatch |
| H2 | Commissioning masks suppress eligible 2022 production | Correct only year eligibility masks | Missing plants or months become available with traceable dates |
| H3 | `p_max_pu` is an overly generic or clipped offer | Reconstruct plant-specific offers | Saturation pattern becomes physically plausible and plant outputs improve |
| H4 | Hydro marginal prices are wrong | Test documented price tiers and opportunity-cost alternatives | Hydro/thermal merit order changes without excessive shedding or curtailment |
| H5 | Missing reservoir or water-value constraints distort dispatch | Test a documented water-budget or reservoir representation | Seasonal dispatch improves without annual multiplier fitting |
| H6 | Transmission congestion suppresses hydro | Compare original topology with copperplate and targeted congestion cases | Hydro improvement is localized to constrained plants/corridors |
| H7 | Plant/bus attachment or capacity data are wrong | Audit plant coordinates, buses, voltage, and nominal capacity | Specific plants recover output after evidence-backed attachment correction |
| H8 | CENACE and model boundaries differ | Reconcile monthly totals, imports, exports, losses, and self-generation | The residual and carrier gap are explained by boundary differences |

No hypothesis passes merely because it reduces an error indicator. It must also
preserve feasibility, energy accounting, source provenance, and plausible plant
behavior.

## Work packages

### 1. Input and provenance inventory

Inventory every hydro input used by Notebooks 04/05 and the supporting modules:
plant capacity and commissioning year; plant location and bus; river/catchment
or weather proxy; flow series; plant factor and availability assumptions;
`p_max_pu` construction; source document, table/page, unit, and transformation.

Deliverable: `data/interim/hydro_evaluation_2022/hydro_input_lineage.csv`.

### 2. Plant-level offer reconstruction

Reconstruct independently of solved dispatch:

`available_mw[t, plant] = p_nom[plant] * p_max_pu[t, plant]`

Report minimum, mean, maximum, annual available GWh, seasonal totals, and
commissioning eligibility for every plant. Confirm that the available series
does not contain post-commissioning energy or unintended clipping.

Deliverable: `hydro_offer_diagnostics.csv` and monthly/plant figures.

### 3. Saturation and dispatch attribution

Compare dispatch with available power for every plant and hour. Classify each
plant as resource-limited, power-saturated, transmission-constrained,
economically displaced, disconnected/unavailable, or unresolved. Use both the
original network and copperplate diagnostic, including line loading, prices,
curtailment, shedding, and plant output.

Deliverable: `hydro_dispatch_attribution.csv` and saturation-duration plots.

### 4. CENACE/CNEL evidence reconciliation

Register official sources for monthly or plant-level hydro generation,
installed capacity, operating dates, river-flow statistics, plant factors, and
outages. Preserve original names, values, units, pages/tables, editions, and
source status. Separate official facts from modeling assumptions.

Deliverable: `hydro_reference_evidence.csv` and a source coverage matrix.

### 5. Controlled sensitivity experiments

Run reproducible cases: original network/offer; copperplate/offer; corrected
plant factors only; corrected commissioning masks only; documented cost or
water-value changes only; and the combined evidence-supported correction.
Preserve unrelated inputs in each case and record solver status, balance,
hydro/thermal generation, saturation, losses, curtailment, shedding, and CENACE
indicators.

The sensitivity matrix must explicitly include hydro price cases:

- current uniform hydro price;
- source-supported plant or technology price tiers;
- low hydro opportunity-cost case;
- high hydro opportunity-cost case;
- seasonal or water-value proxy, only if documented and clearly labeled as a
  modeling assumption.

Price tests must report the marginal-cost ordering against thermal, imports,
biomass, wind, and solar. They must also report whether changing price merely
switches generation between hydro and thermal without resolving the hydro
resource shortfall.

### 6. Full sequential notebook reruns

After the diagnostic matrix identifies a candidate solution, run the complete
approved notebook sequence from Notebook 00 through the applicable validation
stage in the `pypsa-earth-ec` environment. Do not validate a hydro change only
with a patched NetCDF or an isolated solve.

For every full rerun, retain:

- executed notebook copies and logs;
- artifact preflight and checksums;
- solver status and termination condition;
- input configuration and hypothesis ID;
- network checksum;
- hydro, thermal, renewable, imports, exports, losses, curtailment, storage,
  shedding, and balance results;
- TVD and carrier percentage-point diagnostics;
- before/after comparison against the Phase 06D baseline.

At least the baseline and every owner-approved candidate solution must complete
the full sequence. Failed runs remain documented with their logs and failure
point.

### 7. Correction registry and implementation

Create `hydro_corrections.csv` with original and proposed values, units,
transformations, source locators, confidence, expected effect, and owner-review
status. Implement only approved corrections through the notebook pipeline; do
not patch solved NetCDF files directly.

### 8. Full-year validation handoff

After approval, rerun all applicable notebooks and compare against Phase 06D.
Report whether the change improves hydro without transferring the discrepancy
into thermal generation, imports, shedding, curtailment, or the 931.4 GWh
accounting residual.

## Acceptance criteria

Phase 06F is complete only when every hydro plant has input lineage; every
hourly offer and dispatch series has been evaluated; saturation is quantified
against nominal and available capacity; resource, economics, topology, and
accounting effects are separated; hydro prices and water-value assumptions are
checked; all competing hypotheses have controlled results; the baseline and
owner-approved candidates have completed the full notebook sequence; CENACE/CNEL
evidence has exact locators; no unsupported multiplier or forced dispatch is
introduced; corrections are source-traceable and owner-reviewed; at least one
defensible solution path is identified or ruled out; and the report
distinguishes explained from unresolved hydro discrepancy.

## Planned deliverables

- `docs/PHASE_06F_HYDRO_EVALUATION_ACTION_PLAN.md`
- `data/interim/hydro_evaluation_2022/hydro_input_lineage.csv`
- `data/interim/hydro_evaluation_2022/hydro_offer_diagnostics.csv`
- `data/interim/hydro_evaluation_2022/hydro_dispatch_attribution.csv`
- `data/interim/hydro_evaluation_2022/hydro_reference_evidence.csv`
- `data/interim/hydro_evaluation_2022/hydro_corrections.csv`
- `results/phase_06f/` controlled-run outputs
- `reports/PHASE_06F_HYDRO_EVALUATION_REPORT.md`
