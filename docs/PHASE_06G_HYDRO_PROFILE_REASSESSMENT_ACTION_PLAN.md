# Phase 06G — Hydro profile and capacity reassessment

## Status and purpose

This is a planning and evidence-review phase. It starts from remote Phase 06F
commit `752dda742a72ec75212a160f33428e0a027e39bf` on the typo-named branch
`v1-pahse-06f-hydrology-evaluation` and is carried by
`v1-phase-06g-hydro-profile-reassessment`. No model correction is approved by
this plan. The purpose is to determine whether hydro offers, plant factors,
capacity definitions, commissioning eligibility, or accounting boundaries
explain the modeled hydro/thermal mix without fitting profiles to an annual
target.

The phase preserves the project-wide acceptance gate: complete sequential
workflow, successful solver termination, explicit energy balance, cited 2022
carrier comparison, 5% tolerance for every in-scope carrier, provenance, and
checksums. Every hypothesis evaluated in execution requires a complete 2022
run with all 8,760 snapshots. A week, month, representative period, patched
solved NetCDF, or smoke run is not evidence of hypothesis impact.

## Scope and exclusions

In scope are the mainland SNI 2022 hydro fleet, flow/profile construction,
plant-to-flow assignments, nominal/effective capacity and commissioning data,
documented outages, offer/dispatch attribution, retained topology diagnostics,
and reference-accounting boundaries. Hydro and thermal shares are primary
outputs; absolute GWh are context.

Out of scope are topology changes, endogenous expansion, detailed reservoir
optimization without traceable parameters, nuclear or other hypothetical
Master Plan assets, and any unsupported annual multiplier or forced dispatch.
The Phase 06F price sweep is not repeated broadly unless a specific unresolved
question is found in the audit.

## Verified Phase 06F starting evidence

The remote head contains the report, action plan, runner, input tables, figures,
and experiment summaries. The exact full-year baseline is `F06F_P45`: 22,754.54
GWh hydro and 3,043.40 GWh thermal, with solver status `ok` and termination
`optimal`. `F06F_P00` gives 22,767.02 GWh hydro and 3,043.27 GWh thermal.
`F06F_A120` gives 24,504.00 GWh hydro and 1,837.80 GWh thermal, close to the
24,499.54 GWh CENACE reference only by coincidence; it is an unsupported
sensitivity and is not validation. `F06F_A100` gives 26,880.05 GWh hydro and
838.04 GWh thermal and is retained only as an all-hydro-`p_max_pu = 1`
upper-bound diagnostic. Reuse either result only after input, settings,
snapshot count, hypothesis coverage, and network checksum verification.

Evidence paths:

- `reports/PHASE_06F_HYDRO_EVALUATION_REPORT.md`
- `docs/PHASE_06F_HYDRO_EVALUATION_ACTION_PLAN.md`
- `src/phase06f.py`, `src/availability.py`
- `data/interim/hydro_evaluation_2022/`
- `results/phase_06f/`, including `experiment_summary.csv`
- `results/phase_06d/baseline/network_solved_2022.nc`
- `data/interim/generation/hydro_profile_assignments_2022.csv`
- `notebooks/03_prepare_renewable_and_hydro_profiles.ipynb` and
  `notebooks/03_c_hydro_profiles.ipynb`

The assignment file has 65 rows, eight flow-series identifiers, and confidence
counts of 4 high, 5 medium, and 56 low. The active production notebook reads
`data/raw/generation/caudals.csv` and the assignment table; `src/availability.py`
implements monthly flow divided by the 95th percentile, clipping to `[0, 1]`,
a trailing three-month mean for reservoir plants, mid-month monthly anchors
with time interpolation to hourly snapshots, and pre-commissioning masks.
Offers are reconstructed as `p_nom * p_max_pu`; Phase 06G must audit where
effective capacity enters before treating it as a production input.

Phase 06F lineage is incomplete for scientific reassessment: annual mean
profiles are blank in the retained lineage, and many source/plant-level fields
are generic or modeling assumptions. Dispatch below offered availability alone
does not prove curtailment or hydrological scarcity; attribution must separate
offered power, dispatch, verified curtailment, network limits, economic
displacement, outages, and unresolved cases.

## Evidence register and audit design

Create or extend a versioned evidence register with: source organization and
document, edition/date, period, page/table/figure, original name/value/unit,
claim supported, transformation, confidence, official-fact versus assumption,
and limitation. Inspect `cap_3_demanda.pdf`, `cap_4_generacion.pdf`, and
`cap_5_transmision.pdf` where available, plus the CENACE 2022 source already
registered by Phase 06F. Do not add PDFs unless licensing permits it. The 2023
Ministry aggregate (5,186.24 MW nominal; 5,146.52 MW effective) is a dated,
definition-mismatched contextual fact, not a verified 2022 plant correction.

The audit must produce plant-level rows covering generator ID/name, source
capacity fields, nominal/effective/turbine distinctions, bus and voltage,
commissioning date and 2022 eligibility, flow source, assignment basis and
confidence, profile method, annual mean and offered GWh, outage evidence, and
all locators/checksums. It must explicitly identify missing evidence rather than
fill gaps with assumptions.

## Ordered experiment matrix

All rows below require the same complete sequential preflight → profile build →
network assembly/repair → solve → accounting → carrier comparison workflow,
using the approved `pypsa-earth-ec` environment. Each run must have 8,760
snapshots, an experiment ID, one controlled change, unchanged assumptions, input
manifest, solver log/status, network and key-input SHA-256 hashes, executed
notebooks, and retained results. A directly applicable Phase 06F result may be
reused only after verification against these requirements.

| Order / ID | Hypothesis and controlled change | Evidence required and unchanged assumptions | Required outputs and decision rule |
|---|---|---|---|
| 0 AUDIT | Read-only lineage, capacity, profile, commissioning, and source audit | Phase 06F artifacts, active notebooks, PDFs/CENACE; no model change | Coverage matrix; identify testable versus untestable hypotheses and missing inputs |
| 1 BASE | Exact Phase 06F baseline reproduction or verified reuse | Retained topology, offers, costs, boundaries, 2022 demand and all other inputs unchanged | Full balance and carrier/share baseline; stop if checksums or settings cannot reproduce the claimed result |
| 2 A100 | All hydro `p_max_pu = 1` upper bound | Verified `F06F_A100`, or full rerun; diagnostic only | Hydro/thermal shares and plant saturation; must never be promoted as correction |
| 3 CAP | Source-supported effective/nominal capacity and commissioning cases, one field/change at a time | Plant-level official evidence; no fleet-wide 2023 substitution | Plant and carrier deltas, eligibility impacts, feasibility; stop unsupported rows |
| 4 ROR | One-at-a-time run-of-river normalization, clipping, and hourly construction alternatives | Plant/basin/flow evidence; no target-GWh fitting | Annual/seasonal offers, shares, saturation and balance; only evidence-supported cases advance |
| 5 RES | One-at-a-time alternatives to trailing three-month reservoir profile | Traceable reservoir/flow evidence; no inferred physics from dispatch | Seasonal dispatch and offer changes; absence of evidence is reported, not guessed |
| 6 ASSIGN | Confidence/plant-group and evidence-supported alternative assignments | Existing 4/5/56 confidence groups; bounded uncertainty only for unresolved mappings | Plant/group contribution, offer and share deltas; label sensitivities, not corrections |
| 7 OUTAGE | Documented 2022 outage/maintenance masks separated from water availability | Dated plant-level outage evidence; unchanged hydrology | Outage energy, offers, dispatch, shares and residuals; no outage inference from low dispatch |
| 8 TECH | Traceable flow/head/efficiency constraints where parameters exist | Source-located parameters only; otherwise bounded sensitivity with explicit assumption | Offer attribution and shares; otherwise record missing evidence |
| 9 TEMP | Month-centering, interpolation, year-boundary and commissioning-mask alternatives | Calendar/profile evidence; one processing choice changed at a time | Seasonal and annual offer effects; check no 8,760-hour or boundary defect |
| 10 DISP | Revised attribution on retained topology: offered power vs dispatch, congestion, displacement, outage and curtailment | Same offers and topology as baseline; full-year outputs | Plant/hour categories, line loading, prices, shedding, curtailment; no scarcity claim from dispatch alone |
| 11 NET | Retained-topology diagnostics to quantify network/dispatch contribution | No topology edits; copperplate or diagnostic comparison only where already verified or fully rerun | Localized constraint effects and carrier shares; do not add lines/transformers |
| 12 ACCT | Full-year accounting-boundary recomputation | Verified full-year model result; reclassify imports/exports, losses, own-use, storage, shedding and residuals | Denominator and carrier-universe sensitivity; accounting-only cases cannot change dispatch |
| 13 COMB | Combined evidence-supported case after isolated tests | Only owner-approved corrections with source traceability; all other baseline inputs unchanged | Full workflow, balance, shares, TVD/5% screens, provenance; advance only if scientifically defensible |

For each testable hypothesis, the execution registry must name one or more
scenario IDs and a directly corresponding full-year run. If evidence cannot
define a defensible scenario, mark it `not testable`, identify the missing
input, and do not claim a result. A uniform ×1.20 correction, “design power ×
1.20,” forced dispatch, or profile fitted to CENACE annual GWh is prohibited;
only a clearly labeled diagnostic multiplier may be retained for context.

## Metrics and accounting

For each run report hydro and thermal generation shares first, then all agreed
carriers, imports/exports, storage, losses, curtailment, load shedding, demand,
and residual. For a declared carrier universe and denominator, report model
share, reference share, percentage-point difference, relative share error where
defined, and absolute GWh difference. Recompute each documented boundary rather
than silently mixing CENACE supply, model generation, and net imports. State
whether own-use, isolated systems, autoproducers, storage, losses, exports, or
residual categories are excluded or included. Preserve the project-wide 5%
tolerance; do not omit failing carriers.

## Reproducibility deliverables

Later execution should retain a manifest and checksum record, preflight log,
experiment registry, plant audit, evidence register, profile/offer diagnostics,
dispatch attribution, accounting table, executed notebook copies, solver logs,
per-run summaries, network NetCDF checksums, comparison tables, and validation
figures under phase-specific `data/interim/`, `results/phase_06g/`, and
`reports/figures/phase_06g/` paths. Reports must distinguish verified facts,
hypotheses, assumptions, sensitivities, and unresolved evidence.

## Review gate and recommendation

Before implementation, the owner reviews the evidence register, plant-level
audit, experiment registry, denominators/boundaries, and the claim that every
testable hypothesis has a complete 8,760-hour run. No correction enters the
production pipeline until its source locator, transformation, confidence,
expected effect, and owner approval are recorded. The recommended next phase
is execution only after this gate; if primary plant/river evidence remains
unavailable, retain the current offers and document the unresolved discrepancy.

## This branch task completion boundary

This branch creates the plan and branch only. No simulation, notebook
execution, model implementation, commit, or push is performed here.
