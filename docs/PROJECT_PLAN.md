# Project plan

Development is phase-gated. Every phase has a dedicated branch, focused tests,
acceptance criteria, a technical report, and owner review before merge. Branch
names below are the canonical recommendation; the owner may map an already-created
branch to a phase explicitly.

## Phase 00 — Forensic diagnostic and governing documentation

Branch: `refactor/v1-phase-00-diagnostic`

Heavy execution: forbidden.

Deliverables:

- root `AGENTS.md` and an honest status `README.md`;
- project objectives, development workflow, data/artifact contract, and report
  template;
- `docs/PHASE_00_DIAGNOSTIC.md` with repository evidence, gap analysis, risks,
  known external files, and recommended target workflow;
- this phase plan.

Tests: Markdown/path/link sanity checks and read-only environment/artifact probes.

Acceptance: owner confirms the diagnostic, notebook-led architecture, environment
policy, scope, and phase order. No repository cleanup occurs in this phase.

Report: `reports/PHASE_00_TECHNICAL_REPORT.md` after owner review resolves Phase 0
decisions.

## Phase 01 — Evidence preservation and repository structure

Branch: `refactor/v1-phase-01-structure`

Heavy execution: forbidden.

Tasks:

- classify every notebook, module, table, network, result, and source document;
- map the working 2017/2018 chain and partial 2022 chain as a dependency graph;
- identify reusable manual repairs, mappings, and PDF extractions;
- establish target directories and naming conventions;
- move/archive/remove only after provenance is migrated and owner approval is
  recorded;
- separate official Master Plan rows from hypothetical nuclear additions;
- create initial schemas and small data-quality tests.

Acceptance: no scientific evidence is lost; active and legacy paths are clear;
notebook-directory outputs are mapped to proper data zones; supported code has no
machine-specific path; test suite runs cheaply.

## Phase 02 — Artifact manifest and mandatory preflight

Branch: `refactor/v1-phase-02-data-preflight`

Heavy execution: limited to owner-authorized publication/download verification.

Tasks:

- prepare the external manifest for the required upstream `base.nc`, `solar.nc`,
  and `onwind.nc` artifacts;
- exclude the full ERA5/atlite cutout from the student preflight and keep its
  regeneration responsibility in the companion data project;
- exclude current `ec_network_2022.nc` and `network_base_filled.nc` instances from
  the source/input publication set because they are locally generated,
  scientifically unvalidated products;
- prepare Zenodo-ready release inventory, provenance, and checksums;
- owner publishes artifacts and supplies persistent record/file URIs;
- implement configuration/manifest, checksum validation, safe cached download,
  atomic placement, and dependency closure by stage;
- implement a Notebook 00/preflight entry point used by every later notebook;
- document use of the existing `pypsa-earth-ec` environment and reconcile its
  tested package set with `config/environment_ec.yaml` without installing.

Tests: manifest/schema, existing-file detection, checksum failure, mocked download,
atomic placement, offline behavior, and stage dependency resolution.

Acceptance: before Stage 1, one command/check reports all required data and obtains
all missing publishable artifacts without later manual interruption.

## Phase 03 — Demand and statistical reference

Branch: `refactor/v1-phase-03-demand-statistics`

Heavy execution: forbidden.

Tasks:

- refactor the 2022 demand profile/projection work into explained notebooks plus
  tested functions;
- establish the official 2022 annual demand/supply statistical reference table;
- define system boundary and units;
- freeze carrier taxonomy and mapping, including imports/exports, biomass,
  non-renewable thermal fuels, isolated systems, storage, and residuals;
- define the multi-criteria TVD acceptance model and zero/small-carrier treatment;
- validate hourly totals, peak, annual energy, spatial allocation, and leap/time
  conventions.

Tests: profile scaling, time index, units, aggregation, carrier mapping, and
tolerance calculation.

Acceptance: demand and the statistical comparison contract are source-traceable,
machine-readable, and independently testable.

## Phase 04 — Existing generation and availability profiles

Branch: `refactor/v1-phase-04-generation-profiles`

Heavy execution: only short/profile checks unless explicitly authorized.

Tasks:

- normalize the 2022 existing plant fleet with source-row provenance;
- reconcile plant names, carriers, capacities, coordinates, commissioning and
  retirement fields;
- refactor hydro matching and availability estimation;
- attach verified solar/onshore-wind profiles and document the 2013 weather-year
  choice relative to a 2022 simulation;
- save match-quality and generation-capacity validation plots;
- quarantine future and nuclear assets from the baseline.

Tests: schemas, capacity reconciliation, carrier mapping, profile bounds and
coverage, hydro matching, unmatched/duplicate plants, and time alignment.

Acceptance: every baseline generator and time series is traceable and valid; no
future/hypothetical capacity leaks into 2022.

## Phase 05 — Network repair and topology validation

Branch: `refactor/v1-phase-05-network`

Heavy execution: no annual solve.

Tasks:

- reproduce the useful pruning/manual-fix logic from the legacy consistency
  notebooks as explicit transformations;
- establish whether supported voltage scope is 138/230/500 kV and document any
  necessary exception;
- validate buses, branches, transformers, electrical parameters, connected
  components, geographic coordinates, duplicate assets, and orphan handling;
- attach loads/generators without silent nearest-bus decisions;
- save before/after topology and diagnostic figures;
- produce a deterministic candidate assembled 2022 network.

Tests: repair rules on fixtures, invariants, topology, attachments, serialization,
and structural fingerprint of the candidate artifact.

Acceptance: the candidate network passes consistency checks, reloads identically,
and every manual mutation has evidence/rationale. The candidate remains a
generated product and is not published as authoritative until Phase 6 passes.

## Phase 06 — 2022 solve, calibration, and scientific validation

Branch: `refactor/v1-phase-06-2022-validation`

Heavy execution: owner-authorized short and full-year runs.

Tasks:

- create explained assembly/smoke-run and full-year validation notebooks;
- define objective, constraints, snapshot weighting, solver configuration, and
  reproducibility metadata;
- diagnose infeasibility/load shedding without concealing defects;
- quantify whether the contracted 69 kV load paths materially affect feasibility,
  congestion, losses, or load shedding; retain selective 69 kV restoration as a
  documented future refinement unless the owner separately approves its
  evidence-backed implementation;
- implement annual energy balance and carrier comparison;
- calibrate only through source-supported parameters and record each decision;
- save solved network, summaries, logs, tables, and validation plots.
- after acceptance, prepare the validated assembled network for optional versioned
  publication as a generated release artifact, clearly separated from upstream
  source inputs.

Tests: tiny solve, energy accounting, KPI extraction, result schema, and statistical
comparison. Full 8,760-hour solve remains a recorded manual scientific run.

Acceptance: network runs; energy balance reconciles; all agreed carrier comparisons
pass the configured TVD/carrier screens; material load shedding and unexplained
residuals are absent;
artifacts and report are reproducible and checksummed.

Future topology refinement: if the contracted-load comparison shows material
effects, retain only the necessary problematic 69 kV paths and their explicit
69/138 kV boundary transformers. Do not restore the complete medium-voltage
network, invent missing transformer connections, or clear the attachment flags
by weakening the distance thresholds.

## Phase 06E — Historical transmission capacity audit

Branch: `v1-phase-06e-topology-evaluation`

Heavy execution: forbidden during the audit-only stage. Full-year execution
requires explicit owner authorization after correction-registry review.

Purpose: systematically extract the historical 2022 transmission lines and
transformers documented by official Ecuadorian sources and reconcile them
against every Line and Transformer in the Phase 06D network. This phase does
not add infrastructure, alter topology, change generation or demand inputs,
or apply capacity corrections before owner approval.

Tasks:

- register and checksum relevant Plan Maestro de Electricidad 2022, CENACE,
  CELEC EP TRANSELECTRIC, and other official regulatory sources;
- inspect indexes, transmission chapters, technical tables, annexes, and
  substation descriptions, distinguishing existing 2022 assets from planned
  infrastructure;
- create complete official line and transformer inventories preserving source
  wording, values, units, status, page, table, and conversion provenance;
- independently extract every Phase 06D PyPSA Line and Transformer, including
  endpoints, voltage evidence, `s_nom`, `s_max_pu`, effective rating, and
  network checksum;
- reconcile both inventories with `VERIFIED`, `PROBABLE`, `AMBIGUOUS`, and
  `UNMATCHED` statuses, including official assets missing from PyPSA;
- compare line and transformer ratings component by component while separating
  physical thermal ratings from operational/security limits;
- create the six required audit CSVs and a source register with mandatory
  provenance fields;
- implement automated completeness, uniqueness, unit, status, and provenance
  checks;
- prepare a correction registry for owner review without modifying the network;
- after approval only, inspect Notebooks 04/05 for the reproducible integration
  point, apply verified corrections, and preserve the Phase 06D baseline;
- after separate owner authorization, rerun the sequential notebook workflow
  and compare scientific metrics, energy balance, congestion, utilization, and
  CENACE indicators before and after correction.

Acceptance: documentary coverage is demonstrably comprehensive; every modeled
Line and Transformer appears exactly once in the audit; every modeled component
has a match status; missing/ambiguous ratings and official assets absent from
PyPSA are explicit; every proposed correction is source-traceable and pending
owner review; automated checks pass; and no network change occurs during the
audit-only stage.

Deliverables: `docs/PHASE_06E_ACTION_PLAN.md`, the six tables under
`data/interim/topology_audit_2022/`, and
`reports/PHASE_06E_TECHNICAL_REPORT.md` after the corresponding work is
performed. The detailed methodology and later execution gates are defined in
the action plan.

## Phase 06F — Hydro resource and dispatch evaluation

Branch: `v1-phase-06f-hydro-evaluation`

Purpose: explain and, where supported by CENACE/CNEL evidence, correct the
approximately 1,743.7 GWh hydroelectric shortfall remaining after Phase 06D/06E.
This phase audits river-flow simplifications, plant factors, `p_max_pu`,
commissioning masks, dispatch economics, plant attachments, congestion, and
the distinction between nominal and available capacity.

Starting evidence is recorded in
`docs/PHASE_06F_HYDRO_EVALUATION_ACTION_PLAN.md`. The Phase 06D network
dispatches 22,755.84 GWh of hydro against the CENACE reference of 24,499.54
GWh. Copperplating raises hydro only to 23,116.08 GWh, so transmission is
contributory but not the sole cause. All 65 modeled hydro plants reach their
available limit at least once, and 61 do so for more than half the year.

Tasks:

- reconstruct complete hydro input lineage from source data through `p_max_pu`;
- evaluate every plant's nominal capacity, hourly offer, annual available GWh,
  dispatch, saturation, commissioning, and bus attachment;
- separate resource-limited, power-saturated, economically displaced,
  transmission-constrained, and unavailable behavior;
- reconcile simplified river-flow and plant-factor assumptions with official
  CENACE/CNEL/CELEC evidence;
- test competing resource, commissioning, `p_max_pu`, hydro-price,
  water-value, topology, plant-attachment, and accounting hypotheses;
- check hydro marginal prices and opportunity-cost assumptions against available
  source evidence and thermal/import merit order;
- run controlled resource, mask, economic, and copperplate sensitivities;
- rerun the complete sequential notebook workflow for the baseline and every
  owner-approved candidate solution, retaining executed notebooks, logs,
  checksums, solver outcomes, and validation comparisons;
- prepare source-traceable hydro corrections without forced dispatch or circular
  annual scaling;
- after owner approval, implement corrections through the notebook pipeline and
  rerun the full validation workflow;
- report hydro, thermal, nonconventional, imports, balance residual, curtailment,
  shedding, congestion, and TVD before and after.

Acceptance: every hydro plant has input lineage and an offer/dispatch
diagnostic; saturation is evaluated against nominal and available capacity;
resource, economics, topology, and accounting effects are separated; official
source evidence and exact locators are retained; corrections are reviewed and
reproducible; and the report distinguishes explained from unresolved hydro
discrepancy.

Deliverables: the Phase 06F action plan, hydro interim tables and controlled-run
outputs under `data/interim/hydro_evaluation_2022/` and `results/phase_06f/`,
and `reports/PHASE_06F_HYDRO_EVALUATION_REPORT.md`.

## Phase 06G — Hydro profile and capacity reassessment

Branch: `v1-phase-06g-hydro-profile-reassessment`

Purpose: reassess hydro flow-to-availability profiles, plant factors, capacity
definitions, commissioning eligibility, assignment confidence, outages,
temporal processing, dispatch/network attribution, and accounting boundaries
after the Phase 06F diagnostic matrix. This is a planning and evidence phase;
it does not approve a hydro correction.

Tasks:

- verify the remote Phase 06F baseline and retained full-year artifacts;
- audit 65 plant assignments, eight legacy flow series, capacity fields,
  commissioning, profile lineage, and authoritative source locators;
- define isolated, evidence-backed hypotheses and bounded sensitivities;
- require a complete 8,760-snapshot sequential workflow for every hypothesis;
- compare hydro and thermal shares with explicit denominators, boundaries,
  percentage-point differences, relative errors, and the project-wide 5% gate;
- separate offers, dispatch, curtailment, congestion, displacement, outages,
  and unresolved cases;
- retain `F06F_A120` as unsupported sensitivity context and `F06F_A100` as an
  upper-bound diagnostic only;
- obtain owner review before any implementation-phase correction.

Acceptance: the action plan identifies evidence, missing inputs, controlled
full-year runs, outputs, metrics, checksums, and stop/advance criteria for each
testable hypothesis; no unsupported multiplier or forced dispatch is promoted;
and the later execution phase has a reviewable handoff.

Deliverable: `docs/PHASE_06G_HYDRO_PROFILE_REASSESSMENT_ACTION_PLAN.md`.

## Phase 07 — Master Plan expansion data products

Branch: `refactor/v1-phase-07-master-plan`

Heavy execution: forbidden except parsing verification.

Tasks:

- audit and reuse existing manual PDF-to-table work;
- normalize future generation, lines, transformers, and substations into distinct,
  source-traceable tables with stable IDs and commissioning years;
- retain raw official timing separately from later scenario delay assumptions;
- encode dependencies between plants, buses, substations, transformers, and lines;
- validate endpoints, voltage levels, units, duplicates, unknown locations, and
  aggregation by target year;
- explicitly remove/quarantine nuclear placeholders from official tables.

Tests: schemas, source citations, unique IDs, referential integrity, chronology,
units, year filtering, and deterministic aggregation.

Acceptance: a student can ask “what official expansion exists by year Y?” and
obtain an auditable, model-ready portfolio without re-reading the PDFs.

## Phase 08 — Student scenario framework and examples

Branch: `refactor/v1-phase-08-scenarios`

Heavy execution: tiny examples only unless explicitly authorized.

Tasks:

- reassess and replace the current scenario registry around the validated baseline;
- make scenario definitions data/configuration-driven;
- support official planned expansion as a reference layer and separate student
  changes such as additional renewable plants, buses, lines, and transformers;
- define hydrology, demand, retirement, interconnection, and optional hypothetical
  technology dimensions without preselecting students' research conclusions;
- provide notebook templates for building, checking, running, and comparing a
  scenario.

Tests: schema, composition ordering, baseline immutability, year filtering,
conflict detection, provenance, and tiny solve.

Acceptance: students can create a scenario without editing core model logic, and
official projects remain distinguishable from scenario assumptions.

## Phase 09 — Final documentation and handover audit

Branch: `refactor/v1-phase-09-handover`

Heavy execution: use already validated artifacts; rerun only if explicitly needed.

Tasks:

- finalize installation, execution, methodology, data/provenance, validation,
  scenario-authoring, result-interpretation, citation, and licensing docs;
- run a fresh-user walkthrough using only documented entry points and external
  artifacts;
- verify all notebooks run in order and reports/plots regenerate;
- audit for stale copies, hidden machine paths, accidental binaries, obsolete
  nuclear assumptions, and undocumented manual steps;
- prepare the student assignment boundary and extension guide.

Acceptance: a technically competent new student can obtain data, reproduce the
validated 2022 result, inspect planned expansions, define a scenario, and interpret
results without repository archaeology or an unplanned manual download.

## Phase 10 — Final pruning and clean-branch release

Branch: `refactor/v1-phase-10-prune`

Heavy execution: forbidden. Use validation evidence produced by earlier phases.

Purpose: remove development and migration artifacts only after their useful
scientific content and provenance have been incorporated into the supported
project. This is a deletion phase, not an opportunity to change model behavior.

Tasks:

- build a complete tracked-file and relevant ignored-artifact inventory;
- classify each item as required runtime input, authoritative source, supported
  notebook/code, reproducible output, retained provenance, or removable debris;
- identify obsolete `_old`/`_alt` material, copies, scratch notebooks, stale
  registries, superseded tables, generated notebook-directory files, abandoned
  results, temporary logs, caches, and duplicate artifacts;
- prove that any retained scientific logic, manual correction, citation, or
  provenance has been migrated before deleting its only source;
- delete files and artifacts classified as unnecessary;
- update manifests, documentation, notebook references, ignore rules, and tests so
  no supported workflow references a removed item;
- run the complete cheap test suite and the documented fresh-user smoke workflow;
- inspect the final tree for unsupported binaries, workstation-specific paths,
  broken links, stale references, empty directories, and unexplained duplicates;
- create a dedicated reviewed commit containing the approved removals and the
  documentation changes required by those removals.

Tests: repository reference/link scan, import checks, manifest validation, all
cheap unit/integration tests, notebook dependency inspection, and the documented
short smoke workflow. Do not repeat the annual scientific solve unless earlier
artifacts fail integrity or compatibility checks.

Acceptance: every remaining file has a documented purpose; no active workflow
depends on legacy or copied material; no unnecessary generated or binary artifact
remains; the validated baseline and Master Plan provenance remain reproducible;
all tests pass; and the removals exist in a dedicated, owner-reviewed commit.

Report: `reports/PHASE_10_TECHNICAL_REPORT.md`, including the deletion inventory,
recovery source for intentionally removed historical material, tests, final tree
summary, and removal commit identifier.
