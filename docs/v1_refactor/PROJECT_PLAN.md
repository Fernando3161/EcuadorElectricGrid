# PROJECT_PLAN.md

## 1. Purpose

This document defines the modular implementation plan for the v1 refactor of `EcuadorElectricGrid`.

The refactor converts the current research prototype into a clean, reproducible, public, student-ready PyPSA project for Ecuador.

The plan is intentionally phased. Each phase must be reviewable and independently auditable. The repository owner manually controls branches, commits, and pushes.

The project does not aim to complete the students' future-scenario research before handover. It aims to provide the validated baseline, architecture, inputs, scenario framework, execution workflow, provenance, and documentation needed for students to perform that research independently.

## 2. Global development protocol

### 2.1 Branch-before-phase rule

Before each phase begins, the repository owner manually creates or selects a dedicated branch. The branch should contain only the work for that phase. Agent-assisted work begins only after the phase branch is active.

Recommended branch naming:

- `refactor/v1-phase-00-specification`
- `refactor/v1-phase-01-cleanup`
- `refactor/v1-phase-02-environment-data`
- `refactor/v1-phase-03-baseline`
- `refactor/v1-phase-04-scenarios`
- `refactor/v1-phase-05-results`
- `refactor/v1-phase-06-documentation`
- `refactor/v1-phase-07-acceptance`

The exact names may change. The one-branch-per-phase principle must remain.

The repository owner manually reviews, commits, and pushes the work.

### 2.2 Post-phase report rule

After implementation and testing for every phase are complete, create `docs/v1_refactor/REPORT_PHASE_XX.md`.

The report records what was actually done and must include objective, branch, changed files, removals/archives, architecture changes, dependency changes, data changes, tests and results, deviations, known limitations, unresolved issues, and next-phase recommendations.

### 2.3 Testing rule

Testing lives under `/tests`.

Use unit tests by default, tiny deterministic fixtures, and short integration tests only when genuinely required.

Prohibited:

- long integration tests;
- full-year optimization as an automated test;
- full scenario batches as automated tests;
- repeated large external downloads in the normal test suite.

Full annual scientific runs are validation or research executions, not software tests.

## 3. Scientific baseline to preserve

The refactor must preserve these decisions:

- supported base year: 2022;
- system: Ecuadorian SNI;
- Galapagos/isolated systems excluded;
- supported voltage levels: 138/230/500 kV;
- 69 kV excluded;
- future generation/transmission exogenous;
- future demand = complete 2022 profile multiplied by one annual factor;
- the factor may derive from official projections or a documented annual-growth method;
- central demand case: D2 PME Case Base;
- canonical delays: +5, +8, +12 years;
- generation and required transmission normally delayed together;
- v1 hydro represented through availability profiles;
- hydrology cases: normal, dry, severe dry;
- nuclear: additive SMR sensitivity;
- SMR additions: +0.5 GW in 2035, 2040, 2045, 2050;
- nuclear grid connection explicitly modeled and justified;
- PME/PET is the principal transmission basis;
- N-1 optional for v1;
- official anchor years: 2022, 2030, 2035, 2040, 2050;
- full annual hourly runs supported for scientific execution.

# Phase 0 — Freeze specification and refactor baseline

## Objective

Create a reviewable record of the pre-refactor state and requirements that later phases must preserve.

## Main tasks

- Add root `AGENTS.md`.
- Add `docs/v1_refactor/PROJECT_PLAN.md`.
- Add `docs/v1_refactor/README_v1.md`.
- Add `docs/v1_refactor/REPORT_v0.md`.
- Confirm scientific assumptions.
- Define v1 scope/non-scope.
- Identify the canonical validated 2022 artifact.
- Identify large inputs to externalize.
- Establish provenance fields.
- Establish mandatory results/KPIs.
- Establish licensing/attribution targets.
- Record known technical debt.

## Tests

No model tests required. Only file/path and Markdown sanity review.

## Acceptance criteria

The owner agrees the scope is correctly represented and the pre-refactor state is documented sufficiently for later comparison.

# Phase 1 — Repository cleanup and target architecture

## Objective

Remove active dependence on legacy research debris and establish a clear production-oriented structure without changing validated scientific behavior.

## Main tasks

Audit and remove from the supported active workflow:

- `_old`;
- `_alt`;
- `Kopie` / duplicate notebooks/scripts;
- obsolete pre-2022 scenario workflows;
- duplicate large CSVs;
- generated files stored inside notebook directories;
- dead experimental code;
- stale path/environment references;
- accidental outputs treated as inputs.

Historical material should remain recoverable through Git history/branches rather than cluttering the final main branch.

Establish clear locations for source code, configuration, authoritative small raw inputs, processed inputs, externally downloaded artifacts, generated intermediate data where necessary, tests, results, examples/notebooks, and documentation.

Ensure notebooks are explicitly non-production.

## Deliverables

- cleaned active tree;
- documented target architecture;
- no supported dependency on legacy modules/copied notebooks;
- clear data-category separation.

## Tests

Minimal path/import/schema tests only.

## Acceptance criteria

Production code no longer imports legacy folders; notebooks are unnecessary for production; duplicate large files are identified for removal/externalization; scientific behavior has not intentionally changed.

## Report

Create `REPORT_PHASE_01.md`.

# Phase 2 — Local environment and external artifact bootstrap

## Objective

Make the repository self-contained from a software-environment perspective while retrieving large canonical artifacts from a stable public archive.

## Main tasks

### Local environment

Make repository-local `.venv` the only supported development/runtime environment.

- `.venv` is Git-ignored.
- It is created if absent during first setup.
- Supported dependencies are installed there.
- `requirements.txt` is authoritative.
- New dependencies introduced later must be added to `requirements.txt`.
- Undeclared global dependencies are unsupported.

Audit existing environment files and remove their role as the primary v1 installation mechanism.

### External artifact manifest

Define a machine-readable manifest for required large artifacts with logical name, URI/DOI/archive reference, expected local destination, checksum, archive/version metadata, and purpose.

### First-run retrieval

Implement supported first-run behavior that checks local files, downloads missing artifacts, validates integrity, places them correctly, and reuses valid local files later.

The validated 2022 PyPSA `.nc` is mandatory.

Other large canonical hourly datasets may also be externalized where appropriate.

## Tests

Dependency/import smoke checks, manifest parsing, local-file detection, checksum tests on small fixtures, mocked/minimal download behavior, and representative network-file opening if practical.

No annual optimization and no repeated full-archive download.

## Acceptance criteria

A clean machine can follow the documented process without a pre-existing external environment; dependencies are declared; required artifacts can be obtained automatically and integrity-checked; students do not rebuild canonical network artifacts.

## Report

Create `REPORT_PHASE_02.md`.

# Phase 3 — Canonical 2022 baseline workflow

## Objective

Replace notebook-dependent baseline execution with a clean supported Python workflow centered on the externally supplied validated 2022 network.

## Supported boundary

Canonical validated 2022 `.nc` → load → validate → attach/confirm required time-dependent inputs → run → summarize → save results.

Rebuilding the network from the original PyPSA-Earth workflow is not part of the normal student workflow.

## Main tasks

Extract supported baseline notebook behavior into modular code for:

- loading the canonical network;
- validating topology/components;
- enforcing supported voltage scope;
- loading demand;
- loading generation availability;
- solver configuration;
- running the baseline;
- calculating KPIs;
- saving standardized outputs.

Document reference characteristics of the validated artifact so corruption can be identified without an annual solve.

## Tests

Unit tests for loaders, validation, configuration, and KPI logic.

At most one tiny integration test with a miniature network if needed.

A manually invoked short baseline smoke run is allowed.

## Acceptance criteria

No notebook is required; the canonical `.nc` is not rebuilt; the baseline loads and sanity-validates; short execution works through production interfaces; full annual execution is supported for scientific use.

## Report

Create `REPORT_PHASE_03.md`.

# Phase 4 — Master Plan translation and scenario framework

## Objective

Create the data model and scenario-composition framework students will use for future Ecuador scenarios.

## Demand

For every future year, the complete 2022 demand profile is multiplied by one annual scaling factor.

The annual factor is derived from the selected demand case and growth method.

Support D1 Tendential, D2 PME Case Base, and D3 high-growth stress.

## Master Plan timing

Support +5, +8, +12 year delay cases.

Generation and required transmission normally shift together.

## Generation

Normalize PME projects/blocks across hydro, solar, wind, biomass, geothermal, thermal, firm capacity, and replacement/repowering.

Generic or geographically undefined assignments must be explicit assumptions with provenance.

## Thermal retirement

Represent documented retirement/replacement behavior as scenario assumptions.

## Hydrology

Support normal, dry, severe dry using availability profiles.

## Interconnections

Support autonomous adequacy and Colombia/Peru cases.

Where used, interconnection economic assumptions should have documented reference prices and sources.

## SMR

Support additive SMR sensitivity cases with the canonical pathway.

Location and grid connection must be explicit scenario data.

Do not automatically remove PME firm capacity.

## Scenario registry

Create a clear registry/schema containing the dimensions needed to build a scenario without editing model code.

## Provenance

Every manually derived/assigned infrastructure row should support traceability metadata.

## Tests

Unit tests for demand scaling, delay transformations, scenario validation, project filtering, provenance, generation/transmission coupling, and SMR cumulative-capacity logic.

Use only a tiny integration test if needed.

## Acceptance criteria

Adding a scenario does not require editing core model logic; canonical dimensions are validated; ambiguous mappings are not silently invented; provenance is retained; anchor-year filtering is deterministic.

## Report

Create `REPORT_PHASE_04.md`.

# Phase 5 — Scenario execution, batch execution, results, and KPIs

## Objective

Provide consistent execution and output behavior for one scenario or the registered scenario set.

## Main tasks

Provide a supported single-scenario workflow that resolves the scenario, loads/builds required data, validates the scenario, solves it, summarizes it, and saves outputs.

Provide a batch workflow that can run selected/all registered scenarios, isolate outputs, report failures, record run metadata, and permit selected reruns.

Each scenario receives an isolated result directory.

Preferred formats:

- JSON for metadata and KPIs;
- Parquet for time series/tabular outputs;
- NetCDF for solved networks where retained.

Minimum KPI support:

- annual demand;
- peak demand;
- generation by technology;
- curtailment;
- hydro generation/availability;
- load shedding/ENS;
- peak load shedding;
- imports/exports;
- line loading;
- transformer loading;
- installed capacity;
- total system cost;
- system cost per kWh;
- solver status;
- runtime.

Add congestion/emissions where consistently supported.

## Tests

Unit tests for KPI/result calculations and metadata.

Tiny execution integration test only where required.

No annual Ecuador run in automated tests.

## Acceptance criteria

A registered scenario can be resolved, executed, and stored through one supported path; multiple scenarios can run without source-code edits; outputs are deterministic and machine-readable.

## Report

Create `REPORT_PHASE_05.md`.

# Phase 6 — Documentation, examples, licensing, and teaching handover

## Objective

Turn the implemented repository into a self-explanatory public research artifact.

## Main tasks

Rewrite the final root `README.md` to describe the implemented v1 repository.

Create/update:

- installation guide;
- first-run artifact guide;
- execution guide;
- scenario-authoring guide;
- data/provenance guide;
- methodology;
- architecture overview;
- baseline validation documentation;
- result interpretation guide;
- useful optional examples/notebooks;
- `LICENSE`;
- `CITATION.cff`;
- third-party attribution notices.

The installation guide must start from a clean machine and local `.venv`.

The scenario guide must explain anchor years, demand factors, delays, generation, hydrology, transmission, SMR, provenance, and validation.

## Tests

Documentation link/path checks where practical and smoke checks for documented entry points.

No long model runs.

## Acceptance criteria

A technically competent student unfamiliar with repository history can understand what the project does, install it, obtain data, run it, inspect results, add scenarios, and understand assumptions.

## Report

Create `REPORT_PHASE_06.md`.

# Phase 7 — Final acceptance and student-handover audit

## Objective

Verify v1 meets the handover definition and remove remaining implementation-era debris.

## Main tasks

Perform a fresh-clone-style audit of:

- `.venv` creation;
- dependency installation;
- external artifact retrieval;
- checksum validation;
- baseline loading;
- baseline sanity validation;
- short baseline execution;
- scenario registry discovery;
- scenario construction;
- output creation.

Audit final tree for obsolete copies, legacy execution dependencies, temporary migration files, hidden machine paths, accidental outputs, and duplicated binaries.

Audit documentation against implemented behavior.

Verify refactoring did not silently change voltage scope, base year, demand scaling, delay definitions, hydrology cases, SMR pathway, or exogenous scenario philosophy.

## Tests

Only minimal acceptance smoke tests.

Full annual scenario runs remain separate scientific validation unless explicitly requested.

## Acceptance criteria

The repository is ready when a fresh user can install it with local `.venv`, retrieve required artifacts, load the validated 2022 system without rebuilding PyPSA-Earth, understand and run the baseline, define scenarios through configuration/data, obtain standardized results, and extend the study without redesigning core architecture.

## Report

Create `REPORT_PHASE_07.md`.

## 4. Work intentionally deferred

Not required for successful v1 refactoring:

- completion of all future scientific scenarios;
- optimal generation expansion;
- optimal transmission expansion;
- explicit reservoir optimization;
- comprehensive N-1 analysis;
- demand response;
- independent future load-shape evolution;
- distribution-network modeling;
- Galapagos modeling;
- rebuilding canonical PyPSA-Earth artifacts;
- production dashboard implementation.

Students are expected to use the finished platform to investigate at minimum:

- renewable-expansion pathways for future anchor years;
- at least one SMR scenario;
- a severe hydrological-resilience case.

## 5. Change-control principle

If implementation reveals that a scientific assumption or phase boundary is wrong, do not silently adapt the model.

Record the issue, propose the change, and obtain an explicit project decision before altering the governing specification.

The purpose of the phased process is not only clean code. It is preservation of scientific provenance.
