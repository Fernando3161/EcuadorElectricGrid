# AGENTS.md

## 1. Purpose and authority

This file governs all agent-assisted development in the `EcuadorElectricGrid` repository during the v1 refactor and thereafter unless explicitly superseded by a later repository-level instruction.

The repository is a public, open-source scientific artifact for the Ecuadorian Sistema Nacional Interconectado (SNI). Its purpose is to provide a clean PyPSA-based platform for studying future electricity-system scenarios for Ecuador from a validated 2022 reference system.

The primary development objective is not to redesign the scientific study. It is to convert the current research prototype into a reliable, reproducible, well-documented student handover artifact.

If a task conflicts with this file, stop and report the conflict rather than silently changing the governing assumptions.

## 2. Core v1 outcome

At the end of the v1 refactor, a new student must be able to clone the repository, create and use the repository-local Python environment, automatically obtain required externally hosted artifacts on first execution, load the validated 2022 Ecuador SNI network without rebuilding it from PyPSA-Earth, run the supported 2022 workflow, understand the source and assumptions of future inputs, define a new scenario without modifying core model logic, run supported scenarios, inspect standardized outputs and KPIs, and understand how to extend the project in a controlled way.

Students must not be required to reconstruct the 2022 `.nc` network from scratch, reproduce obsolete pre-2022 studies, or understand legacy notebook chains before doing scientific work.

## 3. Scientific scope that must not be changed during refactoring

### 3.1 Reference system

- The sole supported reference year is 2022.
- The supported national network represents the Ecuadorian SNI.
- Galapagos and isolated subnetworks are outside scope.
- The transmission model is restricted to 138 kV, 230 kV, and 500 kV assets.
- 69 kV assets are excluded from the supported macro-level model.
- Colombia and Peru interconnection points may be represented.
- The principal adequacy reference is autonomous operation; interconnection cases may be added separately.

### 3.2 Demand

The 2022 spatial and temporal demand profile is the base profile.

For a future year, the complete 2022 demand profile is multiplied by one annual scaling factor. The factor may be derived from official annual projections, linear growth, compound annual growth, or another explicitly defined annual-growth method.

The annual-growth method determines the factor. The profile is then scaled directly by that factor. The hourly and spatial shape is not independently re-profiled in v1.

If future infrastructure explicitly introduces new demand nodes, those additions must be documented as scenario assumptions.

The main demand scenario family is:

- D1: Tendential;
- D2: PME Case Base / singular-load case and central planning case;
- D3: high-growth stress case.

Beyond the PME projection horizon, compound annual growth is the current default unless a scenario explicitly states otherwise.

### 3.3 Generation

Future generation additions are exogenous scenario assumptions, not endogenous capacity-expansion decisions.

Supported technologies include hydro, solar PV, wind, biomass, geothermal, thermal generation, firm-generation blocks, thermal replacement or repowering, and SMR nuclear sensitivity cases.

Existing thermal capacity may be retired based on age and documented retirement or replacement plans.

PME firm thermal additions are a possible adequacy strategy. They are not to be silently removed when nuclear sensitivity cases are introduced.

### 3.4 Hydropower

The supported v1 hydro representation uses availability profiles such as `p_max_pu` derived from hydrological information.

Official hydrology cases are normal, dry, and severe dry.

Explicit reservoir dispatch and water-balance modeling are future extensions and are not required for v1.

### 3.5 Master Plan timing

The scenario framework must support global delays applied to Master Plan project dates:

- optimistic: +5 years;
- baseline: +8 years;
- pessimistic: +12 years.

Generation and the transmission infrastructure required for it are normally delayed together.

### 3.6 Nuclear

Nuclear scenarios are additive SMR sensitivity scenarios.

They do not automatically remove PME firm capacity. Research scenarios may later compare nuclear against alternative firm-capacity assumptions, but replacement is not implicit.

Canonical cumulative SMR pathway:

| Year | New SMR capacity | Cumulative SMR capacity |
|---|---:|---:|
| 2035 | 0.5 GW | 0.5 GW |
| 2040 | 0.5 GW | 1.0 GW |
| 2045 | 0.5 GW | 1.5 GW |
| 2050 | 0.5 GW | 2.0 GW |

SMR sites should preferentially be located close to major transmission nodes.

Every SMR scenario must explicitly model and document its grid connection. New transformers, buses, lines, and 500 kV infrastructure must be added where electrically required rather than automatically.

### 3.7 Transmission

The PME/PET expansion is the principal future transmission basis.

Additional lines or substations may be introduced only when explicitly documented and justified.

Formal N-1 analysis is desirable but is not a v1 refactor requirement. It may be included if it can be implemented with little additional effort using existing project or PyPSA capabilities. Otherwise it must be recorded as a future extension.

### 3.8 Scenario years

Official anchor years are 2022, 2030, 2035, 2040, and 2050.

### 3.9 Simulation resolution

Final scientific runs should support a full hourly year.

Short runs are encouraged for development, smoke testing, and debugging.

A full annual solve is scientific execution, not an automated software test.

## 4. Git and branch rules

Before any project phase begins, the repository owner creates or selects a dedicated branch for that phase.

The agent must not push to any remote, merge branches, force-push, rebase shared history, delete branches, modify `main` directly, or create tags/releases unless explicitly instructed.

The repository owner manually reviews, commits, and pushes all changes.

One branch should correspond to one refactor phase. Recommended naming convention: `refactor/v1-phase-XX-short-description`.

Do not begin work for a new phase while the previous phase has unresolved acceptance failures unless the repository owner explicitly authorizes it.

## 5. Mandatory phase workflow

Every phase follows this order:

1. Confirm the phase objective and active branch.
2. Inspect the current repository state relevant to the phase.
3. Implement only the phase scope.
4. Update documentation affected by the changes.
5. Run the minimum required tests.
6. Review changed/generated files for accidental large artifacts, secrets, legacy dependencies, or machine-specific paths.
7. Create a detailed phase report after implementation and testing are complete.
8. Stop for manual owner review, commit, and push.

A phase report must be written as `docs/v1_refactor/REPORT_PHASE_XX.md`.

Each report must include the phase objective, branch name, files added/changed/moved/removed, architectural decisions, scientific behavior changed or intentionally preserved, tests and results, skipped tests and reasons, dependencies added/removed, data changes, documentation changes, deviations from plan, unresolved issues, and recommended next actions.

The phase report is written after the work, not before it.

## 6. Python environment and dependency rules

The project must use a repository-local, Git-ignored `.venv`.

No external Conda environment, global Python environment, neighboring repository environment, or developer-specific environment is part of the supported v1 workflow.

On first implementation or first supported execution:

- if `.venv` does not exist, it must be created locally;
- supported dependencies must be installed into that `.venv`;
- `.venv` must remain Git-ignored;
- `requirements.txt` is the authoritative pip dependency list.

If any project task requires an additional runtime or test dependency:

- install it only in the repository-local `.venv`;
- add it to `requirements.txt`;
- document why it was introduced if its purpose is not obvious;
- avoid unnecessary dependency additions;
- remove obsolete dependencies when their code paths are removed.

Do not rely on undeclared packages that happen to exist on a developer machine.

The final installation documentation must begin from a clean-machine assumption.

## 7. External data and first-run artifact rules

Large canonical artifacts must not be duplicated in Git when they can be reliably hosted in a permanent public archive such as Zenodo.

The v1 repository will use an external archive for required large artifacts, including the validated 2022 PyPSA network and other large canonical inputs as appropriate.

First-run behavior must detect whether each required local artifact exists, download only missing artifacts, place each file in its documented repository location, verify file integrity using a checksum or equivalent manifest, fail clearly if an artifact cannot be obtained or validated, and record the archive version or DOI/URI used.

Later runs must reuse valid local files and must not redownload them unnecessarily.

Small authoritative source tables, schemas, manifests, and metadata should remain in Git when practical.

Large generated or downloaded files must not be committed accidentally. Duplicate large files are prohibited.

## 8. Repository architecture rules

Required principles:

- production logic lives in Python modules or explicit executable entry points;
- notebooks are optional examples or visualization tools only;
- no notebook may be required to build/load the supported network, prepare canonical inputs, run a scenario, or summarize results;
- active code must not depend on `_old`, `_alt`, copy files, backup notebooks, or historical prototypes;
- legacy work may remain in Git history or non-active branches, but must not clutter the final supported `main`;
- raw, processed, external, generated, and result data must have distinct meanings and locations;
- generated outputs must never be treated as hand-edited source data;
- paths must be repository-relative or configuration-driven;
- hard-coded developer-machine paths are prohibited.

The final root `README.md` is part of the refactor and must describe the repository as actually implemented.

`docs/v1_refactor/README_v1.md` documents the refactor work itself and is not a substitute for the final root README.

## 9. Scenario architecture rules

Scenario definitions must be data/configuration driven.

A student should be able to define a scenario without editing core modeling code.

The scenario system must separate scenario identity and metadata, anchor year, demand case, Master Plan delay case, generation additions, thermal retirement or repowering assumptions, hydrology case, transmission expansion, interconnection assumptions, SMR assumptions, and run configuration.

Scenario IDs must be stable, descriptive, and machine-safe.

Every manually assigned project, bus, connection, or future infrastructure assumption must be traceable.

Recommended provenance fields include source document, source chapter, source table, source page, source text/reference, whether a value is an assumption, assumption reason, confidence, creator, and last verification date.

## 10. Result and dashboard-readiness rules

Simulation and dashboard concerns must remain separate.

The scientific model produces structured outputs. A future dashboard reads those outputs and must not run PyPSA simulations itself.

Each scenario should have an isolated result directory containing resolved input/scenario metadata, outputs, time series, logs, figures where produced, `metadata.json`, `summary.json`, and optionally a solved network artifact.

Preferred public formats:

- JSON for metadata and KPIs;
- Parquet for large tabular/time-series outputs;
- NetCDF for PyPSA network artifacts.

Pickle must not be the only long-term public result format.

SQL may be introduced later if the dashboard requires it; it is not mandatory for v1.

Mandatory KPI support includes annual demand, peak demand, generation by technology, load shedding/ENS, peak load shedding, renewable curtailment, imports/exports when enabled, line loading, transformer loading, congestion indicators where supported, installed capacity by technology, total system cost, system cost per kWh, solver status, and runtime.

## 11. Testing rules

Testing must remain intentionally small and lives under `/tests`.

Unit tests are the default and should use deterministic minimal fixtures.

Long integration tests are prohibited.

A small integration test may be added only when interaction between components cannot be meaningfully tested in isolation. Such tests must use tiny networks, small datasets, short time horizons, and deterministic inputs.

Do not use an 8,760-hour Ecuador optimization as an automated test.

Do not use a full scenario batch as an automated test.

Scientific validation runs are separate from the software test suite.

If external artifacts are introduced, an explicitly invoked smoke test may verify manifest parsing, download behavior with a minimal fixture or mocked source where possible, checksum validation, network opening, expected component presence, and schema compatibility.

Do not create tests simply to maximize coverage. Tests must protect behavior that is important, fragile, or scientifically consequential.

## 12. Coding standards

Code must be suitable for student handover.

Use descriptive names, explicit units in names where ambiguity is possible, type hints where they improve clarity, concise docstrings for public APIs, comments for scientific assumptions/non-obvious decisions, small composable functions, explicit error messages, and deterministic behavior where possible.

Avoid unexplained abbreviations, giant procedural scripts, duplicated model-building logic, hidden mutation of shared state, magic numbers, undocumented unit conversions, silent fallback behavior, and broad exception swallowing.

Scientific assumptions belong in configuration/data and documentation rather than being buried in implementation code whenever practical.

## 13. Documentation requirements

The v1 refactor must ultimately provide:

- final root `README.md`;
- installation guide;
- execution guide;
- scenario-authoring guide;
- data and provenance guide;
- methodology documentation;
- architecture overview;
- validation documentation;
- source and attribution documentation;
- license;
- citation information;
- phase reports.

Documentation must match implemented behavior.

## 14. Licensing and attribution

Original project code is intended to use the MIT License unless a later explicit project decision changes this.

Third-party data, documentation, and derived artifacts retain their own applicable terms.

Do not claim ownership of or relicense third-party content.

The final repository should include `LICENSE`, `CITATION.cff`, and `THIRD_PARTY_NOTICES.md` or equivalent.

Attribution must cover, as applicable, Ecuadorian electricity Master Plan sources, PyPSA, PyPSA-Earth, meteorological/hydrological data sources, and other external datasets.

## 15. Prohibited refactor behavior

Unless explicitly authorized, do not alter the validated scientific meaning of the 2022 base case while merely reorganizing code, invent missing project data, silently resolve ambiguous bus/project mappings, silently change units, replace documented assumptions with agent-selected alternatives, convert exogenous planning scenarios into endogenous capacity expansion, add a new modeling framework, build the future dashboard during the refactor, make the dashboard execute simulations, restore 69 kV modeling to the supported baseline, require students to rebuild PyPSA-Earth source artifacts, preserve obsolete files in active paths merely “just in case,” or add large binary artifacts to Git without explicit approval.

When uncertain, document the ambiguity and preserve scientific behavior rather than guessing.

## 16. Definition of phase completion

A phase is complete only when its planned tasks are implemented, acceptance criteria are satisfied, required documentation is updated, required minimal tests pass, no prohibited long tests were introduced, dependency changes are reflected in `requirements.txt`, no accidental large artifacts or local paths are tracked, and its phase report has been written.

The repository owner then reviews, commits, and pushes manually.

The agent must stop after the agreed phase scope rather than opportunistically continuing into the next phase.
