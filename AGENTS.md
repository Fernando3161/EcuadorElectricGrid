# Repository guidance for coding agents

## Mission

Refactor `EcuadorElectricGrid` into a reproducible, notebook-led teaching and
research project. The first scientific release must build and run an Ecuador
2022 PyPSA network and validate its annual generation by energy carrier against
authoritative 2022 statistics.

The repository is not yet validated. Treat existing notebooks, networks, CSVs,
and solved results as evidence to investigate, not automatically as canonical
outputs.

## Read before changing the project

1. `docs/PROJECT_OBJECTIVES.md`
2. `docs/PHASE_00_DIAGNOSTIC.md`
3. `docs/PROJECT_PLAN.md`
4. `docs/DATA_AND_ARTIFACTS.md`
5. the active phase report and the notebooks/modules relevant to the task

## Scientific acceptance gate

The 2022 baseline is acceptable only when all of the following are true:

- the complete staged workflow runs in the approved environment;
- the optimization terminates successfully with recorded solver status;
- annual demand, generation, imports/exports, storage, losses, curtailment, and
  load shedding reconcile through an explicit energy balance;
- simulated annual supply by the agreed carrier taxonomy is compared with a
  versioned, cited 2022 statistical reference table;
- every in-scope carrier passes the agreed 5% tolerance rule;
- the comparison, formulas, exclusions, and residual category are reported;
- the validated input network and result artifacts have checksums and provenance.

Do not weaken the tolerance, omit failing carriers, or tune inputs solely to
make the comparison pass. Record and explain calibration decisions.

## Supported development style

Notebooks are the student-facing workflow. Keep them sequential, explanatory,
restartable, and independently useful. They must explain the purpose of each
stage, inputs, outputs, sources, units, assumptions, and validation checks.

Move reusable and testable logic into small modules under `src/`; notebook
cells should orchestrate those functions and present diagnostics. Do not hide
the scientific workflow behind a large class or require students to edit core
Python logic to define a scenario.

Plots displayed in notebooks must also be saved to the phase's figure/output
directory when they provide validation evidence. Technical reports reference
those reproducible plots.

## Environment rules

The approved environment is the existing Conda environment
`pypsa-earth-ec`.

- Do not create a repository-local environment, `.venv`, temporary virtual
  environment, or replacement Conda environment.
- Do not install or upgrade packages without the owner's explicit approval.
- Activate or invoke the environment by name through the user's existing Conda
  installation. Do not encode workstation-specific installation locations in
  committed documentation, configuration, notebooks, or source code.
- Execute notebooks through this environment, never the system Python.
- Keep `config/environment_ec.yaml` as an auditable environment specification,
  but first reconcile it against the working environment before using it to
  reproduce an installation.

Supported code and notebook paths must remain repository-relative or
configuration-driven.

## Data preflight is mandatory

No stage notebook may start scientific processing before a common preflight has:

1. read the artifact manifest;
2. checked every input required by that stage;
3. verified size and SHA-256 where specified;
4. downloaded missing external artifacts from their configured persistent URI;
5. validated the downloaded file before placing it at its authoritative path;
6. failed once, early, and clearly if any artifact remains unavailable.

Downloads must be cached and restartable. Never silently substitute another
weather year, network, statistical edition, or source. Zenodo publication and
URI/checksum entry are manual owner actions.

## Data and provenance rules

- Keep small authoritative source and normalized tables in Git when licensing
  permits; keep large NetCDF, bulk downloads, generated results, and source PDFs
  out of Git unless explicitly approved.
- Distinguish `raw`, `interim`, `processed`, and `results`. Notebook-directory
  CSV files are not an acceptable final data interface.
- Every derived planning row must retain source document, edition, chapter,
  table/page, original name/value/unit, transformation, confidence, and whether
  it is an official fact or a modeling assumption.
- Preserve manual PDF parsing and network repairs. Migrate them into normalized
  tables and tested transformation rules before retiring the legacy artifact.
- Never present auto-generated nuclear connectors or other hypothetical assets
  as Master Plan projects.

## Phase and Git workflow

Each phase in `docs/PROJECT_PLAN.md` has a dedicated branch, tests, acceptance
criteria, and technical report. Work only on the active phase. The owner reviews,
commits, merges, and pushes unless explicitly requesting otherwise.

Do not delete or bulk-move legacy files during Phase 0. Later cleanup must first
classify each item as authoritative source, reusable transformation, reproducible
output, legacy evidence, or disposable duplicate. Git history is not a substitute
for migrating otherwise undocumented scientific provenance.

## Testing and execution safety

- Use small deterministic unit tests for parsing, schemas, carrier mappings,
  energy accounting, paths, downloads, checksums, and network repair rules.
- Use tiny PyPSA fixtures for integration tests.
- Full-year Ecuador solves are manual scientific validation runs, not CI tests.
- Do not launch a large download, annual solve, or batch scenario run unless the
  owner explicitly requests that execution in the current conversation.
- Retain logs and partial reusable downloads after failure; never leave a partial
  file under an authoritative filename.

## Scenarios and Master Plan boundary

The validated 2022 baseline comes before future scenarios. The 2022 Master Plan
expansion portfolio must be formalized as source-traceable generation, line,
transformer, and substation tables that can be filtered by commissioning year and
aggregated into a future network.

Nuclear is not part of the official Master Plan baseline. Existing nuclear data,
registry entries, buses, and auto-generated connections are quarantined as legacy
hypothetical work until a later student scenario explicitly defines and justifies
them. Do not let nuclear assumptions contaminate official expansion tables.

## Completion discipline

Do not claim a phase complete until its acceptance criteria, tests, report,
figures, provenance, and manual-run handoff are present. Record commands that were
not run and why. When scientific meaning is ambiguous, document the question and
stop before encoding a guess.

The final pruning phase must identify and remove files and artifacts that are no
longer needed. Its deletions must be reviewed and recorded in a dedicated commit
so the final branch contains only supported project material and intentional
provenance.


## Codex subagent routing and context budget

Project-scoped specialist configurations live in `.codex/agents/`; `.codex/config.toml` sets a conservative concurrency cap and low-cost fallback for spawned agents. These instructions supplement, never override, the mission, scientific acceptance gate, environment policy, provenance rules, phase boundaries, and manual-execution restrictions above. The parent Codex model is selected in the Codex session; agent selection is instruction-guided rather than guaranteed.

- **Trivial changes:** The parent should handle one-file typo fixes, simple renames, or formatting directly. Do not spawn agents merely because they exist.
- **`research` (Luna):** Bounded read-only evidence search; 2022 CENACE reference data, carrier mapping, file lineage, checksum/source documentation, and discrepancies between datasets. Return compact findings with file/page, year, units, and citations.
- **`documentation` (Luna):** README, phase-report structure, notebook explanations and Markdown based on already verified results. Never turn unvalidated results into scientific claims.
- **`developer` (Sol, medium):** Small Python changes in `src/`, sequential notebook edits, data transformation functions, unit tests and tiny PyPSA fixtures. Scientific decisions must be clarified first.
- **`scientific_reviewer` (Sol, high):** Read-only, higher-stakes examination of the 2022 energy balance, feasibility/solver diagnostics, data validity, validation metrics and new scientific interpretations. It must not relax the established 5% gate or alter carrier exclusions without owner approval.

### Delegation and escalation

1. Prefer one specialist for a bounded request; spawn multiple only for genuinely independent work or for a clearly staged handoff. Cap ordinary work at two subagents and complex work at three. These are upper bounds, not targets.
2. For scientific writing with new conclusions: `research` gathers evidence, `scientific_reviewer` verifies interpretations, then `documentation` drafts from that **compact verified handoff**. For writing about already verified results, use only `documentation`.
3. For ordinary implementation use `developer`. For algorithmic or physics-changing work, get acceptance criteria from `scientific_reviewer` first; then let `developer` implement and run targeted tests.
4. Do not delegate duplicate repo-wide scans. Each subagent should receive a specific question, relevant path(s), expected deliverable and explicit exclusion of unrelated work.
5. Escalate to a stronger agent only if a low-cost agent encounters substantive ambiguity, a failed check, or potential scientific impact. Do not repeatedly retry the same unresolved task at the same effort.
6. Subagents may not make independent commits, push branches, relax data/validation requirements, install packages, or launch large calculations. The owner retains the Git and full-year-run approval gates above.

### Context and execution economy

- Keep the existing **Read before changing the project** document list as required for substantive changes, but satisfy it through focused heading/range inspection when sufficient rather than dumping entire files into every subagent's context. Include the active phase report and only relevant `src/`, `notebooks/`, `config/`, `docs/` and test files.
- Reuse compact evidence summaries; do not make every agent independently reread the full project. Create a new Codex conversation for a distinct phase or large unrelated task after recording decisions in tracked phase docs.
- Prefer deterministic scripts, existing tests and narrow calculations over repeated language-model arithmetic. Full-year Ecuador simulations remain manual, owner-authorized validation runs.
- In each task summary, state agent(s) used when relevant, files changed, checks executed, checks skipped, provenance/assumptions affected, and any issue requiring owner resolution.
