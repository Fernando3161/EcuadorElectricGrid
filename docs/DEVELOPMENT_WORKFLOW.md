# Development workflow

## Phase lifecycle

Each project-plan phase uses its own branch and follows this sequence:

1. owner creates or selects the phase branch;
2. confirm scope and read the current diagnostic/plan;
3. run `python -m src.preflight --stage 00_preflight` in the approved environment;
4. inspect relevant evidence before moving or rewriting it;
5. implement only the active phase;
6. run cheap automated tests in `pypsa-earth-ec`;
7. run only the explicitly authorized scientific/manual execution;
8. regenerate validation figures and write the phase report;
9. audit the diff for binaries, machine paths, credentials, and lost provenance;
10. owner reviews, commits, merges, and pushes.

Do not begin the next phase merely because it is convenient.

## Notebook contract

Supported notebooks use a stable numeric prefix and have:

- a title, scope, and learning/scientific objective;
- explicit input/output tables with units and artifact IDs;
- cited data sources and a distinction between facts and assumptions;
- a first executable cell that calls the common preflight;
- repository-relative paths obtained from shared path/config helpers;
- short orchestration cells calling functions in `src/`;
- validation checks that fail loudly;
- conclusions, limitations, and a pointer to the next stage;
- deterministic saved figures and outputs.

Notebooks should run top-to-bottom from a clean kernel. Clear stale outputs before
release or deliberately retain only small outputs that are part of the teaching
artifact. Execution during development must use the approved environment.
Use `python -m src.notebook_runner <notebook-path>` for command-line execution
when a persistent environment-specific kernelspec is unavailable.

## Testing layers

- Static/schema tests: configuration, manifests, tables, paths, units.
- Unit tests: parsing, matching, repair rules, profile normalization, carrier
  mapping, tolerance formulas, energy balance, and KPIs.
- Tiny integration tests: miniature PyPSA networks and mocked downloads.
- Notebook smoke tests: short horizon or no-solve stages.
- Scientific validation: full 8,760-hour baseline, manually authorized and
  recorded in the phase report.

Passing software tests does not establish scientific validation.

## Reports and plots

Each phase ends with `reports/PHASE_XX_TECHNICAL_REPORT.md`, following
`docs/TECHNICAL_REPORT_TEMPLATE.md`. Figures must answer a technical question and
be reproducible from a named notebook/script, input artifact version, and commit.

## Heavy-operation boundary

Annual optimization, multi-gigabyte download, weather-cutout processing, and
scenario batches are heavy operations. The agent may implement and cheaply test
their interfaces but runs them only after explicit owner authorization. The report
lists exact commands the owner must run when a heavy validation was deferred.
