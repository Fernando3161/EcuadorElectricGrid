# Phase 07 planning data and interfaces

Phase 07 records the published expansion schedule in the owner-selected
adjustment and prepares a configurable teaching schedule. It does not establish
that a listed project was built, remains pending, or is electrically ready to
add to a network. Physical and electrical assumptions are deliberately deferred
for later teaching scenarios, as directed by the owner.

## Governing source and verified inventory

The source is the owner-supplied `AJUSTE-PME-2026.pdf`, whose printed title is
*Anexos ajuste PME 2023-2032*. The four-page PDF is available at
[`literature/PlanMaestroElectrifEc/AJUSTE-PME-2026.pdf`](../literature/PlanMaestroElectrifEc/AJUSTE-PME-2026.pdf)
and from the [Ministry source URL](https://www.ambienteyenergia.gob.ec/wp-content/uploads/2026/06/Anexos-A-B-y-C_ajuste_pme_20260611.pdf).
Its SHA-256 is
`3f131642a88e1923461529591d2ff5cc8417149b0a271b0bf19074c1f984c3e0`.
The source is an adjustment schedule; financing in management is not proof of
construction or commissioning.

Visual review of all pages established this inventory:

| Source section | PDF page | Recorded rows |
|---|---:|---:|
| Annex A — generation | 1–2 | 50 |
| Annex B — grid projects | 3 | 59 |
| Annex C — grid projects | 4 | 14 |
| Total | 1–4 | 123 |

The raw transcriptions are in
[`data/raw/planning/`](../data/raw/planning/): `adjustment_2026_generation.csv`,
`adjustment_2026_grid_projects.csv`, and `adjustment_2026_totals.csv`. The
normalized inventory retains each source page, annex, source text and source
fingerprint in [`data/processed/planning/projects.csv`](../data/processed/planning/projects.csv).
Four direct-asset views contain 77 rows: 50 generators, 9 lines, 16 substations,
and 2 transformers. The full `projects.csv` also retains 46 rows that do not
map one-to-one to those asset types: 28 compound grid systems, 12 reactive
installations, 4 mobile-equipment entries, 1 plant modernization, and 1 storage
entry. Keep these in the full inventory; do not silently drop them or invent an
asset decomposition.

The generation annex row sums are **8,818.75 MW** of capacity and **48,759.01
GWh/year** of mean energy. The annex prints **47,413 GWh/year** for mean energy,
leaving a **1,346.01 GWh/year** difference. These values and the discrepancy are
retained in `data/raw/planning/adjustment_2026_totals.csv` and
`results/phase_07/source_total_reconciliation.csv`; the difference is unresolved
and must not be used as dispatch or control data.

`results/phase_07/generation_source_checks.csv` separately flags A049,
Tugaduaja 2 - Fase 2: 2,058 GWh/year exceeds the 1,314 GWh nameplate-times-8,760h
bound for its listed 150 MW. The source numbers remain unchanged; resolving this
warning requires evidence, not a silent model correction.

The restored chapters 3, 4, and 5 (`cap_3_demanda.pdf`,
`cap_4_generacion.pdf`, and `cap_5_transmision.pdf`) are older, 2017-origin
planning material. They remain legacy evidence and are not relabeled as the
2026 adjustment. The 187 earlier candidate records and 11 nuclear-quarantine
records are retained under `data/interim/planning/legacy_candidates/` and
`data/interim/planning/nuclear_quarantine.csv`; they are not merged into the
123-row adjustment inventory. The nuclear entries remain outside the official
planning products.

## Published dates and exercise delay

`data/processed/planning/exercise_schedule.csv` preserves the published source
date and calculates an exercise date as:

\[
  y_{exercise} = y_{published} + d
\]

where `d` is the configurable delay in whole years. The pinned teaching default
is **7 years**, set in [`config/planning.yaml`](../config/planning.yaml). It
does not change a source date or claim that the adjustment is a forecast. The
resulting dates span 2030–2039; any half-year source timing is preserved.
`build_products(delay_years=...)` supports another explicit delay for an
exercise. The notebook exposes the setting without requiring edits to core
logic. Every repeated project has its own `EX_REPEAT_*` scenario identifier.

The published-date inventory and delayed exercise filter answer different
questions. Use `query_inventory(projects, target_year, delay_years=0)` for
published timing across all project categories. The four direct-asset views
can be filtered with `filter_published_by_year`; this view is explicit about
its narrower scope. The delayed schedule is a pedagogical repeat of the
published portfolio, not a statement that projects remain unbuilt in the
exercise year.

## Coordinates and readiness boundary

Coordinates are assigned only through an exact-name legacy match or a unique
canton match. Those coordinates are marked as assumptions with low confidence;
they do not establish a project site. The current inventory assigns coordinates
to 48 rows and leaves 75 unresolved, recorded in
[`results/phase_07/coordinate_coverage.csv`](../results/phase_07/coordinate_coverage.csv).
No nearest-bus repair is performed. Same-name or province/canton evidence does
not establish electrical connectivity.

All 123 projects currently have `model_ready=False`. Missing facts include
electrical ratings, connections, detailed endpoints and decompositions for
compound works. “Not specified” is retained separately from “not normalized.”
The strict model-readiness query raises when asked to publish these records as
network assets. Later teaching assumptions must be explicit, configurable and
distinguished from source facts before any network transformation uses them.

Potential project relationships remain unresolved: B039/B045 (Ibarra–Pimampiro),
B052/B055 (Shushufindi–Orellana), and B026/B031 (Portoviejo). Preserve these
in [`data/processed/planning/project_relationships.csv`](../data/processed/planning/project_relationships.csv)
as review questions, not as established duplicates or additive projects.

## Reproduction and provenance

Use the existing `pypsa-earth-ec` Conda environment. Stage 07 preflight checks
all required artifacts and their configured size/checksum; the missing PDF is
retrieved from its configured persistent URI and checked before it is placed at
the registered path. The restored older chapter PDFs are manual inputs with
registered fingerprints. Notebook processing reads the reviewed CSV
transcriptions, so it does not need a PDF-reading package at runtime.

```powershell
conda activate pypsa-earth-ec
python -m src.preflight --stage 07_planning --offline
python -m scripts.migrate_phase07_planning
python -m src.notebook_runner notebooks/07_prepare_master_plan_expansion.ipynb --output results/phase_07/07_prepare_master_plan_expansion.executed.ipynb
python -m unittest discover -s tests -p 'test_phase07*.py' -v
```

The migration manifest at `data/interim/planning/migration_manifest.json`
records source and product fingerprints. The notebook writes validation outputs
under `results/phase_07/` and saves its evidence figures under
`reports/figures/phase_07/`. These figures describe source totals and coordinate
coverage, not a solved or validated network. The Python API is in
[`src/planning_products.py`](../src/planning_products.py) and
[`src/planning.py`](../src/planning.py).

Phase 07's source inventory and delayed teaching schedule are reproducible.
They do not pass the physical model-readiness gate, and do not validate the
Ecuador 2022 baseline or authorize a scenario solve. Resolve the source-total
discrepancy and project relationships from evidence; define any later teaching
assumptions explicitly before network use.
