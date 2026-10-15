# Repository architecture

## Status of this document

This is the Phase 1 target architecture. It separates the supported project from
legacy evidence without moving or deleting the evidence prematurely. A path being
listed here does not make its current contents scientifically validated.

## Design principles

- Notebooks are the student-facing stages; modules contain reusable, tested logic.
- Every stage runs a common data preflight before scientific processing.
- Paths are repository-relative and resolved by shared code.
- Raw evidence is immutable. Reproducible transformations write to `interim`,
  normalized model inputs to `processed`, and run products to `results`.
- Official evidence, derived values, and modeling assumptions are separate data
  classes, not labels inferred from filenames.
- A generated network is never used as an unexplained raw input.
- Legacy material remains readable until its unique logic and provenance have
  migrated and the owner approves its Phase 10 disposition.

## Target tree

```text
EcuadorElectricGrid/
|-- config/
|   |-- artifacts.yaml                 # Phase 2 external-artifact manifest
|   |-- environment_ec.yaml            # reconciled environment specification
|   `-- schemas/                        # manifest and normalized-table contracts
|-- data/
|   |-- raw/                            # immutable source evidence/external inputs
|   |-- interim/                        # reproducible parses, matches, and audit logs
|   `-- processed/                      # normalized model-ready tables
|-- docs/                               # project, methods, provenance, and handover docs
|-- literature/                         # source-document register; PDFs external if needed
|-- notebooks/
|   |-- 00_preflight_and_data_inventory.ipynb
|   |-- 01_prepare_2022_demand.ipynb
|   |-- 02_prepare_2022_generation.ipynb
|   |-- 03_prepare_renewable_and_hydro_profiles.ipynb
|   |-- 04_build_and_validate_2022_topology.ipynb
|   |-- 05_assemble_and_smoke_test_2022_network.ipynb
|   |-- 06_run_and_validate_2022_full_year.ipynb
|   |-- 07_prepare_master_plan_expansions.ipynb
|   |-- 08_build_future_scenario.ipynb
|   `-- 09_compare_scenarios.ipynb
|-- reports/
|   `-- figures/                        # curated reproducible validation evidence
|-- results/
|   |-- runs/<run_id>/                  # network, tables, metadata, and logs per run
|   `-- experiments/                    # explicitly non-canonical diagnostics
|-- src/                                # small domain modules used by notebooks
`-- tests/                              # deterministic schema, unit, and tiny fixtures
```

The present `_alt` and `_old` directories are migration areas, not supported
entry points. Generated CSVs currently stored beside notebooks will move only
when their producing logic and consumers are rebuilt.

## Data-zone contract

| Zone | Allowed content | Required behavior |
|---|---|---|
| `data/raw/` | source tables and checksum-pinned external inputs | immutable after acquisition; retain source identity |
| `data/interim/` | parsing, matching, and repair audit outputs | fully reproducible; may expose unresolved rows |
| `data/processed/` | normalized tables consumed by model stages | schema-validated and produced from named inputs |
| `results/runs/<run_id>/` | solved network, KPIs, energy balance, tables, logs | immutable run directory with metadata and input hashes |
| `reports/figures/` | selected evidence used in reports | reproducible from a named stage and run |

## Supported module boundaries

The target does not use one large pipeline class. Expected modules are small and
stage-oriented:

| Concern | Target responsibility |
|---|---|
| paths/configuration | repository-relative locations and configuration loading |
| artifacts/preflight | dependency closure, download, checksum, structural validation |
| demand | source parsing, scaling, totals, and forecast checks |
| generation | plant normalization, carrier mapping, capacity checks |
| profiles | hydro matching and renewable-profile alignment |
| topology | network checks, explicit repairs, pruning audit, invariants |
| assembly | deterministic component attachment and short-run construction |
| accounting | weighted annual energy balance and carrier aggregation |
| validation | official-statistics comparison and fixed acceptance formula |
| planning | source-traceable project normalization and year filtering |
| scenarios | declarative assumptions applied after the baseline is validated |
| reporting | standardized figures, tables, metadata, and run summaries |

Existing functions in `src/ec_network_eval.py`, `src/helpers.py`, and
`src/paths.py` are candidates for extraction, not automatically supported APIs.
The old scenario pipeline and registry remain quarantined until the baseline
contract is complete.

## Naming conventions

- Notebook prefixes are two digits and define execution order.
- Artifact IDs use lower-case snake case and do not encode a local path.
- Normalized planning IDs use stable upper-case identifiers independent of row
  order.
- Tables include units in field names when one unit is fixed, or in a dedicated
  unit column when the value type varies.
- Generated run directories use an explicit run ID and contain machine-readable
  metadata; filenames such as `TEST_365` are not canonical identifiers.
- Copies, backups, and words such as `final` are not versioning mechanisms.

## Planning-data boundary

Normalized official Master Plan tables will use the common schema initiated in
`config/schemas/planning_project.schema.json`. Each row records its source
document, edition, chapter, page/table, original value and unit, transformation,
confidence, and evidence class.

Nuclear rows, buses, connections, costs, and scenario definitions presently in
the repository are modeling assumptions. They cannot enter a table whose
`evidence_class` is `official_fact`. Phase 8 may preserve them only in a separate
student-scenario input after explicit definition and justification.

## Migration and deletion gates

A legacy item may be moved or removed only when:

1. its writer and readers are known;
2. any unique formula, repair, mapping, citation, or manual decision is captured;
3. replacement data has a schema and provenance;
4. replacement logic has a deterministic test;
5. supported notebooks no longer reference the legacy item;
6. the owner approves its classification; and
7. removal occurs in the dedicated Phase 10 branch and commit.

Git history alone does not satisfy these gates for undocumented scientific work.
