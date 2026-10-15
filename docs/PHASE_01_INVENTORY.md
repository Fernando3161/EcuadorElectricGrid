# Phase 1 evidence inventory and disposition

Date: 2026-10-15

Branch: `refactor/v1-phase-01-structure`

## Inventory contract

`config/repository_inventory.json` is the machine-readable Phase 1 inventory.
It classifies every tracked file through non-overlapping rules, registers every
notebook individually, records external/ignored artifacts that matter to the
workflow, and names critical evidence that cannot be removed before migration.

Classification answers “what is this now?” It does not answer “is this
scientifically correct?” Candidate inputs and outputs remain unvalidated until
their owning phase establishes provenance, schema compliance, and scientific
checks.

## Repository findings

The tracked tree contains these main scientific groups:

| Group | Classification | Phase 1 disposition |
|---|---|---|
| current 2022 notebooks and two preparation scripts | candidate active | reconstruct into explained target stages |
| `_alt` notebooks and `temp_demand.py` | legacy evidence | preserve until formulas, repairs, and checks migrate |
| `src/ec_network_eval.py`, `src/helpers.py`, `src/paths.py` | candidate reusable code | extract and test incrementally |
| `src/_old/` and existing scenario registry | legacy evidence | quarantine until baseline validation |
| raw demand, geography, flow, carrier, and cost CSVs | source candidates | verify provenance and units in owning phases |
| processed demand/generation/profile CSVs | generated candidates | reproduce from named inputs before acceptance |
| topology repair and hydro matching tables | manual evidence | high-priority migration inputs |
| notebook-directory CSV/JSON files | generated audit/results | map to interim/results; leave in place for now |
| experiment result CSV | experimental result | retain as non-canonical diagnostic evidence |
| planning CSVs containing nuclear rows | mixed official/hypothetical | quarantine and split in Phase 8 |

The exact rule match for any path can be obtained with
`src.repository_inventory.matching_rules`. Automated tests fail on any tracked
or newly added non-ignored file that lacks exactly one classification.

At the Phase 1 review point, 108 tracked or pending non-ignored files are covered
by 28 non-overlapping rules. Eighteen additional rules cover all relevant ignored
artifacts currently present under `data/`, `literature/`, and `results/`, plus the
root runtime log. The detailed registers contain 24 notebooks, 12 Python modules,
15 local source documents/extracts, and eight external or generated
large-artifact records.

## Source-document and local-artifact register

The following ignored files are material evidence and must not be lost merely
because Git does not list them.

| Artifact family | Role | Disposition |
|---|---|---|
| `literature/PlanMaestroElectrifEc/cap_3_demanda.pdf` and table extracts | official-plan source candidate and manual extraction evidence | preserve; add edition/page/table provenance and licensing decision |
| `literature/PlanMaestroElectrifEc/cap_4_generacion.pdf` and generation/flow extracts | official-plan source candidate and manual extraction evidence | preserve; normalize in Phases 4 and 8 |
| `literature/PlanMaestroElectrifEc/cap_5_transmision.pdf` | official-plan source candidate | preserve; normalize in Phase 8 |
| `literature/_old/*.pdf` | earlier source-document versions | preserve until edition relationship is recorded |
| `data/raw/generation/generation_existing_2017.xlsx` | legacy plant source | preserve for lineage comparison |
| `data/raw/generation/generation_existing_2022.xlsx` | 2022 plant source candidate | preserve and validate in Phase 4 |
| `data/raw/generation/generation_future_2022.xlsx` | future plant source candidate | preserve and normalize in Phase 8 |
| `data/raw/gadm/Cantons.xlsx`, `data/raw/generation/Cantons.xlsx` | location lookup candidates | compare and consolidate with provenance |
| `data/raw/gadm/gadm41_ECU.gpkg` | plotting/boundary source candidate | decide Phase 2 retrieval/licensing policy |
| `data/raw/networks/base.nc` | required upstream network | published and verified; add to the executable Phase 2 manifest |
| `data/raw/cutouts/solar.nc`, `data/raw/cutouts/onwind.nc` | required upstream profiles | published and verified; add to the executable Phase 2 manifest |
| full ERA5/atlite cutout | upstream regeneration only | exclude from student preflight and this repository's Zenodo bundle |
| `data/processed/networks/ec_network_2022.nc` | generated topology candidate | reproduce in Phase 5; do not upload as input |
| `data/processed/networks/network_base_filled.nc` | generated assembled candidate | reproduce in Phase 5; publish only after Phase 6 validation if useful |
| alternate/intermediate networks and solved results | legacy/experimental evidence | retain locally until their unique role is migrated or ruled out |

Phase 2 must add every required ignored input to the formal artifact manifest.
Source PDFs and workbooks require a separate licensing/distribution decision; a
manifest may describe a manually supplied artifact without pretending it can be
redistributed.

## Manual scientific evidence migration register

| Evidence | What must survive | Migration phase |
|---|---|---:|
| current and legacy `line_fix.csv` | affected line identity, before/after values, rationale, source or modeling basis | 5 |
| orphan/LV-only bus reports | exact removal set and the rule that produced it | 5 |
| hydro aliases, anchors, match report, confidence/unmatched tables | manual aliases, distance/fuzzy decisions, unresolved cases | 4 |
| plant attachment and canton-not-found reports | normalized plant identity and location decision | 4–5 |
| 2017 consistency notebooks | pruning order, load shedding, transformer repair, solve checks, energy accounting | 5–6 |
| parsed planning CSVs and source extracts | original wording/value/unit, source location, transformation, confidence | 8 |
| expansion skip reports | why an official candidate could not be applied | 8 |

No item in this table is eligible for deletion until its migration gate has a
test and an owner-reviewed replacement.

## Nuclear and official-plan quarantine

The audit found hypothetical nuclear content in:

- `config/config.yaml`;
- `data/raw/generation/generation_future_renewable_2022.csv`;
- `data/raw/networks/EC_bus_expansion_check.csv`;
- `data/raw/networks/EC_bus_expansion_masterplan_2022.csv`;
- `data/raw/networks/EC_line_expansion_masterplan_2022.csv`;
- `data/processed/generation/nuclear_layout.csv`;
- `data/processed/generation/powerplants_all.csv` and
  `powerplants_future.csv`;
- the old scenario modules, registry, scenario notebooks, and scenario-design
  document;
- ignored alternate networks and generated expansion reports.

Three line rows explicitly state that they are auto-generated connectors and not
from the PDF. Three future-generation rows are labeled assumptions. Corresponding
buses are also present in tables named as Master Plan tables. The filenames are
therefore misleading: these files are classified as mixed and cannot be consumed
as official expansion inputs.

Phase 8 will create clean official tables from cited rows and put any retained
nuclear concept in a separate assumption table. Phase 1 does not rewrite the
mixed files because doing so before row-level provenance is established could
discard manual work.

## Initial schemas

Two initial contracts are now present:

- `config/schemas/artifact_manifest.schema.json` defines the minimum metadata
  needed for checksum-pinned external artifacts;
- `config/schemas/planning_project.schema.json` defines common provenance and
  official-versus-assumption fields for generation, line, transformer, and
  substation projects.

They are version-one design inputs. Phase 2 will instantiate and test the artifact
manifest. Phase 8 will refine and apply the planning schema to normalized tables.

## Cleanup candidates—not deletion decisions

Likely Phase 10 candidates include unrelated/scratch notebooks, exact or semantic
copies, stale notebook-directory outputs, obsolete scenario registries, temporary
logs, old solved networks, and superseded source extracts. Their presence in this
list is not approval to delete them. The migration gates in
`docs/REPOSITORY_ARCHITECTURE.md` apply.

## Phase 1 conclusion

The tree is now classifiable without relying on filenames alone. Active
reconstruction inputs, legacy evidence, generated candidates, experimental
results, and hypothetical scenario content have explicit boundaries. No tracked
or local scientific artifact was moved, deleted, downloaded, or regenerated.
