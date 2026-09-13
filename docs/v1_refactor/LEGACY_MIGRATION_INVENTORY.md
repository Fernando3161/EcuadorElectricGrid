# Legacy migration inventory

## Purpose

This document preserves scientific and implementation knowledge found during
Phase 1 before obsolete notebooks and modules were removed from the supported
tree. It is a migration checklist, not an endorsement of every historical
assumption. Git history remains the source for the original implementation.

## Historical 2022 network construction

`04_EC_network_base.ipynb` loaded `base.nc`, normalized network voltage labels,
checked topology, removed true orphan buses, applied five manually defined
138 kV line fixes, and exported `data/processed/networks/ec_network_2022.nc`.
The historical voltage normalization considered 48, 69, 138, 230, and 500 kV;
the supported v1 runtime scope is instead explicitly 138, 230, and 500 kV.
The five retained topology fixes are in `data/processed/networks/line_fix.csv`:
bus pairs 181-277, 280-91, 60-102, 278-256, and 24-64. Comments in the notebook
said that 60-102 and 24-64 replace connectivity lost with 69 kV filtering.
These are modeling decisions requiring provenance review, not newly verified
official infrastructure.

The notebook reported pre-cleaning voltage levels and transformer bridge
checks, including a missing 48-69 kV bridge. Those lower-voltage diagnostics
are historical construction evidence and are not v1 supported-scope
requirements.

`src/ec_network_eval.py` implemented construction-era primary/foreign-key,
coordinate, voltage, line geometry/rating, orphan-bus, topology, and
transformer-bridge diagnostics. Its defaults included 48/69 kV bridge checks,
so it was removed from the active package rather than silently changing its
scientific meaning. Phase 3 should reimplement the applicable structural checks
against the canonical 138/230/500 kV artifact.

## Historical runnable 2022 workflow

`05_EC_network_base_run.ipynb` loaded `ec_network_2022.nc`, pruned buses below
137 kV, attached the processed 2022 demand profile, mapped unmatched loads and
plants by coordinates, attached renewable and hydro availability, merged
technology costs, added load-shedding generators, patched zero transformer
impedances, solved with HiGHS, and calculated operational diagnostics.

Items that later phases must migrate deliberately and validate include:

- load and generator bus remapping, including all manually/nearest-coordinate
  mappings rather than silently recomputing them;
- carrier normalization from Spanish/source labels to PyPSA carrier names;
- hydropower `p_max_pu` attachment by generator identifier;
- renewable profile assignment by nearest province centroid;
- load-shedding carrier and generators (historical penalty was 1,000,000 per
  MWh and capacity was 10,000 MW per bus);
- transformer fallback values used experimentally (`r_pu=0.01`; historical
  cells used both `x_pu=0.10` and `x_pu=0.05` in different patches);
- line impedance overrides (`x=0.1`, `r=0.01`) that must not become canonical
  without scientific confirmation;
- KPI calculations for demand, generation, curtailment, load shedding, line
  and transformer loading, and installed capacity.

The notebook depended on a neighboring `pypsa-earth` checkout and referenced
the missing `.nc` artifacts. Neither dependency is supported after v1.

## Demand preparation

The two Python demand scripts were moved to `scripts/preprocessing/`. One
scales each month of the complete hourly/nodal template to the observed 2022
monthly energy total and moves timestamps to 2022. The second combines the
observed 2022 annual total with 2023-2032 Master Plan projections and performs
a historical linear extension. These behaviors are preserved unchanged except
for repository-root resolution after the move. Phase 4 must reconcile the
forecast script with the canonical D1/D2/D3 and configurable growth-method
specification; it must not silently treat this historical linear extension as
the only supported methodology.

## Generation and hydrology preparation

`03_a_process_generation.ipynb` normalized existing/future plant names,
technologies, locations, and capacities. `03_c_hydro_profiles.ipynb` generated
hourly availability from monthly caudal series. Its matching order was manual
alias, token rule, exact normalized identifier/name, fuzzy match (cutoff 0.72),
then nearest geographic anchor. Run-of-river and reservoir profiles used
different transformations. The manual aliases, match report, and anchor report
remain under `data/processed/generation/hidro_max_profiles/`; empty duplicate
unmatched/low-confidence reports were removed. Phase 3 or 4 must migrate and
test this logic before regeneration.

## Transmission expansion provenance

The generated notebook-directory expansion report was moved to
`data/processed/networks/expansion_line_application_report.csv`. It retains
project names, source text, mapped endpoints, timing, and modeling notes. The
associated `skipped_expansion_lines.csv` preserves unresolved applications.
Notably, `LT_Delsitanisagua_Cumbaratza_138` maps both modeled endpoints to bus
97 although the source names distinct substations. This ambiguity remains
unresolved and must not be silently repaired.

Future transformer rows were observed to repeat generic electrical defaults.
Later normalization must distinguish source parameters from modeling defaults.

## Removed scenario implementations

The `_old` dataclasses/registry/pipeline and the copied expansion/scenario
notebooks contained useful architectural ideas: typed scenario dimensions,
load scaling, network selection, generator filtering, renewable/nuclear
additions, run orchestration, and saved outputs. They also contained broken
imports, 2018/2024 reference years, a neighboring-repository dependency, and
conflicting nuclear paths (0.9/2.1/3.0 GW) and delay cases (5/7/9 years).

`literature/scenario_deve_documentation.md` repeated the obsolete 7-year
baseline delay and replacement-mode nuclear cases. `nuclear_layout.csv`
contained a different 300 MW-unit layout. These are explicitly superseded by
AGENTS.md: delays are 5/8/12 years and additive SMR additions are 0.5 GW in
2035, 2040, 2045, and 2050. Phase 4 should reuse concepts, not values or code,
from the removed implementations.

The minimal JSON registry under `src/` was also removed. It referenced a
nonexistent generation file, empty paths, and the missing network, and did not
represent the required scenario dimensions. Phase 4 will create the
authoritative configuration-driven registry.

## Historical outputs removed from notebooks

Notebook-local CSV/JSON files were generated diagnostics or exports, not
authoritative inputs. They included buses/lines/transformers exports, attached
plants, load/generator time series, orphan/LV buses, load shedding, and skipped
application logs. The uniquely relevant expansion reports were migrated as
described above; transformer sensitivity evidence remains in
`results/experiments/experiment_x_defaults_results.csv`. All other notebook
outputs remain recoverable through Git history and should be regenerated from
the canonical artifact only after Phase 3 defines the supported workflow.

## Missing network references

Historical files referred to `base.nc`, `ec_network_2022.nc`,
`network_base_filled.nc`, and solved network variants. No `.nc` file was
present during Phase 1. Phase 2 will define the archive manifest/bootstrap;
Phase 3 will identify the canonical validation fingerprint. No network was
reconstructed or invented during this cleanup.
