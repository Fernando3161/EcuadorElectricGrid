# Data and external artifact contract

## Principle

The repository must know every required input before Stage 1 begins. A shared
manifest and preflight—not ad hoc notebook cells—will determine whether each file
is present, valid, downloadable, or blocking.

## Target manifest fields

Each artifact entry must contain:

- stable logical ID and description;
- stage(s) that require it;
- authoritative repository-relative destination;
- source organization and source record/edition;
- persistent URI or DOI plus exact file URL;
- expected byte size and SHA-256;
- media type and optional archive member;
- license/terms and required attribution;
- whether the file is source, external canonical input, or generated product;
- validation routine and expected structural metadata.

Unknown URIs and checksums must be explicit placeholders that make preflight fail
with an actionable message. They must never be replaced by an unversioned fallback.

## Required upstream external artifacts (Phase 0 conclusion)

The supported notebook workflow requires three upstream PyPSA-Earth-derived
artifacts. They are inputs to this repository and are owned by the companion
`EcuadorElectricGridData` project:

| Candidate | Repository destination | Observed size | Observed structure/use |
|---|---|---:|---|
| PyPSA-Earth base grid | `data/raw/networks/base.nc` | 1,346,893 B | 8,760 snapshots; 313 buses; 247 lines; 62 transformers |
| Solar profiles | `data/raw/cutouts/solar.nc` | 1,760,868 B | 8,760 × 24 buses; `profile`, `capacities` |
| Onshore-wind profiles | `data/raw/cutouts/onwind.nc` | 1,760,868 B | 8,760 × 24 buses; `profile`, `capacities` |

The active 2022 notebooks load `solar.nc` and `onwind.nc` directly. They do not
load the full ERA5/atlite cutout. These three files—not the cutout or an already
cleaned Ecuador network—form the current minimum external NetCDF input set.

The three inputs are published in Zenodo record
[`10.5281/zenodo.23392049`](https://doi.org/10.5281/zenodo.23392049), version
1.0, under CC BY 4.0. Phase 1 downloaded each published file to temporary storage
and verified that its byte size and SHA-256 exactly match the values above.

The direct URLs and the safe Phase 2 retrieval-test procedure are recorded in
`docs/ZENODO_HANDOFF.md`. Automated retrieval is not complete until Phase 2
instantiates the manifest and tests missing, cached, invalid, and interrupted
download behavior.

## Generated network products—not source artifacts

The following files are produced by this repository's notebooks and must not be
treated as raw or upstream inputs:

| Generated product | Writer | Main transformations before export |
|---|---|---|
| `data/processed/networks/ec_network_2022.nc` | `notebooks/04_EC_network_base.ipynb` | loads `base.nc`; snaps voltage levels; identifies/removes orphan elements; applies line fixes and manual topology edits |
| `data/processed/networks/network_base_filled.nc` | `notebooks/05_EC_network_base_run.ipynb` | loads `ec_network_2022.nc`; prunes voltage scope; attaches demand and plants; assigns solar, wind, and hydro profiles; adds costs/carriers; repairs transformer parameters; adds load-shedding generators |

Both files are ignored by Git and have no file history. The instances observed in
Phase 0 are unvalidated reconstruction evidence. Do not upload or publish them as
authoritative artifacts. Phases 3–6 must make their complete transformation chain
reproducible from upstream inputs and reviewed small tables.

After Phase 6 passes scientific validation, the resulting validated assembled
network may be published as a versioned convenience/release artifact with its
manifest, checksum, source revisions, and validation report. That future release
artifact would remain a generated product; it would not become raw source data.

## Full ERA5/atlite cutout classification

`data/raw/cutouts/cutout-2013-era5.nc` is an approximately 18.66 GB upstream
regeneration artifact. Repository inspection found it referenced only by
`notebooks/_alt/07_EC_renewables_estimation.ipynb` and the legacy scratch notebook
`notebooks/_alt/07_EC_saved.ipynb`. The former uses the cutout to create or replace
`solar.nc` and `onwind.nc`; when those derived files already exist, it loads them
instead.

The active 2022 assembly and expansion notebooks consume `solar.nc` and
`onwind.nc` and never open the full cutout. Therefore the cutout is not a required
download for students, is not part of this repository's normal preflight, and
should not be included in this repository's Zenodo input bundle. Its generation,
storage, and provenance belong to the companion `EcuadorElectricGridData` project.

## Other ignored evidence

The audited workspace also contains:

- 2017 and 2022 existing/future generation Excel sources;
- Ecuador Master Plan demand, generation, and transmission chapter PDFs;
- PDF-extracted Excel tables;
- alternate/intermediate networks and multiple solved experiment NetCDF files.

These are not automatically canonical. Phase 1 must classify each file, identify
its source and license, and decide whether to commit a small normalized derivative,
publish it externally, regenerate it, archive it as evidence, or discard it.

## Phase 3 demand and statistical inputs

The two official CENACE annual reports supporting the extracts are committed
under `literature/CENACE/`. Their original publication URLs, byte sizes, and
SHA-256 checksums are recorded alongside them. They are provenance evidence, not
runtime notebook inputs, so the stage preflight checks the extracted CSVs rather
than repeatedly parsing the PDFs.

Stage 01 adds three checksum-pinned repository inputs to the common manifest:

| Artifact | Role | Runtime treatment |
|---|---|---|
| `data/raw/demand/demand_profiles_EC.csv` | PyPSA-Earth 0.7.0 hourly shape and bus-share template | tracked, checksum-verified, never auto-substituted |
| `data/raw/demand/cenace_sni_2022_monthly_demand.csv` | official monthly 2022 energy and peak controls | tracked source extract with a CENACE table locator |
| `data/raw/statistics/cenace_2022_net_supply.csv` | source-faithful CENACE net-supply rows | tracked source extract with page/table locators |

The demand template is numerically identical to `loads_t.p_set` in its forensic
PyPSA-Earth source network. It supplies only normalized chronology and spatial
shares; official CENACE values control the 2022 energy and peak.

The Stage 01 notebook generates
`data/processed/demand/demand_profiles_2022_mw.csv`. This approximately 40 MB
time series is ignored because it is reproducible. Its values are MW, unlike the
legacy tracked `demand_profiles_2022.csv`, whose values are kW. The legacy file
remains evidence until Phase 10; supported assembly must consume the MW product.

The small normalized carrier table
`data/processed/statistics/ecuador_sni_2022_supply_reference.csv` is tracked so
the later annual validation reads an inspectable, schema-tested interface. See
`VALIDATION_2022.md` for the boundary and formula.

## Phase 4 generation and profile artifacts

The private/local Excel files are migration evidence, not runtime dependencies.
Their supported open interfaces are checksum-pinned repository artifacts:

| Artifact | Zone | Role |
|---|---|---|
| `data/raw/generation/generation_existing_2022.csv` | raw | all 300 source rows with original values and row provenance |
| `data/interim/generation/plant_location_evidence_2022.csv` | interim | one-to-one location evidence for the 133 SNI rows |
| `data/raw/generation/plant_location_overrides_2022.csv` | raw | reviewed plant-specific mapped coordinates that supersede canton proxies |
| `data/raw/generation/cenace_2022_commissioning_events.csv` | raw | official in-year operating dates retained separately from conflicting plant capacities |
| `data/processed/generation/generators_2022.csv` | processed | normalized and reconciled model-facing fleet |
| `data/raw/generation/caudals.csv` | raw | eight legacy monthly flow proxy series with unresolved primary provenance |
| `data/interim/generation/hydro_profile_assignments_2022.csv` | interim | reviewed plant-to-flow mapping with basis and confidence |

The workbook migration can be audited with
`scripts/migrate_phase04_generation_sources.py`, but ordinary notebook execution
does not require those workbooks. Stage 03 also requires the external `solar.nc`
and `onwind.nc` files already managed by Zenodo preflight.

The hourly hydro and variable-renewable output CSVs are ignored generated
products. They are recreated by the Stage 03 notebook and checked for 8,760-hour
alignment, exact plant coverage, finite values, and `[0, 1]` bounds. See
`GENERATION_AND_PROFILES_2022.md` for assumptions and evidence limitations.

## Phase 5 topology and assembled-network products

`data/interim/network/topology_repairs_2022.csv` is the reviewed small interface
for the five legacy 138 kV corridor additions and the transformer-resistance
assumption. Every row records its rationale, notebook locator, confidence, and
that it is not an official Master Plan fact. Deterministic mutation, connected-
component, load-attachment, and generator-attachment audits are written beside
it by the Stage 04 and 05 notebooks.

`data/interim/network/generator_connection_evidence_2022.csv` is the reviewed
interface for plant-specific physical connections that override geographic
nearest-bus attachment. It currently records the official 138 kV S/E Esmeraldas
connections for Termoesmeraldas I and II and the separately verified mapping to
model bus 42.

The Phase 5 topology figures use the checksum-pinned GADM 4.1 Ecuador
GeoPackage at `data/raw/gadm/gadm41_ECU.gpkg`. Its license restricts use to the
GADM terms; it remains an external input and is not committed as project data.

The supported transmission boundary is 138/230/500 kV. The Stage 04 generated
topology contains 110 buses, 111 lines, 19 transformers, and four connected
components. The smaller components are reported rather than silently joined.

Stage 05 regenerates `data/processed/networks/network_base_filled.nc` as an
**unsolved candidate** containing 8,760 snapshots, 251 source-preserving loads,
133 real generators, and 110 separately labeled emergency load-shedding
generators. It conserves 27,561.62 GWh of annual demand and 7,489.65 MW of real
nominal capacity. Its run-specific checksum and structural metadata are written
beside the Phase 5 audits. It remains a generated product and must not be described or
published as a validated network before the Phase 06 acceptance gate passes.

## Target data zones

| Zone | Meaning | Hand edited? | Git policy |
|---|---|---|---|
| `data/raw/` | immutable source tables or external canonical files | never | small/licensed sources only; large files ignored |
| `data/interim/` | reproducible parsing and matching outputs | never | normally ignored; selected audit tables may be tracked |
| `data/processed/` | normalized, model-ready tables | only through reviewed source changes | small tables tracked; large networks external |
| `results/` | run-specific outputs and figures | never | ignored except curated examples/reports |
| `reports/figures/` | curated phase evidence regenerated by code | never | tracked when useful and reasonably sized |

Generated CSVs must not live in `notebooks/` in the target structure.

## Preflight behavior

The Phase 2 implementation is configured by `config/artifacts.yaml` and exposed
through `python -m src.preflight`. See `docs/PREFLIGHT.md` for commands, status
definitions, stage dependency closure, and recovery behavior.

The preflight must:

1. determine the selected stage and its complete dependency closure;
2. validate existing files before deciding they are reusable;
3. download to a temporary sibling path with resume support where practical;
4. verify size/checksum and structural expectations;
5. atomically move the valid file into place;
6. produce a concise inventory with `valid`, `downloaded`, `missing`, `invalid`,
   or `not-required` status;
7. stop all downstream work if a required artifact is unresolved.

Credentials, if ever required, must be checked during preflight and never stored
in the repository. A failed run must retain useful logs and downloads but must not
leave a partial authoritative artifact.
