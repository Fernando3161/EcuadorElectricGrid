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

No Zenodo DOI, exact download URL, or release checksum is currently recorded in
this repository. Publication is therefore a Phase 2 owner action and automated
retrieval cannot yet be considered complete.

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
