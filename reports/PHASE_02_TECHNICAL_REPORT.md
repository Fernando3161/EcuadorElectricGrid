# Phase 02 technical report

## Report metadata

- Phase: 02 — Artifact manifest and mandatory preflight
- Branch: `refactor/v1-phase-02-data-preflight`
- Starting revision: `6a403a3`
- Author: Codex, under repository-owner direction
- Reviewer: repository owner; accepted by authorizing Phase 03
- Date: 2026-10-15
- Environment: existing `pypsa-earth-ec` Conda environment
- Heavy execution: limited to the owner-authorized three-file Zenodo retrieval

## Objective and acceptance criteria

Phase 02 must ensure that missing external inputs are discovered and safely
obtained before scientific processing begins.

| Acceptance criterion | Status | Evidence |
|---|---|---|
| Versioned manifest contains the required external inputs | Pass | `config/artifacts.yaml` |
| Exact Zenodo URI, size, SHA-256, license, and provenance are pinned | Pass | manifest and schema tests |
| Full cutout is excluded | Pass | manifest boundary test |
| Generated cleaned/assembled networks are excluded as source inputs | Pass | manifest boundary test |
| Stage dependency closure is deterministic | Pass | all ten stage IDs tested |
| Cached files are verified before reuse | Pass | unit test and real offline run |
| Missing files download safely and atomically | Pass | mocked test and empty-root Zenodo run |
| Invalid files never become authoritative | Pass | partial and replacement tests |
| Downloads are resumable | Pass | HTTP Range unit test |
| NetCDF structure is validated | Pass | fixture test and three real files |
| Approved environment is checked without installation | Pass | runtime report and environment tests |
| Notebook 00 uses the common preflight | Pass | notebook contract test and execution |
| One command obtains all currently published external inputs | Pass | empty-root real retrieval |
| Owner accepts the complete phase | Pass | owner proceeded to Phase 03 after reviewing the Phase 02 handoff |

## Inputs and provenance

The manifest pins Zenodo version record
[`10.5281/zenodo.23392049`](https://doi.org/10.5281/zenodo.23392049), version 1.0,
published under CC BY 4.0.

| Artifact ID | Destination | Bytes | SHA-256 |
|---|---|---:|---|
| `pypsa_earth_base_ec` | `data/raw/networks/base.nc` | 1,346,893 | `376db45456d13049a9564947f1dac73bf96e6aff09fc5d0b30c07102ad1cf9af` |
| `renewable_profile_solar_ec` | `data/raw/cutouts/solar.nc` | 1,760,868 | `28c6ab3a6bd537869bf6811371d3bb76210fa744b4a11ef46205be91755578c1` |
| `renewable_profile_onwind_ec` | `data/raw/cutouts/onwind.nc` | 1,760,868 | `ace2b9524dec52f2235aabe1b398909f24f7c096f21a20d7f0dfb06c17922332` |

The base-network structural contract requires 8,760 snapshots, 313 buses, 247
lines, 62 transformers, the corresponding index variables, and network name
`PyPSA-Earth`. Each renewable profile requires 8,760 hourly rows, 24 buses, and
the `profile` and `capacities` variables.

## Methods and decisions

### Manifest and dependency resolution

The YAML manifest is validated against a JSON Schema and then receives semantic
checks for unique IDs/destinations, safe repository-relative paths, known stage
references, and acyclic stage dependencies. Artifact selection is the union of
the chosen stage and its dependency closure.

Stage 00 deliberately resolves all three published external inputs so a student
does not reach a later supported stage before discovering one is absent. Later
phases must extend the manifest before accepting any new untracked runtime input.

### Download and installation safety

Retrieval writes to a sibling `.part` path. Existing partial data is resumed with
an HTTP Range request when supported. The partial file must pass size, SHA-256,
and structural validation before atomic installation. If an authoritative path
contains an invalid file, that file is preserved with a timestamped suffix before
replacement. Failures retain the partial data but never give it the authoritative
name.

### Environment reconciliation

The existing environment was inspected, not changed. The environment YAML typo
and empty dependency were corrected, and packages already present for manifest,
download, NetCDF, and notebook handling were made explicit. Runtime preflight
checks the environment name, Python version, and package version constraints.

The generic installed Jupyter kernelspec resolves to another Python installation.
To prevent accidental system-Python execution, `src/notebook_runner.py` creates a
temporary kernelspec pointing to the current approved interpreter. It is removed
after execution and does not install or register anything persistently.

## Implementation

Added:

- `config/artifacts.yaml`;
- `src/artifacts.py`;
- `src/environment.py`;
- `src/preflight.py`;
- `src/notebook_runner.py`;
- `notebooks/00_preflight_and_data_inventory.ipynb`;
- `tests/test_artifact_preflight.py`;
- `docs/PREFLIGHT.md`;
- `docs/ENVIRONMENT.md`;
- this report.

Updated:

- the artifact-manifest JSON Schema;
- `config/environment_ec.yaml`;
- repository inventory/notebook/module classifications;
- README, development workflow, and data/artifact documentation.

No existing data, source document, scientific notebook, generated network, or
result was changed or deleted. Temporary downloaded test artifacts were removed
after verification.

## Tests and execution

### Automated suite

```powershell
python -m unittest discover -s tests -v
```

Executed through the approved environment: 29 tests passed in under one second.

The Phase 2 tests cover:

- manifest schema and semantic validation;
- exact published input boundary;
- stage dependency closure and unknown stages;
- valid cache reuse without a download;
- missing and invalid offline behavior;
- checksum failure;
- mocked download and atomic installation;
- preservation of an invalid prior authoritative file;
- rejection and retention of an invalid partial download;
- resumption through an HTTP Range request;
- NetCDF dimension, variable, and attribute checks;
- path traversal rejection;
- approved environment identity/package constraints;
- early actionable CLI rejection when a different environment is active;
- reconciled environment YAML;
- temporary kernelspec interpreter identity;
- Notebook 00's first code cell calling common environment/artifact preflight;
- all retained Phase 1 repository-contract tests.

### Real cached/offline verification

```powershell
python -m src.preflight --stage 00_preflight --offline
```

Result: environment PASS; all three cached artifacts `valid`.

### Real missing-file retrieval

The owner authorized downloading the three small files. Preflight was run with an
empty temporary root rather than moving or deleting the working copies. Result:

- first run: all three artifacts `downloaded` and verified;
- second run with `--offline`: all three artifacts `valid` from cache;
- temporary root removed after success.

### Notebook execution

```powershell
python -m src.notebook_runner notebooks/00_preflight_and_data_inventory.ipynb
```

Result: successful top-to-bottom execution through the approved interpreter. No
executed copy was written and the source notebook remains output-free.

Not run:

- no demand/generation preparation;
- no network mutation or assembly;
- no optimization or annual solve;
- no scenario batch;
- no full ERA5 cutout retrieval.

## Results

The repository now has one command and one explanatory notebook that verify the
runtime and make the external input set available without an unplanned manual
download. A fresh empty root recovered all three inputs directly from the pinned
Zenodo version and immediately reused them offline.

The environment specification is syntactically usable for audit purposes, and
the command refuses a wrong interpreter or incompatible preflight package before
downloading data.

## Validation figures

No plot is needed for this infrastructure phase. The pass/fail tables emitted by
preflight are the relevant evidence; they are recorded as commands and results
above rather than as a decorative image.

## Scientific validation table

Not applicable. Phase 02 validates artifact integrity and availability only. It
does not run or validate the 2022 energy model.

## Limitations and unresolved issues

- The manifest currently covers the three published external NetCDF inputs.
  Later phases must register any additional untracked runtime source before their
  notebook is accepted.
- Source PDFs and workbooks remain provenance evidence with unresolved
  redistribution decisions; they are not silently downloaded or substituted.
- The environment YAML is an audited specification, not a fully solved lock file.
- GIS/weather configuration warnings remain assigned to the first phase that
  exercises those operations.
- HTTP interruption recovery is covered by range-request logic and a deterministic
  unit test; an intentionally interrupted live Zenodo transfer was not necessary
  for these small files.

## Reproducibility handoff

Review with:

```powershell
git status --short --branch
git diff --check
python -m unittest discover -s tests -v
python -m src.preflight --stage 00_preflight --offline
python -m src.notebook_runner notebooks/00_preflight_and_data_inventory.ipynb
```

Do not delete the working NetCDF files to test retrieval. The documented empty
temporary-root method proves the same missing-file path without risking the only
local cache.

## Conclusion and next-phase recommendation

Phase 02 is accepted. All acceptance criteria pass. The next phase is
`refactor/v1-phase-03-demand-statistics`, which must select and version the
authoritative 2022 statistics, define the comparison boundary and carrier
crosswalk, and fix the 5% acceptance formula before model calibration begins.
