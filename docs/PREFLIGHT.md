# Mandatory environment and data preflight

## Purpose

Preflight is the only supported entry point before scientific processing. It
checks the approved runtime, resolves the selected stage's dependency closure,
validates checksum-pinned external and repository artifacts, obtains missing
published files, and refuses
to continue while any required artifact is unresolved.

The executable manifest is `config/artifacts.yaml`. It contains the three external
NetCDF inputs established by the Phase 0/1 audit and the checksum-pinned tracked
inputs needed by Stage 01. A later phase must add every new runtime input before
its notebook is accepted.

## Commands

After activating the existing `pypsa-earth-ec` environment, run:

```powershell
python -m src.preflight --stage 00_preflight
```

This checks and, when necessary, retrieves all currently published external
inputs. To validate an existing cache without network access:

```powershell
python -m src.preflight --stage 00_preflight --offline
```

Machine-readable output is available with `--json`.

Notebook 00 is the explanatory interface to the same functions:

```powershell
python -m src.notebook_runner notebooks/00_preflight_and_data_inventory.ipynb
```

The runner creates a temporary kernelspec whose interpreter is the currently
running approved Python. It does not install a package, create an environment, or
register a persistent kernel.

## Status meanings

| Status | Meaning |
|---|---|
| `valid` | authoritative cached file passed size, SHA-256, and structural checks |
| `downloaded` | missing/invalid file was downloaded, verified, and atomically installed |
| `missing` | required file is absent and could not be obtained |
| `invalid` | authoritative file exists but fails validation and no valid replacement was installed |
| `not-required` | artifact exists in the manifest but is outside the selected stage closure |

Any `missing` or `invalid` result makes the command fail. All artifacts are
reported together so the user receives one complete early diagnostic.

## Retrieval safety

For each required artifact, preflight:

1. resolves a repository-relative destination and rejects path traversal;
2. validates an existing authoritative file before reusing it;
3. downloads to a sibling `.part` file, resuming with an HTTP range request when
   the server supports it;
4. validates byte size, SHA-256, and configured NetCDF structure on the partial
   path;
5. preserves an invalid prior authoritative file with an `.invalid-<timestamp>`
   suffix;
6. atomically moves only the verified partial file into the authoritative path;
7. retains an incomplete/invalid partial download for diagnosis and retry.

A failed download never receives the authoritative filename. A tracked repository
artifact has no download URL: if it is missing or altered, preflight fails and
asks the user to restore the reviewed repository version instead of substituting
an upstream document or unversioned file.

## Stage dependency closure

| Stage | Artifacts resolved |
|---|---|
| `00_preflight` | base network, solar profiles, onshore-wind profiles |
| `01_demand` | tracked hourly demand template, official monthly demand controls, official net-supply table |
| `02_generation` | none currently; Phase 4 must register any retained external source |
| `03_profiles` | solar and onshore-wind profiles |
| `04_topology` | base network |
| `05_assembly` | all three external inputs plus the three Stage 01 inputs through dependencies |
| `06_validation` | all six through dependencies |
| `07_planning` | none currently; Phase 8 must register any distributed source document |
| `08_scenario`, `09_comparison` | all six through dependencies |

This table is generated conceptually from manifest dependencies; code uses the
manifest rather than a hard-coded stage list.

## Published input boundary

Included:

- `base.nc`;
- `solar.nc`;
- `onwind.nc`.

Excluded deliberately:

- the full ERA5/atlite cutout, because profile regeneration belongs to the
  companion data project;
- `ec_network_2022.nc` and `network_base_filled.nc`, because they are generated,
  unvalidated products of this repository;
- solved and hypothetical scenario networks.

## Phase 2 verification evidence

The production manifest passed three independent paths:

- offline validation of the existing repository cache;
- online retrieval of all three files into an empty temporary root;
- a second offline run against the newly downloaded cache.

All files matched their pinned sizes, SHA-256 values, and NetCDF structures. The
temporary retrieval root was removed after the test. No scientific notebook or
network optimization was run.
