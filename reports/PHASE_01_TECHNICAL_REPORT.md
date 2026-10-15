# Phase 01 technical report

## Report metadata

- Phase: 01 — Evidence preservation and repository structure
- Branch: `refactor/v1-phase-01-structure`
- Starting revision: `cc6a929`
- Author: Codex, under repository-owner direction
- Reviewer: repository owner; final review pending
- Date: 2026-10-15
- Environment: existing `pypsa-earth-ec` Conda environment
- Heavy execution: forbidden; no notebook, download, or optimization run

## Objective and acceptance criteria

Phase 01 creates a complete, evidence-preserving map of the disordered repository
and defines the structure into which it will be reconstructed.

| Acceptance criterion | Status | Evidence |
|---|---|---|
| Every tracked file has one classification | Pass | machine-readable inventory and coverage test |
| Every notebook has an individual disposition | Pass | 24-entry notebook register |
| 2017/2018 and 2022 chains are mapped | Pass | `docs/NOTEBOOK_LINEAGE.md` |
| Manual repairs, matches, and source extracts are protected | Pass | critical-evidence migration gates |
| Active, generated, legacy, experimental, and hypothetical material are distinct | Pass | inventory taxonomy and target architecture |
| Notebook-directory outputs have target data zones | Pass | lineage output map |
| Official Master Plan candidates are separated conceptually from nuclear assumptions | Pass | mixed tables quarantined and tested |
| Initial manifest and planning-row schemas exist | Pass | `config/schemas/` |
| Supported documentation/code contains no machine-specific paths | Pass | automated scan |
| Cheap Phase 1 tests pass | Pass | eleven tests, no failures |
| Owner accepts the phase | Pending | requires review of this report and branch diff |

## Inputs and provenance

The inventory covers the tracked tree plus relevant ignored files present in the
workspace: source PDFs/workbooks, required upstream NetCDF inputs, generated
networks, alternate networks, solved results, figures, and logs. It uses notebook
source/output inspection, CSV headers and row counts, import tracing, filename
writer/reader tracing, and Phase 0 network fingerprints.

The required upstream files remain:

| Artifact | Destination | Bytes | SHA-256 |
|---|---|---:|---|
| PyPSA-Earth base network | `data/raw/networks/base.nc` | 1,346,893 | `376db45456d13049a9564947f1dac73bf96e6aff09fc5d0b30c07102ad1cf9af` |
| solar profiles | `data/raw/cutouts/solar.nc` | 1,760,868 | `28c6ab3a6bd537869bf6811371d3bb76210fa744b4a11ef46205be91755578c1` |
| onshore-wind profiles | `data/raw/cutouts/onwind.nc` | 1,760,868 | `ace2b9524dec52f2235aabe1b398909f24f7c096f21a20d7f0dfb06c17922332` |

The checksums were rechecked by the test suite and then independently against
downloads from Zenodo record `10.5281/zenodo.23392049`. All three remote files
matched the local fingerprints exactly. Zenodo reports record version 1.0 and a
CC BY 4.0 license.

## Methods and decisions

### Classification method

Each tracked or pending non-ignored path must match exactly one inventory rule.
The allowed classifications distinguish governance, active candidates, manual
evidence, generated candidates, legacy evidence, mixed official/hypothetical
tables, hypothetical scenarios, experiments, and external artifacts.

The classification deliberately avoids calling the current 2022 chain “active”
without qualification. It is reconstruction material and not yet the supported
student workflow.

### Evidence preservation

No existing file was moved or deleted. Critical evidence has an explicit
migration gate. This is especially important for line repairs, hydro aliases and
matching decisions, orphan/pruning outputs, plant attachment decisions, the two
consistency notebooks, and manually parsed planning tables.

### Nuclear boundary

The audit confirmed nuclear assumptions in current files whose names imply
official Master Plan content. Three connector rows explicitly say they were
auto-generated and not from the PDF. Related future-generation rows identify
themselves as assumptions. Those tables are now machine-classified as mixed and
are forbidden as official-plan inputs until Phase 8 creates source-traceable
official tables and separate assumption tables.

### Zenodo publication

The owner published `base.nc`, `solar.nc`, and `onwind.nc` in versioned Zenodo
record `10.5281/zenodo.23392049`. Direct downloads were verified by byte size,
SHA-256, and Zenodo-reported MD5. The full cutout and the two generated
cleaned/assembled networks remain correctly excluded. Phase 2 can now instantiate
the executable manifest and test preflight retrieval.

## Implementation

Added:

- `config/repository_inventory.json`;
- `config/schemas/artifact_manifest.schema.json`;
- `config/schemas/planning_project.schema.json`;
- `src/repository_inventory.py`;
- `tests/test_phase01_repository_contract.py`;
- `docs/REPOSITORY_ARCHITECTURE.md`;
- `docs/NOTEBOOK_LINEAGE.md`;
- `docs/PHASE_01_INVENTORY.md`;
- `docs/ZENODO_HANDOFF.md`;
- this report.

Updated:

- `README.md` to reflect Phase 1 and link the new documents;
- `docs/DATA_AND_ARTIFACTS.md` to link the upload handoff;
- the Phase 0 report to record owner acceptance.

No scientific data, notebook, existing module, network, source document, or
result was modified, relocated, downloaded, or deleted.

## Tests and execution

Command:

```powershell
python -m unittest discover -s tests -v
```

The command was executed through the approved environment interpreter. Result:
eleven tests passed in approximately 0.10 seconds.

The owner additionally authorized a one-time small-file publication check. Each
direct Zenodo URL was downloaded to unique temporary storage. All byte sizes and
SHA-256 values matched the table above, and the downloaded MD5 values matched the
Zenodo API metadata. The temporary copies were not placed in the repository.
They were removed after verification.

Tested behavior:

- every tracked or pending non-ignored file matches exactly one classification;
- every relevant ignored artifact present under the scientific data, literature,
  and results areas matches exactly one classification;
- every notebook is registered exactly once;
- every Python module under `src/` and `notebooks/` is registered exactly once;
- source-document entries are unique and repository-relative;
- named critical evidence is present;
- the three required NetCDF files match expected sizes and SHA-256 values when
  present;
- all three required artifact records contain the verified version DOI, direct
  HTTPS download URL, and CC BY 4.0 license;
- contaminated planning/generation tables are classified as mixed;
- initial JSON schemas parse and declare required fields;
- governance documents and candidate Python source contain no machine-specific
  paths;
- relative Markdown links resolve.

Not run by design:

- notebooks and Python preparation scripts;
- source PDF parsing or workbook conversion;
- network mutation, assembly, optimization, or annual validation;
- preflight-driven downloads or missing/corrupt/cache behavior;
- file moves or deletions.

## Results

The repository now has three complementary controls:

1. a machine-readable inventory that makes unclassified additions fail tests;
2. a human-readable dependency and migration map for both development histories;
3. target schemas/architecture that later phases can implement without losing
   the scientific decisions embedded in legacy files.

At review time, 108 tracked or pending non-ignored files match 28 classification
rules, while 18 rules cover the relevant ignored-artifact tree. Separate exact
registers cover 24 notebooks, 12 Python modules, 15 source documents/extracts,
and eight external or generated large-artifact records.

The audit also resolved a dangerous naming problem: current files labeled as
Master Plan inputs contain hypothetical nuclear buses, lines, and generators.
They remain preserved, but are quarantined from official-plan use.

## Validation figures

No binary figure is warranted for this structural phase. The dependency graphs
in `docs/NOTEBOOK_LINEAGE.md` are the reproducible visual evidence. They show the
writers/readers of the two notebook chains and the handoff into the target stages.

## Scientific validation table

Not applicable. Phase 01 changes no model inputs or outputs and makes no claim
about the 2022 carrier-composition tolerance.

## Limitations and unresolved issues

- Source-document licensing and redistribution rights are not yet fully recorded.
- The current planning tables need row-level citations and may contain additional
  assumptions beyond the confirmed nuclear rows.
- Existing CSV encodings and headers are inconsistent and require normalization
  in their owning phases.
- `config/environment_ec.yaml` and runtime configuration are retained as legacy
  evidence and require Phase 2 reconciliation.
- Existing notebook saved output may contain stale machine paths; the supported
  refactored notebooks will be created and output-cleared in later phases.
- The current scenario registry points at placeholder or generated artifacts and
  remains unsupported.

These limitations are assigned to later phases and do not justify deleting the
evidence during Phase 1.

## Reproducibility handoff

Review with:

```powershell
git status --short --branch
git diff --check
python -m unittest discover -s tests -v
```

After owner acceptance, commit Phase 1. Phase 2 can then place the verified DOI,
direct file URLs, license, sizes, and checksums from `docs/ZENODO_HANDOFF.md` into
the executable manifest and exercise the preflight without inventing artifact
provenance.

## Conclusion and next-phase recommendation

Phase 01 is implementation-complete and ready for owner review. Its acceptance
criteria pass except for owner approval. The next phase is
`refactor/v1-phase-02-preflight`: reconcile the environment specification, create
the executable artifact manifest, implement atomic checksum-pinned retrieval, and
test missing/corrupt/cached behavior without running scientific notebooks.
