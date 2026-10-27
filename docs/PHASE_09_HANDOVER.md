# Phase 09 handover audit

This document records the Phase 09 fresh-user and reproducibility audit. The
approved environment is `pypsa-earth-ec`; installation evidence is retained in
`results/phase_09_handover/environment/`.

## Documented entry points

From the repository root, use PowerShell and the existing Conda installation:

```powershell
.\scripts\create_environment.ps1 -Update -EvidenceDirectory results/phase_09_handover/environment
& 'C:\Software\Anaconda3\Scripts\conda.exe' run --no-capture-output -n pypsa-earth-ec python -m src.notebook_runner notebooks/00_preflight_and_data_inventory.ipynb --output results/phase_09_handover/notebooks/00_preflight_and_data_inventory.executed.ipynb
```

Run the canonical notebooks sequentially, using a fresh Conda process for
each notebook. Notebook 08 is a full-year scenario screen and must not be run
concurrently with another annual solve. Notebook 08A is a separate experiment
and is excluded from the Phase 09 chain.

## Execution evidence

The following Phase 09 copies completed and are retained:

- notebooks 00 through 07, including the full-year notebook 06;
- notebook 08 for both the tendencial and expansion 2030 cases;
- environment creation and runtime verification evidence;
- solver logs and scenario summaries produced by the full-year runs.

Notebook 08 expansion required HiGHS IPM with crossover disabled to avoid an
additional basis-allocation failure; this execution method is recorded in the
notebook output and log. No report or report-image generation is required for
the handover acceptance.

## Required inputs and generated outputs

The student prerequisite list is
`results/phase_09_handover/required_files.txt`. It contains only external/raw
inputs and the execution contract. Notebook-generated CSVs, intermediate
networks, solved NetCDFs, executed notebooks, runtime verification, reports,
and Phase 08A utilities are deliberately excluded. The chain must create
`results/phase_06/network_solved_2022.nc` in Notebook 06; Notebook 08 consumes
that exact path.

## Student workflow boundary

Students should edit scenario parameters in a notebook or a small scenario
runner, not core network-construction logic. A scenario definition must state
the year, demand source, hydrology or weather perturbation, import limits,
assets added or removed, and output directory. It must preserve the preflight,
provenance, solver-status, and energy-balance checks. Official Master Plan
tables are separate from hypothetical assets; nuclear remains quarantined
legacy material.

## Handover conclusion

The documented Phase 09 notebook chain is complete for the approved runtime:
00, 01, 02, 03, 04, 05, 06, 07, and 08. Notebook 08A remains a separate
experiment and is intentionally excluded. The full-year runs were performed in
isolated Conda processes, with the solver model released before NetCDF export;
the expansion case additionally used IPM without crossover to control memory.

## Archival provenance and generated evidence

The generated files remain available as audit evidence, but are no longer
presented as student prerequisites. They are produced by the notebook chain
and may be retained in the repository's evidence area or in the release
archive.

One important consequence: the previously executed Notebook 08 copies were
produced before the final Phase 06 output-path correction. They should be
rerun once more so the retained execution evidence matches the final chain.

For long-term source traceability, create a Zenodo record through the GitHub
release integration and upload a complete archival bundle containing:

1. repository source at the release commit;
2. raw source tables;
3. permitted source PDFs;
4. `config/artifacts.yaml`;
5. a provenance and checksum manifest;
6. the environment specification and lock/evidence files.

Preserve the government URL, retrieval date, edition, document title,
page/table locator, license status, and SHA-256 for every source. Use the
Zenodo concept DOI in project documentation; each release receives its own
version DOI.

Do not redistribute PDFs whose licensing is unclear. For those sources,
archive the metadata, checksum, and source URL, and let preflight download the
file. This preserves durable provenance if a government website disappears,
without incorrectly treating generated results or audit evidence as
installation inputs.
