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

The complete file inventory is
`results/phase_09_handover/required_files.txt`. It is generated from the
documented notebook inputs, the artifact manifest, configuration, reusable
source modules, and the environment evidence. Large solved NetCDF files remain
runtime artifacts and are not copied into this inventory or committed by
default.

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
