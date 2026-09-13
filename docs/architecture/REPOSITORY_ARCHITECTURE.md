# Repository architecture

## Status

This is the Phase 1 production-oriented directory contract. It separates
source facts, derived data, runtime code, optional analysis, and results. Later
phases will fill the interfaces without changing this separation.

| Location | Responsibility | Tracked policy |
|---|---|---|
| `src/ecuador_grid/` | Importable production Python code | Tracked |
| `scripts/` | Explicit entry points and reproducible preprocessing | Tracked |
| `config/` | Runtime configuration; scenario definitions later | Tracked |
| `data/raw/` | Small authoritative source tables | Tracked when licensing/size permit |
| `data/processed/` | Model-ready derived tables and provenance reports | Tracked when small; large candidates move to the archive |
| `data/external/` | Downloaded canonical artifacts | Contents ignored; Phase 2 managed |
| `data/generated/` | Reproducible intermediates | Contents ignored |
| `results/` | Isolated simulation and analysis outputs | Contents ignored except documented provenance exceptions |
| `tests/` | Small deterministic software tests | Tracked |
| `notebooks/` | Optional examples and result visualization | Tracked selectively; never production |
| `docs/` | Architecture, methodology, provenance, guides, and reports | Tracked |

## Dependency direction

Production scripts may import `ecuador_grid`; production modules must not
import notebooks, results, generated data, legacy paths, or a neighboring
PyPSA-Earth checkout. Notebooks may import production modules and read saved
results. Results must never be used as hand-edited model inputs.

The normal v1 boundary will begin with the externally archived, validated 2022
network. Rebuilding that network from PyPSA-Earth is provenance/maintenance
work, not a student runtime step.

## Current Phase 1 boundary

`project_paths.py` is the only retained production module. The historical
network evaluator depended on the construction-era 48/69 kV checks and was
removed for deliberate replacement in Phase 3. The demand preprocessing
scripts are explicit scripts rather than notebook companions. No baseline
runner or scenario engine is claimed yet.

The broad legacy configuration files remain visible because simplifying them
correctly depends on the Phase 2 environment decision and later runtime work.
Their status is documented in `config/README.md`.
