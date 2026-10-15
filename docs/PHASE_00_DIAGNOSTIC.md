# Phase 0 diagnostic: pre-refactor repository

Date: 2026-10-15

Branch inspected: `refactor_documentation` at `89f7a23`

## Executive assessment

The repository contains substantial, reusable scientific work, but it is not yet
ready for student handover and the 2022 baseline is not yet proven validated.

There are two overlapping development histories:

- a legacy 2017/2018 notebook chain that contains the most complete consistency,
  repair, optimization, and scenario experiments;
- a newer 2022 chain that prepares demand, generation, hydro profiles, topology,
  and an assembled network, but lacks a documented statistical acceptance test.

The correct refactor is controlled reconstruction, not blind cleanup and not a
restart. Manual PDF parsing, plant matching, network fixes, orphan decisions, and
successful legacy checks must be extracted as evidence and reusable rules before
duplicates are removed.

## Confirmed repository shape

The tracked project contains 24 notebooks, 11 Python files, 46 CSV files, three
JSON files, two YAML files, and only minimal Markdown documentation at the current
head. The active areas are `config/`, `data/`, `literature/`, `notebooks/`,
`results/`, and `src/`.

The current branch is aligned with `origin/scenarios` and
`origin/refactor_documentation`. Historical refactor commits/branches already
contain useful planning text, but some earlier rules conflict with the owner's
current requirements (notably a repository-local environment and a Python-only
production workflow). This Phase 0 documentation supersedes those points by using
the existing Conda environment and retaining notebooks as the student interface.

## Environment diagnostic

`conda` is not on PowerShell's `PATH`, but the existing installation contains the
requested environment and the environment is directly usable. Workstation-specific
installation locations are intentionally omitted from project documentation.

- environment name: `pypsa-earth-ec`
- Python: 3.11.14
- PyPSA: 0.30.3
- pandas: 2.3.3
- NumPy: 1.26.4
- xarray: 2025.1.2
- HiGHS Python bindings: available

No installation was performed. Imports emitted GDAL/ecCodes data-library warnings,
which do not prevent the files inspected here from loading but must be diagnosed
before GIS/weather processing is called reproducible.

The environment is not currently listed as a dedicated Jupyter kernelspec; several
notebooks nevertheless record display name `pypsa-earth-ec` while using kernel name
`python3`. Notebook execution must therefore be invoked explicitly through the
environment until kernel registration is deliberately standardized.

`config/environment_ec.yaml` is not yet a reliable lock/specification: it contains
at least a likely typo (`ploty=6.3.1`) and a dangling empty dependency entry. It
must be reconciled with the known-working environment rather than used to create a
new environment during the early phases.

## Notebook diagnostic

### Legacy chain (`notebooks/_alt/`)

The approximate intended sequence is demand forecast, demand profile scaling,
generation preparation/plots, network evaluation/plots, load/generation matching,
renewable estimation, consistency checks, and scenario building.

High-value evidence includes:

- `08_2017_Consistency_Check.ipynb`: 49 code cells; substantial assembly,
  diagnostics, pruning, renewable attachment, and solve logic. The saved notebook
  contains a final `KeyError`, so “fully functional” must be reconstructed from
  earlier execution state and related notebooks/results rather than inferred from
  its last cell alone.
- `09_2017_Consistency_Check_no_pruning.ipynb`: 43 code cells, 42 executed, and no
  saved error. This is strong evidence for a working comparison path.
- `10_EC_Scenario_Building.ipynb` and its copy: mostly executed with no saved
  errors; they include reusable output/KPI and attachment logic but also obsolete
  scenario assumptions.

The legacy chain has very little explanatory Markdown, inconsistent kernel
metadata, duplicated/copy notebooks, and stale errors. It is valuable forensic
material, not a handover workflow.

### Current 2022 chain (`notebooks/`)

Observed stages are:

1. Python scripts for demand profile generation and projection extension;
2. `03_a_process_generation.ipynb`;
3. `03_b_generation_plots.ipynb`;
4. `03_c_hydro_profiles.ipynb`;
5. `04_EC_network_base.ipynb`;
6. `05_EC_network_base_run.ipynb`;
7. `06_EC_network_expansion_run - Kopie.ipynb`;
8. `06_EC_network_plots.ipynb`.

The newer chain is closer to the desired year and contains significant results,
but most notebooks have zero Markdown cells, depend on generated CSVs inside the
`notebooks/` directory, mix preparation/repair/solve/report concerns, and do not
implement the required statistical 2022 carrier validation. The expansion notebook
has a saved `IndexError`.

This chain must be decomposed into explained notebooks that call shared functions;
it should not be discarded.

## Data and artifact diagnostic

Large and ignored files exist in the audited workspace even though they are absent
from `rg --files` and Git. Confirmed NetCDF inventory includes the upstream
`base.nc`, `solar.nc`, `onwind.nc`, an 18.66 GB ERA5 cutout, multiple
processed/alternate networks, and nine solved result networks. See
`docs/DATA_AND_ARTIFACTS.md` for the artifact classification.

The companion `EcuadorElectricGridData` repository explicitly owns generation of
`base.nc`, `solar.nc`, and `onwind.nc`; this repository owns `ec_network_2022.nc`,
`network_base_filled.nc`, demand/generation/hydro processing, expansion scenarios,
and simulations. That boundary should be retained.

Notebook writer tracing confirms that `ec_network_2022.nc` and
`network_base_filled.nc` are generated processed products, not external source
artifacts. The former is exported by `04_EC_network_base.ipynb` after topology
cleaning and manual fixes. The latter is exported by
`05_EC_network_base_run.ipynb` after additional pruning and attachment of demand,
generation, profiles, costs, carriers, transformer fixes, and load shedding. The
current instances must not be uploaded as authoritative inputs before their
transformation chain is reproduced and the final network passes Phase 6.

The active 2022 notebooks read `solar.nc` and `onwind.nc` directly. Only two
legacy `_alt` notebooks reference the full ERA5/atlite cutout, and one of those
uses it to regenerate the two smaller profile files when they are absent. The
18.66 GB cutout is therefore not required by this repository's supported student
workflow or preflight; it belongs to upstream profile regeneration in the
companion data project.

The current `.gitignore` correctly ignores `.nc`, PDFs, Excel workbooks, and common
images, but it also means essential local evidence can be invisible to a normal
tracked-file inventory. A manifest is mandatory. The repository presently has no
URI/checksum configuration that can restore these files on a fresh clone.

## Network evidence

Read-only loading through the approved environment produced these fingerprints:

| Artifact | Snapshots | Buses | Lines | Transformers | Generators | Loads |
|---|---:|---:|---:|---:|---:|---:|
| `data/raw/networks/base.nc` | 8,760 | 313 | 247 | 62 | 0 | 0 |
| `data/processed/networks/ec_network_2022.nc` | 8,760 | 295 | 252 | 62 | 0 | 0 |
| `data/processed/networks/network_base_filled.nc` | 8,760 | 110 | 111 | 19 | 247 | 208 |
| legacy `_alt/network_base_filled.nc` | 8,760 | 111 | 109 | 19 | 244 | 251 |

The candidate 2022 assembled network has nominal capacity of approximately
5,184.19 MW hydro, 2,061.88 MW oil, 152.62 MW biomass, 66.48 MW onshore wind, and
28.48 MW solar, plus artificial load-shedding generators. These values require
source reconciliation before acceptance.

The very large reduction from the raw topology to the assembled network may be
scientifically justified pruning, but it is currently encoded largely in notebook
state and generated audit CSVs. Phase 5 must make every removal and merge
reproducible and explain the supported voltage/system boundary.

## Existing solve evidence and validation gap

Several saved NetCDF results contain dispatch and demonstrate that variants of the
network can solve. This is important feasibility evidence, but filenames such as
`TEST_7`, `TEST_90`, and `TEST_365` do not constitute reproducible run metadata.

The apparent full-year `network_solved_TEST_365.nc` contains 8,760 snapshots and,
when load-shedding generation is excluded from the denominator, approximately:

| Carrier | Simulated generation share |
|---|---:|
| Hydro | 85.71% |
| Oil | 11.71% |
| Biomass | 2.30% |
| Solar | 0.22% |
| Onshore wind | 0.06% |

It also records about 75.3 GWh of load shedding against 23,934.8 GWh of demand
(about 0.31%). Snapshot weighting and result provenance must be verified before
these numbers are treated as definitive.

No canonical official-2022 carrier table, carrier-crosswalk, 5% formula, automated
pass/fail comparison, or documented calibration history was found. Therefore:

- a network can run, but the exact candidate and command are not canonical;
- successful optimization is not yet equivalent to validation;
- the required “within 5%” result is currently unproven.

## Master Plan and future-expansion diagnostic

The repository contains valuable manually prepared inputs:

- demand chapter PDFs and extracted Excel tables;
- generation chapter, existing/future generation extracts, and hydro flow data;
- transmission chapter;
- `EC_bus_expansion_masterplan_2022.csv`;
- `EC_line_expansion_masterplan_2022.csv`;
- `EC_transformer_expansion_masterplan_2022.csv`;
- existing/future generation tables and a renewable-future CSV.

These assets are a strong starting point, not yet a finished planning database.
They need stable IDs, row-level citations, source-versus-assumption flags, unit
schemas, referential integrity, and deterministic year filtering.

The current line-expansion CSV includes auto-generated nuclear connector rows
explicitly labeled as placeholders and “not from PDF.” Nuclear also appears in the
configuration, generator data, scenario registry, and historical network variants.
Because the owner has deemed those scenarios infeasible/currently unsuitable,
Phase 1 must quarantine them from official Master Plan data and Phase 8 must not
restore them except as a clearly separate, explicitly chosen student hypothesis.

## Source-code diagnostic

Reusable code exists in `src/helpers.py`, `src/ec_network_eval.py`, path utilities,
and scenario-registry modules. There are also `_old` pipeline/registry/template
implementations. Their behavior overlaps notebook code and the scenario registry
precedes a validated baseline.

The correct sequence is to extract tested domain functions from the demonstrated
notebooks, then simplify/rebuild the scenario layer after validation. Refactoring
the current registry first would preserve the wrong priority and risk embedding
obsolete nuclear assumptions.

## Gap analysis against the required outcome

| Required outcome | Current evidence | Gap |
|---|---|---|
| All required data checked before work | Candidate files exist | No manifest, URI, checksum, dependency closure, or preflight |
| Use existing environment | Environment is present and usable | Conda shell initialization, kernel identity, and YAML need reconciliation |
| Explained sequential notebooks | Two partial chains exist | Sparse Markdown, duplication, hidden state, inconsistent paths |
| Runnable 2022 model | Candidate annual results exist | Canonical command/artifact/solver metadata not fixed |
| Carrier mix within 5% | Candidate mix can be calculated | Official table, taxonomy, formula, and pass/fail test absent |
| Reproducible network fixes | Manual work and audit CSVs exist | Rules/provenance largely trapped in notebooks/generated files |
| Ready-to-aggregate future assets | Several parsed CSV/XLSX inputs exist | Schemas, citations, IDs, integrity, and fact/assumption split incomplete |
| Student scenario framework | Registry and old pipelines exist | Premature/obsolete assumptions; not anchored to validated baseline |
| Phase reports with plots | Many notebook plots/results exist locally | No standardized report/figure provenance workflow |
| Clean repository | Useful content is present | Copies, `_alt`/`_old`, mixed outputs, minimal docs, large log |

## Principal risks

1. **False validation:** treating a solved network or visually plausible mix as a
   validated 2022 model.
2. **Evidence loss:** deleting legacy notebooks/CSVs before extracting the only
   record of manual corrections.
3. **Boundary mismatch:** comparing mainland-SNI simulation with national
   statistics that include isolated systems or use different net/gross accounting.
4. **Carrier mismatch:** comparing “oil” against a statistical taxonomy that splits
   fuel oil, diesel, gas, and other thermal generation.
5. **Hidden infeasibility:** accepting load-shedding generators as normal supply.
6. **Data interruption:** discovering a missing ignored file midway through a
   notebook instead of during preflight.
7. **Scenario contamination:** retaining hypothetical nuclear assets in tables
   presented as official Master Plan expansion.
8. **Environment drift:** inadvertently executing with system Python 3.13 or
   changing the working environment while trying to reproduce it.
9. **Non-reproducible pruning:** changing topology while “cleaning” without a
   before/after invariant and source-supported rationale.

## Recommended target notebook series

The final names may be refined in Phase 1, but the intended teaching flow is:

1. `00_preflight_and_data_inventory.ipynb`
2. `01_prepare_2022_demand.ipynb`
3. `02_prepare_2022_generation.ipynb`
4. `03_prepare_renewable_and_hydro_profiles.ipynb`
5. `04_build_and_validate_2022_topology.ipynb`
6. `05_assemble_and_smoke_test_2022_network.ipynb`
7. `06_run_and_validate_2022_full_year.ipynb`
8. `07_prepare_master_plan_expansions.ipynb`
9. `08_build_future_scenario.ipynb`
10. `09_compare_scenarios.ipynb`

Notebooks 08–09 are templates/frameworks, not completed student research.

## Phase 0 decisions requiring owner confirmation

The documentation currently recommends, but does not silently lock, these points:

- use the current `refactor_documentation` branch as the Phase 0 branch or rename it
  to the plan's canonical branch name;
- compare carrier values using a clearly defined metric (relative percentage error
  is the likely interpretation, but percentage-point error may be preferable for
  mix shares and small carriers need special handling);
- choose the authoritative 2022 statistical publication and confirm whether the
  model boundary is mainland SNI or all Ecuador;
- decide which candidate assembled network is the starting evidence for Phase 5,
  without prematurely declaring it canonical.

These decisions affect scientific acceptance and artifact size, so they should be
resolved before their respective implementation phases rather than guessed now.

## Phase 0 conclusion

The project is recoverable and already holds much of the hard scientific work.
The shortest trustworthy path is: preserve evidence, establish preflight, freeze
the statistical contract, normalize baseline inputs, reproduce network repairs,
validate the annual 2022 run, and only then formalize Master Plan and student
scenarios. The phase sequence in `docs/PROJECT_PLAN.md` implements that order.
