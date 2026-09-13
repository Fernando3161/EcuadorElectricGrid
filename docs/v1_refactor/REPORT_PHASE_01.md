# Phase 1 report: repository cleanup and target architecture

## Phase objective and governance

Phase 1 removed active dependence on legacy research debris and established a
production-oriented repository structure without intentionally changing the
validated scientific meaning of the project.

Detected branch: `v1_refactor_base`.

The branch name differs from the recommended example
`refactor/v1-phase-01-cleanup`, but the repository owner stated that the phase
branch had already been created. No branch, commit, remote, or history operation
was performed. This report ends Phase 1; no Phase 2 work was started.

## Original state relevant to Phase 1

The initial working tree was clean. It contained 24 tracked notebooks, four
legacy scenario modules under `src/_old`, a broken `src/helpers.py`, a minimal
placeholder JSON scenario registry, generated CSV/JSON outputs mixed into the
notebook directory, `_alt` trees under notebooks and data, and a copied
expansion notebook named `Kopie`.

The repository held approximately 166.9 MiB of CSV files and 18.2 MiB of
notebooks. The notebook directory alone contained a 38.7 MiB generated demand
table (`loads_p_set.csv`) plus generation, topology, load-shedding, and
expansion diagnostics. No `.nc` file was present, as expected.

The active Python layout was a flat `src/` directory. `src/helpers.py` imported
missing `scenarios`/`paths` modules and a nonexistent `PROC_LOAD_DIR`, so it was
not a valid production interface. Historical notebooks and the old scenario
pipeline added a neighboring `pypsa-earth/scripts` directory to `sys.path`.

`config/config.yaml` and `config/environment_ec.yaml` were broad legacy
PyPSA-Earth/Conda configurations. There was no `tests/` directory. The root
README was minimal. These observations agree with `REPORT_v0.md`.

## Architecture established

The directory contract is documented in
`docs/architecture/REPOSITORY_ARCHITECTURE.md`:

- `src/ecuador_grid/`: importable production package;
- `scripts/`: explicit entry points and preprocessing;
- `config/`: runtime configuration and, later, scenario definitions;
- `data/raw/`: small authoritative source inputs;
- `data/processed/`: derived/model-ready inputs and provenance reports;
- `data/external/`: ignored downloaded canonical artifacts;
- `data/generated/`: ignored reproducible intermediates;
- `results/`: ignored generated results, with one documented historical
  experiment exception;
- `tests/`: small deterministic tests;
- `notebooks/`: optional examples/analysis only;
- `docs/`: architecture, provenance, guides, and phase reports.

Production code has no import from a notebook, `_old`, `_alt`, copied file,
results directory, neighboring repository, or machine-specific path.

## Files and directories added

- `config/README.md`
- `data/README.md`
- `data/external/README.md`
- `data/generated/README.md`
- `docs/architecture/REPOSITORY_ARCHITECTURE.md`
- `docs/v1_refactor/LEGACY_MIGRATION_INVENTORY.md`
- `docs/v1_refactor/REPORT_PHASE_01.md`
- `notebooks/README.md`
- `results/README.md`
- `scripts/README.md`
- `src/ecuador_grid/__init__.py`
- `tests/test_repository_structure.py`

New directory boundaries are `data/external/`, `data/generated/`,
`docs/architecture/`, `scripts/preprocessing/`, `src/ecuador_grid/`, and
`tests/`.

## Files moved

- `notebooks/01_generate_demand_profiles_2022.py` to
  `scripts/preprocessing/generate_demand_profiles_2022.py`;
- `notebooks/02_extend_demand_prognosis.py` to
  `scripts/preprocessing/extend_demand_prognosis.py`;
- `src/paths.py` to `src/ecuador_grid/project_paths.py`;
- `notebooks/EC_line_expansion_masterplan_nreport.csv` to
  `data/processed/networks/expansion_line_application_report.csv`;
- `notebooks/skipped_expansion_lines.csv` to
  `data/processed/networks/skipped_expansion_lines.csv`.

The demand scripts changed only their repository-root calculation to account
for the new location. The path module gained a stable `PROJECT_ROOT`, external
and generated data constants, and `.venv` traversal exclusion.

## Files and directories removed

The complete `src/_old/` directory was removed:

- `scenario_pipeline.py`
- `scenario_reg.py`
- `scenario_run_template.py`
- `scenarios.py`

Broken or misleading flat-source files removed:

- `src/helpers.py`
- `src/scenario_registry.py`
- `src/scenario_registry.json`
- `src/ec_network_eval.py`

The complete `notebooks/_alt/` directory was removed:

- `01_EC_year_demand_forecast.ipynb`
- `01_a_EC_year_demand_forecast.ipynb`
- `02_EC_profiles_forecast.ipynb`
- `03_EC_generation.ipynb`
- `04_EC_generation_plots.ipynb`
- `05_EC_network_eval.ipynb`
- `06_EC_network_plots.ipynb`
- `07_EC_load_n_gen_match.ipynb`
- `07_EC_renewables_estimation.ipynb`
- `07_EC_saved.ipynb`
- `08_2017_Consistency_Check.ipynb`
- `08_EC_cluster_network copy.ipynb`
- `09_2017_Consistency_Check_no_pruning.ipynb`
- `10_EC_Scenario_Building copy.ipynb`
- `10_EC_Scenario_Building.ipynb`
- `999_Vietnam.ipynb`
- `temp_demand.py`
- `test.ipynb`

Former top-level notebooks removed after migration review:

- `03_a_process_generation.ipynb`
- `03_b_generation_plots.ipynb`
- `03_c_hydro_profiles.ipynb`
- `04_EC_network_base.ipynb`
- `05_EC_network_base_run.ipynb`
- `06_EC_network_expansion_run - Kopie.ipynb`
- `06_EC_network_plots.ipynb`

Notebook-local generated files removed:

- `buses_final_2022.csv`
- `cantons_not_found.csv`
- `generators_postopt.csv`
- `generators_t_pmax_pu.csv`
- `lines_data.csv`
- `lines_final_2022.csv`
- `load_shedding_buses.csv`
- `loads_p_set.csv`
- `lv_only_buses.csv`
- `lv_only_buses_by_subnetwork.json`
- `network_lines.csv`
- `ppl_attached.csv`
- `removed_orphan_buses.csv`
- `skipped_expansion_trafos.csv`
- `transformers_final_2022.csv`

Other legacy/duplicate files removed:

- `data/processed/networks/_alt/line_fix.csv` and its `_alt` directory;
- `data/raw/demand/_alt/demand_ec_2018_2027.csv` and its `_alt` directory;
- `data/raw/generation/_alt/carrier_name_mapping.json` and its `_alt` directory;
- `data/processed/generation/nuclear_layout.csv`;
- empty duplicate reports
  `data/processed/generation/hidro_max_profiles/low_confidence_matches.csv` and
  `unmatched_plants.csv`;
- `literature/scenario_deve_documentation.md` and the now-empty `literature/`
  directory.

All removed tracked content remains recoverable from Git history. No `.nc` or
other external artifact was deleted because none existed locally.

## Files materially modified

- `.gitignore`: explicitly ignores downloaded external data, generated
  intermediates, and results while retaining category READMEs and the existing
  historical transformer experiment.
- `src/ecuador_grid/project_paths.py`: package-relative root resolution,
  external/generated constants, and `.venv` exclusion.
- the two moved demand scripts: corrected root resolution after their move.

The Phase 0 specification files, root README, scientific CSV values, legacy
environment files, and experiment results were not rewritten.

## Legacy material removed from the active workflow

All pre-2022 scenario workflows, copied scenario-building notebooks, the
Vietnam experiment, broken helper/pipeline modules, and notebook-only build/run
chains were removed. `notebooks/` now contains only a policy README. There is no
active scenario engine in Phase 1; this is deliberate because the removed
placeholder did not satisfy the required schema and Phase 4 owns its
replacement.

The old network evaluator was not silently adapted: its default checks included
48/69 kV construction-era bridges inconsistent with the supported v1 runtime
scope. Applicable checks are documented for evidence-based reimplementation in
Phase 3.

## Scientific and provenance information preserved or migrated

`LEGACY_MIGRATION_INVENTORY.md` records the complete later-phase migration
boundary: demand scaling, generator/load mappings, carrier normalization,
hydro/renewable profile assignment, load shedding, transformer/line impedance
defaults, topology cleanup, KPI logic, and conflicting historical scenario
assumptions.

Specific retained data includes:

- the five 138 kV manual topology fixes in `line_fix.csv`, including their
  endpoints and construction comments;
- hydro alias, anchor, and match reports;
- the Master Plan line application report with source text and modeling notes;
- all eight skipped expansion-line records;
- the unresolved Delsitanisagua-Cumbaratza row whose two mapped endpoints are
  both bus 97;
- historical transformer sensitivity results in
  `results/experiments/experiment_x_defaults_results.csv`.

Conflicting old assumptions were documented but not migrated into active data:
5/7/9-year delay cases, 0.9/2.1/3.0 GW nuclear pathways, replacement-mode
nuclear, and the separate 300 MW-unit nuclear layout. AGENTS.md remains
authoritative: 5/8/12-year delays and additive 0.5 GW SMR steps.

No bus mapping, capacity, demand value, voltage-scope decision, hydro method,
Master Plan timing, or nuclear assumption was newly invented or scientifically
changed.

## Data changes and large-artifact review

The 38.7 MiB notebook-local `loads_p_set.csv` was removed as generated output;
the authoritative processed demand profile remains. Other notebook-generated
network/generator exports were removed after their unique migration signals
were documented.

Five tracked files larger than 5 MiB remain intentionally for Phase 2 review:

- `data/raw/demand/demand_profiles_EC.csv` (38.80 MiB);
- `data/processed/demand/demand_profiles_2022.csv` (38.75 MiB);
- `data/processed/generation/renewable_energy_generation.csv` (24.93 MiB);
- `pmax_pu_hydro_2022_hourly.csv` (10.58 MiB);
- `pmax_mw_hydro_2022_hourly.csv` (10.27 MiB).

They are scientifically relevant and not exact duplicates, so Phase 1 did not
delete or externalize them without the Phase 2 manifest/archive decision.

## Configuration and dependency effects

No dependency was added, removed, installed, or newly imported. No `.venv` was
created. `requirements.txt` was not introduced because the authoritative pip
environment is Phase 2 work.

`config/environment_ec.yaml` remains as a documented legacy file. Its known
issues include a suspicious `ploty=6.3.1` entry and a blank dependency item.
`config/config.yaml` still contains broad PyPSA-Earth extraction settings.
Removing settings or packages safely requires knowing the supported Phase 2/3
runtime, so both were intentionally deferred.

## Tests performed

Command: `python -B -m unittest discover -s tests -v`

Result: 7 tests passed in 0.008 seconds. Tests verify the directory contract,
absence of active legacy directories, absence of notebook CSV outputs,
repository-root path resolution, exclusion of `.git`/`.venv` traversal, the
five-row line-fix schema, and preservation of the unresolved same-bus expansion
mapping.

Earlier in the final audit, `python -m compileall -q src scripts` also completed
successfully. Generated bytecode caches were removed afterward.

Read-only audits found no prohibited legacy/copy/backup names in the resulting
active tree and no production import from legacy folders or notebooks.

## Tests intentionally not performed

- no annual or short PyPSA optimization;
- no canonical network loading or topology validation;
- no `.nc`-dependent test;
- no scenario execution or batch run;
- no external download or large-data generation;
- no clean environment installation/dependency resolution.

These tests either require the intentionally absent canonical network or belong
to Phase 2 and later.

## Known missing `.nc` references

Historical workflows referred to `base.nc`, `ec_network_2022.nc`,
`network_base_filled.nc`, and solved network variants. The retained migration
inventory documents these names. The target canonical location is reserved
under `data/external/networks/ec_network_2022.nc`, but the directory/file will
be populated only by the Phase 2 manifest/bootstrap after a permanent archive
record and checksum exist.

No attempt was made to reconstruct, regenerate, download, or invent a network.

## Deferred issues

Phase 2:

- establish repository-local `.venv` and authoritative `requirements.txt`;
- review/remove the legacy Conda definition after dependency validation;
- decide which five remaining large tracked tables belong in the archive;
- implement the artifact manifest, checksums, and first-run acquisition.

Phase 3:

- implement supported 2022 loading/validation/execution from the canonical
  artifact;
- verify all historical load/generator mappings and impedance assumptions;
- designate authoritative baseline validation evidence;
- migrate applicable network checks for 138/230/500 kV only.

Phase 4:

- normalize source facts versus model assumptions in generation/transmission
  tables;
- resolve or formally flag ambiguous project mappings;
- establish D1/D2/D3 growth behavior and the 5/8/12-year delay model;
- implement the authoritative scenario registry and additive SMR pathway.

Later phases own standardized outputs, final documentation, licensing/citation
completion, and acceptance testing.

## Deviations and assumptions

There was no scientific deviation from `PROJECT_PLAN.md`. The detected branch
name differs from the recommendation but was accepted as owner-selected.

The cleanup assumed that obsolete implementations can be removed after unique
knowledge is migrated because AGENTS.md explicitly identifies Git history as
the historical archive. It also assumed that large scientifically relevant
tables should remain until Phase 2 can externalize them reproducibly.

The only behavioral edit was repository-root resolution in moved demand
scripts; their calculations were not altered.

## Recommended manual review before commit

1. Confirm that `v1_refactor_base` is the intended Phase 1 branch.
2. Review `LEGACY_MIGRATION_INVENTORY.md`, especially the five topology fixes,
   transformer/line impedance defaults, hydrology matching order, and the
   unresolved bus-97 expansion mapping.
3. Confirm the removal of all notebooks and old scenario implementations is
   acceptable given their preservation in Git history.
4. Inspect the moved Master Plan application/skipped reports and demand scripts
   for rename detection and unchanged contents.
5. Confirm the ignored `data/external`, `data/generated`, and `results` policies
   match the intended Phase 2/5 workflow.
6. Review the five remaining large tracked files as Phase 2 archive candidates;
   do not commit newly generated large artifacts.
7. Run `python -B -m unittest discover -s tests -v` once more after staging if
   desired, then review `git diff` and `git status` before the owner’s manual
   commit and push.
