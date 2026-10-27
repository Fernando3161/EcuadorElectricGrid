# EcuadorElectricGrid

PyPSA teaching/research model of Ecuador's interconnected electricity system for reference year 2022. It prepares demand, existing generation, renewable and hydro availability, transmission topology, an unsolved network, an annual solve, planning screening, and a separate 2026 hydro-stress experiment.

The 2022 baseline is not scientifically validated: current evidence remains FAIL/PARTIAL for the carrier comparison and energy-accounting acceptance gate. The 2026 experiment is not baseline validation.

## Environment

Use the existing Conda environment; do not create a repository-local environment.

```powershell
conda activate pypsa-earth-ec
python -m src.preflight --stage 00_preflight
```

For a fresh environment, run `scripts/create_environment.ps1` with the existing Conda installation. The manifest `config/artifacts.yaml` retrieves and verifies missing external inputs; `config/environment_ec.yaml` is the auditable environment specification.

## Supported sequence

Run these notebooks from the repository root, in this order, using `python -m src.notebook_runner` and a fresh process for each:

1. `notebooks/00_preflight_and_data_inventory.ipynb`
2. `notebooks/01_prepare_2022_demand.ipynb`
3. `notebooks/02_prepare_2022_generation.ipynb`
4. `notebooks/03_prepare_renewable_and_hydro_profiles.ipynb`
5. `notebooks/04_build_and_validate_2022_topology.ipynb`
6. `notebooks/05_assemble_and_smoke_test_2022_network.ipynb`
7. `notebooks/06_run_and_validate_2022_full_year.ipynb`
8. `notebooks/06a_diagnose_2022_validation.ipynb`
9. `notebooks/07_prepare_master_plan_expansion.ipynb`
10. `notebooks/08_future_masterplan_screening.ipynb`
11. `notebooks/08a_hydro_shortage_2026.ipynb`

Stages create intermediate products under `data/processed/` and `data/interim/`; later notebooks consume those products. Solved and analysis outputs are created locally under `results/` and are intentionally not committed.

## Inputs and interpretation

Required raw inputs are listed in `config/artifacts.yaml` and include the pinned PyPSA-Earth network/weather files, Ecuador demand and generation tables, planning tables, and cited CENACE/Master Plan source documents. Inputs are checked by preflight before scientific processing. The model boundary, carrier taxonomy, units, assumptions, and limitations are encoded in `config/validation_2022.yaml` and the notebook cells.

Interpret results as reproducible model outputs, not official historical measurements or operational forecasts. The baseline must satisfy the documented carrier and energy-balance criteria before being called validated. Cite this repository together with the CENACE and Master Plan sources named in the manifest. See `LICENSE` for reuse terms.
