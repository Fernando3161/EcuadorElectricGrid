# README_v1.md

## Refactor specification for EcuadorElectricGrid v1

This document describes the work to be performed during the v1 refactor of the `EcuadorElectricGrid` repository.

It is not the final repository README.

The final root `README.md` is itself a required refactor deliverable and must describe the implemented repository after the refactor is complete.

## 1. Why the refactor is required

The current repository contains valuable scientific work and a functioning 2022 PyPSA-based Ecuador model, but it evolved as a research workspace rather than as a clean reusable software artifact.

The present state includes notebook-centered execution, duplicated and historical files, legacy scenario code, large tracked datasets, canonical `.nc` artifacts that are used by workflows but are not reproducibly distributed through the repository, incomplete root documentation, partial scenario infrastructure, assumptions distributed across notebooks/CSVs/old modules, and environment definitions inherited partly from broader PyPSA-Earth workflows.

The refactor must preserve the validated scientific base while removing the need for a new student to reconstruct the development history.

## 2. v1 goal

The target is a ready-to-hand-over open-source research artifact.

A student should receive:

- a validated 2022 Ecuador SNI starting network;
- automatic retrieval of required large public artifacts;
- a repository-local reproducible Python environment;
- a clean supported baseline workflow;
- normalized official-planning inputs;
- a scenario registry and scenario-building framework;
- standardized execution/result handling;
- transparent provenance for assumptions;
- complete instructions for installing, running, understanding, and extending the project.

The student should begin with scientific scenario development, not repository archaeology.

## 3. Scientific project purpose

The project supports an open-source simulation study of possible future developments of the Ecuadorian electric power system.

It starts from the 2022 status of the SNI and uses official electricity Master Plan information to represent future electricity demand, renewable generation, hydropower, thermal/firm generation, retirement/replacement of older assets, and transmission infrastructure.

The project additionally supports hypothetical SMR nuclear sensitivity scenarios.

The model is intended for comparative scenario analysis rather than endogenous generation/transmission investment optimization.

## 4. Supported v1 scientific scope

### Base system

- Ecuadorian SNI only.
- Base year 2022.
- 138/230/500 kV.
- 69 kV excluded from the supported macro-model.
- Galapagos/isolated systems excluded.
- Colombia/Peru interconnection points may be represented.

### Demand

The complete 2022 demand profile is the reference shape.

Future demand is constructed using one annual factor:

`future profile = complete 2022 profile × annual factor`

The method used to determine the annual factor may differ by scenario. It may come from official Master Plan values, linear growth, compound annual growth, or another documented annual-growth assumption.

The spatial/hourly shape is not independently changed in the base v1 methodology.

Canonical demand families:

- D1 Tendential;
- D2 PME Case Base / central planning case;
- D3 high-growth stress.

### Master Plan delays

Canonical delay cases:

- +5 years;
- +8 years;
- +12 years.

Generation and required transmission are normally delayed together.

### Generation

Supported additions include hydro, solar PV, wind, biomass, geothermal, thermal, firm-generation blocks, and replacement/repowering.

Thermal retirement should use documented age/retirement information where available.

### Hydrology

Supported cases are normal, dry, severe dry.

Hydro is represented using availability profiles in v1.

### SMR nuclear sensitivity

Nuclear is additive by default.

Canonical additions:

- 2035: +0.5 GW;
- 2040: +0.5 GW;
- 2045: +0.5 GW;
- 2050: +0.5 GW.

SMR locations should favor major transmission nodes.

Required network connections must be explicitly modeled and justified.

### Anchor years

2022, 2030, 2035, 2040, 2050.

## 5. What the refactor must change

### 5.1 Remove notebook dependence

Production workflows must move to modular Python code.

Notebooks may remain only for examples, visualization, and exploratory analysis.

They may not be required for loading/building the supported baseline, constructing scenarios, solving scenarios, or producing standard summaries.

### 5.2 Clean the repository

The active final repository should contain no visible development debris such as copy notebooks, active `_old`/`_alt` dependencies, obsolete 2017/2018 workflows, duplicated generated data, or temporary migration artifacts.

Historical provenance remains available through Git history and branches.

### 5.3 Establish a supported local Python environment

The project must stop relying on external Python environments.

The supported environment is repository-local `.venv`, Git-ignored, created during first setup if missing, and populated using `requirements.txt`.

Every later dependency addition must be added both to `.venv` and `requirements.txt`.

### 5.4 Externalize large canonical artifacts

The validated 2022 PyPSA network and other appropriate large canonical inputs will be hosted in a permanent public archive such as Zenodo.

On first execution, required files are checked, missing files are downloaded, integrity is verified, and files are placed correctly.

On later executions, valid existing files are reused.

Students are not expected to regenerate the canonical `.nc` network from PyPSA-Earth.

### 5.5 Formalize the baseline

The v1 project must provide a clean supported path to obtain, load, validate, run, and summarize the validated 2022 system.

The historical process used to produce that artifact is provenance, not the student's normal runtime workflow.

### 5.6 Normalize Master Plan inputs

Demand, generation, and transmission expansion information must be represented in clean authoritative tables/configuration.

Manual mappings must be explicit.

Where an official project has no precise model location, the repository may contain a documented modeling assumption, but it must be identifiable as an assumption rather than official source data.

### 5.7 Build a real scenario framework

The current scenario-registry concept must become a usable framework.

A scenario should be definable without editing core source code.

It must be possible to select or define anchor year, demand case/factor, delay case, generation additions, thermal retirement/repowering, hydrology, transmission, interconnection behavior, SMR additions, and run settings.

### 5.8 Standardize results

Every scenario must produce an isolated reproducible result package.

The project should prepare data for later dashboard use without building simulation into the dashboard.

Preferred public formats are JSON for metadata/KPIs, Parquet for time series, and NetCDF for solved networks where useful.

Minimum result coverage includes demand, peak demand, generation, curtailment, hydro behavior, load shedding, imports/exports, line/transformer loading, installed capacity, total system cost, system cost per kWh, solver status, and runtime.

### 5.9 Add a master scenario runner

The supported workflow must include running one selected scenario and running the registered scenario set without requiring users to edit model internals.

### 5.10 Add small tests

Use unit tests by default, small deterministic fixtures, tiny integration tests only where needed, and external-data smoke tests where appropriate.

Do not use annual Ecuador optimization, full scenario batches, or repeated large-data downloads in the normal automated suite.

## 6. Documentation to produce during the refactor

The final repository must contain clear current documentation for installation, first-run data retrieval, project structure, scientific methodology, data sources, provenance/assumptions, baseline execution, scenario definition/execution, results, student extension, testing, licensing, and citation.

The root `README.md` is rewritten near the end of the refactor so it describes the actual implemented v1 repository.

This `README_v1.md` remains a record of what the refactor set out to do.

## 7. Required development discipline

Before each phase, the repository owner creates or selects a new branch.

During the phase, the agent works only within the agreed scope and the owner retains control of commits/pushes.

After the phase, minimal tests are run, `REPORT_PHASE_XX.md` is written, and the owner reviews the diff before committing/pushing.

No phase should silently change scientific assumptions merely to simplify implementation.

## 8. Licensing and openness

The project is intended to remain public and open source.

Original project code should use a permissive license, preferably MIT.

The repository must separately acknowledge and respect third-party terms.

Attribution must include applicable use of Ecuadorian electricity Master Plans, PyPSA, PyPSA-Earth, meteorological datasets, hydrological datasets, and other externally derived scientific data.

The final repository should include licensing, citation, and third-party notice files.

## 9. Student handover boundary

Students should not be expected to reconstruct the validated 2022 `.nc`, recreate PyPSA-Earth extraction from scratch, reproduce obsolete pre-2022 studies, understand historical notebook chains, or redesign the architecture before research.

Students should be able to focus on renewable expansion, hydro/drought resilience, thermal retirement/firm-capacity alternatives, SMR sensitivity, and transmission adequacy.

## 10. Definition of successful v1 refactoring

The refactor is successful when the repository has changed from a researcher-specific workspace into a stable scientific platform.

A new student should be able to obtain the repository, create the local environment, retrieve canonical data, understand the validated 2022 base, and begin defining future scenarios without reconstructing the previous development process.

The scientific value of v1 is the combination of validated baseline, explicit assumptions, traceable source data, modular scenario machinery, reproducible execution, standardized results, maintainable code, and teaching-quality documentation.
