# REPORT_v0.md

## Technical pre-refactor assessment of EcuadorElectricGrid

**Document purpose:** establish the technical and scientific baseline before the v1 repository refactor.

**Status:** pre-refactor technical report.

**Scope:** repository state, current workflows, reproducibility limitations, scientific assumptions, technical debt, and v1 refactor objectives.

This report is intended to remain as provenance. It should not be rewritten into a post-refactor description. Later phase reports should document how the repository changes relative to this baseline.

# 1. Executive summary

`EcuadorElectricGrid` currently contains a functioning research prototype for a PyPSA-based representation of the Ecuadorian electricity system, together with demand preparation, generation data, network-expansion data, preliminary scenario infrastructure, and evidence of successful optimization experiments.

The project has already reached an important scientific milestone: the 2022 base system has been assembled and used as the starting point for scenario work.

However, the repository is not yet suitable for independent student handover.

The principal problem is not that the project lacks scientific content. The problem is that scientific content is distributed across notebooks, generated files, legacy modules, duplicated data, partially normalized CSVs, and local/external artifacts whose acquisition is not reproducible from a fresh clone.

The v1 refactor should not restart the model. It should preserve the validated work, isolate the canonical 2022 artifact, formalize data provenance, convert production logic into modular Python, establish a supported scenario registry, provide standardized execution/results, and remove the requirement for future students to understand the original development history.

The target state is a public, permissively licensed research artifact that a student can install, populate with externally archived canonical data, run from the validated 2022 system, and extend with future scenarios without rebuilding the network from scratch.

# 2. Project objective

The scientific objective is to evaluate future adequacy and operation of the Ecuadorian electric power system under predefined development scenarios.

The study considers renewable generation, expanded thermal/firm generation, hydropower, transmission expansion, retirement/replacement of older generation, hydrological stress, international interconnection, and hypothetical SMR nuclear additions.

Future assets are primarily exogenous scenario assumptions based on the Ecuadorian electricity Master Plan and explicitly documented extensions.

The project is not intended to be an endogenous generation/transmission capacity-expansion optimizer.

# 3. Intended scientific contribution

The project is intended to become an open-source, reproducible PyPSA artifact for country-scale evaluation of Ecuador.

Its scientific value should emphasize reproducibility, transparent methodology, explicit assumptions, source traceability, reusable scenario definitions, and accessible teaching/research workflows.

Any eventual publication claim that no equivalent open artifact exists for Ecuador or Latin America should be verified separately through a formal literature review. The refactor should not turn that current working belief into an unverified scientific claim.

# 4. Repository state reviewed before refactor

The evaluated `scenarios` branch was a functioning development branch rather than a clean release branch.

At the time of evaluation, the repository root included the major areas `config`, `data`, `literature`, `notebooks`, `results`, and `src`.

The root README was minimal and did not provide enough information for installation, execution, scientific scope, data acquisition, or scenario development.

The branch contained both current and historical approaches.

The overall assessment was:

**Functioning research prototype, not yet reproducible handover project.**

# 5. Major repository-level issues

## 5.1 Notebook-centered production workflow

Important current behavior is embedded in notebooks.

The existing notebook sequence contains meaningful model-building and validation logic, but notebook dependence makes it difficult to provide deterministic execution, testable functions, stable interfaces, maintainable scenario composition, and student-friendly extension.

The agreed v1 policy is that notebooks may remain for example evaluations, visualization, and exploration only. They must not be required for production model loading/building, scenario construction, production runs, or standard result export.

## 5.2 Canonical network artifacts are not reproducibly distributed

Current workflows reference `.nc` network artifacts including a base network used during skeleton/cleanup work and a processed `ec_network_2022.nc` used for the runnable 2022 model.

The repository tree evaluated during the assessment did not contain the required `.nc` artifacts as tracked files, and the Git ignore rules exclude `.nc` files.

A fresh clone therefore cannot currently obtain the validated baseline using only repository contents.

This is the highest-priority reproducibility issue.

### v1 decision

The validated 2022 PyPSA network and other appropriate large canonical files will be published in a permanent public archive such as Zenodo.

The repository will contain a manifest with artifact locations and integrity information.

On first supported execution, missing files are downloaded, checksums are verified, and files are placed in their expected locations.

Later executions reuse verified local copies.

Students will not be required to rebuild the canonical `.nc` network from PyPSA-Earth.

## 5.3 Large and duplicated data

The repository includes large time-series data and historical duplicates.

Examples observed during evaluation included large demand, renewable, and hydro profile files and duplicate load-profile material under more than one path.

This causes repository bloat, ambiguity over which copy is authoritative, unnecessary Git-history growth, and risk of stale/inconsistent copies.

### v1 decision

Small authoritative tables and metadata remain in Git where appropriate.

Large canonical artifacts move to the external public archive where appropriate.

Large generated files should be reproducibly generated or downloaded.

Duplicate files should be removed from the active tree.

Each external artifact should have version/checksum metadata.

Generated outputs must not be confused with source data.

# 6. Python environment status

The repository currently contains environment/configuration material inherited partly from PyPSA-Earth-style workflows.

The reviewed environment contains the expected scientific stack, but it also requires cleanup and validation. During evaluation, at least one suspicious dependency spelling was observed and the environment definition appeared to include legacy/unused entries.

The exact final package set must be validated during the refactor rather than copied blindly.

### v1 decision

The supported environment becomes a repository-local `.venv`.

Rules:

- `.venv` is Git-ignored;
- it is created locally during first setup if missing;
- all supported runtime/test dependencies are installed into it;
- `requirements.txt` is authoritative;
- no external Conda/global/developer environment is part of the supported workflow;
- dependencies added later must be added to `requirements.txt`;
- obsolete dependencies should be removed with obsolete code paths.

This is mandatory.

# 7. Current 2022 network workflow

The current baseline model is conceptually split between two important notebooks.

## 7.1 Network skeleton / cleanup stage

`04_EC_network_base.ipynb` creates/processes the electrical network skeleton.

Its role includes buses, lines, transformers, voltage normalization, topology checks, and network sanity/cleanup.

The evaluated workflow included mapping voltage levels toward Ecuadorian standard levels and network-quality checks.

The notebook produced evidence of topology issues that required cleaning, including orphan/disconnected areas and transformer consistency checks.

This work is historically valuable, but should not be required for normal student use.

## 7.2 Runnable 2022 baseline stage

`05_EC_network_base_run.ipynb` takes the processed network, attaches relevant time-dependent model data, and runs optimization to verify that the 2022 model is usable.

The evaluated workflow loads the processed 2022 network, uses 2022 hourly snapshots, restricts the supported system to high-voltage transmission assets, attaches load/generation information, optimizes with HiGHS, and evaluates network behavior/load shedding.

Development history indicates that the 2022 base scenario is considered operational and is the intended starting point for future scenario work.

### v1 interpretation

The scientific boundary should become explicit:

historical construction of canonical network → validated `ec_network_2022.nc` → supported v1 workflow begins.

The v1 project should support loading, sanity-validating, executing, and analyzing that artifact.

It does not require a student to regenerate it from the raw PyPSA-Earth process.

# 8. Supported voltage scope

The current model work identified lower-voltage subnetworks and applied a high-voltage restriction.

The v1 decision is explicit:

- retain 138 kV;
- retain 230 kV;
- retain 500 kV;
- remove 69 kV from the supported macro-study.

Reason: including 69 kV introduces infeasibilities and detail outside the intended national transmission-scale study.

This is a documented model-scope decision, not a temporary workaround.

# 9. Current demand-processing status

The repository already contains useful Python-based demand preparation logic.

An existing demand-processing script scales an hourly/nodal demand template to observed 2022 monthly energy totals and moves resulting timestamps into the 2022 calendar.

A second demand script extends annual demand projections using Master Plan-derived values and longer-term extrapolation.

## 9.1 Clarified v1 demand model

For future year `y`:

**future profile(y) = annual scaling factor(y) × complete 2022 profile**

The scaling factor applies to the existing 2022 time and spatial pattern as a whole.

The annual factor itself may be determined from an official Master Plan annual value, linear annual growth, compound annual growth, or another explicitly documented scenario method.

The annual-growth method determines the scalar. The scalar then multiplies the complete profile.

The v1 methodology does not independently reshape hourly profiles or spatial load shares.

If a future infrastructure scenario introduces an explicit new load location, that change must be documented separately.

## 9.2 Demand cases

Canonical scenario families:

- D1: Tendential;
- D2: PME Case Base / central case;
- D3: high-growth stress.

Beyond the official PME horizon, compound annual growth is the default continuation method unless a scenario specifies otherwise.

# 10. Current generation-expansion status

The repository contains future-generation tables derived from Master Plan work and manual modeling decisions.

These include renewable, hydro, thermal/firm, and other future additions.

The data demonstrates substantial effort to map projects into the modeled network.

However, evaluated generation inputs mix official project information, normalized naming, manually chosen buses, synthetic splits, generic blocks assigned to plausible locations, and scenario-specific additions.

These categories should not remain indistinguishable.

### v1 requirement

Each future project must separate official source information, transformed/normalized fields, and model-specific assumptions.

Where the Master Plan specifies capacity but not a precise network location, assignment to a plausible bus is allowed, but must be explicitly marked as an assumption with justification, source reference, and confidence.

# 11. Current nuclear-data inconsistency

The evaluated repository contains more than one historical nuclear assumption.

At least three conceptually different nuclear pathways have appeared across future-generation data, older processed generation data, and previous scenario-development documents.

This is a strong example of why the refactor needs one authoritative scenario registry.

### Canonical v1 decision

Technology: SMR.

Scientific role: hypothetical firm low-carbon capacity used as sensitivity analysis to evaluate whether/when nuclear becomes useful under strong renewable development.

Deployment: additive by default. PME firm capacity is not automatically removed. Later scientific comparisons may explicitly define replacement cases.

Canonical additions:

| Year | Added | Cumulative |
|---|---:|---:|
| 2035 | 0.5 GW | 0.5 GW |
| 2040 | 0.5 GW | 1.0 GW |
| 2045 | 0.5 GW | 1.5 GW |
| 2050 | 0.5 GW | 2.0 GW |

Siting should preferentially use major transmission nodes.

Grid connection must always be explicit. Transformers, buses, lines, and 500 kV infrastructure are added only where electrically required.

Legacy nuclear definitions must be removed from active scenario logic or clearly archived as historical material.

# 12. Current transmission-expansion status

The repository contains Master Plan-derived future bus, line, and transformer CSVs.

The line-expansion table is comparatively rich in metadata and includes project names, endpoint mappings, original/source descriptions, year information, and modeling notes.

This is a strong basis for provenance-aware v1 inputs.

However, the evaluated data contains unresolved/questionable mappings.

One example identified during repository review was an apparent line mapping where both modeled endpoints resolved to the same bus despite the source project referring to two distinct named substations.

This type of row must not be silently repaired during general refactoring.

It must be corrected using source evidence, flagged as unresolved, or represented as an explicit modeling assumption.

# 13. Transformer expansion assumptions

The reviewed future-transformer table contains many entries using uniform electrical assumptions such as repeated reactance/resistance/capacity values.

These appear to function partly as topology/modeling defaults rather than verified equipment-specific parameters.

This is acceptable for a macro-study only if explicit.

### v1 requirement

Future transformer data must distinguish official equipment parameters, values derived from standards, generic modeling defaults, and unknown/unresolved parameters.

Repeated default values must not appear to be reported by the Master Plan when they were not.

# 14. Master Plan timing and delay logic

The original Master Plan contains project dates beginning shortly after the 2022 planning base.

By 2026, the project assumption is that the original schedule is no longer a realistic current implementation timeline.

The v1 framework therefore supports global date shifts:

- optimistic: +5 years;
- baseline: +8 years;
- pessimistic: +12 years.

Generation and required transmission are normally shifted together.

The refactor does not need a project-by-project real-world status database.

The original official project date should remain as provenance even when the modeled commissioning year is shifted.

# 15. Hydropower scope

Hydropower is central to Ecuadorian adequacy and drought resilience.

The current project contains hydro-related time-series work and explicitly intends to evaluate dry conditions.

### v1 representation

Use availability profiles such as `p_max_pu`.

Do not require explicit multi-reservoir water-balance optimization for v1.

Canonical hydrology scenarios:

- normal;
- dry;
- severe dry.

A future student research scenario should include strong reduction in hydro availability representing drought-type stress comparable in concept to the risk observed in Ecuador during 2024.

This should be scientifically documented rather than an arbitrary undocumented derating.

# 16. International interconnections

The study considers Ecuador primarily from the perspective of domestic adequacy.

The autonomous case remains the principal security-of-supply reference.

Separate cases may allow Colombia and Peru interconnection.

The scientific intention is to represent interconnection economics using documented reference assumptions, including 2026 reference prices where appropriate, so imports are not automatically treated as an unrealistically cheap solution.

Exact price sources and implementation must be documented when this scenario is formalized.

# 17. Thermal-generation treatment

The project will not assume indefinitely available 2022 thermal capacity without examination.

Thermal assets should be retired based on age, documented retirement, and documented replacement/repowering plans.

This intentionally increases future adequacy stress where old capacity disappears.

PME firm-generation blocks remain one possible adequacy strategy and are not automatically replaced by SMR.

# 18. Existing scenario registry

The current active scenario-registry implementation is minimal.

It establishes scenario identity, year, description, input-file references, and tags/metadata, but functions more as a placeholder than a complete scenario engine.

### v1 requirement

The registry must become the authoritative scenario-definition layer.

It should support anchor year, demand case, annual scaling/growth assumption, delay case, generation additions, thermal retirement, hydrology, transmission, interconnections, SMR, run configuration, and provenance.

Core model code should not need editing when a student adds a normal new scenario.

# 19. Legacy scenario modules

The repository contains older scenario-related modules under legacy paths.

They include earlier dataclasses and scenario pipelines based on older years and assumptions.

The evaluated legacy pipeline contains useful conceptual ideas but is not suitable as the v1 production base.

Observed concerns include older reference years, stale interfaces, mismatched fields, partially commented-out workflows, incomplete logic, and overlap with the newer registry concept.

### v1 decision

Do not invest substantial effort repairing the old pipeline.

Use it only as an idea archive where valuable.

The active v1 scenario engine should follow the current scientific specification rather than backward compatibility with obsolete code.

# 20. Current optimization/validation evidence

The repository contains result artifacts from optimization experiments.

One evaluated experiment varied transformer reactance assumptions and reported consistent high-level system metrics for the tested setup.

This demonstrates that the repository has moved beyond pure data preparation and that network optimization has been performed successfully.

Existing result files are useful provenance.

However, they should not automatically be treated as the sole formal acceptance reference for the final 2022 model unless their exact configuration is confirmed as canonical.

### v1 requirement

Create explicit baseline validation documentation containing which network artifact is canonical, which input versions belong to it, which structural checks identify corruption, and which scientific reference run is authoritative if one is designated.

# 21. Current configuration status

The project configuration contains settings inherited from broader PyPSA/PyPSA-Earth practice, including technology carriers, solver settings, weather/prediction-year concepts, and renewable technology options.

The configuration is broader than the final Ecuador-specific v1 runtime needs.

### refactor requirement

Separate settings genuinely required by the supported Ecuador workflow from historical PyPSA-Earth extraction settings and obsolete/unused options.

The v1 runtime configuration should be understandable without detailed knowledge of the upstream PyPSA-Earth repository.

# 22. PyPSA-Earth dependency boundary

The project was originally developed from PyPSA-Earth-derived network and climate information.

That provenance must remain visible and credited.

However, final v1 student use must not require a neighboring PyPSA-Earth checkout.

PyPSA-Earth is an upstream source/provenance dependency for canonical artifacts, not a required neighboring runtime repository.

Any process for regenerating those artifacts belongs to advanced provenance/maintenance documentation rather than the normal student quick start.

# 23. Testing status and required change

Historical research development emphasized interactive validation rather than a compact automated test suite.

A naive test strategy would be inappropriate because a full annual PyPSA solve is computationally expensive.

### v1 testing policy

Use `/tests`.

Default: unit tests with small deterministic fixtures.

Allowed: small integration tests only if necessary, short smoke tests, external artifact/schema checks.

Prohibited: long integration tests, annual Ecuador solves in the automated suite, full registered scenario batches as tests, repeated full archive downloads.

Scientific validation is distinct from software testing.

# 24. Result/output architecture

The project eventually intends to expose results through a dashboard/web interface.

Simulation and dashboard must remain separate.

The future dashboard reads saved result data and must not run PyPSA simulations.

### v1 result policy

Each scenario should have an isolated directory containing resolved inputs/metadata, outputs, time-series data, logs, figures where generated, `metadata.json`, `summary.json`, and optionally a solved network state.

Preferred formats:

- JSON for metadata/KPIs;
- Parquet for large tabular/time-series outputs;
- NetCDF for network artifacts.

Pickle may be used internally if useful but must not be the only durable public output format.

SQL can be added later if the dashboard architecture requires it.

# 25. Mandatory KPI requirements

At minimum:

- annual demand;
- peak demand;
- generation by technology;
- generator dispatch summaries;
- renewable curtailment;
- hydro availability/generation;
- load shedding/ENS;
- peak load shedding;
- imports/exports when enabled;
- line loading;
- transformer loading;
- installed capacity;
- total system cost;
- system cost per kWh;
- solver status;
- runtime.

Congestion indicators and emissions should be added where model data supports them consistently.

# 26. Documentation deficiencies

The current repository does not yet contain sufficient documentation for independent use.

The final v1 requires:

- complete root README;
- installation guide;
- first-run data guide;
- execution guide;
- scenario-authoring guide;
- methodology;
- architecture;
- data/source provenance;
- baseline validation;
- result interpretation;
- licensing;
- citation;
- third-party attribution.

Documentation is part of the scientific artifact, not optional project notes.

# 27. Provenance requirements

Future model inputs must distinguish source facts from modeling decisions.

Recommended provenance fields:

- `source_document`;
- `source_chapter`;
- `source_table`;
- `source_page`;
- `source_text`;
- `assumption`;
- `assumption_reason`;
- `confidence`;
- `created_by`;
- `last_verified`.

Not every source naturally provides every field. The governing principle is that a student should be able to determine where a value came from, whether it was directly stated, how it was transformed, whether it was assumed, and why a model location was selected.

# 28. Source hierarchy

Principal scientific sources are the Ecuadorian Master Plan demand documentation, generation-expansion/PEG documentation, transmission-expansion/PET documentation, documented 2022 generation data, documented 2022 transmission/network data, hydrological/flow data used for hydro availability, and PyPSA-Earth-derived network/climate artifacts where applicable.

Final documentation should identify exact editions, archive links, and citations rather than relying on informal filenames alone.

# 29. Licensing and public-access objective

The project is intended to remain publicly accessible.

The preferred license for original project code is MIT.

“No copyright” should not be used as the legal mechanism for openness because original work normally receives copyright automatically.

Instead, original code should be permissively licensed; third-party material retains its own terms; sources are attributed; and third-party data is not relicensed without authority.

Final repository targets include `LICENSE`, `CITATION.cff`, and `THIRD_PARTY_NOTICES.md` or equivalent.

# 30. Code-quality objective

The repository is intended for students who will extend it.

Required characteristics include descriptive variable names, explicit units where useful, modular functions, readable modules, public API docstrings, documented assumptions, minimal duplication, configuration-driven scientific choices, clear errors, and stable file conventions.

The refactor should favor clarity over cleverness.

# 31. Branch and phase governance

The repository owner manually creates a branch before each refactor phase.

The intended cycle is:

owner creates/selects phase branch → agent implements phase → agent performs minimal tests → agent writes post-phase report → owner reviews → owner commits/pushes manually.

This protects scientific provenance and prevents autonomous repository-history changes.

# 32. Scientific decisions frozen for v1

## 32.1 Base year

2022 only. Older 2017/2018 work is legacy.

## 32.2 Voltage scope

138/230/500 kV. 69 kV excluded.

## 32.3 Demand shape

2022 spatial/temporal shape retained.

Future demand is a scalar multiple of the complete 2022 profile.

The scalar may result from different annual growth methods.

## 32.4 Demand cases

D1 Tendential; D2 PME Case Base central; D3 high-growth stress.

## 32.5 Master Plan delay

+5 / +8 / +12 years.

## 32.6 Delay coupling

Generation and supporting transmission normally shift together.

## 32.7 Generation technologies

Include all relevant PME categories.

## 32.8 Undefined generation locations

Model assignments allowed with explicit assumption/provenance.

## 32.9 Thermal plants

Retire based on age/documented plans.

## 32.10 Firm generation

A possible adequacy strategy, not the only future solution.

## 32.11 Hydro

Availability profiles; normal/dry/severe-dry cases.

## 32.12 Interconnections

Autonomous adequacy primary; Colombia/Peru cases supported separately.

## 32.13 Nuclear

SMR; additive default; +0.5 GW in 2035, 2040, 2045, 2050; siting near major nodes preferred; grid reinforcement explicit and requirement-driven.

## 32.14 Transmission

PME/PET primary. Extra infrastructure requires documentation/justification.

## 32.15 N-1

Desirable but not mandatory for v1.

## 32.16 Anchor years

2022, 2030, 2035, 2040, 2050.

## 32.17 Final run resolution

Full hourly year supported. Short runs used for development/testing.

# 33. Student research boundary after v1

The refactor does not have to complete the full future research program.

Expected student work includes at least:

1. future renewable-expansion scenarios;
2. at least one SMR scenario;
3. a hydro-resilience/drought scenario.

Additional work may include thermal retirement, firm-capacity alternatives, transmission stress, interconnection sensitivity, improved renewable assumptions, improved hydrology, and optional N-1 analysis.

Students should not be required to regenerate the canonical 2022 `.nc`, rebuild the model from PyPSA-Earth, redo obsolete legacy studies, or understand historical notebook execution order.

# 34. v1 architecture target

The refactor should converge toward a layered structure.

## Layer 1 — Canonical validated baseline

Supported 2022 artifact and its validation/loading logic.

## Layer 2 — Authoritative planning translation

Normalized demand, generation, and transmission planning inputs with provenance.

## Layer 3 — Scenario definitions

Registry/configuration expressing year, demand, delay, hydrology, generation, retirement, transmission, interconnection, and SMR.

## Layer 4 — Scenario construction

Applies scenario definition to the canonical baseline without requiring core-code editing.

## Layer 5 — Execution

Solves a selected scenario or batch.

## Layer 6 — Results

Stores standardized machine-readable outputs and KPIs.

## Layer 7 — Analysis/presentation

Optional notebooks and a later dashboard consume saved outputs. They do not participate in simulation.

# 35. Main risks during refactor

## Risk 1 — Changing science while cleaning code

Bus mappings, capacities, demand scaling, carrier definitions, or solver assumptions could change accidentally.

Mitigation: freeze scientific decisions, separate structural refactors from scientific changes, and document deviations.

## Risk 2 — Losing provenance when removing old files

Legacy files may contain the only explanation of a modeling assumption.

Mitigation: inspect before deletion, migrate relevant provenance into documentation/data fields, and rely on Git history for obsolete implementation rather than undocumented scientific assumptions.

## Risk 3 — Treating assumed infrastructure as official data

Manual project mappings may be mistaken for Master Plan facts.

Mitigation: explicit assumption fields and source/assumption separation.

## Risk 4 — Heavy automated tests

Full-year optimization can make development impractical.

Mitigation: unit tests, tiny integration fixtures, scientific runs separate from CI/tests.

## Risk 5 — External archive drift

A changed archive file could silently change the model.

Mitigation: versioned archive, DOI/URI, checksum manifest, artifact version in run metadata.

## Risk 6 — Environment drift

Developers may rely on globally installed packages.

Mitigation: mandatory local `.venv`, authoritative `requirements.txt`, clean-install acceptance audit.

# 36. Refactor objectives

## Objective A — Reproducible installation

A fresh user can create `.venv` and install declared dependencies.

## Objective B — Reproducible artifact acquisition

Required large files are automatically retrieved and integrity-checked.

## Objective C — Stable baseline

The validated 2022 model is a formal external canonical artifact with documented validation.

## Objective D — Notebook-independent production

The production workflow runs without notebooks.

## Objective E — Scenario extensibility

A new scenario can be defined without changing core model code.

## Objective F — Provenance

Official data and modeling assumptions are distinguishable and traceable.

## Objective G — Small test suite

Important behavior is protected without expensive scientific simulations.

## Objective H — Standardized outputs

Scenario results are machine-readable and dashboard-ready.

## Objective I — Teaching-quality documentation

A student can independently install, run, understand, and extend the repository.

## Objective J — Clean public main branch

The final active repository contains the supported v1 system, not the debris of every development iteration.

# 37. Explicit non-objectives

The refactor does not itself need to complete every future scientific scenario, optimize generation/transmission investments, build a production dashboard, run simulations from a web page, model detailed distribution networks/Galapagos, implement detailed reservoir dispatch, perform comprehensive N-1 studies, introduce another modeling framework, or regenerate PyPSA-Earth source artifacts as part of the normal student workflow.

# 38. Definition of v1 handover readiness

The repository is ready when a new user can clone it, create the local `.venv`, install dependencies, retrieve and verify missing canonical artifacts, load the validated 2022 system, understand its scope/assumptions, perform a supported baseline execution, inspect standardized outputs, understand future planning inputs, define a scenario through the supported framework, and understand where important assumptions originated.

The student does not need to rebuild the validated network from scratch.

# 39. Planned v1 student starting point

The intended student research begins after handover.

At minimum, students should be able to develop a strong renewable-development scenario over future anchor years, an SMR sensitivity scenario, and a drought/resilience scenario with strongly reduced hydro availability.

The infrastructure supplied by v1 should make the scientific questions the difficult part. Repository mechanics should not be the difficult part.

# 40. Conclusion

The current repository contains the scientific foundation needed for the intended Ecuador scenario study.

The correct next step is not a wholesale rebuild. It is controlled consolidation.

The refactor should freeze the validated 2022 system as the canonical starting point, externalize large durable artifacts, normalize Master Plan-derived future inputs, formalize scenario composition, separate simulation from presentation, establish a local reproducible environment, and document the project at a level appropriate for independent student research.

The v1 artifact therefore represents a transition from a researcher-specific development history to a public, traceable, modular scientific platform.
