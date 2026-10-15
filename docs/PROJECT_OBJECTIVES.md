# Project objectives

## Purpose

Convert a researcher-specific collection of notebooks, intermediate files, and
experiments into a reproducible, understandable project that a student group can
run and extend without reconstructing its history.

## Primary scientific objective

Produce a runnable Ecuador 2022 PyPSA network and validate annual simulated
electricity supply by carrier against authoritative 2022 statistics. The target
is agreement within 5% for each in-scope carrier under a documented comparison
formula and common carrier taxonomy.

The statistical reference, carrier mapping, handling of imports/exports, storage,
autoproducers, isolated systems, losses, and any residual category must be fixed
before the acceptance test is implemented. Until that contract exists and passes,
the term “validated 2022 network” must not be used for an artifact.

## Secondary objectives

- Preserve and explain the successful legacy consistency-check methodology.
- Replace duplicated notebook logic with tested functions while retaining an
  explanatory, sequential notebook workflow for students.
- Make all required inputs discoverable before computation begins and retrieve
  missing large artifacts automatically from persistent, checksum-pinned URIs.
- Preserve manual network fixes and PDF-derived information with provenance.
- Normalize 2022 Master Plan generation, transmission-line, transformer, and
  substation expansions into tables ready to filter and aggregate by year.
- Separate official planning evidence from researcher assumptions and student
  scenarios.
- Produce phase technical reports containing reproducible validation plots.
- Provide a clean scenario extension point after the baseline is validated.

## Student handover outcome

A new student should be able to:

1. clone the repository and use the documented existing Conda environment;
2. run one preflight that obtains or clearly reports every missing artifact;
3. execute the numbered notebooks in order;
4. reproduce the accepted 2022 validation and its figures;
5. inspect the provenance and transformation of Master Plan projects;
6. define future assumptions without editing baseline model logic;
7. run a selected future year and interpret standardized results.

## Scope boundaries

In scope for the refactor:

- mainland Ecuador/SNI 2022 baseline;
- demand, existing generation, hydro/renewable availability, transmission,
  network repair, optimization, energy accounting, and statistical validation;
- source-traceable future generation and grid expansion inputs;
- short software tests and explicit full-year scientific validation;
- notebook explanations, saved figures, reports, licensing, and citation.

Deferred to later phases or students:

- selection and completion of future scenario families;
- hypothetical nuclear/SMR scenario design;
- endogenous capacity expansion;
- detailed reservoir water-balance optimization;
- distribution-grid and Galapagos modeling;
- a production dashboard.

## Quality priorities

In order: scientific correctness, provenance, reproducibility, clarity for
students, recoverability, and performance. A clear failure is preferable to a
plausible but untraceable result.
