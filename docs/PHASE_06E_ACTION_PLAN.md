# Phase 06E Historical Transmission Capacity Audit

## Purpose and boundary

Phase 06E will determine whether the transmission capacities represented by the
Phase 06D 2022 network are supported by official Ecuadorian technical
documentation. It is a documentary extraction and reconciliation phase, not a
network redesign or a new validation run. The Phase 06D unrestricted,
transmission-constrained model remains the sole comparison baseline.

This phase may inspect line and transformer endpoints, voltage levels, circuit
representation, impedances, and lengths to identify physical assets. It may
propose changes only to existing line nominal capacities, existing line
operational maximum-loading factors, existing transformer ratings, and
documented transformer operational limits. It must not add or remove buses,
substations, lines, transformers, voltage levels, generators, or connections;
change demand, profiles, costs, endpoints, impedances, or thermal capacity;
or apply artificial generation restrictions. No correction is applied until
the owner has reviewed and approved the correction registry.

The current session ends after this action plan and the corresponding project
plan update. It does not perform PDF extraction, alter the network, run the
annual solve, or claim the audit is complete.

## Work products and phase gates

The audit will produce the following versioned interim tables:

- `data/interim/topology_audit_2022/official_lines_inventory.csv`
- `data/interim/topology_audit_2022/official_transformers_inventory.csv`
- `data/interim/topology_audit_2022/lines_audit.csv`
- `data/interim/topology_audit_2022/transformers_audit.csv`
- `data/interim/topology_audit_2022/capacity_corrections.csv`
- `data/interim/topology_audit_2022/source_register.csv`

The technical interpretation and unresolved evidence will be recorded in
`reports/PHASE_06E_TECHNICAL_REPORT.md`. Official assets absent from PyPSA will
be reported separately and will not be added to the model.

The work is gated as follows:

1. **Source discovery gate.** The source register identifies every relevant
   repository PDF, external official source, edition, coverage, and retrieval
   status. Indexes, chapters, annexes, technical tables, and substation
   descriptions are explicitly reviewed or marked not applicable with a reason.
2. **Official inventory gate.** Every identifiable 2022 line and transformer
   asset is entered once, preserving the original name, value, unit, status,
   and exact documentary locator. Missing ratings are recorded as
   `NOT_DOCUMENTED`, never inferred silently.
3. **Model inventory gate.** Every Line and Transformer in the actual Phase
   06D network is extracted independently from the network artifact, including
   endpoints, voltage evidence, `s_nom`, `s_max_pu` where present, effective
   rating, and source-network provenance.
4. **Reconciliation gate.** Each modeled component appears exactly once in its
   audit table and has a documentary match status: `VERIFIED`, `PROBABLE`,
   `AMBIGUOUS`, or `UNMATCHED`. Official assets not matched to PyPSA are listed
   separately.
5. **Correction-review gate.** Each proposed correction has a source locator,
   original and proposed values, conversion formula, confidence, and rationale.
   The owner reviews this registry before any notebook or network change.
6. **Implementation gate.** Only after approval are verified corrections
   implemented through the reproducible Notebook 04/05 pipeline. The original
   Phase 06D artifact is retained and checksummed.
7. **Scientific rerun gate.** A full-year rerun requires explicit owner
   authorization. It must use the mandatory preflight and sequential notebook
   workflow and report before/after metrics against the Phase 06D baseline.

## Source discovery and documentary coverage

The investigation starts with a read-only inventory of `literature/`, `docs/`,
`data/`, legacy evidence directories, manifests, and existing citation tables.
Each candidate source receives a stable `source_id` before extraction. The
preferred evidence hierarchy is:

1. Plan Maestro de Electricidad 2022 transmission-system chapters,
   infrastructure inventories, technical tables, annexes, and substation
   descriptions;
2. CENACE technical and annual reports describing the 2022 Sistema Nacional
   Interconectado and its transmission infrastructure;
3. CELEC EP TRANSELECTRIC line, substation, and transformer specifications;
4. official Ecuadorian regulatory or institutional publications used to fill a
   documented gap.

For each source, the extractor records filename or stable URI, institution,
title, edition/publication date, retrieval date, checksum when available,
language, relevant page range, chapters/annexes/tables inspected, and whether
the source describes existing 2022 infrastructure or future/planned assets.
The review log must distinguish a source that was searched from one that
contains usable technical evidence. Planned infrastructure is excluded from
the 2022 existing-asset inventory unless the document explicitly establishes
its operational status in 2022.

The PDF workflow is reproducible and staged:

1. register and checksum the source without modifying the original;
2. extract searchable text and page images where needed;
3. inspect the document index and identify all transmission chapters, asset
   inventories, annexes, tables, and substation sections;
4. search alternate names, voltage levels, corridors, substations, and
   Spanish technical terms for lines, circuits, current, MVA, transformers,
   and operational status;
5. transcribe the complete applicable rows, including rows without ratings;
6. retain the original wording/value/unit and record any normalization or
   conversion separately;
7. cross-check conflicting editions or institutions and preserve the conflict
   rather than choosing an undocumented value;
8. record page and table locators that another reviewer can reproduce.

External searches are limited to official institutional domains. A missing
repository PDF may be downloaded only through a configured persistent URI,
validated against its expected size/checksum, and placed at its authoritative
path after validation. A failed or unavailable source is an explicit audit
finding, not a reason to substitute another year or edition.

## Official-document inventories

The official line inventory contains one row per identifiable physical line or
line circuit/corridor representation, with stable fields for:

`official_asset_id`, `official_name`, `alternate_name`, `from_substation`,
`to_substation`, `voltage_kv`, `capacity_value`, `capacity_unit`,
`current_value`, `current_unit`, `circuit_count`, `status_2022`,
`asset_scope` (`EXISTING_2022`, `PLANNED`, or `UNCLEAR`), `source_id`,
`source_filename`, `source_page`, `source_table`, `source_quote_or_label`,
`original_value_text`, `conversion_formula`, `evidence_quality`, and
`notes`.

The official transformer inventory uses analogous fields and adds
`substation`, `voltage_high_kv`, `voltage_low_kv`, `rated_mva`, `rated_unit`,
`unit_count`, `bank_configuration`, `status_2022`, and `operational_limit`.
Where a source gives only current or only a bank total, the original value is
preserved and the conversion to MVA is recorded with the voltage basis and
formula. No capacity is fabricated when voltage, current, or bank scope is
not documented.

## Independent Phase 06D PyPSA inventory

The model inventory is generated from the actual Phase 06D network artifact,
not from an existing audit CSV. The extraction records the network checksum,
creation/provenance metadata, and the exact component table used. For every
modeled line it records:

`pypsa_component_id`, `bus0`, `bus1`, `bus0_voltage_kv`, `bus1_voltage_kv`,
`s_nom_mva`, `s_max_pu`, `effective_rating_mva`, `circuit_or_equivalent_note`,
`source_network`, `network_checksum`, and `inventory_status`.

For every modeled transformer it records the same identity/provenance fields
plus `voltage_ratio`, `s_nom_mva`, `s_max_pu` if applicable, and the modeled
bank/unit representation. Effective rating is defined explicitly from the
stored attributes and is not treated as a physical thermal rating without
documentary support.

## Matching and reconciliation method

Matching is performed using normalized substation names, aliases, endpoint
direction independent of bus ordering, voltage level, corridor names, circuit
counts, coordinates where available, and known source-network identifiers.
The matching worksheet retains candidate matches and the evidence used to
accept or reject them. A PyPSA component is classified as:

- `VERIFIED` when documentary identity and technical context establish the
  physical match;
- `PROBABLE` when the evidence supports one best match but omits a confirming
  identifier or specification;
- `AMBIGUOUS` when two or more physical assets remain plausible; or
- `UNMATCHED` when no defensible documentary correspondence is found.

The reconciliation is many-to-one aware: a physical transformer bank may be
represented by one or several PyPSA components, and a modeled equivalent line
may represent multiple circuits. Such representation decisions are stated in
the audit rather than treated as automatic capacity equality. Every modeled
component must occur once and only once in the final line or transformer audit;
duplicate component IDs, missing components, and unassigned official assets
are quality-gate failures.

## Component-level comparison and correction registry

`lines_audit.csv` compares official voltage/current or MVA evidence with
PyPSA `s_nom`, `s_max_pu`, and effective operational rating. It includes
`mva_discrepancy`, `percent_discrepancy`, `circuit_representation`,
`match_status`, `evidence_quality`, `recommended_action`, and all source
locators. Physical thermal ratings are kept separate from operational/security
limits; a documented limit below the thermal rating is not automatically
converted into `s_max_pu = 1.0`.

`transformers_audit.csv` compares physical transformer/bank identity,
substation, voltage levels, official MVA, unit count, operational limits, and
modeled values. Bank totals and per-unit ratings are not combined without an
explicit documented interpretation.

`capacity_corrections.csv` is a review registry, not an instruction to mutate
the network. It contains `correction_id`, `pypsa_component_id`, `component_type`,
`original_attribute`, `original_value`, `proposed_value`, `unit`,
`correction_type`, `source_id`, `source_page`, `source_table`,
`conversion_formula`, `match_status`, `confidence`, `rationale`,
`owner_review_status`, and `implementation_note`. Only `VERIFIED` corrections
approved by the owner may later be applied.

## Automated quality checks

A cheap validation script/test suite will check:

- all six output paths and required columns exist;
- every modeled Line and Transformer from the Phase 06D artifact appears once;
- modeled component IDs and official asset IDs are unique within their scopes;
- every modeled row has one allowed match status;
- every official evidence row has source ID, filename/URI, page, and table or
  an explicit `NOT_DOCUMENTED` locator reason;
- missing ratings are represented as `NOT_DOCUMENTED`, not empty or invented;
- units belong to an allowed set and conversions carry a formula and voltage
  basis where needed;
- no planned/unclear asset is silently classified as existing in 2022;
- every proposed correction has traceable provenance and review status;
- no correction changes endpoints or introduces a new component;
- official assets missing from PyPSA are listed and excluded from corrections.

The checks validate record completeness and provenance, but they do not turn
automatically generated rows into documentary evidence. The technical report
must include the source coverage matrix, extraction log, inventory counts,
match-status counts, unmatched official assets, unresolved ambiguities,
missing ratings, and reviewer sign-off state.

## Integration after owner approval

Only after the correction registry is reviewed will the implementation point be
confirmed by inspecting Notebooks 04 and 05. Corrections will be represented
as a source-traceable transformation in the notebook-led pipeline, preserving
the original capacity and network checksum. The final NetCDF will not be
patched directly. The transformation must assert that topology, endpoints,
component counts, voltage levels, generation, demand, and profiles are
unchanged, except for the explicitly approved capacity attributes.

The corrected candidate will pass preflight, consistency tests, serialization
checks, and a capacity-diff report before any solve. The Phase 06D baseline is
retained for direct comparison and is never overwritten.

## Full workflow re-execution and scientific validation

After explicit owner authorization, the sequential workflow will run from
Notebook 00 through the applicable 06A/06 validation stage in the approved
`pypsa-earth-ec` environment. Mandatory preflight must read the artifact
manifest, verify required inputs and checksums, and fail early if an input is
unavailable. The run log records commands, environment, solver status, network
checksums, correction registry version, and skipped commands.

Before/after tables compare the corrected result with the unrestricted Phase
06D baseline for hydroelectric generation, thermal generation, imports,
renewable generation, load shedding, congestion and line loading, transformer
utilization, CENACE validation indicators, energy-balance residual, and the
remaining thermal-generation discrepancy. The accounting includes annual
demand, generation, imports/exports, storage, losses, curtailment, and load
shedding with explicit units and residual formulas. The copperplate experiment
remains diagnostic evidence and is not substituted for the corrected grid.

## Technical report structure

`PHASE_06E_TECHNICAL_REPORT.md` will contain:

1. scope, branch, baseline artifact, and prohibited changes;
2. source register, documentary coverage, checksums, and extraction method;
3. official line inventory summary and source limitations;
4. official transformer inventory summary and source limitations;
5. independent PyPSA inventory counts and provenance;
6. exhaustive reconciliation results, including every component count and
   match-status distribution;
7. line-level and transformer-level discrepancy summaries with links to the
   audit tables;
8. official assets missing from PyPSA and components with missing ratings;
9. correction registry, confidence, unresolved issues, and owner review state;
10. quality-check results and commands executed or intentionally skipped;
11. implementation handoff for Notebooks 04/05;
12. authorized before/after validation results, if and only if that later gate
    has occurred.

The report will not call the topology audit complete until documentary source
coverage and the two-sided inventory gates are satisfied. It will not call the
2022 network scientifically revalidated merely because a capacity comparison
has been completed.

## Acceptance criteria

Phase 06E is ready for owner review only when:

- all relevant repository and official external source families have been
  identified, with chapters, annexes, tables, and technical sections reviewed;
- the official line and transformer inventories preserve original values,
  units, status, and exact source locators;
- every Phase 06D PyPSA Line and Transformer appears exactly once in the final
  audit tables;
- every modeled component has a documentary match status, including
  `UNMATCHED` where necessary;
- official assets missing from PyPSA are reported without being added;
- missing and ambiguous ratings are explicit and never silently inferred;
- every proposed capacity correction is traceable to official evidence and is
  marked pending owner approval;
- automated checks pass for completeness, duplicates, units, statuses, and
  provenance;
- the technical report summarizes all components and the documentary limits;
- no network or model change has occurred during the audit-only stage.

The later implementation and scientific rerun are accepted only if the owner
approves the correction registry, the pipeline proves that prohibited topology
and dispatch inputs are unchanged, the solver terminates successfully, and the
full energy balance and contracted validation criteria are reported against the
Phase 06D baseline.
