# Phase 06C action plan — 2022 availability, accounting, and validation boundary

Branch: `v1-phase-06c-2022-validation-accounting-availability`  
Parent phase: `v1-phase-06b-2022-validation-calibration`  
Status: planned; no acceptance claim is made by this document.

## Purpose

Phase 06C investigates why the regenerated 2022 model still fails the unchanged
validation screens after NLR wind and seasonal biomass corrections. The work
must distinguish physical/modeling causes from statistical-boundary differences
before changing any capacity, cost, profile, or validation contract.

The phase explicitly defers reservoir/cascade refinement and does not relax or
reconsider validation thresholds.

## Starting evidence

The latest annual solve is optimal and internally balanced:

| Quantity | Result |
|---|---:|
| Simulated total supply | 27,753.18 GWh |
| Domestic demand | 27,561.62 GWh |
| Exports | 191.57 GWh |
| Modeled balance residual | approximately 0 GWh |
| Hydro | 22,756.12 GWh vs 24,499.54 GWh reference (-7.12%) |
| Thermal | 4,349.36 GWh vs 3,393.35 GWh reference (+28.17%) |
| Nonconventional | 158.58 GWh vs 325.89 GWh reference (-51.34%) |
| Unresolved source-boundary residual | 931.43 GWh |

The model's internal balance closing does not prove that the modeled boundary
matches the CENACE or ARCONEL accounting boundary. The 931.43 GWh residual must
not be inserted as an arbitrary loss load.

## Work package A — Wind profile calibration and source equivalence

### Objective

Make wind availability consistent with a documented “weather + turbine” model
before it reaches Stage 05, while avoiding arbitrary scaling to the 57.00 GWh
CENACE/CNEL observation.

### Tasks

1. Preserve the NLR Sup3rWind CSVs and their checksums as raw external inputs.
2. Add a configuration-driven turbine specification for each plant:
   manufacturer/model or documented proxy, hub height, cut-in speed, rated
   speed, cut-out speed, power-curve source, losses, and time convention.
3. Compare at least three profiles in notebook 03:
   - legacy provincial `onwind.nc`;
   - direct NLR wind-speed profile converted through the documented turbine
     curve;
   - NLR-derived profile with documented plant losses and commissioning mask.
4. Report separately:
   - raw resource potential;
   - turbine-converted potential;
   - post-commissioning potential;
   - dispatch from the solved network;
   - CENACE/CNEL annual observation.
5. Apply the selected profile to the processed availability product before
   notebook 05. Do not use the annual observation as both a scaling target and
   an independent validation target.
6. If a scaling experiment is useful, write it as a named sensitivity output;
   do not make it the baseline without owner review.

### Acceptance checks

- Complete, unique 8,760-hour calendar for each site.
- No silent zero-filling or missing-hour interpolation.
- Values remain in `[0, 1]` after turbine conversion and losses.
- Commissioning masks match the documented in-year events.
- Annual candidate energy and capacity factors are reported by plant.
- The candidate decision and all parameters are recorded in a provenance table.

## Work package B — Hydro shortfall without reservoir/cascade refinement

### Objective

Determine whether the 1,743.42 GWh hydro deficit is caused by the hydro resource
profiles, plant capacities, temporal assignment, network transfer limits, or
the statistical boundary. Reservoir/cascade physics remain deferred.

### Tasks

1. Reconcile hydro capacity and plant identity against the source fleet table;
   flag capacity proxies and plants whose model capacity is not directly
   comparable with CENACE gross production.
2. Produce monthly hydro potential, dispatch, and curtailment by plant,
   province, and major cascade group.
3. Compare model gross/available energy with the cited CENACE/ARCONEL monthly
   observations, preserving gross-versus-net labels.
4. Separate hydro energy that is:
   - unavailable because of the input profile;
   - available but curtailed by the linear network;
   - displaced by cheaper imports or other generation;
   - not comparable because of source boundary or gross/net definitions.
5. Run controlled analytical diagnostics only: copperplate merit order,
   transmission-relaxed comparison, and plant-level potential/dispatch tables.
   Do not add reservoir state variables or cascade links in Phase 06C.

### Acceptance checks

- Every missing hydro GWh is assigned to a documented category or marked
  unresolved/non-comparable.
- Monthly profiles and figures are saved under `results/phase_06c/` and
  `reports/figures/phase_06c/`.
- No hydro input is changed solely to close the annual gap.

## Work package C — Thermal maintenance and availability

### Objective

Test whether the 956.01 GWh thermal overproduction is explained by historical
maintenance, outages, deratings, fuel availability, or source-boundary scope.

### Tasks

1. Build a plant-level thermal availability evidence table with plant ID,
   technology, capacity, source document, reporting period, outage/maintenance
   dates, unavailable MW, confidence, and `official_fact`.
2. Search CENACE annual/monthly operational statistics, CELEC plant reports,
   maintenance notices, and official generation tables for 2022 evidence.
3. Distinguish:
   - planned maintenance;
   - forced outage;
   - derating;
   - fuel/dispatch restriction;
   - unavailable evidence.
4. Implement availability only through reusable time-series functions and a
   source-backed table. Keep a no-outage baseline and an evidence-based
   availability variant.
5. Compare plant-level and monthly thermal dispatch before and after the
   availability variant. Do not reduce thermal capacity or alter marginal costs
   simply to pass the carrier screen.

### Acceptance checks

- Every availability reduction has a source locator or is explicitly labeled a
  modeling sensitivity.
- The model never treats an undocumented outage as an official fact.
- Thermal changes are reported by plant, month, GWh, and validation impact.
- Fuel constraints and maintenance constraints remain distinguishable.

## Work package D — Nonconventional supply decomposition

### Objective

Resolve the remaining 167.31 GWh nonconventional shortfall without hiding
biomass calibration inside an independent validation claim.

### Tasks

1. Report wind, biomass, biogas, and solar separately in all validation tables.
2. Keep biomass's ARCONEL monthly shape and CENACE annual control labeled as a
   calibration input.
3. Search plant-level biomass and biogas production, outage, and fuel records.
4. Compare modeled monthly biomass and biogas dispatch with CENACE/ARCONEL
   observations where the accounting boundary is comparable.
5. Evaluate whether the current p_max ceiling is too restrictive or whether
   dispatch is displaced by thermal/import economics. Do not force dispatch to
   the annual target.

### Acceptance checks

- Biogas remains separate from biomass.
- Gross ARCONEL observations are not silently compared with CENACE net values.
- Calibration inputs are excluded from independent validation claims.
- Monthly zeros and seasonal limits are tested after Stage 05 assembly.

## Work package E — Accounting boundary and loss investigation

### Objective

Determine whether CENACE/ARCONEL losses, self-consumption, non-block
transactions, station service, or gross/net definitions explain the apparent
generation gap.

### Tasks

1. Extend the existing accounting bridge with one row per term and fields for
   organization, edition, year, table/page, gross/net basis, metering point,
   included generator classes, imports, exports, transmission losses,
   distribution losses, station service, self-consumption, and confidence.
2. Reconcile at least these identities separately:
   - CENACE net supply + imports − demand − exports;
   - ARCONEL gross production + imports − transmission losses − public-service
     deliveries;
   - distribution availability − final consumption − distribution losses.
3. Compare the model boundary with each published identity. Do not call the
   931.43 GWh residual “losses” unless a source-backed bridge proves it.
4. Add explicit model loss components only after the bridge identifies a
   comparable quantity and the owner approves the representation.
5. Keep customer demand, exports, curtailment, load shedding, and physical
   losses as separate accounting terms.

### Acceptance checks

- The internal PyPSA balance closes independently of the source-boundary bridge.
- Every source residual is quantified and explained or marked unresolved.
- No loss load is introduced merely to reduce TVD.
- Total-energy screening remains disabled until the common boundary is closed.

## Work package F — Retroactive PDF provenance audit

Effective immediately, every PDF consulted by the project must be downloaded to
the repository's documentation evidence area or explicitly recorded as an
owner-approved external artifact. A PDF may not be cited only by a web URL.

### Required registry fields

Create or extend `data/interim/provenance/pdf_source_registry.csv` with:

`source_id,organization,title,edition,publication_year,source_url,local_path,
download_date,bytes,sha256,license,locator_used,topics,consulted_by,notes`

### Retroactive audit scope

Audit every PDF referenced by:

- `AGENTS.md`, phase reports, and technical reports;
- notebooks and source modules;
- `config/artifacts.yaml` and provenance tables;
- current CENACE, ARCONEL, PME, CELEC, and transmission evidence.

At minimum, verify the two CENACE annual PDFs and the Plan Maestro PDFs already
present under `literature/`. Missing PDFs must be downloaded only from the
documented official URL, checksum-pinned, and placed under
`literature/<organization>/` or `data/raw/statistics/` according to source role.
If licensing or access prevents committing a PDF, record the failed retrieval,
license status, URL, checksum if available, and owner decision in the registry.

### Acceptance checks

- Every PDF citation has a local path or an explicit blocked-download record.
- Every local PDF has size and SHA-256 recorded.
- Every numerical extraction retains page/table/figure locator.
- No report claims a PDF-based fact without a registry entry.

## Work package G — Regenerated validation and reporting

After A–F, rerun generated notebooks 00–05, then the owner-authorized annual 06
solve and 06a diagnostic. Write a new Phase 06C interim report containing:

- solver status and checksum;
- energy balance and source-boundary bridge;
- carrier results before/after each approved correction;
- wind and thermal plant-level evidence;
- hydro monthly diagnostics without reservoir refinement;
- calibration versus independent validation labels;
- unchanged thresholds and explicit remaining failures;
- commands not run and reasons.

## Branch and handoff

All Phase 06C work belongs on:

`v1-phase-06c-2022-validation-accounting-availability`

The owner reviews source evidence, approves any loss representation and thermal
availability assumptions, and performs the final annual scientific acceptance
run. Phase 06C is not complete until the action-package tests, PDF registry,
figures, report, checksums, and manual-run handoff are present.
