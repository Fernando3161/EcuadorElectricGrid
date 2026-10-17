# Phase 06b interim report: 2022 validation calibration

Date: 2026-10-17

Status: evidence and implementation plan. This report does not authorize tuning
inputs solely to pass validation, weakening thresholds, or treating the current
network as validated. Phase 06b must preserve the distinction between source
correction, model calibration, and independent validation.

Proposed branch: `v1-phase-06b-2022-validation-calibration`.

## Executive conclusion

The Phase 06a solve is optimal and its modeled energy balance closes. The two
material islands were repaired successfully: load shedding fell from about
924.76 GWh to 61.13 GWh and disappeared at buses 291 and 159. The remaining
carrier mismatch is not explained by one defective hydro scaling factor. It has
four separable causes:

1. the Santa Elena-Chongon corridor is represented with one generic circuit,
   while the official transmission inventory contains two circuits with 113 MVA
   and 266 MVA ratings;
2. the provincial `onwind.nc` profiles are dimensionally labelled correctly but
   have implausibly low wind resource at the two Ecuador wind plants;
3. biomass and biogas are modeled as continuously fuel-available generators,
   contrary to the strongly seasonal official Ecuador biomass record;
4. the CENACE production table and modeled demand boundary still differ. An
   ARCONEL national energy balance now provides the correct source framework for
   resolving that difference, but it does not justify assigning the complete
   931.43 GWh residual directly to losses.

Hydro remains important, but it is not the first calibration target. The
existing profiles permit 26,894.11 GWh, above the CENACE 24,499.54 GWh annual
reference. The model curtails 4,299.99 GWh because of hourly availability,
network restrictions, and economic dispatch. The largest cascade is already
close in aggregate annual energy, although its allocation among plants is not.

The correct Phase 06b sequence is therefore: repair documented transmission
ratings, rebuild wind profiles, add source-backed bioenergy availability,
formalize the energy-accounting bridge, add a reproducible copperplate
diagnostic, and only then reassess validation criteria. Threshold relaxation is
not an acceptable substitute for this work.

## 1. Phase 06a result reproduced

| Quantity | Phase 06a result |
|---|---:|
| Domestic demand | 27,561.620 GWh |
| Fixed exports | 191.570 GWh |
| Generation plus imports | 27,692.060 GWh |
| Load shedding | 61.130 GWh |
| Energy-balance residual | effectively zero |
| Reference production plus imports | 28,684.610 GWh |
| TVD | 0.03819 |
| Solver termination | optimal |

The supply identity is:

```text
27,692.0595 generation/imports + 61.1305 shedding
= 27,561.6200 domestic demand + 191.5700 exports
```

The carrier screen fails for hydro, thermal, and non-conventional generation;
interconnection passes only because imports reach the upper edge of the
historical tolerance band.

## 2. Bus 58 and line 47: Santa Elena-Chongon

### Model diagnosis

The coordinates identify bus 58 as the Santa Elena area and bus 145 as the
Chongon area:

| Model element | Longitude | Latitude | Voltage |
|---|---:|---:|---:|
| Bus 58 | -80.3437 | -2.6328 | 138 kV |
| Bus 145 | -80.3203 | -2.2068 | 138 kV |

Line 47 is the only retained connection at bus 58. It is modeled as one generic
138 kV circuit with `s_nom = 154.17 MVA` and `s_max_pu = 0.7`, giving an effective
active-power limit of 107.92 MW. It is at or above 99% utilization for about
5,215 hours. Bus 58 carries 904.56 GWh of annual demand and accounts for 61.127
of the 61.130 GWh remaining shedding.

Four original 69 kV loads (`load_57`, `load_139`, `load_201`, and `load_282`) are
aggregated at bus 58 by the voltage-reduction workflow. This aggregation must be
reviewed together with the corridor rating; correcting the line alone must not
hide an incorrect load attachment.

### Official evidence

The Plan Maestro de Electricidad 2023-2032, Chapter 5, transmission inventory,
PDF page 12 (printed page 168), lists two distinct 138 kV circuits:

| Official line | Length | Capacity | Circuits/conductor record |
|---|---:|---:|---|
| Chongon-Santa Elena C1 | 81.60 km | 113 MVA | one ACSR 397.5 circuit |
| Chongon-Santa Elena C2 | 84.67 km | 266 MVA | one ACAR 1200 circuit |

The same chapter, PDF page 24 (printed page 180), recognizes voltage limitations
in the Chongon-Santa Elena-Posorja zone and identifies future reinforcement. This
supports preserving the real corridor limits, but not reducing the 2022 corridor
to one 107.92 MW equivalent.

### Phase 06b decision

High-confidence correction candidate: replace/update model line 47 through the
CSV repair mechanism so the two official circuits are represented explicitly,
or as a documented equivalent only if PyPSA parallel-line treatment requires it.
Do not copy the current straight-line geometry as an official route length; use
the official 81.60 km and 84.67 km values. Preserve each circuit's conductor and
rating separately where practical.

Before implementation, confirm the bus-name mapping against the official SNT
single-line diagram and review the four load attachments. The expected outcome
is elimination of artificial shedding at bus 58 without inventing generation.

## 3. La Concordia corridor circuit count and capacity

The Phase 06a repair uses one 113 MVA Santo Domingo-La Concordia segment, one
113 MVA La Concordia-Quininde segment, and one 202.57 MVA La
Concordia-Pedernales segment.

The current Master Plan transmission inventory supports those 2022 values:

| Official line | Voltage | Length | Capacity | Circuits |
|---|---:|---:|---:|---:|
| Santo Domingo-La Concordia | 138 kV | 39.10 km | 113 MVA | 1 |
| La Concordia-Quininde | 138 kV | 41.40 km | 113 MVA | 1 |
| La Concordia-Pedernales | 138 kV | 79.70 km | 202.57 MVA | 1 |

Source: Plan Maestro 2023-2032, Chapter 5, PDF page 13 (printed page
169), Table 5-7. Section 5.9.4.10, PDF page 48, states that La Concordia
sections one circuit of Santo Domingo-Esmeraldas. PDF page 24 states that
sectioning the second circuit at La Concordia and Quininde is planned for 2028.

An earlier CELEC construction notice described La Concordia-Pedernales as a
double-circuit line. That describes the project/build configuration, while the
later in-service planning inventory records one operating circuit. Phase 06b
must use the in-service 2022 inventory unless a 2022 switching or operating
diagram demonstrates that the second circuit was energized.

The repaired Santo Domingo-La Concordia segment is saturated for about 7,230
hours in the Phase 06a solve. This may represent a genuine system limitation,
not an erroneous rating. The action is to verify the 2022 operating state and
retain the current one-circuit rating by default, not automatically increase it.

Official web evidence:

- CELEC, construction-stage description of the system:
  https://www.celec.gob.ec/transelectric/uncategorized/nuevo-sistema-de-transmision-la-concordia-pedernales-138-kv/
- CELEC, final system context and SNT sectioning evidence already cited in the
  Phase 06a interim report.

## 4. Wind-profile audit

### What `onwind.nc` actually contains

`data/raw/cutouts/onwind.nc` is a 2013 hourly dataset with:

| Variable | Dimensions | Declared units | Meaning used by the code |
|---|---|---|---|
| `profile` | 8,760 hours x 24 provinces | p.u. | generator `p_max_pu` |
| `capacities` | 24 provinces | MW | upstream provincial potential; not used to scale existing plants |

The current loader correctly treats `profile` as per-unit availability. It does
not multiply or divide the profile by the NetCDF `capacities` variable. Therefore
the defect is not presently explained by a MW-versus-p.u. conversion or a simple
decimal error in `src.availability`.

Dataset-wide diagnostics:

| Statistic | Value |
|---|---:|
| Mean provincial profile | 0.02132 p.u. |
| Median sample | 0.00180 p.u. |
| Maximum sample | 0.84544 p.u. |
| Sum of provincial `capacities` | 12,119.38 MW |

Relevant province profiles:

| Province | Mean profile | Full-load hours | Profile maximum |
|---|---:|---:|---:|
| Azuay | 0.01853 | 162.33 h/year | 0.43395 |
| Loja | 0.01296 | 113.51 h/year | 0.33164 |

Plant results:

| Plant | Capacity | Current 2022 profile potential |
|---|---:|---:|
| Huascachaca | 49.98 MW | 0.163 GWh after its December commissioning mask |
| Villonaco | 16.50 MW | 1.873 GWh |

CENACE reports 57.00 GWh of net wind generation in 2022. Huascachaca groups A
and B entered operation only on 2 and 24 December 2022, respectively (CENACE
2022 Annual Report, PDF page 111). The annual wind total must therefore be
dominated by Villonaco. A 16.5 MW plant producing approximately 57 GWh has an
order-of-magnitude capacity factor near 39%, compared with 1.30% in the Loja
profile. The Master Plan also uses energy/capacity combinations around 40-54%
for future Ecuador wind projects. The current wind profile is demonstrably
unfit for historical validation.

### Most likely causes to test

The NetCDF has no global provenance metadata identifying turbine model, hub
height, wind-speed extrapolation, cutout-generation configuration, spatial
weighting, or exclusion rules. The low values are already present in the source
NetCDF, before the Phase 4 plant mapping. Plausible causes are:

1. an unsuitable turbine power curve or hub height;
2. wind-speed height conversion omitted or applied incorrectly;
3. provincial averaging diluted high-elevation plant sites with large low-wind
   areas;
4. an availability matrix or land eligibility mask used as an unintended
   weight in the per-unit profile;
5. profile normalization against broad provincial potential rather than plant-
   site capacity;
6. corrupted or incomplete upstream metadata, making the artifact impossible to
   reproduce scientifically.

### Corrective direction

Do not repair this by multiplying the existing profile by 30. Regenerate wind
availability from the checksum-pinned weather cutout or another versioned ERA5
source at the actual Villonaco and Huascachaca coordinates. Record turbine,
hub height, wind-speed extrapolation, weather year, interpolation, clipping,
commissioning masks, and resulting capacity factors. Compare at least:

- Villonaco modeled annual capacity factor versus its independent CENACE annual
  production;
- Huascachaca December production/availability versus its commissioning dates;
- 2013 weather sensitivity versus a 2022 weather run if a 2022 cutout can be
  obtained without silent substitution.

The current `onwind.nc` should remain preserved as rejected upstream evidence,
not overwritten in place.

## 5. Biomass and biogas availability

### Current defect

The model gives the 144.30 MW biomass fleet and 8.32 MW biogas fleet continuous
unit availability. With marginal costs of 95 and 85 USD/MWh, respectively, the
optimizer uses them to displace 180 USD/MWh oil generation:

| Carrier | Phase 06a generation | CENACE net reference |
|---|---:|---:|
| Biomass | 600.51 GWh | 195.68 GWh |
| Biogas | 36.40 GWh | 40.06 GWh |

Biogas is already close annually. Biomass is overproduced by about 405 GWh and
is the dominant non-conventional error.

### Ecuador-specific monthly evidence is available

No foreign-country proxy is required. ARCONEL's *Estadistica Anual y Multianual
del Sector Electrico Ecuatoriano 2022*, Table 54, reports monthly Ecuador biomass
production for autogenerators:

| Month | Biomass GWh |
|---|---:|
| Jan | 5.34 |
| Feb-May | 0.00 |
| Jun | 14.03 |
| Jul | 52.76 |
| Aug | 52.05 |
| Sep | 50.66 |
| Oct | 49.13 |
| Nov | 47.37 |
| Dec | 17.15 |

This is an official, Ecuador-specific harvest-season shape. Its annual gross
scope (288.49 GWh for autogenerators, and 348.08 GWh in the broader national
technology total) differs from CENACE's 195.68 GWh net block-transaction value.
It may therefore define temporal shape, but must not be copied as the CENACE
validation total without a boundary reconciliation.

Sources:

- ARCONEL publication page:
  https://arconel.gob.ec/publicaciones-estadistica-del-sector-electrico-2/
- ARCONEL 2022 statistics PDF:
  https://arconel.gob.ec/wp-content/uploads/downloads/2025/01/EstadisticaAnual2022-baja-1.pdf
- CENACE 2022 Annual Report, Tables/Figures on PDF pages 73-75.

### Recommended model

1. Create a source-faithful monthly biomass shape table from ARCONEL Table 54.
2. Investigate plant-level production or harvest records for San Carlos,
   Ecoelectric, and Ecudos before assigning the aggregate shape to each plant.
3. If plant-level records remain unavailable, apply the national monthly shape
   with an explicit shared-profile assumption and sensitivity test.
4. Represent biomass using monthly energy budgets or monthly `p_max_pu`, rather
   than continuous annual fuel availability.
5. Keep CENACE 195.68 GWh as the independent net validation observation unless
   the same value is deliberately adopted as a calibration control. If used as
   a calibration control, remove biomass annual energy from the independent
   acceptance gate and validate its monthly shape instead.
6. For biogas, first compare monthly CENACE/ARCONEL data. A mild availability
   profile or annual fuel budget may be sufficient; do not impose the biomass
   harvest shape on landfill gas.

## 6. Hydro monthly and cascade audit

### Aggregate finding

| Quantity | GWh |
|---|---:|
| Current hydro availability ceiling | 26,894.11 |
| Phase 06a hydro dispatch | 22,594.12 |
| Current hydro curtailment | 4,299.99 |
| CENACE net hydro reference | 24,499.54 |

The annual profile ceiling is not too low. A post-processed copperplate merit
order using the same hourly profiles and solved import chronology dispatches
about 23,118.02 GWh of hydro. Approximately 524 GWh of the difference between
that value and the network result is therefore associated with spatial network
constraints. The remaining difference reflects hourly availability, generation
competition, and the statistical boundary.

### Largest plants and cascades

CENACE 2022 Annual Report, PDF pages 31-32, reports annual gross production by
large hydro plant. Selected comparisons are:

| Plant | Model dispatch | CENACE gross production | Difference |
|---|---:|---:|---:|
| Coca Codo Sinclair | 7,282.97 | 6,828.15 | +454.82 |
| Paute Molino | 5,273.31 | 5,150.65 | +122.66 |
| Sopladora | 2,264.35 | 2,544.99 | -280.64 |
| San Francisco | 772.86 | 1,137.17 | -364.31 |
| Marcel Laniado | 699.41 | 1,025.45 | -326.04 |
| Minas San Francisco | 1,089.80 | 993.78 | +96.02 |
| Agoyan | 570.17 | 901.10 | -330.93 |
| Delsitanisagua | 641.94 | 755.17 | -113.23 |
| Mazar | 853.23 | 679.68 | +173.55 |
| Pucara | 297.08 | 221.84 | +75.24 |

Gross-versus-net definitions prevent these rows from being acceptance tests as
written, but they are useful profile diagnostics.

The official Mazar-Paute-Sopladora complex produced 8,375.32 GWh. The model
produces about 8,390.89 GWh across those three plants, a difference of only
15.57 GWh, while reallocating energy among the plants. This is strong evidence
that the cascade's aggregate annual water-energy magnitude is reasonable.

The model contains only independent `Generator` components: no reservoirs,
stores, storage units, water balances, cascade links, or upstream/downstream
coupling. CENACE identifies Mazar and Marcel Laniado as the most representative
reservoirs and describes the eastern wet season as April-September and dry
season as October-March (PDF pages 72-73). Students can later improve reservoir
operation, but Phase 06b should first provide an exploratory monthly comparison.

Recommended low-priority work:

1. extract monthly hydro production from CENACE or ARCONEL into a cited table;
2. compare monthly modeled potential, dispatch, and curtailment at aggregate and
   major-cascade level;
3. flag timing errors separately from transmission curtailment;
4. document the absence of reservoir/cascade physics;
5. do not add detailed reservoir optimization in Phase 06b unless required to
   explain a material remaining validation failure.

## 7. Resolving the 931.43 GWh accounting boundary

The Phase 06a residual is:

```text
28,684.62 CENACE net production plus imports
- 27,561.62 CENACE national demand control
-    191.57 exports
=    931.43 GWh
```

Profile or topology corrections cannot make this accounting residual disappear;
they change modeled dispatch, not the definitions of the source totals.

ARCONEL's official 2022 national balance now supplies the missing accounting
framework. It separately reports gross production, imports, energy delivered
for public service, transmission losses, energy available for public service,
exports, distribution availability, final consumption, and distribution
losses. Relevant published values include:

| ARCONEL quantity | GWh |
|---|---:|
| SNI gross production | 28,863.00 |
| Imports | 465.83 |
| Energy delivered for public service | 29,123.37 |
| Transmission losses | 1,292.73 |
| Energy available for public service | 27,830.63 |
| Exports | 192.14 |
| Available to distribution systems | 27,638.49 |

Source: ARCONEL 2022 statistics, national energy balance tables:
https://arconel.gob.ec/wp-content/plugins/download-monitor/download.php?force=0&id=273

The 931.43 GWh residual is not numerically equal to the 1,292.73 GWh official
transmission-loss value. The CENACE and ARCONEL tables also differ between net
and gross production and between block-transaction participants and the broader
SNI. Therefore Phase 06b must construct a formal source-boundary bridge rather
than insert a 931.43 GWh loss load.

Required reconciliation table:

1. source organization, table, page, year, and gross/net status;
2. included generator and autogenerator classes;
3. station service/auxiliary consumption;
4. imports and exports;
5. transmission input, output, and losses;
6. distribution deliveries and losses;
7. self-consumption and non-block transactions;
8. numerical residual after every documented bridge item.

Decision rule: only model a loss component after the bridge proves which loss
quantity belongs inside the PyPSA boundary. Because the current PyPSA linear
network is lossless, an accepted historical-loss representation could be a
source-backed zonal loss load or a documented scalar delivery efficiency. It
must remain separate from customer demand and exports. The target is source
equivalence, not merely reducing the residual to an arbitrary "acceptable"
percentage.

## 8. Copperplate diagnostic notebook

Add `notebooks/06a_diagnose_2022_validation.ipynb` before the annual validation
notebook. This is a diagnostic stage, not a second accepted model.

It should:

1. run preflight and load the same assembled candidate and validation contract;
2. compute annual and monthly potential, dispatch, curtailment, and capacity
   factors by plant and carrier;
3. produce a no-new-solve analytical merit-order diagnostic using aggregate
   hourly availability and sink profiles;
4. optionally build and solve a reproducible one-bus PyPSA copperplate network
   with the same generators, loads, import control, and snapshot weights;
5. compare copperplate and network dispatch by carrier and month;
6. identify generators curtailed while higher-cost generation or shedding is
   active elsewhere;
7. list binding lines and transformers, effective capacities, and hours binding;
8. report shedding by actual generator bus, not by inferred generator-name
   suffix;
9. save CSV/JSON diagnostics and figures under `results/phase_06a/` and
   `reports/figures/phase_06a/`;
10. never overwrite the solved Phase 06 network.

The notebook should normally consume an existing solved network. Its analytical
diagnostics must work without another 8,760-hour solve. The optional one-bus
solve is materially smaller than the transmission model, but remains a manual
scientific run rather than a CI test.

## 9. Validation criteria and anti-overfitting rule

Do not relax TVD or carrier-share thresholds during the source-correction work.
Reconsider them only after:

- the Santa Elena corridor is corrected and bus 58 shedding is explained;
- wind profiles pass order-of-magnitude and provenance checks;
- biomass/biogas availability reflects documented fuel seasonality;
- the accounting bridge establishes comparable totals;
- copperplate and network diagnostics separate temporal/profile error from
  transmission error.

If an official annual or monthly statistic is used directly as a model
constraint, it becomes a calibration input and may not simultaneously count as
independent validation. The report must label each observable as one of:

- calibration input;
- independent validation target;
- diagnostic-only evidence;
- unresolved/non-comparable.

The preferred validation outcome is that the system reproduces the historical
mix from defensible capacities, resource profiles, network limits, costs, and
operating constraints. Threshold changes are the last measure and require a
scientific justification independent of the observed model error.

## 10. Codex implementation plan for Phase 06b

### Work package A - evidence tables

1. Create a reviewed transmission-evidence CSV for line 47 with both official
   Chongon-Santa Elena circuits, source page, original units, conductor, rating,
   circuit status, bus mapping, confidence, and `official_fact`.
2. Extend the topology repair CSV using `add`, `remove`, or `update` rows; never
   hard-code the correction in a notebook.
3. Create normalized monthly source tables for ARCONEL biomass and available
   hydro/biogas observations. Preserve gross/net and boundary fields.
4. Add every new source and generated artifact to `config/artifacts.yaml` with
   size/checksum after review.

Acceptance: every numerical correction has a row-level locator and no source
table silently mixes CENACE net values with ARCONEL gross values.

### Work package B - topology and load attachment

1. Confirm buses 58 and 145 against the official SNT diagram.
2. Represent Chongon-Santa Elena C1 and C2 with official lengths and ratings.
3. Review `load_57`, `load_139`, `load_201`, and `load_282` and retain or revise
   their bus-58 attachment only through connection-evidence CSV rows.
4. Keep the 2022 La Concordia corridor at one in-service circuit unless new
   contemporaneous evidence proves otherwise.
5. Rebuild with notebook 04 and run the notebook 05 smoke test.

Acceptance: a short representative solve has no unexplained shedding at bus 58;
the structural fingerprint changes only as documented; La Concordia remains
source-consistent.

### Work package C - wind profile reconstruction

1. Quarantine `onwind.nc` as rejected-for-validation evidence without deleting
   it.
2. Locate or generate a reproducible site-level wind resource for Villonaco and
   Huascachaca, preferably from the existing full cutout if scientifically
   adequate.
3. Record weather year, turbine, hub height, interpolation, clipping, and
   commissioning masks.
4. Add tests for units, [0,1] bounds, nontrivial positive-hour coverage, annual
   full-load-hour sanity ranges, and explicit weather year.
5. Do not use CENACE 57 GWh as both a scaling target and validation observation.

Acceptance: Villonaco is within a defensible order of magnitude before any
validation comparison; the profile can be regenerated from recorded inputs.

### Work package D - bioenergy constraints

1. Encode the official ARCONEL monthly biomass shape in a normalized table.
2. Search for plant-level San Carlos, Ecoelectric, Ecudos, El Inga, and Pichacay
   production/fuel evidence; document negative search results.
3. Implement monthly energy budgets or monthly availability through reusable
   `src/` functions and configuration switches.
4. Keep biomass and biogas controls separate.
5. Add deterministic tests proving monthly and annual energy limits and future-
   scenario disablement.

Acceptance: bioenergy can no longer operate as unlimited year-round fuel, and
the historical constraint is removable for future scenarios.

### Work package E - accounting contract

1. Add a versioned CENACE-to-ARCONEL reconciliation table.
2. Extend validation output with gross/net basis, station service, transmission
   losses, self-consumption, exports, and unresolved residual.
3. Keep total-energy screening disabled until the reconciliation closes at a
   documented common boundary.
4. If a loss representation is approved, add it as an explicit component with
   separate reporting and tests; never rescale customer demand silently.

Acceptance: every term in the annual energy identity has one definition and one
source, and the remaining residual is explained rather than tolerated.

### Work package F - diagnostic notebook and rerun

1. Implement `06a_diagnose_2022_validation.ipynb` and small reusable diagnostic
   functions under `src/`.
2. Unit-test copperplate aggregation and congestion accounting with tiny
   networks; do not place annual solves in CI.
3. Run notebooks 03, 04, and 05 after approved input changes.
4. Run notebook 06a against the rebuilt candidate and/or existing solved result.
5. Only then manually run notebook 06 for the complete year.
6. Record before/after carrier metrics, monthly profiles, congestion, shedding,
   source-boundary reconciliation, checksums, and solver status.

Acceptance: the final report can attribute each remaining discrepancy to
resource/profile, temporal constraint, transmission, economics, or source
boundary without relying on unexplained residuals.

## 11. Priority order

| Priority | Action | Reason |
|---:|---|---|
| 1 | Correct and verify Chongon-Santa Elena | Direct official evidence; explains nearly all remaining shedding |
| 1 | Rebuild wind profiles | Current resource is demonstrably invalid by more than an order of magnitude |
| 1 | Add biomass seasonality | Largest non-conventional overproduction; official monthly data exists |
| 1 | Build accounting bridge | Required before total-energy validation has scientific meaning |
| 2 | Add notebook 06a copperplate/congestion diagnostics | Separates network and profile causes reproducibly |
| 2 | Verify La Concordia operating state | Current one-circuit model is supported; preserve unless contradicted |
| 3 | Monthly hydro comparison | Useful diagnostic after higher-impact corrections |
| 3 | Reservoir/cascade refinement | Important teaching extension, but aggregate major-cascade energy is already close |
| Last | Reconsider thresholds | Must not compensate for correctable data/model defects |

## Decision gate for implementation

Proceed with Phase 06b implementation for the Santa Elena corridor evidence,
wind-profile reconstruction, Ecuador-specific biomass seasonality, accounting
reconciliation, and notebook 06a diagnostics. Preserve the present La Concordia
one-circuit ratings unless better 2022 operating evidence is found. Treat hydro
reservoir/cascade enhancement as exploratory and lower priority. Do not perform
the final annual solve until the owner reviews the new evidence tables and the
short diagnostic/smoke results.
