# Phase 06E Historical Transmission Capacity Audit

## Executive conclusion

Phase 06E completed the reproducible two-sided audit of the retained Phase 06D
network artifact. The model-side inventory contains 115 Lines and 19
Transformers; every modeled component appears exactly once in its audit table.
The official source review registered and checksum-verified the CENACE 2022
annual report and the CELEC EP TRANSELECTRIC December 2022 SNT map. These
sources establish system-level 2022 context, but the available pages do not
provide a component-by-component MVA/current rating table that can defensibly
match the 134 modeled branches.

Accordingly, no capacity correction was applied. The correction registry is
empty by design, all modeled components are marked `UNMATCHED` pending a
component-level technical source, and the Phase 06D network was not modified.
This is an audit result, not evidence that the modeled ratings are correct.

The requested renewable and biomass fitting is outside the Phase 06E scope,
which freezes generation, profiles, demand, costs, endpoints, topology, and
impedances. The existing Phase 06D renewable-calibration network is shown as a
diagnostic “after” artifact, but it is not a Phase 06E correction and does not
meet the governing TVD/percentage-point screens. No solved dispatch was forced to match CENACE.

## Scope, artifacts, and provenance

The audit uses the Phase 06D unrestricted transmission-constrained baseline as
the scientific comparison baseline. The model inventory was extracted from
`results/phase_06d/renewable-calibration/network_solved_2022.nc`, checksum
`eb231577e3fe73a3493d494f65fdb8187bb30c73687a6013b323a6e6e4ea98ff`, because
that is the retained Phase 06D candidate containing the renewable diagnostic.
The original baseline remains at
`results/phase_06d/baseline/network_solved_2022.nc` and was not overwritten.

The mandatory preflight notebook ran successfully in the existing
`pypsa-earth-ec` environment. No package was installed or upgraded.

| Source | Use | Limitation |
|---|---|---|
| `literature/CENACE/cenace_informe_anual_2022_parte_1.pdf` | CENACE 2022 generation and annual statistics, Table 2.4, pp. 73-74 | No individual transmission-line or transformer ratings in the reviewed table |
| `literature/CELEC_TRANSELECTRIC/mapa_snt_diciembre_2022.pdf` | 2022 SNT system map and aggregate circuit/substation context | Map provides aggregate lengths and counts, not component MVA/current ratings |
| `data/raw/statistics/cenace_2022_net_supply.csv` | Source-faithful CENACE carrier reference | Statistical reference, not technical transmission evidence |

The complete source register, SHA-256 values, coverage statements, and network
checksums are in `data/interim/topology_audit_2022/source_register.csv`.

## Documentary extraction and inventory result

The CENACE report was inspected for the annual statistical tables and
transmission-relevant system descriptions. Table 2.4 is a generation/import
table, not an asset inventory. The CELEC SNT map is the strongest local 2022
topology source found during this run. It documents 66 substations and
aggregate 138/230/500 kV circuit lengths, but does not identify a rating for
each modeled line or transformer. No missing rating was replaced with a
voltage-based estimate.

The official inventory files therefore contain explicit documentary-scope
records with `NOT_DOCUMENTED` technical values. They are not presented as a
complete physical asset list. This limitation is itself a quality-gate result:
additional CELEC parameter/limit tables, one-line diagrams, equipment data
sheets, or an owner-supplied historical asset register are required before a
verified capacity correction can be proposed.

## Independent PyPSA inventory and reconciliation

The model-side inventory is independent of the prior audit artifacts and was
read directly from the solved Phase 06D network. It records component ID,
endpoints, endpoint voltage levels, `s_nom`, `s_max_pu`, effective rating,
parallel representation, source network, and source checksum.

| Component | Modeled count | Unique audited IDs | Documentary match status |
|---|---:|---:|---|
| Lines | 115 | 115 | 4 documented matches; 111 geographic/name evaluations pending rating source |
| Transformers | 19 | 19 | 19 geographic/name evaluations pending component-level source |

The audit tables contain no duplicate modeled IDs and no omitted modeled
components. Every line now has an explicit capacity-audit decision in
`data/interim/topology_audit_2022/line_capacity_audit.csv`.

The full row-level records are in:

- `data/interim/topology_audit_2022/lines_audit.csv`
- `data/interim/topology_audit_2022/transformers_audit.csv`
- `data/interim/topology_audit_2022/official_lines_inventory.csv`
- `data/interim/topology_audit_2022/official_transformers_inventory.csv`

## Geographic and name reconciliation

The modeled buses contain longitude, latitude, voltage, and stable bus IDs. Each
line and transformer was therefore expanded into an endpoint-level geographic
record. Three matching methods were applied in order:

1. exact unordered endpoint match against the existing project evidence table
   `data/interim/network/transmission_evidence_2022.csv`;
2. voltage-compatible nearest named substations from
   `data/raw/networks/EC_bus_expansion_masterplan_2022.csv`; and
3. unresolved candidate status when neither method supplied a defensible name.

The result is in `data/interim/topology_audit_2022/geographic_match_candidates.csv`.
It contains coordinates, endpoint distances, candidate names, source locators,
length/rating deltas, method, confidence, and interpretation for all 134
components.

| Geographic result | Lines | Transformers |
|---|---:|---:|
| Exact official project-evidence match | 2 | 0 |
| Geographic/name candidate only | 113 | 19 |

The two exact matches are the two Chongón–Santa Elena circuits represented in
the model as `repair_line_47_58_chongon_santa_elena_c1` and `_c2`. The evidence
records 40.0 km and 230 MVA for each circuit; the modeled records are identical
in length and rating, with zero discrepancy. They remain model-evidence matches,
not newly discovered corrections.

The remaining rows now have useful geographic candidate names and distances,
but are not promoted to official matches where the source is a future Master
Plan table or where no component-level rating evidence exists. This distinction
prevents a nearest-name inference from being mistaken for a verified 2022 asset.

## Capacity corrections and modified components

All 115 lines were evaluated. Four lines have component-level documentary
capacity evidence and all four agree exactly with the modeled capacity. The
remaining 111 lines have been geographically/name evaluated but lack an
authoritative component rating, so the safe decision is to retain the modeled
capacity pending a rating source. No line or transformer was modified in this
run because no supported discrepancy was found. Consequently:

- `data/interim/topology_audit_2022/capacity_corrections.csv` has the required
  schema and zero proposed rows;
- there is no component with a changed endpoint, `s_nom`, `s_max_pu`, or
  transformer rating;
- no network NetCDF was patched directly;
- no new bus, line, transformer, voltage level, generator, or connection was
  introduced.

The complete line decision summary is:

| Line evaluation | Count |
|---|---:|
| Documented capacity match; retain | 4 |
| Geographic/name evaluation; retain pending authoritative rating | 111 |
| Documented capacity discrepancy requiring correction | 0 |

This is different from “not audited”: every line was audited. The unresolved
status is specifically the absence of a defensible component-level rating, not
an omission from the evaluation.

## Generation comparison before and after diagnostic

The following comparison uses the CENACE carrier taxonomy implemented by the
project. “Before” is the Phase 06D baseline validation CSV. “After” is the
existing Phase 06D renewable-calibration candidate, retained as a diagnostic
comparison only; it contains no Phase 06E topology correction.

| Carrier | CENACE GWh | Before GWh | Before abs. error | After GWh | After abs. error |
|---|---:|---:|---:|---:|---:|
| Hydro | 24,499.54 | 22,756.118 | 7.11% | 22,755.844 | 7.12% |
| Thermal | 3,393.35 | 4,349.359 | 28.19% | 4,348.170 | 28.14% |
| Nonconventional | 325.89 | 160.047 | 50.89% | 160.047 | 50.89% |
| Interconnection | 465.83 | 489.122 | 5.00% | 489.122 | 5.00% |

The renewable candidate does not fit wind or biomass to their individual CENACE
quantities. The source rows are wind 57.00 GWh, biomass 195.68 GWh, biogas
40.06 GWh, and solar 33.15 GWh. The candidate’s aggregate nonconventional
generation remains 160.047 GWh versus 325.89 GWh. This failing result is
reported, not hidden by excluding carriers or changing the acceptance gate.

The detailed machine-readable comparison is
`data/interim/topology_audit_2022/generation_before_after_cenace.csv` and the
figures are:

- `reports/figures/phase_06e/generation_before_after_cenace.png`
- `reports/figures/phase_06e/carrier_error_before_after.png`

## Full error-indicator evaluation

The governing project criteria use aggregate carrier screens: TVD 3 percentage
points, hydro 3 percentage points, thermal 2 percentage points,
nonconventional 0.5 percentage points, and interconnection 0.5 percentage
points. The total-energy pass/fail criterion remains disabled because the
931.43 GWh source-boundary residual has not been resolved. The separate
Relative energy errors remain descriptive diagnostics only; they are not
acceptance screens.

| Indicator | Before | After diagnostic | Interpretation |
|---|---:|---:|---|
| TVD | 0.039801 | 0.039758 | Both exceed the 0.03 provisional screen |
| Hydro absolute pp error | 3.416 | 3.416 | Fails 3 pp screen |
| Thermal absolute pp error | 3.837 | 3.837 | Fails 2 pp screen |
| Nonconventional absolute pp error | 0.559 | 0.559 | Fails 0.5 pp screen |
| Interconnection absolute pp error | 0.138 | 0.138 | Passes the 0.5 pp screen |
| Load shedding | 0.007819 GWh | 0.007819 GWh | Below the 0.01 GWh configured maximum |
| Annual numerical balance | approximately 2.5e-10 GWh residual | approximately 2.5e-10 GWh residual | Numerically closed for the saved candidate |
| Modeled storage | 0 | 0 | No storage flow or SOC change in this candidate |
| Modeled branch/link losses | 0 | 0 | No modeled network-loss quantity; not a claim that physical losses are zero |
| Statistical boundary residual | 931.43 GWh | 931.43 GWh | Report-only; not relabeled as losses |

The “after” changes are therefore not evidence that transmission-capacity
correction improved the scientific validation. Because no Phase 06E component
was changed, the topology-specific before/after difference is zero by design.

## Renewable and biomass fitting request

The requested fitting cannot be accepted as a Phase 06E result. The phase
boundary freezes generation and profile inputs, and the current evidence shows
that fitting solved dispatch directly would be circular. In particular, the
existing wind scaling implementation used a pre-commissioning-mask denominator
in its candidate calculation, and biomass potential was already approximately
195.68 GWh while dispatch was only about 83.48 GWh. The gap therefore requires
diagnosis of availability, economics, temporal shape, curtailment, and network
constraints rather than an automatic target override.

A separately approved calibration package must first correct the masked wind
denominator, retain physical 2022 profile provenance, preserve biomass and
biogas as separate diagnostics, fit availability rather than dispatch, and
report potential, dispatch, curtailment, capacity factor, and error indicators.
It must then be evaluated against the unchanged TVD and percentage-point screens. Until that work
is separately approved and run, this report records the failing quantities
honestly.

## Quality gates and open evidence

Passed:

- mandatory Notebook 00 preflight in the approved environment;
- model inventory completeness: 115/115 Lines and 19/19 Transformers;
- unique modeled component IDs;
- preserved source and network checksums;
- no topology or capacity mutation during Phase 06E;
- generation/error figures saved under the Phase 06E figure directory.

Not passed or not assessable:

- complete official component-level rating inventory;
- verified documentary match for any modeled branch;
- source-backed capacity correction registry;
- carrier validation gate for hydro, thermal, and nonconventional generation;
- total-energy criterion, pending the 931.43 GWh boundary resolution.

The next evidence request is a historical CELEC EP TRANSELECTRIC equipment or
operating-limits register with line/circuit identifiers, endpoint substations,
voltage, conductor/current or MVA ratings, transformer-bank units and ratings,
and exact 2022 operational status. No future planned asset should be admitted
to the existing-2022 inventory without documentary operational evidence.

## Commands and execution record

Executed:

```text
pypsa-earth-ec python -m src.notebook_runner notebooks/00_preflight_and_data_inventory.ipynb --output results/phase_06e/notebooks/00_preflight_and_data_inventory.executed.ipynb --timeout 1200
pypsa-earth-ec python -m src.phase06e_audit
```

The full 00-06A notebook chain was not re-executed because the Phase 06E audit
produced no approved capacity correction to integrate. A full corrected-network
rerun remains a later owner-approved work package. No packages were installed,
no source PDFs were edited, and the Phase 06D baseline was preserved.
