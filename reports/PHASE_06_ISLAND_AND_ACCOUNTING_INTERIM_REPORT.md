# Phase 6 interim report: islands, interchange, and accounting boundary

Date: 2026-10-16

Status: diagnostic only. No network, input table, or notebook correction is
authorized by this report. Candidate repairs must be encoded as reviewed,
source-traceable rules and tested before the annual model is solved again.

## Executive conclusion

The annual modeled load is already correct at 27,561.62 GWh. The two material
islands are not evidence that their local plants are undersized. They are
topology omissions in the upstream `base.nc`:

1. buses 186/291 retain the La Concordia-Pedernales 138 kV feeder but omit La
   Concordia's sectioning connection into the Quininde-Santo Domingo corridor;
2. buses 159/279 retain most of the Minas San Francisco-La Union-San Ildefonso
   230 kV evacuation corridor but omit the short final tie from bus 279 to the
   co-located 230 kV side at bus 34.

Both conclusions are supported by official CELEC/CENACE evidence and by the
geometry and component structure of the model. Increasing the Pedernales or
Minas San Francisco plant capacity is not supported. The recommended next step
is to implement the two missing network connections as explicit Phase 5 repair
rules, test them on the unsolved network, and perform a short solve before a new
annual run.

The 2022 import volume should be a historical-mode annual constraint, removable
for future scenarios. Exports should be explicit border sinks, separated by
country. The 931.43 GWh residual must not yet be represented as a loss load:
CENACE describes the demand series as being measured at generation terminals
and its figure labels national demand as including SNT losses. The residual's
definition remains unresolved and requires a source-table reconciliation.

## 1. Reproduced energy balance

The solved result contains 26,636.856 GWh of actual generation/import supply and
924.764 GWh of load shedding. Their sum is exactly 27,561.620 GWh, so demand
preparation and annual snapshot weighting close correctly.

| Quantity | GWh |
|---|---:|
| Simulated generation plus imports | 26,636.856 |
| Load shedding | 924.764 |
| Modeled demand | 27,561.620 |

The load shedding is concentrated at four buses:

| Bus | Load shedding (GWh) | Component |
|---|---:|---|
| 291 | 816.652 | 186-291 island |
| 58 | 61.127 | main SNI component |
| 159 | 46.613 | 159-279 island |
| 181 | 0.373 | main SNI component |

The remaining 61.50 GWh on the main component is a separate congestion or
local-delivery problem and should be diagnosed after the two islands are fixed.

## 2. Buses 186 and 291: La Concordia-Pedernales

### Model evidence

| Item | Bus 186 | Bus 291 |
|---|---:|---:|
| Voltage | 138 kV | 138 kV |
| Longitude | -79.4962 | -79.9195 |
| Latitude | -0.0596 | 0.0176 |
| Existing modeled connection | line 151 to bus 291 | line 151 to bus 186 |

The component has 860.452 GWh of annual load but only the 5 MW Pedernales oil
plant, which produces 43.8 GWh. The resulting 816.652 GWh of shedding is therefore
structural, not evidence that the plant should be enlarged.

The Phase 4 fleet records the Pedernales plant at 5 MW from the official Master
Plan generation inventory. There is no evidence that it was intended to supply
the entire northern Manabi demand island. It is attached near bus 291 because of
its location, not because bus 291 is intended to be electrically isolated.

### Official network evidence

CELEC states that the La Concordia-Pedernales system was energized on 25
September 2020 and comprises an 80 km, 138 kV line (insulated for 230 kV), the
Pedernales 138/69 kV substation, and the La Concordia sectioning substation.
CENACE's 2020 annual report lists the line and both substations as incorporated
into the SNI. The Master Plan transmission chapter lists:

- La Concordia-Pedernales: 138 kV, 79.70 km, 202.57 MVA, one circuit;
- La Concordia-Quininde: 138 kV, 41.40 km, 113 MVA, one circuit;
- La Concordia as a sectioning point on the Santo Domingo-Esmeraldas 138 kV
  corridor.

Sources:

- CELEC, "Sistema de transmision La Concordia-Pedernales energizado",
  https://www.celec.gob.ec/gensur/noticias/celec-ep-energizo-el-nuevo-sistema-de-transmision-la-concordia-pedernales/
- CENACE, Informe Anual 2020, Tables 3.2-3.3, page 104,
  https://www.cenace.gob.ec/wp-content/uploads/downloads/2021/04/Informe-Anual-CENACE-2020-Parte-1.pdf
- PME 2023-2032, Chapter 5, Tables 5-7 and 5-12 and section 5.9.4.10,
  `literature/PlanMaestroElectrifEc/cap_5_transmision.pdf`.

### Bus identification and proposed repair

The coordinate/topology match is strong:

- bus 291 corresponds to the Pedernales end;
- bus 186 corresponds to La Concordia;
- bus 62 is on the Esmeraldas side and is consistent with Quininde;
- bus 86 is on the Santo Domingo side;
- existing line 149 directly connects buses 62 and 86, bypassing bus 186.

This suggests the upstream extraction retained the old direct corridor and the
new Pedernales feeder, but failed to section line 149 at La Concordia. The
scientifically supported repair is not a nearest-bus connector. It is to replace
the direct 62-86 representation with two segments, 62-186 and 186-86, while
retaining 186-291. Exact line parameters and whether one or both original
circuits are sectioned must be taken from the official SNT data; the report does
not authorize copying arbitrary nearest-line parameters.

Confidence: high for the physical connection; medium for the exact model bus
name mapping until it is recorded in a reviewed evidence table.

## 3. Buses 159 and 279: Minas San Francisco-San Ildefonso

### Model evidence

| Item | Bus 159 | Bus 279 |
|---|---:|---:|
| Voltage | 230 kV | 230 kV |
| Longitude | -79.6002 | -79.7595 |
| Latitude | -3.3087 | -3.2083 |
| Existing modeled connection | line 105 to bus 279 | line 105 to bus 159 |

The component has 911.746 GWh of annual load and the 270 MW Minas San Francisco
plant. It produces 865.133 GWh under the assigned availability profile, leaving
46.613 GWh unserved. The 270 MW capacity agrees with CENACE and CELEC: three 90
MW units. Increasing capacity is therefore unsupported.

CELEC states that Minas San Francisco exports at 230 kV over a 42 km line to San
Ildefonso and from there to the SNI. Later official project documents refer to
the existing corridor as Minas San Francisco-La Union-San Ildefonso. This is
consistent with model line 159-279 representing the first part of the corridor.

Bus 34 is the strongest candidate for the omitted SNI terminal:

- it is 230 kV and only 8.81 km from bus 279;
- it is connected by an existing transformer to co-located bus 33;
- it has no 230 kV transmission line of its own;
- adding the missing 279-34 segment produces the expected continuous
  Minas-La Union-San Ildefonso evacuation path.

Sources:

- CELEC SUR, "Central Hidroelectrica Minas San Francisco",
  https://www.celec.gob.ec/celecsur/informacion-tecnica/central-hidroelectrica-minas-san-francisco/
- CELEC project evidence describing the existing Minas San Francisco-La
  Union-San Ildefonso line and its future sectioning at Pasaje,
  https://www.celec.gob.ec/transelectric/wp-content/uploads/2024/10/BOLETIN-DE-ACLARACIONES-No.-1.pdf
- CENACE Informe Anual 2022, Table 2.3, pages 72-73,
  `literature/CENACE/cenace_informe_anual_2022_parte_1.pdf`.

The proposed repair is one reviewed 230 kV connection from bus 279 to bus 34,
with parameters derived from official corridor evidence or the established
repository line-parameter method. A map/page review should confirm bus 34 as San
Ildefonso before implementation.

Confidence: high that the island requires the San Ildefonso SNI tie; medium-high
that model bus 34 is the correct terminal.

## 4. Imports: historical control versus future scenarios

The 1,121.127 GWh optimized import is not evidence that the price is simply
wrong. A 500 MW power limit permits up to 4,380 GWh annually, and cost
minimization will import whenever doing so is cheaper than domestic marginal
generation or required to avoid load shedding.

For the historical 2022 run, use two independent limits:

1. a physical power-transfer limit based on documented interconnector capacity;
2. an annual historical energy band centered on 465.83 GWh, with the agreed 5%
   data tolerance if the owner retains that tolerance for this input control.

The annual constraint must be explicitly mode-specific. Future scenarios should
disable the historical energy band while retaining scenario-appropriate physical
capacity and prices. A suggested configuration structure is:

```yaml
interconnection:
  mode: historical_2022  # or future_market
  historical_energy_gwh: 465.83
  historical_energy_tolerance_fraction: 0.05
  enforce_historical_energy_band: true
```

The present single northern interface is also a simplification. CENACE reports
four 230 kV circuits from Pimampiro to Jamondino and one 138 kV circuit from
Tulcan to Panamericana. A later implementation should attach imports to those
documented interfaces rather than merely the geometrically northernmost bus.

## 4A. Test of the removed-lower-voltage-bridge hypothesis

The proposed failure mode is technically plausible in general: if a connected
path has retained high-voltage endpoints but intermediate 48/69 kV buses, simply
deleting all lower-voltage buses can leave the retained endpoints as artificial
islands. A safe topology reduction must detect such paths before deletion and
either contract an evidence-supported electrical interface or retain the
necessary lower-voltage path and transformers.

For buses 186/291 and 159/279, however, this is not what happened in the current
repository. Read-only graph checks were performed on every relevant network
artifact that preserves the original bus identifiers:

| Artifact | Buses below 138 kV | Component containing 186/291 | Component containing 159/279 |
|---|---:|---:|---:|
| `data/raw/networks/base.nc` | 202 | 2 buses | 2 buses |
| `data/raw/networks/elec.nc` | 202 | 2 buses | 2 buses |
| `data/processed/networks/network_original.nc` | 202 | 2 buses | 2 buses |
| `data/processed/networks/_alt/network_snapped.nc` | 202 | 2 buses | 2 buses |
| `data/processed/networks/_alt/network_base.nc` | 185 | 2 buses | 2 buses |
| `data/processed/networks/_alt/network_2018.nc` | 185 | 2 buses | 2 buses |
| `data/processed/networks/ec_network_2022.nc` | 0 | 2 buses | 2 buses |

Thus both islands already exist in the 313-bus upstream network before the
Phase 5 voltage filter removes any 48/69 kV bus. There is no modeled path of any
voltage from either pair to the main component, and therefore no deleted A-level
bridge to restore from these NetCDF files.

This does not disprove that lower-voltage infrastructure exists physically. For
example, Pedernales has an official 138/69 kV transformer and downstream 69 kV
network. It establishes only that the missing SNI ties cannot be recovered by
undoing the repository's voltage pruning. They must be reconstructed from
official transmission evidence. In both material cases, that evidence identifies
missing 138/230 kV transmission connections, not a distribution-voltage bridge:

- La Concordia sections the 138 kV Santo Domingo-Esmeraldas corridor and feeds
  Pedernales at 138 kV;
- Minas San Francisco evacuates to San Ildefonso at 230 kV.

The lower-voltage-bridge risk remains relevant to the other Phase 5 load
attachments. It should become a general regression check rather than the repair
mechanism for these two islands.

## 5. Export sinks

CENACE reports 191.57 GWh of exports in 2022, split as approximately:

- Colombia: 159.15 GWh;
- Peru: 32.42 GWh.

Historical exports should be represented as separately labeled fixed border
loads or links, never merged into national customer demand. The Colombian sink
belongs at the documented Pimampiro/Jamondino and/or Tulcan/Panamericana
interfaces. The Peruvian sink needs a separately documented southern transfer
point consistent with the 2022 cross-border arrangement.

If hourly observations are unavailable, monthly values visible in CENACE's TIE
figures are preferable to a flat annual profile. Any interpolation must be
recorded as a modeling assumption. Future-scenario export behavior should be
configurable and should not inherit the fixed 2022 export profile automatically.

## 6. The unresolved 931.43 GWh residual

The arithmetic residual is:

```text
28,684.62 net production plus imports
- 27,561.62 national demand
-    191.57 exports
=    931.43 GWh
```

It is tempting to label this residual as transmission losses or station service,
but current source evidence does not justify doing so. CENACE calls 28,684.62
GWh "produccion neta" for plants participating in block transactions. It also
reports demand at generation terminals and labels the demand curve as national
demand including SNT losses plus exports. These definitions mean that adding a
new 931.43 GWh loss load now risks double-counting losses.

Required follow-up before modeling the residual:

1. reconcile CENACE pages 23, 38-40, 73-74, and 85 at a common metering boundary;
2. determine whether the production table and demand table cover the same set of
   generators/autoproducers and transactions;
3. identify auxiliary consumption, unallocated generation, and any source-table
   reconciliation terms;
4. only then choose between explicit loss loads, lossy network formulation, or a
   reported non-modeled residual.

Until resolved, the annual total-supply comparison must be `NOT_COMPARABLE` or
`PARTIAL`, even though the carrier-mix diagnostics remain informative.

## 7. Validation taxonomy correction

The final official-category comparison should aggregate:

- hydro;
- thermal;
- non-conventional = wind + biomass + biogas + solar;
- interconnection.

Threshold keys must map to actual row names. A missing category match must not
silently pass. Metric calculation should remain independent of screening.

## 8. Recommended implementation sequence

1. Add a pre-pruning connectivity audit that evaluates the complete multi-voltage
   graph. For every retained 138/230/500 kV component, search for paths to the
   main component whose internal nodes are 48/69 kV. Record the path, voltage
   transitions, transformers, and whether contraction is supported. Fail if a
   path disappears silently during voltage filtering.
2. Record that this audit finds no recoverable lower-voltage path for 186/291 or
   159/279 in any available upstream artifact. Preserve the negative result as
   provenance so these islands are not later attributed to Phase 5 pruning.
3. Create reviewed repair evidence for La Concordia-Pedernales. Confirm the bus
   identification on the official map, then model La Concordia as the sectioning
   point by replacing/breaking the direct 62-86 corridor representation into
   62-186 and 186-86 while retaining 186-291. Preserve the original line and
   replacement rows in the mutation audit.
4. Create reviewed repair evidence for Minas San Francisco-La Union-San
   Ildefonso. Confirm bus 34 as the 230 kV San Ildefonso terminal, then add the
   missing 279-34 segment. Do not change the official 270 MW plant capacity.
5. Derive voltage, circuit count, conductor/capacity, and lengths from official
   evidence. Where a parameter remains unavailable, label the value as a modeling
   assumption with confidence; do not use a generic nearest-bus connector.
6. Add deterministic topology tests requiring buses 186, 291, 159, and 279 to be
   in the main SNI component after repair. Also assert that the number of
   components and structural fingerprint change only as reviewed.
7. Rebuild the unsolved candidate and run consistency checks plus a short
   representative solve. Confirm that shedding at buses 291 and 159 disappears
   and inspect line loading around both repairs.
8. Diagnose any remaining main-component shedding, especially buses 58 and 181.
   This must be treated as congestion/local-delivery analysis, not automatically
   solved by adding generation.
9. Update the repair table, mutation audit, topology component table, topology
   figures, Phase 5 report, artifact checksum, and notebook narrative before the
   annual rerun.
10. Replace the single generic Colombia interface with documented border buses or
   explicitly retain it as a declared simplification.
11. Add the historical import-energy band behind a scenario/configuration switch.
12. Add separately labeled Colombia and Peru export sinks from source-backed
   annual/monthly data.
13. Resolve the 931.43 GWh accounting residual before adding any loss component.
14. Correct category aggregation and threshold lookup.
15. Rebuild and rerun the full annual model only after the short solve has no
    unexplained material shedding.

## Decision gate

The evidence supports proceeding with topology-repair implementation for both
islands. It does not support increasing either plant capacity. It supports a
historical import-energy constraint that is disabled for future scenarios, and
explicit country-specific export sinks. It does not yet support encoding 931.43
GWh as losses.

## Phase 6a implementation status

Implemented on branch `v1-pahse-06a-2022-valiidation-fixes` as source-traceable,
CSV-driven repairs. The La Concordia and Minas-San Ildefonso corrections are in
`data/interim/network/topology_repairs_2022.csv`; no plant capacity was changed.
Notebook 04 now emits the pre-pruning lower-voltage bridge audit, notebook 05
rebuilds the corrected candidate with separate export loads and documented
Colombia interfaces, and notebook 06 applies the historical import-energy band
and the four-category validation screen.

The audit also identifies an unrelated retained component (`120|161|75`) whose
only upstream path to the main graph crosses buses `119|112|162|18` below 138 kV.
It remains deliberately disconnected: no reviewed evidence currently supports
contracting that path into a high-voltage equivalent. This finding is written to
the audit CSV so voltage pruning cannot hide it. The 931.43 GWh accounting
residual likewise remains report-only.
