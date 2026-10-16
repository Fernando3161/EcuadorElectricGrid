# Ecuador 2022 topology and assembly method

## Boundary

The candidate represents mainland Ecuador's Sistema Nacional Interconectado at
138, 230, and 500 kV. The upstream 48 and 69 kV buses are outside the supported
transmission boundary. Demand and generator locations below that boundary are
aggregated to retained buses; the model does not claim to reproduce the
distribution grid.

The voltage choice reconstructs the threshold used by the legacy 2022 assembly
(`v_nom >= 137 kV`) while naming its actual retained levels explicitly. One
upstream 145 kV bus and line are snapped to 138 kV; six 46 kV buses and five
46 kV lines are snapped to 48 kV before lower-voltage removal.

## Reproducible mutations

Stage 04 starts from the checksum-pinned `base.nc`. It removes 18 buses that have
no component reference, removes all out-of-scope buses and incident branches,
then applies `data/interim/network/topology_repairs_2022.csv`.

That table preserves five corridor additions from the legacy notebook. The
available evidence is the notebook rationale, so the repairs are classified as
low/medium-confidence modeling decisions rather than official asset facts. Line
length is recomputed from endpoint coordinates; the reviewed legacy conductor
type, parallel count, and rating determine the electrical parameters.

All 19 retained transformers have zero upstream series resistance and 0.10 p.u.
reactance. Stage 04 sets resistance to 0.01 p.u. as an explicit numerical
assumption and retains the reactance. This prevents a hidden singularity repair
but is not equipment-level validation.

The candidate topology contains one 103-bus main component and three small
components of 3, 2, and 2 buses. No unsupported line is added to hide these
islands. Phase 6 must report their role in feasibility and any load shedding.

Stage 04 saves a combined voltage-coded topology map and separate bus-labeled
maps for the 500, 230, and 138 kV layers. They reuse the project's established
blue/brown/dark-blue palette and GADM provincial-boundary presentation. The GADM
4.1 GeoPackage is checksum-pinned in the common artifact manifest.

## Attachments

Stage 05 uses source coordinates from the upstream grid for all 251 Stage 1 load
columns. A load at a retained bus keeps that bus; otherwise it is assigned to the
nearest retained bus by great-circle distance. This preserves the complete
27,561.62 GWh demand total instead of deleting lower-voltage demand. The audit
records each source/target pair and distance. The longest load aggregation is
106.62 km and requires scientific review.

Generators without plant-specific connection evidence are assigned from their
qualified coordinates to the nearest retained bus. Reviewed connection evidence
overrides this fallback. Termoesmeraldas I and II use representative mapped
facility coordinates and are explicitly assigned to bus 42, the 138 kV side of
the modeled S/E Esmeraldas group. Bus 43 is the co-located 230 kV side and is
linked to bus 42 by `transf_18_1`; it is therefore not used merely because it is
slightly closer geographically. CELEC documents Esmeraldas I's 13.8/138 kV unit
transformer and the substation's 138 kV SNI connection, while pre-2022 CELEC /
TRANSELECTRIC evidence identifies the dedicated 138 kV bays. The attachment
audit retains the source and locator. Remaining nearest-bus assignments are
transparent modeling aggregations, not physical connection evidence.

Stage 05 also performs a non-mutating materiality screen. Generator bands of
<=1, >1-10, >10-50, >50-100, and >100 MW allow respectively 100, 75, 50, 35,
and 25 km. Annual-load bands of <=25, >25-100, >100-250, >250-500, and >500
GWh/year use the same descending limits. These are project QA conventions, not
engineering interconnection standards. Rows beyond their limit are labeled
`not acceptable - fix this bus`; 1 generator row (Villonaco, 16.50 MW) and 12
load rows (2,492.96 GWh/year) are currently flagged. The screening tables preserve the
selected buses and provide the review list for a later evidence-backed repair.

Hydro, solar, and wind receive the complete Phase 4 hourly availability tables.
Other generator availability remains unconstrained at one. Carrier-level cost
rows are retained modeling assumptions for Phase 6 review. Nuclear rows from the
legacy carrier table are not imported.

One high-cost `load_shedding` generator is added per retained bus so Phase 6 can
measure infeasibility explicitly. These generators are not supply capacity and
must always be excluded from official carrier comparisons.

## Candidate status

`network_base_filled.nc` is an unsolved generated product with a run-specific
checksum sidecar. Stage 05 checks component identifiers, structural fingerprint,
demand, availability, and serialization. It does not establish optimization success, energy balance,
or agreement with 2022 statistics; those remain Phase 6 acceptance criteria.
