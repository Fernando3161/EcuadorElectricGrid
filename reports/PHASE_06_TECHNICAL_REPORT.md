# Phase 6 technical report — 2022 validation implementation

## Scope

This phase implements the requested multi-criteria historical benchmark against
CENACE's 2022 supply statistics and adds Colombia import interfaces at the
modeled Pimampiro 230 kV and Tulcan 138 kV buses. It does not claim that the annual scientific
acceptance gate has passed: the full-year solve remains an owner-authorized
manual run.

## Validation method

The workflow first checks annual calendar coverage and comparability. It then
calculates reference/model GWh, normalized shares, signed and absolute
percentage-point errors, relative errors, total annual energy relative error,
and TVD. Thresholds are applied afterward from `config/validation_2022.yaml`.
The configured values are provisional engineering screens, not official CENACE
tolerances and not calibration targets. The criteria and status vocabulary are
documented in `docs/PHASE_06_VALIDATION_CRITERIA.md`.

The legacy uniform 5% carrier helper remains only for backwards-compatible
evidence and is not the Phase 6 acceptance mechanism.

## Colombia interconnection

`src.assembly._add_colombia_interconnection` creates an external market bus, a
500 MW supply generator, and links to the documented Pimampiro and Tulcan interfaces.
The marginal cost is 45.20 USD/MWh, derived from CENACE Informe Anual 2022
Figure 2.30's 4.52 cUSD/kWh average Colombia import price. The rating is a
clearly labeled modeling assumption, not an official interconnector rating. The
historical solve constrains annual imports to 465.83 GWh +/-5%; future scenarios
disable this historical energy band. Colombia and Peru exports are separate
fixed sinks totaling 191.57 GWh.

## Phase 6a topology corrections

The traceable topology repair table now restores the official La Concordia
sectioning and Pedernales feeder and completes the Minas San Francisco-La
Union-San Ildefonso 230 kV evacuation path. The repairs connect the former
186/291 and 159/279 islands to the main SNI without changing plant capacities.
The full upstream multi-voltage audit confirms these islands existed before
48/69 kV pruning. The repaired topology has a 107-bus main component and one
remaining three-bus component.

## Verification and handoff

The student-facing rebuild sequence is notebooks 04, 05, and 06. Notebook 04
regenerates the topology and writes the mutation, component, and pre-pruning
low-voltage-bridge audits. Notebook 05 assembles the candidate with domestic
demand, country-specific exports, and the two Colombia import interfaces.
Notebook 06 is restartable, starts with stage-06 preflight, writes the requested annual
CSV/JSON artifacts and three figures, and records the solved-network checksum.

The Phase 5/6 focused suite passes in the approved `pypsa-earth-ec` environment,
including topology fingerprints, repaired connectivity, export accounting,
four-category screening, and a tiny HiGHS solve proving the import-energy band.

The currently present files under `results/phase_06/` were generated before the
Phase 6a topology and accounting changes. They are retained only as diagnostic
evidence and must be overwritten by the manual rerun before review.

Manual handoff: execute notebooks 04 and 05, inspect their topology and smoke-test
outputs, then execute notebook 06 for the complete 8,760
hour network, record solver termination, energy balance, load shedding,
curtailment, imports/exports, validation CSV/JSON/report, plots, and checksums.
Do not label the network validated until those outputs satisfy the documented
criteria and are reviewed.
