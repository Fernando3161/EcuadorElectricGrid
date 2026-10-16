# Phase 6 validation criteria

The historical benchmark is a reproducible screening experiment, not a claim
that CENACE prescribes these tolerances. The reference is CENACE's 2022 annual
statistics and the model must first pass comparability checks: year, complete
annual snapshots, snapshot weighting, boundary, accounting basis, and an
exhaustive carrier mapping.

For the comparable carrier set, the workflow reports annual energy in GWh,
normalizes model and reference shares, and calculates total-energy relative
error, signed and absolute percentage-point errors, carrier relative errors, and
TVD = 0.5 * sum(abs(model share - reference share)). Missing or unmapped
categories are warnings/errors, not silently discarded. Load shedding,
curtailment, storage flows, exports, losses, and residuals are reported in the
energy balance and are not counted as actual generation.

The provisional screens in `config/validation_2022.yaml` are independently
configurable: total energy 5%, TVD 3 pp, hydro 3 pp, thermal 2 pp,
non-conventional 0.5 pp, and interconnection 0.5 pp. The final status is
`NOT_COMPARABLE`, `PARTIAL`, `PASS`, or `FAIL`, with each failed criterion named.
No threshold may be changed to make a run pass, and no model input may be tuned
solely for this benchmark.
