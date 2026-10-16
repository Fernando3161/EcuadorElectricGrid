# Phase 6 technical report — 2022 validation implementation

## Scope

This phase implements the requested multi-criteria historical benchmark against
CENACE's 2022 supply statistics and adds a Colombia import interface at the
northernmost retained SNI bus. It does not claim that the annual scientific
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

`src.assembly._add_colombia_interconnection` creates an external bus, a
500 MW supply generator, and a 500 MW link to the northernmost retained bus.
The marginal cost is 45.20 USD/MWh, derived from CENACE Informe Anual 2022
Figure 2.30's 4.52 cUSD/kWh average Colombia import price. The rating is a
clearly labeled modeling assumption, not an official interconnector rating.

## Verification and handoff

The student-facing workflow is `notebooks/06_run_and_validate_2022_full_year.ipynb`.
It is restartable, starts with stage-06 preflight, writes the requested annual
CSV/JSON artifacts and three figures, and records the solved-network checksum.

The focused Phase 6 unit tests pass. The existing Phase 5 assembly suite could
not be used as a clean integration signal in the current shell because the
loaded PyPSA version is 1.1.2 while the repository artifact was written by
PyPSA 0.30.3; it fails before the new interconnection logic at the pre-existing
`Network.madd` call. Re-run in the approved `pypsa-earth-ec` environment.

Manual handoff: execute the preflighted Phase 6 notebook for the complete 8,760
hour network, record solver termination, energy balance, load shedding,
curtailment, imports/exports, validation CSV/JSON/report, plots, and checksums.
Do not label the network validated until those outputs satisfy the documented
criteria and are reviewed.
