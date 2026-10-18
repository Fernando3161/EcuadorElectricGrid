# Phase 06G — Hydro profile reassessment report

## Executive status

Phase 06G implementation and reporting infrastructure are complete, and the
planned runnable matrix has completed. The retained Phase 06F networks provide
a useful legacy diagnostic comparison. All new Phase 06G cases enforce the
configured 465.83 GWh annual import constraint and fixed 191.57 GWh exports,
use all 8,760 snapshots, and have `ok/optimal` solver metadata.

The legacy Phase 06F results must still be read as diagnostic evidence, not as
the Phase 06G validated baseline. Their solve path omitted the historical
import callback and produced approximately 1,917.27 GWh of imports rather than
the configured 465.83 GWh band.

## What was implemented

`src/phase06g.py` now provides:

- an experiment registry covering baseline, upper-bound, effective-capacity,
  run-of-river, reservoir, temporal, assignment, outage, technical, network,
  and accounting hypotheses;
- full-year PyPSA execution paths with 8,760-snapshot checks;
- source effective-capacity handling and isolated profile alternatives;
- corrected hydro/thermal/nonconventional/interconnection taxonomy;
- import-constraint callback wiring for new solves;
- per-plant hydro offer/dispatch diagnostics;
- carrier GWh, comparable-universe shares, percentage-point share errors,
  relative errors, and TVD outputs;
- experiment manifests, registries, checksums, CSV summaries, and seaborn
  figures under `results/phase_06g/`, `data/interim/hydro_reassessment_2022/`,
  and `reports/figures/phase_06g/`.

The six profile alternatives are intentionally labeled bounded sensitivities.
They are not production corrections because the underlying plant/river
evidence is not yet sufficient. Assignment, outage, and technical hypotheses
remain explicitly `not testable` until traceable evidence exists.

## Registry and experiment meaning

| ID | Meaning | Status |
|---|---|---|
| `G06G_BASE` | New baseline with historical import constraint | Started; not completed |
| `G06G_A100` | Matched all-hydro `p_max_pu=1` upper bound | Not accepted until constrained solve |
| `G06G_F06F_P45_LEGACY` | Retained Phase 06F P45 result | Reused legacy diagnostic |
| `G06G_CAP_EFFECTIVE` | Replace hydro nominal `p_nom` with source effective capacity | Requires completed constrained run |
| `G06G_ROR_Q90`, `G06G_ROR_Q100` | Run-of-river rating quantile sensitivities | Requires completed constrained runs |
| `G06G_RES_1M`, `G06G_RES_6M` | Reservoir smoothing-window sensitivities | Requires completed constrained runs |
| `G06G_TEMP_START` | Month-start rather than mid-month anchors | Requires completed constrained run |
| `G06G_ASSIGN` | Alternative plant-flow mapping | Not testable: no traceable alternative mapping |
| `G06G_OUTAGE` | Plant-level 2022 outage masks | Not testable: evidence is incomplete/aggregate |
| `G06G_TECH` | Flow/head/efficiency constraints | Not testable: plant parameters unavailable |
| `G06G_NET` | Retained-topology attribution | Diagnostic/output-only until matched annual result exists |
| `G06G_ACCT` | Accounting-boundary recomputation | Can use a verified full-year result; cannot change dispatch |

The old `F06F_A120` result is not validation. Its apparent annual hydro match
is an unsupported availability multiplier. `F06F_A100` remains an upper-bound
diagnostic only.

## Legacy diagnostic comparison

The exact Phase 06F P45 result reports 22,754.54 GWh hydro and 2,953.94 GWh
thermal after correcting the taxonomy so biomass/biogas are not counted as
thermal. The retained Phase 06F availability diagnostics are:

| Case | Hydro GWh | Thermal GWh | Interpretation |
|---|---:|---:|---|
| P45 | 22,754.54 | 2,953.94 | Legacy baseline; imports unconstrained |
| P00 | 22,767.02 | 2,953.84 | Hydro price diagnostic; immaterial change |
| A105 | 23,261.76 | 2,683.23 | Unsupported availability sensitivity |
| A110 | 23,712.92 | 2,368.73 | Unsupported availability sensitivity |
| A120 | 24,504.00 | 1,837.80 | Near annual hydro reference by coincidence; not validation |
| A130 | 25,167.17 | 1,436.26 | Unsupported availability sensitivity |
| A100 | 26,880.05 | 838.04 | All-hydro upper-bound diagnostic |

The CENACE hydro reference is 24,499.54 GWh. The monotonic response supports
the conclusion that the availability/profile chain is influential, but it does
not identify a physically defensible factor.

## Shares, TVD, and accounting basis

The comparable carrier universe is hydro, thermal, nonconventional, and
interconnection. Shares are calculated over the sum of those same four model
categories, excluding load shedding, exports, storage, curtailment, and network
losses from the carrier denominator. The reference denominator is 28,684.61
GWh, excluding the separately reported source-rounding residual.

For every completed run the report tables contain:

- weighted annual GWh;
- model and reference shares;
- signed and absolute share error in percentage points;
- signed and absolute GWh error;
- relative share/GWh error where defined;
- `TVD = 0.5 × Σ|model share − reference share|`;
- demand, imports, exports, load shedding, and estimated line-loss diagnostics.

The legacy P45 share comparison is not an acceptance result because its
interconnection supply is outside the configured historical band. Its hydro
share is materially below the reference share and its thermal share is above
the reference share. A120 can reduce hydro’s absolute GWh difference while
still changing the thermal and total carrier composition; this is why the
share/TVD view is more informative than hydro GWh alone.

The 931.43 GWh CENACE/model metering-boundary residual remains report-only and
is not inserted as guessed losses. The electrical balance must still be closed
on the model side for every accepted run.

## Figures

Generated figure paths include:

- `reports/figures/phase_06g/tvd_by_experiment.png`
- `reports/figures/phase_06g/carrier_shares_by_experiment.png`
- `reports/figures/phase_06g/share_error_pp_by_experiment.png`
- `reports/figures/phase_06g/hydro_thermal_tradeoff.png`

These figures are descriptive for the currently available legacy outputs and
must not be presented as evidence that the constrained Phase 06G matrix passed.
The expected scientific reading is a hydro/thermal trade-off: relaxing offers
raises hydro and reduces thermal, but does not establish that the relaxed offer
is physically supported.

## Limitations and next execution gate

The next run must resume with a single constrained baseline and verify before
any sensitivity:

1. solver status and termination are recorded as `ok`/`optimal`;
2. imports lie in the configured 465.83 GWh annual band;
3. all 8,760 snapshots and unit weights are present;
4. model-side energy balance closes;
5. carrier taxonomy excludes biomass/biogas from thermal;
6. profile isolation tests confirm ROR cases leave reservoir offers unchanged;
7. only then are the six new profile cases executed and compared.

No scientific acceptance claim is made in this report. The full runnable
matrix is complete, but the scientific acceptance gate remains open because
the nonconventional carrier screen and cross-boundary accounting issue remain
unresolved.

## Final execution results

The planned runnable matrix is now complete. Every new Phase 06G case below
used 8,760 snapshots, terminated with solver status `ok` and condition
`optimal`, enforced 465.83 GWh annual imports, and retained the fixed 191.57
GWh export sinks. The legacy P45 row is shown only for comparison because its
import constraint was not enforced.

| Experiment | Hydro GWh | Thermal GWh | Nonconventional GWh | Load shedding GWh | TVD |
|---|---:|---:|---:|---:|---:|
| `G06G_BASE` | 22,756.12 | 4,372.65 | 158.58 | 0.0078 | 0.03980 |
| `G06G_A100` | 26,414.21 | 838.04 | 35.10 | 0.0000 | 0.09820 |
| `G06G_CAP_EFFECTIVE` | 22,699.66 | 4,428.04 | 159.64 | 0.0106 | 0.04180 |
| `G06G_ROR_Q90` | 23,919.36 | 3,229.49 | 138.51 | 0.0000 | **0.00830** |
| `G06G_ROR_Q100` | 21,554.58 | 5,551.27 | 181.24 | 0.2721 | 0.08227 |
| `G06G_RES_1M` | 22,055.04 | 5,056.11 | 176.12 | 0.0918 | 0.06443 |
| `G06G_RES_6M` | 23,952.56 | 3,193.87 | 140.93 | 0.0000 | 0.00950 |
| `G06G_TEMP_START` | 22,706.55 | 4,406.19 | 174.61 | 0.0078 | 0.04101 |

The full machine-readable results are in
`results/phase_06g/experiment_summary.csv` and
`results/phase_06g/carrier_share_comparison.csv`. Each accepted network has a
run metadata file and SHA-256 checksum under its experiment directory.

## Final share findings

The reference shares are hydro 85.410%, thermal 11.830%, nonconventional
1.136%, and interconnection 1.624%. The two strongest share cases are:

| Experiment | Hydro error pp | Thermal error pp | Nonconventional error pp | Interconnection error pp | TVD |
|---|---:|---:|---:|---:|---:|
| `G06G_ROR_Q90` | +0.776 | −0.193 | −0.637 | +0.055 | 0.00830 |
| `G06G_RES_6M` | +0.896 | −0.322 | −0.628 | +0.055 | 0.00950 |

`G06G_ROR_Q90` is the best completed case by TVD and hydro/thermal shares.
`G06G_RES_6M` is a close second. Both still miss the nonconventional share
screen by more than 0.5 percentage points, so neither passes the complete
carrier acceptance gate.

The effective-capacity case does not improve the baseline: reducing hydro
`p_nom` to the source effective-capacity values lowers hydro generation and
raises thermal generation. The 1-month reservoir case is also unfavorable.
The 6-month reservoir sensitivity improves the mix substantially, but the
three-month smoothing rule is a modeling assumption and the six-month result
is not evidence of actual reservoir physics. The month-start temporal case is
close to the baseline and does not explain the discrepancy. The Q100
run-of-river case is materially worse than Q90, supporting a bounded
normalization sensitivity but not identifying the correct percentile.

## Final recommendations

1. **Do not promote Q90, Q100, RES-1M, RES-6M, or any multiplier to the
   production baseline.** The experiments establish model sensitivity, not
   physical correctness.
2. **Use `G06G_ROR_Q90` as the leading research diagnostic** and `G06G_RES_6M`
   as the leading reservoir-shape diagnostic for figures and further evidence
   collection. They are the best share fits, but both remain hypotheses.
3. **Prioritize primary evidence collection** for plant/basin flow series,
   turbine rated flow, head/efficiency, reservoir operation, and plant-level
   2022 availability. The current 65-plant mapping still contains 56 low-
   confidence regional proxies.
4. **Investigate the nonconventional shortfall separately.** Even the best
   hydro cases remain below the nonconventional reference share; changing hydro
   profiles alone cannot close that carrier gap.
5. **Retain the baseline as the supported model input** until owner approval is
   based on source evidence. The current results do not justify calling any
   network validated: the nonconventional share screen fails and the 931.43 GWh
   cross-boundary residual remains report-only.

## Completion statement

Phase 06G full-year execution is complete for all runnable hypotheses in the
registry. Assignment, outage, and flow/head/efficiency hypotheses remain
explicitly not testable because the repository lacks defensible plant-level
inputs. No topology was changed, no dispatch was forced, and no profile was
fit to the CENACE annual hydro total. The final decision is therefore:
**retain the current production baseline, use Q90 and RES6M only as labeled
diagnostic sensitivities, and obtain primary hydro/nonconventional evidence
before implementing a correction.**

## Combined candidate: ROR-Q90 plus RES-6M

The combined experiment `G06G_COMB_ROR_Q90_RES6M` applies Q90 normalization to
run-of-river plants and a six-month trailing flow mean to reservoir plants in
the same 8,760-hour solve. Imports remain 465.83 GWh and exports 191.57 GWh;
the solver terminated `ok/optimal`.

| Case | Hydro GWh | Thermal GWh | Nonconventional GWh | Load shedding GWh | TVD |
|---|---:|---:|---:|---:|---:|
| `G06G_ROR_Q90` | 23,919.36 | 3,229.49 | 138.51 | 0.0000 | **0.00830** |
| `G06G_RES_6M` | 23,952.56 | 3,193.87 | 140.93 | 0.0000 | 0.00950 |
| `G06G_COMB_ROR_Q90_RES6M` | 24,960.52 | 2,205.31 | 121.54 | 0.0000 | 0.04582 |

The combination does **not** improve the result. It overshoots the hydro
reference by 460.98 GWh, shifts too much of the comparable supply mix away
from thermal, and has a TVD more than five times the Q90-only case. The
individual sensitivities are therefore not additive: their simultaneous
profile changes interact through dispatch and network constraints. The combined
case should remain a diagnostic interaction experiment, not a recommended
hydro correction.

## Complete comparison figures

The final figure set is saved under `reports/figures/phase_06g/`:

- `annual_generation_by_experiment.png` — absolute hydro, thermal, and
  nonconventional generation by experiment;
- `carrier_shares_by_experiment.png` — model/reference carrier shares;
- `share_error_pp_by_experiment.png` — signed percentage-point share errors;
- `share_error_heatmap.png` — carrier-by-experiment share-error matrix;
- `tvd_by_experiment.png` — TVD ranking against the configured screen;
- `hydro_thermal_tradeoff.png` — hydro/thermal dispatch trade-off;
- `load_shedding_by_experiment.png` — annual shedding and the 0.01 GWh screen.

These plots support the decision directly: Q90-only and RES6M-only are the
closest share compositions, while their combination overcorrects hydro and
reduces thermal too far. None is promoted to production because the profile
parameters remain unsupported sensitivities and the nonconventional carrier
screen still fails.
