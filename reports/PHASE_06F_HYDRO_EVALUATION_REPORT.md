# Phase 06F — Hydro resource and dispatch evaluation

## Executive result

Phase 06F completed the registered full-year diagnostic matrix on the retained
Phase 06D baseline. All runs used 8,760 snapshots, returned solver status
`ok`/termination `optimal`, and wrote an experiment-specific solved network
and SHA-256 checksum under `results/phase_06f/<experiment_id>/`.

The baseline reproduction is `F06F_P45`: 22,754.54 GWh hydro versus the
CENACE reference of 24,499.54 GWh, a shortfall of 1,745.00 GWh. This is close
to, but not identical with, the earlier Phase 06D rounded evidence (22,755.84
GWh); the retained network checksum and exact run output are the Phase 06F
record.

The controlled price sweep does not explain the shortfall. Reducing hydro cost
from 45 to 0 USD/MWh changes hydro generation by only 12.48 GWh. Availability
relaxation has a much larger effect: the 1.20 diagnostic multiplier produces
24,504.00 GWh, but this is an unsupported calibration and is not proposed as a
correction. The phase therefore identifies resource/profile assumptions as a
major driver, while leaving the physically defensible correction unresolved.

## Reproducibility and outputs

The runner is `src/phase06f.py`. Baseline input metadata, plant lineage, offer
diagnostics, dispatch attribution, evidence, corrections, and the registry are
in `data/interim/hydro_evaluation_2022/`. The aggregate experiment table is
`results/phase_06f/experiment_summary.csv`; figures are saved under
`reports/figures/phase_06f/`.

The baseline network is the preserved Phase 06D artifact
`results/phase_06d/baseline/network_solved_2022.nc`; no Phase 06D file was
modified.

## Results

| Experiment | Change | Hydro GWh | Difference vs CENACE GWh | Thermal GWh |
|---|---|---:|---:|---:|
| F06F_P45 | baseline, 45 USD/MWh | 22,754.54 | 1,745.00 | 3,043.40 |
| F06F_P00 | hydro cost 0 USD/MWh | 22,767.02 | 1,732.52 | 3,043.27 |
| F06F_A105 | availability ×1.05, capped | 23,261.76 | 1,237.78 | 2,683.23 |
| F06F_A110 | availability ×1.10, capped | 23,712.92 | 786.62 | 2,368.73 |
| F06F_A120 | availability ×1.20, capped | 24,504.00 | -4.46 | 1,837.80 |
| F06F_A130 | availability ×1.30, capped | 25,167.17 | -667.63 | 1,436.26 |
| F06F_A100 | p_max_pu = 1 | 26,880.05 | -2,380.51 | 838.04 |

The price cases P20–P45 are practically identical. The low-price case is a
diagnostic lower bound, not evidence for changing hydro costs. The availability
cases demonstrate sensitivity, not validation: A120 happens to match the annual
reference but is explicitly retained as a diagnostic only.

## Input lineage and offer findings

All 65 hydro generators have rows in `hydro_input_lineage.csv`. The retained
offers are reconstructed as `p_nom × p_max_pu` and summarized in
`hydro_offer_diagnostics.csv`. The lineage confirms that the current profile
chain includes eight monthly legacy `caudals.csv` series and regional proxy
assignments. The primary provenance of those legacy flow series is unresolved;
this is the main evidence gap preventing an evidence-backed profile correction.

`hydro_dispatch_attribution.csv` compares hourly dispatch with hourly offered
power, quantifies available-energy utilization and potential curtailment, and
classifies plants as resource-limited or power-saturated. These classifications
are diagnostic and do not infer a physical reservoir model from dispatch alone.

## Hypothesis conclusions

* Resource/profile availability: supported as a material contributor by the
  monotonic A105–A130 response and the A100 upper-bound case. The exact factor
  is not identified by this experiment alone.
* Hydro marginal cost: not a primary contributor; the complete 0–45 USD/MWh
  sweep changes annual hydro by only about 0.05%.
* Transmission: not re-estimated in this phase; prior copperplate evidence
  shows a contribution but leaves a substantial residual. The Phase 06F
  availability experiments were run on the retained topology.
* Commissioning, plant attachment, nominal capacity, water value, and boundary
  reconciliation: not promoted to corrections. They require additional
  source-backed tests before encoding assumptions.

## Corrections and limitations

`hydro_corrections.csv` contains no proposed production correction. No profile
was multiplied into the validated model, no dispatch was forced, and no solved
NetCDF was edited manually. The CENACE annual reference and source limitations
are registered in `hydro_reference_evidence.csv`.

The phase does not pass the overall scientific acceptance gate: the accounting
boundary residual and other carrier validation issues remain unresolved, and
the diagnostic A120 result is not an approved physical correction. The next
owner decision is whether to obtain primary 2022 plant/river evidence or retain
the current hydro offers and document the unresolved discrepancy.

## Execution record

Executed in the approved `pypsa-earth-ec` environment:

```text
py -m src.phase06f
py -m src.phase06f --run-experiments
```

The first command regenerated the audit tables and figures. The second command
ran the full-year experiment registry. No packages were installed or upgraded.

## Complete experiment comparison

Metrics were recomputed from every saved full-year NetCDF using the four in-scope
carriers: hydro, thermal, nonconventional, and interconnection. The complete
row-level tables are `results/phase_06f/experiment_carrier_metrics.csv` and
`results/phase_06f/experiment_total_metrics.csv`.

| Experiment | Model total GWh | Reference total GWh | Absolute total error GWh | Relative total error % | TVD |
|---|---:|---:|---:|---:|---:|
| F06F_A100 | 27,753.190 | 28,684.610 | 931.420 | 3.247 | 0.114 |
| F06F_A105 | 27,753.190 | 28,684.610 | 931.420 | 3.247 | 0.048 |
| F06F_A110 | 27,753.190 | 28,684.610 | 931.420 | 3.247 | 0.043 |
| F06F_A120 | 27,753.190 | 28,684.610 | 931.420 | 3.247 | 0.062 |
| F06F_A130 | 27,753.190 | 28,684.610 | 931.420 | 3.247 | 0.077 |
| F06F_P00/P20/P25/P30/P35/P40/P45 | 27,753.182 | 28,684.610 | 931.428 | 3.247 | 0.053 |

CENACE carrier shares are hydro 85.41%, thermal 11.83%, nonconventional 1.14%,
and interconnection 1.62%. The model shares for each experiment are:

| Experiment | Hydro | Thermal | Nonconventional | Interconnection |
|---|---:|---:|---:|---:|
| F06F_A100 | 96.85 | 3.02 | 0.13 | 0.00 |
| F06F_A105 | 83.82 | 9.38 | 0.42 | 6.38 |
| F06F_A110 | 85.44 | 8.28 | 0.39 | 5.89 |
| F06F_A120 | 88.29 | 6.43 | 0.33 | 4.95 |
| F06F_A130 | 90.68 | 5.05 | 0.26 | 4.01 |
| F06F_P00/P20/P25/P30 | 82.03 | 10.64 | 0.41 | 6.91 |
| F06F_P35 | 82.01 | 10.64 | 0.44 | 6.91 |
| F06F_P40/P45 | 81.99 | 10.64 | 0.46 | 6.91 |

For each carrier, the detail CSV reports absolute GWh error, relative error
percentage, simulated/reference share, signed and absolute share error in
percentage points. The total metrics report total absolute/relative error and
TVD. Under the requested closest-share rule, **F06F_A110 is the winner** with
TVD 0.04295 and total absolute share error 8.589 percentage points. F06F_A105
is second at TVD 0.04754; the baseline F06F_P45 is 0.05284. F06F_A120 is
closest in absolute hydro GWh, but not in overall carrier shares. A110 remains
an unsupported diagnostic multiplier and is not an approved correction.
