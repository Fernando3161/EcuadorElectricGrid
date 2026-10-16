# 2022 statistical validation contract

Phase 6 uses a multi-criteria historical benchmark. It replaces the retired
uniform 5% carrier rule; the criteria below are provisional engineering screens,
not official CENACE tolerances and not calibration targets.

The comparable set is the exhaustive normalized CENACE set: hydro, thermal,
non-conventional, and interconnection where available in the source table. The
implementation reports annual GWh, derived shares, signed/absolute GWh error,
percentage-point error, relative error, total annual energy error, and total
variation distance (TVD). A result is `NOT_COMPARABLE` if the calendar year,
annual hourly coverage, accounting boundary, weighting, or carrier mapping is
not demonstrably compatible. Otherwise the configurable screen reports `PASS`
or `FAIL`; `PARTIAL` is reserved for a technically comparable run with an
incomplete diagnostic set.

Configured exploratory screens are: total energy relative error <= 5%, TVD <=
3 percentage points, hydro <= 3 pp, thermal <= 2 pp, non-conventional <= 0.5 pp,
and interconnection <= 0.5 pp. Metrics are calculated independently from these
thresholds, so changing a screen cannot alter the underlying benchmark.

The Colombia import interface is attached to the northernmost retained SNI bus.
It has a 500 MW modeling-assumption maximum and a 45.20 USD/MWh marginal cost,
from CENACE's 2022 average Colombia import price of 4.52 cUSD/kWh (Figure 2.30).
The price source is recorded in `config/validation_2022.yaml`; the capacity is
not presented as an official interconnector rating.

## Purpose

This document fixes the comparison rules before the 2022 model is calibrated or
solved. The machine-readable contract is `config/validation_2022.yaml`; this page
explains the scientific choices for students and reviewers.

## System boundary

The baseline represents mainland Ecuador's Sistema Nacional Interconectado (SNI).
It includes generators participating in CENACE block transactions and electricity
imports from Colombia and Peru. It excludes Galapagos and other isolated systems,
as well as explicit distribution-grid topology.

The statistical basis is net electricity delivered to the SNI. This is the
closest counterpart to PyPSA generator dispatch at network buses. International
imports are an in-scope supply category. Exports are a sink in the energy balance,
not a generation carrier.

The later annual energy balance must report, separately:

- national demand;
- domestic generation by validation carrier;
- imports and exports;
- storage charging, discharging, and state-of-charge change;
- network losses;
- curtailment;
- load shedding; and
- the final numerical residual.

## Official sources

The primary supply source is CENACE's
[Informe Anual 2022 - Parte 1](../literature/CENACE/cenace_informe_anual_2022_parte_1.pdf)
([official publication](https://www.cenace.gob.ec/wp-content/uploads/downloads/2023/04/Parte-1-Informe-Anual-2022.pdf)),
especially Table 2.4 on pages 73-74. The report covers CENACE's SNI operating
boundary and publishes net supply by generation type and international
interconnection.

Monthly 2022 demand energy and peak values are taken from Table 1.8 on page 42 of
CENACE's
[Informe Anual 2023 - Parte 1](../literature/CENACE/cenace_informe_anual_2023_parte_1.pdf)
([official publication](https://www.cenace.gob.ec/wp-content/uploads/downloads/2024/04/Parte-1-Informe-Anual-CENACE-2023.pdf)),
which republishes the prior year's monthly values at two-decimal precision. The
2022 report independently confirms annual national demand of 27,561.62 GWh and a
maximum of 4,388.06 MW.

The Master Plan Chapter 3 demand series remains planning evidence. Its rounded
2022 monthly values sum to 28,180 GWh while its displayed annual total is
28,181 GWh. That series also uses a different stated demand boundary. It is not
silently substituted for the internally consistent CENACE operational baseline.

## Demand method

The raw CENACE monthly values sum to 27,561.64 GWh because each month is displayed
to 0.01 GWh. One common factor scales all twelve monthly energy values to the
published annual control of 27,561.62 GWh. This changes no relative monthly
weight. Monthly peak values are preserved exactly.

The hourly shape and bus shares come from a 2013 PyPSA-Earth 0.7.0 load profile.
Forensic comparison found exact numerical identity between the tracked CSV and
`loads_t.p_set` in its source PyPSA-Earth network. The source network records
PyPSA-Earth commit `4bfb13da0f24a5fc98f541babe00dfd72f7ba3f8`, SSP2-2.6,
weather year 2013, and prediction year 2030. These values are a modeling template,
not observed 2022 load measurements.

For each month, the national historical shape receives one affine transformation
that makes its mean energy and maximum equal the official controls. Each hour's
historical bus shares are then applied to that transformed national value. The
output is in MW, which is the PyPSA load convention.

The accepted time index is the complete 2022 calendar year: 8,760 hourly,
timezone-naive timestamps representing Ecuador mainland local standard time
(UTC-05:00). Ecuador does not use daylight-saving time. Leap-day differences are
rejected rather than interpolated implicitly.

## Carrier taxonomy

The source table remains at CENACE's published detail. The validation table uses
the smallest aggregation that the current model can represent without pretending
to know plant-level fuels that are not encoded.

| Validation carrier | Official source rows | Reference (GWh) |
|---|---|---:|
| `hydro` | Hydroelectric | 24,499.54 |
| `wind` | Wind | 57.00 |
| `bioenergy` | Biomass + biogas | 235.74 |
| `solar` | Photovoltaic | 33.15 |
| `thermal_nonrenewable` | Diesel + fuel oil + natural gas + naphtha + residue | 3,393.35 |
| `imports` | Colombia + Peru imports | 465.83 |

The displayed component rows total 28,684.61 GWh. CENACE reports 28,684.62 GWh;
the 0.01 GWh display-rounding residual is retained as an explicit, out-of-gate row.

The legacy model carrier `oil` maps to `thermal_nonrenewable`. This mapping does
not claim that all thermal generation burns oil. Phase 4 should preserve the
official fuel detail where the plant data supports it, while the acceptance gate
remains stable at the declared aggregate level. Biomass and biogas map to
`bioenergy` because the current assembly already combines them.

Storage discharge is not primary generation and cannot be counted as a supply
carrier. Charging, discharging, and annual state-of-charge change belong in the
energy balance. Load shedding is never counted as valid supply.

## Five percent rule

For every positive in-scope reference carrier:

```text
relative_error_percent =
    100 * abs(simulated_gwh - reference_gwh) / reference_gwh
```

A carrier passes when `relative_error_percent <= 5.0`. The same rule applies to
small positive carriers such as solar and wind. A missing simulated carrier is
treated as zero and therefore fails a positive reference.

The official in-scope table has no zero reference. If a future version introduces
one, relative error is undefined and the absolute simulated value must be no more
than 0.01 GWh. This zero-only numerical rule is not a relaxed tolerance for small
positive carriers.

The model passes the carrier gate only when every in-scope row passes. The complete
scientific acceptance gate in `AGENTS.md` additionally requires a successful
optimization, explicit energy balance, recorded exclusions/residual, checksums,
and provenance. A carrier table alone cannot validate the network.
