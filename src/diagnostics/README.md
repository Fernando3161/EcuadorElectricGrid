# Bounded copperplate diagnostic

Run from the repository root in the existing `pypsa-earth-ec` environment:

```powershell
conda run -n pypsa-earth-ec python -m unittest discover -s tests -p test_copperplate_diagnostic.py -v
conda run -n pypsa-earth-ec python -m src.diagnostics.copperplate --network data/processed/networks/network_base_filled.nc --start 2022-01-01 --hours 24 --period-import-policy capacity-only
```

The second command first runs the common `06_validation` environment/artifact
preflight **offline**. An unresolved input stops execution before network loading.
It never downloads or repairs files. The selected generated candidate is then
hashed separately; it is evidence, not an authoritative validated network.
No network is exported or overwritten. A unique directory under
`results/diagnostics/copperplate-*` retains the original-to-diagnostic bus map,
solver log, source checksum/provenance, software version and period summary.
Solver exceptions retain a summary with the error; unsuccessful termination
does not produce dispatch accounting or a successful exit status.

`build_copperplate` returns an in-memory copy for 1--168 consecutive actual
hourly snapshots, with unit weights only. All domestic AC buses, including
disconnected islands, map to the first domestic bus. Domestic Lines and
Transformers are removed. The original network, generator/load IDs, capacities,
carriers, signs, profiles, costs, exports and source metadata are preserved.
Colombia's external bus (`interconnection_colombia`) remains separate; the
reviewed two-port import Links retain their capacity and efficiency. See
`src/assembly.py::_add_colombia_interconnection` and
`config/validation_2022.yaml` for the existing interface boundary.

This intentionally relaxes domestic transmission and island restrictions.
It is not a proposed physical topology or a network repair. The caller must
explicitly select **capacity-only imports**: the annual historical import-energy
callback from `src/validation.py` is neither applied to 24 hours nor prorated.
This is a diagnostic assumption, not a change to the baseline contract. Storage,
shunts, investment, unit commitment, global constraints, active generator energy
budgets and other Link arrangements fail clearly pending explicit semantics.

The summary reports raw-carrier supply and withdrawals in MWh, including exports
and emergency load shedding as separate raw carriers. It accounts for PyPSA
component signs. Unit hourly weights give `MWh = sum(MW * 1 hour)`; Link losses
are `sum(p0 + p1)`, and the internal balance residual is supply minus withdrawals
minus Link losses. No annual extrapolation, carrier acceptance comparison, source
boundary residual adjustment or modification of the established TVD-based validation screens occurs.

The deterministic synthetic 24-hour smoke checks dispatch, import capacity,
profiles, source immutability, mapping, balance, sign-aware shedding and guards.
Its hand-computable fixture supplies 216 MWh: 120 hydro, 48 imports and 48 oil;
withdrawals are 192 demand and 24 exports. This tests code behavior, not Ecuador
scientific validity. A January day is an explicit short sample, not evidence of
annual or seasonal representativeness. Select a different start explicitly.

Inspection on 2026-10-18 found offline `06_validation` preflight blocked by
`renewable_availability_2022`: expected 3,158,162 bytes, found 3,157,404 bytes.
No Ecuador diagnostic solve was run and no manifest pins or input files were
changed. Owner reconciliation of that artifact is required before the second
command can proceed. Full-year solves and batch scenarios remain out of scope.
