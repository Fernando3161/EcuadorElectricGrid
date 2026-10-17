"""Tiny deterministic 24-hour smoke; no repository network inputs or downloads."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import pypsa

from src.diagnostics.copperplate import build_copperplate, main, solve_period


def fixture():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2022-01-01", periods=48, freq="h"))
    for carrier in ("AC", "hydro", "oil", "electricity", "export", "load_shedding", "interconnection_colombia"):
        n.add("Carrier", carrier)
    for bus in ("a", "b", "interconnection_colombia"):
        n.add("Bus", bus, carrier="AC")
    n.add("Line", "bottleneck", bus0="a", bus1="b", x=0.1, s_nom=1)
    n.add("Generator", "hydro", bus="a", carrier="hydro", p_nom=10,
          p_max_pu=pd.Series(0.5, index=n.snapshots), marginal_cost=1)
    n.add("Generator", "thermal", bus="b", carrier="oil", p_nom=10, marginal_cost=20)
    n.add("Generator", "shed", bus="b", carrier="load_shedding", p_nom=100, marginal_cost=1000)
    n.add("Generator", "colombia_import_supply", bus="interconnection_colombia",
          carrier="interconnection_colombia", p_nom=10, marginal_cost=10)
    n.add("Link", "import", bus0="interconnection_colombia", bus1="b",
          carrier="interconnection_colombia", p_nom=2, efficiency=1)
    n.add("Load", "demand", bus="b", carrier="electricity", p_set=8)
    n.add("Load", "export", bus="a", carrier="export", p_set=1)
    n.meta = {"provenance": {"source": "synthetic fixture"}}
    return n


class CopperplateTests(unittest.TestCase):
    def test_copy_preserves_profiles_capacities_carriers_and_source(self):
        n = fixture()
        before = n.generators.copy(deep=True)
        profiles = n.generators_t.p_max_pu.copy(deep=True)
        d, mapping = build_copperplate(n, n.snapshots[0])
        pd.testing.assert_frame_equal(n.generators, before)
        pd.testing.assert_frame_equal(n.generators_t.p_max_pu, profiles)
        # PyPSA copy normalizes integer widths (build_year int64 -> int32).
        pd.testing.assert_frame_equal(d.generators.drop(columns="bus"), before.drop(columns="bus"), check_dtype=False)
        pd.testing.assert_frame_equal(d.generators_t.p_max_pu, profiles.iloc[:24])
        self.assertEqual(len(n.lines), 1)
        self.assertEqual(len(d.lines), 0)
        self.assertEqual(mapping["b"], "a")
        self.assertEqual(mapping["interconnection_colombia"], "interconnection_colombia")
        self.assertEqual(d.links.loc["import", "p_nom"], 2)
        d.meta["provenance"]["source"] = "changed copy"
        self.assertEqual(n.meta["provenance"]["source"], "synthetic fixture")

    def test_24_hour_smoke_dispatch_import_limit_and_balance(self):
        n = fixture()
        d, _ = build_copperplate(n, n.snapshots[0])
        with TemporaryDirectory() as directory:
            log = Path(directory) / "highs.log"
            report = solve_period(d, log_path=log)
            self.assertTrue(log.is_file())
        self.assertEqual((report["solver_status"], report["termination_condition"]), ("ok", "optimal"))
        supply = report["generator_supply_mwh_by_raw_carrier"]
        self.assertAlmostEqual(supply["hydro"], 120)
        self.assertAlmostEqual(supply["interconnection_colombia"], 48)
        self.assertAlmostEqual(supply["oil"], 48)
        self.assertAlmostEqual(supply["load_shedding"], 0)
        self.assertAlmostEqual(report["balance_residual_mwh"], 0)
        self.assertEqual(report["load_withdrawal_mwh_by_raw_carrier"], {"electricity": 192, "export": 24})
        self.assertFalse(report["annual_comparable"])
        self.assertTrue(n.generators_t.p.empty)

    def test_transformer_removal_and_sign_aware_shedding(self):
        n = fixture()
        n.add("Bus", "island", carrier="AC")
        n.add("Transformer", "transformer", bus0="a", bus1="b", x=0.1, s_nom=1)
        n.generators.loc["shed", "sign"] = 1000.0
        n.generators.loc["shed", "marginal_cost"] = 1_000_000.0
        n.generators.loc["thermal", "p_nom"] = 0.0
        d, mapping = build_copperplate(n, n.snapshots[0])
        self.assertTrue(d.transformers.empty)
        self.assertEqual(mapping["island"], "a")
        report = solve_period(d)
        self.assertEqual(report["termination_condition"], "optimal")
        self.assertAlmostEqual(report["generator_supply_mwh_by_raw_carrier"]["load_shedding"], 48)
        self.assertAlmostEqual(report["balance_residual_mwh"], 0)
        self.assertEqual(len(n.transformers), 1)

    def test_reject_long_unavailable_or_weighted_period(self):
        n = fixture()
        for start, hours in ((n.snapshots[0], 8760), (n.snapshots[-1], 24), (n.snapshots[0], 0)):
            with self.assertRaises(ValueError):
                build_copperplate(n, start, hours)
        n.snapshot_weightings.loc[:, "generators"] = 365
        with self.assertRaisesRegex(ValueError, "unit hourly"):
            build_copperplate(n, n.snapshots[0])

    def test_reject_undefined_period_semantics(self):
        for component, name, kwargs in (
            ("Store", "store", {"bus": "a", "e_nom": 10}),
            ("GlobalConstraint", "annual", {"constant": 100}),
            ("Link", "internal", {"bus0": "a", "bus1": "b", "p_nom": 10}),
        ):
            n = fixture()
            n.add(component, name, **kwargs)
            with self.assertRaises(ValueError):
                build_copperplate(n, n.snapshots[0])

    def test_preflight_failure_blocks_network_loading(self):
        with patch("src.preflight.main", return_value=1), patch("pypsa.Network") as load:
            self.assertEqual(main(["--network", "unused.nc", "--start", "2022-01-01",
                                   "--period-import-policy", "capacity-only"]), 1)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
