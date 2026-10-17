import unittest
import pandas as pd
import pypsa

from src.validation import (calculate_validation_metrics, enforce_historical_import_energy,
                            screen_validation, validate_annual_coverage)


class Phase06ValidationTests(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame({"carrier": ["hydro", "thermal", "nonconventional", "interconnection"],
                                       "reference_gwh": [80.0, 14.0, 1.0, 5.0], "in_scope": [True] * 4})

    def test_identical_distribution_has_zero_tvd(self):
        rows, metrics = calculate_validation_metrics(pd.Series({"hydro": 80, "thermal": 14, "nonconventional": 1, "interconnection": 5}), self.reference)
        self.assertEqual(metrics["tvd"], 0.0)
        self.assertTrue((rows.absolute_error_pp == 0).all())

    def test_order_does_not_change_metrics(self):
        a = calculate_validation_metrics(pd.Series({"hydro": 80, "thermal": 14, "nonconventional": 1, "interconnection": 5}), self.reference)[1]
        b = calculate_validation_metrics(pd.Series({"interconnection": 5, "nonconventional": 1, "hydro": 80, "thermal": 14}), self.reference)[1]
        self.assertEqual(a, b)

    def test_tvd_can_pass_while_carrier_screen_fails(self):
        rows, metrics = calculate_validation_metrics(pd.Series({"hydro": 81, "thermal": 13, "nonconventional": 1, "interconnection": 5}), self.reference)
        result = screen_validation(metrics, rows, {"max_tvd": 0.03, "max_hydro_error_pp": 0.5})
        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"]["max_tvd"])
        self.assertFalse(result["checks"]["max_hydro_error_pp"])

    def test_missing_screen_category_cannot_silently_pass(self):
        rows, metrics = calculate_validation_metrics(pd.Series({"hydro": 80, "thermal": 14, "nonconventional": 1, "interconnection": 5}), self.reference)
        result = screen_validation(metrics, rows, {"max_unknown_error_pp": 99})
        self.assertFalse(result["passed"])

    def test_incomplete_year_is_not_comparable(self):
        comparable, reasons = validate_annual_coverage(pd.date_range("2022-01-01", periods=24, freq="h"))
        self.assertFalse(comparable)
        self.assertTrue(reasons)

    def test_zero_reference_relative_error_is_safe(self):
        reference = pd.DataFrame({"carrier": ["hydro", "zero"], "reference_gwh": [100.0, 0.0], "in_scope": [True, True]})
        rows, _ = calculate_validation_metrics(pd.Series({"hydro": 100, "zero": 0}), reference)
        self.assertTrue(pd.isna(rows.loc[rows.carrier == "zero", "relative_error"]).iloc[0])

    def test_historical_import_energy_band_is_enforced(self):
        network = pypsa.Network()
        network.set_snapshots(pd.date_range("2022-01-01", periods=2, freq="h"))
        network.add("Bus", "b")
        network.add("Load", "demand", bus="b", p_set=1.0)
        network.add("Generator", "colombia_import_supply", bus="b", p_nom=1.0, marginal_cost=1.0)
        network.add("Generator", "domestic", bus="b", p_nom=2.0, marginal_cost=10.0)
        contract = {"interconnection_colombia": {
            "enforce_historical_energy_band": True,
            "historical_energy_gwh": 0.001,
            "historical_energy_tolerance_fraction": 0.0,
        }}
        status, condition = network.optimize(
            solver_name="highs",
            extra_functionality=lambda n, snapshots: enforce_historical_import_energy(
                n, snapshots, contract
            ),
        )
        self.assertEqual((status, condition), ("ok", "optimal"))
        self.assertAlmostEqual(network.generators_t.p["colombia_import_supply"].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
