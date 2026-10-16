import unittest
import pandas as pd

from src.validation import calculate_validation_metrics, screen_validation, validate_annual_coverage


class Phase06ValidationTests(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame({"carrier": ["hydro", "thermal", "imports"],
                                       "reference_gwh": [80.0, 15.0, 5.0], "in_scope": [True] * 3})

    def test_identical_distribution_has_zero_tvd(self):
        rows, metrics = calculate_validation_metrics(pd.Series({"hydro": 80, "thermal": 15, "imports": 5}), self.reference)
        self.assertEqual(metrics["tvd"], 0.0)
        self.assertTrue((rows.absolute_error_pp == 0).all())

    def test_order_does_not_change_metrics(self):
        a = calculate_validation_metrics(pd.Series({"hydro": 80, "thermal": 15, "imports": 5}), self.reference)[1]
        b = calculate_validation_metrics(pd.Series({"imports": 5, "hydro": 80, "thermal": 15}), self.reference)[1]
        self.assertEqual(a, b)

    def test_tvd_can_pass_while_carrier_screen_fails(self):
        rows, metrics = calculate_validation_metrics(pd.Series({"hydro": 81, "thermal": 14, "imports": 5}), self.reference)
        result = screen_validation(metrics, rows, {"max_tvd": 0.03, "max_hydro_error_pp": 0.5})
        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"]["max_tvd"])
        self.assertFalse(result["checks"]["max_hydro_error_pp"])

    def test_incomplete_year_is_not_comparable(self):
        comparable, reasons = validate_annual_coverage(pd.date_range("2022-01-01", periods=24, freq="h"))
        self.assertFalse(comparable)
        self.assertTrue(reasons)

    def test_zero_reference_relative_error_is_safe(self):
        reference = pd.DataFrame({"carrier": ["hydro", "zero"], "reference_gwh": [100.0, 0.0], "in_scope": [True, True]})
        rows, _ = calculate_validation_metrics(pd.Series({"hydro": 100, "zero": 0}), reference)
        self.assertTrue(pd.isna(rows.loc[rows.carrier == "zero", "relative_error"]).iloc[0])


if __name__ == "__main__":
    unittest.main()
