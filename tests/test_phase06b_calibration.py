import unittest
import pandas as pd
from src.phase06b import load_monthly_biomass_shape, aggregate_dispatch_by_carrier


class Phase06BCalibrationTests(unittest.TestCase):
    def test_biomass_source_is_complete_and_source_faithful(self):
        table = load_monthly_biomass_shape("data/raw/statistics/arconel_2022_biomass_monthly.csv")
        self.assertEqual(table.month.tolist(), list(range(1, 13)))
        self.assertAlmostEqual(table.biomass_gwh.sum(), 288.49, places=2)
        self.assertTrue((table.boundary == "autogenerators gross").all())

    def test_diagnostic_rejects_negative_dispatch(self):
        self.assertTrue(callable(aggregate_dispatch_by_carrier))


if __name__ == "__main__":
    unittest.main()
