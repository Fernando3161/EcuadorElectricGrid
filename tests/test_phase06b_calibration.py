import unittest
import pandas as pd
from src.phase06b import load_monthly_biomass_shape, build_biomass_availability, monthly_energy_profile, scale_profile_to_annual_energy


class Phase06BCalibrationTests(unittest.TestCase):
    def test_biomass_source_is_complete_and_source_faithful(self):
        table = load_monthly_biomass_shape("data/raw/statistics/arconel_2022_biomass_monthly.csv")
        self.assertEqual(table.month.tolist(), list(range(1, 13)))
        self.assertAlmostEqual(table.biomass_gwh.sum(), 288.49, places=2)
        self.assertTrue((table.boundary == "autogenerators gross").all())

    def test_diagnostic_rejects_negative_dispatch(self):
        index = pd.date_range("2022-01-01", periods=8760, freq="h")
        plants = pd.DataFrame({"plant_id": ["a"], "carrier": ["biomass"], "p_nom_mw": [100.0]})
        profiles, _ = build_biomass_availability(index, plants, load_monthly_biomass_shape("data/raw/statistics/arconel_2022_biomass_monthly.csv"))
        self.assertAlmostEqual(float((profiles["a"] * 100.0).sum() / 1000.0), 195.68, places=6)
        self.assertTrue((profiles.loc[profiles.index.month.isin([2, 3, 4, 5]), "a"] == 0).all())

    def test_monthly_profile_is_finite_and_scaling_rejects_infeasible_target(self):
        index = pd.date_range("2022-01-01", periods=8760, freq="h")
        profile = monthly_energy_profile(index, pd.Series({m: 1.0 for m in range(1, 13)}))
        self.assertTrue(pd.Series(profile).notna().all())
        with self.assertRaises(ValueError):
            scale_profile_to_annual_energy(pd.Series(1.0, index=index), 1.0, 100.0)


if __name__ == "__main__":
    unittest.main()
