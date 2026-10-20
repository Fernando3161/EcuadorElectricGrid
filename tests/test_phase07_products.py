"""Real small-table checks for the published adjustment, not a future-grid solve."""
import hashlib
import json
import unittest

import pandas as pd

from src.artifacts import ROOT, load_manifest, required_artifact_ids
from src.planning import PlanningValidationError, filter_published_by_year, validate_published_portfolio, select_exercise_by_year
from src.planning_products import build_products, decimal_es, grid_kind, location_for, query_inventory


class AdjustmentProductsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projects, cls.tables, cls.exercise, cls.metadata = build_products()

    def test_all_source_records_reconcile_and_citations_are_complete(self):
        self.assertEqual(123, len(self.projects))
        self.assertEqual(123, self.projects.project_id.nunique())
        self.assertEqual(50, len(self.tables["generation"]))
        self.assertTrue(validate_published_portfolio(self.tables).empty)
        for source, frame in self.projects.groupby("source_artifact"):
            self.assertEqual({hashlib.sha256((ROOT / source).read_bytes()).hexdigest()}, set(frame.source_sha256))
        self.assertFalse(self.projects.official_name.str.contains("nuclear", case=False).any())
        self.assertFalse(self.projects.official_name.str.startswith("Proyecto / Descripción").any())

    def test_quantity_arithmetic_and_reported_discrepancy_preserved(self):
        self.assertAlmostEqual(8818.75, self.tables["generation"].normalized_value.sum())
        self.assertAlmostEqual(48759.01, self.tables["generation"].energy_gwh_year.sum())
        totals = pd.read_csv(ROOT / "results/phase_07/source_total_reconciliation.csv")
        self.assertAlmostEqual(1346.01, totals.loc[totals.comparison == "generation_mean_energy", "difference"].iloc[0])
        self.assertEqual(1200.0, decimal_es("1.200,00"))
        checks = pd.read_csv(ROOT / "results/phase_07/generation_source_checks.csv")
        flagged = checks.loc[checks.energy_above_nameplate_8760h]
        self.assertEqual(["PME26_A_049"], flagged.project_id.tolist())
        self.assertEqual(2058.0, flagged.iloc[0].energy_gwh_year)

    def test_uniform_delay_all_assets_and_half_year_preserved(self):
        self.assertTrue((self.exercise.exercise_year - self.exercise.published_year == 7).all())
        self.assertEqual(2030, self.exercise.exercise_year.min())
        self.assertEqual(2039, self.exercise.exercise_year.max())
        self.assertEqual("1er sem 2033", self.exercise.set_index("project_id").loc["PME26_B_002", "exercise_timing"])
        self.assertTrue(self.exercise.scenario_asset_id.str.startswith("EX_REPEAT_").all())
        self.assertEqual(2023, self.tables["generation"].commissioning_year.min())

    def test_queries_distinguish_published_and_delayed_cutoffs(self):
        original = filter_published_by_year(self.tables, 2023)
        delayed = filter_published_by_year(self.tables, 2030, delay_years=7)
        self.assertEqual(3, len(original["generation"]))
        self.assertEqual(original["generation"].project_id.tolist(), delayed["generation"].project_id.tolist())
        with self.assertRaises(PlanningValidationError):
            select_exercise_by_year(self.tables, 2039, repeat_mode=True)
        self.assertEqual(123, len(query_inventory(self.projects, 2039, delay_years=7)))
        self.assertEqual(3, len(query_inventory(self.projects, 2023)))

    def test_aliases_identify_known_baseline_overlap_without_asserting_unknown_absence(self):
        exercise = self.exercise.set_index("project_id")
        huascachaca = exercise.loc["PME26_A_002"]
        self.assertTrue(huascachaca.baseline_overlap)
        self.assertIn("Huascachaca", huascachaca.baseline_asset_id)
        self.assertTrue(self.exercise.loc[self.exercise.overlap_status == "not_established", "baseline_overlap"].isna().all())

    def test_unknown_electrical_inputs_are_not_fabricated(self):
        self.assertFalse(self.exercise.model_ready.any())
        self.assertTrue(self.projects.latitude.isna().all())
        self.assertTrue(self.projects.endpoint0_id.isna().all())
        self.assertEqual(2, len(self.tables["transformers"]))
        self.assertEqual({225.0}, set(self.tables["transformers"].normalized_value))
        self.assertEqual("transformer", grid_kind("Subestación Quevedo 230/69 kV, 225 MVA, reposición autotransformador"))
        self.assertEqual("reactive", grid_kind("Subestación Nueva Prosperina 230 kV, 2x60 MVAR"))
        collectora = self.projects.set_index("project_id").loc["PME26_C_005"]
        self.assertEqual(600.0, collectora.normalized_value)
        self.assertEqual("MVA", collectora.normalized_unit)

    def test_location_assumptions_require_context_and_uniqueness(self):
        cantons = {("loja", "loja"): [(-4.0, -79.2, "fixture.csv", "Loja")]}
        result = location_for("Unknown", "Loja", {}, cantons, {}, province="Loja")
        self.assertEqual("unique_canton", result["coordinate_method"])
        self.assertEqual("modeling_assumption", result["coordinate_evidence_class"])
        self.assertEqual("unresolved", location_for("Unknown", "Loja", {}, cantons, {}, province="Other")["coordinate_method"])
        self.assertEqual("unresolved", location_for("Unknown", "Por definir", {}, {}, {})["coordinate_method"])

    def test_manifest_covers_every_source_and_lookup(self):
        manifest = load_manifest()
        required = set(required_artifact_ids(manifest, "07_planning"))
        self.assertTrue({"planning_adjustment_pdf", "planning_adjustment_generation_source", "planning_adjustment_grid_source", "planning_exercise_config", "planning_adjustment_totals_source", "generation_location_evidence_2022", "generation_fleet_2022_normalized"} <= required)

    def test_generated_tables_and_fingerprints_are_deterministic(self):
        paths = [path for path in (ROOT / "data/processed/planning").glob("*") if path.is_file()]
        before = {path: path.read_bytes() for path in paths}
        build_products()
        self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_zero_delay_override_preserves_all_source_dates(self):
        try:
            projects, _, exercise, metadata = build_products(delay_years=0)
            self.assertTrue((exercise.exercise_year == exercise.published_year).all())
            self.assertEqual(0, metadata["exercise_delay_years"])
            self.assertEqual(self.projects.original_payload_json.tolist(), projects.original_payload_json.tolist())
        finally:
            build_products()

    def test_full_inventory_query_does_not_accept_unverified_rows_or_invalid_units(self):
        for field, value in [("evidence_class", "legacy_unverified"), ("normalized_unit", "MWh")]:
            projects = self.projects.copy(deep=True)
            projects.loc[0, field] = value
            with self.assertRaises(PlanningValidationError):
                query_inventory(projects, 2039)


if __name__ == "__main__":
    unittest.main()
