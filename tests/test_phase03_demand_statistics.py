from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import nbformat
import pandas as pd
from jsonschema import Draft202012Validator

from src.demand import (
    assert_demand_acceptance,
    expected_hourly_index,
    load_hourly_template,
    load_monthly_targets,
    prepare_hourly_demand,
    reconcile_monthly_energy,
    validate_hourly_index,
)
from src.validation import (
    aggregate_model_supply,
    build_supply_reference,
    calculate_validation_metrics,
    load_raw_supply_reference,
    load_validation_contract,
    map_model_carriers,
    screen_validation,
)


ROOT = Path(__file__).resolve().parents[1]


class DemandPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_validation_contract()
        cls.targets = load_monthly_targets(
            ROOT / cls.contract["demand"]["monthly_targets"]
        )

    def test_official_monthly_target_schema_and_controls(self) -> None:
        schema = json.loads(
            (ROOT / "config/schemas/demand_monthly_target.schema.json").read_text(
                encoding="utf-8"
            )
        )
        records = json.loads(self.targets.to_json(orient="records"))
        errors = [
            error.message
            for record in records
            for error in Draft202012Validator(schema).iter_errors(record)
        ]
        self.assertEqual([], errors)
        self.assertAlmostEqual(27561.64, self.targets["energy_gwh"].sum(), places=6)
        self.assertAlmostEqual(4388.06, self.targets["peak_mw"].max(), places=6)

    def test_rounding_reconciliation_preserves_one_common_factor(self) -> None:
        reconciled = reconcile_monthly_energy(
            self.targets,
            annual_energy_gwh=self.contract["demand"]["annual_energy_gwh"],
        )
        factors = reconciled["model_energy_gwh"] / reconciled["energy_gwh"]
        self.assertAlmostEqual(27561.62, reconciled["model_energy_gwh"].sum(), places=9)
        self.assertLess(float(factors.max() - factors.min()), 1e-14)

    def test_profile_scaling_matches_energy_peak_time_and_spatial_controls(self) -> None:
        index = expected_hourly_index(2013)
        hour = np.arange(len(index))
        national = 3000.0 + 600.0 * np.sin(2 * np.pi * hour / 24.0) + 150.0 * np.cos(
            2 * np.pi * hour / (24.0 * 7.0)
        )
        template = pd.DataFrame(
            {
                "bus_a": national * (0.35 + 0.05 * np.sin(2 * np.pi * hour / 168.0)),
                "bus_b": national * (0.65 - 0.05 * np.sin(2 * np.pi * hour / 168.0)),
            },
            index=index,
        )
        source_shares = template.div(template.sum(axis=1), axis=0).to_numpy()
        result = prepare_hourly_demand(
            template,
            self.targets,
            target_year=2022,
            annual_energy_gwh=self.contract["demand"]["annual_energy_gwh"],
        )
        assert_demand_acceptance(result)
        validate_hourly_index(result.profile_mw.index, year=2022)
        self.assertEqual(8760, len(result.profile_mw))
        self.assertAlmostEqual(27561.62, result.summary["annual_energy_gwh"], places=8)
        self.assertAlmostEqual(4388.06, result.summary["annual_peak_mw"], places=8)
        output_shares = result.profile_mw.div(result.profile_mw.sum(axis=1), axis=0)
        self.assertLess(
            float(np.abs(output_shares.to_numpy() - source_shares).max()), 1e-12
        )

    def test_leap_year_mismatch_is_rejected(self) -> None:
        leap_index = expected_hourly_index(2012)
        template = pd.DataFrame({"bus": np.ones(len(leap_index))}, index=leap_index)
        with self.assertRaisesRegex(ValueError, "leap-day handling"):
            prepare_hourly_demand(
                template,
                self.targets,
                target_year=2022,
                annual_energy_gwh=27561.62,
            )

    def test_real_template_has_explicit_valid_hourly_structure(self) -> None:
        profile = load_hourly_template(ROOT / self.contract["demand"]["template"])
        self.assertEqual((8760, 251), profile.shape)
        validate_hourly_index(profile.index, year=2013)


class StatisticalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_validation_contract()
        cls.raw = load_raw_supply_reference(
            ROOT / cls.contract["supply_reference"]["raw_table"]
        )
        cls.reference = build_supply_reference(cls.raw, cls.contract)

    def test_official_reference_aggregates_and_reconciles(self) -> None:
        expected = {
            "hydro": 24499.54,
            "nonconventional": 325.89,
            "thermal": 3393.35,
            "interconnection": 465.83,
        }
        actual = self.reference.set_index("carrier")["reference_gwh"].to_dict()
        for carrier, value in expected.items():
            self.assertAlmostEqual(value, actual[carrier], places=8)
        self.assertAlmostEqual(
            self.contract["supply_reference"]["reported_total_gwh"],
            self.reference["reference_gwh"].sum(),
            places=8,
        )
        residual = self.reference.set_index("carrier").loc["source_rounding_residual"]
        self.assertFalse(bool(residual["in_scope"]))
        self.assertAlmostEqual(0.01, residual["reference_gwh"], places=8)

    def test_committed_official_reports_match_contract_checksums(self) -> None:
        for source in self.contract["sources"].values():
            repository_path = source.get("repository_path")
            expected_sha256 = source.get("sha256")
            if not repository_path:
                continue

            report = ROOT / repository_path
            payload = report.read_bytes()
            self.assertTrue(payload.startswith(b"%PDF-"), repository_path)
            self.assertTrue(payload.rstrip().endswith(b"%%EOF"), repository_path)
            self.assertEqual(
                expected_sha256,
                hashlib.sha256(payload).hexdigest(),
                repository_path,
            )

    def test_normalized_reference_schema(self) -> None:
        schema = json.loads(
            (ROOT / "config/schemas/statistical_reference.schema.json").read_text(
                encoding="utf-8"
            )
        )
        records = json.loads(self.reference.to_json(orient="records"))
        errors = [
            error.message
            for record in records
            for error in Draft202012Validator(schema).iter_errors(record)
        ]
        self.assertEqual([], errors)

    def test_model_carrier_mapping_is_explicit(self) -> None:
        mapped = map_model_carriers(
            ["hydro", "onwind", "biomass", "biogas", "solar", "oil", "imports"],
            self.contract,
        )
        self.assertEqual(
            [
                "hydro",
                "nonconventional",
                "nonconventional",
                "nonconventional",
                "nonconventional",
                "thermal",
                "interconnection",
            ],
            mapped.tolist(),
        )
        with self.assertRaisesRegex(ValueError, "unmapped model carriers"):
            map_model_carriers(["nuclear"], self.contract)

    def test_model_supply_aggregation_uses_official_categories(self) -> None:
        model = pd.Series(
            {"hydro": 10.0, "biomass": 2.0, "biogas": 1.0, "oil": 3.0}
        )
        aggregated = aggregate_model_supply(model, self.contract)
        self.assertEqual(3.0, aggregated["nonconventional"])
        self.assertEqual(3.0, aggregated["thermal"])

    def test_multi_criteria_screen_accepts_exact_reference(self) -> None:
        base = self.reference.loc[self.reference["in_scope"]].set_index("carrier")[
            "reference_gwh"
        ]
        rows, metrics = calculate_validation_metrics(base, self.reference)
        result = screen_validation(metrics, rows, self.contract["acceptance"])
        self.assertTrue(result["passed"])

    def test_missing_positive_category_is_zero_and_fails(self) -> None:
        rows, metrics = calculate_validation_metrics(
            pd.Series({"hydro": 24499.54}), self.reference
        )
        indexed = rows.set_index("carrier")
        self.assertEqual(0.0, indexed.loc["interconnection", "simulated_gwh"])
        result = screen_validation(metrics, rows, self.contract["acceptance"])
        self.assertFalse(result["passed"])

    def test_metric_calculation_keeps_zero_reference_relative_error_null(self) -> None:
        reference = pd.DataFrame([
            {"carrier": "zero_test", "reference_gwh": 0.0, "in_scope": True},
            {"carrier": "hydro", "reference_gwh": 1.0, "in_scope": True},
        ])
        rows, _ = calculate_validation_metrics(
            pd.Series({"zero_test": 0.01, "hydro": 1.0}), reference
        )
        self.assertTrue(np.isnan(rows.set_index("carrier").loc["zero_test", "relative_error"]))


class Stage01NotebookContractTests(unittest.TestCase):
    def test_supported_notebook_is_explanatory_output_free_and_starts_with_preflight(
        self,
    ) -> None:
        notebook_path = ROOT / "notebooks/01_prepare_2022_demand.ipynb"
        notebook = nbformat.read(notebook_path, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        markdown_cells = [cell for cell in notebook.cells if cell.cell_type == "markdown"]

        self.assertGreaterEqual(len(markdown_cells), 6)
        self.assertTrue(code_cells)
        self.assertIn("run_preflight", code_cells[0].source)
        self.assertIn('"01_demand"', code_cells[0].source)
        self.assertTrue(
            all(cell.execution_count is None and not cell.outputs for cell in code_cells)
        )

    def test_supported_demand_output_declares_mw_in_name_and_contract(self) -> None:
        contract = load_validation_contract()
        output = contract["demand"]["output"]
        self.assertTrue(output.endswith("_mw.csv"))
        self.assertEqual("MW", contract["demand"]["output_unit"])


if __name__ == "__main__":
    unittest.main()
