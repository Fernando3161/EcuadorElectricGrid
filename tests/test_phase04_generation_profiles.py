from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import nbformat
import pandas as pd
from jsonschema import Draft202012Validator

from src.availability import (
    assert_profile_acceptance,
    attach_renewable_availability,
    build_hydro_availability,
    load_hydro_assignments,
    load_monthly_flows,
)
from src.demand import expected_hourly_index
from src.generation import (
    assert_generation_acceptance,
    load_commissioning_events,
    load_generation_source,
    load_location_evidence,
    prepare_existing_generation,
)


ROOT = Path(__file__).resolve().parents[1]


class Phase04GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = load_generation_source(
            ROOT / "data/raw/generation/generation_existing_2022.csv"
        )
        cls.locations = load_location_evidence(
            ROOT / "data/interim/generation/plant_location_evidence_2022.csv"
        )
        cls.commissioning = load_commissioning_events(
            ROOT / "data/raw/generation/cenace_2022_commissioning_events.csv"
        )
        cls.result = prepare_existing_generation(
            cls.source, cls.locations, cls.commissioning
        )

    def test_source_transcription_retains_all_rows(self) -> None:
        self.assertEqual(len(self.source), 300)
        self.assertEqual(self.source["source_row"].tolist(), list(range(2, 302)))
        self.assertAlmostEqual(self.source["nominal_capacity_mw_original"].sum(), 8878.11)
        self.assertAlmostEqual(self.source["effective_capacity_mw_original"].sum(), 8239.44)

    def test_sni_fleet_reconciles_exactly(self) -> None:
        assert_generation_acceptance(self.result)
        expected = {
            "hydro": (65, 5184.19, 5144.61),
            "oil": (37, 2061.88, 1796.82),
            "biomass": (3, 144.30, 136.40),
            "biogas": (2, 8.32, 7.20),
            "onwind": (2, 66.48, 66.48),
            "solar": (24, 24.48, 23.60),
        }
        actual = self.result.carrier_summary.set_index("carrier")
        for carrier, (plants, nominal, effective) in expected.items():
            self.assertEqual(int(actual.loc[carrier, "plants"]), plants)
            self.assertAlmostEqual(actual.loc[carrier, "nominal_capacity_mw"], nominal)
            self.assertAlmostEqual(actual.loc[carrier, "effective_capacity_mw"], effective)

    def test_stable_ids_resolve_legitimate_duplicate_names(self) -> None:
        generators = self.result.generators
        self.assertTrue(generators["plant_id"].is_unique)
        loreto = generators[generators["name"] == "Loreto"]
        self.assertEqual(len(loreto), 2)
        self.assertEqual(set(loreto["carrier"]), {"hydro", "oil"})

    def test_no_future_or_nuclear_assets(self) -> None:
        generators = self.result.generators
        self.assertEqual(set(generators["baseline_status"]), {"existing_2022"})
        self.assertFalse(generators["carrier"].str.contains("nuclear", case=False).any())
        self.assertEqual(int(generators["commissioning_year"].notna().sum()), 3)
        self.assertEqual(
            set(generators.loc[generators["commissioning_year"].notna(), "commissioning_year"]),
            {2022},
        )
        self.assertTrue(generators["retirement_year"].isna().all())

    def test_in_year_operating_dates_are_source_linked(self) -> None:
        generators = self.result.generators.set_index("name")
        self.assertEqual(str(generators.loc["Sarapullo", "operating_start_date_2022"].date()), "2022-07-07")
        self.assertEqual(str(generators.loc["Sabanilla", "operating_start_date_2022"].date()), "2022-12-01")
        self.assertEqual(str(generators.loc["Huascachaca", "operating_start_date_2022"].date()), "2022-12-02")
        self.assertTrue(
            (generators.loc[["Sarapullo", "Sabanilla", "Huascachaca"], "operating_start_basis"]
             == "cenace_2022_commissioning_event").all()
        )

    def test_location_coverage_is_one_to_one(self) -> None:
        self.assertEqual(len(self.locations), 133)
        self.assertTrue(self.locations["source_row"].is_unique)
        self.assertFalse(self.locations[["latitude", "longitude"]].isna().any().any())
        self.assertEqual(
            set(self.locations["location_method"]),
            {"exact_2017", "manual_legacy", "capacity_proxy_2017"},
        )

    def test_normalized_rows_satisfy_json_schema(self) -> None:
        schema = json.loads(
            (ROOT / "config/schemas/generation_fleet.schema.json").read_text(
                encoding="utf-8"
            )
        )
        validator = Draft202012Validator(schema)
        rows = json.loads(
            self.result.generators.to_json(orient="records", date_format="iso")
        )
        for row in rows:
            self.assertEqual([], list(validator.iter_errors(row)))


class Phase04NotebookContractTests(unittest.TestCase):
    def test_supported_notebooks_are_output_free_and_start_with_preflight(self) -> None:
        for filename, stage in [
            ("02_prepare_2022_generation.ipynb", "02_generation"),
            ("03_prepare_renewable_and_hydro_profiles.ipynb", "03_profiles"),
        ]:
            notebook = nbformat.read(ROOT / "notebooks" / filename, as_version=4)
            code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
            markdown_cells = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
            self.assertGreaterEqual(len(markdown_cells), 4, filename)
            self.assertIn("run_preflight", code_cells[0].source, filename)
            self.assertIn(stage, code_cells[0].source, filename)
            self.assertTrue(
                all(not cell.get("outputs") for cell in code_cells), filename
            )
            self.assertTrue(
                all(cell.get("execution_count") is None for cell in code_cells), filename
            )


class Phase04AvailabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = load_generation_source(
            ROOT / "data/raw/generation/generation_existing_2022.csv"
        )
        locations = load_location_evidence(
            ROOT / "data/interim/generation/plant_location_evidence_2022.csv"
        )
        commissioning = load_commissioning_events(
            ROOT / "data/raw/generation/cenace_2022_commissioning_events.csv"
        )
        cls.generators = prepare_existing_generation(
            source, locations, commissioning
        ).generators
        cls.flows = load_monthly_flows(ROOT / "data/raw/generation/caudals.csv")
        cls.assignments = load_hydro_assignments(
            ROOT / "data/interim/generation/hydro_profile_assignments_2022.csv"
        )

    def test_flow_and_assignment_schemas(self) -> None:
        self.assertEqual(self.flows.shape, (8, 12))
        self.assertEqual(len(self.assignments), 65)
        self.assertEqual(
            set(self.assignments["confidence"]), {"high", "medium", "low"}
        )
        schema = json.loads(
            (ROOT / "config/schemas/hydro_assignment.schema.json").read_text(
                encoding="utf-8"
            )
        )
        validator = Draft202012Validator(schema)
        for row in self.assignments.to_dict(orient="records"):
            self.assertEqual([], list(validator.iter_errors(row)))

    def test_hydro_profiles_have_complete_coverage_and_bounds(self) -> None:
        profile, diagnostics = build_hydro_availability(
            self.generators, self.flows, self.assignments
        )
        self.assertEqual(profile.shape, (8760, 65))
        self.assertEqual(len(diagnostics), 65)
        self.assertTrue(profile.index.equals(expected_hourly_index(2022)))
        self.assertGreaterEqual(profile.to_numpy().min(), 0.0)
        self.assertLessEqual(profile.to_numpy().max(), 1.0)
        by_name = self.generators.set_index("name")
        for plant, final_preoperation_hour in {
            "Sarapullo": "2022-07-06 23:00:00",
            "Sabanilla": "2022-11-30 23:00:00",
        }.items():
            plant_id = by_name.loc[plant, "plant_id"]
            self.assertEqual(
                float(profile.loc[:final_preoperation_hour, plant_id].max()), 0.0
            )

    def test_missing_hydro_assignment_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not exactly cover"):
            build_hydro_availability(
                self.generators, self.flows, self.assignments.iloc[:-1]
            )

    def test_profile_validator_rejects_out_of_bounds_value(self) -> None:
        profile = pd.DataFrame(
            {"plant": np.zeros(8760)}, index=expected_hourly_index(2022)
        )
        profile.iloc[0, 0] = 1.01
        with self.assertRaisesRegex(AssertionError, "outside"):
            assert_profile_acceptance(profile, expected_columns={"plant"}, year=2022)

    def test_external_renewable_profiles_attach_to_every_plant(self) -> None:
        profile, diagnostics = attach_renewable_availability(
            self.generators,
            solar_profile_path=ROOT / "data/raw/cutouts/solar.nc",
            wind_profile_path=ROOT / "data/raw/cutouts/onwind.nc",
        )
        self.assertEqual(profile.shape, (8760, 26))
        self.assertEqual(set(diagnostics["weather_year"]), {2013})
        self.assertEqual(set(diagnostics["simulation_year"]), {2022})
        self.assertEqual(diagnostics.groupby("carrier").size().to_dict(), {"onwind": 2, "solar": 24})
        huascachaca = self.generators.set_index("name").loc["Huascachaca", "plant_id"]
        self.assertEqual(float(profile.loc[:"2022-12-01 23:00:00", huascachaca].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
