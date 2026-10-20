"""Tiny synthetic planning fixtures; no real portfolio or solve is validated."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.planning import (
    PlanningValidationError, aggregate_by_year, filter_official_by_year,
    filter_published_by_year, load_portfolio, scheduled_portfolio,
    select_exercise_by_year, validate_portfolio, validate_published_portfolio,
)


def project(pid, kind, year=2030, **changes):
    row = dict(
        project_id=pid, asset_type=kind, official_name=f"Synthetic {pid}",
        commissioning_year=year, evidence_class="official_fact",
        source_document="Synthetic fixture only", source_edition="2022 fixture",
        source_chapter="Fixture chapter", source_page_or_table="Fixture table 1",
        original_name=pid, original_value="10", original_unit="MW",
        normalized_value=10.0, normalized_unit="MW", transformation="Identity",
        confidence="high", source_artifact="tests/synthetic.csv",
        source_sha256="a" * 64, source_row=1, official_timing=str(year),
        review_status="verified", blockers="", dependencies="",
        latitude=-1.0, longitude=-79.0, voltage_kv=230.0,
    )
    row.update(changes)
    return row


def portfolio():
    return {
        "generation": pd.DataFrame([project("GEN_01", "generator", substation_id="SUB_01")]),
        "lines": pd.DataFrame([project("LINE_01", "line", endpoint0_id="SUB_01", endpoint1_id="SUB_02", normalized_unit="km")]),
        "transformers": pd.DataFrame([project("TR_01", "transformer", endpoint0_id="SUB_01", endpoint1_id="SUB_03", voltage0_kv=230.0, voltage1_kv=138.0, normalized_unit="MVA")]),
        "substations": pd.DataFrame([
            project("SUB_01", "substation", 2028, normalized_unit="MVA"),
            project("SUB_02", "substation", 2028, normalized_unit="MVA"),
            project("SUB_03", "substation", 2028, voltage_kv=138.0, normalized_unit="MVA"),
        ]),
    }


class PlanningTests(unittest.TestCase):
    def test_synthetic_official_fixture_and_inclusive_year_boundary(self):
        tables = portfolio()
        self.assertTrue(validate_portfolio(tables).empty)
        self.assertTrue(filter_official_by_year(tables, 2029)["generation"].empty)
        self.assertEqual(len(filter_official_by_year(tables, 2030)["generation"]), 1)

    def test_unknown_year_cannot_silently_disappear(self):
        tables = portfolio()
        tables["generation"].loc[0, "commissioning_year"] = None
        with self.assertRaises(PlanningValidationError):
            filter_official_by_year(tables, 2022)
        self.assertIn("unknown_year", set(validate_portfolio(tables).code))

    def test_legacy_evidence_blocks_only_when_within_horizon(self):
        tables = portfolio()
        tables["generation"].loc[0, "evidence_class"] = "legacy_unverified"
        tables["generation"].loc[0, "review_status"] = "unverified"
        self.assertIn("unverified", set(validate_portfolio(tables).code))
        filter_official_by_year(tables, 2029)
        with self.assertRaises(PlanningValidationError):
            aggregate_by_year(tables, 2030)

    def test_official_nuclear_is_rejected(self):
        for field in ("official_name", "original_name", "carrier"):
            with self.subTest(field=field):
                tables = portfolio()
                tables["generation"].loc[0, field] = "nuclear"
                self.assertIn("nuclear_official", set(validate_portfolio(tables).code))

    def test_unique_ids_across_asset_types(self):
        tables = portfolio()
        tables["generation"].loc[0, "project_id"] = "SUB_01"
        self.assertIn("duplicate_id", set(validate_portfolio(tables).code))
        with self.assertRaises(PlanningValidationError):
            filter_official_by_year(tables, 2030)

    def test_voltage_self_loop_missing_endpoint_and_coordinates(self):
        mutations = [
            ("lines", "voltage_kv", 138.0, "voltage"),
            ("lines", "endpoint1_id", "SUB_01", "self_loop"),
            ("lines", "endpoint0_id", "MISSING", "missing_endpoint"),
            ("substations", "latitude", 91.0, "coordinates"),
            ("generation", "longitude", None, "coordinates"),
            ("transformers", "voltage1_kv", 230.0, "voltage"),
        ]
        for table, column, value, code in mutations:
            with self.subTest(table=table, column=column):
                tables = portfolio()
                tables[table].loc[0, column] = value
                self.assertIn(code, set(validate_portfolio(tables).code))

    def test_dependencies_chronology_missing_cycle_and_quarantine(self):
        tables = portfolio()
        tables["generation"].loc[0, "dependencies"] = "SUB_01"
        tables["substations"].loc[0, "commissioning_year"] = 2031
        self.assertIn("dependency_chronology", set(validate_portfolio(tables).code))
        tables["generation"].loc[0, "dependencies"] = "ABSENT"
        self.assertIn("missing_dependency", set(validate_portfolio(tables).code))
        tables = portfolio()
        tables["generation"].loc[0, "dependencies"] = "LINE_01"
        tables["lines"].loc[0, "dependencies"] = "GEN_01"
        self.assertIn("dependency_cycle", set(validate_portfolio(tables).code))
        tables = portfolio()
        tables["substations"].loc[0, "review_status"] = "quarantined"
        with self.assertRaises(PlanningValidationError):
            filter_official_by_year(tables, 2030)

    def test_quantity_and_official_provenance_are_required(self):
        for value in (float("inf"), -1.0, "unknown", None):
            with self.subTest(value=value):
                tables = portfolio()
                tables["generation"]["normalized_value"] = tables["generation"]["normalized_value"].astype(object)
                tables["generation"].loc[0, "normalized_value"] = value
                self.assertIn("invalid_quantity", set(validate_portfolio(tables).code))
        for field in ("source_page_or_table", "source_sha256", "original_unit"):
            tables = portfolio()
            tables["generation"].loc[0, field] = None
            self.assertIn("missing_evidence", set(validate_portfolio(tables).code))

    def test_aggregation_retains_separate_units_and_stable_order(self):
        tables = portfolio()
        extra = project("GEN_00", "generator", substation_id="SUB_01", normalized_value=2.0, normalized_unit="kW")
        tables["generation"] = pd.concat([tables["generation"], pd.DataFrame([extra])], ignore_index=True)
        result = aggregate_by_year(tables, 2030)
        units = result[result.asset_type == "generator"].set_index("normalized_unit")
        self.assertEqual(units.loc["MW", "normalized_value"], 10.0)
        self.assertEqual(units.loc["kW", "normalized_value"], 2.0)
        self.assertEqual(filter_official_by_year(tables, 2030)["generation"].project_id.tolist(), ["GEN_00", "GEN_01"])
        shuffled = {key: frame.iloc[::-1].copy() for key, frame in tables.items()}
        pd.testing.assert_frame_equal(result, aggregate_by_year(shuffled, 2030))

    def test_csv_roundtrip_preserves_raw_strings_and_nulls(self):
        tables = portfolio()
        tables["generation"].loc[0, "original_value"] = "010 MW (raw)"
        with tempfile.TemporaryDirectory() as directory:
            for key, frame in tables.items():
                frame.to_csv(Path(directory) / f"{key}.csv", index=False)
            loaded = load_portfolio(directory)
            self.assertEqual(loaded["generation"].loc[0, "original_value"], "010 MW (raw)")
            self.assertTrue(validate_portfolio(loaded).empty)
            (Path(directory) / "lines.csv").unlink()
            with self.assertRaises(PlanningValidationError):
                load_portfolio(directory)

    def test_prebaseline_and_invalid_year_selection(self):
        tables = portfolio()
        tables["generation"].loc[0, "commissioning_year"] = 2021
        self.assertIn("baseline_overlap", set(validate_portfolio(tables).code))
        with self.assertRaises(PlanningValidationError):
            filter_official_by_year(tables, 2030)
        for year in (True, 2030.5, "2030", 2021):
            with self.assertRaises(ValueError):
                filter_official_by_year(portfolio(), year)

    def test_empty_malformed_table_cannot_bypass_schema(self):
        tables = portfolio()
        tables["lines"] = pd.DataFrame()
        self.assertIn("missing_columns", set(validate_portfolio(tables).code))
        with self.assertRaises(PlanningValidationError):
            filter_official_by_year(tables, 2030)

    def test_generator_connection_must_exist_and_be_commissioned(self):
        for connection in (None, "ABSENT", "LINE_01"):
            with self.subTest(connection=connection):
                tables = portfolio()
                tables["generation"].loc[0, "substation_id"] = connection
                self.assertIn("missing_connection", set(validate_portfolio(tables).code))
                with self.assertRaises(PlanningValidationError):
                    filter_official_by_year(tables, 2030)
        tables = portfolio()
        tables["substations"].loc[0, "commissioning_year"] = 2031
        generator_issues = validate_portfolio(tables).query("project_id == 'GEN_01'")
        self.assertIn("dependency_chronology", set(generator_issues.code))

    def test_generator_connection_participates_in_cycle_and_selection_closure(self):
        tables = portfolio()
        tables["substations"].loc[0, "commissioning_year"] = 2030
        tables["substations"].loc[0, "dependencies"] = "GEN_01"
        self.assertIn("dependency_cycle", set(validate_portfolio(tables).code))
        tables = portfolio()
        # Remove branches to isolate the generator's physical reference.
        tables["lines"] = tables["lines"].iloc[:0]
        tables["transformers"] = tables["transformers"].iloc[:0]
        tables["substations"].loc[0, "review_status"] = "quarantined"
        with self.assertRaises(PlanningValidationError):
            filter_official_by_year(tables, 2030)

    def test_units_are_compatible_with_asset_type(self):
        for table in ("generation", "lines", "transformers", "substations"):
            for unit in ("unknown", "MWh"):
                with self.subTest(table=table, unit=unit):
                    tables = portfolio()
                    tables[table].loc[0, "normalized_unit"] = unit
                    self.assertIn("invalid_unit", set(validate_portfolio(tables).code))
                    with self.assertRaises(PlanningValidationError):
                        aggregate_by_year(tables, 2030)

    def test_published_evidence_does_not_imply_physical_readiness(self):
        tables = portfolio()
        for key, frame in tables.items():
            tables[key] = frame.drop(columns=[
                "latitude", "longitude", "voltage_kv", "voltage0_kv",
                "voltage1_kv", "substation_id", "endpoint0_id", "endpoint1_id",
            ], errors="ignore")
        self.assertTrue(validate_published_portfolio(tables).empty)
        self.assertEqual(len(filter_published_by_year(tables, 2030)["generation"]), 1)
        with self.assertRaises(PlanningValidationError) as caught:
            select_exercise_by_year(tables, 2037)
        self.assertIn("missing_connection", set(caught.exception.issues.code))
        selected = select_exercise_by_year(tables, 2037, require_model_ready=False)
        self.assertEqual(len(selected["generation"]), 1)

    def test_published_and_delayed_inclusive_dates_never_mutate_sources(self):
        tables = portfolio()
        before = {key: frame.copy(deep=True) for key, frame in tables.items()}
        self.assertTrue(filter_published_by_year(tables, 2029)["generation"].empty)
        self.assertEqual(len(filter_published_by_year(tables, 2030)["generation"]), 1)
        self.assertTrue(filter_published_by_year(tables, 2036, delay_years=7)["generation"].empty)
        delayed = select_exercise_by_year(tables, 2037)
        row = delayed["generation"].iloc[0]
        self.assertEqual(row.commissioning_year, 2030)
        self.assertEqual(row.published_year, 2030)
        self.assertEqual(row.exercise_year, 2037)
        self.assertEqual(row.official_timing, "2030")
        for key in tables:
            pd.testing.assert_frame_equal(before[key], tables[key])

    def test_scheduling_rejects_invalid_delay_and_unknown_source_year(self):
        for delay in (-1, True, 7.5, "7"):
            with self.subTest(delay=delay), self.assertRaises(ValueError):
                scheduled_portfolio(portfolio(), delay_years=delay)
        tables = portfolio()
        tables["generation"].loc[0, "commissioning_year"] = None
        with self.assertRaises(PlanningValidationError):
            filter_published_by_year(tables, 2022)

    def test_source_query_accepts_prebaseline_facts_but_exercise_needs_review(self):
        tables = portfolio()
        tables["generation"].loc[0, "commissioning_year"] = 2021
        self.assertEqual(len(filter_published_by_year(tables, 2021)["generation"]), 1)
        with self.assertRaises(PlanningValidationError) as caught:
            select_exercise_by_year(tables, 2028, require_model_ready=False)
        self.assertIn("baseline_overlap_review", set(caught.exception.issues.code))
        tables["generation"]["baseline_overlap"] = True
        with self.assertRaises(PlanningValidationError):
            select_exercise_by_year(tables, 2028, require_model_ready=False)
        selected = select_exercise_by_year(tables, 2028, require_model_ready=False, repeat_mode=True)
        self.assertEqual(selected["generation"].iloc[0].scenario_asset_id, "EX_REPEAT_GEN_01")
        self.assertTrue(selected["generation"].iloc[0].baseline_overlap)

    def test_published_queries_exclude_legacy_and_quarantined_evidence(self):
        tables = portfolio()
        legacy = project("LEGACY_GEN", "generator", year=None, evidence_class="legacy_unverified", review_status="unverified")
        quarantined = project("QUARANTINE_GEN", "generator", year=None, review_status="quarantined")
        tables["generation"] = pd.concat([tables["generation"], pd.DataFrame([legacy, quarantined])], ignore_index=True)
        self.assertEqual(filter_published_by_year(tables, 2030)["generation"].project_id.tolist(), ["GEN_01"])

    def test_published_queries_still_enforce_citations_units_and_nuclear_exclusion(self):
        for field, value, code in (
            ("source_page_or_table", None, "missing_evidence"),
            ("normalized_unit", "MWh", "invalid_unit"),
            ("official_name", "Nuclear project", "nuclear_official"),
        ):
            with self.subTest(field=field):
                tables = portfolio()
                tables["generation"].loc[0, field] = value
                self.assertIn(code, set(validate_published_portfolio(tables).code))
                with self.assertRaises(PlanningValidationError):
                    filter_published_by_year(tables, 2030)

    def test_explicit_unquantified_source_remains_a_fact_and_requires_model_inputs(self):
        tables = portfolio()
        tables["lines"]["quantity_status"] = "not_specified"
        for field in ("original_value", "original_unit", "normalized_value", "normalized_unit"):
            tables["lines"][field] = None
        self.assertTrue(validate_published_portfolio(tables).empty)
        self.assertEqual(len(filter_published_by_year(tables, 2030)["lines"]), 1)
        with self.assertRaises(PlanningValidationError) as caught:
            select_exercise_by_year(tables, 2037)
        self.assertIn("invalid_quantity", set(caught.exception.issues.code))
        tables["lines"]["normalized_value"] = "invalid"
        self.assertIn("invalid_quantity", set(validate_published_portfolio(tables).code))


if __name__ == "__main__":
    unittest.main()
