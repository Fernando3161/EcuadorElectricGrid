"""Small real-evidence migration regression checks; no PDF parsing or solve."""

import csv
import hashlib
import json
import unittest

import pandas as pd

from scripts.migrate_phase07_planning import ROOT, SOURCES, migrate
from src.planning import PlanningValidationError, aggregate_by_year, load_portfolio, validate_portfolio


class PlanningMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        migrate()
        cls.tables = load_portfolio(ROOT / "data/interim/planning/legacy_candidates")
        cls.quarantine = pd.read_csv(ROOT / "data/interim/planning/nuclear_quarantine.csv", keep_default_na=False)

    def test_every_source_record_is_preserved_exactly_once(self):
        combined = pd.concat([*self.tables.values(), self.quarantine], ignore_index=True)
        self.assertEqual(198, len(combined))
        self.assertEqual(198, combined.project_id.nunique())
        for _, relative, *_ in SOURCES:
            path = ROOT / relative
            with path.open(encoding="utf-8-sig", newline="") as handle:
                original = list(csv.DictReader(handle))
            migrated = combined[combined.source_artifact == relative]
            self.assertEqual(len(original), len(migrated))
            self.assertEqual({hashlib.sha256(path.read_bytes()).hexdigest()}, set(migrated.source_sha256))
            for row in migrated.to_dict("records"):
                self.assertEqual(original[int(row["source_row"]) - 1], json.loads(row["original_payload_json"]))

    def test_no_units_or_official_facts_are_invented(self):
        for frame in self.tables.values():
            self.assertEqual({"legacy_unverified"}, set(frame.evidence_class))
            self.assertTrue(frame.normalized_value.isna().all())
            self.assertTrue(frame.normalized_unit.isna().all())
            self.assertFalse(frame.official_name.str.contains("nuclear", case=False).any())
        self.assertEqual(11, len(self.quarantine))
        self.assertEqual({"quarantined"}, set(self.quarantine.review_status))
        self.assertEqual({"modeling_assumption"}, set(self.quarantine.evidence_class))

    def test_real_candidates_fit_schema_but_cannot_publish(self):
        issues = validate_portfolio(self.tables)
        self.assertNotIn("schema", set(issues.code))
        with self.assertRaises(PlanningValidationError):
            aggregate_by_year(self.tables, 2030)

    def test_regeneration_is_byte_deterministic(self):
        paths = [*sorted((ROOT / "data/interim/planning/legacy_candidates").glob("*.csv")),
                 *sorted(path for path in (ROOT / "data/interim/planning").glob("*") if path.is_file())]
        before = {path: path.read_bytes() for path in paths}
        migrate()
        self.assertEqual(before, {path: path.read_bytes() for path in paths})


if __name__ == "__main__":
    unittest.main()
