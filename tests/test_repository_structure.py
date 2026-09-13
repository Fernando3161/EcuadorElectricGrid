import csv
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecuador_grid.project_paths import PROJECT_ROOT, all_dirs  # noqa: E402


class RepositoryStructureTests(unittest.TestCase):
    def test_required_top_level_directories_exist(self) -> None:
        for relative in (
            "config",
            "data/raw",
            "data/processed",
            "data/external",
            "data/generated",
            "docs",
            "notebooks",
            "results",
            "scripts",
            "src/ecuador_grid",
            "tests",
        ):
            self.assertTrue((ROOT / relative).is_dir(), relative)

    def test_legacy_active_directories_are_absent(self) -> None:
        for relative in ("src/_old", "notebooks/_alt"):
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_notebook_directory_contains_no_generated_tables(self) -> None:
        generated = list((ROOT / "notebooks").glob("*.csv"))
        self.assertEqual(generated, [])

    def test_canonical_network_location_is_external(self) -> None:
        network = ROOT / "data/external/networks/ec_network_2022.nc"
        self.assertTrue(network.is_relative_to(ROOT / "data/external"))

    def test_project_paths_resolve_to_repository(self) -> None:
        self.assertEqual(PROJECT_ROOT, ROOT)
        directories = all_dirs()
        self.assertEqual(Path(directories[""]), ROOT)
        self.assertNotIn(".git", directories)
        self.assertNotIn(".venv", directories)

    def test_preserved_line_fix_schema(self) -> None:
        path = ROOT / "data/processed/networks/line_fix.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 5)
        self.assertTrue({"Line", "bus0", "bus1", "v_nom", "description"} <= rows[0].keys())

    def test_unresolved_same_bus_expansion_mapping_is_preserved(self) -> None:
        path = ROOT / "data/processed/networks/skipped_expansion_lines.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        unresolved = [
            row
            for row in rows
            if row["project"] == "LT_Delsitanisagua_Cumbaratza_138"
        ]
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["bus0"], unresolved[0]["bus1"])
        self.assertEqual(unresolved[0]["reason"], "bus0_equals_bus1")


if __name__ == "__main__":
    unittest.main()
