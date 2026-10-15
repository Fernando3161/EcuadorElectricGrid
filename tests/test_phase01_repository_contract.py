from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path
from urllib.parse import unquote

from src.repository_inventory import (
    ROOT,
    load_inventory,
    matching_ignored_rules,
    matching_rules,
    validate_inventory,
)


def repository_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    return [line.replace("\\", "/") for line in output.splitlines() if line]


class RepositoryInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = load_inventory()
        cls.files = repository_files()

    def test_every_tracked_or_pending_file_has_one_classification(self) -> None:
        errors = validate_inventory(self.inventory, self.files)
        self.assertEqual([], errors, "\n".join(errors))

    def test_relevant_ignored_artifacts_have_one_classification(self) -> None:
        output = subprocess.check_output(
            ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        )
        relevant = [
            path.replace("\\", "/")
            for path in output.splitlines()
            if path == "logs.log"
            or path.startswith(("data/", "literature/", "results/"))
        ]
        failures = []
        for path in relevant:
            matches = matching_ignored_rules(path, self.inventory)
            if len(matches) != 1:
                failures.append(f"{path}: {[row['id'] for row in matches]}")
        self.assertEqual([], failures, "\n".join(failures))

    def test_notebook_register_is_complete(self) -> None:
        actual = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "notebooks").rglob("*.ipynb")
        }
        registered = {row["path"] for row in self.inventory["notebook_register"]}
        self.assertEqual(actual, registered)

    def test_module_register_is_complete(self) -> None:
        actual = {
            path.relative_to(ROOT).as_posix()
            for directory in (ROOT / "src", ROOT / "notebooks")
            for path in directory.rglob("*.py")
        }
        registered = {row["path"] for row in self.inventory["module_register"]}
        self.assertEqual(actual, registered)

    def test_source_document_register_has_relative_unique_paths(self) -> None:
        rows = self.inventory["source_document_register"]
        paths = [row["path"] for row in rows]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(not Path(path).is_absolute() for path in paths))

    def test_critical_evidence_is_present(self) -> None:
        for evidence in self.inventory["critical_evidence"]:
            for relative in evidence["paths"]:
                self.assertTrue(
                    (ROOT / relative).is_file(),
                    f"critical evidence is missing: {relative}",
                )

    def test_required_external_artifacts_match_phase_0_fingerprints(self) -> None:
        required = [
            row
            for row in self.inventory["external_artifacts"]
            if row["classification"] == "external_required"
        ]
        self.assertEqual(3, len(required))
        for artifact in required:
            self.assertEqual("10.5281/zenodo.23392049", artifact["doi"])
            self.assertTrue(artifact["record_url"].startswith("https://"))
            self.assertTrue(artifact["download_url"].startswith("https://"))
            self.assertIn(Path(artifact["path"]).name, artifact["download_url"])
            self.assertEqual("CC-BY-4.0", artifact["license"])
            path = ROOT / artifact["path"]
            if not path.exists():
                continue
            self.assertEqual(artifact["bytes"], path.stat().st_size)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(artifact["sha256"], digest)

    def test_nuclear_contaminated_tables_are_quarantined(self) -> None:
        contaminated = [
            "data/raw/generation/generation_future_renewable_2022.csv",
            "data/raw/networks/EC_bus_expansion_masterplan_2022.csv",
            "data/raw/networks/EC_line_expansion_masterplan_2022.csv",
            "data/processed/generation/powerplants_all.csv",
            "data/processed/generation/powerplants_future.csv",
        ]
        for path in contaminated:
            rules = matching_rules(path, self.inventory)
            self.assertEqual(1, len(rules), path)
            self.assertEqual("mixed_official_and_hypothetical", rules[0]["classification"])

    def test_initial_json_schemas_are_well_formed(self) -> None:
        schema_dir = ROOT / "config" / "schemas"
        for path in schema_dir.glob("*.schema.json"):
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("$schema", schema)
            self.assertEqual("object", schema.get("type"))
            self.assertTrue(schema.get("required"), path.name)

    def test_governance_and_candidate_python_have_no_machine_paths(self) -> None:
        scan_paths = [ROOT / "README.md", ROOT / "AGENTS.md"]
        scan_paths.extend((ROOT / "docs").glob("*.md"))
        scan_paths.extend((ROOT / "reports").glob("*.md"))
        scan_paths.extend((ROOT / "src").glob("*.py"))
        scan_paths.extend((ROOT / "notebooks").glob("*.py"))
        machine_path = re.compile(r"(?:[A-Z]:[\\/]|/(?:home|Users)/[^/]+/)")
        findings = []
        for path in scan_paths:
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if machine_path.search(line):
                    findings.append(f"{path.relative_to(ROOT)}:{line_number}")
        self.assertEqual([], findings, f"machine-specific paths found: {findings}")

    def test_relative_markdown_links_resolve(self) -> None:
        markdown_files = [ROOT / "README.md", ROOT / "AGENTS.md"]
        markdown_files.extend((ROOT / "docs").glob("*.md"))
        markdown_files.extend((ROOT / "reports").glob("*.md"))
        pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
        broken = []
        for document in markdown_files:
            text = document.read_text(encoding="utf-8")
            for target in pattern.findall(text):
                target = target.strip().strip("<>")
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                relative = unquote(target.split("#", 1)[0])
                if relative and not (document.parent / relative).exists():
                    broken.append(f"{document.relative_to(ROOT)} -> {target}")
        self.assertEqual([], broken, f"broken Markdown links: {broken}")


if __name__ == "__main__":
    unittest.main()
