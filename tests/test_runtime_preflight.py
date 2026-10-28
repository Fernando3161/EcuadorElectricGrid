"""Focused checks for documentary PDF scope and unlimited notebook execution."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat

from src.artifacts import load_manifest, required_artifact_ids, run_preflight
from src import notebook_runner


LEGACY_PDFS = {
    "planning_older_demand_pdf",
    "planning_older_generation_pdf",
    "planning_older_transmission_pdf",
}


class RuntimePreflightTests(unittest.TestCase):
    def test_documentary_pdfs_are_not_runtime_dependencies(self):
        manifest = load_manifest()
        for stage in ("06_validation", "07_planning", "08_scenario"):
            required = set(required_artifact_ids(manifest, stage))
            self.assertFalse(required & LEGACY_PDFS, stage)
        required = set(required_artifact_ids(manifest, "07_planning"))
        self.assertIn("planning_adjustment_pdf", required)
        self.assertIn("planning_adjustment_generation_source", required)
        self.assertIn("planning_adjustment_grid_source", required)

    def test_explicit_reference_audit_still_requires_all_three_pdfs(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = run_preflight(
                "planning_reference_audit", root=Path(temporary), offline=True
            )
        self.assertFalse(report.ok)
        self.assertEqual(set(report.required_ids), LEGACY_PDFS)
        self.assertEqual({r.artifact_id for r in report.unresolved()}, LEGACY_PDFS)
        self.assertTrue(all(r.status == "missing" for r in report.unresolved()))

    def test_unlimited_default_executes_in_a_real_kernel(self):
        args = notebook_runner.build_parser().parse_args(["tiny.ipynb"])
        self.assertEqual(args.timeout, -1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "tiny.ipynb", root / "executed.ipynb"
            nbformat.write(nbformat.v4.new_notebook(cells=[
                nbformat.v4.new_code_cell("import time\ntime.sleep(0.1)\nprint(6 * 7)")
            ]), source)
            with patch.object(
                notebook_runner, "NotebookClient", wraps=notebook_runner.NotebookClient
            ) as factory:
                notebook_runner.execute_notebook(
                    source, output_path=output, working_directory=root
                )
            self.assertEqual(factory.call_args.kwargs["timeout"], -1)
            executed = nbformat.read(output, as_version=4)
            self.assertEqual(executed.cells[0].outputs[0].text.strip(), "42")


if __name__ == "__main__":
    unittest.main()
