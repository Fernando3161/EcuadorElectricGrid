"""Small integrity checks for environment acceptance tooling; no solves or downloads."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

spec = importlib.util.spec_from_file_location(
    "environment_acceptance", Path(__file__).resolve().parents[1] / "scripts/verify_environment_pipeline.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PinnedInputRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "config").mkdir()
        (self.root / "data").mkdir()
        self.path = self.root / "data/source.csv"
        self.blob = b'name,note\na,"line one\nline two"\n'
        self.expected = b'name,note\r\na,"line one\nline two"\r\n'
        self.artifact = {"destination": "data/source.csv", "availability": "repository",
                         "role": "source", "bytes": len(self.expected),
                         "sha256": hashlib.sha256(self.expected).hexdigest()}

    def recover(self):
        (self.root / "config/artifacts.yaml").write_text(
            yaml.safe_dump({"artifacts": [self.artifact]}), encoding="utf-8"
        )
        git_result = SimpleNamespace(returncode=0, stdout=self.blob)
        with patch.object(module.subprocess, "run", return_value=git_result):
            return module.restore_pinned_inputs(self.root)

    def test_restores_exact_records_and_preserves_embedded_provenance_newline(self):
        self.path.write_bytes(self.blob)
        self.assertEqual(len(self.recover()), 1)
        self.assertEqual(self.path.read_bytes(), self.expected)
        self.assertEqual(self.recover(), [])

    def test_unmatched_checksum_never_replaces_the_input(self):
        self.path.write_bytes(self.blob)
        self.artifact["sha256"] = "0" * 64
        self.assertEqual(self.recover(), [])
        self.assertEqual(self.path.read_bytes(), self.blob)

    def test_generated_product_is_left_to_its_notebook(self):
        self.artifact["role"] = "generated_product"
        self.assertEqual(self.recover(), [])
        self.assertFalse(self.path.exists())

    def test_destination_cannot_escape_the_checkout(self):
        self.artifact["destination"] = "../outside.csv"
        with self.assertRaisesRegex(ValueError, "escapes checkout"):
            self.recover()


if __name__ == "__main__":
    unittest.main()
