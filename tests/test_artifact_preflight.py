from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import h5netcdf
import nbformat
import yaml

from src.artifacts import (
    ArtifactValidationError,
    ManifestError,
    load_manifest,
    http_download,
    required_artifact_ids,
    run_preflight,
    validate_artifact,
)
from src.environment import EXPECTED_ENVIRONMENT, check_environment
from src.notebook_runner import TEMP_KERNEL_NAME, _write_temporary_kernelspec
from src.preflight import main as preflight_main


ROOT = Path(__file__).resolve().parents[1]


def make_manifest(payload: bytes, *, destination: str = "data/raw/test.bin") -> dict:
    return {
        "manifest_version": 1,
        "project": "test",
        "record": {
            "title": "Test artifact",
            "version": "1",
            "doi": "10.0000/test",
            "record_url": "https://example.test/record",
            "license": "test-only",
        },
        "stages": {
            "root": {"description": "root", "depends_on": []},
            "child": {"description": "child", "depends_on": ["root"]},
            "unrelated": {"description": "unrelated", "depends_on": []},
        },
        "artifacts": [
            {
                "id": "test_artifact",
                "description": "deterministic test payload",
                "stages": ["root"],
                "destination": destination,
                "role": "external_input",
                "source": {"organization": "tests"},
                "persistent_uri": "https://example.test/record",
                "download_url": "https://example.test/test.bin",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "media_type": "application/octet-stream",
                "license": "test-only",
                "validation": {"kind": "checksum_only"},
            }
        ],
    }


def write_manifest(directory: Path, manifest: dict) -> Path:
    path = directory / "artifacts.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


class PublishedManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()

    def test_manifest_has_exact_published_input_set(self) -> None:
        self.assertEqual(
            {
                "pypsa_earth_base_ec",
                "renewable_profile_solar_ec",
                "renewable_profile_onwind_ec",
            },
            {
                row["id"]
                for row in self.manifest["artifacts"]
                if row.get("availability", "external") == "external"
            },
        )

    def test_stage_one_checks_all_demand_and_reference_inputs(self) -> None:
        self.assertEqual(
            {
                "demand_template_2013_ec",
                "cenace_sni_2022_monthly_demand",
                "cenace_2022_net_supply",
            },
            set(required_artifact_ids(self.manifest, "01_demand")),
        )

    def test_generated_networks_and_full_cutout_are_excluded(self) -> None:
        destinations = {row["destination"] for row in self.manifest["artifacts"]}
        self.assertNotIn("data/raw/cutouts/cutout-2013-era5.nc", destinations)
        self.assertNotIn("data/processed/networks/ec_network_2022.nc", destinations)
        self.assertNotIn("data/processed/networks/network_base_filled.nc", destinations)

    def test_stage_dependency_closure(self) -> None:
        expected = {
            "00_preflight": 3,
            "01_demand": 3,
            "02_generation": 0,
            "03_profiles": 2,
            "04_topology": 1,
            "05_assembly": 6,
            "06_validation": 6,
            "07_planning": 0,
            "08_scenario": 6,
            "09_comparison": 6,
        }
        for stage, count in expected.items():
            self.assertEqual(count, len(required_artifact_ids(self.manifest, stage)), stage)

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaises(ManifestError):
            required_artifact_ids(self.manifest, "does_not_exist")

    def test_stage_zero_notebook_calls_common_preflight_first(self) -> None:
        notebook = nbformat.read(
            ROOT / "notebooks" / "00_preflight_and_data_inventory.ipynb",
            as_version=4,
        )
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        self.assertTrue(code_cells)
        self.assertIn("run_preflight", code_cells[0].source)
        self.assertIn("check_environment", code_cells[0].source)


class ArtifactBehaviorTests(unittest.TestCase):
    payload = b"verified artifact payload\n"

    def test_valid_cached_file_does_not_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = write_manifest(root, make_manifest(self.payload))
            destination = root / "data" / "raw" / "test.bin"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(self.payload)

            def forbidden_download(*_args) -> None:
                raise AssertionError("downloader must not run for a valid cached file")

            report = run_preflight(
                "child", manifest_path=manifest_path, root=root, downloader=forbidden_download
            )
            self.assertTrue(report.ok)
            self.assertEqual("valid", report.results[0].status)

    def test_missing_file_fails_cleanly_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = write_manifest(root, make_manifest(self.payload))
            report = run_preflight(
                "root", manifest_path=manifest_path, root=root, offline=True
            )
            self.assertFalse(report.ok)
            self.assertEqual("missing", report.results[0].status)
            self.assertIn("offline", report.results[0].message)

    def test_missing_repository_artifact_never_attempts_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest(self.payload)
            manifest["artifacts"][0]["availability"] = "repository"
            manifest["artifacts"][0]["role"] = "source"
            manifest["artifacts"][0]["download_url"] = None
            manifest_path = write_manifest(root, manifest)

            def forbidden_download(*_args) -> None:
                raise AssertionError("repository artifacts must never be downloaded")

            report = run_preflight(
                "root", manifest_path=manifest_path, root=root, downloader=forbidden_download
            )
            self.assertFalse(report.ok)
            self.assertEqual("missing", report.results[0].status)
            self.assertIn("restore", report.results[0].message)

    def test_checksum_failure_is_reported_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = write_manifest(root, make_manifest(self.payload))
            destination = root / "data" / "raw" / "test.bin"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"wrong but present")
            report = run_preflight(
                "root", manifest_path=manifest_path, root=root, offline=True
            )
            self.assertFalse(report.ok)
            self.assertEqual("invalid", report.results[0].status)

    def test_mocked_download_is_verified_and_atomically_installed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = write_manifest(root, make_manifest(self.payload))

            def downloader(_url: str, partial: Path, _size: int | None) -> None:
                partial.parent.mkdir(parents=True, exist_ok=True)
                partial.write_bytes(self.payload)

            report = run_preflight(
                "root", manifest_path=manifest_path, root=root, downloader=downloader
            )
            destination = root / "data" / "raw" / "test.bin"
            self.assertTrue(report.ok)
            self.assertEqual("downloaded", report.results[0].status)
            self.assertEqual(self.payload, destination.read_bytes())
            self.assertFalse(destination.with_name("test.bin.part").exists())

    def test_invalid_authoritative_file_is_preserved_before_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = write_manifest(root, make_manifest(self.payload))
            destination = root / "data" / "raw" / "test.bin"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"old invalid file")

            def downloader(_url: str, partial: Path, _size: int | None) -> None:
                partial.write_bytes(self.payload)

            report = run_preflight(
                "root", manifest_path=manifest_path, root=root, downloader=downloader
            )
            backups = list(destination.parent.glob("test.bin.invalid-*"))
            self.assertTrue(report.ok)
            self.assertEqual(1, len(backups))
            self.assertEqual(b"old invalid file", backups[0].read_bytes())
            self.assertEqual(self.payload, destination.read_bytes())

    def test_failed_download_never_becomes_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = write_manifest(root, make_manifest(self.payload))

            def downloader(_url: str, partial: Path, _size: int | None) -> None:
                partial.parent.mkdir(parents=True, exist_ok=True)
                partial.write_bytes(b"partial")

            report = run_preflight(
                "root", manifest_path=manifest_path, root=root, downloader=downloader
            )
            destination = root / "data" / "raw" / "test.bin"
            self.assertFalse(report.ok)
            self.assertFalse(destination.exists())
            self.assertTrue(destination.with_name("test.bin.part").exists())

    def test_http_download_resumes_an_existing_partial_file(self) -> None:
        class FakeResponse:
            status_code = 206

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                self.chunk_size = chunk_size
                yield b"payload\n"

            def close(self) -> None:
                return None

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            partial = Path(temporary) / "artifact.part"
            partial.write_bytes(b"verified ")
            response = FakeResponse()
            with patch("requests.get", return_value=response) as request:
                http_download(
                    "https://example.test/artifact",
                    partial,
                    expected_bytes=len(b"verified payload\n"),
                )
            self.assertEqual(b"verified payload\n", partial.read_bytes())
            self.assertEqual(
                "bytes=9-", request.call_args.kwargs["headers"]["Range"]
            )

    def test_netcdf_structure_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.nc"
            with h5netcdf.File(path, "w") as dataset:
                dataset.dimensions = {"time": 2}
                dataset.create_variable("profile", ("time",), dtype="f4")[:] = [0.1, 0.2]
                dataset.attrs["kind"] = "fixture"
            artifact = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "validation": {
                    "kind": "netcdf",
                    "dimensions": {"time": 2},
                    "required_variables": ["profile"],
                    "attributes": {"kind": "fixture"},
                },
            }
            validate_artifact(path, artifact)
            artifact["validation"]["dimensions"]["time"] = 3
            with self.assertRaises(ArtifactValidationError):
                validate_artifact(path, artifact)

    def test_unsafe_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = make_manifest(self.payload, destination="../escape.bin")
            path = write_manifest(root, manifest)
            with self.assertRaises(ManifestError):
                load_manifest(path)


class EnvironmentTests(unittest.TestCase):
    def test_approved_environment_is_active_and_complete(self) -> None:
        report = check_environment()
        self.assertEqual(EXPECTED_ENVIRONMENT, report.detected_name)
        self.assertTrue(report.ok, report.as_dict())

    def test_environment_specification_has_no_empty_or_misspelled_dependencies(self) -> None:
        specification = yaml.safe_load(
            (ROOT / "config" / "environment_ec.yaml").read_text(encoding="utf-8")
        )
        dependencies = specification["dependencies"]
        self.assertNotIn(None, dependencies)
        text_dependencies = {
            re.split(r"[<>=!~ ]", row, maxsplit=1)[0].strip()
            for row in dependencies
            if isinstance(row, str)
        }
        self.assertIn("plotly", text_dependencies)
        self.assertNotIn("ploty", text_dependencies)
        self.assertTrue(
            {"h5netcdf", "pyyaml", "requests", "jsonschema", "nbclient", "nbformat"}
            .issubset(text_dependencies)
        )

    def test_temporary_kernelspec_uses_current_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_temporary_kernelspec(root)
            kernel = json.loads(
                (root / "kernels" / TEMP_KERNEL_NAME / "kernel.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(sys.executable, kernel["argv"][0])

    def test_cli_rejects_wrong_environment_before_artifact_work(self) -> None:
        output = io.StringIO()
        with patch.dict("os.environ", {"CONDA_DEFAULT_ENV": "wrong-environment"}):
            with patch.object(sys, "prefix", "not-the-approved-environment"):
                with redirect_stdout(output):
                    exit_code = preflight_main(["--offline"])
        self.assertEqual(2, exit_code)
        self.assertIn("expected pypsa-earth-ec", output.getvalue())


if __name__ == "__main__":
    unittest.main()
