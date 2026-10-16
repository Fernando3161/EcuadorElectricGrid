from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pypsa

from src.assembly import (
    assemble_2022_network,
    assert_reload_equivalent,
    evaluate_attachment_materiality,
    read_hourly_table,
    summarize_attachment_screen,
)
from src.topology import (
    SUPPORTED_VOLTAGES_KV,
    build_2022_topology,
    load_repair_table,
)
from src.topology_plots import VOLTAGE_COLORS, load_mainland_provinces, plot_topology_voltage_map


ROOT = Path(__file__).resolve().parents[1]


class Phase05TopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = pypsa.Network(ROOT / "data/raw/networks/base.nc")
        cls.repairs = load_repair_table(
            ROOT / "data/interim/network/topology_repairs_2022.csv"
        )
        cls.result = build_2022_topology(cls.base, cls.repairs)

    def test_supported_voltage_boundary_and_structural_fingerprint(self) -> None:
        self.assertEqual(set(self.result.network.buses.v_nom), set(SUPPORTED_VOLTAGES_KV))
        self.assertEqual(
            self.result.fingerprint,
            {
                "snapshots": 8760,
                "buses": 110,
                "lines": 111,
                "transformers": 19,
                "loads": 0,
                "generators": 0,
                "bus_voltage_counts": {"138.0": 64, "230.0": 40, "500.0": 6},
                "connected_components": 4,
            },
        )

    def test_reviewed_repairs_are_explicit_and_complete(self) -> None:
        self.assertEqual(len(self.repairs.query("component == 'Line'")), 5)
        self.assertEqual(set(self.repairs.official_fact.astype(str).str.lower()), {"false"})
        self.assertFalse(self.repairs.rationale.str.strip().eq("").any())
        self.assertFalse(self.repairs.evidence_locator.str.strip().eq("").any())
        added = self.result.mutation_audit.query("mutation == 'add_reviewed_line'")
        self.assertEqual(set(added.name), set(self.repairs.query("component == 'Line'").repair_id))
        patched = self.result.mutation_audit.query(
            "mutation == 'patch_transformer_resistance'"
        )
        self.assertEqual(len(patched), 19)
        self.assertTrue((self.result.network.transformers.r > 0).all())

    def test_orphan_and_voltage_mutations_are_audited(self) -> None:
        counts = self.result.mutation_audit.mutation.value_counts().to_dict()
        self.assertEqual(counts["remove_orphan_bus"], 18)
        self.assertEqual(counts["remove_out_of_scope_bus"], 185)
        self.assertEqual(counts["snap_voltage"], 12)

    def test_islands_are_reported_not_silently_bridged(self) -> None:
        self.assertEqual(self.result.component_summary.n_buses.tolist(), [103, 3, 2, 2])


class Phase05AssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base = pypsa.Network(ROOT / "data/raw/networks/base.nc")
        topology = build_2022_topology(
            base,
            load_repair_table(ROOT / "data/interim/network/topology_repairs_2022.csv"),
        ).network
        cls.demand = read_hourly_table(
            ROOT / "data/processed/demand/demand_profiles_2022_mw.csv"
        )
        cls.generators = pd.read_csv(
            ROOT / "data/processed/generation/generators_2022.csv"
        )
        hydro = read_hourly_table(
            ROOT / "data/processed/generation/hydro_availability_2022.csv"
        )
        renewable = read_hourly_table(
            ROOT / "data/processed/generation/renewable_availability_2022.csv"
        )
        costs = pd.read_csv(
            ROOT / "data/raw/generation/technology_cost.csv", comment="#"
        )
        carriers = pd.read_csv(
            ROOT / "data/raw/generation/carriers.csv", index_col=0
        )
        connection_evidence = pd.read_csv(
            ROOT / "data/interim/network/generator_connection_evidence_2022.csv"
        )
        load_connection_evidence = pd.read_csv(
            ROOT / "data/interim/network/load_connection_evidence_2022.csv"
        )
        cls.result = assemble_2022_network(
            topology,
            base.buses,
            cls.demand,
            cls.generators,
            hydro,
            renewable,
            costs,
            carriers,
            connection_evidence=connection_evidence,
            load_connection_evidence=load_connection_evidence,
        )

    def test_attachment_is_complete_and_demand_is_conserved(self) -> None:
        self.assertEqual(len(self.result.load_attachments), self.demand.shape[1])
        self.assertEqual(len(self.result.generator_attachments), len(self.generators))
        self.assertTrue((self.result.load_attachments.distance_km >= 0).all())
        self.assertTrue((self.result.generator_attachments.distance_km >= 0).all())
        self.assertAlmostEqual(
            self.result.network.loads_t.p_set.to_numpy().sum() / 1000,
            27561.62,
            places=6,
        )

    def test_flagged_load_investigation_is_complete_and_non_mutating(self) -> None:
        investigation = pd.read_csv(
            ROOT / "data/interim/network/load_connection_investigation_2022.csv"
        )
        self.assertEqual(len(investigation), 12)
        self.assertEqual(set(investigation.source_voltage_kv), {69})
        self.assertEqual(investigation.load_id.nunique(), 12)
        self.assertFalse(investigation.evidence_source.str.strip().eq("").any())
        self.assertAlmostEqual(investigation.annual_consumption_gwh.sum(), 2492.96059, places=5)
        self.assertEqual(
            set(investigation.query("upstream_connectivity == 'orphan_in_upstream_model'").load_id),
            {"load_180", "load_241"},
        )
        self.assertEqual(
            dict(zip(investigation.load_id, investigation.recommended_target_bus))["load_241"],
            102,
        )
        self.assertEqual(
            dict(zip(investigation.load_id, investigation.recommended_target_bus))["load_262"],
            52,
        )

    def test_reviewed_load_connections_are_applied_and_traceable(self) -> None:
        attachments = self.result.load_attachments.set_index("load_id")
        expected = {
            "load_126": "42", "load_127": "28", "load_179": "42",
            "load_180": "42", "load_194": "50", "load_195": "50",
            "load_205": "38", "load_208": "64", "load_220": "24",
            "load_241": "102", "load_262": "52", "load_287": "42",
        }
        self.assertEqual(
            attachments.loc[list(expected), "target_bus"].astype(str).to_dict(), expected
        )
        self.assertAlmostEqual(attachments.loc["load_262", "distance_km"], 56.653993, places=5)
        self.assertEqual(
            attachments.loc["load_262", "connection_review_status"],
            "not_acceptable_requires_source_validation",
        )
        self.assertEqual(
            set(attachments.loc[list(expected), "connection_source"].isna()), {False}
        )
        self.assertTrue(attachments.loc[list(expected), "modeling_assumption"].all())
        self.assertFalse(attachments.loc[list(expected), "official_fact"].any())

    def test_fleet_capacity_profiles_and_nuclear_exclusion(self) -> None:
        network = self.result.network
        real = network.generators.query("carrier != 'load_shedding'")
        self.assertEqual(len(real), 133)
        self.assertAlmostEqual(real.p_nom.sum(), 7489.65)
        self.assertNotIn("nuclear", set(network.generators.carrier))
        self.assertNotIn("nuclear", set(network.carriers.index))
        self.assertEqual(len(network.generators_t.p_max_pu.columns), 133)
        self.assertTrue(np.isfinite(network.generators_t.p_max_pu.to_numpy()).all())

    def test_load_shedding_is_explicit_and_separate(self) -> None:
        emergency = self.result.network.generators.query("carrier == 'load_shedding'")
        self.assertEqual(len(emergency), len(self.result.network.buses))
        self.assertTrue((emergency.marginal_cost == 1_000_000.0).all())

    def test_attachment_materiality_screen_is_size_dependent_and_non_mutating(self) -> None:
        before_load_buses = self.result.load_attachments.target_bus.copy()
        before_generator_buses = self.result.generator_attachments.target_bus.copy()
        load_screen, generator_screen = evaluate_attachment_materiality(
            self.result.load_attachments,
            self.result.generator_attachments,
            self.demand,
            self.generators,
        )
        self.assertEqual(set(load_screen.screening_status), {"acceptable", "not acceptable - fix this bus"})
        self.assertEqual(set(generator_screen.screening_status), {"acceptable", "not acceptable - fix this bus"})
        self.assertAlmostEqual(load_screen.annual_consumption_gwh.sum(), 27561.62, places=6)
        self.assertAlmostEqual(generator_screen.p_nom_mw.sum(), 7489.65, places=6)
        esmeraldas = generator_screen[generator_screen.name.isin(["Esmeraldas I", "Esmeraldas II"])]
        self.assertEqual(set(esmeraldas.target_bus.astype(str)), {"42"})
        self.assertEqual(set(esmeraldas.attachment_method), {"verified_physical_connection"})
        self.assertTrue((esmeraldas.connection_voltage_kv == 138.0).all())
        self.assertTrue((esmeraldas.distance_km < 2.5).all())
        self.assertTrue((esmeraldas.screening_status == "acceptable").all())
        self.assertTrue(before_load_buses.equals(self.result.load_attachments.target_bus))
        self.assertTrue(before_generator_buses.equals(self.result.generator_attachments.target_bus))
        summary = summarize_attachment_screen(
            generator_screen, "capacity_band", "p_nom_mw"
        )
        self.assertEqual(summary.attachments.sum(), 133)

    def test_serialization_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.nc"
            self.result.network.export_to_netcdf(path)
            assert_reload_equivalent(self.result.network, pypsa.Network(path))


class Phase05NotebookContractTests(unittest.TestCase):
    def test_supported_notebooks_are_explanatory_output_free_and_preflighted(self) -> None:
        for filename, stage in (
            ("04_build_and_validate_2022_topology.ipynb", "04_topology"),
            ("05_assemble_and_smoke_test_2022_network.ipynb", "05_assembly"),
        ):
            notebook = nbformat.read(ROOT / "notebooks" / filename, as_version=4)
            code = [cell for cell in notebook.cells if cell.cell_type == "code"]
            markdown = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
            self.assertGreaterEqual(len(markdown), 4)
            self.assertIn("run_preflight", code[0].source)
            self.assertIn(stage, code[0].source)
            self.assertTrue(all(not cell.outputs for cell in code))
            self.assertTrue(all(cell.execution_count is None for cell in code))
        topology_notebook = nbformat.read(
            ROOT / "notebooks/04_build_and_validate_2022_topology.ipynb", as_version=4
        )
        self.assertIn(
            "save_phase05_topology_maps",
            "\n".join(cell.source for cell in topology_notebook.cells),
        )


class Phase05TopologyPlotTests(unittest.TestCase):
    def test_owner_voltage_palette_and_single_voltage_render(self) -> None:
        self.assertEqual(
            VOLTAGE_COLORS,
            {138.0: "#377eb8", 230.0: "#a65628", 500.0: "#00008B"},
        )
        base = pypsa.Network(ROOT / "data/raw/networks/base.nc")
        topology = build_2022_topology(
            base,
            load_repair_table(ROOT / "data/interim/network/topology_repairs_2022.csv"),
        ).network
        provinces = load_mainland_provinces(
            ROOT / "data/raw/gadm/gadm41_ECU.gpkg"
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "topology_500kv.png"
            figure, _ = plot_topology_voltage_map(
                topology,
                provinces,
                voltage_kv=500,
                annotate_bus_ids=True,
                save_path=destination,
                dpi=80,
            )
            self.assertTrue(destination.is_file())
            self.assertGreater(destination.stat().st_size, 1000)
            import matplotlib.pyplot as plt

            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
