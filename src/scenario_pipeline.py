"""Scenario execution pipeline.

Restructured version of the original ``scenario_run_template.py`` script.
It keeps the existing workflow (load → scale → build network → validate → run)
while exposing each step as a dedicated method so that future scenarios can be
plugged in via the ``scenario_reg`` registry.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pypsa

from paths import (
    PROC_LOAD_DIR,
    PROC_NETWORKS_DIR,
    RAW_DEMANDS_DIR,
    RAW_GENERATION_DIR,
    RESULTS_DIR,
)
from paths import all_dirs
from scenarios import (
    DemandConfig,
    HydroConfig,
    NetworkConfig,
    NuclearConfig,
    REConfig,
    Scenario,
)
from scenario_reg import SCENARIOS as REGISTERED_SCENARIOS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario registry ---------------------------------------------------------
# ---------------------------------------------------------------------------

BASE_SCENARIO_2018 = Scenario(
    id="BASE_2018",
    year=2018,
    demand=DemandConfig(
        mode="projected",
        family="base",
        projection="lin",
        scaling_csv="demand_ec_2018_2027.csv",
        profile_template_name="load_base_2030_linear.csv",
        year_column="year",
    ),
    hydro=HydroConfig(case="normal"),
    network=NetworkConfig(
        base_nc="network_base.nc",
        year=2018,
        result_tag="base_2018",
    ),
    nuclear=NuclearConfig(wave="none", total_capacity_mw=0.0),
    re=REConfig(level="none"),
    description="Base 2018 scenario derived from the original template workflow.",
    validate_1day=True,
    run_full_year=True,
)

SCENARIO_REGISTRY: Dict[str, Scenario] = dict(REGISTERED_SCENARIOS)
SCENARIO_REGISTRY.setdefault(BASE_SCENARIO_2018.id, BASE_SCENARIO_2018)


# ---------------------------------------------------------------------------
# Helpers ------------------------------------------------------------------
# ---------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a simple logging handler if none exists."""

    if logging.getLogger().handlers:
        return

    log_path = Path(RESULTS_DIR) / "scenario_runs.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


@dataclass
class ScenarioPaths:
    """Convenience container for frequently used directories."""

    load_dir: Path = Path(PROC_LOAD_DIR)
    network_dir: Path = Path(PROC_NETWORKS_DIR)
    raw_demand_dir: Path = Path(RAW_DEMANDS_DIR)
    raw_generation_dir: Path = Path(RAW_GENERATION_DIR)
    results_dir: Path = Path(RESULTS_DIR)

    @classmethod
    def create(cls) -> "ScenarioPaths":
        dirs = all_dirs()  # ensures directories exist when running interactively
        _ = dirs  # intentionally unused but keeps behaviour consistent with template
        return cls()


# ---------------------------------------------------------------------------
# Pipeline -----------------------------------------------------------------
# ---------------------------------------------------------------------------


class ScenarioPipeline:
    """Encapsulates the full scenario workflow from load prep to final run."""

    DEFAULT_NETWORK_STACK: Tuple[str, ...] = (
        "network_original.nc",
        "network_snapped.nc",
        "network_expanded.nc",
        "network_expanded_no_orphans.nc",
        "network_nuclear.nc",
        "network_prod_mix.nc",
        "network_base.nc",
    )

    def __init__(
        self,
        scenario_id: str,
        paths: Optional[ScenarioPaths] = None,
    ) -> None:
        setup_logging()
        self.paths = paths or ScenarioPaths.create()
        self.scenario = self._get_scenario(scenario_id)
        self.networks: Dict[str, pypsa.Network] = {}
        self.network: Optional[pypsa.Network] = None

    # ------------------------------------------------------------------
    # Scenario management
    # ------------------------------------------------------------------
    @staticmethod
    def _get_scenario(scenario_id: str) -> Scenario:
        try:
            return SCENARIO_REGISTRY[scenario_id]
        except KeyError as exc:
            raise ValueError(f"Unknown scenario '{scenario_id}'.") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info("Starting pipeline for %s", self.scenario.id)
        profile_template = self.load_profile_template()
        load_profile = self.scale_load_template(profile_template)
        self.load_network_stack()
        self.prepare_active_network()
        self.attach_load_profiles(load_profile)
        installed = self.load_installed_generators()
        missing = self.load_future_generators()
        merged_generators = self.merge_generator_tables(installed, missing)
        merged_generators = self.apply_re_new_builds(merged_generators)
        merged_generators = self.apply_nuclear_builds(merged_generators)
        self.attach_generators(merged_generators)
        if self.scenario.validate_1day:
            self.validate_network()
            self.save_network(tag="validated")
        if self.scenario.run_full_year:
            self.run_full_year()
        self.save_network(tag="results")
        logger.info("Scenario %s finished", self.scenario.id)

    # ------------------------------------------------------------------
    # Step 1: demand profiles
    # ------------------------------------------------------------------
    def load_profile_template(self) -> pd.DataFrame:
        config = self.scenario.demand
        profile_path = self.paths.load_dir / config.profile_template_name
        if not profile_path.exists():
            raise FileNotFoundError(f"Load template not found: {profile_path}")
        logger.info("Loading load profile template from %s", profile_path)
        profile = pd.read_csv(profile_path, index_col=0, parse_dates=True)
        profile.index.name = "snapshot"
        return profile

    def load_scaling_factor(self) -> float:
        config = self.scenario.demand
        if config.mode == "historical":
            logger.info("Demand mode is historical → scaling factor = 1.0")
            return 1.0
        scaling_path = self.paths.raw_demand_dir / config.scaling_csv
        if not scaling_path.exists():
            logger.warning("Scaling CSV %s not found; using factor=1.0", scaling_path)
            return 1.0
        df = pd.read_csv(scaling_path)
        if config.year_column not in df.columns:
            raise ValueError(
                f"Column '{config.year_column}' missing in scaling file {scaling_path}"
            )
        row = df[df[config.year_column] == self.scenario.year]
        if row.empty:
            raise ValueError(
                f"Year {self.scenario.year} not present in scaling file {scaling_path}"
            )
        column_candidates = []
        if config.family and config.projection:
            column_candidates.append(f"{config.family}_{config.projection}")
        if config.family:
            column_candidates.append(config.family)
        column_candidates.append("factor")  # generic fallback
        for column in column_candidates:
            if column in df.columns:
                factor = float(row.iloc[0][column])
                logger.info(
                    "Using scaling factor %.3f from column '%s' (year %s)",
                    factor,
                    column,
                    self.scenario.year,
                )
                return factor
        raise ValueError(
            f"None of the scaling columns {column_candidates} found in {scaling_path}"
        )

    def scale_load_template(self, profile: pd.DataFrame) -> pd.DataFrame:
        factor = self.load_scaling_factor()
        logger.info("Scaling load template by factor %.3f", factor)
        return profile * factor

    # ------------------------------------------------------------------
    # Step 3: networks
    # ------------------------------------------------------------------
    def load_network_stack(self) -> None:
        """Load all available NC files to reuse metadata and coordinates."""

        logger.info("Loading available networks from %s", self.paths.network_dir)
        for name in self.DEFAULT_NETWORK_STACK:
            path = self.paths.network_dir / name
            if not path.exists():
                logger.debug("Skipping missing network file %s", path)
                continue
            try:
                self.networks[name] = pypsa.Network(str(path))
                logger.info("  ✓ loaded %s", name)
            except Exception:
                logger.exception("Failed to load network %s", path)
        config_path = self.paths.network_dir / self.scenario.network.base_nc
        if config_path.name not in self.networks:
            logger.info("Loading scenario network %s", config_path)
            self.networks[config_path.name] = pypsa.Network(str(config_path))
        logger.info("Loaded %d network variants", len(self.networks))

    def prepare_active_network(self) -> None:
        base_name = self.scenario.network.base_nc
        try:
            raw_network = self.networks[base_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Scenario base network {base_name} was not loaded."
            ) from exc
        self.network = prune_network_min_voltage(raw_network, v_threshold_kv=137)
        logger.info(
            "Active network has %d buses, %d lines, %d loads",
            len(self.network.buses),
            len(self.network.lines),
            len(self.network.loads),
        )

    def attach_load_profiles(self, load_profile: pd.DataFrame) -> None:
        if self.network is None:
            raise RuntimeError("Network must be loaded before attaching loads.")
        logger.info("Attaching load profiles to network")
        load_profile.columns = load_profile.columns.astype(str)
        buses = self.network.buses.copy()
        buses.index = buses.index.astype(str)
        load_buses = self._remap_load_buses(load_profile.columns.tolist(), buses)
        self.network.madd(
            "Load",
            load_profile.columns.astype(str).tolist(),
            bus=load_buses,
            p_set=load_profile,
        )
        logger.info(
            "Total peak demand %.2f GW",
            self.network.loads_t.p_set.sum(axis=1).max() / 1e3,
        )

    def _remap_load_buses(
        self, load_columns: List[str], buses: pd.DataFrame
    ) -> List[str]:
        existing = set(buses.index)
        coords = buses[[c for c in ["lon", "lat"] if c in buses.columns]]
        if coords.empty:
            coords = buses[[c for c in ["x", "y"] if c in buses.columns]]
        if coords.empty:
            raise ValueError("Network buses lack coordinate columns (lon/lat or x/y)")
        arr = coords.to_numpy()
        from scipy.spatial import cKDTree as KDTree

        tree = KDTree(arr)
        assignments: List[str] = []
        for name in load_columns:
            if name in existing:
                assignments.append(name)
                continue
            idx = int(tree.query(arr.mean(axis=0))[1])
            assignments.append(buses.index[idx])
            logger.debug("Mapped missing load '%s' to bus '%s'", name, buses.index[idx])
        return assignments

    # ------------------------------------------------------------------
    # Step 4: generation tables
    # ------------------------------------------------------------------
    def load_installed_generators(self) -> pd.DataFrame:
        path = self.paths.raw_generation_dir / "generation_filtered.xlsx"
        if not path.exists():
            logger.warning("Installed generation file %s not found", path)
            return pd.DataFrame(columns=["name", "bus", "carrier", "p_nom"])
        logger.info("Reading installed generators from %s", path)
        df = pd.read_excel(path)
        return self._normalize_generation_table(df, tag="installed")

    def load_future_generators(self) -> pd.DataFrame:
        path = self.paths.raw_generation_dir / "generation_future.xlsx"
        if not path.exists():
            logger.info("Future generation file %s not found", path)
            return pd.DataFrame(columns=["name", "bus", "carrier", "p_nom"])
        logger.info("Reading future generators from %s", path)
        df = pd.read_excel(path)
        return self._normalize_generation_table(df, tag="future")

    def _normalize_generation_table(
        self,
        df: pd.DataFrame,
        tag: str,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        renamed = df.rename(
            columns={
                "Name": "name",
                "Plant": "name",
                "Bus": "bus",
                "bus": "bus",
                "Technology": "carrier",
                "Carrier": "carrier",
                "Capacity": "p_nom",
                "p_nom (MW)": "p_nom",
                "MW": "p_nom",
            }
        )
        needed = ["name", "bus", "carrier", "p_nom"]
        missing = [c for c in needed if c not in renamed.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} missing after normalising {tag} generation table"
            )
        renamed["name"] = renamed["name"].astype(str)
        renamed["bus"] = renamed["bus"].astype(str)
        renamed["carrier"] = renamed["carrier"].astype(str)
        renamed["p_nom"] = pd.to_numeric(renamed["p_nom"], errors="coerce").fillna(0.0)
        renamed = renamed[renamed["p_nom"] > 0.0]
        logger.info("Normalised %d %s generator entries", len(renamed), tag)
        return renamed

    def merge_generator_tables(
        self,
        installed: pd.DataFrame,
        future: pd.DataFrame,
    ) -> pd.DataFrame:
        if installed.empty and future.empty:
            return pd.DataFrame(columns=["name", "bus", "carrier", "p_nom"])
        merged = pd.concat([installed, future], ignore_index=True)
        merged = merged.drop_duplicates(subset="name", keep="last")
        logger.info("Merged generator table has %d entries", len(merged))
        return merged

    # ------------------------------------------------------------------
    # Step 5: Renewables (placeholder)
    # ------------------------------------------------------------------
    def apply_re_new_builds(self, generators: pd.DataFrame) -> pd.DataFrame:
        if self.scenario.re.level == "none":
            return generators
        logger.info(
            "Applying renewable build-out level '%s' (placeholder)",
            self.scenario.re.level,
        )
        # Placeholder: scale existing renewable entries according to configured factors.
        out = generators.copy()
        for carrier, factor in [
            ("onwind", self.scenario.re.onwind_factor),
            ("solar", self.scenario.re.solar_factor),
        ]:
            if factor <= 0 or carrier not in out["carrier"].unique():
                continue
            mask = out["carrier"].str.lower() == carrier
            out.loc[mask, "p_nom"] *= 1.0 + factor
        return out

    # ------------------------------------------------------------------
    # Step 6: Nuclear additions
    # ------------------------------------------------------------------
    def apply_nuclear_builds(self, generators: pd.DataFrame) -> pd.DataFrame:
        config = self.scenario.nuclear
        if config.total_capacity_mw <= 0:
            return generators
        logger.info(
            "Adding nuclear capacity %.1f MW (wave %s)",
            config.total_capacity_mw,
            config.wave,
        )
        if self.network is None:
            raise RuntimeError("Network must be loaded before adding nuclear units")
        bus = self.network.buses.index[0]
        n_units = config.n_units or 1
        unit_capacity = config.unit_capacity_mw or (
            config.total_capacity_mw / max(n_units, 1)
        )
        nuclear = pd.DataFrame(
            {
                "name": [f"nuclear_{config.wave}_{i+1}" for i in range(n_units)],
                "bus": [bus] * n_units,
                "carrier": ["nuclear"] * n_units,
                "p_nom": [unit_capacity] * n_units,
            }
        )
        return pd.concat([generators, nuclear], ignore_index=True)

    # ------------------------------------------------------------------
    # Step 7: Attach generation
    # ------------------------------------------------------------------
    def attach_generators(self, generators: pd.DataFrame) -> None:
        if self.network is None:
            raise RuntimeError("Network must be loaded before attaching generators")
        if generators.empty:
            logger.warning("Generator table empty; no units added to network")
            return
        logger.info("Attaching %d generators to the network", len(generators))
        self.network.madd(
            "Generator",
            generators["name"].astype(str).tolist(),
            bus=generators["bus"].astype(str).tolist(),
            carrier=generators["carrier"].astype(str).tolist(),
            p_nom=generators["p_nom"].values,
        )

    # ------------------------------------------------------------------
    # Step 8: Validation and full run
    # ------------------------------------------------------------------
    def validate_network(self) -> None:
        if self.network is None:
            raise RuntimeError("Network must be ready before validation")
        logger.info("Validating scenario via 1-day optimization")
        tmp = self.network.copy()
        snapshots = tmp.snapshots[:24]
        tmp.optimize(snapshots=snapshots)
        logger.info("Validation run completed")

    def run_full_year(self) -> None:
        if self.network is None:
            raise RuntimeError("Network must be ready before optimization")
        logger.info("Running full-year optimization")
        self.network.optimize()
        logger.info("Full-year optimization completed")

    def save_network(self, tag: str) -> None:
        if self.network is None:
            return
        result_dir = self.paths.results_dir / "networks"
        result_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{self.scenario.id.lower()}_{tag}.nc"
        target = result_dir / file_name
        logger.info("Saving network to %s", target)
        self.network.export_to_netcdf(target)


# ---------------------------------------------------------------------------
# Utilities -----------------------------------------------------------------
# ---------------------------------------------------------------------------


def prune_network_min_voltage(n: pypsa.Network, v_threshold_kv: float = 137.0) -> pypsa.Network:
    """Replicates the pruning helper from the template script."""

    m = n.copy()
    keep = m.buses.index[m.buses["v_nom"] >= float(v_threshold_kv)].astype(str)
    drop = set(m.buses.index.astype(str)) - set(keep)
    logger.info(
        "Voltage pruning keeps %d/%d buses (threshold %.1f kV)",
        len(keep),
        len(m.buses),
        v_threshold_kv,
    )
    component_map = {
        "Load": "loads",
        "Generator": "generators",
        "Store": "stores",
        "StorageUnit": "storage_units",
        "ShuntImpedance": "shunt_impedances",
        "Line": "lines",
        "Transformer": "transformers",
        "Link": "links",
    }
    for component, attr in component_map.items():
        table = getattr(m, attr, None)
        if table is None or table.empty:
            continue
        if component in {"Line", "Transformer", "Link"}:
            cols = ["bus0", "bus1"]
        else:
            cols = ["bus"]
        mask = np.zeros(len(table), dtype=bool)
        for col in cols:
            if col not in table.columns:
                continue
            mask |= ~table[col].astype(str).isin(keep)
        for name in table.index[mask]:
            try:
                m.remove(component, name)
            except Exception:
                logger.exception("Failed to remove %s '%s' during pruning", component, name)
    m.buses = m.buses.loc[keep]
    return m


# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Scenario pipeline runner")
    parser.add_argument(
        "scenario_id",
        nargs="?",
        default=BASE_SCENARIO_2018.id,
        help="Scenario ID to execute (default: BASE_2018)",
    )
    args = parser.parse_args(args=list(argv) if argv is not None else None)
    pipeline = ScenarioPipeline(args.scenario_id)
    pipeline.run()


if __name__ == "__main__":
    main()
