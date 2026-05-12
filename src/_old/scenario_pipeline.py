"""Scenario execution pipeline.

Restructured version of the original ``scenario_run_template.py`` script.
It keeps the existing workflow (load → scale → build network → validate → run)
while exposing each step as a dedicated method so that future scenarios can be
plugged in via the ``scenario_reg`` registry.
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from helpers import ScenarioPaths, setup_logging, patch_transformers_all, prune_network_min_voltage
from scenarios import (
    DemandConfig,
    HydroConfig,
    NetworkConfig,
    NuclearConfig,
    REConfig,
    Scenario,
)
from EcuadorElectricGrid.src._old.scenario_reg import SCENARIOS as REGISTERED_SCENARIOS
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
        scaling_csv="demand_projection_2018_2050_all_scenarios.csv",
        profile_template_name="load_base_2030_linear.csv",
        year_column="year",
    ),
    hydro=HydroConfig(case="normal"),
    network=NetworkConfig(
        base_nc="network_base.nc",
        year=2018,
        result_tag=f"base_2018_linear",
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
        self.installed_plants: Optional[pd.DataFrame] = None

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
        #missing = self.load_future_generators()
        #merged_generators = self.merge_generator_tables(installed, missing)
        merged_generators = self.apply_re_new_builds(merged_generators)
        merged_generators = self.apply_nuclear_builds(merged_generators)
        self.attach_generators(merged_generators)
        self.apply_network_postprocessing()
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

    def load_scaling_factor(self, ) -> float:
        # first get the whole year demand of the loaded profile

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
        total_demand_profile = profile.sum().sum()
        scenario_year_demand = self.load_scaling_factor()
        factor = scenario_year_demand / total_demand_profile
        logger.info("Scaling load template by factor %.3f", factor)
        logger.info("Total year demand of loaded profile %.3f", total_demand_profile)
        logger.info("Total year demand of scenario %.3f", total_demand_profile)
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
        path = self.paths.processed_generation_dir / "powerplants_all.csv"
        if not path.exists():
            logger.warning("Power plants CSV not found: %s", path)
            return pd.DataFrame(columns=["name", "bus", "carrier", "p_nom"])

        logger.info("Reading installed power plants from %s", path)
        raw = pd.read_csv(path, index_col=0)
        normalized = self._apply_powerplantmatching_normalization(raw)
        normalized = self._normalize_powerplant_fields(normalized)
        filtered = self._filter_powerplants(normalized)
        attached = self._attach_plants_to_lv_buses(filtered)
        self.installed_plants = attached.copy()

        needed = ["name", "bus", "carrier", "p_nom"]
        missing = [col for col in needed if col not in attached.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} missing after processing installed generators"
            )

        result = attached[needed].copy()
        logger.info("Prepared %d installed generators after normalization", len(result))
        return result


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

    def _apply_powerplantmatching_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        ppl = df.copy()
        try:
            import powerplantmatching as pm  # type: ignore

            config_path = (
                Path(__file__).resolve().parents[2]
                / "pypsa-earth"
                / "configs"
                / "powerplantmatching_config.yaml"
            )
            config = pm.get_config(str(config_path))
            config["target_countries"] = ["EC"]
            ppl_f = ppl.powerplant.fill_missing_commissioning_years()
            ppl_p = ppl_f.powerplant.to_pypsa_names()
            ppl = ppl_p.dropna(axis=1, how="all")
            logger.info("Applied powerplantmatching normalization (fill years + names).")
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("powerplantmatching step skipped (using raw CSV): %s", exc)
        return ppl

    def _normalize_powerplant_fields(self, plants: pd.DataFrame) -> pd.DataFrame:
        ppl = plants.copy()
        if "name" not in ppl.columns:
            ppl["name"] = ppl.index.astype(str)
        else:
            ppl["name"] = ppl["name"].astype(str)
        if "p_nom" in ppl.columns:
            ppl["p_nom"] = (
                ppl["p_nom"].astype(str).str.replace("'", "", regex=False).str.replace(",", ".", regex=False)
            )
            ppl["p_nom"] = pd.to_numeric(ppl["p_nom"], errors="coerce").fillna(0.0)
        else:
            logger.warning("Column 'p_nom' not found in plants table after normalization.")
        if "carrier" in ppl.columns:
            ppl["carrier_norm"] = ppl["carrier"].apply(self._normalize_text)
            carrier_map_es2en = {
                "hidraulica": "hydro",
                "hidroelectrico": "hydro",
                "termica": "oil",
                "termoelectrico": "oil",
                "biomasa": "biomass",
                "fotovoltaica": "solar",
                "solar": "solar",
                "eolica": "onwind",
                "eolico": "onwind",
                "biogas": "biomass",
                "ernc": "onwind",
                "nuclear": "nuclear",
                "geotermica": "geothermal",
                "carbon": "coal",
                "carbón": "coal",
                "lignito": "lignite",
                "gas natural": "natural gas",
                "gas": "natural gas",
                "phs": "PHS",
                "pasada": "ror",
                "embalse": "hydro",
                "offshore": "offwind-ac",
                "offshore_dc": "offwind-dc",
            }
            ppl["carrier"] = ppl["carrier_norm"].map(carrier_map_es2en).fillna(ppl["carrier_norm"])
        else:
            logger.warning("Column 'carrier' not found in plants table after normalization.")
        return ppl

    def _filter_powerplants(self, ppl: pd.DataFrame) -> pd.DataFrame:
        filtering_power = 0.0
        date_in = 2017
        ppl_connected = None
        ppl_connected_current = None
        if {"component", "DateIn", "p_nom"}.issubset(ppl.columns):
            ppl_connected = ppl[ppl["component"] == "S.N.I."]
            ppl_connected_current = ppl_connected[ppl_connected["DateIn"] <= date_in]
            ppl_filtered = ppl_connected_current[
                ppl_connected_current["p_nom"] >= filtering_power
            ].copy()
        else:
            logger.warning("Required columns for filtering missing; skipping filter.")
            ppl_filtered = ppl.copy()

        def safe_sum(series: pd.Series) -> float:
            try:
                return float(series.sum())
            except Exception:
                return float("nan")

        total_capacity = safe_sum(ppl.get("p_nom", pd.Series(dtype=float)))
        total_connected = (
            safe_sum(ppl_connected.get("p_nom", pd.Series(dtype=float)))
            if ppl_connected is not None
            else float("nan")
        )
        total_connected_current = (
            safe_sum(ppl_connected_current.get("p_nom", pd.Series(dtype=float)))
            if ppl_connected_current is not None
            else float("nan")
        )
        total_filtered = safe_sum(ppl_filtered.get("p_nom", pd.Series(dtype=float)))
        logger.info(
            "Capacities [kW]\n"
            f"  total:                {total_capacity:,.0f}\n"
            f"  connected (SNI):      {total_connected:,.0f}\n"
            f"  connected <= {date_in}: {total_connected_current:,.0f}\n"
            f"  filtered (>= {filtering_power:.0f} kW): {total_filtered:,.0f}"
        )
        return ppl_filtered

    @staticmethod
    def _normalize_text(value: object) -> Optional[str]:
        if pd.isna(value):
            return None
        text = str(value).strip().lower()
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"\s+", " ", text)
        return text or None

    def _attach_plants_to_lv_buses(self, plants: pd.DataFrame) -> pd.DataFrame:
        if self.network is None or plants.empty:
            return plants
        if "substation_lv" not in self.network.buses.columns:
            logger.warning("Network is missing 'substation_lv' column; skipping LV attachment.")
            return plants
        if not {"lon", "lat"}.issubset(plants.columns):
            logger.warning("Plant table lacks 'lon'/'lat' columns; skipping LV attachment.")
            return plants
        lv_mask = self.network.buses["substation_lv"].astype(bool)
        sub_idx = self.network.buses.index[lv_mask]
        if sub_idx.empty:
            logger.warning("No LV substations found (substation_lv == True).")
            return plants
        bus_coords_cols = [c for c in ["lon", "lat"] if c in self.network.buses.columns]
        if len(bus_coords_cols) < 2:
            bus_coords_cols = [c for c in ["x", "y"] if c in self.network.buses.columns]
        if len(bus_coords_cols) < 2:
            logger.warning("Network buses lack coordinate columns; skipping LV attachment.")
            return plants
        from scipy.spatial import cKDTree as KDTree

        bus_coords = self.network.buses.loc[sub_idx, bus_coords_cols].astype(float)
        tree = KDTree(bus_coords.values)
        coords_mask = plants[["lon", "lat"]].notna().all(axis=1)
        if not coords_mask.all():
            logger.warning(
                "Skipping %d plants without valid coordinates for LV attachment.",
                (~coords_mask).sum(),
            )
        valid_plants = plants.loc[coords_mask].copy()
        if valid_plants.empty:
            return plants
        plant_coords = valid_plants[["lon", "lat"]].astype(float)
        nearest_idx = tree.query(plant_coords.values)[1]
        attached = plants.copy()
        attached.loc[valid_plants.index, "bus"] = sub_idx[nearest_idx].astype(str).values
        missing = attached["bus"].isna()
        if missing.any():
            logger.warning("Found %d plants without LV bus assignment.", missing.sum())
        attached = attached[~missing].copy()
        attached.loc[:, "bus"] = attached["bus"].astype(str)
        logger.info("Attached %d plants to nearest LV buses", len(attached))
        return attached

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

    def apply_network_postprocessing(self) -> None:
        if self.network is None:
            return
        self.ensure_load_shedding_generators()
        patch_transformers_all(self.network)
        self.fix_line_impedances()
        self.assign_renewable_profiles()

    def ensure_load_shedding_generators(self) -> None:
        if self.network is None:
            return
        carriers_cols = [
            "co2_emissions",
            "color",
            "nice_name",
            "max_growth",
            "max_relative_growth",
        ]
        if "load_shedding" not in self.network.carriers.index:
            self.network.carriers.loc["load_shedding", carriers_cols] = [
                0.0,
                "#DD1818",
                "Load Shedding",
                np.inf,
                0.0,
            ]
        marginal_cost = 100e4
        capital_cost = 1e10
        missing_names: List[str] = []
        missing_buses: List[str] = []
        for bus in self.network.buses.index.astype(str):
            name = f"LS_{bus}"
            if name not in self.network.generators.index:
                missing_names.append(name)
                missing_buses.append(bus)
        if missing_names:
            self.network.madd(
                "Generator",
                names=missing_names,
                bus=missing_buses,
                carrier="load_shedding",
                p_nom=10000.0,
                marginal_cost=marginal_cost,
                capital_cost=capital_cost,
            )
            logger.info("Added %d load-shedding generators (cost=%s).", len(missing_names), marginal_cost)

    def fix_line_impedances(self) -> None:
        if self.network is None or self.network.lines.empty:
            return
        for col, value in {"x": 0.1, "r": 0.01}.items():
            if col not in self.network.lines.columns:
                self.network.lines[col] = value
            else:
                self.network.lines.loc[:, col] = value
        logger.info("Patched line impedances: set x=0.1, r=0.01 for all lines.")

    def assign_renewable_profiles(self) -> None:
        if self.network is None or self.network.generators.empty:
            return
        if "carrier" not in self.network.generators.columns:
            logger.warning("Generator table lacks 'carrier'; cannot assign renewable profiles.")
            return
        carriers = {"solar", "onwind"}
        mask = self.network.generators["carrier"].astype(str).str.lower().isin(carriers)
        ren_gens = self.network.generators.loc[mask]
        if ren_gens.empty:
            return
        if not {"lon", "lat"}.issubset(self.network.buses.columns):
            logger.warning("Bus table lacks lon/lat; cannot build renewable profiles.")
            return
        centroids_path = self.paths.raw_gadm_dir / "EC_centroids.csv"
        if not centroids_path.exists():
            logger.warning("Centroid CSV not found at %s; skipping renewable profiles.", centroids_path)
            return
        solar_path = self.paths.raw_cutouts_dir / "solar.nc"
        wind_path = self.paths.raw_cutouts_dir / "onwind.nc"
        if not solar_path.exists() or not wind_path.exists():
            logger.warning("Cutout files missing; expected %s and %s", solar_path, wind_path)
            return
        try:
            centroids = pd.read_csv(centroids_path)
        except Exception as exc:
            logger.warning("Failed to read centroid CSV: %s", exc)
            return
        for col in ["province", "longitude", "latitude"]:
            if col not in centroids.columns:
                logger.warning("Centroid CSV missing column '%s'", col)
                return
        centroids["province_norm"] = centroids["province"].map(self._norm_province_name)
        cent_lon = centroids["longitude"].astype(float).values
        cent_lat = centroids["latitude"].astype(float).values
        cent_name_norm = centroids["province_norm"].values
        try:
            solar_ds = xr.open_dataset(solar_path)
            wind_ds = xr.open_dataset(wind_path)
        except Exception as exc:
            logger.warning("Failed to open cutout datasets: %s", exc)
            return
        try:
            provinces_raw = pd.Index(solar_ds.bus.data)
            provinces_norm = provinces_raw.to_series().map(self._norm_province_name)
            norm_to_raw = dict(zip(provinces_norm.values, provinces_raw.values))
            solar_profiles = solar_ds["profile"].to_pandas()
            solar_profiles.columns = provinces_raw
            wind_profiles = wind_ds["profile"].to_pandas()
            wind_profiles.columns = provinces_raw
            solar_profiles = solar_profiles.reindex(index=self.network.snapshots)
            wind_profiles = wind_profiles.reindex(index=self.network.snapshots)
        finally:
            solar_ds.close()
            wind_ds.close()
        if "province" not in self.network.generators.columns:
            self.network.generators["province"] = pd.NA
        p_max_df = self._ensure_generator_p_max_table()

        def find_province_for_coord(lon: float, lat: float) -> Optional[str]:
            d2 = (cent_lon - lon) ** 2 + (cent_lat - lat) ** 2
            idx = int(np.argmin(d2))
            return cent_name_norm[idx] if len(cent_name_norm) else None

        assigned = 0
        for gen_id, row in ren_gens.iterrows():
            bus_id = row["bus"]
            try:
                bus_lon = float(self.network.buses.at[bus_id, "lon"])
                bus_lat = float(self.network.buses.at[bus_id, "lat"])
            except Exception:
                logger.warning("Bus %s missing lon/lat; skipping %s", bus_id, gen_id)
                continue
            prov_norm = find_province_for_coord(bus_lon, bus_lat)
            if prov_norm is None:
                logger.warning("No centroid match for generator %s", gen_id)
                continue
            if prov_norm not in norm_to_raw:
                logger.warning(
                    "Province '%s' not in cutout provinces; skipping %s",
                    prov_norm,
                    gen_id,
                )
                continue
            prov_raw = norm_to_raw[prov_norm]
            carrier = str(row["carrier"]).lower()
            if carrier == "solar":
                prof = solar_profiles[prov_raw].astype(float)
            elif carrier == "onwind":
                prof = wind_profiles[prov_raw].astype(float)
            else:
                continue
            prof = prof.reindex(self.network.snapshots)
            max_val = prof.max()
            if pd.notna(max_val) and max_val > 0:
                prof = (prof / max_val).fillna(0.0)
            else:
                prof = pd.Series(0.0, index=self.network.snapshots)
                logger.warning(
                    "Profile for %s (province %s) has non-positive max; assigning zeros.",
                    gen_id,
                    prov_raw,
                )
            p_max_df.loc[:, gen_id] = prof.values
            self.network.generators.at[gen_id, "province"] = prov_raw
            assigned += 1
        self.network.generators_t["p_max_pu"] = p_max_df
        if assigned:
            logger.info("Assigned renewable profiles to %d generators.", assigned)

    def _ensure_generator_p_max_table(self) -> pd.DataFrame:
        if self.network is None:
            raise RuntimeError("Network must be loaded before ensuring p_max_pu table")
        if "p_max_pu" not in self.network.generators_t:
            df = pd.DataFrame(
                1.0,
                index=self.network.snapshots,
                columns=self.network.generators.index,
            )
            self.network.generators_t["p_max_pu"] = df
            return df
        df = self.network.generators_t["p_max_pu"]
        df = df.reindex(index=self.network.snapshots, columns=self.network.generators.index, fill_value=1.0)
        self.network.generators_t["p_max_pu"] = df
        return df

    @staticmethod
    def _norm_province_name(name: object) -> str:
        text = str(name)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.strip().lower().replace(" ", "_").replace("-", "_")
        while "__" in text:
            text = text.replace("__", "_")
        return text

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
