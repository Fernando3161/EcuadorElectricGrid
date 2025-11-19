from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
from os.path import join

from scenarios import Scenario
from paths import all_dirs
from paths import (
    PROC_GENERATION_DIR,
    PROC_LOAD_DIR,
    PROC_NETWORKS_DIR,
    RAW_CUTOUTS_DIR,
    RAW_DEMANDS_DIR,
    RAW_GADM_DIR,
    RAW_GENERATION_DIR,
    RESULTS_DIR,
)
# =============================================================================
# Helper accessors (optional but handy)
# =============================================================================


def get_demand_scaling_factor(scenario: Scenario, dirs: Dict[str, str]) -> float:
    """
    Use scenario.demand to get the scalar factor from the CSV.

    CSV format:

        year,trend_lin,trend_exp,base_lin,base_exp,productive_mix_lin,productive_mix_exp
        ...

    - For mode = "historical": returns 1.0.
    - For mode = "projected": builds column name "<family>_<projection>".
    """
    dcfg = scenario.demand

    if dcfg.mode == "historical":
        return 1.0

    if dcfg.family is None or dcfg.projection is None:
        raise ValueError(
            f"DemandConfig for scenario {scenario.id} has mode='projected' "
            f"but family/projection not set."
        )

    col_name = f"{dcfg.family}_{dcfg.projection}"   # e.g. "base_lin"
    scaling_path = join(dirs["data/inputs"], dcfg.scaling_csv)
    df = pd.read_csv(scaling_path)

    row = df.loc[df[dcfg.year_column] == scenario.year]
    if row.empty:
        raise ValueError(
            f"No scaling row for year={scenario.year} in {dcfg.scaling_csv}"
        )

    row = row.iloc[0]
    if col_name not in row.index:
        raise ValueError(
            f"Column '{col_name}' not found in {dcfg.scaling_csv} "
            f"for year={scenario.year}"
        )

    factor = float(row[col_name])
    return factor


def get_re_factors(scenario: Scenario) -> Dict[str, float]:
    rcfg = scenario.re
    return {
        "onwind": rcfg.onwind_factor,
        "solar": rcfg.solar_factor,
        "other": rcfg.other_re_factor,
    }



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
    processed_generation_dir: Path = Path(PROC_GENERATION_DIR)
    raw_demand_dir: Path = Path(RAW_DEMANDS_DIR)
    raw_generation_dir: Path = Path(RAW_GENERATION_DIR)
    raw_gadm_dir: Path = Path(RAW_GADM_DIR)
    raw_cutouts_dir: Path = Path(RAW_CUTOUTS_DIR)
    results_dir: Path = Path(RESULTS_DIR)

    @classmethod
    def create(cls) -> "ScenarioPaths":
        dirs = all_dirs()  # ensures directories exist when running interactively
        _ = dirs  # intentionally unused but keeps behaviour consistent with template
        return cls()



