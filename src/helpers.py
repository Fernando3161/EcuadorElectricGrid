from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from os.path import join

import pypsa

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
import logging
logger = logging.getLogger(__name__)

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


def patch_transformers_all(
    n: pypsa.Network,
    r_pu_default: float = 0.01,
    x_pu_default: float = 0.05,
    r_abs_default: float = 0.01,
    x_abs_default: float = 0.05,
) -> None:
    if n.transformers.empty:
        return
    tr = n.transformers.copy()
    for col in ["r_pu", "x_pu"]:
        if col not in tr.columns:
            tr[col] = np.nan
    if hasattr(n, "transformer_types") and not n.transformer_types.empty and "type" in tr.columns:
        tt_cols = [c for c in ["r_pu", "x_pu"] if c in n.transformer_types.columns]
        if tt_cols:
            tr = tr.join(n.transformer_types[tt_cols], on="type", rsuffix="_type")
            for col in ["r_pu", "x_pu"]:
                type_col = f"{col}_type"
                if type_col in tr.columns:
                    need = tr[col].isna() | (tr[col] == 0)
                    tr.loc[need & tr[type_col].notna(), col] = tr.loc[need, type_col]
            drop_cols = [c for c in ["r_pu_type", "x_pu_type"] if c in tr.columns]
            tr.drop(columns=drop_cols, inplace=True)
    tr.loc[tr["r_pu"].isna() | (tr["r_pu"] == 0), "r_pu"] = r_pu_default
    tr.loc[tr["x_pu"].isna() | (tr["x_pu"] == 0), "x_pu"] = x_pu_default
    if "r" not in tr.columns:
        tr["r"] = np.nan
    if "x" not in tr.columns:
        tr["x"] = np.nan
    tr.loc[tr["r"].isna() | (tr["r"] == 0), "r"] = r_abs_default
    tr.loc[tr["x"].isna() | (tr["x"] == 0), "x"] = x_abs_default
    if "s_nom" in tr.columns:
        tr.loc[tr["s_nom"].fillna(0) <= 0, "s_nom"] = 1.0
    n.transformers.loc[tr.index, tr.columns] = tr
    n_rpu_zero = (n.transformers["r_pu"].fillna(0) == 0).sum()
    n_xpu_zero = (n.transformers["x_pu"].fillna(0) == 0).sum()
    n_r_zero = (n.transformers["r"].fillna(0) == 0).sum()
    n_x_zero = (n.transformers["x"].fillna(0) == 0).sum()
    logger.info(
        "Transformer patch summary: r_pu zeros=%d, x_pu zeros=%d, r zeros=%d, x zeros=%d",
        n_rpu_zero,
        n_xpu_zero,
        n_r_zero,
        n_x_zero,
    )


# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

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



