import pandas as pd
from os.path import join
from __future__ import annotations
from typing import Dict
from scenarios import Scenario

# =============================================================================
# Helper accessors (optional but handy)
# =============================================================================

import pandas as pd
from os.path import join


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

