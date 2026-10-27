"""Reproduce the legacy 2022–2050 demand prognosis used by Phase 08.

The 2022 baseline comes from the legacy monthly table, not the Phase 01
CENACE SNI demand boundary. Values through 2032 are preserved from the
legacy prognosis; 2033–2050 are a linear modelling extrapolation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS = ("Tendencial", "Expansion")


def build_demand_prognosis(monthly_path: Path, annual_path: Path) -> pd.DataFrame:
    """Build the historical 2022–2032 rows and linear 2033–2050 extension (GWh)."""
    monthly = pd.read_csv(monthly_path, encoding="utf-8-sig")
    annual = pd.read_csv(annual_path, encoding="utf-8-sig")
    monthly.columns = monthly.columns.str.strip()
    annual.columns = annual.columns.str.strip()
    if "Expasion" in annual and "Expansion" not in annual:
        annual = annual.rename(columns={"Expasion": "Expansion"})
    if "2022" not in monthly or not {"Year", *SCENARIOS}.issubset(annual):
        raise ValueError("Demand source columns are missing")
    if len(monthly) != 12 or monthly["Month"].nunique() != 12:
        raise ValueError("Expected 12 distinct months in the legacy 2022 demand table")
    if annual["Year"].tolist() != list(range(2023, 2033)):
        raise ValueError("Expected one ordered prognosis row per year, 2023–2032")
    baseline = pd.to_numeric(monthly["2022"], errors="raise")
    for scenario in SCENARIOS:
        annual[scenario] = pd.to_numeric(annual[scenario], errors="raise")
    if baseline.isna().any() or annual[list(SCENARIOS)].isna().any().any():
        raise ValueError("Demand source contains missing numeric values")
    if (baseline <= 0).any() or (annual[list(SCENARIOS)] <= 0).any().any():
        raise ValueError("Demand source contains nonpositive energy")

    total_2022 = float(baseline.sum())
    historical = pd.concat(
        [pd.DataFrame({"Year": [2022], "Tendencial": [total_2022],
                       "Expansion": [total_2022]}),
         annual[["Year", *SCENARIOS]]],
        ignore_index=True,
    )
    fit = historical.loc[historical["Year"].between(2026, 2032)]
    future_years = np.arange(2033, 2051)
    future = pd.DataFrame({"Year": future_years})
    for scenario in SCENARIOS:
        future[scenario] = np.polyval(
            np.polyfit(fit["Year"], fit[scenario], deg=1), future_years
        )
    return pd.concat([historical, future], ignore_index=True)


def write_demand_prognosis(
    monthly_path: Path, annual_path: Path, output_path: Path
) -> pd.DataFrame:
    """Validate inputs and atomically write the reproducible forecast CSV."""
    result = build_demand_prognosis(monthly_path, annual_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv",
            prefix=".demand_prognosis_", dir=output_path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            result.to_csv(handle, index=False, lineterminator="\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return result
