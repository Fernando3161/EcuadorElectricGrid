# %% [markdown]
# #Electricity demand projection up to 2050 from a 2018–2027 CSV file.
# This file is to generate CSV with load forecast for the different scenarios
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import all dirs
parent_dir = Path(os.getcwd()).parents[0]
sys.path.append(str(parent_dir))
from src.paths import all_dirs
dirs = all_dirs()
dirs.keys()

# %%
"""
Electricity demand projection up to 2050 for all scenarios (tendencial, caso_base, matriz_productiva)
based on 2018–2027 historical data.
"""

# ---------------------------
# SETTINGS
# ---------------------------
path = dirs["data/raw/demand"]
csv_file = "demand_ec_2018_2027.csv"  # input file (2018–2027)
csv_path_hist = os.path.join(path, csv_file)
scenarios = ["trend", "base", "productive_mix"]  # base scenarios to project
year_projection_end = 2050
calibration_window = (2022, 2027)  # used for slope and CAGR
output_dir = dirs["data/processed"]


# ---------------------------
# FUNCTIONS
# ---------------------------
def load_historical(csv_path: str) -> pd.DataFrame:
    """Load and interpolate the historical data (2018–2027)."""
    df = pd.read_csv(csv_path)
    if "year" not in df.columns:
        raise ValueError("CSV must contain a 'year' column.")
    df = df.set_index("year").sort_index()
    years = pd.Index(range(int(df.index.min()), int(df.index.max()) + 1), name="year")
    df = df.reindex(years).interpolate().round(3)
    return df


def calc_slope(df: pd.DataFrame, col: str, y1: int, y2: int) -> float:
    """Linear slope (GWh/year)."""
    return (float(df.loc[y2, col]) - float(df.loc[y1, col])) / (y2 - y1)


def calc_cagr(df: pd.DataFrame, col: str, y1: int, y2: int) -> float:
    """Compound annual growth rate (CAGR)."""
    v1, v2 = float(df.loc[y1, col]), float(df.loc[y2, col])
    if v1 <= 0:
        raise ValueError(f"CAGR invalid: {col} value in {y1} <= 0.")
    return (v2 / v1) ** (1 / (y2 - y1)) - 1


def project_linear(
    base_value: float, slope: float, years: list[int], base_year: int
) -> np.ndarray:
    """Linear projection: v = v_base + slope*(year - base_year)."""
    return np.array([base_value + slope * (y - base_year) for y in years], dtype=float)


def project_exponential(
    base_value: float, cagr: float, years: list[int], base_year: int
) -> np.ndarray:
    """Exponential projection: v = v_base * (1 + cagr)^(year - base_year)."""
    return np.array(
        [base_value * ((1 + cagr) ** (y - base_year)) for y in years], dtype=float
    )


def avg_growth(vals: np.ndarray) -> float:
    """Average year-to-year increment."""
    diffs = np.diff(vals)
    return float(np.mean(diffs))



# %%
# ---------------------------
# MAIN PIPELINE
# ---------------------------
hist = load_historical(csv_path_hist)
hist = hist.round(2)                     # ensure historical values also max 2 decimals
y1, y2 = calibration_window

hist_start_year = int(hist.index.min())
hist_end_year = int(hist.index.max())
full_years = pd.Index(range(hist_start_year, year_projection_end + 1), name="year")

combined = pd.DataFrame(index=full_years)

results_df = {}

for s in scenarios:
    if s not in hist.columns:
        print(f"[WARN] Scenario '{s}' not found in historical data, skipping.")
        continue

    # --- compute linear slope & CAGR ---
    slope = calc_slope(hist, s, y1, y2)
    cagr = calc_cagr(hist, s, y1, y2)
    v_base = float(hist.loc[hist_end_year, s])

    proj_years = list(range(hist_end_year + 1, year_projection_end + 1))

    # --- compute projections ---
    proj_lin = project_linear(v_base, slope, proj_years, base_year=hist_end_year)
    proj_exp = project_exponential(v_base, cagr, proj_years, base_year=hist_end_year)

    # ensure 2 decimals
    proj_lin = np.round(proj_lin, 2)
    proj_exp = np.round(proj_exp, 2)

    # --- build full-length series ---
    col_lin = hist[s].reindex(full_years)
    col_exp = hist[s].reindex(full_years)

    col_lin.loc[proj_years] = proj_lin
    col_exp.loc[proj_years] = proj_exp

    col_lin = col_lin.astype(float).round(2)
    col_exp = col_exp.astype(float).round(2)

    # store in combined
    combined[f"{s}_lin"] = col_lin
    combined[f"{s}_exp"] = col_exp

    results_df[s] = pd.DataFrame(
        {
            "proj_linear_gwh": col_lin,
            "proj_exponential_gwh": col_exp,
        },
        index=full_years,
    )

    print(f"\n=== {s.upper()} ===")
    print(f"Calibration window: {y1}–{y2}")
    print(f"Linear slope ≈ {slope:.3f} GWh/year")
    print(f"CAGR ≈ {100 * cagr:.3f}% per year")
    print(f"Base value ({hist_end_year}): {v_base:.2f} GWh")

# ---------------------------
# FINAL OUTPUT TABLE
# ---------------------------
out_df = combined.reset_index()        # convert index → column `year`
out_df = out_df.round(2)               # enforce 2-decimal limit on all columns
out_df = out_df[["year"] + [c for c in out_df.columns if c != "year"]]

# ---------------------------
# SAVE CSV
# ---------------------------
output_dir_demand = os.path.join(dirs["data/processed/scaled_loads"])
os.makedirs(output_dir_demand, exist_ok=True)

output_path = os.path.join(
    output_dir_demand,
    "demand_projection_2018_2050_all_scenarios.csv"
)
out_df.to_csv(output_path, index=False)

print(f"\n[OK] Saved combined demand projection to: {output_path}")




