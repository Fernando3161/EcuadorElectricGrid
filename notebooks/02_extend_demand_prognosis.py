"""Extend Ecuador demand prognosis through 2050.

This script combines the 2022 observed annual demand with the Master Plan
prognosis for 2023–2032 and linearly extends both the trend and expansion
scenarios to 2050. A full prognosis table is written to
``data/processed/demand_prognosis_2022_2050.csv`` and a publication-ready
figure is saved to ``results/graphs/demand``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_MONTH_PATH = BASE_DIR / "data" / "raw" / "demand" / "demand_month.csv"
RAW_YEAR_PROGNOSIS_PATH = (
    BASE_DIR / "data" / "raw" / "demand" / "demand_year_prognosis.csv"
)
OUTPUT_TABLE_PATH = BASE_DIR / "data" / "processed" / "demand_prognosis_2022_2050.csv"
FIGURE_PATH = BASE_DIR / "results" / "graphs" / "demand" / "demand_prognosis_extension.png"


TREND_COLOR = "#1f77b4"  # blue tone
EXPANSION_COLOR = "#ff7f0e"  # orange tone


def _clean_columns(columns: Iterable[str]) -> List[str]:
    """Strip BOM and whitespace artifacts from column names."""
    return [col.replace("\ufeff", "").strip() for col in columns]


def load_monthly_totals(path: Path) -> pd.DataFrame:
    """Load monthly demand (GWh) with cleaned column names."""
    df = pd.read_csv(path)
    df.columns = _clean_columns(df.columns)
    return df


def calculate_annual_total(monthly_df: pd.DataFrame, column: str = "2022") -> float:
    """Return the annual total (GWh) for the specified column."""
    if column not in monthly_df.columns:
        raise ValueError(f"Column '{column}' not found in monthly demand data.")
    return float(monthly_df[column].sum())


def load_master_plan(path: Path) -> pd.DataFrame:
    """Load master plan prognosis for 2023–2032 with normalized column names."""
    df = pd.read_csv(path)
    df.columns = _clean_columns(df.columns)
    # Normalize scenario naming
    if "Expasion" in df.columns:
        df.rename(columns={"Expasion": "Expansion"}, inplace=True)
    return df[["Year", "Tendencial", "Expansion"]]


def build_historical_series(annual_2022: float, prognosis_df: pd.DataFrame) -> pd.DataFrame:
    """Return table for 2022–2032 combining observed and prognosis data."""
    baseline = pd.DataFrame(
        {"Year": [2022], "Tendencial": [annual_2022], "Expansion": [annual_2022]}
    )
    combined = pd.concat([baseline, prognosis_df], ignore_index=True)
    return combined


def extend_linear_forecast(historical: pd.DataFrame, target_year: int = 2050) -> pd.DataFrame:
    """Linearly extend each scenario from the historical years to the target year."""
    latest_year = int(historical["Year"].max())
    if target_year <= latest_year:
        raise ValueError("Target year must be greater than historical maximum.")

    forecast_years = np.arange(latest_year + 1, target_year + 1)
    forecast_df = pd.DataFrame({"Year": forecast_years})

    for scenario in ["Tendencial", "Expansion"]:
        coeffs = np.polyfit(historical["Year"], historical[scenario], deg=1)
        forecast_df[scenario] = np.polyval(coeffs, forecast_years)

    return pd.concat([historical, forecast_df], ignore_index=True)


def write_prognosis_table(prognosis: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prognosis.to_csv(path, index=False)


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]


def plot_prognosis(prognosis: pd.DataFrame, figure_path: Path) -> None:
    configure_plot_style()
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    given_mask = prognosis["Year"] <= 2032
    extension_mask = prognosis["Year"] > 2032

    for scenario, color in [("Tendencial", TREND_COLOR), ("Expansion", EXPANSION_COLOR)]:
        # Master Plan data
        sns.lineplot(
            data=prognosis.loc[given_mask],
            x="Year",
            y=scenario,
            label=f"{scenario} – Master Plan Prognosis",
            ax=ax,
            color=color,
            linestyle="-",
        )

        # Extension forecast
        sns.lineplot(
            data=prognosis.loc[extension_mask],
            x="Year",
            y=scenario,
            label=f"{scenario} – Extension",
            ax=ax,
            color=color,
            linestyle="--",
        )

    ax.set_title("Energy demand prognosis extension (GWh)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Demand (GWh)")
    ax.legend(title="Scenario", frameon=True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    monthly_totals = load_monthly_totals(RAW_MONTH_PATH)
    annual_2022 = calculate_annual_total(monthly_totals)

    master_plan = load_master_plan(RAW_YEAR_PROGNOSIS_PATH)
    historical = build_historical_series(annual_2022, master_plan)

    full_prognosis = extend_linear_forecast(historical, target_year=2050)
    write_prognosis_table(full_prognosis, OUTPUT_TABLE_PATH)
    plot_prognosis(full_prognosis, FIGURE_PATH)

    print(f"Saved full prognosis table to {OUTPUT_TABLE_PATH}")
    print(f"Saved prognosis plot to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
