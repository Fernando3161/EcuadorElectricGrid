"""Generate 2022 demand profiles by scaling the 2013 template.

This script reads the 2013 hourly demand profiles (in kW) and scales each
month so the monthly energy (kWh) matches the 2022 totals provided in
``data/raw/demand/demand_month.csv`` (given in GWh). It then writes the
scaled profiles to ``data/processed/demand/demand_profiles_2022.csv`` and
produces summary visualizations.
"""
from __future__ import annotations

import calendar
import datetime as dt
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# File locations
BASE_DIR = Path(__file__).resolve().parent.parent
PROFILE_PATH = BASE_DIR / "data" / "raw" / "demand" / "demand_profiles_EC.csv"
MONTH_TOTALS_PATH = BASE_DIR / "data" / "raw" / "demand" / "demand_month.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "demand" / "demand_profiles_2022.csv"
FIGURES_DIR = BASE_DIR / "results" / "graphs" / "demand"

MONTH_NAME_TO_NUMBER = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def read_monthly_totals(path: Path) -> Dict[int, float]:
    """Return mapping of month number -> 2022 energy total in kWh."""
    monthly_totals: Dict[int, float] = {}
    df = pd.read_csv(path)
    df.columns = [col.replace("\ufeff", "") for col in df.columns]
    for _, row in df.iterrows():
        month_name = row["Month"]
        total_2022 = float(row["2022"])
        month_number = MONTH_NAME_TO_NUMBER[month_name]
        monthly_totals[month_number] = total_2022 * 1_000_000  # GWh -> kWh
    return monthly_totals


def load_2013_profiles(path: Path) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """Load 2013 profiles and compute monthly energy totals."""
    df = pd.read_csv(path, parse_dates=[0])
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    bus_columns = df.columns[1:]
    df["month"] = df["timestamp"].dt.month
    df["total"] = df[bus_columns].sum(axis=1)
    monthly_sums = df.groupby("month")["total"].sum().to_dict()
    return df, monthly_sums


def scale_profiles(
    profiles: pd.DataFrame,
    monthly_sums: Dict[int, float],
    monthly_targets: Dict[int, float],
) -> pd.DataFrame:
    """Scale monthly profiles to match target energy totals."""
    bus_columns = profiles.columns[1:-2]  # exclude timestamp, month, total
    scaled_profiles = profiles.copy()

    for month in range(1, 13):
        source_sum = monthly_sums[month]
        target_sum = monthly_targets[month]
        if source_sum == 0:
            raise ValueError(f"Source monthly sum for month {month} is zero; cannot scale.")
        factor = target_sum / source_sum
        month_mask = scaled_profiles["month"] == month
        scaled_profiles.loc[month_mask, bus_columns] *= factor
        scaled_profiles.loc[month_mask, "total"] *= factor

    scaled_profiles["timestamp"] = scaled_profiles["timestamp"].apply(
        lambda ts: ts.replace(year=2022)
    )
    return scaled_profiles.drop(columns=["month"])


def assert_totals(new_profiles: pd.DataFrame, expected_kwh: float) -> None:
    bus_columns = new_profiles.columns[1:-1]
    total = new_profiles[bus_columns].to_numpy().sum()
    if not math.isclose(total, expected_kwh, rel_tol=1e-6):
        raise AssertionError(
            f"Total energy mismatch: expected {expected_kwh:.2f} kWh, got {total:.2f} kWh"
        )


def write_profiles(path: Path, profiles: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profiles.drop(columns=["total"], errors="ignore").to_csv(path, index=False)


def plot_march_window(profiles: pd.DataFrame, output_dir: Path) -> None:
    """Plot total demand for the first 15 days of March."""
    bus_columns = profiles.columns[1:-1]
    march_mask = (profiles["timestamp"] >= dt.datetime(2022, 3, 1)) & (
        profiles["timestamp"] < dt.datetime(2022, 3, 16)
    )
    march_data = profiles.loc[march_mask].copy()
    march_data["total"] = march_data[bus_columns].sum(axis=1)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=march_data, x="timestamp", y="total")
    plt.title("Total demand (kW) – March 1-15, 2022")
    plt.xlabel("Timestamp")
    plt.ylabel("Total demand (kW)")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "march_total_timeseries.png")
    plt.close()


def plot_monthly_totals(profiles: pd.DataFrame, output_dir: Path) -> None:
    """Plot total monthly energy as a column chart."""
    bus_columns = profiles.columns[1:-1]
    profiles["total"] = profiles[bus_columns].sum(axis=1)
    monthly_totals = (
        profiles.groupby(profiles["timestamp"].dt.month)["total"].sum().reset_index()
    )
    monthly_totals["month_name"] = monthly_totals["timestamp"].apply(
        lambda m: calendar.month_name[m]
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=monthly_totals, x="month_name", y="total", color="skyblue")
    plt.title("Total monthly energy (kWh) – 2022 profiles")
    plt.xlabel("Month")
    plt.ylabel("Energy (kWh)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "monthly_energy_totals.png")
    plt.close()


def main() -> None:
    monthly_targets = read_monthly_totals(MONTH_TOTALS_PATH)
    profiles_2013, monthly_sums_2013 = load_2013_profiles(PROFILE_PATH)
    scaled_profiles = scale_profiles(profiles_2013, monthly_sums_2013, monthly_targets)

    # Verify totals before writing
    expected_total = sum(monthly_targets.values())
    assert_totals(scaled_profiles, expected_total)

    write_profiles(OUTPUT_PATH, scaled_profiles)
    print(f"Wrote scaled 2022 profiles to {OUTPUT_PATH}")

    plot_march_window(scaled_profiles, FIGURES_DIR)
    plot_monthly_totals(scaled_profiles, FIGURES_DIR)
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
