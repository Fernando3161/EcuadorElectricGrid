"""
Generate 2022 demand profiles by scaling the 2013 template.

This script reads the 2013 hourly demand profiles (in kW) and scales each
month so the monthly energy (kWh) matches the 2022 totals provided in
``data/raw/demand/demand_month.csv`` (given in GWh).
"""
from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# File locations
BASE_DIR = Path(__file__).resolve().parent.parent
PROFILE_PATH = BASE_DIR / "data" / "raw" / "demand" / "demand_profiles_EC.csv"
MONTH_TOTALS_PATH = BASE_DIR / "data" / "raw" / "demand" / "demand_month.csv"
OUTPUT_PATH = BASE_DIR / "data" / "raw" / "demand" / "demand_profiles_2022.csv"


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
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Remove BOM if present in the first column name
        if header and header[0].startswith("\ufeff"):
            header[0] = header[0].replace("\ufeff", "")
        for row in reader:
            month_name, _, total_2022 = row
            month_number = MONTH_NAME_TO_NUMBER[month_name]
            monthly_totals[month_number] = float(total_2022) * 1_000_000  # GWh -> kWh
    return monthly_totals


def load_2013_profiles(path: Path) -> Tuple[List[str], Dict[int, List[Tuple[dt.datetime, List[float]]]], Dict[int, float]]:
    """Load 2013 profiles grouped by month and return header and sums."""
    profiles_by_month: Dict[int, List[Tuple[dt.datetime, List[float]]]] = {m: [] for m in range(1, 13)}
    monthly_sums: Dict[int, float] = {m: 0.0 for m in range(1, 13)}

    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            timestamp = dt.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            month = timestamp.month
            values = [float(value) for value in row[1:]]
            row_sum = sum(values)
            monthly_sums[month] += row_sum
            profiles_by_month[month].append((timestamp, values))
    return header, profiles_by_month, monthly_sums


def scale_profiles(
    profiles_by_month: Dict[int, List[Tuple[dt.datetime, List[float]]]],
    monthly_sums: Dict[int, float],
    monthly_targets: Dict[int, float],
) -> List[List[float]]:
    """Scale monthly profiles to match target energy totals and return rows."""
    scaled_rows: List[List[float]] = []
    for month in range(1, 13):
        source_sum = monthly_sums[month]
        target_sum = monthly_targets[month]
        if source_sum == 0:
            raise ValueError(f"Source monthly sum for month {month} is zero; cannot scale.")
        factor = target_sum / source_sum
        for timestamp, values in profiles_by_month[month]:
            scaled_values = [value * factor for value in values]
            new_time = timestamp.replace(year=2022)
            scaled_rows.append([new_time.strftime("%Y-%m-%d %H:%M:%S"), *scaled_values])
    return scaled_rows


def write_profiles(path: Path, header: Iterable[str], rows: Iterable[List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def assert_totals(new_rows: Iterable[List[float]], expected_kwh: float) -> None:
    total = 0.0
    for row in new_rows:
        total += sum(float(value) for value in row[1:])
    if not math.isclose(total, expected_kwh, rel_tol=1e-6):
        raise AssertionError(
            f"Total energy mismatch: expected {expected_kwh:.2f} kWh, got {total:.2f} kWh"
        )


def main() -> None:
    monthly_targets = read_monthly_totals(MONTH_TOTALS_PATH)
    header, profiles_by_month, monthly_sums = load_2013_profiles(PROFILE_PATH)
    scaled_rows = scale_profiles(profiles_by_month, monthly_sums, monthly_targets)

    # Verify totals before writing
    expected_total = sum(monthly_targets.values())
    assert_totals(scaled_rows, expected_total)

    write_profiles(OUTPUT_PATH, header, scaled_rows)
    print(f"Wrote scaled 2022 profiles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
