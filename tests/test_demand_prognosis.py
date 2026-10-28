"""Small deterministic checks for the legacy Phase 08 forecast transformation."""

from pathlib import Path

import pandas as pd
import pytest

from src.demand_prognosis import build_demand_prognosis, write_demand_prognosis


ROOT = Path(__file__).resolve().parents[1]
MONTHLY = ROOT / "data/raw/demand/demand_month.csv"
ANNUAL = ROOT / "data/raw/demand/demand_year_prognosis.csv"


def test_reproduces_legacy_forecast(tmp_path: Path) -> None:
    output = tmp_path / "demand_prognosis_2022_2050.csv"
    forecast = write_demand_prognosis(MONTHLY, ANNUAL, output)
    assert forecast.Year.tolist() == list(range(2022, 2051))
    assert forecast.loc[forecast.Year.eq(2022), "Tendencial"].item() == 28180
    assert forecast.loc[forecast.Year.eq(2030), "Tendencial"].item() == 38042
    assert forecast.loc[forecast.Year.eq(2030), "Expansion"].item() == 46084
    pd.testing.assert_frame_equal(pd.read_csv(output), forecast, check_exact=False, rtol=1e-12)


def test_rejects_missing_year_without_overwriting_output(tmp_path: Path) -> None:
    annual = pd.read_csv(ANNUAL)
    broken = tmp_path / "broken.csv"
    annual.loc[annual.Year.ne(2030)].to_csv(broken, index=False)
    output = tmp_path / "forecast.csv"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="2023–2032"):
        write_demand_prognosis(MONTHLY, broken, output)
    assert output.read_text(encoding="utf-8") == "keep"


def test_source_data_is_not_replaced_by_current_sni_boundary() -> None:
    forecast = build_demand_prognosis(MONTHLY, ANNUAL)
    assert forecast.loc[forecast.Year.eq(2022), "Tendencial"].item() != 27561.64
