# src/scenarios.py  (continuation)
from __future__ import annotations
from typing import Dict
from scenarios import *



SCENARIOS: Dict[str, Scenario] = {}


def _add(s: Scenario):
    SCENARIOS[s.id] = s


# File-name constants (adapt to your project layout)
DEMAND_TEMPLATE = "load_template.csv"
DEMAND_SCALING_CSV = "demand_scaling.csv"
NUCLEAR_LAYOUT_CSV = "nuclear_layout_smrs.csv"
CUTOUT_WIND = "cutout-2013-era5.nc"
CUTOUT_SOLAR = "cutout-2013-era5.nc"

# You can adjust network file names as needed:
NET_2017 = "network_2017.nc"
NET_2024 = "network_2024.nc"
NET_2030 = "network_2030_MP.nc"
NET_2035 = "network_2035_MP.nc"
NET_2045 = "network_2045_MP.nc"
NET_2050 = "network_2050_MP.nc"


# -----------------------------------------------------------------------------
# 1. Calibration & Current System (2017–2024)
# -----------------------------------------------------------------------------

_add(Scenario(
    id="CAL_2017",
    year=2017,
    demand=DemandConfig(
        mode="historical",                      # reproduce historical operation
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",                         # 2017-like
        inflow_profile_name="hydro_inflows_2017.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2017,
        year=2017,
        result_tag="cal_2017",
    ),
    nuclear=NuclearConfig(
        wave="none",
        total_capacity_mw=0.0,
    ),
    re=REConfig(level="none"),
    description="Calibration against 2017 historical production and ENS."
))

_add(Scenario(
    id="REF_2024",
    year=2024,
    demand=DemandConfig(
        mode="projected",
        family="base",                         # treat “Medium” as base_lin
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2024.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2024,
        year=2024,
        result_tag="ref_2024",
    ),
    nuclear=NuclearConfig(
        wave="none",
        total_capacity_mw=0.0,
    ),
    re=REConfig(level="none"),
    description="Reference current system (2024) under typical hydrology."
))

_add(Scenario(
    id="CRISIS_2024",
    year=2024,
    demand=DemandConfig(
        mode="projected",
        family="base",                         # same demand as REF_2024
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="severe_dry",                     # severe drought
        inflow_profile_name="hydro_inflows_2024.nc",
        inflow_scaling_factor=0.4,             # example: 40% inflows
    ),
    network=NetworkConfig(
        base_nc=NET_2024,
        year=2024,
        result_tag="crisis_2024",
    ),
    nuclear=NuclearConfig(
        wave="none",
        total_capacity_mw=0.0,
    ),
    re=REConfig(level="none"),
    description="Stress test representing 2024-like hydro crisis."
))


# -----------------------------------------------------------------------------
# 2. Master Plan 2030 Baseline (No Nuclear Yet)
# -----------------------------------------------------------------------------

_add(Scenario(
    id="REF_2030_MP",
    year=2030,
    demand=DemandConfig(
        mode="projected",
        family="base",                         # Medium case ~ base_lin
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2030.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2030,
        year=2030,
        result_tag="ref_2030_mp",
    ),
    nuclear=NuclearConfig(
        wave="none",
        total_capacity_mw=0.0,
    ),
    re=REConfig(
        level="none",
    ),
    description="Master Plan 2030 baseline without extra RE beyond MP."
))

_add(Scenario(
    id="RE_2030_L",
    year=2030,
    demand=DemandConfig(
        mode="projected",
        family="base",
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2030.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2030,
        year=2030,
        result_tag="re_2030_l",
    ),
    nuclear=NuclearConfig(
        wave="none",
        total_capacity_mw=0.0,
    ),
    re=REConfig(
        level="low",
        onwind_factor=0.05,                   # ~5% extra RE
        solar_factor=0.05,
        wind_cutout_nc=CUTOUT_WIND,
        solar_cutout_nc=CUTOUT_SOLAR,
    ),
    description="REF_2030_MP + low extra RE (~5% reference)."
))

_add(Scenario(
    id="RE_2030_H",
    year=2030,
    demand=DemandConfig(
        mode="projected",
        family="base",
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2030.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2030,
        year=2030,
        result_tag="re_2030_h",
    ),
    nuclear=NuclearConfig(
        wave="none",
        total_capacity_mw=0.0,
    ),
    re=REConfig(
        level="high",
        onwind_factor=0.15,                   # ~10–15% extra RE
        solar_factor=0.15,
        wind_cutout_nc=CUTOUT_WIND,
        solar_cutout_nc=CUTOUT_SOLAR,
    ),
    description="REF_2030_MP + high extra RE (~10–15% reference)."
))


# -----------------------------------------------------------------------------
# 3. First SMR Wave in 2035 (0.9 GW)
# -----------------------------------------------------------------------------

_add(Scenario(
    id="REF_2035_MP",
    year=2035,
    demand=DemandConfig(
        mode="projected",
        family="base",
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2035.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2035,
        year=2035,
        result_tag="ref_2035_mp",
    ),
    nuclear=NuclearConfig(
        wave="none",
        total_capacity_mw=0.0,
    ),
    re=REConfig(level="none"),
    description="2035 baseline (MP 2030 + 2035 additions, no nuclear)."
))

_add(Scenario(
    id="NUC_2035_W1",
    year=2035,
    demand=DemandConfig(
        mode="projected",
        family="base",                         # Medium demand
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2035.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2035,
        year=2035,
        result_tag="nuc_2035_w1",
    ),
    nuclear=NuclearConfig(
        wave="W1",
        total_capacity_mw=900.0,               # 0.9 GW
        layout_csv=NUCLEAR_LAYOUT_CSV,
        n_units=3,
        unit_capacity_mw=300.0,
    ),
    re=REConfig(level="none"),
    description="2035 with SMR Wave 1: 0.9 GW nuclear."
))

_add(Scenario(
    id="NUC_2035_W1_H",
    year=2035,
    demand=DemandConfig(
        mode="projected",
        family="productive_mix",               # High demand = productive_mix
        projection="lin",                      # (use _exp later if you like)
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2035.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2035,
        year=2035,
        result_tag="nuc_2035_w1_h",
    ),
    nuclear=NuclearConfig(
        wave="W1",
        total_capacity_mw=900.0,
        layout_csv=NUCLEAR_LAYOUT_CSV,
        n_units=3,
        unit_capacity_mw=300.0,
    ),
    re=REConfig(level="none"),
    description="2035 high-demand case with 0.9 GW SMR (Wave 1)."
))

_add(Scenario(
    id="RE_2035_H",
    year=2035,
    demand=DemandConfig(
        mode="projected",
        family="base",
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2035.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2035,
        year=2035,
        result_tag="re_2035_h",
    ),
    nuclear=NuclearConfig(
        wave="none",
        total_capacity_mw=0.0,
    ),
    re=REConfig(
        level="high",
        onwind_factor=0.20,                   # 10–20% RE
        solar_factor=0.20,
        wind_cutout_nc=CUTOUT_WIND,
        solar_cutout_nc=CUTOUT_SOLAR,
    ),
    description="2035 RE-only alternative with strong RE buildout."
))

_add(Scenario(
    id="MIX_2035_W1_RE",
    year=2035,
    demand=DemandConfig(
        mode="projected",
        family="base",
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2035.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2035,
        year=2035,
        result_tag="mix_2035_w1_re",
    ),
    nuclear=NuclearConfig(
        wave="W1",
        total_capacity_mw=900.0,
        layout_csv=NUCLEAR_LAYOUT_CSV,
        n_units=3,
        unit_capacity_mw=300.0,
    ),
    re=REConfig(
        level="medium",
        onwind_factor=0.10,
        solar_factor=0.10,
        wind_cutout_nc=CUTOUT_WIND,
        solar_cutout_nc=CUTOUT_SOLAR,
    ),
    description="2035 hybrid path: 0.9 GW SMR + moderate RE (~10%)."
))


# -----------------------------------------------------------------------------
# 4. Second SMR Wave in 2045 (2.1 GW Total)
# -----------------------------------------------------------------------------

_add(Scenario(
    id="REF_2045_MP",
    year=2045,
    demand=DemandConfig(
        mode="projected",
        family="base",
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2045.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2045,
        year=2045,
        result_tag="ref_2045_mp",
    ),
    nuclear=NuclearConfig(
        wave="none",
        total_capacity_mw=0.0,
    ),
    re=REConfig(level="none"),
    description="2045 baseline (REF_2035 updated + MP 2040 pipeline, no nuclear)."
))

_add(Scenario(
    id="NUC_2045_W2",
    year=2045,
    demand=DemandConfig(
        mode="projected",
        family="base",                         # Medium demand
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2045.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2045,
        year=2045,
        result_tag="nuc_2045_w2",
    ),
    nuclear=NuclearConfig(
        wave="W2",
        total_capacity_mw=2100.0,              # 2.1 GW total (W1+W2)
        layout_csv=NUCLEAR_LAYOUT_CSV,
    ),
    re=REConfig(level="none"),
    description="2045 nuclear pathway with 2.1 GW SMR (W1+W2)."
))

_add(Scenario(
    id="NUC_2045_W2_H",
    year=2045,
    demand=DemandConfig(
        mode="projected",
        family="productive_mix",               # High demand
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2045.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2045,
        year=2045,
        result_tag="nuc_2045_w2_h",
    ),
    nuclear=NuclearConfig(
        wave="W2",
        total_capacity_mw=2100.0,
        layout_csv=NUCLEAR_LAYOUT_CSV,
    ),
    re=REConfig(level="none"),
    description="2045 high-demand nuclear adequacy scenario (2.1 GW)."
))

_add(Scenario(
    id="RE_2045_H",
    year=2045,
    demand=DemandConfig(
        mode="projected",
        family="base",
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2045.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2045,
        year=2045,
        result_tag="re_2045_h",
    ),
    nuclear=NuclearConfig(
        wave="none",
        total_capacity_mw=0.0,
    ),
    re=REConfig(
        level="high",
        onwind_factor=0.25,                   # 15–25% RE
        solar_factor=0.25,
        wind_cutout_nc=CUTOUT_WIND,
        solar_cutout_nc=CUTOUT_SOLAR,
    ),
    description="2045 high-RE-only pathway (no nuclear, strong RE)."
))

_add(Scenario(
    id="MIX_2045_W2_RE",
    year=2045,
    demand=DemandConfig(
        mode="projected",
        family="productive_mix",               # High demand
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2045.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2045,
        year=2045,
        result_tag="mix_2045_w2_re",
    ),
    nuclear=NuclearConfig(
        wave="W2",
        total_capacity_mw=2100.0,
        layout_csv=NUCLEAR_LAYOUT_CSV,
    ),
    re=REConfig(
        level="medium",
        onwind_factor=0.15,
        solar_factor=0.15,
        wind_cutout_nc=CUTOUT_WIND,
        solar_cutout_nc=CUTOUT_SOLAR,
    ),
    description="2045 hybrid path: 2.1 GW nuclear + substantial RE."
))


# -----------------------------------------------------------------------------
# 5. Full SMR Deployment by 2050 (3.0 GW)
# -----------------------------------------------------------------------------

_add(Scenario(
    id="REF_2050_NO_NUC",
    year=2050,
    demand=DemandConfig(
        mode="projected",
        family="base",                         # Medium baseline
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2050.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2050,
        year=2050,
        result_tag="ref_2050_no_nuc",
    ),
    nuclear=NuclearConfig(
        wave="none",
        total_capacity_mw=0.0,
    ),
    re=REConfig(level="none"),
    description="2050 long-term non-nuclear alternative (baseline)."
))

_add(Scenario(
    id="NUC_2050_W3",
    year=2050,
    demand=DemandConfig(
        mode="projected",
        family="base",
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2050.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2050,
        year=2050,
        result_tag="nuc_2050_w3",
    ),
    nuclear=NuclearConfig(
        wave="W3",
        total_capacity_mw=3000.0,              # 3.0 GW total (W1+W2+W3)
        layout_csv=NUCLEAR_LAYOUT_CSV,
    ),
    re=REConfig(level="none"),
    description="2050 full SMR deployment: 3.0 GW nuclear backbone."
))

_add(Scenario(
    id="NUC_2050_W3_H",
    year=2050,
    demand=DemandConfig(
        mode="projected",
        family="productive_mix",               # High electrification
        projection="lin",
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2050.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2050,
        year=2050,
        result_tag="nuc_2050_w3_h",
    ),
    nuclear=NuclearConfig(
        wave="W3",
        total_capacity_mw=3000.0,
        layout_csv=NUCLEAR_LAYOUT_CSV,
    ),
    re=REConfig(level="none"),
    description="2050 high-demand full nuclear scenario (3.0 GW)."
))

_add(Scenario(
    id="RE_2050_XL",
    year=2050,
    demand=DemandConfig(
        mode="projected",
        family="base",                         # could also use base_exp
        projection="exp",                      # make this a bit more extreme
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2050.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2050,
        year=2050,
        result_tag="re_2050_xl",
    ),
    nuclear=NuclearConfig(
        wave="none",
        total_capacity_mw=0.0,
    ),
    re=REConfig(
        level="xl",
        onwind_factor=0.35,                    # ~30–40% RE
        solar_factor=0.35,
        wind_cutout_nc=CUTOUT_WIND,
        solar_cutout_nc=CUTOUT_SOLAR,
    ),
    description="2050 very high RE pathway (no nuclear, base_exp demand)."
))

_add(Scenario(
    id="MIX_2050_DEEP",
    year=2050,
    demand=DemandConfig(
        mode="projected",
        family="productive_mix",               # High demand
        projection="exp",                      # aggressive electrification
        scaling_csv=DEMAND_SCALING_CSV,
        profile_template_name=DEMAND_TEMPLATE,
    ),
    hydro=HydroConfig(
        case="normal",
        inflow_profile_name="hydro_inflows_2050.nc",
    ),
    network=NetworkConfig(
        base_nc=NET_2050,
        year=2050,
        result_tag="mix_2050_deep",
    ),
    nuclear=NuclearConfig(
        wave="W3",
        total_capacity_mw=3000.0,
        layout_csv=NUCLEAR_LAYOUT_CSV,
    ),
    re=REConfig(
        level="high",
        onwind_factor=0.25,
        solar_factor=0.25,
        wind_cutout_nc=CUTOUT_WIND,
        solar_cutout_nc=CUTOUT_SOLAR,
    ),
    description="2050 deep decarbonisation: 3 GW nuclear + strong RE under high demand."
))
