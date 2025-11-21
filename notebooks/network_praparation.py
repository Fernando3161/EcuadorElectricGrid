# %% [markdown]
# Setup environment and load the base PyPSA-Earth network for a specified country.
# 

# %%
# --- Core imports
import os
import sys
import copy
import logging
import warnings
from os.path import join
from pathlib import Path

# --- Numerical / data
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

# --- Power systems / geospatial
import pypsa
import atlite
from shapely.geometry import Point, box
from shapely.ops import unary_union
import shapely
import numpy as np
from scipy.spatial import cKDTree as KDTree

# Optional plotting (only used in helper plot function)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Optional: powerplantmatching (picks up config but we keep local CSV as source of truth)
try:
    import powerplantmatching as pm
    HAVE_PPM = True
except Exception:
    HAVE_PPM = False

# --- Silence noisy warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.filterwarnings('ignore')


# --- Logging
parent_dir = Path(os.getcwd()).parents[0]          # project/
LOG_FILE = join(parent_dir, "logs.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Project paths helper
sys.path.append(str(parent_dir))
from src.paths import all_dirs  # must exist in your repo
dirs = all_dirs()
from src.scenario_reg import SCENARIOS

# --- Add PyPSA-Earth scripts to PATH (assumes repo layout: <project>/../pypsa-earth/scripts)
scripts_path = os.path.join(parent_dir.parents[0], "pypsa-earth", "scripts")
assert os.path.isdir(scripts_path), f"Path not found: {scripts_path}"
sys.path.append(scripts_path)

scenario= SCENARIOS["CAL_2018"]


# %%
# 1) Load Networks
def load_network(scenario):
    network_name= scenario.network.name

    network_dir = dirs["data/processed/networks"]
    network_file = join(network_dir, network_name)

    try:
        network = pypsa.Network(network_file)
        logger.info(f"✓ loaded {network_name}.nc")
    except Exception as e:
        logger.warning(f"⚠ failed to load {network_name}.nc: {e}")

    return network

network = load_network(scenario)

# %%
#1.a) Prune Network to work only at 137kV level

def prune_network_min_voltage(n, v_threshold_kv=137.0):
    """
    Keep only buses with v_nom >= v_threshold_kv and remove any components
    (transformers, lines, links, loads, generators, stores, storage_units, shunts)
    that reference removed buses.
    """
    import numpy as np
    n = n.copy()

    # 1) Decide which buses to keep
    buses_orig = n.buses.index.astype(str)
    keep_buses = n.buses.index[n.buses["v_nom"] >= float(v_threshold_kv)].astype(str)
    keep_set   = set(keep_buses)
    drop_set   = set(buses_orig) - keep_set

    print(f"Keeping {len(keep_buses)}/{len(buses_orig)} buses (v_nom >= {v_threshold_kv} kV).")
    if drop_set:
        print(f"Dropping {len(drop_set)} buses due to voltage threshold.")

    # helper to drop rows by bus column name if bus not in keep_set
    def _drop_by_bus(df, bus_col):
        if df.empty or bus_col not in df.columns:
            return df.index[:0]
        mask = ~df[bus_col].astype(str).isin(keep_set)
        return df.index[mask]

    # 2) Collect components to drop because they reference removed buses
    to_drop = {
        "Load":          _drop_by_bus(n.loads, "bus"),
        "Generator":     _drop_by_bus(n.generators, "bus"),
        "Store":         _drop_by_bus(getattr(n, "stores"), "bus") if hasattr(n, "stores") else [],
        "StorageUnit":   _drop_by_bus(n.storage_units, "bus") if hasattr(n, "storage_units") else [],
        "ShuntImpedance": _drop_by_bus(getattr(n, "shunt_impedances"), "bus") if hasattr(n, "shunt_impedances") else [],
        "Line":          n.lines.index[
                            ~n.lines["bus0"].astype(str).isin(keep_set) |
                            ~n.lines["bus1"].astype(str).isin(keep_set)
                        ],
        "Transformer":   n.transformers.index[
                            ~n.transformers["bus0"].astype(str).isin(keep_set) |
                            ~n.transformers["bus1"].astype(str).isin(keep_set)
                        ],
        "Link":          n.links.index[
                            ~n.links["bus0"].astype(str).isin(keep_set) |
                            ~n.links["bus1"].astype(str).isin(keep_set)
                        ] if len(n.links) else []
    }

    # 3) Remove them from the network using PyPSA's API
    for comp, idx in to_drop.items():
        idx = list(idx)
        if not idx:
            continue
        for name in idx:
            try:
                n.remove(comp, name)
            except Exception as e:
                # be robust if component table missing etc.
                pass
        print(f"Removed {len(idx):4d} {comp}(s) referencing dropped buses.")

    # 4) Finally, restrict the buses table itself to the kept set
    n.buses = n.buses.loc[keep_buses]

    # Optional: clean orphaned carriers/sub-networks etc. if needed

    return n

# ---- use it ----
network = prune_network_min_voltage(network, v_threshold_kv=137)
                                    #137.0)



# %%
def total_energy_demand(scenario) -> float:
    # first get the whole year demand of the loaded profile

    config = scenario.demand
    year = scenario.year
    demand_path = config.scaling_csv
    if not os.path.isfile(demand_path):
        logger.warning("Demand CSV %s not found; using factor=1.0", demand_path)
        return 1.0
    df = pd.read_csv(demand_path, index_col= "year")
    if year not in df.index:
        raise ValueError(
            f"Year '{config.year_column}' missing in  file {demand_path}"
        )
    prognosis_type = scenario.demand.family + "_"+ scenario.demand.projection
   
    demand = df.loc[year,prognosis_type]
    return demand




# %%
# 2) Loads: add p_set time series
# 2) Loads: add p_set time series with nearest-bus remapping for unknown buses
def read_and_scale_loads(scenario):
    path_loads = scenario.demand.profile_template_name
    logger.info(f"Loading load profile: {path_loads}")
    load_profile = pd.read_csv(path_loads, index_col=0, parse_dates=True)
    total_reference_energy = load_profile.sum().sum()*1e-6 #GW
    total_target_year_energy = total_energy_demand(scenario)
    scaling_factor = total_target_year_energy/total_reference_energy
    load_profile = load_profile*scaling_factor
    print(total_reference_energy)
    print(total_target_year_energy)
    logger.info(f"Total reference energy: {total_reference_energy:.2f}")
    logger.info(f"Total target-year energy: {total_target_year_energy:.2f}")
    logger.info(f"Scaled all load profile values by factor {scaling_factor:.2f}")
    return load_profile
    
load_profile = read_and_scale_loads(scenario)    


# %%
import os
import numpy as np
from os.path import join
from scipy.spatial import cKDTree as KDTree
import pypsa

# assumes: dirs, logger are defined globally


def ensure_string_load_columns(load_profile):
    """Ensure load_profile columns are strings and return (profile, columns_list)."""
    load_profile = load_profile.copy()
    load_cols = [str(c) for c in load_profile.columns]
    load_profile.columns = load_cols
    return load_profile, load_cols


def get_existing_buses(network):
    """Return (buses_now, existing_buses_set). Index is cast to str."""
    buses_now = network.buses.copy()
    buses_now.index = buses_now.index.astype(str)
    existing_buses = set(buses_now.index)
    return buses_now, existing_buses


def log_missing_buses(load_cols, existing_buses):
    """Log whether there are missing buses in the current network."""
    missing = [c for c in load_cols if c not in existing_buses]
    if missing:
        logger.warning(
            f"{len(missing)} load buses not found; reassigning to nearest existing bus."
        )
    else:
        logger.info("All load buses found in current network.")


def build_kdtree_for_buses(buses_now):
    """
    Build KDTree for current network buses.

    Returns (coords_now, kdt_now, use_lonlat).
    """
    use_lonlat = {"lon", "lat"}.issubset(buses_now.columns)
    if use_lonlat:
        coords_now = buses_now.loc[:, ["lon", "lat"]].to_numpy(dtype=float)
    else:
        coords_now = buses_now.loc[:, ["x", "y"]].to_numpy(dtype=float)

    kdt_now = KDTree(coords_now)
    return coords_now, kdt_now, use_lonlat


def load_fallback_networks():
    """
    Load fallback networks from disk.

    Returns a dict {name: pypsa.Network}.
    """
    network_dir_fb = dirs["data/processed/networks"]
    network_files_fb = ["network_original"]
    fallback_networks = {}

    for nf in network_files_fb:
        f = join(network_dir_fb, f"{nf}.nc")
        if os.path.isfile(f):
            fallback_networks[nf] = pypsa.Network(f)

    return fallback_networks


def find_bus_coords_in_fallbacks(bus_name, fallback_networks):
    """
    Try to get coordinates for a missing bus from any fallback network.

    Returns:
        ("lonlat", lon, lat) or ("xy", x, y) or None.
    """
    for nf, net_fb in fallback_networks.items():
        try:
            df = net_fb.buses.copy()
            df.index = df.index.astype(str)
            if bus_name in df.index:
                if {"lon", "lat"}.issubset(df.columns):
                    return (
                        "lonlat",
                        float(df.loc[bus_name, "lon"]),
                        float(df.loc[bus_name, "lat"]),
                    )
                elif {"x", "y"}.issubset(df.columns):
                    return (
                        "xy",
                        float(df.loc[bus_name, "x"]),
                        float(df.loc[bus_name, "y"]),
                    )
        except Exception:
            # ignore failures on individual fallback networks
            pass

    return None


def build_bus_assignments(
    load_cols, existing_buses, buses_now, coords_now, kdt_now, use_lonlat, fallback_networks
):
    """
    Build bus assignment list for each load column and collect a remap report.

    Returns (bus_assignments, remap_report).
    """
    bus_assignments = []
    remap_report = []

    # precompute centroid for “no info” cases
    centroid = np.nanmedian(coords_now, axis=0)[None, :]

    for c in load_cols:
        if c in existing_buses:
            bus_assignments.append(c)  # unchanged
            continue

        # Try to recover coordinates for this bus label from other networks
        info = find_bus_coords_in_fallbacks(c, fallback_networks)

        if info is not None:
            kind, bx, by = info

            # Query KDTree (always built for current network in the coordinate type we have)
            if use_lonlat and kind == "lonlat":
                q = np.array([[bx, by]])
            elif (not use_lonlat) and kind == "xy":
                q = np.array([[bx, by]])
            else:
                # Coordinate kind mismatch; cannot reliably transform → use crude nearest
                # by lon/lat if both exist, else by x/y. If mismatch, use network centroid.
                remap_report.append((c, None, None, "coord_kind_mismatch"))
                q = centroid

            idx = int(kdt_now.query(q)[1][0])
            nearest_bus = buses_now.index[idx]
            bus_assignments.append(nearest_bus)
            remap_report.append((c, bx, by, f"mapped_to:{nearest_bus}"))

        else:
            # No coordinates found anywhere → attach to globally nearest by network centroid
            idx = int(kdt_now.query(centroid)[1][0])
            nearest_bus = buses_now.index[idx]
            bus_assignments.append(nearest_bus)
            remap_report.append(
                (c, None, None, f"mapped_to:{nearest_bus};reason:no_coords")
            )

    return bus_assignments, remap_report


def log_remap_report(remap_report):
    """Log detailed remapping information."""
    if not remap_report:
        return

    logger.info("Load bus remapping performed (missing → nearest existing):")
    for r in remap_report:
        logger.info(f"  load '{r[0]}' → {r[3]} (src_coords: {r[1]}, {r[2]})")


def log_peak_load(network):
    """Compute and log peak load in GW from loads_t.p_set."""
    peak_load = round(network.loads_t.p_set.T.sum().max() / 1000, 1)
    logger.info(f"Peak load in p_set time series: {peak_load} GW")


def add_profiles_to_nearest_bus(load_profile, network):
    """
    Main wrapper: keep behaviour identical, but orchestrate smaller helpers.
    """
    try:
        # --- Ensure column names are strings
        load_profile, load_cols = ensure_string_load_columns(load_profile)

        # --- Existing buses in the active network
        buses_now, existing_buses = get_existing_buses(network)

        # --- Find missing bus labels and log
        log_missing_buses(load_cols, existing_buses)

        # --- Build KDTree on current network buses
        coords_now, kdt_now, use_lonlat = build_kdtree_for_buses(buses_now)

        # --- Load fallback networks
        fallback_networks = load_fallback_networks()

        # --- Build a bus assignment vector aligned with load columns
        bus_assignments, remap_report = build_bus_assignments(
            load_cols,
            existing_buses,
            buses_now,
            coords_now,
            kdt_now,
            use_lonlat,
            fallback_networks,
        )

        # --- Log remapping report
        log_remap_report(remap_report)

        # --- Finally add Loads with reassigned buses
        network.madd("Load", load_cols, bus=bus_assignments, p_set=load_profile)

        # --- Peak (GW)
        log_peak_load(network)

    except Exception as e:
        logger.warning(f"Could not attach loads: {e}")

    return network


# usage
network = add_profiles_to_nearest_bus(load_profile, network)


# %%
# Couple of helpers
# ---- Normalize capacities and carriers (ES->EN) on the (possibly PPM-normalized) table
def _normalize_text(s):
    import unicodedata, re
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s)
    return s

# ---- Totals
def _safe_sum(series):
    try: return float(series.sum())
    except: return np.nan

import json
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except UnicodeDecodeError:
        # fallback if file was saved incorrectly (Latin-1 / Windows-1252)
        with open(path, "r", encoding="latin-1") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None


# %%
import os
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa


def _load_raw_power_plants():
    ppl_csv = join(dirs["data/processed/generation"], "powerplants_all.csv")
    assert os.path.isfile(ppl_csv), f"Power plants CSV not found: {ppl_csv}"
    return pd.read_csv(ppl_csv, index_col=0)


def _normalize_with_powerplantmatching(ppl_raw: pd.DataFrame) -> pd.DataFrame:
    ppl = ppl_raw.copy()
    try:
        import powerplantmatching as pm

        ppmatching = os.path.join(
            Path(os.getcwd()).parents[1],
            "pypsa-earth",
            "configs",
            "powerplantmatching_config.yaml",
        )
        config = pm.get_config(ppmatching)
        config["target_countries"] = ["EC"]

        ppl_f = ppl.powerplant.fill_missing_commissioning_years()
        ppl_p = ppl_f.powerplant.to_pypsa_names()

        ppl = ppl_p.dropna(axis=1, how="all")

        logger.info(
            "Applied powerplantmatching normalization (fill years + to_pypsa_names)."
        )
    except Exception as e:
        logger.warning(f"powerplantmatching step skipped (using raw CSV): {e}")
        ppl = ppl_raw.copy()

    return ppl


def _clean_p_nom(ppl: pd.DataFrame) -> pd.DataFrame:
    if "p_nom" not in ppl.columns:
        logger.warning("Column 'p_nom' not found in plants table after normalization.")
        return ppl

    ppl = ppl.copy()
    ppl["p_nom"] = (
        ppl["p_nom"]
        .astype(str)
        .str.replace("'", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    ppl["p_nom"] = pd.to_numeric(ppl["p_nom"], errors="coerce").fillna(0.0)
    return ppl


def _map_carriers(ppl: pd.DataFrame) -> pd.DataFrame:
    if "carrier" not in ppl.columns:
        logger.warning("Column 'carrier' not found in plants table after normalization.")
        return ppl

    ppl = ppl.copy()
    ppl["carrier_norm"] = ppl["carrier"].apply(_normalize_text)

    file_es2en = join(dirs["data/raw/generation"], "carrier_name_mapping.json")
    carrier_map_es2en = load_json(file_es2en)

    ppl["carrier"] = (
        ppl["carrier_norm"].map(carrier_map_es2en).fillna(ppl["carrier_norm"])
    )
    return ppl


def _filter_by_sni_year_and_size(
    ppl: pd.DataFrame, scenario, filtering_power_kw: float
) -> pd.DataFrame:
    date_in = int(scenario.year - 1)

    if {"component", "DateIn", "p_nom"}.issubset(ppl.columns):
        ppl_connected = ppl[ppl["component"] == "S.N.I."]
        ppl_connected_current = ppl_connected[ppl_connected["DateIn"] <= date_in]
        ppl_filtered = ppl_connected_current[
            ppl_connected_current["p_nom"] >= filtering_power_kw
        ].copy()
    else:
        logger.warning("Required columns for filtering missing; skipping filter.")
        ppl_connected = None
        ppl_connected_current = None
        ppl_filtered = ppl.copy()

    return ppl_filtered, ppl_connected, ppl_connected_current, date_in


def _log_capacity_stats(
    ppl: pd.DataFrame,
    ppl_connected: pd.DataFrame | None,
    ppl_connected_current: pd.DataFrame | None,
    date_in: int,
    filtering_power_kw: float,
):
    total_capacity = _safe_sum(ppl.get("p_nom", pd.Series(dtype=float)))

    if ppl_connected is not None:
        total_connected = _safe_sum(ppl_connected.get("p_nom", pd.Series(dtype=float)))
    else:
        total_connected = np.nan

    if ppl_connected_current is not None:
        total_connected_current = _safe_sum(
            ppl_connected_current.get("p_nom", pd.Series(dtype=float))
        )
    else:
        total_connected_current = np.nan

    total_filtered = _safe_sum(
        ppl_connected_current.get("p_nom", pd.Series(dtype=float))
        if ppl_connected_current is not None
        else pd.Series(dtype=float)
    )

    logger.info(
        "Capacities [kW]\n"
        f"  total:                  {total_capacity:,.0f}\n"
        f"  connected (SNI):        {total_connected:,.0f}\n"
        f"  connected <= {date_in}: {total_connected_current:,.0f}\n"
        f"  filtered (>= {filtering_power_kw:.0f} kW): {total_filtered:,.0f}"
    )


def create_power_plants_table(scenario):
    ppl_raw = _load_raw_power_plants()
    ppl = _normalize_with_powerplantmatching(ppl_raw)
    ppl = _clean_p_nom(ppl)
    ppl = _map_carriers(ppl)

    filtering_power_kw = 0.0
    (
        ppl_filtered,
        ppl_connected,
        ppl_connected_current,
        date_in,
    ) = _filter_by_sni_year_and_size(ppl, scenario, filtering_power_kw)

    _log_capacity_stats(
        ppl=ppl,
        ppl_connected=ppl_connected,
        ppl_connected_current=ppl_connected_current,
        date_in=date_in,
        filtering_power_kw=filtering_power_kw,
    )

    return ppl_filtered

ppl_filtered = create_power_plants_table(scenario)
ppl_filtered

# %%
# s= ppl_filtered.p_nom
# plt.figure(figsize=(8,5))
# plt.hist(s, bins=30)   # choose number of bins
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.title("Histogram of Series")
# plt.show()

# %%
network.buses.v_nom.unique()

# %%
from scipy.spatial import cKDTree as KDTree


# -----------------------------
# 1) Helpers
# -----------------------------
def _categorize_plant_power(p_nom, low_thr=50.0, high_thr=300.0):
    """
    Categorize plant size based on installed power.
    Adjust thresholds to your units (kW/MW).
    """
    if p_nom >= high_thr:
        return "high"
    elif p_nom >= low_thr:
        return "medium"
    else:
        return "low"


def _select_buses_by_voltage(buses_df, idx, target_v, tol=10.0):
    """Return index of buses within target_v +/- tol."""
    if "v_nom" not in buses_df.columns:
        return idx[:0]  # empty Index
    mask = (buses_df.loc[idx, "v_nom"] >= target_v - tol) & (
        buses_df.loc[idx, "v_nom"] <= target_v + tol
    )
    return idx[mask]


def _build_kdtree_lonlat(buses_df, idx):
    """KDTree on lon/lat for given bus index subset."""
    if len(idx) == 0:
        return None, None
    coords = buses_df.loc[idx, ["lon", "lat"]].values
    return KDTree(coords), idx


def _query_bus_within_radius(plant_lonlat, kdt, idx_values, radius_km):
    """
    Query nearest bus within radius_km (approx using lon/lat degrees).
    Returns bus label or None.
    """
    if kdt is None or len(idx_values) == 0:
        return None

    dist_deg, pos = kdt.query(plant_lonlat, k=1)
    # 1 degree ~ 111 km near equator (OK for Ecuador-scale distances)
    dist_km = dist_deg * 111.0
    if dist_km <= radius_km:
        return idx_values[pos]
    return None


def _attach_simple_nearest(net, plants_df, sub_idx):
    """
    Old simple behaviour: nearest LV substation only, no voltage / size logic.
    Used as fallback if lon/lat / v_nom not available.
    """
    kdt = KDTree(net.buses.loc[sub_idx, ["x", "y"]].values)

    use_lonlat_tree = {"lon", "lat"}.issubset(net.buses.columns)
    if use_lonlat_tree:
        kdt_ll = KDTree(net.buses.loc[sub_idx, ["lon", "lat"]].values)
        tree_i = kdt_ll.query(plants_df.loc[:, ["lon", "lat"]].values)[1]
    else:
        tree_i = kdt.query(plants_df.loc[:, ["lon", "lat"]].values)[1]

    attached = plants_df.copy()
    attached["bus"] = sub_idx.append(pd.Index([np.nan]))[tree_i].astype(str)
    missing = ~attached["bus"].isin(net.buses.index)
    if missing.any():
        logger.warning(
            f"Found {missing.sum()} plants with non-existing bus assignments."
        )
    return attached


# -----------------------------
# 2) Main function
# -----------------------------
def attach_to_nearest_lv_bus(net: pypsa.Network, plants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach plants to buses using a power-to-voltage merit order:

    - HIGH plants:
        1) 500 kV within 25 km
        2) 230 kV within 50 km
        3) nearest LV substation (any voltage)

    - MEDIUM plants:
        1) 230 kV within 25 km
        2) nearest LV substation (any voltage)

    - LOW plants:
        1) 130/138 kV within 25 km
        2) nearest LV substation (any voltage)

    If lon/lat or v_nom are not available for buses, falls back to the
    old "nearest LV substation" behaviour.
    """
    if "substation_lv" not in net.buses.columns:
        raise ValueError("network.buses must contain a boolean 'substation_lv' column")

    if not {"lon", "lat"}.issubset(plants_df.columns):
        raise ValueError("plants_df must contain 'lon' and 'lat' columns")

    sub_idx = net.buses.query("substation_lv").index
    if len(sub_idx) == 0:
        raise ValueError("No LV substations found (substation_lv == True).")

    # If buses don’t have lon/lat or v_nom, use old simple behaviour
    if not {"lon", "lat", "v_nom"}.issubset(net.buses.columns):
        logger.warning(
            "Buses lack lon/lat or v_nom; using simple nearest-LV-substation attachment."
        )
        return _attach_simple_nearest(net, plants_df, sub_idx)

    buses_lv = net.buses  # shorthand

    # Voltage targets & distances (km) – tweak if you like
    HV_LEVEL = 500.0
    MV_LEVEL = 230.0
    LV_LEVEL = 138.0  # or 130.0, depending on your data

    DIST_HIGH_PRIMARY = 25.0  # 500 kV search radius
    DIST_HIGH_SECONDARY = 50.0  # 230 kV fallback radius for high plants
    DIST_MEDIUM = 25.0  # 230 kV search radius for medium plants
    DIST_LOW = 25.0  # 138 kV search radius for low plants

    V_TOL = 10.0  # tolerance around v_nom (kV)

    # Prepare bus groups by voltage
    idx_500 = _select_buses_by_voltage(buses_lv, sub_idx, HV_LEVEL, tol=V_TOL)
    idx_230 = _select_buses_by_voltage(buses_lv, sub_idx, MV_LEVEL, tol=V_TOL)
    idx_138 = _select_buses_by_voltage(buses_lv, sub_idx, LV_LEVEL, tol=V_TOL)

    # KD-trees in lon/lat space for each group and for all LV substations
    kdt_500, idx_500 = _build_kdtree_lonlat(buses_lv, idx_500)
    kdt_230, idx_230 = _build_kdtree_lonlat(buses_lv, idx_230)
    kdt_138, idx_138 = _build_kdtree_lonlat(buses_lv, idx_138)
    kdt_all, idx_all = _build_kdtree_lonlat(buses_lv, sub_idx)

    # Ensure we have size categories
    plants = plants_df.copy()
    if "p_nom" not in plants.columns:
        raise ValueError("plants_df must contain 'p_nom' column for size categorization.")
    if "size_cat" not in plants.columns:
        plants["size_cat"] = plants["p_nom"].apply(_categorize_plant_power)

    buses_assigned = []

    for _, row in plants.iterrows():
        plant_coord = np.array([row["lon"], row["lat"]])
        size_cat = row["size_cat"]
        bus_label = None

        if size_cat == "high":
            # 1) 500 kV within 25 km
            bus_label = _query_bus_within_radius(
                plant_coord, kdt_500, idx_500, DIST_HIGH_PRIMARY
            )
            # 2) 230 kV within 50 km
            if bus_label is None:
                bus_label = _query_bus_within_radius(
                    plant_coord, kdt_230, idx_230, DIST_HIGH_SECONDARY
                )
        elif size_cat == "medium":
            # 1) 230 kV within 25 km
            bus_label = _query_bus_within_radius(
                plant_coord, kdt_230, idx_230, DIST_MEDIUM
            )
        else:  # "low"
            # 1) 138 kV within 25 km (skip higher levels)
            bus_label = _query_bus_within_radius(
                plant_coord, kdt_138, idx_138, DIST_LOW
            )

        # Final fallback: nearest LV substation of any voltage
        if bus_label is None:
            bus_label = _query_bus_within_radius(
                plant_coord, kdt_all, idx_all, radius_km=10_000.0
            )  # effectively "any distance"

        buses_assigned.append(str(bus_label) if bus_label is not None else np.nan)

    attached = plants.copy()
    attached["bus"] = buses_assigned

    missing = ~attached["bus"].isin(net.buses.index)
    if missing.any():
        logger.warning(
            f"Found {missing.sum()} plants with non-existing bus assignments."
        )

    return attached


# Usage (unchanged call-site)
try:
    ppl_attached = attach_to_nearest_lv_bus(network, ppl_filtered)
    logger.info("Attached plants to nearest bus with power/voltage merit order.")
except Exception as e:
    logger.warning(f"Could not attach plants to buses: {e}")
    ppl_attached = ppl_filtered.copy()
    ppl_attached["bus"] = np.nan


ppl_attached.to_csv("ppl_attached.csv")

# %%
# 5) Safely add generators to the network

def add_generators_safe(net: pypsa.Network, pp_df: pd.DataFrame):
    needed = ["bus", "carrier", "p_nom"]
    if not set(needed).issubset(pp_df.columns):
        raise ValueError(f"Missing columns in plant df. Need at least: {needed}")

    # PyPSA expects per-generator rows. We’ll use index as names.
    added = 0
    for i in pp_df.index:
        row = pp_df.loc[[i]]
        try:
            
            net.madd("Generator",
                     row.index,
                     bus=row["bus"].values,
                     carrier=row["carrier"].values,
                     p_nom=row["p_nom"].values)
            added += 1
        except Exception as e:
            logger.warning(f"  ⚠ could not add {i}: {e}")
    logger.info(f"Generators added: {added}")

try:
    add_generators_safe(network, ppl_attached)
except Exception as e:
    logger.warning(f"Adding generators failed: {e}")


# %%
import os
import numpy as np
import pandas as pd
import xarray as xr
import unicodedata

#re_dict = {}

def _norm_province_name(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    # strip accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    # lower, trim, unify separators
    name = name.strip().lower()
    name = name.replace(" ", "_").replace("-", "_")
    # collapse multiple underscores
    while "__" in name:
        name = name.replace("__", "_")
    return name


# -----------------------------
# 1) Data loading / preparation
# -----------------------------
def load_province_centroids(centroids_csv_path: str):
    centroids = pd.read_csv(centroids_csv_path)

    for col in ["province", "longitude", "latitude"]:
        if col not in centroids.columns:
            raise ValueError(f"Centroid CSV must contain column '{col}'")

    centroids["province_norm"] = centroids["province"].map(_norm_province_name)

    cent_lon = centroids["longitude"].astype(float).values
    cent_lat = centroids["latitude"].astype(float).values
    cent_name_norm = centroids["province_norm"].values

    return centroids, cent_lon, cent_lat, cent_name_norm


def load_cutout_profiles(cutouts_dir: str, snapshots):
    solar_ds = xr.open_dataset(os.path.join(cutouts_dir, "solar.nc"))
    wind_ds  = xr.open_dataset(os.path.join(cutouts_dir, "onwind.nc"))

    # provinces from solar_ds.bus.data
    provinces_raw = pd.Index(solar_ds.bus.data)
    provinces_norm = provinces_raw.to_series().map(_norm_province_name)

    # map normalized -> original name/index
    norm_to_raw = dict(zip(provinces_norm.values, provinces_raw.values))

    # profile DataFrames (time x province_raw)
    solar_profiles = solar_ds["profile"].to_pandas()
    solar_profiles.columns = provinces_raw

    wind_profiles = wind_ds["profile"].to_pandas()
    wind_profiles.columns = provinces_raw

    # align to snapshots
    solar_profiles = solar_profiles.reindex(index=snapshots)
    wind_profiles  = wind_profiles.reindex(index=snapshots)

    return solar_profiles, wind_profiles, norm_to_raw


# -----------------------------
# 2) Network preparation
# -----------------------------
def ensure_generator_static_province(network):
    if "province" not in network.generators.columns:
        network.generators["province"] = pd.NA
    return network


def ensure_generators_p_max_pu_table(network):
    if "p_max_pu" not in network.generators_t:
        network.generators_t["p_max_pu"] = pd.DataFrame(
            1.0,
            index=network.snapshots,
            columns=network.generators.index,
        )
    return network


# -----------------------------
# 3) Geometry / province helpers
# -----------------------------
def find_province_for_coord(lon, lat, cent_lon, cent_lat, cent_name_norm) -> str:
    d2 = (cent_lon - lon) ** 2 + (cent_lat - lat) ** 2
    idx = int(np.argmin(d2))
    return cent_name_norm[idx]  # normalized name


def get_profile_for_generator(
    carrier: str,
    prov_raw: str,
    solar_profiles: pd.DataFrame,
    wind_profiles: pd.DataFrame,
    snapshots,
    gen_id: str,
):
    if carrier == "solar":
        prof = solar_profiles[prov_raw].astype(float)
    elif carrier == "onwind":
        prof = wind_profiles[prov_raw].astype(float)
    else:
        # only solar/onwind should reach here
        return pd.Series(0.0, index=snapshots)

    # normalize to max = 1
    max_val = prof.max()
    if pd.notna(max_val) and max_val > 0:
        prof = prof / max_val
        prof = prof.fillna(0.0)
    else:
        logger.warning(
            f"Profile for {gen_id} (province {prov_raw}) has non-positive max; using zeros."
        )
        prof = pd.Series(0.0, index=snapshots)

    return prof


def assign_profile_to_single_generator(
    gen_id: str,
    row: pd.Series,
    network,
    solar_profiles: pd.DataFrame,
    wind_profiles: pd.DataFrame,
    norm_to_raw: dict,
    cent_lon: np.ndarray,
    cent_lat: np.ndarray,
    cent_name_norm: np.ndarray,
):
    carrier = row["carrier"]
    bus_id = row["bus"]

    try:
        bus_lon = float(network.buses.at[bus_id, "lon"])
        bus_lat = float(network.buses.at[bus_id, "lat"])
    except KeyError:
        logger.warning(f"Bus {bus_id} not found in network.buses – skipping {gen_id}")
        return

    prov_norm = find_province_for_coord(bus_lon, bus_lat, cent_lon, cent_lat, cent_name_norm)

    # match centroid province to cutout province via normalized name
    if prov_norm not in norm_to_raw:
        logger.warning(
            f"Province '{prov_norm}' (from centroids) not in cutout provinces – skipping {gen_id}"
        )
        return

    prov_raw = norm_to_raw[prov_norm]  # name as used in solar_ds/wind_ds

    prof = get_profile_for_generator(
        carrier, prov_raw, solar_profiles, wind_profiles, network.snapshots, gen_id
    )

    # # --- Save to dict (keyed by generator id) ---
    # re_dict[gen_id] = {
    #     "bus": bus_id,
    #     "carrier": carrier,
    #     "province": prov_raw,
    #     "p_max_pu": prof.copy(),  # keep as Series
    # }

    # --- Place profile in network.generators_t ---
    network.generators_t["p_max_pu"].loc[:, gen_id] = prof

    # --- Store province in static generators table ---
    network.generators.at[gen_id, "province"] = prov_raw


# -----------------------------
# 4) Main pipeline function
# -----------------------------
def assign_re_profiles(network):
    # --- Select renewable generators ---
    mask = ppl_attached["carrier"].isin(["solar", "onwind"])
    ren_gens = ppl_attached[mask]

    # --- Load centroids CSV ---
    centroids_csv_path = os.path.join(dirs["data/raw/gadm"], "EC_centroids.csv")
    centroids, cent_lon, cent_lat, cent_name_norm = load_province_centroids(centroids_csv_path)

    # --- Load cutouts ONCE ---
    cutouts_dir = dirs["data/raw/cutouts"]
    solar_profiles, wind_profiles, norm_to_raw = load_cutout_profiles(
        cutouts_dir, network.snapshots
    )

    # --- Ensure network tables/columns ---
    network = ensure_generator_static_province(network)
    network = ensure_generators_p_max_pu_table(network)

    # f) & g) loop over renewable generators
    for gen_id, row in ren_gens.iterrows():
        assign_profile_to_single_generator(
            gen_id=gen_id,
            row=row,
            network=network,
            solar_profiles=solar_profiles,
            wind_profiles=wind_profiles,
            norm_to_raw=norm_to_raw,
            cent_lon=cent_lon,
            cent_lat=cent_lat,
            cent_name_norm=cent_name_norm,
        )

    logger.info(
        "Assigned province-based p_max_pu profiles for solar/onwind generators "
        "and stored them in network.generators_t['p_max_pu']."
    )

    return network

network = assign_re_profiles(network)

# keep using global re_dict as before
#re_dict


# %%
# 6) Technology costs merge
def merge_tech_costs(network):
    try:
        cost_filename = join(dirs["data/raw/generation"], "technology_cost.csv")
        costs = pd.read_csv(cost_filename, index_col=0, comment="#")
        # Expecting index to be carrier names; columns e.g. marginal_cost, capital_cost, etc.
        # Remove any pre-existing columns to avoid duplicate merge keys
        for c in ["marginal_cost", "capital_cost"]:
            if c in network.generators.columns:
                network.generators.drop(columns=[c], inplace=True)

        network.generators = pd.merge(
            network.generators,
            costs,
            left_on="carrier",
            right_index=True,
            how="left",
            suffixes=("", "_cost"),
        )
        logger.info("Merged tech costs into network.generators")
    except Exception as e:
        logger.warning(f"Could not merge technology costs: {e}")

    return network

merge_tech_costs(network)

# %%
def merge_carriers(network):
    # Load carriers and assign WITHOUT reset_index (critical!)
    carriers_path = join(dirs["data/raw/generation"], "carriers.csv")
    carriers_df = pd.read_csv(carriers_path, index_col=0)

    # Optional: ensure expected columns exist (PyPSA only requires a table indexed by carrier name;
    # extra columns like color/nice_name are fine)
    # Example expected index entries (must include every carrier used by any component):
    needed_carriers = set(network.generators.carrier.unique()) \
                    | set(network.lines.get("carrier", pd.Series([], dtype=str)).astype(str).unique()) \
                    | set(network.buses.get("carrier", pd.Series([], dtype=str)).astype(str).unique())

    missing_in_table = sorted(c for c in needed_carriers if c not in carriers_df.index)
    if missing_in_table:
        # Add stubs so PyPSA considers them "defined"
        for c in missing_in_table:
            carriers_df.loc[c, ["co2_emissions","color","nice_name","max_growth","max_relative_growth"]] = [0.0, "#999999", c, np.inf, 0.0]

    network.carriers = carriers_df  # <- keep the index as carrier names (do NOT reset_index)
    logger.info(f"Carriers assigned. Count={len(network.carriers)}; added stubs: {missing_in_table}")
    return network

merge_carriers(network)



# %%
# 5) Double-check that every component’s carrier exists
def _check_carrier_defined(df, label):
    if "carrier" not in df.columns or df.empty:
        return
    used = set(df["carrier"].astype(str).unique())
    missing = sorted(c for c in used if c not in network.carriers.index)
    if missing:
        logger.warning(f"{label}: carriers not defined in network.carriers: {missing}")
    else:
        logger.info(f"{label}: all carriers are defined.")

_check_carrier_defined(network.generators, "Generators")
_check_carrier_defined(network.lines, "Lines")
_check_carrier_defined(network.buses, "Buses")



#6) (Optional) Fix bus/line carriers quickly
# If buses/lines have blank carriers but you want them defined:
def fix_bus_line_carriers(network):
    # Buses
    if "carrier" in network.buses.columns:
        network.buses["carrier"] = network.buses["carrier"].fillna("AC").astype(str)
    else:
        network.buses["carrier"] = "AC"

    # Lines
    if "carrier" in network.lines.columns:
        network.lines["carrier"] = network.lines["carrier"].replace({None: "AC"}).fillna("AC").astype(str)
    else:
        network.lines["carrier"] = "AC"

    # Ensure 'AC' entry exists in carriers table
    if "AC" not in network.carriers.index:
        network.carriers.loc["AC", ["co2_emissions","color","nice_name","max_growth","max_relative_growth"]] = [
            0.0, "#888888", "AC", np.inf, 0.0
        ]
    return network

fix_bus_line_carriers(network)

# %% [markdown]
# Assertion that the buses RHS and LHS of the equation are balanced

# %%


# %%
# Patch transformers with zero r/x (use realistic small per-unit values)
def patch_transformers(network):
    if len(network.transformers):
        tr = network.transformers

        # If r_pu/x_pu columns missing, create them
        if "r_pu" not in tr.columns: tr["r_pu"] = np.nan
        if "x_pu" not in tr.columns: tr["x_pu"] = np.nan

        zero_r = tr["r_pu"].fillna(0) == 0
        zero_x = tr["x_pu"].fillna(0) == 0

        # Assign small but sensible defaults
        tr.loc[zero_r, "r_pu"] = 0.01    # ~1% p.u. resistance
        tr.loc[zero_x, "x_pu"] = 0.10    # ~10% p.u. reactance

        network.transformers = tr
        logger.info(f"Patched transformers: r_pu->0.01 for {zero_r.sum()} rows; x_pu->0.10 for {zero_x.sum()} rows.")

patch_transformers(network)

# %%
def create_load_shedding(network):
    # Ensure a 'load_shedding' carrier exists
    if "load_shedding" not in network.carriers.index:
        network.carriers.loc["load_shedding", ["co2_emissions","color","nice_name","max_growth","max_relative_growth"]] = [
            0.0, "#DD1818", "Load Shedding", np.inf, 0.0
        ]

    # Add a per-bus emergency generator (very high cost)
    LS_COST = 100e4  # €/MWh — ensure above any real generator marginal cost
    CS_COST = 1e10
    to_add = []
    for b in network.buses.index:
        name = f"LS_{b}"
        if name not in network.generators.index:
            to_add.append(name)

    if to_add:
        network.madd(
            "Generator",
            names=to_add,
            bus=[b for b in network.buses.index],
            carrier="load_shedding",
            p_nom=10000,                 # plenty of capacity
            marginal_cost=LS_COST,
            capital_cost = CS_COST,
        )
        logger.info(f"Added {len(to_add)} load-shedding generators (cost={LS_COST}).")

    return network

create_load_shedding(network)

# %%
def patch_transformers_all(n: pypsa.Network,
                           r_pu_default=0.01,   # 1% p.u. resistance
                           x_pu_default=0.05,   # 10% p.u. reactance
                           r_abs_default=0.01,  # absolute fallback if 'r' exists
                           x_abs_default=0.05): # absolute fallback if 'x' exists
    if n.transformers.empty:
        return

    tr = n.transformers.copy()

    # --- ensure per-unit columns exist
    for col in ["r_pu", "x_pu"]:
        if col not in tr.columns:
            tr[col] = np.nan

    # --- try to fill from transformer_types (preferred)
    if hasattr(n, "transformer_types") and not n.transformer_types.empty and "type" in tr.columns:
        tt_cols = [c for c in ["r_pu","x_pu"] if c in n.transformer_types.columns]
        if tt_cols:
            tr = tr.join(n.transformer_types[tt_cols], on="type", rsuffix="_type")
            # where missing/zero, take values from *_type
            for col in ["r_pu","x_pu"]:
                if f"{col}_type" in tr.columns:
                    need = tr[col].isna() | (tr[col] == 0)
                    tr.loc[need & tr[f"{col}_type"].notna(), col] = tr.loc[need, f"{col}_type"]
            tr.drop(columns=[c for c in ["r_pu_type","x_pu_type"] if c in tr.columns], inplace=True)

    # --- per-unit defaults
    tr.loc[tr["r_pu"].isna() | (tr["r_pu"] == 0), "r_pu"] = r_pu_default
    tr.loc[tr["x_pu"].isna() | (tr["x_pu"] == 0), "x_pu"] = x_pu_default

    # --- absolute columns: some PyPSA versions/tables keep r/x as absolute fields
    # create if missing (so we can silence "zero r" warnings that look at 'r')
    if "r" not in tr.columns: tr["r"] = np.nan
    if "x" not in tr.columns: tr["x"] = np.nan

    # If you know the formula to convert p.u. -> absolute in your setup, apply it here.
    # Lacking nameplate/base data, set small non-zero fallbacks to avoid singularities:
    tr.loc[tr["r"].isna() | (tr["r"] == 0), "r"] = r_abs_default
    tr.loc[tr["x"].isna() | (tr["x"] == 0), "x"] = x_abs_default

    # --- make sure rating is positive (otherwise no flow vars)
    if "s_nom" in tr.columns:
        tr.loc[tr["s_nom"].fillna(0) <= 0, "s_nom"] = 1.0

    # write back
    n.transformers.loc[tr.index, tr.columns] = tr

    # --- final assertions in logs
    n_rpu_zero = (n.transformers["r_pu"].fillna(0) == 0).sum()
    n_xpu_zero = (n.transformers["x_pu"].fillna(0) == 0).sum()
    n_r_zero   = (n.transformers["r"].fillna(0)   == 0).sum()
    n_x_zero   = (n.transformers["x"].fillna(0)   == 0).sum()
    logger.info(f"Transformer patch summary: r_pu zeros={n_rpu_zero}, x_pu zeros={n_xpu_zero}, r zeros={n_r_zero}, x zeros={n_x_zero}")

# run before optimize()
patch_transformers_all(network)


# %%
import time

def patch_lines(network):
    network.lines['x'] = 0.1
    network.lines['r'] = 0.01

patch_lines(network)


# %%


