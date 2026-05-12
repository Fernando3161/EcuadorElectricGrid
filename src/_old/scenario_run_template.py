# %% [markdown]
# Setup environment and load the base PyPSA-Earth network for a specified country.
# 

# %%
# General Factors for scenario testing
SCENARIO = "A"
SNAPPING_H = 1
SCALING_FACTOR = 23/49.2
LEN_OPT = 24*30

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
parent_dir = Path(os.getcwd())     
     # project/
print(parent_dir)
LOG_FILE = join(parent_dir, "logs.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- Project paths helper
#sys.path.append(str(parent_dir))
#from src.paths import all_dirs  # must exist in your repo
#dirs = all_dirs()

# --- Add PyPSA-Earth scripts to PATH (assumes repo layout: <project>/../pypsa-earth/scripts)
scripts_path = os.path.join(parent_dir, "pypsa-earth", "scripts")
assert os.path.isdir(scripts_path), f"Path not found: {scripts_path}"
sys.path.append(scripts_path)

from paths import all_dirs  # must exist in your repo
dirs = all_dirs()

# %%
# 1) Load Networks
network_dir = dirs["data/processed/networks"]
network_files = [
    "network_original",
    "network_snapped",
    "network_expanded",
    "network_expanded_no_orphans",
    "network_nuclear",
    "network_prod_mix",
    "network_base",
]

logger.info("Loading .nc networks …")
networks_dict = {}
for nf in network_files:
    f = join(network_dir, f"{nf}.nc")
    if os.path.isfile(f):
        try:
            networks_dict[nf] = pypsa.Network(f)
            logger.info(f"  ✓ loaded {nf}.nc")
        except Exception as e:
            logger.warning(f"  ⚠ failed to load {nf}.nc: {e}")
    else:
        logger.warning(f"  ⚠ missing file: {f}")

# Work on a copy of the base network
assert "network_base" in networks_dict, "network_base.nc not found/loaded."
network = networks_dict["network_base"].copy()



# %%
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
network = prune_network_min_voltage(networks_dict["network_base"], v_threshold_kv=137)
                                    #137.0)



# %%
# 2) Loads: add p_set time series
# 2) Loads: add p_set time series with nearest-bus remapping for unknown buses
try:
    path_loads = dirs["data/processed/scaled_loads"]
    load_profile_name = "load_base_2030_linear.csv"
    load_profile_file = join(path_loads, load_profile_name)
    logger.info(f"Loading load profile: {load_profile_file}")
    load_profile = pd.read_csv(load_profile_file, index_col=0, parse_dates=True)
    scaling_factor = SCALING_FACTOR
    load_profile = load_profile*scaling_factor
    logger.info(f"Scaled all load profile values by factor {scaling_factor}")
    # --- Ensure column names are strings
    load_cols = [str(c) for c in load_profile.columns]
    load_profile.columns = load_cols

    # --- Existing buses in the active network
    buses_now = network.buses.copy()
    buses_now.index = buses_now.index.astype(str)
    existing_buses = set(buses_now.index)

    # --- Find missing bus labels
    missing = [c for c in load_cols if c not in existing_buses]
    if missing:
        logger.warning(f"{len(missing)} load buses not found; reassigning to nearest existing bus.")
    else:
        logger.info("All load buses found in current network.")

    # --- Build KDTree on current network buses
    from scipy.spatial import cKDTree as KDTree
    use_lonlat = {"lon", "lat"}.issubset(buses_now.columns)
    if use_lonlat:
        coords_now = buses_now.loc[:, ["lon", "lat"]].to_numpy(dtype=float)
    else:
        coords_now = buses_now.loc[:, ["x", "y"]].to_numpy(dtype=float)
    kdt_now = KDTree(coords_now)

    # --- Helper: try to get coordinates for a missing bus from any fallback network
    def find_bus_coords_in_fallbacks(bus_name: str):
        for nf, net_fb in networks_dict.items():
            try:
                df = net_fb.buses.copy()
                df.index = df.index.astype(str)
                if bus_name in df.index:
                    if {"lon", "lat"}.issubset(df.columns):
                        return ("lonlat",
                                float(df.loc[bus_name, "lon"]),
                                float(df.loc[bus_name, "lat"]))
                    elif {"x", "y"}.issubset(df.columns):
                        return ("xy",
                                float(df.loc[bus_name, "x"]),
                                float(df.loc[bus_name, "y"]))
            except Exception:
                pass
        return None

    # --- Build a bus assignment vector aligned with load columns
    bus_assignments = []
    remap_report = []

    for c in load_cols:
        if c in existing_buses:
            bus_assignments.append(c)  # unchanged
            continue

        # Try to recover coordinates for this bus label from other networks
        info = find_bus_coords_in_fallbacks(c)
        if info is not None:
            kind, bx, by = info
            # Query KDTree (always built for current network in the coordinate type we have)
            if use_lonlat and kind == "lonlat":
                q = np.array([[bx, by]])
            elif (not use_lonlat) and kind == "xy":
                q = np.array([[bx, by]])
            else:
                # Coordinate kind mismatch; cannot reliably transform → use crude nearest by lon/lat if both exist,
                # else by x/y. If mismatch, try whichever exists in current network.
                if use_lonlat and kind == "xy":
                    # We only have xy for missing, but KDTree is lon/lat → fallback to closest by index 0
                    # (transparent about this in logs)
                    remap_report.append((c, None, None, "coord_kind_mismatch"))
                    # Pick the closest *by index* fallback: use network's geometric center
                    # (KDTree needs a point; we’ll use median lon/lat)
                    q = np.nanmedian(coords_now, axis=0)[None, :]
                elif (not use_lonlat) and kind == "lonlat":
                    remap_report.append((c, None, None, "coord_kind_mismatch"))
                    q = np.nanmedian(coords_now, axis=0)[None, :]
                else:
                    q = np.nanmedian(coords_now, axis=0)[None, :]
            idx = int(kdt_now.query(q)[1][0])
            nearest_bus = buses_now.index[idx]
            bus_assignments.append(nearest_bus)
            remap_report.append((c, bx, by, f"mapped_to:{nearest_bus}"))
        else:
            # No coordinates found anywhere → attach to globally nearest by network centroid
            idx = int(kdt_now.query(np.nanmedian(coords_now, axis=0)[None, :])[1][0])
            nearest_bus = buses_now.index[idx]
            bus_assignments.append(nearest_bus)
            remap_report.append((c, None, None, f"mapped_to:{nearest_bus};reason:no_coords"))

    if remap_report:
        logger.info("Load bus remapping performed (missing → nearest existing):")
        for r in remap_report:
            logger.info(f"  load '{r[0]}' → {r[3]} (src_coords: {r[1]}, {r[2]})")

    # --- Finally add Loads with reassigned buses
    network.madd("Load", load_cols, bus=bus_assignments, p_set=load_profile)

    # Peak (GW)
    peak_load = round(network.loads_t.p_set.T.sum().max() / 1000, 1)
    logger.info(f"Peak load in p_set time series: {peak_load} GW")

except Exception as e:
    logger.warning(f"Could not attach loads: {e}")



# %%
# ======================================================================
#                      3) Power plants table (CSV)
# ======================================================================
ppl_csv = join(dirs["data/processed/generation"], "powerplants_all.csv")
assert os.path.isfile(ppl_csv), f"Power plants CSV not found: {ppl_csv}"
ppl_raw = pd.read_csv(ppl_csv, index_col=0)

# --- OPTIONAL (recommended): normalize using powerplantmatching
ppl = ppl_raw.copy()
try:
    import powerplantmatching as pm

    # Load PPM config (pypsa-earth config shipped in your repo)
    ppmatching = os.path.join(
        Path(os.getcwd()), "pypsa-earth", "configs", "powerplantmatching_config.yaml"
    )
    print(ppmatching)
    config = pm.get_config(ppmatching)
    config["target_countries"] = ["EC"]  # Ecuador

    # These utilities are in pm.powerplant
    # 1) fill missing commissioning years
    ppl_f = ppl.powerplant.fill_missing_commissioning_years()
    # 2) convert to PyPSA-compatible column names (bus/generator naming conventions)
    ppl_p = ppl_f.powerplant.to_pypsa_names()

        # Drop all-empty columns that may be introduced by upstream merges
    ppl = ppl_p.dropna(axis=1, how="all")

    logger.info("Applied powerplantmatching normalization (fill years + to_pypsa_names).")
except Exception as e:
    logger.warning(f"powerplantmatching step skipped (using raw CSV): {e}")
    ppl = ppl_raw.copy()


ppl.columns

# ---- Normalize capacities and carriers (ES->EN) on the (possibly PPM-normalized) table
def _normalize_text(s):
    import unicodedata, re
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s)
    return s

# capacity as float (handles "1'500,00")
if "p_nom" in ppl.columns:
    ppl["p_nom"] = (
        ppl["p_nom"].astype(str).str.replace("'", "", regex=False).str.replace(",", ".", regex=False)
    )
    ppl["p_nom"] = pd.to_numeric(ppl["p_nom"], errors="coerce").fillna(0.0)
else:
    logger.warning("Column 'p_nom' not found in plants table after normalization.")

# map carriers
if "carrier" in ppl.columns:
    ppl["carrier_norm"] = ppl["carrier"].apply(_normalize_text)
    carrier_map_es2en = {
        "hidraulica": "hydro",
        "hidroelectrico": "hydro",
        "termica": "oil",            # general fallback, could also be "natural gas" or "coal" depending on subtype
        "termoelectrico": "oil",
        "biomasa": "biomass",
        "fotovoltaica": "solar",
        "solar": "solar",
        "eolica": "onwind",
        "eolico": "onwind",
        "biogas": "biomass",         # or "biogas" if you keep it distinct later
        "ernc": "onwind",            # “Energía Renovable No Convencional”, usually wind/solar, adjust if needed
        "nuclear": "nuclear",
        "geotermica": "geothermal",
        "carbón": "coal",
        "carbon": "coal",
        "lignito": "lignite",
        "gas natural": "natural gas",
        "gas": "natural gas",
        "phs": "PHS",
        "pasada": "ror",             # hydro run-of-river
        "embalse": "hydro",          # reservoir-type hydro
        "offshore": "offwind-ac",    # adjust if you know DC vs AC
        "offshore_dc": "offwind-dc",
    }
    ppl["carrier"] = ppl["carrier_norm"].map(carrier_map_es2en).fillna(ppl["carrier_norm"])
else:
    logger.warning("Column 'carrier' not found in plants table after normalization.")

# ---- Filter: SNI, by year and minimum size
FILTERING_POWER = 0.0   # kW threshold
DATE_IN = 2017
if {"component", "DateIn", "p_nom"}.issubset(ppl.columns):
    ppl_connected = ppl[ppl["component"] == "S.N.I."]
    ppl_connected_current = ppl_connected[ppl_connected["DateIn"] <= DATE_IN]
    ppl_filtered = ppl_connected_current[ppl_connected_current["p_nom"] >= FILTERING_POWER].copy()
else:
    logger.warning("Required columns for filtering missing; skipping filter.")
    ppl_filtered = ppl.copy()

# ---- Totals
def safe_sum(series):
    try: return float(series.sum())
    except: return np.nan

total_capacity = safe_sum(ppl.get("p_nom", pd.Series(dtype=float)))
total_connected = safe_sum(ppl_connected.get("p_nom", pd.Series(dtype=float))) if 'ppl_connected' in locals() else np.nan
total_connected_current = safe_sum(ppl_connected_current.get("p_nom", pd.Series(dtype=float))) if 'ppl_connected_current' in locals() else np.nan
total_filtered = safe_sum(ppl_filtered.get("p_nom", pd.Series(dtype=float)))
logger.info(
    "Capacities [kW]\n"
    f"  total:                {total_capacity:,.0f}\n"
    f"  connected (SNI):      {total_connected:,.0f}\n"
    f"  connected <= {DATE_IN}: {total_connected_current:,.0f}\n"
    f"  filtered (>= {FILTERING_POWER:.0f} kW): {total_filtered:,.0f}"
)


# %%
# 4) Attach filtered plants to nearest LV substation

from scipy.spatial import cKDTree as KDTree

def attach_to_nearest_lv_bus(net: pypsa.Network, plants_df: pd.DataFrame) -> pd.DataFrame:
    """Attach plants to nearest bus among those marked substation_lv."""
    if "substation_lv" not in net.buses.columns:
        raise ValueError("network.buses must contain a boolean 'substation_lv' column")

    if not {"lon", "lat"}.issubset(plants_df.columns):
        raise ValueError("plants_df must contain 'lon' and 'lat' columns")

    sub_idx = net.buses.query("substation_lv").index
    if len(sub_idx) == 0:
        raise ValueError("No LV substations found (substation_lv == True).")

    kdt = KDTree(net.buses.loc[sub_idx, ["x", "y"]].values)

    # Query nearest bus for each plant lon/lat -> need plant x/y; network.buses stores x,y in projected units.
    # If network.buses has lon/lat columns, prefer those:
    if {"lon", "lat"}.issubset(net.buses.columns):
        # Build a lon/lat->index KDTree as fallback using lon/lat and not x/y
        # But your buses KDTree above already used x/y; keep consistent by translating plants to bus CRS when available.
        pass

    # Best-effort: assume bus x/y ~ proj coords corresponding to lon/lat mapping already in network
    # If network carries bus lon/lat, we project plants into the same 'x/y' via a crude nearest in lon/lat space:
    # For robustness (and speed), use lon/lat KDTree when bus lon/lat exist:
    use_lonlat_tree = {"lon", "lat"}.issubset(net.buses.columns)
    if use_lonlat_tree:
        kdt_ll = KDTree(net.buses.loc[sub_idx, ["lon", "lat"]].values)
        tree_i = kdt_ll.query(plants_df.loc[:, ["lon", "lat"]].values)[1]
    else:
        # fallback to x/y KDTree (will work if plant lon/lat columns are actually x/y already)
        tree_i = kdt.query(plants_df.loc[:, ["lon", "lat"]].values)[1]

    attached = plants_df.copy()
    attached["bus"] = sub_idx.append(pd.Index([np.nan]))[tree_i].astype(str)
    missing = ~attached["bus"].isin(net.buses.index)
    if missing.any():
        logger.warning(f"Found {missing.sum()} plants with non-existing bus assignments.")
    return attached

try:
    ppl_attached = attach_to_nearest_lv_bus(network, ppl_filtered)
except Exception as e:
    logger.warning(f"Could not attach plants to LV buses: {e}")
    ppl_attached = ppl_filtered.copy()
    ppl_attached["bus"] = np.nan


# %%


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
# gens_clean from previous step
gens_clean = (
    network.generators[
        ~network.generators["carrier"].str.contains("shedding", case=False, na=False)
    ]
    .sort_values("p_nom")
)

# Add the bus voltage (v_nom) to gens_clean
gens_clean = gens_clean.join(
    network.buses["v_nom"].rename("bus_v_nom"),
    on="bus"
)

gens_clean[["bus", "carrier", "p_nom", "bus_v_nom"]]



# %%
import os
import numpy as np
import pandas as pd
import xarray as xr
import unicodedata

re_dict = {}

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

# --- Select renewable generators ---
mask = ppl_attached["carrier"].isin(["solar", "onwind"])
ren_gens = ppl_attached[mask]

# --- Load centroids CSV ---
centroids_csv_path = os.path.join(dirs["data/raw/gadm"], "EC_centroids.csv")
centroids = pd.read_csv(centroids_csv_path)

for col in ["province", "longitude", "latitude"]:
    if col not in centroids.columns:
        raise ValueError(f"Centroid CSV must contain column '{col}'")

centroids["province_norm"] = centroids["province"].map(_norm_province_name)
cent_lon = centroids["longitude"].astype(float).values
cent_lat = centroids["latitude"].astype(float).values
cent_name_norm = centroids["province_norm"].values

# --- Load cutouts ONCE ---
solar_ds = xr.open_dataset(os.path.join(dirs["data/raw/cutouts"], "solar.nc"))
wind_ds  = xr.open_dataset(os.path.join(dirs["data/raw/cutouts"], "onwind.nc"))

# a) provinces from solar_ds.bus.data
provinces_raw = pd.Index(solar_ds.bus.data)
provinces_norm = provinces_raw.to_series().map(_norm_province_name)

# map normalized -> original name/index
norm_to_raw = dict(zip(provinces_norm.values, provinces_raw.values))

# b) profile DataFrames (time x province_raw)
solar_profiles = solar_ds["profile"].to_pandas()
solar_profiles.columns = provinces_raw

wind_profiles = wind_ds["profile"].to_pandas()
wind_profiles.columns = provinces_raw

# align to snapshots
solar_profiles = solar_profiles.reindex(index=network.snapshots)
wind_profiles  = wind_profiles.reindex(index=network.snapshots)

# --- Ensure province column in generators (static) ---
if "province" not in network.generators.columns:
    network.generators["province"] = pd.NA

# --- Ensure p_max_pu time-series table exists ---
if "p_max_pu" not in network.generators_t:
    network.generators_t["p_max_pu"] = pd.DataFrame(
        1.0,
        index=network.snapshots,
        columns=network.generators.index,
    )

def find_province_for_coord(lon, lat) -> str:
    d2 = (cent_lon - lon) ** 2 + (cent_lat - lat) ** 2
    idx = int(np.argmin(d2))
    return cent_name_norm[idx]  # normalized name

# f) & g) loop over renewable generators
for gen_id, row in ren_gens.iterrows():
    carrier = row["carrier"]
    bus_id = row["bus"]

    try:
        bus_lon = float(network.buses.at[bus_id, "lon"])
        bus_lat = float(network.buses.at[bus_id, "lat"])
    except KeyError:
        logger.warning(f"Bus {bus_id} not found in network.buses – skipping {gen_id}")
        continue

    prov_norm = find_province_for_coord(bus_lon, bus_lat)

    # match centroid province to cutout province via normalized name
    if prov_norm not in norm_to_raw:
        logger.warning(
            f"Province '{prov_norm}' (from centroids) not in cutout provinces – skipping {gen_id}"
        )
        continue

    prov_raw = norm_to_raw[prov_norm]  # name as used in solar_ds/wind_ds

    if carrier == "solar":
        prof = solar_profiles[prov_raw].astype(float)
    elif carrier == "onwind":
        prof = wind_profiles[prov_raw].astype(float)
    else:
        continue

    # normalize to max = 1
    max_val = prof.max()
    if pd.notna(max_val) and max_val > 0:
        prof = prof / max_val
        prof = prof.fillna(0.0)   # reassign!
    else:
        logger.warning(
            f"Profile for {gen_id} (province {prov_raw}) has non-positive max; using zeros."
        )
        prof = pd.Series(0.0, index=network.snapshots)

    # --- Save to dict (keyed by generator id) ---
    re_dict[gen_id] = {
        "bus": bus_id,
        "carrier": carrier,
        "province": prov_raw,
        "p_max_pu": prof.copy(),  # keep as Series
    }

    # --- Place profile in network.generators_t ---
    network.generators_t["p_max_pu"].loc[:, gen_id] = prof

    # --- Store province in static generators table ---
    network.generators.at[gen_id, "province"] = prov_raw

logger.info(
    "Assigned province-based p_max_pu profiles for solar/onwind generators "
    "and stored them in network.generators_t['p_max_pu']."
)

re_dict


# %%
# 6) Technology costs merge
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


# %% [markdown]
# Assertion that the buses RHS and LHS of the equation are balanced

# %%
import pandas as pd
import networkx as nx

n = network  # just a shorter alias

# 1) Check you actually have snapshots and they match your time series
print("Snapshots:", n.snapshots[:3], "... total:", len(n.snapshots))

# 2) Buses that carry any non-zero RHS (net fixed injections/withdrawals)
rhs_load = (n.loads_t.p_set.groupby(n.loads.bus, axis=1).sum()
            if not n.loads_t.p_set.empty else pd.DataFrame(index=n.snapshots))
rhs_other = 0
# add other fixed RHS terms if you use them, e.g. fixed generator p_set, shunts, etc.

rhs_by_bus = rhs_load.sum(axis=0)  # sum over time
rhs_nonzero = rhs_by_bus[abs(rhs_by_bus) > 1e-6]

print("Buses with non-zero RHS count:", rhs_nonzero.shape[0])

# 3) Buses that have any LHS variables attached (anything that can balance)
lhs_buses = set()

# Generators with capacity (fixed or extendable)
if len(n.generators):
    cap = n.generators.p_nom.copy()
    cap[n.generators.p_nom_extendable.fillna(False)] = cap.where(~n.generators.p_nom_extendable, 1.0)
    lhs_buses |= set(n.generators.bus[cap > 0])

# StorageUnits / Stores
if len(n.storage_units):
    cap = n.storage_units.p_nom.copy()
    cap[n.storage_units.p_nom_extendable.fillna(False)] = cap.where(~n.storage_units.p_nom_extendable, 1.0)
    lhs_buses |= set(n.storage_units.bus[cap > 0])

if len(n.stores):
    cap = n.stores.e_nom.copy()
    cap[n.stores.e_nom_extendable.fillna(False)] = cap.where(~n.stores.e_nom_extendable, 1.0)
    lhs_buses |= set(n.stores.bus[cap > 0])

# Lines / Transformers (any positive rating yields flow variables)
if len(n.lines):
    lhs_buses |= set(n.lines.bus0[n.lines.s_nom > 0])
    lhs_buses |= set(n.lines.bus1[n.lines.s_nom > 0])

if len(n.transformers):
    lhs_buses |= set(n.transformers.bus0[n.transformers.s_nom > 0])
    lhs_buses |= set(n.transformers.bus1[n.transformers.s_nom > 0])

# Links (DC/HVDC links also provide variables)
if len(n.links):
    lhs_buses |= set(n.links.bus0[n.links.p_nom > 0])
    lhs_buses |= set(n.links.bus1[n.links.p_nom > 0])

problem_buses = [b for b in rhs_nonzero.index if b not in lhs_buses]
print("Buses with RHS≠0 but no LHS variables:", problem_buses)

# 4) Also check for islands without supply
G = n.graph()
islands = list(nx.connected_components(G))
print(f"Connected components: {len(islands)}")


# %%
b = "90"  # o 90, según tu índice; abajo normalizamos a str

def _mask_bus(df, col="bus", bus=b):
    if df.empty or col not in df.columns: 
        return df.iloc[0:0]
    return df[df[col].astype(str) == str(bus)]

loads_90        = _mask_bus(network.loads, "bus")
gens_90         = _mask_bus(network.generators, "bus")
stores_90       = _mask_bus(network.stores, "bus")
sus_90          = _mask_bus(network.storage_units, "bus")
shunts_90       = _mask_bus(network.shunt_impedances, "bus") if hasattr(network, "shunt_impedances") else network.buses.iloc[0:0]

lines_90_0      = _mask_bus(network.lines, "bus0")
lines_90_1      = _mask_bus(network.lines, "bus1")
trafos_90_0     = _mask_bus(network.transformers, "bus0")
trafos_90_1     = _mask_bus(network.transformers, "bus1")
links_90_0      = _mask_bus(network.links, "bus0")
links_90_1      = _mask_bus(network.links, "bus1")

print("Loads @90:\n", loads_90)
print("Generators @90:\n", gens_90)
print("Stores @90:\n", stores_90)
print("StorageUnits @90:\n", sus_90)
print("Shunts @90:\n", shunts_90)

print("Lines touching 90 (as bus0):\n", lines_90_0.index.tolist())
print("Lines touching 90 (as bus1):\n", lines_90_1.index.tolist())
print("Transformers touching 90 (bus0):\n", trafos_90_0.index.tolist())
print("Transformers touching 90 (bus1):\n", trafos_90_1.index.tolist())
print("Links touching 90 (bus0):\n", links_90_0.index.tolist())
print("Links touching 90 (bus1):\n", links_90_1.index.tolist())


# %%
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

# %%
# Patch transformers with zero r/x (use realistic small per-unit values)
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


# %%
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
network.lines['x'] = 0.1
network.lines['r'] = 0.01



# %%

network.generators_t.p_max_pu


# %%
ren_gens.to_csv("re_gens.csv")

# %%
output_path = dirs["data/processed/networks"]

filename = join(output_path, "network_base_filled.nc")
network.export_to_netcdf(filename)
logging.info(f"Network exported to {filename} ")

# %%
network.generators.to_csv("network_gens.csv")

# %%
solver = "highs"
# Select first 4 snapshots
snapshots_subset = network.snapshots[:LEN_OPT]
logging.info(f"Starting optimization with solver='{solver}' for {len(snapshots_subset)} snapshots")

start_time = time.time()
# Optimize only over those 4 snapshots

try:
    if SNAPPING_H and SNAPPING_H>1:
       # 3-hourly subset
        opts = {"threads": 8, "presolve": "on"}  # set to your core count
        snapshots_dt = pd.to_datetime(network.snapshots)
        snap_h = snapshots_dt[::SNAPPING_H]
        network.set_snapshots(snap_h)
        network.optimize(snapshots=snap_h, solver_name="highs",solver_options=opts) 
        logging.info(f"Optimization done with {SNAPPING_H}h snaps")
    else:
        opts = {"threads": 8, "presolve": "on"}  # set to your core count
        network.optimize(snapshots=snapshots_subset, solver_name="highs", solver_options=opts)
    #network.optimize(snapshots=snapshots_subset, solver_name=solver)
    elapsed = time.time() - start_time
    logging.info(f"Optimization completed successfully in {elapsed:.2f} seconds.")
except Exception as e:
    elapsed = time.time() - start_time
    logging.error(f"Optimization failed after {elapsed:.2f} seconds. Error: {e}")

# %%
import os

output_path = dirs["results/networks"]

# Ensure the directory exists
os.makedirs(output_path, exist_ok=True)


# Save each network to NetCDF in the output folder
days = int(LEN_OPT/24)
filename = join(output_path, f"network_solved_{SCENARIO}_{days}.nc")
network.export_to_netcdf(filename)
logging.info(f"Solved network exported to {filename} ")