import numpy as np
import pandas as pd
import networkx as nx
from shapely import wkt
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


# -------- Public API -------- #

def evaluate_network(
    n,
    *,
    lon_min: float = -92.0,
    lon_max: float = -74.0,
    lat_min: float = -6.0,
    lat_max: float = 3.0,
    lv_threshold_kv: float = 69.0,   # kept for compatibility; currently unused
    allowed_trafo_pairs: Optional[set] = None,
    degree_expectations: Optional[List[Tuple[Any, int]]] = None,
    required_bridge_pairs: Optional[set] = None,
    downstream_path: Optional[Tuple[int, int]] = None,
) -> Tuple[Any, List[Tuple[str, Any]]]:
    """
    Evaluate a PyPSA network topology and data sanity checks *as it is*.

    Parameters
    ----------
    n : pypsa.Network
        Network with .buses, .lines, .transformers.
    lon_min/lon_max/lat_min/lat_max : float
        Bounding box for bus coordinates.
    lv_threshold_kv : float
        Kept for interface compatibility (not used directly here).
    allowed_trafo_pairs : set of (kv, kv)
        Allowed (v_nom0, v_nom1) transformer voltage pairs (unordered).
    degree_expectations : list of (bus_id, min_degree)
        Expected minimum degree for specific buses (based on lines only).
    required_bridge_pairs : set of (kv, kv)
        Required voltage-level bridges over transformers (unordered).
    downstream_path : (kv_hi, kv_lo) or None
        Optional check: is there a path in voltage-space graph between
        kv_hi and kv_lo via transformers?

    Returns
    -------
    (n, issues) : (Network, list)
        The original network and a list of (issue_key, payload).
    """

    issues: List[Tuple[str, Any]] = []

    # Defaults
    if allowed_trafo_pairs is None:
        allowed_trafo_pairs = {(500, 230), (230, 138), (230, 69), (138, 69), (69, 48), (138, 48), (230, 48)}
    if required_bridge_pairs is None:
        required_bridge_pairs = {
            (48, 69),
            (48, 138),
            (69, 138),
            (69, 230),
            (138, 230),
            (230, 500),
        }
    if degree_expectations is None:
        degree_expectations = []

    # ---------------------------
    # 1. Prepare bus / line / trafo frames from network
    # ---------------------------

    # Buses
    buses_all = n.buses.copy()
    buses_all = buses_all.assign(Bus=buses_all.index)

    # Normalise lon/lat
    if "lon" not in buses_all.columns and "x" in buses_all.columns:
        buses_all["lon"] = buses_all["x"]
    if "lat" not in buses_all.columns and "y" in buses_all.columns:
        buses_all["lat"] = buses_all["y"]

    # Normalised bus IDs
    buses_all["Bus_norm"] = _to_str_id(buses_all["Bus"])

    # Lines and transformers from network
    line_df = n.lines.copy()
    trafo_df = n.transformers.copy()

    # Add ID columns (for reporting) if missing
    if "Line" not in line_df.columns:
        line_df = line_df.assign(Line=line_df.index)
    if "Transformer" not in trafo_df.columns:
        trafo_df = trafo_df.assign(Transformer=trafo_df.index)

    # Normalised foreign keys for lines/trafo
    for col in ("bus0", "bus1"):
        if col in line_df.columns:
            line_df[col + "_norm"] = _to_str_id(line_df[col])
        if col in trafo_df.columns:
            trafo_df[col + "_norm"] = _to_str_id(trafo_df[col])

    # ---------------------------
    # 2. Core checks
    # ---------------------------

    _check_pk_uniqueness(buses_all, line_df, trafo_df, issues)
    _check_fk_existence(buses_all, line_df, trafo_df, issues)
    _check_bus_coords(buses_all, lon_min, lon_max, lat_min, lat_max, issues)
    _check_voltage_sanity(buses_all, line_df, trafo_df, allowed_trafo_pairs, issues)
    _check_line_ratings(line_df, issues)
    _check_line_geometry(line_df, issues)
    _check_line_length(line_df, issues)
    _check_orphan_buses(buses_all, n, issues)

    # Report voltage levels and transformer pairs directly from the network
    _report_voltage_levels_and_pairs(n, issues)

    topo_report = _build_topology_report(buses_all, line_df)
    if topo_report:
        issues.append(("topology_summary", topo_report))

    _check_degree_expectations(line_df, buses_all, degree_expectations, issues)

    # Optional: check required transformer bridges in voltage space
    _check_transformer_bridges(
        buses_all,
        trafo_df,
        required_bridge_pairs,
        issues,
        downstream_path=downstream_path,
    )

    return n, issues


# -------- Internal helpers -------- #

def _to_str_id(s: pd.Series) -> pd.Series:
    def conv(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer)):
            return str(x)
        if isinstance(x, (float, np.floating)):
            return str(int(x)) if float(x).is_integer() else str(x)
        return str(x).strip()

    return s.map(conv)


def _check_pk_uniqueness(
    buses_all: pd.DataFrame,
    line_df: pd.DataFrame,
    trafo_df: pd.DataFrame,
    issues: List[Tuple[str, Any]],
) -> None:
    # Buses
    if "Bus" in buses_all.columns and buses_all["Bus"].duplicated().any():
        issues.append(
            ("bus_pk_dup", buses_all[buses_all["Bus"].duplicated()]["Bus"].tolist())
        )

    # Lines (use "Line" column we added)
    if "Line" in line_df.columns and line_df["Line"].duplicated().any():
        issues.append(
            ("line_pk_dup", line_df[line_df["Line"].duplicated()]["Line"].tolist())
        )

    # Transformers
    if "Transformer" in trafo_df.columns and trafo_df["Transformer"].duplicated().any():
        issues.append(
            (
                "trafo_pk_dup",
                trafo_df[trafo_df["Transformer"].duplicated()][
                    "Transformer"
                ].tolist(),
            )
        )


def _check_fk_existence(
    buses_all: pd.DataFrame,
    line_df: pd.DataFrame,
    trafo_df: pd.DataFrame,
    issues: List[Tuple[str, Any]],
) -> None:
    bus_set = set(buses_all.get("Bus_norm", pd.Series(dtype=object)).dropna())

    def fk_check(df: pd.DataFrame, kind: str):
        id_col = "Line" if kind == "line" else "Transformer"
        b0, b1 = "bus0_norm", "bus1_norm"
        if id_col not in df.columns:
            issues.append((f"{kind}_missing_id_col", f"Missing '{id_col}'"))
            return
        if b0 not in df.columns or b1 not in df.columns:
            issues.append(
                (f"{kind}_missing_fk_col", f"Missing '{b0}'/'{b1}' in {kind} dataframe")
            )
            return
        miss0 = df.loc[~df[b0].isin(bus_set), [id_col, b0]]
        miss1 = df.loc[~df[b1].isin(bus_set), [id_col, b1]]
        if not miss0.empty:
            issues.append((f"{kind}_missing_bus0", miss0))
        if not miss1.empty:
            issues.append((f"{kind}_missing_bus1", miss1))

    fk_check(line_df, "line")
    fk_check(trafo_df, "trafo")


def _check_bus_coords(
    buses_all: pd.DataFrame,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    issues: List[Tuple[str, Any]],
) -> None:
    if {"lon", "lat"}.issubset(buses_all.columns):
        bad_coords = buses_all[
            (buses_all["lon"] < lon_min)
            | (buses_all["lon"] > lon_max)
            | (buses_all["lat"] < lat_min)
            | (buses_all["lat"] > lat_max)
        ]
        if not bad_coords.empty:
            issues.append(("bad_coords", bad_coords[["Bus", "lon", "lat"]]))


def _check_voltage_sanity(
    buses_all: pd.DataFrame,
    line_df: pd.DataFrame,
    trafo_df: pd.DataFrame,
    allowed_trafo_pairs: set,
    issues: List[Tuple[str, Any]],
) -> None:
    if "v_nom" not in buses_all.columns:
        return

    bmap = buses_all.set_index("Bus")["v_nom"]
    bmap_r = bmap.round(0)

    # Lines: v_nom(line) equals both buses (if line has v_nom)
    if {"bus0", "bus1"}.issubset(line_df.columns) and "v_nom" in line_df.columns:
        v0 = bmap_r.reindex(line_df["bus0"]).values
        v1 = bmap_r.reindex(line_df["bus1"]).values
        lv = line_df["v_nom"].round(0).values
        ln_bad_v = line_df[(v0 != lv) | (v1 != lv)]
        if not ln_bad_v.empty:
            cols = [
                c for c in ["Line", "v_nom", "bus0", "bus1"] if c in ln_bad_v.columns
            ]
            issues.append(("line_vnom_mismatch", ln_bad_v[cols]))

    # Transformers: different voltage levels and in allowed pairs
    if {"bus0", "bus1"}.issubset(trafo_df.columns):
        t_v0 = bmap_r.reindex(trafo_df["bus0"]).values
        t_v1 = bmap_r.reindex(trafo_df["bus1"]).values
        same = np.isfinite(t_v0) & np.isfinite(t_v1) & (t_v0 == t_v1)
        tr_same = trafo_df[same]
        if not tr_same.empty:
            cols = [
                c for c in ["Transformer", "bus0", "bus1"] if c in tr_same.columns
            ]
            issues.append(("trafo_same_voltage", tr_same[cols]))

        pairs = pd.DataFrame({"v0": t_v0, "v1": t_v1}, index=trafo_df.index).dropna()

        def is_allowed(a: float, b: float) -> bool:
            try:
                aa, bb = int(round(a)), int(round(b))
            except Exception:
                return False
            return (aa, bb) in allowed_trafo_pairs or (bb, aa) in allowed_trafo_pairs

        bad_mask = ~pairs.apply(lambda s: is_allowed(s["v0"], s["v1"]), axis=1)
        bad_pairs = trafo_df.loc[pairs.index[bad_mask]]
        if not bad_pairs.empty:
            cols = [
                c for c in ["Transformer", "bus0", "bus1"] if c in bad_pairs.columns
            ]
            issues.append(("trafo_unexpected_voltage_pair", bad_pairs[cols]))


def _check_line_ratings(line_df: pd.DataFrame, issues: List[Tuple[str, Any]]) -> None:
    # These columns might not exist in all networks; only check if they do
    rating_cols = ["num_parallel", "s_nom", "s_max_pu"]
    missing = [c for c in rating_cols if c not in line_df.columns]
    if missing:
        issues.append(("line_missing_rating_col", f"Missing {missing}"))
        return

    ln_bad_rating = line_df[
        (line_df.get("num_parallel", 1) < 1)
        | (line_df.get("s_nom", 0) < 0)
        | (line_df.get("s_max_pu", 1.0) <= 0)
        | (line_df.get("s_max_pu", 1.0) > 1.2)
    ]
    if not ln_bad_rating.empty:
        cols = [
            c
            for c in ["Line", "num_parallel", "s_nom", "s_max_pu"]
            if c in ln_bad_rating.columns
        ]
        issues.append(
            ("line_rating_sanity", ln_bad_rating[cols] if cols else ln_bad_rating)
        )


def _geom_ok(wkt_str: Any) -> bool:
    try:
        g = wkt.loads(wkt_str)
        if g.is_empty:
            return False
        if g.geom_type == "MultiLineString":
            return any(len(seg.coords) >= 2 for seg in g.geoms)
        if g.geom_type == "LineString":
            return len(g.coords) >= 2
        return False
    except Exception:
        return False


def _check_line_geometry(line_df: pd.DataFrame, issues: List[Tuple[str, Any]]) -> None:
    if "geometry" in line_df.columns:
        ln_bad_geom = line_df[~line_df["geometry"].apply(_geom_ok)]
        if not ln_bad_geom.empty:
            cols = [c for c in ["Line", "geometry"] if c in ln_bad_geom.columns]
            issues.append(("line_geom_invalid", ln_bad_geom[cols].head(20)))


def _check_line_length(line_df: pd.DataFrame, issues: List[Tuple[str, Any]]) -> None:
    if "length" not in line_df.columns:
        return
    ln_bad_len = line_df[(line_df["length"] <= 0) | (line_df["length"] > 600_000)]
    if ln_bad_len.empty and line_df["length"].max() < 2000:
        ln_bad_len = line_df[(line_df["length"] <= 0) | (line_df["length"] > 600)]
    if not ln_bad_len.empty:
        cols = [c for c in ["Line", "length"] if c in ln_bad_len.columns]
        issues.append(("line_length_bounds", ln_bad_len[cols] if cols else ln_bad_len))


def _check_orphan_buses(
    buses_all: pd.DataFrame, n, issues: List[Tuple[str, Any]]
) -> None:
    line_df = n.lines.copy()
    trafo_df = n.transformers.copy()

    for col in ("bus0", "bus1"):
        if col in line_df.columns:
            line_df[col + "_norm"] = _to_str_id(line_df[col])
        if col in trafo_df.columns:
            trafo_df[col + "_norm"] = _to_str_id(trafo_df[col])

    deg = defaultdict(int)
    for b in buses_all.get("Bus_norm", pd.Series(dtype=object)):
        deg[b] = 0

    if {"bus0_norm", "bus1_norm"}.issubset(line_df.columns):
        for b0, b1 in line_df[["bus0_norm", "bus1_norm"]].itertuples(
            index=False, name=None
        ):
            if pd.notna(b0):
                deg[b0] += 1
            if pd.notna(b1):
                deg[b1] += 1

    if {"bus0_norm", "bus1_norm"}.issubset(trafo_df.columns):
        for b0, b1 in trafo_df[["bus0_norm", "bus1_norm"]].itertuples(
            index=False, name=None
        ):
            if pd.notna(b0):
                deg[b0] += 1
            if pd.notna(b1):
                deg[b1] += 1

    orphans_norm = [b for b, d in deg.items() if d == 0]
    if orphans_norm:
        orphan_rows = buses_all[
            buses_all.get("Bus_norm", pd.Series(dtype=object)).isin(orphans_norm)
        ]
        cols_pref = ["Bus", "v_nom", "lon", "lat"]
        cols = [c for c in cols_pref if c in orphan_rows.columns]
        issues.append(
            (
                "orphan_buses",
                orphan_rows[cols] if cols else orphan_rows,
            )
        )


def _build_topology_report(
    buses_all: pd.DataFrame, line_df: pd.DataFrame
) -> Dict[str, Any]:
    def build_graph(df_lines: pd.DataFrame, v_target: float) -> nx.Graph:
        if not {"bus0", "bus1", "v_nom"}.issubset(df_lines.columns):
            return nx.Graph()
        dfv = df_lines[np.isclose(df_lines["v_nom"].round(0), v_target)]
        G = nx.Graph()
        if {"v_nom"}.issubset(buses_all.columns):
            nodes = buses_all.loc[
                np.isclose(buses_all["v_nom"].round(0), v_target), "Bus"
            ].tolist()
            G.add_nodes_from(nodes)
        G.add_edges_from(dfv[["bus0", "bus1"]].itertuples(index=False, name=None))
        return G

    topo_report: Dict[str, Any] = {}
    voltages_to_check: List[float] = []
    if "v_nom" in line_df.columns:
        voltages_to_check = sorted(pd.unique(line_df["v_nom"].round(0).dropna()))
    for v in voltages_to_check:
        Gv = build_graph(line_df, v)
        comps = list(nx.connected_components(Gv))
        topo_report[f"{int(v)}kV"] = {
            "n_components": len(comps),
            "sizes": sorted([len(c) for c in comps], reverse=True)[:5],
        }
    return topo_report


def _check_degree_expectations(
    line_df: pd.DataFrame,
    buses_all: pd.DataFrame,
    degree_expectations: List[Tuple[Any, int]],
    issues: List[Tuple[str, Any]],
) -> None:
    bus_set = set(buses_all.get("Bus", pd.Series(dtype=object)))

    def degree_at(bus_id: Any, df_lines: pd.DataFrame) -> int:
        if not {"bus0", "bus1"}.issubset(df_lines.columns):
            return 0
        return int((df_lines["bus0"].eq(bus_id) | df_lines["bus1"].eq(bus_id)).sum())

    deg_expect_violations = []
    for bus_id, min_deg in degree_expectations:
        if bus_id in bus_set:
            d = degree_at(bus_id, line_df)
            if d < min_deg:
                deg_expect_violations.append((bus_id, d, min_deg))
    if deg_expect_violations:
        issues.append(
            (
                "degree_expectations",
                pd.DataFrame(
                    deg_expect_violations, columns=["bus", "degree", "min_expected"]
                ),
            )
        )


def _check_transformer_bridges(
    buses_all: pd.DataFrame,
    trafo_df: pd.DataFrame,
    required_bridge_pairs: set,
    issues: List[Tuple[str, Any]],
    *,
    downstream_path: Optional[Tuple[int, int]] = None,
) -> None:
    if "Bus_norm" not in buses_all.columns or "v_nom" not in buses_all.columns:
        issues.append(
            ("missing_bus_norm_or_vnom",
             "buses_all must contain Bus_norm and v_nom")
        )
        return

    if not {"bus0_norm", "bus1_norm"}.issubset(trafo_df.columns):
        # No normalised FK → we cannot analyse voltage bridges
        return

    bus_v = (
        buses_all.dropna(subset=["Bus_norm"])
        .drop_duplicates("Bus_norm", keep="last")
        .set_index("Bus_norm")["v_nom"]
        .round(0)
    )

    v0_s = trafo_df["bus0_norm"].map(bus_v)
    v1_s = trafo_df["bus1_norm"].map(bus_v)
    mask_diff = (
        (~v0_s.isna()).to_numpy()
        & (~v1_s.isna()).to_numpy()
        & (v0_s.to_numpy() != v1_s.to_numpy())
    )

    bridge_pairs_found: set = set()
    Gv = nx.Graph()

    if mask_diff.any():
        pairs_arr = np.column_stack(
            (v0_s.to_numpy()[mask_diff], v1_s.to_numpy()[mask_diff])
        )
        pairs_arr = np.round(pairs_arr, 0).astype(int, copy=False)
        for a, b in pairs_arr:
            aa, bb = int(a), int(b)
            bridge_pairs_found.add(tuple(sorted((aa, bb))))
            Gv.add_edge(aa, bb)

    # Report all unique bridge pairs found
    if bridge_pairs_found:
        issues.append(("trafo_pairs_found", sorted(bridge_pairs_found)))

    # Required bridge pairs present?
    missing = [p for p in required_bridge_pairs if p not in bridge_pairs_found]
    if missing:
        issues.append(
            (
                "missing_transformer_bridges",
                {
                    "required": sorted(required_bridge_pairs),
                    "found": sorted(bridge_pairs_found),
                    "missing": sorted(missing),
                },
            )
        )

    # Optional: check downstream connectivity between two levels (e.g., 500 -> 48)
    if downstream_path is not None and len(Gv) > 0:
        v_hi, v_lo = int(downstream_path[0]), int(downstream_path[1])
        if not (Gv.has_node(v_hi) and Gv.has_node(v_lo)):
            issues.append(
                (
                    "missing_downstream_path",
                    {
                        "from": v_hi,
                        "to": v_lo,
                        "reason": "one or both voltage levels absent in transformer graph",
                    },
                )
            )
        elif not nx.has_path(Gv, v_hi, v_lo):
            issues.append(
                (
                    "missing_downstream_path",
                    {
                        "from": v_hi,
                        "to": v_lo,
                        "reason": "no path in voltage-level transformer graph",
                    },
                )
            )


def _report_voltage_levels_and_pairs(n, issues: List[Tuple[str, Any]]) -> None:
    """
    Append overall bus voltage levels and unique transformer voltage pairs from the network.

    - bus voltage levels are reported as sorted unique integer kV (rounded from v_nom)
    - transformer voltage pairs are unordered unique (hi, lo) integer kV pairs
    """
    if "v_nom" in n.buses.columns:
        v_series = n.buses["v_nom"].dropna()
        try:
            levels = sorted(pd.unique(v_series.round(0).astype(int)))
        except Exception:
            levels = sorted(pd.unique(v_series))
        issues.append(("bus_voltage_levels", levels))

    if len(n.transformers) > 0 and {"bus0", "bus1"}.issubset(n.transformers.columns):
        pairs = set()
        buses = n.buses
        for _, row in n.transformers.iterrows():
            b0 = row.get("bus0")
            b1 = row.get("bus1")
            if b0 in buses.index and b1 in buses.index:
                v0 = buses.at[b0, "v_nom"] if "v_nom" in buses.columns else None
                v1 = buses.at[b1, "v_nom"] if "v_nom" in buses.columns else None
                if pd.notna(v0) and pd.notna(v1) and v0 != v1:
                    try:
                        a, b = int(round(float(v0))), int(round(float(v1)))
                        pairs.add(tuple(sorted((a, b))))
                    except Exception:
                        continue
        if pairs:
            issues.append(("trafo_pairs_found_network", sorted(pairs)))
