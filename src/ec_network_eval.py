import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import networkx as nx
from shapely import wkt


# Public API
def evaluate_network(
    n_exp,
    *,
    data_dir: Optional[str] = None,
    bus_csv: Optional[str] = None,
    line_csv: Optional[str] = None,
    trafo_csv: Optional[str] = None,
    lon_min: float = -92.0,
    lon_max: float = -74.0,
    lat_min: float = -6.0,
    lat_max: float = 3.0,
    lv_threshold_kv: float = 69.0,
    allowed_trafo_pairs: Optional[set] = None,
    degree_expectations: Optional[List[Tuple[Any, int]]] = None,
    required_bridge_pairs: Optional[set] = None,
    downstream_path: Optional[Tuple[int, int]] = None,
) -> Tuple[Any, List[Tuple[str, Any]]]:
    """
    Evaluate Ecuador network expansion data and topology sanity checks.

    Parameters
    - n_exp: PyPSA network (or compatible) with .buses, .lines, .transformers
    - data_dir: Base directory containing the expansion CSVs (used if explicit CSV paths not provided)
    - bus_csv, line_csv, trafo_csv: Optional explicit filepaths to expansion CSVs
    - lon_min/lon_max/lat_min/lat_max: Bounding box for bus coordinates
    - lv_threshold_kv: Threshold for LV exclusion in orphan checks (kept for compatibility; currently unused)
    - allowed_trafo_pairs: Set of allowed (kv, kv) transformer voltage pairs
    - degree_expectations: List of (bus_id, min_degree) pairs to check
    - required_bridge_pairs: Set of required voltage bridge pairs for transformers

    Returns
    - (n_exp, issues): original network object and list of (issue_key, payload)
    """

    issues: List[Tuple[str, Any]] = []

    # Defaults
    if allowed_trafo_pairs is None:
        allowed_trafo_pairs = {(500, 230), (230, 138), (230, 69), (138, 69)}
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

    # Resolve file paths
    if not bus_csv or not line_csv or not trafo_csv:
        if not data_dir:
            issues.append(
                ("missing_data_paths", "Provide data_dir or explicit CSV paths")
            )
            return n_exp, issues
        bus_csv = bus_csv or os.path.join(data_dir, "EC_buses_expansion.csv")
        line_csv = line_csv or os.path.join(data_dir, "EC_lines_expansion.csv")
        trafo_csv = trafo_csv or os.path.join(data_dir, "EC_trafo_expansion.csv")

    # Load expansion CSVs
    bus_exp = _load_csv_utf8(bus_csv, issues)
    line_exp = _load_csv_utf8(line_csv, issues)
    trafo_exp = _load_csv_utf8(trafo_csv, issues)

    # Build unified buses from network + expansion
    buses_all = _unified_buses(n_exp, bus_exp, issues)

    # Normalize identifiers for robust PK/FK checks
    buses_all["Bus_norm"] = (
        _to_str_id(buses_all["Bus"])
        if "Bus" in buses_all.columns
        else pd.Series(dtype=object)
    )
    line_exp = line_exp.copy()
    trafo_exp = trafo_exp.copy()
    for col in ("bus0", "bus1"):
        if col in line_exp.columns:
            line_exp[col + "_norm"] = _to_str_id(line_exp[col])
        if col in trafo_exp.columns:
            trafo_exp[col + "_norm"] = _to_str_id(trafo_exp[col])

    # Core checks
    _check_pk_uniqueness(buses_all, line_exp, trafo_exp, issues)
    _check_fk_existence(buses_all, line_exp, trafo_exp, issues)
    _check_bus_coords(buses_all, lon_min, lon_max, lat_min, lat_max, issues)
    _check_voltage_sanity(buses_all, line_exp, trafo_exp, allowed_trafo_pairs, issues)
    _check_line_ratings(line_exp, issues)
    _check_line_geometry(line_exp, issues)
    _check_line_length(line_exp, issues)
    _check_orphan_buses(buses_all, n_exp, issues)

    # Report voltage levels and transformer pairs directly from the network
    _report_voltage_levels_and_pairs(n_exp, issues)

    topo_report = _build_topology_report(buses_all, line_exp)
    if topo_report:
        issues.append(("topology_summary", topo_report))

    _check_degree_expectations(line_exp, buses_all, degree_expectations, issues)
    # _check_transformer_bridges(
    #     buses_all,
    #     trafo_exp,
    #     required_bridge_pairs,
    #     issues,
    #     downstream_path=downstream_path,
    # )

    return n_exp, issues


# ---------------- Internal helpers ----------------


def _load_csv_utf8(path: str, issues: List[Tuple[str, Any]]=None) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError as e:
        issues.append(("utf8_decode_error", f"{os.path.basename(path)}: {e}"))
        raise
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        if issues:
            issues.append(
                ("all_empty_columns", {"file": os.path.basename(path), "cols": empty_cols})
            )
    return df


def _unified_buses(
    n_exp, bus_exp: pd.DataFrame, issues: List[Tuple[str, Any]]
) -> pd.DataFrame:
    bus_net = n_exp.buses.copy()
    bus_net = bus_net.assign(Bus=bus_net.index)
    if "lon" not in bus_net and "x" in bus_net:
        bus_net["lon"] = bus_net["x"]
    if "lat" not in bus_net and "y" in bus_net:
        bus_net["lat"] = bus_net["y"]

    if "Bus" not in bus_exp.columns:
        issues.append(("bus_pk_missing", "Expansion buses CSV missing 'Bus' column"))

    common_bus_cols = bus_net.columns.intersection(bus_exp.columns).tolist()
    if "Bus" not in common_bus_cols:
        common_bus_cols.append("Bus")
    bus_exp_norm = bus_exp.loc[
        :, [c for c in common_bus_cols if c in bus_exp.columns]
    ].copy()

    buses_all = pd.concat(
        [bus_net.loc[:, common_bus_cols], bus_exp_norm],
        ignore_index=True,
    ).drop_duplicates("Bus", keep="last")
    return buses_all


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
    line_exp: pd.DataFrame,
    trafo_exp: pd.DataFrame,
    issues: List[Tuple[str, Any]],
) -> None:
    if "Bus" in buses_all.columns and buses_all["Bus"].duplicated().any():
        issues.append(
            ("bus_pk_dup", buses_all[buses_all["Bus"].duplicated()]["Bus"].tolist())
        )

    if "Line" in line_exp.columns and line_exp["Line"].duplicated().any():
        issues.append(
            ("line_pk_dup", line_exp[line_exp["Line"].duplicated()]["Line"].tolist())
        )

    if (
        "Transformer" in trafo_exp.columns
        and trafo_exp["Transformer"].duplicated().any()
    ):
        issues.append(
            (
                "trafo_pk_dup",
                trafo_exp[trafo_exp["Transformer"].duplicated()][
                    "Transformer"
                ].tolist(),
            )
        )


def _check_fk_existence(
    buses_all: pd.DataFrame,
    line_exp: pd.DataFrame,
    trafo_exp: pd.DataFrame,
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
                (f"{kind}_missing_fk_col", f"Missing '{b0}'/'{b1}'; normalize earlier")
            )
            return
        miss0 = df.loc[~df[b0].isin(bus_set), [id_col, b0]]
        miss1 = df.loc[~df[b1].isin(bus_set), [id_col, b1]]
        if not miss0.empty:
            issues.append((f"{kind}_missing_bus0", miss0))
        if not miss1.empty:
            issues.append((f"{kind}_missing_bus1", miss1))

    fk_check(line_exp, "line")
    fk_check(trafo_exp, "trafo")


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
    line_exp: pd.DataFrame,
    trafo_exp: pd.DataFrame,
    allowed_trafo_pairs: set,
    issues: List[Tuple[str, Any]],
) -> None:
    if "v_nom" not in buses_all.columns:
        return

    bmap = buses_all.set_index("Bus")["v_nom"]
    bmap_r = bmap.round(0)

    # Lines: v_nom(line) equals both buses
    if {"bus0", "bus1", "v_nom"}.issubset(line_exp.columns):
        v0 = bmap_r.reindex(line_exp["bus0"]).values
        v1 = bmap_r.reindex(line_exp["bus1"]).values
        lv = line_exp["v_nom"].round(0).values
        ln_bad_v = line_exp[(v0 != lv) | (v1 != lv)]
        if not ln_bad_v.empty:
            cols = [
                c for c in ["Line", "v_nom", "bus0", "bus1"] if c in ln_bad_v.columns
            ]
            issues.append(("line_vnom_mismatch", ln_bad_v[cols]))

    # Transformers: different voltage levels and in allowed pairs
    if {"bus0", "bus1"}.issubset(trafo_exp.columns):
        t_v0 = bmap_r.reindex(trafo_exp["bus0"]).values
        t_v1 = bmap_r.reindex(trafo_exp["bus1"]).values
        same = np.isfinite(t_v0) & np.isfinite(t_v1) & (t_v0 == t_v1)
        tr_same = trafo_exp[same]
        if not tr_same.empty:
            cols = [c for c in ["Transformer", "bus0", "bus1"] if c in tr_same.columns]
            issues.append(("trafo_same_voltage", tr_same[cols]))

        pairs = pd.DataFrame({"v0": t_v0, "v1": t_v1}, index=trafo_exp.index).dropna()

        def is_allowed(a: float, b: float) -> bool:
            try:
                aa, bb = int(round(a)), int(round(b))
            except Exception:
                return False
            return (aa, bb) in allowed_trafo_pairs or (bb, aa) in allowed_trafo_pairs

        bad_mask = ~pairs.apply(lambda s: is_allowed(s["v0"], s["v1"]), axis=1)
        bad_pairs = trafo_exp.loc[pairs.index[bad_mask]]
        if not bad_pairs.empty:
            cols = [
                c for c in ["Transformer", "bus0", "bus1"] if c in bad_pairs.columns
            ]
            issues.append(("trafo_unexpected_voltage_pair", bad_pairs[cols]))


def _check_line_ratings(line_exp: pd.DataFrame, issues: List[Tuple[str, Any]]) -> None:
    for req in ["num_parallel", "s_nom", "s_max_pu"]:
        if req not in line_exp.columns:
            issues.append(("line_missing_rating_col", f"Missing '{req}'"))
    ln_bad_rating = line_exp[
        (line_exp.get("num_parallel", 1) < 1)
        | (line_exp.get("s_nom", 0) < 0)
        | (line_exp.get("s_max_pu", 1.0) <= 0)
        | (line_exp.get("s_max_pu", 1.0) > 1.2)
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


def _check_line_geometry(line_exp: pd.DataFrame, issues: List[Tuple[str, Any]]) -> None:
    if "geometry" in line_exp.columns:
        ln_bad_geom = line_exp[~line_exp["geometry"].apply(_geom_ok)]
        if not ln_bad_geom.empty:
            cols = [c for c in ["Line", "geometry"] if c in ln_bad_geom.columns]
            issues.append(("line_geom_invalid", ln_bad_geom[cols].head(20)))


def _check_line_length(line_exp: pd.DataFrame, issues: List[Tuple[str, Any]]) -> None:
    if "length" not in line_exp.columns:
        return
    ln_bad_len = line_exp[(line_exp["length"] <= 0) | (line_exp["length"] > 600_000)]
    if ln_bad_len.empty and line_exp["length"].max() < 2000:
        ln_bad_len = line_exp[(line_exp["length"] <= 0) | (line_exp["length"] > 600)]
    if not ln_bad_len.empty:
        cols = [c for c in ["Line", "length"] if c in ln_bad_len.columns]
        issues.append(("line_length_bounds", ln_bad_len[cols] if cols else ln_bad_len))


def _check_orphan_buses(
    buses_all: pd.DataFrame, n_exp, issues: List[Tuple[str, Any]]
) -> None:
    from collections import defaultdict

    line_df = n_exp.lines.copy()
    trafo_df = n_exp.transformers.copy()

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
    buses_all: pd.DataFrame, line_exp: pd.DataFrame
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
    if "v_nom" in line_exp.columns:
        voltages_to_check = sorted(pd.unique(line_exp["v_nom"].round(0).dropna()))
    for v in voltages_to_check:
        Gv = build_graph(line_exp, v)
        comps = list(nx.connected_components(Gv))
        topo_report[f"{int(v)}kV"] = {
            "n_components": len(comps),
            "sizes": sorted([len(c) for c in comps], reverse=True)[:5],
        }
    return topo_report


def _check_degree_expectations(
    line_exp: pd.DataFrame,
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
            d = degree_at(bus_id, line_exp)
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
    trafo_exp: pd.DataFrame,
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

    bus_v = (
        buses_all.dropna(subset=["Bus_norm"])
        .drop_duplicates("Bus_norm", keep="last")
        .set_index("Bus_norm")["v_nom"]
        .round(0)
    )

    if {"bus0_norm", "bus1_norm"}.issubset(trafo_exp.columns):
        v0_s = trafo_exp["bus0_norm"].map(bus_v)
        v1_s = trafo_exp["bus1_norm"].map(bus_v)
        mask_diff = (
            (~v0_s.isna()).to_numpy()
            & (~v1_s.isna()).to_numpy()
            & (v0_s.to_numpy() != v1_s.to_numpy())
        )
        # Build bridge set and an undirected graph of voltage levels
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
                Gv.add_edge(aa, bb)  # undirected: pair order does not matter
        # Report all unique bridge pairs found (unordered pairs)
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
        if downstream_path is not None:
            v_hi, v_lo = int(downstream_path[0]), int(downstream_path[1])
            if len(Gv) == 0 or not (Gv.has_node(v_hi) and Gv.has_node(v_lo)):
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


def _report_voltage_levels_and_pairs(n_exp, issues: List[Tuple[str, Any]]) -> None:
    """Append overall bus voltage levels and unique transformer voltage pairs from the network.

    - bus voltage levels are reported as sorted unique integer kV (rounded from v_nom)
    - transformer voltage pairs are unordered unique (hi, lo) integer kV pairs
    """
    if "v_nom" in n_exp.buses.columns:
        v_series = n_exp.buses["v_nom"].dropna()
        try:
            levels = sorted(pd.unique(v_series.round(0).astype(int)))
        except Exception:
            levels = sorted(pd.unique(v_series))
        issues.append(("bus_voltage_levels", levels))

    # Compute unique unordered pairs from transformers using bus voltages
    if len(n_exp.transformers) > 0 and {"bus0", "bus1"}.issubset(n_exp.transformers.columns):
        pairs = set()
        buses = n_exp.buses
        for _, row in n_exp.transformers.iterrows():
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
    else:
        issues.append(
            ("trafo_norm_fk_missing", "trafo_exp must contain bus0_norm and bus1_norm")
        )
