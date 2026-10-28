"""Publication-quality maps for the Ecuador transmission topology."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
import numpy as np


VOLTAGE_COLORS = {
    138.0: "#377eb8",  # blue, retained from the owner's topology plots
    230.0: "#a65628",  # brown
    500.0: "#00008B",  # dark blue
}

VOLTAGE_WIDTHS = {138.0: 1.5, 230.0: 2.3, 500.0: 3.4}
VOLTAGE_BUS_SIZES = {138.0: 32.0, 230.0: 46.0, 500.0: 64.0}
ECUADOR_MAINLAND_EXTENT = (-81.1, -75.1, -4.8, 1.7)


def load_mainland_provinces(path: Path | str) -> gpd.GeoDataFrame:
    """Load GADM level-one boundaries and exclude Galapagos."""
    provinces = gpd.read_file(path, layer="ADM_ADM_1")
    if provinces.crs is None:
        raise ValueError("GADM province layer has no coordinate reference system")
    provinces = provinces.to_crs(4326)
    name_candidates = [
        column
        for column in provinces.columns
        if column.upper().endswith("NAME_1")
        or column.lower() in {"name_1", "name1", "provincia", "province"}
    ]
    if not name_candidates:
        raise ValueError("Could not identify the province-name column in GADM")
    names = provinces[name_candidates[0]].astype(str)
    return provinces.loc[~names.str.contains("Galápagos|Galapagos", case=False, regex=True)].copy()


def _coordinates(network, bus: str) -> tuple[float, float]:
    row = network.buses.loc[bus]
    if {"x", "y"}.issubset(network.buses.columns):
        return float(row.x), float(row.y)
    return float(row.lon), float(row.lat)


def _segments(network, table) -> list[list[tuple[float, float]]]:
    return [
        [_coordinates(network, str(row.bus0)), _coordinates(network, str(row.bus1))]
        for row in table.itertuples()
    ]


def plot_topology_voltage_map(
    network,
    provinces: gpd.GeoDataFrame,
    *,
    voltage_kv: float | None = None,
    annotate_bus_ids: bool = False,
    title: str | None = None,
    save_path: Path | str | None = None,
    dpi: int = 300,
):
    """Plot the combined topology or a single nominal-voltage layer.

    Lines and buses use the fixed palette from the owner's legacy topology
    notebook. Transformers are neutral dashed connectors because they join two
    voltage levels and therefore do not belong to one line-voltage class.
    """
    from matplotlib.collections import LineCollection

    available = sorted(set(network.buses.v_nom.astype(float)))
    unsupported = set(available) - set(VOLTAGE_COLORS)
    if unsupported:
        raise ValueError(f"No approved plot color for voltage levels: {sorted(unsupported)}")
    if voltage_kv is not None and float(voltage_kv) not in VOLTAGE_COLORS:
        raise ValueError(f"Unsupported requested voltage: {voltage_kv}")

    selected_voltages: Iterable[float] = (
        [float(voltage_kv)] if voltage_kv is not None else available
    )
    figure_size = (13.5, 13.5) if voltage_kv is not None else (10.5, 10.5)
    fig, ax = plt.subplots(figsize=figure_size)
    provinces.plot(ax=ax, color="#f7f4ed", edgecolor="#8c8c8c", linewidth=0.55, zorder=0)
    provinces.boundary.plot(ax=ax, color="#676767", linewidth=0.45, zorder=1)

    # Transformers remain visible as the cross-voltage structure.
    transformers = network.transformers
    if voltage_kv is not None:
        selected_buses = set(
            network.buses.index[np.isclose(network.buses.v_nom, float(voltage_kv))]
        )
        transformers = transformers[
            transformers.bus0.isin(selected_buses) | transformers.bus1.isin(selected_buses)
        ]
    if not transformers.empty:
        ax.add_collection(
            LineCollection(
                _segments(network, transformers),
                colors="#777777",
                linewidths=1.0,
                linestyles="dashed",
                alpha=0.7,
                zorder=2,
            )
        )

    present = []
    for voltage in selected_voltages:
        buses = network.buses[np.isclose(network.buses.v_nom, voltage)]
        lines = network.lines[np.isclose(network.lines.v_nom, voltage)]
        if buses.empty:
            continue
        present.append(voltage)
        if not lines.empty:
            ax.add_collection(
                LineCollection(
                    _segments(network, lines),
                    colors=VOLTAGE_COLORS[voltage],
                    linewidths=VOLTAGE_WIDTHS[voltage],
                    alpha=0.9,
                    zorder=3,
                )
            )
        x_column, y_column = (("x", "y") if {"x", "y"}.issubset(buses.columns) else ("lon", "lat"))
        ax.scatter(
            buses[x_column],
            buses[y_column],
            s=VOLTAGE_BUS_SIZES[voltage],
            c=VOLTAGE_COLORS[voltage],
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        if annotate_bus_ids:
            for bus_id, bus in buses.iterrows():
                ax.annotate(
                    str(bus_id),
                    (float(bus[x_column]), float(bus[y_column])),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.5,
                    color="#202020",
                    zorder=5,
                    path_effects=[path_effects.withStroke(linewidth=2.0, foreground="white")],
                )

    legend = [
        Line2D(
            [0], [0],
            color=VOLTAGE_COLORS[voltage],
            linewidth=VOLTAGE_WIDTHS[voltage],
            marker="o",
            markerfacecolor=VOLTAGE_COLORS[voltage],
            markeredgecolor="white",
            markersize=7,
            label=f"{int(voltage)} kV",
        )
        for voltage in present
    ]
    if not transformers.empty:
        legend.append(Line2D([0], [0], color="#777777", linestyle="--", label="Transformer"))
    legend.append(Patch(facecolor="#f7f4ed", edgecolor="#8c8c8c", label="Province boundary"))
    ax.legend(handles=legend, title="Topology", loc="lower right", frameon=True, framealpha=0.96)

    minx, maxx, miny, maxy = ECUADOR_MAINLAND_EXTENT
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="#d7d7d7", linewidth=0.4, linestyle=":", alpha=0.7)
    if title is None:
        title = (
            f"Ecuador SNI candidate topology — {int(voltage_kv)} kV"
            if voltage_kv is not None
            else "Ecuador SNI candidate topology by nominal voltage"
        )
    ax.set_title(title, fontweight="bold", fontsize=14, pad=12)
    fig.tight_layout()
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, ax


def save_phase05_topology_maps(network, boundary_path: Path | str, figure_dir: Path | str):
    """Save the combined map and separate 138/230/500 kV evidence maps."""
    provinces = load_mainland_provinces(boundary_path)
    output_dir = Path(figure_dir)
    figures = {}
    figures["combined"] = plot_topology_voltage_map(
        network,
        provinces,
        title="Ecuador SNI candidate topology by nominal voltage",
        save_path=output_dir / "candidate_topology_by_voltage_2022.png",
    )[0]
    for voltage in sorted(VOLTAGE_COLORS, reverse=True):
        figures[str(int(voltage))] = plot_topology_voltage_map(
            network,
            provinces,
            voltage_kv=voltage,
            annotate_bus_ids=True,
            title=f"Ecuador SNI candidate topology — {int(voltage)} kV",
            save_path=output_dir / f"candidate_topology_{int(voltage)}kv_2022.png",
        )[0]
    return figures
