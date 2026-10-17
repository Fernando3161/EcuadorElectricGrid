"""Assembly of the unsolved Ecuador SNI 2022 candidate network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import hashlib

import numpy as np
import pandas as pd

from .availability import expected_hourly_index
from .topology import assert_topology_invariants, haversine_km, structural_fingerprint


# Phase 5 materiality screens. These are review rules for the reduced-voltage
# model, not physical interconnection standards.
GENERATOR_ATTACHMENT_BANDS = (
    (1.0, "<= 1 MW", 100.0),
    (10.0, "> 1-10 MW", 75.0),
    (50.0, "> 10-50 MW", 50.0),
    (100.0, "> 50-100 MW", 35.0),
    (float("inf"), "> 100 MW", 25.0),
)
LOAD_ATTACHMENT_BANDS = (
    (25.0, "<= 25 GWh/year", 100.0),
    (100.0, "> 25-100 GWh/year", 75.0),
    (250.0, "> 100-250 GWh/year", 50.0),
    (500.0, "> 250-500 GWh/year", 35.0),
    (float("inf"), "> 500 GWh/year", 25.0),
)


@dataclass(frozen=True)
class AssemblyResult:
    network: object
    load_attachments: pd.DataFrame
    generator_attachments: pd.DataFrame
    fingerprint: dict[str, object]


def read_hourly_table(path: Path | str, year: int = 2022) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "snapshot"
    expected = expected_hourly_index(year)
    if not frame.index.equals(expected):
        raise ValueError(f"{path} does not contain the exact {year} hourly index")
    if frame.columns.duplicated().any() or frame.isna().any().any():
        raise ValueError(f"{path} contains duplicate columns or missing values")
    return frame.astype(float)


def _nearest_bus(
    buses: pd.DataFrame, longitude: float, latitude: float
) -> tuple[str, float]:
    distances = buses.apply(
        lambda row: haversine_km(latitude, longitude, float(row.y), float(row.x)), axis=1
    )
    bus = str(distances.idxmin())
    return bus, float(distances.loc[bus])


def build_load_attachment_table(
    topology,
    source_buses: pd.DataFrame,
    demand_columns,
    connection_evidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map every source load to a retained bus and expose every decision."""
    evidence = pd.DataFrame() if connection_evidence is None else connection_evidence.copy()
    if not evidence.empty:
        required = {
            "load_id", "source_bus", "target_bus", "target_voltage_kv",
            "attachment_method", "connection_basis", "official_fact",
            "modeling_assumption", "confidence", "source_document",
            "source_locator", "source_url", "rationale", "review_status",
        }
        missing = required.difference(evidence.columns)
        if missing:
            raise ValueError(f"Load connection evidence is missing columns: {sorted(missing)}")
        if evidence.load_id.duplicated().any():
            raise ValueError("Load connection evidence must have one row per load")
        evidence = evidence.set_index("load_id")
    rows = []
    retained = topology.buses
    for source in map(str, demand_columns):
        if source not in source_buses.index:
            raise ValueError(f"Demand source bus {source} is absent from the upstream network")
        coordinates = source_buses.loc[source, ["x", "y"]].astype(float)
        load_id = f"load_{source}"
        if load_id in evidence.index:
            connection = evidence.loc[load_id]
            if str(connection.source_bus) != source:
                raise ValueError(f"Load evidence source bus does not match {load_id}")
            target = str(connection.target_bus)
            if target not in retained.index.astype(str):
                raise ValueError(f"Load connection evidence for {load_id} references absent bus {target}")
            actual_voltage = float(retained.loc[target, "v_nom"])
            if not np.isclose(actual_voltage, float(connection.target_voltage_kv)):
                raise ValueError(f"Load connection evidence voltage for {load_id} does not match bus {target}")
            distance = haversine_km(
                coordinates.y, coordinates.x,
                float(retained.loc[target, "y"]), float(retained.loc[target, "x"]),
            )
            method = str(connection.attachment_method)
            rationale = str(connection.rationale)
            basis = str(connection.connection_basis)
            source_document = str(connection.source_document)
            source_locator = str(connection.source_locator)
            source_url = str(connection.source_url)
            official_fact = bool(connection.official_fact)
            modeling_assumption = bool(connection.modeling_assumption)
            confidence = str(connection.confidence)
            review_status = str(connection.review_status)
        elif source in retained.index:
            target, distance, method = source, 0.0, "exact_retained_bus"
            rationale = "demand location is already a retained transmission bus"
            basis = "exact retained bus"
            source_document = pd.NA
            source_locator = pd.NA
            source_url = pd.NA
            official_fact = False
            modeling_assumption = False
            confidence = "high"
            review_status = "acceptable_exact_retained_bus"
        else:
            target, distance = _nearest_bus(retained, coordinates.x, coordinates.y)
            method = "nearest_retained_transmission_bus"
            rationale = "preserve Stage 1 demand while aggregating distribution-level locations to the supported transmission boundary"
            basis = "geographic proximity"
            source_document = pd.NA
            source_locator = pd.NA
            source_url = pd.NA
            official_fact = False
            modeling_assumption = True
            confidence = "low"
            review_status = "unreviewed_geographic_fallback"
        rows.append(
            {
                "load_id": load_id,
                "source_bus": source,
                "source_longitude": coordinates.x,
                "source_latitude": coordinates.y,
                "target_bus": target,
                "distance_km": distance,
                "attachment_method": method,
                "connection_basis": basis,
                "connection_source": source_document,
                "connection_locator": source_locator,
                "connection_source_url": source_url,
                "official_fact": official_fact,
                "modeling_assumption": modeling_assumption,
                "confidence": confidence,
                "connection_review_status": review_status,
                "rationale": rationale,
            }
        )
    result = pd.DataFrame(rows)
    if result.load_id.duplicated().any() or len(result) != len(demand_columns):
        raise AssertionError("Load attachment must be one-to-one with demand columns")
    return result


def build_generator_attachment_table(
    topology, generators: pd.DataFrame, connection_evidence: pd.DataFrame | None = None
) -> pd.DataFrame:
    evidence = pd.DataFrame() if connection_evidence is None else connection_evidence.copy()
    if not evidence.empty:
        required = {
            "plant_id", "connection_substation", "connection_voltage_kv", "target_bus",
            "official_fact", "confidence", "source_document", "source_locator", "source_url",
        }
        missing = required.difference(evidence.columns)
        if missing:
            raise ValueError(f"Generator connection evidence is missing columns: {sorted(missing)}")
        if evidence.plant_id.duplicated().any():
            raise ValueError("Generator connection evidence must have one row per plant")
        evidence = evidence.set_index("plant_id")
    rows = []
    for plant in generators.itertuples(index=False):
        nearest_target, distance = _nearest_bus(
            topology.buses, float(plant.longitude), float(plant.latitude)
        )
        if plant.plant_id in evidence.index:
            connection = evidence.loc[plant.plant_id]
            target = str(connection.target_bus)
            if target not in topology.buses.index.astype(str):
                raise ValueError(f"Connection evidence for {plant.plant_id} references absent bus {target}")
            actual_voltage = float(topology.buses.loc[target, "v_nom"])
            if not np.isclose(actual_voltage, float(connection.connection_voltage_kv)):
                raise ValueError(
                    f"Connection evidence voltage for {plant.plant_id} does not match bus {target}"
                )
            distance = haversine_km(
                float(plant.latitude), float(plant.longitude),
                float(topology.buses.loc[target, "y"]), float(topology.buses.loc[target, "x"]),
            )
            method = "verified_physical_connection"
            connection_substation = connection.connection_substation
            connection_voltage_kv = float(connection.connection_voltage_kv)
            connection_source = connection.source_document
            connection_locator = connection.source_locator
            rationale = str(connection.get("rationale", "source-verified physical connection"))
        else:
            target = nearest_target
            method = "nearest_retained_transmission_bus"
            connection_substation = pd.NA
            connection_voltage_kv = pd.NA
            connection_source = pd.NA
            connection_locator = pd.NA
            rationale = "attach the Phase 4 source-qualified plant location to the supported transmission boundary"
        rows.append(
            {
                "plant_id": plant.plant_id,
                "source_longitude": float(plant.longitude),
                "source_latitude": float(plant.latitude),
                "target_bus": target,
                "distance_km": distance,
                "attachment_method": method,
                "location_method": plant.location_method,
                "location_source": plant.location_source,
                "connection_substation": connection_substation,
                "connection_voltage_kv": connection_voltage_kv,
                "connection_source": connection_source,
                "connection_locator": connection_locator,
                "rationale": rationale,
            }
        )
    result = pd.DataFrame(rows)
    if result.plant_id.duplicated().any() or set(result.plant_id) != set(generators.plant_id):
        raise AssertionError("Generator attachment must cover each plant exactly once")
    return result


def _apply_attachment_bands(
    frame: pd.DataFrame,
    value_column: str,
    bands: tuple[tuple[float, str, float], ...],
    band_column: str,
) -> pd.DataFrame:
    """Apply a transparent size-dependent distance screen without changing buses."""
    result = frame.copy()
    values = result[value_column].astype(float)
    labels = []
    limits = []
    for value in values:
        _, label, limit = next(band for band in bands if value <= band[0])
        labels.append(label)
        limits.append(limit)
    result[band_column] = labels
    result["acceptable_distance_km"] = limits
    result["distance_exceedance_km"] = (
        result.distance_km - result.acceptable_distance_km
    ).clip(lower=0.0)
    result["screening_status"] = np.where(
        result.distance_km <= result.acceptable_distance_km,
        "acceptable",
        "not acceptable - fix this bus",
    )
    return result


def evaluate_attachment_materiality(
    load_attachments: pd.DataFrame,
    generator_attachments: pd.DataFrame,
    demand: pd.DataFrame,
    generators: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen attachment distances relative to plant/load materiality.

    The returned tables are diagnostics only: target buses are never modified.
    Annual load energy is calculated from the hourly MW profiles.
    """
    load_energy = demand.sum(axis=0).div(1000.0)
    load_screen = load_attachments.copy()
    load_screen["annual_consumption_gwh"] = load_screen.source_bus.map(load_energy)
    if load_screen.annual_consumption_gwh.isna().any():
        raise ValueError("Annual consumption is missing for an attached load")
    load_screen = _apply_attachment_bands(
        load_screen,
        "annual_consumption_gwh",
        LOAD_ATTACHMENT_BANDS,
        "consumption_band",
    )
    load_screen["distance_energy_index_gwh_km"] = (
        load_screen.distance_km * load_screen.annual_consumption_gwh
    )

    plant_attributes = generators.set_index("plant_id")[["name", "carrier", "p_nom_mw"]]
    generator_screen = generator_attachments.join(
        plant_attributes, on="plant_id", validate="one_to_one"
    )
    generator_screen = _apply_attachment_bands(
        generator_screen,
        "p_nom_mw",
        GENERATOR_ATTACHMENT_BANDS,
        "capacity_band",
    )
    generator_screen["distance_capacity_index_mw_km"] = (
        generator_screen.distance_km * generator_screen.p_nom_mw
    )
    return load_screen, generator_screen


def summarize_attachment_screen(
    screen: pd.DataFrame, band_column: str, materiality_column: str
) -> pd.DataFrame:
    """Summarize counts and materiality passed/flagged in each ordered band."""
    rows = []
    for band, group in screen.groupby(band_column, sort=False):
        flagged = group.screening_status.ne("acceptable")
        rows.append(
            {
                band_column: band,
                "attachments": len(group),
                "materiality_total": float(group[materiality_column].sum()),
                "not_acceptable": int(flagged.sum()),
                "materiality_not_acceptable": float(group.loc[flagged, materiality_column].sum()),
                "maximum_distance_km": float(group.distance_km.max()),
                "acceptable_distance_km": float(group.acceptable_distance_km.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _add_carriers(network, generators: pd.DataFrame, carrier_source: pd.DataFrame) -> None:
    used = set(generators.carrier.astype(str)) | {"AC", "electricity", "export", "load_shedding", "interconnection_colombia"}
    carrier_source = carrier_source.copy()
    carrier_source.index = carrier_source.index.astype(str)
    defaults = {
        "AC": {"co2_emissions": 0.0, "color": "#888888", "nice_name": "Alternating current"},
        "electricity": {"co2_emissions": 0.0, "color": "#808080", "nice_name": "Electricity demand"},
        "load_shedding": {"co2_emissions": 0.0, "color": "#DD1818", "nice_name": "Emergency load shedding"},
        "biogas": {"co2_emissions": 0.0, "color": "#4d8f45", "nice_name": "Biogas"},
        "interconnection_colombia": {"co2_emissions": 0.0, "color": "#6a3d9a", "nice_name": "Colombia import"},
        "export": {"co2_emissions": 0.0, "color": "#cab2d6", "nice_name": "International export"},
    }
    for carrier in sorted(used):
        values = defaults.get(carrier, {})
        if carrier in carrier_source.index:
            source = carrier_source.loc[carrier]
            values = {
                "co2_emissions": float(source.get("co2_emissions", 0.0)),
                "color": str(source.get("color", "#999999")),
                "nice_name": str(source.get("nice_name", carrier)),
            }
        if carrier in network.carriers.index:
            for key, value in values.items():
                network.carriers.loc[carrier, key] = value
        else:
            network.add("Carrier", carrier, **values)
    unused = network.carriers.index.difference(sorted(used))
    for carrier in unused:
        network.remove("Carrier", carrier)


def _add_loads(network, demand: pd.DataFrame, attachments: pd.DataFrame) -> None:
    names = attachments.load_id.astype(str).tolist()
    p_set = demand.loc[:, attachments.source_bus].copy()
    p_set.columns = names
    network.madd(
        "Load",
        names,
        bus=attachments.target_bus.astype(str).to_numpy(),
        p_set=p_set,
        carrier="electricity",
    )


def _add_generators(
    network,
    generators: pd.DataFrame,
    attachments: pd.DataFrame,
    hydro: pd.DataFrame,
    renewable: pd.DataFrame,
    costs: pd.DataFrame,
) -> None:
    plants = generators.set_index("plant_id", drop=False).copy()
    attachment_bus = attachments.set_index("plant_id").target_bus
    costs = costs.set_index("carrier")
    names = plants.index.astype(str).tolist()
    profiles = pd.DataFrame(1.0, index=network.snapshots, columns=names)
    profiles.loc[:, hydro.columns] = hydro
    profiles.loc[:, renewable.columns] = renewable
    network.madd(
        "Generator",
        names,
        bus=attachment_bus.loc[names].to_numpy(),
        carrier=plants.loc[names, "carrier"].astype(str).to_numpy(),
        p_nom=plants.loc[names, "p_nom_mw"].astype(float).to_numpy(),
        marginal_cost=plants.loc[names, "carrier"].map(costs.marginal_cost).astype(float).to_numpy(),
        capital_cost=plants.loc[names, "carrier"].map(costs.capital_cost).astype(float).to_numpy(),
        p_nom_extendable=False,
        p_max_pu=profiles,
    )


def _add_emergency_load_shedding(network, cost: float = 1_000_000.0) -> None:
    buses = network.buses.index.astype(str).tolist()
    network.madd(
        "Generator",
        [f"load_shedding_{bus}" for bus in buses],
        bus=buses,
        carrier="load_shedding",
        p_nom=10_000.0,
        marginal_cost=cost,
        capital_cost=0.0,
        p_nom_extendable=False,
    )


def _add_colombia_interconnection(
    network, config: Mapping | None = None
) -> dict[str, object]:
    """Add the documented Pimampiro and Tulcan Colombia interfaces.

    The 500 MW rating is a modeling assumption. The 45.20 USD/MWh marginal cost
    is the CENACE 2022 average Colombia import price (4.52 cUSD/kWh).
    """
    config = dict(config or {})
    max_capacity_mw = float(config.get("max_capacity_mw", 500.0))
    marginal_cost_usd_per_mwh = float(config.get("marginal_cost_usd_per_mwh", 45.2))
    interfaces = config.get("interfaces", {})
    buses = [str(interfaces.get("pimampiro_230_bus", "84")), str(interfaces.get("tulcan_138_bus", "36"))]
    split = list(interfaces.get("representative_power_split", [0.8, 0.2]))
    if any(bus not in network.buses.index for bus in buses) or not np.isclose(sum(split), 1.0):
        raise ValueError("Colombia interface buses must exist and power split must sum to one")
    external_bus = "interconnection_colombia"
    network.add("Bus", external_bus, x=float(network.buses.loc[buses[0], "x"]),
                y=float(network.buses.loc[buses[0], "y"]), v_nom=230.0,
                carrier="AC")
    network.add("Generator", "colombia_import_supply", bus=external_bus,
                carrier="interconnection_colombia", p_nom=max_capacity_mw,
                marginal_cost=marginal_cost_usd_per_mwh, capital_cost=0.0,
                p_nom_extendable=False)
    for name, bus, fraction in zip(("pimampiro", "tulcan"), buses, split):
        network.add("Link", f"colombia_import_link_{name}", bus0=external_bus, bus1=bus,
                    carrier="interconnection_colombia", p_nom=max_capacity_mw * float(fraction),
                    efficiency=1.0, capital_cost=0.0, marginal_cost=0.0,
                    p_nom_extendable=False)
    return {"internal_buses": buses, "external_bus": external_bus,
            "max_capacity_mw": max_capacity_mw, "marginal_cost_usd_per_mwh": marginal_cost_usd_per_mwh}


def _add_historical_exports(network, demand: pd.DataFrame, config: Mapping | None = None) -> None:
    config = dict(config or {})
    national_shape = demand.sum(axis=1)
    normalized = national_shape / national_shape.sum()
    for country in ("colombia", "peru"):
        bus = str(config.get(f"{country}_bus", "84" if country == "colombia" else "123"))
        annual_gwh = float(config.get(f"{country}_gwh", 159.15 if country == "colombia" else 32.42))
        if bus not in network.buses.index:
            raise ValueError(f"Export interface bus {bus} for {country} is missing")
        network.add("Load", f"export_{country}_2022", bus=bus, carrier="export",
                    p_set=normalized * annual_gwh * 1000.0)


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_assembly_invariants(
    network,
    demand: pd.DataFrame,
    generators: pd.DataFrame,
    hydro: pd.DataFrame,
    renewable: pd.DataFrame,
) -> None:
    assert_topology_invariants(network)
    if not network.snapshots.equals(demand.index):
        raise AssertionError("Network snapshots do not match demand")
    domestic = network.loads.index[network.loads.carrier.eq("electricity")]
    if not np.isclose(network.loads_t.p_set.loc[:, domestic].to_numpy().sum(), demand.to_numpy().sum()):
        raise AssertionError("Load attachment did not conserve annual demand")
    real = network.generators.query("~carrier.isin(['load_shedding', 'interconnection_colombia'])")
    if set(real.index) != set(generators.plant_id):
        raise AssertionError("Network generator IDs differ from the Phase 4 fleet")
    expected_capacity = generators.groupby("carrier").p_nom_mw.sum().sort_index()
    actual_capacity = real.groupby("carrier").p_nom.sum().sort_index()
    if not np.allclose(expected_capacity, actual_capacity):
        raise AssertionError("Generator capacity was not conserved by attachment")
    profile_ids = set(hydro.columns) | set(renewable.columns)
    if not profile_ids.issubset(network.generators_t.p_max_pu.columns):
        raise AssertionError("Availability profiles are incomplete in the network")
    if not bool(network.generators.loc[list(profile_ids), "carrier"].isin(["hydro", "solar", "onwind", "biomass"]).all()):
        raise AssertionError("Availability profile assigned to an unexpected carrier")
    if set(network.generators.carrier) - set(network.carriers.index):
        raise AssertionError("Every generator carrier must be defined")


def assemble_2022_network(
    topology,
    source_buses: pd.DataFrame,
    demand: pd.DataFrame,
    generators: pd.DataFrame,
    hydro: pd.DataFrame,
    renewable: pd.DataFrame,
    costs: pd.DataFrame,
    carriers: pd.DataFrame,
    connection_evidence: pd.DataFrame | None = None,
    load_connection_evidence: pd.DataFrame | None = None,
    provenance_paths: dict[str, Path | str] | None = None,
    include_colombia_interconnection: bool = True,
    interconnection_config: Mapping | None = None,
    exports_config: Mapping | None = None,
) -> AssemblyResult:
    """Return a complete, unsolved candidate network and attachment audits."""
    network = topology.copy()
    network.set_snapshots(demand.index)
    load_attachments = build_load_attachment_table(
        network, source_buses, demand.columns, load_connection_evidence
    )
    generator_attachments = build_generator_attachment_table(
        network, generators, connection_evidence
    )
    carrier_input = carriers.copy()
    if "interconnection_colombia" not in carrier_input.index:
        carrier_input.loc["interconnection_colombia", :] = np.nan
    _add_carriers(network, generators, carrier_input)
    _add_loads(network, demand, load_attachments)
    _add_historical_exports(network, demand, exports_config)
    _add_generators(network, generators, generator_attachments, hydro, renewable, costs)
    interconnection = None
    if include_colombia_interconnection:
        interconnection = _add_colombia_interconnection(network, interconnection_config)
    _add_emergency_load_shedding(network)
    network.meta = {
        "project": "EcuadorElectricGrid",
        "model_year": 2022,
        "system_boundary": "mainland Ecuador SNI",
        "status": "candidate unsolved Phase 5 product; not scientifically validated",
        "colombia_interconnection": interconnection,
        "transformation": "src.topology + src.assembly",
        "input_sha256": {
            key: _sha256(path) for key, path in sorted((provenance_paths or {}).items())
        },
    }
    assert_assembly_invariants(network, demand, generators, hydro, renewable)
    fingerprint = structural_fingerprint(network)
    fingerprint.update(
        {
            "real_generators": int((network.generators.carrier != "load_shedding").sum()),
            "load_shedding_generators": int((network.generators.carrier == "load_shedding").sum()),
            "annual_demand_gwh": float(
                network.loads_t.p_set.loc[
                    :, network.loads.index[network.loads.carrier.eq("electricity")]
                ].to_numpy().sum() / 1000.0
            ),
            "annual_exports_gwh": float(
                network.loads_t.p_set.loc[
                    :, network.loads.index[network.loads.carrier.eq("export")]
                ].to_numpy().sum() / 1000.0
            ),
            "nominal_capacity_mw": float(
                network.generators.query("carrier != 'load_shedding'").p_nom.sum()
            ),
        }
    )
    return AssemblyResult(network, load_attachments, generator_attachments, fingerprint)


def assert_reload_equivalent(original, reloaded) -> None:
    if structural_fingerprint(original) != structural_fingerprint(reloaded):
        raise AssertionError("Reloaded network structural fingerprint differs")
    for component in ("buses", "lines", "transformers", "loads", "generators"):
        left, right = getattr(original, component), getattr(reloaded, component)
        if not left.index.equals(right.index):
            raise AssertionError(f"Reload changed {component} identifiers")
    if not np.allclose(original.loads_t.p_set, reloaded.loads_t.p_set):
        raise AssertionError("Reload changed demand time series")
    def effective_availability(network) -> pd.DataFrame:
        names = network.generators.index[network.generators.carrier != "load_shedding"]
        result = pd.DataFrame(index=network.snapshots, columns=names, dtype=float)
        for name in names:
            if name in network.generators_t.p_max_pu.columns:
                result.loc[:, name] = network.generators_t.p_max_pu[name]
            else:
                result.loc[:, name] = float(network.generators.at[name, "p_max_pu"])
        return result

    if not np.allclose(
        effective_availability(original),
        effective_availability(reloaded),
    ):
        raise AssertionError("Reload changed generator availability")
