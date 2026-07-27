"""
tests/test_supernetwork.py

Unit tests for the core network-building logic. These exercise the
specific bugs that were actually caught during development (the
directed-edge dedup bug that made the network only routable one way,
and the path-component leg-segmentation logic) rather than being purely
illustrative - they'd have caught real regressions.
"""

import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.network import supernetwork
from src.data import gtfs_loader
from src.models import path_utils


def test_haversine_distance_known_points():
    # Sydney Opera House to Sydney Harbour Bridge, ~650m apart
    d = gtfs_loader.haversine_distance(-33.8568, 151.2153, -33.8523, 151.2108)
    assert 400 < d < 900


def test_haversine_distance_zero_for_same_point():
    d = gtfs_loader.haversine_distance(-33.86, 151.20, -33.86, 151.20)
    assert d == pytest.approx(0, abs=1e-6)


def test_gtfs_time_to_seconds():
    assert gtfs_loader.gtfs_time_to_seconds("01:02:03") == 3723
    assert gtfs_loader.gtfs_time_to_seconds("25:00:00") == 25 * 3600  # GTFS allows hours > 24
    assert pd.isna(gtfs_loader.gtfs_time_to_seconds("garbage"))


def test_collapse_platforms_to_stations():
    stops = pd.DataFrame({
        "stop_id": ["A", "A_PF1", "A_PF2", "B"],
        "stop_name": ["Station A", "Station A, Platform 1", "Station A, Platform 2", "Station B"],
        "parent_station": [None, "A", "A", None],
        "stop_lat": [1.0, 1.0001, 1.0002, 2.0],
        "stop_lon": [1.0, 1.0001, 1.0002, 2.0],
    })
    stations_df, stop_to_station = gtfs_loader.collapse_platforms_to_stations(stops)
    assert len(stations_df) == 2  # A's 3 rows collapse to 1 station
    assert stop_to_station["A_PF1"] == "A"
    assert stop_to_station["A_PF2"] == "A"
    a_row = stations_df[stations_df["station_id"] == "A"].iloc[0]
    assert a_row["station_name"] == "Station A"  # "Platform N" suffix stripped


def test_collapse_platforms_prefers_real_station_row_over_stand_rows():
    """Regression test for a real bug: Sydney's stops.txt has light-rail/bus
    'Stand' rows sharing a parent_station with the real train station row.
    Picking whichever row appears FIRST in the file (rather than the real
    location_type==1 station row) silently produced garbled names like
    "Parramatta Station, Stand B1" instead of "Parramatta Station" - which
    also broke matching against entry_exit.csv, since that file uses the
    clean station name."""
    stops = pd.DataFrame({
        "stop_id": ["PRRA_STAND", "PRRA"],
        "stop_name": ["Parramatta Station, Stand B1", "Parramatta Station"],
        "parent_station": [None, None],
        "location_type": [2, 1],   # stand row appears FIRST in the file
        "stop_lat": [-33.8, -33.8], "stop_lon": [151.0, 151.0],
    })
    # both rows share a station_id via parent_station in the real data - simulate that:
    stops["parent_station"] = ["PRRA", None]
    stations_df, _ = gtfs_loader.collapse_platforms_to_stations(stops)
    assert stations_df["station_name"].iloc[0] == "Parramatta Station"
    assert "Stand" not in stations_df["station_name"].iloc[0]


def test_transit_edges_are_bidirectional():
    """Regression test for the real bug: duration-averaging deduped each
    station pair to ONE row, silently making the network only routable in
    one direction. Every transit edge must have a mirrored reverse edge."""
    edges = pd.DataFrame([
        {"from_stop": "A", "to_stop": "B", "route_id": "R1", "duration": 100, "link_type": "transit"},
        {"from_stop": "B", "to_stop": "C", "route_id": "R1", "duration": 100, "link_type": "transit"},
    ])
    # simulate the mirroring step from build_metro_edges
    reversed_edges = edges.rename(columns={"from_stop": "to_stop", "to_stop": "from_stop"})
    both = pd.concat([edges, reversed_edges], ignore_index=True)

    G = nx.from_pandas_edgelist(both, source="from_stop", target="to_stop",
                                 edge_attr=True, create_using=nx.DiGraph())
    assert nx.has_path(G, "A", "C")
    assert nx.has_path(G, "C", "A")  # this direction is what the bug broke


def test_walking_transfers_skip_transit_connected_pairs():
    stations = pd.DataFrame({
        "station_id": ["A", "B"],
        "station_name": ["A", "B"],
        "stop_lat": [1.0, 1.0001],   # ~11m apart, well within walk radius
        "stop_lon": [1.0, 1.0001],
    })
    existing_transit = pd.DataFrame([
        {"from_stop": "A", "to_stop": "B", "route_id": "R1", "duration": 60, "link_type": "transit"},
    ])
    transfers = supernetwork.build_walking_transfers(stations, existing_transit, max_walk_distance_m=500)
    assert len(transfers) == 0  # already transit-connected, no redundant walk edge


def test_walking_transfers_created_for_nearby_unconnected_stations():
    stations = pd.DataFrame({
        "station_id": ["A", "B"],
        "station_name": ["A", "B"],
        "stop_lat": [1.0, 1.0001],
        "stop_lon": [1.0, 1.0001],
    })
    empty_edges = pd.DataFrame(columns=["from_stop", "to_stop", "route_id", "duration", "link_type"])
    transfers = supernetwork.build_walking_transfers(stations, empty_edges, max_walk_distance_m=500)
    assert len(transfers) == 2  # bidirectional
    assert set(transfers["link_type"]) == {"transfer"}


def test_path_components_first_boarding_not_counted_as_transfer():
    """The very first boarding of a trip should contribute wait time but NOT
    a transfer - only route changes AFTER the first boarding are transfers."""
    edge_seq = [
        {"link_type": "transit", "route_id": "R1", "duration": 300},
        {"link_type": "transit", "route_id": "R1", "duration": 300},
    ]
    ivt, wait, walk, n_transfers = path_utils.compute_path_components(edge_seq, {"R1": 10.0})
    assert n_transfers == 0
    assert wait == pytest.approx(5.0)  # headway/2
    assert ivt == pytest.approx(10.0)  # 600 sec total


def test_path_components_route_change_counts_as_transfer():
    edge_seq = [
        {"link_type": "transit", "route_id": "R1", "duration": 300},
        {"link_type": "transit", "route_id": "R2", "duration": 300},  # route change = boarding + transfer
    ]
    ivt, wait, walk, n_transfers = path_utils.compute_path_components(edge_seq, {"R1": 10.0, "R2": 8.0})
    assert n_transfers == 1
    assert wait == pytest.approx(5.0 + 4.0)  # both routes' half-headways


def test_path_components_walk_resets_boarding():
    edge_seq = [
        {"link_type": "transit", "route_id": "R1", "duration": 300},
        {"link_type": "transfer", "route_id": "walk", "duration": 120},
        {"link_type": "transit", "route_id": "R1", "duration": 300},  # same route, but after a walk = new boarding+transfer
    ]
    ivt, wait, walk, n_transfers = path_utils.compute_path_components(edge_seq, {"R1": 10.0})
    assert n_transfers == 1
    assert walk == pytest.approx(2.0)


def test_path_size_of_single_path_is_one():
    """A path with no overlap alternatives in its choice set should have
    Path Size == 1 (no correction needed)."""
    links = [("A", "B", 100), ("B", "C", 100)]
    ps = path_utils.path_size([links], [200])
    assert ps[0] == pytest.approx(1.0)


def test_path_size_penalizes_full_overlap_duplicate():
    """Two identical paths in a choice set should each get Path Size 0.5
    (fully shared, so each gets half credit)."""
    links = [("A", "B", 100), ("B", "C", 100)]
    ps = path_utils.path_size([links, links], [200, 200])
    assert ps[0] == pytest.approx(0.5)
    assert ps[1] == pytest.approx(0.5)


def test_cast_node_id_keeps_string_when_graph_nodes_are_strings():
    """Regression test: a real Sydney run crashed with a KeyError because
    string-typed graph nodes (Sydney's light-rail 'Stand' stops force the
    whole stop_id column to string dtype) were being silently coerced back
    to int by the path-casting logic, so a genuinely valid path like
    "276610|276710" no longer matched the graph's actual string nodes."""
    assert path_utils.cast_node_id("276610", example_node="200020") == "276610"
    assert isinstance(path_utils.cast_node_id("276610", example_node="200020"), str)


def test_cast_node_id_converts_to_int_when_graph_nodes_are_int():
    assert path_utils.cast_node_id("276610", example_node=200020) == 276610
    assert isinstance(path_utils.cast_node_id("276610", example_node=200020), int)


def test_cast_path_matches_graph_node_dtype():
    G = nx.DiGraph()
    G.add_edge("276610", "276710")  # string-typed nodes, like the real Sydney bug
    path = path_utils.cast_path(["276610", "276710"], G)
    assert all(isinstance(p, str) for p in path)
    # this is exactly the operation that crashed in production - must not KeyError
    assert G.edges[path[0], path[1]] is not None