"""
src.network.supernetwork

Builds the routable multimodal network for a city:
  1. Collapse GTFS platforms into stations, select the canonical branch/
     pattern per route (src.data.gtfs_loader), and compute duration-
     averaged transit edges - BOTH DIRECTIONS (a real bug from the
     original ad hoc script: duration-averaging deduped each station pair
     down to a single row, silently making the network only routable in
     one direction - fixed here by mirroring every transit edge).
  2. Generate walking-transfer edges between nearby stations (<=
     MAX_WALK_DISTANCE_M), skipping pairs already transit-connected.
  3. (Bengaluru only) Pull in parallel BMTC bus routes near the network's
     boundary stations, so the small test network has genuine route
     redundancy (a pure single-mode tree has exactly one physical path
     per OD pair - nothing for EPS/NRL to actually choose between).

Output: (stations_df, edges_df) with columns
  stations_df: station_id, station_name, stop_lat, stop_lon
  edges_df:    from_stop, to_stop, route_id, duration, link_type
"""

import numpy as np
import pandas as pd
import networkx as nx

from src import config
from src.data import gtfs_loader


def build_metro_edges(city: str, gtfs_tables: dict):
    """Builds duration-weighted, branch-clean, bidirectional transit edges
    for a city's metro/rail routes, per src.config.CITY_CONFIGS."""
    cfg = config.CITY_CONFIGS[city]
    stops_df, trips_df, stop_times_df = gtfs_tables["stops"], gtfs_tables["trips"], gtfs_tables["stop_times"]

    stations_df, stop_to_station = gtfs_loader.collapse_platforms_to_stations(stops_df)
    stations_lookup = stations_df.set_index("station_id")

    calendar_df = gtfs_loader.load_calendar(city)
    service_id_map = gtfs_loader.resolve_weekday_service_ids(
        city, trips_df, calendar_df, cfg["target_route_ids"], cfg["selected_headsigns"])
    for route_id, sid in service_id_map.items():
        if sid is None:
            print(f"  WARNING: no weekday service_id resolved for {route_id} - canonical pattern "
                  f"selection will combine ALL calendars for this route, which can pick a "
                  f"short-turn/shuttle pattern instead of the genuine full-length route.")

    canonical_trip_ids, pattern_info = gtfs_loader.select_canonical_trips(
        trips_df, stop_times_df,
        cfg["target_route_ids"], cfg["selected_headsigns"],
        service_id_filter=cfg.get("service_id_filter"),
        service_id_map=service_id_map,
        pattern_selection=cfg.get("pattern_selection"),
    )
    for route_id, info in pattern_info.items():
        print(f"  {route_id}: {info['n_stops']} stops/trip, "
              f"{info['n_matching_trips']}/{info['n_candidate_trips']} candidate trips match canonical pattern")

    if not canonical_trip_ids:
        raise ValueError(
            f"No canonical trips found for city={city!r}. This usually means the GTFS files in "
            f"data/raw/{city}/ don't match the route_ids/headsigns configured in src/config.py "
            f"(target_route_ids={cfg['target_route_ids']}, selected_headsigns={cfg['selected_headsigns']}). "
            f"Check trips_df['route_id'].unique() and trips_df['trip_headsign'].unique() against the config."
        )

    st = stop_times_df[stop_times_df["trip_id"].isin(canonical_trip_ids)].copy()
    st["arr_sec"] = st["arrival_time"].apply(gtfs_loader.gtfs_time_to_seconds)
    st["dep_sec"] = st["departure_time"].apply(gtfs_loader.gtfs_time_to_seconds)
    st = st.sort_values(["trip_id", "stop_sequence"])

    trip_route_map = trips_df.set_index("trip_id")["route_id"].to_dict()
    st["route_id"] = st["trip_id"].map(trip_route_map)
    st["station_id"] = st["stop_id"].map(stop_to_station)
    st["next_station_id"] = st.groupby("trip_id")["station_id"].shift(-1)
    st["next_arr_sec"] = st.groupby("trip_id")["arr_sec"].shift(-1)

    links = st.dropna(subset=["next_station_id"]).copy()
    links = links[links["station_id"] != links["next_station_id"]]
    links["travel_time"] = links["next_arr_sec"] - links["dep_sec"]
    links = links[links["travel_time"] > 0]

    transit_edges = (
        links.groupby(["station_id", "next_station_id", "route_id"])["travel_time"]
        .mean()
        .reset_index()
    )
    transit_edges.columns = ["from_stop", "to_stop", "route_id", "duration"]

    transit_edges["pair_key"] = transit_edges.apply(
        lambda r: tuple(sorted([r["from_stop"], r["to_stop"]])), axis=1)
    transit_edges = (
        transit_edges.groupby(["pair_key", "route_id"])
        .agg(duration=("duration", "mean"))
        .reset_index()
    )
    transit_edges["from_stop"] = transit_edges["pair_key"].apply(lambda p: p[0])
    transit_edges["to_stop"] = transit_edges["pair_key"].apply(lambda p: p[1])
    transit_edges["link_type"] = "transit"
    transit_edges = transit_edges[["from_stop", "to_stop", "route_id", "duration", "link_type"]]

    # mirror both directions - duration averaging above collapsed each station
    # pair to one row, but transit obviously runs both ways
    reversed_edges = transit_edges.rename(columns={"from_stop": "to_stop", "to_stop": "from_stop"})
    transit_edges = pd.concat([transit_edges, reversed_edges], ignore_index=True)

    used_station_ids = set(transit_edges["from_stop"]) | set(transit_edges["to_stop"])
    stations_df = stations_df[stations_df["station_id"].isin(used_station_ids)].reset_index(drop=True)

    return stations_df, transit_edges, stations_lookup


def build_walking_transfers(stations_df: pd.DataFrame, existing_edges: pd.DataFrame,
                             max_walk_distance_m=None, walk_speed_mps=None):
    """Generates bidirectional walking-transfer edges between all station
    pairs within max_walk_distance_m, skipping pairs already transit-connected."""
    max_walk_distance_m = max_walk_distance_m or config.MAX_WALK_DISTANCE_M
    walk_speed_mps = walk_speed_mps or config.WALK_SPEED_MPS

    transit_pairs = set(
        tuple(sorted([r.from_stop, r.to_stop])) for r in existing_edges.itertuples()
    )
    coords = stations_df[["station_id", "stop_lat", "stop_lon"]].values

    rows = []
    for i in range(len(coords)):
        sid_i, lat_i, lon_i = coords[i]
        for j in range(i + 1, len(coords)):
            sid_j, lat_j, lon_j = coords[j]
            if tuple(sorted([sid_i, sid_j])) in transit_pairs:
                continue
            dist = gtfs_loader.haversine_distance(lat_i, lon_i, lat_j, lon_j)
            if dist <= max_walk_distance_m:
                duration = dist / walk_speed_mps
                rows.append({"from_stop": sid_i, "to_stop": sid_j, "route_id": "walk",
                             "duration": duration, "link_type": "transfer"})
                rows.append({"from_stop": sid_j, "to_stop": sid_i, "route_id": "walk",
                             "duration": duration, "link_type": "transfer"})
    return pd.DataFrame(rows)


def trim_metro_branches(stations_df, edges_df, branch_boundary_names):
    """Drops any metro station beyond the configured branch boundary stations
    (e.g. Bengaluru's small test network is truncated at Rajajinagar/Lalbagh/
    Mahatma Gandhi Road rather than running each line end-to-end)."""
    name_to_id = dict(zip(stations_df["station_name"], stations_df["station_id"]))
    boundary_ids = {name_to_id[n] for n in branch_boundary_names.values() if n in name_to_id}

    G = nx.from_pandas_edgelist(edges_df, source="from_stop", target="to_stop",
                                 edge_attr=True, create_using=nx.DiGraph())
    if G.number_of_nodes() == 0:
        return stations_df, edges_df
    hub = max(G.nodes(), key=lambda n: G.degree(n))

    keep_ids = {hub}
    for boundary_id in boundary_ids:
        if nx.has_path(G, hub, boundary_id):
            keep_ids.update(nx.shortest_path(G, hub, boundary_id, weight="duration"))

    stations_out = stations_df[stations_df["station_id"].isin(keep_ids)].reset_index(drop=True)
    edges_out = edges_df[edges_df["from_stop"].isin(keep_ids) & edges_df["to_stop"].isin(keep_ids)].reset_index(drop=True)
    return stations_out, edges_out


def add_parallel_bus_routes(city: str, stations_df: pd.DataFrame, edges_df: pd.DataFrame):
    """(Bengaluru only) Pulls in parallel BMTC bus routes near the network's
    boundary stations, giving genuine route redundancy - a pure single-mode
    tree network has exactly one physical path per OD pair, which gives
    EPS/NRL nothing to actually choose between.

    Requires the BMTC GTFS feed at data/raw/bengaluru/bmtc/ (see
    github.com/Vonter/bmtc-gtfs). Each configured bus route is trimmed to
    the segment between its two anchor stations (or an explicit
    trim_at_stop_id, e.g. 314-D is trimmed at Mayohall rather than running
    all the way to Indiranagar) using the same canonical-pattern-selection
    logic as the metro network.

    Also computes each bus route's real headway from actual departure times
    at its reference stop (the anchor_a end of the trimmed segment), using
    ALL of that route's trips (not just the canonical-pattern-matching
    subset used for topology) - frequency should reflect the whole day's
    timetable, not just the trips sharing one exact stopping pattern.
    Returns (combined_stations, combined_edges, bus_headways) where
    bus_headways is {"BUS_<short_name>": headway_min}.
    """
    cfg = config.CITY_CONFIGS[city]
    bus_gtfs_dir = config.raw_dir(city) / "bmtc"
    bus_trips = pd.read_csv(bus_gtfs_dir / "trips.txt")
    bus_stops = pd.read_csv(bus_gtfs_dir / "stops.txt")
    stop_name = dict(zip(bus_stops["stop_id"], bus_stops["stop_name"]))
    stop_lat = dict(zip(bus_stops["stop_id"], bus_stops["stop_lat"]))
    stop_lon = dict(zip(bus_stops["stop_id"], bus_stops["stop_lon"]))

    name_to_id = dict(zip(stations_df["station_name"], stations_df["station_id"]))

    all_bus_stations, all_bus_edges, bus_headways = {}, [], {}

    for short_name, route_cfg in cfg["parallel_bus_routes"].items():
        route_id = route_cfg["route_id"]
        anchor_a_name, anchor_b_name = route_cfg["anchors"]
        anchor_a_coord = _station_coord(stations_df, name_to_id[anchor_a_name])
        anchor_b_coord = _station_coord(stations_df, name_to_id[anchor_b_name])

        stop_to_anchor = _tag_stops_near_anchors(
            bus_stops, {anchor_a_name: anchor_a_coord, anchor_b_name: anchor_b_coord})

        route_trip_ids = set(bus_trips[bus_trips["route_id"] == route_id]["trip_id"])
        st = _read_bus_stop_times_chunked(bus_gtfs_dir / "stop_times.txt", route_trip_ids)

        sigs = st.groupby("trip_id")["stop_id"].apply(tuple)
        if sigs.empty:
            print(f"  WARNING: no stop_times found for bus route {short_name} (route_id={route_id})")
            continue
        mode_sig = sigs.value_counts().idxmax()
        matching_trip_ids = set(sigs[sigs == mode_sig].index)

        edge_durations = _duration_weighted_edges(st, matching_trip_ids)

        rep_trip_id = next(iter(matching_trip_ids))
        rep_seq = st[st["trip_id"] == rep_trip_id].sort_values("stop_sequence")["stop_id"].tolist()
        tags = [stop_to_anchor.get(sid, set()) for sid in rep_seq]

        idx_a = next(i for i, t in enumerate(tags) if anchor_a_name in t)
        idx_b_name = anchor_b_name
        trim_at = route_cfg.get("trim_at_stop_id")
        if trim_at is not None:
            idx_b = rep_seq.index(trim_at)
        else:
            idx_b = next(i for i, t in enumerate(tags) if idx_b_name in t)
        lo, hi = sorted([idx_a, idx_b])
        segment = rep_seq[lo:hi + 1]

        # headway: use ALL trips of this route_id (not just matching_trip_ids)
        # departing from the anchor_a end of the segment - a true frequency
        # measure, not restricted to one exact stopping pattern
        reference_stop_id = rep_seq[idx_a]
        headway = _compute_bus_headway(bus_gtfs_dir, route_trip_ids, reference_stop_id)
        bus_headways[f"BUS_{short_name}"] = headway if headway else config.DEFAULT_HEADWAY_MIN

        for sid in segment:
            all_bus_stations[sid] = {"name": stop_name[sid], "lat": stop_lat[sid], "lon": stop_lon[sid]}
        for a, b in zip(segment[:-1], segment[1:]):
            row = edge_durations[(edge_durations["stop_id"] == a) & (edge_durations["next_stop"] == b)]
            dur = row["travel_time"].iloc[0] if not row.empty else np.nan
            all_bus_edges.append({"from_stop": f"BUS_{a}", "to_stop": f"BUS_{b}",
                                   "route_id": f"BUS_{short_name}", "duration": dur, "link_type": "transit"})
            all_bus_edges.append({"from_stop": f"BUS_{b}", "to_stop": f"BUS_{a}",
                                   "route_id": f"BUS_{short_name}", "duration": dur, "link_type": "transit"})

        print(f"  {short_name}: {len(segment)} stops, canonical pattern from {len(matching_trip_ids)}/{len(sigs)} trips, "
              f"headway={bus_headways[f'BUS_{short_name}']:.1f} min (reference stop={reference_stop_id})")

    bus_station_rows = [{"station_id": f"BUS_{sid}", "station_name": info["name"],
                          "stop_lat": info["lat"], "stop_lon": info["lon"]}
                         for sid, info in all_bus_stations.items()]
    bus_stations_df = pd.DataFrame(bus_station_rows).drop_duplicates(subset="station_id")
    bus_edges_df = pd.DataFrame(all_bus_edges)

    combined_stations = pd.concat([stations_df, bus_stations_df], ignore_index=True).drop_duplicates(subset="station_id")
    combined_transit_edges = pd.concat([edges_df, bus_edges_df], ignore_index=True)

    bus_transfer_edges = build_walking_transfers(
        pd.concat([stations_df, bus_stations_df], ignore_index=True),
        combined_transit_edges,
    )
    combined_edges = pd.concat([combined_transit_edges, bus_transfer_edges], ignore_index=True).drop_duplicates()

    return combined_stations, combined_edges, bus_headways


def _compute_bus_headway(bus_gtfs_dir, route_trip_ids, reference_stop_id, max_gap_sec=7200):
    """Median headway (minutes) for a bus route at a reference stop, using
    ALL trips on the route_id (not just those matching the canonical
    stopping pattern used for topology) - frequency should reflect the
    whole day's timetable. max_gap_sec is wider than the metro default
    (7200 vs 3600) since some BMTC routes run infrequently (e.g. 314-D's
    real headway is ~50 min - a 3600s/1hr cutoff would wrongly exclude some
    of its genuine gaps)."""
    st = _read_bus_stop_times_chunked(bus_gtfs_dir / "stop_times.txt", route_trip_ids)
    ref_st = st[st["stop_id"] == reference_stop_id].copy()
    ref_st["dep_sec"] = ref_st["departure_time"].apply(gtfs_loader.gtfs_time_to_seconds)
    ref_st = ref_st.dropna(subset=["dep_sec"]).sort_values("dep_sec")

    gaps = ref_st["dep_sec"].diff().dropna()
    gaps = gaps[(gaps > 0) & (gaps < max_gap_sec)]
    if gaps.empty:
        return None
    return gaps.median() / 60


def _station_coord(stations_df, station_id):
    row = stations_df[stations_df["station_id"] == station_id].iloc[0]
    return (row["stop_lat"], row["stop_lon"])


def _tag_stops_near_anchors(bus_stops_df, anchors: dict, radius_m=None):
    radius_m = radius_m or config.MAX_WALK_DISTANCE_M
    stop_to_anchor = {}
    for name, (lat, lon) in anchors.items():
        d = gtfs_loader.haversine_distance(lat, lon, bus_stops_df["stop_lat"].values, bus_stops_df["stop_lon"].values)
        for sid in bus_stops_df[d <= radius_m]["stop_id"]:
            stop_to_anchor.setdefault(sid, set()).add(name)
    return stop_to_anchor


def _read_bus_stop_times_chunked(stop_times_path, trip_ids, chunksize=300_000):
    frames = []
    for chunk in pd.read_csv(stop_times_path, chunksize=chunksize,
                              usecols=["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"]):
        sub = chunk[chunk["trip_id"].isin(trip_ids)]
        if not sub.empty:
            frames.append(sub)
    return pd.concat(frames, ignore_index=True).sort_values(["trip_id", "stop_sequence"])


def _duration_weighted_edges(stop_times_df, matching_trip_ids):
    seg = stop_times_df[stop_times_df["trip_id"].isin(matching_trip_ids)].copy()
    seg["arr_sec"] = seg["arrival_time"].apply(gtfs_loader.gtfs_time_to_seconds)
    seg["dep_sec"] = seg["departure_time"].apply(gtfs_loader.gtfs_time_to_seconds)
    seg = seg.sort_values(["trip_id", "stop_sequence"])
    seg["next_stop"] = seg.groupby("trip_id")["stop_id"].shift(-1)
    seg["next_arr"] = seg.groupby("trip_id")["arr_sec"].shift(-1)
    seg = seg.dropna(subset=["next_stop"])
    seg["travel_time"] = seg["next_arr"] - seg["dep_sec"]
    seg = seg[seg["travel_time"] > 0]
    return seg.groupby(["stop_id", "next_stop"])["travel_time"].mean().reset_index()


def build_supernetwork(city: str, save=True):
    """Full pipeline: metro edges -> walking transfers -> (Bengaluru) parallel
    bus routes -> branch trimming. Returns (stations_df, edges_df)."""
    cfg = config.CITY_CONFIGS[city]
    print(f"Building supernetwork for {city}...")

    gtfs_tables = gtfs_loader.load_gtfs_tables(city)
    stations_df, transit_edges, _ = build_metro_edges(city, gtfs_tables)
    transfer_edges = build_walking_transfers(stations_df, transit_edges)
    edges_df = pd.concat([transit_edges, transfer_edges], ignore_index=True)
    print(f"  metro: {len(stations_df)} stations, {len(edges_df)} directed edges")

    if "branch_boundaries" in cfg:
        stations_df, edges_df = trim_metro_branches(stations_df, edges_df, cfg["branch_boundaries"])
        print(f"  after branch trim: {len(stations_df)} stations, {len(edges_df)} directed edges")

    if "parallel_bus_routes" in cfg:
        stations_df, edges_df, bus_headways = add_parallel_bus_routes(city, stations_df, edges_df)
        print(f"  after bus integration: {len(stations_df)} stations, {len(edges_df)} directed edges")
        if save:
            pd.DataFrame(
                [{"route_id": k, "headway_min": v} for k, v in bus_headways.items()]
            ).to_csv(config.processed_dir(city) / "bus_route_headways.csv", index=False)

    edges_df = edges_df.drop_duplicates()

    if save:
        out_dir = config.processed_dir(city)
        stations_df.to_csv(out_dir / "stations.csv", index=False)
        edges_df.to_csv(out_dir / "supernetwork_edges.csv", index=False)
        print(f"  saved to {out_dir}")

    return stations_df, edges_df


def load_bus_headways(city: str) -> dict:
    """Loads bus_route_headways.csv (written by build_supernetwork when the
    city has parallel_bus_routes configured) as a route_id -> headway_min
    dict. Returns {} if the file doesn't exist (e.g. city has no bus routes,
    or build_supernetwork hasn't been run yet)."""
    path = config.processed_dir(city) / "bus_route_headways.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["route_id"], df["headway_min"]))


def to_networkx(edges_df: pd.DataFrame) -> nx.DiGraph:
    return nx.from_pandas_edgelist(edges_df, source="from_stop", target="to_stop",
                                    edge_attr=True, create_using=nx.DiGraph())