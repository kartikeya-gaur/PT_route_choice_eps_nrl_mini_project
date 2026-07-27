"""
src.data.gtfs_loader

GTFS bounding-box filtering and chunked table parsing. This module
generalizes the branch-selection logic originally built ad hoc for
Sydney (T1/T2 covering multiple physical branches) and Bengaluru
(GREEN/PURPLE lines), plus the headway-from-departure-times calculation
used for both cities' route-choice wait-time components.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

# Explicit dtypes for the large GTFS tables, used instead of
# low_memory=False - buffering an entire multi-million-row stop_times.txt
# for type inference can exceed available memory (seen in practice on
# Sydney's full-network feed). Being explicit is strictly better: same (or
# better) correctness, far lower peak memory, no DtypeWarning.
_STOP_TIMES_DTYPES = {
    "trip_id": str, "stop_id": str, "stop_sequence": "int32",
    "arrival_time": str, "departure_time": str,
}
_STOP_TIMES_USECOLS = list(_STOP_TIMES_DTYPES.keys())

_TRIPS_DTYPES = {
    "trip_id": str, "route_id": str, "service_id": str,
    "trip_headsign": str, "direction_id": "Int8", "shape_id": str, "block_id": str,
}

# GTFS enum columns in stops.txt that must be numeric, not string, for
# downstream int comparisons (e.g. collapse_platforms_to_stations' location_type
# == 1 check) to work - read everything else in stops.txt as str to avoid
# mixed-type inference warnings (e.g. platform_code), but these two need
# their real dtype restored explicitly.
_STOPS_NUMERIC_COLS = ["stop_lat", "stop_lon", "location_type"]


def gtfs_time_to_seconds(time_str):
    """Converts a GTFS HH:MM:SS string (including hours > 24) to seconds."""
    try:
        h, m, s = str(time_str).strip().split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return np.nan


def haversine_distance(lat1, lon1, lat2, lon2):
    """Earth-surface distance in meters between two coordinate pairs (vectorized)."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.subtract(lat2, lat1))
    dlambda = np.radians(np.subtract(lon2, lon1))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_gtfs_tables(city: str, gtfs_dir: Path = None):
    """Loads stops/trips/routes/stop_times.txt for a city from data/raw/<city>/.

    stop_times.txt and trips.txt use explicit dtypes + usecols rather than
    low_memory=False, to avoid buffering the entire raw file for type
    inference (can OOM on a large feed). stops.txt reads everything as str
    by default (avoids mixed-type-column warnings like platform_code) EXCEPT
    stop_lat/stop_lon/location_type, which are coerced back to numeric -
    location_type in particular MUST be numeric, since
    collapse_platforms_to_stations compares it against the int 1, and a
    string "1" would silently never match, breaking station-name resolution.
    """
    gtfs_dir = gtfs_dir or config.raw_dir(city)

    stops = pd.read_csv(gtfs_dir / "stops.txt", dtype=str)
    for col in _STOPS_NUMERIC_COLS:
        if col in stops.columns:
            stops[col] = pd.to_numeric(stops[col], errors="coerce")

    trips_cols = pd.read_csv(gtfs_dir / "trips.txt", nrows=0).columns
    trips_dtype = {k: v for k, v in _TRIPS_DTYPES.items() if k in trips_cols}
    trips = pd.read_csv(gtfs_dir / "trips.txt", dtype=trips_dtype)

    routes = pd.read_csv(gtfs_dir / "routes.txt", dtype=str)

    stop_times_cols = pd.read_csv(gtfs_dir / "stop_times.txt", nrows=0).columns
    usecols = [c for c in _STOP_TIMES_USECOLS if c in stop_times_cols]
    missing = set(_STOP_TIMES_USECOLS) - set(usecols)
    if missing:
        raise ValueError(
            f"stop_times.txt at {gtfs_dir} is missing expected column(s) {missing} - "
            f"available columns: {list(stop_times_cols)}"
        )
    stop_times = pd.read_csv(
        gtfs_dir / "stop_times.txt",
        usecols=usecols,
        dtype={k: v for k, v in _STOP_TIMES_DTYPES.items() if k in usecols},
    )

    return {"stops": stops, "trips": trips, "routes": routes, "stop_times": stop_times}


def collapse_platforms_to_stations(stops_df: pd.DataFrame):
    """Collapses platform-level stops into station-level nodes via parent_station.

    Returns (stations_df, stop_to_station) where stations_df has one row per
    physical station (station_id, station_name, stop_lat, stop_lon).

    IMPORTANT: station_name is taken from the location_type==1 row (the real
    station-level GTFS record) when one exists, NOT from whichever row
    happens to appear first in the raw file. Some feeds (Sydney's included)
    have multiple rows sharing a parent_station - the real station plus
    light-rail/bus "Stand" rows - and picking the file-order-first name can
    silently grab a stand label like "Parramatta Station, Stand B1" instead
    of "Parramatta Station". This was a real production bug: it also broke
    matching against entry_exit.csv and against config-specified
    branch_boundaries names, since those use the clean station-level name.

    NOTE: this relies on location_type being read as a real numeric dtype
    (see load_gtfs_tables) - if it's a string "1" instead of the int 1, the
    == 1 check below silently never matches and this fix has no effect.
    """
    stops_df = stops_df.copy()
    stops_df["station_id"] = stops_df["parent_station"].fillna(stops_df["stop_id"]).astype(str)
    stops_df["clean_name"] = (
        stops_df["stop_name"]
        .str.replace(r",\s*Platform\s*\d+$", "", regex=True)
        .str.strip()
    )

    def pick_station_name(group):
        station_rows = group[group.get("location_type") == 1] if "location_type" in group else group.iloc[0:0]
        if len(station_rows):
            return station_rows["clean_name"].iloc[0]
        return group["clean_name"].iloc[0]  # fallback: no location_type==1 row, use file order

    name_by_station = stops_df.groupby("station_id", group_keys=False).apply(pick_station_name)
    coords_by_station = stops_df.groupby("station_id").agg(stop_lat=("stop_lat", "mean"), stop_lon=("stop_lon", "mean"))
    stations_df = coords_by_station.assign(station_name=name_by_station).reset_index()[
        ["station_id", "station_name", "stop_lat", "stop_lon"]]

    stop_to_station = dict(zip(stops_df["stop_id"], stops_df["station_id"]))
    return stations_df, stop_to_station


def select_canonical_trips(trips_df, stop_times_df, target_route_ids, selected_headsigns,
                            service_id_filter=None, service_id_map=None, pattern_selection=None):
    """Resolves the branch-selection problem: for each target route, restrict to
    the given headsign (or the fullest pattern if headsign is None), then further
    narrow to the MOST COMMON stop-sequence signature among matching trips - a
    headsign can still contain a short-turn/express variant alongside the real
    full-length pattern (seen in both Sydney's and Bengaluru's raw feeds).

    IMPORTANT: also restrict to a single representative day PER ROUTE before
    pattern-matching (service_id_map), not just a flat filter applied to every
    route the same way (service_id_filter). Without per-route restriction,
    combining trips across many different individual calendar dates can make
    "most common stop sequence" pick a short-turn/shuttle pattern instead of
    the genuine full-length route, if the shuttle pattern happens to repeat
    identically across more individual dates than the full-length one does.

    service_id_map takes priority over service_id_filter if both given;
    service_id_filter (a single flat string applied to every route) remains
    for the Bengaluru-style single-city-wide-calendar case.

    pattern_selection: optional dict of route_id -> "most_common" (default)
    or "longest" - use "longest" for routes where short-turn trips share a
    headsign with the genuine full-length route.

    Returns (canonical_trip_ids, pattern_info) where pattern_info is a dict of
    route_id -> {"n_stops": int, "n_matching_trips": int, "n_candidate_trips": int}.
    """
    pattern_selection = pattern_selection or {}
    trips_df = trips_df[trips_df["route_id"].isin(target_route_ids)].copy()
    if service_id_map is None and service_id_filter is not None:
        trips_df = trips_df[trips_df["service_id"] == service_id_filter]

    headsign_candidates = {}
    for route_id in target_route_ids:
        headsign = selected_headsigns.get(route_id)
        sub = trips_df[trips_df["route_id"] == route_id]
        if headsign is not None:
            sub = sub[sub["trip_headsign"] == headsign]
        if service_id_map is not None:
            route_service_id = service_id_map.get(route_id)
            if route_service_id is not None:
                sub = sub[sub["service_id"] == route_service_id]
        headsign_candidates[route_id] = set(sub["trip_id"])

    all_candidate_ids = set()
    for ids in headsign_candidates.values():
        all_candidate_ids |= ids

    all_seq = (
        stop_times_df[stop_times_df["trip_id"].isin(all_candidate_ids)]
        [["trip_id", "stop_id", "stop_sequence"]]
        .sort_values(["trip_id", "stop_sequence"])
    )
    trip_signatures = all_seq.groupby("trip_id")["stop_id"].apply(tuple)

    canonical_trip_ids = set()
    pattern_info = {}
    for route_id, candidates in headsign_candidates.items():
        sigs = trip_signatures.reindex(list(candidates)).dropna()
        if sigs.empty:
            pattern_info[route_id] = {"n_stops": 0, "n_matching_trips": 0, "n_candidate_trips": 0}
            continue
        if pattern_selection.get(route_id) == "longest":
            unique_sigs = sigs.unique()
            target_sig = max(unique_sigs, key=len)
        else:
            target_sig = sigs.value_counts().idxmax()
        matching = set(sigs[sigs == target_sig].index)
        canonical_trip_ids |= matching
        pattern_info[route_id] = {
            "n_stops": len(target_sig),
            "n_matching_trips": len(matching),
            "n_candidate_trips": len(sigs),
        }

    return canonical_trip_ids, pattern_info


def find_reference_stop(trips_df, stop_times_df, route_id, headsign):
    """Finds the most common first-stop (stop_sequence == 1) among trips matching
    a route_id + headsign - used as the departure-time reference point for
    headway calculation when no explicit reference stop is configured."""
    trip_ids = trips_df[(trips_df["route_id"] == route_id) & (trips_df["trip_headsign"] == headsign)]["trip_id"]
    first_stops = stop_times_df[(stop_times_df["trip_id"].isin(trip_ids)) & (stop_times_df["stop_sequence"] == 1)]
    if first_stops.empty:
        return None
    return first_stops["stop_id"].value_counts().idxmax()


def compute_route_headway(trips_df, stop_times_df, route_id, headsign=None,
                           reference_stop_id=None, service_id=None,
                           max_gap_sec=3600):
    """Computes median headway (minutes) for a route/direction from real GTFS
    departure times at a reference stop."""
    sub_trips = trips_df[trips_df["route_id"] == route_id]
    if headsign is not None:
        sub_trips = sub_trips[sub_trips["trip_headsign"] == headsign]
    if service_id is not None:
        sub_trips = sub_trips[sub_trips["service_id"] == service_id]

    if reference_stop_id is None and headsign is not None:
        reference_stop_id = find_reference_stop(trips_df, stop_times_df, route_id, headsign)
    if reference_stop_id is None:
        return None

    trip_ids = set(sub_trips["trip_id"])
    st = stop_times_df[(stop_times_df["trip_id"].isin(trip_ids)) & (stop_times_df["stop_id"] == reference_stop_id)].copy()
    st["dep_sec"] = st["departure_time"].apply(gtfs_time_to_seconds)
    st = st.dropna(subset=["dep_sec"]).sort_values("dep_sec")

    gaps = st["dep_sec"].diff().dropna()
    gaps = gaps[(gaps > 0) & (gaps < max_gap_sec)]
    if gaps.empty:
        return None
    return gaps.median() / 60


def load_calendar(city: str, gtfs_dir=None):
    """Loads calendar.txt if present. Returns None if the feed doesn't have
    one (some feeds, like Bengaluru's, use a simple 'weekday'/'weekend'
    service_id string directly in trips.txt instead - see
    CITY_CONFIGS[city]['service_id_filter'] for that case)."""
    gtfs_dir = gtfs_dir or config.raw_dir(city)
    cal_path = gtfs_dir / "calendar.txt"
    if not cal_path.exists():
        return None
    cal = pd.read_csv(cal_path)
    cal.columns = [c.strip().lstrip("\ufeff") for c in cal.columns]
    return cal


def find_weekday_service_id(trips_df, calendar_df, route_id, headsign):
    """Finds the single most-used service_id for a route/headsign that
    represents a genuine weekday (runs on at least one of Mon-Fri, never on
    a weekend) - built from real calendar.txt, not hardcoded.

    "runs on at least one weekday, no weekend" is deliberately broader than
    "runs on all 5 weekdays" - many real-world feeds encode almost every
    service_id as a single specific calendar date rather than a recurring
    Mon-Fri pattern, so requiring all 5 flags can find almost nothing. A
    service_id that only runs on, say, Fridays is still a legitimate
    weekday snapshot for headway/topology purposes.
    """
    if calendar_df is None:
        return None
    weekday_mask = (
        ((calendar_df["monday"] == 1) | (calendar_df["tuesday"] == 1) | (calendar_df["wednesday"] == 1)
         | (calendar_df["thursday"] == 1) | (calendar_df["friday"] == 1))
        & (calendar_df["saturday"] == 0) & (calendar_df["sunday"] == 0)
    )
    weekday_ids = set(calendar_df[weekday_mask]["service_id"])

    sub = trips_df[(trips_df["route_id"] == route_id) & (trips_df["trip_headsign"] == headsign)]
    sub = sub[sub["service_id"].isin(weekday_ids)]
    if sub.empty:
        return None
    return sub["service_id"].value_counts().idxmax()


def resolve_weekday_service_ids(city: str, trips_df, calendar_df, target_route_ids, selected_headsigns):
    """Resolves ONE representative weekday service_id per route, using the
    same logic as find_weekday_service_id. Used to filter BOTH the
    topology-building step (select_canonical_trips) and headway calculation
    to the SAME single representative day - critical, because combining
    every service_id for topology selection can make "most common stop
    sequence" pick a short-turn/shuttle pattern instead of the genuine
    full-length route.

    Falls back to cfg['headway_reference_service_id'] (manual override) or
    cfg['service_id_filter'] (flat string, Bengaluru-style) if calendar.txt
    isn't available or a route/headsign has no weekday match.
    """
    cfg = config.CITY_CONFIGS[city]
    service_id_map = {}
    for route_id in target_route_ids:
        headsign = selected_headsigns.get(route_id)
        service_id = None
        if calendar_df is not None and headsign is not None:
            service_id = find_weekday_service_id(trips_df, calendar_df, route_id, headsign)
        if service_id is None:
            service_id = cfg.get("headway_reference_service_id", {}).get(route_id)
        if service_id is None:
            service_id = cfg.get("service_id_filter")
        service_id_map[route_id] = service_id
    return service_id_map


def build_headway_table(city: str, gtfs_tables: dict) -> pd.DataFrame:
    """Builds the full route_id -> headway_min table for a city using its
    config.CITY_CONFIGS settings. Bus routes (Bengaluru only) are computed
    separately in network/supernetwork.py since they come from a different
    GTFS feed (BMTC) not covered by this function.

    Uses resolve_weekday_service_ids for the SAME per-route service_id
    resolution used to build network topology (build_metro_edges /
    select_canonical_trips) - important that these match, so headway and
    travel time both come from literally the same representative day's
    schedule.
    """
    cfg = config.CITY_CONFIGS[city]
    trips_df, stop_times_df = gtfs_tables["trips"], gtfs_tables["stop_times"]
    calendar_df = load_calendar(city)

    service_id_map = resolve_weekday_service_ids(
        city, trips_df, calendar_df, cfg["target_route_ids"], cfg["selected_headsigns"])

    rows = []
    for route_id in cfg["target_route_ids"]:
        headsign = cfg["selected_headsigns"].get(route_id)
        service_id = service_id_map[route_id]
        if service_id is None:
            print(f"  WARNING: no weekday service_id resolved for {route_id} - combining ALL "
                  f"calendars, which may overstate frequency if the feed uses per-date calendars.")

        headway = compute_route_headway(trips_df, stop_times_df, route_id, headsign=headsign,
                                          service_id=service_id)
        print(f"  {route_id}: headway={headway:.1f} min (service_id={service_id!r})"
              if headway else f"  {route_id}: headway calculation failed, using default")
        rows.append({"route_id": route_id, "headway_min": headway if headway else config.DEFAULT_HEADWAY_MIN})
    return pd.DataFrame(rows)