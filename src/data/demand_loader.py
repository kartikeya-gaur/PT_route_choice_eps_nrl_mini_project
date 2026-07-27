"""
src.data.demand_loader
"""

import networkx as nx
import numpy as np
import pandas as pd

from src import config


def load_opal_demand(entry_exit_path, stations_df):
    ee = pd.read_csv(entry_exit_path)
    ee["Station"] = ee["Station"].str.replace(r"\s+", " ", regex=True).str.strip()
    ee["Trip_clean"] = pd.to_numeric(ee["Trip"].astype(str).str.replace(",", ""), errors="coerce")
    ee.loc[ee["Trip"] == "Less than 50", "Trip_clean"] = 25  # suppressed small-count proxy

    n_uncoerced = ee["Trip_clean"].isna().sum()
    if n_uncoerced:
        bad_values = ee.loc[ee["Trip_clean"].isna(), "Trip"].unique()[:10]
        print(f"WARNING: {n_uncoerced} entry_exit.csv rows had a 'Trip' value that didn't "
              f"parse to a number and weren't 'Less than 50' - treated as 0 by groupby().sum() "
              f"(NaN is skipped, not summed as 0-contributing, so this silently UNDERCOUNTS demand "
              f"for any station where this happens). Sample unparsed values: {list(bad_values)}")

    net_names = set(stations_df["station_name"].str.strip())
    net_ee = ee[ee["Station"].isin(net_names)]
    unmatched = net_names - set(net_ee["Station"])
    if unmatched:
        print(f"WARNING: {len(unmatched)} network stations not found in entry_exit.csv: {unmatched}")

    totals = net_ee.groupby(["Station", "Entry_Exit"])["Trip_clean"].sum().unstack(fill_value=0)
    print(f"load_opal_demand: Entry_Exit categories found in data: {list(totals.columns)}")
    if "Entry" not in totals.columns or "Exit" not in totals.columns:
        print("WARNING: expected 'Entry'/'Exit' categories not both present in the "
              "Entry_Exit column - total_entries/total_exits will be all-zero for the "
              "missing category, which will silently zero out the gravity model downstream. "
              f"Check the actual category strings above against what load_opal_demand expects.")

    totals = totals.rename(columns={"Entry": "total_entries", "Exit": "total_exits"})
    # guarantee these columns exist even if the Entry_Exit category names didn't
    # match "Entry"/"Exit" above - otherwise downstream code (dropna, etc.) hits
    # a confusing raw KeyError instead of the WARNING already printed above
    for col in ("total_entries", "total_exits"):
        if col not in totals.columns:
            totals[col] = 0.0
    totals["total_demand"] = totals["total_entries"] + totals["total_exits"]
    totals.index.name = "station_name"
    totals = totals.reset_index()

    return stations_df.merge(totals, on="station_name", how="left")


def load_bmrcl_demand(station_hourly_path, station_hourly_exits_path, stations_df):
    entries = pd.read_csv(station_hourly_path, sep=";")
    exits = pd.read_csv(station_hourly_exits_path, sep=";")

    entry_totals = entries.groupby("Station")["Ridership"].sum().rename("total_entries")
    exit_totals = exits.groupby("Station")["Ridership"].sum().rename("total_exits")

    demand = pd.concat([entry_totals, exit_totals], axis=1).fillna(0)
    demand["total_demand"] = demand["total_entries"] + demand["total_exits"]
    demand.index.name = "station_name"
    demand = demand.reset_index()

    return stations_df.merge(demand, on="station_name", how="left")


def load_bmrcl_od(stationpair_hourly_path, stations_df):
    sp = pd.read_csv(stationpair_hourly_path, sep=";")
    net_names = set(stations_df["station_name"])

    covered = sp[sp["Origin Station"].isin(net_names) & sp["Destination Station"].isin(net_names)]
    print(f"OD rows covered by the network: {len(covered)}/{len(sp)}")

    od_matrix = (
        covered.groupby(["Origin Station", "Destination Station"])["Ridership"]
        .sum()
        .reset_index()
        .rename(columns={"Origin Station": "origin_station_name",
                          "Destination Station": "dest_station_name",
                          "Ridership": "total_trips"})
        .sort_values("total_trips", ascending=False)
    )

    name_to_id = dict(zip(stations_df["station_name"], stations_df["station_id"]))
    od_matrix["origin_station_id"] = od_matrix["origin_station_name"].map(name_to_id)
    od_matrix["dest_station_id"] = od_matrix["dest_station_name"].map(name_to_id)
    assert od_matrix["origin_station_id"].notna().all() and od_matrix["dest_station_id"].notna().all(), \
        "Some OD station names failed to map to station_id - name mismatch with stations_df"

    return od_matrix


def build_gravity_od(stations_with_demand, graph: nx.DiGraph, beta=0.05,
                      target_total_trips=18000, n_iterations=15,
                      min_trip_threshold=0.5):
    """Estimates a station-to-station OD matrix via a doubly-constrained
    gravity model (Furness/IPF balancing), calibrated so row/column sums
    match real entry/exit totals, with real network shortest-path travel
    time as the impedance (exp(-beta * time_minutes)).

    This is an ESTIMATE, not measured data - use load_bmrcl_od (or an
    equivalent real OD source) whenever one is available instead.
    """
    n_before_dropna = len(stations_with_demand)
    stations_with_demand = stations_with_demand.dropna(subset=["total_entries", "total_exits"])
    n_after_dropna = len(stations_with_demand)
    if n_after_dropna < n_before_dropna:
        print(f"build_gravity_od: dropped {n_before_dropna - n_after_dropna} stations with "
              f"missing (NaN) total_entries/total_exits (no demand-data match)")

    station_ids = [sid for sid in stations_with_demand["station_id"] if sid in graph.nodes()]
    print(f"build_gravity_od: {n_after_dropna} stations with demand data, "
          f"{len(station_ids)} of those found in graph.nodes()")
    if len(station_ids) < 2:
        raise ValueError(
            f"Only {len(station_ids)} station(s) with demand data matched a graph node - "
            f"can't build an OD matrix. Check dtype/value alignment between "
            f"stations_with_demand['station_id'] and graph.nodes(), e.g.:\n"
            f"  stations_with_demand['station_id'] sample: {list(stations_with_demand['station_id'])[:5]}\n"
            f"  graph.nodes() sample: {list(graph.nodes())[:5]}"
        )

    demand_idx = stations_with_demand.set_index("station_id")
    O = np.array([demand_idx.loc[sid, "total_entries"] for sid in station_ids], dtype=float)
    D = np.array([demand_idx.loc[sid, "total_exits"] for sid in station_ids], dtype=float)
    print(f"build_gravity_od: total_entries sum={O.sum():.1f} (min={O.min():.1f}, max={O.max():.1f}), "
          f"total_exits sum={D.sum():.1f} (min={D.min():.1f}, max={D.max():.1f})")
    if O.sum() <= 0 or D.sum() <= 0:
        raise ValueError(
            "total_entries and/or total_exits sum to zero across every station with demand "
            "data - the gravity model has nothing to distribute. This usually means the "
            "Entry_Exit category names in entry_exit.csv don't match what load_opal_demand "
            "expects ('Entry'/'Exit' exactly), or 'Trip_clean' failed to parse for most rows - "
            "check the WARNINGs printed by load_opal_demand above."
        )

    times = dict(nx.all_pairs_dijkstra_path_length(graph, weight="duration"))
    n = len(station_ids)
    T_min = np.full((n, n), np.nan)
    for i, si in enumerate(station_ids):
        for j, sj in enumerate(station_ids):
            if si == sj:
                continue
            t = times.get(si, {}).get(sj)
            if t is not None:
                T_min[i, j] = t / 60

    n_reachable_pairs = np.isfinite(T_min).sum()
    n_total_pairs = n * (n - 1)
    print(f"build_gravity_od: {n_reachable_pairs}/{n_total_pairs} directed station pairs "
          f"reachable in the graph")
    if n_reachable_pairs == 0:
        raise ValueError(
            "No station pairs are mutually reachable in the graph - every shortest-path "
            "lookup returned nothing. Check that `graph` passed in is actually the routable "
            "supernetwork graph (built via supernetwork.to_networkx), not an empty or "
            "disconnected one."
        )

    impedance = np.nan_to_num(np.exp(-beta * T_min), nan=0.0)
    np.fill_diagonal(impedance, 0)

    Tij = np.outer(O, D) * impedance
    for _ in range(n_iterations):
        row_sums = Tij.sum(axis=1)
        row_sums[row_sums == 0] = 1
        Tij = Tij * (O / row_sums)[:, None]
        col_sums = Tij.sum(axis=0)
        col_sums[col_sums == 0] = 1
        Tij = Tij * (D / col_sums)[None, :]

    print(f"build_gravity_od: Tij.sum() after Furness balancing = {Tij.sum():.4f}")
    if Tij.sum() <= 0:
        raise ValueError(
            "Gravity model produced an all-zero OD matrix after Furness balancing, despite "
            "positive total_entries/total_exits sums - likely caused by zero reachable pairs "
            "between stations with nonzero demand (demand is concentrated on stations that "
            "can't reach each other in the graph). Check the reachable-pairs count above."
        )

    scale = target_total_trips / Tij.sum()
    Tij_scaled = Tij * scale

    n_above_threshold = int((Tij_scaled >= min_trip_threshold).sum())
    print(f"build_gravity_od: {n_above_threshold}/{n_total_pairs} pairs above the "
          f"{min_trip_threshold}-trip threshold after scaling to target_total_trips={target_total_trips}")
    if n_above_threshold == 0:
        raise ValueError(
            f"Every estimated OD pair fell below min_trip_threshold={min_trip_threshold} after "
            f"scaling to target_total_trips={target_total_trips} - demand is spread too thinly "
            f"across too many station pairs ({n_total_pairs} pairs) for any single pair to clear "
            f"the threshold. Either raise target_total_trips, lower min_trip_threshold, or check "
            f"whether O/D and impedance above look realistic (e.g. impedance near-zero everywhere "
            f"would spread Tij almost uniformly instead of concentrating it on nearby/likely pairs)."
        )

    rows = []
    for i, si in enumerate(station_ids):
        for j, sj in enumerate(station_ids):
            if si == sj or Tij_scaled[i, j] < min_trip_threshold:
                continue
            rows.append({"origin_station_id": si, "dest_station_id": sj,
                         "synthetic_total_trips": Tij_scaled[i, j]})

    od_df = pd.DataFrame(rows).sort_values("synthetic_total_trips", ascending=False)
    name_lookup = stations_with_demand.set_index("station_id")["station_name"]
    od_df["origin_station_name"] = od_df["origin_station_id"].map(name_lookup)
    od_df["dest_station_name"] = od_df["dest_station_id"].map(name_lookup)
    return od_df


def load_demand(city: str, stations_df: pd.DataFrame, graph: nx.DiGraph = None):
    raw = config.raw_dir(city)

    if city == "sydney":
        stations_with_demand = load_opal_demand(raw / "entry_exit.csv", stations_df)
        if graph is None:
            raise ValueError("Sydney has no real OD source - pass `graph` to estimate one via gravity model.")
        od_matrix = build_gravity_od(stations_with_demand, graph)
        od_matrix = od_matrix.rename(columns={"synthetic_total_trips": "total_trips"})
        return stations_with_demand, od_matrix

    elif city == "bengaluru":
        stations_with_demand = load_bmrcl_demand(
            raw / "station-hourly.csv", raw / "station-hourly-exits.csv", stations_df)
        od_matrix = load_bmrcl_od(raw / "stationpair-hourly.csv", stations_df)
        return stations_with_demand, od_matrix

    raise ValueError(f"Unknown city: {city}")