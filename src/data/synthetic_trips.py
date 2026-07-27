"""
src.data.synthetic_trips

Generates disaggregate synthetic path-choice observations - the data EPS
and NRL actually need to be estimated/validated against, since neither
real OD counts nor patronage data reveal which link sequence a traveler
used. See src.models.path_utils and src.config for the "true" Path Size
Logit generating model and utility parameters.

City-agnostic: takes a graph, headway map, and OD matrix (with columns
origin_station_id/dest_station_id/total_trips - real for Bengaluru, a
gravity-model estimate for Sydney) and produces the same disaggregate
trip table shape for either city.
"""

import numpy as np
import pandas as pd
import networkx as nx

from src import config
from src.models import path_utils


def generate_synthetic_trips(graph: nx.DiGraph, od_matrix: pd.DataFrame, headway_map: dict,
                              k_paths=None, scale_trips=0.001, min_trips_per_od=2,
                              beta: dict = None, random_seed=None):
    """
    od_matrix: dataframe with origin_station_id, dest_station_id, total_trips
               (plus optional origin_station_name/dest_station_name for readability)
    scale_trips: multiplier applied to total_trips before rounding to an
                 integer number of synthetic trips to simulate per OD pair -
                 real cumulative demand is usually far larger than a
                 tractable disaggregate sample size.
    """
    k_paths = k_paths or config.K_PATHS
    beta = beta or config.TRUE_UTILITY_PARAMS
    rng = np.random.default_rng(random_seed or config.RANDOM_SEED)

    has_names = "origin_station_name" in od_matrix.columns

    rows = []
    skipped_no_path = 0

    for _, od_row in od_matrix.iterrows():
        o_id, d_id = od_row["origin_station_id"], od_row["dest_station_id"]
        real_trips = od_row["total_trips"]
        if o_id == d_id or not nx.has_path(graph, o_id, d_id):
            skipped_no_path += 1
            continue

        try:
            gen = nx.shortest_simple_paths(graph, o_id, d_id, weight="duration")
            candidate_paths = []
            for p in gen:
                candidate_paths.append(p)
                if len(candidate_paths) >= k_paths:
                    break
        except nx.NetworkXNoPath:
            skipped_no_path += 1
            continue

        path_records, paths_link_sets, path_lengths = [], [], []
        for path in candidate_paths:
            edge_seq = [graph.edges[u, v] for u, v in zip(path[:-1], path[1:])]
            ivt, wait, walk, n_transfers = path_utils.compute_path_components(edge_seq, headway_map)
            total_dur = sum(e["duration"] for e in edge_seq) or 1e-6
            link_set = [(u, v, e["duration"]) for (u, v), e in zip(zip(path[:-1], path[1:]), edge_seq)]
            paths_link_sets.append(link_set)
            path_lengths.append(total_dur)
            path_records.append({"path": path, "ivt_min": ivt, "wait_min": wait,
                                  "walk_min": walk, "n_transfers": n_transfers})

        ps_values = path_utils.path_size(paths_link_sets, path_lengths)

        utilities = []
        for rec, ps in zip(path_records, ps_values):
            v = path_utils.systematic_utility(rec["ivt_min"], rec["wait_min"], rec["walk_min"],
                                               rec["n_transfers"], ps, beta)
            utilities.append(v)
            rec["path_size"] = ps
            rec["systematic_utility"] = v

        utilities = np.array(utilities)
        probs = np.exp(utilities - utilities.max())
        probs = probs / probs.sum()

        n_trips = max(min_trips_per_od, int(round(real_trips * scale_trips)))
        chosen_idx = rng.choice(len(path_records), size=n_trips, p=probs)

        o_name = od_row["origin_station_name"] if has_names else o_id
        d_name = od_row["dest_station_name"] if has_names else d_id

        for i, idx in enumerate(chosen_idx):
            rec = path_records[idx]
            rows.append({
                "trip_id": f"{o_id}_{d_id}_{i}",
                "origin_station_id": o_id, "origin_station_name": o_name,
                "dest_station_id": d_id, "dest_station_name": d_name,
                "chosen_path": "|".join(str(s) for s in rec["path"]),
                "n_stops": len(rec["path"]),
                "ivt_min": round(rec["ivt_min"], 2), "wait_min": round(rec["wait_min"], 2),
                "walk_min": round(rec["walk_min"], 2), "n_transfers": rec["n_transfers"],
                "path_size": round(rec["path_size"], 4),
                "systematic_utility": round(rec["systematic_utility"], 4),
                "choice_set_size": len(path_records),
                "od_total_trips": real_trips,
            })

    trips_df = pd.DataFrame(rows)
    print(f"Skipped {skipped_no_path} unreachable/same-node OD pairs")
    print(f"Generated {len(trips_df)} synthetic trips, avg choice set size "
          f"{trips_df['choice_set_size'].mean():.1f}" if len(trips_df) else "Generated 0 synthetic trips")
    return trips_df


def train_test_split(trips_df: pd.DataFrame, test_fraction=None, random_seed=None):
    """80/20 (default) split, stratified by OD pair so both splits see every pair."""
    test_fraction = test_fraction or config.TEST_FRACTION
    random_seed = random_seed or config.RANDOM_SEED

    trips_df = trips_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    grp = trips_df.groupby(["origin_station_id", "dest_station_id"])
    test_mask = grp.cumcount(ascending=False) < (grp["trip_id"].transform("count") * test_fraction).round()
    return trips_df[~test_mask].reset_index(drop=True), trips_df[test_mask].reset_index(drop=True)
