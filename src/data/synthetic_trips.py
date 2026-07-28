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
                              k_paths=None, scale_trips=0.01, min_trips_per_od=2,
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


import numpy as np


def generate_synthetic_trips_universal(graph: nx.DiGraph, origin, dest, headway_map: dict,
                                        n_obs=3000, max_paths=500, beta: dict = None,
                                        random_seed=None):
    """
    Paper-exact synthetic generation for ONE OD pair (Section 6.1's protocol):
    enumerates the UNIVERSAL choice set U (ALL simple paths between origin
    and dest, not a k-shortest-paths shortlist), computes each path's
    systematic utility, then simulates n_obs independent observations by
    drawing i.i.d. Gumbel(0,1) shocks per alternative and taking the argmax
    (Eq. 15: U_in = V_in + eps_in, choice = argmax_i U_in). This is
    mathematically equivalent in distribution to softmax/multinomial
    sampling - implemented via explicit shocks + argmax to match the
    paper's simulation protocol exactly, not just its resulting probabilities.

    Differs from generate_synthetic_trips in a way that matters, not just
    in scope: that function truncates to config.K_PATHS candidates per OD
    via nx.shortest_simple_paths (a SAMPLED/truncated choice set), so "true"
    choices there are only ever picked from a shortlist. This function's
    "true" choices are picked from the full universal set, which is the
    whole point of Section 6.1's design - it lets you check whether EPS's
    sampling-based estimation and NRL's exact recursion both recover the
    true beta when the DGP itself used the complete choice set, not an
    approximation of it.

    max_paths is a safety cutoff: enumerating ALL simple paths is only
    tractable when the OD pair/network is small enough (the paper's own
    network was deliberately stripped of loops to make |U|=170 enumerable -
    see the loops discussion: real, cyclic networks can make the simple-path
    count between two nodes explode combinatorially as the network grows).
    Raises rather than silently truncating, since a silently truncated
    "universal" set defeats the purpose of this function - use
    generate_synthetic_trips's k_paths-truncated shortlist instead if this
    OD pair's full simple-path count isn't tractable.

    Returns (trips_df, candidate_paths, systematic_utilities) - the latter
    two let you compute exact universal-set probabilities (softmax over
    systematic_utilities) for a Fig.-4-style plot without re-enumerating.
    """
    rng = np.random.default_rng(random_seed or config.RANDOM_SEED)
    beta = beta or config.TRUE_UTILITY_PARAMS

    if origin == dest or not nx.has_path(graph, origin, dest):
        raise ValueError(f"No path {origin} -> {dest}")

    candidate_paths = []
    for path in nx.all_simple_paths(graph, origin, dest):
        candidate_paths.append(path)
        if len(candidate_paths) > max_paths:
            raise ValueError(
                f"Universal simple-path set for {origin} -> {dest} exceeds "
                f"max_paths={max_paths} before finishing enumeration - this "
                f"OD pair/network isn't small enough to fully enumerate the "
                f"way the paper's stripped, loop-free Borlange subgraph was "
                f"(|U|=170 there). Pick a sparser OD pair, raise max_paths "
                f"if you've confirmed the true count is still tractable, or "
                f"use generate_synthetic_trips's k_paths-truncated shortlist "
                f"instead (a genuinely different, sampled-choice-set "
                f"generating process, not the universal one)."
            )

    print(f"Universal choice set |U| = {len(candidate_paths)} simple paths for {origin} -> {dest}")

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

    ps_values = path_utils.path_size(paths_link_sets, path_lengths)  # Eq. 16, over the full universal set

    systematic_utilities = np.array([
        path_utils.systematic_utility(rec["ivt_min"], rec["wait_min"], rec["walk_min"],
                                       rec["n_transfers"], ps, beta)
        for rec, ps in zip(path_records, ps_values)
    ])
    for rec, ps, v in zip(path_records, ps_values, systematic_utilities):
        rec["path_size"] = ps
        rec["systematic_utility"] = v

    # Eq. 15's simulation protocol: independent Gumbel(0,1) shock per
    # alternative per observation, choice = argmax of shocked utility.
    gumbel_shocks = rng.gumbel(loc=0.0, scale=1.0, size=(n_obs, len(candidate_paths)))
    total_utilities = systematic_utilities[None, :] + gumbel_shocks
    chosen_idx = total_utilities.argmax(axis=1)

    rows = []
    for i, idx in enumerate(chosen_idx):
        rec = path_records[idx]
        rows.append({
            "trip_id": f"{origin}_{dest}_{i}",
            "origin_station_id": origin, "dest_station_id": dest,
            "chosen_path": "|".join(str(s) for s in rec["path"]),
            "n_stops": len(rec["path"]),
            "ivt_min": round(rec["ivt_min"], 2), "wait_min": round(rec["wait_min"], 2),
            "walk_min": round(rec["walk_min"], 2), "n_transfers": rec["n_transfers"],
            "path_size": round(rec["path_size"], 4),
            "systematic_utility": round(rec["systematic_utility"], 4),
            "choice_set_size": len(candidate_paths),
        })

    trips_df = pd.DataFrame(rows)
    print(f"Generated {len(trips_df)} synthetic observations over the full universal set (|U|={len(candidate_paths)})")
    return trips_df, candidate_paths, systematic_utilities


def train_test_split(trips_df, test_fraction=None, random_seed=None):
    """80/20 (default) split, stratified by OD pair so both splits see every
    pair with >=2 trips. Guarantees at least 1 test trip per such group -
    naive rounding ((count * test_fraction).round()) sends every small
    group's test allocation to 0 (e.g. 2 trips * 20% = 0.4 rounds down to
    0), which can make the ENTIRE test set empty if most OD pairs sit at
    min_trips_per_od (a real bug hit in production: build_estimation_data
    got 0 rows, producing a columnless empty dataframe that crashed
    log_likelihood with a confusing KeyError instead of a clear message).
    Groups with exactly 1 trip can't be split at all and stay entirely in
    train.

    Requires `config` (src.config) and `numpy as np` to be importable in
    this module - both should already be present if you're pasting this
    into the existing synthetic_trips.py.
    """
    from src import config  # local import if this isn't already imported at module level

    test_fraction = test_fraction or config.TEST_FRACTION
    random_seed = random_seed or config.RANDOM_SEED

    trips_df = trips_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    grp = trips_df.groupby(["origin_station_id", "dest_station_id"])
    group_sizes = grp["trip_id"].transform("count")
    n_test_per_group = np.where(group_sizes >= 2, np.maximum(1, np.round(group_sizes * test_fraction)), 0)
    test_mask = grp.cumcount(ascending=False) < n_test_per_group
    return trips_df[~test_mask].reset_index(drop=True), trips_df[test_mask].reset_index(drop=True)