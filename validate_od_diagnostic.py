"""
validate_od_diagnostic.py

Standalone script to test the "K-shortest-paths starves beta_ivt of
identifying variation" theory: fits EPS/NRL against the FULL universal
choice set for a single, well-chosen OD pair (src.data.synthetic_trips.
generate_synthetic_trips_universal_od), instead of the network-wide
K-shortest-paths pipeline in main.py.

Run from the project root:
    python validate_od_diagnostic.py --city bengaluru

If parameter recovery is clean here but bad on the full network-wide
results, that confirms the choice-set construction (not the estimators
themselves) is the problem.
"""

import argparse

import networkx as nx
import pandas as pd

from src import config
from src.data import gtfs_loader, synthetic_trips
from src.network import supernetwork
from src.models.eps_model import EPSModel
from src.models.nrl_model import NRLModel
from src.evaluation import metrics


def pick_a_good_od_pair(graph, min_hops=3, max_hops=8):
    """Finds an OD pair with a moderate hop count (not trivially short, not
    huge) so the universal choice set is meaningfully sized but still fast
    to enumerate. Picks the pair with the largest |U| among a sample of
    candidates, to maximize how informative this diagnostic run is."""
    nodes = list(graph.nodes())
    best = None
    for o in nodes[:30]:
        for d in nodes[:30]:
            if o == d or not nx.has_path(graph, o, d):
                continue
            hops = nx.shortest_path_length(graph, o, d)
            if not (min_hops <= hops <= max_hops):
                continue
            paths = synthetic_trips.enumerate_universal_choice_set(graph, o, d, max_hops=hops + 2)
            if best is None or len(paths) > best[2]:
                best = (o, d, len(paths))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["sydney", "bengaluru"])
    parser.add_argument("--origin", default=None)
    parser.add_argument("--dest", default=None)
    parser.add_argument("--n-observations", type=int, default=3000)
    parser.add_argument("--max-hops", type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("VALIDATE_OD_DIAGNOSTIC.PY - universal choice set test")
    print("(if you don't see this banner, you're running the wrong script)")
    print("=" * 70)

    out_dir = config.processed_dir(args.city)
    stations_df = pd.read_csv(out_dir / "stations.csv")
    edges_df = pd.read_csv(out_dir / "supernetwork_edges.csv")
    graph = supernetwork.to_networkx(edges_df)

    headway_df = pd.read_csv(out_dir / "route_headways.csv")
    headway_map = dict(zip(headway_df["route_id"], headway_df["headway_min"]))
    headway_map.update(supernetwork.load_bus_headways(args.city))
    transit_route_ids = edges_df.loc[edges_df["link_type"] == "transit", "route_id"].unique()
    for route_id in transit_route_ids:
        if route_id not in headway_map:
            headway_map[route_id] = config.DEFAULT_HEADWAY_MIN

    if args.origin and args.dest:
        origin, dest = args.origin, args.dest
    else:
        print("No --origin/--dest given - searching for a good OD pair (moderate |U|)...")
        origin, dest, n_paths = pick_a_good_od_pair(graph)
        print(f"Picked OD pair {origin} -> {dest} with |U|={n_paths}")

    trips_df, path_records = synthetic_trips.generate_synthetic_trips_universal_od(
        graph, origin, dest, headway_map,
        n_observations=args.n_observations, max_hops=args.max_hops)

    # quick diagnostic: how much does ivt_min actually vary across the choice set?
    print("\nAttribute variation across the universal choice set (this is the thing to check):")
    attrs = pd.DataFrame(path_records)[["ivt_min", "wait_min", "walk_min", "n_transfers"]]
    print(attrs.describe())

    train_df, test_df = synthetic_trips.train_test_split(trips_df, test_fraction=0.2)

    print("\nFitting EPS on universal-choice-set data...")
    eps_model = EPSModel(graph, headway_map)
    eps_fitted = eps_model.fit(train_df, n_draws=20)
    print("EPS fitted:", eps_fitted)
    print(metrics.parameter_bias(eps_fitted))

    print("\nFitting NRL on universal-choice-set data...")
    nrl_model = NRLModel(graph, headway_map)
    nrl_fitted = nrl_model.fit(train_df)
    print("NRL fitted:", nrl_fitted)
    print(metrics.parameter_bias(nrl_fitted))


if __name__ == "__main__":
    main()