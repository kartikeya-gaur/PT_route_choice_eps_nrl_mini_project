"""
main.py

Master execution script. Runs the full pipeline end-to-end for one or
both cities:

  1. Build the supernetwork from raw GTFS (src.network.supernetwork)
  2. Integrate real-world demand / OD (src.data.demand_loader)
  3. Compute real GTFS-derived headways (src.data.gtfs_loader)
  4. Generate synthetic disaggregate path-choice data (src.data.synthetic_trips)
  5. Fit EPS and NRL on the training split (src.models)
  6. Evaluate both on the held-out test split (src.evaluation.metrics)
  7. Save figures, tables, and fitted model parameters (outputs/)

Usage:
    python main.py --city sydney
    python main.py --city bengaluru
    python main.py --city both          # default
    python main.py --city both --skip-models      # data prep only, skip EPS/NRL fitting
    python main.py --city both --model eps         # only fit EPS
    python main.py --city both --model nrl         # only fit NRL
    python main.py --city both --model both        # fit both (default)
"""

import argparse
import json

import pandas as pd

from src import config
from src.data import gtfs_loader, demand_loader, synthetic_trips
from src.network import supernetwork
from src.models.eps_model import EPSModel
from src.models.nrl_model import NRLModel
from src.evaluation import metrics, visualization


def run_city(city: str, skip_models: bool = False, model: str = "both", n_eps_draws: int = 20):
    print(f"\n{'=' * 70}\n{city.upper()}\n{'=' * 70}")

    # 1. network
    stations_df, edges_df = supernetwork.build_supernetwork(city)
    graph = supernetwork.to_networkx(edges_df)

    # 2. demand / OD
    stations_with_demand, od_matrix = demand_loader.load_demand(city, stations_df, graph)
    out_dir = config.processed_dir(city)
    stations_with_demand.to_csv(out_dir / "stations_with_demand.csv", index=False)
    od_matrix.to_csv(out_dir / "od_matrix.csv", index=False)

    # 3. headways
    gtfs_tables = gtfs_loader.load_gtfs_tables(city)
    headway_df = gtfs_loader.build_headway_table(city, gtfs_tables)
    headway_map = dict(zip(headway_df["route_id"], headway_df["headway_min"]))
    headway_map.update(supernetwork.load_bus_headways(city))  # real bus headways, if this city has any
    # 'walk' (transfer edges) never has a headway concept and is never
    # actually consulted in wait-time calculation (see path_utils -
    # transfer edges don't look up headway_map at all) - only backfill a
    # default for genuine TRANSIT route_ids that are still missing one.
    transit_route_ids = edges_df.loc[edges_df["link_type"] == "transit", "route_id"].unique()
    for route_id in transit_route_ids:
        if route_id not in headway_map:
            headway_map[route_id] = config.DEFAULT_HEADWAY_MIN
    headway_df.to_csv(out_dir / "route_headways.csv", index=False)
    print(f"Headways: {headway_map}")

    # 4. synthetic disaggregate trips
    trips_df = synthetic_trips.generate_synthetic_trips(graph, od_matrix, headway_map)
    train_df, test_df = synthetic_trips.train_test_split(trips_df)
    trips_df.to_csv(out_dir / "synthetic_trips.csv", index=False)
    train_df.to_csv(out_dir / "synthetic_trips_train.csv", index=False)
    test_df.to_csv(out_dir / "synthetic_trips_test.csv", index=False)
    print(f"Synthetic trips: {len(trips_df)} total / {len(train_df)} train / {len(test_df)} test")

    # 5. figures
    routes_df = gtfs_tables["routes"]
    visualization.plot_network(stations_df, edges_df, routes_df, title=f"{city.title()} Supernetwork",
                                save_path=config.FIGURES_OUT_DIR / f"{city}_network.png")
    visualization.folium_map(stations_df, edges_df, routes_df,
                              save_path=config.FIGURES_OUT_DIR / f"{city}_network.html")

    if skip_models:
        print(f"--skip-models set: skipping EPS/NRL fitting for {city}")
        return

    run_eps = model in ("eps", "both")
    run_nrl = model in ("nrl", "both")
    eps_fitted, nrl_fitted = None, None

    # 6. EPS
    if run_eps:
        print("\nFitting EPS...")
        eps_model = EPSModel(graph, headway_map)
        eps_fitted = eps_model.fit(train_df, n_draws=n_eps_draws)
        eps_test_ll = eps_model.log_likelihood(test_df, eps_fitted, n_draws=n_eps_draws)
        print(f"EPS fitted: {eps_fitted}")
        print(f"EPS test log-likelihood: {eps_test_ll:.2f}")

        eps_bias = metrics.parameter_bias(eps_fitted)
        eps_bias.to_csv(config.TABLES_OUT_DIR / f"{city}_eps_parameter_bias.csv", index=False)
        with open(config.MODELS_OUT_DIR / f"{city}_eps_params.json", "w") as f:
            json.dump(eps_fitted, f, indent=2, default=str)

    # 7. NRL
    if run_nrl:
        print("\nFitting NRL...")
        nrl_model = NRLModel(graph, headway_map)
        nrl_fitted = nrl_model.fit(train_df)
        nrl_fitted_full = dict(nrl_fitted, beta_pathsize=0.0)
        nrl_test_ll = nrl_model.log_likelihood(test_df, nrl_fitted_full)
        print(f"NRL fitted: {nrl_fitted}")
        print(f"NRL test log-likelihood: {nrl_test_ll:.2f}")

        nrl_bias = metrics.parameter_bias(nrl_fitted)
        nrl_bias.to_csv(config.TABLES_OUT_DIR / f"{city}_nrl_parameter_bias.csv", index=False)
        with open(config.MODELS_OUT_DIR / f"{city}_nrl_params.json", "w") as f:
            json.dump(nrl_fitted, f, indent=2, default=str)

    # 8. comparison table - only meaningful with both fitted, but still
    # useful as a single-model summary if only one was run
    if run_eps or run_nrl:
        comparison = metrics.summarize_comparison(eps_fitted or {}, nrl_fitted or {})
        visualization.save_comparison_table(comparison, config.TABLES_OUT_DIR / f"{city}_eps_vs_nrl.csv")
        print(f"\n{city} results:\n{comparison.to_string(index=False)}")


def main():
    parser = argparse.ArgumentParser(description="Transit route-choice pipeline (EPS vs NRL)")
    parser.add_argument("--city", choices=["sydney", "bengaluru", "both"], default="both")
    parser.add_argument("--skip-models", action="store_true",
                         help="Only run data prep (network/demand/synthetic trips), skip EPS/NRL fitting entirely")
    parser.add_argument("--model", choices=["eps", "nrl", "both"], default="both",
                         help="Which model(s) to fit - useful to run just one at a time, "
                              "e.g. while NRL's slower value-iteration fitting is a work in progress "
                              "you can iterate on EPS alone with --model eps")
    parser.add_argument("--eps-draws", type=int, default=20,
                         help="Number of biased-random-walk draws per OD pair for EPS choice-set sampling")
    args = parser.parse_args()

    cities = ["sydney", "bengaluru"] if args.city == "both" else [args.city]
    for city in cities:
        run_city(city, skip_models=args.skip_models, model=args.model, n_eps_draws=args.eps_draws)


if __name__ == "__main__":
    main()