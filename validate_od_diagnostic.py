"""
validate_od_diagnostic.py

Standalone script to test the "K-shortest-paths starves beta_ivt of
identifying variation" theory: fits EPS/NRL against the FULL universal
choice set for a single, well-chosen OD pair
(src.data.synthetic_trips.generate_synthetic_trips_universal), instead of
the network-wide K-shortest-paths pipeline in main.py.

Run from the project root:
    python validate_od_diagnostic.py --city bengaluru

If parameter recovery is clean here but bad on the full network-wide
results, that confirms the choice-set construction (not the estimators
themselves) is the problem.

IMPORTANT: the EPS draw-count sensitivity sweep below fits against the
SAME universal-choice-set trips_df generated earlier in this run - not
reloaded from main.py's old synthetic_trips_train.csv/test.csv on disk.
Reloading those would silently go back to testing the K-shortest-paths
data this script exists to move past, defeating the whole comparison.
"""

import argparse
import time

import networkx as nx
import pandas as pd
import numpy as np

from src import config
from src.data import synthetic_trips
from src.network import supernetwork
from src.models.eps_model import EPSModel
from src.models.nrl_model import NRLModel
from src.evaluation import metrics


def _timed(fn, *args, **kwargs):
    """Runs fn(*args, **kwargs), returns (result, elapsed_seconds). Centralized
    here so every train/test timing in this script uses the same wall-clock
    convention (time.time(), not process/CPU time) - matters if you ever
    compare these numbers against numbers timed a different way elsewhere."""
    start = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - start


def pick_a_good_od_pair(graph, min_paths=2, max_paths=500):
    """Finds the OD pair with the richest TRACTABLE universal choice set -
    ranks candidates by actual |U| (full nx.all_simple_paths enumeration),
    not a hop-count proxy.

    Earlier version ranked by unweighted hop count over only the first 30
    of graph.nodes() as a cheap stand-in for |U|, to avoid enumerating
    all_simple_paths for every candidate pair up front. Both parts of that
    tradeoff turned out to matter on real data: (1) node iteration order
    follows edge-list insertion order, not anything meaningful, so
    "first 30 nodes" silently excluded whichever stations happened to be
    added last to the edges file - on the Sydney network this cut out
    exactly the western corridor (Parramatta, Harris Park, Granville,
    Clyde, Auburn) where nearly all the real path-choice richness lives;
    (2) even among the nodes it did search, hop count correlated poorly
    with |U| - it picked a hops=8 pair with |U|=17 over a hops=7 pair with
    |U|=272 purely because 8 > 7. On a graph this size (tens of nodes),
    directly enumerating |U| for every pair is cheap enough that there's no
    real need to proxy through hop count at all.

    Any candidate whose |U| exceeds max_paths is skipped rather than
    counted at its true (larger) value, since generate_synthetic_trips_universal
    would refuse to run against it anyway - no point recommending an OD
    pair this script can't actually generate data for.
    """
    import itertools

    nodes = list(graph.nodes())
    best = None  # (origin, dest, n_paths, hops)
    for o in nodes:
        for d in nodes:
            if o == d or not nx.has_path(graph, o, d):
                continue
            # count up to max_paths+1 only - we just need to know whether
            # it's tractable and, if so, its exact count; anything beyond
            # max_paths+1 gets treated as "too big" without finishing the
            # (potentially very large) enumeration
            gen = nx.all_simple_paths(graph, o, d)
            n_paths = sum(1 for _ in itertools.islice(gen, max_paths + 1))
            if n_paths > max_paths or n_paths < min_paths:
                continue
            if best is None or n_paths > best[2]:
                hops = nx.shortest_path_length(graph, o, d)
                best = (o, d, n_paths, hops)

    if best is None:
        raise ValueError(f"No OD pair found with {min_paths}<=|U|<=max_paths({max_paths}) "
                          f"anywhere in the graph - widen min_paths/max_paths or pass "
                          f"--origin/--dest explicitly.")
    return best[0], best[1], best[3]  # origin, dest, hops (kept for the existing print statement)


def run_eps_sensitivity_analysis(graph, headway_map, trips_df, city,
                                  draw_levels=(10, 20, 40, 80)):
    """Sweeps EPS's n_draws over the SAME universal-choice-set trips_df
    generated earlier in this run (passed in directly, never reloaded from
    disk) - isolates the effect of draw count from the effect of choice-set
    construction, which is the whole point of this script."""
    print("\n" + "=" * 90)
    print(f"RUNNING EPS DRAW SENSITIVITY SWEEP ({city.upper()}, universal choice set) - "
          f"draws={list(draw_levels)}")
    print("=" * 90)

    true_params = config.TRUE_UTILITY_PARAMS
    train_df, test_df = synthetic_trips.train_test_split(trips_df, test_fraction=0.2)

    avg_dgp_set_size = (
        train_df["choice_set_size"].mean() if "choice_set_size" in train_df.columns else np.nan
    )

    results_rows = []
    for draws in draw_levels:
        print(f"\n  [Draws = {draws}] Fitting EPS...")
        eps_model = EPSModel(graph, headway_map)

        start_time = time.time()
        fitted = eps_model.fit(train_df, n_draws=draws)
        elapsed = time.time() - start_time

        test_ll = eps_model.log_likelihood(test_df, fitted, n_draws=draws)

        # reuse fit()'s own estimation dataframe instead of rebuilding it a
        # second time just to compute the average sampled choice-set size
        est_df_train = eps_model.build_estimation_data(train_df, n_draws=draws)
        avg_sampled_choice_set_size = est_df_train.groupby("obs_id").size().mean()

        row = {
            "Draws (R)": draws,
            "DGP Choice Set Size": round(avg_dgp_set_size, 2) if pd.notna(avg_dgp_set_size) else "N/A",
            "Sampled Choice Set Size": round(avg_sampled_choice_set_size, 2),
            "Obs Retained": fitted["n_observations"],
            "Train LL": round(fitted["log_likelihood"], 2),
            "Test LL": round(test_ll, 2) if pd.notna(test_ll) else "N/A",
            "Run-time (s)": round(elapsed, 1),
        }
        for p in ["beta_ivt", "beta_wait", "beta_walk", "beta_transfer", "beta_pathsize"]:
            row[f"Est {p}"] = round(fitted.get(p, np.nan), 4)
        for p in ["beta_ivt", "beta_wait", "beta_walk", "beta_transfer", "beta_pathsize"]:
            row[f"Bias {p}"] = round(fitted.get(p, 0.0) - true_params.get(p, 0.0), 4)
        for p in ["beta_ivt", "beta_wait", "beta_walk", "beta_transfer", "beta_pathsize"]:
            row[f"z {p}"] = (
                round(fitted[f"z_{p}"], 3)
                if f"z_{p}" in fitted and pd.notna(fitted[f"z_{p}"])
                else "N/A"
            )

        results_rows.append(row)

    results_df = pd.DataFrame(results_rows)

    table_path = config.TABLES_OUT_DIR / f"{city}_eps_draw_sensitivity_universal.csv"
    results_df.to_csv(table_path, index=False)

    print("\n" + "-" * 90)
    print(f"SENSITIVITY COMPARISON TABLE FOR {city.upper()} (universal choice set)")
    print(f"Full results saved to: {table_path}")
    print("-" * 90)

    summary_cols = (
        ["Draws (R)", "DGP Choice Set Size", "Sampled Choice Set Size", "Obs Retained",
         "Train LL", "Test LL", "Run-time (s)"]
        + [f"Est {p}" for p in ["beta_ivt", "beta_wait", "beta_walk", "beta_transfer", "beta_pathsize"]]
    )
    print(results_df[summary_cols].to_string(index=False))

    print("\nEstimated Parameter Biases (closer to 0.0 is better):")
    bias_cols = ["Draws (R)"] + [
        f"Bias {p}" for p in ["beta_ivt", "beta_wait", "beta_walk", "beta_transfer", "beta_pathsize"]
    ]
    print(results_df[bias_cols].to_string(index=False))

    print("\nz-stats (estimate / SE, from a numerical Hessian at each draw level's optimum):")
    z_cols = ["Draws (R)"] + [
        f"z {p}" for p in ["beta_ivt", "beta_wait", "beta_walk", "beta_transfer", "beta_pathsize"]
    ]
    print(results_df[z_cols].to_string(index=False))
    print("=" * 90)

    return results_df


def print_eps_diagnostics(eps_fitted, trips_df, candidate_paths):
    """True-vs-estimated table (all 5 params, including beta_pathsize),
    attribute variation, correlation matrix, transfer distribution, and
    path-size stats for the universal choice set."""
    print("\n" + "=" * 70)
    print("EPS DIAGNOSTICS")
    print("=" * 70)

    true_params = config.TRUE_UTILITY_PARAMS
    compare_params = ["beta_ivt", "beta_wait", "beta_walk", "beta_transfer", "beta_pathsize"]

    diagnostic_df = pd.DataFrame({
        "parameter": compare_params,
        "true_value": [true_params[k] for k in compare_params],
        "estimated_value": [eps_fitted[k] for k in compare_params],
    })
    diagnostic_df["bias"] = diagnostic_df["estimated_value"] - diagnostic_df["true_value"]
    diagnostic_df["pct_bias"] = 100 * diagnostic_df["bias"] / diagnostic_df["true_value"]

    # se_<param>/z_<param>/pval_<param> are only present if fit() was called
    # with compute_se=True (EPS defaults to True; see eps_model.py). Missing
    # gracefully as NaN rather than raising a KeyError, so this script still
    # runs unchanged against an eps_fitted dict from an older EPSModel.fit
    # call or one made with compute_se=False.
    diagnostic_df["se"] = [eps_fitted.get(f"se_{k}", np.nan) for k in compare_params]
    diagnostic_df["z"] = [eps_fitted.get(f"z_{k}", np.nan) for k in compare_params]
    diagnostic_df["pval"] = [eps_fitted.get(f"pval_{k}", np.nan) for k in compare_params]

    print("\nParameter Recovery")
    print(diagnostic_df.round(4))

    attribute_cols = ["ivt_min", "wait_min", "walk_min", "n_transfers"]
    if "path_size" in trips_df.columns:
        attribute_cols.append("path_size")

    attrs = trips_df.drop_duplicates(subset="chosen_path")[attribute_cols]

    print("\nAttribute Summary")
    print(attrs.describe().round(3))

    print("\nCorrelation Matrix")
    print(attrs.corr().round(3))

    print("\nTransfer Distribution")
    print(attrs["n_transfers"].value_counts().sort_index())

    print("\nChoice Set Information")
    print(f"Universal alternatives : {len(candidate_paths)}")
    print(f"Chosen at least once   : {len(attrs)}")

    if "path_size" in attrs.columns:
        print("\nPath Size Statistics")
        print(attrs["path_size"].describe().round(3))

    print("=" * 70)


def run_city_eps(city: str, args):
    """Phase 1: network/data setup + EPS fitting/diagnostics/sweep for one
    city. Returns the state run_city_nrl's phase 2 needs (graph,
    headway_map, train_df) so NRL doesn't reload or regenerate anything -
    it fits against the EXACT SAME universal-choice-set trips_df EPS just
    used, just a smaller subsample of rows (see --nrl-sample-size)."""
    print("\n" + "#" * 70)
    print(f"# {city.upper()} - EPS")
    print("#" * 70)

    out_dir = config.processed_dir(city)
    stations_df = pd.read_csv(out_dir / "stations.csv")
    edges_df = pd.read_csv(out_dir / "supernetwork_edges.csv")
    graph = supernetwork.to_networkx(edges_df)

    headway_df = pd.read_csv(out_dir / "route_headways.csv")
    headway_map = dict(zip(headway_df["route_id"], headway_df["headway_min"]))
    headway_map.update(supernetwork.load_bus_headways(city))
    transit_route_ids = edges_df.loc[edges_df["link_type"] == "transit", "route_id"].unique()
    for route_id in transit_route_ids:
        if route_id not in headway_map:
            headway_map[route_id] = config.DEFAULT_HEADWAY_MIN

    if args.origin and args.dest:
        origin, dest = args.origin, args.dest
    else:
        print("No --origin/--dest given - searching for a good OD pair (richest tractable |U|)...")
        origin, dest, hops = pick_a_good_od_pair(graph, max_paths=args.max_paths)
        print(f"Picked OD pair {origin} -> {dest} ({hops} hops, unweighted)")

    trips_df, candidate_paths, systematic_utilities = synthetic_trips.generate_synthetic_trips_universal(
        graph, origin, dest, headway_map,
        n_obs=args.n_observations, max_paths=args.max_paths)

    print("\nAttribute variation across the universal choice set (this is the thing to check):")
    attrs = trips_df.drop_duplicates(subset="chosen_path")[["ivt_min", "wait_min", "walk_min", "n_transfers"]]
    print(f"({len(attrs)}/{len(candidate_paths)} distinct alternatives were chosen at least once)")
    print(attrs.describe())

    train_df, test_df = synthetic_trips.train_test_split(trips_df, test_fraction=0.2)

    print("\nFitting EPS on universal-choice-set data...")
    eps_model = EPSModel(graph, headway_map)
    eps_fitted, eps_train_time = _timed(eps_model.fit, train_df, n_draws=50)
    print(f"EPS train fit took {eps_train_time:.1f}s")
    print("EPS fitted:", eps_fitted)
    print(metrics.parameter_bias(eps_fitted))

    print("\nEvaluating EPS on held-out test_df (same n_draws=50 as training fit)...")
    eps_test_ll, eps_test_time = _timed(eps_model.log_likelihood, test_df, eps_fitted, n_draws=50)
    print(f"EPS test log-likelihood: {eps_test_ll:.2f} (took {eps_test_time:.1f}s)")
    eps_timing = {
        "train_s": eps_train_time,
        "test_s": eps_test_time,
        "total_s": eps_train_time + eps_test_time,
    }

    print_eps_diagnostics(eps_fitted, trips_df, candidate_paths)

    if not args.skip_sweep:
        run_eps_sensitivity_analysis(graph, headway_map, trips_df, city)

    return {
        "graph": graph, "headway_map": headway_map,
        "train_df": train_df, "test_df": test_df, "trips_df": trips_df,
        "origin": origin, "dest": dest,
        "eps_timing": eps_timing,
    }


def run_city_nrl(city: str, args, state):
    """Phase 2: NRL fitting for one city, reusing the graph/headway_map/
    train_df from that city's run_city_eps call - no reloading from disk,
    no regenerating the universal-choice-set data. Fits on a
    --nrl-sample-size subsample of train_df, since NRL's fit() cost scales
    with sample size (and runs the objective twice - warm-start +
    full-batch, each evaluation re-solving the Bellman value-iteration
    recursion) in a way EPS's sampling-based estimation doesn't. Returns a
    timing dict ({'train_s', 'test_s', 'total_s', 'skipped'}) that main()
    aggregates into the final cross-city/cross-model runtime summary."""
    if args.skip_nrl:
        print(f"\n--skip-nrl set: skipping NRL fitting for {city}")
        return {"train_s": None, "test_s": None, "total_s": None, "skipped": True}

    print("\n" + "#" * 70)
    print(f"# {city.upper()} - NRL")
    print("#" * 70)

    graph, headway_map, train_df, test_df = (
        state["graph"], state["headway_map"], state["train_df"], state["test_df"]
    )

    nrl_n = min(args.nrl_sample_size, len(train_df))
    nrl_train_df = train_df.sample(n=nrl_n, random_state=config.RANDOM_SEED).reset_index(drop=True)

    if nrl_n < len(train_df):
        print(f"NRL: using a {nrl_n}/{len(train_df)}-observation subsample of train_df "
              f"(--nrl-sample-size={args.nrl_sample_size}) - NOT the full universal-choice-set "
              f"training data EPS just used above.")
    if nrl_n > 100:
        print(f"WARNING: --nrl-sample-size={nrl_n} is well above the ~60-observation scale "
              f"benchmarked to reliably finish in reasonable time with L-BFGS-B (see the "
              f"NRL optimizer-cost discussion this script's history is built on). A 300-"
              f"observation 'full'-spec fit did NOT finish within a comparable test window "
              f"in that benchmark - expect this to potentially take many minutes to fit per "
              f"city, not seconds. Re-run with a smaller --nrl-sample-size if this stalls "
              f"longer than you're willing to wait.")

    print("\nFitting NRL on universal-choice-set data...")
    nrl_model = NRLModel(graph, headway_map)
    nrl_fitted, nrl_train_time = _timed(nrl_model.fit, nrl_train_df, compute_se=args.nrl_se)
    print(f"NRL train fit took {nrl_train_time:.1f}s")
    print("NRL fitted:", nrl_fitted)
    print(metrics.parameter_bias(nrl_fitted))

    # NRL.log_likelihood re-solves the Bellman recursion per path just like
    # fit() does, so test-set cost scales with sample size the same way
    # training does - subsample test_df to the same --nrl-sample-size cap
    # rather than evaluating on the (potentially much larger) full test_df,
    # or this "test" step could quietly cost far more than training did.
    nrl_test_n = min(args.nrl_sample_size, len(test_df))
    nrl_test_df = test_df.sample(n=nrl_test_n, random_state=config.RANDOM_SEED).reset_index(drop=True)
    if nrl_test_n < len(test_df):
        print(f"NRL: evaluating on a {nrl_test_n}/{len(test_df)}-observation subsample of "
              f"test_df, capped the same way as --nrl-sample-size, for the same reason.")

    print("\nEvaluating NRL on held-out (subsampled) test data...")
    nrl_test_ll, nrl_test_time = _timed(nrl_model.log_likelihood, nrl_test_df, nrl_fitted)
    print(f"NRL test log-likelihood: {nrl_test_ll:.2f} (took {nrl_test_time:.1f}s)")

    return {
        "train_s": nrl_train_time,
        "test_s": nrl_test_time,
        "total_s": nrl_train_time + nrl_test_time,
        "skipped": False,
    }


def print_runtime_summary(timing_rows):
    """Consolidated train/test/total runtime table across every
    (city, model) combination run this invocation. `timing_rows` is a list
    of dicts with keys city, model, train_s, test_s, total_s - any run that
    was skipped (--skip-nrl) or never reached (an exception in an earlier
    city) has train_s/test_s/total_s left as None and prints as 'N/A'
    rather than being silently dropped from the table, so a skipped run is
    still visible as skipped rather than looking like it never happened.
    """
    print("\n" + "=" * 90)
    print("RUNTIME SUMMARY (train / test / total, seconds)")
    print("=" * 90)

    df = pd.DataFrame(timing_rows)
    display_df = df.copy()
    for col in ["train_s", "test_s", "total_s"]:
        display_df[col] = display_df[col].apply(lambda v: round(v, 1) if pd.notna(v) else "N/A")
    print(display_df[["city", "model", "train_s", "test_s", "total_s"]].to_string(index=False))

    numeric_total = df["total_s"].dropna().sum()
    print(f"\nGrand total across all runs shown above: {numeric_total:.1f}s "
          f"(excludes any 'N/A'/skipped rows)")

    table_path = config.TABLES_OUT_DIR / "runtime_summary.csv"
    df.to_csv(table_path, index=False)
    print(f"Full runtime table saved to: {table_path}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["sydney", "bengaluru", "both"],
                         help="'both' runs sydney then bengaluru in one invocation (matches "
                              "main.py's --city convention). Cannot be combined with "
                              "--origin/--dest, since those are city-specific station IDs and "
                              "it would be ambiguous which city they apply to - pass a single "
                              "city if you need to pin a specific OD pair.")
    parser.add_argument("--origin", default=None)
    parser.add_argument("--dest", default=None)
    parser.add_argument("--n-observations", type=int, default=3000)
    parser.add_argument("--max-paths", type=int, default=500,
                         help="Safety cutoff for full simple-path enumeration - "
                              "generate_synthetic_trips_universal raises if |U| exceeds this "
                              "before finishing, rather than silently truncating.")
    parser.add_argument("--skip-sweep", action="store_true",
                         help="Skip the EPS draw-count sensitivity sweep (it refits EPS 4x - "
                              "slow on long paths/large n_observations, see the draws-vs-time discussion)")
    parser.add_argument("--skip-nrl", action="store_true",
                         help="Skip NRL fitting (slow - see the runtime discussion for this script)")
    parser.add_argument("--nrl-sample-size", type=int, default=60,
                         help="NRL's fit() cost scales with sample size AND runs the objective "
                              "twice (warm-start + full-batch), each evaluation re-solving the "
                              "Bellman value-iteration recursion - it does NOT scale the way EPS "
                              "does, so NRL is fit on its own, smaller subsample of train_df "
                              "rather than the full universal-choice-set n_observations. Default "
                              "of 60 matches what was benchmarked to actually finish in reasonable "
                              "time with L-BFGS-B on a comparable choice-set size; raise this only "
                              "if you've confirmed the runtime is acceptable for your case first.")
    parser.add_argument("--nrl-se", action="store_true",
                         help="Also compute standard errors/z-stats for NRL via a numerical "
                              "Hessian (see NRLModel.fit's compute_se docstring). Off by default: "
                              "for 6 parameters this re-solves the Bellman recursion ~84 more "
                              "times, roughly doubling this phase's runtime on top of an already-"
                              "slow fit. EPS gets its z-stats unconditionally since its NLL is cheap.")
    args = parser.parse_args()

    if args.city == "both" and (args.origin or args.dest):
        parser.error("--origin/--dest cannot be combined with --city both - station IDs are "
                      "city-specific, so it's ambiguous which city they'd apply to. Run each "
                      "city separately (--city sydney / --city bengaluru) if you need to pin "
                      "a specific OD pair.")

    print("=" * 70)
    print("VALIDATE_OD_DIAGNOSTIC.PY - universal choice set test")
    print("(if you don't see this banner, you're running the wrong script)")
    print("=" * 70)

    cities = ["sydney", "bengaluru"] if args.city == "both" else [args.city]
    timing_rows = []
    for city in cities:
        state = run_city_eps(city, args)
        nrl_timing = run_city_nrl(city, args, state)

        timing_rows.append({"city": city, "model": "EPS", **state["eps_timing"]})
        timing_rows.append({
            "city": city, "model": "NRL",
            "train_s": nrl_timing["train_s"],
            "test_s": nrl_timing["test_s"],
            "total_s": nrl_timing["total_s"],
        })

    print_runtime_summary(timing_rows)


if __name__ == "__main__":
    main()
