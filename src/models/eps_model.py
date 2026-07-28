import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
from tqdm import tqdm  # Imported tqdm

from src import config
from src.models import path_utils

START = "START"

class EPSModel:
    def __init__(self, graph: nx.DiGraph, headway_map: dict, beta: dict = None,
                 kumaraswamy_a=2.0, kumaraswamy_b=2.0, max_steps=25):
        self.G = graph
        self.headway_map = headway_map
        self.beta = beta or dict(config.TRUE_UTILITY_PARAMS)
        self.kumaraswamy_a = kumaraswamy_a
        self.kumaraswamy_b = kumaraswamy_b
        self.max_steps = max_steps
        self._remaining_time_cache = {}

    def _remaining_time(self, node, dest):
        key = (node, dest)
        if key not in self._remaining_time_cache:
            try:
                self._remaining_time_cache[key] = nx.shortest_path_length(self.G, node, dest, weight="duration")
            except nx.NetworkXNoPath:
                self._remaining_time_cache[key] = np.inf
        return self._remaining_time_cache[key]

    def _link_probs(self, current, dest, successors):
        """Eq. (8)-(9): probability distribution over `successors` (already
        filtered to the walk's own valid-next-node set) leaving `current`
        toward `dest`. Shared by the stochastic sampler and the deterministic
        path-probability evaluator below, so both use IDENTICAL weights -
        needed for q(i) computed after the fact to be the true probability
        under the same protocol that generates paths in the first place.
        Returns None if `current` can't reach `dest` (sp_v invalid).
        """
        sp_v = self._remaining_time(current, dest)
        if not np.isfinite(sp_v) or sp_v <= 0:
            return None

        # Convert sp_v from seconds to minutes
        sp_v_min = sp_v / 60

        f_vals = []
        for s in successors:
            link_time = self.G.edges[current, s]["duration"] / 60  # Minutes
            sp_s = self._remaining_time(s, dest) / 60              # Convert to Minutes

            cond_time = link_time + sp_s
            if not np.isfinite(cond_time) or cond_time <= 0:
                x_l = 0.0
            else:
                x_l = np.clip(sp_v_min / cond_time, 0.0, 1.0)      # Ratio of minutes/minutes

            f_val = 1.0 - (1.0 - x_l ** self.kumaraswamy_a) ** self.kumaraswamy_b
            f_vals.append(max(f_val, 1e-12))

        f_vals = np.array(f_vals)
        return f_vals / f_vals.sum()

    def _sample_one_path(self, origin, dest, rng):
        path = [origin]
        log_q = 0.0
        current = origin
        visited = {origin}

        for _ in range(self.max_steps):
            if current == dest:
                return path, log_q
            successors = list(self.G.successors(current))
            successors = [s for s in successors if s not in visited or s == dest]
            if not successors:
                return None, None

            probs = self._link_probs(current, dest, successors)
            if probs is None:
                return None, None

            choice_idx = rng.choice(len(successors), p=probs)
            chosen = successors[choice_idx]
            log_q += np.log(probs[choice_idx])

            path.append(chosen)
            visited.add(chosen)
            current = chosen

        return None, None

    def _path_log_prob(self, path):
        """Deterministic log q(path) under the SAME biased-random-walk
        protocol as _sample_one_path (Eq. 9-10), but evaluating the exact
        probability of each link the path actually uses instead of drawing
        one at random. Used for observed/chosen paths the random walk
        didn't happen to generate within n_draws attempts - without this,
        force_include_path was assigning those paths an arbitrary
        placeholder q instead of their true sampling probability, which
        corrupts the ln(k_in/q(i)) correction (and, via the shared
        Expanded-Path-Size denominator, every other path in that choice
        set too).

        Returns None if the path isn't executable under the walk's own
        visited-node rule (e.g. it revisits a node) or hits an
        unreachable/degenerate step - signals the caller should fall back
        to a floor probability instead, since q(i) is then genuinely ~0
        under this protocol, not just a computation we skipped.
        """
        dest = path[-1]
        visited = {path[0]}
        log_q = 0.0

        for current, chosen in zip(path[:-1], path[1:]):
            successors = [s for s in self.G.successors(current) if s not in visited or s == dest]
            if chosen not in successors:
                return None

            probs = self._link_probs(current, dest, successors)
            if probs is None:
                return None

            log_q += np.log(probs[successors.index(chosen)])
            visited.add(chosen)

        return log_q

    def sample_choice_set(self, origin, dest, n_draws=20, rng=None, force_include_path=None):
        rng = rng or np.random.default_rng(config.RANDOM_SEED)
        drawn = {}

        for _ in range(n_draws):
            path, log_q = self._sample_one_path(origin, dest, rng)
            if path is None:
                continue
            key = tuple(path)
            if key not in drawn:
                drawn[key] = {"n_sampled": 0, "log_q": log_q}
            drawn[key]["n_sampled"] += 1

        if force_include_path is not None:
            key = tuple(force_include_path)
            if key not in drawn:
                log_q = self._path_log_prob(list(force_include_path))
                if log_q is None:
                    # The walk's own visited-node rule would never generate this
                    # path at all (e.g. it revisits a station) - q(i) is
                    # genuinely ~0 under this protocol. Use a floor rather than
                    # a fabricated value so ln(k_in/q(i)) stays finite, and flag
                    # it since this means the observed path is structurally
                    # incompatible with the sampling protocol, not just unlucky.
                    print(f"WARNING: observed path {key} isn't reachable under the "
                          f"biased random walk's own successor rule - using a floor "
                          f"probability (1e-6) instead of its true q(i).")
                    log_q = np.log(1e-6)
                drawn[key] = {"n_sampled": 1, "log_q": log_q}

        return drawn

    def _path_records(self, choice_set: dict, n_draws: int):
        paths = list(choice_set.keys())
        paths_link_sets, path_lengths = [], []

        for path in paths:
            edge_seq = [self.G.edges[u, v] for u, v in zip(path[:-1], path[1:])]
            total_dur = sum(e["duration"] for e in edge_seq) or 1e-6
            link_set = [(u, v, e["duration"]) for (u, v), e in zip(zip(path[:-1], path[1:]), edge_seq)]
            paths_link_sets.append(link_set)
            path_lengths.append(total_dur)

        phi = {}
        for path in paths:
            info = choice_set[path]
            q_path = np.exp(info["log_q"])
            phi[path] = info["n_sampled"] / (n_draws * q_path)

        # PRECOMPUTE: Extract link node sets once per path outside the loop
        paths_link_node_sets = [
            {(u_j, v_j) for u_j, v_j, _ in link_set} 
            for link_set in paths_link_sets
        ]

        eps_values = []
        for i, path_i in enumerate(paths):
            links_i = paths_link_sets[i]
            length_i = path_lengths[i]
            
            eps_i = 0.0
            for u, v, length_a in links_i:
                denom = 0.0
                for j, path_j in enumerate(paths):
                    # Fast O(1) set membership test with zero inner allocations
                    if (u, v) in paths_link_node_sets[j]:
                        denom += phi[path_j]
                
                if denom > 0:
                    eps_i += (length_a / length_i) * (1.0 / denom)
            eps_values.append(eps_i)

        records = []
        for i, path in enumerate(paths):
            info = choice_set[path]
            edge_seq = [self.G.edges[u, v] for u, v in zip(path[:-1], path[1:])]
            ivt, wait, walk, n_transfers = path_utils.compute_path_components(edge_seq, self.headway_map)
            
            correction = np.log(max(info["n_sampled"], 1) / (n_draws * np.exp(info["log_q"])))
            records.append({
                "path": path, "ivt_min": ivt, "wait_min": wait, "walk_min": walk,
                "n_transfers": n_transfers, "expanded_path_size": eps_values[i], "sampling_correction": correction,
            })
        return records
    
    def build_estimation_data(self, trips_df: pd.DataFrame, n_draws=20, path_col="chosen_path", sep="|"):
        rng = np.random.default_rng(config.RANDOM_SEED)
        rows = []
        # Added tqdm progress bar for choice set generation
        for i, row in tqdm(trips_df.iterrows(), total=len(trips_df), desc="Sampling Choice Sets (EPS)"):
            origin, dest = row["origin_station_id"], row["dest_station_id"]
            observed_path = None
            if isinstance(row[path_col], str):
                observed_path = tuple(path_utils.cast_path(row[path_col].split(sep), self.G))
                
            choice_set = self.sample_choice_set(origin, dest, n_draws=n_draws, rng=rng,
                                                 force_include_path=observed_path)
            records = self._path_records(choice_set, n_draws)
            for rec in records:
                rows.append({
                    "obs_id": i,
                    "path": rec["path"],
                    "chosen": int(rec["path"] == observed_path),
                    "ivt_min": rec["ivt_min"], "wait_min": rec["wait_min"],
                    "walk_min": rec["walk_min"], "n_transfers": rec["n_transfers"],
                    "expanded_path_size": rec["expanded_path_size"], "sampling_correction": rec["sampling_correction"],
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _neg_log_likelihood(params, est_df):
        if est_df.empty:
            # No informative rows to evaluate - e.g. called on an empty
            # trips_df, or every observation got filtered out upstream.
            # Returning 0.0 (rather than crashing on a KeyError from a
            # columnless empty dataframe) makes this safe to call from
            # scipy.optimize.minimize during fit() without special-casing
            # every call site; log_likelihood() below is the one that
            # decides whether an empty result is actually meaningful and
            # surfaces a clear warning + NaN instead of a silent 0.0.
            return 0.0
        b_ivt, b_wait, b_walk, b_transfer, b_ps = params
        # b_ivt, b_ps = params
        util = (
            b_ivt * est_df["ivt_min"] 
            + b_wait * est_df["wait_min"]
            + b_walk * est_df["walk_min"] 
            + b_transfer * est_df["n_transfers"]
            + b_ps * np.log(est_df["expanded_path_size"].clip(lower=1e-9))
            + est_df["sampling_correction"]  # Mathematically exact offset
        )
        est_df = est_df.assign(util=util)
        ll = 0.0
        for _, grp in est_df.groupby("obs_id"):
            u = grp["util"].values
            p = np.exp(u - u.max())
            p = p / p.sum()
            chosen_idx = np.argmax(grp["chosen"].values)
            if grp["chosen"].sum() == 0:
                continue
            ll += np.log(max(p[chosen_idx], 1e-12))
        return -ll

    def fit(self, trips_df: pd.DataFrame, n_draws=20, x0=None,
            min_choice_set_size=2, coef_bound=5.0):
        """Estimates beta_ivt, beta_wait, beta_walk, beta_transfer, beta_pathsize
        by maximum likelihood on the sampled + observed choice sets using Scipy's L-BFGS-B.

        min_choice_set_size: observations whose sampled choice set (after
        forcing the observed path in) ends up with fewer than this many
        distinct paths are dropped before estimation.

        coef_bound: box constraint applied to all coefficients during
        optimization (L-BFGS-B) - keeps the optimizer in a sane region.
        """
        est_df = self.build_estimation_data(trips_df, n_draws=n_draws)

        informative_obs = est_df.groupby("obs_id").size()
        informative_obs = informative_obs[informative_obs >= min_choice_set_size].index
        n_dropped = est_df["obs_id"].nunique() - len(informative_obs)
        if n_dropped:
            print(f"EPS.fit: dropping {n_dropped} observations with a degenerate "
                  f"(<{min_choice_set_size}-alternative) choice set - uninformative for estimation.")
        est_df = est_df[est_df["obs_id"].isin(informative_obs)]

        if est_df.empty:
            raise ValueError("No informative observations left after filtering degenerate choice sets.")

        x0 = x0 or [self.beta["beta_ivt"], self.beta["beta_wait"], self.beta["beta_walk"],
                     self.beta["beta_transfer"], self.beta["beta_pathsize"]]

        # x0 = x0 or [self.beta["beta_ivt"], self.beta["beta_pathsize"]]

        bounds = [(-coef_bound, coef_bound)] * len(x0)
        result = minimize(self._neg_log_likelihood, x0=x0, args=(est_df,), method="L-BFGS-B", bounds=bounds)
        names = ["beta_ivt", "beta_wait", "beta_walk", "beta_transfer", "beta_pathsize"]
        # names = ["beta_ivt", "beta_pathsize"]
        fitted = dict(zip(names, result.x))
        fitted["log_likelihood"] = -result.fun
        fitted["n_observations"] = est_df["obs_id"].nunique()
        fitted["converged"] = result.success
        self.fitted_params = fitted
        return fitted

    def log_likelihood(self, trips_df: pd.DataFrame, beta: dict, n_draws=20):
        if trips_df.empty:
            print("WARNING: log_likelihood called with an empty trips_df (0 rows) - "
                  "returning NaN. This usually means your test split came out empty; "
                  "check train_test_split's per-OD-pair rounding if most of your OD "
                  "pairs have very few trips (see synthetic_trips.train_test_split - "
                  "naive rounding sends small groups' test allocation to 0).")
            return np.nan
        est_df = self.build_estimation_data(trips_df, n_draws=n_draws)
        if est_df.empty:
            print("WARNING: build_estimation_data produced no informative rows "
                  "(e.g. every path in trips_df was unreachable in the current graph) "
                  "- returning NaN.")
            return np.nan
        params = [beta["beta_ivt"], beta["beta_wait"], beta["beta_walk"],
                  beta["beta_transfer"], beta["beta_pathsize"]]
        # params = [beta["beta_ivt"], beta["beta_pathsize"]]
        return -self._neg_log_likelihood(params, est_df)