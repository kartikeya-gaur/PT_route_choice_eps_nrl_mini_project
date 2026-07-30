import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
from tqdm import tqdm  # Imported tqdm
from src.models import path_utils

START = "START"

class NRLModel:
    def __init__(self, graph: nx.DiGraph, headway_map: dict, beta: dict = None,
                 max_iter=300, tol=1e-3):
        self.G = graph
        self.headway_map = headway_map
        default_beta = {
            "beta_ivt": 1.0,
            "beta_wait": 1.0,
            "beta_walk": 1.0,
            "beta_transfer": 0.5,
            "beta_scale_transfer": 0.0,
            "beta_scale_regular": 0.0
        }
        if beta:
            default_beta.update(beta)
        self.beta = default_beta
        
        self.max_iter = max_iter
        self.tol = tol
        self._all_links = list(graph.edges())
        
        self.links = self._all_links + [(START, n) for n in self.G.nodes()]
        self.link_to_idx = {link: i for i, link in enumerate(self.links)}
        
        self._current_beta_tuple = None
        self.succ_indices = []
        self.succ_utilities = []
        self.mu = None

    def _link_attrs(self, link):
        if link[0] == START:
            return {"route_id": None, "link_type": "start", "duration": 0.0}
        return self.G.edges[link]

    def _successors(self, link):
        if link[0] == START:
            node = link[1]
            return [(node, w) for w in self.G.successors(node)]
        _, v = link
        return [(v, w) for w in self.G.successors(v)]

    def _head_node(self, link):
        return link[1]

    def _instantaneous_utility(self, prev_link, next_link, beta):
        prev_attrs = self._link_attrs(prev_link)
        next_attrs = self._link_attrs(next_link)

        if next_attrs["link_type"] == "transfer":
            walk_min = next_attrs["duration"] / 60
            return -beta["beta_walk"] * walk_min

        ivt_min = next_attrs["duration"] / 60
        is_boarding = (
            prev_attrs["link_type"] in ("start", "transfer")
            or prev_attrs["route_id"] != next_attrs["route_id"]
        )
        if not is_boarding:
            return -beta["beta_ivt"] * ivt_min

        headway = self.headway_map.get(next_attrs["route_id"], 10.0)
        wait_min = headway / 2
        is_transfer = prev_attrs["link_type"] != "start"
        transfer_term = beta["beta_transfer"] if is_transfer else 0.0
        return -beta["beta_ivt"] * ivt_min - beta["beta_wait"] * wait_min - transfer_term

    def _get_scale(self, link, beta):
        attrs = self._link_attrs(link)
        if attrs["link_type"] == "start":
            return 1.0
            
        if attrs["link_type"] == "transfer":
            scale_param = beta.get("beta_scale_transfer", 0.0)
        else:
            scale_param = beta.get("beta_scale_regular", 0.0)
            
        mu = 1.0 / (1.0 + np.exp(-scale_param))
        return np.clip(mu, 1e-4, 1.0)

    def _precompute_transitions(self, beta):
        N = len(self.links)
        self.succ_indices = []
        self.succ_utilities = []
        self.mu = np.empty(N)
        
        for i, link_i in enumerate(self.links):
            self.mu[i] = self._get_scale(link_i, beta)
            
            successors = self._successors(link_i)
            indices = []
            utilities = []
            for link_j in successors:
                if link_j in self.link_to_idx:
                    indices.append(self.link_to_idx[link_j])
                    utilities.append(self._instantaneous_utility(link_i, link_j, beta))
                    
            self.succ_indices.append(np.array(indices, dtype=np.int32))
            self.succ_utilities.append(np.array(utilities, dtype=np.float64))
            
        self._current_beta_tuple = tuple(sorted(beta.items()))

    def _solve_value_functions_nrl(self, dest, beta):
        beta_tuple = tuple(sorted(beta.items()))
        if self._current_beta_tuple != beta_tuple:
            self._precompute_transitions(beta)
            
        N = len(self.links)
        dest_indices = [i for i, link in enumerate(self.links) if self._head_node(link) == dest]
        dest_set = set(dest_indices)
        non_dest_indices = [i for i in range(N) if i not in dest_set]
        
        V = np.zeros(N)
        V[non_dest_indices] = -10.0
        
        converged = False
        for _ in range(self.max_iter):
            V_new = np.zeros(N)
            
            for i in non_dest_indices:
                succs = self.succ_indices[i]
                if len(succs) == 0:
                    V_new[i] = -1e6
                    continue
                    
                mu_i = self.mu[i]
                terms = (self.succ_utilities[i] + V[succs]) / mu_i
                
                max_term = np.max(terms)
                V_new[i] = mu_i * (max_term + np.log(np.sum(np.exp(terms - max_term))))
                
            max_diff = np.max(np.abs(V_new[non_dest_indices] - V[non_dest_indices]))
            V = V_new
            if max_diff < self.tol:
                converged = True
                break
                
        return {link: V[i] for link, i in self.link_to_idx.items()}

    def path_log_prob(self, path, beta):
        dest = path[-1]
        V_dict = self._solve_value_functions_nrl(dest, beta)

        links_seq = [(START, path[0])] + [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        
        log_prob = 0.0
        for i in range(len(links_seq) - 1):
            prev_link = links_seq[i]
            next_link = links_seq[i+1]
            
            prev_idx = self.link_to_idx[prev_link]
            
            u = self._instantaneous_utility(prev_link, next_link, beta)
            mu_i = self.mu[prev_idx]
            V_prev = V_dict[prev_link]
            V_next = V_dict[next_link]
            
            log_prob += (1.0 / mu_i) * (u + V_next - V_prev)
            
        return log_prob

    def log_likelihood(self, trips_df: pd.DataFrame, beta: dict, path_col="chosen_path", sep="|"):
        ll = 0.0
        # Added tqdm progress bar for batch evaluations
        for _, row in tqdm(trips_df.iterrows(), total=len(trips_df), desc="NRL Log-Likelihood"):
            raw_path = row[path_col].split(sep) if isinstance(row[path_col], str) else row[path_col]
            path = path_utils.cast_path(raw_path, self.G)
            ll += self.path_log_prob(path, beta)
        return ll

    def _cast_paths(self, df, path_col, sep):
        return [
            path_utils.cast_path(
                row[path_col].split(sep) if isinstance(row[path_col], str) else row[path_col], self.G)
            for _, row in df.iterrows()
        ]

    def _make_objective(self, paths, tag=""):
        label = f" [{tag}]" if tag else ""

        def neg_ll(params):
            b = {
                "beta_ivt": params[0], 
                "beta_wait": params[1],
                "beta_walk": params[2], 
                "beta_transfer": params[3],
                "beta_scale_transfer": params[4],
                "beta_scale_regular": params[5],
                "beta_pathsize": 0.0
            }
            # Added tqdm with leave=False to show current evaluation progress cleanly
            nll = -sum(
                self.path_log_prob(p, b) 
                for p in tqdm(paths, desc=f"Evaluating NLL{label}", leave=False)
            )
            neg_ll.n_calls += 1
            if neg_ll.n_calls % 10 == 0:
                print(f"  NRL.fit{label}: objective evaluation {neg_ll.n_calls}, "
                      f"nll={nll:.2f}, params={np.round(params, 4)}")
            return nll
        neg_ll.n_calls = 0
        return neg_ll

    def fit(self, trips_df: pd.DataFrame, x0=None, path_col="chosen_path", sep="|",
            method="L-BFGS-B", coef_bound=5.0, warm_start_frac=0.25,
            warm_start_min_n=50, random_seed=None):
        x0 = x0 or [
            self.beta["beta_ivt"], 
            self.beta["beta_wait"],
            self.beta["beta_walk"], 
            self.beta["beta_transfer"],
            self.beta["beta_scale_transfer"],
            self.beta["beta_scale_regular"]
        ]
        bounds = [(-coef_bound, coef_bound)] * len(x0)

        rng = np.random.default_rng(random_seed if random_seed is not None else 42)
        n_warm = max(warm_start_min_n, int(round(len(trips_df) * warm_start_frac)))
        n_warm = min(n_warm, len(trips_df))

        if n_warm < len(trips_df):
            warm_df = trips_df.sample(n=n_warm, random_state=int(rng.integers(0, 2**31 - 1)))
            warm_paths = self._cast_paths(warm_df, path_col, sep)
            print(f"NRL.fit: phase 1 (warm start) on {len(warm_paths)}/{len(trips_df)} trips "
                  f"({len({p[-1] for p in warm_paths})} unique destinations)")
            warm_obj = self._make_objective(warm_paths, tag="warm")
            warm_result = minimize(warm_obj, x0=x0, method=method, bounds=bounds)
            x0 = warm_result.x
            print(f"NRL.fit: phase 1 complete ({warm_obj.n_calls} evaluations), "
                  f"warm-started params={np.round(x0, 4)}")
        else:
            print("NRL.fit: warm_start_frac/warm_start_min_n covers the full dataset - "
                  "skipping straight to full-batch fitting")

        paths = self._cast_paths(trips_df, path_col, sep)
        print(f"NRL.fit: phase 2 (full batch) optimizing over {len(paths)} observed paths "
              f"({len({p[-1] for p in paths})} unique destinations)")
        full_obj = self._make_objective(paths, tag="full")
        result = minimize(full_obj, x0=x0, method=method, bounds=bounds, options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 300, "maxfev": 300})

        names = [
            "beta_ivt", "beta_wait", "beta_walk", "beta_transfer", 
            "beta_scale_transfer", "beta_scale_regular"
        ]
        fitted = dict(zip(names, result.x))
        fitted["log_likelihood"] = -result.fun
        fitted["n_observations"] = len(paths)
        fitted["converged"] = result.success
        self.fitted_params = fitted
        return fitted