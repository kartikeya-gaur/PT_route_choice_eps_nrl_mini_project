"""
src.evaluation.metrics

Validation metrics for comparing EPS and NRL against the held-out test
split and against the TRUE synthetic-generation parameters.
"""

import numpy as np
import pandas as pd

from src import config


def rho_squared(ll_model, ll_null):
    """McFadden's rho-squared: 1 - LL(model)/LL(null). ll_null should be the
    log-likelihood of a naive equal-probability model over each choice set
    (ln(1/choice_set_size) summed over observations) - NOT a constants-only
    model, since that requires re-estimating; equal-probability is the
    standard cheap null for route-choice applications."""
    return 1 - (ll_model / ll_null)


def null_log_likelihood(trips_df: pd.DataFrame, choice_set_size_col="choice_set_size"):
    """LL of an equal-probability model: sum of ln(1/choice_set_size) over
    all observations. Requires the choice_set_size column produced by the
    synthetic trip generator (or equivalent)."""
    sizes = trips_df[choice_set_size_col].clip(lower=1)
    return -np.log(sizes).sum()


def out_of_sample_log_likelihood(model, test_df: pd.DataFrame, fitted_beta: dict):
    """Evaluates a fitted model's log-likelihood on held-out test trips.
    `model` is an EPSModel or NRLModel instance; fitted_beta should include
    beta_pathsize=0.0 for NRL (no path-size term in that model)."""
    return model.log_likelihood(test_df, fitted_beta)


def parameter_bias(fitted_params: dict, true_params: dict = None):
    """Signed and absolute bias of each estimated coefficient vs. the TRUE
    synthetic-generation parameters (src.config.TRUE_UTILITY_PARAMS)."""
    true_params = true_params or config.TRUE_UTILITY_PARAMS
    rows = []
    for name, true_val in true_params.items():
        if name not in fitted_params:
            continue
        est_val = fitted_params[name]
        rows.append({
            "parameter": name, "true_value": true_val, "estimated_value": est_val,
            "bias": est_val - true_val,
            "pct_bias": 100 * (est_val - true_val) / true_val if true_val != 0 else np.nan,
        })
    return pd.DataFrame(rows)


def hit_rate(model, test_df: pd.DataFrame, fitted_beta: dict, path_col="chosen_path", sep="|"):
    """Fraction of test trips where the model's single most-likely predicted
    path matches the observed path. For EPS this re-samples a choice set per
    trip (stochastic); for NRL this requires enumerating/ranking candidate
    paths, which src.models.nrl_model doesn't do directly - this metric is
    most meaningful for EPS, or for NRL if you separately supply a
    candidate-path enumeration.
    """
    correct = 0
    total = 0
    for _, row in test_df.iterrows():
        observed = tuple(row[path_col].split(sep)) if isinstance(row[path_col], str) else tuple(row[path_col])
        origin, dest = row["origin_station_id"], row["dest_station_id"]
        try:
            choice_set = model.sample_choice_set(origin, dest, n_draws=30)
        except AttributeError:
            continue  # model doesn't support sample_choice_set (e.g. NRL) - skip
        if not choice_set:
            continue
        records = model._path_records(choice_set, 30)
        utils = [
            -fitted_beta["beta_ivt"] * r["ivt_min"] - fitted_beta["beta_wait"] * r["wait_min"]
            - fitted_beta["beta_walk"] * r["walk_min"] - fitted_beta["beta_transfer"] * r["n_transfers"]
            + fitted_beta["beta_pathsize"] * np.log(max(r["path_size"], 1e-9))
            for r in records
        ]
        best = records[int(np.argmax(utils))]["path"]
        correct += int(best == observed)
        total += 1
    return correct / total if total else np.nan


def summarize_comparison(eps_results: dict, nrl_results: dict) -> pd.DataFrame:
    rows = []
    for name in ["beta_ivt", "beta_wait", "beta_walk", "beta_transfer", "beta_pathsize",
                 "log_likelihood", "n_observations"]:
        rows.append({
            "metric": name,
            "EPS": eps_results.get(name),
            "NRL": nrl_results.get(name),   # correctly shows blank/NaN for NRL, which has no path-size term
            "true_value": config.TRUE_UTILITY_PARAMS.get(name),
        })
    return pd.DataFrame(rows)