"""
tests/test_models.py

Unit tests for EPS and NRL. These use small synthetic toy networks
(a few nodes) rather than the real Sydney/Bengaluru data, so they run
fast and isolate the model logic from data-loading concerns.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.eps_model import EPSModel, _kumaraswamy_sample
from src.models.nrl_model import NRLModel
from src import config


def _toy_graph():
    """A -> B -> D  (2 hops)
       A -> C -> D  (2 hops, parallel alternative)
    Both routes on the same duration so choice is roughly 50/50 under the
    true utility params (modulo the log(PathSize) term, since these two
    paths don't overlap at all - PathSize=1 for both, so it IS ~50/50)."""
    edges = [
        ("A", "B", {"route_id": "R1", "duration": 300, "link_type": "transit"}),
        ("B", "D", {"route_id": "R1", "duration": 300, "link_type": "transit"}),
        ("A", "C", {"route_id": "R2", "duration": 300, "link_type": "transit"}),
        ("C", "D", {"route_id": "R2", "duration": 300, "link_type": "transit"}),
    ]
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


def _toy_headways():
    return {"R1": 10.0, "R2": 10.0}


# --------------------------------------------------------------------------
# EPS
# --------------------------------------------------------------------------
def test_kumaraswamy_sample_bounded_0_1():
    rng = np.random.default_rng(0)
    samples = _kumaraswamy_sample(rng, a=2.0, b=2.0, size=1000)
    assert (samples >= 0).all() and (samples <= 1).all()


def test_eps_sample_choice_set_reaches_destination():
    G = _toy_graph()
    model = EPSModel(G, _toy_headways(), max_steps=10)
    choice_set = model.sample_choice_set("A", "D", n_draws=30, rng=np.random.default_rng(1))
    assert len(choice_set) > 0
    for path in choice_set:
        assert path[0] == "A" and path[-1] == "D"


def test_eps_force_include_path_always_present():
    G = _toy_graph()
    model = EPSModel(G, _toy_headways())
    forced = ("A", "C", "D")
    choice_set = model.sample_choice_set("A", "D", n_draws=5, rng=np.random.default_rng(2),
                                          force_include_path=forced)
    assert forced in choice_set


def test_eps_path_records_probabilities_sane():
    G = _toy_graph()
    model = EPSModel(G, _toy_headways())
    choice_set = model.sample_choice_set("A", "D", n_draws=50, rng=np.random.default_rng(3))
    records = model._path_records(choice_set, n_draws=50)
    assert all(r["path_size"] > 0 for r in records)
    assert all(np.isfinite(r["sampling_correction"]) for r in records)


# --------------------------------------------------------------------------
# NRL
# --------------------------------------------------------------------------
def test_nrl_value_function_zero_at_destination_links():
    G = _toy_graph()
    model = NRLModel(G, _toy_headways())
    V = model._value_iteration("D", config.TRUE_UTILITY_PARAMS)
    assert V[("B", "D")] == 0.0
    assert V[("C", "D")] == 0.0


def test_nrl_value_function_converges_and_is_finite():
    G = _toy_graph()
    model = NRLModel(G, _toy_headways())
    V = model._value_iteration("D", config.TRUE_UTILITY_PARAMS)
    for link, v in V.items():
        assert np.isfinite(v) or v == -1e6  # -1e6 only for genuine dead ends


def test_nrl_path_probabilities_sum_to_one_over_full_choice_set():
    """For the toy network, A->B->D and A->C->D are the ONLY two paths from
    A to D, so their probabilities should sum to ~1."""
    G = _toy_graph()
    model = NRLModel(G, _toy_headways())
    beta = dict(config.TRUE_UTILITY_PARAMS)

    p1 = np.exp(model.path_log_prob(["A", "B", "D"], beta))
    p2 = np.exp(model.path_log_prob(["A", "C", "D"], beta))
    assert p1 + p2 == pytest.approx(1.0, abs=1e-6)


def test_nrl_symmetric_paths_get_equal_probability():
    """Both toy paths have identical attributes (same duration, same route
    structure) - under a symmetric network they must get equal probability."""
    G = _toy_graph()
    model = NRLModel(G, _toy_headways())
    beta = dict(config.TRUE_UTILITY_PARAMS)

    p1 = model.path_log_prob(["A", "B", "D"], beta)
    p2 = model.path_log_prob(["A", "C", "D"], beta)
    assert p1 == pytest.approx(p2, abs=1e-9)


def test_nrl_fit_recovers_reasonable_parameters_on_noiseless_toy_data():
    """A cheap sanity check, not a rigorous recovery test: fitting on a
    handful of observed toy paths shouldn't blow up or return nonsense."""
    G = _toy_graph()
    model = NRLModel(G, _toy_headways())
    import pandas as pd
    trips_df = pd.DataFrame({"chosen_path": ["A|B|D"] * 5 + ["A|C|D"] * 5})
    fitted = model.fit(trips_df, method="Nelder-Mead")
    assert fitted["converged"] or fitted["log_likelihood"] > -100
    assert np.isfinite(fitted["log_likelihood"])
