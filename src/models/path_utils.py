"""
src.models.path_utils

Shared path-level utility computation: segmenting a path into legs by
route_id (to correctly detect boarding/transfer events), computing the
in-vehicle/wait/walk/transfer components, the Ben-Akiva & Bierlaire Path
Size correction, and the systematic (deterministic) utility of a path
under a given set of coefficients.

Used by both src.models.eps_model (enumerated/sampled path choice sets)
and the synthetic trip generator. src.models.nrl_model does NOT use this
module directly - NRL operates at the link level via a line-graph Bellman
recursion (see nrl_model.py), which needs the same underlying utility
logic but applied edge-by-edge rather than to a whole path at once.
"""

from src import config


def generalized_time(ivt_min, wait_min, walk_min, wait_weight=2.0, walk_weight=2.0):
    """Combines ivt/wait/walk into one generalized-time value using fixed
    relative weights, rather than estimating separate coefficients for each.

    ivt_min/wait_min/walk_min and n_transfers showed VIF ~5.5-6.8 against
    each other in the synthetic Bengaluru/Sydney data - transferring
    mechanically adds wait+walk+often ivt, so trying to freely estimate
    separate beta_ivt/beta_wait/beta_walk sits on a near-flat log-likelihood
    ridge. Collapsing to one beta_time coefficient (applied to this
    function's output) with fixed weights drops VIF to ~3.3-3.9.

    This is shared by nrl_model.py's and eps_model.py's "generalized_time"
    utility_spec so wait_weight/walk_weight can't silently drift apart
    between the two models. NRL calls this per-link with zeros for whichever
    component that link type doesn't have (e.g. a transfer link passes
    ivt_min=wait_min=0.0); EPS calls it once per whole estimation dataframe
    with all three populated. Works with scalars or pandas Series/arrays
    either way, since it's pure arithmetic.
    """
    return ivt_min + wait_weight * wait_min + walk_weight * walk_min


def compute_path_components(edge_seq, headway_map, default_headway_min=None):
    """Segments a path's edge sequence into legs by route_id to correctly
    count boardings/transfers.

    edge_seq: list of edge-attribute dicts (each with 'link_type',
              'route_id', 'duration') in path order.
    headway_map: dict of route_id -> headway_min.

    Returns (ivt_min, wait_min, walk_min, n_transfers).
    """
    default_headway_min = default_headway_min or config.DEFAULT_HEADWAY_MIN
    ivt_sec, wait_sec, walk_sec = 0.0, 0.0, 0.0
    n_transfers = 0
    current_route = None
    first_transit_leg = True

    for e in edge_seq:
        if e["link_type"] == "transfer":
            walk_sec += e["duration"]
            current_route = None  # next transit edge is a fresh boarding
        else:
            if e["route_id"] != current_route:
                headway = headway_map.get(e["route_id"], default_headway_min)
                wait_sec += (headway * 60) / 2
                if not first_transit_leg:
                    n_transfers += 1
                first_transit_leg = False
                current_route = e["route_id"]
            ivt_sec += e["duration"]

    return ivt_sec / 60, wait_sec / 60, walk_sec / 60, n_transfers


def path_size(paths_link_sets, path_lengths):
    """Ben-Akiva & Bierlaire (2003) Path Size for each path in a choice set.

    paths_link_sets: list (one per path) of lists of (from_node, to_node,
                      link_length) tuples.
    path_lengths: list of each path's total length (same units as link_length;
                  duration is used as the impedance measure throughout this
                  project, as is standard when physical distance isn't the
                  natural impedance).
    """
    ps_values = []
    for i, links_i in enumerate(paths_link_sets):
        L_i = path_lengths[i]
        ps = 0.0
        for link in links_i:
            l_a = link[2]
            n_sharing = sum(1 for links_j in paths_link_sets if link in links_j)
            ps += (l_a / L_i) * (1.0 / n_sharing)
        ps_values.append(ps)
    return ps_values


def systematic_utility(ivt_min, wait_min, walk_min, n_transfers, path_size_value, beta: dict):
    """V(path) = -b_ivt*IVT -b_wait*WAIT -b_walk*WALK -b_transfer*n_transfers
                 + b_pathsize * ln(PathSize)

    beta: dict with keys beta_ivt, beta_wait, beta_walk, beta_transfer,
          beta_pathsize (see src.config.TRUE_UTILITY_PARAMS for the default/
          "true" values used to generate synthetic data).
    """
    import numpy as np
    return (
        - beta["beta_ivt"] * ivt_min
        - beta["beta_wait"] * wait_min
        - beta["beta_walk"] * walk_min
        - beta["beta_transfer"] * n_transfers
        + beta["beta_pathsize"] * np.log(max(path_size_value, 1e-9))
    )


def softmax_utilities(utilities):
    """Exact multinomial-logit probabilities over a fixed set of systematic
    utilities: softmax(utilities). Equivalent in distribution to Eq. 15's
    Gumbel-shock-argmax simulation protocol (synthetic_trips.
    generate_synthetic_trips_universal), just returning the analytic
    probabilities directly rather than simulated choice frequencies - use
    this to reproduce a Fig.-4-style plot exactly (no simulation noise)
    from the systematic_utilities array that function also returns.
    """
    import numpy as np
    u = np.asarray(utilities, dtype=float)
    p = np.exp(u - u.max())
    return p / p.sum()


def cast_node_id(x, example_node=None):
    """Station ids may be numeric (Sydney, usually) or string (Sydney's own
    light-rail 'Stand' stops can force the WHOLE stop_id column to string
    dtype in pandas; Bengaluru's BUS_ prefix nodes are always strings).
    Paths are stored as pipe-joined strings, so splitting always yields
    strings even for originally-numeric node ids - this must be undone
    before comparing against or indexing into the graph, or lookups
    silently fail (or worse, raise a confusing KeyError deep in networkx).

    If example_node is given, casts to match its type EXACTLY - including
    the case where example_node is itself a string, in which case x is
    returned as-is rather than being (wrongly) parsed into a number.
    """
    if example_node is not None:
        if isinstance(example_node, str):
            return str(x)
        try:
            return type(example_node)(x)
        except (ValueError, TypeError):
            return x
    try:
        f = float(x)
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        return x


def cast_path(path, graph=None):
    """Casts every node in a path (list/tuple) to match the graph's node
    dtype, inferred from an arbitrary existing node if `graph` is given."""
    example_node = next(iter(graph.nodes())) if graph is not None and graph.number_of_nodes() else None
    return [cast_node_id(p, example_node) for p in path]