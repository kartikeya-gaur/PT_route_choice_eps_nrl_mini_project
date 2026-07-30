"""
Global configuration for the transit choice project.

Centralizes everything that was previously hardcoded/scattered across the
one-off Sydney and Bengaluru scripts: file paths, per-city GTFS route
selection (the T1/T2/L2 and GREEN/PURPLE/bus branch-selection problem),
walking transfer parameters, and the shared "true" utility parameters used
for synthetic choice-set generation and as EPS/NRL estimation starting
values.

Import this module rather than hardcoding paths/constants elsewhere -
that was the main source of drift/bugs in the original notebook-driven
workflow (e.g. the directed-edge dedup bug, the stray incremental-trim
inconsistency) and is exactly what this refactor is meant to prevent.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_OUT_DIR = OUTPUTS_DIR / "models"
FIGURES_OUT_DIR = OUTPUTS_DIR / "figures"
TABLES_OUT_DIR = OUTPUTS_DIR / "tables"

for _d in (MODELS_OUT_DIR, FIGURES_OUT_DIR, TABLES_OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def raw_dir(city: str) -> Path:
    return RAW_DIR / city


def processed_dir(city: str) -> Path:
    d = PROCESSED_DIR / city
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# Walking transfer parameters (shared across cities)
# --------------------------------------------------------------------------
MAX_WALK_DISTANCE_M = 500      # meters - max distance to generate a walking transfer edge
WALK_SPEED_MPS = 1.2           # meters/second (~4.3 km/h)

# --------------------------------------------------------------------------
# Per-city GTFS / route configuration
# --------------------------------------------------------------------------
# SELECTED_HEADSIGNS resolves the branch-selection problem: a route_id can
# cover multiple physical branches or directions (Sydney's T1 covers both
# the Western and North Shore branches; T2 covers both the City Circle loop
# and standalone Inner West patterns). None means "no real branching, just
# take the most common stop-sequence pattern" (e.g. Bengaluru's Yellow Line
# before it was dropped from the small subnetwork).
CITY_CONFIGS = {
    "sydney": {
        "gtfs_files": ["stops.txt", "trips.txt", "routes.txt", "stop_times.txt"],
        "target_route_ids": ["2-T1-W-sj2-1", "2-T2-sj2-1", "78-L2-sj2-1"],
        "selected_headsigns": {
            "2-T1-W-sj2-1": "Penrith Via Parramatta",
            "2-T2-sj2-1": "City Circle Via Museum",
            "78-L2-sj2-1": None,
        },
        # T1's "Penrith Via Parramatta" headsign is shared by short-turn
        # trips (e.g. starting mid-line at Blacktown) that outnumber genuine
        # full Central->Penrith runs even within a single day - "most
        # common" picks the short pattern, so force "longest" here instead.
        # Verify this is still what you want after checking real station
        # names in your own environment (this sandbox's stops.txt was stale
        # Bengaluru data, so pattern content couldn't be directly verified).
        "pattern_selection": {
            "2-T1-W-sj2-1": "longest",
        },
        # truncate T1 at Parramatta rather than running the full line to
        # Penrith/Emu Plains/Richmond - T2 doesn't extend this far west
        # (stops around Lidcombe/Strathfield in this network), so only T1
        # needs a boundary here. Replaces an old hardcoded "longitude >=
        # 151" filter with an explicit, checkable station name instead.
        # "branch_boundaries": {
        #     "T1_west": "Parramatta Station",
        # },
        # single service_id per route to use for headway calculation -
        # the raw feed has ~1 service_id per calendar date with no
        # calendar.txt, so combining all of them overstates frequency
        "headway_reference_service_id": {
            "2-T1-W-sj2-1": "TA+d9+7",
            "2-T2-sj2-1": "TA+p2+7",
            "78-L2-sj2-1": "TA+c9+44",
        },
        "demand_source": "entry_exit.csv",   # Opal tap-on/tap-off patronage
        "od_source": None,                    # no real OD file available - use gravity model fallback
    },
    "bengaluru": {
        "gtfs_files": ["stops.txt", "trips.txt", "routes.txt", "stop_times.txt"],
        "target_route_ids": ["GREEN", "PURPLE"],  # Yellow Line excluded from the small test network
        "selected_headsigns": {
            "GREEN": "Silk Institute",
            "PURPLE": "Whitefield (Kadugodi)",
        },
        "service_id_filter": "weekday",
        # boundary stations truncating each branch of the small test network
        "branch_boundaries": {
            "GREEN_north": "Rajajinagar",
            "GREEN_south": "Lalbagh",
            "PURPLE_east": "Mahatma Gandhi Road",
        },
        # parallel bus routes added for genuine route redundancy (BMTC GTFS:
        # github.com/Vonter/bmtc-gtfs), trimmed to the same branch boundaries
        "parallel_bus_routes": {
            "252-F": {"route_id": 1621, "anchors": ("Nadaprabhu Kempegowda Station, Majestic", "Rajajinagar")},
            "25-A": {"route_id": 1771, "anchors": ("Lalbagh", "Nadaprabhu Kempegowda Station, Majestic")},
            "314-D": {"route_id": 1127, "anchors": ("Nadaprabhu Kempegowda Station, Majestic", "Mahatma Gandhi Road"),
                      "trim_at_stop_id": "22278"},  # Mayohall - nearest bus stop to MG Road
        },
        "demand_source": "bmrcl-ridership-hourly",  # github.com/Vonter/bmrcl-ridership-hourly
        "od_source": "bmrcl-ridership-hourly",       # real station-pair OD, unlike Sydney
    },
}

# --------------------------------------------------------------------------
# Route-choice model parameters
# --------------------------------------------------------------------------
K_PATHS = 10                # choice set size per OD pair (k shortest simple paths by duration)
DEFAULT_HEADWAY_MIN = 10.0  # fallback headway for any route_id missing from the headway table
RANDOM_SEED = 42

# "TRUE" utility parameters used to generate synthetic disaggregate choice
# data (Path Size Logit) - the same values are used as EPS/NRL estimation
# starting points, so recovering something close to these is the sanity
# check for both estimators.
TRUE_UTILITY_PARAMS = {
    "beta_ivt": 0.025,        # per in-vehicle minute
    "beta_wait": 0.05,        # per waiting minute (perceived worse than in-vehicle)
    "beta_walk": 0.035,       # per walking minute
    "beta_transfer": 0.4,     # fixed penalty per transfer (minutes-equivalent)
    "beta_pathsize": 1.0,     # standard Path Size Logit coefficient
}

# train/test split fraction held out for validation
TEST_FRACTION = 0.2

# synthetic trip generation is scaled DOWN from real cumulative demand to a
# tractable disaggregate sample size (real OD/patronage totals are often in
# the hundreds of thousands - simulating one synthetic trip per real trip
# would be both slow and pointless, since duplicate OD pairs just resample
# the same small choice set over and over). TARGET_TOTAL_SYNTHETIC_TRIPS
# auto-computes scale_trips = target / sum(od_matrix.total_trips).
TARGET_TOTAL_SYNTHETIC_TRIPS = 1800
MIN_TRIPS_PER_OD = 2