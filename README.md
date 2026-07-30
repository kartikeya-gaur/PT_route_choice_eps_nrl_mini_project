# Transit Choice Project: EPS vs. NRL Route Choice Comparison

Comparative implementation of two route-choice estimation methods —
**Expanded Path Size Logit** (EPS; Frejinger, Bierlaire & Ben-Akiva, 2009)
and **Nested Recursive Logit** (NRL; Mai, Fosgerau & Frejinger, 2015) — on
real transit networks in **Sydney** and **Bengaluru**.

> [!IMPORTANT]
> 🚧 **Work in Progress**
>
> The repository is currently under active development. At present, the
> estimation and evaluation pipeline runs on **a single OD pair** for each
> city (Sydney and Bengaluru). Full-network experiments will be added in a
> future update.

## Why synthetic choice data

Neither method can be estimated from OD counts or patronage data alone —
both need **disaggregate observations of which specific path each traveler
took**, which isn't publicly available for either city. This project
instead builds the real network + real demand from open data, then
simulates a "true" Path Size Logit generating process over that real
network to produce disaggregate synthetic trips, and uses those as ground
truth to fit and validate EPS and NRL. See `src/data/synthetic_trips.py`
and `src/models/path_utils.py` for the full rationale.

**Sydney vs. Bengaluru data quality is asymmetric and this matters for
interpreting results:**

| | Sydney | Bengaluru |
|---|---|---|
| Network | real GTFS (T1/T2/L2) | real GTFS (Green/Purple Metro + BMTC bus) |
| Station demand | real (Opal `entry_exit.csv`) | real (`bmrcl-ridership-hourly`) |
| OD matrix | **estimated** (gravity model, Furness-balanced to real entry/exit totals) | **real** (`bmrcl-ridership-hourly` station-pair data) |
| Route redundancy | rail-branch overlap + 6 walking transfers | rail + parallel bus + walking transfers (deliberately added for genuine path choice) |

Sydney's synthetic trips are therefore "doubly synthetic" (estimated OD +
simulated path choice) vs. Bengaluru's "singly synthetic" (real OD +
simulated path choice only).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data you need to provide

This repo does **not** bundle raw GTFS/demand data (see `.gitignore`) -
place the following under `data/raw/<city>/` yourself:

**`data/raw/sydney/`**
- `stops.txt`, `trips.txt`, `routes.txt`, `stop_times.txt` (Sydney Trains +
  Light Rail GTFS)
- `entry_exit.csv` (Opal tap-on/tap-off patronage; Transport for NSW Open
  Data Hub)

**`data/raw/bengaluru/`**
- `stops.txt`, `trips.txt`, `routes.txt`, `stop_times.txt` (BMRCL Namma
  Metro GTFS)
- `station-hourly.csv`, `station-hourly-exits.csv`, `stationpair-hourly.csv`
  (semicolon-delimited; [github.com/Vonter/bmrcl-ridership-hourly](https://github.com/Vonter/bmrcl-ridership-hourly))
- `bmtc/stops.txt`, `bmtc/trips.txt`, `bmtc/routes.txt`, `bmtc/stop_times.txt`
  (full-city BMTC bus GTFS, used only for the parallel-bus segments
  configured in `src/config.py`'s `parallel_bus_routes`;
  [github.com/Vonter/bmtc-gtfs](https://github.com/Vonter/bmtc-gtfs))

## Running the pipeline

```bash
python main.py --city both              # everything, both cities
python main.py --city sydney            # one city only
python main.py --city both --skip-models   # data prep only (network/demand/synthetic trips), skip EPS/NRL fitting
python main.py --city both --eps-draws 50  # more biased-random-walk draws per OD pair for EPS
```

Outputs land in:
- `data/processed/<city>/` - built network, demand, OD matrix, headways, synthetic trips (+ train/test split)
- `outputs/figures/` - static network plots (`.png`) and interactive maps (`.html`)
- `outputs/tables/` - parameter bias tables, EPS vs. NRL comparison CSVs
- `outputs/models/` - fitted EPS/NRL parameters (`.json`)

## Package structure

```
src/
├── config.py              # paths, per-city GTFS/branch-selection config, "true" utility params
├── data/
│   ├── gtfs_loader.py      # GTFS parsing, platform->station collapsing, branch selection, headways
│   ├── demand_loader.py    # Opal/BMRCL demand, gravity-model OD fallback
│   └── synthetic_trips.py  # Path Size Logit synthetic disaggregate trip generator
├── network/
│   └── supernetwork.py     # transit + walking-transfer edges, Bengaluru bus integration
├── models/
│   ├── path_utils.py       # shared leg-segmentation / utility / Path Size logic
│   ├── eps_model.py        # biased random walk sampler, sampling correction, MNL estimation
│   └── nrl_model.py        # link-based Bellman value iteration, custom MLE
└── evaluation/
    ├── metrics.py           # out-of-sample LL, parameter bias, hit rate
    └── visualization.py     # matplotlib + folium network maps, comparison tables
```

## Known limitations / honesty notes

- **EPS's sampling protocol** (biased random walk + Kumaraswamy noise +
  sampling correction) is implemented faithfully in spirit, not as a
  literal line-by-line reproduction of Frejinger et al. (2009)'s exact
  closed-form derivations. See the module docstring in `eps_model.py`.
- **NRL's estimation** uses a self-contained scipy-based MLE rather than a
  specialized recursive-logit estimation package, to keep the codebase
  runnable without external dependency/version risk.
- **Sydney's OD matrix is estimated, not measured** (see table above) -
  any EPS/NRL comparison across cities should account for this rather than
  treating both as equally grounded in real demand.
- **Headway computation** assumes a single representative service_id per
  route where the raw GTFS feed doesn't cleanly separate weekday/weekend
  calendars (Sydney's raw feed has ~1 service_id per calendar date with no
  `calendar.txt` to disambiguate) - see `gtfs_loader.compute_route_headway`.

## Tests

```bash
pytest tests/ -v
```
