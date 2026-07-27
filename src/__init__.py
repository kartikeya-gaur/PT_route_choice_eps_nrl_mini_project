"""
transit_choice_project.src

Core reusable package for the EPS vs. NRL route-choice comparison across
Sydney and Bengaluru transit networks.

Subpackages:
    data        - GTFS parsing/filtering and demand (OD/patronage) loading
    network     - Supernetwork (transit + walking transfer graph) construction
    models      - EPS (Expanded Path Size Logit) and NRL (Nested Recursive Logit)
    evaluation  - Validation metrics and visualization
"""

__version__ = "0.1.0"
