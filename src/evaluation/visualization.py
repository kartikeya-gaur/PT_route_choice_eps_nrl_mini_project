"""
src.evaluation.visualization

Static network plots (matplotlib), interactive maps (folium), and
comparison table exports - generalizes the original plot_supernetwork.py
script into a reusable module.
"""

import folium
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src import config

FALLBACK_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def plot_network(stations_df, edges_df, routes_df=None, title="Supernetwork",
                  label_all_stations=False, save_path=None):
    """Static matplotlib plot: transit edges colored by route, walking
    transfers as dashed grey lines, multi-route stations marked solid
    black, walking-transfer stations ringed in blue."""
    stations = stations_df.set_index("station_id")

    route_color_map, route_name_map = {}, {}
    if routes_df is not None:
        route_color_map = dict(zip(routes_df["route_id"], "#" + routes_df["route_color"]))
        route_name_map = dict(zip(routes_df["route_id"], routes_df["route_short_name"]))

    transit_routes = sorted(edges_df.loc[edges_df["link_type"] == "transit", "route_id"].unique())
    for i, rid in enumerate(transit_routes):
        route_color_map.setdefault(rid, FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)])
        route_name_map.setdefault(rid, rid)

    routes_per_station = {}
    for row in edges_df[edges_df["link_type"] == "transit"].itertuples():
        for sid in (row.from_stop, row.to_stop):
            routes_per_station.setdefault(sid, set()).add(row.route_id)
    interchange_ids = {sid for sid, rs in routes_per_station.items() if len(rs) > 1}

    fig, ax = plt.subplots(figsize=(12, 12))

    transfer_df = edges_df[edges_df["link_type"] == "transfer"]
    for row in transfer_df.itertuples():
        a, b = stations.loc[row.from_stop], stations.loc[row.to_stop]
        ax.plot([a.stop_lon, b.stop_lon], [a.stop_lat, b.stop_lat],
                color="#999999", linewidth=1, linestyle="--", zorder=1, alpha=0.7)

    transit_df = edges_df[edges_df["link_type"] == "transit"]
    for row in transit_df.itertuples():
        a, b = stations.loc[row.from_stop], stations.loc[row.to_stop]
        ax.plot([a.stop_lon, b.stop_lon], [a.stop_lat, b.stop_lat],
                color=route_color_map[row.route_id], linewidth=2.5, zorder=2)

    all_used_ids = set(edges_df["from_stop"]) | set(edges_df["to_stop"])
    used_stations = stations.loc[stations.index.intersection(all_used_ids)]
    ax.scatter(used_stations["stop_lon"], used_stations["stop_lat"], s=45,
               color="white", edgecolor="black", zorder=3)

    interchange_pts = used_stations.loc[used_stations.index.intersection(interchange_ids)]
    ax.scatter(interchange_pts["stop_lon"], interchange_pts["stop_lat"], s=90, color="black", zorder=4)

    transfer_station_ids = set(transfer_df["from_stop"]) | set(transfer_df["to_stop"])
    transfer_pts = used_stations.loc[used_stations.index.intersection(transfer_station_ids)]
    ax.scatter(transfer_pts["stop_lon"], transfer_pts["stop_lat"], s=160,
               facecolors="none", edgecolor="#1f77b4", linewidth=2, zorder=5)

    to_label = used_stations if label_all_stations else interchange_pts
    for row in to_label.itertuples():
        ax.annotate(row.station_name, (row.stop_lon, row.stop_lat),
                    xytext=(5, 5), textcoords="offset points", fontsize=8, zorder=5)

    legend_handles = [Line2D([0], [0], color=route_color_map[rid], lw=2.5, label=route_name_map.get(rid, rid))
                      for rid in transit_routes]
    legend_handles.append(Line2D([0], [0], color="#999999", lw=1, linestyle="--", label="walking transfer"))
    legend_handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
                                  markersize=9, label="multi-route station"))
    legend_handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                                  markeredgecolor="#1f77b4", markeredgewidth=2, markersize=11,
                                  label="walking-transfer station"))
    ax.legend(handles=legend_handles, loc="best", fontsize=9)

    stats_text = (
        f"Total nodes: {len(all_used_ids)}\n"
        f"Total edges: {len(edges_df)}\n"
        f"Transfer edges: {len(transfer_df)}\n"
        f"Walking-transfer stations: {len(transfer_station_ids)}\n"
        f"Multi-route stations: {len(interchange_ids)}"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999", alpha=0.9))

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"Plot saved to:\n--> {save_path}")
    return fig, ax


def folium_map(stations_df, edges_df, routes_df=None, save_path=None):
    """Interactive folium map of the network, colored the same way as
    plot_network."""
    center_lat = stations_df["stop_lat"].mean()
    center_lon = stations_df["stop_lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

    route_color_map = {}
    if routes_df is not None:
        route_color_map = dict(zip(routes_df["route_id"], "#" + routes_df["route_color"]))
    transit_routes = sorted(edges_df.loc[edges_df["link_type"] == "transit", "route_id"].unique())
    for i, rid in enumerate(transit_routes):
        route_color_map.setdefault(rid, FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)])

    stations = stations_df.set_index("station_id")
    for row in edges_df.itertuples():
        a, b = stations.loc[row.from_stop], stations.loc[row.to_stop]
        color = "#999999" if row.link_type == "transfer" else route_color_map.get(row.route_id, "#333333")
        dash = "5, 5" if row.link_type == "transfer" else None
        folium.PolyLine(
            [(a.stop_lat, a.stop_lon), (b.stop_lat, b.stop_lon)],
            color=color, weight=3, opacity=0.8, dash_array=dash,
        ).add_to(m)

    for sid, row in stations.iterrows():
        folium.CircleMarker(
            location=(row.stop_lat, row.stop_lon), radius=4,
            color="black", fill=True, fill_color="white", fill_opacity=1,
            popup=row.station_name,
        ).add_to(m)

    if save_path:
        m.save(str(save_path))
        print(f"Interactive map saved to:\n--> {save_path}")
    return m


def save_comparison_table(comparison_df, path):
    comparison_df.to_csv(path, index=False)
    print(f"Comparison table saved to:\n--> {path}")
