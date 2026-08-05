#!/usr/bin/env python3
"""
Fetch public repos for a GitHub user, compute a knowledge graph
based on shared languages and topics, output:
  1. docs/graph_data.json  — consumed by the interactive 3d-force-graph page
  2. assets/graph_preview.gif — rotating 3D preview embedded in the README
"""

import json
import os
import math

import requests
import networkx as nx
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter

USERNAME = os.environ.get("GITHUB_USERNAME", "Indiguana")
API = f"https://api.github.com/users/{USERNAME}/repos"

LANG_COLORS = {
    "Python": "#3572A5",
    "JavaScript": "#f1e05a",
    "TypeScript": "#3178c6",
    "C++": "#f34b7d",
    "C#": "#178600",
    "C": "#555555",
    "Java": "#b07219",
    "Astro": "#ff5a03",
    "TeX": "#3D6117",
    "HTML": "#e34c26",
    "CSS": "#563d7c",
    "Rust": "#dea584",
    "Go": "#00ADD8",
    "Shell": "#89e051",
    "Jupyter Notebook": "#DA5B0B",
}
DEFAULT_COLOR = "#8b949e"


def fetch_repos():
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    repos, page = [], 1
    while True:
        resp = requests.get(
            API, params={"per_page": 100, "page": page, "type": "owner"}, headers=headers
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        repos.extend(batch)
        page += 1

    return [r for r in repos if not r["fork"] and r["name"].lower() != USERNAME.lower()]


def build_graph(repos):
    nodes = []
    for r in repos:
        lang = r.get("language") or "Unknown"
        nodes.append(
            {
                "id": r["name"],
                "language": lang,
                "color": LANG_COLORS.get(lang, DEFAULT_COLOR),
                "description": r.get("description") or "",
                "stars": r.get("stargazers_count", 0),
                "url": r.get("html_url", ""),
                "topics": r.get("topics", []),
                "val": min(12, max(3, math.log10((r.get("size") or 1) + 1) * 3)),
            }
        )

    links = []
    for i, a in enumerate(repos):
        for j, b in enumerate(repos):
            if i >= j:
                continue
            reasons = []
            if a.get("language") and a["language"] == b.get("language"):
                reasons.append(a["language"])
            shared = set(a.get("topics", [])) & set(b.get("topics", []))
            reasons.extend(sorted(shared))
            if reasons:
                links.append({
                    "source": a["name"],
                    "target": b["name"],
                    "weight": len(reasons),
                    "label": " + ".join(reasons),
                })

    return {"nodes": nodes, "links": links}


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def generate_gif(data, path):
    G = nx.Graph()
    node_map = {}
    for n in data["nodes"]:
        G.add_node(n["id"])
        node_map[n["id"]] = n
    for e in data["links"]:
        G.add_edge(e["source"], e["target"], weight=e["weight"], label=e.get("label", ""))

    pos2d = nx.spring_layout(G, k=3.0, iterations=100, seed=42)
    rng = np.random.default_rng(42)
    pos3d = {n: (*xy, rng.uniform(-0.3, 0.3)) for n, xy in pos2d.items()}

    nodes = list(G.nodes())
    xs = [pos3d[n][0] for n in nodes]
    ys = [pos3d[n][1] for n in nodes]
    zs = [pos3d[n][2] for n in nodes]
    colors = [hex_to_rgb(node_map[n]["color"]) for n in nodes]
    sizes = [max(100, node_map[n]["val"] * 28) for n in nodes]

    fig = plt.figure(figsize=(8, 5.5), dpi=120)
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0d1117")

    total_frames = 72

    def draw(frame):
        ax.clear()
        ax.set_facecolor("#0d1117")
        ax.view_init(elev=25, azim=frame * (360 / total_frames))

        # edges — draw glow layer then solid line on top
        for e in data["links"]:
            s, t = e["source"], e["target"]
            if s not in pos3d or t not in pos3d:
                continue
            lx = [pos3d[s][0], pos3d[t][0]]
            ly = [pos3d[s][1], pos3d[t][1]]
            lz = [pos3d[s][2], pos3d[t][2]]
            w = e["weight"]
            ax.plot(lx, ly, lz, color="#58a6ff", alpha=0.12, linewidth=6 + w * 2, solid_capstyle="round")
            ax.plot(lx, ly, lz, color="#58a6ff", alpha=0.55 + w * 0.1, linewidth=1.5 + w * 0.5, solid_capstyle="round")

            # edge label at midpoint
            mx = (lx[0] + lx[1]) / 2
            my = (ly[0] + ly[1]) / 2
            mz = (lz[0] + lz[1]) / 2
            label = e.get("label", "")
            if label:
                ax.text(mx, my, mz, label, fontsize=4.5, color="#58a6ff", alpha=0.7,
                        ha="center", va="center", fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d1117", edgecolor="none", alpha=0.8))

        # nodes — outer glow then solid dot
        ax.scatter(xs, ys, zs, c=colors, s=[s * 2.5 for s in sizes], alpha=0.08, edgecolors="none")
        ax.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.95, edgecolors="white", linewidths=0.6, depthshade=True)

        # node labels
        for i, name in enumerate(nodes):
            short = name[:22] + ".." if len(name) > 24 else name
            ax.text(xs[i], ys[i], zs[i] + 0.08, short, fontsize=5.2, color="#e6edf3",
                    ha="center", va="bottom", fontfamily="monospace", fontweight="bold")

        pad = 0.4
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_zlim(min(zs) - pad, max(zs) + pad)
        ax.axis("off")
        ax.grid(False)

    anim = FuncAnimation(fig, draw, frames=total_frames, interval=70)
    anim.save(path, writer=PillowWriter(fps=14), dpi=120)
    plt.close()
    print(f"  -> {path}")


def main():
    print(f"Fetching repos for {USERNAME}...")
    repos = fetch_repos()
    print(f"  {len(repos)} repos (excluding forks & profile repo)")

    data = build_graph(repos)
    print(f"  {len(data['nodes'])} nodes, {len(data['links'])} edges")

    os.makedirs("docs", exist_ok=True)
    with open("docs/graph_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print("  -> docs/graph_data.json")

    os.makedirs("assets", exist_ok=True)
    generate_gif(data, "assets/graph_preview.gif")


if __name__ == "__main__":
    main()
