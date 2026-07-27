"""
Fairness-Goodness Algorithm (FGA)
=================================

Kumar, Spezzano, Subrahmanian, Faloutsos,
"Edge Weight Prediction in Weighted Signed Networks", ICDM 2016.

Implements the mutually-recursive Fairness (f: V -> [0, 1]) and
Goodness (g: V -> [-1, 1]) scores for a Weighted Signed Network (WSN)
G = (V, E, W), W: E -> [-1, 1], following Figure 3 of the paper:

    f^0(u) = 1,  g^0(u) = 1,  for all u in V
    t = -1
    repeat:
        t = t + 1
        g^{t+1}(v) = (1 / |in(v)|) * sum_{u in in(v)} f^t(u) * W(u, v)
        f^{t+1}(u) = 1 - (1 / (2 * |out(u)|)) * sum_{v in out(u)} |W(u, v) - g^{t+1}(v)|
    until sum_u |f^{t+1}(u) - f^t(u)| <= eps  and  sum_v |g^{t+1}(v) - g^t(v)| <= eps
    return f^{t+1}, g^{t+1}

Notes on edge cases not explicitly covered by the paper's pseudocode:
  * A vertex with no incoming edges has no defined average in Eq. (1);
    we leave its goodness at the previous iteration's value (effectively
    it never updates, matching "we don't know anything new about it").
  * Symmetrically, a vertex with no outgoing edges has no defined
    average in Eq. (2); its fairness is likewise held at the previous
    value (a vertex that never rates anyone cannot be judged unfair).
This is the natural convention used in the reference SNAP implementation
of FGA and keeps every score well defined for real, sparse WSNs.
"""

from __future__ import annotations

import argparse
import gzip
import io
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx


def _sanitize_filename_part(value: str) -> str:
    """Convert a CLI-derived value into a filesystem-friendly token."""
    sanitized = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            sanitized.append(char)
        else:
            sanitized.append("-")
    return "".join(sanitized).strip("-_.") or "value"


def _get_dataset_label(dataset: Optional[str]) -> str:
    """Return a short dataset label for plot file names."""
    if dataset is None:
        return "toy-example"

    dataset_name = Path(dataset).name
    for suffix in (".csv.gz", ".csv", ".gz"):
        if dataset_name.endswith(suffix):
            dataset_name = dataset_name[: -len(suffix)]
            break

    return _sanitize_filename_part(dataset_name)


def build_distribution_output_path(
    dataset: Optional[str],
    algorithm: str,
    eps: float,
    max_iter: int,
    output_dir: Optional[str] = None,
) -> Path:
    """Build an output path that reflects the run configuration."""
    base_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    file_name = (
        f"{_get_dataset_label(dataset)}_{algorithm}_"
        f"eps-{eps:g}_max-iter-{max_iter}_distributions.png"
    )
    return base_dir / file_name


def save_score_distributions(
    fairness: Dict,
    goodness: Dict,
    output_path: Path,
) -> Path:
    """Save fairness and goodness histograms using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to save distributions. Install it with 'pip install matplotlib'."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    goodness_values = list(goodness.values())
    fairness_values = list(fairness.values())

    axes[0].hist(goodness_values, bins=30, range=(-1.0, 1.0), color="steelblue", edgecolor="black")
    axes[0].set_title("Goodness Distribution")
    axes[0].set_xlabel("Goodness")
    axes[0].set_ylabel("Node Count")
    axes[0].set_xlim(-1.0, 1.0)

    axes[1].hist(fairness_values, bins=30, range=(0.0, 1.0), color="darkorange", edgecolor="black")
    axes[1].set_title("Fairness Distribution")
    axes[1].set_xlabel("Fairness")
    axes[1].set_ylabel("Node Count")
    axes[1].set_xlim(0.0, 1.0)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_toy_visualization_output_path(
    algorithm: str,
    eps: float,
    max_iter: int,
    temporal: bool = False,
    dataset_label: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """Build an output path for toy graph visualization."""
    base_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    temporal_suffix = "_temporal" if temporal else ""
    dataset_prefix = _sanitize_filename_part(dataset_label) if dataset_label else "toy-example"
    file_name = f"{dataset_prefix}_{algorithm}_eps-{eps:g}_max-iter-{max_iter}{temporal_suffix}_graph.png"
    return base_dir / file_name


def save_toy_graph_visualization(
    G: "nx.DiGraph",
    fairness: Dict,
    goodness: Dict,
    output_path: Path,
    show_time_labels: bool = False,
) -> Path:
    """Save a toy-example graph plot with node color=goodness and size=fairness."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to visualize the toy graph. Install it with 'pip install matplotlib'."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7), constrained_layout=True)
    pos = nx.spring_layout(G, seed=42)
    nodes = list(G.nodes())

    node_colors = [float(goodness.get(node, 0.0)) for node in nodes]
    node_sizes = [300.0 + 900.0 * max(0.0, min(1.0, float(fairness.get(node, 0.0)))) for node in nodes]
    edge_colors = ["#2ca02c" if _get_edge_weight(G, u, v) >= 0.0 else "#d62728" for u, v in G.edges()]
    edge_widths = [1.0 + 2.0 * abs(_get_edge_weight(G, u, v)) for u, v in G.edges()]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=14,
        alpha=0.7,
    )
    node_artist = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=nodes,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.get_cmap("RdYlGn"),
        vmin=-1.0,
        vmax=1.0,
        linewidths=1.0,
        edgecolors="black",
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    if show_time_labels:
        edge_labels = {
            (u, v): f"w={_get_edge_weight(G, u, v):+.2f}\nt={_get_edge_time(G, u, v):.2f}"
            for u, v in G.edges()
        }
    else:
        edge_labels = {(u, v): f"{_get_edge_weight(G, u, v):+.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        ax=ax,
        edge_labels=edge_labels,
        font_size=8,
        font_color="#303030",
        rotate=False,
        bbox={"alpha": 0.6, "edgecolor": "none", "facecolor": "white"},
    )

    colorbar = fig.colorbar(node_artist, ax=ax, shrink=0.85)
    colorbar.set_label("Goodness")
    title = "Toy Example Graph (color=goodness, size=fairness, edge label=weight"
    if show_time_labels:
        title += ", time)"
    else:
        title += ")"
    ax.set_title(title)
    ax.axis("off")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path

def _get_edge_weight(G: "nx.DiGraph", u, v) -> float:
    """Return the edge weight or 0.0 when the edge or weight attribute is missing."""
    edge_data = G.get_edge_data(u, v)
    if not edge_data:
        return 0.0
    weight = edge_data.get("weight", 0.0)
    if weight is None:
        return 0.0
    try:
        return float(weight)
    except (TypeError, ValueError):
        return 0.0


def _get_edge_time(G: "nx.DiGraph", u, v) -> float:
    """Return the edge time or 0.0 when the edge or time attribute is missing."""
    edge_data = G.get_edge_data(u, v)
    if not edge_data:
        return 0.0
    time_value = edge_data.get("time", 0.0)
    if time_value is None:
        return 0.0
    try:
        return float(time_value)
    except (TypeError, ValueError):
        return 0.0


def _load_dataset_graph(dataset: str) -> Tuple["nx.DiGraph", bool, Optional[str]]:
    """Load a graph from a dataset path or one of the built-in toy dataset names.

    Returns a tuple of (graph, show_time_labels, toy_dataset_label).
    """
    toy_dataset_loaders = {
        "toy_example": (toy_example, False),
        "toy_example_temporal": (toy_example_temporal, True),
        "toy_example_temporal_p_long": (toy_example_temporal_p_long, True),
        "toy_example_temporal_p_short": (toy_example_temporal_p_short, True),
    }
    loader_info = toy_dataset_loaders.get(dataset)
    if loader_info is not None:
        loader, show_time_labels = loader_info
        return loader(), show_time_labels, dataset

    raise FileNotFoundError(dataset)

def compute_fairness_goodness(
    G: "nx.DiGraph",
    eps: float = 0.001,
    max_iter: int = 1000,
    verbose: bool = False,
) -> Tuple[Dict, Dict]:
    """Compute Fairness and Goodness scores for every node of a WSN.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph whose edges carry a numeric 'weight' attribute
        in [-1, 1] (i.e. W(u, v) = G[u][v]['weight']).
    eps : float
        Convergence threshold epsilon, as in the paper (default 0.001).
    max_iter : int
        Safety cap on the number of iterations.
    verbose : bool
        If True, print the per-iteration total change.

    Returns
    -------
    (fairness, goodness) : (dict, dict)
        fairness[u]  in [0, 1] for every node u
        goodness[v]  in [-1, 1] for every node v
    """
    nodes = list(G.nodes())

    # Precompute in/out neighbor lists once (Fig. 3 uses in(v), out(u)).
    in_nbrs = {v: list(G.predecessors(v)) for v in nodes}
    out_nbrs = {u: list(G.successors(u)) for u in nodes}

    # Line 3: f^0(u) = 1, g^0(u) = 1 for all u in V
    f_prev = {u: 1.0 for u in nodes}
    g_prev = {u: 1.0 for u in nodes}

    t = -1
    while True:
        t += 1

        # Line 7: g^{t+1}(v) = (1/|in(v)|) * sum_{u in in(v)} f^t(u) * W(u, v)
        g_next: Dict = {}
        for v in nodes:
            preds = in_nbrs[v]
            if not preds:
                # No incoming ratings -> goodness undefined by Eq. (1);
                # keep previous value (initially 1.0, i.e. neutral-good prior).
                g_next[v] = g_prev[v]
                continue
            s = 0.0
            for u in preds:
                w = _get_edge_weight(G, u, v)
                s += f_prev[u] * w
            g_next[v] = s / len(preds)

        # Line 8: f^{t+1}(u) = 1 - (1/(2|out(u)|)) * sum_{v in out(u)} |W(u,v) - g^{t+1}(v)|
        f_next: Dict = {}
        for u in nodes:
            succs = out_nbrs[u]
            if not succs:
                # Vertex rates no one -> fairness undefined by Eq. (2);
                # keep previous value (initially 1.0, i.e. fully-fair prior).
                f_next[u] = f_prev[u]
                continue
            s = 0.0
            for v in succs:
                w = _get_edge_weight(G, u, v)
                s += abs(w - g_next[v])
            f_next[u] = 1.0 - s / (2.0 * len(succs))

        # Line 9: convergence check
        delta_f = sum(abs(f_next[u] - f_prev[u]) for u in nodes)
        delta_g = sum(abs(g_next[u] - g_prev[u]) for u in nodes)

        if verbose:
            print(f"iter {t}: delta_f={delta_f:.6f}  delta_g={delta_g:.6f}")

        f_prev, g_prev = f_next, g_next

        if (delta_f <= eps and delta_g <= eps) or t >= max_iter:
            break

    return f_prev, g_prev


def get_evaluation_count(G: "nx.DiGraph", u) -> int:
    """Return the number of evaluations made by node u.

    In this module, an evaluation corresponds to an outgoing edge from u.
    """
    return G.out_degree(u)


def get_evaluation_period(G: "nx.DiGraph", u) -> Optional[float]:
    """Return the span of time covered by u's evaluations.

    The function looks for an edge attribute named "time" on outgoing edges
    from u and returns the difference between the maximum and minimum
    timestamp found. If no such timestamps are available, it returns None.
    """
    times = []
    for _, _, data in G.out_edges(u, data=True):
        if "time" in data and data["time"] is not None:
            times.append(float(data["time"]))

    if not times:
        return None

    return max(times) - min(times)


def _map_to_unit_interval(x: float) -> float:
    """Apply the logistic sigmoid function $1 / (1 + e^{-x})."""
    return 1.0 / (1.0 + math.exp(-x))


def get_normalized_evaluation_count(G: "nx.DiGraph", u) -> float:
    """Normalize the evaluation count for node u into [0, 1] via sigmoid."""
    return _map_to_unit_interval(get_evaluation_count(G, u))


def get_normalized_evaluation_period(G: "nx.DiGraph", u) -> Optional[float]:
    """Normalize the evaluation period for node u into [0, 1] via sigmoid."""
    period = get_evaluation_period(G, u)
    if period is None:
        return None
    return _map_to_unit_interval(period)


def compute_fairness_goodness_with_evaluation_weights(
    G: "nx.DiGraph",
    eps: float = 0.001,
    max_iter: int = 1000,
    verbose: bool = False,
) -> Tuple[Dict, Dict]:
    """Compute fairness and goodness while weighting by evaluation statistics.

    The original FGA update is modified so that:
    - fairness for node u is scaled by both the normalized evaluation count
      and the normalized evaluation period of u;
        - goodness for node v is first computed by the original FGA update and
            then scaled by the normalized evaluation period of v itself.
    """
    nodes = list(G.nodes())

    in_nbrs = {v: list(G.predecessors(v)) for v in nodes}
    out_nbrs = {u: list(G.successors(u)) for u in nodes}

    f_prev = {u: 1.0 for u in nodes}
    g_prev = {u: 1.0 for u in nodes}

    t = -1
    while True:
        t += 1

        g_next: Dict = {}
        for v in nodes:
            preds = in_nbrs[v]
            if not preds:
                g_next[v] = g_prev[v]
                continue
            s = 0.0
            for u in preds:
                w = _get_edge_weight(G, u, v)
                s += f_prev[u] * w
            base_goodness = s / len(preds)
            period_weight = get_normalized_evaluation_period(G, v)
            if period_weight is None:
                period_weight = 1.0
            g_next[v] = base_goodness * period_weight

        f_next: Dict = {}
        for u in nodes:
            succs = out_nbrs[u]
            if not succs:
                f_next[u] = f_prev[u]
                continue
            s = 0.0
            for v in succs:
                w = _get_edge_weight(G, u, v)
                s += abs(w - g_next[v])
            base_fairness = 1.0 - s / (2.0 * len(succs))
            count_weight = get_normalized_evaluation_count(G, u)
            period_weight = get_normalized_evaluation_period(G, u)
            if period_weight is None:
                period_weight = 1.0
            f_next[u] = base_fairness * count_weight * period_weight

        delta_f = sum(abs(f_next[u] - f_prev[u]) for u in nodes)
        delta_g = sum(abs(g_next[u] - g_prev[u]) for u in nodes)

        if verbose:
            print(f"iter {t}: delta_f={delta_f:.6f}  delta_g={delta_g:.6f}")

        f_prev, g_prev = f_next, g_next

        if (delta_f <= eps and delta_g <= eps) or t >= max_iter:
            break

    return f_prev, g_prev

# ---------------------------------------------------------------------------
# Bitcoin-OTC loader
# ---------------------------------------------------------------------------
def load_bitcoin_otc(path: str) -> "nx.DiGraph":
    """Load the SNAP soc-sign-bitcoinotc.csv(.gz) file into a WSN.

    Dataset: https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
    Format per line: SOURCE, TARGET, RATING, TIME
        RATING is an integer in [-10, 10] (excluding 0).
    We rescale RATING to W(u, v) in [-1, 1] via division by 10,
    exactly as described in the paper (Section V, "Bitcoin networks").
    """
    G = nx.DiGraph()

    data_path = Path(path)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path

    opener = gzip.open if str(data_path).endswith(".gz") else open
    with opener(data_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            src, tgt, rating, timestamp = parts[0], parts[1], parts[2], parts[3]
            u, v = int(src), int(tgt)
            w = float(rating) / 10.0
            w = max(-1.0, min(1.0, w))  # clip, just in case
            # If a (u, v) pair repeats, keep the latest occurrence
            # (SNAP's file is already deduplicated per pair in practice).
            G.add_edge(u, v, weight=w)
    return G


def load_bitcoin_otc_temporal(path: str) -> "nx.DiGraph":
    """Load the Bitcoin OTC dataset while preserving the edge timestamps.

    The CSV format is SOURCE,TARGET,RATING,TIME. This loader parses the
    timestamp column, normalizes the values to the range [0, 1], and stores
    the normalized time as an edge attribute named "time".
    """
    G = nx.DiGraph()

    data_path = Path(path)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path

    timestamps = []
    opener = gzip.open if str(data_path).endswith(".gz") else open
    with opener(data_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            src, tgt, rating, timestamp = parts[0], parts[1], parts[2], parts[3]
            timestamps.append(float(timestamp))

    if not timestamps:
        return G

    min_time = min(timestamps)
    max_time = max(timestamps)
    if max_time == min_time:
        normalized_range = 1.0
    else:
        normalized_range = max_time - min_time

    opener = gzip.open if str(data_path).endswith(".gz") else open
    with opener(data_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            src, tgt, rating, timestamp = parts[0], parts[1], parts[2], parts[3]
            u, v = int(src), int(tgt)
            w = float(rating) / 10.0
            w = max(-1.0, min(1.0, w))
            ts = float(timestamp)
            normalized_time = (ts - min_time) / normalized_range
            G.add_edge(u, v, weight=w, time=normalized_time)

    return G


# ---------------------------------------------------------------------------
# Small worked example (sanity check), loosely in the spirit of Fig. 2(a):
# a small set of "fair/good" raters and two "unfair, very negative" raters.
# ---------------------------------------------------------------------------
def _toy_example_edges():
    """Return the canonical toy-example edge list."""
    return [
        # fair vertices rating each other positively/consistently
        ("a", "c", 0.9), ("b", "c", 0.8), ("e", "c", 0.7),
        ("a", "b", 0.8), ("c", "b", 0.6), ("e", "b", 0.5),
        ("a", "e", 0.9), ("b", "e", 0.8), ("c", "e", 0.85),
        ("b", "a", 0.85), ("c", "a", 0.8), ("e", "a", 0.9),
        # fair vertices rating d well (d is good, just not fair)
        ("a", "d", 0.85), ("b", "d", 0.9), ("c", "d", 0.8), ("e", "d", 0.9),
        # d rates erratically: sometimes very positive, sometimes very negative
        ("d", "a", 0.9), ("d", "b", -0.9), ("d", "c", 0.9), ("d", "e", -0.85),
        # fair vertices rating the trolls f, g very negatively
        ("a", "f", -0.9), ("b", "f", -0.85), ("c", "f", -0.9), ("e", "f", -0.8),
        ("a", "g", -0.85), ("b", "g", -0.9), ("c", "g", -0.85), ("e", "g", -0.9),
        # trolls f, g rate everyone very negatively (undifferentiated / hostile)
        ("f", "a", -0.9), ("f", "b", -0.9), ("f", "c", -0.9), ("f", "e", -0.9),
        ("g", "a", -0.85), ("g", "b", -0.9), ("g", "c", -0.85), ("g", "e", -0.9),
    ]


def toy_example() -> "nx.DiGraph":
    """Every node both rates and is rated, so f and g are both meaningful.

    a, b, c, e: mutually consistent, fair raters -> high fairness, high goodness
    d: rates wildly inconsistently (helps a lot, hurts a lot) -> low fairness,
       but is itself rated well by the fair nodes -> high goodness
    f, g: trolls that dump very negative ratings on everyone
          -> low fairness (deviates from consensus) and very negative goodness
             (the fair raters, in turn, rate them very negatively)
    """
    G = nx.DiGraph()
    for u, v, w in _toy_example_edges():
        G.add_edge(u, v, weight=w)
    return G


def toy_example_temporal() -> "nx.DiGraph":
    """Toy example with the same ratings plus normalized time attributes.

    The edge order is mapped linearly onto [0, 1] so the toy graph can be used
    with the temporal evaluation-weighted FGA variant.
    """
    G = nx.DiGraph()
    edges = _toy_example_edges()
    last_index = max(len(edges) - 1, 1)
    for index, (u, v, w) in enumerate(edges):
        G.add_edge(u, v, weight=w, time=index / last_index)
    return G


def _toy_example_temporal_with_probe_node(
    probe_node: str,
    probe_time_values,
) -> "nx.DiGraph":
    """Return the toy graph extended with one probe node and custom probe times."""
    G = toy_example_temporal()
    probe_edges = [
        (probe_node, "a", 0.9),
        (probe_node, "b", 0.7),
        (probe_node, "c", -0.8),
        (probe_node, "d", 0.6),
        (probe_node, "e", 0.85),
        (probe_node, "f", -0.9),
        (probe_node, "g", 0.75),
        ("a", probe_node, 0.8),
        ("b", probe_node, 0.65),
        ("c", probe_node, -0.75),
        ("d", probe_node, 0.55),
        ("e", probe_node, 0.9),
        ("f", probe_node, -0.85),
        ("g", probe_node, 0.7),
    ]

    probe_time_values = list(probe_time_values)
    if len(probe_time_values) != len(probe_edges):
        raise ValueError("probe_time_values must match the number of probe edges")

    for (u, v, w), time_value in zip(probe_edges, probe_time_values):
        G.add_edge(u, v, weight=w, time=time_value)
    return G


def toy_example_temporal_p_long() -> "nx.DiGraph":
    """Toy example extended with p_long, whose probe edges span [0, 1]."""
    probe_time_values = [index / 13.0 for index in range(14)]
    return _toy_example_temporal_with_probe_node("p_long", probe_time_values)


def toy_example_temporal_p_short() -> "nx.DiGraph":
    """Toy example extended with p_short, whose probe edges cluster near 0.5."""
    probe_time_values = [0.48 + 0.0015 * index for index in range(14)]
    return _toy_example_temporal_with_probe_node("p_short", probe_time_values)


def main():
    parser = argparse.ArgumentParser(description="Fairness-Goodness Algorithm (FGA)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to soc-sign-bitcoinotc.csv(.gz). If omitted, runs the toy example.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["basic", "weighted"],
        default="weighted",
        help=(
            "Algorithm variant to run: 'basic' uses the original FGA update, "
            "'weighted' uses evaluation-count/time weighted updates."
        ),
    )
    parser.add_argument("--eps", type=float, default=0.001)
    parser.add_argument("--top", type=int, default=10, help="Show top-k by goodness/fairness")
    parser.add_argument("--max-iter", type=int, default=50, help="Maximum FGA iterations")
    parser.add_argument(
        "--save-distributions",
        action="store_true",
        help="Save goodness/fairness distribution histograms as a PNG file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where distribution images are saved (default: fga directory).",
    )
    parser.add_argument(
        "--visualize-toy",
        action="store_true",
        help="Save a toy-graph visualization PNG when running without --dataset.",
    )
    args = parser.parse_args()

    toy_visualization = False
    toy_dataset_label: Optional[str] = None
    toy_visualization_show_time_labels = False

    if args.dataset:
        print(f"Loading WSN from {args.dataset} ...")
        try:
            G, toy_visualization_show_time_labels, toy_dataset_label = _load_dataset_graph(args.dataset)
            toy_visualization = True
            print(f"Loaded toy graph: |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}")
        except FileNotFoundError:
            if args.algorithm == "basic":
                G = load_bitcoin_otc(args.dataset)
            else:
                G = load_bitcoin_otc_temporal(args.dataset)
            print(f"Loaded graph: |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}")
    else:
        print("No --dataset given, running toy example instead.")
        G = toy_example_temporal() if args.algorithm == "weighted" else toy_example()
        toy_visualization = True
        toy_visualization_show_time_labels = args.algorithm == "weighted"

    if args.algorithm == "basic":
        fairness, goodness = compute_fairness_goodness(
            G, eps=args.eps, max_iter=args.max_iter, verbose=False
        )
    else:
        fairness, goodness = compute_fairness_goodness_with_evaluation_weights(
            G, eps=args.eps, max_iter=args.max_iter, verbose=False
        )

    if args.save_distributions:
        output_path = build_distribution_output_path(
            dataset=args.dataset,
            algorithm=args.algorithm,
            eps=args.eps,
            max_iter=args.max_iter,
            output_dir=args.output_dir,
        )
        saved_path = save_score_distributions(fairness, goodness, output_path)
        print(f"\nSaved score distributions to {saved_path}")

    if args.visualize_toy:
        if not toy_visualization:
            print("\n--visualize-toy was ignored because --dataset was provided.")
        else:
            output_path = build_toy_visualization_output_path(
                algorithm=args.algorithm,
                eps=args.eps,
                max_iter=args.max_iter,
                temporal=toy_visualization_show_time_labels,
                dataset_label=toy_dataset_label,
                output_dir=args.output_dir,
            )
            saved_path = save_toy_graph_visualization(
                G,
                fairness,
                goodness,
                output_path,
                show_time_labels=toy_visualization_show_time_labels,
            )
            print(f"\nSaved toy graph visualization to {saved_path}")

    print("\nTop nodes by goodness:")
    for node, g in sorted(goodness.items(), key=lambda kv: kv[1], reverse=True)[: args.top]:
        print(f"  {node!r:>10}  f={fairness[node]:.3f}  g={g:.3f}")

    print("\nBottom nodes by goodness (likely trolls/scammers):")
    for node, g in sorted(goodness.items(), key=lambda kv: kv[1])[: args.top]:
        print(f"  {node!r:>10}  f={fairness[node]:.3f}  g={g:.3f}")

    print("\nLowest fairness nodes (unreliable / erratic raters):")
    for node, f in sorted(fairness.items(), key=lambda kv: kv[1])[: args.top]:
        print(f"  {node!r:>10}  f={f:.3f}  g={goodness[node]:.3f}")


if __name__ == "__main__":
    main()
