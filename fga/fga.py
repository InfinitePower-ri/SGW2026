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
                w = G[u][v]["weight"]
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
                w = G[u][v]["weight"]
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


def _sigmoid(x: float) -> float:
    """Apply the logistic sigmoid function $1 / (1 + e^{-x})."""
    return 1.0 / (1.0 + math.exp(-x))


def get_normalized_evaluation_count(G: "nx.DiGraph", u) -> float:
    """Normalize the evaluation count for node u into [0, 1] via sigmoid."""
    return _sigmoid(get_evaluation_count(G, u))


def get_normalized_evaluation_period(G: "nx.DiGraph", u) -> Optional[float]:
    """Normalize the evaluation period for node u into [0, 1] via sigmoid."""
    period = get_evaluation_period(G, u)
    if period is None:
        return None
    return _sigmoid(period)


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
    - goodness for node v is scaled by the normalized evaluation period of
      each rater u that contributes to v.
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
                period_weight = get_normalized_evaluation_period(G, u)
                if period_weight is None:
                    period_weight = 1.0
                w = G[u][v]["weight"]
                s += f_prev[u] * w * period_weight
            g_next[v] = s / len(preds)

        f_next: Dict = {}
        for u in nodes:
            succs = out_nbrs[u]
            if not succs:
                f_next[u] = f_prev[u]
                continue
            s = 0.0
            for v in succs:
                w = G[u][v]["weight"]
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


def fxg_predict(G: "nx.DiGraph", fairness: Dict, goodness: Dict, u, v) -> float:
    """Predicted edge weight for (u, v), the FxG score: f(u) * g(v)."""
    return fairness[u] * goodness[v]


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
    edges = [
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
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G


def main():
    parser = argparse.ArgumentParser(description="Fairness-Goodness Algorithm (FGA)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to soc-sign-bitcoinotc.csv(.gz). If omitted, runs the toy example.",
    )
    parser.add_argument("--eps", type=float, default=0.001)
    parser.add_argument("--top", type=int, default=10, help="Show top-k by goodness/fairness")
    args = parser.parse_args()

    if args.dataset:
        print(f"Loading WSN from {args.dataset} ...")
        G = load_bitcoin_otc(args.dataset)
        print(f"Loaded graph: |V|={G.number_of_nodes()}  |E|={G.number_of_edges()}")
    else:
        print("No --dataset given, running toy example instead.")
        G = toy_example()

    fairness, goodness = compute_fairness_goodness_with_evaluation_weights(
        G, eps=args.eps, verbose=True
    )

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
