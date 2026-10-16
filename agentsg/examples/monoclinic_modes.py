"""Four monoclinic clusters, and the cost of comparing a million of them.

The cloud is a mixture of four modes. ``a`` sits at 30 Å or 35 Å, each with
standard deviation 0.75 Å. ``beta`` sits at 89° or 91°, each with standard
deviation 0.5°. ``b``, ``c``, ``alpha`` and ``gamma`` stay fixed, so the two
choices of ``a`` and the two choices of ``beta`` make four clusters.

The comparison that is timed on the whole list is the sorted linear key and
a KD-tree of its neighbours. A Selling-closure distance on this near-right
angle is much slower, so that distance is timed on a few pairs only.

Run::

    python agentsg/examples/monoclinic_modes.py
    python agentsg/examples/monoclinic_modes.py --svd --n 24000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from agentsg.cell.canonical import closure_distance
from agentsg.cell.rootform import sorted_linear_key

A_MEANS = (30.0, 35.0)
A_SIGMA = 0.75
BETA_MEANS = (89.0, 91.0)
BETA_SIGMA = 0.5
B = 40.0
C = 55.0
ALPHA = 90.0
GAMMA = 90.0
CLUSTER_NAMES = (
    "a=30, beta=89",
    "a=30, beta=91",
    "a=35, beta=89",
    "a=35, beta=91",
)


def monoclinic_cloud(n, seed=0):
    """``n`` monoclinic cells from the four-mode mixture.

    Returns ``cells`` of shape ``(n, 6)`` and integer ``labels`` in ``0..3``.
    Label ``2 * ia + ib`` is mean ``a`` index ``ia`` and mean ``beta`` index ``ib``.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, 2, size=n)
    ib = rng.integers(0, 2, size=n)
    a = np.array(A_MEANS, dtype=np.float64)[ia] + rng.normal(0.0, A_SIGMA, size=n)
    beta = np.array(BETA_MEANS, dtype=np.float64)[ib] + rng.normal(0.0, BETA_SIGMA, size=n)
    cells = np.column_stack([
        a,
        np.full(n, B, dtype=np.float64),
        np.full(n, C, dtype=np.float64),
        np.full(n, ALPHA, dtype=np.float64),
        beta,
        np.full(n, GAMMA, dtype=np.float64),
    ])
    return cells, ia * 2 + ib


def linear_keys(cells):
    """Sorted linear keys, and the seconds they took."""
    started = time.perf_counter()
    keys = np.empty((len(cells), 6), dtype=np.float64)
    for i, row in enumerate(cells):
        keys[i] = sorted_linear_key(row)
    return keys, time.perf_counter() - started


def time_linear_neighbours(cells, k=8):
    """Build keys, a KD-tree, and the k neighbours of every cell.

    Returns the neighbour indices, distances, and a timing dict. Distances
    are ångström in the sorted linear key. Self is excluded.
    """
    from scipy.spatial import cKDTree

    n = len(cells)
    if k < 1:
        raise ValueError("k must be at least 1")
    keys, key_seconds = linear_keys(cells)
    started = time.perf_counter()
    tree = cKDTree(keys)
    build_seconds = time.perf_counter() - started
    take = min(k + 1, n)
    started = time.perf_counter()
    dists, nbrs = tree.query(keys, k=take, workers=-1)
    query_seconds = time.perf_counter() - started
    dists = np.atleast_2d(np.asarray(dists, dtype=np.float64))
    nbrs = np.atleast_2d(np.asarray(nbrs, dtype=np.int64))
    indices = np.full((n, k), -1, dtype=np.int64)
    distances = np.full((n, k), np.inf, dtype=np.float64)
    for i in range(n):
        kept = 0
        for j, dist in zip(nbrs[i].tolist(), dists[i].tolist()):
            j = int(j)
            if j == i or not np.isfinite(dist):
                continue
            indices[i, kept] = j
            distances[i, kept] = dist
            kept += 1
            if kept == k:
                break
    return {
        "indices": indices,
        "distances": distances,
        "key_seconds": key_seconds,
        "build_seconds": build_seconds,
        "query_seconds": query_seconds,
    }


def svd_embedding(cells):
    """Mean-centred SVD of the sorted linear keys.

    Scores are ångström in the basis of the first two right singular vectors.
    The same construction as :func:`agentsg.serve.scatter.root_svd`.
    """
    from agentsg.serve.scatter import root_svd

    keys, key_seconds = linear_keys(cells)
    fit = root_svd(keys)
    fit["key_seconds"] = key_seconds
    return fit


def write_svd_png(scores, labels, variance_frac, path):
    """Scatter of PC1 against PC2, one colour per mixture component."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    cmap = plt.get_cmap("tab10")
    xy = np.asarray(scores, dtype=np.float64)
    for lab, name in enumerate(CLUSTER_NAMES):
        mask = np.asarray(labels) == lab
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=8, alpha=0.45, color=cmap(lab), label=name, linewidths=0,
        )
    pc1 = 100.0 * variance_frac[0] if variance_frac else 0.0
    pc2 = 100.0 * variance_frac[1] if len(variance_frac) > 1 else 0.0
    ax.set_xlabel(f"PC1 ({pc1:.1f}% of linear-key variance, Å)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}% of linear-key variance, Å)")
    ax.set_title("SVD of the sorted linear key")
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def time_closure_pairs(cells, pairs):
    """Seconds for each Selling-closure distance. ``pairs`` are index couples."""
    seconds = []
    distances = []
    for i, j in pairs:
        started = time.perf_counter()
        dist = closure_distance(tuple(cells[i]), tuple(cells[j]), boundary_rel=0.0)[1]
        seconds.append(time.perf_counter() - started)
        distances.append(float(dist))
    return seconds, distances


def _sample_pairs(labels, n_pairs, seed):
    """One pair inside a cluster and the rest across clusters, when possible."""
    rng = np.random.default_rng(seed)
    pairs = []
    for lab in range(4):
        members = np.flatnonzero(labels == lab)
        if len(members) >= 2 and len(pairs) < n_pairs:
            pick = rng.choice(members, size=2, replace=False)
            pairs.append((int(pick[0]), int(pick[1])))
    while len(pairs) < n_pairs:
        i, j = rng.integers(0, len(labels), size=2)
        if i != j:
            pairs.append((int(i), int(j)))
    return pairs[:n_pairs]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--closure-pairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--svd", action="store_true", help="plot the linear-key SVD and stop")
    parser.add_argument(
        "--svd-out", type=Path,
        default=Path(__file__).resolve().parent / "out" / "monoclinic_svd.png",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    cells, labels = monoclinic_cloud(args.n, seed=args.seed)
    draw_seconds = time.perf_counter() - started
    counts = np.bincount(labels, minlength=4)
    print(f"{args.n:,} cells in {draw_seconds:.3f}s")
    for name, count in zip(CLUSTER_NAMES, counts):
        print(f"  {name}: {int(count):,}")
    if args.svd:
        fit = svd_embedding(cells)
        frac = fit["svd"]["variance_frac"]
        write_svd_png(fit["scores"], labels, frac, args.svd_out)
        print(
            f"linear keys {fit['key_seconds']:.2f}s; "
            f"PC1 {100 * frac[0]:.1f}%, PC2 {100 * frac[1]:.1f}% of key variance"
        )
        print(args.svd_out)
        return {"cells": cells, "labels": labels, "svd": fit}
    result = time_linear_neighbours(cells, k=args.k)
    compared = args.n * args.k
    query = result["query_seconds"]
    print(
        f"linear keys {result['key_seconds']:.2f}s "
        f"({result['key_seconds'] / args.n * 1e6:.1f} µs/cell)"
    )
    print(f"KD-tree build {result['build_seconds']:.2f}s")
    print(
        f"k={args.k} neighbour query {query:.2f}s "
        f"({compared:,} comparisons, {query / compared * 1e6:.2f} µs each)"
    )
    pairs = _sample_pairs(labels, args.closure_pairs, args.seed)
    seconds, dists = time_closure_pairs(cells, pairs)
    mean_closure = float(np.mean(seconds))
    print(
        f"Selling-closure distance on {len(pairs)} pairs: "
        f"{mean_closure * 1e3:.1f} ms/pair"
    )
    full = mean_closure * compared
    print(
        f"rescoring all {compared:,} neighbours that way would take "
        f"{full / 3600:.1f} h"
    )
    return {
        "cells": cells,
        "labels": labels,
        "linear": result,
        "closure_seconds": seconds,
        "closure_distances": dists,
    }


if __name__ == "__main__":
    main()
