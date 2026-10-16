"""Landmarks inside each linear-key cluster, then a Selling-closure Isomap.

Cells are the four-mode monoclinic mixture. In the sorted linear key those
modes form two clusters (beta 89° and 91° are the same key). Within each
cluster the keys are a multivariate normal. Landmarks are a 9×9 Gauss–Hermite
grid for a standard isotropic Gaussian, stretched by the square root of that
covariance and centred on the class mean.

Each landmark keeps its nearest cells. The exact Selling-closure distance
between those cells is a batch of aligned linear distances, run on the GPU
through CuPy when a device is present. Each linear-key cluster is then
embedded from its own distances, and the two maps are placed apart by the
median distance between the clusters.

Run::

    python agentsg/examples/monoclinic_landmarks.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from itertools import permutations

import numpy as np

from agentsg.cell.metric import metric_tensor
from agentsg.cell.selling_closure import selling_superbase_closure

from linear_knn_isomap import _classical_mds
from monoclinic_modes import linear_keys, monoclinic_cloud

HERE = Path(__file__).resolve().parent
HERMITE_ORDER = 9
VARIANCE_TOL = 1e-3
NEIGHBOURS = 4
_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_PAIR_INDEX = {tuple(sorted(ij)): slot for slot, ij in enumerate(_PAIRS)}
_SLOT = np.zeros((24, 6), dtype=np.int64)
for _p, _perm in enumerate(permutations(range(4))):
    for _s, (_i, _j) in enumerate(_PAIRS):
        _SLOT[_p, _s] = _PAIR_INDEX[tuple(sorted((_perm[_i], _perm[_j])))]


def _cupy():
    """CuPy module when a CUDA device is available, else None."""
    try:
        import cupy as cp
    except ImportError:
        return None
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
    except Exception:
        return None
    return cp


def key_clusters(keys):
    """Two clusters from the sign of the leading principal component.

    The clouds sit on opposite sides of the centred keys. Which side is
    cluster 0 depends on the SVD sign; callers that have the cells can swap
    so that cluster 0 is the smaller ``a``.
    """
    centred = np.asarray(keys, dtype=np.float64)
    centred = centred - centred.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    score = centred @ vt[0]
    labels = np.zeros(len(score), dtype=np.int64)
    labels[score >= 0] = 1
    return labels, score


def gaussian_landmarks(keys, order=HERMITE_ORDER, var_tol=VARIANCE_TOL):
    """Gauss–Hermite nodes of N(μ, Σ), placed in the principal axes of Σ.

    ``keys`` is ``(n, 6)``. Axes with eigenvalue below ``var_tol`` times the
    largest are dropped, so the tensor grid stays on the span that carries
    the class. Nodes are the probabilists' Gauss–Hermite points (the
    quadrature for a standard normal), then scaled by ``sqrt(λ)`` and rotated.
    """
    sample = np.asarray(keys, dtype=np.float64)
    mu = sample.mean(axis=0)
    cov = np.cov(sample, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    floor = var_tol * float(max(evals[-1], 0.0))
    keep = evals > floor
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    if not np.any(keep):
        return mu.reshape(1, -1), {
            "mean": mu,
            "eigenvalues": evals,
            "kept": 0,
            "nodes": nodes,
            "weights": weights,
        }
    lam = evals[keep]
    axes = evecs[:, keep]
    grids = np.meshgrid(*([nodes] * len(lam)), indexing="ij")
    standard = np.column_stack([grid.ravel() for grid in grids])
    stretched = standard * np.sqrt(np.maximum(lam, 0.0))
    points = mu + stretched @ axes.T
    return points, {
        "mean": mu,
        "eigenvalues": evals,
        "kept": int(len(lam)),
        "nodes": nodes,
        "weights": weights,
    }


def nearest_cells(keys, landmarks, k):
    """Unique row indices of the ``k`` nearest keys to each landmark.

    The distance matrix is evaluated on the GPU when CuPy has a device.
    """
    keys = np.asarray(keys, dtype=np.float64)
    landmarks = np.asarray(landmarks, dtype=np.float64)
    cp = _cupy()
    if cp is not None and len(keys) > 0 and len(landmarks) > 0:
        host_keys = cp.asarray(keys)
        host_marks = cp.asarray(landmarks)
        delta = host_marks[:, None, :] - host_keys[None, :, :]
        nearest = cp.argpartition(cp.sum(delta * delta, axis=-1), kth=k - 1, axis=1)[:, :k]
        return np.unique(cp.asnumpy(nearest).ravel())
    from scipy.spatial import cKDTree

    tree = cKDTree(keys)
    _dists, nbrs = tree.query(landmarks, k=k)
    return np.unique(np.atleast_2d(np.asarray(nbrs, dtype=np.int64)).ravel())


def closure_conorms(cell):
    """Normalised conorms of each Selling-closure superbase, shape ``(m, 6)``."""
    metric = metric_tensor(cell)
    members = selling_superbase_closure(cell)
    out = np.empty((len(members), 6), dtype=np.float64)
    for row, superbase in enumerate(members):
        conorm = np.empty(6, dtype=np.float64)
        for slot, (i, j) in enumerate(_PAIRS):
            left = superbase[i]
            right = superbase[j]
            dot = 0.0
            for a in range(3):
                for b in range(3):
                    dot += left[a] * metric[a][b] * right[b]
            conorm[slot] = -dot
        total = float(conorm.sum())
        out[row] = conorm / np.sqrt(total) if total > 0.0 else conorm
    return out


def pairwise_closure_distances(conorms):
    """Exact closure distances. The reduction runs on the GPU when possible.

    ``conorms`` is a list of ``(m, 6)`` arrays, one Selling closure per cell.
    Padding past each closure is ignored. The result is symmetric, in Å.
    """
    n = len(conorms)
    width = max(array.shape[0] for array in conorms)
    packed = np.zeros((n, width, 6), dtype=np.float64)
    counts = np.empty(n, dtype=np.int32)
    for i, array in enumerate(conorms):
        packed[i, :array.shape[0]] = array
        counts[i] = array.shape[0]
    cp = _cupy()
    if cp is None:
        return _pairwise_closure_numpy(packed, counts)
    slot = cp.asarray(_SLOT)
    bank = cp.asarray(packed)
    count = cp.asarray(counts)
    permuted = bank[:, :, slot]
    valid = cp.arange(width)[None, :] < count[:, None]
    dist = cp.empty((n, n), dtype=cp.float64)
    for i in range(n):
        query = bank[i, :int(counts[i])]
        diff = query[:, None, None, None, :] - permuted[None]
        gap = cp.linalg.norm(diff, axis=-1)
        gap = cp.where(valid[None, :, :, None], gap, cp.inf)
        dist[i] = gap.min(axis=(0, 2, 3))
    out = cp.asnumpy(dist)
    out = np.minimum(out, out.T)
    np.fill_diagonal(out, 0.0)
    return out


def _pairwise_closure_numpy(packed, counts):
    """Same reduction as :func:`pairwise_closure_distances`, on the CPU."""
    n, width, _six = packed.shape
    permuted = packed[:, :, _SLOT]
    dist = np.empty((n, n), dtype=np.float64)
    valid = np.arange(width)[None, :] < counts[:, None]
    for i in range(n):
        query = packed[i, :int(counts[i])]
        diff = query[:, None, None, None, :] - permuted[None]
        gap = np.linalg.norm(diff, axis=-1)
        gap = np.where(valid[None, :, :, None], gap, np.inf)
        dist[i] = gap.min(axis=(0, 2, 3))
    dist = np.minimum(dist, dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def closure_graph(cells):
    """Pairwise exact Selling-closure distances."""
    n = len(cells)
    started = time.perf_counter()
    conorms = [closure_conorms(tuple(float(x) for x in row)) for row in cells]
    enumerated = time.perf_counter() - started
    device = "GPU" if _cupy() is not None else "CPU"
    started = time.perf_counter()
    full = pairwise_closure_distances(conorms)
    compared = time.perf_counter() - started
    print(
        f"  closures enumerated in {enumerated:.2f}s; "
        f"{n * (n - 1) // 2} distances on {device} in {compared:.2f}s",
        flush=True,
    )
    return full


def embed_clusters(full, labels):
    """One closure-distance map per linear-key cluster, then the gap between them.

    Neighbours are not allowed to cross clusters. The inner Gauss–Hermite
    landmarks of one class sit closer to the other class than to the far side
    of their own, so a shared neighbour graph erases the partition. Each
    class is embedded from its own distances, and the second class is shifted
    by the median cross-cluster closure distance.
    """
    out = np.zeros((len(full), 2), dtype=np.float64)
    labels = np.asarray(labels)
    for lab in np.unique(labels):
        nodes = np.flatnonzero(labels == lab)
        xy = _classical_mds(full[np.ix_(nodes, nodes)])
        xy -= xy.mean(axis=0)
        out[nodes] = xy
    sides = np.unique(labels)
    if len(sides) < 2:
        return out, 0.0
    left = np.flatnonzero(labels == sides[0])
    right = np.flatnonzero(labels == sides[1])
    gap = float(np.median(full[np.ix_(left, right)]))
    out[right, 0] += gap
    return out, gap


def nearest_landmark_cells(keys, landmarks):
    """Row index of the single nearest cell to each landmark, in landmark order."""
    keys = np.asarray(keys, dtype=np.float64)
    landmarks = np.asarray(landmarks, dtype=np.float64)
    cp = _cupy()
    if cp is not None and len(keys) > 0 and len(landmarks) > 0:
        delta = cp.asarray(landmarks)[:, None, :] - cp.asarray(keys)[None, :, :]
        return cp.asnumpy(cp.argmin(cp.sum(delta * delta, axis=-1), axis=1))
    from scipy.spatial import cKDTree

    _dists, nbrs = cKDTree(keys).query(landmarks, k=1)
    return np.asarray(nbrs, dtype=np.int64)


def linear_key_matrix(keys):
    """Euclidean distances between sorted linear keys, in Å.

    The pairwise reduction runs on the GPU when CuPy has a device.
    """
    sample = np.asarray(keys, dtype=np.float64)
    cp = _cupy()
    if cp is not None and len(sample) > 0:
        bank = cp.asarray(sample)
        delta = bank[:, None, :] - bank[None, :, :]
        return cp.asnumpy(cp.sqrt(cp.sum(delta * delta, axis=-1)))
    delta = sample[:, None, :] - sample[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def _block_lines(ax, labels):
    """White cuts between contiguous cluster blocks."""
    edges = np.flatnonzero(np.diff(np.asarray(labels))) + 1
    for edge in edges:
        ax.axhline(edge - 0.5, color="white", linewidth=1.0)
        ax.axvline(edge - 0.5, color="white", linewidth=1.0)


def write_key_comparison_png(closure, linear, labels, path):
    """Closure matrix, linear-key matrix, and closure minus linear key."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    closure = np.asarray(closure, dtype=np.float64)
    linear = np.asarray(linear, dtype=np.float64)
    gap = closure - linear
    shared = max(float(closure.max()), float(linear.max()))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    panels = (
        (axes[0], closure, "Selling closure", "viridis", 0.0, shared, "Å"),
        (axes[1], linear, "Sorted linear key", "viridis", 0.0, shared, "Å"),
        (axes[2], gap, "Closure minus linear key", "magma", None, None, "Å"),
    )
    for ax, values, title, cmap, vmin, vmax, unit in panels:
        image = ax.imshow(values, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
        _block_lines(ax, labels)
        ax.set_title(title)
        ax.set_xlabel("Nearest cell")
        ax.set_xticks([])
        ax.set_yticks([])
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label(unit)
    axes[0].set_ylabel("Nearest cell")
    fig.suptitle("Landmark distances: closure and linear key")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_distance_matrix_png(matrix, labels, path):
    """Heat map of landmark closure distances, blocked by the cell's cluster.

    ``labels`` follow the nearest cell, not the landmark that queried it.
    The matrix is already ordered so each cluster is one contiguous block.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(labels)
    names = {0: "a ≈ 30 Å", 1: "a ≈ 35 Å"}
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, cmap="viridis", origin="upper")
    _block_lines(ax, labels)
    edges = np.flatnonzero(np.diff(labels)) + 1
    cuts = np.concatenate([[0], edges, [len(labels)]])
    ticks = []
    ticklabels = []
    for start, stop in zip(cuts[:-1], cuts[1:]):
        if stop <= start:
            continue
        ticks.append(0.5 * (start + stop - 1))
        ticklabels.append(names.get(int(labels[start]), str(int(labels[start]))))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    ax.set_xlabel("Nearest cell")
    ax.set_ylabel("Nearest cell")
    ax.set_title("Selling-closure distance between landmarks (Å)")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Closure distance (Å)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_isomap_png(
    xy, cluster, path,
    title="Closure-distance embedding of each cluster",
    xlabel="Coordinate 1 (Å)",
    ylabel="Coordinate 2 (Å)",
):
    """Isomap scatter coloured by linear-key cluster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    cmap = plt.get_cmap("tab10")
    xy = np.asarray(xy, dtype=np.float64)
    for lab, name in ((0, "a ≈ 30 Å"), (1, "a ≈ 35 Å")):
        mask = np.asarray(cluster) == lab
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=36, color=cmap(lab), label=name, zorder=2,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def landmark_isomap(n=20000, seed=0, k_neighbours=NEIGHBOURS):
    """Fit class landmarks, keep their nearest cells, and embed the closures."""
    cells, _mixture = monoclinic_cloud(n, seed=seed)
    keys, key_seconds = linear_keys(cells)
    clusters, _score = key_clusters(keys)
    if cells[clusters == 0, 0].mean() > cells[clusters == 1, 0].mean():
        clusters = 1 - clusters
    selected = []
    reports = []
    for lab in (0, 1):
        block = keys[clusters == lab]
        points, info = gaussian_landmarks(block)
        hits = nearest_cells(keys, points, k_neighbours)
        crossed = int(np.sum(clusters[hits] != lab))
        selected.append(hits)
        reports.append({
            "n": int(len(block)),
            "landmarks": points,
            "kept_axes": info["kept"],
            "eigenvalues": info["eigenvalues"],
            "neighbours": hits,
            "crossed": crossed,
            "mean_a": float(cells[clusters == lab, 0].mean()),
        })
        print(
            f"cluster {lab}: n={len(block):,}, mean a={reports[-1]['mean_a']:.2f} Å, "
            f"{info['kept']} axes, {len(points)} landmarks, {len(hits)} cells, "
            f"{crossed} from the other cluster",
            flush=True,
        )
    chosen = np.unique(np.concatenate(selected))
    print(f"{len(chosen)} cells after merging neighbours", flush=True)
    # One nearest cell per landmark. Its label is the cell's own linear-key
    # cluster, which can differ from the landmark's cluster.
    pairs = []
    for lab, report in enumerate(reports):
        found = nearest_landmark_cells(keys, report["landmarks"])
        for source, cell_index in zip(np.full(len(found), lab), found):
            pairs.append((int(clusters[cell_index]), int(source), int(cell_index)))
    pairs.sort()
    landmark_index = np.array([cell_index for _cell_lab, _source, cell_index in pairs], dtype=np.int64)
    landmark_labels = np.array([cell_lab for cell_lab, _source, _cell_index in pairs], dtype=np.int64)
    crossed_landmarks = sum(cell_lab != source for cell_lab, source, _i in pairs)
    print(
        f"{len(pairs)} landmarks, {crossed_landmarks} nearest cells belong to the other cluster",
        flush=True,
    )
    landmark_matrix = closure_graph(cells[landmark_index])
    landmark_xy = _classical_mds(landmark_matrix)
    landmark_xy -= landmark_xy.mean(axis=0)
    landmark_linear = linear_key_matrix(keys[landmark_index])
    gap = landmark_matrix - landmark_linear
    upper = np.triu_indices(len(gap), k=1)
    print(
        f"closure minus linear key: mean {gap[upper].mean():.4f} Å, "
        f"max {gap[upper].max():.4f} Å",
        flush=True,
    )
    full = closure_graph(cells[chosen])
    cluster = clusters[chosen]
    xy, gap = embed_clusters(full, cluster)
    print(f"median closure distance between clusters {gap:.2f} Å", flush=True)
    return {
        "cells": cells[chosen],
        "cluster": cluster,
        "xy": xy,
        "closure": full,
        "landmark_matrix": landmark_matrix,
        "landmark_xy": landmark_xy,
        "landmark_linear": landmark_linear,
        "landmark_index": landmark_index,
        "landmark_labels": landmark_labels,
        "reports": reports,
        "key_seconds": key_seconds,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    parser.add_argument(
        "--out", type=Path, default=HERE / "out" / "monoclinic_landmarks_isomap.png",
    )
    parser.add_argument(
        "--matrix-out", type=Path,
        default=HERE / "out" / "monoclinic_landmark_distances.png",
    )
    parser.add_argument(
        "--embed-out", type=Path,
        default=HERE / "out" / "monoclinic_closure_embedding.png",
    )
    parser.add_argument(
        "--compare-out", type=Path,
        default=HERE / "out" / "monoclinic_landmark_key_vs_closure.png",
    )
    args = parser.parse_args(argv)
    result = landmark_isomap(
        n=args.n, seed=args.seed,
        k_neighbours=args.neighbours,
    )
    write_isomap_png(
        result["xy"], result["cluster"], args.out,
        xlabel="Isomap 1 (closure distance, Å)",
        ylabel="Isomap 2 (closure distance, Å)",
    )
    write_distance_matrix_png(
        result["landmark_matrix"], result["landmark_labels"], args.matrix_out,
    )
    write_key_comparison_png(
        result["landmark_matrix"], result["landmark_linear"],
        result["landmark_labels"], args.compare_out,
    )
    write_isomap_png(
        result["landmark_xy"], result["landmark_labels"], args.embed_out,
        title="Classical MDS of the Selling-closure matrix",
        xlabel="MDS 1 (Å)",
        ylabel="MDS 2 (Å)",
    )
    print(args.out)
    print(args.matrix_out)
    print(args.embed_out)
    print(args.compare_out)
    return result


if __name__ == "__main__":
    main()
