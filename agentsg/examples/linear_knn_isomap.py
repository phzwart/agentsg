"""Nearest-neighbour graph on the sorted linear key, rescored by Selling closure.

Pass any cells. A Niggli-reduced cell is the usual input. Two changes of
basis are stored separately and are not multiplied here:

1. :func:`to_selling` — submitted cell to one obtuse cell, and that COB.
2. :func:`closure_cobs` — the Selling closure of that obtuse cell, every COB
   in the reduced basis.

The KD-tree proposes neighbours in the sorted linear key
``sort(p_ij / sqrt(Σ p))``, in Å. Each proposed pair is then measured with
:func:`agentsg.cell.canonical.closure_distance`, which walks both full
Selling-superbase closures. Those neighbour distances are the edges of an
Isomap, drawn as a 2-D scatter. Combining a step-1 COB with a step-2 COB
is a later step (:func:`combine_cobs`).

With ``boundary_rel=0`` the closure distance is at least the linear-key
distance, so the candidate list widens until the k-th closure distance sits
inside the linear-key ball. With ``boundary_rel>0`` near-obtuse superbases
are admitted and that lower bound does not hold. The candidate window is
then ``k * margin`` (default 4). That window is a search margin, not a proof.

Run::

    python agentsg/examples/linear_knn_isomap.py
    python agentsg/examples/linear_knn_isomap.py --boundary-rel 0.001

Requires ``pip install -e ".[db,plot]"`` (scipy and matplotlib).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agentsg.cell.canonical import closure_distance
from agentsg.cell.primitive import primitive_cell
from agentsg.cell.rootform import sorted_linear_key

from primitive_batch import primitive_cells
from selling_steps import closure_cobs, to_selling

HERE = Path(__file__).resolve().parent

IDEAL = {
    "cubic": (5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
    "tetragonal": (5.0, 5.0, 8.0, 90.0, 90.0, 90.0),
    "hexagonal": (5.0, 5.0, 8.0, 90.0, 90.0, 120.0),
    "trigonal": (5.0, 5.0, 5.0, 70.0, 70.0, 70.0),
    "orthorhombic": (5.0, 6.0, 7.0, 90.0, 90.0, 90.0),
    "monoclinic": (5.0, 6.0, 7.0, 90.0, 95.0, 90.0),
    "triclinic": (5.0, 6.0, 7.0, 80.0, 85.0, 95.0),
}


def _require_scipy():
    try:
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components, shortest_path
    except ImportError as exc:
        raise ImportError(
            'linear_knn_isomap needs scipy: pip install -e ".[db]"'
        ) from exc
    return cKDTree, csr_matrix, connected_components, shortest_path


def _candidates(tree, key, i, take):
    """Indices and linear-key distances of ``take`` neighbours other than ``i``."""
    n = tree.n
    dists, nbrs = tree.query(key, k=min(take + 1, n))
    dists = np.atleast_1d(np.asarray(dists, dtype=np.float64))
    nbrs = np.atleast_1d(np.asarray(nbrs, dtype=np.int64))
    js = []
    linear = []
    for j, dist in zip(nbrs.tolist(), dists.tolist()):
        j = int(j)
        if j == i or not np.isfinite(dist):
            continue
        js.append(j)
        linear.append(float(dist))
        if len(js) == take:
            break
    return js, linear


def _cached_closure(i, j, cells, boundary_rel, cache):
    key = (i, j) if i < j else (j, i)
    dist = cache.get(key)
    if dist is None:
        dist = float(closure_distance(cells[i], cells[j], boundary_rel=boundary_rel)[1])
        cache[key] = dist
    return dist


def _best(js, dists, k):
    order = np.argsort(np.asarray(dists, dtype=np.float64), kind="mergesort")
    take = min(k, len(order))
    sel = order[:take]
    return [js[t] for t in sel], [float(dists[t]) for t in sel]


def _neighbours_one(i, tree, keys, cells, k, boundary_rel, margin, cache):
    """k other cells nearest to ``i`` under the Selling-closure distance."""
    limit = len(cells) - 1
    if limit <= 0:
        return [], []
    if boundary_rel > 0:
        take = min(limit, max(k, int(np.ceil(k * margin))))
        js, _linear = _candidates(tree, keys[i], i, take)
        dists = [_cached_closure(i, j, cells, boundary_rel, cache) for j in js]
        return _best(js, dists, k)

    take = min(limit, max(2 * k, k + 1))
    while True:
        js, linear = _candidates(tree, keys[i], i, take)
        dists = [_cached_closure(i, j, cells, boundary_rel, cache) for j in js]
        chosen, chosen_d = _best(js, dists, k)
        if not chosen_d:
            return [], []
        radius = max(linear) if linear else 0.0
        certified = chosen_d[-1] < radius or len(js) >= limit
        if certified and len(chosen_d) >= min(k, limit):
            return chosen, chosen_d
        if len(js) >= limit:
            return chosen, chosen_d
        grown = min(limit, max(len(js) * 2, len(js) + 1))
        if grown <= len(js):
            return chosen, chosen_d
        take = grown


def _classical_mds(dist):
    """Two classical-MDS coordinates of a dense distance matrix."""
    d2 = np.asarray(dist, dtype=np.float64) ** 2
    row = d2.mean(axis=1, keepdims=True)
    gram = -0.5 * (d2 - row - row.T + d2.mean())
    gram = 0.5 * (gram + gram.T)
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    coords = np.zeros((len(dist), 2), dtype=np.float64)
    for axis in range(2):
        ev = float(evals[order[axis]])
        if ev <= 1e-12:
            break
        coords[:, axis] = evecs[:, order[axis]] * np.sqrt(ev)
    return coords


def _isomap(indices, distances, csr_matrix, connected_components, shortest_path):
    """Isomap of an undirected k-neighbour graph. Pieces sit side by side."""
    n, k = indices.shape
    coords = np.zeros((n, 2), dtype=np.float64)
    if n < 2:
        return coords
    acc: dict[tuple[int, int], list[float]] = {}
    for i in range(n):
        for t in range(k):
            j = int(indices[i, t])
            dist = float(distances[i, t])
            if j < 0 or j == i or not np.isfinite(dist):
                continue
            edge = (i, j) if i < j else (j, i)
            acc.setdefault(edge, []).append(dist)
    if not acc:
        return coords
    rows = []
    cols = []
    data = []
    for (a, b), vals in acc.items():
        weight = float(np.mean(vals))
        rows.extend((a, b))
        cols.extend((b, a))
        data.extend((weight, weight))
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    graph.sum_duplicates()
    _n_comp, labels = connected_components(graph, directed=False)
    pieces = [np.flatnonzero(labels == lab) for lab in range(int(labels.max()) + 1)]
    pieces.sort(key=len, reverse=True)
    cursor = 0.0
    started = False
    for nodes in pieces:
        if len(nodes) == 1:
            xy = np.zeros((1, 2), dtype=np.float64)
        else:
            dist = shortest_path(graph[nodes][:, nodes], directed=False)
            finite = np.isfinite(dist)
            if not finite.all():
                cap = float(np.max(dist[finite])) if finite.any() else 1.0
                dist = np.where(finite, dist, cap * 2.0)
            xy = _classical_mds(dist)
            xy = xy - xy.mean(axis=0)
        width = float(xy[:, 0].max() - xy[:, 0].min()) if len(xy) else 0.0
        if started:
            xy = xy.copy()
            xy[:, 0] += cursor - float(xy[:, 0].min())
        coords[nodes] = xy
        cursor = float(coords[nodes, 0].max()) + 0.18 * max(width, 1.0)
        started = True
    return coords


def linear_knn_isomap(cells, symbols=None, *, k=8, boundary_rel=0.0, margin=4.0):
    """k nearest lattices under the Selling-closure distance, and their Isomap.

    Parameters
    ----------
    cells :
        Any cells, shape ``(n, 6)``. A Niggli cell is the usual case.
        Selling reduction is not required of the caller. Step 1 stores it.
    symbols :
        Centering letters or space-group strings, aligned with the rows.
        Missing entries stay as given.
    k :
        Neighbours stored per cell, self excluded.
    boundary_rel :
        Near-obtuse width passed to :func:`closure_distance`. ``0`` uses the
        exact closure and a certified candidate expansion. A positive value
        admits near-zero conorm flips and switches to the ``margin`` window.
    margin :
        Used only when ``boundary_rel > 0``. The KD-tree returns
        ``ceil(k * margin)`` candidates. Default 4.

    Returns
    -------
    dict
        ``primitive`` ``(n, 6)`` is the input after centring removal, still
        in the submitted setting (Niggli, if that is what was passed).
        ``selling`` is step 1, one dict per cell (``reduced``, ``cob``).
        ``closure_cobs`` is step 2, a list of reduced-basis matrices per cell.
        Those two are not multiplied. ``indices`` ``(n, k)`` (``-1`` if
        missing), ``distances`` ``(n, k)`` in Å, ``xy`` ``(n, 2)``.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if boundary_rel < 0:
        raise ValueError("boundary_rel must be >= 0")
    if boundary_rel > 0 and margin < 1:
        raise ValueError("margin must be >= 1 when boundary_rel > 0")
    cKDTree, csr_matrix, connected_components, shortest_path = _require_scipy()
    prim = primitive_cells(cells, symbols)
    n = prim.shape[0]
    selling = []
    closures = []
    for row in prim:
        step = to_selling(tuple(float(x) for x in row))
        selling.append(step)
        closures.append(closure_cobs(step["reduced"], boundary_rel=boundary_rel))
    keys = np.array([sorted_linear_key(step["cell"]) for step in selling], dtype=np.float64)
    if n == 0:
        empty = np.zeros((0, k), dtype=np.int64)
        return {
            "primitive": prim,
            "selling": selling,
            "closure_cobs": closures,
            "indices": empty,
            "distances": np.zeros((0, k), dtype=np.float64),
            "xy": np.zeros((0, 2), dtype=np.float64),
        }
    tree = cKDTree(keys)
    cells_t = [tuple(float(x) for x in row) for row in prim]
    indices = np.full((n, k), -1, dtype=np.int64)
    distances = np.full((n, k), np.inf, dtype=np.float64)
    cache: dict[tuple[int, int], float] = {}
    for i in range(n):
        chosen, chosen_d = _neighbours_one(
            i, tree, keys, cells_t, k, boundary_rel, margin, cache,
        )
        indices[i, :len(chosen)] = chosen
        distances[i, :len(chosen_d)] = chosen_d
    xy = _isomap(indices, distances, csr_matrix, connected_components, shortest_path)
    return {
        "primitive": prim,
        "selling": selling,
        "closure_cobs": closures,
        "indices": indices,
        "distances": distances,
        "xy": xy,
    }


def demo_cloud(copies=2, seed=0):
    """Ideal cells, a few noisy copies, and a C-centred cell with its primitive twin.

    Returns ``(cells, symbols, labels)``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    symbols = []
    labels = []
    for name, cell in IDEAL.items():
        rows.append(cell)
        symbols.append(None)
        labels.append(name)
        for _ in range(copies):
            edge = np.array(cell[:3], dtype=np.float64) * (1.0 + rng.normal(0.0, 0.01, size=3))
            ang = np.array(cell[3:], dtype=np.float64) + rng.normal(0.0, 0.4, size=3)
            rows.append(tuple(edge.tolist()) + tuple(ang.tolist()))
            symbols.append(None)
            labels.append(name)
    conventional = (10.0, 12.0, 7.0, 90.0, 90.0, 90.0)
    rows.append(conventional)
    symbols.append("C 2 2 21")
    labels.append("C conventional")
    rows.append(tuple(primitive_cell(conventional, "C")))
    symbols.append(None)
    labels.append("C primitive")
    return np.asarray(rows, dtype=np.float64), symbols, labels


def write_png(xy, labels, path: Path):
    """Scatter the Isomap. One colour per label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    names = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab10")
    for t, name in enumerate(names):
        mask = np.array([lab == name for lab in labels])
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=28, color=cmap(t % 10), label=name, zorder=2,
        )
    ax.set_xlabel("Isomap 1")
    ax.set_ylabel("Isomap 2")
    ax.set_title("Selling-closure neighbour distances (Å)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--boundary-rel", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=4.0)
    parser.add_argument(
        "--out", type=Path, default=HERE / "out" / "linear_knn_isomap.png",
    )
    args = parser.parse_args(argv)
    cells, symbols, labels = demo_cloud()
    result = linear_knn_isomap(
        cells, symbols, k=args.k, boundary_rel=args.boundary_rel, margin=args.margin,
    )
    write_png(result["xy"], labels, args.out)
    twin = result["distances"][-2]
    twin_d = float(twin[twin < np.inf].min()) if np.isfinite(twin).any() else float("inf")
    print(
        f"{len(cells)} cells, k={args.k}, boundary_rel={args.boundary_rel:g}, "
        f"C cell to its primitive twin {twin_d:.4g} Å"
    )
    print(args.out)
    return result


if __name__ == "__main__":
    main()
