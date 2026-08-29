"""KD-tree index over precomputed Kurlin root invariants (6D Euclidean, Å).

The root invariant is a fixed 6-vector per lattice, so ``scipy.spatial.cKDTree``
gives exact nearest-neighbour / radius queries with much lower overhead than the
pure-Python :class:`~agentsg.cell.neartree.NearTree` (which remains for
arbitrary metrics such as boundary-aware G6/S6 distances).

Requires scipy (``pip install agentsg[db]``).
"""
from __future__ import annotations

import numpy as np

from .primitive import primitive_cell
from .rootform import root_invariant


def _primitive_for_roots(cell, sg_hm):
    if not sg_hm:
        return cell
    try:
        return primitive_cell(cell, sg_hm)
    except (ValueError, ZeroDivisionError):
        return cell


def build_root_index(roots_with_ids):
    """Build a :class:`RootIndex` from ``(root_tuple, payload)`` pairs.

    ``root_tuple`` is a length-6 sequence ``(r0,..,r5)``; ``payload`` is
    returned with query results (typically a PDB id).
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "RootIndex needs scipy: pip install agentsg[db]") from exc
    roots_with_ids = list(roots_with_ids)
    if not roots_with_ids:
        tree = cKDTree(np.empty((0, 6)))
        return RootIndex(tree, [])
    roots = np.asarray([r[0] for r in roots_with_ids], dtype=np.float64)
    ids = [r[1] for r in roots_with_ids]
    return RootIndex(cKDTree(roots), ids)


class RootIndex:
    """Prebuilt cKDTree over root invariants for fast repeated cell queries.

    Construct via :func:`build_root_index` or
    :meth:`~agentsg.cell.celldb.CellDatabase.build_index`. Queries take a cell
    ``(a,b,c,alpha,beta,gamma)``; the query root is computed once per call and
    matched against stored roots by exact Euclidean distance (Å).
    """

    __slots__ = ("_tree", "_ids")

    def __init__(self, tree, ids):
        self._tree = tree
        self._ids = ids

    def __len__(self):
        return len(self._ids)

    def _query_root(self, cell, sg_hm=None):
        return np.asarray(
            root_invariant(_primitive_for_roots(cell, sg_hm)), dtype=np.float64)

    def k_nearest(self, cell, k=10, sg_hm=None):
        """Return the k nearest ``(payload, distance)`` to ``cell``."""
        if not self._ids:
            return []
        k = min(k, len(self._ids))
        q = self._query_root(cell, sg_hm)
        dists, idxs = self._tree.query(q, k=k)
        if k == 1:
            return [(self._ids[int(idxs)], float(dists))]
        return [(self._ids[int(i)], float(d)) for d, i in zip(dists, idxs)]

    def nearest(self, cell, sg_hm=None):
        """Return the single nearest ``(payload, distance)`` to ``cell``."""
        return self.k_nearest(cell, k=1, sg_hm=sg_hm)[0]

    def within(self, cell, radius, sg_hm=None):
        """Return all ``(payload, distance)`` within ``radius`` of ``cell``."""
        if not self._ids:
            return []
        q = self._query_root(cell, sg_hm)
        idxs = self._tree.query_ball_point(q, radius, return_sorted=True)
        out = [
            (self._ids[i], float(np.linalg.norm(q - self._tree.data[i])))
            for i in idxs
        ]
        out.sort(key=lambda t: t[1])
        return out

    # aliases used by lattice_index
    def nearest_cell(self, cell, sg_hm=None):
        return self.nearest(cell, sg_hm=sg_hm)

    def k_nearest_cells(self, cell, k, sg_hm=None):
        return self.k_nearest(cell, k=k, sg_hm=sg_hm)

    def within_cells(self, cell, radius, sg_hm=None):
        return self.within(cell, radius, sg_hm=sg_hm)
