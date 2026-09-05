"""
NearTree: a metric-space nearest-neighbour index for the lattice manifold.

Once lattices are points in G6/S6 with a boundary-aware distance
(:mod:`agentsg.cell.g6`), "find lattices like this one" becomes a
nearest-neighbour query on a manifold. A NearTree (Andrews 2001, "A template for
the nearest neighbour problem"; the structure underlying Andrews & Bernstein's
SAUC lattice search) indexes points so that queries prune large parts of the
space using only the triangle inequality -- no coordinates or embedding assumed,
just a metric. That is exactly right here: the boundary-aware G6/S6 distance is a
true metric but is NOT a plain Euclidean norm (it minimises over boundary
transforms), so a KD-tree on the raw 6 coordinates would give wrong neighbours
near reduction boundaries, while a NearTree on the metric itself is correct.

This is a pure-Python, dependency-free implementation. It is exact (returns true
nearest neighbours, not approximate). For persistence and SQL prefiltering over a
large database (e.g. all of the PDB) see :mod:`agentsg.cell.celldb`, which stores
the S6 coordinates in DuckDB and hands a candidate set to a NearTree for exact
ranking.

Reference: M. G. Andrews, J. Appl. Cryst. 34, 663-668 (2001).
"""
from __future__ import annotations


class NearTree:
    """Exact nearest-neighbour index over an arbitrary metric.

    Parameters
    ----------
    distance : callable(a, b) -> float
        A metric on the payload objects. Must satisfy the triangle inequality
        for correctness (the boundary-aware G6/S6 distances do).

    Insert points with :meth:`insert`, query with :meth:`nearest` (single) or
    :meth:`k_nearest` / :meth:`within` (batch). Each point carries an arbitrary
    ``payload`` returned with the result.
    """

    __slots__ = ("_distance", "_left", "_right", "_left_max", "_right_max",
                 "_left_obj", "_right_obj", "_n")

    def __init__(self, distance):
        self._distance = distance
        self._left = None          # left child NearTree
        self._right = None         # right child NearTree
        self._left_max = -1.0      # radius of left subtree
        self._right_max = -1.0     # radius of right subtree
        self._left_obj = None      # (point, payload) anchor for left
        self._right_obj = None     # (point, payload) anchor for right
        self._n = 0

    def __len__(self):
        return self._n

    def insert(self, point, payload=None):
        """Insert a point (any object the distance function accepts)."""
        obj = (point, payload)
        if self._left_obj is None:
            self._left_obj = obj
            self._n += 1
            return
        if self._right_obj is None:
            self._right_obj = obj
            self._n += 1
            return
        dl = self._distance(point, self._left_obj[0])
        dr = self._distance(point, self._right_obj[0])
        if dl <= dr:
            if self._left is None:
                self._left = NearTree(self._distance)
            if dl > self._left_max:
                self._left_max = dl
            self._left.insert(point, payload)
        else:
            if self._right is None:
                self._right = NearTree(self._distance)
            if dr > self._right_max:
                self._right_max = dr
            self._right.insert(point, payload)
        self._n += 1

    def nearest(self, query, radius=float("inf")):
        """Return (payload, distance) of the single nearest point within radius.

        Returns (None, inf) if the tree is empty or nothing is within radius.
        """
        best = [radius, None]
        self._nearest(query, best)
        return (best[1], best[0]) if best[1] is not None else (None, float("inf"))

    def _nearest(self, query, best):
        """Recursive branch-and-bound nearest neighbor search using triangle inequality."""
        if self._left_obj is not None:
            d = self._distance(query, self._left_obj[0])
            if d < best[0]:
                best[0] = d; best[1] = self._left_obj[1]
        if self._right_obj is not None:
            d = self._distance(query, self._right_obj[0])
            if d < best[0]:
                best[0] = d; best[1] = self._right_obj[1]
        # branch-and-bound with the triangle inequality
        if self._left is not None and self._left_obj is not None:
            dl = self._distance(query, self._left_obj[0])
            if dl - self._left_max <= best[0]:
                self._left._nearest(query, best)
        if self._right is not None and self._right_obj is not None:
            dr = self._distance(query, self._right_obj[0])
            if dr - self._right_max <= best[0]:
                self._right._nearest(query, best)

    def within(self, query, radius):
        """Return all (payload, distance) with distance <= radius, sorted."""
        out = []
        self._within(query, radius, out)
        out.sort(key=lambda t: t[1])
        return out

    def _within(self, query, radius, out):
        """Recursive range search collecting all points within radius into out."""
        if self._left_obj is not None:
            d = self._distance(query, self._left_obj[0])
            if d <= radius:
                out.append((self._left_obj[1], d))
        if self._right_obj is not None:
            d = self._distance(query, self._right_obj[0])
            if d <= radius:
                out.append((self._right_obj[1], d))
        if self._left is not None and self._left_obj is not None:
            dl = self._distance(query, self._left_obj[0])
            if dl - self._left_max <= radius:
                self._left._within(query, radius, out)
        if self._right is not None and self._right_obj is not None:
            dr = self._distance(query, self._right_obj[0])
            if dr - self._right_max <= radius:
                self._right._within(query, radius, out)

    def k_nearest(self, query, k):
        """Return the k nearest (payload, distance), sorted by distance.

        Simple exact approach: collect all within an expanding radius is
        overkill; here we gather everything via a bounded traversal that keeps
        the k best. For large k or large trees prefer a DuckDB prefilter
        (celldb) to shrink the candidate set first.
        """
        heap = []  # list of (distance, payload), kept sorted, length <= k
        self._knn(query, k, heap)
        return [(p, d) for d, p in heap]

    def _knn(self, query, k, heap):
        """Recursive bounded search maintaining the top-k nearest neighbors."""
        def consider(obj):
            """Evaluate candidate object against current top-k heap."""
            if obj is None:
                return
            d = self._distance(query, obj[0])
            if len(heap) < k:
                heap.append((d, obj[1])); heap.sort(key=lambda t: t[0])
            elif d < heap[-1][0]:
                heap[-1] = (d, obj[1]); heap.sort(key=lambda t: t[0])
        consider(self._left_obj)
        consider(self._right_obj)
        bound = heap[-1][0] if len(heap) == k else float("inf")
        if self._left is not None and self._left_obj is not None:
            dl = self._distance(query, self._left_obj[0])
            if dl - self._left_max <= bound:
                self._left._knn(query, k, heap)
        if self._right is not None and self._right_obj is not None:
            dr = self._distance(query, self._right_obj[0])
            if dr - self._right_max <= bound:
                self._right._knn(query, k, heap)


def build_neartree(points, distance):
    """Build a NearTree from an iterable of (point, payload) pairs.

    Insertion order affects balance; for large sets shuffle first for a more
    balanced tree (the index is exact regardless of balance).
    """
    t = NearTree(distance)
    for point, payload in points:
        t.insert(point, payload)
    return t


# ----------------------------------------------------------------------------
# Lattice index: a KD-tree keyed on the Kurlin root invariant
# ----------------------------------------------------------------------------
def lattice_index(cells_with_ids):
    """Build a KD-tree over lattices, keyed on the root invariant.

    Parameters
    ----------
    cells_with_ids : iterable of (cell, payload)
        ``cell`` is a (a,b,c,alpha,beta,gamma) tuple; ``payload`` is anything
        you want back (e.g. a PDB id).

    The index uses the root invariant (Kurlin 2022) as the point and plain
    Euclidean distance on it as the metric -- a single vector per lattice, no
    orbit minimisation, continuous across the reduction-flip boundary. Queries
    (``nearest_cell``/``k_nearest_cells``/``within_cells``) take a query *cell*;
    distances are in Angstrom (root-product units).

    Requires scipy (``pip install agentsg[db]``).
    """
    from .rootindex import build_root_index
    from .rootform import root_invariant
    pts = [(root_invariant(cell), payload) for cell, payload in cells_with_ids]
    return build_root_index(pts)
