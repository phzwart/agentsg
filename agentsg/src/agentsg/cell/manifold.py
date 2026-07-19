"""
Deformation-manifold scaffold: landmarks (inducing points) for path-consistent
reindexing across continuous lattice deformation.

Why this exists
---------------
Under a continuous deformation of a lattice -- a shrinkage series, a thermal or
pressure trajectory, an operando sweep -- each state is a *slightly different*
lattice. A pairwise reindexing operator P between two states satisfies
``P^T G_1 P = G_2`` only approximately, and the residual GROWS with the
deformation (it is exact only when the two lattices are identical; see
:func:`agentsg.cell.canonical.best_reindex_with_residual`). So two ends of a long
trajectory can be far enough apart that no single integer operator relates them
within tolerance, even though every ADJACENT pair matches cleanly. The reindexing
is path-dependent: compose adjacent operators along the path and the frame is
consistent; compare the ends directly and they "don't match up" (monodromy).

The fix is a scaffold of LANDMARKS (inducing points, in the sparse-GP sense):
fix a small set of reference states spanning the manifold, express every observed
state relative to its NEAREST landmark (a short, low-residual hop where the
operator is cleanly integer), and propagate the landmark-to-landmark frame along
the graph. All comparisons route through the scaffold, never through long direct
hops, so the global frame is path-consistent.

Two layers (mirroring the reindexing design)
--------------------------------------------
* The ROOT INVARIANT (isometry invariant, continuous across reduction flips)
  gives the path-consistent manifold COORDINATE and the distance used to build
  the deformation graph. It is blind to the setting, so it never flips.
* The FIXED-SETTING METRIC locates SYMMETRY JUNCTIONS -- states where the lattice
  momentarily gains symmetry during the deformation (a metric degeneracy, e.g.
  a == c or an angle -> 90 deg). The invariant cannot see these by construction,
  so they are found on the metric. Symmetry junctions are physically
  distinguished landmarks: they are where the deformation graph branches.

Landmarks with a physical reading
---------------------------------
Landmarks are placed at physically-distinguished states, not arbitrary points:
  * ENDPOINTS / extremal states  -- extrema of the deformation coordinate,
  * SYMMETRY JUNCTIONS           -- higher-symmetry states along the path,
  * (optionally) FARTHEST-POINT samples to fill large gaps evenly.
The deformation graph's Fiedler coordinate (2nd Laplacian eigenvector) orders the
states along the dominant deformation axis, turning "shrinkage state" into a
continuous coordinate rather than a discrete label.

Dependencies
------------
The graph, landmark selection, and path-consistent reindexing are pure-Python and
dependency-free. The optional spectral (Fiedler) coordinate uses numpy if it is
importable; without it, :meth:`DeformationManifold.fiedler_coordinate` raises and
the farthest-point / shortest-path machinery still works.
"""
from __future__ import annotations
from heapq import heappush, heappop

from .rootform import root_invariant, root_distance
from .canonical import best_reindex_with_residual, reindexing_via_canonical


# --------------------------------------------------------------- graph core ----
def deformation_graph(cells, k=None, distances=None):
    """k-nearest-neighbour graph on the root-invariant distances between cells.

    Each edge connects states related by a small physical deformation step; the
    edge weight is the root-invariant distance (linear in Angstrom, a physical
    displacement in lattice-edge space). Returns ``(D, adj)`` where ``D`` is the
    full symmetric distance matrix (list of lists) and ``adj`` is the adjacency
    ``{i: {j: dist, ...}}`` of the kNN graph (symmetrised).

    ``k`` defaults to ``min(len-1, max(2, round(sqrt(len))))``. Pass a precomputed
    ``distances`` (root-invariant distance matrix) to skip recomputation.
    """
    n = len(cells)
    if distances is None:
        RI = [root_invariant(c) for c in cells]
        D = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = sum((RI[i][t] - RI[j][t]) ** 2 for t in range(len(RI[i]))) ** 0.5
                D[i][j] = D[j][i] = d
    else:
        D = [list(row) for row in distances]
    if k is None:
        k = min(n - 1, max(2, round(n ** 0.5)))
    adj = {i: {} for i in range(n)}
    for i in range(n):
        order = sorted((j for j in range(n) if j != i), key=lambda j: D[i][j])
        for j in order[:k]:
            w = D[i][j]
            adj[i][j] = w
            adj[j][i] = w              # symmetrise
    return D, adj


def _dijkstra(adj, src, n):
    """Shortest-path distances from src over the weighted graph (pure Python)."""
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heappush(pq, (nd, v))
    return dist, prev


# ------------------------------------------------------- symmetry junctions ----
def _metric_degeneracies(cell):
    """Fixed-setting metric near-degeneracy signals that flag a symmetry junction.

    Returns ``(length_signal, angle_signal)`` in Angstrom-comparable units:
      length_signal = smallest pairwise |edge_i - edge_j|  (e.g. a == c),
      angle_signal  = smallest |angle - 90| or |angle - 120|, scaled to length by
                      the mean edge.
    Small => close to a higher-symmetry metric. These are FIXED-SETTING
    quantities -- the root invariant is blind to them, so junctions are found on
    the metric. The two signals are kept SEPARATE so a constant offset in one
    (e.g. a fixed monoclinic beta across a whole trajectory) does not mask a
    genuine local degeneracy in the other.
    """
    a, b, c, al, be, ga = cell
    length_signal = min(abs(a - b), abs(a - c), abs(b - c))
    angle_deg = min(abs(al - 90.0), abs(be - 90.0), abs(ga - 90.0),
                    abs(al - 120.0), abs(be - 120.0), abs(ga - 120.0))
    mean_edge = (a + b + c) / 3.0
    angle_signal = angle_deg * mean_edge * 3.14159265 / 180.0
    return length_signal, angle_signal


def symmetry_junctions(cells, rel_tol=0.02):
    """Indices of states that are STRICT local minima of a metric-degeneracy
    signal AND within ``rel_tol`` (fraction of mean edge) of an actual
    degeneracy. These are the higher-symmetry states along the path.

    A strict local minimum is required (``< both neighbours``, not ``<=``): a
    constant signal -- e.g. a monoclinic beta that never changes along the
    trajectory -- is FLAT, and a flat region has no strict interior minimum, so
    it produces no spurious junctions. Only a signal that genuinely dips (a
    length or angle passing through a degeneracy) is flagged. The length and
    angle signals are tested independently: a junction fires if EITHER dips to a
    strict local minimum below tolerance, so a real length degeneracy is not
    hidden by a constant nonzero angle offset (or vice versa).
    """
    n = len(cells)
    sigs = [_metric_degeneracies(c) for c in cells]
    junc = []
    for i in range(n):
        mean_edge = sum(cells[i][:3]) / 3.0
        thr = rel_tol * mean_edge
        is_junction = False
        for s in (0, 1):                       # length signal, then angle signal
            here = sigs[i][s]
            lo = sigs[i - 1][s] if i > 0 else float("inf")
            hi = sigs[i + 1][s] if i < n - 1 else float("inf")
            # strict interior minimum, or an endpoint that is itself degenerate
            interior_min = here < lo and here < hi
            endpoint_deg = (i == 0 or i == n - 1) and here <= thr and (
                here < (hi if i == 0 else lo))
            if here <= thr and (interior_min or endpoint_deg):
                is_junction = True
                break
        if is_junction:
            junc.append(i)
    return junc


# ------------------------------------------------------ landmark selection ----
def farthest_point_landmarks(D, n_landmarks, seed=0):
    """Farthest-point (k-center) sampling on the distance matrix: greedily pick
    the state farthest from all current landmarks. Guarantees even coverage.
    """
    n = len(D)
    n_landmarks = min(n_landmarks, n)
    chosen = [seed]
    while len(chosen) < n_landmarks:
        best_i, best_d = -1, -1.0
        for i in range(n):
            if i in chosen:
                continue
            d = min(D[i][c] for c in chosen)
            if d > best_d:
                best_d, best_i = d, i
        if best_i < 0:
            break
        chosen.append(best_i)
    return sorted(chosen)


def select_landmarks(cells, D=None, n_fill=0, include_junctions=True):
    """Physically-motivated landmark set: endpoints (extrema of the deformation
    coordinate) + symmetry junctions + optional farthest-point fill.

    Returns a sorted list of landmark indices. The endpoints are the two states
    farthest apart on the manifold (the extremes of the deformation); junctions
    are the higher-symmetry states; fill points spread evenly to cover gaps.
    """
    n = len(cells)
    if D is None:
        D, _ = deformation_graph(cells)
    # endpoints: the pair of maximum mutual root-invariant distance
    i0 = j0 = 0
    best = -1.0
    for i in range(n):
        for j in range(i + 1, n):
            if D[i][j] > best:
                best, i0, j0 = D[i][j], i, j
    landmarks = {i0, j0}
    if include_junctions:
        landmarks.update(symmetry_junctions(cells))
    if n_fill > 0:
        for idx in farthest_point_landmarks(D, len(landmarks) + n_fill, seed=i0):
            landmarks.add(idx)
    return sorted(landmarks)


# ------------------------------------------------- path-consistent reindex ----
class DeformationManifold:
    """A landmark scaffold over a set of lattice states for path-consistent
    reindexing across continuous deformation.

    Construct from a list of unit cells (a trajectory, or a sampled ensemble).
    The scaffold builds the deformation graph, selects landmarks, and provides:

    * :meth:`nearest_landmark`     -- the landmark closest to a state (short hop),
    * :meth:`reindex_to_landmark`  -- the exact integer operator state -> landmark
                                      (canonical, low-residual because the hop is
                                      short),
    * :meth:`fiedler_coordinate`   -- the deformation coordinate (needs numpy),
    * :attr:`landmarks`            -- the landmark indices, with their physical
                                      role (endpoint / junction / fill).
    """

    def __init__(self, cells, k=None, n_fill=0, include_junctions=True):
        self.cells = [tuple(c) for c in cells]
        self.D, self.adj = deformation_graph(self.cells, k=k)
        self.landmarks = select_landmarks(self.cells, self.D, n_fill=n_fill,
                                          include_junctions=include_junctions)
        self._junctions = set(symmetry_junctions(self.cells)) if include_junctions else set()
        # endpoints = the extreme pair
        eps = select_landmarks(self.cells, self.D, n_fill=0, include_junctions=False)
        self._endpoints = set(eps)

    # -- roles ---------------------------------------------------------------
    def landmark_role(self, idx):
        """'endpoint', 'junction', or 'fill' for a landmark index."""
        if idx in self._endpoints:
            return "endpoint"
        if idx in self._junctions:
            return "junction"
        return "fill"

    # -- nearest landmark (graph shortest path, not direct distance) ---------
    def nearest_landmark(self, i):
        """Landmark reachable from state ``i`` by the shortest graph path (the
        physically shortest deformation route, not the direct chord)."""
        dist, _ = _dijkstra(self.adj, i, len(self.cells))
        return min(self.landmarks, key=lambda L: dist[L])

    def path_to_landmark(self, i):
        """(landmark, node path) from state ``i`` to its nearest landmark along
        the graph -- the chain of short deformation hops."""
        dist, prev = _dijkstra(self.adj, i, len(self.cells))
        L = min(self.landmarks, key=lambda Lk: dist[Lk])
        path = [L]
        u = L
        # prev encodes the tree rooted at i, so walk back from L to i
        while u != i and prev[u] != -1:
            u = prev[u]
            path.append(u)
        path.reverse()
        return L, path

    # -- exact reindexing operator for a short hop ---------------------------
    def reindex_to_landmark(self, i, verify_rel=1e-3):
        """Exact integer operator ``P`` reindexing state ``i`` onto its nearest
        landmark's setting, plus the metric residual and the landmark index.

        Because the landmark is the nearest state, the hop is short and the
        canonical operator is essentially exact (small residual). Returns
        ``(landmark_index, P, residual)`` with ``P is None`` if even the nearest
        landmark is beyond ``verify_rel`` (the state is off the sampled manifold).
        """
        L = self.nearest_landmark(i)
        P, resid = best_reindex_with_residual(self.cells[i], self.cells[L])
        from .metric import UnitCell
        GB = UnitCell(*self.cells[L]).metric_tensor()
        tol = verify_rel * (abs(GB[0][0]) + abs(GB[1][1]) + abs(GB[2][2]))
        return (L, P, resid) if (P is not None and resid <= tol) else (L, None, resid)

    # -- composed, path-consistent operator ---------------------------------
    def reindex_along_path(self, i):
        """Path-consistent operator from state ``i`` to its landmark, composed
        from the short adjacent hops along the graph path (NOT a single direct
        operator). Returns ``(landmark, P_total, max_hop_residual)``.

        Composing short hops keeps every intermediate residual small, so the
        end-to-end operator is recovered even when a direct ``i -> landmark``
        match would exceed tolerance (the monodromy fix).
        """
        L, path = self.path_to_landmark(i)
        # accumulate P so that G_path[m+1] = P_step^T G_path[m] P_step;
        # total maps state i onto landmark L.
        P_total = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        max_res = 0.0
        for a, b in zip(path[:-1], path[1:]):
            ops = reindexing_via_canonical(self.cells[a], self.cells[b],
                                           verify_rel=1e-2)
            if not ops:
                Pstep, res = best_reindex_with_residual(self.cells[a], self.cells[b])
            else:
                Pstep = ops[0]
                from .metric import UnitCell
                GA = UnitCell(*self.cells[a]).metric_tensor()
                GB = UnitCell(*self.cells[b]).metric_tensor()
                Gp = [[sum((sum(Pstep[u][r] * GA[u][w] for u in range(3))) * Pstep[w][c]
                            for w in range(3)) for c in range(3)] for r in range(3)]
                res = max(abs(Gp[u][w] - GB[u][w]) for u in range(3) for w in range(3))
            max_res = max(max_res, res)
            P_total = _matmul_int(P_total, Pstep)
        return L, tuple(tuple(row) for row in P_total), max_res

    def fiedler_coordinate(self):
        """The deformation coordinate: 2nd-smallest Laplacian eigenvector of the
        (Gaussian-weighted) deformation graph. Orders states along the dominant
        deformation axis. Requires numpy.
        """
        try:
            import numpy as np
        except ImportError as e:
            raise RuntimeError("fiedler_coordinate requires numpy") from e
        n = len(self.cells)
        vals = [self.D[i][j] for i in range(n) for j in range(n) if i != j and self.D[i][j] > 0]
        med = sorted(vals)[len(vals) // 2] if vals else 1.0
        W = np.zeros((n, n))
        for i in range(n):
            for j, w in self.adj[i].items():
                aff = np.exp(-(w ** 2) / (2 * med ** 2))
                W[i, j] = W[j, i] = aff
        Lap = np.diag(W.sum(axis=1)) - W
        evals, evecs = np.linalg.eigh(Lap)
        return evecs[:, 1]


def _matmul_int(A, B):
    return [[sum(A[r][k] * B[k][c] for k in range(3)) for c in range(3)]
            for r in range(3)]
