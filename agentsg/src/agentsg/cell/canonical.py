"""
Bet-free reindexing via the canonical (Delaunay/Selling) superbase -- Kurlin 2022.

The reduction-flip problem
--------------------------
Two settings of the SAME lattice, indexed on different frames of a serial
experiment, can Niggli-reduce to *different* canonical cells when the cell sits
near a reduction-cone boundary (a<->c or a<->b near-degeneracy). Methods that
recover the relating operator by enlarging a symmetry group with an angular
tolerance (LePage max_delta; the tolerance metric-automorphism coset used by
:mod:`agentsg.cell.ambiguity` and, in spirit, by dials.cosym) then make a *bet*:
the flip operator is recovered only if it happens to fall inside the enlarged
group. Past the tolerance -- a loose near-degeneracy, or noise larger than the
delta -- the operator drops out and the flip is silently missed.

The bet-free alternative
------------------------
The Delaunay/Selling reduction underlying Kurlin's root invariant is *continuous
across the flip*: both settings reduce to the SAME obtuse superbase (identical
conorms), so the reindexing operator is recovered by matching the two superbases
over the FINITE group of 24 index permutations of the four superbase vectors
(Kurlin 2022, Def 5.1) -- a fixed, bounded search, with no tolerance-thresholded
symmetry group to fall out of.

Concretely, tracking the superbase in INTEGER lattice coordinates (not Cartesian)
makes the recovered change of basis exact: if A and B are the same lattice, the
operator P with ``P^T G_A P = G_B`` is an exact integer matrix, recovered as
``P = U . W^{-1}`` where U, W are the integer superbase coordinates of A and B
under a conorm-matching index permutation. The metric is used only to drive the
reduction and to verify candidates; the operator itself is integer-exact.

Functions
---------
* :func:`canonical_superbase`   -- integer superbase coords + conorms of a cell.
* :func:`reindexing_via_canonical` -- the reindexing coset A -> B, bet-free.
* :func:`best_reindex_with_residual` -- best operator + raw metric residual, no
  threshold (the primitive for measuring deformation / calibrating tolerances).
* :func:`calibrate_verify_tol`  -- derive the residual threshold from a baseline
  of known same- and different-lattice pairs, because a relative/absolute
  tolerance is only meaningful relative to the scales present in the data.

These complement :func:`agentsg.cell.reindex.reindexing_operators` (brute
unimodular enumeration -- also exact, but O(6960) per call) and
:class:`agentsg.cell.ambiguity.ReindexingReference` (fast tolerance-coset,
which makes the bet). This module is the exact, tolerance-group-free route.

Reference: V. Kurlin, "A complete isometry classification of 3-dimensional
lattices", arXiv:2201.10543 (2022), Def 5.1 (24 index permutations), Lemma 6.2
(superbase reconstruction). Dependency-free; exact integer/rational arithmetic
for the operator, float only to drive the reduction.
"""
from __future__ import annotations
from fractions import Fraction
from itertools import permutations

from .metric import UnitCell

# the four superbase vectors are indexed 0..3 with v0 = -(v1+v2+v3);
# the six unordered pairs whose conorms characterise the lattice:
_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
# the 24 permutations of the four index labels (Kurlin Def 5.1)
_PERMS = tuple(permutations(range(4)))


def _metric(cell):
    return UnitCell(*cell).metric_tensor()


def _dotG(ci, cj, G):
    """Scalar product of two integer coordinate vectors under metric G."""
    return sum(ci[a] * G[a][b] * cj[b] for a in range(3) for b in range(3))


def canonical_superbase(cell, max_iter=1000, rel_eps=1e-9):
    """Delaunay/Selling-reduce ``cell`` to an obtuse superbase, tracked in
    INTEGER lattice coordinates.

    Returns ``(C, P)`` where

      * ``C`` is a list of the four superbase vectors as integer coordinate
        triples in the ORIGINAL cell basis (``C[0] = -(C[1]+C[2]+C[3])``), reduced
        so that every pairwise scalar product ``C[i].C[j] <= 0`` (obtuse), and
      * ``P`` is the ``4x4`` conorm matrix ``p_ij = -C[i].C[j]`` (>= 0 off the
        diagonal) evaluated under the cell metric.

    The reduction move (negate ``v_i``, add it to the two vectors other than
    ``v_j``) preserves the superbase sum and the lattice, so ``C`` is an exact
    integer description of the same lattice. A move fires only for a scalar
    product positive beyond ``rel_eps * scale`` so that an exactly-orthogonal cell
    (whose right angles evaluate as ~6e-17, not 0) does not wander.
    """
    G = _metric(cell)
    C = [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    scale = max(abs(_dotG(C[i], C[i], G)) for i in range(4)) or 1.0
    eps = rel_eps * scale
    for _ in range(max_iter):
        worst = eps
        wi = wj = -1
        for i in range(4):
            for j in range(i + 1, 4):
                d = _dotG(C[i], C[j], G)
                if d > worst:
                    worst = d
                    wi, wj = i, j
        if wi < 0:
            Pm = [[-_dotG(C[i], C[j], G) if i != j else 0.0 for j in range(4)]
                  for i in range(4)]
            return C, Pm
        i, j = wi, wj
        others = [k for k in range(4) if k != i and k != j]
        vi = C[i][:]
        C[i] = [-x for x in vi]
        for k in others:
            C[k] = [C[k][t] + vi[t] for t in range(3)]
    raise RuntimeError("Delaunay reduction did not converge")


def superbase_variants(cell, boundary_rel=1e-3, max_variants=64):
    """All obtuse superbases within ``boundary_rel`` of the reduced one.

    The obtuse (Delaunay/Selling) superbase is UNIQUE only in the interior of a
    Delaunay type. On a boundary -- a conorm ``p_ij = -v_i.v_j`` passing through
    zero (e.g. a monoclinic angle near 90 deg, where ``a.c ~ 0``) -- the Selling
    flip of that pair produces an equally-valid obtuse superbase. Two settings of
    one lattice straddling such a boundary reduce to genuinely DIFFERENT integer
    superbases, so matching within the 24 relabellings of a single superbase
    misses the operator relating them (the reduction-flip problem, reappearing
    inside the superbase). Enumerating the finite closure of boundary flips
    restores completeness.

    Returns a list of superbases (each four integer coordinate triples in the
    original cell basis). ``boundary_rel`` is relative to the largest squared
    superbase edge; raise it to treat a near-degeneracy as a boundary (needed for
    noisy cells), lower it toward 0 to keep only the single reduced superbase.
    """
    G = _metric(cell)
    C0, _ = canonical_superbase(cell)
    scale = max(abs(_dotG(C0[i], C0[i], G)) for i in range(4)) or 1.0
    tol = boundary_rel * scale

    def key(C):
        return tuple(tuple(v) for v in C)

    seen = {}
    frontier = [C0]
    while frontier and len(seen) < max_variants:
        C = frontier.pop()
        k = key(C)
        if k in seen:
            continue
        seen[k] = C
        for i in range(4):
            for j in range(i + 1, 4):
                if abs(_dotG(C[i], C[j], G)) <= tol:      # a Delaunay boundary
                    others = [t for t in range(4) if t != i and t != j]
                    Cn = [row[:] for row in C]
                    vi = Cn[i][:]
                    Cn[i] = [-x for x in vi]
                    for t in others:
                        Cn[t] = [Cn[t][s] + vi[s] for s in range(3)]
                    if key(Cn) not in seen:
                        frontier.append(Cn)
    return list(seen.values())


def _int_or_none(fracM):
    """If a 3x3 Fraction matrix is integral, return it as int tuple; else None."""
    out = []
    for row in fracM:
        r = []
        for x in row:
            if x.denominator != 1:
                return None
            r.append(int(x))
        out.append(tuple(r))
    return tuple(out)


def _inv3_frac(M):
    """Exact inverse of a 3x3 integer matrix as Fractions (or None if singular)."""
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if det == 0:
        return None
    adj = [[(e * i - f * h), -(b * i - c * h), (b * f - c * e)],
           [-(d * i - f * g), (a * i - c * g), -(a * f - c * d)],
           [(d * h - e * g), -(a * h - b * g), (a * e - b * d)]]
    return [[Fraction(adj[r][col], det) for col in range(3)] for r in range(3)]


def _matmul_frac(A, B):
    return [[sum(Fraction(A[r][k]) * B[k][col] for k in range(3))
             for col in range(3)] for r in range(3)]


def _transform_metric_int(G, P):
    """G' = P^T G P for an integer matrix P (float G)."""
    PtG = [[sum(P[k][r] * G[k][b] for k in range(3)) for b in range(3)]
           for r in range(3)]
    return [[sum(PtG[r][k] * P[k][b] for k in range(3)) for b in range(3)]
            for r in range(3)]


def _rootmat(p):
    """Root-product matrix r_ij = sqrt(max(p_ij, 0)) -- linear in Angstrom, so a
    length-scaled tolerance is meaningful (conorms are quadratic in length)."""
    from math import sqrt
    return [[sqrt(p[i][j]) if p[i][j] > 0 else 0.0 for j in range(4)]
            for i in range(4)]


def _roots_close(rA, rB, perm, tol):
    """Do A's root products match B's under label permutation ``perm`` within
    ``tol`` (Angstrom)?"""
    for i in range(4):
        for j in range(4):
            if abs(rA[i][j] - rB[perm[i]][perm[j]]) > tol:
                return False
    return True


def reindexing_via_canonical(cell_A, cell_B, boundary_rel=1e-3,
                             verify_rel=1e-6, verify_abs=0.0, conorm_tol=None):
    """Reindexing operators A -> B via canonical-superbase matching (bet-free).

    Returns the list of integer operators ``P`` (as row-tuples) with
    ``P^T G_A P == G_B`` -- the reindexing coset -- recovered by matching the two
    Delaunay superbases over the finite group of 24 index permutations, WITHOUT
    building any tolerance-thresholded symmetry group. Empty list if A and B are
    not the same lattice (no candidate operator reproduces B's metric within
    tolerance).

    Parameters
    ----------
    boundary_rel : float
        Relative tolerance (fraction of the largest squared superbase edge) for
        treating a conorm as a Delaunay boundary and enumerating the alternative
        obtuse superbase there. Default ``1e-3`` catches the near-90-deg
        monoclinic case; raise it for noisy cells whose boundary conorms are
        displaced by measurement error, lower it toward 0 to match only the single
        reduced superbase (faster, but incomplete on boundaries).
    conorm_tol : float, optional
        Unused (kept for backward compatibility); superseded by the variant
        enumeration, which is complete rather than a pruning heuristic.
    verify_rel : float
        RELATIVE tolerance on the metric residual ``|P^T G_A P - G_B|``, as a
        fraction of the metric scale ``tr|G_B|``. Default ``1e-6`` accepts only
        essentially-exact operators (two settings of the *same* lattice). The
        residual grows with any true lattice DEFORMATION between A and B (it is
        exact only when they are the identical lattice), so raise ``verify_rel``
        to treat near-lattices as matching -- e.g. for a short deformation hop
        between adjacent trajectory states. This is the one knob that decides
        "how different may B be and still count as A's lattice".
    verify_abs : float
        Absolute floor on the residual tolerance (default 0). The effective
        tolerance is ``max(verify_abs, verify_rel * tr|G_B|)``.

    Notes
    -----
    Because the operator is built as ``P = U . W^{-1}`` over exact integer
    superbase coordinates and inverted with rational arithmetic, an accepted
    ``P`` is an EXACT integer unimodular matrix -- there is no float operator to
    round. The tolerances act only on the (possibly noisy or deformed) metric
    comparison, never on the operator, which is always exact.
    """
    GA = _metric(cell_A)
    GB = _metric(cell_B)
    tol = max(verify_abs,
              verify_rel * (abs(GB[0][0]) + abs(GB[1][1]) + abs(GB[2][2])))

    # Enumerate obtuse-superbase VARIANTS of both cells. The reduced superbase is
    # unique only in the interior of a Delaunay type; on a boundary (a conorm
    # ~ 0, e.g. a monoclinic angle near 90 deg) two settings of one lattice
    # reduce to different superbases. Matching across the finite boundary-variant
    # closure of both cells is what makes this complete -- it removes the residual
    # "reduction flip inside the superbase" that a single-superbase match misses.
    vA = superbase_variants(cell_A, boundary_rel=boundary_rel)
    vB = superbase_variants(cell_B, boundary_rel=boundary_rel)

    found = set()
    for CA in vA:
        U = [[CA[1][r], CA[2][r], CA[3][r]] for r in range(3)]   # 3x3 int, columns
        for CB in vB:
            for perm in _PERMS:
                for s in (1, -1):
                    W = [[s * CB[perm[1]][r], s * CB[perm[2]][r], s * CB[perm[3]][r]]
                         for r in range(3)]
                    Winv = _inv3_frac(W)
                    if Winv is None:
                        continue
                    # P = U . W^{-1}: exact integer change of basis matching A's
                    # superbase (variant CA) to B's (variant CB) under this
                    # relabelling. P maps B-basis coords to A-basis, so
                    # P^T G_A P = G_B is the reindexing we want.
                    P = _int_or_none(_matmul_frac(U, Winv))
                    if P is None:
                        continue
                    det = (P[0][0] * (P[1][1] * P[2][2] - P[1][2] * P[2][1])
                           - P[0][1] * (P[1][0] * P[2][2] - P[1][2] * P[2][0])
                           + P[0][2] * (P[1][0] * P[2][1] - P[1][1] * P[2][0]))
                    if abs(det) != 1:
                        continue
                    Gp = _transform_metric_int(GA, P)
                    resid = max(abs(Gp[a][b] - GB[a][b])
                                for a in range(3) for b in range(3))
                    if resid <= tol:
                        found.add(tuple(tuple(int(x) for x in row) for row in P))
    return sorted(found)


def reindexing_operator_via_canonical(cell_A, cell_B, **kw):
    """One reindexing operator A -> B via canonical superbase, or None."""
    ops = reindexing_via_canonical(cell_A, cell_B, **kw)
    return ops[0] if ops else None


def calibrate_verify_tol(same_pairs, different_pairs=None):
    """Derive a data-driven ``verify_abs`` residual threshold from a baseline.

    A relative/absolute residual threshold is only meaningful relative to the
    residual scales actually present in the data: how far apart genuine
    same-lattice pairs sit (measurement noise + small deformation) versus how far
    unrelated lattices sit. This estimates the threshold from that baseline.

    Parameters
    ----------
    same_pairs : iterable of (cell_A, cell_B)
        Pairs KNOWN to be the same lattice (e.g. successive frames of one
        crystal, or adjacent states on a deformation trajectory). Their residuals
        set the upper scale a match may reach.
    different_pairs : iterable of (cell_A, cell_B), optional
        Pairs known to be DIFFERENT lattices. If given, the threshold is placed
        in the gap between the two residual populations (geometric mean of the
        same-pair max and the different-pair min), which both accepts every true
        match and rejects every true non-match when the populations separate.

    Returns
    -------
    dict with keys:
        ``verify_abs``   -- recommended absolute residual threshold,
        ``same_max``     -- largest residual among same-lattice pairs,
        ``diff_min``     -- smallest residual among different-lattice pairs
                            (``inf`` if none supplied),
        ``separated``    -- bool: do the two populations separate cleanly?
        ``same_residuals``, ``diff_residuals`` -- the raw residual lists.

    Notes
    -----
    Use the returned ``verify_abs`` as the ``verify_abs`` argument to
    :func:`reindexing_via_canonical` (with ``verify_rel=0`` to use the absolute
    floor alone). If ``separated`` is False the lattice family is genuinely
    ambiguous at this deformation scale -- no single threshold cleanly separates
    match from non-match, which is itself the useful diagnostic (it says the
    trajectory needs landmark routing, not a looser tolerance).
    """
    same_res = []
    for A, B in same_pairs:
        _, r = best_reindex_with_residual(A, B)
        same_res.append(r)
    diff_res = []
    if different_pairs is not None:
        for A, B in different_pairs:
            _, r = best_reindex_with_residual(A, B)
            diff_res.append(r)
    same_max = max(same_res) if same_res else 0.0
    diff_min = min(diff_res) if diff_res else float("inf")
    if diff_res and diff_min > same_max > 0:
        from math import sqrt as _sqrt
        verify_abs = _sqrt(same_max * diff_min)      # geometric mean of the gap
        separated = True
    elif diff_res and diff_min <= same_max:
        verify_abs = same_max                        # overlap: accept up to same-max
        separated = False
    else:
        verify_abs = same_max * 3.0 if same_max > 0 else 0.0   # no negatives given
        separated = bool(same_res)
    return {"verify_abs": verify_abs, "same_max": same_max, "diff_min": diff_min,
            "separated": separated, "same_residuals": same_res,
            "diff_residuals": diff_res}


def best_reindex_with_residual(cell_A, cell_B, boundary_rel=1e-3):
    """Best canonical reindexing operator A -> B and its metric residual.

    Unlike :func:`reindexing_via_canonical`, this applies NO acceptance
    threshold: it returns ``(P, resid)`` for the integer operator ``P`` that
    minimises ``|P^T G_A P - G_B|`` over all 48 signed index permutations, where
    ``resid`` is that minimum residual (in metric-tensor units). This is the
    primitive the manifold layer uses to MEASURE the deformation between two
    lattice states along a trajectory: ``resid`` is ~0 for the same lattice and
    grows smoothly with deformation, so a short hop has a small residual and a
    long hop a large one -- the signal that says "route through a landmark".
    Returns ``(None, inf)`` only if no integer unimodular candidate exists (which
    should not happen for valid cells).
    """
    GA = _metric(cell_A)
    GB = _metric(cell_B)
    vA = superbase_variants(cell_A, boundary_rel=boundary_rel)
    vB = superbase_variants(cell_B, boundary_rel=boundary_rel)
    best_P, best_res = None, float("inf")
    for CA in vA:
        U = [[CA[1][r], CA[2][r], CA[3][r]] for r in range(3)]
        for CB in vB:
            for perm in _PERMS:
                for s in (1, -1):
                    W = [[s * CB[perm[1]][r], s * CB[perm[2]][r], s * CB[perm[3]][r]]
                         for r in range(3)]
                    Winv = _inv3_frac(W)
                    if Winv is None:
                        continue
                    P = _int_or_none(_matmul_frac(U, Winv))
                    if P is None:
                        continue
                    det = (P[0][0] * (P[1][1] * P[2][2] - P[1][2] * P[2][1])
                           - P[0][1] * (P[1][0] * P[2][2] - P[1][2] * P[2][0])
                           + P[0][2] * (P[1][0] * P[2][1] - P[1][1] * P[2][0]))
                    if abs(det) != 1:
                        continue
                    Gp = _transform_metric_int(GA, P)
                    resid = max(abs(Gp[a][b] - GB[a][b])
                                for a in range(3) for b in range(3))
                    if resid < best_res:
                        best_res, best_P = resid, P
    return best_P, best_res
