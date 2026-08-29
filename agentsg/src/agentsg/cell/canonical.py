"""
Bet-free reindexing via the typed Selling-superbase closure (main_v5 / Kurlin).

Search vs certification
-----------------------
Continuous Euclidean retrieval uses the *sorted* six root products
(:mod:`agentsg.cell.rootform`). Exact identity and reindexing use the finite
type-dependent Selling-superbase closure (:mod:`agentsg.cell.selling_closure`):
one isometry class for V1 (S4 x {+/-I}), and 2/3/3/4 non-isometric classes for
V2--V5. Matching over that closure recovers an exact integer ``P`` with
``P^T G_A P = G_B`` without tolerance-gated symmetry enumeration.

``superbase_variants`` remains available as an optional *noise* expander for
near-boundary measured cells; it is not the definition of the typed closure.

Functions
---------
* :func:`canonical_superbase`   -- integer superbase coords + conorms of a cell.
* :func:`reindexing_via_canonical` -- the reindexing coset A -> B over the
  Selling-superbase closure.
* :func:`best_reindex_with_residual` -- best operator + raw metric residual.
* :func:`calibrate_verify_tol`  -- data-driven residual threshold.

These complement :func:`agentsg.cell.reindex.reindexing_operators` (brute
unimodular enumeration) and :class:`agentsg.cell.ambiguity.ReindexingReference`
(tolerance-coset). Reference: Kurlin Lemmas 4.1--4.5; manuscript/main_v5.tex.
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
    """Optional noise expander: obtuse superbases near the reduced one.

    For exact typed closure use :func:`agentsg.cell.selling_closure.selling_superbase_closure`.
    This BFS expands Selling flips whenever a conorm is within ``boundary_rel`` of
    zero — useful for measured cells whose near-orthogonal angles are displaced
    by noise. It is *not* Kurlin's type-dependent class list (default
    ``max_variants=64`` can miss members of a 32-element V5 closure).

    Returns a list of superbases (each four integer coordinate triples in the
    original cell basis).
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


def _inv3_unimod(M):
    """Exact INTEGER inverse of a unimodular (det = +-1) integer matrix.

    Returns None if ``det`` is not +-1. For a unimodular integer matrix the
    inverse is ``det * adjugate`` and is itself integer, so no rational
    arithmetic is needed. This is the fast path for lattice reindexing, where
    the matrix W is built from three superbase vectors and is always unimodular
    (any three of the four zero-sum superbase vectors form a primitive basis).
    Fractional-coordinate / space-group work, where the denominator is genuinely
    non-trivial, must use :func:`_inv3_frac` instead.
    """
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if det not in (1, -1):
        return None
    adj = [[(e * i - f * h), -(b * i - c * h), (b * f - c * e)],
           [-(d * i - f * g), (a * i - c * g), -(a * f - c * d)],
           [(d * h - e * g), -(a * h - b * g), (a * e - b * d)]]
    return [[adj[r][col] * det for col in range(3)] for r in range(3)]


def _matmul_int(A, B):
    """Integer 3x3 matrix product (no rational arithmetic)."""
    return [[sum(A[r][k] * B[k][col] for k in range(3))
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


def _closure_for_match(cell, boundary_rel=0.0, use_typed_closure=True,
                       angle_sigma=None):
    """Full typed Selling-superbase closure, plus optional noise variants.

    Matching must run over the *full* closure, not class representatives:
    members within one isometry class are related by lattice automorphisms,
    and those automorphisms *are* the reindexing coset. Representatives alone
    under-count the coset on high-symmetry cells.

    When ``boundary_rel > 0``, near-zero conorm flips from
    :func:`superbase_variants` are merged in for noisy measured cells.
    ``angle_sigma`` (degrees) widens the typed-closure zero tolerance so
    near-symmetric noisy cells are classified as V2--V5 rather than V1.
    """
    from .selling_closure import selling_superbase_closure
    seen = {}
    if use_typed_closure:
        for C in selling_superbase_closure(cell, angle_sigma=angle_sigma):
            seen[tuple(tuple(v) for v in C)] = C
        if boundary_rel > 0:
            for C in superbase_variants(cell, boundary_rel=boundary_rel):
                seen[tuple(tuple(v) for v in C)] = C
    else:
        br = boundary_rel if boundary_rel > 0 else 1e-3
        for C in superbase_variants(cell, boundary_rel=br):
            seen[tuple(tuple(v) for v in C)] = C
    if not seen:
        C0, _ = canonical_superbase(cell)
        seen[tuple(tuple(v) for v in C0)] = C0
    return list(seen.values())


def reindexing_via_canonical(cell_A, cell_B, boundary_rel=None,
                             verify_rel=1e-6, verify_abs=0.0, conorm_tol=None,
                             use_typed_closure=True, angle_sigma=None):
    """Reindexing operators A -> B via Selling-superbase closure matching.

    Returns the list of integer operators ``P`` (as row-tuples) with
    ``P^T G_A P == G_B`` -- the reindexing coset -- recovered by matching the
    finite typed Selling-superbase closures of A and B over S4 index
    permutations and central inversion. Empty list if A and B are not the same
    lattice within tolerance.

    Parameters
    ----------
    boundary_rel : float or None
        Relative tolerance for the optional :func:`superbase_variants` noise
        expander. Default ``None`` uses ``1e-3`` (serial / measured cells).
        Pass ``0`` for exact archive metrics so only the typed Kurlin closure
        is used. Raise further (e.g. ``1e-2``) for very noisy frames.
    angle_sigma : float, optional
        Angular noise σ in degrees for typed-closure zero classification.
        When set, near-zero conorms within a few σ of the invariant noise
        floor are treated as zeros so V2--V5 types are recovered on noisy
        high-symmetry cells.
    use_typed_closure : bool
        If True (default), start from the full
        :func:`~agentsg.cell.selling_closure.selling_superbase_closure`.
        If False, fall back to ``superbase_variants`` only (legacy).
    conorm_tol : float, optional
        Unused (kept for backward compatibility).
    verify_rel : float
        RELATIVE tolerance on the metric residual ``|P^T G_A P - G_B|``, as a
        fraction of the metric scale ``tr|G_B|``. Default ``1e-6``.
    verify_abs : float
        Absolute floor on the residual tolerance (default 0). The effective
        tolerance is ``max(verify_abs, verify_rel * tr|G_B|)``.
    """
    if boundary_rel is None:
        boundary_rel = 1e-3
    GA = _metric(cell_A)
    GB = _metric(cell_B)
    tol = max(verify_abs,
              verify_rel * (abs(GB[0][0]) + abs(GB[1][1]) + abs(GB[2][2])))

    vA = _closure_for_match(cell_A, boundary_rel, use_typed_closure,
                            angle_sigma=angle_sigma)
    vB = _closure_for_match(cell_B, boundary_rel, use_typed_closure,
                            angle_sigma=angle_sigma)

    found = set()
    for CA in vA:
        U = [[CA[1][r], CA[2][r], CA[3][r]] for r in range(3)]   # 3x3 int, columns
        for CB in vB:
            for perm in _PERMS:
                for s in (1, -1):
                    W = [[s * CB[perm[1]][r], s * CB[perm[2]][r], s * CB[perm[3]][r]]
                         for r in range(3)]
                    # W is built from three of the four zero-sum superbase
                    # vectors, so it is always unimodular and its inverse is
                    # exactly integer -- take the fast integer path. (Should W
                    # ever be non-unimodular, fall back to exact rationals so
                    # correctness is never at risk.)
                    Winv = _inv3_unimod(W)
                    if Winv is not None:
                        # P = U . W^{-1}: exact integer change of basis matching
                        # A's superbase (variant CA) to B's (variant CB) under
                        # this relabelling. P maps B-basis coords to A-basis, so
                        # P^T G_A P = G_B is the reindexing we want.
                        P = _matmul_int(U, Winv)
                    else:
                        Winv_f = _inv3_frac(W)
                        if Winv_f is None:
                            continue
                        P = _int_or_none(_matmul_frac(U, Winv_f))
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


def best_reindex_with_residual(cell_A, cell_B, boundary_rel=0.0):
    """Best canonical reindexing operator A -> B and its metric residual.

    Unlike :func:`reindexing_via_canonical`, this applies NO acceptance
    threshold: it returns ``(P, resid)`` for the integer operator ``P`` that
    minimises ``|P^T G_A P - G_B|`` over the typed Selling-superbase closure.
    Returns ``(None, inf)`` only if no integer unimodular candidate exists.
    """
    GA = _metric(cell_A)
    GB = _metric(cell_B)
    vA = _closure_for_match(cell_A, boundary_rel)
    vB = _closure_for_match(cell_B, boundary_rel)
    best_P, best_res = None, float("inf")
    for CA in vA:
        U = [[CA[1][r], CA[2][r], CA[3][r]] for r in range(3)]
        for CB in vB:
            for perm in _PERMS:
                for s in (1, -1):
                    W = [[s * CB[perm[1]][r], s * CB[perm[2]][r], s * CB[perm[3]][r]]
                         for r in range(3)]
                    Winv = _inv3_unimod(W)
                    if Winv is not None:
                        P = _matmul_int(U, Winv)
                    else:
                        Winv_f = _inv3_frac(W)
                        if Winv_f is None:
                            continue
                        P = _int_or_none(_matmul_frac(U, Winv_f))
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


def _reindex_coset(cell_A, cell_B, boundary_rel, band_rel=1e-3):
    """All integer operators tied (within a relative band) with the minimum
    metric residual -- the reindexing coset P.H (H = lattice holohedry). No
    acceptance gate; the caller decides whether to accept via the root distance.
    """
    GA = _metric(cell_A)
    GB = _metric(cell_B)
    scale = abs(GB[0][0]) + abs(GB[1][1]) + abs(GB[2][2])
    vA = _closure_for_match(cell_A, boundary_rel)
    vB = _closure_for_match(cell_B, boundary_rel)
    scored = []
    best_res = float("inf")
    for CA in vA:
        U = [[CA[1][r], CA[2][r], CA[3][r]] for r in range(3)]
        for CB in vB:
            for perm in _PERMS:
                for s in (1, -1):
                    W = [[s * CB[perm[1]][r], s * CB[perm[2]][r], s * CB[perm[3]][r]]
                         for r in range(3)]
                    Winv = _inv3_unimod(W)
                    if Winv is not None:
                        P = _matmul_int(U, Winv)
                    else:
                        Winv_f = _inv3_frac(W)
                        if Winv_f is None:
                            continue
                        P = _int_or_none(_matmul_frac(U, Winv_f))
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
                    scored.append((resid, tuple(tuple(int(x) for x in row) for row in P)))
                    if resid < best_res:
                        best_res = resid
    if not scored:
        return []
    band = best_res + band_rel * scale
    return sorted({P for resid, P in scored if resid <= band})


def reindex(cell_A, cell_B, max_volume_frac=None, max_root_dist=None,
            boundary_rel=6e-2, band_rel=1e-3):
    """Reindex ``cell_A`` onto ``cell_B``, gated solely by the Kurlin root distance.

    This is the recommended entry point for cell reindexing. It splits the
    problem into the two questions that are actually distinct:

    1. **Should we reindex at all?** -- decided by the *setting-invariant* Kurlin
       root distance between the two lattices. If ``root_distance(A, B) >
       max_root_dist`` the cells are not the same lattice within the accepted
       deformation and an empty coset is returned.
    2. **What is the reindexing?** -- the Selling/canonical-superbase coset,
       returned in full when (1) passes.

    The metric residual ``|P^T G_A P - G_B|`` is deliberately NOT used as the
    acceptance gate: it is setting-dependent (two settings of the *same* lattice
    can have a large residual), which is exactly the ambiguity the Selling route
    exists to defeat. The root distance is a lattice invariant -- blind to
    setting, sensitive only to genuine lattice difference -- so it is the correct
    quantity to threshold. See :func:`agentsg.cell.rootform.root_distance`.

    The gate is expressed as a **fractional volume change**, which is
    scale-free: the absolute root distance grows ~linearly with cell size for
    the *same* fractional deformation (``root_distance ~ ||RI(cell)||``), so a
    fixed Angstrom threshold is too tight for large cells and too loose for
    small ones. The volume-fraction gate divides that scale out. Internally it
    becomes the per-cell root radius
    ``symmetry_cutoff(cell_A, volume_tol=max_volume_frac) =
    |(1+max_volume_frac)**(1/3) - 1| * ||RI(cell_A)||`` -- the root distance a
    pure isotropic volume change of ``max_volume_frac`` would produce.

    Parameters
    ----------
    cell_A, cell_B : (a, b, c, alpha, beta, gamma)
        The cell to reindex and the reference, respectively.
    max_volume_frac : float, optional
        Acceptance threshold as a fractional volume change (e.g. ``0.05`` = "treat
        cells within a 5 % isotropic volume change as the same lattice"). This is
        the recommended, **scale-free** knob: it transfers unchanged across cell
        sizes and crystal systems. Exactly one of ``max_volume_frac`` /
        ``max_root_dist`` must be given.
    max_root_dist : float, optional
        Absolute root-distance threshold in Angstrom -- a legacy/expert override
        for when you want a raw distance rather than a volume fraction. NOT
        scale-free: a value tuned on an 8 A cell will not transfer to an 80 A
        cell. Prefer ``max_volume_frac``. Exactly one of the two must be given.
    boundary_rel : float
        Delaunay boundary-variant tolerance for superbase enumeration. The
        default ``6e-2`` is wide enough to stay complete under substantial
        deformation; lower it toward ``1e-3`` for near-exact-lattice work.
    band_rel : float
        Relative width of the residual band defining the coset (operators within
        ``band_rel * tr|G_B|`` of the minimum residual are all returned).

    Returns
    -------
    (ops, root_dist) : (list of 3x3 int tuples, float)
        ``ops`` is the reindexing coset (empty if the root gate rejects), and
        ``root_dist`` is the measured Kurlin root distance (always returned, so
        the caller can inspect the decision).

    Notes
    -----
    The returned coset is a coset, not a single operator, because the reindexing
    solution is intrinsically ``P.H`` with ``H`` the lattice metric-symmetry
    group. A unique operator is selected only by a tie-breaker outside geometry
    (an intensity correlation, or a fixed reference-frame convention) acting over
    this same coset.
    """
    from .rootform import root_distance, symmetry_cutoff
    if (max_volume_frac is None) == (max_root_dist is None):
        raise ValueError("give exactly one of max_volume_frac or max_root_dist")
    rd = root_distance(cell_A, cell_B)
    if max_volume_frac is not None:
        gate = symmetry_cutoff(cell_A, volume_tol=max_volume_frac)
    else:
        gate = max_root_dist
    if rd > gate:
        return [], rd
    return _reindex_coset(cell_A, cell_B, boundary_rel, band_rel=band_rel), rd
