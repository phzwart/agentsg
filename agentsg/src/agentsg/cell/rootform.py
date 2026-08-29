"""
Sorted root-product search key from Kurlin's root products.

Kurlin (2022 / April 2026) builds six root products ``r_ij = sqrt(p_ij)`` from
the obtuse (Selling / Delaunay) superbase and arranges them in a type-dependent
``2x3`` root form that retains opposite-edge pairing. That structured form is
the complete isometry classification. This module deliberately uses a coarser
object for Euclidean retrieval:

  * the six root products, globally sorted into a nondecreasing 6-tuple
    (``sorted_root_key`` / ``root_invariant``);

Properties of the sorted key:

  * invariant   : independent of the chosen basis (same multiset of root
                  products on every obtuse-superbase branch of one lattice);
  * continuous  : sorting is continuous through product ties; the key also
                  stays continuous across Voronoi-type boundaries because all
                  members of a lattice's Selling-superbase closure share the
                  same sorted multiset;
  * Euclidean   : plain L2 distance on the 6-tuple; by the rearrangement
                  inequality this distance lower-bounds any physically allowed
                  relabelling-orbit distance (conservative radius filter);
  * many-to-one : not injective for Voronoi types V1, V2, V4 (forgotten pairing);
                  injective for V3 and V5. Equality of sorted keys is never a
                  proof of lattice identity — certify with the exact operator
                  test over the Selling-superbase closure.

``root_invariant`` / ``root_distance`` remain as back-compat aliases for
``sorted_root_key`` / ``sorted_root_distance``. They are *not* Kurlin Def. 5.1.

Pipeline
--------
1. Cartesian basis (v1,v2,v3) of the lattice from the unit cell.
2. Delaunay/Selling reduction to an OBTUSE superbase {v0,v1,v2,v3},
   v0 = -(v1+v2+v3), all conorms p_ij = -v_i.v_j >= 0.
3. Six conorms -> six slot values via a monotone map f (default f=sqrt;
   optional floored / soft-threshold / linear stabilisations for noisy data).
4. Sort the six values into a nondecreasing 6-tuple (the search key).

Optional stabilisations (``stabilize=``, default ``None`` = plain √) tame the
Hölder-½ cusp of √ at vanishing conorms; see ``pair_noise_scales`` and
``sorted_conorm_key``. A stabilised key is a different metric from Kurlin's.

Reference: V. Kurlin, "A complete isometry classification of 3-dimensional
lattices" (April 2026 revision); building on B. Delone (1932), E. Selling
(1874), J. H. Conway & N. J. A. Sloane, "Low-dimensional lattices VI" (1992).
See manuscript/main_v5.tex for the search/certify architecture.

Dependency-free; float arithmetic (distances/roots are inherently numeric).
"""
from __future__ import annotations
from math import sqrt

from .metric import UnitCell


# the six unordered index pairs of {0,1,2,3}
_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _cart_basis(cell):
    """Cartesian lattice basis vectors (v1,v2,v3) as rows, from a unit cell."""
    O = UnitCell(*cell).orthogonalization_matrix()   # columns map frac -> cart
    # basis vector i is O applied to the i-th unit fractional vector = column i
    v1 = (O[0][0], O[1][0], O[2][0])
    v2 = (O[0][1], O[1][1], O[2][1])
    v3 = (O[0][2], O[1][2], O[2][2])
    return [list(v1), list(v2), list(v3)]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def delaunay_superbase(cell, max_iter=1000):
    """Reduce to an obtuse superbase; return the 4 superbase vectors.

    Selling/Delaunay reduction: while some pair has a positive scalar product
    v_i.v_j > 0 (conorm p_ij < 0), apply the reduction move that negates v_i,
    adds it to the two vectors other than v_j, and leaves v_j fixed -- this
    keeps the superbase sum zero and decreases the sum of squared lengths, so it
    terminates at an obtuse superbase (all v_i.v_j <= 0).
    """
    v1, v2, v3 = _cart_basis(cell)
    v0 = [-(v1[k] + v2[k] + v3[k]) for k in range(3)]
    S = [v0, v1, v2, v3]
    # A move is triggered only by a scalar product that is positive beyond a
    # relative tolerance. Without this, an orthogonal cell -- whose right angles
    # come from cos(90 deg) evaluated as ~6e-17 rather than exactly 0 -- would
    # show tiny POSITIVE products and fire spurious reduction moves, wandering to
    # a non-canonical obtuse superbase whose tetrahedron edge structure is not
    # reachable by index permutation from the trivial one.
    scale = max(abs(_dot(S[i], S[i])) for i in range(4)) or 1.0
    eps = 1e-9 * scale
    for _ in range(max_iter):
        # find the pair with the most positive scalar product
        worst = eps
        wi = wj = -1
        for i in range(4):
            for j in range(i + 1, 4):
                d = _dot(S[i], S[j])
                if d > worst:
                    worst = d; wi, wj = i, j
        if wi < 0:                      # all products <= tol -> obtuse
            return S
        i, j = wi, wj
        others = [k for k in range(4) if k != i and k != j]
        vi = S[i]
        S[i] = [-x for x in vi]
        for k in others:
            S[k] = [S[k][t] + vi[t] for t in range(3)]
        # S[j] unchanged
    raise RuntimeError("Delaunay reduction did not converge")


def conorms(cell):
    """The six conorms p_ij = -v_i.v_j (>= 0) of the obtuse superbase."""
    S = delaunay_superbase(cell)
    return {(i, j): -_dot(S[i], S[j]) for (i, j) in _PAIRS}


def _clamp0(x):
    # tiny negatives from float noise -> 0 before sqrt
    return 0.0 if -1e-9 < x < 0 else x


def _superbase_lengths(cell):
    """Lengths |v_i| of the obtuse superbase vectors."""
    S = delaunay_superbase(cell)
    return [sqrt(max(_dot(S[i], S[i]), 0.0)) for i in range(4)]


def pair_noise_scales(cell, angle_sigma_deg):
    """Per-pair conorm noise floors ``s_ij ≈ |v_i| |v_j| σ_θ`` (Angstrom²).

    Angular noise is approximately Gaussian in conorm space. Pair-local floors
    keep the long axis from setting the floor on short-edge pairs. ``angle_sigma_deg``
    is the one-sigma angular noise in degrees.
    """
    import math
    lengths = _superbase_lengths(cell)
    sig = math.radians(float(angle_sigma_deg))
    return {(i, j): lengths[i] * lengths[j] * sig for (i, j) in _PAIRS}


def _resolve_floors(cell, floors, angle_sigma):
    if floors is not None:
        return floors
    if angle_sigma is None:
        raise ValueError(
            "stabilize mode needs per-pair floors=... or angle_sigma=... (degrees)"
        )
    return pair_noise_scales(cell, angle_sigma)


def _slot_map(p, s, stabilize, kappa, length_scale):
    """Monotone per-slot map f(p); default None/'sqrt' is Kurlin √p."""
    p = max(_clamp0(p), 0.0)
    if stabilize is None or stabilize == "sqrt":
        return sqrt(p)
    if stabilize == "floored":
        # Wiener-style: √(p+s)-√s → Lipschitz near 0, ~√p for p ≫ s
        s = max(float(s), 0.0)
        if s <= 0.0:
            return sqrt(p)
        return sqrt(p + s) - sqrt(s)
    if stabilize == "soft_threshold":
        # √(max(p - κ s, 0)) — shrinks near-zero slots onto the symmetry stratum
        s = max(float(s), 0.0)
        return sqrt(max(p - float(kappa) * s, 0.0))
    if stabilize == "linear":
        # p / L with L a length → Angstrom units (Lipschitz, Gaussian-noise preserving)
        L = max(float(length_scale), 1e-12)
        return p / L
    raise ValueError(
        f"unknown stabilize={stabilize!r}; use None/'sqrt'/'floored'/"
        f"'soft_threshold'/'linear'"
    )


def root_products(cell, stabilize=None, angle_sigma=None, kappa=2.0, floors=None):
    """Slot-wise root (or stabilised) products, keyed by index pair.

    Parameters
    ----------
    stabilize : None | 'sqrt' | 'floored' | 'soft_threshold' | 'linear'
        Default ``None`` (same as ``'sqrt'``) is Kurlin ``r_ij = sqrt(p_ij)``.
        ``floored`` uses ``sqrt(p+s)-sqrt(s)``; ``soft_threshold`` uses
        ``sqrt(max(p-κs, 0))``; ``linear`` uses ``p/L`` with
        ``L = max_i |v_i|``. Any monotone per-slot map preserves sorting
        invariance and the rearrangement lower bound in the chosen metric.
    angle_sigma : float, optional
        Angular noise σ in degrees; used to build per-pair floors
        ``s_ij = |v_i||v_j| σ_θ`` when ``floors`` is omitted.
    kappa : float
        Soft-threshold multiple of ``s`` (table default 2).
    floors : dict, optional
        Explicit per-pair ``s_ij`` (Angstrom²), overriding ``angle_sigma``.

    A stabilised key is a *different* metric from Kurlin's √ root products: the
    floor chooses the resolution at which near-zero conorms are treated as
    symmetric. Archive search should keep the default; serial/noisy frames may
    prefer ``floored``, ``soft_threshold``, or :func:`sorted_conorm_key`.
    """
    p = conorms(cell)
    if stabilize is None or stabilize == "sqrt":
        return {ij: sqrt(_clamp0(p[ij])) for ij in _PAIRS}

    lengths = _superbase_lengths(cell)
    length_scale = max(lengths) if lengths else 1.0
    s_map = None
    if stabilize in ("floored", "soft_threshold"):
        s_map = _resolve_floors(cell, floors, angle_sigma)

    out = {}
    for ij in _PAIRS:
        s = 0.0 if s_map is None else s_map[ij]
        out[ij] = _slot_map(p[ij], s, stabilize, kappa, length_scale)
    return out


def sorted_conorm_key(cell):
    """Sorted six conorms ``sort(p)`` in Angstrom² (Lipschitz / S⁶-like key).

    Preferred for noisy per-frame statistics (e.g. XFEL PCA): linear in the
    metric tensor, no Hölder-½ amplification at vanishing conorms. Still one
    key per lattice and still a pure sort, so the rearrangement lower bound
    holds. Not in length units — use √ roots for archival length-unit search.
    """
    p = conorms(cell)
    return tuple(sorted(_clamp0(p[ij]) for ij in _PAIRS))


def sorted_conorm_distance(cell_A, cell_B):
    """Euclidean distance between sorted conorm keys (Angstrom²)."""
    a = sorted_conorm_key(cell_A)
    b = sorted_conorm_key(cell_B)
    return sqrt(sum((a[i] - b[i]) ** 2 for i in range(6)))


def vonorms_from_conorms(p):
    """Seven vonorms from six conorms (Kurlin Def. 2.6 / ABS D7).

    Four vertex vonorms ``v_i^2 = sum_{j≠i} p_ij`` and three opposite-edge
    pair vonorms ``v_{ij}^2 = p_ik+p_il+p_jk+p_jl`` for complementary pairs
    ``{i,j}`` / ``{k,l}``. Returns a length-7 list (squared lengths).
    """
    def _p(i, j):
        return p[(i, j) if i < j else (j, i)]

    vertex = []
    for i in range(4):
        others = [j for j in range(4) if j != i]
        vertex.append(sum(_p(i, j) for j in others))
    # opposite edge pairs of the tetrahedron K4
    opp = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))
    pairs = []
    for i, j, k, l in opp:
        pairs.append(_p(i, k) + _p(i, l) + _p(j, k) + _p(j, l))
    return vertex + pairs


def vonorms(cell):
    """Seven vonorms (squared lengths) of the obtuse superbase of ``cell``."""
    return vonorms_from_conorms(conorms(cell))


def sorted_vonorm_key(cell):
    """Sorted square roots of the seven vonorms (Angstrom; ABS D7 / Kurlin voform)."""
    return tuple(sorted(sqrt(_clamp0(v)) for v in vonorms(cell)))


def sorted_concat_key(cell, **kw):
    """Concatenated sorted 6-root ‖ sorted √vonorm key in R^13."""
    return sorted_root_key(cell, **kw) + sorted_vonorm_key(cell)


def _canonical_tuple(rp):
    """Sort the six root products into the Euclidean search key.

    The 24 index permutations of the superbase act on the six root products by
    permutation, so the multiset of root products is the permutation-invariant
    content. We project further to the globally sorted six-tuple, which forgets
    tetrahedral edge pairing.

    This is *not* Kurlin Definition 5.1 (the type-dependent ordered 2x3 root
    form). Sorting is continuous through product ties and yields one key per
    lattice because all members of the Selling-superbase closure share the same
    sorted multiset. Injectivity of the sorted key: V3 and V5 yes; V1, V2, V4
    no (finite pairing collisions). Use the exact operator test for identity.
    """
    return tuple(sorted(rp[ij] for ij in _PAIRS))


def sorted_root_key(cell, stabilize=None, angle_sigma=None, kappa=2.0, floors=None):
    """Return the sorted six-slot search key (default: √ conorms, Angstrom).

    Continuous and basis-invariant, but deliberately many-to-one except on
    Voronoi types V3 and V5. Optional ``stabilize`` selects a monotone slot map
    (see :func:`root_products`); default ``None`` preserves archive √ behaviour.
    Do not treat equality as a lattice-identity proof.
    """
    return _canonical_tuple(root_products(
        cell, stabilize=stabilize, angle_sigma=angle_sigma,
        kappa=kappa, floors=floors,
    ))


def root_invariant(cell, **kw):
    """Back-compat alias for :func:`sorted_root_key`.

    Historical name retained; this is the sorted search key, not Kurlin's
    complete ordered root invariant. Keyword args forwarded to
    :func:`sorted_root_key` (e.g. ``stabilize=``).
    """
    return sorted_root_key(cell, **kw)


def sorted_root_distance(cell_A, cell_B, **kw):
    """Euclidean distance between sorted root keys (conservative search metric)."""
    a = sorted_root_key(cell_A, **kw)
    b = sorted_root_key(cell_B, **kw)
    return sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def root_distance(cell_A, cell_B, **kw):
    """Back-compat alias for :func:`sorted_root_distance`."""
    return sorted_root_distance(cell_A, cell_B, **kw)


def sorted_key_lower_bound(x, y, G=None):
    """Rearrangement lower bound: ``||sort(x)-sort(y)|| <= min_σ∈G ||x-σy||``.

    When ``G`` is omitted, uses all of ``S_6`` and the equality
    ``||sort(x)-sort(y)|| = min_{σ∈S6} ||x-σy||`` holds. For any physically
    allowed relabelling group ``G ⊆ S_6`` the sorted distance is therefore a
    certified lower bound on the orbit distance (main_v5 Lemma).
    """
    from itertools import permutations
    sx = tuple(sorted(x))
    sy = tuple(sorted(y))
    sorted_d = sqrt(sum((sx[i] - sy[i]) ** 2 for i in range(len(sx))))
    if G is None:
        return sorted_d, sorted_d
    best = float("inf")
    y = tuple(y)
    for sigma in G:
        d = sqrt(sum((x[i] - y[sigma[i]]) ** 2 for i in range(len(x))))
        if d < best:
            best = d
    return sorted_d, best


def _cell_volume(cell):
    """Unit-cell volume from parameters (local, dependency-free)."""
    import math
    a, b, c, al, be, ga = cell
    ca, cb, cg = (math.cos(math.radians(x)) for x in (al, be, ga))
    return a * b * c * math.sqrt(max(
        1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg, 0.0))


def root_distance_to_volume_ratio(distance, cell):
    """Convert a sorted-key distance to the equivalent isotropic volume ratio.

    For a *pure isotropic* volume change the root key scales linearly with
    the length scale factor, so ``distance = |(V'/V)**(1/3) - 1| * ||key||``.
    Inverting gives the fractional volume change a given distance corresponds to::

        V'/V = (1 + distance / ||key||)**3

    Returns the volume ratio ``V'/V >= 1`` (a magnitude -- the sign of the change
    is not recoverable from an unsigned distance). Exact only for pure scaling;
    for a general cell pair apply it to the ``volume_component`` from
    :func:`root_volume_decomposition`, not the total distance.
    """
    nrho = sqrt(sum(x * x for x in sorted_root_key(cell)))
    if nrho <= 0:
        return 1.0
    return (1.0 + distance / nrho) ** 3


def volume_ratio_to_root_distance(volume_ratio, cell):
    """Sorted-key distance produced by a pure isotropic volume change.

    The inverse of :func:`root_distance_to_volume_ratio`::

        distance = |volume_ratio**(1/3) - 1| * ||key||

    Use it to turn a volume tolerance ("treat cells within 5 % volume as the
    same") into a scale-correct cutoff for a specific cell.
    """
    nrho = sqrt(sum(x * x for x in sorted_root_key(cell)))
    return abs(volume_ratio ** (1.0 / 3.0) - 1.0) * nrho


def symmetry_cutoff(cell, volume_tol=None, noise_frac=None, z=11.0):
    """Scale-correct sorted-key cutoff for accepting a symmetrised cell.

    A Kurlin symmetry deficiency (distance from a cell to its Reynolds-symmetrised
    metric) has units of length and grows with cell size, so an absolute Angstrom
    cutoff does not transfer between cells. Both sensible references are
    proportional to the cell's own key norm ``||key||``:

    * ``volume_tol`` -- accept when the deficiency is no larger than a pure
      isotropic volume change of this fraction (e.g. ``0.05`` for 5 %). Returns
      ``|(1+volume_tol)**(1/3) - 1| * ||key||``. The interpretable knob.
    * ``noise_frac`` -- accept when the deficiency is within measurement noise of
      fractional size ``noise_frac`` (e.g. ``0.01`` for 1 % cell precision).
      Returns ``z * noise_frac * ||key||``; the default ``z=11`` is the p95 of the
      noise null distribution (``z=12.4`` for p99), empirically scale-invariant
      under *edge-length* perturbations. It is not calibrated for angular noise
      at vanishing conorms (Hölder-½ regime of plain √); use a stabilised key or
      sorted conorms when that regime dominates.

    Exactly one of ``volume_tol`` / ``noise_frac`` must be given. In both cases
    the returned cutoff is ``(dimensionless) * ||key||``, so it automatically
    tracks cell scale and the per-system spread (cubic/trigonal rhombohedral
    primitives included) without a separate per-system table.
    """
    if (volume_tol is None) == (noise_frac is None):
        raise ValueError("give exactly one of volume_tol or noise_frac")
    nrho = sqrt(sum(x * x for x in sorted_root_key(cell)))
    if volume_tol is not None:
        return abs((1.0 + volume_tol) ** (1.0 / 3.0) - 1.0) * nrho
    return z * noise_frac * nrho


def similarity_invariant(cell):
    """Volume-normalised sorted key ``key / V**(1/3)`` (a *similarity* key).

    Dividing by the cube root of the cell volume removes the overall length
    scale: two lattices are *similar* (identical up to isotropic scaling) when
    their similarity keys coincide (up to the known many-to-one collisions of
    the sorted projection). Returns a 6-tuple (dimensionless).
    """
    s = _cell_volume(cell) ** (1.0 / 3.0)
    ri = sorted_root_key(cell)
    return tuple(r / s for r in ri)


def similarity_distance(cell_A, cell_B):
    """Euclidean distance between volume-normalised sorted keys.

    Zero when the two lattices are similar at the level of the sorted key
    (isotropic-scale copies, up to pairing collisions). Shape-only counterpart
    of :func:`sorted_root_distance`, blind to volume.
    """
    a = similarity_invariant(cell_A); b = similarity_invariant(cell_B)
    return sqrt(sum((a[i] - b[i]) ** 2 for i in range(6)))


def root_cutoff_for_edge_tolerance(max_edge_change, cell=None, n_edges=1):
    """Sorted-key distance cutoff corresponding to an accepted cell-edge change.

    Answers: "I am willing to treat two lattices as the same if their cell edges
    differ by at most ``max_edge_change`` Angstrom -- what key-space radius is
    that?"

    The sorted key carries units of length and, for an orthogonal cell, is
    exactly ``sorted(0, 0, 0, a, b, c)``: changing one edge by delta moves one
    component by exactly delta, so ``sorted_root_distance == delta`` (slope k = 1),
    and changing all three edges by delta gives exactly ``sqrt(3) * delta``.
    Empirically the single-edge slope has median k = 1.00 for near-orthogonal
    lattices (tetragonal, orthorhombic, hexagonal, monoclinic, triclinic in the
    PDB), but it is LARGER for cells whose primitive basis is strongly
    non-orthogonal -- notably cubic groups, whose primitive cells are rhombohedra
    (F -> 60 deg, I -> 109.47 deg): there a single conventional-edge change couples
    into several root components (median k ~ 1.3, tail up to ~4.5). The analytic
    ``n_edges`` bound is therefore a guide, not a guarantee; pass ``cell`` for an
    exact, conservative per-cell cutoff whenever the lattice may be
    non-orthogonal.

    Parameters
    ----------
    max_edge_change : float
        The largest per-edge length change (Angstrom) you are willing to accept.
    cell : tuple, optional
        If given, the cutoff is *calibrated exactly for this cell* by perturbing
        each of its edges by ``max_edge_change`` and taking the largest resulting
        key distance -- exact rather than the generic bound (captures the
        angle/length coupling of a non-orthogonal cell).
    n_edges : int
        Number of edges assumed to change simultaneously for the analytic bound
        (1 = a single edge, worst-typical; 3 = all edges, the ``sqrt(3)`` upper
        envelope). Ignored when ``cell`` is provided.

    Returns
    -------
    float
        The sorted-key distance cutoff (Angstrom). Two lattices within this
        distance differ by at most ``max_edge_change`` per edge, to first order.
    """
    if cell is not None:
        base = sorted_root_key(cell)
        a, b, c, al, be, ga = cell
        worst = 0.0
        for i in range(3):
            for sgn in (+1.0, -1.0):
                e = [a, b, c]
                e[i] = max(e[i] + sgn * max_edge_change, 1e-3)
                pert = (e[0], e[1], e[2], al, be, ga)
                d = sqrt(sum((base[j] - sorted_root_key(pert)[j]) ** 2
                             for j in range(6)))
                if d > worst:
                    worst = d
        # all edges together (upper envelope for this cell)
        e = (max(a + max_edge_change, 1e-3), max(b + max_edge_change, 1e-3),
             max(c + max_edge_change, 1e-3), al, be, ga)
        d = sqrt(sum((base[j] - sorted_root_key(e)[j]) ** 2 for j in range(6)))
        return max(worst, d)
    return float(n_edges) ** 0.5 * float(max_edge_change)


def root_volume_decomposition(cell_A, cell_B):
    """Split the sorted-key distance between two lattices into volume and shape.

    The sorted key scales *linearly* with the cell's length scale factor
    ``s = (V_B / V_A)**(1/3)`` (conorms carry units of length**2, roots their
    square root), so a pure isotropic volume change contributes an exactly
    predictable amount to the key distance. This function factors an observed
    distance into

    * ``volume_component`` -- the distance from ``cell_A`` to the isotropically
      rescaled ``cell_A`` whose volume equals ``V_B``: ``|s - 1| * ||key(A)||``.
      This is the part of the separation forced purely by the volume change.
    * ``shape_residual`` -- the key distance between that rescaled ``cell_A``
      and ``cell_B``: the genuine shape change at matched volume. Equivalently
      ``||V_B**(1/3)|| * similarity_distance(A, B)`` up to the scaling of A.
    * ``coupling_angle_deg`` -- the angle (degrees) between the volume leg and
      the shape leg in key space. 90 deg means shape change is independent of
      the volume change; smaller angles mean the two are coupled (e.g. an
      anisotropic dehydration series, where losing volume also changes shape).

    Returns a dict with keys ``total`` (== :func:`sorted_root_distance`),
    ``volume_component``, ``shape_residual``, ``scale_factor`` (s),
    ``volume_ratio`` (V_B / V_A) and ``coupling_angle_deg``. ``total`` and the
    two legs satisfy ``total <= volume_component + shape_residual`` (triangle
    inequality) and, when the legs are orthogonal,
    ``total**2 == volume_component**2 + shape_residual**2``.
    """
    import math
    VA = _cell_volume(cell_A); VB = _cell_volume(cell_B)
    s = (VB / VA) ** (1.0 / 3.0)
    riA = sorted_root_key(cell_A)
    a, b, c, al, be, ga = cell_A
    scaled_A = (a * s, b * s, c * s, al, be, ga)
    riS = sorted_root_key(scaled_A)
    riB = sorted_root_key(cell_B)

    leg_vol = [riS[i] - riA[i] for i in range(6)]
    leg_shape = [riB[i] - riS[i] for i in range(6)]
    total = sqrt(sum((riB[i] - riA[i]) ** 2 for i in range(6)))
    vcomp = sqrt(sum(x * x for x in leg_vol))
    scomp = sqrt(sum(x * x for x in leg_shape))
    dot = sum(leg_vol[i] * leg_shape[i] for i in range(6))
    if vcomp > 1e-12 and scomp > 1e-12:
        cosang = max(-1.0, min(1.0, dot / (vcomp * scomp)))
        angle = math.degrees(math.acos(cosang))
    else:
        angle = float("nan")
    return {
        "total": total,
        "volume_component": vcomp,
        "shape_residual": scomp,
        "scale_factor": s,
        "volume_ratio": VB / VA,
        "coupling_angle_deg": angle,
    }
