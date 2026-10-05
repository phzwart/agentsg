"""
Sorted linear search key from the Selling conorms.

The archive search key is the six Selling conorms of an obtuse superbase,
scaled by the closure-invariant length ``sqrt(T)`` with ``T = Σ p_ij`` and
sorted into a nondecreasing 6-tuple:

    sorted_linear_key = sort(p_ij / sqrt(T))

``sorted_root_key`` is that key. Pass ``stabilize='sqrt'`` for Kurlin's root
products ``r_ij = sqrt(p_ij)``, which remain the complete-classification
quantity (the type-dependent 2×3 root form keeps opposite-edge pairing; this
module does not). ``root_invariant`` / ``root_distance`` still return the √
key so stored √ columns and old callers do not change meaning. They are
deprecated as the search key.

Why the linear map is the search key
-------------------------------------
√ has a Hölder-½ cusp at a vanishing conorm. A lattice on a Selling boundary
(one conorm exactly 0: a C-centred monoclinic cell with a = b, or any cell
with a right angle) compared with the same lattice pushed just across that
boundary gets a key distance that is almost entirely cusp. On the Scotty set
(McCoy, Andrews, Bernstein & Read, Acta Cryst. D82, 813, 2026) two of five
cases then fail retrieval: 1fe5 → 1u4j / 1g2x at 16.7 / 16.4 Å (rank about
170) and 4nl7 → 2yht at 13.8 Å, while NCDist reports 2.1, 2.1 and 2.3. The
linear key gives 2.7 / 2.9 and 5.4 Å on those pairs and stays within a factor
of about 2 of NCDist on all eleven pairs.

Properties of the sorted linear key:

  * one key per lattice : ``T`` is closure-invariant, and every member of the
    Selling closure carries the same sorted multiset;
  * rearrangement lower bound : the sorted-key distance is at most the aligned
    distance over any superbase pairing (and equals the minimum over all of
    S6);
  * Lipschitz : continuous in the metric tensor, including at a vanishing
    conorm, where √ is only Hölder-½;
  * ångström units and length scaling : ``key(λΛ) = λ · key(Λ)``.

Cube identity: for a cube of edge ``a`` the key is
``(0, 0, 0, a/√3, a/√3, a/√3)``, and an isotropic change ``Δa`` moves the key
by exactly ``Δa``.

The key is many-to-one for Voronoi types V1, V2 and V4 (forgotten pairing) and
injective for V3 and V5. Equality is never a proof of lattice identity.
Certify with the exact operator test over the Selling-superbase closure.

``floored`` and ``soft_threshold`` are noise models for frame-level serial
data. They are not the archive search key.

Pipeline
--------
1. Cartesian basis (v1, v2, v3) of the lattice from the unit cell.
2. Delaunay/Selling reduction to an obtuse superbase {v0, v1, v2, v3},
   v0 = -(v1+v2+v3), all conorms p_ij = -v_i·v_j >= 0.
3. Six conorms -> six slot values via ``p / sqrt(T)`` (or an explicit
   ``stabilize`` map).
4. Sort the six values into a nondecreasing 6-tuple.

Reference: V. Kurlin, "A complete isometry classification of 3-dimensional
lattices" (April 2026 revision); building on B. Delone (1932), E. Selling
(1874), J. H. Conway & N. J. A. Sloane, "Low-dimensional lattices VI" (1992).

Dependency-free; float arithmetic (distances are inherently numeric).
"""
from __future__ import annotations
import warnings
from math import sqrt

from .metric import UnitCell
from ..tolerances import REL_EPS, ROOT_SNAP_DECIMALS, ROOT_SNAP_REL, ROOT_STABILIZE_KAPPA, SYMMETRY_CUTOFF_Z


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
    """Euclidean dot product of two 3-vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def selling_reduced_cell(cell):
    """One Selling-reduced cell: ``(a, b, c, alpha, beta, gamma)`` of ``v1,v2,v3``.

    Built from the obtuse superbase returned by :func:`delaunay_superbase`.
    This is a single representative, not the Selling orbit. Its labeling
    follows the input basis; other obtuse bases are recovered by running the
    orbit on a reference cell. Pass the primitive cell when the deposited
    cell is centred.
    """
    from .metric import params_from_metric
    S = delaunay_superbase(cell)
    G = [[_dot(S[i], S[j]) for j in (1, 2, 3)] for i in (1, 2, 3)]
    return params_from_metric(G)


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
    eps = REL_EPS * scale
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


def _clamp0(x, scale=None):
    """Map float-noise negatives to 0 using the same relative 1e-9 as Delaunay.

    ``scale`` is a positive Å² magnitude (typically max |v_i|² of the superbase).
    An absolute −1e−9 cutoff is wrong for large cells, where noise grows as cell².
    """
    if x >= 0.0:
        return float(x)
    tol = REL_EPS * (float(scale) if scale is not None else 1.0)
    return 0.0 if x > -tol else float(x)


def _superbase_lengths(cell):
    """Lengths |v_i| of the obtuse superbase vectors."""
    S = delaunay_superbase(cell)
    return [sqrt(max(_dot(S[i], S[i]), 0.0)) for i in range(4)]


def conorms(cell):
    """The six conorms p_ij = -v_i.v_j (>= 0) of the obtuse superbase."""
    S = delaunay_superbase(cell)
    scale = max(abs(_dot(S[i], S[i])) for i in range(4)) or 1.0
    return {(i, j): _clamp0(-_dot(S[i], S[j]), scale) for (i, j) in _PAIRS}


def conorm_sum(cell):
    """Invariant T = Σ_{i<j} p_ij (same on every Selling-closure member)."""
    p = conorms(cell)
    return sum(_clamp0(p[ij]) for ij in _PAIRS)


def noise_floor(cell, angle_sigma_deg, c=1.0):
    """Closure-invariant global conorm floor ``s = c · σ_θ · T`` (Angstrom²).

    ``T = Σ p_ij`` is unchanged by Selling flips (Kurlin Lemma A.1 with ε=0
    permutes the conorms). Per-pair floors ``|v_i||v_j|σ_θ`` are *not*
    closure-invariant (even-class superbases have different edge lengths).
    """
    import math
    T = conorm_sum(cell)
    return float(c) * math.radians(float(angle_sigma_deg)) * T


def pair_noise_scales(cell, angle_sigma_deg, c=1.0):
    """Global invariant floor broadcast to all six pairs (API compat).

    Historically named for per-pair scales; now returns the same
    :func:`noise_floor` value for every pair so stabilised keys remain
    one-per-lattice across the Selling closure.
    """
    s = noise_floor(cell, angle_sigma_deg, c=c)
    return {ij: s for ij in _PAIRS}


def _resolve_floors(cell, floors, angle_sigma):
    """Normalize user-supplied noise floors into a dict mapping superbase pair to floor value."""
    if floors is not None:
        if isinstance(floors, (int, float)):
            s = float(floors)
            return {ij: s for ij in _PAIRS}
        return floors
    if angle_sigma is None:
        raise ValueError(
            "stabilize mode needs floors=... or angle_sigma=... (degrees)"
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
        # p / L with L = √T (closure-invariant length) → Angstrom units
        L = max(float(length_scale), 1e-12)
        return p / L
    raise ValueError(
        f"unknown stabilize={stabilize!r}; use None/'sqrt'/'floored'/"
        f"'soft_threshold'/'linear'"
    )


def root_products(cell, stabilize=None, angle_sigma=None, kappa=ROOT_STABILIZE_KAPPA, floors=None):
    """Slot-wise root (or stabilised) products, keyed by index pair.

    Parameters
    ----------
    cell : tuple
        ``(a, b, c, alpha, beta, gamma)``, angles in degrees.
    stabilize : str, optional
        ``None`` (the default, same as ``'sqrt'``) is Kurlin
        ``r_ij = sqrt(p_ij)``. Also ``'floored'``, ``'soft_threshold'``, or
        ``'linear'``. ``floored`` uses ``sqrt(p+s)-sqrt(s)``.
        ``soft_threshold`` uses ``sqrt(max(p-κs, 0))``. ``linear`` uses
        ``p/√T`` with invariant ``T = Σ p_ij``. A monotone per-slot map
        preserves one key per lattice, and the rearrangement lower bound,
        only when it depends on ``p`` and lattice invariants.
    angle_sigma : float, optional
        Angular noise in degrees. Builds the invariant floor
        ``s = σ_θ · T`` when ``floors`` is omitted.
    kappa : float
        Soft-threshold multiple of ``s``. Table default 2.
    floors : float or dict, optional
        Explicit global floor (Å²) or a per-pair dict. Overrides
        ``angle_sigma``. Prefer a scalar or other invariant floor.

    Notes
    -----
    ``None`` and ``'sqrt'`` are Kurlin's root products. The archive search
    key is :func:`sorted_linear_key`, not this default. ``floored`` and
    ``soft_threshold`` are noise models for frame-level serial data: the
    floor chooses the resolution at which near-zero conorms are treated as
    symmetric. They are not used for archive search.
    """
    p = conorms(cell)
    if stabilize is None or stabilize == "sqrt":
        return {ij: sqrt(_clamp0(p[ij])) for ij in _PAIRS}

    T = sum(_clamp0(p[ij]) for ij in _PAIRS)
    length_scale = sqrt(max(T, 0.0))  # √T: closure-invariant length
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
        """Extract conorm p_ij using canonical index ordering i < j."""
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


def _present_key(values):
    """Snap noise-floor components to 0 and round the rest to 10 decimals.

    A component whose absolute value is at most ``max(ROOT_SNAP_REL, ROOT_SNAP_REL * max|r|)``
    is written as 0.0. That is the float dust on an orthogonal root, not a
    change to a root that is actually angstroms long. Rounding stops a value
    such as ``5.000000000000002`` from leaving the key.
    """
    vals = [float(v) for v in values]
    scale = max((abs(v) for v in vals), default=0.0)
    tol = max(ROOT_SNAP_REL, ROOT_SNAP_REL * scale)
    out = []
    for value in vals:
        if abs(value) <= tol:
            out.append(0.0)
        else:
            rounded = round(value, ROOT_SNAP_DECIMALS)
            out.append(0.0 if rounded == 0 else rounded)
    return tuple(out)


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
    return tuple(sorted(_present_key(rp[ij] for ij in _PAIRS)))


def sorted_root_key(cell, stabilize=None, angle_sigma=None, kappa=ROOT_STABILIZE_KAPPA, floors=None):
    """Return the sorted six-slot search key (default: linear, ångström).

    ``stabilize=None`` means ``'linear'``: ``sort(p_ij / sqrt(T))`` with
    ``T = Σ p_ij``. Pass ``stabilize='sqrt'`` for Kurlin's root products.
    ``'floored'`` and ``'soft_threshold'`` are noise models for frame-level
    data, not for archive search. Continuous and basis-invariant, but
    deliberately many-to-one except on Voronoi types V3 and V5. Do not treat
    equality as a lattice-identity proof.
    """
    if stabilize is None:
        stabilize = "linear"
    return _canonical_tuple(root_products(
        cell, stabilize=stabilize, angle_sigma=angle_sigma,
        kappa=kappa, floors=floors,
    ))


def sorted_linear_key(cell):
    """Sorted linear search key ``sort(p_ij / sqrt(Σ p))``, in ångström."""
    return sorted_root_key(cell, stabilize="linear")


def sorted_linear_distance(cell_A, cell_B):
    """Euclidean distance between sorted linear keys, in ångström."""
    return sorted_root_distance(cell_A, cell_B, stabilize="linear")


def root_invariant(cell, **kw):
    """Kurlin √ key. Deprecated as the archive search key.

    Always requests ``stabilize='sqrt'`` unless the caller passes
    ``stabilize`` explicitly, so stored √ keys and old callers keep their
    meaning. The search key is :func:`sorted_linear_key`.
    """
    warnings.warn(
        "root_invariant returns Kurlin's square-root key; the archive search "
        "key is sorted_linear_key",
        DeprecationWarning,
        stacklevel=2,
    )
    kw.setdefault("stabilize", "sqrt")
    return sorted_root_key(cell, **kw)


def sorted_root_distance(cell_A, cell_B, **kw):
    """Euclidean distance between sorted keys (default: the linear key)."""
    a = sorted_root_key(cell_A, **kw)
    b = sorted_root_key(cell_B, **kw)
    return sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def root_distance(cell_A, cell_B, **kw):
    """Kurlin √ distance. Deprecated as the archive search distance.

    Always requests ``stabilize='sqrt'`` unless the caller passes
    ``stabilize`` explicitly. The search distance is
    :func:`sorted_linear_distance`.
    """
    warnings.warn(
        "root_distance returns Kurlin's square-root distance; the archive "
        "search distance is sorted_linear_distance",
        DeprecationWarning,
        stacklevel=2,
    )
    kw.setdefault("stabilize", "sqrt")
    return sorted_root_distance(cell_A, cell_B, **kw)


def aligned_linear_distance(pA, pB):
    """Aligned linear distance of two conorm dicts on one labelling.

    ``sqrt(Σ (pA_ij − pB_ij)²) / (T_A · T_B)^(1/4)``, with ``T = Σ p``.
    When ``T_A = T_B = T`` the normaliser is ``1/sqrt(T)``, which is the
    linear slot map. This is the exact-stage counterpart of the sorted-key
    distance: the sorted distance lower-bounds it when both superbases are
    obtuse.
    """
    diff2 = 0.0
    tA = 0.0
    tB = 0.0
    for ij in _PAIRS:
        a = float(pA[ij])
        b = float(pB[ij])
        diff2 += (a - b) * (a - b)
        tA += a
        tB += b
    denom = (max(tA, 0.0) * max(tB, 0.0)) ** 0.25
    if denom <= 0.0:
        return 0.0
    return sqrt(diff2) / denom


def sorted_key_lower_bound(x, y, G=None):
    """Rearrangement lower bound: ``||sort(x)-sort(y)|| <= min_σ∈G ||x-σy||``.

    When ``G`` is omitted, uses all of ``S_6`` and the equality
    ``||sort(x)-sort(y)|| = min_{σ∈S6} ||x-σy||`` holds. For any physically
    allowed relabelling group ``G ⊆ S_6`` the sorted distance is therefore a
    certified lower bound on the orbit distance (main_v5 Lemma).
    """
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
    """Convert a linear-key distance to the equivalent isotropic volume ratio.

    An isotropic volume change by the factor ``(1+f)`` scales the linear key
    by ``(1+f)**(1/3)``, so
    ``distance = |(V'/V)**(1/3) - 1| * ||key||``. Inverting gives::

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
    """Linear-key distance produced by a pure isotropic volume change.

    The inverse of :func:`root_distance_to_volume_ratio`. A volume factor
    ``(1+f)`` scales the key by ``(1+f)**(1/3)``::

        distance = |volume_ratio**(1/3) - 1| * ||key||

    Use it to turn a volume tolerance ("treat cells within 5 % volume as the
    same") into a scale-correct cutoff for a specific cell.
    """
    nrho = sqrt(sum(x * x for x in sorted_root_key(cell)))
    return abs(volume_ratio ** (1.0 / 3.0) - 1.0) * nrho


def symmetry_cutoff(cell, volume_tol=None, noise_frac=None, z=SYMMETRY_CUTOFF_Z):
    """Scale-correct sorted-key cutoff for accepting a symmetrised cell.

    A linear-key deficiency (distance from a cell to its Reynolds-symmetrised
    metric) has units of length and grows with cell size, so an absolute ångström
    cutoff does not transfer between cells. Both sensible references are
    proportional to the cell's own linear-key norm ``||key||``:

    * ``volume_tol`` -- accept when the deficiency is no larger than a pure
      isotropic volume change of this fraction (e.g. ``0.05`` for 5 %). Returns
      ``|(1+volume_tol)**(1/3) - 1| * ||key||``. The interpretable knob.
    * ``noise_frac`` -- accept when the deficiency is within measurement noise of
      fractional size ``noise_frac`` (e.g. ``0.01`` for 1 % cell precision).
      Returns ``z * noise_frac * ||key||``; the default ``z=11`` is the p95 of the
      noise null distribution (``z=12.4`` for p99), empirically scale-invariant
      under *edge-length* perturbations. The linear key is Lipschitz at a
      vanishing conorm; the square-root key is only Hölder-½ there.

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
    return _present_key(r / s for r in ri)


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

    The sorted linear key carries units of length. For a cube of edge ``a`` it
    is ``(0, 0, 0, a/√3, a/√3, a/√3)``, and an isotropic edge change ``Δa``
    moves the key by exactly ``Δa``. A single conventional-edge change does not
    move one slot by ``Δa``: the slots are ``p_ij/√T``, so length and ``T``
    both change. The analytic ``n_edges`` bound (``sqrt(n) * delta``, which is
    exact for the √ key on an orthogonal cell) is therefore only a guide for
    the linear key. Pass ``cell`` for an exact, conservative per-cell cutoff.

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
        # Every sign pattern of moving all three edges. For the linear key the
        # worst box corner is not always the all-positive one.
        for sa in (+1.0, -1.0):
            for sb in (+1.0, -1.0):
                for sc in (+1.0, -1.0):
                    e = (max(a + sa * max_edge_change, 1e-3),
                         max(b + sb * max_edge_change, 1e-3),
                         max(c + sc * max_edge_change, 1e-3), al, be, ga)
                    d = sqrt(sum((base[j] - sorted_root_key(e)[j]) ** 2
                                 for j in range(6)))
                    if d > worst:
                        worst = d
        return worst
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
