"""
Root form: a continuous lattice similarity key from Kurlin's root invariant.

This is the Kurlin (2022) root invariant, built on the obtuse superbase of
Delone (Delaunay) and Selling and the conorms of Conway & Sloane. Unlike the
G6/S6 embedding — where the same lattice maps to many points and comparison
needs a minimisation over a basis-transform orbit — the root invariant is a
SINGLE vector per lattice, canonicalised by a fixed finite group of 24 index
permutations (relabelling the four superbase vectors). No orbit search, no
reduction-flip discontinuity. Properties:

  * invariant   : independent of the chosen basis, preserved under isometry;
  * continuous  : changes continuously under perturbation of the cell
                  (the property Niggli / Buerger reduction lacks);
  * complete    : for Voronoi types V2–V5 (higher symmetry), RI(L1) == RI(L2)
                  iff L1, L2 are isometric. For generic triclinic (V1) the
                  sorted six-tuple is a collision-free-in-practice similarity
                  key, not Kurlin's full complete invariant (which also fixes
                  the 2×3 root-form column pairing).

Pipeline
--------
1. Cartesian basis (v1,v2,v3) of the lattice from the unit cell.
2. Delaunay/Selling reduction to an OBTUSE superbase {v0,v1,v2,v3},
   v0 = -(v1+v2+v3), all conorms p_ij = -v_i.v_j >= 0.
3. Six conorms -> six root products r_ij = sqrt(p_ij).
4. Canonicalise over the 24 index permutations of {0,1,2,3} to a unique
   ordered invariant (the root invariant RI).

Reference: V. Kurlin, "A complete isometry classification of 3-dimensional
lattices", arXiv:2201.10543 (2022); building on B. Delone (1932), E. Selling
(1874), J. H. Conway & N. J. A. Sloane, "Low-dimensional lattices VI" (1992).

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


def root_products(cell):
    """The six root products r_ij = sqrt(p_ij), keyed by index pair."""
    p = conorms(cell)
    return {ij: sqrt(_clamp0(p[ij])) for ij in _PAIRS}


def _canonical_tuple(rp):
    """Canonicalise the six root products into a unique ordered invariant.

    The 24 index permutations of the superbase act on the six root products by
    permutation, so the multiset of root products is the permutation-invariant
    content. We canonicalise to the SORTED six-tuple.

    Why sorting (rather than a fixed-layout lex-min): for the higher-symmetry
    Voronoi types V2-V5 -- every orthorhombic, tetragonal, hexagonal, cubic and
    monoclinic lattice, i.e. one or more zero conorms -- the obtuse superbase is
    not unique up to index permutation, so a layout-dependent canonical form is
    setting-dependent (two axis orderings yield superbases with genuinely
    different tetrahedron edge structures). Kurlin's Definition 5.1 handles this
    by reducing each Voronoi type's root form precisely to its sorted non-zero
    products; the sorted six-tuple reproduces those per-type invariants uniformly
    and is invariant across all settings.

    Completeness: for V2-V5 the sorted products ARE Kurlin's complete invariant.
    For the generic triclinic type V1 the complete invariant additionally fixes
    the column pairing of the 2x3 root form; the sorted multiset drops that, but
    is empirically collision-free on random triclinic lattices -- sufficient for
    metric similarity search. See docs and Kurlin (2022), Definition 5.1.
    """
    return tuple(sorted(rp[ij] for ij in _PAIRS))


def root_invariant(cell):
    """Return the root invariant RI(cell) as an ordered 6-tuple (Angstrom units).

    A continuous isometry invariant (sorted root products). For Voronoi types
    V2–V5 it is Kurlin's complete invariant: equal tuples iff the lattices are
    isometric. For generic triclinic (V1) it is a collision-free-in-practice
    similarity key — sufficient for metric search, but not the full Kurlin
    complete invariant (which also fixes the 2×3 root-form column pairing).
    Compare lattices with plain Euclidean distance on these tuples — no orbit
    minimisation, continuous across the reduction-flip boundary.
    """
    return _canonical_tuple(root_products(cell))


def root_distance(cell_A, cell_B):
    """Euclidean distance between root invariants (a true, continuous metric)."""
    a = root_invariant(cell_A); b = root_invariant(cell_B)
    return sqrt(sum((a[i] - b[i]) ** 2 for i in range(6)))


def _cell_volume(cell):
    """Unit-cell volume from parameters (local, dependency-free)."""
    import math
    a, b, c, al, be, ga = cell
    ca, cb, cg = (math.cos(math.radians(x)) for x in (al, be, ga))
    return a * b * c * math.sqrt(max(
        1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg, 0.0))


def root_distance_to_volume_ratio(distance, cell):
    """Convert a Kurlin root distance to the equivalent isotropic volume ratio.

    For a *pure isotropic* volume change the root invariant scales linearly with
    the length scale factor, so ``distance = |(V'/V)**(1/3) - 1| * ||RI(cell)||``.
    Inverting gives the fractional volume change a given distance corresponds to::

        V'/V = (1 + distance / ||RI(cell)||)**3

    Returns the volume ratio ``V'/V >= 1`` (a magnitude -- the sign of the change
    is not recoverable from an unsigned distance). Exact only for pure scaling;
    for a general cell pair apply it to the ``volume_component`` from
    :func:`root_volume_decomposition`, not the total distance.
    """
    nrho = sqrt(sum(x * x for x in root_invariant(cell)))
    if nrho <= 0:
        return 1.0
    return (1.0 + distance / nrho) ** 3


def volume_ratio_to_root_distance(volume_ratio, cell):
    """Root distance produced by a pure isotropic volume change of ``volume_ratio``.

    The inverse of :func:`root_distance_to_volume_ratio`::

        distance = |volume_ratio**(1/3) - 1| * ||RI(cell)||

    Use it to turn a volume tolerance ("treat cells within 5 % volume as the
    same") into a scale-correct Kurlin cutoff for a specific cell.
    """
    nrho = sqrt(sum(x * x for x in root_invariant(cell)))
    return abs(volume_ratio ** (1.0 / 3.0) - 1.0) * nrho


def symmetry_cutoff(cell, volume_tol=None, noise_frac=None, z=11.0):
    """Scale-correct Kurlin cutoff for accepting a symmetrised cell.

    A Kurlin symmetry deficiency (distance from a cell to its Reynolds-symmetrised
    metric) has units of length and grows with cell size, so an absolute Angstrom
    cutoff does not transfer between cells. Both sensible references are
    proportional to the cell's own root-invariant norm ``||RI(cell)||``:

    * ``volume_tol`` -- accept when the deficiency is no larger than a pure
      isotropic volume change of this fraction (e.g. ``0.05`` for 5 %). Returns
      ``|(1+volume_tol)**(1/3) - 1| * ||RI||``. The interpretable knob.
    * ``noise_frac`` -- accept when the deficiency is within measurement noise of
      fractional size ``noise_frac`` (e.g. ``0.01`` for 1 % cell precision).
      Returns ``z * noise_frac * ||RI||``; the default ``z=11`` is the p95 of the
      noise null distribution (``z=12.4`` for p99), empirically scale-invariant.

    Exactly one of ``volume_tol`` / ``noise_frac`` must be given. In both cases
    the returned cutoff is ``(dimensionless) * ||RI(cell)||``, so it automatically
    tracks cell scale and the per-system spread (cubic/trigonal rhombohedral
    primitives included) without a separate per-system table.
    """
    if (volume_tol is None) == (noise_frac is None):
        raise ValueError("give exactly one of volume_tol or noise_frac")
    nrho = sqrt(sum(x * x for x in root_invariant(cell)))
    if volume_tol is not None:
        return abs((1.0 + volume_tol) ** (1.0 / 3.0) - 1.0) * nrho
    return z * noise_frac * nrho


def similarity_invariant(cell):
    """Volume-normalised root invariant RI(cell) / V**(1/3) (a *similarity* key).

    Dividing the root invariant by the cube root of the cell volume removes the
    overall length scale: two lattices are *similar* (identical up to isotropic
    scaling) iff their similarity invariants coincide, exactly as in the
    manuscript's similarity relation RI(A)/V(A)**(1/3) = RI(B)/V(B)**(1/3).
    Returns a 6-tuple (dimensionless).
    """
    s = _cell_volume(cell) ** (1.0 / 3.0)
    ri = root_invariant(cell)
    return tuple(r / s for r in ri)


def similarity_distance(cell_A, cell_B):
    """Euclidean distance between volume-normalised root invariants.

    Zero iff the two lattices are similar (isotropic-scale copies). This is the
    shape-only counterpart of :func:`root_distance`, blind to volume.
    """
    a = similarity_invariant(cell_A); b = similarity_invariant(cell_B)
    return sqrt(sum((a[i] - b[i]) ** 2 for i in range(6)))


def root_cutoff_for_edge_tolerance(max_edge_change, cell=None, n_edges=1):
    """Root-distance cutoff corresponding to an accepted cell-edge change.

    Answers: "I am willing to treat two lattices as the same if their cell edges
    differ by at most ``max_edge_change`` Angstrom -- what root-distance radius is
    that?"

    The root invariant carries units of length and, for an orthogonal cell, is
    exactly ``sorted(0, 0, 0, a, b, c)``: changing one edge by delta moves one root
    component by exactly delta, so ``root_distance == delta`` (slope k = 1), and
    changing all three edges by delta gives exactly ``sqrt(3) * delta``.
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
        root distance -- exact rather than the generic bound (captures the
        angle/length coupling of a non-orthogonal cell).
    n_edges : int
        Number of edges assumed to change simultaneously for the analytic bound
        (1 = a single edge, worst-typical; 3 = all edges, the ``sqrt(3)`` upper
        envelope). Ignored when ``cell`` is provided.

    Returns
    -------
    float
        The root-distance cutoff (Angstrom). Two lattices within this root
        distance differ by at most ``max_edge_change`` per edge, to first order.
    """
    if cell is not None:
        base = root_invariant(cell)
        a, b, c, al, be, ga = cell
        worst = 0.0
        for i in range(3):
            for sgn in (+1.0, -1.0):
                e = [a, b, c]
                e[i] = max(e[i] + sgn * max_edge_change, 1e-3)
                pert = (e[0], e[1], e[2], al, be, ga)
                d = sqrt(sum((base[j] - root_invariant(pert)[j]) ** 2
                             for j in range(6)))
                if d > worst:
                    worst = d
        # all edges together (upper envelope for this cell)
        e = (max(a + max_edge_change, 1e-3), max(b + max_edge_change, 1e-3),
             max(c + max_edge_change, 1e-3), al, be, ga)
        d = sqrt(sum((base[j] - root_invariant(e)[j]) ** 2 for j in range(6)))
        return max(worst, d)
    return float(n_edges) ** 0.5 * float(max_edge_change)


def root_volume_decomposition(cell_A, cell_B):
    """Split the root distance between two lattices into volume and shape parts.

    The root invariant scales *linearly* with the cell's length scale factor
    ``s = (V_B / V_A)**(1/3)`` (conorms carry units of length**2, roots their
    square root), so a pure isotropic volume change contributes an exactly
    predictable amount to the root distance. This function factors an observed
    root distance into

    * ``volume_component`` -- the distance from ``cell_A`` to the isotropically
      rescaled ``cell_A`` whose volume equals ``V_B``: ``|s - 1| * ||RI(A)||``.
      This is the part of the separation forced purely by the volume change.
    * ``shape_residual`` -- the root distance between that rescaled ``cell_A``
      and ``cell_B``: the genuine shape change at matched volume. Equivalently
      ``||V_B**(1/3)|| * similarity_distance(A, B)`` up to the scaling of A.
    * ``coupling_angle_deg`` -- the angle (degrees) between the volume leg and
      the shape leg in root space. 90 deg means shape change is independent of
      the volume change; smaller angles mean the two are coupled (e.g. an
      anisotropic dehydration series, where losing volume also changes shape).

    Returns a dict with keys ``total`` (== :func:`root_distance`),
    ``volume_component``, ``shape_residual``, ``scale_factor`` (s),
    ``volume_ratio`` (V_B / V_A) and ``coupling_angle_deg``. ``total`` and the
    two legs satisfy ``total <= volume_component + shape_residual`` (triangle
    inequality) and, when the legs are orthogonal,
    ``total**2 == volume_component**2 + shape_residual**2``.
    """
    import math
    VA = _cell_volume(cell_A); VB = _cell_volume(cell_B)
    s = (VB / VA) ** (1.0 / 3.0)
    riA = root_invariant(cell_A)
    a, b, c, al, be, ga = cell_A
    scaled_A = (a * s, b * s, c * s, al, be, ga)
    riS = root_invariant(scaled_A)
    riB = root_invariant(cell_B)

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
