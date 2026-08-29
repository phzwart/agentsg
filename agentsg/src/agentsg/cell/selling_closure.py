"""Typed Selling-superbase closure (Kurlin Lemmas 4.1--4.5 / main_v5).

Selling reduction turns the infinite primitive-basis ambiguity into a finite set
of obtuse superbases. For generic Voronoi type V1 that set is a single isometry
class acted on by S4 x {+I,-I} (order 48). At V2--V5 additional *non-isometric*
obtuse-superbase classes appear (2, 3, 3, 4 classes respectively). Exact
reindexing must enumerate the full type-dependent closure, not only the 24
relabellings of one reduced superbase.

This module classifies the Voronoi type of an obtuse superbase and builds the
finite closure used by :mod:`agentsg.cell.canonical` for certification.

The S4 x {+/-I} group in :mod:`agentsg.cell.selling_group` remains the V1 /
single-class orbit; it is not by itself the V2--V5 closure.
"""
from __future__ import annotations

from .canonical import canonical_superbase, _metric, _dotG


def _conorm_tol(C, G, rel=1e-9):
    scale = max(abs(_dotG(C[i], C[i], G)) for i in range(4)) or 1.0
    return rel * scale


def _zero_pairs(C, G, tol):
    """Index pairs (i,j) with |v_i · v_j| <= tol (orthogonal / boundary)."""
    out = []
    for i in range(4):
        for j in range(i + 1, 4):
            if abs(_dotG(C[i], C[j], G)) <= tol:
                out.append((i, j))
    return out


def _is_obtuse(C, G, tol):
    for i in range(4):
        for j in range(i + 1, 4):
            if _dotG(C[i], C[j], G) > tol:
                return False
    return True


def _ukey(C):
    """Unordered superbase identity (label order ignored; ± kept distinct)."""
    return frozenset(tuple(int(x) for x in v) for v in C)


def _opposite_pair(i, j):
    return tuple(sorted(k for k in range(4) if k != i and k != j))


def _conorm_sum(C, G):
    """T = Σ_{i<j} p_ij with p_ij = -v_i·v_j (invariant on the closure)."""
    T = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            T += max(0.0, -_dotG(C[i], C[j], G))
    return T


def _zero_tol(C, G, tol_rel=1e-9, angle_sigma=None, noise_mult=3.0):
    """Absolute conorm tolerance for zero detection / obtuseness.

    Default is a tiny relative floor. With ``angle_sigma`` (degrees), widen to
    a few σ of the invariant global noise scale ``σ_θ · T / 6`` so noisy
    high-symmetry cells are classified as V2--V5 rather than V1.
    """
    tol = _conorm_tol(C, G, tol_rel)
    if angle_sigma is not None:
        import math
        sig = math.radians(float(angle_sigma))
        T = _conorm_sum(C, G)
        tol = max(tol, float(noise_mult) * sig * T / 6.0)
    return tol


def voronoi_type_from_superbase(C, G, tol=None):
    """Classify Voronoi type V1--V5 from an obtuse superbase's zero conorms.

    Uses Kurlin's zero-pattern characterisation:
      V1: no zeros; V2: one zero; V3: two opposite (complementary) zeros;
      V4: two zeros sharing a vertex (one vector orthogonal to two others);
      V5: three or more zeros (cuboid).
    """
    if tol is None:
        tol = _conorm_tol(C, G)
    zeros = _zero_pairs(C, G, tol)
    n = len(zeros)
    if n == 0:
        return 1
    if n == 1:
        return 2
    if n >= 3:
        return 5
    (a, b), (c, d) = zeros[0], zeros[1]
    if set(zeros[1]) == set(_opposite_pair(a, b)):
        return 3
    return 4


def voronoi_type(cell, tol_rel=1e-9, angle_sigma=None):
    """Voronoi type (1..5) of ``cell`` from its Selling-reduced superbase.

    Pass ``angle_sigma`` (degrees) to treat near-zero conorms within a few σ
    of the invariant noise floor as zeros (needed for noisy serial frames).
    """
    G = _metric(cell)
    C, _ = canonical_superbase(cell)
    return voronoi_type_from_superbase(
        C, G, tol=_zero_tol(C, G, tol_rel, angle_sigma=angle_sigma),
    )


def _selling_flip(C, i, j):
    """Selling move on pair (i,j): negate v_i, add to the two others except j."""
    Cn = [row[:] for row in C]
    vi = Cn[i][:]
    Cn[i] = [-x for x in vi]
    for t in range(4):
        if t != i and t != j:
            Cn[t] = [Cn[t][s] + vi[s] for s in range(3)]
    return Cn


def _v5_even_reps(C0):
    """Three even-class representatives for a cuboid odd superbase (Lemma 4.5)."""
    reps = []
    for i, j, k in ((1, 2, 3), (1, 3, 2), (2, 3, 1)):
        vi, vj, vk = C0[i], C0[j], C0[k]
        vk_minus_vi = [vk[t] - vi[t] for t in range(3)]
        minus_vk_vj = [-vk[t] - vj[t] for t in range(3)]
        reps.append([vi[:], vj[:], vk_minus_vi, minus_vk_vj])
    return reps


def _class_seeds(C0, G, tol):
    """Seed superbases: reduced one plus Kurlin alternate-class maps."""
    vtype = voronoi_type_from_superbase(C0, G, tol)
    seeds = [C0]
    zeros = _zero_pairs(C0, G, tol)
    if vtype == 1:
        return seeds
    if vtype in (2, 3, 4):
        for i, j in zeros:
            seeds.append(_selling_flip(C0, i, j))
            seeds.append(_selling_flip(C0, j, i))
        return seeds
    # V5
    seeds.extend(_v5_even_reps(C0))
    for i, j in zeros:
        seeds.append(_selling_flip(C0, i, j))
        seeds.append(_selling_flip(C0, j, i))
    return seeds


def selling_superbase_closure(cell, tol_rel=1e-9, angle_sigma=None):
    """Finite Selling-superbase closure of ``cell`` (typed, Kurlin 4.1--4.5).

    Returns distinct obtuse superbases as ordered 4-tuples of integer coordinate
    triples in the input cell basis. Uniqueness is up to unordered set of the
    four vectors. A primitive orthorhombic V5 lattice has 32 superbases in 4
    isometry classes; generic V1 has a single reduced representative (the match
    loop in :mod:`canonical` still applies S4 x {+/-I}).

    Pass ``angle_sigma`` (degrees) to widen zero detection for noisy cells.

    All members share the same sorted root-product multiset (main_v5).
    """
    G = _metric(cell)
    C0, _ = canonical_superbase(cell)
    tol = _zero_tol(C0, G, tol_rel, angle_sigma=angle_sigma)

    seen = {}
    frontier = []
    for S in _class_seeds(C0, G, tol):
        if _is_obtuse(S, G, tol):
            frontier.append(S)

    while frontier:
        C = frontier.pop()
        k = _ukey(C)
        if k in seen:
            continue
        seen[k] = C
        for i, j in _zero_pairs(C, G, tol):
            for a, b in ((i, j), (j, i)):
                Cn = _selling_flip(C, a, b)
                if _is_obtuse(Cn, G, tol) and _ukey(Cn) not in seen:
                    frontier.append(Cn)

    if not seen:
        seen[_ukey(C0)] = C0
    return list(seen.values())


def _s4_canonical_coform(C, G):
    """Lex-minimal 6-conorm tuple over S4 index relabellings (class signature)."""
    import itertools
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    best = None
    for perm in itertools.permutations(range(4)):
        vals = []
        for a, b in pairs:
            ia, ib = perm[a], perm[b]
            vals.append(round(-_dotG(C[ia], C[ib], G), 10))
        t = tuple(vals)
        if best is None or t < best:
            best = t
    return best


def selling_closure_representatives(cell, tol_rel=1e-9, angle_sigma=None):
    """One obtuse superbase per isometry class (diagnostic / class count).

    Deduplicates the full closure by the S4-canonical coform (lex-minimal
    6-tuple of conorms under index relabelling). This is *not* a substitute
    for matching over the full closure: members within one class are related
    by lattice automorphisms that form the reindexing coset. Use
    :func:`selling_superbase_closure` for certification / coset recovery.
    """
    G = _metric(cell)
    closure = selling_superbase_closure(
        cell, tol_rel=tol_rel, angle_sigma=angle_sigma,
    )
    by_class = {}
    for C in closure:
        sig = _s4_canonical_coform(C, G)
        by_class.setdefault(sig, C)
    return list(by_class.values())


def closure_class_count(cell, tol_rel=1e-9, angle_sigma=None):
    """Number of isometry classes in the closure (1 for V1; up to 4 for V5)."""
    return len(selling_closure_representatives(
        cell, tol_rel=tol_rel, angle_sigma=angle_sigma,
    ))
