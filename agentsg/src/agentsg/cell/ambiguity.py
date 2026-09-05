"""
Indexing-ambiguity (reindexing) operators for serial crystallography.

In a serial dataset every frame indexes to the same cell and space group, so the
set of reindexing operators is a *dataset-level constant* -- computing it per
frame (or, worse, regenerating all unimodular matrices per call) is wasted work.
The exact, minimal object is the coset decomposition of the crystal's Laue group
in the lattice holohedry:

    reindexing operators  =  left-coset representatives of  L_Laue  in  M_lattice
    number of ambiguities =  |M_lattice| / |L_Laue|

The Laue group (crystal point group + inversion) is used rather than the bare
rotation group because Friedel's law makes the diffraction pattern
centrosymmetric; the lattice holohedry is centrosymmetric too. The count is
typically 1 (no ambiguity), 2, 3 or 4 -- the ambiguity is non-trivial only when
the lattice carries higher metric symmetry than the diffraction symmetry
(merohedry / pseudo-merohedry). Examples: P4 -> 2 (h,k,l vs k,-h... i.e. the
in-plane two-fold), P3 -> 4, F23 -> 2, all orthorhombic and lower (in their own
metric) -> 1.

Everything here is EXACT integer matrix algebra, so the cached operator set does
not drift over a million frames. The result is memoised on
(space-group key, rounded reduced-cell signature, tolerance).

Attribution and scope
---------------------
The operator-generation method is NOT new: selecting the reindexing operators by
coset decomposition of the crystal Laue group in a tolerance-widened lattice
group is exactly the approach used by dials.cosym (Gildea & Winter, Acta Cryst.
D74, 405-410, 2018), extending Brehm & Diederichs (Acta Cryst. D70, 101-109,
2014). The "large tolerance to accommodate pseudosymmetry" choice is theirs too.
This module is a clean, dependency-free, exact-arithmetic re-implementation of
that standard method, not a new algorithm.

On the reduction flip specifically (why the surfaced coset must include
cell-choice transforms, and how production software handles the same problem),
see docs/REDUCTION_FLIP_LITERATURE.md -- it traces the boundary-discontinuity
literature (Grosse-Kunstleve et al. 2004; Andrews & Bernstein 1988/2014) and the
combinatorial candidate-enumeration approach used by dials.refine_bravais_settings
and dials.cosym.

IMPORTANT limitation of the reference-anchored resolver
(:class:`ReindexingReference`): it chooses a branch by *metric geometry* (which
operator brings the reference cell closest to the frame cell). That is valid and
cheap for **pseudo-merohedral / cell-choice** ambiguity, where the branches have
slightly different metrics (e.g. a monoclinic cell with beta near 90, or the
a~b Niggli reduction flip). It is BLIND to **true merohedral / polar**
ambiguity, where every branch gives an identical metric (e.g. P4 with a==b
exactly, or P3_1 vs P3_2): there the ambiguity can only be resolved by analysing
reflection *intensities* (dials.cosym's intensity-correlation clustering, or
cctbx.xfel's reindex_to_reference against model F_calc). This module does not
attempt intensity-based resolution.
"""
from __future__ import annotations
from fractions import Fraction as Fr
from functools import lru_cache

from ..linalg import Matrix3, Vector3, IDENTITY3
from ..symmetry_op import SymmetryOp
from ..group import close_group, point_group
from ..space_groups import space_group, SpaceGroup
from ..lattice_symmetry import tolerance_metric_symmetry
from .reduction import niggli_reduce

_NEG_I = Matrix3([[Fr(-1), Fr(0), Fr(0)],
                  [Fr(0), Fr(-1), Fr(0)],
                  [Fr(0), Fr(0), Fr(-1)]])


def _laue_matrices(sg: SpaceGroup) -> frozenset:
    """Rotation matrices of the crystal Laue group (point group closed with -I)."""
    rots = point_group(sg.operations())
    ops = [SymmetryOp(R, Vector3((0, 0, 0))) for R in rots]
    ops.append(SymmetryOp(_NEG_I, Vector3((0, 0, 0))))
    return frozenset(op.W for op in close_group(ops))


def _coset_reps(L: frozenset, H: frozenset) -> list:
    """Left-coset representatives of subgroup L in group H (identity coset first)."""
    reps = [IDENTITY3]
    covered = set(L)
    # deterministic order for reproducibility
    for g in sorted(H, key=lambda M: M.rows):
        if g in covered:
            continue
        reps.append(g)
        covered |= {g @ l for l in L}
    return reps


@lru_cache(maxsize=256)
def _cached_ambiguity(sg_key, cell_sig, M_sig, len_tol, ang_tol):
    """Coset reps in the *input* basis.

    ``cell_sig`` is the Niggli-reduced cell (cache coalescing). ``M_sig`` is the
    integer change-of-basis input→reduced. Metric symmetry H is computed in the
    reduced basis; the crystal Laue group L is conjugated into that basis so the
    quotient is well-defined, then coset representatives are mapped back to the
    input basis via M. No special cases: just change-of-basis conjugation.
    """
    sg = space_group(sg_key)
    reduced = tuple(v / 1000.0 for v in cell_sig)
    M = Matrix3([[Fr(x) for x in row] for row in M_sig])
    Minv = M.inverse()
    L_in = _laue_matrices(sg)
    L = frozenset(Minv @ R @ M for R in L_in)
    H = frozenset(op.W for op in
                  tolerance_metric_symmetry(reduced, length_tol_pct=len_tol, angle_tol_deg=ang_tol))
    if not (L <= H):
        reps_in = [IDENTITY3]
    else:
        reps_red = _coset_reps(L, H)
        reps_in = [M @ R @ Minv for R in reps_red]
    # Deterministic order: identity first, then by matrix entries.
    reps_in = sorted(reps_in, key=lambda R: (R != IDENTITY3, R.rows))
    if IDENTITY3 in reps_in:
        reps_in.remove(IDENTITY3)
        reps_in.insert(0, IDENTITY3)
    return tuple(SymmetryOp(R, Vector3((0, 0, 0))) for R in reps_in)


def reindexing_ambiguity_operators(space_group_key, cell,
                                   length_tol_pct: float = 2.0,
                                   angle_tol_deg: float = 2.0):
    """Return the reindexing-ambiguity operators for a (space group, cell).

    Parameters
    ----------
    space_group_key : space-group number, Hermann-Mauguin or Hall symbol, or a
        SpaceGroup instance.
    cell : (a, b, c, alpha, beta, gamma), angles in degrees. It is Niggli-reduced
        internally so equivalent cells hit the same cache entry; operators are
        always returned in the *input* basis (Laue and metric symmetry are
        aligned through the Niggli change of basis).
    length_tol_pct : tolerance (percent) on edge lengths for the metric-symmetry
        determination.
    angle_tol_deg : tolerance (degrees) on angles.

    Returns a tuple of :class:`SymmetryOp` (exact integer rotations, zero
    translation) -- coset representatives of the crystal Laue group in the
    *tolerance* metric-automorphism group, identity first. Because the quotient
    is taken within the tolerance group, the result includes not only the exact
    reindexings but also the pseudo-symmetry branches (e.g. a monoclinic cell
    with beta near 90 gets its pseudo-orthorhombic partner) and the cell-choice
    transforms across Niggli reduction boundaries. The result is memoised; the
    same (space group, reduced cell, Niggli CoB, tolerances) never recomputes.

    Apply an operator to Miller indices with :func:`apply_to_hkl_batch` (or
    ``op.W`` directly). Picking the correct branch per frame (correlation to a
    reference) is left to the caller; this function supplies the *candidates*.
    """
    if isinstance(space_group_key, SpaceGroup):
        sg_key = space_group_key.number
    else:
        sg_key = space_group_key
    # canonicalise the cell via Niggli reduction, then a milliangstrom/millidegree
    # integer signature so nearby cells share a cache slot. Keep M so Laue (input
    # basis) and H (reduced basis) are compared in the same frame.
    reduced, M_raw = niggli_reduce(*cell)
    cell_sig = tuple(int(round(v * 1000)) for v in reduced)
    M_sig = tuple(tuple(int(x) for x in row) for row in M_raw)
    return _cached_ambiguity(
        sg_key, cell_sig, M_sig, round(length_tol_pct, 3), round(angle_tol_deg, 3),
    )


def apply_to_hkl_batch(op, hkl):
    """Apply a reindexing operator to an (N, 3) array of Miller indices.

    ``op`` is a SymmetryOp (or Matrix3); ``hkl`` is any nested sequence or numpy
    array of shape (N, 3). Returns a list of 3-tuples of ints. Miller indices
    transform WITH the rotation matrix (as row vectors): h' = h W. If numpy is
    available the caller can instead do ``hkl @ W`` directly -- this helper keeps
    the package dependency-free and works on plain lists.
    """
    W = op.W if isinstance(op, SymmetryOp) else op
    rows = W.rows
    out = []
    for h in hkl:
        hh, kk, ll = int(h[0]), int(h[1]), int(h[2])
        out.append((
            int(hh * rows[0][0] + kk * rows[1][0] + ll * rows[2][0]),
            int(hh * rows[0][1] + kk * rows[1][1] + ll * rows[2][1]),
            int(hh * rows[0][2] + kk * rows[1][2] + ll * rows[2][2]),
        ))
    return out


def ambiguity_index(space_group_key, cell,
                    length_tol_pct: float = 2.0, angle_tol_deg: float = 2.0) -> int:
    """Number of indexing ambiguities = |tolerance metric symmetry| / |Laue group|."""
    return len(reindexing_ambiguity_operators(
        space_group_key, cell, length_tol_pct, angle_tol_deg))


class ReindexingReference:
    """A fixed reference setting for a serial dataset.

    Serial crystallography indexes many frames to (approximately) the same cell
    and space group. The correct, stable design is to anchor to ONE reference
    basis and compute the reindexing-operator set ONCE for that reference -- never
    re-reducing per frame. Per-frame Niggli reduction is discontinuous (a hair of
    cell noise near a reduction boundary flips the canonical cell to a different
    but equivalent one), so anything keyed on the per-frame reduced cell is
    fragile. Because the reference tolerance metric-automorphism group already
    contains those cell-choice / reduction-flip transforms as elements, resolving
    each frame against the fixed reference set is stable across the boundary.

    Construct once with the reference cell + space group; then call
    :meth:`resolve` per frame with that frame's cell (and, optionally, an integer
    reindexing candidate) to pick the operator that best brings the frame onto the
    reference. The candidate operator set is a dataset constant -- O(1) in the
    number of frames, no per-frame enumeration.
    """
    __slots__ = ("cell", "operators", "_G_ref", "_len_tol", "_ang_tol",
                 "_sg_key", "_laue_rows", "_ref_asu")

    def __init__(self, space_group_key, cell, length_tol_pct: float = 2.0,
                 angle_tol_deg: float = 2.0):
        self.cell = tuple(cell)
        self._len_tol = length_tol_pct
        self._ang_tol = angle_tol_deg
        self._sg_key = (space_group_key.number
                        if isinstance(space_group_key, SpaceGroup) else space_group_key)
        self.operators = reindexing_ambiguity_operators(
            space_group_key, cell, length_tol_pct, angle_tol_deg)
        from .metric import UnitCell
        self._G_ref = UnitCell(*cell).metric_tensor()
        self._laue_rows = None      # lazily built on first intensity call
        self._ref_asu = None        # reference intensities merged to ASU

    def __len__(self):
        return len(self.operators)

    def resolve(self, frame_cell):
        """Pick the reindexing operator bringing ``frame_cell`` onto the reference.

        Returns (best_operator, residual) where residual is the max metric
        mismatch (a blend of percent-length and degree-angle deviation) after
        applying the operator to the reference metric and comparing to the frame.
        The operator is one of :attr:`operators`; identity means the frame is
        already in the reference branch. This is the per-frame O(1) step -- it
        loops only over the handful of dataset-constant operators, never
        enumerating or re-reducing.
        """
        from .metric import UnitCell, params_from_metric
        Gf = UnitCell(*frame_cell).metric_tensor()
        pf = params_from_metric(Gf)
        best = None
        best_res = None
        for op in self.operators:
            W = op.W.rows
            # transform reference metric by this operator: G' = W^T G_ref W
            WtG = [[sum(W[k][i] * self._G_ref[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
            Gp = [[sum(WtG[i][k] * W[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
            pp = params_from_metric(Gp)
            dl = max(abs(pp[i] - pf[i]) / pf[i] * 100.0 for i in range(3))
            da = max(abs(pp[3 + i] - pf[3 + i]) for i in range(3))
            res = max(dl, da)
            if best_res is None or res < best_res:
                best_res = res
                best = op
        return best, best_res

    # -- intensity tie-breaker -------------------------------------------------
    def set_reference_intensities(self, intensities):
        """Register the reference reflection intensities: {(h,k,l): I}.

        Stored merged onto ASU keys so per-frame resolution is a fast dict
        lookup. Call once after construction; then use :meth:`resolve_intensities`
        per frame. This is what breaks the branch tie by *physics* rather than
        geometry -- required for true merohedral ambiguity, where every geometric
        branch has an identical metric.
        """
        self._laue_rows = _laue_rows(self._sg_key)
        self._ref_asu = _merge_to_asu(
            intensities, self._laue_rows, style="ccp4", sg_key=self._sg_key,
        )
        return self

    def resolve_intensities(self, frame_intensities, min_common: int = 3):
        """Pick the branch whose reindexed intensities best correlate with the reference.

        ``frame_intensities`` is {(h,k,l): I} for one frame. Each candidate coset
        operator is applied to the frame's Miller indices, merged to the ASU, and
        Pearson-correlated against the reference. Returns an
        :class:`AmbiguityResolution` exposing EVERY branch's CC (the geometric
        choices, surfaced) and the winning operator + margin.

        The geometric layer enumerates the candidates; the intensities decide.
        """
        if self._ref_asu is None:
            raise ValueError("call set_reference_intensities(...) before resolve_intensities(...)")
        laue = self._laue_rows
        items = list(frame_intensities.items() if isinstance(frame_intensities, dict)
                     else frame_intensities)
        scores = []
        for op in self.operators:
            W = op.W.rows
            reindexed = []
            for h, I in items:
                hh, kk, ll = int(h[0]), int(h[1]), int(h[2])
                hp = (hh * W[0][0] + kk * W[1][0] + ll * W[2][0],
                      hh * W[0][1] + kk * W[1][1] + ll * W[2][1],
                      hh * W[0][2] + kk * W[1][2] + ll * W[2][2])
                reindexed.append((hp, I))
            merged = _merge_to_asu(
                reindexed, laue, style="ccp4", sg_key=self._sg_key,
            )
            common = [(merged[k], self._ref_asu[k]) for k in merged if k in self._ref_asu]
            cc = _pearson(common) if len(common) >= min_common else float("-inf")
            scores.append((op, cc, len(common)))
        scores.sort(key=lambda t: t[1], reverse=True)
        best = scores[0][0]
        margin = (scores[0][1] - scores[1][1]) if len(scores) > 1 and scores[1][1] != float("-inf") else float("inf")
        return AmbiguityResolution(best, scores, margin)


# ----------------------------------------------------------------------------
# Intensity tie-breaker
# ----------------------------------------------------------------------------
# The geometric layer above SURFACES the candidate branches (the coset
# operators). Which branch is physically correct is a property of the structure,
# not of the cell metric, so the tie is broken by reflection INTENSITIES --
# exactly the Brehm & Diederichs (2014) / dials.cosym / reindex_to_reference
# idea. This is the piece a metric-only method cannot supply, and it is the ONLY
# thing that resolves a *true merohedral* ambiguity (identical metrics on every
# branch). Kept dependency-free: Pearson CC in plain Python.

class AmbiguityResolution:
    """Result of an intensity-based branch decision.

    Attributes
    ----------
    best : SymmetryOp        the chosen reindexing operator (identity = frame
                             already in the reference branch).
    scores : list of (SymmetryOp, cc, n_common)
                             every candidate branch with its correlation to the
                             reference and the number of common reflections --
                             the geometric choices, surfaced, so the decision is
                             inspectable rather than hidden.
    margin : float           best_cc minus second-best_cc; small margin => the
                             intensities did not clearly discriminate (e.g. weak
                             data, or a genuine tie).
    """
    __slots__ = ("best", "scores", "margin")

    def __init__(self, best, scores, margin):
        self.best = best
        self.scores = scores
        self.margin = margin

    def __repr__(self):
        ccs = ", ".join(f"{cc:.3f}" for _, cc, _ in self.scores)
        return f"AmbiguityResolution(best_cc={self.scores[0][1]:.3f}, margin={self.margin:.3f}, ccs=[{ccs}])"


def _laue_rows(sg_key):
    """Rotation matrix rows for the crystal Laue group."""
    sg = space_group(sg_key) if not isinstance(sg_key, SpaceGroup) else sg_key
    return [W.rows for W in _laue_matrices(sg)]


def _map_to_asu_lexmax(h, laue_rows):
    """Lexicographically-maximum Laue image (legacy merging key)."""
    hh, kk, ll = h
    best = None
    for r in laue_rows:
        img = (hh * r[0][0] + kk * r[1][0] + ll * r[2][0],
               hh * r[0][1] + kk * r[1][1] + ll * r[2][1],
               hh * r[0][2] + kk * r[1][2] + ll * r[2][2])
        if best is None or img > best:
            best = img
    return best


def _map_to_asu(h, laue_rows, style="ccp4", sg_key=None):
    """Canonical ASU representative of a reflection under the Laue group.

    ``style='ccp4'`` (default) uses :class:`agentsg.asu.ReciprocalAsu` when
    ``sg_key`` is available; otherwise falls back to lex-max. ``style='lexmax'``
    always returns the lexicographically-maximum Laue image.
    """
    h = (int(h[0]), int(h[1]), int(h[2]))
    if style == "lexmax" or sg_key is None:
        return _map_to_asu_lexmax(h, laue_rows)
    if style != "ccp4":
        raise ValueError("style must be 'ccp4' or 'lexmax'")
    from ..asu import ReciprocalAsu
    from ..symmetry_op import SymmetryOp
    rasu = ReciprocalAsu.from_space_group(sg_key)
    ops = [SymmetryOp(Matrix3(r), Vector3((0, 0, 0))) for r in laue_rows]
    hkl, _isym = rasu.to_asu(h, ops)
    return hkl


def _pearson(pairs):
    """Compute Pearson correlation coefficient between pairs of scalar observations."""
    n = len(pairs)
    if n < 2:
        return 0.0
    sx = sum(x for x, _ in pairs); sy = sum(y for _, y in pairs)
    mx = sx / n; my = sy / n
    sxx = syy = sxy = 0.0
    for x, y in pairs:
        dx = x - mx; dy = y - my
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return sxy / (sxx * syy) ** 0.5


def _merge_to_asu(intensities, laue_rows, style="ccp4", sg_key=None):
    """Fold a {(h,k,l): I} mapping onto ASU keys, averaging duplicates."""
    acc = {}
    for h, I in (intensities.items() if isinstance(intensities, dict) else intensities):
        key = _map_to_asu(
            (int(h[0]), int(h[1]), int(h[2])), laue_rows,
            style=style, sg_key=sg_key,
        )
        if key in acc:
            s, c = acc[key]; acc[key] = (s + I, c + 1)
        else:
            acc[key] = (I, 1)
    return {k: s / c for k, (s, c) in acc.items()}


# ----------------------------------------------------------------------------
# Surfacing the complete geometrically-allowed operator set
# ----------------------------------------------------------------------------
# Contract for cell comparison / reindexing: the geometric layer returns the
# COMPLETE set of operators allowed by the metric within tolerance -- reduction
# flips and cell-choice transforms included -- and *nothing here decides* which
# is correct. That is the intensity layer's job. This function makes the set and
# each operator's geometric provenance explicit and inspectable.

class GeometricOperator:
    """One geometrically-allowed reindexing operator, with provenance.

    Attributes
    ----------
    op : SymmetryOp                  the exact integer reindexing operator.
    residual : float                 metric mismatch (max of %-length and
                                     deg-angle) after applying it to the cell;
                                     0 for an exact metric symmetry.
    is_identity : bool               True for the reference branch.
    is_metric_symmetry : bool        True if the operator maps the cell metric
                                     onto itself within tolerance (residual ~ 0)
                                     -- i.e. geometry alone cannot distinguish it
                                     from identity (merohedral-type), so only
                                     intensities can decide this branch.
    """
    __slots__ = ("op", "residual", "is_identity", "is_metric_symmetry")

    def __init__(self, op, residual, is_identity, is_metric_symmetry):
        self.op = op
        self.residual = residual
        self.is_identity = is_identity
        self.is_metric_symmetry = is_metric_symmetry

    def __repr__(self):
        rows = tuple(tuple(int(x) for x in r) for r in self.op.W.rows)
        tag = "identity" if self.is_identity else ("metric-sym" if self.is_metric_symmetry else "cell-change")
        return f"GeometricOperator({rows}, residual={self.residual:.4f}, {tag})"


def surface_geometric_operators(space_group_key, cell,
                                length_tol_pct: float = 2.0,
                                angle_tol_deg: float = 2.0,
                                metric_sym_tol: float = 1e-6):
    """Return the COMPLETE list of geometrically-allowed reindexing operators.

    This is the authoritative surface consumed by cell comparison and reindexing:
    the coset of the crystal Laue group in the tolerance metric-automorphism
    group of ``cell`` -- exactly the operators the metric permits within
    tolerance, reduction flips and cell-choice transforms included. Identity is
    first.

    Each operator is wrapped in :class:`GeometricOperator` carrying its metric
    residual and two flags: ``is_metric_symmetry`` marks the branches geometry
    *cannot* tell from the identity (residual ~ 0 -- the merohedral-type
    ambiguity that needs intensities), while the rest are distinguishable
    cell-change operators (pseudo-merohedral / reduction-flip) that geometry
    alone can order by residual. The geometric layer only *surfaces and
    annotates*; it does not decide.
    """
    from .metric import UnitCell, params_from_metric
    ops = reindexing_ambiguity_operators(space_group_key, cell,
                                         length_tol_pct, angle_tol_deg)
    G = UnitCell(*cell).metric_tensor()
    p0 = params_from_metric(G)
    out = []
    for i, op in enumerate(ops):
        W = op.W.rows
        WtG = [[sum(W[k][r] * G[k][j] for k in range(3)) for j in range(3)] for r in range(3)]
        Gp = [[sum(WtG[r][k] * W[k][j] for k in range(3)) for j in range(3)] for r in range(3)]
        pp = params_from_metric(Gp)
        dl = max(abs(pp[j] - p0[j]) / p0[j] * 100.0 for j in range(3))
        da = max(abs(pp[3 + j] - p0[3 + j]) for j in range(3))
        res = max(dl, da)
        out.append(GeometricOperator(op, res, i == 0, res <= metric_sym_tol))
    return out
