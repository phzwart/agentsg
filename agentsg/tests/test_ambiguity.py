"""Indexing-ambiguity / reindexing-operator tests for serial crystallography.

The reindexing operators are the coset representatives of the crystal Laue group
in the (tolerance) metric-automorphism group of the cell. This is a dataset-level
constant; the reference-anchored resolver is stable across Niggli reduction
boundaries where the per-frame reduced cell flips.
"""
import pytest
from agentsg.cell.ambiguity import (
    reindexing_ambiguity_operators, ambiguity_index, apply_to_hkl_batch,
    ReindexingReference,
)
from agentsg.linalg import IDENTITY3


@pytest.mark.parametrize("sg,cell,expected", [
    (75, (50, 50, 80, 90, 90, 90), 2),      # P4  -> tetragonal metric, 4/mmm : 4/m
    (75, (80, 80, 50, 90, 90, 90), 2),      # same SG with c unique short (Niggli reorders)
    (19, (40, 50, 60, 90, 90, 90), 1),      # P212121 orthorhombic -> none
    (1, (40, 50, 60, 88, 92, 103), 1),      # P1 triclinic -> none
    (143, (50, 50, 80, 90, 90, 120), 4),    # P3 -> 6/mmm : -3
    (196, (50, 50, 50, 90, 90, 90), 2),     # F23 cubic -> m-3m : m-3
])
def test_known_ambiguity_counts(sg, cell, expected):
    # tight tolerance -> exact metric symmetry
    assert ambiguity_index(sg, cell, length_tol_pct=0.1, angle_tol_deg=0.1) == expected


def test_identity_is_first_operator():
    ops = reindexing_ambiguity_operators(75, (50, 50, 80, 90, 90, 90))
    assert ops[0].W == IDENTITY3


def test_operators_are_exact_integers_zero_translation():
    for op in reindexing_ambiguity_operators(143, (50, 50, 80, 90, 90, 120)):
        assert op.w.v == (0, 0, 0)
        for row in op.W.rows:
            for x in row:
                assert x.denominator == 1


def test_pseudosymmetry_caught_by_tolerance():
    # monoclinic C2 with beta = 90.5: exact says 1, tolerance says 2
    cell = (40.0, 50.0, 60.0, 90.0, 90.5, 90.0)
    assert ambiguity_index(5, cell, length_tol_pct=0.1, angle_tol_deg=0.1) == 1
    assert ambiguity_index(5, cell, length_tol_pct=2.0, angle_tol_deg=2.0) == 2


def test_pseudosymmetry_collapses_when_clearly_monoclinic():
    cell = (40.0, 50.0, 60.0, 90.0, 95.0, 90.0)
    assert ambiguity_index(5, cell, length_tol_pct=2.0, angle_tol_deg=2.0) == 1


def test_apply_to_hkl_batch_returns_ints():
    ops = reindexing_ambiguity_operators(75, (50, 50, 80, 90, 90, 90))
    out = apply_to_hkl_batch(ops[1], [(1, 2, 3), (4, 0, 1)])
    assert all(isinstance(v, int) for h in out for v in h)


def test_caching_reuses_result():
    from agentsg.cell.ambiguity import _cached_ambiguity
    _cached_ambiguity.cache_clear()
    reindexing_ambiguity_operators(75, (50.0, 50.0, 80.0, 90, 90, 90))
    # identical cell -> cache hit
    reindexing_ambiguity_operators(75, (50.0, 50.0, 80.0, 90, 90, 90))
    info = _cached_ambiguity.cache_info()
    assert info.hits >= 1


def test_reference_resolver_stable_across_reduction_boundary():
    # same crystal + noise straddling the a~b Niggli boundary must all resolve
    # to the identity branch with small residual
    ref = ReindexingReference(3, (40.0, 40.0, 60.0, 90, 91, 90),
                              length_tol_pct=2.0, angle_tol_deg=2.0)
    for f in [(40.00, 40.03, 60.0, 90, 91, 90),
              (40.00, 39.97, 60.0, 90, 91, 90),
              (40.02, 40.00, 60.0, 90, 91, 90),
              (39.98, 40.00, 60.0, 90, 91, 90)]:
        op, res = ref.resolve(f)
        assert op.W == IDENTITY3
        assert res < 0.5


def test_reference_resolver_detects_flipped_frame():
    # a frame that IS the reindexed partner should resolve to a non-identity op
    ref = ReindexingReference(75, (50.0, 50.0, 80.0, 90, 90, 90),
                              length_tol_pct=1.0, angle_tol_deg=1.0)
    assert len(ref) == 2  # P4 has a 2-fold ambiguity


# --- intensity tie-breaker (true merohedral case geometry cannot resolve) ----
def _p4_reference():
    from agentsg.cell.ambiguity import ReindexingReference
    ref = ReindexingReference(75, (50.0, 50.0, 80.0, 90, 90, 90),
                              length_tol_pct=0.1, angle_tol_deg=0.1)
    hkls = [(h, k, l) for h in range(1, 7) for k in range(1, 7) for l in range(1, 5)]
    trueI = lambda h, k, l: 100.0 + 30.0 * h - 10.0 * k + 5.0 * l  # asymmetric in h,k
    ref.set_reference_intensities({hkl: trueI(*hkl) for hkl in hkls})
    return ref, hkls, trueI


def test_geometry_cannot_break_true_merohedral_tie():
    from agentsg.cell.ambiguity import ReindexingReference
    ref = ReindexingReference(75, (50.0, 50.0, 80.0, 90, 90, 90),
                              length_tol_pct=0.1, angle_tol_deg=0.1)
    assert len(ref.operators) == 2
    # identical metric on both branches -> residual is zero for the true cell
    _, res = ref.resolve((50.0, 50.0, 80.0, 90, 90, 90))
    assert res < 1e-9


def test_intensity_resolves_correctly_indexed_frame():
    ref, hkls, trueI = _p4_reference()
    frame = {hkl: trueI(*hkl) for hkl in hkls}
    r = ref.resolve_intensities(frame)
    assert r.best.W == ref.operators[0].W          # identity branch
    assert r.scores[0][1] > 0.9 and r.margin > 0.5


def test_intensity_resolves_misindexed_frame():
    ref, hkls, trueI = _p4_reference()
    op1 = ref.operators[1]
    W = op1.W.rows
    def apply(h):
        hh, kk, ll = h
        return (hh * W[0][0] + kk * W[1][0] + ll * W[2][0],
                hh * W[0][1] + kk * W[1][1] + ll * W[2][1],
                hh * W[0][2] + kk * W[1][2] + ll * W[2][2])
    frame = {apply(hkl): trueI(*hkl) for hkl in hkls}
    r = ref.resolve_intensities(frame)
    assert r.best.W == op1.W                         # the mis-indexing branch
    assert r.scores[0][1] > 0.9


def test_resolve_intensities_requires_reference():
    from agentsg.cell.ambiguity import ReindexingReference
    ref = ReindexingReference(75, (50.0, 50.0, 80.0, 90, 90, 90))
    import pytest
    with pytest.raises(ValueError):
        ref.resolve_intensities({(1, 0, 0): 1.0})


# --- geometric surfacing contract -------------------------------------------
def test_surface_includes_reduction_flip():
    """The complete coset must surface the reduction-flip when geometry allows it."""
    from agentsg.cell.ambiguity import surface_geometric_operators
    gos = surface_geometric_operators(3, (40, 40, 60, 90, 91, 90), 2.0, 2.0)
    Ws = {tuple(tuple(int(x) for x in r) for r in g.op.W.rows) for g in gos}
    assert ((0, -1, 0), (-1, 0, 0), (0, 0, -1)) in Ws     # the a~b reduction flip
    assert len(gos) == 4


def test_surface_identity_first_and_zero_residual():
    from agentsg.cell.ambiguity import surface_geometric_operators
    gos = surface_geometric_operators(5, (40, 50, 60, 90, 90.5, 90), 2.0, 2.0)
    assert gos[0].is_identity and gos[0].residual < 1e-9


def test_surface_flags_true_merohedry_as_metric_symmetry():
    """True merohedral branch (P4, a==b) is residual-0 -> flagged geometry-blind."""
    from agentsg.cell.ambiguity import surface_geometric_operators
    gos = surface_geometric_operators(75, (50, 50, 80, 90, 90, 90), 0.1, 0.1)
    assert len(gos) == 2
    assert gos[1].is_metric_symmetry and gos[1].residual < 1e-6


def test_surface_pseudomerohedry_is_distinguishable():
    """Pseudo-orthorhombic partner has non-zero residual -> NOT geometry-blind."""
    from agentsg.cell.ambiguity import surface_geometric_operators
    gos = surface_geometric_operators(5, (40, 50, 60, 90, 90.5, 90), 2.0, 2.0)
    assert len(gos) == 2
    assert not gos[1].is_metric_symmetry and gos[1].residual > 1e-3


def test_clearly_monoclinic_surfaces_only_identity():
    from agentsg.cell.ambiguity import surface_geometric_operators
    gos = surface_geometric_operators(5, (40, 50, 60, 90, 95, 90), 2.0, 2.0)
    assert len(gos) == 1 and gos[0].is_identity
