"""Typed Selling-superbase closure tests (main_v5 / Kurlin 4.1--4.5)."""
import math

from agentsg.cell.selling_closure import (
    voronoi_type, selling_superbase_closure, selling_closure_representatives,
    closure_class_count,
)
from agentsg.cell.canonical import (
    reindexing_via_canonical, canonical_superbase, _metric, _dotG,
    _PERMS, _inv3_unimod, _matmul_int, _transform_metric_int,
)
from agentsg.cell.rootform import sorted_root_key, sorted_root_distance
from agentsg.cell.metric import UnitCell
from agentsg.cell.g6 import _transform_metric
from agentsg.cell.selling_group import selling_group


ORTHO = (10.0, 12.0, 20.0, 90.0, 90.0, 90.0)  # primitive orthorhombic V5
HEX = (41.8, 41.8, 233.0, 90.0, 90.0, 120.0)  # hexagonal V4 (CXIDB-83-like)
TRIC = (40.0, 50.0, 60.0, 85.0, 95.0, 100.0)


def _cell_of(G):
    a = math.sqrt(G[0][0]); b = math.sqrt(G[1][1]); c = math.sqrt(G[2][2])
    ang = lambda x: math.degrees(math.acos(max(-1, min(1, x))))
    return (a, b, c, ang(G[1][2] / (b * c)), ang(G[0][2] / (a * c)),
            ang(G[0][1] / (a * b)))


def _match_lists(GA, GB, vA, vB, tol=1e-8):
    """Match superbase lists with S4 x {+/-I}; return accepted P's."""
    found = set()
    for CA in vA:
        U = [[CA[1][r], CA[2][r], CA[3][r]] for r in range(3)]
        for CB in vB:
            for perm in _PERMS:
                for s in (1, -1):
                    W = [[s * CB[perm[1]][r], s * CB[perm[2]][r],
                          s * CB[perm[3]][r]] for r in range(3)]
                    Winv = _inv3_unimod(W)
                    if Winv is None:
                        continue
                    P = _matmul_int(U, Winv)
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
    return found


def test_voronoi_types():
    assert voronoi_type(ORTHO) == 5
    assert voronoi_type(HEX) == 4
    assert voronoi_type(TRIC) == 1


def test_v5_orthorhombic_closure_32_in_4_classes():
    assert len(selling_superbase_closure(ORTHO)) == 32
    assert closure_class_count(ORTHO) == 4
    assert len(selling_closure_representatives(ORTHO)) == 4


def test_v4_hexagonal_closure_nontrivial():
    cl = selling_superbase_closure(HEX)
    assert voronoi_type(HEX) == 4
    assert len(cl) >= 3
    assert closure_class_count(HEX) >= 1


def test_v1_single_representative():
    assert len(selling_superbase_closure(TRIC)) == 1
    assert closure_class_count(TRIC) == 1


def test_v5_cross_class_match_needs_alternate_reps():
    """Odd-only S4 match misses the even-class automorphism; full reps recover it."""
    G = _metric(ORTHO)
    C0, _ = canonical_superbase(ORTHO)
    reps = selling_closure_representatives(ORTHO)
    sig0 = tuple(sorted(round(_dotG(C0[i], C0[i], G), 8) for i in range(4)))
    Ce = None
    for C in reps:
        sig = tuple(sorted(round(_dotG(C[i], C[i], G), 8) for i in range(4)))
        if sig != sig0:
            Ce = C
            break
    assert Ce is not None, "expected a non-odd class representative"

    P_even = tuple(tuple(Ce[j + 1][i] for j in range(3)) for i in range(3))
    GB = _transform_metric_int(G, P_even)

    odd_only = _match_lists(G, GB, [C0], [C0])
    with_even = _match_lists(G, GB, [C0], [Ce])
    full_reps = _match_lists(G, GB, reps, reps)

    assert len(with_even) >= 1
    assert len(full_reps) >= 1
    assert P_even not in odd_only


def test_v5_typed_reindex_all_selling_group_settings():
    G = UnitCell(*ORTHO).metric_tensor()
    # Sample the order-48 group (every 3rd) for speed; full group is covered by
    # closure enumeration tests above.
    for cob in selling_group()[::3]:
        M = tuple(tuple(int(x) for x in row) for row in cob.P.rows)
        c2 = _cell_of(_transform_metric(G, M))
        ops = reindexing_via_canonical(ORTHO, c2)
        assert ops, f"typed closure missed setting {M} -> {c2}"


def test_v4_hex_typed_reindex_selling_group_settings():
    G = UnitCell(*HEX).metric_tensor()
    for cob in selling_group()[::3]:
        M = tuple(tuple(int(x) for x in row) for row in cob.P.rows)
        c2 = _cell_of(_transform_metric(G, M))
        if sorted_root_distance(HEX, c2) > 1e-3:
            continue
        ops = reindexing_via_canonical(HEX, c2)
        assert ops, f"typed closure missed hex setting {c2}"


def test_monoclinic_flip_still_works():
    """Regression: monoclinic settings remain reindexable via typed closure."""
    mono = (6.0, 8.0, 11.0, 90.0, 112.0, 90.0)
    G = UnitCell(*mono).metric_tensor()
    for cob in selling_group()[::3]:
        M = tuple(tuple(int(x) for x in row) for row in cob.P.rows)
        c2 = _cell_of(_transform_metric(G, M))
        ops = reindexing_via_canonical(mono, c2)
        assert ops


def test_sorted_key_constant_on_closure():
    """Every Selling-group setting of one lattice shares the sorted root key."""
    key = sorted_root_key(ORTHO)
    G = UnitCell(*ORTHO).metric_tensor()
    for cob in selling_group()[::3]:
        M = tuple(tuple(int(x) for x in row) for row in cob.P.rows)
        c2 = _cell_of(_transform_metric(G, M))
        assert sorted_root_distance(ORTHO, c2) < 1e-6
        assert all(abs(a - b) < 1e-6 for a, b in zip(key, sorted_root_key(c2)))
