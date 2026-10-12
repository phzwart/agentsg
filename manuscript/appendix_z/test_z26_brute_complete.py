"""z:brute-complete — when a {-1,0,1} unimodular search is complete.

Stratum-wide proposition (machine proof per Voronoi type): between any two
Selling-reduced settings of one lattice a relating operator with entries in
{-1,0,1} exists, but the full ambiguity coset escapes that ball for V4 and
V5 (entries 2 and 3); the closure route returns the whole coset. Between
different lattices the certificate radius governs and anisotropic pairs
overrule the radius-1 truncation. Niggli: empirical, incl. adversarial.
Also pins the Niggli CoB identity N^T G N == G(red).
"""
from __future__ import annotations

import itertools
import math
import random

import pytest

from helpers import (
    SEED_REINDEX_PROC, sample_unimodular, metric_tensor, transform_metric,
    cell_from_metric, det3, matmul_int, inv3, TYPE_CONORMS, cell_from_conorms,
    perturb_angles, key_l2,
)

pytestmark = [pytest.mark.zcheck]

# Fixed seed for this check
SEED_BRUTE = 29


def _max_abs_entry(M):
    return max(abs(int(x)) for row in M for x in row)


def _unimodular_pm1():
    from agentsg.cell.g6 import _unimodular_pm1
    return _unimodular_pm1()


def test_i_most_random_resettings_need_entry_mag_2():
    """(i) 200 random M with |entries|<=2: most have max|entry|>1."""
    rng = random.Random(SEED_BRUTE)
    Ms = sample_unimodular(rng, entry_max=2, n=200)
    n_large = sum(1 for M in Ms if _max_abs_entry(M) > 1)
    # manuscript observed 192/200; pin ±10%
    assert 170 <= n_large <= 200, f"got {n_large}/200 with max|entry|>1"


def test_iv_niggli_cob_identity_and_convergence():
    """(iv) N^T G N == G(red), |det|=1; converge on noisy cells."""
    from agentsg.cell.reduction import niggli_reduce
    from agentsg.cell.reduction import _transform_metric, _gram_from_params

    rng = random.Random(SEED_BRUTE + 1)
    base = (40.0, 50.0, 60.0, 80.0, 95.0, 110.0)
    n_ok = 0
    n_fail_conv = 0
    for _ in range(500):  # fast subset; full 10^4 is slow-marked below
        cell = (
            base[0] * (1 + rng.gauss(0, 0.02)),
            base[1] * (1 + rng.gauss(0, 0.02)),
            base[2] * (1 + rng.gauss(0, 0.02)),
            base[3] + rng.gauss(0, 0.5),
            base[4] + rng.gauss(0, 0.5),
            base[5] + rng.gauss(0, 0.5),
        )
        try:
            red, N = niggli_reduce(*cell, max_iter=2000)
        except RuntimeError:
            n_fail_conv += 1
            continue
        G = metric_tensor(cell)
        Gp = _transform_metric(G, N)
        Gr = _gram_from_params(*red)
        err = max(abs(Gp[i][j] - Gr[i][j]) for i in range(3) for j in range(3))
        scale = max(abs(x) for row in Gr for x in row) or 1.0
        assert err <= 1e-9 * max(scale, 1.0) or err < 1e-6
        assert abs(det3(N)) == 1
        n_ok += 1
    assert n_ok >= 480, f"too many non-convergences: fail={n_fail_conv}"
    # After the CoB fix, noisy convergence should be rare
    assert n_fail_conv / max(n_ok + n_fail_conv, 1) < 0.05


@pytest.mark.slow
def test_iv_niggli_cob_10k_noisy():
    """Full manuscript 10^4 noisy-cell CoB + convergence check."""
    from agentsg.cell.reduction import niggli_reduce, _transform_metric, _gram_from_params

    rng = random.Random(SEED_BRUTE + 2)
    base = (40.0, 50.0, 60.0, 80.0, 95.0, 110.0)
    n_fail = 0
    for _ in range(10_000):
        cell = (
            base[0] * (1 + rng.gauss(0, 0.02)),
            base[1] * (1 + rng.gauss(0, 0.02)),
            base[2] * (1 + rng.gauss(0, 0.02)),
            base[3] + rng.gauss(0, 0.5),
            base[4] + rng.gauss(0, 0.5),
            base[5] + rng.gauss(0, 0.5),
        )
        try:
            red, N = niggli_reduce(*cell, max_iter=2000)
        except RuntimeError:
            n_fail += 1
            continue
        G = metric_tensor(cell)
        err = max(
            abs(_transform_metric(G, N)[i][j] - _gram_from_params(*red)[i][j])
            for i in range(3) for j in range(3)
        )
        assert err < 1e-6
        assert abs(det3(N)) == 1
    assert n_fail / 10000 < 0.02


def _max_entry_closure_match(cell):
    """Max |P_ij| over typed-closure matching (canonical U W^{-1} path)."""
    from agentsg.cell.selling_closure import selling_superbase_closure
    from agentsg.cell.canonical import _inv3_unimod, _matmul_int, _PERMS

    cl = selling_superbase_closure(cell)
    worst = 0
    worst_P = None
    for CA in cl:
        U = [[CA[1][r], CA[2][r], CA[3][r]] for r in range(3)]
        for CB in cl:
            for perm in _PERMS:
                for s in (1, -1):
                    W = [
                        [s * CB[perm[1]][r], s * CB[perm[2]][r], s * CB[perm[3]][r]]
                        for r in range(3)
                    ]
                    Winv = _inv3_unimod(W)
                    if Winv is None:
                        continue
                    P = _matmul_int(U, Winv)
                    if abs(det3(P)) != 1:
                        continue
                    m = _max_abs_entry(P)
                    if m > worst:
                        worst = m
                        worst_P = P
    return worst, worst_P


def _max_entry_s4_orbit(cell):
    """Max |P_ij| within the S4×{±I} orbit of one reduced superbase."""
    from agentsg.cell.canonical import canonical_superbase, _inv3_unimod, _matmul_int, _PERMS

    C0, _ = canonical_superbase(cell)
    orb = []
    for perm in _PERMS:
        for s in (1, -1):
            orb.append([[s * C0[perm[i]][j] for j in range(3)] for i in range(4)])
    worst = 0
    for CA in orb:
        for drop_a in range(4):
            idx_a = [i for i in range(4) if i != drop_a]
            A = [[CA[idx_a[j]][r] for j in range(3)] for r in range(3)]
            if abs(det3(A)) != 1:
                continue
            for CB in orb:
                for drop_b in range(4):
                    idx_b = [i for i in range(4) if i != drop_b]
                    B = [[CB[idx_b[j]][r] for j in range(3)] for r in range(3)]
                    Binv = _inv3_unimod(B)
                    if Binv is None:
                        continue
                    P = _matmul_int(A, Binv)
                    if abs(det3(P)) != 1:
                        continue
                    worst = max(worst, _max_abs_entry(P))
    return worst


# ---------------------------------------------------------------------------
# (iii) Stratum-wide proposition.  On each Voronoi stratum the closure,
# in the coordinates of one obtuse superbase, is a FIXED set of integer
# superbases: the flip graph at the stratum's zero slots is decided by
# integer conorm-coefficient vectors, not by parameter values, so one
# breadth-first enumeration per type covers every lattice of that type.
# ---------------------------------------------------------------------------

_SLOTS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_PATTERNS = {  # which conorm slots (01,02,03,12,13,23) are free on the stratum
    1: (1, 1, 1, 1, 1, 1),
    2: (1, 1, 1, 1, 1, 0),
    3: (0, 1, 1, 1, 1, 0),
    4: (1, 1, 1, 1, 0, 0),
    5: (1, 1, 1, 0, 0, 0),
}


def _gram_coeff_mats(pattern):
    """One integer Gram matrix per free conorm parameter (G is linear)."""
    out = []
    for s, free in enumerate(pattern):
        if not free:
            continue
        p = [0] * 6
        p[s] = 1
        p01, p02, p03, p12, p13, p23 = p
        out.append((
            (p01 + p12 + p13, -p12, -p13),
            (-p12, p02 + p12 + p23, -p23),
            (-p13, -p23, p03 + p13 + p23),
        ))
    return out


def _conorm_coeffs(Gk, wi, wj):
    return tuple(
        -sum(wi[a] * G[a][b] * wj[b] for a in range(3) for b in range(3))
        for G in Gk
    )


def _canon_superbase(member):
    best = None
    for perm in itertools.permutations(range(4)):
        for s in (1, -1):
            t = tuple(tuple(s * x for x in member[p]) for p in perm)
            if best is None or t < best:
                best = t
    return best


def _symbolic_closure(Gk):
    """Flip-BFS with exact coefficient conorms; valid for every stratum point."""
    base = [(-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    seen = {_canon_superbase(base)}
    members, frontier = [base], [base]
    while frontier:
        nxt = []
        for m in frontier:
            for (i, j) in _SLOTS:
                if any(_conorm_coeffs(Gk, m[i], m[j])):
                    continue  # flip only at identically-zero slots
                k, l = [x for x in range(4) if x not in (i, j)]
                f = [None] * 4
                f[i] = tuple(-x for x in m[i])
                f[j] = m[j]
                f[k] = tuple(a + b for a, b in zip(m[k], m[i]))
                f[l] = tuple(a + b for a, b in zip(m[l], m[i]))
                assert all(sum(v[r] for v in f) == 0 for r in range(3))
                c = _canon_superbase(f)
                if c not in seen:
                    seen.add(c)
                    members.append(f)
                    nxt.append(f)
        frontier = nxt
    return members


def _member_bases(members):
    out = []
    for m in members:
        for drop in range(4):
            idx = [i for i in range(4) if i != drop]
            out.append([[m[idx[j]][r] for j in range(3)] for r in range(3)])
    return out


def _generic_automorphisms(Gk):
    """Unimodular Q with Q^T Gk Q == Gk for every coefficient matrix."""
    import numpy as np

    vals = np.arange(-2, 3)
    grids = np.meshgrid(*([vals] * 9), indexing="ij")
    M = np.stack([g.ravel() for g in grids], axis=1).reshape(-1, 3, 3)
    det = np.linalg.det(M.astype(float)).round()
    M = M[np.abs(det) == 1]
    keep = np.ones(len(M), dtype=bool)
    for G in Gk:
        Gm = np.array(G, dtype=np.int64)
        res = np.einsum("nji,jk,nkl->nil", M, Gm, M) - Gm
        keep &= ~np.any(res != 0, axis=(1, 2))
    return M[keep]


def test_iii_stratum_closures_fixed_and_obtuse():
    """Closure classes 1/2/3/6/16 (mod S4, sign), obtuse on the whole stratum."""
    expected = {1: 1, 2: 2, 3: 3, 4: 6, 5: 16}
    for vt, nexp in expected.items():
        Gk = _gram_coeff_mats(_PATTERNS[vt])
        mem = _symbolic_closure(Gk)
        assert len(mem) == nexp, (vt, len(mem))
        for m in mem:
            for (i, j) in _SLOTS:
                cc = _conorm_coeffs(Gk, m[i], m[j])
                assert all(c >= 0 for c in cc), (vt, m, (i, j), cc)


def test_iii_relating_operator_in_pm1_exists_for_every_pair():
    """Coset minimum: between any two Selling-reduced settings of any lattice,
    some relating operator has entries in {-1,0,1} (machine proof per type;
    signed permutations act on the outside and larger point groups only
    enlarge the coset, so the bound covers every lattice)."""
    import numpy as np

    for vt in (1, 2, 3, 4, 5):
        Gk = _gram_coeff_mats(_PATTERNS[vt])
        bases = _member_bases(_symbolic_closure(Gk))
        auts = _generic_automorphisms(Gk).astype(float)
        assert int(np.abs(auts).max()) == 1
        for A in bases:
            Af = np.array(A, dtype=float)
            for B in bases:
                Binv = np.linalg.inv(np.array(B, dtype=float))
                coset = np.einsum("ij,njk,kl->nil", Binv, auts, Af)
                ents = np.abs(np.round(coset)).max(axis=(1, 2)).astype(int)
                assert ents.min() <= 1, (vt, A, B)


def test_iii_coset_escapes_pm1_for_v4_v5_not_v1_v3():
    """Coset maximum: V1-V3 cosets stay inside {-1,0,1}; most V4/V5 reduced
    bases have SELF-coset operators with an entry of 2, and V5 cross-class
    pairs need an entry of 3. The radius-one list is provably not
    coset-complete after Selling reduction."""
    import numpy as np

    stats = {}
    for vt in (1, 2, 3, 4, 5):
        Gk = _gram_coeff_mats(_PATTERNS[vt])
        bases = _member_bases(_symbolic_closure(Gk))
        auts = _generic_automorphisms(Gk).astype(float)
        self_escapes = 0
        cross_max = 0
        for A in bases:
            Af = np.array(A, dtype=float)
            Ainv = np.linalg.inv(Af)
            conj = np.einsum("ij,njk,kl->nil", Ainv, auts, Af)
            if int(np.abs(np.round(conj)).max()) > 1:
                self_escapes += 1
            for B in bases:
                Binv = np.linalg.inv(np.array(B, dtype=float))
                coset = np.einsum("ij,njk,kl->nil", Binv, auts, Af)
                cross_max = max(cross_max, int(np.abs(np.round(coset)).max()))
        stats[vt] = (self_escapes, len(bases), cross_max)
    assert stats[1] == (0, 4, 1)
    assert stats[2] == (0, 8, 1)
    assert stats[3] == (0, 12, 1)      # V3 never escapes: the old claim's type
    assert stats[4] == (18, 24, 2)
    assert stats[5] == (60, 64, 3)


def test_iii_cuboid_coset_entry3_and_closure_route_returns_it():
    """Numeric instance: cuboid 3x4x5. Two reduced settings from different
    arrangement classes have an 8-operator coset with entries up to 3; the
    {-1,0,1} list finds only a strict subset; the closure route returns all."""
    import numpy as np
    from agentsg.cell.canonical import reindexing_via_canonical
    from agentsg.cell.g6 import _unimodular_pm1
    from agentsg.cell.metric import params_from_metric

    G = np.diag([9.0, 16.0, 25.0])
    A = np.array([[-1, 0, 0], [-1, 1, 0], [-1, 0, 1]], dtype=float)
    B = np.array([[0, 0, 1], [-1, 1, 0], [-1, 0, 1]], dtype=float)
    GA, GB = A.T @ G @ A, B.T @ G @ B
    auts = [np.diag(d).astype(float)
            for d in itertools.product((1, -1), repeat=3)]
    coset = [np.linalg.inv(B) @ Q @ A for Q in auts]
    ents = sorted(int(np.abs(np.round(c)).max()) for c in coset)
    assert ents == [1, 1, 1, 1, 2, 2, 3, 3]
    for c in coset:
        assert np.allclose(c.T @ GB @ c, GA, atol=1e-9)

    # the {-1,0,1} scan recovers exactly the four small coset members
    n_small = 0
    for M in _unimodular_pm1():
        Mf = np.array(M, dtype=float)
        if np.allclose(Mf.T @ GB @ Mf, GA, atol=1e-6):
            n_small += 1
    assert n_small == 4

    # the closure route returns the full coset, big entries included
    # (the direction whose congruence is P^T G_B P = G_A, matching the scan)
    cellA = tuple(params_from_metric(GA.tolist()))
    cellB = tuple(params_from_metric(GB.tolist()))
    ops = reindexing_via_canonical(cellB, cellA, boundary_rel=0.0,
                                   verify_rel=1e-6)
    assert len(ops) == 8
    assert sorted(_max_abs_entry(P) for P in ops) == [1, 1, 1, 1, 2, 2, 3, 3]
    # the opposite direction escapes the radius-one ball as well
    ops_rev = reindexing_via_canonical(cellA, cellB, boundary_rel=0.0,
                                       verify_rel=1e-6)
    assert len(ops_rev) == 8
    assert max(_max_abs_entry(P) for P in ops_rev) == 2

    # self-coset of a single reduced V4 cell already needs an entry of 2
    Gk4 = _gram_coeff_mats(_PATTERNS[4])
    G4 = np.zeros((3, 3))
    for w, Gm in zip((2.0, 3.0, 4.0, 5.0), Gk4):
        G4 += w * np.array(Gm, dtype=float)
    A4 = np.array([[-1, 0, 0], [-1, 1, 0], [-1, 0, 1]], dtype=float)
    GA4 = A4.T @ G4 @ A4
    c4 = tuple(params_from_metric(GA4.tolist()))
    ops4 = reindexing_via_canonical(c4, c4, boundary_rel=0.0, verify_rel=1e-6)
    assert len(ops4) == 4
    assert max(_max_abs_entry(P) for P in ops4) == 2


def test_iii_retracted_v3_construction_was_not_reduced():
    """Regression for a retracted claim: the V3 'entry-2 pair' obtained by
    pairing closure members under arbitrary signed permutations is NOT a
    pair of Selling-reduced settings -- the constructed cell's superbase
    completion has a negative conorm, so it belongs to the unreduced
    category (i), not to the reduced-settings statement."""
    import numpy as np

    c3 = cell_from_conorms(TYPE_CONORMS[3]())  # conorms 0,2,3,4,5,0
    w3, P3 = _max_entry_closure_match(c3)
    assert w3 >= 2 and P3 is not None  # the construction does reach entry 2
    G = np.array(metric_tensor(c3))
    C = np.array(P3, dtype=float)  # the constructed cell's basis, base coords
    v = [C[:, 0], C[:, 1], C[:, 2]]
    v0 = -(v[0] + v[1] + v[2])
    ws = [v0] + v
    conorms = [-float(ws[i] @ G @ ws[j])
               for i in range(4) for j in range(i + 1, 4)]
    assert min(conorms) < -1e-9, "completion is obtuse; claim would stand"

    # between GENUINE reduced settings of this V3 lattice all relating
    # operators (coset included) stay inside {-1,0,1}: see the stratum test


def test_truncation_fails_between_different_anisotropic_lattices():
    """Between DIFFERENT lattices the certificate governs: on a reduced
    needle pair with R >> 2 the radius-2 ball strictly improves the
    radius-1 minimum, so the 6960 truncation returns a wrong ideal there."""
    from agentsg.cell.deformation import (_ball_min, _metric, _radius,
                                          _unimodular_ball)

    # Selling-reduced cells (all angles >= 90); same search as the SI recipe
    A = (7.47, 47.69, 7.15, 91.69, 90.81, 98.53)
    B = (7.27, 42.89, 7.01, 91.26, 90.24, 92.72)
    GA, GB = _metric(A), _metric(B)
    v1, _ = _ball_min(GA, GB, _unimodular_ball(1))
    v2, M2 = _ball_min(GA, GB, _unimodular_ball(2))
    assert v2 < 0.5 * v1              # observed ratio ~2.8
    assert _max_abs_entry(M2.tolist()) == 2
    assert _radius(GA, GB, v2) > 2.0  # the certificate said so in advance


def test_ii_niggli_adversarial_entries_stay_pm1():
    """Adversarial Niggli check: anisotropic and near-degenerate lattices,
    exact resettings with entries up to 2; the direct relating operator
    P = (M2 N2)^{-1} (M1 N1) between the two Niggli cells is integer,
    satisfies the metric identity, and keeps entries in {-1,0,1}."""
    from agentsg.cell.reduction import niggli_reduce

    rng = random.Random(SEED_BRUTE + 7)
    refs = [
        ("needle-hP", (41.8, 41.8, 233.0, 90.0, 90.0, 120.0)),
        ("plate", (80.0, 80.0, 3.4, 90.0, 90.0, 120.0)),
        ("near-tie", (10.0, 10.001, 10.002, 89.99, 90.01, 90.0)),
        ("near-cube", (10.0, 10.0, 10.0, 89.9, 90.1, 90.05)),
        ("cuboid", (3.0, 4.0, 5.0, 90.0, 90.0, 90.0)),
        ("skew-needle", (7.47, 47.69, 7.15, 91.69, 90.81, 98.53)),
    ]
    worst = 0
    n_done = 0
    for label, cell in refs:
        G = metric_tensor(cell)
        for _ in range(25):
            M1 = sample_unimodular(rng, 2, 1)[0]
            M2 = sample_unimodular(rng, 2, 1)[0]
            c1 = cell_from_metric(transform_metric(G, M1))
            c2 = cell_from_metric(transform_metric(G, M2))
            try:
                r1, N1 = niggli_reduce(*c1)
                r2, N2 = niggli_reduce(*c2)
            except RuntimeError:
                continue
            T1 = matmul_int(M1, N1)
            T2 = matmul_int(M2, N2)
            T2inv = [[int(round(x)) for x in row] for row in inv3(T2)]
            P = matmul_int(T2inv, T1)
            G1 = transform_metric(G, T1)
            G2 = transform_metric(G, T2)
            Gp = transform_metric(G2, P)
            err = max(abs(Gp[i][j] - G1[i][j])
                      for i in range(3) for j in range(3))
            assert err < 1e-6, (label, err)
            worst = max(worst, _max_abs_entry(P))
            n_done += 1
    assert n_done >= 120
    assert worst <= 1, f"adversarial Niggli relating op max|entry|={worst}"


def test_ii_niggli_cells_relating_op_in_pm1():
    """(ii) Between Niggli cells of one lattice, relating op has |entries|<=1."""
    from agentsg.cell.reduction import niggli_reduce
    from agentsg.cell.primitive import primitive_cell

    rng = random.Random(SEED_BRUTE + 3)
    refs = [
        ("oP", (3.0, 4.0, 5.0, 90.0, 90.0, 90.0), None),
        ("cP", (5.0, 5.0, 5.0, 90.0, 90.0, 90.0), None),
        ("hP", (3.0, 3.0, 5.0, 90.0, 90.0, 120.0), None),
        ("mP", (120.0, 189.1, 120.3, 90.0, 91.2, 90.0), None),
        ("tric", (40.0, 50.0, 60.0, 80.0, 95.0, 110.0), None),
        ("oI", (3.0, 4.0, 5.0, 90.0, 90.0, 90.0), "I222"),
        ("cF", (5.0, 5.0, 5.0, 90.0, 90.0, 90.0), "F23"),
    ]
    worst = 0
    for label, cell, sym in refs:
        R = primitive_cell(cell, sym) if sym else cell
        G0 = metric_tensor(R)
        for _ in range(30):  # 30 x 7 = 210; manuscript 150 per type is slow
            M1 = sample_unimodular(rng, 2, 1)[0]
            M2 = sample_unimodular(rng, 2, 1)[0]
            # noise before resetting
            noisy = perturb_angles(
                (
                    R[0] * (1 + rng.gauss(0, 0.02 / R[0])),
                    R[1] * (1 + rng.gauss(0, 0.02 / R[1])),
                    R[2] * (1 + rng.gauss(0, 0.02 / R[2])),
                    R[3], R[4], R[5],
                ),
                0.05, rng,
            )
            Gn = metric_tensor(noisy)
            c1 = cell_from_metric(transform_metric(Gn, M1))
            c2 = cell_from_metric(transform_metric(Gn, M2))
            r1, N1 = niggli_reduce(*c1)
            r2, N2 = niggli_reduce(*c2)
            # P relating r1 -> r2 in reduced bases: approx via reindexing
            from agentsg.cell.canonical import reindexing_via_canonical
            ops = reindexing_via_canonical(r1, r2, boundary_rel=0, verify_rel=1e-3)
            if not ops:
                continue
            for P in ops:
                worst = max(worst, _max_abs_entry(P))
    assert worst <= 1, f"Niggli-cell relating op had max|entry|={worst}"
