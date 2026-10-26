"""z66 — statements of the paper that no other claim checks, and the data of the figures.

Each claim recomputes a number or a fact printed in the main text or the supporting
information from the definitions (U, the flips, the strain, the price); the printed values
are written here as they appear in the paper.
"""
from __future__ import annotations

import ast
import itertools
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from agentsg.cell import strain_closure as sc
from agentsg.cell.metric import UnitCell
from sellingproofs import paths
from sellingproofs import pin
from sellingproofs.registry import claim

from .test_z44_widened_complete import S0 as S0_LIST, _cn, _generate
from .test_z46_selling_groupoid import _orbits_and_shapes
from .test_z61_appendix_statements import _flip

S0 = np.array(S0_LIST, float)
SL = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
WORKED = (30.02, 39.97, 50.03, 90.25, 92.05, 89.80)
FIGSRC = paths.PAPER / "figures" / "src"

# Table 1 of the main text: flip at, new basis, matrix (columns are the new basis vectors)
FLIPS = {
    "ab": [[-1, 0, 1], [0, 1, 0], [0, 0, 1]],
    "ac": [[-1, 1, 0], [0, 1, 0], [0, 0, 1]],
    "bc": [[1, 0, 0], [1, -1, 0], [0, 0, 1]],
    "da": [[1, -1, -1], [0, 0, -1], [0, -1, 0]],
    "db": [[0, 0, -1], [-1, 1, -1], [-1, 0, 0]],
    "dc": [[0, -1, 0], [-1, 0, 0], [-1, -1, 1]],
}
IDX = {"d": 0, "a": 1, "b": 2, "c": 3}


def _G(cell, sg=None):
    return np.array(UnitCell(*cell).metric_tensor(), float)


def _strain(A, B):
    w, V = np.linalg.eigh(A)
    R = V @ np.diag(w ** -0.5) @ V.T
    return float(np.linalg.norm(R @ (B - A) @ R))


def _rows(W):
    """superbase rows (d, a, b, c) of a column operator W."""
    W = np.asarray(W, float)
    return np.vstack([-W.sum(1), W.T])


def _word(w):
    M = np.eye(3)
    for f in w.split("."):
        M = M @ np.array(FLIPS[f], float)
    return M


@pytest.fixture(scope="module")
def U():
    return sc.universal_operators()


@pytest.fixture(scope="module")
def swaps(U):
    k0 = sc._key(S0)
    Q = np.array([W for W in U if sc._key(_rows(W)) == k0])
    assert len(Q) == 48
    return Q


# ---- U ------------------------------------------------------------------------------

@claim("z66", "main §3.3", "main Lemma A2", kind="exhaustive")
def test_every_operator_is_swap_times_small_matrix_times_swap(U, swaps):
    """Every element of U is a swap, times a matrix with entries in {-1, 0, 1}, times a swap:
    relabelling both cells removes every entry of 2 and 3."""
    Qs = swaps.astype(np.int64)
    best = np.full(len(U), 99)
    for Q1 in Qs:
        A = np.einsum("ij,njk->nik", Q1, U.astype(np.int64))
        B = np.einsum("nik,qkl->nqil", A, Qs)
        best = np.minimum(best, np.abs(B).max((2, 3)).min(1))
    assert best.max() == 1
    assert np.abs(U).max() == 3


@claim("z66", "main §4.2", kind="exhaustive")
def test_only_768_transposes_lie_in_U(U):
    """Only 768 of the 6096 transposes of the column operators lie in U."""
    keys = {tuple(W.ravel()) for W in U}
    assert len(keys) == 6096
    assert sum(tuple(W.T.ravel()) in keys for W in U) == 768


@claim("z66", "main §4.2", "main Table 6", "main Remark A8", kind="exhaustive")
def test_3048_operators_of_each_determinant(U):
    """U has 3048 operators of determinant +1 and 3048 of determinant -1, so a proper
    candidate needs the 3048 with det W = det B_R det B_N."""
    d = np.round(np.linalg.det(U.astype(float))).astype(int)
    assert (d == 1).sum() == 3048 and (d == -1).sum() == 3048


@claim("z66", "SI §S3, Why four", kind="exhaustive")
def test_why_four_flip_distances():
    """For a lattice with a, b, c mutually perpendicular the sixteen members lie at flip
    distances 0, 1, 2, 3, 4 from (a, b, c) in numbers 1, 3, 6, 3, 3; four orthogonal
    superbases and twelve with a face diagonal."""
    G = sc._gram([900.0, 1600.0, 2500.0, 0.0, 0.0, 0.0])
    dist = {sc._key(S0): 0}
    frontier = [S0]
    while frontier:
        nxt = []
        for V in frontier:
            for i, j in SL:
                for s, t in ((i, j), (j, i)):
                    if abs(V[s] @ G @ V[t]) < 1e-9:
                        F = _flip(V, s, t)
                        k = sc._key(F)
                        if k not in dist:
                            dist[k] = dist[sc._key(V)] + 1
                            nxt.append(F)
        frontier = nxt
    counts = [list(dist.values()).count(n) for n in range(max(dist.values()) + 1)]
    assert counts == [1, 3, 6, 3, 3]
    orth = [k for k in dist if sum(abs(np.array(v)).sum() == 1 for v in k) >= 3]
    assert len(orth) == 4


# ---- Table 1 and the flip ---------------------------------------------------------------

@claim("z66", "main §2.2", "main Table 1", kind="exhaustive")
def test_table_1_flip_matrices():
    """Table 1: each flip matrix has as columns the new basis listed, determinant -1 and
    squares to the identity; it is the flip of the superbase (a, b, c, d) at that pair."""
    for name, M in FLIPS.items():
        M = np.array(M, float)
        assert round(np.linalg.det(M)) == -1 and np.array_equal(M @ M, np.eye(3)), name
        x, y = IDX[name[0]], IDX[name[1]]
        F = _flip(S0.copy(), x, y)                       # x is negated, added to the other two
        assert sc._key(F) == sc._key(_rows(M)), name
        assert np.array_equal(F[1:], M.T) or name[0] == "d", name
        if name[0] == "d":                               # d negated: the new a, b, c are listed
            assert np.array_equal(F[1:], M.T), name


@claim("z66", "main §2.2", kind="sampled")
def test_flip_conorm_update():
    """A flip at p_ab, (a, b, c, d) -> (-a, b, c+a, d+a), gives p'_ab = -p_ab, p'_ac = p_da + p_ab,
    p'_da = p_ac + p_ab, p'_bc = p_bc + p_ab, p'_db = p_db + p_ab, p'_dc = p_dc - p_ab."""
    rng = np.random.default_rng(66)
    for _ in range(500):
        A = rng.normal(size=(3, 3)); G = A @ A.T
        p = sc.conorms(S0, G)
        da, db, dc, ab, ac, bc = p
        a, b, c = np.eye(3); d = -a - b - c
        new = np.array([d + a, -a, b, c + a])          # rows d', a', b', c'
        q = sc.conorms(new, G)
        want = [ac + ab, db + ab, dc - ab, -ab, da + ab, bc + ab]   # da, db, dc, ab, ac, bc
        assert np.allclose(q, want)


# ---- tolerances and bands -----------------------------------------------------------------

@claim("z66", "main Table 5", kind="sampled")
def test_table_5_first_two_rows():
    """Table 5, first two rows, to first order: an angle between two vectors changed by delta
    with the edges fixed is a strain of at least sqrt2 sin delta (7.4 % at 3 deg), reached for an
    orthogonal pair; an edge changed by a fraction f with the angles fixed is a strain of at
    least 2f (10 % at 5 %), reached for an axis normal to the other two."""
    rng = np.random.default_rng(5)
    f, dl = 1e-4, np.radians(1e-2)
    for _ in range(400):
        L = rng.uniform(5, 100, 3); ang = rng.uniform(60, 120, 3)
        cell = (*L, *ang)
        try:
            G = _G(cell)
        except Exception:
            continue
        if np.linalg.eigvalsh(G)[0] <= 0:
            continue
        D = np.diag([1 + f, 1, 1])
        assert _strain(G, D @ G @ D) >= 2 * f * (1 - 1e-3)
        c2 = list(cell); c2[5] += np.degrees(dl)                    # gamma, between a and b
        assert _strain(G, _G(tuple(c2))) >= np.sqrt(2) * np.sin(dl) * (1 - 1e-3)
    G = _G((30, 40, 50, 90, 90, 90))
    assert abs(_strain(G, _G((30, 40, 50, 90, 90, 93))) - 0.074) < 0.0006
    D = np.diag([1.05, 1, 1])
    assert round(100 * _strain(G, D @ G @ D)) == 10
    assert abs(_strain(G, D @ G @ D) / 0.10 - 1) < 0.03


@claim("z66", "main §4.3", "main Table 6", kind="exhaustive")
def test_two_and_three_degree_bands():
    """A 2 deg lattice-symmetry search corresponds to a price band of 9.9 % and a 3 deg one to
    14.8 %; the reference is then widened at 10.4 % or 16.0 % strain."""
    p2 = 2 * np.sqrt(2) * np.arctanh(np.sin(np.radians(2)))
    p3 = 2 * np.sqrt(2) * np.arctanh(np.sin(np.radians(3)))
    assert round(100 * p2, 1) == 9.9 and round(100 * p3, 1) == 14.8
    assert round(100 * sc.band_for(0.099), 1) == 10.4 and round(100 * sc.band_for(0.148), 1) == 16.0
    assert round(100 * sc.band_for(p2), 1) == 10.4 and round(100 * sc.band_for(p3), 1) == 16.0


@claim("z66", "SI §S4", kind="exhaustive")
def test_lambda1_over_32_is_about_1_8_degrees():
    """For an isotropic cell the hypothesis lambda1 > 32 tau of the band theorem allows an
    angular error of about 1.8 deg: a conorm L^2 sin(delta) below L^2/32."""
    assert round(np.degrees(np.arcsin(1 / 32)), 1) == 1.8


# ---- the worked cell ----------------------------------------------------------------------

@claim("z66", "main §5.2", kind="exhaustive")
def test_worked_cell_error_is_0_82_percent():
    """The measured worked cell 30.02, 39.97, 50.03 A, 90.25, 92.05, 89.80 deg is the cell
    30 x 40 x 50 A, beta = 92 deg, perturbed by a metric error of strain 0.82 %."""
    G0, G = _G((30, 40, 50, 90, 92, 90)), _G(WORKED)
    assert round(100 * _strain(G0, G), 2) == 0.82
    assert round(100 * _strain(G, G0), 2) == 0.82


@claim("z66", "main Proposition A10", "main Table 9", kind="exhaustive")
def test_worked_cell_self_maps():
    """The worked cell's proper self-maps cost 0, 1.55 % and 10.2 %, so the condition of
    Proposition A10(iii) fails at 7.4 %, yet H_eps = {I, 2_y} is a group."""
    W = sc.widened_closure(WORKED, eps=0.2)
    maps = sc._self_maps(W, sc.accept_for(0.2), True)
    prices = sorted(round(100 * c.price, 2) for c in maps)
    assert prices[0] == 0.0 and prices[1] == 1.55 and all(round(p, 1) == 10.2 for p in prices[2:])
    assert len(prices) == 4
    H = [np.array([[float(x) for x in r] for r in c.operator.rows]) for c in maps if c.price <= 0.074]
    got = {tuple(np.round(h).astype(int).ravel()) for h in H}
    assert got == {(1, 0, 0, 0, 1, 0, 0, 0, 1), (-1, 0, 0, 0, 1, 0, 0, 0, -1)}
    assert any(0.074 < c.price <= 0.148 for c in maps)                   # the gap condition fails


# SI Table S6: word, conorms da db dc ab ac bc, depth (%)
TABLE_S6 = [
    ("e", (49.5, 1588.9, 2444.7, 4.2, 847.5, 4.5), 0),
    ("bc", (45.0, 8.7, 2449.3, 1593.4, 852.0, -4.5), 0.28),
    ("ab", (851.7, 1593.1, 2440.6, -4.2, 53.7, 8.7), 0.49),
    ("bc.db", (2458.0, -8.7, 53.7, 1602.1, 843.3, 4.2), 0.62),
    ("ab.bc", (842.9, 4.5, 2449.3, 1601.8, 62.5, -8.7), 0.78),
    ("ab.bc.db", (2453.8, -4.5, 847.5, 1606.3, 57.9, -4.2), 0.78),
    ("da", (-49.5, 2494.3, 1638.4, 53.7, 897.0, -45.0), 3.76),
    ("ab.bc.ac", (1664.2, -57.9, 2511.7, 905.4, -62.5, 53.7), 4.56),
    ("da.ab", (950.7, 2548.0, 1584.7, -53.7, 4.2, 8.7), 5.06),
    ("da.ab.ac", (-49.5, 2543.8, 1588.9, 954.9, -4.2, 12.9), 5.08),
    ("ab.ac", (49.5, 1539.3, 2494.3, 905.4, -53.7, 62.5), 5.08),
    ("da.ab.bc", (942.0, -45.0, 1593.4, 2556.7, 12.9, -8.7), 5.09),
    ("bc.db.dc", (45.0, 2511.7, -53.7, 1548.4, 897.0, 57.9), 5.09),
    ("ab.ac.bc.ac", (1610.5, 959.1, 2565.5, -4.2, -8.7, -53.7), 5.11),
    ("da.ab.bc.ac", (2569.6, -57.9, 1606.3, 954.9, -12.9, 4.2), 5.11),
    ("ab.ac.bc", (-12.9, 967.8, 2556.7, 1601.8, 8.7, -62.5), 5.11),
]


def _worked_frame():
    from agentsg.cell.canonical import canonical_superbase
    G = _G(WORKED)
    F0 = np.array(canonical_superbase(WORKED)[0], float)
    return F0[1:4] @ G @ F0[1:4].T


@claim("z66", "SI Table S6", "main §5.2", kind="exhaustive")
def test_table_s6_words_conorms_and_depths():
    """SI Table S6: each of the sixteen members, reached from the reduced cell by its flip
    word (Table 1, read left to right), has the printed conorms under the measured metric and
    enters at the printed depth; all sixteen are distinct and members at 7.7 %."""
    Gf = _worked_frame()
    keys = set()
    for w, cn, depth in TABLE_S6:
        M = np.eye(3) if w == "e" else _word(w)
        S = _rows(M)
        q = sc.conorms(S, Gf)
        assert np.allclose(q, cn, atol=0.051), (w, np.round(q, 1))
        d = sc.distance_to_obtuse(Gf, S)
        assert abs(100 * d - depth) < 0.0051, (w, 100 * d)
        assert d <= sc.band_for(0.074)
        keys.add(sc._key(S))
    assert len(keys) == 16


@claim("z66", "SI §S4", kind="exhaustive")
def test_worked_cell_at_tau_20():
    """For the worked cell at tau = 20 A^2, within the hypothesis lambda1/32 = 28.2 A^2, the
    absolute band returns the six members of the ring."""
    Gf = _worked_frame()
    p0 = _cn(S0, Gf)
    lam1 = min(float(s @ Gf @ s) for s in S0)
    assert round(lam1 / 32, 1) == 28.2
    _, frames = _generate(p0, 4 * 20.0, 20.0)                 # the closure of the idealised lattice
    kept = [S for S in frames.values() if _cn(S, Gf).min() >= -20.0]
    assert len(frames) == 16 and len(kept) == 6
    assert all(sc.distance_to_obtuse(Gf, S) < 0.008 for S in kept)  # the ring, below 0.8 %


# ---- Discussion arithmetic ---------------------------------------------------------------

@claim("z66", "main §6, Discussion", kind="exhaustive")
def test_discussion_arithmetic():
    """At about 5 % of price per degree of obliquity, the 90th percentile of non-isomorphism
    (4.2 %) is about 0.85 deg; Le Page's 3 deg prices a twofold at two and a half to six times
    the 90th percentile of the Bravais types that have twofolds (Table 10); 8.6 % of entries is
    about one in twelve."""
    per_deg = 2 * np.sqrt(2) * np.arctanh(np.sin(np.radians(1)))
    assert round(100 * per_deg, 1) == 4.9
    assert round(0.042 / per_deg, 2) == 0.85
    q90 = {"mP": 5.38, "mC": 5.06, "oP": 4.70, "oS": 3.57, "oI": 5.12, "oF": 5.87,
           "tP": 2.59, "tI": 2.76, "hR": 3.54, "hP": 2.87}               # Table 10, types with twofolds
    p3 = 100 * sc.LEPAGE_PRICE
    lo, hi = p3 / max(q90.values()), p3 / min(q90.values())
    assert 2.4 < lo < 2.6 and 5.5 < hi < 6.0, (lo, hi)
    assert round(1 / 0.086) == 12


# ---- implementation constants -------------------------------------------------------------

@claim("z66", "main Table 7", "main §4.1", kind="exhaustive")
def test_membership_margin_is_1e_9():
    """Membership in the widened set is decided by a floating-point depth with a relative
    margin of 1e-9."""
    import inspect
    src = inspect.getsource(sc.widened_closure) + inspect.getsource(sc._ball_superbases)
    assert "eps * (1 + 1e-9)" in src
    G = sc.reduce_cell(WORKED).G
    W = sc.widened_closure(WORKED, eps=0.005)
    assert all(m.depth <= 0.005 * (1 + 1e-9) for m in W.members)


@claim("z66", "main §4", "main Data availability", "SI §S7", kind="exhaustive")
def test_pinned_library_and_tables():
    """The checks run against agentsg 0.5.0, the pinned data files have the recorded SHA-256
    hashes, and the library's U equals the deposited column table element for element."""
    import json
    assert pin.agentsg_version() == pin.AGENTSG_VERSION
    for rel, want in pin.SHA256.items():
        assert pin.sha256(rel) == want, rel
    dep = np.array(json.load(open(paths.U_COLUMNS))["W"])
    assert np.array_equal(dep, sc.universal_operators())


# ---- partially checked statements ---------------------------------------------------------

@claim("z66", "main §5.2, Needles and plates", kind="exhaustive", needs=("slow",))
def test_needle_from_0_05_to_5_2_percent():
    """On the needle 5.58 x 5.21 x 131.97 A, gamma = 91.12 deg, the exact enumeration and
    brute force agree at strains from 0.05 to 5.2 %."""
    from .test_z58_projection_closure import strain_set
    cell = (5.583, 5.21, 131.965, 90.0, 90.0, 91.118)
    G = sc.reduce_cell(cell).G
    for eps in (0.0005, 0.002, 0.005, 0.01, 0.02, 0.035, 0.052):
        W = sc.widened_closure(cell, eps=eps)
        got = {sc._key(m.superbase) for m in W.members}
        ref, _ = strain_set(G, eps, 3)
        assert got == ref, (eps, len(got), len(ref))


SAMPLES_S5 = [  # name, BLDConograph distance (A^2), printed A^2, printed %
    ("HERMES_Sr327_250K", 0.5672, 0.57, 1.24),
    ("1339prbasrcuo_bs1", 0.05641, 0.056, 0.10),
    ("sample3", 2.0738, 2.07, 0.78),
    ("CeO2_matsushita", 0.084427, 0.084, 0.61),
    ("M402-pf-119673", 20.549, 20.5, 5.28),
    ("Acrinol-B-D8CuKa1", 0.71525, 0.72, 0.35),
    ("Alumina676", 1.9315, 1.93, 8.11),
]


@claim("z66", "SI Table S5", kind="exhaustive")
def test_table_s5_distance_columns():
    """SI Table S5: BLDConograph's minimal distance to the program's type (A^2) and the same
    divided by the mean squared edge (%)."""
    from .test_z51_conograph_samples import SAMPLES
    cells = {s[0]: s[1] for s in SAMPLES}
    for name, dist, a2, pct in SAMPLES_S5:
        decimals = len(repr(a2).split(".")[1])
        assert round(dist, decimals) == a2, name
        L = np.array(cells[name][:3])
        assert round(100 * dist / np.mean(L ** 2), 2) == pct, (name, 100 * dist / np.mean(L ** 2))


@claim("z66", "SI §S4, From a metric error to a band", kind="exhaustive")
def test_cuboid_frames_with_constant_2_sqrt3():
    """The largest constant c_ij = ||sym(u_i u_j^T)||_F on a zero conorm is 2 sqrt3 in six of
    the sixteen labelled cuboid frames (three zero conorms: four triangles and twelve paths),
    those whose path of zeros has d in its interior, and at most sqrt5 in the other ten."""
    from agentsg.cell.canonical import _closure_for_match
    from ._helpers import cell_from_conorms
    rnd = np.random.default_rng(66)
    kaps = []
    for Z in itertools.combinations(range(6), 3):
        free = [SL[k] for k in range(6) if k not in Z]
        par = list(range(4))

        def f(x):
            while par[x] != x:
                x = par[x]
            return x
        for i, j in free:
            par[f(i)] = f(j)
        if len({f(i) for i in range(4)}) > 1:
            continue
        p = rnd.uniform(900, 2500, 6); p[list(Z)] = 0
        cell = tuple(cell_from_conorms(list(p)))
        G = _G(cell)
        k = 0.0
        for M in _closure_for_match(cell, boundary_rel=0.0):
            M = np.array(M, float)
            for i, j in SL:
                if abs(-M[i] @ G @ M[j]) < 1e-9 * np.abs(G).max():
                    u, v = M[i], M[j]
                    k = max(k, ((u @ u * (v @ v) + (u @ v) ** 2) / 2) ** 0.5)
        deg0 = sum(0 in SL[z] for z in Z)
        kaps.append((k, deg0))
    assert len(kaps) == 16
    big = [d for k, d in kaps if abs(k - 12 ** 0.5) < 1e-9]
    assert len(big) == 6 and all(d == 2 for d in big)
    assert all(k <= 5 ** 0.5 + 1e-9 for k, _ in kaps if abs(k - 12 ** 0.5) >= 1e-9)


@claim("z66", "SI §S4, Bands per slot", kind="exhaustive")
def test_slot_band_about_25_times_larger():
    """On the needle, the members built on a+c carry the dot product of a and b on a slot whose
    band sigma |a+c| |b| is about 25 times that of the slot ab, sigma |a| |b|."""
    G = _G((5.583, 5.21, 131.965, 90.0, 90.0, 91.118))
    a, c = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    ratio = np.sqrt((a + c) @ G @ (a + c)) / np.sqrt(a @ G @ a)
    assert 20 < ratio < 30, ratio


# SI Table S1: lattice -> [(n, printed cell)]
TABLE_S1 = {
    "cP": (((30, 30, 30, 90, 90, 90), None), [(4, (30, 30, 30, 90, 90, 90)), (12, (30, 30, 42.4, 90, 135, 90))]),
    "tP": (((30, 30, 50, 90, 90, 90), None), [(4, (30, 30, 50, 90, 90, 90)), (4, (30, 30, 58.3, 90, 121.0, 90)),
                                              (8, (30, 50, 42.4, 90, 135, 90))]),
    "oP": (((30, 40, 55, 90, 90, 90), None), [(4, (30, 40, 55, 90, 90, 90)), (4, (30, 40, 62.6, 90, 118.6, 90)),
                                              (4, (30, 55, 50, 90, 126.9, 90)), (4, (40, 55, 50, 90, 143.1, 90))]),
    "oS": (((30, 40, 50, 90, 90, 90), "C 2 2 2"), [(2, (25, 50, 25, 90, 106.3, 90)), (4, (25, 50, 30, 90, 126.9, 90))]),
    "mP": (((30, 40, 50, 90, 100, 90), None), [(2, (30, 40, 50, 90, 100, 90)), (2, (30, 40, 53.7, 90, 113.4, 90)),
                                               (2, (50, 40, 53.7, 90, 146.6, 90))]),
    "O3": (([900, 1600, 0, 0, 1600, 900], "conorms"), [(1, (50, 50, 50, 111.1, 129.8, 90)),
                                                         (2, (42.4, 50, 50, 90, 115.1, 115.1))]),
    "M2B": (([900, 1600, 2500, 0, 1600, 900], "conorms"), [(1, (50, 50, 70.7, 104.7, 116.9, 90)),
                                                             (1, (50, 50, 65.6, 105.9, 105.9, 90))]),
    "M3": (([2500, 1600, 0, 0, 1600, 900], "conorms"), [(1, (50, 50, 64.0, 120, 90, 111.1)),
                                                          (2, (50, 56.6, 58.3, 90, 108.0, 124.4))]),
    "A2": (([2500, 400, 1200, 0, 1600, 900], "conorms"), [(1, (36.1, 60.8, 64.0, 114.3, 90, 114.2)),
                                                            (1, (36.1, 56.6, 64.0, 116.2, 90, 101.3))]),
    "A3": (([2500, 400, 0, 0, 1600, 900], "conorms"), [(1, (36.1, 50, 53.9, 90, 101.9, 119.9)),
                                                         (1, (44.7, 50, 53.9, 90, 99.6, 135.7)),
                                                         (1, (36.1, 44.7, 58.3, 90, 115.3, 104.4))]),
}
_PERMS = [[{e: n for n, e in enumerate(SL)}[tuple(sorted((g[a], g[b])))] for a, b in SL]
          for g in itertools.permutations(range(4))]


def _shapes_of(spec):
    src, kind = spec
    if kind == "conorms":
        p = np.array(src, float)
    else:
        r = sc.reduce_cell(src, kind)
        p = sc.conorms(S0, r.G)
        p[np.abs(p) < 1e-6 * np.abs(p).max()] = 0.0
    G = sc._gram(p)
    mem, _, shape_of, _ = _orbits_and_shapes(p, G)
    count = {}
    for k in mem:
        count[shape_of[k]] = count.get(shape_of[k], 0) + 1
    return count


@claim("z66", "SI Table S1", "main §3.2", kind="exhaustive")
def test_table_s1_cell_shapes():
    """SI Table S1: for each lattice, the printed cells are obtuse superbases, one of each of
    its cell shapes, with the printed number n of members of that shape; the third shape of
    the 100 deg monoclinic cell is 50, 40, 53.7 A, beta = 146.6 deg."""
    for name, (spec, rows) in TABLE_S1.items():
        shapes = _shapes_of(spec)
        matched = set()
        for n, cell in rows:
            assert all(x >= 90 - 0.05 for x in cell[3:]), (name, cell)
            q = sc.conorms(S0, _G(cell))
            tol = 0.012 * np.abs(q).max()
            assert q.min() >= -tol, (name, cell, q)
            hits = [s for s in shapes if min(np.abs(q[pm] - np.array(s)).max() for pm in _PERMS) < tol]
            assert len(hits) == 1, (name, cell, len(hits))
            assert shapes[hits[0]] == n, (name, cell, shapes[hits[0]], n)
            matched.add(hits[0])
        assert len(matched) == len(shapes), f"{name}: the table lists {len(rows)} of {len(shapes)} shapes"


# ---- figures -----------------------------------------------------------------------------

def _literals(path, names):
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tg in node.targets:
                if isinstance(tg, ast.Name) and tg.id in names:
                    out[tg.id] = ast.literal_eval(node.value)
                elif isinstance(tg, ast.Tuple):
                    for el, v in zip(tg.elts, getattr(node.value, "elts", [])):
                        if isinstance(el, ast.Name) and el.id in names:
                            try:
                                out[el.id] = ast.literal_eval(v)
                            except ValueError:
                                pass
    return out


@claim("z66", "main Fig. 1", "main §3.2", kind="exhaustive")
def test_fig_1_ring():
    """Fig. 1: the six superbases of the 100 deg cell on the ring M0-M1-M5-M4-M3-M2, coloured by
    the three shapes as labelled; the word ab.bc.db goes from M0 to M4 by flips at zero conorms,
    and the twofold and the mirror are that word followed by a swap."""
    lit = _literals(FIGSRC / "fig_ring.py", {"SHAPE", "LAB", "ring", "path"})
    p = np.array([639.5277335, 1600.0, 2239.5277335, 0.0, 260.4722665, 0.0])
    G = sc._gram(p)
    mem, _, shape_of, _ = _orbits_and_shapes(p, G)
    assert len(mem) == 6
    # walk the word, each flip at a zero conorm of the current superbase
    V, walk = S0.copy(), [S0.copy()]
    for f in ("ab", "bc", "db"):
        i, j = IDX[f[0]], IDX[f[1]]
        assert abs(V[i] @ G @ V[j]) < 1e-9 * p.max(), f
        V = _flip(V, i, j)
        walk.append(V)
    assert np.allclose(_rows(_word("ab.bc.db")), walk[-1])
    names = ["M0", "M1", "M5", "M4"]
    ks = [sc._key(S) for S in walk]
    assert len(set(ks)) == 4 and all(k in mem for k in ks)
    assert lit["ring"][:4] == names and lit["path"] == {("M0", "M1"): "1: ab", ("M1", "M5"): "2: bc",
                                                         ("M5", "M4"): "3: db"}
    # shapes: the same colour exactly for the same shape
    shp = {m: shape_of[k] for m, k in zip(names, ks)}
    for m1, m2 in itertools.combinations(names, 2):
        assert (shp[m1] == shp[m2]) == (lit["SHAPE"][m1] == lit["SHAPE"][m2]), (m1, m2)
    # the ring: M3 and M2 are the remaining members, neighbours of M4 and M0
    rest = [k for k in mem if k not in ks]
    assert len(rest) == 2
    # labels: each colour's cell is that shape
    for m, k in zip(names[:3], ks[:3]):
        lab = lit["LAB"][lit["SHAPE"][m]]
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", lab)]
        if len(nums) == 4:
            a, b, c, beta = nums
        else:
            raise AssertionError(lab)
        q = sc.conorms(S0, _G((a, b, c, 90, beta, 90)))
        assert min(np.abs(q[pm] - np.array(shape_of[k])).max() for pm in _PERMS) < 0.012 * p.max(), (m, lab)
    # twofold and mirror = F_ab F_bc F_db times a swap
    F = _word("ab.bc.db")
    U = sc.universal_operators().astype(float)
    aut = [W for W in U if np.allclose(W.T @ G @ W, G, atol=1e-7 * p.max())]
    two = [W for W in aut if round(np.linalg.det(W)) == 1 and not np.allclose(W, np.eye(3))]
    mir = [W for W in aut if round(np.linalg.det(W)) == -1 and not np.allclose(W, -np.eye(3))]
    assert len(two) == 1 and len(mir) == 1
    k0 = sc._key(S0)
    for g in (two[0], mir[0]):
        Q = np.linalg.solve(F, g)
        assert np.allclose(Q, np.round(Q)) and sc._key(_rows(np.round(Q))) == k0


@claim("z66", "main Fig. 2", kind="exhaustive")
def test_fig_2_depths():
    """Fig. 2: the sixteen members by depth as drawn are those of SI Table S6; the six of the
    ring enter below 0.8 % strain, the other ten between 3.8 and 5.1 %."""
    rows = _literals(FIGSRC / "fig_depths.py", {"ROWS"})["ROWS"]
    assert [(w, d) for w, d, _ in rows] == [(w, d) for w, _, d in TABLE_S6]
    Gf = _worked_frame()
    for w, d, colour in rows:
        S = _rows(np.eye(3) if w == "e" else _word(w))
        dd = 100 * sc.distance_to_obtuse(Gf, S)
        assert abs(dd - d) < 0.0051, (w, dd)
        assert (colour == 0) == (dd < 0.8)
    deep = [d for _, d, k in rows if k]
    assert len(deep) == 10 and round(min(deep), 1) >= 3.7 and round(max(deep), 1) <= 5.1


@claim("z66", "main Fig. 3", "main §5.4", kind="pdb", needs=("pdb", "slow"))
def test_fig_3_data():
    """Fig. 3: the 90th and 95th percentiles of non-isomorphism drawn as vertical lines are 4.2
    and 6.7 %, and the orange curve of the cheapest pseudo-symmetric alternative ends near
    14.8 %, the price of a twofold at Le Page's 3 deg."""
    import importlib.util
    sys.path.insert(0, str(paths.REPO / "agentsg" / "src"))
    spec = importlib.util.spec_from_file_location("fig_pdb_prices", FIGSRC / "fig_pdb_prices.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ni, ps = mod.nonisomorphism(), mod.pseudo()
    q90, q95 = np.percentile(ni, [90, 95])
    assert round(100 * q90, 1) == 4.2 and round(100 * q95, 1) == 6.7
    assert ps.max() <= sc.LEPAGE_PRICE * (1 + 1e-3) and ps.max() > 0.14


def _steps(name):
    import json
    return json.load(open(FIGSRC / f"steps_{name}.json"))


@claim("z66", "SI Fig. S1", "SI Fig. S2", kind="exhaustive")
def test_figs_s1_s2_data():
    """SI Figs S1 and S2: the drawn sort distances are the sort spectra of the measured worked
    cell (A2 0.28 %, M4 0.78 %, A3 3.8 %, O5 5.1 %) and of the lysozyme cell, whose twelve sorts
    from aP to tP lie within 0.22 % with every other sort at least 25 % away."""
    w = _steps("worked")["rows"]["measured"]
    for s, v in {"A2": 0.28, "M4": 0.78, "A3": 3.8, "O5": 5.1}.items():
        assert round(100 * w[s], 2 if v < 1 else 1) == v, (s, w[s])
    ly = _steps("lysozyme")["rows"]["measured"]
    near = [s for s, v in ly.items() if v <= 0.0022]
    assert len(near) == 12
    assert {sc.BRAVAIS.get(s, "aP") for s in near} == {"aP", "mP", "mC", "oP", "oS", "tP"}
    assert all(v >= 0.25 for s, v in ly.items() if s not in near)


@claim("z66", "SI Fig. S1", "SI Fig. S2", kind="reproduce", needs=("slow",))
def test_figs_s1_s2_data_regenerate(tmp_path):
    """The data files of SI Figs S1 and S2 (steps_worked.json, steps_lysozyme.json) are
    regenerated identically by sort_steps.py."""
    import json
    for name in ("worked", "lysozyme"):
        dep = _steps(name)
        out = tmp_path / f"steps_{name}.json"
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(paths.REPO / "agentsg" / "src"),
                                                          str(paths.SUPPLEMENT / "src")]))
        r = subprocess.run([sys.executable, str(FIGSRC / "sort_steps.py"), *map(str, dep["cell"]), str(out)],
                           capture_output=True, text=True, env=env, timeout=3600)
        assert r.returncode == 0, r.stderr[-2000:]
        new = json.load(open(out))
        assert new["order"] == dep["order"]
        for row in dep["rows"]:
            for k, v in dep["rows"][row].items():
                assert abs(new["rows"][row][k] - v) <= 1e-9 * max(1.0, abs(v)), (name, row, k)
