"""Shared fixtures and utilities for Appendix Z computational checks.

RNG policy: every stochastic helper takes an explicit ``rng`` (or uses a named
SEED_* constant). Never read global ``random`` / ``numpy.random`` state.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Sequence

# --- fixed seeds (named; do not rely on global RNG) ---
SEED_REDUCTION = 101
SEED_LOWERBOUND = 202
SEED_EUCLID = 303
SEED_FIBRE = 404
SEED_D7 = 20260828
SEED_NOISE = 11
SEED_VERIFY = 7
SEED_NOISY_FRAMES = 13
SEED_REINDEX_PROC = 17
SEED_BENCH = 2026
SEED_TIMING = 42
SEED_EMBED = 99

# --- manuscript fixtures ---
P63 = (41.8, 41.8, 233.0, 90.0, 90.0, 120.0)
CUBOID = (3.0, 4.0, 5.0, 90.0, 90.0, 90.0)
ORTHO = (10.0, 12.0, 20.0, 90.0, 90.0, 90.0)
TRICLINIC = (40.0, 50.0, 60.0, 85.0, 95.0, 100.0)
MONOCLINIC_P = (6.0, 8.0, 11.0, 90.0, 112.0, 90.0)
TETRAGONAL = (78.0, 78.0, 38.0, 90.0, 90.0, 90.0)
TRICLINIC_NOISE = (41.8, 55.2, 73.1, 81.0, 97.0, 112.0)

# Table C1 printed medians (Å) for P6_3 under angular noise
TABLE_C1 = {
    # stabilize mode -> sigma_deg -> printed median
    "sqrt": {0.01: 1.58, 0.05: 3.12, 0.10: 4.63},
    "floored": {0.01: 0.28, 0.05: 0.64, 0.10: 0.68},
    "soft_threshold": {0.01: 0.03, 0.05: 0.17, 0.10: 0.35},
    "linear": {0.01: 0.01, 0.05: 0.06, 0.10: 0.10},
}

# Kurlin Table: type -> (n_obtuse, n_classes, key_fibre)
CLOSURE_TABLE = {
    1: (2, 1, 30),   # with ±I: 2; class count 1
    2: (4, 2, 15),
    3: (6, 3, 1),
    4: (12, 3, 4),
    5: (32, 4, 1),
}

_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def assert_within_pct(got, expected, pct=10.0, label=""):
    """Pin a manuscript printed value with ±pct relative margin."""
    expected = float(expected)
    got = float(got)
    if expected == 0.0:
        tol = max(1e-12, abs(got) * pct / 100.0)
        assert abs(got) <= tol, f"{label}: got {got}, expected ~0 ±{tol}"
        return
    # For very small printed values, also allow a tiny absolute floor so
    # 0.01 ±10% is not ruined by 0.0015 Monte Carlo noise.
    rel = abs(expected) * pct / 100.0
    abs_floor = 0.005 if abs(expected) < 0.05 else 0.0
    tol = max(rel, abs_floor)
    lo, hi = expected - tol, expected + tol
    assert lo <= got <= hi, (
        f"{label}: got {got:.6g}, expected {expected:.6g} ±{pct}% "
        f"[{lo:.6g}, {hi:.6g}]"
    )


def key_l2(a, b):
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(len(a))))


def metric_tensor(cell):
    from agentsg.cell.metric import metric_tensor as _mt
    return _mt(cell)


def cell_from_metric(G):
    from agentsg.cell.metric import cell_from_metric
    return cell_from_metric(G)


def transform_metric(G, P):
    """G' = P^T G P for 3x3 lists / row-tuples."""
    P = [list(row) for row in P]
    PtG = [[sum(P[k][r] * G[k][b] for k in range(3)) for b in range(3)]
           for r in range(3)]
    return [[sum(PtG[r][k] * P[k][b] for k in range(3)) for b in range(3)]
            for r in range(3)]


def det3(P):
    P = [list(row) for row in P]
    return (P[0][0] * (P[1][1] * P[2][2] - P[1][2] * P[2][1])
            - P[0][1] * (P[1][0] * P[2][2] - P[1][2] * P[2][0])
            + P[0][2] * (P[1][0] * P[2][1] - P[1][1] * P[2][0]))


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def inv3(M):
    """Inverse of 3x3 float matrix."""
    d = det3(M)
    if abs(d) < 1e-15:
        raise ValueError("singular")
    a, b, c = M[0]
    d0, e, f = M[1]
    g, h, i = M[2]
    return [
        [(e * i - f * h) / d, (c * h - b * i) / d, (b * f - c * e) / d],
        [(f * g - d0 * i) / d, (a * i - c * g) / d, (c * d0 - a * f) / d],
        [(d0 * h - e * g) / d, (b * g - a * h) / d, (a * e - b * d0) / d],
    ]


def perturb_angles(cell, sigma_deg, rng):
    """Perturb the three angles by N(0, sigma_deg). Requires ``rng``."""
    a, b, c, al, be, ga = cell
    return (
        a, b, c,
        al + rng.gauss(0.0, sigma_deg),
        be + rng.gauss(0.0, sigma_deg),
        ga + rng.gauss(0.0, sigma_deg),
    )


def conorm_dict_from_six(vals):
    """Map a length-6 sequence onto ``_PAIRS`` order."""
    return {pair: float(vals[k]) for k, pair in enumerate(_PAIRS)}


def gram_from_conorms(p):
    """3x3 Gram of (v1,v2,v3) from conorms with v0=-(v1+v2+v3).

    v_i·v_j = -p_ij for i,j in 1..3;  |v_i|^2 = sum_{j≠i} p_ij.
    """
    def _p(i, j):
        return float(p[(i, j) if i < j else (j, i)])

    # indices 1,2,3 for the three basis vectors
    G = [[0.0] * 3 for _ in range(3)]
    for i in range(1, 4):
        others = [j for j in range(4) if j != i]
        G[i - 1][i - 1] = sum(_p(i, j) for j in others)
    for i in range(1, 4):
        for j in range(i + 1, 4):
            G[i - 1][j - 1] = G[j - 1][i - 1] = -_p(i, j)
    return G


def cell_from_conorms(vals_or_dict):
    """Build a cell from six conorms (dict or sequence in _PAIRS order)."""
    if isinstance(vals_or_dict, dict):
        p = vals_or_dict
    else:
        p = conorm_dict_from_six(vals_or_dict)
    return cell_from_metric(gram_from_conorms(p))


# Test conorm patterns for V1--V5 (manuscript Z.2 examples)
def v1_conorms():
    return (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def v2_conorms():
    # one zero: p01=0
    return (0.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def v3_conorms():
    # two opposite: p01 and p23 = 0
    return (0.0, 2.0, 3.0, 4.0, 5.0, 0.0)


def v4_conorms():
    # two sharing a vertex: p01=p02=0 (vertex 0 orthogonal to 1 and 2)
    return (0.0, 0.0, 3.0, 4.0, 5.0, 6.0)


def v5_conorms():
    # three zeros of cuboid form: p01=p02=p12=0 (orthogonal triad among 0,1,2
    # is illegal); cuboid has three zeros on edges of a matching.
    # Standard: p01=p02=p03=0 is singular. Cuboid: three pairwise orthogonal
    # among the four? Kurlin V5: three zero conorms forming a "triangle" on
    # three edges that meet? Table: 3 zeros. Odd cuboid: v1,v2,v3 orthogonal
    # => p12=p13=p23=0, and p01,p02,p03 > 0.
    return (1.0, 2.0, 3.0, 0.0, 0.0, 0.0)


TYPE_CONORMS = {
    1: v1_conorms,
    2: v2_conorms,
    3: v3_conorms,
    4: v4_conorms,
    5: v5_conorms,
}


def type_test_cell(vtype: int):
    return cell_from_conorms(TYPE_CONORMS[vtype]())


def s4_slot_permutations():
    """The 24 permutations of the six edge slots induced by S4 on vertices."""
    perms = []
    for sigma in itertools.permutations(range(4)):
        inv = {sigma[i]: i for i in range(4)}
        slot_perm = []
        for a, b in _PAIRS:
            ia, ib = inv[a], inv[b]
            if ia > ib:
                ia, ib = ib, ia
            slot_perm.append(_PAIRS.index((ia, ib)))
        perms.append(tuple(slot_perm))
    # unique as permutations of 0..5
    return list(dict.fromkeys(perms))


def apply_slot_perm(x, perm):
    """Apply slot permutation: out[i] = x[perm[i]]? We want σ·y with permuted coords.
    For orbit distance min_σ ||x - σ y||, (σ y)_i = y[σ^{-1}(i)] if σ acts on indices.
    Here ``perm`` maps new_slot -> old_slot index of the value that lands there
    after vertex relabelling (same as concat_key_fibres).
    """
    return tuple(x[perm[i]] for i in range(6))


def sample_unimodular(rng, entry_max=2, n=50):
    """Sample ``n`` distinct unimodular 3x3 integer matrices with |entries|<=max."""
    found = []
    seen = set()
    # deterministic scan with rng-shuffled candidates
    vals = list(range(-entry_max, entry_max + 1))
    attempts = 0
    while len(found) < n and attempts < 200000:
        attempts += 1
        M = tuple(tuple(rng.choice(vals) for _ in range(3)) for _ in range(3))
        if M in seen:
            continue
        if abs(det3(M)) != 1:
            continue
        seen.add(M)
        found.append(M)
    if len(found) < n:
        raise RuntimeError(f"only found {len(found)} unimodular matrices")
    return found


def brute_automorphisms(G, entry_max=2, tol=1e-8):
    """All integer P with |entries|<=entry_max, |det|=1, P^T G P == G."""
    vals = range(-entry_max, entry_max + 1)
    aut = []
    for rows in itertools.product(vals, repeat=9):
        P = (rows[0:3], rows[3:6], rows[6:9])
        if abs(det3(P)) != 1:
            continue
        Gp = transform_metric(G, P)
        resid = max(abs(Gp[a][b] - G[a][b]) for a in range(3) for b in range(3))
        if resid <= tol:
            aut.append(tuple(tuple(int(x) for x in row) for row in P))
    return aut


def even_cuboid_cell(odd=CUBOID):
    """Even-class basis of a cuboid via a non-odd closure member."""
    from agentsg.cell.canonical import _metric, canonical_superbase, _dotG
    from agentsg.cell.selling_closure import selling_superbase_closure

    G = _metric(odd)
    C0, _ = canonical_superbase(odd)
    sig0 = tuple(sorted(round(_dotG(C0[i], C0[i], G), 8) for i in range(4)))
    Ce = next(
        C for C in selling_superbase_closure(odd)
        if tuple(sorted(round(_dotG(C[i], C[i], G), 8) for i in range(4))) != sig0
    )
    P = tuple(tuple(Ce[j + 1][i] for j in range(3)) for i in range(3))
    return cell_from_metric(transform_metric(G, P)), P


def lepage_delta_deg(P, G):
    """Rough Le Page angular residual (degrees) of a candidate 2-fold-like P.

    For the monoclinic a<->c exchange, measures how far P is from being a
    metric symmetry of G: residual of P^T G P vs G, converted to an angle-like
    scale via acos on the off-diagonal mismatch of the transformed metric.
    """
    Gp = transform_metric(G, P)
    # Angular proxy: max |angle change| of the three cell angles under Gp vs G
    from agentsg.cell.metric import cell_from_metric
    c0 = cell_from_metric(G)
    c1 = cell_from_metric(Gp)
    return max(abs(c0[i] - c1[i]) for i in (3, 4, 5))


# --- PREPARE_REFERENCE / REINDEX_FRAME (thin wrappers for Z.15 / Z.19) ---

@dataclass
class ReferenceCache:
    cell: tuple
    symbol: str | None
    prim_cell: tuple
    C_matrix: list  # conventional -> primitive (columns)
    G_prim: list
    key: tuple
    radius: float
    aut: list
    closure_call_count: int = 0
    _closure: list = field(default_factory=list, repr=False)


def prepare_reference(R_conv, sigma_theta=None, symbol=None, radius=None):
    """Cache reference lattice objects (App. D PREPARE_REFERENCE)."""
    from agentsg.cell.primitive import primitive_cell, primitive_transform
    from agentsg.cell.rootform import sorted_root_key
    from agentsg.cell.selling_closure import selling_superbase_closure
    from agentsg.cell.canonical import reindexing_via_canonical

    if symbol:
        C = primitive_transform(symbol)
        prim = primitive_cell(R_conv, symbol)
    else:
        C = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        prim = tuple(R_conv)

    G = metric_tensor(prim)
    cl = selling_superbase_closure(prim, angle_sigma=sigma_theta)
    key = sorted_root_key(prim)
    if radius is None:
        if sigma_theta is None:
            radius = 1.0
        else:
            # rough noise radius in key space
            radius = max(0.5, 50.0 * math.radians(float(sigma_theta)))
    aut = reindexing_via_canonical(
        prim, prim, boundary_rel=0, verify_rel=1e-9, angle_sigma=sigma_theta,
    )
    ref = ReferenceCache(
        cell=tuple(R_conv),
        symbol=symbol,
        prim_cell=tuple(prim),
        C_matrix=C,
        G_prim=G,
        key=key,
        radius=float(radius),
        aut=aut,
        closure_call_count=1,
        _closure=cl,
    )
    return ref


def reindex_frame(ref: ReferenceCache, N_conv, sigma_theta=None, verify_rel=1e-6,
                  screen=True):
    """Per-frame reindex onto a prepared reference (App. D REINDEX_FRAME)."""
    from agentsg.cell.primitive import primitive_cell
    from agentsg.cell.rootform import sorted_root_key, sorted_root_distance
    from agentsg.cell.canonical import reindexing_via_canonical

    if ref.symbol:
        n_prim = primitive_cell(N_conv, ref.symbol)
    else:
        n_prim = tuple(N_conv)

    if screen:
        d = sorted_root_distance(ref.prim_cell, n_prim)
        if d > ref.radius:
            return []

    # Reference closure is NOT re-enumerated: match via reindexing_via_canonical
    # which will recompute N's closure only. Count stays on ref.
    ops = reindexing_via_canonical(
        ref.prim_cell, n_prim,
        boundary_rel=0,
        verify_rel=verify_rel,
        angle_sigma=sigma_theta,
    )
    return ops


def reindex_to_reference(R, N, sigma_theta=None, symbol=None, verify_rel=1e-6):
    """One-shot REINDEX_TO_REFERENCE = prepare + frame."""
    ref = prepare_reference(R, sigma_theta=sigma_theta, symbol=symbol)
    return reindex_frame(ref, N, sigma_theta=sigma_theta, verify_rel=verify_rel,
                         screen=False), ref


def coset_equals_M_times_aut(ops, M, aut):
    """True if set(ops) equals {M @ a for a in aut} as integer matrices."""
    M = [list(row) for row in M]
    expected = set()
    for a in aut:
        Pa = matmul(M, [list(r) for r in a])
        # integerize
        Pt = tuple(tuple(int(round(x)) for x in row) for row in Pa)
        expected.add(Pt)
    got = set(tuple(tuple(int(x) for x in row) for row in P) for P in ops)
    return got == expected


def matmul_int(A, B):
    return tuple(
        tuple(sum(int(A[i][k]) * int(B[k][j]) for k in range(3)) for j in range(3))
        for i in range(3)
    )
