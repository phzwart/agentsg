"""z:reduction-step — Selling move identities + random termination."""
from __future__ import annotations

import math
import random

import pytest

from helpers import SEED_REDUCTION

pytestmark = [pytest.mark.zcheck]


@pytest.mark.symbolic
def test_selling_move_conorm_identities_sympy():
    sympy = pytest.importorskip("sympy")
    # Symbolic 3-vectors with v0 = -(v1+v2+v3)
    comps = {(i, k): sympy.symbols(f"v{i}_{k}", real=True) for i in range(1, 4) for k in range(3)}
    v = {1: sympy.Matrix([comps[(1, k)] for k in range(3)]),
         2: sympy.Matrix([comps[(2, k)] for k in range(3)]),
         3: sympy.Matrix([comps[(3, k)] for k in range(3)])}
    v[0] = -(v[1] + v[2] + v[3])

    def dot(a, b):
        return (a.T * b)[0]

    # Apply move at (i,j)=(0,1): u0=-v0, u1=v1, u2=v0+v2, u3=v0+v3
    i, j = 0, 1
    u = dict(v)
    vi = v[i]
    u[i] = -vi
    u[j] = v[j]
    for t in (2, 3):
        u[t] = vi + v[t]

    # zero sum preserved
    assert sympy.simplify(u[0] + u[1] + u[2] + u[3]) == sympy.Matrix([0, 0, 0])

    eps = dot(v[i], v[j])  # = -p_ij before move when obtuse? move when positive product
    # Conorms p = -dot; after move the manuscript formulas use ε = v_i·v_j
    p = {(a, b): -dot(v[a], v[b]) for a in range(4) for b in range(a + 1, 4)}
    q = {(a, b): -dot(u[a], u[b]) for a in range(4) for b in range(a + 1, 4)}

    # Manuscript: q_ij=ε, q_jk=p_jk-ε, q_jl=p_jl-ε, q_ik=p_il-ε, q_il=p_ik-ε, q_kl=p_kl+ε
    # with ε = v_i·v_j = -p_ij when using conorm convention... Manuscript says
    # ε = v_i·v_j and q_ij = ε. With p = -dot, before move p_ij = -ε.
    # Stated: q_ij=ε (= -p_ij flipped sign of that slot).
    k, l = 2, 3
    claimed = {
        (i, j): eps,
        (j, k): p[(min(j, k), max(j, k))] - eps,
        (j, l): p[(min(j, l), max(j, l))] - eps,
        (i, k): p[(min(i, l), max(i, l))] - eps,  # q_ik = p_il - ε
        (i, l): p[(min(i, k), max(i, k))] - eps,  # q_il = p_ik - ε
        (k, l): p[(k, l)] + eps,
    }
    for pair, expr in claimed.items():
        a, b = (pair if pair[0] < pair[1] else (pair[1], pair[0]))
        diff = sympy.simplify(q[(a, b)] - expr)
        assert diff == 0, f"pair {a},{b}: {diff}"

    sum_p = sum(p.values())
    sum_q = sum(q.values())
    assert sympy.simplify(sum_p - sum_q - eps) == 0


def test_random_reduction_terminates_and_formulas():
    """3000 random bases: move formulas hold; terminates obtuse; |det| unchanged."""
    from agentsg.cell.selling_closure import _selling_flip
    from agentsg.cell.canonical import _metric, _dotG

    rng = random.Random(SEED_REDUCTION)
    max_steps = 0
    n_ok = 0
    for _ in range(3000):
        # random nearly-generic cell
        cell = (
            rng.uniform(20, 80),
            rng.uniform(20, 80),
            rng.uniform(20, 80),
            rng.uniform(70, 110),
            rng.uniform(70, 110),
            rng.uniform(70, 110),
        )
        G = _metric(cell)
        # integer identity superbase in cell basis, then reduce with flips
        C = [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # Start from Cartesian-free integer coords; use Selling flips on G
        steps = 0
        for _step in range(500):
            # most positive scalar product (acute)
            worst = 1e-12
            wi = wj = -1
            for i in range(4):
                for j in range(i + 1, 4):
                    d = _dotG(C[i], C[j], G)
                    if d > worst:
                        worst = d
                        wi, wj = i, j
            if wi < 0:
                break
            # check sum decrease of conorms (= -scalar products)
            sum_before = sum(
                -_dotG(C[a], C[b], G) for a in range(4) for b in range(a + 1, 4)
            )
            eps = _dotG(C[wi], C[wj], G)
            Cn = _selling_flip(C, wi, wj)
            sum_after = sum(
                -_dotG(Cn[a], Cn[b], G) for a in range(4) for b in range(a + 1, 4)
            )
            assert sum_after < sum_before - 0.5 * abs(eps) + 1e-6 or abs(eps) < 1e-9
            C = Cn
            steps += 1
        max_steps = max(max_steps, steps)
        # obtuse
        for i in range(4):
            for j in range(i + 1, 4):
                assert _dotG(C[i], C[j], G) <= 1e-6
        # det of three vectors unchanged magnitude
        U = [[C[1][r], C[2][r], C[3][r]] for r in range(3)]
        det = (U[0][0] * (U[1][1] * U[2][2] - U[1][2] * U[2][1])
               - U[0][1] * (U[1][0] * U[2][2] - U[1][2] * U[2][0])
               + U[0][2] * (U[1][0] * U[2][1] - U[1][1] * U[2][0]))
        assert abs(abs(det) - 1) < 1e-9 or abs(det) == 1
        n_ok += 1

    assert n_ok == 3000
    assert 1 <= max_steps < 500
    # manuscript observed 103; we only require termination bound
