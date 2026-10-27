"""z67 — the reindexing workflow (supporting information, Section S10).

The steps of the workflow rest on four identities: the input bases drop out of every price
(step 2), the determinant rule fixes the sign of each ordered basis (step 3), every
candidate has the same best scale (step 4), and the necessary condition of step 5 never
discards an ordering within the band.  Each is checked here on seeded random cells, the
worked example is recomputed against the numbers of Tables S17-S19, and the comparison with
the direct form on the benchmark is read from the deposited results and regenerated.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from sellingproofs import paths
from sellingproofs.registry import claim

from agentsg.cell import strain_closure as sc
from agentsg.cell.metric import UnitCell, params_from_metric

WORKED = (30.02, 39.97, 50.03, 90.25, 92.05, 89.80)
RESULTS = paths.BENCH / "results_reindex_workflow.txt"


def _metric(cell):
    return np.array(UnitCell(*cell).metric_tensor(), float)


def _strained(G, eta, rng):
    w, V = np.linalg.eigh(G); Gh = V @ np.diag(w ** .5) @ V.T
    X = rng.normal(size=(3, 3)); X = X + X.T; X *= eta / np.linalg.norm(X)
    return Gh @ (np.eye(3) + X) @ Gh


def _unimodular(rng, k=2):
    while True:
        U = rng.integers(-k, k + 1, (3, 3))
        if round(abs(np.linalg.det(U))) == 1:
            return U


def _floats(M):
    return np.array([[float(x) for x in row] for row in M.rows])


def _random_case(rng, top=0.04):
    p = rng.uniform(5, 100, 6) * np.exp(rng.uniform(0, np.log(4), 6))
    p[rng.choice(6, int(rng.integers(0, 3)), replace=False)] = 0.0   # at most two zeros: K4 stays connected
    G = sc._gram(p)
    cell = tuple(params_from_metric(G.tolist()))
    U = _unimodular(rng)
    new = tuple(params_from_metric((U.T @ _strained(G, rng.uniform(0.001, top), rng) @ U).tolist()))
    return cell, new


@claim("z67", "SI §S10", kind="sampled")
def test_input_bases_drop_out_of_every_price():
    """Step 2: for P = B_R W B_N^-1 the price between the input metrics equals the price
    between the new cell's reduced metric and W^T G_R W, for every member, labelling and
    seeded cell, centred references included."""
    rng = np.random.default_rng(671)
    cases = [_random_case(rng) for _ in range(30)]
    cases += [((50.0, 30.0, 40.0, 90, 105, 90), None), ((30.0, 40.0, 50.0, 90, 90, 90), None)]
    for cell, new in cases:
        sg = None
        if new is None:                          # a centred reference against a primitive new cell
            sg = "C 1 2 1" if cell[4] != 90 else "I 2 2 2"
            Bf = _floats(sc.reduce_cell(cell, sg).basis)
            new = tuple(params_from_metric((Bf.T @ _metric(cell) @ Bf).tolist()))
        ref = sc.widened_closure(cell, sg)
        red_n = sc.reduce_cell(new, None)
        BR, BN = _floats(ref.reduced.basis), _floats(red_n.basis)
        GR = BR.T @ _metric(ref.reduced.cell) @ BR
        GN = BN.T @ _metric(red_n.cell) @ BN
        for m in ref.members[:6]:
            for B in list(sc._relabellings(np.rint(np.asarray(m.superbase)).astype(int)))[:12]:
                W = B.T.astype(float)
                P = BR @ W @ np.linalg.inv(BN)
                direct = sc._price(_metric(red_n.cell), P.T @ _metric(ref.reduced.cell) @ P)[0]
                reduced = sc._price(GN, W.T @ GR @ W)[0]
                assert abs(direct - reduced) < 1e-9, (cell, direct, reduced)


@claim("z67", "SI §S10", kind="exhaustive")
def test_determinant_rule_and_common_scale():
    """Steps 3 and 4: every ordered basis of every member has determinant +-1, the sign
    c det B sgn det B_R sgn det B_N is the sign of det P, the two signs give the same metric,
    and every predicted metric has the reference's determinant, so one scale serves all
    candidates."""
    rng = np.random.default_rng(672)
    for _ in range(20):
        cell, new = _random_case(rng)
        ref = sc.widened_closure(cell)
        red_n = sc.reduce_cell(new)
        BR, BN = _floats(ref.reduced.basis), _floats(red_n.basis)
        GR = BR.T @ _metric(ref.reduced.cell) @ BR
        for m in ref.members:
            for B in sc._ordered_bases(np.rint(np.asarray(m.superbase)).astype(int)):
                dB = round(np.linalg.det(B))
                assert abs(dB) == 1
                for c in (1, -1):
                    P = BR @ (c * B).T @ np.linalg.inv(BN)
                    assert np.sign(np.linalg.det(P)) == c * dB * np.sign(np.linalg.det(BR)) * np.sign(np.linalg.det(BN))
                H = B @ GR @ B.T
                assert np.allclose(H, (-B) @ GR @ (-B).T)
                assert abs(np.linalg.det(H) / np.linalg.det(GR) - 1) < 1e-9


@claim("z67", "SI §S10", kind="sampled")
def test_necessary_condition_never_discards_a_candidate():
    """Step 5: for seeded cells and bands from 1 to 20 %, every ordering whose price is within
    the band satisfies the ten inequalities |s u_i^T H u_j - u_i.u_j| <= eta |u_i||u_j|
    exactly (no tolerance), and the workflow returns the same operators as the direct form."""
    rng = np.random.default_rng(673)
    S0 = np.array(sc._S0, float)
    checked = 0
    for _ in range(80):
        cell, new = _random_case(rng, 0.015)
        accept = float(rng.uniform(0.02, 0.2))
        ref = sc.widened_closure(cell, eps=sc.band_for(accept))
        red_n = sc.reduce_cell(new)
        BR, BN = _floats(ref.reduced.basis), _floats(red_n.basis)
        GR = BR.T @ _metric(ref.reduced.cell) @ BR
        GN = BN.T @ _metric(red_n.cell) @ BN
        s = (np.linalg.det(GN) / np.linalg.det(GR)) ** (1 / 3)
        V = S0 @ GN @ S0.T
        nrm = np.sqrt(np.diag(V))
        eta = np.expm1(accept)
        for m in ref.members:
            for B in sc._ordered_bases(np.rint(np.asarray(m.superbase)).astype(int)):
                H = B @ GR @ B.T
                price = sc._price(GN, H)[0]
                if price <= accept:
                    checked += 1
                    assert np.all(np.abs(s * (S0 @ H @ S0.T) - V) <= eta * np.outer(nrm, nrm)), (cell, price)
        a, _ = sc._reindex_direct(ref, new, accept=accept)
        b, _ = sc.reindex(ref, new, accept=accept)
        A = {tuple(c.operator.rows): c.price for c in a}
        Bp = {tuple(c.operator.rows): c.price for c in b}
        assert set(A) == set(Bp) and all(abs(A[k] - Bp[k]) < 1e-9 for k in A)
    assert checked > 100


@claim("z67", "SI §S10", "SI Table S17", "SI Table S18", "SI Table S19", kind="exhaustive")
def test_worked_example_step_by_step():
    """SI Tables S17-S19: on the worked cell, with the new cell in the basis (a, b, a+c) and a
    strain of 0.3 %, the 16 members give 384 ordered bases, 4 pass the necessary condition,
    2 lie within the band, at 0.212 % (the true operator) and 1.381 % (with the
    pseudo-twofold), and the other two are priced 10.194 % and 10.208 %; the common scale is
    1.001225 and the direct form builds 768 operators to return the same two."""
    rng = np.random.default_rng(1)
    ref = sc.widened_closure(WORKED)
    P0 = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    new = params_from_metric((P0.T @ _strained(_metric(WORKED), 0.003, rng) @ P0).tolist())
    wf = sc.reindex_workflow(ref, new)
    assert (wf.n_members, wf.n_orderings, wf.n_signed, wf.n_prefilter, wf.n_within, wf.n_candidates) == \
        (16, 384, 384, 4, 2, 2)
    assert f"{wf.scale:.6f}" == "1.001225" and f"{wf.eta:.6f}" == "0.076807"
    assert [f"{100 * p[3]:.3f}" for p in wf.priced] == ["0.212", "1.381", "10.194", "10.208"]
    assert [(k, o, c) for k, o, c, _ in wf.priced] == [(0, 4, -1), (4, 18, -1), (10, 22, -1), (14, 16, 1)]
    ops = [[[int(x) for x in r] for r in c.operator.rows] for c in wf.candidates]
    assert ops == [[[1, 0, 1], [0, 1, 0], [0, 0, 1]], [[-1, 0, -1], [0, 1, 0], [0, 0, -1]]]
    assert [f"{x:.2f}" for x in wf.new_conorms] == ["50.61", "2448.09", "1594.50", "847.80", "3.51", "2.88"]
    assert [f"{x:.2f}" for x in wf.reference_conorms] == ["49.54", "1588.88", "2444.74", "4.19", "847.47", "4.54"]
    assert [f"{x:.1f}" for x in wf.bounds[0]] == ["314.4", "147.6", "282.2", "196.6"]
    assert len(sc._reindex_direct(ref, new)[0]) == 2 and 48 * wf.n_members == 768
    depths = [f"{100 * m.depth:.3f}" for m in ref.members]
    assert depths[:6] == ["0.000", "0.280", "0.494", "0.617", "0.777", "0.777"]
    assert min(float(d) for d in depths[6:]) > 3.7 and max(float(d) for d in depths[6:]) < 5.12


def _results():
    return RESULTS.read_text()


@claim("z67", "SI §S10", "SI Table S16", "main §4.2", kind="benchmark")
def test_workflow_against_the_direct_form_on_the_test_cases():
    """The workflow returns the same operators as the direct form in all 780 benchmark cases
    and all 24 centred cases, prices agreeing to 3e-12; members median 3 and at most 56;
    ordered bases median 72 and at most 1344; 5.9 % of them pass the necessary condition
    (median 4, at most 64); exact operators built 238 224 in the direct form and 5 797 in the
    workflow, which is the number returned; about a twentieth of the time (about 27 s against
    1.3 s in the deposited run)."""
    t = _results()
    assert "the same set of operators in 780 of 780 cases" in t
    assert "the same set of operators in 24 of 24 cases" in t
    d = float(re.search(r"largest price difference ([\d.e+-]+)\n  members", t).group(1))
    assert f"{d:.0e}" == "3e-12"
    assert "members: median 3, max 56" in t
    assert "ordered bases per case: median 72, max 1344" in t
    m = re.search(r"passing the necessary condition: median (\d+), max (\d+); ([\d.]+) %", t)
    assert (m.group(1), m.group(2), f"{float(m.group(3)):.1f}") == ("4", "64", "5.9")
    assert "exact operators built: direct form 238224, workflow 5797; distinct operators returned 5797" in t
    sec = re.search(r"seconds: direct form ([\d.]+), workflow ([\d.]+)", t)
    assert (round(float(sec.group(1))), round(float(sec.group(2)), 1)) == (27, 1.3)   # SI Table S16
    f = float(re.search(r"\(factor (\d+)\)", t).group(1))
    assert 15 <= f <= 30                     # 'about a twentieth', from the deposited run


@claim("z67", "SI §S10", kind="reproduce", needs=("slow",))
def test_reproduce_reindex_workflow():
    """reindex_workflow.py regenerates results_reindex_workflow.txt identically, apart from
    the lines that report wall-clock times ('seconds')."""
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, SELLINGPROOFS_OUT=d,
                   PYTHONPATH=os.pathsep.join([str(paths.REPO / "agentsg" / "src"), str(paths.PACKAGE.parent)]))
        r = subprocess.run([sys.executable, str(paths.BENCH / "reindex_workflow.py")], env=env,
                           capture_output=True, text=True, timeout=3600)
        assert r.returncode == 0, r.stderr[-2000:]
        new = open(os.path.join(d, "results_reindex_workflow.txt")).read()
    keep = lambda txt: [l for l in txt.splitlines() if "seconds" not in l]          # noqa: E731
    assert keep(new) == keep(_results())
