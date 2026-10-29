"""Origin/setting alignment of cluster members (agentsg.cell.cluster_frames), synthetic cases."""
import math
from fractions import Fraction as Fr

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from agentsg.cell.cluster_frames import Model, _Reference, _orth, place
from agentsg.change_of_basis import ChangeOfBasis
from agentsg.linalg import Matrix3, Vector3
from agentsg.space_groups import space_group


def _cell_from_basis(B):
    a, b, c = (B[:, k] for k in range(3))
    L = np.linalg.norm
    ang = lambda u, v: math.degrees(math.acos(u @ v / L(u) / L(v)))
    return (L(a), L(b), L(c), ang(b, c), ang(a, c), ang(a, b))


def _chain(rng, n=150):
    """A compact random walk of Cα-like points (3.8 A steps), as Cartesian."""
    steps = rng.normal(size=(n, 3))
    steps *= 3.8 / np.linalg.norm(steps, axis=1, keepdims=True)
    x = np.cumsum(steps, axis=0)
    return x - x.mean(axis=0) + np.array([9.0, 7.0, 5.0])


def _make(cell, sg, xyz_cart, pid, unp0=1):
    frac = xyz_cart @ np.linalg.inv(_orth(cell)).T
    return Model(pid, tuple(cell), sg, 1.5,
                 {"A": ("P12345", np.arange(unp0, unp0 + len(frac)), frac)})


def _derive(ref_model, Pc, pc, op, noise, rng, sg_new):
    """Member = reference coordinates after symmetry op `op`, written in the setting
    reached by ChangeOfBasis(Pc, pc) (old = reference), with Cartesian noise."""
    cob = ChangeOfBasis(Matrix3([[Fr(x) for x in r] for r in Pc]), Vector3([Fr(x) for x in pc]))
    Pm = np.array(Pc, float)
    _, unp, x = next(iter(ref_model.chains.values()))
    W = np.array([[float(v) for v in r] for r in op.W.rows])
    w = np.array([float(v) for v in op.w.v])
    x = x @ W.T + w
    Bref = _orth(ref_model.cell)
    Bnew = Bref @ Pm
    new_cell = _cell_from_basis(Bnew)
    xm = (x - np.array(pc, float)) @ np.linalg.inv(Pm).T
    xm += (rng.normal(scale=noise, size=xm.shape)) @ np.linalg.inv(_orth(new_cell)).T
    # the member's operators must be the standard ones of sg_new
    ops_new = {cob.apply_to_op(o) for o in space_group(ref_model.sg_hm).operations()}
    assert ops_new == set(space_group(sg_new).operations())
    return Model("MEM", new_cell, sg_new, 2.0, {"B": ("P12345", unp.copy(), xm)})


@pytest.mark.parametrize("case", [
    # P212121, cyclic axis relabel, origin shift (1/2, 0, 0), symmetry op 2
    dict(sg="P 21 21 21", cell=(40.0, 50.0, 60.0, 90, 90, 90),
         Pc=[[0, 0, 1], [1, 0, 0], [0, 1, 0]], pc=[0.5, 0, 0], sg_new="P 21 21 21", op=2),
    # P21: floating origin along b (0.37) plus discrete shift (1/2, 0, 1/2)
    dict(sg="P 1 21 1", cell=(42.2, 41.3, 72.0, 90, 104.3, 90),
         Pc=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], pc=[0.5, 0.37, 0.5], sg_new="P 1 21 1", op=1),
    # P43212 with a normalizer axis swap a <-> b (c reversed to stay right-handed)
    dict(sg="P 43 21 2", cell=(79.1, 79.1, 37.9, 90, 90, 90),
         Pc=[[0, 1, 0], [1, 0, 0], [0, 0, -1]], pc=[0, 0, 0], sg_new="P 43 21 2", op=3),
    # C2 deposited as I2 (a' = -a - c, c' = -c)
    dict(sg="C 1 2 1", cell=(113.3, 53.2, 44.7, 90, 102.3, 90),
         Pc=[[-1, 0, 0], [0, 1, 0], [-1, 0, -1]], pc=[0, 0, 0], sg_new="I 1 2 1", op=1),
    # C2 deposited as I2 and moved by the allowed origin (0, 0, 1/2)
    dict(sg="C 1 2 1", cell=(113.3, 53.2, 44.7, 90, 102.3, 90),
         Pc=[[-1, 0, 0], [0, 1, 0], [-1, 0, -1]], pc=[0, 0.21, 0.5], sg_new="I 1 2 1", op=2),
])
def test_recovers_setting_and_origin(case):
    rng = np.random.default_rng(7)
    ref = _make(case["cell"], case["sg"], _chain(rng), "REF")
    ops = sorted(space_group(case["sg"]).operations(), key=lambda o: (o.W.rows, o.w.v))
    mem = _derive(ref, case["Pc"], case["pc"], ops[case["op"] % len(ops)], 0.2, rng, case["sg_new"])
    p = place(_Reference(ref), mem)
    assert p.status == "placed", p.status
    assert p.rmsd < 0.5                       # noise is 0.2 A per coordinate (~0.35 A rms)
    assert p.asu_rmsd < 0.5
    assert p.margin is None or p.margin > 2.0


def test_different_space_group_is_skipped():
    rng = np.random.default_rng(1)
    ref = _make((40.0, 50.0, 60.0, 90, 90, 90), "P 21 21 21", _chain(rng), "REF")
    mem = _make((40.0, 50.0, 60.0, 90, 90, 90), "P 1 21 1", _chain(rng), "MEM")
    assert place(_Reference(ref), mem).status == "different_space_group"


def _close_mod_lattice(t1, t2, tol=0.02):
    d = np.asarray(t1, float) - np.asarray(t2, float)
    return np.all(np.abs(d - np.round(d)) < tol)


def test_composed_frame_matches_direct():
    from agentsg.cell.cluster_frames import compose
    rng = np.random.default_rng(11)
    ref = _make((40.0, 50.0, 60.0, 90, 90, 90), "P 21 21 21", _chain(rng), "REF")
    ops = sorted(space_group("P 21 21 21").operations(), key=lambda o: (o.W.rows, o.w.v))
    anchor = _derive(ref, [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [0.5, 0, 0], ops[1], 0.0, rng,
                     "P 21 21 21")
    anchor.pdb_id = "ANC"
    member = _derive(anchor, [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [0, 0.5, 0.5], ops[3], 0.0, rng,
                     "P 21 21 21")
    direct = place(_Reference(ref), member)
    fa = place(_Reference(ref), anchor)
    fm = place(_Reference(anchor), member)
    L, t = compose(fa.L, fa.t, fm.L, fm.t)
    # the same placement up to the group: compare where the member's atoms land
    x = member.chains["B"][2]
    from agentsg.cell.cluster_frames import _mat
    a = x @ _mat(direct.L).T + direct.t
    b = x @ _mat(L).T + t
    ops_f = [(np.array([[float(v) for v in r] for r in o.W.rows]),
              np.array([float(v) for v in o.w.v])) for o in space_group("P 21 21 21").operations()]
    assert any(_close_mod_lattice((b @ W.T + w).mean(0), a.mean(0)) and
               np.allclose(b @ W.T + w - np.round((b @ W.T + w - a).mean(0)), a, atol=1e-6)
               for W, w in ops_f)


def test_accession_fallback_finds_numbering_shift():
    rng = np.random.default_rng(5)
    ref = _make((42.2, 41.3, 72.0, 90, 104.3, 90), "P 1 21 1", _chain(rng), "REF", unp0=3264)
    acc, unp, x = ref.chains["A"]
    mem = Model("MEM", ref.cell, "P 1 21 1", 2.0, {"A": ("A0A7U3EDN3", unp - 22, x.copy())})
    p = place(_Reference(ref), mem)
    assert p.status == "placed" and p.acc_fallback
    assert p.unp_offset == 22 and p.rmsd < 1e-6


def test_register_shift_found_from_structure():
    """Same accession, numbered 37 residues off: the register comes from the structure."""
    rng = np.random.default_rng(13)
    ref = _make((42.2, 41.3, 72.0, 90, 104.3, 90), "P 1 21 1", _chain(rng), "REF", unp0=20)
    acc, unp, x = ref.chains["A"]
    mem = Model("MEM", ref.cell, "P 1 21 1", 2.0, {"A": (acc, unp + 37, x.copy())})
    p = place(_Reference(ref), mem)
    assert p.status == "placed" and not p.acc_fallback
    assert p.unp_offset == -37 and p.rmsd < 1e-6


def test_second_copy_on_noncrystallographic_shift_is_not_chosen():
    """Two copies in the asymmetric unit, B = A moved by a non-crystallographic shift. Both
    score the same before the origin snap; only A sits on an allowed origin."""
    rng = np.random.default_rng(21)
    cell = (42.2, 41.3, 72.0, 90, 104.3, 90)
    ref = _make(cell, "P 1 21 1", _chain(rng), "REF")
    acc, unp, x = ref.chains["A"]
    shifted = x + np.array([0.13, 0.0, 0.27])          # not an allowed origin of P21
    noisy = x + rng.normal(scale=0.2, size=x.shape) @ np.linalg.inv(_orth(cell)).T
    # B scores better before the snap (an exact copy), A after it (0.2 A of noise)
    mem = Model("MEM", cell, "P 1 21 1", 2.0, {"B": (acc, unp.copy(), shifted),
                                                "A": (acc, unp.copy(), noisy)})
    p = place(_Reference(ref), mem)
    assert p.member_chain == "A" and p.rmsd < 0.5
    assert p.margin is not None and p.margin > 1.0


def test_composed_frame_brought_back_onto_reference_copy():
    """A frame that puts the molecule on a symmetry mate is moved back by a group operator."""
    from agentsg.cell.cluster_frames import Placement, _onto_reference_copy
    from fractions import Fraction
    rng = np.random.default_rng(4)
    ref = _make((58.5, 147.0, 64.5, 90, 99.96, 90), "P 1 21 1", _chain(rng), "REF")
    mem = Model("MEM", ref.cell, "P 1 21 1", 2.0, {"A": ref.chains["A"]})
    R = _Reference(ref)
    screw = next(op for op in R.ops if op.W.rows != Matrix3([[Fraction(int(i == j)) for j in range(3)] for i in range(3)]).rows)
    W = np.array([[float(v) for v in r] for r in screw.W.rows]); w = np.array([float(v) for v in screw.w.v])
    q = Placement("MEM", "REF", "placed", L=screw.W, t=w)        # lands on the 2_1 mate
    from agentsg.cell.cluster_frames import _evaluate
    _evaluate(R, mem, q)
    assert q.rmsd > 5
    _onto_reference_copy(R, mem, q)
    _evaluate(R, mem, q)
    assert q.rmsd < 1e-6
