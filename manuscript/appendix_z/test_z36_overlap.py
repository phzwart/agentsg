"""z:overlap — the widened closure is a union of on-cone Kurlin guarantees.

Two directions. Overlap: for lattices within eps (Frobenius on G, shared
frame), each one's EXACT closure lies inside the other's widened closure at
tau = 3 eps, same integer coordinates -- because closure-member coordinates
have entries <= 1, so each conorm moves by at most 3 eps. Clipping: every
widened member is an EXACT obtuse superbase (hence a Kurlin-certified
closure member) of an explicitly constructed lattice within 2*sqrt(6)*tau.
So no guarantee is ever used off the cone: the band is bookkeeping for a
family of exact closures, one per admissible perturbation.
"""
from __future__ import annotations

import itertools
import math
import random

import pytest

pytestmark = [pytest.mark.zcheck]

from helpers import TYPE_CONORMS, cell_from_conorms  # noqa: E402


def _np():
    import numpy
    return numpy


def _gram_from_p(p):
    p01, p02, p03, p12, p13, p23 = p
    return [[p01 + p12 + p13, -p12, -p13],
            [-p12, p02 + p12 + p23, -p23],
            [-p13, -p23, p03 + p13 + p23]]


def _conorms(member, G):
    np = _np()
    W = [np.array(v, float) for v in member]
    return [-(W[i] @ G @ W[j]) for i in range(4) for j in range(i + 1, 4)]


def _canonset(members):
    out = set()
    for m in members:
        best = None
        for perm in itertools.permutations(range(4)):
            for s in (1, -1):
                t = tuple(tuple(int(s * x) for x in m[p]) for p in perm)
                if best is None or t < best:
                    best = t
        out.add(best)
    return out


def test_parent_closure_inside_widened_daughter():
    """All five types, seeded: penetration <= 3 eps and full containment."""
    import numpy as np
    from agentsg.cell.canonical import canonical_superbase, superbase_variants
    from agentsg.cell.metric import UnitCell, params_from_metric
    from agentsg.cell.selling_closure import selling_superbase_closure

    rng = random.Random(3)
    for vt in (1, 2, 3, 4, 5):
        for _ in range(2):
            cell = cell_from_conorms(
                [x * (25 + 10 * rng.random()) for x in TYPE_CONORMS[vt]()])
            G = np.array(UnitCell(*cell).metric_tensor())
            A = np.array([[rng.gauss(0, 1) for _ in range(3)]
                          for _ in range(3)])
            E = 0.5 * (A + A.T)
            E /= np.linalg.norm(E)
            eps = 0.01 * np.linalg.norm(G)
            Gp = G + eps * E
            if np.linalg.eigvalsh(Gp)[0] <= 0:
                continue
            cellp = tuple(params_from_metric(Gp.tolist()))
            parent = selling_superbase_closure(cell)
            pen = max(max(0.0, -min(_conorms(m, Gp))) for m in parent)
            assert pen <= 3.0 * eps + 1e-9, (vt, pen / eps)
            C0, _ = canonical_superbase(cellp)
            scale = max(abs(np.array(v) @ Gp @ np.array(v)) for v in C0)
            wid = superbase_variants(cellp, boundary_rel=3.0 * eps / scale,
                                     max_variants=512)
            assert _canonset(parent) <= _canonset(wid), vt


def test_clipping_yields_exact_member_of_nearby_lattice():
    """Every widened member with a negative conorm becomes an exact obtuse
    superbase of the clipped lattice, within 2*sqrt(6)*tau of the original."""
    import numpy as np
    from agentsg.cell.canonical import canonical_superbase, superbase_variants
    from agentsg.cell.metric import UnitCell

    delta = 0.8  # two small conorms put the lattice near walls
    cell = cell_from_conorms((delta, 50.0, 75.0, 100.0, 125.0, 2.0 * delta))
    G = np.array(UnitCell(*cell).metric_tensor())
    C0, _ = canonical_superbase(cell)
    scale = max(abs(np.array(v) @ G @ np.array(v)) for v in C0)
    tau = 3.0 * (3.0 * delta)  # a band wide enough to cross both walls
    wid = superbase_variants(cell, boundary_rel=tau / scale, max_variants=512)
    n_clipped = 0
    for m in wid:
        p = _conorms(m, G)
        if min(p) >= -1e-9:
            continue  # already exactly obtuse
        assert min(p) >= -tau * 2.0 - 1e-6  # within the (2x) band
        # rebuild the member Gram in its own frame and clip there
        order = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        pm = dict(zip(order, p))
        q = [max(0.0, pm[k]) for k in order]
        Hc = np.array(_gram_from_p(q))
        Hm = np.array(_gram_from_p([pm[k] for k in order]))
        n_clipped += 1
        # exact obtuseness of the standard superbase under the clipped Gram
        std = [(-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        assert min(_conorms(std, Hc)) >= -1e-9
        assert np.linalg.eigvalsh(Hc)[0] > 0
        assert np.linalg.norm(Hc - Hm) <= 2.0 * math.sqrt(6.0) * (2.0 * tau)
    assert n_clipped >= 1  # the band genuinely contained non-obtuse frames
