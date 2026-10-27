"""z:anisotropy-shapes — per-slot bands for anisotropic cells, and the classical
ambiguities as cell shapes of the closure.

(i) With a band per slot proportional to the two vector lengths (angular-error
model, tau_ij = sigma |u_i||u_j|), the idealise-enumerate-filter construction
is exact on needles far beyond the uniform hypothesis; when it zeroes a
conorm comparable to the shortest vector the idealised lattice is singular,
which is the per-slot hypothesis failing and is detected. Plates at ordinary
angular precision are a genuine floor: the widened set exceeds sixteen
members. (ii) In every primitive sort with several cell shapes the extra
shapes are monoclinic-looking cells; C-centred orthorhombic has two
monoclinic-looking shapes; primitive monoclinic has three monoclinic
settings.
"""
from __future__ import annotations

from sellingproofs.registry import claim  # noqa: E402

import itertools

import pytest

from .test_z44_widened_complete import S0, SL, _cn, _flip, _gram, _key  # noqa: E402
from .test_z46_selling_groupoid import SORTS, _orbits_and_shapes  # noqa: E402


@pytest.fixture(scope="module")
def superbases():
    import numpy as np
    r = np.arange(-2, 3)
    M = np.array(np.meshgrid(*[r] * 9, indexing="ij")).reshape(9, -1).T.reshape(-1, 3, 3)
    B = M[np.abs(np.round(np.linalg.det(M))) == 1].astype(float)
    return np.concatenate([-(B.sum(1))[:, None, :], B], 1)


def _band(S, G, sig):
    import numpy as np
    L = np.sqrt(np.einsum("ia,ab,ib->i", S, G, S))
    return np.array([sig * L[i] * L[j] for i, j in SL])


def _brute_rel(SB, G, sig):
    import numpy as np
    P = -np.einsum("nia,ab,njb->nij", SB, G, SB)
    q = np.stack([P[:, i, j] for i, j in SL], 1)
    L = np.sqrt(np.einsum("nia,ab,nib->ni", SB, G, SB))
    band = np.stack([sig * L[:, i] * L[:, j] for i, j in SL], 1)
    keep = (q >= -band - 1e-9 * np.abs(G).max()).all(1)
    return {_key(S) for S in SB[keep]}


def _construct_rel(G, sig, factor=4.0):
    """returns (set, 'ok') or (None, 'hypothesis') when the idealised lattice
    is singular, i.e. a conorm comparable to lambda_1 was zeroed."""
    import numpy as np
    s0 = np.array(S0, float)
    p0 = _cn(s0, G)
    ps = np.where(p0 <= factor * _band(s0, G, sig), 0.0, p0)
    Gs = _gram(ps)
    if np.linalg.eigvalsh(Gs).min() <= 1e-9 * np.abs(G).max():
        return None, "hypothesis"           # singular idealised lattice
    lam1 = min(float(s @ Gs @ s) for s in s0)   # shortest of the four superbase vectors
    zeroed = ps <= 0.0
    status = "ok"
    if zeroed.any() and 32 * _band(s0, G, sig)[zeroed].max() >= lam1:
        status = "hypothesis"               # per-slot form of lambda_1 > 32 tau: the
                                            # bands on the idealised slots
    tol = 1e-9 * np.abs(G).max()
    seen, front = {_key(s0): s0}, [s0]
    while front:
        S = front.pop()
        p = _cn(S, Gs)
        for n, (i, j) in enumerate(SL):
            if abs(p[n]) < tol:
                U = _flip(S, i, j)
                k = _key(U)
                if k not in seen and _cn(U, Gs).min() >= -tol:
                    seen[k] = U
                    front.append(U)
    return {k for k, S in seen.items() if (_cn(S, G) >= -_band(S, G, sig) - tol).all()}, status


def _nice_cell(S, G):
    """the member's cell with all angles >= 90, most right angles first."""
    import numpy as np
    best = None
    for pm in itertools.permutations(range(4)):
        for sg in (1, -1):
            B = (sg * S[list(pm)])[1:4]
            Gm = B @ G @ B.T
            a, b, c = np.sqrt(np.diag(Gm))
            ang = np.degrees(np.arccos([Gm[1, 2] / (b * c), Gm[0, 2] / (a * c), Gm[0, 1] / (a * b)]))
            if (ang < 90 - 1e-6).any():
                continue
            n90 = int((abs(ang - 90) < 1e-6).sum())
            key = (-n90, round(a, 3), round(b, 3), round(c, 3))
            if best is None or key < best[0]:
                best = (key, (a, b, c, *ang), n90)
    return best[1], best[2]


@claim("z49", "SI Table S1", kind="exhaustive")
def test_classical_ambiguities_are_cell_shapes():
    r"""SI Table S1: the classical ambiguities of each type are its cell shapes, with the stated right angles per shape (oP 3,2,2,2; tP 3,2,2; cP 3,2; oC 2,2; mP 2,2,2)."""
    import numpy as np
    expected = {"C5": [3, 2], "T5": [3, 2, 2], "O5": [3, 2, 2, 2], "O4": [2, 2],
                "M4": [2, 2, 2], "O3": [1, 1], "M2B": [1, 1], "M3": [1, 1],
                "A2": [1, 1], "A3": [1, 1, 1]}
    for name, bt, pat in SORTS:
        if name not in expected:
            continue
        letters = sorted(set(pat) - {"0"})
        vals = dict(zip(letters, [900., 1600., 2500., 400., 1200., 700.][:len(letters)]))
        s = [0.0 if ch == "0" else vals[ch] for ch in pat]
        p = np.array([s[3], s[4], s[5], s[2], s[1], s[0]])
        G = _gram(p)
        mem, orbit_of, shape_of, _ = _orbits_and_shapes(p, G)
        shapes = {}
        for k, S in mem.items():
            shapes.setdefault(shape_of[k], S)
        n90 = sorted((_nice_cell(S, G)[1] for S in shapes.values()), reverse=True)
        assert n90 == expected[name], (name, n90)
    # the primitive orthorhombic 30 x 40 x 55 cell: its three monoclinic-looking shapes
    p = np.array([900., 1600., 3025., 0., 0., 0.])
    G = _gram(p)
    mem, orbit_of, shape_of, _ = _orbits_and_shapes(p, G)
    cells = sorted({tuple(np.round(_nice_cell(S, G)[0], 1)) for S in mem.values()})
    assert cells == [(30.0, 40.0, 55.0, 90.0, 90.0, 90.0), (30.0, 40.0, 62.6, 90.0, 118.6, 90.0),
                     (30.0, 50.0, 55.0, 90.0, 90.0, 126.9), (40.0, 50.0, 55.0, 90.0, 90.0, 143.1)], cells
