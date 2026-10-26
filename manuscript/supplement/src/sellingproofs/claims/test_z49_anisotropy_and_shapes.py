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


@claim("z49", "SI §S4, Bands per slot", kind="sampled")
def test_per_slot_bands_on_needles_and_plates(superbases):
    r"""On needles and plates with angular errors, a band per slot reproduces the brute-force widened set."""
    import numpy as np
    from agentsg.cell.canonical import canonical_superbase
    from agentsg.cell.metric import UnitCell, params_from_metric
    rnd = np.random.default_rng(11)
    cells = {"needle mP": (5, 5, 100, 90, 92, 90), "needle oP": (5, 5, 100, 90, 90, 90),
             "needle hP": (40, 40, 300, 90, 90, 120), "plate oP": (100, 100, 5, 90, 90, 90)}
    tally = {}
    for name, c in cells.items():
        G0 = np.array(UnitCell(*c).metric_tensor())
        for sdeg in (0.02, 0.05, 0.2, 0.5):
            sig = np.radians(sdeg)
            for _ in range(4):
                E = rnd.normal(size=(3, 3)); E = (E + E.T) / 2; np.fill_diagonal(E, 0)
                L = np.sqrt(np.diag(G0))
                G = G0 + np.outer(L, L) * E * sig * 0.7
                cell = tuple(params_from_metric(G))
                F0 = np.array(canonical_superbase(cell)[0], float)
                Gf = F0[1:4] @ G @ F0[1:4].T
                truth = _brute_rel(superbases, Gf, sig)
                got, status = _construct_rel(Gf, sig)
                key = (name, sdeg)
                t_ = tally.setdefault(key, {"ok": 0, "wrong": 0, "beyond ok": 0,
                                            "beyond wrong": 0, "singular": 0, "max": 0})
                t_["max"] = max(t_["max"], len(truth))
                if got is None:
                    t_["singular"] += 1
                elif status == "ok":
                    t_["ok" if got == truth else "wrong"] += 1
                else:
                    t_["beyond ok" if got == truth else "beyond wrong"] += 1
    # within the per-slot hypothesis the construction is exact, every cell and precision
    assert all(v["wrong"] == 0 for v in tally.values()), tally
    # needles: hypothesis holds and exact through 0.05 deg; beyond it (0.2,
    # 0.5 deg) the set is still exact whenever the idealised lattice exists --
    # the constant 32 is conservative
    for name in ("needle mP", "needle oP", "needle hP"):
        assert tally[(name, 0.02)]["ok"] == 4 and tally[(name, 0.05)]["ok"] == 4, name
        for sdeg in (0.2, 0.5):
            assert tally[(name, sdeg)]["beyond wrong"] == 0, (name, sdeg)
    # plate: the floor -- at 0.2 and 0.5 deg the hypothesis fails, the widened
    # set exceeds sixteen members and the sixteen-member closure misses some
    for sdeg in (0.2, 0.5):
        t_ = tally[("plate oP", sdeg)]
        assert t_["ok"] == 0 and t_["beyond ok"] == 0 and t_["max"] > 16, t_
    # (exact at 0.02 and 0.05 deg, mostly already beyond the hypothesis)
    for sdeg in (0.02, 0.05):
        t_ = tally[("plate oP", sdeg)]
        assert t_["ok"] + t_["beyond ok"] == 4 and t_["wrong"] + t_["beyond wrong"] == 0, t_


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
