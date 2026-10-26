"""Generate the 'conditions for trouble' benchmark: measured cells constructed
from the places where the Selling closure says two correct descriptions of
one lattice can disagree.

Families
  shapes          a lattice of a sort with several cell shapes, measured in a
                  shape other than the reference's (the three monoclinic cell
                  choices, the monoclinic-looking cells of oP/tP/cP, ...)
  specialisation  a lattice of sort X a distance delta from a codimension-one
                  condition that makes it sort Y (Table 'special' of the
                  paper), delta from 0.5 to 8 times the metric error
  hidden          a cell at a three-wall corner with two small conorms in the
                  strip z <= tau < y <= z + tau: a member of the widened
                  closure reachable only through a non-member
  anisotropy      needles and plates with an angular error, inside and beyond
                  the hypothesis lambda_1 > 32 tau
  boundary        the three boundary sorts O3, M2B, M3 and their neighbours

Every case carries the exact reference lattice, the measured cell (reference
plus a metric error of Frobenius norm eta in the reference frame, then
reindexed by a random unimodular matrix), the true operator, eta, the
suggested band tau = 3 eta, lambda_1, whether lambda_1 > 32 tau, and the
expected answers: the true Delone sort, the nearby sort and its distance
delta where there is one, and the number of members of the widened closure
at tau by brute force.

    python make_benchmark.py            -> benchmark.jsonl (seed 2026)
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SELLINGPROOFS_OUT", HERE)          # output directory (default: here)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from sellingproofs.claims import test_z44_widened_complete as t  # noqa: E402
from sellingproofs.claims.test_z46_selling_groupoid import SORTS, _orbits_and_shapes  # noqa: E402
from sellingproofs.claims.test_z50_specialisations import (  # noqa: E402
    EXPECTED as SPECIAL, _constraints, _member_maps, _param_matrix, _positive_nullspace_point,
)

PERM4 = list(itertools.permutations(range(4)))
SEED = 2026
SCALE = 900.0           # parameters are scaled so that cells are tens of Angstrom


def _superbases():
    r = np.arange(-2, 3)
    M = np.array(np.meshgrid(*[r] * 9, indexing="ij")).reshape(9, -1).T.reshape(-1, 3, 3)
    B = M[np.abs(np.round(np.linalg.det(M))) == 1].astype(float)
    return np.concatenate([-(B.sum(1))[:, None, :], B], 1)


SB = None


def brute_count(G, tau):
    global SB
    if SB is None:
        SB = _superbases()
    P = -np.einsum("nia,ab,njb->nij", SB, G, SB)
    q = np.stack([P[:, i, j] for i, j in t.SL], 1)
    keep = (q >= -tau - 1e-9 * np.abs(G).max()).all(1)
    keys = {t._key(S) for S in SB[keep]}
    return len(keys)


def cell_of(G):
    a, b, c = np.sqrt(np.diag(G))
    ang = np.degrees(np.arccos([G[1, 2] / (b * c), G[0, 2] / (a * c), G[0, 1] / (a * b)]))
    return [float(x) for x in (a, b, c, *ang)]


def random_unimodular(rnd):
    while True:
        M = rnd.integers(-2, 3, (3, 3)).astype(float)
        if abs(round(np.linalg.det(M))) == 1:
            return M


def random_error(rnd, eta):
    E = rnd.normal(size=(3, 3)); E = (E + E.T) / 2
    return E * (eta / np.linalg.norm(E))


def lam1_of(p):
    G = t._gram(p)
    return min(float(s @ G @ s) for s in np.array(t.S0, float))


def make_case(rnd, family, sort, p, eta, extra, cid, member_basis=None, E=None, GN0=None, tau=None):
    """p: reduced conorms of the reference (slots da,db,dc,ab,ac,bc).
    member_basis: 3x3 integer rows = a member's basis in reference coordinates,
    so the measured cell is that member's cell (family 'shapes')."""
    GR = t._gram(p)
    B = np.eye(3) if member_basis is None else np.array(member_basis, float)
    M = np.round(B.T @ random_unimodular(rnd))      # columns: the measured basis in reference coordinates
    if GN0 is None:
        E = random_error(rnd, eta) if E is None else E
        GN0 = GR + E
    eta = float(np.linalg.norm(GN0 - GR))
    GN = M.T @ GN0 @ M
    lam1 = lam1_of(p)
    tau = 3 * eta if tau is None else tau
    bt = {n: b for n, b, _ in SORTS}[sort]
    from agentsg.cell.reduction import niggli_gk
    nig, Mn = niggli_gk(tuple(cell_of(GN)))
    case = {
        "id": cid, "family": family, "sort": sort, "bravais": bt,
        "reference_cell": cell_of(GR), "reference_conorms": [float(x) for x in p],
        "measured_cell": cell_of(GN), "true_operator": M.astype(int).tolist(),
        "measured_niggli_cell": [float(x) for x in nig],
        "niggli_from_measured": np.array(Mn).astype(int).tolist(),
        "eta": float(eta), "tau": float(tau), "lambda1": float(lam1),
        "hypothesis_ok": bool(lam1 > 32 * tau),
        "expected": {"widened_members": brute_count(GR, tau), **extra},
    }
    return case


def generic_params(pat, rnd):
    letters = [ch for ch in "rstuvw" if ch in pat]
    return np.sort(rnd.uniform(0.3, 3.0, len(letters)))[::-1] * SCALE, letters


def family_shapes(rnd, cases):
    etas = [1.0, 3.0, 10.0]
    for name, bt, pat in SORTS:
        A, letters = _param_matrix(pat)
        theta, _ = generic_params(pat, rnd)
        p = A @ theta
        G = t._gram(p)
        mem, orbit_of, shape_of, _ = _orbits_and_shapes(p, G)
        shapes = {}
        for k, S in mem.items():
            shapes.setdefault(shape_of[k], S)
        if len(shapes) < 2:
            continue
        for si, (sh, S) in enumerate(shapes.items()):
            for eta in etas:
                cases.append(make_case(rnd, "shapes", name, p, eta,
                                       {"true_sort": name, "shape_index": si, "shapes": len(shapes)},
                                       f"shapes-{name}-{si}-{eta:g}", member_basis=S[1:4]))


def family_specialisation(rnd, cases):
    for (x, y), (conds, nshapes) in sorted(SPECIAL.items()):
        A, letters = _param_matrix(next(pat for n, b, pat in SORTS if n == x))
        k = len(letters)
        for cond in sorted(conds):
            # condition row
            row = np.zeros(k)
            lhs = cond.replace(" = 0", "")
            if "-" in lhs:
                a_, b_ = lhs.split(" - ")
                row[letters.index(a_)] = 1; row[letters.index(b_)] = -1
            else:
                row[letters.index(lhs)] = 1
            x0 = _positive_nullspace_point(row[None, :], k, rnd)
            if x0 is None:
                continue
            x0 = x0 / x0.max() * SCALE * 2
            r = row / np.linalg.norm(row)
            for eta in (1.0, 3.0):
                for mult in (0.5, 1.0, 2.0, 4.0, 8.0):
                    delta = mult * eta
                    theta = x0 + delta * r
                    if (theta < 0).any():
                        theta = x0 - delta * r
                    if (theta < -1e-9).any():
                        continue
                    theta = np.maximum(theta, 0.0)
                    p = A @ theta
                    if np.linalg.eigvalsh(t._gram(p)).min() <= 1e-6 * p.max():
                        continue
                    cases.append(make_case(rnd, "specialisation", x, p, eta,
                                           {"true_sort": x, "near_sort": y, "condition": cond, "delta": float(delta)},
                                           f"special-{x}-{y}-{cond.replace(' ', '')}-{eta:g}-{mult:g}"))


def family_hidden(rnd, cases):
    """the measured cell is constructed so that its only reduced superbase is
    the hidden member of the reference's widened closure (as in test_z46)."""
    for name, base in (("O5", [900.0, 1600.0, 2500.0, 0, 0, 0]), ("C5", [900.0, 900.0, 900.0, 0, 0, 0])):
        base = np.array(base, float)
        lam = lam1_of(base)
        for frac in (0.1, 0.5):
            tau = frac * lam / 32
            for (y, z) in ((0.5, 1.2), (0.25, 1.1), (0.75, 1.5), (1.0, 1.9), (0.5, 1.4)):
                p = base.copy()
                p[3], p[4], p[5] = y * tau, z * tau, 0.0
                GR = t._gram(p)
                thm = [S for S in t._generate(p, 4 * tau, tau)[1].values() if t._cn(S, GR).min() >= -tau - 1e-9]
                known = {t._key(S) for S in t._generate(p, tau, tau)[1].values()}
                hidden = [S for S in thm if t._key(S) not in known]
                assert hidden, (name, y, z)
                for hi, H in enumerate(hidden):
                    q = t._cn(H, GR)
                    Binv = np.linalg.inv(H[1:4].T)
                    GN0 = Binv.T @ t._gram(np.where(q <= 0, 0.25 * tau, q)) @ Binv
                    cases.append(make_case(rnd, "hidden", name, p, 0.0,
                                           {"true_sort": "A2", "near_sort": name, "hidden_member": True,
                                            "walls_in_tau": [y, z]},
                                           f"hidden-{name}-{frac:g}-{y:g}-{z:g}-{hi}", GN0=GN0, tau=tau))


def family_anisotropy(rnd, cases):
    from agentsg.cell.canonical import canonical_superbase
    from agentsg.cell.metric import UnitCell
    cells = {"needle-mP": ((5, 5, 100, 90, 92, 90), "M4"), "needle-oP": ((5, 5, 100, 90, 90, 90), "O5"),
             "plate-oP": ((100, 100, 5, 90, 90, 90), "O5"), "needle-hP": ((40, 40, 300, 90, 90, 120), "H4")}
    for name, (c, sort) in cells.items():
        G0 = np.array(UnitCell(*c).metric_tensor())
        F0 = np.array(canonical_superbase(tuple(float(v) for v in c))[0], float)
        Gf = F0[1:4] @ G0 @ F0[1:4].T
        p = t._cn(np.array(t.S0, float), Gf)
        L = np.sqrt(np.diag(G0))
        for sdeg in (0.02, 0.05, 0.2, 0.5):
            sig = np.radians(sdeg)
            Lf = np.sqrt(np.diag(Gf))
            Z = rnd.normal(size=(3, 3)); Z = (Z + Z.T) / 2; np.fill_diagonal(Z, 0)
            E = np.outer(Lf, Lf) * Z * sig * 0.7                   # angular error model, reduced frame
            cases.append(make_case(rnd, "anisotropy", sort, p, 0.0,
                                   {"true_sort": sort, "angular_error_deg": sdeg, "cell_name": name,
                                    "per_slot_sigma_rad": float(sig)},
                                   f"aniso-{name}-{sdeg:g}", E=E))


def family_boundary(rnd, cases):
    for name in ("O3", "M2B", "M3"):
        A, letters = _param_matrix(next(pat for n, b, pat in SORTS if n == name))
        theta, _ = generic_params("".join(letters), rnd)
        p = A @ theta
        for eta in (1.0, 3.0, 10.0):
            cases.append(make_case(rnd, "boundary", name, p, eta, {"true_sort": name}, f"boundary-{name}-{eta:g}"))


def main(path=None):
    path = path or os.path.join(OUT, "benchmark.jsonl")
    rnd = np.random.default_rng(SEED)
    cases = []
    family_shapes(rnd, cases)
    family_specialisation(rnd, cases)
    family_hidden(rnd, cases)
    family_anisotropy(rnd, cases)
    family_boundary(rnd, cases)
    with open(path, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    fam = {}
    for c in cases:
        fam[c["family"]] = fam.get(c["family"], 0) + 1
    print(len(cases), "cases", fam)


if __name__ == "__main__":
    main()
