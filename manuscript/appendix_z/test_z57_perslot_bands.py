"""z:per-slot — the per-slot construction of Appendix B is not exact under its
stated condition; the frame-consistent idealisation is.

Per-slot bands are frame-covariant: a superbase F with vectors u_i has band
tau_ij(F) = sigma |u_i||u_j| on each slot, and the widened set is
S_sigma(G) = {F : p_ij(F;G) >= -sigma |u_i||u_j| for all ij}, the superbases
whose pairwise angles are all at least 90 degrees minus arcsin(sigma).

Construction B (Appendix B, test_z49): idealise the slots of F0 with
p0_kl <= 4 sigma l_k l_l (l_k the lengths of F0's own vectors), enumerate the
closure, filter per slot; condition lambda_1 > 32 sigma max_Z l_k l_l.

Counterexample. The needle a=5.58, b=5.21, c=131.97 A, gamma=91.12 deg
(p_ab = 0.568 A^2, p_ac = p_bc = 0) at sigma with lambda_1 / (sigma l_a l_c)
= 33 or 40: construction B returns 6 superbases, the brute-force widened set
has 12. The slot ab survives idealisation because its own band sigma l_a l_b
is tiny, but the same dot product a.b appears in the member slot (a+c, b),
whose band sigma |a+c||b| ~ 25 times larger covers it; the members built on
a+c are delta-obtuse for G and are missed. Exactness returns only when
sigma |a+c||b| < p_ab (lambda_1 / (sigma l_a l_c) ~ 45).

Consistent idealisation (fixpoint). Start from Z_0 as in B. For every member
of the current closure and every slot of it that is surviving (positive for
Lambda*) but at most four times that member's own band, add to Z the
surviving F0 slots that feed that conorm and are themselves at most four
times that band; repeat until nothing is added (Z only grows, so it stops in
at most six rounds); refuse when Lambda* becomes singular; filter per slot.
On the needle this adds ab and returns 12 of 12. On 240 random cells
(needles, plates, isotropic, monoclinic; 0.1-3 deg) it is never wrong:
exact in 124, refused in 116; construction B is wrong in 9 of the 240.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.zcheck]

import test_z44_widened_complete as t  # noqa: E402
from test_z49_anisotropy_and_shapes import _construct_rel  # noqa: E402

SL = t.SL
NEEDLE = np.array([30.606, 26.579, 17414.674, 0.568, 0.0, 0.0])   # conorms da,db,dc,ab,ac,bc


@pytest.fixture(scope="module")
def superbases():
    r = np.arange(-2, 3)
    M = np.array(np.meshgrid(*[r] * 9, indexing="ij")).reshape(9, -1).T.reshape(-1, 3, 3)
    B = M[np.abs(np.round(np.linalg.det(M))) == 1].astype(float)
    return np.concatenate([-(B.sum(1))[:, None, :], B], 1)


def brute(SB, G, sigma):
    P = -np.einsum("nia,ab,njb->nij", SB, G, SB)
    q = np.stack([P[:, i, j] for i, j in SL], 1)
    L = np.sqrt(np.einsum("nia,ab,nib->ni", SB, G, SB))
    band = np.stack([sigma * L[:, i] * L[:, j] for i, j in SL], 1)
    keep = (q >= -band - 1e-9 * np.abs(G).max()).all(1)
    return {t._key(S) for S in SB[keep]}


def closure_members(Gs, tol):
    s0 = np.array(t.S0, float)
    seen, front = {t._key(s0): s0}, [s0]
    while front:
        S = front.pop()
        p = t._cn(S, Gs)
        for n, (i, j) in enumerate(SL):
            if abs(p[n]) < tol:
                U = t._flip(S, i, j)
                k = t._key(U)
                if k not in seen and t._cn(U, Gs).min() >= -tol:
                    seen[k] = U
                    front.append(U)
    return seen


def consistent_construct(G, sigma):
    """(set or None, Z, status, rounds): the frame-consistent per-slot construction."""
    s0 = np.array(t.S0, float)
    p0 = t._cn(s0, G)
    tol = 1e-9 * np.abs(G).max()
    L0 = np.sqrt(np.einsum("ia,ab,ib->i", s0, G, s0))
    Z = {n for n, (i, j) in enumerate(SL) if p0[n] <= 4 * sigma * L0[i] * L0[j]}
    rounds = 0
    while True:
        rounds += 1
        ps = np.where([n in Z for n in range(6)], 0.0, p0)
        Gs = t._gram(ps)
        if np.linalg.eigvalsh(Gs).min() <= tol:
            return None, Z, "singular", rounds
        mem = closure_members(Gs, tol)
        added = set()
        for S in mem.values():
            L = np.sqrt(np.einsum("ia,ab,ib->i", S, G, S))
            pL = t._cn(S, Gs)
            for n, (i, j) in enumerate(SL):
                band = sigma * L[i] * L[j]
                if not (tol < pL[n] <= 4 * band):
                    continue
                cu, cv = np.array([0.0, *S[i]]), np.array([0.0, *S[j]])   # superbase coordinates in F0
                for m, (k, l) in enumerate(SL):
                    if m not in Z and (cu[k] - cu[l]) * (cv[k] - cv[l]) != 0 and p0[m] <= 4 * band:
                        added.add(m)
        if not added:
            break
        Z |= added
    bands0 = np.array([sigma * L0[i] * L0[j] for i, j in SL])
    status = "ok" if (not Z or t._lam1(p0) > 32 * max(bands0[n] for n in Z)) else "hypothesis"
    out = set()
    for k, S in mem.items():
        L = np.sqrt(np.einsum("ia,ab,ib->i", S, G, S))
        band = np.array([sigma * L[i] * L[j] for i, j in SL])
        if (t._cn(S, G) >= -band - tol).all():
            out.add(k)
    return out, Z, status, rounds


def closure_construct(G, sigma):
    """(set or None, Z): idealise while enumerating the closure.

    Flip at the zeros of the current idealised lattice; at every member reached,
    any slot with 0 < p_ij(member; Lambda*) <= 4 sigma |u_i||u_j| (the member's own
    band) zeroes, at once, the surviving F0 slots that feed it and are below the
    same threshold; Lambda* is updated and every member found so far is queued
    again, since it may now have new zeros to flip at. By Lemma A8 a member found
    under fewer zeros stays a member. Refuse when Lambda* becomes singular.
    Filter per slot under the measured metric at the end."""
    s0 = np.array(t.S0, float)
    p0 = t._cn(s0, G)
    tol = 1e-9 * np.abs(G).max()
    L0 = np.sqrt(np.einsum("ia,ab,ib->i", s0, G, s0))
    Z = {n for n, (i, j) in enumerate(SL) if p0[n] <= 4 * sigma * L0[i] * L0[j]}

    def ideal(Z):
        Gs = t._gram(np.where([n in Z for n in range(6)], 0.0, p0))
        return None if np.linalg.eigvalsh(Gs).min() <= tol else Gs

    Gs = ideal(Z)
    if Gs is None:
        return None, Z
    seen, front = {t._key(s0): s0}, [s0]
    while front:
        S = front.pop()
        p = t._cn(S, Gs)
        L = np.sqrt(np.einsum("ia,ab,ib->i", S, G, S))
        added = set()
        for n, (i, j) in enumerate(SL):
            band = sigma * L[i] * L[j]
            if tol < p[n] <= 4 * band:
                cu, cv = np.array([0.0, *S[i]]), np.array([0.0, *S[j]])
                for m, (k, l) in enumerate(SL):
                    if m not in Z and (cu[k] - cu[l]) * (cv[k] - cv[l]) != 0 and p0[m] <= 4 * band:
                        added.add(m)
        if added:
            Z |= added
            Gs = ideal(Z)
            if Gs is None:
                return None, Z
            front = list(seen.values())            # re-examine every member under the new zeros
            continue
        for n, (i, j) in enumerate(SL):
            if abs(p[n]) < tol:
                U = t._flip(S, i, j)
                k = t._key(U)
                if k not in seen and t._cn(U, Gs).min() >= -tol:
                    seen[k] = U
                    front.append(U)
    out = set()
    for k, S in seen.items():
        L = np.sqrt(np.einsum("ia,ab,ib->i", S, G, S))
        band = np.array([sigma * L[i] * L[j] for i, j in SL])
        if (t._cn(S, G) >= -band - tol).all():
            out.add(k)
    return out, Z


def zero_filtration(G):
    """The zeroing as a filtration on the closure, in cosine units t.

    At level t a slot of F0 is zero when p0_kl <= t l_k l_l, and a member slot with
    0 < p_ij(member; Lambda*) <= t |u_i||u_j| zeroes the surviving F0 slots that feed
    it and lie below the same level. Events are processed in increasing t, so the
    sequence is canonical. Returns (levels, limit): levels is a list of
    (t, Z, members) at each event, members the closure of Lambda* at that level as
    {key: superbase}; limit is the level at which Lambda* would become singular
    (the geometric limit), or inf when every level is non-singular."""
    s0 = np.array(t.S0, float)
    p0 = t._cn(s0, G)
    tol = 1e-9 * np.abs(G).max()

    def ideal(Z):
        Gs = t._gram(np.where([n in Z for n in range(6)], 0.0, p0))
        return None if np.linalg.eigvalsh(Gs).min() <= tol else Gs

    def candidates(Z, mem, Gs):
        """(level, slots to zero) for every pending event."""
        ev = []
        L0 = np.sqrt(np.einsum("ia,ab,ib->i", s0, G, s0))
        for m, (k, l) in enumerate(SL):
            if m not in Z:
                ev.append((p0[m] / (L0[k] * L0[l]), {m}))
        for S in mem.values():
            p = t._cn(S, Gs)
            L = np.sqrt(np.einsum("ia,ab,ib->i", S, G, S))
            for n, (i, j) in enumerate(SL):
                if p[n] > tol:
                    lev = p[n] / (L[i] * L[j])
                    cu, cv = np.array([0.0, *S[i]]), np.array([0.0, *S[j]])
                    feed = {m for m, (k, l) in enumerate(SL)
                            if m not in Z and (cu[k] - cu[l]) * (cv[k] - cv[l]) != 0
                            and p0[m] <= lev * L[i] * L[j] + tol}
                    if feed:
                        ev.append((lev, feed))
        return ev

    Z = {m for m in range(6) if p0[m] <= tol}
    Gs = ideal(Z)
    mem = closure_members(Gs, tol)
    levels = [(0.0, set(Z), dict(mem))]
    while True:
        ev = candidates(Z, mem, Gs)
        if not ev:
            return levels, np.inf
        lev = min(e[0] for e in ev)
        add = set().union(*(e[1] for e in ev if e[0] <= lev * (1 + 1e-12) + 1e-15))
        Z2 = Z | add
        Gs2 = ideal(Z2)
        if Gs2 is None:
            return levels, lev
        Z, Gs = Z2, Gs2
        mem = closure_members(Gs, tol)
        levels.append((lev, set(Z), dict(mem)))


def filtration_select(levels, G, sigma, t_stop):
    """members enumerated at the last level <= t_stop, filtered per slot at sigma."""
    tol = 1e-9 * np.abs(G).max()
    mem = [m for (lev, Z, m) in levels if lev <= t_stop][-1]
    out = set()
    for k, S in mem.items():
        L = np.sqrt(np.einsum("ia,ab,ib->i", S, G, S))
        band = np.array([sigma * L[i] * L[j] for i, j in SL])
        if (t._cn(S, G) >= -band - tol).all():
            out.add(k)
    return out


def test_needle_counterexample_to_construction_B(superbases):
    G = t._gram(NEEDLE)
    s0 = np.array(t.S0, float)
    L0 = np.sqrt(np.einsum("ia,ab,ib->i", s0, G, s0))
    lam1 = t._lam1(NEEDLE)
    for Cstar, exact_B in ((33, False), (40, False), (60, True)):
        sigma = lam1 / (Cstar * L0[1] * L0[3])            # lambda_1 / (sigma l_a l_c) = C*, Z_0 = {ac, bc}
        true = brute(superbases, G, sigma)
        got, status = _construct_rel(G, sigma)
        assert status == "ok"                              # the stated condition holds for C* >= 32
        assert (got == true) is exact_B, (Cstar, len(got), len(true))
        if not exact_B:
            assert len(got) == 6 and len(true) == 12
        out, Z, _, rounds = consistent_construct(G, sigma)
        assert out == true and rounds == 2 and 3 in Z      # the fixpoint adds ab and is exact


def test_consistent_construction_is_never_wrong(superbases):
    from agentsg.cell.metric import metric_tensor
    from agentsg.cell.selling_cob import selling_matrix
    rng = np.random.default_rng(1)
    n_exact = n_refused = n_wrong_B = 0
    for _ in range(60):
        kind = rng.choice(["needle", "plate", "iso", "mono"])
        if kind == "needle":
            a, b, c = 5 * rng.uniform(0.8, 1.2), 5 * rng.uniform(0.8, 1.2), rng.uniform(20, 150)
        elif kind == "plate":
            a, b, c = rng.uniform(40, 150), rng.uniform(40, 150), 5 * rng.uniform(0.8, 1.2)
        elif kind == "iso":
            a, b, c = rng.uniform(5, 50, 3)
        else:
            a, b, c = rng.uniform(5, 30), rng.uniform(5, 30), rng.uniform(5, 60)
        al, be, ga = 90 + rng.normal(0, 3, 3) * rng.choice([0, 1], 3)
        cell = (a, b, c, al, be, ga)
        Gc = np.array(metric_tensor(cell), float)
        P = np.array([[float(x) for x in row] for row in selling_matrix(cell).rows], float)
        p0 = t._cn(np.array(t.S0, float), P.T @ Gc @ P)
        if p0.min() < -1e-9 * abs(Gc).max():
            continue
        G = t._gram(np.maximum(p0, 0))
        for deg in (0.1, 0.5, 1.0, 3.0):
            sigma = np.sin(np.radians(deg))
            true = brute(superbases, G, sigma)
            out, _, status, _ = consistent_construct(G, sigma)
            if out is None:
                n_refused += 1
            else:
                assert out == true, (kind, deg, cell)
                n_exact += 1
            if status == "ok":
                assert out is not None and out == true
            got, _ = _construct_rel(G, sigma)
            n_wrong_B += got is not None and got != true
    assert (n_exact, n_refused, n_wrong_B) == (124, 116, 9)


def delta_obtuse(G, sigma, K):
    """every superbase with basis coordinates in [-K, K] whose six conorms are at least
    -sigma |u_i||u_j| (built from pairs of admissible vectors, completed by the fourth)."""
    import itertools
    r = np.arange(-K, K + 1)
    R = np.array([v for v in itertools.product(r, repeat=3) if any(v)], float)
    n = np.sqrt(np.einsum("ia,ab,ib->i", R, G, R))
    ok = -(R @ G @ R.T) >= -sigma * np.outer(n, n) - 1e-9 * abs(G).max()
    idx = {tuple(v.astype(int)): i for i, v in enumerate(R)}
    found, maxent = set(), 0
    for a in range(len(R)):
        bs = np.where(ok[a])[0]
        bs = bs[bs > a]
        for b in bs:
            for c in bs[(bs > b) & ok[b, bs]]:
                d = -(R[a] + R[b] + R[c])
                j = idx.get(tuple(d.astype(int)))
                if j is None or not (ok[j, a] and ok[j, b] and ok[j, c]):
                    continue
                Bm = np.array([R[a], R[b], R[c]])
                if abs(round(np.linalg.det(Bm))) != 1:
                    continue
                S = np.vstack([d, Bm])
                found.add(t._key(S))
                maxent = max(maxent, int(abs(S).max()))
    return found, maxent


def test_window_beyond_the_hypothesis():
    """a 7.73 x 47.54 x 77.30 A cell, beta = 87.0 deg, at a 3 deg idealisation (sigma = sin 3/4):
    the idealisation is refused, and the widened set needs a basis coordinate of 3."""
    p0 = np.array([0.0, 28.4690554, 5943.77198, 2260.08703, 0.0, 31.2814792])
    G = t._gram(p0)
    sigma = np.sin(np.radians(3.0)) / 4
    out, _, _, _ = consistent_construct(G, sigma)
    assert out is None
    f2, _ = delta_obtuse(G, sigma, 2)
    f3, m3 = delta_obtuse(G, sigma, 3)
    assert len(f2) == 11 and len(f3) == 12 and m3 == 3


def test_three_degree_idealisation_by_aspect_ratio(superbases):
    """sigma = sin(3 deg)/4 (idealise at 3 deg, filter at about 0.75 deg) on 300 random cells:
    exact up to 3:1, refused on 25 of 56 cells at 5:1 and 37 of 41 at 10:1, never wrong."""
    from agentsg.cell.metric import metric_tensor
    from agentsg.cell.selling_cob import selling_matrix
    import collections
    sigma = np.sin(np.radians(3.0)) / 4
    rng = np.random.default_rng(7)
    tally = collections.Counter()
    for _ in range(300):
        ratio = rng.choice([1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
        s = rng.uniform(5, 15)
        dims = np.array([s, s * rng.uniform(1, ratio), s * ratio])
        rng.shuffle(dims)
        ang = 90 + rng.uniform(-4, 4, 3) * rng.choice([0, 1], 3, p=[0.6, 0.4])
        cell = (*dims, *ang)
        Gc = np.array(metric_tensor(cell), float)
        P = np.array([[float(x) for x in row] for row in selling_matrix(cell).rows], float)
        p0 = t._cn(np.array(t.S0, float), P.T @ Gc @ P)
        if p0.min() < -1e-9 * abs(Gc).max():
            continue
        G = t._gram(np.maximum(p0, 0))
        out, _, _, _ = consistent_construct(G, sigma)
        res = "refused" if out is None else ("exact" if out == brute(superbases, G, sigma) else "wrong")
        tally[(float(ratio), res)] += 1
    assert tally == {(1.0, "exact"): 57, (1.5, "exact"): 51, (2.0, "exact"): 48, (3.0, "exact"): 47,
                     (5.0, "exact"): 31, (5.0, "refused"): 25, (10.0, "exact"): 4, (10.0, "refused"): 37}


def _random_cells():
    from agentsg.cell.metric import metric_tensor
    from agentsg.cell.selling_cob import selling_matrix
    rng = np.random.default_rng(1)
    for _ in range(60):
        kind = rng.choice(["needle", "plate", "iso", "mono"])
        if kind == "needle":
            a, b, c = 5 * rng.uniform(0.8, 1.2), 5 * rng.uniform(0.8, 1.2), rng.uniform(20, 150)
        elif kind == "plate":
            a, b, c = rng.uniform(40, 150), rng.uniform(40, 150), 5 * rng.uniform(0.8, 1.2)
        elif kind == "iso":
            a, b, c = rng.uniform(5, 50, 3)
        else:
            a, b, c = rng.uniform(5, 30), rng.uniform(5, 30), rng.uniform(5, 60)
        al, be, ga = 90 + rng.normal(0, 3, 3) * rng.choice([0, 1], 3)
        cell = (a, b, c, al, be, ga)
        Gc = np.array(metric_tensor(cell), float)
        P = np.array([[float(x) for x in row] for row in selling_matrix(cell).rows], float)
        p0 = t._cn(np.array(t.S0, float), P.T @ Gc @ P)
        if p0.min() < -1e-9 * abs(Gc).max():
            continue
        yield t._gram(np.maximum(p0, 0))


def test_zeroing_on_the_closure_and_its_filtration(superbases):
    """Zeroing while enumerating the closure gives the same sets as the rounds; as a
    filtration in cosine units it is canonical, never wrong when stopped at 4 sigma,
    and one enumeration up to the geometric limit answers every band 4 sigma below it."""
    import collections
    tally = collections.Counter()
    for G in _random_cells():
        levels, limit = zero_filtration(G)
        assert np.isfinite(limit)
        for deg in (0.1, 0.5, 1.0, 3.0):
            sigma = np.sin(np.radians(deg))
            truth = brute(superbases, G, sigma)
            rounds, *_ = consistent_construct(G, sigma)
            onfly, _ = closure_construct(G, sigma)
            assert (rounds is None) == (onfly is None) and (rounds is None or rounds == onfly)
            if 4 * sigma < limit:
                assert filtration_select(levels, G, sigma, 4 * sigma) == truth
                assert filtration_select(levels, G, sigma, np.inf) == truth
                tally["answered"] += 1
            else:
                tally["beyond limit"] += 1
                tally["closure exact beyond limit"] += onfly is not None
    assert dict(tally) == {"answered": 118, "beyond limit": 122, "closure exact beyond limit": 6}
