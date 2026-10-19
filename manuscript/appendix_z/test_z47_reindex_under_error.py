"""z:reindex-error — reindexing under a declared metric error.

A reference R is cached as Theorem 2 prescribes. A new cell N is R plus an
error of Frobenius norm eta in R's frame, then reindexed by a random
unimodular M. With the band tau at the coverage bound of the snapped
stratum's closure: (i) N's reduced frame, written in R's coordinates, is a
member of the cache; (ii) the true operator M is among the operators the
single reduced frame proposes; (iii) the a-priori certificate
sqrt2 eta V / (lambda_min(G_R) - eta), with V the summed squared lengths of
N's reduced basis, bounds the true frame's depth in R. (A cache built by snapping only at tau loses
the true operator when N reduces into a hidden member; random errors here
never do, and that case is constructed in test_z46.)
"""
from __future__ import annotations

import itertools

import pytest

pytestmark = [pytest.mark.zcheck]

from test_z44_widened_complete import LATTICES, _cn, _generate, _gram, _key, _lam1  # noqa: E402

PERMS = list(itertools.permutations(range(4)))


def _tau_cov(p, eta, tau):
    """coverage bound over the closure of the stratum snapped at 4 tau:
    max over members and slots of (c_ij eta - p_ij)."""
    import numpy as np
    G = _gram(p)
    _, frames = _generate(p, 4 * tau, tau)
    worst = 0.0
    for S in frames.values():
        for n, (i, j) in enumerate([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
            u, v = S[i], S[j]
            c = ((u @ u * (v @ v) + (u @ v) ** 2) / 2) ** 0.5
            worst = max(worst, c * eta - float(-u @ G @ v))
    return worst


def _band_tau(p, eta):
    """smallest tau (on a grid) with tau >= tau_cov(eta) for the stratum snapped at 4 tau."""
    tau = 1e-9
    for _ in range(200):
        need = _tau_cov(p, eta, tau)
        if need <= tau:
            return tau
        tau = need
    raise RuntimeError("no fixed point")


def _cache(p, theta, tau):
    G = _gram(p)
    return [S for S in _generate(p, theta, tau)[1].values() if _cn(S, G).min() >= -tau - 1e-9]


def _proposed(cache, SN):
    import numpy as np
    BinvT = np.linalg.inv(SN[1:4].T)
    out = set()
    for S in cache:
        for pm in PERMS:
            for sg in (1.0, -1.0):
                P = np.round((sg * S[list(pm)])[1:4].T @ BinvT)
                out.add(tuple(P.astype(int).ravel()))
    return out


def test_true_operator_is_proposed_under_declared_error():
    import numpy as np
    from agentsg.cell.canonical import canonical_superbase
    from agentsg.cell.metric import params_from_metric
    rnd = np.random.default_rng(47)
    refs = []
    for name in ("oP three walls", "cP", "plate", "mP two walls", "hP"):
        base = np.array(LATTICES[name], float)
        w = np.where(base < 1e-12)[0]
        lam = _lam1(base)
        for walls in ([1.25, 0.5, 0.0], [0.3, 0.8, 0.1], [1.6, 0.9, 0.0]):
            p = base.copy()
            unit = 0.1 * lam / 32
            for k, f in zip(w, walls):
                p[k] = f * unit
            refs.append((name, p, unit))
    lost_by_tau_snap = 0
    trials = 0
    certs, within = [], 0
    for name, p, unit in refs:
        GR = _gram(p)
        eta = 0.25 * unit
        tau = _band_tau(p, eta)
        assert _lam1(p) > 32 * tau, name
        cache = _cache(p, 4 * tau, tau)
        keys = {_key(S) for S in cache}
        cache_tau = _cache(p, tau, tau)
        for _ in range(12):
            E = rnd.normal(size=(3, 3))
            E = (E + E.T) / 2
            E *= eta / np.linalg.norm(E)
            while True:
                M = rnd.integers(-2, 3, (3, 3)).astype(float)
                if abs(round(np.linalg.det(M))) == 1:
                    break
            GN = M.T @ (GR + E) @ M
            SN = np.array(canonical_superbase(tuple(params_from_metric(GN)))[0], float)
            in_R = SN @ M.T                                     # N's reduced frame, R coordinates
            assert _key(in_R) in keys, name                     # (i) a cached member
            true = tuple(np.round(M).astype(int).ravel())
            assert true in _proposed(cache, SN), name           # (ii) the true operator
            # (iii) the a-priori certificate bounds the true frame's depth in R
            lmin = np.linalg.eigvalsh(GR).min()
            V = sum(SN[i] @ GN @ SN[i] for i in (1, 2, 3))
            cert = 2 ** 0.5 * eta * V / (lmin - eta)
            assert max(0.0, -_cn(in_R, GR).min()) <= cert + 1e-9
            certs.append(cert / eta)
            within += cert <= tau
            trials += 1
            lost_by_tau_snap += true not in _proposed(cache_tau, SN)
    assert trials == len(refs) * 12
    # random errors of this size never put N into a hidden frame, so the
    # tau-snap cache did not fail here; that failure is the constructed
    # case of test_z46_selling_groupoid
    assert lost_by_tau_snap == 0
    # the certificate is valid but conservative (median about 13 eta)
    assert 10 < float(np.median(certs)) < 16 and within == 15
