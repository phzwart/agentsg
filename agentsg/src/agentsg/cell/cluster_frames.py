"""Put every entry of a crystal-form cluster on the reference's setting and origin.

The reference of a cluster is its highest-resolution entry among those whose
space group has the cluster's most common IT number. Each other entry of that
IT number is placed on the reference in three steps.

1. **Reindexing coset.** :func:`agentsg.cell.strain_closure.indexing_classes`
   lists the operators ``P`` from the reference's basis to the entry's basis
   whose strain price is within the band, grouped into classes of the
   reference's point group (proper operators only, since the molecules are
   chiral). Each class is one physically distinct indexing; the cell cannot
   choose between them. The band is 3 degrees of obliquity (strain price 14.8 %), so that an
   indexing just outside the 7 % clustering band is still scored near a higher lattice
   symmetry; it widens to 20 % when
   nothing is found (chained clusters).
2. **Operators and origin.** For each class representative the entry's
   space-group operators are carried into the reference basis
   (:class:`agentsg.change_of_basis.ChangeOfBasis`), and the origin shift ``p``
   that turns them into the reference's operators is solved exactly
   (``identify._origin_shift_to_standard``). A class with no such shift does not
   describe the reference's group and is dropped. The entry's fractional
   coordinates in the reference setting are ``P x - p``.
3. **Allowed origins by Cα RMSD.** :func:`agentsg.match_models.match_models`
   searches the proper coset representatives of the Euclidean normalizer (the
   allowed origin shifts and the axis changes that keep the group) composed
   with every operator of the group, removes the floating part of the shift,
   and scores by Cα RMSD. Cα atoms are paired by UniProt accession and residue
   number (:mod:`agentsg.cell.pdb_coords`). The probe is the reference chain
   with the most mapped residues; every entry chain with the same accession is
   tried against it, so a different chain labelling or asymmetric-unit choice
   does not matter.

The best placement over all classes, chains and normalizer elements is kept.
``margin`` is the RMSD of the best *different* placement minus the best one; a
small margin means the coordinates barely decide (pseudo-symmetric packing).
``asu_rmsd`` checks the whole entry after placement: each chain is compared with
the nearest reference chain of the same accession under the group's operators
and lattice translations only, with no further fitting.

The stored frame is an affine map ``x -> L x + t`` from the entry's deposited
fractional coordinates to the reference's fractional coordinates.

Command line::

    python -m agentsg.cell.cluster_frames clusters.duckdb ca.duckdb frames.duckdb [--workers 2]

Needs numpy and scipy.
"""
from __future__ import annotations

import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from fractions import Fraction

BAND_STEPS = (0.148, 0.20)
FALLBACK_RMSD = 3.0
MARGIN_CAP = 5.0
"""Candidates are examined until their lower-bound score exceeds the best placement by this
much (Å); a margin reported at this value is a lower bound."""
MAX_EXACT = 80
"""Most candidates made exact per entry; a margin reached at this cap is a lower bound."""
REGISTER_FIT = 1.5
"""Superposed Cα RMSD (Å) below which a residue-number register is taken as right."""
REGISTER_TRIGGER = 2.5
"""The register search runs only when the deposited numbering superposes worse than this
(Å), and a shift is taken only when it at least halves that RMSD and is below
``REGISTER_FIT``."""
REGISTER_SEARCH = 200
"""Largest residue-number shift tried when the deposited register pairs badly."""
"""When the best placement within the first band is farther than this from the reference
(Å), the classes of the next band are scored as well and the better placement is kept."""
"""Strain-price bands for the reindexing coset. The first is Le Page's 3 degrees
(``strain_closure.LEPAGE_PRICE`` = 2 sqrt(2) artanh(sin 3 deg) = 0.148): every class within
it is scored on the Cα, so an indexing just outside the 7 % clustering band is not missed
when the lattice is close to a higher symmetry. 20 % is scored as well when nothing is found within 3 degrees or the best placement is
still more than ``FALLBACK_RMSD`` from the reference (an indexing just above 14.8 %)."""
MIN_PAIRED = 20
PROBE_MAX = 200
"""Largest number of probe Cα used to rank candidates; the reported RMSD uses all pairs."""


def _np():
    import numpy as np
    return np


# --------------------------------------------------------------------- models
@dataclass
class Model:
    """One entry: deposited cell and symbol, Cα per chain (fractional, deposited basis)."""
    pdb_id: str
    cell: tuple
    sg_hm: str
    resolution: float | None
    chains: dict = field(default_factory=dict)   # chain -> (acc, unp int array, frac (n,3))

    def mapped_chains(self):
        return {c: v for c, v in self.chains.items() if v[0] is not None}


def _orth(cell):
    np = _np()
    a, b, c, al, be, ga = cell
    al, be, ga = (math.radians(x) for x in (al, be, ga))
    v = math.sqrt(1 - math.cos(al) ** 2 - math.cos(be) ** 2 - math.cos(ga) ** 2
                  + 2 * math.cos(al) * math.cos(be) * math.cos(ga))
    return np.array([[a, b * math.cos(ga), c * math.cos(be)],
                     [0, b * math.sin(ga), c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)],
                     [0, 0, c * v / math.sin(ga)]])


def _mat(M):
    np = _np()
    return np.array([[float(x) for x in row] for row in M.rows])


def _vec(v):
    np = _np()
    return np.array([float(x) for x in v.v])


# ------------------------------------------------------------ steps 1 and 2
class _Reference:
    """The reference with its closures (per band) and group cached."""

    def __init__(self, model):
        from ..space_groups import space_group
        self.model = model
        self.group = space_group(model.sg_hm)
        self.ops = frozenset(self.group.operations())
        self._closures = {}
        self.orth = _orth(model.cell)

    def matcher(self):
        """The Euclidean normalizer of the reference's group at its cell, built once, and
        every proper coset representative composed with every group operator as float
        arrays ``x -> A x + b`` for vectorised scoring."""
        m = getattr(self, "_matcher", None)
        if m is None:
            np = _np()
            from ..match_models import _group
            from ..normalizer import euclidean_normalizer
            norm = euclidean_normalizer(self.model.sg_hm, tuple(self.model.cell))
            _ops, _system, rec = _group(self.model.sg_hm)
            ops = list(rec.operations()) if hasattr(rec, "operations") else list(_ops)
            combos, A, b = [], [], []
            for M, mvec, det in norm.coset_reps():
                if det != 1:
                    continue
                Mf, mf = _mat(M), _vec(mvec)
                for op in ops:
                    combos.append((M, mvec, det, op))
                    A.append(Mf @ _mat(op.W))
                    b.append(Mf @ _vec(op.w) + mf)
            m = self._matcher = (norm, ops, combos, np.array(A), np.array(b))
        return m

    def score(self, xa, xb):
        """Cα RMSD of every candidate placement of ``xb`` (fractional, reference basis) on
        ``xa``, after removing the torus-mean shift: the RMSD that
        :func:`agentsg.match_models.match_models` reports, for all candidates at once."""
        np = _np()
        _norm, _ops, _combos, A, b = self.matcher()
        d = xa[None] - (np.einsum("kij,nj->kni", A, xb) + b[:, None, :])
        wrap = lambda v: v - np.ceil(v - 0.5)            # into (-1/2, 1/2]
        sh = d[:, :1] + wrap(d - d[:, :1])
        res = wrap(sh - sh.mean(1, keepdims=True))
        cart = res @ self.orth.T
        return np.sqrt((cart ** 2).sum(2).mean(1))

    def closure(self, accept):
        from .strain_closure import band_for, widened_closure
        W = self._closures.get(accept)
        if W is None:
            W = widened_closure(tuple(self.model.cell), self.model.sg_hm,
                                eps=band_for(accept) * 1.001)
            self._closures[accept] = W
        return W


def reindex_classes(ref, member, steps=BAND_STEPS):
    """The coset step: ``(accept, [(P, price, class_size)])`` at the first band that gives
    candidates, or ``(None, [])``."""
    from .strain_closure import indexing_classes
    for accept in steps:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = indexing_classes(ref.closure(accept), tuple(member.cell), ref.model.sg_hm,
                                       new_sg_hm=member.sg_hm, accept=accept, proper=True,
                                       anomalous=True)
        except ValueError:
            continue
        if res.classes:
            return accept, [(cl.representative.operator, cl.representative.price, cl.size)
                            for cl in res.classes]
    return None, []


def origin_for(ref, member_ops, P):
    """Origin ``p`` with ``(I, p)`` carrying the entry's operators, written in the
    reference basis by ``P``, onto the reference's operators; ``None`` when the class
    does not describe the reference's group."""
    from ..change_of_basis import ChangeOfBasis
    from ..identify import _origin_shift_to_standard
    from ..linalg import ZERO3
    to_ref = ChangeOfBasis(P, ZERO3).inverse()
    try:
        ops_in = frozenset(to_ref.apply_to_op(op) for op in member_ops)
    except (ValueError, ZeroDivisionError):
        return None
    return _origin_shift_to_standard(ops_in, ref.ops)


# ------------------------------------------------------------------ step 3
@dataclass
class Placement:
    pdb_id: str
    reference: str
    status: str
    accept: float | None = None
    n_classes: int = 0
    class_price: float | None = None
    member_chain: str | None = None
    ref_chain: str | None = None
    n_paired: int = 0
    rmsd: float | None = None
    second_rmsd: float | None = None
    margin: float | None = None
    asu_rmsd: float | None = None
    ambiguous_snap: bool = False
    via: str | None = None           # anchor entry when placed through a chain of neighbours
    unp_offset: int = 0              # residue-number offset of the deposited UniProt mapping
    acc_fallback: bool = False       # paired on residue numbers under another accession
    L: object = None                 # exact 3x3 (Matrix3), entry fractional -> reference fractional
    t: object = None                 # float translation (3,)


def _pairs(ref_chain, mem_chain):
    np = _np()
    _acc, ur, xr = ref_chain
    _acc2, um, xm = mem_chain
    common, ir, im = np.intersect1d(ur, um, return_indices=True)
    return ir, im


def _subsample(n, k):
    np = _np()
    if n <= k:
        return np.arange(n)
    return np.linspace(0, n - 1, k).round().astype(int)


def _kabsch_rmsd(P, Q):
    np = _np()
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Q.T @ P)
    S[2] *= np.sign(np.linalg.det(U @ Vt))
    return float(np.sqrt(max((P ** 2).sum() + (Q ** 2).sum() - 2 * S.sum(), 0.0) / len(P)))


def _register(ref_unp, ref_cart, mem_unp, mem_cart, search=REGISTER_SEARCH):
    """The residue-number shift ``k`` that pairs the member's Cα (number + k) with the
    reference's best, by superposed RMSD, when the deposited numbering pairs them badly.

    Returns ``None`` when the deposited numbering superposes better than ``REGISTER_TRIGGER``
    on at least half the shorter chain, or when no shift gets below ``REGISTER_FIT`` and
    half the deposited RMSD. A shift must pair at least half the shorter chain, so a repeat or a
    second domain is not matched on its own.
    """
    np = _np()
    need = max(MIN_PAIRED, 0.5 * min(len(ref_unp), len(mem_unp)))

    def fit(k):
        _, ir, im = np.intersect1d(ref_unp, mem_unp + k, return_indices=True)
        return None if len(ir) < need else _kabsch_rmsd(ref_cart[ir], mem_cart[im])

    f0 = fit(0)
    if f0 is not None and f0 < REGISTER_TRIGGER:
        # small misfits are conformational; on a helix a shift of one turn (3-4 residues)
        # can fit two conformers nearly as well as the true register
        return None
    best = None
    for k in range(-search, search + 1):
        if k == 0:
            continue
        f = fit(k)
        if f is not None and (best is None or f < best[0]):
            best = (f, k)
    if best is None or best[0] >= REGISTER_FIT:
        return None
    if f0 is not None and best[0] > 0.5 * f0:
        return None
    return best[1]


def place(ref, member, member_ops=None):
    """Place one entry on the reference. Returns a :class:`Placement`."""
    np = _np()
    from ..linalg import Matrix3, Vector3
    from ..space_groups import space_group

    R = ref.model
    out = Placement(member.pdb_id, R.pdb_id, "pending")
    try:
        mem_group = space_group(member.sg_hm)
    except Exception:
        out.status = "unreadable_space_group"
        return out
    if mem_group.number != ref.group.number:
        out.status = "different_space_group"
        return out
    if member_ops is None:
        member_ops = list(mem_group.operations())

    best = None
    for band in BAND_STEPS:
        accept, classes = reindex_classes(ref, member, steps=(band,))
        if not classes:
            continue
        cand = _place_classes(ref, member, member_ops, accept, classes)
        if best is None or best.rmsd is None or (cand.rmsd is not None and cand.rmsd < best.rmsd):
            best = cand
        if best.status == "placed" and best.rmsd <= FALLBACK_RMSD:
            break
    if best is None:
        out.status = "no_reindexing_within_band"
        return out
    return best


def _place_classes(ref, member, member_ops, accept, classes):
    """Place one entry on the reference, scoring every reindexing class given."""
    np = _np()
    R = ref.model
    out = Placement(member.pdb_id, R.pdb_id, "pending")
    out.accept, out.n_classes = accept, len(classes)
    ref_chains = R.mapped_chains()
    probe = max(ref_chains, key=lambda c: len(ref_chains[c][1]))
    p_acc, p_unp, p_xyz = ref_chains[probe]
    mem_chains = {c: v for c, v in member.mapped_chains().items() if v[0] == p_acc}
    if not mem_chains:
        # the same protein mapped to another accession (a polyprotein 1a / 1ab, an
        # isoform): the cluster label already says it is one protein, so pair on numbers
        mem_chains = member.mapped_chains()
        out.acc_fallback = True
    if not mem_chains:
        out.status = "no_shared_accession"
        return out
    # numbering shifts to try: none, and, under another accession, the shifts that
    # line the residue-number ranges up at their starts and at their ends
    shifted = {}
    p_cart = p_xyz @ ref.orth.T
    m_orth = _orth(member.cell)
    for c, (acc, unp, xyz) in mem_chains.items():
        offs = {0}
        if out.acc_fallback and len(unp):
            offs |= {int(p_unp.min() - unp.min()), int(p_unp.max() - unp.max())}
        # a deposition numbered on another register (signal peptide, isoform, construct
        # numbering): find the whole-chain register from the structure itself
        k = _register(p_unp, p_cart, unp, xyz @ m_orth.T)
        if k is not None:
            offs.add(k)
        for k in sorted(offs, key=abs):
            shifted[(c, k)] = (acc, unp + k, xyz)

    from ..group import centering_translations
    cens = [_vec(c) for c in centering_translations(ref.ops)] or [np.zeros(3)]

    from ..match_models import _candidate
    norm, gops, combos, _A, _b = ref.matcher()
    pool = []            # (rmsd, combo index, P, price, chain, numbering shift, xa, xb, pf)
    for P, price, _size in classes:
        p = origin_for(ref, member_ops, P)
        if p is None:
            continue
        Pf, pf = _mat(P), _vec(p)
        for (mc, base), (acc, unp, xyz) in shifted.items():
            ir, im = _pairs(ref_chains[probe], (acc, unp, xyz))
            if len(ir) < MIN_PAIRED:
                continue
            sel = _subsample(len(ir), PROBE_MAX)
            xa = p_xyz[ir[sel]]
            xb = xyz[im[sel]] @ Pf.T - pf
            r = ref.score(xa, xb)
            orig = xyz[im[sel]]
            keys = p_unp[ir[sel]]
            pool.extend((float(r[k]), k, P, price, mc, base, xa, xb, pf, orig, keys)
                        for k in range(len(r)))
    if not pool:
        out.status = "too_few_paired_atoms"
        return out
    pool.sort(key=lambda s: s[0])

    def exact(item):
        """The exact placement of one pooled candidate (origin snapped, floating fitted)."""
        r, k, P, price, mc, base, xa, xb, pf, _orig, _keys = item
        M, mvec, det, op = combos[k]
        cand = _candidate(M, mvec, det, op, xa.tolist(), xb.tolist(), R.cell, norm, gops)
        Mf = _mat(cand.reindex)
        Wf, wf = _mat(cand.operator.W), _vec(cand.operator.w)
        Lx = cand.reindex @ cand.operator.W @ P
        t = Mf @ (wf - Wf @ pf) + _vec(cand.discrete_origin) + _vec(cand.floating_shift)
        return cand, Lx, t

    def image(item, Lx, t):
        """Where the candidate puts the paired Cα, by reference residue number."""
        return dict(zip(item[10].tolist(), item[9] @ _mat(Lx).T + t))

    def same_place(img1, img2):
        """True when two placements put the shared Cα in the same positions, up to the
        lattice and centring translations (0.5 Å)."""
        common = sorted(set(img1) & set(img2))
        if len(common) < MIN_PAIRED:
            return False
        d = np.array([img1[k] - img2[k] for k in common])
        for c in cens:
            dd = d + c
            dd = dd - np.round(dd.mean(0))
            if np.sqrt(((dd @ ref.orth.T) ** 2).sum(1).mean()) < 0.5:
                return True
        return False

    def actual(item, Lx, t):
        """RMSD of the probe Cα after the exact placement (origin snapped), no fitting."""
        d = item[6] - (item[9] @ _mat(Lx).T + t)
        best_r = None
        for c in cens:
            dd = d + c
            dd = dd - np.round(dd.mean(0))
            r = float(np.sqrt(((dd @ ref.orth.T) ** 2).sum(1).mean()))
            best_r = r if best_r is None else min(best_r, r)
        return best_r

    # The pool is ranked by the RMSD after the best continuous shift, a lower bound on the
    # RMSD of the exact placement, whose shift is snapped to an allowed origin (they differ
    # when, say, a second copy in the asymmetric unit sits on a non-crystallographic shift).
    # Candidates are made exact in score order until the score passes the second-best
    # distinct placement; many operator combinations reach one placement with one score,
    # so equal scores on the same chain and register are examined once.
    found = []            # [actual, item, cand, Lx, t, image], distinct placements
    examined = set()
    n_exact = 0
    bound = None
    for item in pool:
        key = (round(item[0], 6), item[4], item[5])
        if key in examined:
            continue
        examined.add(key)
        if len(found) >= 2 and item[0] >= found[1][0]:
            break
        if found and item[0] >= found[0][0] + MARGIN_CAP:
            bound = item[0]
            break
        if n_exact >= MAX_EXACT:
            bound = item[0]
            break
        n_exact += 1
        c, L2, t2 = exact(item)
        r = actual(item, L2, t2)
        img = image(item, L2, t2)
        same = next((f for f in found if same_place(f[5], img)), None)
        if same is not None:
            if r < same[0]:
                same[:] = [r, item, c, L2, t2, img]
        else:
            found.append([r, item, c, L2, t2, img])
        found.sort(key=lambda f: f[0])
    r0, item0, cand, Lx, t, _img = found[0]
    P, price, mc, base = item0[2], item0[3], item0[4], item0[5]
    if len(found) > 1:
        second = found[1][0] if bound is None else min(found[1][0], bound)
    else:
        second = bound
    margin = None if second is None else second - r0

    out.second_rmsd, out.margin = second, margin
    out.class_price, out.member_chain, out.ref_chain = float(price), mc, probe
    out.ambiguous_snap = bool(cand.ambiguous_snap)
    out.L, out.t = Lx, t
    if base:
        member = Model(member.pdb_id, member.cell, member.sg_hm, member.resolution,
                       {c: (a, u + base, x) for c, (a, u, x) in member.chains.items()})
    _evaluate(ref, member, out)
    out.unp_offset += base
    out.status = "placed"
    return out


OFFSETS = (0, -1, 1, -2, 2, -3, 3)


def _chain_rmsd(ref, rchain, mchain, Lf, t, ops):
    """Smallest Cα RMSD of a placed entry chain against a reference chain, over the
    given operators (with lattice translations) and small residue-number offsets.
    Returns (rmsd, n_paired, offset) or None."""
    np = _np()
    _ra, runp, rxyz = rchain
    _ma, munp, mxyz = mchain
    xm = mxyz @ Lf.T + t
    best = None
    for k in OFFSETS:
        common, ir, im = np.intersect1d(runp, munp + k, return_indices=True)
        if len(ir) < MIN_PAIRED:
            continue
        for W, w in ops:
            d = rxyz[ir] - (xm[im] @ W.T + w)
            d -= np.round(d.mean(axis=0))
            r = float(np.sqrt(((d @ ref.orth.T) ** 2).sum(axis=1).mean()))
            # an offset must beat the deposited numbering clearly to be used
            if best is None or r < best[0] - (0.5 if k != best[2] and best[2] == 0 else 0.0):
                best = (r, int(len(ir)), k)
    return best


def _evaluate(ref, member, out, asu=True):
    """Fill rmsd, n_paired, unp_offset (probe chain, centring translations allowed) and
    asu_rmsd (every chain, every operator of the group) for the frame in ``out``."""
    from ..group import centering_translations
    np = _np()
    Lf = _mat(out.L)
    cens = [(np.eye(3), _vec(c)) for c in centering_translations(ref.ops)] or [(np.eye(3), np.zeros(3))]
    rch = ref.model.mapped_chains()
    probe = out.ref_chain or max(rch, key=lambda c: len(rch[c][1]))
    out.ref_chain = probe
    cands = [out.member_chain] if out.member_chain else list(member.mapped_chains())
    best = None
    for mc in cands:
        r = _chain_rmsd(ref, rch[probe], member.chains[mc], Lf, out.t, cens)
        if r is not None and (best is None or r[0] < best[0][0]):
            best = (r, mc)
    if best is not None:
        (out.rmsd, out.n_paired, out.unp_offset), out.member_chain = best
    if asu:
        out.asu_rmsd = _asu_rmsd(ref, member, Lf, out.t)


def _asu_rmsd(ref, member, Lf, t):
    """Worst chain RMSD after placement, each chain against its nearest reference
    chain of the same accession (any chain when none shares it) under the group's
    operators and lattice translations, with no further fitting."""
    ops = [(_mat(op.W), _vec(op.w)) for op in ref.ops]
    ref_chains = ref.model.mapped_chains()
    worst = None
    for c, ch in member.mapped_chains().items():
        same = [rc for rc, v in ref_chains.items() if v[0] == ch[0]] or list(ref_chains)
        best = None
        for rc in same:
            r = _chain_rmsd(ref, ref_chains[rc], ch, Lf, t, ops)
            if r is not None and (best is None or r[0] < best):
                best = r[0]
        if best is not None:
            worst = best if worst is None else max(worst, best)
    return worst


# ------------------------------------------------------------ cluster driver
def _load_models(con, ids):
    np = _np()
    ent = {r[0]: r for r in con.execute(
        "SELECT pdb_id, resolution, a, b, c, alpha, beta, gamma, sg_hm FROM entries "
        "WHERE error IS NULL AND pdb_id IN (SELECT unnest(?))", [list(ids)]).fetchall()}
    rows = con.execute(
        "SELECT pdb_id, chain, acc, unp, fx, fy, fz FROM ca WHERE acc IS NOT NULL "
        "AND pdb_id IN (SELECT unnest(?)) ORDER BY pdb_id, chain, acc, unp", [list(ids)]).fetchnumpy()
    models = {}
    for pid, e in ent.items():
        if e[8] is None or any(x is None for x in e[2:8]):
            continue
        models[pid] = Model(pid, tuple(e[2:8]), e[8], e[1])
    if len(rows["pdb_id"]):
        pid_a, ch_a, acc_a = rows["pdb_id"], rows["chain"], rows["acc"]
        xyz = np.stack([rows["fx"], rows["fy"], rows["fz"]], axis=1).astype(float)
        brk = np.flatnonzero((pid_a[1:] != pid_a[:-1]) | (ch_a[1:] != ch_a[:-1])
                             | (acc_a[1:] != acc_a[:-1])) + 1
        segs = list(zip(np.r_[0, brk], np.r_[brk, len(pid_a)]))
        n_acc = {}
        for s0, _e in segs:
            n_acc[(pid_a[s0], ch_a[s0])] = n_acc.get((pid_a[s0], ch_a[s0]), 0) + 1
        for s0, e0 in segs:
            m = models.get(pid_a[s0])
            if m is None:
                continue
            # a chain mapped to several accessions (a fusion) is one unit per accession,
            # named "chain:accession"; residue numbers of different accessions never mix
            name = ch_a[s0] if n_acc[(pid_a[s0], ch_a[s0])] == 1 else f"{ch_a[s0]}:{acc_a[s0]}"
            unp = rows["unp"][s0:e0].astype(int)
            _, first = np.unique(unp, return_index=True)       # one Cα per UniProt residue
            m.chains[name] = (acc_a[s0], unp[first], xyz[s0:e0][first])
    return models


def choose_reference(models):
    """Highest resolution among entries of the most common IT number (ties: pdb id)."""
    from ..space_groups import space_group
    num = {}
    for m in models.values():
        try:
            num[m.pdb_id] = space_group(m.sg_hm).number
        except Exception:
            num[m.pdb_id] = None
    counts = {}
    for n in num.values():
        if n is not None:
            counts[n] = counts.get(n, 0) + 1
    if not counts:
        return None
    major = max(counts, key=lambda n: (counts[n], -n))
    pool = [m for m in models.values() if num[m.pdb_id] == major and m.mapped_chains()]
    if not pool:
        return None
    return min(pool, key=lambda m: (m.resolution if m.resolution is not None else 99.0, m.pdb_id))


def _onto_reference_copy(ref, member, out):
    """A composed frame is right up to the reference group: it can put the molecule on a
    symmetry mate of the reference copy. Multiply by the group operator (W, w) that brings
    it back onto the reference copy: L -> W L, t -> W t + w."""
    np = _np()
    best = None
    for op in ref.ops:
        L2 = op.W @ out.L
        t2 = _mat(op.W) @ out.t + _vec(op.w)
        q = Placement(out.pdb_id, out.reference, out.status, L=L2, t=t2)
        _evaluate(ref, member, q, asu=False)
        if q.rmsd is not None and (best is None or q.rmsd < best[0]):
            best = (q.rmsd, L2, t2)
    if best is not None:
        out.L, out.t = best[1], best[2]


def compose(outer_L, outer_t, inner_L, inner_t):
    """Frame of ``x -> outer(inner(x))``: ``(Lo Li, Lo ti + to)``."""
    return outer_L @ inner_L, _mat(outer_L) @ inner_t + outer_t


def frame_cluster(con, ids, max_anchors=3):
    """Placements of every entry of one cluster (the reference included).

    Entries with no reindexing within 20 % of the reference (the far end of a
    chained cluster) are placed on an already-placed entry that is close in cell
    shape, and the two frames are composed; ``via`` names that entry. The RMSD
    and ``asu_rmsd`` of a composed frame are measured against the reference.
    """
    from ..linalg import Matrix3
    from .rootform import similarity_distance
    np = _np()
    models = _load_models(con, ids)
    out = [Placement(i, "", "no_coordinates") for i in ids if i not in models]
    ref_model = choose_reference(models)
    if ref_model is None:
        return out + [Placement(i, "", "no_reference") for i in models]
    ref = _Reference(ref_model)
    ident = Matrix3([[Fraction(int(i == j)) for j in range(3)] for i in range(3)])
    results = {ref_model.pdb_id: Placement(ref_model.pdb_id, ref_model.pdb_id, "reference",
                                           rmsd=0.0, asu_rmsd=0.0, L=ident, t=np.zeros(3))}
    for pid, m in models.items():
        if pid == ref_model.pdb_id:
            continue
        try:
            results[pid] = place(ref, m)
        except Exception as exc:                  # pragma: no cover - logged, not fatal
            results[pid] = Placement(pid, ref_model.pdb_id, f"error: {type(exc).__name__}: {exc}"[:200])

    anchors = {}
    progress = True
    while progress:
        progress = False
        placed = [p for p, r in results.items() if r.L is not None]
        for pid, r in list(results.items()):
            if r.status != "no_reindexing_within_band":
                continue
            m = models[pid]
            near = sorted((a for a in placed if a != pid),
                          key=lambda a: similarity_distance(models[a].cell, m.cell))[:max_anchors]
            best_q = None
            for a in near:
                if a not in anchors:
                    anchors[a] = _Reference(models[a])
                try:
                    q = place(anchors[a], m)
                except Exception:
                    continue
                if q.status != "placed":
                    continue
                fa = results[a]
                q.L, q.t = compose(fa.L, fa.t, q.L, q.t)
                q.reference, q.via = ref_model.pdb_id, a
                q.member_chain = None
                q.ref_chain = None
                _onto_reference_copy(ref, m, q)
                _evaluate(ref, m, q)
                if best_q is None or (q.rmsd is not None and q.rmsd < best_q.rmsd):
                    best_q = q
            if best_q is not None:
                results[pid] = best_q
                placed.append(pid)
                progress = True
    return out + list(results.values())


_WORKER_CON = None


def _rows_for(cid, ids, con):
    rows = []
    for p in frame_cluster(con, ids):
        Ls = None if p.L is None else ";".join(",".join(str(x) for x in r) for r in p.L.rows)
        t = [None] * 3 if p.t is None else [float(x) for x in p.t]
        rows.append((p.pdb_id, cid, p.reference, p.status, p.accept, p.n_classes,
                     p.class_price, p.member_chain, p.ref_chain, p.n_paired, p.rmsd,
                     p.second_rmsd, p.margin, p.asu_rmsd, p.ambiguous_snap, p.via,
                     p.unp_offset, p.acc_fallback, Ls, *t))
    return rows


def _worker_init(ca_path):
    import duckdb
    global _WORKER_CON
    _WORKER_CON = duckdb.connect(ca_path, read_only=True)


def _worker(task):
    import warnings as _w
    _w.simplefilter("ignore")
    cid, ids = task
    return cid, len(ids), _rows_for(cid, ids, _WORKER_CON)


def frame_clusters(cluster_path, ca_path, out_path, cluster_ids=None, workers=1, progress=True):
    """Run :func:`frame_cluster` over clusters and write a ``frames`` table (resumable:
    clusters already in the table are skipped). ``workers`` > 1 runs clusters in
    separate processes, largest first."""
    import duckdb
    cl = duckdb.connect(cluster_path, read_only=True)
    groups = cl.execute("SELECT cluster_id, list(pdb_id ORDER BY pdb_id) FROM members "
                        "GROUP BY cluster_id ORDER BY cluster_id").fetchall()
    cl.close()
    if cluster_ids is not None:
        keep = set(cluster_ids)
        groups = [g for g in groups if g[0] in keep]
    out = duckdb.connect(out_path)
    out.execute("""CREATE TABLE IF NOT EXISTS frames (pdb_id VARCHAR, cluster_id INTEGER,
        reference VARCHAR, status VARCHAR, accept DOUBLE, n_classes INTEGER, class_price DOUBLE,
        member_chain VARCHAR, ref_chain VARCHAR, n_paired INTEGER, rmsd DOUBLE,
        second_rmsd DOUBLE, margin DOUBLE, asu_rmsd DOUBLE, ambiguous_snap BOOLEAN,
        via VARCHAR, unp_offset INTEGER, acc_fallback BOOLEAN,
        L VARCHAR, t0 DOUBLE, t1 DOUBLE, t2 DOUBLE)""")
    done = {r[0] for r in out.execute("SELECT DISTINCT cluster_id FROM frames").fetchall()}
    todo = sorted((g for g in groups if g[0] not in done), key=lambda g: -len(g[1]))
    t0 = time.time()

    def write(k, cid, n, rows):
        out.executemany("INSERT INTO frames VALUES (" + ",".join("?" * 22) + ")", rows)
        if progress:
            print(f"  cluster {cid} n={n} [{k}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)

    if workers <= 1:
        import duckdb as _d
        con = _d.connect(ca_path, read_only=True)
        for k, (cid, ids) in enumerate(todo, start=1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write(k, cid, len(ids), _rows_for(cid, ids, con))
        con.close()
    else:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(ca_path,)) as ex:
            futs = [ex.submit(_worker, g) for g in todo]
            for k, f in enumerate(cf.as_completed(futs), start=1):
                cid, n, rows = f.result()
                write(k, cid, n, rows)
    out.close()


def _cli(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3:
        print(__doc__)
        return 2
    ids, workers = None, 1
    if "--workers" in argv:
        k = argv.index("--workers")
        workers = int(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    if "--clusters" in argv:
        k = argv.index("--clusters")
        ids = [int(x) for x in argv[k + 1].split(",")]
        argv = argv[:k] + argv[k + 2:]
    frame_clusters(argv[0], argv[1], argv[2], cluster_ids=ids, workers=workers)
    return 0


if __name__ == "__main__":               # pragma: no cover
    raise SystemExit(_cli())
