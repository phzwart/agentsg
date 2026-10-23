"""Real cells: the 500 COD entries flagged by the census (manuscript/census/flagged.jsonl),
which sit near zero conorms or other places where reduction is ambiguous.

(1) Reindexing under each entry's own error model. For the 210 entries whose
uncertainties come from deposited esds (tau_kind = esd), a second measurement is drawn
by perturbing each conorm of the reduced cell by its propagated standard deviation
(sig_p, the 'esd' field), and is relabelled by a random proper unimodular matrix. The
band is set by the recipe of the paper, 2.5 times the root-mean-square strain of such a
perturbation (estimated from 200 draws), floored at 1e-3. The closure, the two
small-matrix searches and the cctbx pipeline of compare_methods.py are scored.

(2) Symmetry. For every entry with a reported crystal system, the Bravais type of the
reported space group against the sort spectrum: is it the highest type within a sort
distance of 0.074 (one twofold at Le Page's 3 degrees), and is it the gap type?

    PYTHONPATH=agentsg/src python manuscript/benchmark/cod_real_cells.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as cm                                       # noqa: E402  (imports cctbx first)
import numpy as np                                                 # noqa: E402

from agentsg.cell.metric import params_from_metric                 # noqa: E402
from agentsg.cell.strain_closure import BRAVAIS, _gram, _strain, sort_spectrum  # noqa: E402

ORDER = ["aP", "mP", "mC", "oP", "oS", "oI", "oF", "tP", "tI", "hR", "hP", "cP", "cI", "cF"]
HOLO = {"aP": 2, "mP": 4, "mC": 4, "oP": 8, "oS": 8, "oI": 8, "oF": 8, "tP": 16, "tI": 16, "hR": 12, "hP": 24,
        "cP": 48, "cI": 48, "cF": 48}


def reported_bravais(r):
    sysn = r["reported_system"]
    if not r.get("sg_hm", "").strip():
        return None
    letter = r["sg_hm"].strip()[0].upper()
    fam = {"triclinic": "a", "monoclinic": "m", "orthorhombic": "o", "tetragonal": "t", "trigonal": "h",
           "hexagonal": "h", "cubic": "c"}.get(sysn)
    if fam is None:
        return None
    if sysn == "trigonal":
        return "hR" if (letter == "R" and "H" not in r["sg_hm"].split()[-1:]) or letter == "R" else "hP"
    if fam == "m":
        return "mP" if letter == "P" else "mC"
    if fam == "o":
        return {"P": "oP", "C": "oS", "A": "oS", "B": "oS", "I": "oI", "F": "oF"}[letter]
    if fam in "tc":
        return fam + letter
    return fam + "P"


def reindex_test(rows, rng):
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    bands = []
    for r in rows:
        p = np.array(r["conorms"], float)
        sig = np.array(r["esd"], float)
        G = _gram(np.maximum(p, 0))
        if np.linalg.eigvalsh(G).min() <= 0:
            continue
        draws = []
        for _ in range(200):
            G1 = _gram(np.maximum(p, 0) + rng.normal(size=6) * sig)
            if np.linalg.eigvalsh(G1).min() > 0:
                draws.append(_strain(G, G1, fit_scale=False)[0])
        sigma = float(np.sqrt(np.mean(np.square(draws))))
        band = min(max(2.5 * sigma, 1e-3), 0.2)
        bands.append(band)
        while True:
            G1 = _gram(np.maximum(p, 0) + rng.normal(size=6) * sig)
            if np.linalg.eigvalsh(G1).min() > 0:
                break
        while True:
            U = rng.integers(-2, 3, (3, 3))
            if round(np.linalg.det(U)) == 1:
                break
        ref = params_from_metric(G.tolist())
        new = params_from_metric((U.T @ G1 @ U).tolist())
        for name, f in cm.METHODS.items():
            try:
                Ps, ps = f(ref, new, band)
            except RuntimeError:
                Ps, ps = np.zeros((0, 3, 3), int), np.zeros(0)
            ok, n, rank, wrong = cm.score(Ps, ps, U)
            a = agg[name]
            a["ok"].append(ok); a["n"].append(n)
            if ok:
                a["rank1"].append(rank == 1)
    print(f"(1) reindexing, {len(agg['closure']['ok'])} COD cells with esd-derived errors; band = 2.5 x rms strain, "
          f"median {np.median(bands):.4f}, range {min(bands):.4f}-{max(bands):.4f}")
    print(f"{'method':11s} {'found':>9s} {'cands q50/q90':>14s} {'true cheapest':>14s}")
    for name in cm.METHODS:
        a = agg[name]
        q = np.percentile(a["n"], [50, 90]).astype(int)
        print(f"{name:11s} {sum(a['ok']):4d}/{len(a['ok']):<4d} {q[0]:6d}/{q[1]:<7d} {sum(a['rank1']):6d}/{sum(a['ok']):<6d}")


def _spectrum(r):
    return sort_spectrum(tuple(r["cell"]), r["sg_hm"] if r["sg_hm"][0] in "PABCIFR" else None)


def symmetry_test(rows):
    from concurrent.futures import ProcessPoolExecutor
    tally = collections.Counter()
    disagree = collections.Counter()
    rows = [r for r in rows if reported_bravais(r) is not None]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        spectra = list(ex.map(_spectrum, rows, chunksize=4))
    for r, sp in zip(rows, spectra):
        rb = reported_bravais(r)
        by = {}
        for s, d in sp.items():
            by[BRAVAIS[s]] = min(by.get(BRAVAIS[s], 9), d)
        within = [b for b in by if by[b] <= cm.BAND]
        top = max(within, key=lambda b: (HOLO[b], -by[b]))
        lad = sorted((b for b in by if b != "aP"), key=by.get)
        d = [max(by[b], 1e-6) for b in lad]
        gaps = [np.log10(d[i + 1] / d[i]) for i in range(len(d) - 1)]
        gap_type = lad[int(np.argmax(gaps))] if gaps else "aP"
        tally["n"] += 1
        tally["band=reported"] += top == rb
        tally["band>reported"] += HOLO[top] > HOLO[rb]
        tally["band<reported"] += HOLO[top] < HOLO[rb]
        tally["gap=reported"] += gap_type == rb
        tally["flag metric-above-reported"] += "metric-above-reported" in r["flags"]
        if top != rb:
            disagree[(rb, top)] += 1
    n = tally["n"]
    print(f"\n(2) symmetry, {n} COD cells with a reported crystal system")
    print(f"  highest Bravais type within sort distance 0.074 equals the reported type: {tally['band=reported']}/{n}")
    print(f"  higher than reported: {tally['band>reported']}   lower than reported: {tally['band<reported']}")
    print(f"  gap type equals the reported type: {tally['gap=reported']}/{n}")
    print(f"  (census flag metric-above-reported on {tally['flag metric-above-reported']} of these)")
    print("  most common disagreements (reported -> within band):", disagree.most_common(8))


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(os.path.join(HERE, "..", "census", "flagged.jsonl"))]
    if "--symmetry-only" in sys.argv:
        symmetry_test(rows)
        sys.exit()
    reindex_test([r for r in rows if r["tau_kind"] == "esd"], np.random.default_rng(5))
    symmetry_test(rows)
