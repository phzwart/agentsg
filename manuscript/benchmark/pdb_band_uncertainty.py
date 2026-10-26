"""Uncertainty of the non-isomorphism bands (Table 10) and of the shares of entries whose
cheapest pseudo-symmetric alternative lies within them.

Block bootstrap: the deposition blocks (runs of id-adjacent pairs, pdb_band_controls.py)
are resampled with replacement, which keeps the correlation within a deposition series;
the entries of pdb_lepage.jsonl are resampled independently.  Reported are 95 % intervals
of the 50th/90th/95th percentiles per reported Bravais type (cut 20 %), the number of
blocks per type, and the shares with one pooled band and with the band of the entry's own
type, where a type with fewer than MIN_BLOCKS blocks uses the pooled band.

    PYTHONPATH=agentsg/src python manuscript/benchmark/pdb_band_uncertainty.py [n_boot]
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdb_band_controls as bc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = ["aP", "mP", "mC", "oP", "oS", "oI", "oF", "tP", "tI", "hR", "hP", "cP", "cI", "cF"]
MIN_BLOCKS = 200


def main(n_boot=1000):
    P = bc.pairs()
    fit = np.array([p["fit"] for p in P]); raw = np.array([p["raw"] for p in P])
    blk = np.array([p["block"] for p in P]); bt = np.array([p["bt"] or "" for p in P])
    m = raw < 0.2
    fit, blk, bt = fit[m], blk[m], bt[m]
    blocks = np.unique(blk)
    members = collections.defaultdict(list)
    for i, b in enumerate(blk):
        members[b].append(i)
    rng = np.random.default_rng(11)
    R = [json.loads(l) for l in open(os.path.join(HERE, "pdb_lepage.jsonl"))]
    R = [r for r in R if "error" not in r]
    ent_type = np.array([r["reported"] or "" for r in R])
    ent_min = np.array([min((x for v in r["extra"].values() for x in v), default=np.inf) if r.get("extra") else np.inf
                        for r in R])
    nblk = {t: len(set(blk[bt == t])) for t in TYPES}
    out = [f"Block-bootstrap 95 % intervals (cut 20 %), {n_boot} resamples of {len(blocks)} blocks; {len(R)} entries",
           "", "type   pairs  blocks   p50 [95% CI]          p90 [95% CI]          p95 [95% CI]"]

    def pct(idx, t=None):
        x = fit[idx] if t is None else fit[idx][bt[idx] == t]
        return np.percentile(x, [50, 90, 95]) if len(x) else np.full(3, np.nan)

    allidx = np.arange(len(fit))
    boots = {t: [] for t in TYPES + ["all"]}
    shares = []
    for _ in range(n_boot):
        bs = rng.choice(blocks, len(blocks), replace=True)
        idx = np.concatenate([members[b] for b in bs])
        q_all = pct(idx)
        boots["all"].append(q_all)
        band = {}
        for t in TYPES:
            q = pct(idx, t)
            boots[t].append(q)
            band[t] = q if nblk[t] >= MIN_BLOCKS else q_all
        e = rng.integers(0, len(R), len(R))
        mn, ty = ent_min[e], ent_type[e]
        one90 = (mn <= q_all[1]).mean(); one95 = (mn <= q_all[2]).mean()
        own = np.array([band.get(t, q_all) for t in ty])
        own90 = (mn <= own[:, 1]).mean(); own95 = (mn <= own[:, 2]).mean()
        shares.append((one90, one95, own90, own95))
    for t in TYPES + ["all"]:
        sel = allidx if t == "all" else allidx[bt == t]
        if len(sel) < 30:
            continue
        q = pct(allidx, None if t == "all" else t)
        B = np.array(boots[t])
        lo, hi = np.nanpercentile(B, [2.5, 97.5], axis=0)
        flag = "" if t == "all" or nblk[t] >= MIN_BLOCKS else f"   (fewer than {MIN_BLOCKS} blocks: pooled band used for shares)"
        out.append(f"{t:4s} {len(sel):7d} {len(set(blk[sel])) if t != 'all' else len(blocks):7d}   "
                   + "   ".join(f"{100 * q[k]:5.2f} [{100 * lo[k]:5.2f}-{100 * hi[k]:5.2f}]" for k in range(3)) + flag)
    S = np.array(shares)
    lo, hi = np.percentile(S, [2.5, 97.5], axis=0)
    point = []
    q_all = pct(allidx)
    band = {t: (pct(allidx, t) if nblk[t] >= MIN_BLOCKS else q_all) for t in TYPES}
    own = np.array([band.get(t, q_all) for t in ent_type])
    point = ((ent_min <= q_all[1]).mean(), (ent_min <= q_all[2]).mean(),
             (ent_min <= own[:, 1]).mean(), (ent_min <= own[:, 2]).mean())
    out += ["", "entries whose cheapest alternative (exact cases included) lies within the band:"]
    for k, lab in enumerate(["one band, 90th", "one band, 95th", "own type's band, 90th", "own type's band, 95th"]):
        out.append(f"  {lab:24s} {100 * point[k]:5.2f} %  [{100 * lo[k]:5.2f}-{100 * hi[k]:5.2f}]")
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(HERE, "results_pdb_band_uncertainty.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
