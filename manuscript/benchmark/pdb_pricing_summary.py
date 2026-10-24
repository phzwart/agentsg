"""Rules of thumb from pdb_lepage.jsonl and the non-isomorphism quantiles
(results_pdb_nonisomorphism.txt: 90th 4.2 %, 95th 6.7 % strain after the scale fit):
price against obliquity, and how often the cheapest pseudo-symmetric alternative falls
inside the non-isomorphism band, by transition and by reported type.

    python manuscript/benchmark/pdb_pricing_summary.py
"""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N90, N95 = 0.042, 0.067
R = [json.loads(l) for l in open(os.path.join(HERE, "pdb_lepage.jsonl"))]
ok = [r for r in R if "error" not in r]
ps = [r for r in ok if r.get("extra")]
mn = lambda r: min(x for v in r["extra"].values() for x in v)          # noqa: E731
mx = lambda r: max(x for v in r["extra"].values() for x in v)          # noqa: E731
ex = [r for r in ps if mn(r) < 1e-4]
pr = [r for r in ps if mn(r) >= 1e-4]
print(f"{len(ok)} entries; {len(ps)} pseudo-symmetric at 3 deg; {len(ex)} with an exact extra lattice symmetry "
      f"(price 0); {len(pr)} priced only")
d = np.array([r["delta"] for r in pr]); p = np.array([mx(r) for r in pr]); m = d > 0.05
q = np.percentile(p[m] / (2 * np.sqrt(2) * np.arctanh(np.sin(np.radians(d[m])))), [50, 10, 90])
print(f"largest price / (2 sqrt2 artanh sin delta), delta > 0.05 deg: median {q[0]:.3f}, 10-90 % {q[1]:.3f}-{q[2]:.3f}")
print("\ncheapest pseudo-symmetric alternative, priced cases, by transition (reported -> Le Page)")
print(f"{'transition':12s} {'n':>6s} {'median':>7s} {'<=1%':>6s} {'<=4.2%':>7s} {'<=6.7%':>7s}")
tr = collections.defaultdict(list)
for r in pr:
    tr[(r["reported"], r["lepage"])].append(mn(r))
rows = sorted(tr.items(), key=lambda kv: -len(kv[1]))
for k, v in rows[:16] + [(("all", ""), [mn(r) for r in pr])]:
    v = np.array(v)
    lab = k[0] + ("->" + k[1] if k[1] else "")
    print(f"{lab:12s} {len(v):6d} {100 * np.median(v):6.2f}% {100 * (v <= .01).mean():5.0f}% "
          f"{100 * (v <= N90).mean():6.0f}% {100 * (v <= N95).mean():6.0f}%")
v = np.array([mn(r) for r in pr]); tot = len(ok)
print(f"\nof all entries: priced alternative within 4.2 %: {100 * (v <= N90).sum() / tot:.1f} %, within 6.7 %: "
      f"{100 * (v <= N95).sum() / tot:.1f} %; exact extra lattice symmetry: {100 * len(ex) / tot:.1f} %")
print("\nby reported type: entries with a pseudo-symmetric alternative <= 4.2 % (exact included)")
by = collections.Counter(r["reported"] for r in ok)
hit = collections.Counter(r["reported"] for r in ps if mn(r) <= N90)
for t in ["aP", "mP", "mC", "oP", "oS", "oI", "oF", "tP", "tI", "hR"]:
    print(f"  {t}: {100 * hit[t] / by[t]:4.0f} % of {by[t]}")
