#!/usr/bin/env python3
"""Summarise a saved embedding-explorer selection for follow-up analysis.

Reads ``analysis/data/embedding_selection_latest.json`` by default (or a path
you pass), prints a compact report, and can dump pdb_ids / indices for piping.

Examples::

    ../leanmap/.venv/bin/python analysis/analyze_selection.py
    ../leanmap/.venv/bin/python analysis/analyze_selection.py path/to/selection.json
    ../leanmap/.venv/bin/python analysis/analyze_selection.py --ids-only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LATEST = REPO / "analysis" / "data" / "embedding_selection_latest.json"
META = REPO / "analysis" / "data" / "pdb_embedding_explorer_meta.npz"
TRAJ = REPO / "analysis" / "data" / "deformation_trajectories.npz"


def load_selection(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"no selection at {path} — save one from the explorer first")
    return json.loads(path.read_text())


def report(sel: dict, *, compare_traj: bool = True) -> None:
    s = sel.get("summary") or {}
    print(f"selection: n={sel.get('n')}  created={sel.get('created_utc')}")
    print("crystal systems:")
    for k, v in (s.get("crystal_systems") or {}).items():
        print(f"  {k:14s} {v}")
    if s.get("centerings"):
        print("centring:")
        for k, v in s["centerings"].items():
            print(f"  {k:14s} {v}")
    print("top space groups:")
    for t in (s.get("top_space_groups") or [])[:12]:
        print(f"  {t['sg_number']:3d}  {t['sg_hm']:20s}  n={t['n']}")
    if s.get("volume_median") is not None:
        iqr = s.get("volume_iqr") or [None, None]
        print(f"volume median={s['volume_median']:.0f}  IQR=[{iqr[0]:.0f}, {iqr[1]:.0f}]")
    if s.get("S_mean") is not None:
        print("mean similarity invariant s:", np.array(s["S_mean"]).round(4).tolist())
    if s.get("z_centroid") is not None:
        print("Z centroid:", np.array(s["z_centroid"]).round(4).tolist())

    svd = s.get("svd_roots")
    if svd:
        print(f"\nSVD of Kurlin similarity roots ({svd.get('feature')}, centred={svd.get('centered')})")
        print(f"  n={svd['n']}  effective_rank={svd['effective_rank']:.3f}")
        sig = svd["singular_values"]
        frac = svd["variance_frac"]
        cum = svd["variance_cum"]
        print("  k   σ            var%     cum%")
        for i in range(len(sig)):
            print(f"  {i+1}  {sig[i]:10.6g}  {100*frac[i]:6.2f}  {100*cum[i]:6.2f}")
        for ld in svd.get("top_loadings") or []:
            v = np.array(ld["v"])
            print(f"  PC{ld['component']} loadings s0..s5: {np.array2string(v, precision=3, suppress_small=True)}")

    # pairwise ambient diameter in s (sample if large)
    pts = sel.get("points") or []
    if len(pts) >= 2:
        S = np.array([p["S"] for p in pts], dtype=np.float64)
        if len(S) > 2000:
            rng = np.random.default_rng(0)
            S = S[rng.choice(len(S), 2000, replace=False)]
        # rough diameter via random pairs
        rng = np.random.default_rng(1)
        i = rng.integers(0, len(S), size=min(5000, len(S) * 2))
        j = rng.integers(0, len(S), size=len(i))
        d = np.linalg.norm(S[i] - S[j], axis=1)
        print(f"d_sim among selection (random pairs): median={np.median(d):.4f}  "
              f"p90={np.percentile(d, 90):.4f}  max≈{d.max():.4f}")

    if compare_traj and TRAJ.is_file() and pts:
        t = np.load(TRAJ)
        S = np.array([p["S"] for p in pts], dtype=np.float32)
        Z = np.array([p["z"] for p in pts], dtype=np.float32)
        if "p2_S" in t.files:
            for label, key in (("P2 start", 0), ("trigonal end", -1)):
                s_ref = t["p2_S"][key]
                d = np.linalg.norm(S - s_ref, axis=1)
                print(f"d_sim → {label}: min={d.min():.4f}  median={np.median(d):.4f}")
        if "p2_Z" in t.files:
            for label, key in (("P2 start Z", 0), ("trigonal end Z", -1)):
                z_ref = t["p2_Z"][key]
                d = np.linalg.norm(Z - z_ref, axis=1)
                print(f"‖ΔZ‖ → {label}: min={d.min():.4f}  median={np.median(d):.4f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("selection", type=Path, nargs="?", default=LATEST)
    ap.add_argument("--ids-only", action="store_true", help="print pdb_ids one per line")
    ap.add_argument("--indices-only", action="store_true")
    ap.add_argument("--json-summary", action="store_true", help="print summary dict as JSON")
    args = ap.parse_args(argv)

    sel = load_selection(args.selection)
    if args.ids_only:
        for p in sel.get("pdb_ids") or []:
            print(p)
        return
    if args.indices_only:
        for i in sel.get("indices") or []:
            print(i)
        return
    if args.json_summary:
        json.dump(sel.get("summary") or {}, sys.stdout, indent=2)
        print()
        return
    report(sel)


if __name__ == "__main__":
    main()
