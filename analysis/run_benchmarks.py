#!/usr/bin/env python3
"""Reproduce manuscript benchmarks and write analysis/data/*.csv.

Benchmarks
----------
1. Full-PDB root-invariant k-NN search (cKDTree RootIndex).
2. 3000-cell planted nearest-neighbour recovery (figure 1a).
3. Serial reindexing: brute unimodular vs root-first reference coset (figure 1b).
4. CXIDB 83 XFEL stream summary (figure 2, if stream file present).

Usage::

    python analysis/run_benchmarks.py \\
        --db data/pdb_cells.duckdb \\
        --stream data/blac_new_v0_nomulti.stream

Set ``--stream auto`` to download CXIDB 83 (~212 MB) into ``data/``.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "analysis" / "data"
CXIDB_83_STREAM_URL = "https://www.cxidb.org/data/83/blac_new_v0_nomulti.stream"


def _pct(vals, p):
    s = sorted(vals)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_kv(path: Path, metrics: dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])


def ensure_stream(path: Path, mode: str) -> Path | None:
    if mode == "skip":
        return path if path.is_file() else None
    if path.is_file():
        return path
    if mode != "auto":
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading CXIDB 83 stream -> {path} ...")
    req = urllib.request.Request(
        CXIDB_83_STREAM_URL, headers={"User-Agent": "agentsg-benchmark/1.0"})
    with urllib.request.urlopen(req) as resp, path.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return path if path.is_file() else None


def benchmark_pdb(db_path: Path, n_query: int = 200) -> dict:
    from agentsg.cell.celldb import CellDatabase, _primitive_for_roots
    from agentsg.cell.rootform import root_invariant

    db = CellDatabase(str(db_path))
    n_cells = len(db)
    t0 = time.perf_counter()
    idx = db.build_index()
    index_build_s = time.perf_counter() - t0

    rows = db.sql("SELECT a,b,c,alpha,beta,gamma,sg_hm FROM cells")
    random.seed(42)
    sample = random.sample(rows, min(n_query, len(rows)))
    queries = [((r[0], r[1], r[2], r[3], r[4], r[5]), r[6]) for r in sample]

    times = []
    for cell, sg_hm in queries:
        t0 = time.perf_counter()
        idx.k_nearest(cell, k=10, sg_hm=sg_hm)
        times.append((time.perf_counter() - t0) * 1000)

    allroots = db.sql("SELECT pdb_id,r0,r1,r2,r3,r4,r5 FROM cells")
    exact = 0
    max_diff = 0.0
    for cell, sg_hm in queries[:100]:
        qi = root_invariant(_primitive_for_roots(cell, sg_hm))
        brute = sorted(
            (math.sqrt(sum((qi[i] - r[i + 1]) ** 2 for i in range(6))), r[0])
            for r in allroots)[:10]
        got = idx.k_nearest(cell, k=10, sg_hm=sg_hm)
        got_d = sorted(d for _, d in got)
        brute_d = sorted(d for d, _ in brute)
        if all(abs(a - b) < 1e-9 for a, b in zip(got_d, brute_d)):
            exact += 1
        max_diff = max(max_diff, max(abs(a - b) for a, b in zip(got_d, brute_d)))

    db.close()
    return {
        "n_cells": n_cells,
        "index_backend": "scipy.cKDTree",
        "index_build_s": round(index_build_s, 3),
        "k10_p50_ms": round(statistics.median(times), 4),
        "k10_p90_ms": round(_pct(times, 90), 4),
        "k10_p99_ms": round(_pct(times, 99), 4),
        "k10_max_ms": round(max(times), 4),
        "correctness_kth_distance_exact": f"{exact}/100",
        "correctness_max_abs_dist_diff": f"{max_diff:.2e}",
    }


def _valid_cell(rng: random.Random):
    while True:
        cell = (rng.uniform(20, 120), rng.uniform(20, 120), rng.uniform(20, 120),
                rng.uniform(70, 110), rng.uniform(70, 110), rng.uniform(70, 110))
        try:
            from agentsg.cell.metric import UnitCell
            if UnitCell(*cell).volume() > 1e3:
                return cell
        except Exception:
            pass


def benchmark_planted_knn(n_cells: int = 3000, n_query: int = 40, k: int = 1) -> dict:
    from agentsg.cell.rootform import root_invariant, root_distance
    from agentsg.cell.rootindex import build_root_index

    rng = random.Random(17)
    base = (79.0, 79.0, 38.0, 90.0, 90.0, 90.0)
    planted_id = "PLANT"
    cells = [(base, planted_id)]
    for i in range(n_cells - 1):
        cells.append((_valid_cell(rng), f"C{i:04d}"))

    t0 = time.perf_counter()
    idx = build_root_index((root_invariant(c), pid) for c, pid in cells)
    build_s = time.perf_counter() - t0

    recovered = 0
    times = []
    for trial in range(n_query):
        noise = tuple(v + rng.gauss(0, 0.05 if j < 3 else 0.02)
                      for j, v in enumerate(base))
        t0 = time.perf_counter()
        hits = idx.k_nearest(noise, k=k)
        times.append((time.perf_counter() - t0) * 1000)
        if hits and hits[0][0] == planted_id:
            recovered += 1

    # reduction-boundary robustness: queries on both sides of a=b
    ref = (40.0, 40.0, 60.0, 90.0, 91.0, 90.0)
    ref_cells = [(ref, "REF")] + [(_valid_cell(rng), i) for i in range(500)]
    idx_flip = build_root_index((root_invariant(c), pid) for c, pid in ref_cells)
    hi = idx_flip.nearest_cell((40.0, 40.02, 60.0, 90.0, 91.0, 90.0))
    lo = idx_flip.nearest_cell((40.0, 39.98, 60.0, 90.0, 91.0, 90.0))
    flip_ok = hi[0] == "REF" and lo[0] == "REF"

    return {
        "n_cells": n_cells,
        "index_backend": "scipy.cKDTree",
        "index_build_s": round(build_s, 3),
        "n_queries": n_query,
        "k": k,
        "recovered": recovered,
        "query_mean_ms": round(statistics.mean(times), 4),
        "query_p50_ms": round(statistics.median(times), 4),
        "reduction_flip_same_ref": flip_ok,
        "reduction_flip_d_hi": round(hi[1], 6),
        "reduction_flip_d_lo": round(lo[1], 6),
        "reduction_flip_d_diff": round(abs(hi[1] - lo[1]), 6),
    }


def _perturb(cell, rng, ls=0.15, as_=0.08):
    a, b, c, al, be, ga = cell
    f = lambda v: v * (1 + rng.gauss(0, ls / 100))
    g = lambda v: v + rng.gauss(0, as_)
    return (f(a), f(b), f(c), g(al), g(be), g(ga))


def benchmark_serial_reindexing(n_frames: int = 200) -> dict:
    from agentsg.cell.ambiguity import ReindexingReference
    from agentsg.cell.canonical import reindexing_via_canonical
    from agentsg.cell.g6 import _unimodular_pm1
    from agentsg.cell.metric import UnitCell, params_from_metric
    from agentsg.cell.reindex import reindexing_operators, _transform_metric
    from agentsg.cell.rootform import root_distance

    ref = (120.7, 189.1, 129.4, 90.0, 91.2, 90.0)
    sg = 4  # P 1 21 1
    rng = random.Random(2026)
    reindex_ops = [((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                   ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
                   ((-1, 0, 0), (0, 1, 0), (-1, 0, 1))]

    def reindex_cell(cell, M):
        G = UnitCell(*cell).metric_tensor()
        Gp = _transform_metric(G, M)
        return params_from_metric(Gp)

    frames = []
    for _ in range(n_frames // 2):
        cell = _perturb(reindex_cell(ref, rng.choice(reindex_ops)), rng)
        frames.append(cell)
    for _ in range(n_frames // 2):
        # deliberately straddle a ~ a' near-degeneracy
        a, b, c, al, be, ga = _perturb(ref, rng, ls=0.05, as_=0.03)
        frames.append((a + rng.gauss(0, 0.02), b, c, al, be, ga))

    n_unimod = len(list(_unimodular_pm1()))

    brute_times = []
    for frame in frames:
        t0 = time.perf_counter()
        reindexing_operators(ref, frame, length_tol_pct=2.0, angle_tol_deg=2.0)
        brute_times.append((time.perf_counter() - t0) * 1000)

    ref_cos = ReindexingReference(sg, ref, length_tol_pct=4.0, angle_tol_deg=5.0)
    n_ops = len(ref_cos)

    coset_times = []
    root_full_times = []
    for frame in frames:
        t0 = time.perf_counter()
        ref_cos.resolve(frame)
        coset_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        if root_distance(ref, frame) < 2.0:
            reindexing_via_canonical(ref, frame, verify_rel=0.06)
            ref_cos.resolve(frame)
        root_full_times.append((time.perf_counter() - t0) * 1000)

    return {
        "n_frames": n_frames,
        "n_unimodular_tested": n_unimod,
        "n_reference_operators": n_ops,
        "brute_mean_ms": round(statistics.mean(brute_times), 3),
        "brute_p50_ms": round(statistics.median(brute_times), 3),
        "coset_resolve_mean_ms": round(statistics.mean(coset_times), 3),
        "coset_resolve_p50_ms": round(statistics.median(coset_times), 3),
        "root_first_full_mean_ms": round(statistics.mean(root_full_times), 3),
        "root_first_full_p50_ms": round(statistics.median(root_full_times), 3),
        "speedup_coset_mean": round(statistics.mean(brute_times) / max(statistics.mean(coset_times), 1e-9), 1),
        "speedup_coset_p50": round(statistics.median(brute_times) / max(statistics.median(coset_times), 1e-9), 1),
    }


def benchmark_xfel(stream_path: Path) -> dict:
    from agentsg.cell.crystfel_stream import parse_stream, stream_summary
    from agentsg.cell.rootform import root_invariant

    summary = stream_summary(str(stream_path))
    cells = [d["cell"] for d in parse_stream(str(stream_path))]
    c_vals = [c[2] for c in cells]
    c_median = statistics.median(c_vals)
    c_mad = statistics.median(abs(x - c_median) for x in c_vals)

    # orientation diagnostic: c-scatter vs beam angle to c*
    with_cstar = list(parse_stream(str(stream_path), with_orientation=True))
    parallel, oblique = [], []
    for rec in with_cstar:
        c = rec["cell"][2]
        beam = np.array([0.0, 0.0, 1.0])
        cstar = np.array(rec["cstar"])
        cstar /= np.linalg.norm(cstar)
        cosang = abs(float(np.dot(beam, cstar)))
        (parallel if cosang > math.cos(math.radians(10)) else oblique).append(c)

    def scatter(vals):
        if len(vals) < 2:
            return 0.0
        med = statistics.median(vals)
        return statistics.median(abs(v - med) for v in vals)

    return {
        "stream": stream_path.name,
        "n_frames": summary["n_frames"],
        "n_indexed": summary["n_indexed"],
        "indexing_rate_pct": round(100 * summary["indexing_rate"], 1),
        "cell_mean_a": round(summary["cell_mean"][0], 2),
        "cell_mean_b": round(summary["cell_mean"][1], 2),
        "cell_mean_c": round(summary["cell_mean"][2], 2),
        "c_median_A": round(c_median, 2),
        "c_mad_A": round(c_mad, 3),
        "c_scatter_parallel_cstar_A": round(scatter(parallel), 3),
        "c_scatter_oblique_A": round(scatter(oblique), 3),
        "n_parallel_cstar": len(parallel),
        "n_oblique_cstar": len(oblique),
        "n_with_orientation": len(with_cstar),
        "root_invariant_dim": len(root_invariant(cells[0])),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(REPO / "data" / "pdb_cells.duckdb"))
    p.add_argument("--stream", default=str(REPO / "data" / "blac_new_v0_nomulti.stream"),
                   help="path to CXIDB 83 stream, or 'auto' to download, 'skip' to omit")
    args = p.parse_args()

    print("=== PDB database search ===")
    pdb = benchmark_pdb(Path(args.db))
    for k, v in pdb.items():
        print(f"  {k}: {v}")
    _write_kv(DATA / "query_benchmark.csv", pdb)

    print("\n=== 3000-cell planted k-NN ===")
    planted = benchmark_planted_knn()
    for k, v in planted.items():
        print(f"  {k}: {v}")
    _write_kv(DATA / "planted_knn_benchmark.csv", planted)

    print("\n=== Serial reindexing ===")
    serial = benchmark_serial_reindexing()
    for k, v in serial.items():
        print(f"  {k}: {v}")
    _write_kv(DATA / "serial_reindex_benchmark.csv", serial)

    stream_default = REPO / "data" / "blac_new_v0_nomulti.stream"
    if args.stream == "auto":
        stream_path = ensure_stream(stream_default, "auto")
    elif args.stream == "skip":
        stream_path = None
    else:
        stream_path = ensure_stream(Path(args.stream), "skip")
    if stream_path is not None:
        print(f"\n=== XFEL stream ({stream_path}) ===")
        xfel = benchmark_xfel(stream_path)
        for k, v in xfel.items():
            print(f"  {k}: {v}")
        _write_kv(DATA / "xfel_stream_benchmark.csv", xfel)
    else:
        print("\n=== XFEL stream: skipped (file not found) ===")

    print("\nWrote analysis/data/*.csv")


if __name__ == "__main__":
    main()
