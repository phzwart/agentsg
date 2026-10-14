#!/usr/bin/env python3
"""Selling-closure graph of the restricted cells, and a refinement of their SVD.

Tetragonal, trigonal, hexagonal, and cubic. The neighbour graph is the 50
nearest distinct lattices under the exact Selling-closure distance
(``boundary_rel=0``), not the Euclidean distance of the sorted keys. Because
that closure distance is at least the sorted-key distance, the neighbours are
certified from a sorted-key candidate list.

The embedding starts at the first three principal components of the
volume-normalised sorted key and is moved so its edge lengths agree with the
closure distances.

Writes ``linear_embed/out/closure_restricted/``.

``--all`` writes the same graph for every cell, without the line
refinement, to ``linear_embed/out/closure_all/``.
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from embed import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_PDB,
    SYSTEM_NAMES,
    crystal_system_ids,
    load_features,
)
from query import _cells_for  # noqa: E402

OUT = HERE / "out" / "closure_restricted"
RESTRICTED = ("tetragonal", "trigonal", "hexagonal", "cubic")
K = 50
CANDIDATE_START = 128
CANDIDATE_MAX = 10**9

_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_PERMS = list(permutations(range(4)))
_PAIR_INDEX = {tuple(sorted(ij)): k for k, ij in enumerate(_PAIRS)}
_SLOT = np.zeros((len(_PERMS), 6), dtype=np.int64)
for _pi, _perm in enumerate(_PERMS):
    for _s, (_i, _j) in enumerate(_PAIRS):
        _SLOT[_pi, _s] = _PAIR_INDEX[tuple(sorted((_perm[_i], _perm[_j])))]


def _present_rows(S, meta):
    """Restricted-cache rows whose pdb id still has a cell in the database."""
    import duckdb

    sys_ids = crystal_system_ids(meta["sg_number"])
    wanted = np.array([SYSTEM_NAMES.index(name) for name in RESTRICTED], dtype=np.int32)
    keep = np.flatnonzero(np.isin(sys_ids, wanted))
    ids = np.array([str(x).strip().upper() for x in meta["pdb_id"][keep]])
    db = duckdb.connect(str(DEFAULT_PDB), read_only=True)
    have = set(
        db.execute("SELECT upper(pdb_id) FROM cells").fetchnumpy()["upper(pdb_id)"].tolist()
    )
    db.close()
    ok = np.array([i in have for i in ids])
    dropped = int((~ok).sum())
    if dropped:
        print(f"dropped {dropped} cache ids with no cell row", flush=True)
    return keep[ok], ids[ok]


def _load_cells(ids: list[str]) -> dict[str, tuple]:
    packed: dict[str, tuple] = {}
    step = 4000
    for start in range(0, len(ids), step):
        packed.update(_cells_for(DEFAULT_PDB, ids[start:start + step]))
        print(f"  cells {min(start + step, len(ids)):,} / {len(ids):,}", flush=True)
    return packed


def _norm_conorms(cell) -> np.ndarray:
    """Normalised conorms of each exact Selling-closure superbase, shape (m, 6)."""
    from agentsg.cell.canonical import _closure_for_match, _dotG, _metric

    G = _metric(cell)
    members = _closure_for_match(cell, 0.0)
    out = np.empty((len(members), 6), dtype=np.float64)
    for m, C in enumerate(members):
        p = np.array([-_dotG(C[i], C[j], G) for i, j in _PAIRS], dtype=np.float64)
        total = p.sum()
        out[m] = p / np.sqrt(total) if total > 0.0 else p
    return out


def _closure_distance(qa: np.ndarray, qb: np.ndarray) -> float:
    """Minimum aligned linear distance over the two exact closures."""
    permuted = qb[:, _SLOT]
    diff = qa[:, None, None, :] - permuted[None, :, :, :]
    return float(np.linalg.norm(diff, axis=-1).min())


def _closure_many(qa: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    """Closure distance from one lattice to many others."""
    if len(others) > 512:
        parts = [
            _closure_many(qa, others[start:start + 512])
            for start in range(0, len(others), 512)
        ]
        return np.concatenate(parts)
    width = max(q.shape[0] for q in others)
    packed = np.zeros((len(others), width, 6), dtype=np.float32)
    counts = np.empty(len(others), dtype=np.int32)
    for i, q in enumerate(others):
        packed[i, :q.shape[0]] = q
        counts[i] = q.shape[0]
    permuted = packed[:, :, _SLOT]
    diff = qa.astype(np.float32)[:, None, None, None, :] - permuted[None, :, :, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    for i, count in enumerate(counts):
        dist[:, i, count:, :] = np.inf
    return dist.min(axis=(0, 2, 3)).astype(np.float64)


def _check_against_reference(cells, conorms) -> None:
    from agentsg.cell.canonical import closure_distance

    worst = 0.0
    n = min(6, len(cells))
    for a in range(n):
        for b in range(a + 1, n):
            ref = closure_distance(cells[a], cells[b], boundary_rel=0.0)[1]
            got = _closure_distance(conorms[a], conorms[b])
            worst = max(worst, abs(ref - got))
    if worst > 1e-6:
        raise SystemExit(f"fast closure distance disagrees with closure_distance by {worst}")
    print(f"closure distance matches the reference (worst {worst:.2e})", flush=True)


def _nodes(keys: np.ndarray, cells: list[tuple]):
    """One node per distinct lattice. Identical cells that share a key are one node."""
    view = np.ascontiguousarray(keys).view(
        np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
    ).ravel()
    _, key_inv = np.unique(view, return_inverse=True)
    rounded = np.round(np.asarray(cells, dtype=np.float64), 6)
    cell_view = np.ascontiguousarray(rounded).view(
        np.dtype((np.void, rounded.dtype.itemsize * rounded.shape[1]))
    ).ravel()
    # A node is a (sorted key, cell) pair, so a shared key with two lattices splits.
    code = np.empty(len(keys), dtype=np.int64)
    rep_of_group: dict[tuple[int, bytes], int] = {}
    reps = []
    for i, (k, c) in enumerate(zip(key_inv.tolist(), cell_view.tolist())):
        token = (int(k), c)
        rep = rep_of_group.get(token)
        if rep is None:
            rep = len(reps)
            rep_of_group[token] = rep
            reps.append(i)
        code[i] = rep
    reps = np.asarray(reps, dtype=np.int64)
    print(
        f"{len(keys):,} cells -> {len(reps):,} distinct lattices",
        flush=True,
    )
    return reps, code


def _certified_knn(keys: np.ndarray, conorms: list[np.ndarray]):
    """k nearest other lattices under the closure distance, certified by the sorted key."""
    from scipy.spatial import cKDTree

    n = len(keys)
    tree = cKDTree(keys)
    idx = np.full((n, K), -1, dtype=np.int32)
    dist = np.full((n, K), np.inf, dtype=np.float64)
    pending = np.arange(n, dtype=np.int64)
    width = CANDIDATE_START
    uncertified = 0
    while len(pending):
        take = min(width, n - 1)
        print(f"closure neighbours: {len(pending):,} lattices, {take} candidates", flush=True)
        sorted_d, nbr = tree.query(keys[pending], k=take + 1)
        if take == 1:
            sorted_d = sorted_d[:, None]
            nbr = nbr[:, None]
        nxt = []
        for t, i in enumerate(pending):
            js = np.asarray(nbr[t, 1:], dtype=np.int64)
            qa = conorms[i]
            if len(js) >= 64:
                cd = _closure_many(qa, [conorms[int(j)] for j in js])
            else:
                cd = np.empty(len(js), dtype=np.float64)
                for c, j in enumerate(js):
                    cd[c] = _closure_distance(qa, conorms[int(j)])
            if take >= 512 and (t + 1) % 25 == 0:
                print(f"    {t + 1:,} / {len(pending):,}", flush=True)
            positive = np.flatnonzero(cd > 1e-8)
            if len(positive) < K and take < n - 1 and width < CANDIDATE_MAX:
                nxt.append(int(i))
                continue
            use = positive if len(positive) >= K else np.arange(len(js))
            pick_n = min(K, len(use))
            chosen = use[np.argpartition(cd[use], pick_n - 1)[:pick_n]]
            chosen = chosen[np.argsort(cd[chosen])]
            radius = float(np.max(sorted_d[t]))
            kth = float(cd[chosen[-1]])
            if kth < radius or take >= n - 1 or width >= CANDIDATE_MAX:
                idx[i, :pick_n] = js[chosen]
                dist[i, :pick_n] = cd[chosen]
                if not (kth < radius or take >= n - 1):
                    uncertified += 1
            else:
                nxt.append(int(i))
        print(f"  kept {len(pending) - len(nxt):,}, still open {len(nxt):,}", flush=True)
        if not nxt:
            break
        if width >= CANDIDATE_MAX:
            break
        pending = np.asarray(nxt, dtype=np.int64)
        width = min(width * 2, CANDIDATE_MAX)
    return idx, dist.astype(np.float32), uncertified


def _candidate_distances(qa: np.ndarray, js: np.ndarray, conorms: list[np.ndarray]) -> np.ndarray:
    """Closure distance from one lattice to a candidate list."""
    if len(js) >= 64:
        return _closure_many(qa, [conorms[int(j)] for j in js])
    cd = np.empty(len(js), dtype=np.float64)
    for c, j in enumerate(js):
        cd[c] = _closure_distance(qa, conorms[int(j)])
    return cd


def _certified_knn_parallel(keys: np.ndarray, conorms: list[np.ndarray], workers: int = 8):
    """Same certified neighbours as ``_certified_knn``, scored on several threads."""
    from concurrent.futures import ThreadPoolExecutor
    from scipy.spatial import cKDTree

    n = len(keys)
    tree = cKDTree(keys)
    idx = np.full((n, K), -1, dtype=np.int32)
    dist = np.full((n, K), np.inf, dtype=np.float64)
    pending = np.arange(n, dtype=np.int64)
    width = CANDIDATE_START
    uncertified = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while len(pending):
            take = min(width, n - 1)
            print(f"closure neighbours: {len(pending):,} lattices, {take} candidates", flush=True)
            sorted_d, nbr = tree.query(keys[pending], k=take + 1)
            if take == 1:
                sorted_d = sorted_d[:, None]
                nbr = nbr[:, None]

            def score(t, pending=pending, nbr=nbr, conorms=conorms):
                js = np.asarray(nbr[t, 1:], dtype=np.int64)
                cd = _candidate_distances(conorms[int(pending[t])], js, conorms)
                return js, cd

            scored = [None] * len(pending)
            done = 0
            for t, item in enumerate(pool.map(score, range(len(pending)), chunksize=32)):
                scored[t] = item
                done += 1
                if take >= 512 and (done % 25 == 0 or done == len(pending)):
                    print(f"    {done:,} / {len(pending):,}", flush=True)
                elif done % 20000 == 0 or done == len(pending):
                    print(f"    scored {done:,} / {len(pending):,}", flush=True)
            nxt = []
            for t, i in enumerate(pending):
                js, cd = scored[t]
                positive = np.flatnonzero(cd > 1e-8)
                if len(positive) < K and take < n - 1 and width < CANDIDATE_MAX:
                    nxt.append(int(i))
                    continue
                use = positive if len(positive) >= K else np.arange(len(js))
                pick_n = min(K, len(use))
                chosen = use[np.argpartition(cd[use], pick_n - 1)[:pick_n]]
                chosen = chosen[np.argsort(cd[chosen])]
                radius = float(np.max(sorted_d[t]))
                kth = float(cd[chosen[-1]])
                if kth < radius or take >= n - 1 or width >= CANDIDATE_MAX:
                    idx[i, :pick_n] = js[chosen]
                    dist[i, :pick_n] = cd[chosen]
                    if not (kth < radius or take >= n - 1):
                        uncertified += 1
                else:
                    nxt.append(int(i))
            print(f"  kept {len(pending) - len(nxt):,}, still open {len(nxt):,}", flush=True)
            if not nxt:
                break
            if width >= CANDIDATE_MAX:
                break
            pending = np.asarray(nxt, dtype=np.int64)
            width = min(width * 2, CANDIDATE_MAX)
    return idx, dist.astype(np.float32), uncertified


def build_all():
    """SVD and the certified closure graph for every cell. No line refinement."""
    import duckdb

    out = HERE / "out" / "closure_all"
    out.mkdir(parents=True, exist_ok=True)
    S, meta = load_features(DEFAULT_PDB, DEFAULT_CACHE)
    ids = np.array([str(x).strip().upper() for x in meta["pdb_id"]])
    rows = np.arange(len(ids), dtype=np.int64)
    db = duckdb.connect(str(DEFAULT_PDB), read_only=True)
    have = set(
        db.execute("SELECT upper(pdb_id) FROM cells").fetchnumpy()["upper(pdb_id)"].tolist()
    )
    db.close()
    ok = np.array([i in have for i in ids])
    dropped = int((~ok).sum())
    if dropped:
        print(f"dropped {dropped} cache ids with no cell row", flush=True)
    rows, ids = rows[ok], ids[ok]
    print(f"loading {len(ids):,} cells", flush=True)
    packed = _load_cells(ids.tolist())
    have_cell = np.array([i in packed for i in ids])
    if not have_cell.all():
        print(f"cell load missed {int((~have_cell).sum())}", flush=True)
        rows, ids = rows[have_cell], ids[have_cell]
    keys = np.ascontiguousarray(S[rows])
    cells = [packed[i] for i in ids.tolist()]
    reps, code = _nodes(keys, cells)
    print(f"packing closures for {len(reps):,} lattices", flush=True)
    conorms = []
    for n, r in enumerate(reps):
        conorms.append(_norm_conorms(cells[int(r)]))
        if (n + 1) % 5000 == 0 or n + 1 == len(reps):
            print(f"  closures {n + 1:,} / {len(reps):,}", flush=True)
    _check_against_reference([cells[int(r)] for r in reps[:6]], conorms[:6])

    xc = keys - keys.mean(0)
    _u, singular, vt = np.linalg.svd(xc, full_matrices=False)
    svd = (xc @ vt[:3].T).astype(np.float32)
    variance = (singular ** 2)
    frac = (variance[:3] / variance.sum()).tolist()
    np.save(out / "svd.npy", svd)
    np.save(out / "row_index.npy", rows.astype(np.int64))
    print(f"svd variance {frac[0]:.4f} {frac[1]:.4f} {frac[2]:.4f}", flush=True)

    knn_idx, knn_dist, uncertified = _certified_knn_parallel(keys[reps], conorms)
    safe = np.where(knn_idx >= 0, knn_idx, 0)
    sorted_edge = np.linalg.norm(keys[reps][:, None, :] - keys[reps][safe], axis=-1)
    gap = (knn_dist - sorted_edge)[knn_idx >= 0]
    print(
        f"knn gap closure-sorted  median {np.median(gap):.4f}  "
        f"p90 {np.quantile(gap, 0.9):.4f}  max {gap.max():.4f}  "
        f"uncertified {uncertified}",
        flush=True,
    )
    np.save(out / "knn_idx.npy", knn_idx)
    np.save(out / "knn_dist.npy", knn_dist)
    np.save(out / "rep_row.npy", reps.astype(np.int64))
    np.save(out / "node_of_row.npy", code.astype(np.int32))
    stats = {
        "feature": "exact Selling-closure distance, boundary_rel=0",
        "candidates": "sorted linear key, expanded until the k-th closure neighbour is inside the candidate radius",
        "k": K,
        "n_cells": int(len(rows)),
        "n_lattices": int(len(reps)),
        "uncertified": int(uncertified),
        "gap_median": float(np.median(gap)),
        "gap_p90": float(np.quantile(gap, 0.9)),
        "gap_max": float(gap.max()),
        "svd_variance_fraction": frac,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


def _refine(z0: np.ndarray, knn_idx: np.ndarray, knn_dist: np.ndarray):
    """Move the SVD coordinates so neighbour lengths match the closure distances."""
    import torch

    n, k = knn_idx.shape
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = knn_idx.reshape(-1).astype(np.int64)
    length = knn_dist.reshape(-1).astype(np.float64)
    keep = (dst >= 0) & np.isfinite(length)
    src, dst, length = src[keep], dst[keep], length[keep]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"refine {len(src):,} edges on {device}", flush=True)
    z_init = torch.tensor(z0, dtype=torch.float32, device=device)
    z = z_init.clone().detach().requires_grad_(True)
    src_t = torch.tensor(src, dtype=torch.long, device=device)
    dst_t = torch.tensor(dst, dtype=torch.long, device=device)
    length_t = torch.tensor(length, dtype=torch.float32, device=device)
    opt = torch.optim.Adam([z], lr=1e-3)

    def stress_of(pos):
        embedded = torch.linalg.norm(pos[src_t] - pos[dst_t], dim=1)
        return ((embedded - length_t) ** 2).mean()

    with torch.no_grad():
        stress0 = float(stress_of(z_init))
    # An rms move of 0.05 costs as much as the stress the SVD starts with.
    lam = stress0 / (0.05 ** 2)
    print(f"initial stress {stress0:.6e}  anchor weight {lam:.6e}", flush=True)
    last = stress0
    for step in range(1, 801):
        opt.zero_grad()
        fit = stress_of(z)
        anchor = ((z - z_init) ** 2).mean()
        loss = fit + lam * anchor
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                now = float(stress_of(z))
                move = float(torch.linalg.norm(z - z_init, dim=1).mean())
            print(f"  step {step:4d}  stress {now:.6e}  mean move {move:.4f}", flush=True)
            if step > 1 and abs(last - now) <= 1e-4 * max(last, 1e-12):
                print("  stress settled", flush=True)
                break
            last = now
    with torch.no_grad():
        stress1 = float(stress_of(z))
        move = torch.linalg.norm(z - z_init, dim=1)
        moved = (float(move.mean()), float(move.max()))
    return z.detach().cpu().numpy().astype(np.float32), stress0, stress1, lam, moved


def refine_from_svd(z0: np.ndarray, knn_idx: np.ndarray, knn_dist: np.ndarray, symbols: np.ndarray):
    """Slide each lattice along the straight line already present in the key SVD.

    The line and its centre are fixed from ``z0``, so a space group cannot bend
    or swing onto a new axis. Only the coordinate along that line is free, and
    it moves so the 50 closure neighbours keep their length.
    """
    import torch

    z0 = np.asarray(z0, dtype=np.float64)
    n, k = knn_idx.shape
    center = np.zeros((n, 3), dtype=np.float64)
    direction = np.zeros((n, 3), dtype=np.float64)
    t0 = np.zeros(n, dtype=np.float64)
    free = np.zeros(n, dtype=bool)
    n_lines = 0
    for symbol in np.unique(symbols):
        idx = np.flatnonzero(symbols == symbol)
        pts = z0[idx]
        spread = float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(idx) else 0.0
        if len(idx) < 2 or spread < 0.05:
            center[idx] = np.median(pts, axis=0)
            continue
        origin = pts.mean(axis=0)
        axis = np.linalg.svd(pts - origin, full_matrices=False)[2][0]
        center[idx] = origin
        direction[idx] = axis
        t0[idx] = (pts - origin) @ axis
        free[idx] = True
        n_lines += 1

    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = knn_idx.reshape(-1).astype(np.int64)
    length = knn_dist.reshape(-1).astype(np.float64)
    keep = (dst >= 0) & np.isfinite(length) & (length > 1e-8)
    src, dst, length = src[keep], dst[keep], length[keep]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    center_t = torch.tensor(center, dtype=torch.float32, device=device)
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    t_init = torch.tensor(t0, dtype=torch.float32, device=device)
    free_t = torch.tensor(np.flatnonzero(free), dtype=torch.long, device=device)
    coord = t_init[free_t].clone().detach().requires_grad_(True)
    src_t = torch.tensor(src, dtype=torch.long, device=device)
    dst_t = torch.tensor(dst, dtype=torch.long, device=device)
    length_t = torch.tensor(length, dtype=torch.float32, device=device)
    print(
        f"refine {len(src):,} closure edges along {n_lines} fixed key-component lines, on {device}",
        flush=True,
    )

    def positions(coord_free):
        t = t_init.index_put((free_t,), coord_free)
        return center_t + t[:, None] * direction_t

    def stress_of(pos):
        embedded = torch.linalg.norm(pos[src_t] - pos[dst_t], dim=1)
        return ((embedded - length_t) ** 2).mean()

    with torch.no_grad():
        stress_svd = float(stress_of(center_t + t_init[:, None] * direction_t))
    # A 0.15 rms slide along the line costs as much as the stress the lines start with.
    w_anchor = stress_svd / (0.15 ** 2)
    print(f"line stress {stress_svd:.6e}  anchor {w_anchor:.6e}", flush=True)
    opt = torch.optim.Adam([coord], lr=1e-2)
    for step in range(1, 801):
        opt.zero_grad()
        fit = stress_of(positions(coord))
        anchor = ((coord - t_init[free_t]) ** 2).mean()
        loss = fit + w_anchor * anchor
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                now = float(stress_of(positions(coord)))
                slide = float((coord - t_init[free_t]).abs().mean())
            print(f"  step {step:4d}  stress {now:.6e}  mean slide {slide:.4f}", flush=True)
    with torch.no_grad():
        z = positions(coord)
        stress1 = float(stress_of(z))
        delta = z - torch.tensor(z0, dtype=torch.float32, device=device)
        move = torch.linalg.norm(delta, dim=1)
        off = z - center_t - ((z - center_t) * direction_t).sum(1)[:, None] * direction_t
        # Point groups have a zero direction, so their residual is measured from the centre.
        on_line = torch.linalg.norm(off, dim=1)
        on_line[~torch.tensor(free, device=device)] = 0
    return (
        z.detach().cpu().numpy().astype(np.float32),
        {
            "stress_svd": stress_svd,
            "stress_refined": stress1,
            "off_line_svd": float(np.sqrt(np.mean(np.sum((z0 - center - t0[:, None] * direction) ** 2, axis=1)))),
            "off_line": float(on_line.mean()),
            "mean_move": float(move.mean()),
            "max_move": float(move.max()),
            "w_anchor": float(w_anchor),
        },
    )


def straighten(embedding: np.ndarray, keys: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    """Draw each space group as one straight line.

    The cells of one Hermann–Mauguin symbol are a one-parameter family. They
    are placed on the best-fit line through that family's closure-refined
    coordinates. A family that is already a point stays a point. ``keys`` is
    unused; the line is fit in the embedding.
    """
    del keys
    out = np.array(embedding, dtype=np.float64, copy=True)
    for symbol in np.unique(symbols):
        idx = np.flatnonzero(symbols == symbol)
        pts = out[idx]
        center = pts.mean(axis=0)
        spread = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        if len(idx) < 2 or spread < 0.05:
            out[idx] = np.median(pts, axis=0)
            continue
        direction = np.linalg.svd(pts - center, full_matrices=False)[2][0]
        coord = (pts - center) @ direction
        out[idx] = center + coord[:, None] * direction
    return out.astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S, meta = load_features(DEFAULT_PDB, DEFAULT_CACHE)
    rows, ids = _present_rows(S, meta)
    print(f"loading {len(ids):,} cells", flush=True)
    packed = _load_cells(ids.tolist())
    have = np.array([i in packed for i in ids])
    if not have.all():
        print(f"cell load missed {int((~have).sum())}", flush=True)
        rows, ids = rows[have], ids[have]
    keys = np.ascontiguousarray(S[rows])
    cells = [packed[i] for i in ids.tolist()]
    reps, code = _nodes(keys, cells)
    print(f"packing closures for {len(reps):,} lattices", flush=True)
    conorms = []
    for n, r in enumerate(reps):
        conorms.append(_norm_conorms(cells[int(r)]))
        if (n + 1) % 5000 == 0 or n + 1 == len(reps):
            print(f"  closures {n + 1:,} / {len(reps):,}", flush=True)
    _check_against_reference([cells[int(r)] for r in reps[:6]], conorms[:6])

    knn_idx, knn_dist, uncertified = _certified_knn(keys[reps], conorms)
    sorted_edge = np.linalg.norm(
        keys[reps][:, None, :] - keys[reps][knn_idx], axis=-1
    )
    gap = knn_dist - sorted_edge
    print(
        f"knn gap closure-sorted  median {np.median(gap):.4f}  "
        f"p90 {np.quantile(gap, 0.9):.4f}  max {gap.max():.4f}  "
        f"uncertified {uncertified}",
        flush=True,
    )

    xc = keys - keys.mean(0)
    _u, singular, vt = np.linalg.svd(xc, full_matrices=False)
    svd = (xc @ vt[:3].T).astype(np.float32)
    symbols = np.asarray(meta["sg_hm"])[rows].astype(str)[reps]
    z_nodes, fit_stats = refine_from_svd(svd[reps], knn_idx, knn_dist, symbols)
    embedding = z_nodes[code]

    np.save(OUT / "embedding.npy", np.ascontiguousarray(embedding))
    np.save(OUT / "svd.npy", svd)
    np.save(OUT / "row_index.npy", rows.astype(np.int64))
    np.save(OUT / "knn_idx.npy", knn_idx)
    np.save(OUT / "knn_dist.npy", knn_dist)
    np.save(OUT / "rep_row.npy", reps.astype(np.int64))
    np.save(OUT / "node_of_row.npy", code.astype(np.int32))
    stats = {
        "feature": "exact Selling-closure distance, boundary_rel=0",
        "candidates": "sorted linear key, expanded until the k-th closure neighbour is inside the candidate radius",
        "k": K,
        "n_cells": int(len(rows)),
        "n_lattices": int(len(reps)),
        "uncertified": int(uncertified),
        "gap_median": float(np.median(gap)),
        "gap_p90": float(np.quantile(gap, 0.9)),
        "gap_max": float(gap.max()),
        "stress_svd": fit_stats["stress_svd"],
        "stress_refined": fit_stats["stress_refined"],
        "off_line_svd": fit_stats["off_line_svd"],
        "off_line": fit_stats["off_line"],
        "anchor": fit_stats["w_anchor"],
        "mean_move": fit_stats["mean_move"],
        "max_move": fit_stats["max_move"],
        "svd_variance_fraction": (singular[:3] ** 2 / (singular ** 2).sum()).tolist(),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    if "--all" in sys.argv:
        build_all()
    else:
        main()
