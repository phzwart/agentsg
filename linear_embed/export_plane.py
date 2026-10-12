#!/usr/bin/env python3
"""Place every current PDB cell on the frozen geo 0.1 plane.

The checkpoint is ``linear_embed/out/geo_0.1/model.pt``. Input is the same
volume-normalised sorted linear key that checkpoint was trained on. Coordinates
already in ``embedding.npy`` are copied through by PDB id, so those points stay
put. Cells that are not on that plane keep the coordinates this run produces.

Writes ``linear_embed/out/geo_0.1/cells.npz``.

Usage (leanmap venv)::

    PYTHONPATH="agentsg/src:.venv/lib/python3.12/site-packages" \\
    KMP_DUPLICATE_LIB_OK=TRUE \\
    ../leanmap/.venv/bin/python linear_embed/export_plane.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from embed import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_PDB,
    OUT,
    _ensure_leanmap,
    primitive_volume_scale,
)

MODEL = OUT / "geo_0.1" / "model.pt"
OLD_XY = OUT / "geo_0.1" / "embedding.npy"
DEST = OUT / "geo_0.1" / "cells.npz"


def _uni_list(value):
    if not value:
        return []
    out = []
    for item in value:
        if isinstance(item, dict):
            acc, name, n_res = item.get("acc"), item.get("name"), item.get("n_res")
        else:
            acc, name, n_res = item[0], item[1], item[2]
        out.append((str(acc or ""), str(name or ""), int(n_res or 0)))
    return out


def _load_cells(pdb: Path):
    import duckdb

    if not pdb.is_file():
        raise SystemExit(f"PDB database not found: {pdb}")
    db = duckdb.connect(str(pdb), read_only=True)
    rows = db.execute(
        "SELECT pdb_id, a, b, c, alpha, beta, gamma, volume, sg_number, sg_hm, "
        "l0, l1, l2, l3, l4, l5, deposit_group, uniprots FROM cells "
        "WHERE l0 IS NOT NULL ORDER BY pdb_id"
    ).fetchall()
    meta = dict(db.execute("SELECT key, value FROM schema_meta").fetchall())
    db.close()
    if not rows:
        raise SystemExit(f"{pdb} has no l0..l5")
    pdb_id = np.array([r[0] for r in rows], dtype="U4")
    cell = np.array([r[1:7] for r in rows], dtype=np.float64)
    volume = np.array([r[7] for r in rows], dtype=np.float64)
    sg_number = np.array([r[8] for r in rows], dtype=np.int32)
    sg_hm = np.array([str(r[9] or "") for r in rows], dtype="U32")
    linear = np.array([r[10:16] for r in rows], dtype=np.float64)
    deposit = np.array([str(r[16] or "").strip() for r in rows], dtype=object)
    uniprots = np.array([_uni_list(r[17]) for r in rows], dtype=object)
    return pdb_id, cell, volume, sg_number, sg_hm, linear, deposit, uniprots, meta


def _old_tables():
    if not OLD_XY.is_file() or not DEFAULT_CACHE.is_file():
        raise SystemExit(f"missing {OLD_XY} or {DEFAULT_CACHE}")
    cache = np.load(DEFAULT_CACHE, allow_pickle=True)
    ids = np.asarray(cache["pdb_id"]).astype(str)
    xy = np.load(OLD_XY).astype(np.float32, copy=False)
    S = np.asarray(cache["S"], dtype=np.float32)
    if not (len(ids) == len(xy) == len(S)):
        raise SystemExit(
            f"{OLD_XY} has {len(xy)} rows, the key cache has {len(ids)} ids and {len(S)} keys"
        )
    return {pid: i for i, pid in enumerate(ids)}, xy, S


def _embed(S: np.ndarray) -> tuple[np.ndarray, str]:
    _ensure_leanmap()
    import torch
    from leanmap import load_plane

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"embedding {len(S):,} keys on {device}", flush=True)
    plane = load_plane(MODEL, device=device)
    with torch.no_grad():
        Z, _score = plane.embed(torch.as_tensor(S, dtype=torch.float32))
    return Z.detach().cpu().numpy().astype(np.float32, copy=False), device


def main():
    if not MODEL.is_file():
        raise SystemExit(f"geo 0.1 checkpoint not found: {MODEL}")
    pdb_id, cell, volume, sg_number, sg_hm, linear, deposit, uniprots, meta = _load_cells(DEFAULT_PDB)
    scale = primitive_volume_scale(volume, sg_hm)
    S = (linear / scale[:, None]).astype(np.float32)
    Z, device = _embed(S)

    old_at, old_xy, old_S = _old_tables()
    keep_i = []
    keep_j = []
    fresh = 0
    for i, pid in enumerate(pdb_id):
        j = old_at.get(str(pid))
        if j is None:
            fresh += 1
        else:
            keep_i.append(i)
            keep_j.append(j)
    keep_i = np.asarray(keep_i, dtype=np.int64)
    keep_j = np.asarray(keep_j, dtype=np.int64)
    if len(keep_i):
        xy_delta = float(np.max(np.abs(Z[keep_i] - old_xy[keep_j])))
        key_delta = float(np.max(np.abs(S[keep_i] - old_S[keep_j])))
    else:
        xy_delta = 0.0
        key_delta = 0.0
    print(
        f"rows {len(Z):,}  already on the plane {len(keep_i):,}  new {fresh:,}  "
        f"max |Δxy| {xy_delta:.3e}  max |Δkey| {key_delta:.3e}",
        flush=True,
    )
    if key_delta == 0.0 and xy_delta != 0.0:
        raise SystemExit(
            "geo 0.1 moved a point whose key is unchanged; cells.npz was not written"
        )
    if len(keep_i):
        Z = np.array(Z, dtype=np.float32, copy=True)
        Z[keep_i] = old_xy[keep_j]
        held = float(np.max(np.abs(Z[keep_i] - old_xy[keep_j])))
        if held != 0.0:
            raise SystemExit("failed to keep the stored coordinates")

    snapshot = str(meta.get("uniprot_source", ""))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DEST,
        xy=np.ascontiguousarray(Z, dtype=np.float32),
        pdb_id=pdb_id,
        a=cell[:, 0].astype(np.float32),
        b=cell[:, 1].astype(np.float32),
        c=cell[:, 2].astype(np.float32),
        alpha=cell[:, 3].astype(np.float32),
        beta=cell[:, 4].astype(np.float32),
        gamma=cell[:, 5].astype(np.float32),
        sg_number=sg_number,
        sg_hm=sg_hm,
        deposit_group=deposit,
        uniprots=uniprots,
        lambda_geo=np.float32(0.1),
        model=np.array(str(MODEL)),
        snapshot=np.array(snapshot),
        built=np.array(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        device=np.array(device),
    )
    print(json.dumps({
        "n": int(len(Z)),
        "new": int(fresh),
        "old_max_abs_model": xy_delta,
        "old_max_abs_key": key_delta,
        "path": str(DEST),
        "snapshot": snapshot,
    }), flush=True)


if __name__ == "__main__":
    main()
