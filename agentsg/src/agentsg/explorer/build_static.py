"""Freeze ``cells.npz`` into the static cell-embed site.

The page is the existing explorer: lasso, crystal-system and centring filters,
and the Selling-cell drawing. Coordinates are the geo 0.1 plane. Neighbours
are the k nearest points on that plane.

Usage::

    python -m agentsg.explorer.build_static --out site/ --k 25
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LIMIT = 25 * 1024 * 1024

SYSTEMS = (
    "triclinic", "monoclinic", "orthorhombic", "tetragonal",
    "trigonal", "hexagonal", "cubic",
)
SYS_COLORS = ["#4c78a8", "#f58518", "#e45756", "#54a24b", "#b279a2", "#9d755d", "#72b7b2"]
CENTERINGS = ("P", "C", "I", "F", "R")
CENT_COLORS = ["#e45756", "#f58518", "#4c78a8", "#54a24b", "#b279a2"]
ENCODER = "geo 0.1"


def repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").is_file() and (path / "linear_embed").is_dir():
            return path
    raise SystemExit("repository root not found")


def plane_neighbors(xy: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k nearest other points. Index -1 and a NaN distance pad a short row."""
    from scipy.spatial import cKDTree

    xy = np.ascontiguousarray(xy, dtype=np.float64)
    n = int(len(xy))
    out_i = np.full((n, k), -1, dtype=np.int32)
    out_d = np.full((n, k), np.nan, dtype=np.float32)
    if n <= 1 or k <= 0:
        return out_i, out_d
    kk = min(n, k + 1)
    dist, idx = cKDTree(xy).query(xy, k=kk)
    dist = np.reshape(dist, (n, kk))
    idx = np.reshape(idx, (n, kk))
    for i in range(n):
        chosen = []
        dropped = False
        for j in range(kk):
            nb = int(idx[i, j])
            if not dropped and nb == i:
                dropped = True
                continue
            chosen.append((float(dist[i, j]), nb))
            if len(chosen) == k:
                break
        chosen.sort(key=lambda item: (item[0], item[1]))
        for j, (distance, nb) in enumerate(chosen):
            out_i[i, j] = nb
            out_d[i, j] = np.float32(distance)
    return out_i, out_d


def _le(arr: np.ndarray, dtype: str) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.dtype(dtype).newbyteorder("<"))


def _write(path: Path, blob: bytes) -> None:
    if len(blob) >= LIMIT:
        raise SystemExit(f"{path.name} is {len(blob)} bytes, the limit is {LIMIT}")
    path.write_bytes(blob)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _version(root: Path) -> str:
    for line in (root / "pyproject.toml").read_text().splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _git(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _selling_rows(cell: np.ndarray, sg_hm: np.ndarray) -> np.ndarray:
    root = repo_root()
    linear = str(root / "linear_embed")
    if linear not in sys.path:
        sys.path.insert(0, linear)
    from query import selling_parameters

    out = np.empty((len(cell), 6), dtype=np.float64)
    for i, (row, symbol) in enumerate(zip(cell, sg_hm)):
        packed = selling_parameters(tuple(float(x) for x in row), str(symbol))
        out[i] = (packed["a"], packed["b"], packed["c"], packed["alpha"], packed["beta"], packed["gamma"])
        if i and i % 50000 == 0:
            print(f"  selling {i:,}", flush=True)
    return out.astype(np.float32)


def _string_table(rows: list[list[tuple]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    acc_of = {}
    name_of = {}
    accs: list[str] = []
    names: list[str] = []
    off = np.zeros(len(rows) + 1, dtype=np.int32)
    total = sum(len(row) for row in rows)
    acc_id = np.zeros(total, dtype=np.int32)
    name_id = np.zeros(total, dtype=np.int32)
    n_res = np.zeros(total, dtype=np.int32)
    cursor = 0
    for i, row in enumerate(rows):
        for acc, name, residues in row:
            if acc not in acc_of:
                acc_of[acc] = len(accs)
                accs.append(acc)
            if name not in name_of:
                name_of[name] = len(names)
                names.append(name)
            acc_id[cursor] = acc_of[acc]
            name_id[cursor] = name_of[name]
            n_res[cursor] = int(residues)
            cursor += 1
        off[i + 1] = cursor
    return off, acc_id, name_id, n_res, accs, names


def _pack_points(xy: np.ndarray, system: np.ndarray, centring: np.ndarray) -> tuple[bytes, int]:
    n = len(xy)
    header = np.array([n], dtype="<u4").tobytes()
    header += np.ascontiguousarray(system, dtype=np.uint8).tobytes()
    header += np.ascontiguousarray(centring, dtype=np.uint8).tobytes()
    header += b"\0" * ((4 - (len(header) % 4)) % 4)
    blob = header + _le(xy, "<f4").tobytes()
    return blob, len(header)


def _file_record(path: Path, *, dtype: str, shape) -> dict:
    return {
        "name": path.name,
        "dtype": dtype,
        "shape": list(shape),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build(plane: Path, out: Path, k: int) -> dict:
    root = repo_root()
    linear = str(root / "linear_embed")
    if linear not in sys.path:
        sys.path.insert(0, linear)
    from embed import centring_ids, crystal_system_ids

    if not plane.is_file():
        raise SystemExit(f"missing {plane}; run linear_embed/export_plane.py first")
    data = np.load(plane, allow_pickle=True)
    xy = np.asarray(data["xy"], dtype=np.float32)
    pdb_id = [str(item) for item in data["pdb_id"]]
    sg_hm = [str(item).strip() for item in data["sg_hm"]]
    sg_number = np.asarray(data["sg_number"], dtype=np.int32)
    n = len(pdb_id)
    if len(xy) != n:
        raise SystemExit(f"{plane} xy has {len(xy)} rows and {n} ids")
    system = crystal_system_ids(sg_number).astype(np.uint8)
    centring = centring_ids(np.asarray(sg_hm, dtype=object)).astype(np.uint8)
    deposited = np.column_stack([data["a"], data["b"], data["c"], data["alpha"], data["beta"], data["gamma"]])
    print(f"selling-reducing {n:,} cells", flush=True)
    selling = _selling_rows(deposited, np.asarray(sg_hm, dtype=object))

    proteins = [list(row) for row in data["uniprots"]]
    uni_acc = []
    uni_name = []
    for row in proteins:
        if row:
            uni_acc.append(str(row[0][0]))
            uni_name.append(str(row[0][1] or row[0][0]))
        else:
            uni_acc.append("")
            uni_name.append("")
    deposit = []
    for item in data["deposit_group"]:
        group = str(item or "").strip()
        deposit.append(f"DEPOSITION_GROUP_{group}" if group else "")

    print(f"neighbours k={k}", flush=True)
    nbr_i, nbr_d = plane_neighbors(xy, k)

    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    points, xy_offset = _pack_points(xy, system, centring)
    _write(data_dir / "points.bin", points)
    _write(data_dir / "cells.bin", _le(selling, "<f4").tobytes())
    _write(data_dir / "nbrs_index.bin", _le(nbr_i, "<i4").tobytes())
    _write(data_dir / "nbrs_dist.bin", _le(nbr_d, "<f4").tobytes())

    off, acc_id, name_id, n_res, accs, names = _string_table(proteins)
    proteins_blob = b"".join([
        _le(off, "<i4").tobytes(),
        _le(acc_id, "<i4").tobytes(),
        _le(name_id, "<i4").tobytes(),
        _le(n_res, "<i4").tobytes(),
    ])
    _write(data_dir / "proteins.bin", proteins_blob)
    _write(data_dir / "protein_acc.json", _json_bytes(accs))
    _write(data_dir / "protein_name.json", _json_bytes(names))

    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshot = str(np.asarray(data["snapshot"]))
    version = _version(root)
    meta = {
        "n": n,
        "encoders": [ENCODER],
        "xyOffset": {ENCODER: xy_offset},
        "systems": list(SYSTEMS),
        "systemColors": SYS_COLORS,
        "centerings": list(CENTERINGS),
        "centeringColors": CENT_COLORS,
        "pdb_id": pdb_id,
        "sg_hm": sg_hm,
        "uniAcc": uni_acc,
        "uniName": uni_name,
        "deposit": deposit,
        "proteins": {
            "off": n + 1,
            "m": int(len(acc_id)),
        },
        "built": built,
        "snapshot": snapshot,
        "agentsg": version,
        "plane": ENCODER,
    }
    _write(data_dir / "meta.json", _json_bytes(meta))

    files = {
        "points": [_file_record(data_dir / "points.bin", dtype="bytes", shape=[len(points)])],
        "cells": [_file_record(data_dir / "cells.bin", dtype="float32", shape=[n, 6])],
        "meta": [_file_record(data_dir / "meta.json", dtype="json", shape=[n])],
        "proteins": [_file_record(data_dir / "proteins.bin", dtype="int32", shape=[n + 1, int(len(acc_id))])],
        "protein_acc": [_file_record(data_dir / "protein_acc.json", dtype="json", shape=[len(accs)])],
        "protein_name": [_file_record(data_dir / "protein_name.json", dtype="json", shape=[len(names)])],
        "nbrs_index": [_file_record(data_dir / "nbrs_index.bin", dtype="int32", shape=[n, k])],
        "nbrs_dist": [_file_record(data_dir / "nbrs_dist.bin", dtype="float32", shape=[n, k])],
    }
    manifest = {
        "agentsg": version,
        "built": built,
        "byte_order": "little",
        "files": files,
        "git": _git(root),
        "k": int(k),
        "n": n,
        "pdb_snapshot": snapshot,
        "plane": ENCODER,
        "neighbours": "k nearest other points on the geo 0.1 plane, Euclidean. Index -1 is padding.",
    }
    _write(data_dir / "manifest.json", _json_bytes(manifest))
    print(f"wrote {data_dir}  n={n:,} k={k}", flush=True)
    return manifest


def main():
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "site")
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--plane", type=Path, default=root / "linear_embed" / "out" / "geo_0.1" / "cells.npz")
    args = parser.parse_args()
    if args.k < 1:
        raise SystemExit("--k must be at least 1")
    build(args.plane, args.out, args.k)


if __name__ == "__main__":
    main()
