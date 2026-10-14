#!/usr/bin/env python3
"""Freeze the component explorer into the static cell-embed site.

Coordinates are the classical MDS in ``linear_embed/out/closure_all``.
Neighbours are that directory's Selling-closure graph, so the page can
draw an Isomap of a lasso without a server. Selling cells, UniProt, and
deposition groups come from the PDB cell table.

Usage::

    python linear_embed/build_explorer_site.py --out site
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agentsg" / "src"))
sys.path.insert(0, str(ROOT / "linear_embed"))

from agentsg.cell.deposit_group import deposition_group_label  # noqa: E402
from agentsg.explorer.build_static import (  # noqa: E402
    _file_record,
    _git,
    _json_bytes,
    _string_table,
    _version,
    _write,
)
from embed import DEFAULT_PDB  # noqa: E402
from explorer3d import SAVED, component_view  # noqa: E402
from query import selling_parameters  # noqa: E402

PART = 24 * 1024 * 1024


def _rows(array: np.ndarray, dtype: str) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=dtype)


def _write_parts(directory: Path, stem: str, array: np.ndarray, dtype: str) -> list[Path]:
    array = _rows(array, dtype)
    width = int(np.prod(array.shape[1:])) if array.ndim > 1 else 1
    step = max(1, PART // (array.dtype.itemsize * width))
    paths = []
    start = 0
    while start < len(array):
        path = directory / f"{stem}_{len(paths)}.bin"
        _write(path, array[start:start + step].tobytes())
        paths.append(path)
        start += step
    return paths


def _protein(item) -> tuple[str, str, int]:
    if isinstance(item, dict):
        acc, name, residues = item.get("acc"), item.get("name"), item.get("n_res")
    else:
        acc, name, residues = item[0], item[1], item[2]
    acc = str(acc or "")
    return acc, str(name or acc), int(residues or 0)


def _lookup(pdb_ids: np.ndarray):
    import json

    import duckdb

    db = duckdb.connect(str(DEFAULT_PDB), read_only=True)
    rows = db.execute(
        "SELECT pdb_id, a, b, c, alpha, beta, gamma, sg_hm, uniprots, deposit_group FROM cells"
    ).fetchall()
    db.close()
    found = {str(row[0]).upper(): row for row in rows}
    selling = np.empty((len(pdb_ids), 6), dtype=np.float32)
    proteins = []
    deposit = []
    missing = 0
    for i, pdb_id in enumerate(pdb_ids):
        row = found.get(str(pdb_id).upper())
        if row is None:
            missing += 1
            selling[i] = np.nan
            proteins.append([])
            deposit.append("")
            continue
        packed = selling_parameters(row[1:7], row[7])
        selling[i] = (
            packed["a"], packed["b"], packed["c"],
            packed["alpha"], packed["beta"], packed["gamma"],
        )
        raw = row[8] or []
        if isinstance(raw, str):
            raw = json.loads(raw)
        items = [_protein(item) for item in raw]
        items.sort(key=lambda packed_item: -packed_item[2])
        proteins.append(items)
        group = str(row[9] or "").strip()
        deposit.append(deposition_group_label(group) if group else "")
        if i and i % 50000 == 0:
            print(f"  selling {i:,}", flush=True)
    if missing:
        print(f"  {missing:,} ids had no cell row", flush=True)
    return selling, proteins, deposit


def build(out: Path) -> None:
    print("loading the component view", flush=True)
    view = component_view()
    pdb_id = [str(item) for item in view["pdb_id"]]
    n = len(pdb_id)
    print(f"selling-reducing {n:,} cells", flush=True)
    selling, proteins, deposit = _lookup(view["pdb_id"])
    node = np.load(SAVED / "node_of_row.npy")
    knn_idx = np.load(SAVED / "knn_idx.npy")
    knn_dist = np.load(SAVED / "knn_dist.npy")
    if len(node) != n:
        raise SystemExit(f"node_of_row has {len(node)} rows and the view has {n}")
    if knn_idx.shape != knn_dist.shape:
        raise SystemExit("knn index and distance shapes differ")

    data = out / "data"
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    for name in ("app.js", "d3.min.js"):
        stale = out / name
        if stale.is_file():
            stale.unlink()

    header = np.array([n], dtype="<u4").tobytes()
    header += np.ascontiguousarray(view["sys_id"], dtype=np.uint8).tobytes()
    header += np.ascontiguousarray(view["cent_id"], dtype=np.uint8).tobytes()
    header += b"\0" * ((4 - (len(header) % 4)) % 4)
    points = header + _rows(view["xyz"], "<f4").tobytes()
    _write(data / "points.bin", points)
    _write(data / "cells.bin", _rows(selling, "<f4").tobytes())
    _write(data / "node_of_row.bin", _rows(node, "<i4").tobytes())
    idx_parts = _write_parts(data, "knn_idx", knn_idx, "<i4")
    dist_parts = _write_parts(data, "knn_dist", knn_dist, "<f4")

    off, acc_id, name_id, n_res, accs, names = _string_table(proteins)
    proteins_blob = b"".join([
        _rows(off, "<i4").tobytes(),
        _rows(acc_id, "<i4").tobytes(),
        _rows(name_id, "<i4").tobytes(),
        _rows(n_res, "<i4").tobytes(),
    ])
    _write(data / "proteins.bin", proteins_blob)
    _write(data / "protein_acc.json", _json_bytes(accs))
    _write(data / "protein_name.json", _json_bytes(names))

    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "n": n,
        "systems": ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"],
        "systemColors": ["#4c78a8", "#f58518", "#e45756", "#54a24b", "#b279a2", "#9d755d", "#72b7b2"],
        "centerings": ["P", "C", "I", "F", "R"],
        "centeringColors": ["#e45756", "#f58518", "#4c78a8", "#54a24b", "#b279a2"],
        "variance": [float(v) for v in view["frac"]],
        "pdb_id": pdb_id,
        "sg_hm": [str(item) for item in view["sg_hm"]],
        "deposit": deposit,
        "proteins": {"off": n + 1, "m": int(len(acc_id))},
        "note": view["note"],
        "built": built,
    }
    _write(data / "meta.json", _json_bytes(meta))

    def records(paths, dtype, shape):
        return [_file_record(path, dtype=dtype, shape=shape) for path in paths]

    files = {
        "points": records([data / "points.bin"], "bytes", [len(points)]),
        "cells": records([data / "cells.bin"], "float32", [n, 6]),
        "meta": records([data / "meta.json"], "json", [n]),
        "proteins": records([data / "proteins.bin"], "int32", [n + 1, int(len(acc_id))]),
        "protein_acc": records([data / "protein_acc.json"], "json", [len(accs)]),
        "protein_name": records([data / "protein_name.json"], "json", [len(names)]),
        "node_of_row": records([data / "node_of_row.bin"], "int32", [n]),
        "knn_idx": records(idx_parts, "int32", list(knn_idx.shape)),
        "knn_dist": records(dist_parts, "float32", list(knn_dist.shape)),
    }
    manifest = {
        "agentsg": _version(ROOT),
        "built": built,
        "byte_order": "little",
        "files": files,
        "git": _git(ROOT),
        "k": int(knn_idx.shape[1]),
        "n": n,
        "n_lattices": int(knn_idx.shape[0]),
        "neighbours": "k Selling-closure neighbours among distinct lattices. Index -1 is padding.",
    }
    _write(data / "manifest.json", _json_bytes(manifest))
    shutil.copyfile(ROOT / "linear_embed" / "explorer3d.html", out / "index.html")
    (out / "_headers").write_text(
        "/data/*\n"
        "  Cache-Control: public, max-age=86400\n"
        "\n"
        "/index.html\n"
        "  Cache-Control: no-cache\n"
    )
    print(f"wrote {out}  n={n:,} lattices={knn_idx.shape[0]:,}", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "site")
    args = parser.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
