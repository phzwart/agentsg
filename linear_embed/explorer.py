#!/usr/bin/env python3
"""Interactive explorer for the volume-normalised linear-key plane.

The cloud is drawn on a canvas. D3 handles zoom and axes. Click a point, or
type a PDB id: the panel lists the radius-0.02 shape ball and the
Selling-closure distance of the nearest few, both on unit-volume cells.

Usage::

    PYTHONPATH="agentsg/src:.venv/lib/python3.12/site-packages" \\
    ../leanmap/.venv/bin/python linear_embed/explorer.py

Then open http://127.0.0.1:8051
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from embed import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_PDB,
    OUT,
    centring_ids,
    crystal_system_ids,
    load_features,
)
from agentsg.cell.deposit_group import deposition_group_label  # noqa: E402
from query import (  # noqa: E402
    CLOSURE_K,
    RADIUS,
    _cells_for,
    deposited_conventional,
    selling_parameters,
)

SYSTEMS = (
    "triclinic", "monoclinic", "orthorhombic", "tetragonal",
    "trigonal", "hexagonal", "cubic",
)
SYS_COLORS = ["#4c78a8", "#f58518", "#e45756", "#54a24b", "#b279a2", "#9d755d", "#72b7b2"]
CENTERINGS = ("P", "C", "I", "F", "R")
CENT_COLORS = ["#e45756", "#f58518", "#4c78a8", "#54a24b", "#b279a2"]
UNIPROT_COLORS = [
    "#4c78a8", "#f58518", "#e45756", "#54a24b", "#b279a2", "#72b7b2",
    "#eeca3b", "#ff9da6", "#9d755d", "#1f77b4", "#2ca02c", "#9467bd",
]
UNIPROT_OTHER = -2
UNIPROT_NONE = -1
UNIPROT_TOP = 12
ENCODERS = {
    "geo 0.1": OUT / "geo_0.1" / "embedding.npy",
    "geo 0.5": OUT / "embedding.npy",
}


def load_bundle():
    S, meta = load_features(DEFAULT_PDB, DEFAULT_CACHE)
    planes = {}
    for name, path in ENCODERS.items():
        if not path.is_file():
            continue
        Z = np.load(path).astype(np.float32, copy=False)
        if len(Z) != len(S):
            raise SystemExit(f"{path} has {len(Z)} rows, the key cache has {len(S)}")
        planes[name] = Z
    if not planes:
        raise SystemExit("no embedding.npy next to the key cache")
    ids = np.asarray(meta["pdb_id"]).astype(str)
    sg = np.asarray(meta["sg_number"], dtype=np.int32)
    hm = np.asarray(meta["sg_hm"]).astype(str)
    return {
        "S": S,
        "planes": planes,
        "pdb_id": ids,
        "sg_hm": hm,
        "sg_number": sg,
        "sys_id": crystal_system_ids(sg),
        "cent_id": centring_ids(hm),
        "index": {pid.upper(): i for i, pid in enumerate(ids)},
    }


def shape_neighbors(bundle, q: int, *, radius: float, closure_k: int):
    """Rank the shape ball by sorted distance; closure-correct the nearest few."""
    from agentsg.cell.canonical import closure_distance
    from agentsg.cell.rootform import sorted_linear_distance

    S = bundle["S"]
    nbr = bundle["tree"].query_ball_point(S[q], r=radius)
    nbr = [int(j) for j in nbr if int(j) != q]
    if not nbr:
        return []
    ids = bundle["pdb_id"]
    qid = str(ids[q])
    ball_ids = [str(ids[j]) for j in nbr]
    packed = _cells_for(DEFAULT_PDB, [qid, *ball_ids])
    cell_q = packed.get(qid)
    if cell_q is None:
        return []
    ranked = []
    for j in nbr:
        jid = str(ids[j])
        cell_j = packed.get(jid)
        if cell_j is None:
            continue
        ranked.append((sorted_linear_distance(cell_q, cell_j), j, jid, cell_j))
    ranked.sort()
    rows = []
    for n, (sorted_d, j, jid, cell_j) in enumerate(ranked):
        closure = None
        tweak = None
        if n < closure_k:
            _P, aligned, _resid = closure_distance(cell_q, cell_j, boundary_rel=0.0)
            closure = float(aligned)
            tweak = closure - sorted_d
        rows.append({
            "pdb_id": jid,
            "sg": str(bundle["sg_hm"][j]),
            "system": SYSTEMS[int(bundle["sys_id"][j])],
            "sorted": round(sorted_d, 4),
            "closure": None if closure is None else round(closure, 4),
            "tweak": None if tweak is None else round(tweak, 4),
            "index": j,
        })
    return rows


def longest_uniprot(uniprots):
    """The first list element covers the most residues. Empty means no UniProt."""
    if not uniprots:
        return "", ""
    top = uniprots[0]
    acc = str(top["acc"] or "")
    name = str(top["name"] or acc)
    return acc, name


def protein_label(uniprots):
    """Longest UniProt on the entry. Empty when the entry has none."""
    return longest_uniprot(uniprots)


def _cells_have_column(db, name):
    return any(row[0] == name for row in db.execute("DESCRIBE cells").fetchall())


def deposit_label(group):
    """``(DEPOSITION_GROUP_<id>, same label)``, or empty when the entry has none."""
    group = str(group or "").strip()
    if not group:
        return "", ""
    label = deposition_group_label(group)
    return label, label


def lookup_color_labels(pdb_path, ids, by):
    """Return ``(acc, name)`` for each id.

    ``by`` is ``uniprot`` (the longest accession) or ``deposit`` (the
    deposition-group id). A missing column or a missing value is ``("", "")``.
    """
    import duckdb

    if by not in ("uniprot", "deposit"):
        raise ValueError("by must be uniprot or deposit")
    column = "uniprots" if by == "uniprot" else "deposit_group"
    label_of = protein_label if by == "uniprot" else deposit_label
    wanted = [str(pid).upper() for pid in ids]
    found = {}
    db = duckdb.connect(str(pdb_path), read_only=True)
    try:
        if not _cells_have_column(db, column):
            return [("", "")] * len(wanted)
        db.execute("CREATE TEMP TABLE want (pdb_id VARCHAR)")
        db.executemany("INSERT INTO want VALUES (?)", [(pid,) for pid in wanted])
        rows = db.execute(
            f"SELECT w.pdb_id, c.{column} FROM want w "
            "LEFT JOIN cells c ON c.pdb_id = w.pdb_id"
        ).fetchall()
    finally:
        db.close()
    for row in rows:
        found[str(row[0])] = label_of(row[1])
    return [found.get(pid, ("", "")) for pid in wanted]


def classify_labels(labels, *, kind="uniprot"):
    """Paint the largest labels. The rest share one grey, and missing labels another.

    ``labels`` is ``(acc, name)`` in request order. An empty accession is missing.
    ``kind`` is ``uniprot`` or ``deposit`` and only changes the legend words.
    """
    other_name = "other proteins" if kind == "uniprot" else "other groups"
    none_name = "no UniProt" if kind == "uniprot" else "no deposition group"
    counts = {}
    names = {}
    for acc, name in labels:
        counts[acc] = counts.get(acc, 0) + 1
        if acc and acc not in names:
            names[acc] = name or acc
    ranked = sorted((acc for acc in counts if acc), key=lambda acc: (-counts[acc], acc))
    color_of = {acc: i for i, acc in enumerate(ranked[:UNIPROT_TOP])}
    groups = []
    for acc, code in color_of.items():
        groups.append({
            "acc": acc,
            "name": names[acc],
            "n": counts[acc],
            "code": code,
            "color": UNIPROT_COLORS[code],
        })
    other_n = sum(counts[acc] for acc in ranked[UNIPROT_TOP:])
    if other_n:
        groups.append({
            "acc": "",
            "name": other_name,
            "n": other_n,
            "code": UNIPROT_OTHER,
            "color": "#d0d0d0",
        })
    none_n = counts.get("", 0)
    if none_n:
        groups.append({
            "acc": "",
            "name": none_name,
            "n": none_n,
            "code": UNIPROT_NONE,
            "color": "#666666",
        })
    codes = []
    for acc, _name in labels:
        if not acc:
            codes.append(UNIPROT_NONE)
        elif acc in color_of:
            codes.append(color_of[acc])
        else:
            codes.append(UNIPROT_OTHER)
    return {"groups": groups, "codes": codes}


def annotation_for_id(pdb_path, pdb_id):
    """UniProt list and deposition-group label, stored and returned separately."""
    import duckdb

    db = duckdb.connect(str(pdb_path), read_only=True)
    try:
        uni_col = _cells_have_column(db, "uniprots")
        group_col = _cells_have_column(db, "deposit_group")
        if not uni_col and not group_col:
            return [], ""
        selected = []
        if uni_col:
            selected.append("uniprots")
        if group_col:
            selected.append("deposit_group")
        row = db.execute(
            "SELECT " + ", ".join(selected) + " FROM cells WHERE pdb_id = ?",
            [str(pdb_id).upper()],
        ).fetchone()
    finally:
        db.close()
    if not row:
        return [], ""
    items = row[0] if uni_col else []
    uniprots = [
        {"acc": item["acc"], "name": item["name"], "n_res": item["n_res"]}
        for item in (items or [])
    ]
    group = ""
    if group_col:
        group = row[1] if uni_col else row[0]
    label = deposition_group_label(group) if str(group or "").strip() else ""
    return uniprots, label


def pack_points(bundle):
    """Little-endian blob: n, system codes, centring codes, then each plane."""
    n = len(bundle["pdb_id"])
    names = list(bundle["planes"])
    header = np.array([n], dtype="<u4").tobytes()
    header += bundle["sys_id"].astype(np.uint8).tobytes()
    header += bundle["cent_id"].astype(np.uint8).tobytes()
    pad = (4 - (len(header) % 4)) % 4
    header += b"\0" * pad
    chunks = [header]
    offset = len(header)
    xy_offset = {}
    for name in names:
        raw = np.ascontiguousarray(bundle["planes"][name], dtype="<f4").tobytes()
        xy_offset[name] = offset
        chunks.append(raw)
        offset += len(raw)
    meta = {
        "n": n,
        "encoders": names,
        "xyOffset": xy_offset,
        "systems": list(SYSTEMS),
        "systemColors": SYS_COLORS,
        "centerings": list(CENTERINGS),
        "centeringColors": CENT_COLORS,
        "pdb_id": bundle["pdb_id"].tolist(),
        "sg_hm": bundle["sg_hm"].tolist(),
    }
    return b"".join(chunks), meta


def serve(bundle):
    from scipy.spatial import cKDTree

    bundle["tree"] = cKDTree(bundle["S"])
    blob, meta = pack_points(bundle)
    d3 = (HERE / "d3.min.js").read_bytes()
    meta_bytes = json.dumps(meta).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path)
            if path.path == "/":
                page = (HERE / "explorer.html").read_bytes()
                self._send(200, page, "text/html; charset=utf-8")
                return
            if path.path == "/d3.min.js":
                self._send(200, d3, "text/javascript")
                return
            if path.path == "/meta":
                self._send(200, meta_bytes, "application/json")
                return
            if path.path == "/points":
                self._send(200, blob, "application/octet-stream")
                return
            if path.path == "/cell":
                q = parse_qs(path.query)
                pid = (q.get("id") or [""])[0].strip().upper()
                idx = bundle["index"].get(pid)
                if idx is None:
                    body = json.dumps({"error": f"{pid or 'empty id'} is not in the archive"}).encode()
                    self._send(404, body, "application/json")
                    return
                stored = str(bundle["pdb_id"][idx])
                row = deposited_conventional(DEFAULT_PDB, stored)
                if row is None:
                    body = json.dumps({"error": f"{stored} has no cell row"}).encode()
                    self._send(404, body, "application/json")
                    return
                card = selling_parameters(row[:6], row[6])
                card["pdb_id"] = stored
                card["index"] = idx
                card["uniprots"], card["group_label"] = annotation_for_id(DEFAULT_PDB, stored)
                self._send(200, json.dumps(card).encode(), "application/json")
                return
            if path.path == "/neighbors":
                q = parse_qs(path.query)
                pid = (q.get("id") or [""])[0].strip().upper()
                idx = bundle["index"].get(pid)
                if idx is None:
                    body = json.dumps({"error": f"{pid or 'empty id'} is not in the archive"}).encode()
                    self._send(404, body, "application/json")
                    return
                rows = shape_neighbors(bundle, idx, radius=RADIUS, closure_k=CLOSURE_K)
                body = json.dumps({
                    "index": idx,
                    "status": (
                        f"{pid}  {bundle['sg_hm'][idx]}  "
                        f"{len(rows)} cells within {RADIUS} on the shape key"
                    ),
                    "neighbors": rows,
                }).encode()
                self._send(200, body, "application/json")
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path)
            if path.path not in ("/labels", "/uniprot"):
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 8_000_000:
                body = json.dumps({"error": "expected a JSON body of PDB ids"}).encode()
                self._send(400, body, "application/json")
                return
            try:
                payload = json.loads(self.rfile.read(length))
                ids = payload.get("ids") or []
                if not isinstance(ids, list):
                    raise ValueError("ids must be a list")
                by = payload.get("by") or "uniprot"
                if by not in ("uniprot", "deposit"):
                    raise ValueError("by must be uniprot or deposit")
            except (json.JSONDecodeError, ValueError) as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self._send(400, body, "application/json")
                return
            labels = lookup_color_labels(DEFAULT_PDB, ids, by)
            self._send(200, json.dumps(classify_labels(labels, kind=by)).encode(), "application/json")

    print(f"loaded {len(bundle['pdb_id']):,} cells, encoders: {', '.join(bundle['planes'])}")
    server = ThreadingHTTPServer(("127.0.0.1", 8051), Handler)
    print("explorer on http://127.0.0.1:8051")
    server.serve_forever()


def main():
    serve(load_bundle())


if __name__ == "__main__":
    main()
