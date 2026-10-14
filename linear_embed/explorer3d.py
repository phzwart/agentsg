#!/usr/bin/env python3
"""Four views of the classical MDS for every cell.

Three component planes in a 2x2 grid. The fourth panel is an Isomap of the
lasso on the Selling-closure neighbour graph. A lasso on that panel narrows
the selection. Plane lassos are united.

The coordinates are the first three principal components of the
volume-normalised sorted key, which is classical MDS of the Euclidean
key distance.

Usage::

    PYTHONPATH="agentsg/src:.venv/lib/python3.12/site-packages" \\
    ../leanmap/.venv/bin/python linear_embed/explorer3d.py

Then open http://127.0.0.1:8052
"""
from __future__ import annotations

import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from explorer import (  # noqa: E402
    annotation_for_id,
    classify_labels,
    lookup_color_labels,
)
from query import deposited_conventional, selling_parameters  # noqa: E402
from embed import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_PDB,
    SYSTEM_NAMES,
    centring_ids,
    crystal_system_ids,
    load_features,
)

SAVED = HERE / "out" / "closure_all"
SYS_COLORS = ["#4c78a8", "#f58518", "#e45756", "#54a24b", "#b279a2", "#9d755d", "#72b7b2"]
CENTERINGS = ("P", "C", "I", "F", "R")
CENT_COLORS = ["#e45756", "#f58518", "#4c78a8", "#54a24b", "#b279a2"]
PORT = 8052


def component_view():
    """Principal components of the volume-normalised sorted key, every cell."""
    for name in ("svd.npy", "row_index.npy", "knn_idx.npy", "knn_dist.npy", "node_of_row.npy"):
        if not (SAVED / name).is_file():
            raise SystemExit(f"missing {SAVED / name}; run linear_embed/closure_embed.py --all")
    S, meta = load_features(DEFAULT_PDB, DEFAULT_CACHE)
    del S
    sys_ids = crystal_system_ids(meta["sg_number"])
    keep = np.load(SAVED / "row_index.npy")
    xyz = np.ascontiguousarray(np.load(SAVED / "svd.npy"), dtype=np.float32)
    if xyz.shape != (len(keep), 3):
        raise SystemExit(f"{SAVED / 'svd.npy'} does not match row_index.npy")
    note = (
        "Classical MDS of the volume-normalised sorted key. "
        "The three axes are its first three principal components. "
        "The lower right panel is an Isomap of the lasso, and a lasso "
        "there narrows that selection."
    )
    centered = xyz.astype(np.float64) - xyz.mean(0)
    frac = centered.var(0)
    frac = frac / frac.sum()
    return {
        "xyz": np.ascontiguousarray(xyz, dtype=np.float32),
        "frac": np.asarray(frac[:3], dtype=np.float64),
        "pdb_id": np.asarray(meta["pdb_id"])[keep].astype(str),
        "sg_hm": np.asarray(meta["sg_hm"])[keep].astype(str),
        "sys_id": sys_ids[keep].astype(np.uint8),
        "cent_id": centring_ids(np.asarray(meta["sg_hm"])[keep]).astype(np.uint8),
        "note": note,
    }


def _closure_neighbours():
    """The certified 50-neighbour Selling-closure graph, stored per lattice."""
    saved = SAVED
    return (
        np.load(saved / "knn_idx.npy"),
        np.load(saved / "knn_dist.npy"),
        np.load(saved / "node_of_row.npy"),
    )


def _classical_mds(dist: np.ndarray) -> np.ndarray:
    """Two classical-MDS coordinates of a geodesic distance matrix."""
    d2 = np.asarray(dist, dtype=np.float64) ** 2
    row = d2.mean(axis=1, keepdims=True)
    gram = -0.5 * (d2 - row - row.T + d2.mean())
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    coords = np.zeros((len(dist), 2), dtype=np.float64)
    for axis in range(2):
        ev = float(evals[order[axis]])
        if ev <= 1e-12:
            break
        coords[:, axis] = evecs[:, order[axis]] * np.sqrt(ev)
    return coords


def _landmark_positions(dist_land: np.ndarray, dist_all: np.ndarray) -> np.ndarray:
    """Embed every point from its geodesic distances to the landmarks."""
    landmarks = _classical_mds(dist_land)
    mean_sq = np.mean(np.asarray(dist_land, dtype=np.float64) ** 2, axis=0)
    delta = np.asarray(dist_all, dtype=np.float64) ** 2 - mean_sq[:, None]
    finite = np.isfinite(dist_all).all(axis=0)
    delta = np.where(np.isfinite(delta), delta, 0.0)
    embedded = (-0.5 * np.linalg.pinv(landmarks) @ delta).T
    embedded[~finite] = 0.0
    return embedded


def _embed_component(graph) -> np.ndarray:
    """Isomap of one connected piece of the induced neighbour graph.

    A component of a few hundred lattices uses every geodesic. A larger one
    uses landmark Isomap: farthest-point landmarks, then the same classical
    MDS step from distances to those landmarks.
    """
    from scipy.sparse.csgraph import dijkstra, shortest_path

    n = graph.shape[0]
    if n == 1:
        return np.zeros((1, 2), dtype=np.float64)
    if n <= 450:
        dist = shortest_path(graph, directed=False, method="D")
        finite = np.isfinite(dist)
        if not finite.all():
            cap = float(np.max(dist[finite])) if finite.any() else 1.0
            dist = np.where(finite, dist, cap * 2.0)
        return _classical_mds(dist)
    count = min(160, n)
    chosen = np.empty(count, dtype=np.int32)
    rows = []
    nearest = np.full(n, np.inf)
    start = 0
    for i in range(count):
        chosen[i] = start
        dist = dijkstra(graph, directed=False, indices=int(start))
        rows.append(dist)
        nearest = np.minimum(nearest, dist)
        nearest[chosen[: i + 1]] = -np.inf
        start = int(np.argmax(nearest))
        if not np.isfinite(nearest[start]):
            chosen = chosen[: i + 1]
            break
    dist_all = np.vstack(rows)
    return _landmark_positions(dist_all[:, chosen], dist_all)


def isomap(indices, knn_idx, knn_dist, node_of_row):
    """Two Isomap coordinates of the lasso on its Selling-closure edges.

    Geodesics are shortest paths on the induced 50-neighbour graph. Classical
    MDS of those distances is the map. Lattices with no neighbour in the
    lasso stay at the origin.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    indices = np.asarray(indices, dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < len(node_of_row))]
    if len(indices) == 0:
        return {"n_cells": 0, "n_lattices": 0, "xy": ""}
    lattices, inverse = np.unique(node_of_row[indices], return_inverse=True)
    m = len(lattices)
    coords = np.zeros((m, 2), dtype=np.float64)
    n_edges = 0
    if m >= 2:
        member = np.full(knn_idx.shape[0], -1, dtype=np.int32)
        member[lattices] = np.arange(m, dtype=np.int32)
        nbrs = knn_idx[lattices]
        lengths = knn_dist[lattices].astype(np.float64, copy=False)
        local = np.full(nbrs.shape, -1, dtype=np.int32)
        valid = nbrs >= 0
        local[valid] = member[nbrs[valid]]
        same = local == np.arange(m, dtype=np.int32)[:, None]
        keep = (local >= 0) & np.isfinite(lengths) & (lengths > 0) & ~same
        src = np.repeat(np.arange(m, dtype=np.int32), knn_idx.shape[1])[keep.ravel()]
        dst = local.ravel()[keep.ravel()]
        weight = lengths.ravel()[keep.ravel()]
        if len(weight):
            ones = np.ones(len(weight), dtype=np.float64)
            rows = np.concatenate([src, dst])
            cols = np.concatenate([dst, src])
            graph = csr_matrix((np.concatenate([weight, weight]), (rows, cols)), shape=(m, m))
            mult = csr_matrix((np.concatenate([ones, ones]), (rows, cols)), shape=(m, m))
            graph.sum_duplicates()
            mult.sum_duplicates()
            graph = graph.multiply(mult.power(-1)).tocsr()
            graph.eliminate_zeros()
            n_edges = int(graph.nnz // 2)
            _n_comp, labels = connected_components(graph, directed=False)
            pieces = []
            singletons = []
            for label in range(int(labels.max()) + 1):
                nodes = np.flatnonzero(labels == label)
                if len(nodes) == 1:
                    singletons.append(int(nodes[0]))
                    continue
                xy = _embed_component(graph[nodes][:, nodes])
                pieces.append((nodes, xy - xy.mean(axis=0)))
            pieces.sort(key=lambda item: -len(item[0]))
            cursor = 0.0
            started = False
            for nodes, xy in pieces:
                width = float(xy[:, 0].max() - xy[:, 0].min()) if len(xy) else 0.0
                if started:
                    xy = xy.copy()
                    xy[:, 0] += cursor - float(xy[:, 0].min())
                coords[nodes] = xy
                cursor = float(coords[nodes, 0].max()) + 0.18 * max(width, 1.0)
                started = True
            if singletons and started:
                coords[singletons, 0] = float(coords[:, 0].min()) - 0.18
    cell = coords[inverse]
    return {
        "n_cells": int(len(indices)),
        "n_lattices": int(m),
        "n_edges": n_edges,
        "xy": base64.b64encode(np.ascontiguousarray(cell, dtype="<f4").tobytes()).decode(),
    }


def pack(view):
    n = len(view["pdb_id"])
    header = np.array([n], dtype="<u4").tobytes()
    header += view["sys_id"].tobytes()
    header += view["cent_id"].tobytes()
    header += b"\0" * ((4 - (len(header) % 4)) % 4)
    blob = header + np.ascontiguousarray(view["xyz"], dtype="<f4").tobytes()
    meta = {
        "n": n,
        "xyzOffset": len(header),
        "systems": list(SYSTEM_NAMES),
        "systemColors": SYS_COLORS,
        "centerings": list(CENTERINGS),
        "centeringColors": CENT_COLORS,
        "variance": [float(v) for v in view["frac"]],
        "pdb_id": view["pdb_id"].tolist(),
        "sg_hm": view["sg_hm"].tolist(),
        "note": view.get("note", ""),
    }
    return blob, meta


def serve(view):
    blob, meta = pack(view)
    page_path = HERE / "explorer3d.html"
    meta_bytes = json.dumps(meta).encode()
    print("loading the Selling-closure neighbours", flush=True)
    knn_idx, knn_dist, node_of_row = _closure_neighbours()

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
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, page_path.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/meta":
                self._send(200, meta_bytes, "application/json")
                return
            if path == "/points":
                self._send(200, blob, "application/octet-stream")
                return
            if path == "/cell":
                query = parse_qs(urlparse(self.path).query)
                pid = (query.get("id") or [""])[0].strip().upper()
                row = deposited_conventional(DEFAULT_PDB, pid)
                if row is None:
                    body = json.dumps({"error": f"{pid or 'empty id'} has no cell row"}).encode()
                    self._send(404, body, "application/json")
                    return
                card = selling_parameters(row[:6], row[6])
                for key in ("a", "b", "c", "alpha", "beta", "gamma"):
                    card[key] = float(card[key])
                card["pdb_id"] = pid
                card["uniprots"], card["group_label"] = annotation_for_id(DEFAULT_PDB, pid)
                self._send(200, json.dumps(card).encode(), "application/json")
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            if path not in ("/embed", "/labels"):
                self._send(404, b"not found", "text/plain")
                return
            try:
                payload = json.loads(raw.decode()) if raw else {}
                if path == "/embed":
                    result = isomap(payload.get("indices", []), knn_idx, knn_dist, node_of_row)
                else:
                    by = payload.get("by") or "uniprot"
                    if by not in ("uniprot", "deposit"):
                        raise ValueError("by must be uniprot or deposit")
                    ids = payload.get("ids") or []
                    if not isinstance(ids, list):
                        raise ValueError("ids must be a list")
                    result = classify_labels(lookup_color_labels(DEFAULT_PDB, ids, by), kind=by)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
                return
            self._send(200, json.dumps(result).encode(), "application/json")

    print(f"loaded {len(view['pdb_id']):,} cells")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"3D explorer on http://127.0.0.1:{PORT}")
    server.serve_forever()


def main():
    serve(component_view())


if __name__ == "__main__":
    main()
