"""Explorer for a Crystallography Open Database cell archive.

Three panels show the first three principal components of the
volume-normalised sorted linear key (the key divided by the cube root of the
primitive-cell volume). The fourth panel is a rotatable view of those axes,
or an Isomap of a lasso on that same neighbour graph. A click shows the deposited cell,
the Selling cell, and the chemical formula. The page does not read UniProt
accessions or deposition-group ids.

Usage::

    python -m agentsg.cell.cod_explorer data/cod_cells.duckdb

Then open http://127.0.0.1:8053
"""
from __future__ import annotations

import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..space_groups import SPACE_GROUPS

PAGE = Path(__file__).with_name("cod_explorer.html")
HOST = "127.0.0.1"
PORT = 8053
NEIGHBORS = 8
KNN = 16

SYSTEMS = (
    "triclinic", "monoclinic", "orthorhombic", "tetragonal",
    "trigonal", "hexagonal", "cubic", "unknown",
)
SYS_COLORS = [
    "#4c78a8", "#f58518", "#e45756", "#54a24b", "#b279a2", "#9d755d", "#72b7b2", "#888888",
]
CENTERINGS = ("P", "C", "I", "F", "R", "unknown")
CENT_COLORS = ["#e45756", "#f58518", "#4c78a8", "#54a24b", "#b279a2", "#888888"]
_SYSTEM_OF = {name: i for i, name in enumerate(SYSTEMS)}
_SYSTEM_BY_NUMBER = np.full(231, _SYSTEM_OF["unknown"], dtype=np.uint8)
for _number, _hm, _hall, _crystal in SPACE_GROUPS:
    _SYSTEM_BY_NUMBER[_number] = _SYSTEM_OF[_crystal]
_CENTRING_OF = {"P": 0, "A": 1, "B": 1, "C": 1, "I": 2, "F": 3, "R": 4, "H": 4}
# Deposited conventional volume over primitive volume. ``:R`` is primitive
# and is handled before this table is consulted.
_VOLUME_MULT = {"P": 1.0, "A": 2.0, "B": 2.0, "C": 2.0, "I": 2.0, "F": 4.0, "R": 3.0, "H": 3.0}


def _centring_code(symbol: str) -> int:
    text = str(symbol or "").strip().lstrip("-")
    if not text:
        return _CENTRING_OF_UNKNOWN
    return _CENTRING_OF.get(text[0].upper(), _CENTRING_OF_UNKNOWN)


_CENTRING_OF_UNKNOWN = len(CENTERINGS) - 1


def _as_int(values) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype == object:
        out = np.zeros(len(raw), dtype=np.int32)
        for i, value in enumerate(raw):
            if value is None:
                continue
            out[i] = int(value)
        return out
    return np.nan_to_num(raw.astype(np.float64), nan=0.0).astype(np.int32)


def _as_str(values) -> list[str]:
    return ["" if value is None else str(value) for value in values]


def system_codes(sg_number: np.ndarray) -> np.ndarray:
    """Crystal-system code for each IT number. Missing numbers stay unknown."""
    number = _as_int(sg_number)
    codes = np.full(len(number), _SYSTEM_OF["unknown"], dtype=np.uint8)
    ok = (number >= 1) & (number <= 230)
    codes[ok] = _SYSTEM_BY_NUMBER[number[ok]]
    return codes


def centring_codes(sg_hm) -> np.ndarray:
    return np.fromiter((_centring_code(symbol) for symbol in sg_hm), dtype=np.uint8, count=len(sg_hm))


def primitive_volume_scale(volume, sg_hm) -> np.ndarray:
    """Cube root of the primitive-cell volume.

    ``volume`` is the deposited cell. The linear key is computed on the
    primitive lattice, so a centred cell is divided by its centring
    multiplicity first. A COD ``:R`` symbol is primitive.
    """
    from .cod import centring_symbol

    mult = np.empty(len(sg_hm), dtype=np.float64)
    for i, symbol in enumerate(sg_hm):
        centred = centring_symbol(symbol) or ""
        text = str(centred).strip().lstrip("-")
        letter = text[0].upper() if text else "P"
        mult[i] = _VOLUME_MULT.get(letter, 1.0)
    vol = np.asarray(volume, dtype=np.float64)
    return (vol / mult) ** (1.0 / 3.0)


def volume_normalised_keys(linear, volume, sg_hm) -> np.ndarray:
    """``sorted_linear_key / V_primitive**(1/3)``, dimensionless."""
    scale = primitive_volume_scale(volume, sg_hm)
    linear = np.asarray(linear, dtype=np.float64)
    out = np.zeros_like(linear)
    good = np.isfinite(scale) & (scale > 0)
    out[good] = linear[good] / scale[good, None]
    return out


def principal_axes(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First three principal components of the volume-normalised key."""
    centered = np.asarray(keys, dtype=np.float64)
    centered -= centered.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0, None)
    total = float(values.sum()) or 1.0
    take = min(3, vectors.shape[1])
    xyz = np.zeros((len(centered), 3), dtype=np.float32)
    xyz[:, :take] = (centered @ vectors[:, order[:take]]).astype(np.float32)
    frac = np.zeros(3, dtype=np.float64)
    frac[:take] = values[:take] / total
    return np.ascontiguousarray(xyz), frac


class Archive:
    """In-memory plane and a read-only handle for one cell card at a time."""

    def __init__(self, path: Path):
        import duckdb

        self.path = Path(path)
        self._db = duckdb.connect(str(self.path), read_only=True)
        self._lock = threading.Lock()
        table = self._load()
        self.pdb_id = table["pdb_id"]
        self.sg_hm = table["sg_hm"]
        self.sg_number = table["sg_number"]
        self.sys_id = system_codes(self.sg_number)
        self.cent_id = centring_codes(self.sg_hm)
        self.keys = np.ascontiguousarray(table["keys"], dtype=np.float32)
        self.xyz, self.variance = principal_axes(self.keys)
        self.index = {pid.upper(): i for i, pid in enumerate(self.pdb_id)}
        from scipy.spatial import cKDTree

        self.tree = cKDTree(self.keys)
        self.knn_idx, self.knn_dist = self._neighbour_graph()
        self.node_of_row = np.arange(len(self.pdb_id), dtype=np.int64)

    def close(self):
        self._db.close()

    def _columns(self) -> set[str]:
        return {row[0] for row in self._db.execute("DESCRIBE cells").fetchall()}

    def _load(self) -> dict:
        have = self._columns()
        needed = (
            "pdb_id", "sg_number", "sg_hm", "volume",
            "l0", "l1", "l2", "l3", "l4", "l5",
            "formula", "red_a",
        )
        missing = [name for name in needed if name not in have]
        if missing:
            raise SystemExit(
                f"{self.path} is missing {', '.join(missing)}; "
                "cod_explorer reads a COD cell archive"
            )
        fetched = self._db.execute(
            "SELECT pdb_id, sg_number, sg_hm, volume, l0, l1, l2, l3, l4, l5 "
            "FROM cells ORDER BY pdb_id"
        ).fetchnumpy()
        sg_hm = _as_str(fetched["sg_hm"])
        linear = np.column_stack([fetched[f"l{i}"] for i in range(6)])
        keys = volume_normalised_keys(linear, fetched["volume"], sg_hm)
        return {
            "pdb_id": _as_str(fetched["pdb_id"]),
            "sg_hm": sg_hm,
            "sg_number": _as_int(fetched["sg_number"]),
            "keys": keys,
        }

    def pack(self) -> tuple[bytes, dict]:
        n = len(self.pdb_id)
        header = np.array([n], dtype="<u4").tobytes()
        header += np.ascontiguousarray(self.sys_id, dtype=np.uint8).tobytes()
        header += np.ascontiguousarray(self.cent_id, dtype=np.uint8).tobytes()
        header += b"\0" * ((4 - (len(header) % 4)) % 4)
        blob = header + np.ascontiguousarray(self.xyz, dtype="<f4").tobytes()
        meta = {
            "n": n,
            "xyzOffset": len(header),
            "variance": [round(float(v), 4) for v in self.variance],
            "systems": list(SYSTEMS),
            "systemColors": SYS_COLORS,
            "systemCounts": np.bincount(self.sys_id, minlength=len(SYSTEMS)).tolist(),
            "centerings": list(CENTERINGS),
            "centeringColors": CENT_COLORS,
            "centeringCounts": np.bincount(self.cent_id, minlength=len(CENTERINGS)).tolist(),
            "pdb_id": self.pdb_id,
            "sg_hm": self.sg_hm,
            "note": (
                "Principal components of the volume-normalised sorted linear key. "
                "Each key is divided by the cube root of the primitive-cell volume, "
                "so the axes are shape. The lower right panel turns those three axes "
                "in 3D, or draws an Isomap of the lasso on the same neighbours."
            ),
        }
        return blob, meta

    def _neighbour_graph(self) -> tuple[np.ndarray, np.ndarray]:
        """k nearest other cells in the volume-normalised key. Column -1 pads a short row."""
        n = len(self.keys)
        idx = np.full((n, KNN), -1, dtype=np.int32)
        dist = np.full((n, KNN), np.nan, dtype=np.float32)
        k = min(KNN, n - 1)
        if k <= 0:
            return idx, dist
        found_d, found_i = self.tree.query(self.keys, k=k + 1, workers=-1)
        found_i = np.array(found_i, dtype=np.int32, copy=True)
        found_d = np.array(found_d, dtype=np.float64, copy=True)
        if found_i.ndim == 1:
            found_i = found_i.reshape(-1, 1)
            found_d = found_d.reshape(-1, 1)
        rows = np.arange(n)[:, None]
        is_self = found_i == rows
        present = np.nonzero(is_self.any(axis=1))[0]
        if len(present):
            take = np.argmax(is_self[present], axis=1)
            hold_i = found_i[present, 0].copy()
            hold_d = found_d[present, 0].copy()
            found_i[present, 0] = found_i[present, take]
            found_d[present, 0] = found_d[present, take]
            found_i[present, take] = hold_i
            found_d[present, take] = hold_d
        width = min(k, found_i.shape[1] - 1)
        idx[:, :width] = found_i[:, 1:1 + width]
        dist[:, :width] = found_d[:, 1:1 + width]
        return idx, dist

    def cell_card(self, row: int) -> dict:
        pid = self.pdb_id[row]
        distances, indices = self.tree.query(self.keys[row], k=min(NEIGHBORS + 1, len(self.pdb_id)))
        if len(self.pdb_id) == 1:
            distances = [float(distances)]
            indices = [int(indices)]
        neighbors = []
        for distance, index in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
            index = int(index)
            if index == row:
                continue
            neighbors.append({
                "id": self.pdb_id[index],
                "sg_hm": self.sg_hm[index],
                "distance": round(float(distance), 4),
            })
        with self._lock:
            stored = self._db.execute(
                "SELECT a, b, c, alpha, beta, gamma, volume, sg_number, sg_hm, formula, "
                "sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma, "
                "red_a, red_b, red_c, red_alpha, red_beta, red_gamma "
                "FROM cells WHERE pdb_id = ?",
                [pid],
            ).fetchone()
        if stored is None:
            raise KeyError(pid)
        cell = _params(stored[0:6], stored[6])
        esd = _params(stored[10:16], None)
        selling = _params(stored[16:22], None)
        numeric = pid[3:] if pid.upper().startswith("COD") else pid
        return {
            "id": pid,
            "cod_id": numeric,
            "url": f"https://www.crystallography.net/cod/{numeric}.html",
            "formula": stored[9] or "",
            "sg_number": stored[7],
            "sg_hm": stored[8] or "",
            "system": SYSTEMS[int(self.sys_id[row])],
            "centring": CENTERINGS[int(self.cent_id[row])],
            "cell": cell,
            "esd": esd,
            "selling": selling,
            "neighbors": neighbors,
        }


def _params(values, volume) -> dict:
    names = ("a", "b", "c", "alpha", "beta", "gamma")
    out = {}
    for name, value in zip(names, values):
        out[name] = None if value is None else float(value)
    if volume is not None:
        out["volume"] = float(volume)
    return out


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
    """Isomap of one connected piece of the induced neighbour graph."""
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
    """Two Isomap coordinates of the lasso on its linear-key neighbour edges."""
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


def serve(archive: Archive, host: str, port: int) -> None:
    blob, meta = archive.pack()
    meta_bytes = json.dumps(meta).encode()
    page = PAGE.read_bytes()

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
                self._send(200, page, "text/html; charset=utf-8")
                return
            if path.path == "/meta":
                self._send(200, meta_bytes, "application/json")
                return
            if path.path == "/points":
                self._send(200, blob, "application/octet-stream")
                return
            if path.path == "/cell":
                query = parse_qs(path.query)
                token = (query.get("id") or [""])[0].strip().upper()
                if token.isdigit():
                    token = "COD" + token
                row = archive.index.get(token)
                if row is None:
                    body = json.dumps({"error": f"{token or 'empty id'} is not in this archive"}).encode()
                    self._send(404, body, "application/json")
                    return
                self._send(200, json.dumps(archive.cell_card(row)).encode(), "application/json")
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            if path != "/embed":
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            try:
                payload = json.loads(raw.decode()) if raw else {}
                indices = payload.get("indices", [])
                if not isinstance(indices, list):
                    raise ValueError("indices must be a list")
                result = isomap(indices, archive.knn_idx, archive.knn_dist, archive.node_of_row)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
                return
            self._send(200, json.dumps(result).encode(), "application/json")

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"COD explorer on http://{host}:{port}  ({len(archive.pdb_id):,} cells)", flush=True)
    httpd.serve_forever()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Explore a COD cell archive on the linear-key plane.")
    parser.add_argument("db", type=Path, help="COD cell database, for example data/cod_cells.duckdb")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args(argv)
    if not args.db.is_file():
        raise SystemExit(f"no database at {args.db}")
    archive = Archive(args.db)
    print(
        f"three components of {len(archive.pdb_id):,} cells, "
        f"PC1 {archive.variance[0]:.1%}, PC2 {archive.variance[1]:.1%}, "
        f"PC3 {archive.variance[2]:.1%}",
        flush=True,
    )
    serve(archive, args.host, args.port)


if __name__ == "__main__":
    main()
