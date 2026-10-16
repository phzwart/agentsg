"""Two-pass landmark workflow for a stream of unit cells.

Pass 1 clusters a subsample and stores one landmark catalog. Each landmark
is a submitted cell. Every landmark in a cluster shares that cluster's
medoid, also a submitted cell, and the reindexing operator onto the medoid
is the identity ``((1, 0, 0), (0, 1, 0), (0, 0, 1))``. The catalog also
stores that medoid's reindexing coset, and the symmetry branches when a
space-group symbol is given.

Pass 2 does not rebuild the closure. For each later cell it returns the
sorted-linear-key distance, in ångström, to every landmark, together with that
landmark's coset and branches.

Import::

    from agentsg.cell.landmark_workflow import build_landmarks, score_cells

    catalog = build_landmarks(subsample, sigma_deg=0.05, symmetry="P 1 21 1")
    scored = score_cells(catalog, frames)

``subsample`` and ``frames`` are ``(n, 6)`` arrays ``(a, b, c, alpha, beta,
gamma)``, or cctbx ``unit_cell`` objects (anything with ``.parameters()``).
Miller indices transform as a row, ``h @ P``. The identity leaves ``hkl``
unchanged.

CLI::

    python -m agentsg.cell.landmark_workflow build cells.csv --sigma-deg 0.05 \\
        --subsample 200 --symmetry "P 1 21 1" --out landmarks.json
    python -m agentsg.cell.landmark_workflow score landmarks.json queries.csv \\
        --out scores.json
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .primitive import primitive_cell
from .rootform import sorted_linear_key
from .serial_index import CertifyResult, _jsonable, cluster, cosets, ingest, landmarks

_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def as_cell_rows(cells) -> np.ndarray:
    """``(n, 6)`` array from rows or from objects with ``.parameters()``.

    A cctbx ``unit_cell`` implements ``parameters()`` as
    ``(a, b, c, alpha, beta, gamma)``. One cell may be a length-6 sequence.
    """
    if hasattr(cells, "parameters"):
        return np.asarray([list(cells.parameters())], dtype=np.float64)
    if isinstance(cells, np.ndarray) and np.issubdtype(cells.dtype, np.number):
        rows = np.asarray(cells, dtype=np.float64)
        if rows.ndim == 1:
            return rows.reshape(1, 6)
        return rows
    sequence = list(cells)
    if not sequence:
        return np.zeros((0, 6), dtype=np.float64)
    if hasattr(sequence[0], "parameters"):
        return np.asarray(
            [list(item.parameters()) for item in sequence], dtype=np.float64,
        )
    rows = np.asarray(sequence, dtype=np.float64)
    if rows.ndim == 1:
        return rows.reshape(1, 6)
    return rows


def _prepare(cells, spacegroups):
    rows = as_cell_rows(cells)
    prepared = []
    for index, row in enumerate(rows):
        cell = tuple(float(x) for x in row)
        symbol = None if spacegroups is None else spacegroups[index]
        if symbol:
            cell = tuple(float(x) for x in primitive_cell(cell, str(symbol)))
        prepared.append(cell)
    return prepared


def _choose(n: int, subsample, seed: int) -> np.ndarray:
    """Row indices. An int is a random draw; a sequence is an explicit one."""
    if subsample is None:
        return np.arange(n, dtype=np.int64)
    if isinstance(subsample, (int, np.integer)):
        if int(subsample) <= 0:
            raise ValueError("subsample must be positive")
        if int(subsample) >= n:
            return np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=int(subsample), replace=False))
    chosen = np.asarray(list(subsample), dtype=np.int64).reshape(-1)
    if chosen.size == 0:
        raise ValueError("subsample must be positive")
    if np.any(chosen < 0) or np.any(chosen >= n):
        raise ValueError("subsample index out of range")
    return np.unique(chosen)


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class CatalogCluster:
    cluster_id: int
    size: int
    reference_cell: tuple
    eta: float
    isometric: bool

    def to_dict(self):
        return _jsonable(self.__dict__)

    @classmethod
    def from_dict(cls, payload):
        return cls(
            int(payload["cluster_id"]),
            int(payload["size"]),
            _freeze(payload["reference_cell"]),
            float(payload["eta"]),
            bool(payload["isometric"]),
        )


@dataclass(frozen=True)
class CatalogLandmark:
    landmark_id: int
    cluster_id: int
    source_index: int
    cell: tuple
    key: tuple
    reference_cell: tuple
    to_center: tuple
    operators: tuple
    branches: tuple
    group_closed: bool

    def to_dict(self):
        return _jsonable(self.__dict__)

    @classmethod
    def from_dict(cls, payload):
        return cls(
            int(payload["landmark_id"]),
            int(payload["cluster_id"]),
            int(payload["source_index"]),
            _freeze(payload["cell"]),
            _freeze(payload["key"]),
            _freeze(payload["reference_cell"]),
            _freeze(payload["to_center"]),
            _freeze(payload["operators"]),
            _freeze(payload["branches"]),
            bool(payload["group_closed"]),
        )


@dataclass(frozen=True)
class LandmarkCatalog:
    """Pass-1 catalog. Every landmark cell is a submitted row; ``to_center`` is the identity."""

    sigma_deg: float
    sigma_len: float
    symmetry: str | None
    n_input: int
    subsample: tuple
    clusters: tuple
    landmarks: tuple

    def to_dict(self):
        return _jsonable({
            "sigma_deg": self.sigma_deg,
            "sigma_len": self.sigma_len,
            "symmetry": self.symmetry,
            "n_input": self.n_input,
            "subsample": self.subsample,
            "clusters": [item.to_dict() for item in self.clusters],
            "landmarks": [item.to_dict() for item in self.landmarks],
        })

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, payload):
        return cls(
            float(payload["sigma_deg"]),
            float(payload["sigma_len"]),
            payload.get("symmetry"),
            int(payload["n_input"]),
            _freeze(payload["subsample"]),
            tuple(CatalogCluster.from_dict(item) for item in payload["clusters"]),
            tuple(CatalogLandmark.from_dict(item) for item in payload["landmarks"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "LandmarkCatalog":
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class LandmarkHit:
    landmark_id: int
    cluster_id: int
    key_distance: float
    reference_cell: tuple
    to_center: tuple
    operators: tuple
    branches: tuple

    def to_dict(self):
        return _jsonable(self.__dict__)


@dataclass(frozen=True)
class ScoredCell:
    query_id: int
    hits: tuple

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "hits": [hit.to_dict() for hit in self.hits],
        }


@dataclass(frozen=True)
class ScoreBatch:
    symmetry: str | None
    queries: tuple

    def to_dict(self):
        return {
            "symmetry": self.symmetry,
            "queries": [query.to_dict() for query in self.queries],
        }

    def to_json(self) -> str:
        return json.dumps(_jsonable(self.to_dict()))


def build_landmarks(cells, sigma_deg, sigma_len=0.0, symmetry=None,
                    spacegroups=None, subsample=None, seed=0) -> LandmarkCatalog:
    """Pass 1. Cluster a subsample and store landmarks, centers, and cosets.

    Parameters
    ----------
    cells :
        ``(n, 6)`` array, or cctbx ``unit_cell`` objects.
    sigma_deg :
        Declared angular noise in degrees. It sets every tolerance in the index.
    sigma_len :
        Optional edge noise in ångström.
    symmetry :
        Space-group symbol. When set, each coset carries the branch list:
        proper automorphisms modulo that group's proper point group.
    spacegroups :
        Per-cell centering letter or symbol. Used only to drop to the primitive
        cell. ``None`` keeps every cell as given.
    subsample :
        How many input rows to cluster, or an explicit sequence of row indices.
        ``None`` uses all of them. A random draw can miss a lattice; pass indices
        when each lattice in the stream needs a center.
    seed :
        Seed for an integer subsample draw. Ignored when ``subsample`` is a
        sequence of indices.
    """
    rows = as_cell_rows(cells)
    chosen = _choose(len(rows), subsample, seed)
    groups = None if spacegroups is None else [spacegroups[int(i)] for i in chosen]
    ingested = ingest(
        rows[chosen], groups, sigma_deg=sigma_deg, sigma_len=sigma_len,
    )
    infos, _svd = cluster(ingested)
    marks = landmarks(infos, ingested)
    # Self-cosets only. Pairwise closure certification is serial_index.run.
    blank = (float("inf"),) * len(marks)
    certificate = CertifyResult(
        tuple(blank for _ in marks), tuple(blank for _ in marks), (), False, (),
    )
    cached = cosets(infos, marks, certificate, ingested, symmetry)
    coset_of = {item.landmark_id: item for item in cached.self_cosets}
    reference_of = {}
    for mark in marks:
        reference_of.setdefault(mark.cluster_id, mark.reference_cell)
    clusters = tuple(
        CatalogCluster(
            info.cluster_id,
            len(info.members),
            tuple(float(x) for x in reference_of[info.cluster_id]),
            float(info.eta),
            bool(info.isometric),
        )
        for info in infos
        if info.cluster_id in reference_of
    )
    found = []
    for mark in marks:
        cached_coset = coset_of[mark.landmark_id]
        found.append(CatalogLandmark(
            mark.landmark_id,
            mark.cluster_id,
            int(chosen[mark.cell_index]),
            tuple(float(x) for x in mark.cell),
            tuple(float(x) for x in mark.key),
            tuple(float(x) for x in mark.reference_cell),
            tuple(tuple(int(x) for x in row) for row in mark.to_center),
            tuple(cached_coset.operators),
            tuple(cached_coset.branches),
            bool(cached_coset.group_closed),
        ))
    return LandmarkCatalog(
        float(sigma_deg),
        float(sigma_len or 0.0),
        None if not symmetry else str(symmetry),
        int(len(rows)),
        tuple(int(i) for i in chosen),
        clusters,
        tuple(found),
    )


def score_cells(catalog: LandmarkCatalog, cells, spacegroups=None) -> ScoreBatch:
    """Pass 2. Distance to every landmark, and that landmark's reindexing coset.

    The distance is the sorted linear key, in ångström. ``to_center`` is the
    identity for every hit: that is the operator into the cluster-center
    setting. ``operators`` is the coset of that setting, and ``branches`` is
    the symmetry reduction stored when the catalog was built.
    """
    prepared = _prepare(cells, spacegroups)
    if not catalog.landmarks or not prepared:
        return ScoreBatch(
            catalog.symmetry,
            tuple(ScoredCell(i, ()) for i in range(len(prepared))),
        )
    query_keys = np.asarray(
        [sorted_linear_key(cell) for cell in prepared], dtype=np.float64,
    )
    landmark_keys = np.asarray(
        [mark.key for mark in catalog.landmarks], dtype=np.float64,
    )
    distance = np.linalg.norm(
        query_keys[:, None, :] - landmark_keys[None, :, :], axis=-1,
    )
    queries = []
    for query_id in range(len(prepared)):
        order = np.argsort(distance[query_id], kind="mergesort")
        hits = []
        for slot in order:
            mark = catalog.landmarks[int(slot)]
            hits.append(LandmarkHit(
                mark.landmark_id,
                mark.cluster_id,
                float(distance[query_id, int(slot)]),
                mark.reference_cell,
                mark.to_center,
                mark.operators,
                mark.branches,
            ))
        queries.append(ScoredCell(query_id, tuple(hits)))
    return ScoreBatch(catalog.symmetry, tuple(queries))


def reindex_hkl(hkl, operator) -> tuple:
    """Apply a coset operator to one Miller index. ``h' = h P``."""
    h, k, l = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
    return tuple(
        int(h * operator[0][j] + k * operator[1][j] + l * operator[2][j])
        for j in range(3)
    )


def _fmt_op(operator) -> str:
    return ",".join("".join(str(int(value)) for value in row) for row in operator)


def catalog_summary(catalog: LandmarkCatalog) -> str:
    """Short text report of a landmark catalog."""
    lines = [
        f"input {catalog.n_input}",
        f"subsample {len(catalog.subsample)}",
        f"sigma_deg {catalog.sigma_deg:g}",
        f"symmetry {catalog.symmetry}",
        f"clusters {len(catalog.clusters)}",
        f"landmarks {len(catalog.landmarks)}",
    ]
    for mark in catalog.landmarks:
        cell = " ".join(f"{value:.4g}" for value in mark.cell)
        lines.append(
            f"landmark {mark.landmark_id} cluster {mark.cluster_id} "
            f"source {mark.source_index} cell {cell} "
            f"to_center {_fmt_op(mark.to_center)} "
            f"coset {len(mark.operators)} branches {len(mark.branches)} "
            f"closed {mark.group_closed}"
        )
    return "\n".join(lines)


def score_summary(batch: ScoreBatch) -> str:
    """One line per query: nearest landmark, distance, and its coset."""
    lines = [
        f"queries {len(batch.queries)}",
        f"symmetry {batch.symmetry}",
    ]
    for query in batch.queries:
        if not query.hits:
            lines.append(f"query {query.query_id} no landmarks")
            continue
        nearest = query.hits[0]
        lines.append(
            f"query {query.query_id} nearest landmark {nearest.landmark_id} "
            f"cluster {nearest.cluster_id} distance {nearest.key_distance:.6g} "
            f"to_center {_fmt_op(nearest.to_center)} "
            f"coset {len(nearest.operators)} branches {len(nearest.branches)}"
        )
        for hit in query.hits[1:]:
            lines.append(
                f"  landmark {hit.landmark_id} cluster {hit.cluster_id} "
                f"distance {hit.key_distance:.6g} "
                f"coset {len(hit.operators)} branches {len(hit.branches)}"
            )
    return "\n".join(lines)


def _load_csv(path: Path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        cells = []
        groups = []
        for row in reader:
            cells.append([
                float(row["a"]), float(row["b"]), float(row["c"]),
                float(row["alpha"]), float(row["beta"]), float(row["gamma"]),
            ])
            groups.append(row.get("sg_hm") or None)
    return cells, groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="cluster a subsample into a landmark catalog")
    build.add_argument("input", type=Path)
    build.add_argument("--sigma-deg", type=float, required=True)
    build.add_argument("--sigma-len", type=float, default=0.0)
    build.add_argument("--symmetry", default=None)
    build.add_argument("--subsample", type=int, default=None)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--out", type=Path, required=True)

    score = commands.add_parser("score", help="distances and cosets against a saved catalog")
    score.add_argument("catalog", type=Path)
    score.add_argument("queries", type=Path)
    score.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "build":
        cells, groups = _load_csv(args.input)
        catalog = build_landmarks(
            cells,
            sigma_deg=args.sigma_deg,
            sigma_len=args.sigma_len,
            symmetry=args.symmetry,
            spacegroups=groups,
            subsample=args.subsample,
            seed=args.seed,
        )
        args.out.write_text(catalog.to_json())
        print(catalog_summary(catalog))
        return catalog
    catalog = LandmarkCatalog.from_json(args.catalog.read_text())
    cells, groups = _load_csv(args.queries)
    batch = score_cells(catalog, cells, spacegroups=groups)
    print(score_summary(batch))
    if args.out is not None:
        args.out.write_text(batch.to_json())
    return batch


if __name__ == "__main__":
    main()
