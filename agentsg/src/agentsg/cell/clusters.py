"""Same-protein crystal-form clusters in the cell database.

Two entries are joined by an edge when

* they carry the same label: the exact sorted set of UniProt accessions, or the
  deposition group (``DEPOSITION_GROUP_<id>``) when the entry has no UniProt list;
* their strain price is at most ``band``: the affine-invariant distance between
  the two primitive metrics after the best common scale, minimised over every
  integer operator of the widened Selling closure
  (:func:`agentsg.cell.strain_closure.reindex`), so it does not depend on the
  settings in which the cells were deposited;
* their primitive volumes agree within ``vol_tol``. The price is scale-fitted
  and therefore blind to volume; this gate keeps a cell of the same shape at
  another size from joining.

A cluster is a connected component of that graph (single linkage). It does not
depend on the order in which entries are visited.

The search per label has three stages, each of which leaves the components
unchanged:

1. candidate pairs from a KD-tree on the volume-normalised sorted linear key,
   ``l / V**(1/3)``, within ``key_radius`` (default ``KEY_RATIO * band``), and
   the volume gate;
2. pairs in order of key distance; a pair already inside one component is
   skipped, and a pair whose price through one of the 24 relabellings of the
   reduced superbase is within the band is joined. That price is an upper
   bound on the closure price, so every join is a true edge;
3. the remaining pairs that still join two components are priced exactly
   against the widened closure.

``KEY_RATIO`` was calibrated on every pair of 40 sampled PDB labels (11,988
pairs, 5,591 edges at a band of 0.07): the largest ratio of key distance to
price on an edge was 1.30, and a radius of 1.43 times the band recovered every
edge. ``KEY_RATIO = 1.7`` keeps a margin. :func:`calibrate_key_ratio` repeats
the measurement on any database.

Geodesic connectivity can chain: a cluster can reach two crystal forms through
a run of intermediate cells. Each cluster therefore reports the price from its
medoid to every member and how many members lie beyond the band.

Command line::

    python -m agentsg.cell.pdb_app clusters data/pdb_cells.duckdb clusters.duckdb

Needs numpy and scipy (``pip install agentsg[db]``).
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

DEFAULT_BAND = 0.07
"""Strain price band (7 %), near the 95th percentile of same-form non-isomorphism in the PDB."""

DEFAULT_VOL_TOL = 0.07
"""Largest relative difference of primitive volumes on an edge."""

DEFAULT_MIN_SIZE = 11
"""Smallest cluster reported (more than 10 entries)."""

KEY_RATIO = 1.7
"""Default key radius as a multiple of the band (calibrated maximum 1.30; see the module docstring)."""

_LABEL_SQL = """
SELECT pdb_id, sg_hm, a, b, c, alpha, beta, gamma,
       red_a, red_b, red_c, red_alpha, red_beta, red_gamma,
       l0, l1, l2, l3, l4, l5,
       deposit_group,
       CASE WHEN len(uniprots) > 0 THEN
            array_to_string(list_sort(list_distinct(
                list_transform(uniprots, lambda x: x.acc))), '+')
            WHEN deposit_group <> '' THEN 'DEPOSITION_GROUP_' || deposit_group
            ELSE NULL END AS label,
       CASE WHEN len(uniprots) > 0 THEN uniprots[1].name ELSE NULL END AS name
FROM cells
"""


def _np():
    try:
        import numpy as np
    except ImportError as exc:                             # pragma: no cover
        raise ImportError("clusters needs numpy and scipy: pip install agentsg[db]") from exc
    return np


# ---------------------------------------------------------------- data access
@dataclass
class _Entries:
    rows: list
    col: dict
    red: object          # (n, 6) reduced primitive cells
    G: object            # (n, 3, 3) metrics of the reduced cells
    vprim: object        # (n,) primitive volumes
    skey: object         # (n, 6) volume-normalised sorted linear key
    labels: object       # (n,) label strings


def _connection(db):
    """A DuckDB connection from a path or a CellDatabase."""
    if isinstance(db, str):
        import duckdb
        return duckdb.connect(db, read_only=True), True
    return db._db, False


def _load(db, min_size):
    """Every entry whose label has at least ``min_size`` entries, sorted by label."""
    np = _np()
    from .metric import metric_tensor

    con, own = _connection(db)
    try:
        cols = {r[0] for r in con.execute("DESCRIBE cells").fetchall()}
        missing = {"uniprots", "deposit_group", "red_a", "l0"} - cols
        if missing:
            raise ValueError(
                f"the cells table lacks {sorted(missing)}; clustering needs the PDB "
                "database with UniProt lists, deposition groups, linear keys and "
                "Selling-reduced cells")
        con.execute(f"CREATE OR REPLACE TEMP TABLE _cluster_lab AS {_LABEL_SQL}")
        rel = con.execute(
            """SELECT l.* FROM _cluster_lab l JOIN
               (SELECT label FROM _cluster_lab WHERE label IS NOT NULL
                GROUP BY label HAVING count(*) >= ?) g USING (label)
               WHERE l.red_a IS NOT NULL AND l.l0 IS NOT NULL
               ORDER BY label, pdb_id""", [min_size])
        rows = rel.fetchall()
        names = [d[0] for d in rel.description]
        con.execute("DROP TABLE _cluster_lab")
    finally:
        if own:
            con.close()
    col = {c: k for k, c in enumerate(names)}
    n = len(rows)
    red = np.array([r[col["red_a"]:col["red_gamma"] + 1] for r in rows], float).reshape(n, 6)
    G = np.array([metric_tensor(tuple(c)) for c in red], float).reshape(n, 3, 3)
    vprim = np.sqrt(np.linalg.det(G)) if n else np.zeros(0)
    lkey = np.array([r[col["l0"]:col["l5"] + 1] for r in rows], float).reshape(n, 6)
    skey = lkey / vprim[:, None] ** (1.0 / 3.0) if n else lkey
    labels = np.array([r[col["label"]] for r in rows], dtype=object)
    return _Entries(rows, col, red, G, vprim, skey, labels)


# ------------------------------------------------------------------- pricing
def _relabellings():
    """The 24 operators taking the reduced basis to three of (b1, b2, b3, b4),
    b4 = -(b1 + b2 + b3), in every order. Each is unimodular."""
    np = _np()
    vecs = [np.array(v) for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, -1, -1))]
    return np.array([np.stack([vecs[k] for k in p], axis=1).astype(float)
                     for p in itertools.permutations(range(4), 3)])


_S4 = None


def relabelling_price(G_ref, G_new):
    """Smallest scale-fitted price of ``G_new`` against the 24 relabellings of the
    reduced superbase of ``G_ref``. An upper bound on the closure price."""
    global _S4
    np = _np()
    if _S4 is None:
        _S4 = _relabellings()
    Gp = np.einsum("kba,bc,kcd->kad", _S4, G_ref, _S4)
    w, V = np.linalg.eigh(G_new)
    Gm = V @ np.diag(w ** -0.5) @ V.T
    lv = np.log(np.linalg.eigvalsh(Gm @ Gp @ Gm))
    return float(np.min(np.linalg.norm(lv - lv.mean(axis=1, keepdims=True), axis=1)))


class _ClosurePricer:
    """Exact prices against the widened closure, one closure per reference, cached."""

    def __init__(self, red, band):
        from .strain_closure import band_for
        self.red = red
        self.band = band
        self.eps = band_for(band) * (1 + 1e-3)
        self.cache = {}
        self.calls = 0

    def price(self, i, j):
        """Closure price of entry j against entry i, or inf above the band."""
        import warnings
        from .strain_closure import reindex, widened_closure
        W = self.cache.get(i)
        if W is None:
            W = widened_closure(tuple(self.red[i]), sg_hm=None, eps=self.eps)
            self.cache[i] = W
        self.calls += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cands, _ = reindex(W, tuple(self.red[j]), new_sg_hm=None, accept=self.band,
                                   proper=None, max_scale_change=None)
        except ValueError:                # primitive volumes differ by 1.6x or more
            return math.inf
        return min((c.price for c in cands), default=math.inf)


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[max(ra, rb)] = min(ra, rb)
        return True


# ------------------------------------------------------------------- results
@dataclass
class Cluster:
    cluster_id: int
    label: str
    name: str | None
    medoid: str
    medoid_sg: str
    medoid_cell: tuple
    members: list                      # pdb ids
    member_prices: list                # price from the medoid (upper bound above the band)
    space_groups: list
    key_diameter: float
    deposition_group_fraction: float
    band: float

    @property
    def n(self):
        return len(self.members)

    @property
    def max_price_to_medoid(self):
        return max(self.member_prices)

    @property
    def n_beyond_band(self):
        """Members further than the band from the medoid: the cluster is chained."""
        return sum(p > self.band for p in self.member_prices)


@dataclass
class ClusterResult:
    clusters: list
    edges: list                        # (pdb_a, pdb_b, cluster_id, price, how)
    band: float
    vol_tol: float
    key_radius: float
    min_size: int
    stats: dict = field(default_factory=dict)

    def membership(self):
        """``{pdb_id: cluster_id}`` for every clustered entry."""
        return {m: c.cluster_id for c in self.clusters for m in c.members}


# ---------------------------------------------------------------- clustering
def cluster_cells(db, band=DEFAULT_BAND, vol_tol=DEFAULT_VOL_TOL, min_size=DEFAULT_MIN_SIZE,
                  key_radius=None, progress=False):
    """Connected components of the same-label graph at ``band``.

    ``db`` is a :class:`~agentsg.cell.celldb.CellDatabase` or a path to its file.
    Returns a :class:`ClusterResult` with every component of at least
    ``min_size`` entries, largest first, and the edges that joined them.
    """
    np = _np()
    from scipy.spatial import cKDTree

    t0 = time.time()
    if key_radius is None:
        key_radius = KEY_RATIO * band
    E = _load(db, min_size)
    n_all = len(E.rows)
    stats = dict(entries=n_all, candidates=0, skipped=0, relabel=0, exact=0, exact_joined=0)
    if n_all == 0:
        return ClusterResult([], [], band, vol_tol, key_radius, min_size, stats)
    uniq, start, counts = np.unique(E.labels.astype(str), return_index=True, return_counts=True)
    stats["labels"] = len(uniq)
    if progress:
        print(f"clusters: {n_all:,} entries in {len(uniq):,} labels with >= {min_size} entries",
              flush=True)

    dsu = _DSU(n_all)
    pricer = _ClosurePricer(E.red, band)
    edges = []
    for n_done, k in enumerate(np.argsort(-counts, kind="stable")):
        idx = np.arange(start[k], start[k] + counts[k])
        pairs = cKDTree(E.skey[idx]).query_pairs(key_radius, output_type="ndarray")
        if len(pairs) == 0:
            continue
        i, j = idx[pairs[:, 0]], idx[pairs[:, 1]]
        keep = np.abs(E.vprim[i] / E.vprim[j] - 1) <= vol_tol
        i, j = i[keep], j[keep]
        order = np.argsort(np.linalg.norm(E.skey[i] - E.skey[j], axis=1), kind="stable")
        i, j = i[order].tolist(), j[order].tolist()
        stats["candidates"] += len(i)
        deferred = []
        for a, b in zip(i, j):
            if dsu.find(a) == dsu.find(b):
                stats["skipped"] += 1
                continue
            p = relabelling_price(E.G[a], E.G[b])
            if p <= band:
                dsu.union(a, b)
                edges.append((a, b, p, "relabel"))
                stats["relabel"] += 1
            else:
                deferred.append((a, b))
        for a, b in deferred:
            if dsu.find(a) == dsu.find(b):
                continue
            p = pricer.price(a, b)
            stats["exact"] += 1
            if p <= band:
                dsu.union(a, b)
                edges.append((a, b, p, "closure"))
                stats["exact_joined"] += 1
        if progress and (n_done < 10 or n_done % 250 == 0):
            print(f"  [{n_done + 1}/{len(uniq)}] {uniq[k]} n={counts[k]} "
                  f"{time.time() - t0:.0f}s", flush=True)

    # components, largest first
    comps = {}
    for x in range(n_all):
        comps.setdefault(dsu.find(x), []).append(x)
    big = sorted((m for m in comps.values() if len(m) >= min_size), key=lambda m: (-len(m), m[0]))
    pid = E.col["pdb_id"]
    clusters, cid_of = [], {}
    for cid, m in enumerate(big, start=1):
        clusters.append(_summarise(E, np.array(m), cid, band, pricer))
        for x in m:
            cid_of[x] = cid
    edge_rows = [(E.rows[a][pid], E.rows[b][pid], cid_of[a], float(p), how)
                 for a, b, p, how in edges if a in cid_of]
    stats.update(components=len(comps), clusters=len(clusters),
                 clustered_entries=sum(c.n for c in clusters),
                 closures=len(pricer.cache), seconds=time.time() - t0)
    if progress:
        print(f"clusters: {len(clusters):,} clusters of >= {min_size} covering "
              f"{stats['clustered_entries']:,} entries, {stats['seconds']:.0f}s", flush=True)
    return ClusterResult(clusters, edge_rows, band, vol_tol, key_radius, min_size, stats)


def _summarise(E, m, cid, band, pricer):
    np = _np()
    S = E.skey[m]
    if len(m) <= 4000:
        D = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=2)
        med = int(m[int(np.argmin(D.sum(axis=1)))])
        diam = float(D.max())
    else:                                               # pragma: no cover - memory guard
        med = int(m[int(np.argmin(np.linalg.norm(S - S.mean(0), axis=1)))])
        diam = float("nan")
    prices = []
    for x in m.tolist():
        if x == med:
            prices.append(0.0)
            continue
        p = relabelling_price(E.G[med], E.G[x])
        if p > band:
            p = min(p, pricer.price(med, x))
        prices.append(float(p))
    c, r0 = E.col, E.rows[med]
    return Cluster(
        cluster_id=cid, label=r0[c["label"]], name=r0[c["name"]],
        medoid=r0[c["pdb_id"]], medoid_sg=r0[c["sg_hm"]],
        medoid_cell=tuple(float(v) for v in r0[c["a"]:c["gamma"] + 1]),
        members=[E.rows[x][c["pdb_id"]] for x in m.tolist()], member_prices=prices,
        space_groups=sorted({E.rows[x][c["sg_hm"]] for x in m.tolist()}),
        key_diameter=diam,
        deposition_group_fraction=sum(1 for x in m.tolist() if E.rows[x][c["deposit_group"]]) / len(m),
        band=band)


def write_clusters(result, path):
    """Write ``clusters``, ``members``, ``spanning_edges`` and ``run_info`` tables to a DuckDB file."""
    import duckdb
    con = duckdb.connect(path)
    try:
        con.execute("""CREATE OR REPLACE TABLE clusters (cluster_id INTEGER, label VARCHAR,
            name VARCHAR, n INTEGER, medoid VARCHAR, medoid_sg VARCHAR, a DOUBLE, b DOUBLE,
            c DOUBLE, alpha DOUBLE, beta DOUBLE, gamma DOUBLE, n_space_groups INTEGER,
            space_groups VARCHAR, max_price_to_medoid DOUBLE, n_beyond_band INTEGER,
            key_diameter DOUBLE, deposition_group_fraction DOUBLE)""")
        con.executemany("INSERT INTO clusters VALUES (" + ",".join("?" * 18) + ")", [
            (c.cluster_id, c.label, c.name, c.n, c.medoid, c.medoid_sg, *c.medoid_cell,
             len(c.space_groups), ", ".join(c.space_groups), c.max_price_to_medoid,
             c.n_beyond_band, c.key_diameter, c.deposition_group_fraction)
            for c in result.clusters])
        con.execute("""CREATE OR REPLACE TABLE members (pdb_id VARCHAR, cluster_id INTEGER,
            price_to_medoid DOUBLE)""")
        con.executemany("INSERT INTO members VALUES (?, ?, ?)", [
            (m, c.cluster_id, p) for c in result.clusters
            for m, p in zip(c.members, c.member_prices)])
        con.execute("""CREATE OR REPLACE TABLE spanning_edges (pdb_a VARCHAR, pdb_b VARCHAR,
            cluster_id INTEGER, price DOUBLE, how VARCHAR)""")
        con.executemany("INSERT INTO spanning_edges VALUES (?, ?, ?, ?, ?)", result.edges)
        con.execute("""CREATE OR REPLACE TABLE run_info AS SELECT ? AS band, ? AS vol_tol,
            ? AS key_radius, ? AS min_size""",
                    [result.band, result.vol_tol, result.key_radius, result.min_size])
    finally:
        con.close()


def calibrate_key_ratio(db, band=DEFAULT_BAND, vol_tol=DEFAULT_VOL_TOL, n_labels=40,
                        size_range=(12, 70), seed=0):
    """Price every pair of ``n_labels`` random labels exactly and report the largest
    ratio of key distance to price on an edge (the basis of :data:`KEY_RATIO`).

    Returns ``dict(pairs, edges, max_ratio, max_key_distance)``.
    """
    np = _np()
    E = _load(db, size_range[0])
    uniq, start, counts = np.unique(E.labels.astype(str), return_index=True, return_counts=True)
    rng = np.random.default_rng(seed)
    pick = [k for k in rng.permutation(len(uniq))
            if size_range[0] <= counts[k] <= size_range[1]][:n_labels]
    pricer = _ClosurePricer(E.red, band)
    n_pairs, ratios, dists = 0, [], []
    for k in pick:
        for a, b in itertools.combinations(range(start[k], start[k] + counts[k]), 2):
            n_pairs += 1
            if abs(E.vprim[a] / E.vprim[b] - 1) > vol_tol:
                continue
            p = pricer.price(a, b)
            if p <= band:
                d = float(np.linalg.norm(E.skey[a] - E.skey[b]))
                dists.append(d)
                if p >= 1e-3:                 # the ratio is noise for identical cells
                    ratios.append(d / p)
    return dict(pairs=n_pairs, edges=len(dists),
                max_ratio=max(ratios, default=0.0), max_key_distance=max(dists, default=0.0))
