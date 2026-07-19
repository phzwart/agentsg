"""
celldb: a persistent unit-cell / symmetry database keyed on the root invariant.

Builds and queries a lattice database whose search key is the Kurlin (2022) root
invariant (:mod:`agentsg.cell.rootform`) -- a complete, continuous isometry
invariant, so "find lattices like this" is exact nearest-neighbour search with no
orbit minimisation and no reduction-flip discontinuity.

Two layers:

  * Storage / SQL prefilter -- DuckDB (optional; ``pip install agentsg[db]``).
    An embedded, single-file, columnar SQL engine. Each row stores the PDB id,
    the unit cell, the space group, the cell volume, and the six root-invariant
    components r0..r5. SQL handles coarse prefiltering (by space group, by
    volume band); the exact ranking is done in memory.

  * Exact ranking -- an in-memory NearTree (:mod:`agentsg.cell.neartree`) over
    the root invariants, giving exact k-NN / radius queries in root-product
    (Angstrom) units.

  * PDB ingestion -- :func:`fetch_pdb_cells` pulls unit cell + space group for a
    list of PDB ids (or the entire current holdings) from the RCSB data API
    (holdings REST endpoint for the id list; GraphQL for batched cell/symmetry),
    computes the root invariant, and inserts. Only cell + symmetry + id are
    fetched -- nothing else.

The core agentsg package stays dependency-free; DuckDB is imported lazily and
only :class:`CellDatabase` needs it. The PDB fetch uses only the standard library
(urllib + json).
"""
from __future__ import annotations
import json
import urllib.request
from .rootform import root_invariant
from .metric import UnitCell
from .primitive import primitive_cell


def _primitive_for_roots(cell, sg_hm):
    """Primitive cell for root-invariant computation.

    Uses the lattice letter of ``sg_hm`` to reduce a centred conventional cell
    to its primitive cell. If the symbol is missing or unrecognised the cell is
    assumed primitive and returned unchanged (graceful fallback for query cells
    that carry no symbol).
    """
    if not sg_hm:
        return cell
    try:
        return primitive_cell(cell, sg_hm)
    except (ValueError, ZeroDivisionError):
        return cell


RCSB_HOLDINGS = "https://data.rcsb.org/rest/v1/holdings/current/entry_ids"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"


# ---- PDB ingestion (stdlib only) -------------------------------------------
def list_pdb_ids(timeout=60):
    """Return the list of all current PDB entry ids from RCSB holdings."""
    with urllib.request.urlopen(RCSB_HOLDINGS, timeout=timeout) as fh:
        return json.load(fh)


def _graphql_batch(ids, timeout=60):
    q = ('{entries(entry_ids:%s){rcsb_id '
         'cell{length_a length_b length_c angle_alpha angle_beta angle_gamma} '
         'symmetry{space_group_name_H_M Int_Tables_number}}}' % json.dumps(list(ids)))
    data = json.dumps({"query": q}).encode()
    req = urllib.request.Request(RCSB_GRAPHQL, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)["data"]["entries"]


def fetch_pdb_cells(ids, batch_size=250, progress=False, timeout=60):
    """Fetch (pdb_id, cell, space_group_number, space_group_hm) for PDB ids.

    ``ids`` is an iterable of 4-character PDB ids. Entries with no cell (e.g.
    pure NMR / EM without a crystallographic cell) are skipped. Yields dicts
    with keys: pdb_id, a,b,c,alpha,beta,gamma, sg_number, sg_hm. Batched over
    the RCSB GraphQL endpoint. Only cell + symmetry + id are retrieved.
    """
    ids = list(ids)
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        for e in _graphql_batch(chunk, timeout=timeout):
            cell = e.get("cell") or {}
            sym = e.get("symmetry") or {}
            a = cell.get("length_a")
            if a is None:
                continue                      # no crystallographic cell
            yield {
                "pdb_id": e["rcsb_id"],
                "a": a, "b": cell["length_b"], "c": cell["length_c"],
                "alpha": cell["angle_alpha"], "beta": cell["angle_beta"],
                "gamma": cell["angle_gamma"],
                "sg_number": sym.get("Int_Tables_number"),
                "sg_hm": sym.get("space_group_name_H_M"),
            }
        if progress:
            print(f"  fetched {min(start + batch_size, len(ids))}/{len(ids)}")


# ---- DuckDB-backed database -------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS cells (
    pdb_id     VARCHAR PRIMARY KEY,
    a DOUBLE, b DOUBLE, c DOUBLE,
    alpha DOUBLE, beta DOUBLE, gamma DOUBLE,
    volume DOUBLE,
    sg_number INTEGER, sg_hm VARCHAR,
    r0 DOUBLE, r1 DOUBLE, r2 DOUBLE, r3 DOUBLE, r4 DOUBLE, r5 DOUBLE
);
"""


class CellDatabase:
    """A DuckDB-backed lattice database keyed on the root invariant.

    Parameters
    ----------
    path : str
        DuckDB file path (":memory:" for an ephemeral in-memory database).

    Requires DuckDB (``pip install agentsg[db]``). The core agentsg package does
    not need it.
    """

    def __init__(self, path=":memory:"):
        try:
            import duckdb
        except ImportError as exc:                       # pragma: no cover
            raise ImportError(
                "CellDatabase needs DuckDB: pip install agentsg[db]") from exc
        self._db = duckdb.connect(path)
        self._db.execute(_SCHEMA)

    # -- ingestion --
    def add_cell(self, pdb_id, cell, sg_number=None, sg_hm=None):
        """Insert/replace one cell; computes volume + root invariant.

        The root invariant is computed on the *primitive* cell derived from the
        lattice letter of ``sg_hm`` (centred lattices A/B/C/I/F/R/H are reduced
        first): Kurlin's invariant is a lattice invariant, and the deposited
        conventional cell of a centred group describes only a sublattice. The
        stored cell parameters and volume remain the deposited conventional
        values; only the roots use the primitive lattice.
        """
        try:
            vol = UnitCell(*cell).volume()
            prim = _primitive_for_roots(cell, sg_hm)
            ri = root_invariant(prim)
        except Exception:
            return False
        self._db.execute(
            "INSERT OR REPLACE INTO cells VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [pdb_id, cell[0], cell[1], cell[2], cell[3], cell[4], cell[5],
             vol, sg_number, sg_hm, ri[0], ri[1], ri[2], ri[3], ri[4], ri[5]])
        return True

    def add_pdb(self, ids, batch_size=250, progress=False):
        """Fetch the given PDB ids from RCSB and insert them. Returns count."""
        n = 0
        for rec in fetch_pdb_cells(ids, batch_size=batch_size, progress=progress):
            if self.add_cell(rec["pdb_id"],
                             (rec["a"], rec["b"], rec["c"],
                              rec["alpha"], rec["beta"], rec["gamma"]),
                             rec["sg_number"], rec["sg_hm"]):
                n += 1
        return n

    def __len__(self):
        return self._db.execute("SELECT COUNT(*) FROM cells").fetchone()[0]

    # -- SQL access --
    def sql(self, query, params=None):
        """Run an arbitrary read query; returns list of tuples."""
        return self._db.execute(query, params or []).fetchall()

    # -- nearest-neighbour search --
    def _candidates(self, sg_number=None, volume=None, volume_tol=0.25):
        where, params = [], []
        if sg_number is not None:
            where.append("sg_number = ?"); params.append(sg_number)
        if volume is not None:
            where.append("volume BETWEEN ? AND ?")
            params += [volume * (1 - volume_tol), volume * (1 + volume_tol)]
        sql = "SELECT pdb_id, r0,r1,r2,r3,r4,r5 FROM cells"
        if where:
            sql += " WHERE " + " AND ".join(where)
        return self._db.execute(sql, params).fetchall()

    def nearest(self, cell, k=10, sg_number=None, volume_band=None, sg_hm=None):
        """Return the k nearest lattices to ``cell`` as (pdb_id, distance).

        Optional SQL prefilters shrink the candidate set before exact ranking:
        ``sg_number`` restricts to one space-group number; ``volume_band`` (a
        fractional tolerance, e.g. 0.25) restricts to cells within that fraction
        of the query volume. Ranking is exact root-invariant Euclidean distance.

        ``sg_hm`` is the query cell's space-group symbol; when the query lattice
        is centred, pass it so the query root is computed on the *primitive*
        cell, matching how the stored roots were computed. Without it the query
        cell is assumed primitive.
        """
        import math
        from .neartree import build_neartree
        vol = UnitCell(*cell).volume() if volume_band is not None else None
        rows = self._candidates(sg_number=sg_number, volume=vol,
                                volume_tol=volume_band or 0.25)
        if not rows:
            return []
        def dist(a, b):
            return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(6)))
        tree = build_neartree(((r[1:], r[0]) for r in rows), dist)
        q = root_invariant(_primitive_for_roots(cell, sg_hm))
        return tree.k_nearest(q, k)

    def nearest_with_supercells(self, cell, k=10, max_index=4,
                                length_tol_pct=3.0, angle_tol_deg=5.0,
                                volume_tol=0.05):
        """Volume-spanning similarity: find related lattices INCLUDING super- and
        sub-lattices (doubled / halved / index-n cells), not just isometric ones.

        The root invariant is an *isometry* invariant, so :meth:`nearest` treats a
        cell and its supercell as different lattices. This method spans volume
        changes in two complementary ways:

          * super-lattices of the query -- for each index r in 2..max_index,
            enumerate the query's index-r sublattices (via Hermite normal form),
            reduce each, and look it up in the fast root-invariant index. A hit
            means a database cell IS an index-r supercell of the query.

          * sub-lattices of the query, and the general case -- SQL-prefilter
            database cells whose volume is ~1/r or ~r times the query volume
            (within ``volume_tol``), then run the exact ``compare_cells``
            sublattice search between the query and each candidate.

        Returns a list of dicts sorted by (index, distance):
          {pdb_id, index, distance, relation, cell, sg_number, sg_hm, M}
        where ``relation`` is 'isometric' (index 1), 'db_is_supercell' (db cell =
        query x index), or 'db_is_sublattice' (query = db cell x index), and
        ``M`` is the integer sublattice matrix (None for isometric hits).
        """
        import math
        from .sublattice import generate_sublattices, apply_to_cell
        from .reduction import niggli_reduce
        from .compare import compare_cells

        q_vol = UnitCell(*cell).volume()
        results = {}                       # pdb_id -> best record

        def _consider(pdb_id, index, distance, relation, M):
            row = self._db.execute(
                "SELECT a,b,c,alpha,beta,gamma,sg_number,sg_hm FROM cells "
                "WHERE pdb_id=?", [pdb_id]).fetchone()
            rec = {"pdb_id": pdb_id, "index": index, "distance": distance,
                   "relation": relation,
                   "cell": tuple(row[:6]), "sg_number": row[6], "sg_hm": row[7],
                   "M": M}
            prev = results.get(pdb_id)
            # keep the smaller distance for a given cell; a genuine (small-distance)
            # super/sub-lattice match should win over a far "isometric" mislabel.
            if prev is None or distance < prev["distance"]:
                results[pdb_id] = rec

        # index 1: isometric hits from the fast index, kept only within tolerance
        for pdb_id, dist in self.nearest(cell, k=k):
            if dist <= length_tol_pct:
                _consider(pdb_id, 1, dist, "isometric", None)

        # db_is_supercell: query's index-r sublattices, looked up in the index
        for r in range(2, max_index + 1):
            for M in generate_sublattices(r):
                sup = apply_to_cell(cell, M)
                red, _ = niggli_reduce(*sup)
                for pdb_id, dist in self.nearest(red, k=k):
                    if dist <= length_tol_pct:        # root-dist ~ Angstrom scale
                        _consider(pdb_id, r, dist, "db_is_supercell",
                                  tuple(tuple(row) for row in M))

        # db_is_sublattice / general: volume-banded candidates via compare_cells
        for r in range(2, max_index + 1):
            for target_vol, rel in ((q_vol / r, "db_is_sublattice"),
                                    (q_vol * r, "db_is_supercell")):
                lo, hi = target_vol * (1 - volume_tol), target_vol * (1 + volume_tol)
                rows = self._db.execute(
                    "SELECT pdb_id,a,b,c,alpha,beta,gamma FROM cells "
                    "WHERE volume BETWEEN ? AND ?", [lo, hi]).fetchall()
                for row in rows:
                    other = tuple(row[1:])
                    cmp = compare_cells(cell, other, length_tol_pct=length_tol_pct,
                                        angle_tol_deg=angle_tol_deg, max_index=r)
                    for sol in cmp["solutions"]:
                        if sol.index == r:
                            _consider(row[0], r, sol.max_length_dev, rel,
                                      sol.M)

        out = sorted(results.values(), key=lambda d: (d["index"], d["distance"]))
        return out[:k]

    # -- persistent fast index (build tree once, query many) --
    def build_index(self, sg_number=None, shuffle=True):
        """Build an in-memory root-invariant NearTree from stored r0..r5.

        Unlike :meth:`nearest`, which rebuilds a tree per call, this constructs
        the index once from the precomputed root columns (no root recompute) and
        returns a :class:`RootIndex` supporting repeated sub-millisecond queries.
        Optionally restrict to one space-group number.
        """
        import math
        import random
        from .neartree import build_neartree
        sql = "SELECT pdb_id, r0,r1,r2,r3,r4,r5 FROM cells"
        params = []
        if sg_number is not None:
            sql += " WHERE sg_number = ?"; params.append(sg_number)
        rows = self._db.execute(sql, params).fetchall()
        pts = [(tuple(r[1:]), r[0]) for r in rows if r[1] is not None]
        if shuffle:
            random.shuffle(pts)

        def dist(a, b):
            return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(6)))

        tree = build_neartree(pts, dist)
        return RootIndex(tree, dist)

    def compare_query(self, cell, k=10, sg_number=None, volume_band=None,
                      sg_hm=None):
        """Alias for :meth:`nearest` -- k nearest PDB entries to ``cell``.

        Returns a list of (pdb_id, root_distance) sorted by distance. Pass
        ``sg_hm`` for a centred query lattice. For many repeated queries build a
        :class:`RootIndex` once with :meth:`build_index` instead.
        """
        return self.nearest(cell, k=k, sg_number=sg_number,
                             volume_band=volume_band, sg_hm=sg_hm)

    def close(self):
        self._db.close()


class RootIndex:
    """A prebuilt root-invariant NearTree for fast repeated cell queries.

    Construct via :meth:`CellDatabase.build_index`. Queries take a *cell*
    ``(a,b,c,alpha,beta,gamma)``; the root invariant of the query is computed
    once per call and matched against the precomputed database roots by exact
    Euclidean distance (Angstrom, root-product units).
    """

    __slots__ = ("_tree", "_dist")

    def __init__(self, tree, dist):
        self._tree = tree
        self._dist = dist

    def __len__(self):
        return len(self._tree)

    def k_nearest(self, cell, k=10, sg_hm=None):
        """Return the k nearest (pdb_id, distance) to ``cell``.

        Pass ``sg_hm`` for a centred query lattice so the query root is computed
        on the primitive cell (matching the stored roots).
        """
        q = root_invariant(_primitive_for_roots(cell, sg_hm))
        return self._tree.k_nearest(q, k)

    def nearest(self, cell, sg_hm=None):
        """Return the single nearest (pdb_id, distance) to ``cell``."""
        q = root_invariant(_primitive_for_roots(cell, sg_hm))
        return self._tree.nearest(q)

    def within(self, cell, radius, sg_hm=None):
        """Return all (pdb_id, distance) within ``radius`` of ``cell``."""
        q = root_invariant(_primitive_for_roots(cell, sg_hm))
        return self._tree.within(q, radius)
