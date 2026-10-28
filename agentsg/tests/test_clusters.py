"""Same-protein crystal-form clusters (agentsg.cell.clusters) on small synthetic databases."""
import math
import random

import pytest

duckdb = pytest.importorskip("duckdb")
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from agentsg.cell.celldb import CellDatabase
from agentsg.cell.clusters import (
    _ClosurePricer, _load, cluster_cells, relabelling_price, write_clusters,
)
from agentsg.cell.pdb_app import _cli


def _add(db, pid, cell, sg_hm="P 1", acc=None, group=""):
    assert db.add_cell(pid, cell, None, sg_hm)
    up = "[]" if acc is None else f"[{{'acc': '{acc}', 'name': 'protein {acc}', 'n_res': 100}}]"
    db._db.execute(f"UPDATE cells SET uniprots = {up}, deposit_group = ? WHERE pdb_id = ?",
                   [group, pid])


def _jitter(rng, cell, rel=0.004):
    a, b, c, al, be, ga = cell
    f = lambda x: x * (1 + rng.uniform(-rel, rel))
    return (f(a), f(b), f(c), al, be, ga)


@pytest.fixture
def db():
    rng = random.Random(1)
    d = CellDatabase(":memory:")
    # label A: one tetragonal form, half deposited with the axes permuted
    for k in range(12):
        a, b, c, al, be, ga = _jitter(rng, (79.1, 79.1, 37.9, 90, 90, 90))
        cell = (a, b, c, 90, 90, 90) if k % 2 else (c, a, b, 90, 90, 90)
        _add(d, f"A{k:02d}", cell, acc="P00698")
    # label A, a second, unrelated form: too few to report at min_size 11
    for k in range(5):
        _add(d, f"AO{k}", _jitter(rng, (30.5, 56.3, 73.2, 90, 90, 90)), acc="P00698")
    # label B: the same tetragonal form, but another protein
    for k in range(12):
        _add(d, f"B{k:02d}", _jitter(rng, (79.1, 79.1, 37.9, 90, 90, 90)), acc="Q99999")
    # label C: twelve unrelated cells
    for k in range(12):
        cell = (40 + 9 * k, 55 + 7 * (k % 5), 70 + 11 * (k % 3), 90, 90 + 3 * (k % 4), 90)
        _add(d, f"C{k:02d}", cell, acc="P11111")
    # label D: one shape at two sizes (20 % in volume): the volume gate splits them
    for k in range(11):
        _add(d, f"DS{k:02d}", _jitter(rng, (50, 60, 70, 90, 90, 90)), acc="P22222")
        s = 1.2 ** (1 / 3)
        _add(d, f"DL{k:02d}", _jitter(rng, (50 * s, 60 * s, 70 * s, 90, 90, 90)), acc="P22222")
    # label E: no UniProt, one deposition group
    for k in range(11):
        _add(d, f"E{k:02d}", _jitter(rng, (45.3, 73.1, 52.7, 90, 109.4, 90)), sg_hm="P 1 21 1",
             group="G_1000001")
    # label F: a chain, c grows 3 % per step from 38 to about 60 A
    for k in range(16):
        _add(d, f"F{k:02d}", (79.1, 79.1, 38 * 1.03 ** k, 90, 90, 90), acc="P33333")
    yield d
    d.close()


def _by_label(res):
    out = {}
    for c in res.clusters:
        out.setdefault(c.label, []).append(c)
    return out


def test_clusters_by_label_and_setting(db):
    res = cluster_cells(db)
    by = _by_label(res)
    # A: all twelve, in both settings; the five-member second form is not reported
    (a,) = by["P00698"]
    assert sorted(a.members) == [f"A{k:02d}" for k in range(12)]
    assert a.n_beyond_band == 0 and a.max_price_to_medoid < 0.02
    # B: same form, other protein: its own cluster, never merged with A
    (b,) = by["Q99999"]
    assert all(m.startswith("B") for m in b.members) and b.n == 12
    # C: unrelated cells form nothing
    assert "P11111" not in by
    # D: the volume gate keeps the two sizes apart
    (dl, ds) = sorted(by["P22222"], key=lambda c: c.members[0])
    assert {m[:2] for m in dl.members} == {"DL"} and {m[:2] for m in ds.members} == {"DS"}
    # E: the deposition group stands in for a missing UniProt list
    (e,) = by["DEPOSITION_GROUP_G_1000001"]
    assert e.n == 11 and e.deposition_group_fraction == 1.0


def test_chain_is_one_cluster_and_flagged(db):
    (f,) = _by_label(cluster_cells(db))["P33333"]
    assert f.n == 16
    assert f.n_beyond_band > 0                 # the ends are far beyond 7 % from the medoid


def test_volume_gate_off_merges_sizes(db):
    by = _by_label(cluster_cells(db, vol_tol=0.5))
    (d,) = by["P22222"]
    assert d.n == 22


def test_min_size(db):
    by = _by_label(cluster_cells(db, min_size=5))
    assert sorted(c.n for c in by["P00698"]) == [5, 12]


def test_order_independence():
    rng = random.Random(3)
    cells = [_jitter(rng, (79.1, 79.1, 38 * 1.025 ** (k % 9), 90, 90, 90)) for k in range(30)]
    cells += [_jitter(rng, (30.5, 56.3, 73.2, 90, 90, 90)) for _ in range(12)]
    ids = [f"X{k:02d}" for k in range(len(cells))]
    memberships = []
    for seed in (0, 1):
        order = list(range(len(cells)))
        random.Random(seed).shuffle(order)
        d = CellDatabase(":memory:")
        for k in order:
            _add(d, ids[k], cells[k], acc="P44444")
        res = cluster_cells(d)
        memberships.append(sorted(sorted(c.members) for c in res.clusters))
        d.close()
    assert memberships[0] == memberships[1]
    assert len(memberships[0]) == 2


def test_relabelling_price_is_an_upper_bound(db):
    E = _load(db, 1)
    pricer = _ClosurePricer(E.red, 0.07)
    rng = np.random.default_rng(0)
    n = len(E.rows)
    for _ in range(150):
        i, j = (int(x) for x in rng.integers(0, n, 2))
        exact = pricer.price(i, j)
        ub = relabelling_price(E.G[i], E.G[j])
        if math.isfinite(exact):
            assert ub >= exact - 1e-9
            if ub <= 0.07:
                assert ub == pytest.approx(exact, abs=1e-9)


def test_write_and_cli(db, tmp_path):
    src = tmp_path / "cells.duckdb"
    db._db.execute(f"ATTACH '{src}' AS out")
    db._db.execute("CREATE TABLE out.cells AS SELECT * FROM cells")
    db._db.execute("DETACH out")
    out = tmp_path / "clusters.duckdb"
    assert _cli(["clusters", str(src), str(out)]) == 0
    con = duckdb.connect(str(out), read_only=True)
    n_clusters, n_members = con.execute(
        "SELECT count(*), sum(n) FROM clusters").fetchone()
    assert n_members == con.execute("SELECT count(*) FROM members").fetchone()[0]
    assert con.execute("SELECT count(*) FROM spanning_edges").fetchone()[0] == n_members - n_clusters
    assert con.execute("SELECT band FROM run_info").fetchone()[0] == 0.07
    con.close()
    assert _cli(["clusters", str(src), str(out), "--bogus", "1"]) == 2


def test_cod_database_is_refused():
    d = CellDatabase(":memory:", archive="cod")
    with pytest.raises(ValueError, match="UniProt"):
        cluster_cells(d)
    d.close()
