"""Tests for the resumable PDB-download application.

Network is mocked: a fake fetch_pdb_cells feeds synthetic records so the suite
needs no RCSB access. Covers resume (skip existing), retry/backoff on transient
errors, root-column population, primitive handling for centred symbols, and
nearest-root query correctness against brute force.
"""
import math
import urllib.error
import pytest

pytest.importorskip("duckdb")

from agentsg.cell import pdb_app
from agentsg.cell.celldb import CellDatabase
from agentsg.cell.rootform import root_invariant
from agentsg.cell.celldb import _primitive_for_roots


def _fake_records(n, centred=False):
    recs = []
    for i in range(n):
        recs.append({
            "pdb_id": f"T{i:03d}",
            "a": 40.0 + i, "b": 50.0 + i, "c": 60.0 + i,
            "alpha": 90.0, "beta": 90.0 + (i % 5), "gamma": 90.0,
            "sg_number": 5 if centred else 4,
            "sg_hm": "C 1 2 1" if centred else "P 1 21 1",
        })
    return recs


def _install_fake_fetch(monkeypatch, records, fail_times=0):
    """Patch fetch_pdb_cells to yield from ``records`` by id, optionally raising
    a transient error the first ``fail_times`` calls."""
    state = {"calls": 0}
    by_id = {r["pdb_id"]: r for r in records}

    def fake_fetch(ids, batch_size=250, timeout=60):
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise urllib.error.URLError("transient")
        return [by_id[i] for i in ids if i in by_id]

    monkeypatch.setattr(pdb_app, "fetch_pdb_cells", fake_fetch)
    return state


def test_build_populates_roots(tmp_path, monkeypatch):
    recs = _fake_records(10)
    _install_fake_fetch(monkeypatch, recs)
    dbp = str(tmp_path / "t.duckdb")
    summary = pdb_app.build_pdb_database(
        dbp, ids=[r["pdb_id"] for r in recs], batch_size=4,
        progress_every=0, log=lambda *_: None)
    assert summary["stored"] == 10
    db = CellDatabase(dbp)
    assert len(db) == 10
    # every row has all six roots
    n = db.sql("SELECT COUNT(*) FROM cells WHERE r5 IS NOT NULL")[0][0]
    assert n == 10
    db.close()


def test_resume_skips_existing(tmp_path, monkeypatch):
    recs = _fake_records(10)
    dbp = str(tmp_path / "t.duckdb")
    ids = [r["pdb_id"] for r in recs]
    _install_fake_fetch(monkeypatch, recs)
    # first build: only first 6
    s1 = pdb_app.build_pdb_database(dbp, ids=ids[:6], batch_size=4,
                                    progress_every=0, log=lambda *_: None)
    assert s1["stored"] == 6
    # resume over all 10: 6 skipped, 4 new
    s2 = pdb_app.build_pdb_database(dbp, ids=ids, batch_size=4,
                                    progress_every=0, log=lambda *_: None)
    assert s2["skipped_existing"] == 6
    assert s2["stored"] == 4
    assert s2["rows_in_db"] == 10


def test_retry_then_success(tmp_path, monkeypatch):
    recs = _fake_records(4)
    _install_fake_fetch(monkeypatch, recs, fail_times=2)
    dbp = str(tmp_path / "t.duckdb")
    summary = pdb_app.build_pdb_database(
        dbp, ids=[r["pdb_id"] for r in recs], batch_size=4, retries=5,
        backoff=0.0, progress_every=0, log=lambda *_: None)
    # after 2 transient failures the batch succeeds; nothing lost
    assert summary["stored"] == 4
    assert summary["failed_batches"] == 0


def test_batch_fails_after_retries(tmp_path, monkeypatch):
    recs = _fake_records(4)
    _install_fake_fetch(monkeypatch, recs, fail_times=99)
    dbp = str(tmp_path / "t.duckdb")
    summary = pdb_app.build_pdb_database(
        dbp, ids=[r["pdb_id"] for r in recs], batch_size=4, retries=3,
        backoff=0.0, progress_every=0, log=lambda *_: None)
    assert summary["failed_batches"] == 1
    assert summary["stored"] == 0


def test_centred_symbol_uses_primitive_roots(tmp_path, monkeypatch):
    recs = _fake_records(5, centred=True)
    _install_fake_fetch(monkeypatch, recs)
    dbp = str(tmp_path / "t.duckdb")
    pdb_app.build_pdb_database(dbp, ids=[r["pdb_id"] for r in recs],
                               batch_size=8, progress_every=0,
                               log=lambda *_: None)
    db = CellDatabase(dbp)
    row = db.sql("SELECT a,b,c,alpha,beta,gamma,sg_hm,r0,r1,r2,r3,r4,r5 "
                 "FROM cells LIMIT 1")[0]
    conv = row[:6]
    stored = row[7:13]
    # stored roots are the PRIMITIVE roots, not the conventional ones
    prim_ri = root_invariant(_primitive_for_roots(conv, row[6]))
    conv_ri = root_invariant(conv)
    dprim = math.sqrt(sum((prim_ri[i] - stored[i]) ** 2 for i in range(6)))
    dconv = math.sqrt(sum((conv_ri[i] - stored[i]) ** 2 for i in range(6)))
    assert dprim < 1e-9        # matches primitive
    assert dconv > 1.0         # differs from conventional (the bug we fixed)
    db.close()


def test_index_query_matches_brute(tmp_path, monkeypatch):
    recs = _fake_records(40)
    _install_fake_fetch(monkeypatch, recs)
    dbp = str(tmp_path / "t.duckdb")
    pdb_app.build_pdb_database(dbp, ids=[r["pdb_id"] for r in recs],
                               batch_size=16, progress_every=0,
                               log=lambda *_: None)
    db = CellDatabase(dbp)
    idx = db.build_index()
    allroots = db.sql("SELECT pdb_id,r0,r1,r2,r3,r4,r5 FROM cells")
    q = recs[7]
    qcell = (q["a"], q["b"], q["c"], q["alpha"], q["beta"], q["gamma"])
    qi = root_invariant(_primitive_for_roots(qcell, q["sg_hm"]))
    brute = sorted(
        (math.sqrt(sum((qi[i] - r[i + 1]) ** 2 for i in range(6))), r[0])
        for r in allroots)[:5]
    got = idx.k_nearest(qcell, k=5, sg_hm=q["sg_hm"])
    # k-th distances agree exactly (membership can differ only on ties)
    assert sorted(d for _, d in got) == pytest.approx([d for d, _ in brute])
    db.close()
