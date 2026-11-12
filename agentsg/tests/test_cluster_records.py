"""All-pairs RMSD and record naming (agentsg.cell.cluster_records), synthetic cases."""
import math

import pytest

np = pytest.importorskip("numpy")
duckdb = pytest.importorskip("duckdb")
pytest.importorskip("scipy")

from agentsg.cell.cluster_frames import _orth, frame_cluster
from agentsg.cell.cluster_records import identity_dirname, pairwise_rmsd


def _chain(rng, n=120):
    s = rng.normal(size=(n, 3))
    s *= 3.8 / np.linalg.norm(s, axis=1, keepdims=True)
    x = np.cumsum(s, axis=0)
    return x - x.mean(0) + np.array([9.0, 7.0, 5.0])


def _db(models):
    con = duckdb.connect(":memory:")
    con.execute("""CREATE TABLE entries (pdb_id VARCHAR, resolution DOUBLE, a DOUBLE, b DOUBLE,
        c DOUBLE, alpha DOUBLE, beta DOUBLE, gamma DOUBLE, sg_hm VARCHAR, n_ca INTEGER,
        n_mapped INTEGER, error VARCHAR)""")
    con.execute("CREATE TABLE ca (pdb_id VARCHAR, chain VARCHAR, acc VARCHAR, unp INTEGER, "
                "fx FLOAT, fy FLOAT, fz FLOAT)")
    for pid, cell, sg, res, xyz in models:
        f = xyz @ np.linalg.inv(_orth(cell)).T
        con.execute("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)",
                    [pid, res, *cell, sg, len(f), len(f)])
        con.executemany("INSERT INTO ca VALUES (?,?,?,?,?,?,?)",
                        [(pid, "A", "P12345", k + 1, *map(float, f[k])) for k in range(len(f))])
    return con


def test_pairwise_rmsd_in_common_frame():
    rng = np.random.default_rng(3)
    cell = (40.0, 50.0, 60.0, 90, 90, 90)
    x = _chain(rng)
    # REF; COPY = same molecule after a 2_1 screw and an allowed origin shift;
    # NOISY = REF plus 0.3 A noise per coordinate
    B = _orth(cell)
    f = x @ np.linalg.inv(B).T
    g = np.stack([-f[:, 0] + 0.5, -f[:, 1], f[:, 2] + 0.5], 1) + np.array([0.5, 0, 0])
    models = [("REF", cell, "P 21 21 21", 1.0, x),
              ("COPY", cell, "P 21 21 21", 2.0, g @ B.T),
              ("NOISY", cell, "P 21 21 21", 2.5, x + rng.normal(scale=0.3, size=x.shape))]
    con = _db(models)
    res = frame_cluster(con, ["REF", "COPY", "NOISY"])
    assert {p.status for p in res} == {"reference", "placed"}
    rows = [dict(pdb_id=p.pdb_id, status=p.status, member_chain=p.member_chain,
                 ref_chain=p.ref_chain, unp_offset=p.unp_offset,
                 L=";".join(",".join(str(v) for v in r) for r in p.L.rows),
                 t=tuple(float(v) for v in p.t)) for p in res]
    ids, frame, fit, n = pairwise_rmsd(con, rows)
    k = {pid: i for i, pid in enumerate(ids)}
    assert frame[k["REF"], k["COPY"]] < 1e-3          # the copy lands on the reference exactly
    assert 0.35 < frame[k["REF"], k["NOISY"]] < 0.7   # 0.3 A per coordinate ~ 0.52 A
    assert np.all(fit[~np.isnan(fit)] <= frame[~np.isnan(fit)] + 1e-4)
    assert n[k["REF"], k["NOISY"]] == 120
    assert np.allclose(frame, frame.T, equal_nan=True)


def test_identity_dirname():
    assert identity_dirname("P00918") == "P00918"
    assert identity_dirname("DEPOSITION_GROUP_G_1002238") == "DEPOSITION_GROUP_G_1002238"
    long = "+".join(f"P{k:05d}" for k in range(30))
    name = identity_dirname(long)
    assert len(name) < 64 and name.startswith("P00000+P00001_and28more_")
    assert identity_dirname(long) == name


def test_centring_equivalent_frames_compare_at_zero():
    """In C2 a frame is defined up to (1/2, 1/2, 0); two entries whose frames land one
    centring vector apart are the same placement and must compare at 0 A."""
    rng = np.random.default_rng(8)
    cell = (113.3, 53.2, 44.7, 90, 102.3, 90)
    x = _chain(rng)
    B = _orth(cell)
    f = x @ np.linalg.inv(B).T
    con = _db([("REF", cell, "C 1 2 1", 1.0, x),
               ("SAME", cell, "C 1 2 1", 2.0, (f + np.array([0.5, 0.5, 0.0])) @ B.T)])
    ident = "1,0,0;0,1,0;0,0,1"
    rows = [dict(pdb_id="REF", status="reference", member_chain=None, ref_chain="A",
                 unp_offset=0, L=ident, t=(0.0, 0.0, 0.0)),
            dict(pdb_id="SAME", status="placed", member_chain="A", ref_chain="A",
                 unp_offset=0, L=ident, t=(0.0, 0.0, 0.0))]
    ids, frame, fit, n = pairwise_rmsd(con, rows)
    assert frame[0, 1] < 1e-3 and fit[0, 1] < 1e-3
