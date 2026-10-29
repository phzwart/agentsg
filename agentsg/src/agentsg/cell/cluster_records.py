"""All-pairs Cα RMSD per crystal-form cluster, and one record directory per protein identity.

Inputs are the three files of the cluster pipeline: the cluster file
(:mod:`agentsg.cell.clusters`), the Cα cache (:mod:`agentsg.cell.pdb_coords`) and
the frames (:mod:`agentsg.cell.cluster_frames`).

**Pairwise RMSD.** Every placed entry (and the reference) is put in the reference
frame with its stored transformation ``x_ref = L x + t``. Two numbers are computed
for every pair, on the Cα atoms the two entries share by UniProt residue number
(with the frame's numbering offset):

* ``frame_rmsd``: no fitting. Each entry is orthogonalised with its own cell
  written in the reference setting, so the value contains the shift of the
  molecule in the lattice as well as its internal change;
* ``fit_rmsd``: the same pairs after a Kabsch superposition.

Pairs that share fewer than ``min_common`` Cα atoms are left out.

**Records.** No coordinates are written. Each identity (the exact set of UniProt
accessions, or the deposition group of entries without one) gets a directory::

    <identity>/record.json            identity, name, and a summary per cluster
    <identity>/cluster_<id>/frames.tsv     one row per entry: resolution, deposited
                                      cell and space group, status, the exact L,
                                      t, and the placement statistics
    <identity>/cluster_<id>/pairwise.tsv.gz  pdb_a, pdb_b, frame_rmsd, fit_rmsd, n_common

and ``index.tsv`` at the top maps every PDB id to its directory and cluster. A
coordinate set in the reference frame is rebuilt from the deposited model and
the row of ``frames.tsv``.

Command line::

    python -m agentsg.cell.cluster_records clusters.duckdb ca.duckdb frames.duckdb out_dir

Needs numpy and duckdb.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from fractions import Fraction

MIN_COMMON = 20
FRAMES_COLUMNS = ("pdb_id", "status", "resolution", "sg_hm", "a", "b", "c", "alpha", "beta",
                  "gamma", "L", "t", "rmsd_to_ref", "n_paired", "margin", "asu_rmsd",
                  "accept", "n_classes", "class_price", "via", "member_chain", "ref_chain",
                  "unp_offset", "acc_fallback", "ambiguous_snap")


def _np():
    import numpy as np
    return np


def _orth(cell):
    from .cluster_frames import _orth as o
    return o(cell)


def _metric(cell):
    B = _orth(cell)
    return B.T @ B


def _cell_of_G(G):
    a, b, c = (math.sqrt(G[i, i]) for i in range(3))
    clip = lambda x: max(-1.0, min(1.0, x))
    return (a, b, c, math.degrees(math.acos(clip(G[1, 2] / (b * c)))),
            math.degrees(math.acos(clip(G[0, 2] / (a * c)))),
            math.degrees(math.acos(clip(G[0, 1] / (a * b)))))


def parse_L(text):
    """``"1,0,0;0,1,0;0,0,1"`` (rationals allowed) to a float 3x3 array."""
    np = _np()
    return np.array([[float(Fraction(x)) for x in row.split(",")] for row in text.split(";")])


def identity_dirname(label):
    """A short, filesystem-safe directory name for an identity label."""
    safe = re.sub(r"[^A-Za-z0-9_+.-]", "_", label)
    if len(safe) <= 64:
        return safe
    parts = safe.split("+")
    digest = hashlib.sha1(label.encode()).hexdigest()[:8]
    return f"{'+'.join(parts[:2])}_and{len(parts) - 2}more_{digest}"


# --------------------------------------------------------------- pairwise RMSD
def _placed_chains(ca, frames, ref_chain):
    """Per entry: (pdb_id, unp, Cartesian xyz in the reference frame, own metric)."""
    np = _np()
    ids = [f["pdb_id"] for f in frames]
    ent = {r[0]: r[1:] for r in ca.execute(
        "SELECT pdb_id, a, b, c, alpha, beta, gamma FROM entries WHERE pdb_id IN (SELECT unnest(?))",
        [ids]).fetchall()}
    out = []
    for f in frames:
        chain = f["member_chain"] if f["status"] != "reference" else ref_chain
        if chain is None or f["L"] is None:
            continue
        # "chain:accession" names one accession of a chain mapped to several (a fusion)
        ch, _, acc = chain.partition(":")
        pts = ca.execute("SELECT unp, fx, fy, fz FROM ca WHERE pdb_id = ? AND chain = ? "
                         "AND unp IS NOT NULL AND (? = '' OR acc = ?) ORDER BY acc, unp",
                         [f["pdb_id"], ch, acc, acc]).fetchnumpy()
        if not len(pts["unp"]):
            continue
        unp = pts["unp"].astype(int) + int(f["unp_offset"] or 0)
        x = np.stack([pts["fx"], pts["fy"], pts["fz"]], 1).astype(float)
        unp, first = np.unique(unp, return_index=True)
        x = x[first]
        L = parse_L(f["L"])
        t = np.array(f["t"], float)
        Li = np.linalg.inv(L)
        B = _orth(_cell_of_G(Li.T @ _metric(ent[f["pdb_id"]]) @ Li))
        out.append([f["pdb_id"], unp, x @ L.T + t, B])
    return out


def _centring_vectors(ca, pdb_id):
    """Centring translations (fractional) of the space group written in ``pdb_id``'s
    file, the zero vector included."""
    np = _np()
    from ..group import centering_translations
    from ..space_groups import space_group
    row = ca.execute("SELECT sg_hm FROM entries WHERE pdb_id = ?", [pdb_id]).fetchone()
    vecs = [np.zeros(3)]
    try:
        ops = space_group(row[0]).operations()
        vecs += [np.array([float(x) for x in c.v]) for c in centering_translations(ops)
                 if any(x != 0 for x in c.v)]
    except Exception:
        pass
    return vecs


def pairwise_rmsd(ca, frames, min_common=MIN_COMMON):
    """All-pairs frame and superposed Cα RMSD for one cluster.

    ``frames`` is a list of dicts (rows of the frames table for the cluster).
    Returns ``(ids, frame, fit, n_common)`` with square arrays (NaN below
    ``min_common`` shared residues).
    """
    np = _np()
    placed = [f for f in frames if f["status"] in ("placed", "reference")]
    refs = [f for f in placed if f["status"] == "reference"]
    if not refs:
        return [], None, None, None
    chains_used = [f["ref_chain"] for f in placed if f["status"] == "placed" and f["ref_chain"]]
    ref_chain = max(set(chains_used), key=chains_used.count) if chains_used else None
    if ref_chain is None:
        ref_chain = ca.execute(
            "SELECT chain FROM ca WHERE pdb_id = ? AND acc IS NOT NULL GROUP BY chain "
            "ORDER BY count(*) DESC, chain LIMIT 1", [refs[0]["pdb_id"]]).fetchone()
        ref_chain = ref_chain[0] if ref_chain else None
    rows = _placed_chains(ca, placed, ref_chain)
    if len(rows) < 2:
        return [r[0] for r in rows], None, None, None
    # every entry in the reference's copy: a frame is defined up to the lattice
    # translations, which include the centring translations of a centred group
    ref_id = refs[0]["pdb_id"]
    kref = next((k for k, r in enumerate(rows) if r[0] == ref_id), 0)
    ru, rx = rows[kref][1], rows[kref][2]
    cens = _centring_vectors(ca, ref_id)
    for r in rows:
        _, ir, im = np.intersect1d(ru, r[1], return_indices=True)
        if not len(ir):
            continue
        best = None
        for c in cens:
            y = r[2][im] + c
            shift = np.round((rx[ir] - y).mean(0))
            d = rx[ir] - y - shift
            score = float((d ** 2).sum())
            if best is None or score < best[0]:
                best = (score, c + shift)
        r[2] = r[2] + best[1]
    cols = np.unique(np.concatenate([r[1] for r in rows]))
    N, R = len(rows), len(cols)
    X = np.zeros((N, R, 3))
    M = np.zeros((N, R), bool)
    for k, (_pid, unp, x, B) in enumerate(rows):
        j = np.searchsorted(cols, unp)
        X[k, j] = x @ B.T
        M[k, j] = True
    frame = np.full((N, N), np.nan)
    fit = np.full((N, N), np.nan)
    ncom = np.zeros((N, N), np.int32)
    Mf = M.astype(float)
    for i in range(N):
        w = Mf * Mf[i]
        n = w.sum(1)
        ncom[i] = n
        ok = n >= min_common
        ok[i] = False
        if not ok.any():
            continue
        Wk = w[ok][..., None]
        A, Bi = X[ok] * Wk, X[i][None] * Wk
        nn = n[ok][:, None]
        d = A - Bi
        frame[i, ok] = np.sqrt((d ** 2).sum((1, 2)) / n[ok])
        ca_, cb_ = A.sum(1) / nn, Bi.sum(1) / nn
        A0 = (X[ok] - ca_[:, None]) * Wk
        B0 = (X[i][None] - cb_[:, None]) * Wk
        H = np.einsum("nra,nrb->nab", A0, B0)
        U, S, Vt = np.linalg.svd(H)
        S[:, 2] *= np.sign(np.linalg.det(U @ Vt))
        e0 = (A0 ** 2).sum((1, 2)) + (B0 ** 2).sum((1, 2))
        fit[i, ok] = np.sqrt(np.maximum(e0 - 2 * S.sum(1), 0) / n[ok])
    return [r[0] for r in rows], frame, fit, ncom


# --------------------------------------------------------------------- records
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _summary(v):
    np = _np()
    v = v[~np.isnan(v)]
    if not len(v):
        return None
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"median": round(float(q[2]), 3), "q25": round(float(q[1]), 3),
            "q75": round(float(q[3]), 3), "p5": round(float(q[0]), 3),
            "p95": round(float(q[4]), 3), "max": round(float(v.max()), 3), "n_pairs": int(len(v))}


def write_records(cluster_path, ca_path, frames_path, out_dir, min_common=MIN_COMMON,
                  cluster_ids=None, progress=True):
    """Write every identity directory and ``index.tsv`` under ``out_dir``."""
    import duckdb
    np = _np()
    cl = duckdb.connect(cluster_path, read_only=True)
    clusters = {r[0]: dict(zip(("cluster_id", "label", "name", "n", "medoid", "medoid_sg"), r))
                for r in cl.execute("SELECT cluster_id, label, name, n, medoid, medoid_sg FROM clusters").fetchall()}
    rel = cl.execute("SELECT * FROM run_info")
    info = dict(zip([d[0] for d in rel.description], rel.fetchone()))
    if "min_cluster" in info:                       # files from the standalone first run
        info["min_size"] = info.pop("min_cluster")
    info.pop("source_db", None)
    cl.close()
    fr = duckdb.connect(frames_path, read_only=True)
    rel = fr.execute("SELECT * FROM frames ORDER BY cluster_id, pdb_id")
    names = [d[0] for d in rel.description]
    by_cluster = {}
    for row in rel.fetchall():
        d = dict(zip(names, row))
        d["t"] = None if d["t0"] is None else (d["t0"], d["t1"], d["t2"])
        by_cluster.setdefault(d["cluster_id"], []).append(d)
    fr.close()
    ca = duckdb.connect(ca_path, read_only=True)
    ent = {r[0]: r[1:] for r in ca.execute(
        "SELECT pdb_id, resolution, sg_hm, a, b, c, alpha, beta, gamma FROM entries").fetchall()}

    os.makedirs(out_dir, exist_ok=True)
    by_label = {}
    for cid, c in clusters.items():
        if cluster_ids is not None and cid not in cluster_ids:
            continue
        if cid not in by_cluster:
            if progress:
                print(f"  cluster {cid}: no frames, skipped", flush=True)
            continue
        by_label.setdefault(c["label"], []).append(cid)
    index_rows = []
    t0 = time.time()
    for k, (label, cids) in enumerate(sorted(by_label.items())):
        dname = identity_dirname(label)
        ddir = os.path.join(out_dir, dname)
        os.makedirs(ddir, exist_ok=True)
        rec = {"identity": label, "accessions": [] if label.startswith("DEPOSITION_GROUP_") else label.split("+"),
               "deposition_group": label[len("DEPOSITION_GROUP_"):] if label.startswith("DEPOSITION_GROUP_") else None,
               "name": clusters[cids[0]]["name"], "clusters": []}
        for cid in sorted(cids):
            frames = by_cluster.get(cid, [])
            cdir = os.path.join(ddir, f"cluster_{cid}")
            os.makedirs(cdir, exist_ok=True)
            ref = next((f["pdb_id"] for f in frames if f["status"] == "reference"), None)
            with open(os.path.join(cdir, "frames.tsv"), "w", newline="") as fh:
                w = csv.writer(fh, delimiter="\t", lineterminator="\n")
                w.writerow(FRAMES_COLUMNS)
                for f in frames:
                    e = ent.get(f["pdb_id"], (None,) * 8)
                    w.writerow([f["pdb_id"], f["status"], _fmt(e[0], 2), e[1] or "",
                                *[_fmt(x, 3) for x in e[2:8]],
                                f["L"] or "",
                                "" if f["t"] is None else ",".join(f"{x:.6f}" for x in f["t"]),
                                _fmt(f["rmsd"]), f["n_paired"] or "", _fmt(f["margin"]),
                                _fmt(f["asu_rmsd"]), _fmt(f["accept"], 2), f["n_classes"] or "",
                                _fmt(f["class_price"], 4), f["via"] or "", f["member_chain"] or "",
                                f["ref_chain"] or "", f["unp_offset"] or 0,
                                _fmt(bool(f["acc_fallback"])), _fmt(bool(f["ambiguous_snap"]))])
                    index_rows.append((f["pdb_id"], dname, cid, f["status"], ref or ""))
            ids, frame, fit, ncom = pairwise_rmsd(ca, frames, min_common)
            n_pairs = 0
            with gzip.open(os.path.join(cdir, "pairwise.tsv.gz"), "wt", compresslevel=9) as fh:
                fh.write("pdb_a\tpdb_b\tframe_rmsd\tfit_rmsd\tn_common\n")
                if frame is not None:
                    iu, ju = np.triu_indices(len(ids), 1)
                    keep = ~np.isnan(frame[iu, ju])
                    for a, b in zip(iu[keep], ju[keep]):
                        fh.write(f"{ids[a]}\t{ids[b]}\t{frame[a, b]:.3f}\t{fit[a, b]:.3f}\t{ncom[a, b]}\n")
                    n_pairs = int(keep.sum())
            status = {}
            for f in frames:
                status[f["status"]] = status.get(f["status"], 0) + 1
            placed = [f for f in frames if f["status"] == "placed"]
            re_ = ent.get(ref, (None,) * 8) if ref else (None,) * 8
            iu = np.triu_indices(len(ids), 1) if frame is not None else None
            rec["clusters"].append({
                "cluster_id": cid, "n_members": clusters[cid]["n"], "reference": ref,
                "reference_resolution": re_[0], "space_group": re_[1],
                "reference_cell": None if re_[2] is None else [round(x, 3) for x in re_[2:8]],
                "status_counts": status, "n_pairs": n_pairs,
                "rmsd_to_reference": _summary(np.array([f["rmsd"] for f in placed if f["rmsd"] is not None], float)),
                "pairwise_frame_rmsd": None if frame is None else _summary(frame[iu]),
                "pairwise_fit_rmsd": None if frame is None else _summary(fit[iu]),
                "n_low_margin": sum(1 for f in placed if f["margin"] is not None and f["margin"] < 1.0),
                "n_via_neighbour": sum(1 for f in placed if f["via"]),
            })
        with open(os.path.join(ddir, "record.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
            fh.write("\n")
        if progress and (k < 5 or k % 100 == 0):
            print(f"  [{k + 1}/{len(by_label)}] {dname} {time.time() - t0:.0f}s", flush=True)
    ca.close()
    index_rows.sort()
    with open(os.path.join(out_dir, "index.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(("pdb_id", "identity_dir", "cluster_id", "status", "reference"))
        w.writerows(index_rows)
    with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
        json.dump({"cluster_rule": info, "min_common": min_common,
                   "n_identities": len(by_label), "n_clusters": sum(len(v) for v in by_label.values()),
                   "n_entries": len(index_rows)}, fh, indent=1)
        fh.write("\n")
    return len(by_label)


FAR = 3.0
"""Placements farther than this from the reference without fitting (Å) are flagged."""
SAME = 1.5
"""A far placement whose superposed RMSD to the reference is below this (Å) is the same
structure elsewhere in the lattice; above it, the structures themselves differ."""


def write_flags(out_dir):
    """Write ``flags.tsv``: one quality flag per placed entry, from the records.

    ``far_different_structure``  rmsd_to_ref > FAR and superposed RMSD to the reference
                           >= SAME (or no pair) after the best residue register: the
                           structures differ (conformation, construct, or a different chain);
    ``far_same_structure`` rmsd_to_ref > FAR, superposed RMSD < SAME: another packing in a
                           chained cluster, or a missed origin or setting;
    ``low_margin``         margin < 1 Å: the coordinates barely decide between placements;
    ``ok``                 otherwise.
    """
    import glob
    np = _np()
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*", "cluster_*", "frames.tsv"))):
        cdir = os.path.dirname(f)
        ident, cid = os.path.basename(os.path.dirname(cdir)), int(os.path.basename(cdir).split("_")[1])
        with open(f) as fh:
            frames = list(csv.DictReader(fh, delimiter="\t"))
        ref = next((r["pdb_id"] for r in frames if r["status"] == "reference"), None)
        fit = {}
        if ref:
            with gzip.open(os.path.join(cdir, "pairwise.tsv.gz"), "rt") as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    if r["pdb_a"] == ref:
                        fit[r["pdb_b"]] = float(r["fit_rmsd"])
                    elif r["pdb_b"] == ref:
                        fit[r["pdb_a"]] = float(r["fit_rmsd"])
        for r in frames:
            if r["status"] != "placed":
                continue
            d = float(r["rmsd_to_ref"]) if r["rmsd_to_ref"] else float("nan")
            ft = fit.get(r["pdb_id"], float("nan"))
            m = float(r["margin"]) if r["margin"] else float("nan")
            if d > FAR:
                flag = "far_same_structure" if ft < SAME else "far_different_structure"
            elif m < 1.0:
                flag = "low_margin"
            else:
                flag = "ok"
            rows.append((r["pdb_id"], ident, cid, _fmt(d), _fmt(ft), _fmt(m), r["via"], flag))
    rows.sort()
    with open(os.path.join(out_dir, "flags.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(("pdb_id", "identity_dir", "cluster_id", "rmsd_to_ref", "fit_to_ref", "margin",
                    "via", "flag"))
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r[-1]] = counts.get(r[-1], 0) + 1
    return counts


def _cli(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 4:
        print(__doc__)
        return 2
    write_records(*argv[:4])
    print(write_flags(argv[3]))
    return 0


if __name__ == "__main__":               # pragma: no cover
    raise SystemExit(_cli())
