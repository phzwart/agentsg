"""Resumable full-PDB unit-cell database builder.

Downloads every current PDB entry's unit cell + space group + id from RCSB and
stores them in a DuckDB file with the six root-invariant components precomputed
on insert (see :mod:`agentsg.cell.celldb`). The build is resumable: rerunning
against an existing database skips ids already stored, so an interrupted crawl
resumes where it left off, losing at most one batch.

Command-line usage
------------------
Build (or resume) the whole PDB into ``pdb_cells.duckdb``::

    python -m agentsg.cell.pdb_app build pdb_cells.duckdb

Query the k nearest lattices to a cell::

    python -m agentsg.cell.pdb_app query pdb_cells.duckdb \\
        --cell 79 79 38 90 90 90 -k 5

Only the standard library and DuckDB are used (DuckDB via the optional
``agentsg[db]`` extra); the root precompute needs no extra dependency.
"""
from __future__ import annotations
import sys
import time
import urllib.error

from .celldb import CellDatabase, list_pdb_ids, fetch_pdb_cells


def _existing_ids(db):
    """Set of pdb_ids already present in the database (for resume)."""
    return {row[0] for row in db.sql("SELECT pdb_id FROM cells")}


def build_pdb_database(db_path="pdb_cells.duckdb", ids=None, resume=True,
                       batch_size=250, retries=5, backoff=2.0, timeout=60,
                       progress_every=20, log=print):
    """Download all PDB cells into ``db_path``, precomputing root invariants.

    Parameters
    ----------
    db_path : str
        DuckDB file to build (created if absent, appended to if present).
    ids : iterable of str, optional
        PDB ids to fetch. Defaults to the full RCSB holdings list.
    resume : bool
        Skip ids already stored in the database (default True).
    batch_size : int
        Ids per RCSB GraphQL request.
    retries : int
        Transient-error retry attempts per batch before the batch is skipped.
    backoff : float
        Exponential-backoff base (seconds): wait ``backoff * 2**attempt``.
    timeout : int
        Per-request network timeout (seconds).
    progress_every : int
        Emit a progress line every this many batches.
    log : callable
        Logging sink (default ``print``); pass a no-op to silence.

    Returns
    -------
    dict
        Summary: ids_total, ids_fetched, stored, skipped_existing,
        skipped_no_cell, failed_batches, seconds.
    """
    db = CellDatabase(db_path)
    t0 = time.time()
    if ids is None:
        log("listing RCSB holdings ...")
        ids = list_pdb_ids(timeout=timeout)
    ids = list(ids)
    ids_total = len(ids)

    already = _existing_ids(db) if resume else set()
    todo = [i for i in ids if i not in already]
    skipped_existing = ids_total - len(todo)
    log(f"{ids_total} ids total; {skipped_existing} already stored; "
        f"{len(todo)} to fetch")

    stored = 0
    seen = 0                    # records returned by RCSB (already cell-filtered)
    failed_batches = 0
    n_batches = (len(todo) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(todo), batch_size)):
        chunk = todo[start:start + batch_size]
        recs = None
        for attempt in range(retries):
            try:
                recs = list(fetch_pdb_cells(chunk, batch_size=batch_size,
                                            timeout=timeout))
                break
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, ConnectionError) as exc:
                wait = backoff * (2 ** attempt)
                log(f"  batch {bi + 1}/{n_batches} network error "
                    f"({exc}); retry {attempt + 1}/{retries} in {wait:.0f}s")
                time.sleep(wait)
        if recs is None:
            failed_batches += 1
            log(f"  batch {bi + 1}/{n_batches} FAILED after {retries} "
                f"attempts; skipping {len(chunk)} ids")
            continue
        seen += len(recs)
        for rec in recs:
            if db.add_cell(rec["pdb_id"],
                           (rec["a"], rec["b"], rec["c"],
                            rec["alpha"], rec["beta"], rec["gamma"]),
                           rec["sg_number"], rec["sg_hm"]):
                stored += 1
        if progress_every and (bi + 1) % progress_every == 0:
            el = time.time() - t0
            done = start + len(chunk)
            rate = done / el if el else 0.0
            eta = (len(todo) - done) / rate if rate else 0.0
            log(f"  batch {bi + 1}/{n_batches}: {done}/{len(todo)} fetched, "
                f"{stored} stored, {el:.0f}s elapsed, {rate:.0f} ids/s, "
                f"ETA {eta:.0f}s")

    seconds = time.time() - t0
    total_in_db = len(db)
    summary = dict(ids_total=ids_total, ids_fetched=len(todo),
                   records_with_cell=seen, stored=stored,
                   skipped_existing=skipped_existing,
                   root_rejected=seen - stored,
                   failed_batches=failed_batches, seconds=seconds,
                   rows_in_db=total_in_db)
    log(f"done: {stored} stored this run, {total_in_db} rows in DB, "
        f"{failed_batches} failed batches, {seconds:.0f}s")
    db.close()
    return summary


def _cli(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "build":
        if len(argv) < 2:
            print("usage: pdb_app build <db_path> [batch_size]")
            return 2
        db_path = argv[1]
        bs = int(argv[2]) if len(argv) > 2 else 250
        build_pdb_database(db_path, batch_size=bs)
        return 0
    if cmd == "backfill-similarity":
        if len(argv) < 2:
            print("usage: pdb_app backfill-similarity <db_path>")
            return 2
        db = CellDatabase(argv[1])
        n = db.backfill_similarity_invariants(progress=True)
        print(f"backfilled {n:,} rows in {argv[1]}")
        db.close()
        return 0
    if cmd == "query":
        # query <db> --cell a b c al be ga [-k K] [--sg N]
        db_path = argv[1]
        cell = None
        k = 10
        sg = None
        i = 2
        while i < len(argv):
            if argv[i] == "--cell":
                cell = tuple(float(x) for x in argv[i + 1:i + 7])
                i += 7
            elif argv[i] in ("-k", "--k"):
                k = int(argv[i + 1]); i += 2
            elif argv[i] == "--sg":
                sg = int(argv[i + 1]); i += 2
            else:
                i += 1
        if cell is None:
            print("query needs --cell a b c alpha beta gamma")
            return 2
        db = CellDatabase(db_path)
        for pdb_id, dist in db.nearest(cell, k=k, sg_number=sg):
            print(f"{pdb_id}\t{dist:.4f}")
        db.close()
        return 0
    print(f"unknown command: {cmd!r} (use build | backfill-similarity | query)")
    return 2


if __name__ == "__main__":               # pragma: no cover
    raise SystemExit(_cli())
