"""Resumable full-PDB unit-cell database builder.

``build`` is the whole construction. It downloads current PDB cells from RCSB,
stores each deposited cell with its primitive-lattice keys and one
Selling-reduced cell, then fills UniProt lists from the PDBe SIFTS chain
table and deposition-group ids. The two annotations are stored separately.
Rerunning skips ids and columns already stored, so an interrupted run
resumes where it left off.

A prebuilt snapshot is https://doi.org/10.5281/zenodo.22986222.
Use that unless you need to rebuild.

Command-line usage
------------------
Build (or resume) the whole PDB into ``pdb_cells.duckdb``::

    python -m agentsg.cell.pdb_app build pdb_cells.duckdb

The same command on a random sample, for a check of every step::

    python -m agentsg.cell.pdb_app build pdb_cells.duckdb --sample 100

Query the k nearest lattices to a cell::

    python -m agentsg.cell.pdb_app query pdb_cells.duckdb \\
        --cell 79 79 38 90 90 90 -k 5

The ``backfill-*`` commands repeat one annotation step on a database that
already has cells. ``build`` runs them itself.

``add-cod`` inserts Crystallography Open Database cells (a COD table export,
one CIF, or a directory of CIFs) next to the PDB rows, keyed ``COD<id>`` with
``source = 'cod'``::

    python -m agentsg.cell.pdb_app add-cod pdb_cells.duckdb cod_cells.csv

Only the standard library and DuckDB are used (DuckDB via the optional
``agentsg[db]`` extra); the root precompute needs no extra dependency.
"""
from __future__ import annotations
import random
import sys
import time
import urllib.error

from .celldb import CellDatabase, list_pdb_ids, fetch_pdb_cells


def sample_ids(ids, n, seed=0):
    """Return ``n`` ids from ``ids``, stable for a given seed.

    The whole list is returned when ``n`` is at least its length. Order is the
    random sample order, not the holdings order.
    """
    ids = list(ids)
    if n >= len(ids):
        return ids
    return random.Random(seed).sample(ids, n)


def construct_pdb_database(db_path="pdb_cells.duckdb", ids=None, sample=None,
                           seed=0, batch_size=250, log=print, **kwargs):
    """Build a cell database and fill every derived column.

    Steps, in order: RCSB cell download (primitive-lattice keys and one
    Selling cell are written on insert), then any missing linear keys,
    similarity keys, and Selling cells, then UniProt lists, then
    deposition-group ids on every row. ``sample``
    draws that many ids from ``ids`` or, when ``ids`` is omitted, from the
    current RCSB holdings.
    """
    if sample is not None:
        pool = list(ids) if ids is not None else list_pdb_ids(timeout=kwargs.get("timeout", 60))
        ids = sample_ids(pool, sample, seed)
        log(f"sample {len(ids)} of {len(pool)} PDB ids (seed {seed})")
    summary = build_pdb_database(
        db_path, ids=ids, batch_size=batch_size, log=log, **kwargs,
    )
    db = CellDatabase(db_path)
    try:
        log("linear keys ...")
        summary["linear"] = db.backfill_linear_keys(progress=True)
        log("similarity keys ...")
        summary["similarity"] = db.backfill_similarity_invariants(progress=True)
        log("Selling cells ...")
        summary["selling"] = db.backfill_selling_cells(progress=True)
        log("UniProt lists ...")
        summary["uniprot"] = db.backfill_uniprot(progress=True)
        log("deposition groups ...")
        summary["deposit_group"] = db.backfill_deposit_group(progress=True)
    finally:
        db.close()
    log(
        f"construct done: {summary['rows_in_db']} rows, "
        f"uniprot {summary['uniprot']}, deposition groups {summary['deposit_group']}"
    )
    return summary


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


def _parse_build_args(argv):
    """``db_path``, batch size, sample count, and seed from ``build`` arguments."""
    db_path = argv[0]
    batch = 250
    sample = None
    seed = 0
    i = 1
    if i < len(argv) and not str(argv[i]).startswith("-"):
        batch = int(argv[i])
        i += 1
    while i < len(argv):
        flag = argv[i]
        if flag == "--batch" and i + 1 < len(argv):
            batch = int(argv[i + 1])
            i += 2
        elif flag == "--sample" and i + 1 < len(argv):
            sample = int(argv[i + 1])
            i += 2
        elif flag == "--seed" and i + 1 < len(argv):
            seed = int(argv[i + 1])
            i += 2
        else:
            raise ValueError(f"unrecognised build argument {flag!r}")
        if sample is not None and sample < 1:
            raise ValueError("--sample must be at least 1")
        if batch < 1:
            raise ValueError("--batch must be at least 1")
    return db_path, batch, sample, seed


def _cli(argv=None):
    """Command-line entry point for pdb_app commands."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "build":
        if len(argv) < 2:
            print("usage: pdb_app build <db_path> [batch_size] [--sample N] [--seed N] [--batch N]")
            return 2
        try:
            db_path, batch, sample, seed = _parse_build_args(argv[1:])
        except ValueError as exc:
            print(f"usage: pdb_app build <db_path> [batch_size] [--sample N] [--seed N] [--batch N]\n{exc}")
            return 2
        construct_pdb_database(db_path, sample=sample, seed=seed, batch_size=batch)
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
    if cmd == "backfill-linear":
        if len(argv) < 2:
            print("usage: pdb_app backfill-linear <db_path>")
            return 2
        db = CellDatabase(argv[1])
        n = db.backfill_linear_keys(progress=True)
        print(f"backfilled {n:,} linear keys in {argv[1]}")
        db.close()
        return 0
    if cmd == "backfill-selling":
        if len(argv) < 2:
            print("usage: pdb_app backfill-selling <db_path>")
            return 2
        db = CellDatabase(argv[1])
        n = db.backfill_selling_cells(progress=True)
        print(f"backfilled {n:,} Selling-reduced cells in {argv[1]}")
        db.close()
        return 0
    if cmd == "backfill-uniprot":
        if len(argv) < 2:
            print("usage: pdb_app backfill-uniprot <db_path> [--refresh]")
            return 2
        refresh = "--refresh" in argv[2:]
        db = CellDatabase(argv[1])
        n = db.backfill_uniprot(progress=True, refresh=refresh)
        print(f"backfilled {n:,} UniProt lists in {argv[1]}")
        db.close()
        return 0
    if cmd == "backfill-deposit-group":
        if len(argv) < 2:
            print("usage: pdb_app backfill-deposit-group <db_path> [--refresh]")
            return 2
        refresh = "--refresh" in argv[2:]
        db = CellDatabase(argv[1])
        n = db.backfill_deposit_group(progress=True, refresh=refresh)
        print(f"backfilled {n:,} deposition groups in {argv[1]}")
        db.close()
        return 0
    if cmd == "add-cod":
        if len(argv) < 3:
            print("usage: pdb_app add-cod <db_path> <cod_table.csv | file.cif | cif_dir>")
            return 2
        db = CellDatabase(argv[1])
        n_ok, n_skip = db.add_cod(argv[2], progress=True)
        print(f"added {n_ok:,} COD cells to {argv[1]} ({n_skip:,} skipped)")
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
    print(
        f"unknown command: {cmd!r} "
        "(use build | backfill-similarity | backfill-linear | backfill-selling | "
        "backfill-uniprot | backfill-deposit-group | add-cod | query)"
    )
    return 2


if __name__ == "__main__":               # pragma: no cover
    raise SystemExit(_cli())
