"""Cα coordinates with UniProt residue numbers, for aligning entries of one crystal form.

For each PDB entry the deposited mmCIF is read once and reduced to what the
origin alignment needs:

* the resolution (``_refine.ls_d_res_high``, else ``_reflns.d_resolution_high``);
* the cell and space-group symbol written in the file;
* every Cα of the first model (first alternate location only), with its
  author chain id, and, where ``_struct_ref_seq`` maps the residue to a
  UniProt entry, the accession and UniProt residue number. The UniProt number
  is ``db_align_beg + (label_seq_id - seq_align_beg)`` inside the segment that
  holds the residue, so renumbered constructs, tags and insertion codes do not
  break the pairing between entries.

Coordinates are stored fractional, in the basis of the deposited cell.

Command line::

    python -m agentsg.cell.pdb_coords fetch clusters.duckdb ca.duckdb [--workers 16]

reads the entry ids from the ``members`` table of a cluster file (or any table
with a ``pdb_id`` column given by ``--table``), downloads each mmCIF from
``files.rcsb.org`` and writes ``entries`` and ``ca`` tables. It resumes: ids
already in ``entries`` are skipped.

Needs gemmi, numpy and duckdb.
"""
from __future__ import annotations

import gzip
import sys
import time
import urllib.request

RCSB_CIF = "https://files.rcsb.org/download/{id}.cif.gz"


def _float(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x


def extract_ca(cif_text):
    """Entry summary and Cα rows from one mmCIF text.

    Returns ``(entry, rows)``. ``entry`` holds ``resolution``, ``cell`` and
    ``sg_hm``; each row is ``(chain, acc, unp, fx, fy, fz)`` with ``acc`` and
    ``unp`` ``None`` for residues outside every UniProt segment.
    """
    import gemmi
    block = gemmi.cif.read_string(cif_text).sole_block()
    res = _float(block.find_value("_refine.ls_d_res_high"))
    if res is None:
        res = _float(block.find_value("_reflns.d_resolution_high"))
    cell = tuple(_float(block.find_value(f"_cell.{k}")) for k in
                 ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"))
    sg = block.find_value("_symmetry.space_group_name_H-M")
    sg = gemmi.cif.as_string(sg) if sg else None
    entry = dict(resolution=res, cell=cell, sg_hm=sg)
    if any(c is None for c in cell):
        return entry, []
    uc = gemmi.UnitCell(*cell)

    unp_refs = {r[0] for r in block.find("_struct_ref.", ["id", "db_name"])
                if gemmi.cif.as_string(r[1]) == "UNP"}
    segs = {}
    for r in block.find("_struct_ref_seq.", ["ref_id", "pdbx_strand_id", "seq_align_beg",
                                              "seq_align_end", "db_align_beg",
                                              "pdbx_db_accession"]):
        if r[0] not in unp_refs:
            continue
        try:
            seg = (int(r[2]), int(r[3]), int(r[4]), gemmi.cif.as_string(r[5]))
        except ValueError:
            continue
        segs.setdefault(gemmi.cif.as_string(r[1]), []).append(seg)

    rows, seen = [], set()
    model = None
    tab = block.find("_atom_site.", ["label_atom_id", "label_alt_id", "auth_asym_id",
                                     "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z",
                                     "pdbx_PDB_model_num", "type_symbol"])
    for r in tab:
        if r[0] != "CA" or r[8] != "C":
            continue
        if model is None:
            model = r[7]
        elif r[7] != model:
            break
        chain = gemmi.cif.as_string(r[2])
        try:
            seq = int(r[3])
        except ValueError:
            continue
        if (chain, seq) in seen:           # first alternate location only
            continue
        seen.add((chain, seq))
        acc = unp = None
        for beg, end, dbeg, a in segs.get(chain, ()):
            if beg <= seq <= end:
                acc, unp = a, dbeg + seq - beg
                break
        f = uc.fractionalize(gemmi.Position(float(r[4]), float(r[5]), float(r[6])))
        rows.append((chain, acc, unp, f.x, f.y, f.z))
    return entry, rows


def _download(pdb_id, timeout=60, retries=3):
    url = RCSB_CIF.format(id=pdb_id.upper())
    last = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return gzip.decompress(resp.read()).decode("utf-8", "replace")
        except Exception as exc:          # network errors, 404 for obsoleted ids
            last = exc
            time.sleep(1 + 2 * k)
    raise last


def _work(pdb_id):
    try:
        entry, rows = extract_ca(_download(pdb_id))
        return pdb_id, entry, rows, None
    except Exception as exc:
        return pdb_id, None, None, f"{type(exc).__name__}: {exc}"[:200]


def fetch_ca(ids, out_path, workers=16, progress=True, batch=500):
    """Download and reduce every id in ``ids`` into ``out_path`` (resumable)."""
    import concurrent.futures as cf

    import duckdb
    import pandas as pd

    con = duckdb.connect(out_path)
    con.execute("""CREATE TABLE IF NOT EXISTS entries (pdb_id VARCHAR PRIMARY KEY,
        resolution DOUBLE, a DOUBLE, b DOUBLE, c DOUBLE, alpha DOUBLE, beta DOUBLE,
        gamma DOUBLE, sg_hm VARCHAR, n_ca INTEGER, n_mapped INTEGER, error VARCHAR)""")
    con.execute("""CREATE TABLE IF NOT EXISTS ca (pdb_id VARCHAR, chain VARCHAR, acc VARCHAR,
        unp INTEGER, fx FLOAT, fy FLOAT, fz FLOAT)""")
    done = {r[0] for r in con.execute("SELECT pdb_id FROM entries").fetchall()}
    todo = [i for i in dict.fromkeys(ids) if i not in done]
    if progress:
        print(f"pdb_coords: {len(todo):,} to fetch ({len(done):,} already stored)", flush=True)
    t0 = time.time()
    ent_buf, ca_buf, n = [], [], 0

    def flush():
        if ent_buf:
            con.executemany("INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            ent_buf)
        if ca_buf:
            df = pd.DataFrame(ca_buf, columns=["pdb_id", "chain", "acc", "unp", "fx", "fy", "fz"])
            con.register("ca_df", df)
            con.execute("INSERT INTO ca SELECT * FROM ca_df")
            con.unregister("ca_df")
        ent_buf.clear()
        ca_buf.clear()

    with cf.ThreadPoolExecutor(workers) as ex:
        for pdb_id, entry, rows, err in ex.map(_work, todo):
            n += 1
            if err is not None:
                ent_buf.append((pdb_id, None, None, None, None, None, None, None, None, 0, 0, err))
            else:
                ent_buf.append((pdb_id, entry["resolution"], *entry["cell"], entry["sg_hm"],
                                len(rows), sum(1 for r in rows if r[1] is not None), None))
                ca_buf.extend((pdb_id, *r) for r in rows)
            if n % batch == 0:
                flush()
                if progress:
                    el = time.time() - t0
                    print(f"  {n:,}/{len(todo):,} {n / el:.1f}/s, ETA {(len(todo) - n) / (n / el) / 60:.0f} min",
                          flush=True)
    flush()
    con.close()
    return n


def _cli(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "fetch":
        print(__doc__)
        return 2
    src, out = argv[1], argv[2]
    workers, table = 16, "members"
    rest = argv[3:]
    while rest:
        flag = rest.pop(0)
        if flag == "--workers":
            workers = int(rest.pop(0))
        elif flag == "--table":
            table = rest.pop(0)
        else:
            print(f"unrecognised argument {flag!r}")
            return 2
    import duckdb
    con = duckdb.connect(src, read_only=True)
    ids = [r[0] for r in con.execute(f"SELECT DISTINCT pdb_id FROM {table} ORDER BY pdb_id").fetchall()]
    con.close()
    fetch_ca(ids, out, workers=workers)
    return 0


if __name__ == "__main__":               # pragma: no cover
    raise SystemExit(_cli())
