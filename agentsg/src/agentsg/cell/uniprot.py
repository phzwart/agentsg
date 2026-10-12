"""Ragged UniProt lists for PDB entries, from the PDBe SIFTS chain table.

Each entry keeps every distinct UniProt accession, a short name, and how many
residues that accession covers. The same accession on several chains is one
list element. The list is ordered by coverage. Nothing here decides which
accession is the protein of a complex beyond that order.
"""
from __future__ import annotations

import gzip
import io
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SIFTS_CHAIN_UNIPROT = (
    "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/"
    "pdb_chain_uniprot.tsv.gz"
)
UNIPROT_NAMES = (
    "https://rest.uniprot.org/uniprotkb/stream?"
    + urllib.parse.urlencode({
        "query": "database:pdb",
        "fields": "accession,id,protein_name",
        "format": "tsv",
        "compressed": "true",
    })
)

_LIST_TYPE = "STRUCT(acc VARCHAR, name VARCHAR, n_res INTEGER)[]"
_ACCESSION = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


def uniprot_list_type():
    """DuckDB type of the ``cells.uniprots`` column."""
    return _LIST_TYPE


def assemble_uniprot_lists(chain_uniprot, names=None):
    """Build ``{PDB_ID: [(acc, name, n_res), ...]}`` from the SIFTS chain table.

    ``n_res`` is the summed residue span of every segment that carries the
    accession, using ``RES_BEG``/``RES_END`` and falling back to
    ``SP_BEG``/``SP_END``. Names come from the UniProt table: the recommended
    protein name, otherwise the entry name. The list is sorted by ``n_res``
    descending, then by accession. PDB ids absent from the chain table are
    omitted; the caller stores those as an empty list.
    """
    release, rows = _read_tsv(chain_uniprot)
    span_sum = {}
    accessions = {}
    for row in rows:
        pdb = _pdb_id(row, "PDB", "PDBID")
        acc = _acc(row, "SP_PRIMARY", "ACCESSION", "UNIPROT")
        if not pdb or not acc:
            continue
        span = _span(row, "RES_BEG", "RES_END")
        if span is None:
            span = _span(row, "SP_BEG", "SP_END")
        span_sum[(pdb, acc)] = span_sum.get((pdb, acc), 0) + (span or 0)
        accessions.setdefault(pdb, set()).add(acc)

    name_of = _read_names(names) if names is not None else {}
    out = {}
    for pdb, accs in accessions.items():
        items = [
            (acc, name_of.get(acc), span_sum[(pdb, acc)])
            for acc in accs
        ]
        items.sort(key=lambda item: (-item[2], item[0]))
        out[pdb] = items
    return out, release


def fetch_uniprot_tables(*, log=None):
    """Download the SIFTS chain table and UniProt names, then assemble lists.

    Returns ``(lists, source)`` where ``source`` is the SIFTS release note.
    The downloads live in a temporary directory and are removed before this
    returns.
    """
    with tempfile.TemporaryDirectory() as tmp:
        chain = Path(tmp) / "chain_uniprot.tsv.gz"
        names = Path(tmp) / "names.tsv.gz"
        if log:
            log("downloading SIFTS pdb_chain_uniprot ...")
        _download(SIFTS_CHAIN_UNIPROT, chain)
        if log:
            log("downloading UniProt names ...")
        _download(UNIPROT_NAMES, names)
        return assemble_uniprot_lists(chain, names)


def _download(url, dest):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "agentsg", "Accept": "*/*"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as fh, dest.open("wb") as out:
            shutil.copyfileobj(fh, out)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not download {url}") from exc


def _open_text(source):
    if hasattr(source, "read"):
        return source, False
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and Path(source).is_file()):
        path = Path(source)
        raw = path.read_bytes()[:2]
        if raw == b"\x1f\x8b" or path.suffix == ".gz":
            return gzip.open(path, "rt"), True
        return path.open("rt"), True
    if isinstance(source, str):
        return io.StringIO(source), True
    raise TypeError(f"expected a path, file, or TSV text, got {type(source).__name__}")


def _read_tsv(source):
    """Return ``(release_note, [row dict, ...])``. Comment lines are skipped."""
    fh, close = _open_text(source)
    release = None
    header = None
    rows = []
    try:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                if release is None:
                    release = line[1:].strip()
                continue
            parts = line.split("\t")
            if header is None:
                header = [part.strip() for part in parts]
                continue
            if len(parts) < len(header):
                parts = parts + [""] * (len(header) - len(parts))
            rows.append(dict(zip(header, parts)))
    finally:
        if close:
            fh.close()
    return release, rows


def _read_names(source):
    """Map a UniProt accession to a short display name.

    The UniProt stream header is ``Entry``, ``Entry Name``, ``Protein names``.
    The display name is the recommended protein name, cut before the first
    parenthetical, or the entry name when the protein name is missing.
    """
    fh, close = _open_text(source)
    names = {}
    header = None
    try:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if header is None:
                keys = [part.strip().upper() for part in parts]
                if "ENTRY" in keys or "ACCESSION" in keys or "ENTRY NAME" in keys:
                    header = keys
                    continue
                header = []
            if header:
                row = dict(zip(header, parts))
                acc = _acc(row, "ENTRY", "ACCESSION")
                protein = _text(row, "PROTEIN NAMES", "PROTEIN NAME")
                entry_name = _text(row, "ENTRY NAME", "ID")
            else:
                acc = _accession(parts[0] if parts else "")
                entry_name = parts[1].strip() if len(parts) > 1 else ""
                protein = parts[2].strip() if len(parts) > 2 else ""
            if not acc or acc in names:
                continue
            names[acc] = _display_name(protein, entry_name)
    finally:
        if close:
            fh.close()
    return names


def _display_name(protein, entry_name):
    protein = (protein or "").strip()
    if protein:
        cut = protein.find(" (")
        if cut > 0:
            protein = protein[:cut].strip()
        if protein:
            return protein
    entry_name = (entry_name or "").strip()
    return entry_name or None


def _field(row, *names):
    wanted = {name.upper() for name in names}
    for key, value in row.items():
        if key.upper() in wanted:
            return value.strip() if isinstance(value, str) else value
    return ""


def _text(row, *names):
    value = _field(row, *names)
    return "" if value is None else str(value).strip()


def _pdb_id(row, *names):
    text = _text(row, *names)
    return text.upper() if text else ""


def _accession(text):
    text = str(text or "").strip().upper()
    return text if _ACCESSION.match(text) else ""


def _acc(row, *names):
    return _accession(_text(row, *names))


def _span(row, start_name, end_name):
    try:
        start = int(_text(row, start_name))
        end = int(_text(row, end_name))
    except ValueError:
        return None
    if end < start:
        return None
    return end - start + 1
