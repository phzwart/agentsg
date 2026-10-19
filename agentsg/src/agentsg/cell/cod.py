"""Crystallography Open Database (COD) cell records, from a table export or CIFs.

COD (https://www.crystallography.net/cod/) is the open counterpart of the
ICSD/CSD: ~500k small-molecule and inorganic structures. Its cell table can be
exported from the web interface or the MySQL dump as CSV/TSV with the columns
``file, a, siga, b, sigb, c, sigc, alpha, sigalpha, beta, sigbeta, gamma,
siggamma, vol, sigvol, sg, sgHall, formula, ...``; the structures themselves
are CIFs, one per COD id, obtainable by svn/rsync or per-file download.

This module reads either shape into plain records that
:meth:`agentsg.cell.celldb.CellDatabase.add_cod` inserts next to the PDB
cells. Standard library only; no network.

A record is a dict with ``cod_id`` (the numeric COD id as a string), ``cell``
(a, b, c, alpha, beta, gamma), ``esd`` (six values or None where the file gives
none), ``sg_hm``, ``sg_hall``, ``sg_number`` (as given in the file, or None)
and ``formula``.
"""
from __future__ import annotations

import csv
import os
import re

_NUM_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:\((\d+)\))?\s*$")

_CELL_TAGS = ("_cell_length_a", "_cell_length_b", "_cell_length_c",
              "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")
_HM_TAGS = ("_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M")
_HALL_TAGS = ("_space_group_name_Hall", "_symmetry_space_group_name_Hall")
_NUMBER_TAGS = ("_space_group_IT_number", "_symmetry_Int_Tables_number")
_FORMULA_TAGS = ("_chemical_formula_sum",)
_ID_TAGS = ("_cod_database_code",)

# column aliases in COD table exports (case-insensitive)
_TABLE_COLUMNS = {
    "cod_id": ("file", "cod_id", "codid", "id", "cod_database_code"),
    "a": ("a",), "b": ("b",), "c": ("c",),
    "alpha": ("alpha",), "beta": ("beta",), "gamma": ("gamma",),
    "siga": ("siga", "sig_a"), "sigb": ("sigb", "sig_b"), "sigc": ("sigc", "sig_c"),
    "sigalpha": ("sigalpha", "sig_alpha"), "sigbeta": ("sigbeta", "sig_beta"),
    "siggamma": ("siggamma", "sig_gamma"),
    "sg_hm": ("sg", "sg_hm", "spacegroup", "space_group"),
    "sg_hall": ("sghall", "sg_hall", "hall"),
    "sg_number": ("sgnumber", "sg_number", "it_number"),
    "formula": ("formula", "chemical_formula_sum"),
}


def parse_cif_number(text):
    """``'5.4310(2)'`` -> ``(5.431, 0.0002)``; a bare number gives esd None.

    Returns ``(None, None)`` for CIF placeholders (``.``, ``?``), blanks and
    anything that is not a number.
    """
    if text is None:
        return None, None
    s = str(text).strip().strip("'\"")
    if s in ("", ".", "?", "NULL", "null", "None", "NA", "N/A"):
        return None, None
    m = _NUM_RE.match(s)
    if not m:
        return None, None
    value = float(m.group(1))
    if m.group(2) is None:
        return value, None
    mant = m.group(1)
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    if "e" in mant.lower():                       # esd on an exponent form: rare; take it as written
        decimals = 0
    return value, int(m.group(2)) * 10.0 ** (-decimals)


def _clean_symbol(s):
    if s is None:
        return None
    s = str(s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1].strip()
    return s or None


def _record(cod_id, vals, sigs, sg_hm, sg_hall, sg_number, formula):
    if cod_id is None or any(v is None for v in vals):
        return None
    try:
        number = int(str(sg_number).strip()) if sg_number not in (None, "", ".", "?") else None
    except ValueError:
        number = None
    cod_id = str(cod_id).strip()
    if cod_id.upper().startswith("COD"):
        cod_id = cod_id[3:].strip()
    return {
        "cod_id": cod_id,
        "cell": tuple(float(v) for v in vals),
        "esd": tuple(sigs),
        "sg_hm": _clean_symbol(sg_hm),
        "sg_hall": _clean_symbol(sg_hall),
        "sg_number": number,
        "formula": _clean_symbol(formula),
    }


# ---- table export ---------------------------------------------------------
def _sniff_delimiter(header_line):
    for d in ("\t", ",", ";", "|"):
        if d in header_line:
            return d
    return ","


def cod_cells_from_table(path, *, encoding="utf-8"):
    """Yield records from a COD CSV/TSV export (header row required).

    Leading blank lines and ``#`` comments are skipped. The COD web export
    starts with a comment block before the column names.
    """
    with open(path, "r", encoding=encoding, errors="replace", newline="") as fh:
        first = ""
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            first = line
            break
        if not first:
            return
        delim = _sniff_delimiter(first)
        header = [h.strip().strip('"').lower() for h in first.rstrip("\r\n").split(delim)]
        col = {}
        for key, names in _TABLE_COLUMNS.items():
            for n in names:
                if n in header:
                    col[key] = header.index(n)
                    break
        missing = [k for k in ("cod_id", "a", "b", "c", "alpha", "beta", "gamma") if k not in col]
        if missing:
            raise ValueError(f"COD table {path}: missing columns {missing}; header={header}")

        def get(row, key):
            i = col.get(key)
            return row[i] if i is not None and i < len(row) else None

        for row in csv.reader(fh, delimiter=delim):
            if not row or all(not x.strip() for x in row):
                continue
            vals, sigs = [], []
            for key in ("a", "b", "c", "alpha", "beta", "gamma"):
                v, e = parse_cif_number(get(row, key))
                s, _ = parse_cif_number(get(row, "sig" + key))
                vals.append(v)
                sigs.append(s if s is not None else e)
            rec = _record(get(row, "cod_id"), vals, sigs, get(row, "sg_hm"),
                          get(row, "sg_hall"), get(row, "sg_number"), get(row, "formula"))
            if rec is not None:
                yield rec


# ---- CIF -------------------------------------------------------------------
_TAG_RE = re.compile(r"^(_\S+)\s*(.*)$")


def _cif_items(text):
    """Yield (block_name, {tag: value}) for each data block; loops are skipped.

    A minimal reader: it takes single-line ``_tag value`` items (quoted or not)
    and ``_tag`` followed by a semicolon text field, which covers every cell,
    symmetry and formula tag COD writes.
    """
    blocks, cur, name = [], None, None
    lines = text.splitlines()
    i, n = 0, len(lines)
    in_loop = False
    while i < n:
        line = lines[i]
        s = line.strip()
        i += 1
        if not s or s.startswith("#"):
            continue
        low = s.lower()
        if low.startswith("data_"):
            if cur is not None:
                blocks.append((name, cur))
            name, cur, in_loop = s[5:], {}, False
            continue
        if cur is None:
            continue
        if low.startswith("loop_"):
            in_loop = True
            continue
        if s.startswith("_"):
            m = _TAG_RE.match(s)
            tag, rest = m.group(1), m.group(2).strip()
            if in_loop and not rest:
                continue                    # loop header tag
            in_loop = False
            if not rest:
                # value on the next line, possibly a semicolon field
                if i < n and lines[i].startswith(";"):
                    buf = [lines[i][1:]]
                    i += 1
                    while i < n and not lines[i].startswith(";"):
                        buf.append(lines[i])
                        i += 1
                    i += 1
                    rest = " ".join(x.strip() for x in buf).strip()
                elif i < n:
                    rest = lines[i].strip()
                    i += 1
            cur[tag.lower()] = rest
            continue
        # a bare value line outside a tag: loop body or stray; ignore
    if cur is not None:
        blocks.append((name, cur))
    return blocks


def _first(items, tags):
    for t in tags:
        v = items.get(t.lower())
        if v is not None and v.strip() not in ("", ".", "?"):
            return v
    return None


def cod_cells_from_cif(path, *, encoding="utf-8"):
    """Yield records from one CIF (every data block that carries a cell).

    The COD id is ``_cod_database_code`` when present, else the data block
    name, else the file's stem.
    """
    with open(path, "r", encoding=encoding, errors="replace") as fh:
        text = fh.read()
    stem = os.path.splitext(os.path.basename(path))[0]
    for name, items in _cif_items(text):
        vals, sigs = [], []
        for t in _CELL_TAGS:
            v, e = parse_cif_number(items.get(t.lower()))
            vals.append(v)
            sigs.append(e)
        if any(v is None for v in vals):
            continue
        cod_id = _first(items, _ID_TAGS) or (name if name and name.strip() else stem)
        rec = _record(cod_id, vals, sigs, _first(items, _HM_TAGS), _first(items, _HALL_TAGS),
                      _first(items, _NUMBER_TAGS), _first(items, _FORMULA_TAGS))
        if rec is not None:
            yield rec


# ---- dispatch --------------------------------------------------------------
def iter_cod_cells(path):
    """Yield records from a table export, a CIF, or a directory of CIFs.

    A directory is walked recursively; ``.cif`` files are read as CIFs and
    ``.csv``/``.tsv``/``.txt`` files as table exports.
    """
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                p = os.path.join(root, f)
                ext = os.path.splitext(f)[1].lower()
                if ext == ".cif":
                    yield from cod_cells_from_cif(p)
                elif ext in (".csv", ".tsv", ".txt"):
                    yield from cod_cells_from_table(p)
        return
    ext = os.path.splitext(path)[1].lower()
    if ext == ".cif":
        yield from cod_cells_from_cif(path)
    else:
        yield from cod_cells_from_table(path)


def centring_symbol(sg_hm, sg_hall=None):
    """The symbol whose lattice letter describes the deposited cell's centring.

    COD writes rhombohedral groups on rhombohedral axes as ``R 3 m :R``; that
    cell is primitive, so the centring letter is ``P``. ``:H`` (hexagonal axes)
    is the ordinary R centring. Without an H-M symbol the Hall symbol is used
    (its lattice letter is the same, after a leading ``-``).
    """
    s = sg_hm or sg_hall
    if not s:
        return None
    s = s.strip()
    tail = s.replace(" ", "")
    if tail.endswith(":R"):
        return "P"
    if tail.endswith(":H"):
        return s[: s.rfind(":")].strip()
    return s


__all__ = [
    "parse_cif_number",
    "cod_cells_from_table",
    "cod_cells_from_cif",
    "iter_cod_cells",
    "centring_symbol",
]
