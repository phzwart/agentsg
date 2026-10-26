"""z65 — the PDB numbers (main §4.3, §4.4, §5.4, Table 10, abstract, Conclusions; SI Tables
S8, S9, S14 and Section S9).

Two layers.  The ``pdb`` claims read the deposited results files of
``manuscript/supplement/benchmark`` (and the deposited ``pdb_lepage.jsonl``) and assert every
number the paper prints, at the precision it is printed: a printed value v with d decimals
holds when the file's value lies within half a unit of the last digit of v.  The
``reproduce`` claims rerun the scripts on the PDB snapshot into a temporary directory and
assert that the regenerated output equals the deposited file; none of these outputs carries
a wall-clock timing, so the comparison is exact.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

from sellingproofs import paths
from sellingproofs.registry import claim


# --- reading the deposited files ---------------------------------------------------------

def _read(name: str) -> str:
    p = paths.BENCH / name
    assert p.is_file(), f"deposited file missing: {p}"
    return p.read_text()


def _line(text: str, pattern: str) -> re.Match:
    m = re.search(pattern, text, re.M)
    assert m, f"no line matching {pattern!r}"
    return m


def near(printed: str | float, value: float) -> bool:
    """value rounds to the printed number: |value - printed| <= half a unit of its last digit."""
    s = str(printed).replace(",", "").replace("\\,", "")
    d = len(s.split(".")[1]) if "." in s else 0
    return abs(float(value) - float(s)) <= 0.5 * 10 ** -d + 1e-9


def assert_near(printed, value, what=""):
    assert near(printed, value), f"{what}: paper prints {printed}, file has {value}"


def assert_range(lo, hi, values, what=""):
    """a printed range lo-hi covers the values at its precision: min rounds to lo, max to hi."""
    assert_near(lo, min(values), what + " (low end)")
    assert_near(hi, max(values), what + " (high end)")


def _lepage_entries():
    return [json.loads(l) for l in _read("pdb_lepage.jsonl").splitlines() if l.strip()]


def _controls():
    """results_pdb_band_controls.txt as {key: (n, p50, p90, p95)} for the section-1 rows."""
    t = _read("results_pdb_band_controls.txt")
    rows = {}
    for m in re.finditer(r"^\s+cut\s+(\d+) %  n=\s*(\d+)\s+([\d.]+)/([\d.]+)/([\d.]+)\s*$", t, re.M):
        rows[f"cut{m.group(1)}"] = (int(m.group(2)), *map(float, m.groups()[2:]))
    m = _line(t, r"identical cells dropped  n=\s*(\d+)\s+([\d.]+)/([\d.]+)/([\d.]+)")
    rows["identical"] = (int(m.group(1)), *map(float, m.groups()[1:]))
    m = _line(t, r"each block weighted equally \((\d+) blocks, largest (\d+) pairs\)\s+([\d.]+)/([\d.]+)/([\d.]+)")
    rows["blocks"] = (int(m.group(1)), int(m.group(2)), *map(float, m.groups()[2:]))
    ident, cheap = {}, {}
    sec = t.split("by reported Bravais type (cut 20 %):")[1].split("with the cheapest operator")[0]
    for m in re.finditer(r"^\s+(\w\w)\s+n=\s*(\d+)\s+([\d.]+)/([\d.]+)/([\d.]+)\s*$", sec, re.M):
        ident[m.group(1)] = (int(m.group(2)), *map(float, m.groups()[2:]))
    sec = t.split("with the cheapest operator")[1].split("\n2.")[0]
    for m in re.finditer(r"^\s+(\w\w)\s+n=\s*(\d+)\s+([\d.]+)/([\d.]+)/([\d.]+)\s+\(identity: ([\d.]+)/([\d.]+)/([\d.]+)\)", sec, re.M):
        cheap[m.group(1)] = (int(m.group(2)), *map(float, m.groups()[2:5]))
        assert ident[m.group(1)][1:] == tuple(map(float, m.groups()[5:])), m.group(1)
    return t, rows, ident, cheap


def _bootstrap():
    """results_pdb_band_uncertainty.txt: {type: (pairs, blocks, flagged, [(p, lo, hi)] x 3)}, shares."""
    t = _read("results_pdb_band_uncertainty.txt")
    num = r"([\d.]+) \[\s*([\d.]+)-\s*([\d.]+)\]"
    rows = {}
    for m in re.finditer(rf"^(\w+)\s+(\d+)\s+(\d+)\s+{num}\s+{num}\s+{num}(.*)$", t, re.M):
        g = m.groups()
        q = [tuple(map(float, g[3 + 3 * k: 6 + 3 * k])) for k in range(3)]
        rows[g[0]] = (int(g[1]), int(g[2]), "fewer than 200 blocks" in g[-1], q)
    shares = {}
    for m in re.finditer(r"^\s+(one band|own type's band), (\d\dth)\s+([\d.]+) %\s+\[\s*([\d.]+)-\s*([\d.]+)\]", t, re.M):
        shares[(m.group(1), m.group(2))] = tuple(map(float, m.groups()[2:]))
    return t, rows, shares


TYPES = ["aP", "mP", "mC", "oP", "oS", "oI", "oF", "tP", "tI", "hR", "hP", "cP", "cI", "cF"]


# --- the pdb claims: paper against the deposited files -----------------------------------

@claim("z65", "main §4.4", "main §5.3", kind="pdb")
def test_snapshot_and_entries_analysed():
    """The snapshot holds 206,214 deposited cells; Le Page's analysis covers 206,141 of them,
    31 placeholder cells with an edge below 2 A and 42 cells the cctbx analysis did not
    accept being left out."""
    t = _read("results_pdb_lepage.txt")
    m = _line(t, r"^(\d+) entries, (\d+) not analysed")
    assert int(m.group(1)) == 206214 and int(m.group(2)) == 73
    R = _lepage_entries()
    assert len(R) == 206214
    err = [r["error"] for r in R if "error" in r]
    assert len(err) == 73 and err.count("placeholder") == 31 and len(err) - err.count("placeholder") == 42
    assert len(R) - len(err) == 206141


@claim("z65", "main §5.3", "main Fig. 3", "SI Table S15", kind="pdb")
def test_nonisomorphism_percentiles():
    """Between deposited cells of one crystal form (45,222 pairs) the price after the scale
    fit has a median of 0.98%, a 90th percentile of 4.2% and a 95th of 6.7% (95% intervals
    with deposition blocks resampled, 4.1-4.3 and 6.5-7.0%); without the scale fit the median
    is 1.6%."""
    t = _read("results_pdb_nonisomorphism.txt")
    m = _line(t, r"^(\d+) id-adjacent pairs with the same space group; (\d+) below 20 % strain")
    assert int(m.group(2)) == 45222
    q = {int(m.group(1)): (float(m.group(2)), float(m.group(3)))
         for m in re.finditer(r"^\s+(\d+)%\s+([\d.]+)%\s+([\d.]+)%\s+[\d.]+%$", t, re.M)}
    assert_near("0.98", q[50][1], "median after fit")
    assert_near("4.2", q[90][1], "90th"); assert_near("6.7", q[95][1], "95th")
    assert_near("1.6", q[50][0], "median without fit")
    _, rows, _ = _bootstrap()
    n, b, _, ((p50, *_), (p90, l90, h90), (p95, l95, h95)) = rows["all"]
    assert (n, p50, p90, p95) == (45222, q[50][1], q[90][1], q[95][1])
    assert_near("4.1", l90); assert_near("4.3", h90); assert_near("6.5", l95); assert_near("7.0", h95)


@claim("z65", "main §5.3", kind="pdb")
def test_band_against_the_crystal_form_cut():
    """With the cut between forms at 5, 10, 20 and 50% the 90th percentile is 2.5, 3.4, 4.2 and
    5.6% and the 95th 3.1, 4.7, 6.7 and 11.9%. Dropping the 1,464 pairs of identical cells
    changes little (4.3 and 6.9%), and weighting each deposition block equally, so that
    fragment-screening series of up to 35 pairs count once, gives 5.0 and 8.0%."""
    _, rows, _, _ = _controls()
    for cut, p90, p95 in (("5", "2.5", "3.1"), ("10", "3.4", "4.7"), ("20", "4.2", "6.7"), ("50", "5.6", "11.9")):
        assert_near(p90, rows[f"cut{cut}"][2], f"cut {cut} 90th"); assert_near(p95, rows[f"cut{cut}"][3], f"cut {cut} 95th")
    assert rows["cut20"][0] == 45222
    ident = int(_line(_read("results_pdb_nonisomorphism.txt"), r"exactly identical cells: (\d+)").group(1))
    assert ident == 1464 and rows["identical"][0] == 45222 - 1464
    assert_near("4.3", rows["identical"][2]); assert_near("6.9", rows["identical"][3])
    assert rows["blocks"][1] == 35
    assert_near("5.0", rows["blocks"][3]); assert_near("8.0", rows["blocks"][4])


# Table 10 as printed: type: (pairs, median, 90th, 95th, cheapest 90th, cheapest 95th)
TABLE10 = {
    "aP": (902, "2.06", "7.66", "11.68", "6.14", "9.89"), "mP": (7077, "1.27", "5.38", "9.02", "5.29", "8.80"),
    "mC": (4932, "1.49", "5.06", "8.40", "4.82", "7.90"), "oP": (13615, "1.11", "4.70", "7.43", "4.65", "7.39"),
    "oS": (2271, "1.01", "3.57", "5.50"), "oI": (1353, "1.22", "5.12", "7.91"), "oF": (31, "1.99", "5.87", "7.54"),
    "tP": (5174, "0.52", "2.59", "4.00"), "tI": (805, "0.65", "2.76", "4.91"), "hR": (1270, "0.68", "3.54", "5.86"),
    "hP": (6817, "0.57", "2.87", "4.56"), "cP": (363, "0.00", "0.00", "0.00"), "cI": (385, "0.00", "0.00", "0.00"),
    "cF": (211, "0.00", "0.00", "0.00"),
}


@claim("z65", "main Table 10", kind="pdb")
def test_table10_nonisomorphism_by_type():
    """Table 10: pairs, median, 90th and 95th percentile of the price between deposited cells of
    one crystal form by reported Bravais type, and for aP, mP, mC and oP the 90th and 95th with
    the cheapest operator."""
    _, _, ident, cheap = _controls()
    assert list(ident) == TYPES
    for bt, row in TABLE10.items():
        n, *q = ident[bt]
        assert n == row[0], bt
        for printed, v in zip(row[1:4], q):
            assert_near(printed, v, bt)
        if len(row) > 4:
            assert cheap[bt][0] == n
            assert_near(row[4], cheap[bt][2], bt + " cheapest 90th"); assert_near(row[5], cheap[bt][3], bt + " cheapest 95th")
    assert set(cheap) == {"aP", "mP", "mC", "oP"}


@claim("z65", "main §5.3", kind="pdb")
def test_nonisomorphism_by_type_in_the_text():
    """By reported Bravais type the 90th and 95th percentiles are 2.6 and 4.0% for tP, 2.9 and
    4.6% for hP, 4.7 and 7.4% for oP, 5.1 and 8.4% for mC and 5.4 and 9.0% for mP; for the cubic
    types they are 0. For aP they are 7.7 and 11.7% with the identity as the operator, and 6.1
    and 9.9% with the cheapest operator; for mP, mC and oP the cheapest operator changes them by
    0.5 points or less. At the 20% cut, 2.6-6.1% at the 90th percentile and 4-10% at the 95th
    across the non-cubic types (aP with the cheapest operator). The default band of 7.4% lies
    above the pooled 95th percentile and above the 90th percentile of every type except aP with
    the identity as the operator (7.7%; 95% interval 6.4-9.5%); the oF row rests on 31 pairs in
    22 blocks, and the cubic rows are zero."""
    _, rows, ident, cheap = _controls()
    for bt, a, b in (("tP", "2.6", "4.0"), ("hP", "2.9", "4.6"), ("oP", "4.7", "7.4"), ("mC", "5.1", "8.4"),
                     ("mP", "5.4", "9.0"), ("aP", "7.7", "11.7")):
        assert_near(a, ident[bt][2], bt); assert_near(b, ident[bt][3], bt)
    for bt in ("cP", "cI", "cF"):
        assert ident[bt][1:] == (0.0, 0.0, 0.0)
    assert_near("6.1", cheap["aP"][2]); assert_near("9.9", cheap["aP"][3])
    for bt in ("mP", "mC", "oP"):
        assert max(ident[bt][2] - cheap[bt][2], ident[bt][3] - cheap[bt][3]) <= 0.5 + 1e-9, bt
    noncubic = [bt for bt in TYPES if bt[0] != "c"]
    p90 = [cheap["aP"][2] if bt == "aP" else ident[bt][2] for bt in noncubic]
    p95 = [cheap["aP"][3] if bt == "aP" else ident[bt][3] for bt in noncubic]
    assert_range("2.6", "6.1", p90, "90th across types"); assert_range("4", "10", p95, "95th across types")
    assert rows["cut20"][3] < 7.4
    assert [bt for bt in TYPES if ident[bt][2] >= 7.4] == ["aP"]
    _, boot, _ = _bootstrap()
    assert_near("6.4", boot["aP"][3][1][1]); assert_near("9.5", boot["aP"][3][1][2])
    assert boot["oF"][:2] == (31, 22)


@claim("z65", "main §5.3", "abstract", kind="pdb")
def test_pseudo_symmetry_counts():
    """Le Page's analysis at 3 deg finds a higher lattice than the reported one for 43,201 of the
    206,141 entries analysed (21%); for 545 (0.3%) the higher lattice is exact. For the 42,656
    entries without an exact higher lattice the largest price matches 2 sqrt2 artanh(sin delta)
    (ratio 1.000 at the 10th, 50th and 90th percentiles), and the cheapest operator has a median
    price of 6.7%; the cheapest alternative lies within the 90th percentile of non-isomorphism
    for 35% of the pseudo-symmetric entries and within the 95th for 50%."""
    t = _read("results_pdb_lepage.txt")
    n_ps = int(_line(t, r"pseudo-symmetric entries \(Le Page higher than reported\): (\d+)").group(1))
    higher = [int(m.group(1)) for m in re.finditer(r"^\w\w(?:\s+\d+){14}\s+\d+\s+(\d+)$", t, re.M)]
    assert len(higher) == 14 and sum(higher) == n_ps == 43201
    p = _read("results_pdb_pricing.txt")
    m = _line(p, r"^(\d+) entries; (\d+) pseudo-symmetric at 3 deg; (\d+) with an exact extra lattice symmetry \(price 0\); (\d+) priced only")
    tot, ps, ex, pr = map(int, m.groups())
    assert (tot, ps, ex, pr) == (206141, 43201, 545, 42656) and ex + pr == ps
    assert_near("21", 100 * ps / tot); assert_near("0.3", 100 * ex / tot)
    m = _line(p, r"median ([\d.]+), 10-90 % ([\d.]+)-([\d.]+)")
    assert m.groups() == ("1.000", "1.000", "1.000")
    m = _line(p, r"^all\s+(\d+)\s+([\d.]+)%\s+(\d+)%\s+(\d+)%\s+(\d+)%")
    assert int(m.group(1)) == 42656
    assert_near("6.7", m.group(2)); assert m.group(4) == "35" and m.group(5) == "50"


@claim("z65", "main §5.3", "abstract", "main §6", kind="pdb")
def test_shares_of_entries_with_an_alternative_within_the_band():
    """Over all entries, 7.5% (95% interval 7.4-7.7%) have an alternative, exact or priced,
    within 4.2% and 10.6% (10.4-11.0%) within 6.7%; with the band of the entry's own Bravais
    type, the pooled band standing in for a type with fewer than 200 deposition blocks, 8.6%
    (8.4-9.0%) and 12.5% (12.0-13.0%). (Abstract: the cell alone cannot exclude an alternative
    indexing from a higher lattice for 7.5-8.6% of entries.)"""
    _, rows, shares = _bootstrap()
    want = {("one band", "90th"): ("7.5", "7.4", "7.7"), ("one band", "95th"): ("10.6", "10.4", "11.0"),
            ("own type's band", "90th"): ("8.6", "8.4", "9.0"), ("own type's band", "95th"): ("12.5", "12.0", "13.0")}
    for k, (v, lo, hi) in want.items():
        assert_near(v, shares[k][0], str(k)); assert_near(lo, shares[k][1], str(k)); assert_near(hi, shares[k][2], str(k))
    assert {bt for bt, r in rows.items() if r[2]} == {"oF", "cP", "cI", "cF"}
    assert all(r[1] >= 200 for bt, r in rows.items() if bt != "all" and not r[2])
    c = _read("results_pdb_band_controls.txt")
    assert_near("7.5", _line(c, r"one band, 90th percentile 4.2 %: priced [\d.]+ %, with the 545 exact cases ([\d.]+) %").group(1))
    assert_near("10.6", _line(c, r"one band, 95th percentile 6.7 %: priced [\d.]+ %, with the 545 exact cases ([\d.]+) %").group(1))


def _null():
    t = _read("results_pdb_lepage_null.txt")
    assert "1500 cells per arm" in t
    num = r"([\d.]+)±([\d.]+) / ([\d.]+)±([\d.]+) / ([\d.]+)±([\d.]+)"
    out = {}
    for m in re.finditer(rf"^\s+(.+?)\s+seed (\d): real {num};\s+shuffled {num};\s+shuffled within volume {num}$", t, re.M):
        g = m.groups()
        sg = g[0]
        vals = [tuple(map(float, g[2 + 6 * a: 8 + 6 * a])) for a in range(3)]   # (rate, ±) x 3 per arm
        out.setdefault(sg, []).append(vals)
    assert set(out) == {"P 1", "P 1 21 1", "C 1 2 1", "P 21 21 21", "P 21 21 2"}
    assert all(len(v) == 2 for v in out.values())
    return out


def _arm(out, sg, arm, k):
    """the rates of one arm (0 real, 1 shuffled, 2 within volume; both if arm is None) for
    quantity k (0: higher lattice at 3 deg, 1: <= 1 %, 2: <= 4.2 %), over the two seeds."""
    arms = (1, 2) if arm is None else (arm,)
    return [s[a][2 * k] for s in out[sg] for a in arms]


@claim("z65", "main §5.3, Against chance", "SI §S9, Against chance", kind="pdb")
def test_against_chance():
    """At Le Page's 3 deg the real rate exceeds both shuffled ones for P1 (42-44% against
    30-40%) and P2_12_12_1 (23-24% against 13-17%), exceeds them slightly for P2_1 (46% against
    39-44%), and matches them for C2 and P2_12_12. The share whose cheapest alternative costs
    at most 1% is 10% for real P1 cells against 2-3% shuffled, and 9% against 5-6% for P2_1
    (intervals about +-1 point). At 4.2% the excess remains for P1 (15-16% against 6-9%) but not
    clearly for P2_1 (18% against 13-18%, intervals about +-2 points); two draws of 1500 cells
    per arm."""
    o = _null()
    for sg, k, real, shuf in (("P 1", 0, ("42", "44"), ("30", "40")), ("P 21 21 21", 0, ("23", "24"), ("13", "17")),
                              ("P 1 21 1", 0, ("46", "46"), ("39", "44")),
                              ("P 1", 1, ("10", "10"), ("2", "3")), ("P 1 21 1", 1, ("9", "9"), ("5", "6")),
                              ("P 1", 2, ("15", "16"), ("6", "9")), ("P 1 21 1", 2, ("18", "18"), ("13", "18"))):
        assert_range(*real, _arm(o, sg, 0, k), f"{sg} real {k}")
        assert_range(*shuf, _arm(o, sg, None, k), f"{sg} shuffled {k}")
    for sg in ("P 1", "P 21 21 21", "P 1 21 1"):                                    # exceeds both shuffled
        assert min(_arm(o, sg, 0, 0)) > max(_arm(o, sg, None, 0)), sg
    for sg in ("C 1 2 1", "P 21 21 2"):                                             # matches them
        sh = _arm(o, sg, None, 0)
        assert min(sh) <= min(_arm(o, sg, 0, 0)) and max(_arm(o, sg, 0, 0)) <= max(sh), sg
    pm = lambda sg, k: [s[a][2 * k + 1] for s in o[sg] for a in range(3)]          # noqa: E731
    for sg in ("P 1", "P 1 21 1"):
        assert all(0.5 <= x <= 1.6 for x in pm(sg, 1)), sg                          # about +-1 point
        assert all(1.0 <= x <= 2.0 for x in pm(sg, 2)), sg                          # about +-2 points


@claim("z65", "main §5.3, What the cell decides", "abstract", kind="pdb")
def test_what_the_cell_decides():
    """Reindexing the second cell of 3,000 randomly chosen same-form pairs against the first with
    a band of 7.4%, a wrong class lies within the band at a positive price in 7.9% of pairs, and
    at exactly the identity's price in 17%. A wrong class is strictly cheaper than the
    identity's in 36 pairs (1.2%; 0.8-1.6% with deposition blocks resampled): in 12 by at most
    0.05 points, in 20 through a relabelling of axes, and in 4 otherwise (0.13%, 0.03-0.27%).
    In 125 pairs (4.2%) the two cells differ by more than 7.4%; in 9 of them (0.3%) only a wrong
    class lies within 7.4%. So for about one pair in thirteen the cell leaves a priced
    alternative open, and it ranks a wrong indexing first in between about 0.1% and 1.2% of
    pairs."""
    t = _read("results_pdb_band_controls.txt")
    sec = t.split("3. misranking on ")[1]
    assert sec.startswith("3000 same-form pairs (cut 20 %), band 7.4 %")
    c = {m.group(1).strip(): (int(m.group(2)), float(m.group(3)))
         for m in re.finditer(r"^\s+(.+?): (\d+) \(([\d.]+) %\)$", sec, re.M)}
    assert _line(sec, r"pairs scored (\d+), refused (\d+)").groups() == ("3000", "0")
    n, pc = c["a priced wrong class within the band"]; assert_near("7.9", pc)
    assert_near("17", c["a wrong class at the same price (exact lattice symmetry)"][1])
    n, pc = c["wrong class strictly cheaper"]; assert n == 36; assert_near("1.2", pc)
    assert c["by at most 0.05 points (below the precision of a deposited cell)"][0] == 12
    assert c["by more, through a relabelling of axes (a setting convention)"][0] == 20
    n, pc = c["by more, otherwise"]; assert n == 4; assert_near("0.13", pc)
    assert 12 + 20 + 4 == 36
    n, pc = c["identity beyond the band"]; assert n == 125; assert_near("4.2", pc)
    n, pc = c["of which a wrong class within the band"]; assert n == 9; assert_near("0.3", pc)
    m = _line(sec, r"strictly cheaper ([\d.]+)-([\d.]+) %, by more and not a relabelling ([\d.]+)-([\d.]+) %")
    for printed, v in zip(("0.8", "1.6", "0.03", "0.27"), m.groups()):
        assert_near(printed, v)
    assert round(100 / c["a priced wrong class within the band"][1]) == 13            # one pair in thirteen
    assert_near("0.1", c["by more, otherwise"][1], "abstract, low end")                # 0.1-1.2 %
    assert_near("1.2", c["wrong class strictly cheaper"][1], "abstract, high end")


@claim("z65", "main §4.3", "main §5.3", "main §6", kind="pdb")
def test_band_coverage():
    """Against non-isomorphism, 7.4% covers 95.7% of the pairs of one crystal form in the PDB,
    9.9% covers 97.5% and 14.8% covers 99.4%; the pairs beyond 7.4%, the upper tail of
    non-isomorphism, are 4.3% of all 45,222 same-form pairs, with a median price of 10.7%."""
    t = _read("results_pdb_band_coverage.txt")
    assert _line(t, r"^(\d+) pairs of one crystal form \(cut 20 %\)").group(1) == "45222"
    b = {m.group(1): tuple(map(float, m.groups()[1:])) for m in re.finditer(
        r"^band ([\d.]+) %: ([\d.]+) % of pairs within; ([\d.]+) % beyond \(median price of those ([\d.]+) %\)", t, re.M)}
    assert_near("95.7", b["7.4"][0]); assert_near("97.5", b["9.9"][0]); assert_near("99.4", b["14.8"][0])
    assert_near("4.3", b["7.4"][1]); assert_near("10.7", b["7.4"][2])
    nf = _read("results_pdb_nonisomorphism.txt")
    assert_near(_line(nf, r"pairs within  7.4 %:\s+[\d.]+ % \(no fit\),\s+([\d.]+) % \(fit\)").group(1), b["7.4"][0])


@claim("z65", "main §5.3, Where", kind="pdb")
def test_where_the_pseudo_symmetry_lies():
    """18% of the 35,364 mP entries and 16% of the 8,491 aP entries have an alternative within
    4.2%, against 7% of oP and 1% of tP entries. Among the transitions, mP to hP has a median
    price of 0.71%, 91% within 4.2%; for most transitions the median price of the cheapest
    alternative is 6-7.5%."""
    p = _read("results_pdb_pricing.txt")
    by = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(r"^\s+(\w\w):\s+(\d+) % of (\d+)$", p, re.M)}
    assert by["mP"] == (18, 35364) and by["aP"] == (16, 8491) and by["oP"][0] == 7 and by["tP"][0] == 1
    tr = _transitions(p)
    assert tr["mP->hP"][1] == "0.71" and tr["mP->hP"][3] == "91"
    med = [float(v[1]) for k, v in tr.items() if k != "all"]
    assert sum(6 <= x <= 7.5 for x in med) > len(med) / 2


def _transitions(p):
    return {m.group(1): (int(m.group(2)), m.group(3), m.group(4), m.group(5), m.group(6))
            for m in re.finditer(r"^(\w\w->\w\w|all)\s+(\d+)\s+([\d.]+)%\s+(\d+)%\s+(\d+)%\s+(\d+)%$", p, re.M)}


# SI Table S8 as printed: sort: (type, entries, delta, largest, smallest)
TABLE_S8 = {
    "C1": ("cI", 593, "0.14/1.64/2.73", "0.69/8.10/13.58", "0.32/7.13/13.18"),
    "C3": ("cF", 562, "0.19/1.72/2.76", "0.95/8.51/13.61", "0.50/7.68/13.41"),
    "C5": ("cP", 1349, "0.29/1.70/2.72", "1.43/8.40/13.45", "0.47/5.52/12.59"),
    "H4": ("hP", 2008, "0.08/1.40/2.70", "0.40/6.91/13.32", "0.11/4.03/12.70"),
    "T1": ("tI", 1376, "0.06/1.32/2.67", "0.28/6.60/13.33", "0.17/5.55/12.60"),
    "T2": ("tI", 330, "0.05/1.48/2.69", "0.24/7.30/13.43", "0.11/4.98/12.44"),
    "T5": ("tP", 14816, "0.22/1.47/2.69", "1.07/7.27/13.43", "0.70/6.56/12.98"),
    "R1": ("hR", 96, "0.04/1.55/2.76", "0.22/7.64/13.78", "0.13/2.98/13.13"),
    "R3": ("hR", 393, "0.07/1.82/2.81", "0.37/9.09/13.86", "0.29/7.64/13.72"),
    "O1A": ("oF", 1126, "0.23/1.51/2.70", "1.12/7.45/13.33", "1.02/7.05/13.17"),
    "O1B": ("oI", 331, "0.10/1.58/2.78", "0.49/7.80/13.73", "0.36/7.63/13.73"),
    "O2": ("oI", 1078, "0.14/1.40/2.73", "0.70/6.92/13.50", "0.48/6.33/13.36"),
    "O4": ("oS", 11679, "0.14/1.49/2.74", "0.67/7.36/13.56", "0.62/7.15/13.51"),
    "O5": ("oP", 4760, "0.03/1.08/2.59", "0.17/5.33/12.79", "0.15/5.13/12.69"),
    "M1A": ("mC", 867, "0.72/2.09/2.83", "3.53/10.33/13.98", "3.53/10.33/13.98"),
    "M1B": ("mC", 398, "0.12/1.80/2.72", "0.57/8.91/13.41", "0.57/8.91/13.41"),
    "M2A": ("mC", 854, "0.13/1.75/2.77", "0.66/8.64/13.66", "0.66/8.64/13.66"),
    "M2B": ("mC", 2, "0.61/0.95/1.29", "3.02/4.70/6.38", "3.02/4.70/6.38"),
    "M3": ("mC", 1, "2.39/2.39/2.39", "11.80/11.80/11.80", "11.80/11.80/11.80"),
    "M4": ("mP", 582, "0.10/1.40/2.65", "0.47/6.89/13.07", "0.47/6.89/13.07"),
}


@claim("z65", "SI Table S8", "SI §S8", kind="pdb")
def test_table_s8_pseudo_symmetry_by_sort():
    """SI Table S8: pseudo-symmetric PDB entries by the Delone sort of the Le Page lattice, with
    the 10th/50th/90th percentiles of the largest obliquity and of the largest and smallest
    price. Over the sorts with more than two entries, the median largest price lies between 5.3%
    (O5, oP) and 10.3% (M1A); the median largest obliquity between 1.1 deg and 2.1 deg; within a
    sort the median largest price of the operators of order 3, 4 or 6 is within half a
    percentage point of that of the twofolds."""
    t = _read("results_pdb_lepage.txt")
    got = {}
    for m in re.finditer(r"^(\w+)\s+(\w\w)\s+(\d+)\s+delta\s+(\S+)\s+largest\s+(\S+)\s+smallest\s+(\S+)\s+(.*)$", t, re.M):
        orders = {k: float(v) for k, v in re.findall(r"(\d): ([\d.]+)", m.group(7))}
        got[m.group(1)] = (m.group(2), int(m.group(3)), m.group(4), m.group(5), m.group(6), orders)
    assert "sort not classified" not in t
    assert {k: v[:5] for k, v in got.items()} == TABLE_S8
    big = {k: v for k, v in got.items() if v[1] > 2}
    med_largest = {k: float(v[3].split("/")[1]) for k, v in big.items()}
    med_delta = [float(v[2].split("/")[1]) for v in big.values()]
    assert min(med_largest, key=med_largest.get) == "O5" and max(med_largest, key=med_largest.get) == "M1A"
    assert_near("5.3", med_largest["O5"]); assert_near("10.3", med_largest["M1A"])
    assert_range("1.1", "2.1", med_delta, "median obliquity")
    for k, v in big.items():
        o = v[5]
        assert all(abs(o[n] - o["2"]) <= 0.5 for n in o if n != "2"), k


# SI Table S9 as printed: transition: (entries, median, <=1, <=4.2, <=6.7)
TABLE_S9 = {
    "oP->tP": (12120, "6.98", "9", "31", "48"), "mP->oS": (9474, "7.38", "12", "33", "46"),
    "mP->oP": (4417, "5.78", "20", "42", "55"), "aP->mC": (2113, "9.49", "9", "19", "30"),
    "mC->oS": (1831, "7.41", "12", "33", "45"), "mP->tP": (1542, "3.66", "27", "54", "69"),
    "mC->oI": (1308, "7.13", "11", "30", "48"), "oS->hP": (1169, "7.34", "10", "26", "45"),
    "oI->tI": (1165, "6.25", "19", "38", "52"), "mC->oF": (1067, "7.29", "8", "29", "46"),
    "oS->tP": (880, "7.42", "14", "32", "46"), "tP->cP": (724, "7.10", "7", "28", "47"),
    "aP->mP": (571, "6.97", "15", "30", "48"), "mP->hP": (479, "0.71", "61", "91", "100"),
    "mC->hR": (391, "8.92", "19", "29", "37"), "mC->tI": (391, "4.56", "23", "46", "64"),
    "all": (42656, "6.69", "14", "35", "50"),
}


@claim("z65", "SI Table S9", kind="pdb")
def test_table_s9_transitions():
    """SI Table S9: the cheapest pseudo-symmetric alternative of PDB entries without an exact
    higher lattice, by transition from the reported Bravais type to the one Le Page's 3 deg
    finds: entries, median price, and the shares within 1%, 4.2% and 6.7%; the sixteen most
    frequent transitions and all."""
    assert _transitions(_read("results_pdb_pricing.txt")) == TABLE_S9


@claim("z65", "SI Table S15", "SI §S9, Intervals", kind="pdb")
def test_table_s14_bootstrap_intervals():
    """SI Table S15: non-isomorphism by reported Bravais type with block-bootstrap 95%
    intervals (1000 resamples of the deposition blocks): pairs, blocks, and the median, 90th
    and 95th percentiles with their intervals; the types with fewer than 200 deposition blocks
    are oF, cP, cI and cF."""
    t, rows, _ = _bootstrap()
    assert _line(t, r"(\d+) resamples of (\d+) blocks; (\d+) entries").groups() == ("1000", "20206", "206141")
    S14 = {
        "aP": (902, 535, "2.06 (1.82--2.27) & 7.66 (6.40--9.46) & 11.68 (9.76--13.70)"),
        "mP": (7077, 3277, "1.27 (1.21--1.33) & 5.38 (5.01--5.84) & 9.02 (8.21--9.78)"),
        "mC": (4932, 1977, "1.49 (1.41--1.59) & 5.06 (4.73--5.49) & 8.40 (7.16--9.23)"),
        "oP": (13615, 5987, "1.11 (1.07--1.14) & 4.70 (4.46--4.95) & 7.43 (7.06--7.81)"),
        "oS": (2271, 1108, "1.01 (0.94--1.08) & 3.57 (3.32--4.01) & 5.50 (4.96--6.14)"),
        "oI": (1353, 574, "1.22 (1.07--1.39) & 5.12 (4.28--6.16) & 7.91 (6.40--9.15)"),
        "oF": (31, 22, "1.99 (1.23--3.10) & 5.87 (3.39--8.28) & 7.54 (4.48--8.95)"),
        "tP": (5174, 2061, "0.52 (0.48--0.57) & 2.59 (2.36--2.85) & 4.00 (3.61--4.41)"),
        "tI": (805, 416, "0.65 (0.53--0.78) & 2.76 (2.28--3.34) & 4.91 (3.61--6.03)"),
        "hR": (1270, 597, "0.68 (0.61--0.79) & 3.54 (3.10--3.94) & 5.86 (4.59--6.71)"),
        "hP": (6817, 3167, "0.57 (0.53--0.62) & 2.87 (2.65--3.04) & 4.56 (4.22--4.95)"),
        "cP": (363, 178, "0.00 (0.00--0.00) & 0.00 (0.00--0.00) & 0.00 (0.00--0.00)"),
        "cI": (385, 194, "0.00 (0.00--0.00) & 0.00 (0.00--0.00) & 0.00 (0.00--0.00)"),
        "cF": (211, 106, "0.00 (0.00--0.00) & 0.00 (0.00--0.00) & 0.00 (0.00--0.00)"),
        "all": (45222, 20206, "0.98 (0.96--1.00) & 4.21 (4.09--4.33) & 6.72 (6.52--6.98)"),
    }
    assert list(rows) == TYPES + ["all"]
    for bt, (n, b, cells) in S14.items():
        assert rows[bt][:2] == (n, b), bt
        printed = [tuple(float(x) for x in re.match(r"([\d.]+) \(([\d.]+)--([\d.]+)\)", c.strip()).groups())
                   for c in cells.split("&")]
        assert printed == [tuple(q) for q in rows[bt][3]], bt
    assert {bt for bt, r in rows.items() if r[2]} == {"oF", "cP", "cI", "cF"}


@claim("z65", "SI §S9, Intervals", kind="pdb")
def test_intervals_of_the_shares():
    """The shares of entries with an alternative within the band have intervals of +-0.2 to
    +-0.5 points."""
    _, _, shares = _bootstrap()
    half = [(hi - lo) / 2 for _, lo, hi in shares.values()]
    assert len(half) == 4
    assert_range("0.2", "0.5", half, "half-widths of the share intervals")


@claim("z65", "SI §S9, The crystal-form proxy", kind="pdb")
def test_crystal_form_proxy():
    """Random pairs of entries with one space group whose identifiers do not share their first
    three characters pass the 20% cut in 1.5% of 19,986 cases (0.75% at 5%)."""
    t = _read("results_pdb_proxy_check.txt")
    m = _line(t, r"^(\d+) random same-space-group pairs .*?: ([\d.]+) % below a 5 % cut, ([\d.]+) % below a 10 % cut, ([\d.]+) % below a 20 % cut")
    assert m.group(1) == "19986"
    assert_near("0.75", m.group(2)); assert_near("1.5", m.group(4))


# --- the reproduce claims: rerun the scripts -----------------------------------------------

def _run(script: str, tmp, *args, timeout=4 * 3600, out=True) -> str:
    """run a benchmark script with its outputs in tmp (out=False: read and write the deposited
    directory, for a script that only reads a deposited file and prints)."""
    env = dict(os.environ, SELLINGPROOFS_DATA=str(paths.DATA),
               PYTHONPATH=os.pathsep.join([str(paths.REPO / "agentsg" / "src"), str(paths.SUPPLEMENT / "src"),
                                           os.environ.get("PYTHONPATH", "")]))
    if out:
        env["SELLINGPROOFS_OUT"] = str(tmp)
    else:
        env.pop("SELLINGPROOFS_OUT", None)
    r = subprocess.run([sys.executable, str(paths.BENCH / script), *args], cwd=tmp, env=env,
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def _same(regenerated: str, deposited_name: str):
    dep = _read(deposited_name)
    a, b = regenerated.rstrip("\n").split("\n"), dep.rstrip("\n").split("\n")
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) and not diff, f"{deposited_name}: {len(a)} vs {len(b)} lines; first differences {diff[:5]}"


@claim("z65", "main §5.3", "main Fig. 3", kind="reproduce", needs=("pdb",))
def test_reproduce_pdb_nonisomorphism(tmp_path):
    """pdb_nonisomorphism.py on the snapshot prints results_pdb_nonisomorphism.txt exactly."""
    _same(_run("pdb_nonisomorphism.py", tmp_path), "results_pdb_nonisomorphism.txt")


@claim("z65", "main §5.3", "SI Table S9", kind="reproduce")
def test_reproduce_pdb_pricing_summary(tmp_path):
    """pdb_pricing_summary.py on the deposited pdb_lepage.jsonl prints results_pdb_pricing.txt
    exactly."""
    _same(_run("pdb_pricing_summary.py", tmp_path), "results_pdb_pricing.txt")


@claim("z65", "main §4.3", "main §5.3", kind="reproduce", needs=("pdb",))
def test_reproduce_pdb_band_coverage(tmp_path):
    """pdb_band_coverage.py on the snapshot writes results_pdb_band_coverage.txt exactly."""
    _run("pdb_band_coverage.py", tmp_path)
    _same((tmp_path / "results_pdb_band_coverage.txt").read_text(), "results_pdb_band_coverage.txt")


@claim("z65", "SI §S9", kind="reproduce", needs=("pdb",))
def test_reproduce_pdb_proxy_check(tmp_path):
    """pdb_proxy_check.py on the snapshot writes results_pdb_proxy_check.txt exactly."""
    _run("pdb_proxy_check.py", tmp_path)
    _same((tmp_path / "results_pdb_proxy_check.txt").read_text(), "results_pdb_proxy_check.txt")


@claim("z65", "main §5.3", "main Table 10", kind="reproduce", needs=("pdb", "slow"))
def test_reproduce_pdb_band_controls(tmp_path):
    """pdb_band_controls.py on the snapshot and the deposited pdb_lepage.jsonl prints
    results_pdb_band_controls.txt exactly (the cut controls, Table 10, the shares and the
    misranking of 3000 pairs)."""
    _same(_run("pdb_band_controls.py", tmp_path), "results_pdb_band_controls.txt")


@claim("z65", "SI Table S15", "main §5.3", kind="reproduce", needs=("pdb", "slow"))
def test_reproduce_pdb_band_uncertainty(tmp_path):
    """pdb_band_uncertainty.py (1000 block-bootstrap resamples) writes
    results_pdb_band_uncertainty.txt exactly."""
    _run("pdb_band_uncertainty.py", tmp_path)
    _same((tmp_path / "results_pdb_band_uncertainty.txt").read_text(), "results_pdb_band_uncertainty.txt")


@claim("z65", "SI Table S8", "main §5.3", kind="reproduce", needs=("cctbx",))
def test_reproduce_pdb_lepage_tables(tmp_path):
    """pdb_lepage.py --table on the deposited pdb_lepage.jsonl prints the tables of
    results_pdb_lepage.txt exactly (the file's leading progress counters, written by the full
    run, are not part of the table)."""
    out = _run("pdb_lepage.py", tmp_path, "--table", out=False)
    dep = _read("results_pdb_lepage.txt").split("\n")
    while dep and re.fullmatch(r"\d+", dep[0]):
        dep = dep[1:]
    a, b = out.rstrip("\n").split("\n"), "\n".join(dep).rstrip("\n").split("\n")
    assert a == b, [(x, y) for x, y in zip(a, b) if x != y][:5]


@claim("z65", "main §5.3", "SI Table S8", kind="reproduce", needs=("pdb", "cctbx", "slow"))
def test_reproduce_pdb_lepage_full_run(tmp_path):
    """pdb_lepage.py on every cell of the snapshot regenerates pdb_lepage.jsonl line for line
    and prints results_pdb_lepage.txt exactly (progress counters included)."""
    out = _run("pdb_lepage.py", tmp_path, timeout=12 * 3600)
    new = (tmp_path / "pdb_lepage.jsonl").read_text().splitlines()
    dep = _read("pdb_lepage.jsonl").splitlines()
    assert len(new) == len(dep)
    diff = [i for i, (x, y) in enumerate(zip(new, dep)) if json.loads(x) != json.loads(y)]
    assert not diff, f"{len(diff)} entries differ, first {[dep[i][:120] for i in diff[:3]]}"
    _same(out, "results_pdb_lepage.txt")


@claim("z65", "main §5.3, Against chance", "SI §S9", kind="reproduce", needs=("pdb", "cctbx", "slow"))
def test_reproduce_pdb_lepage_null(tmp_path):
    """pdb_lepage_null.py (two seeds, 1500 cells per arm) prints results_pdb_lepage_null.txt
    exactly."""
    _same(_run("pdb_lepage_null.py", tmp_path, timeout=12 * 3600), "results_pdb_lepage_null.txt")
