"""z64 — the benchmark numbers of the paper, against the deposited results files.

Two layers.  The ``benchmark`` claims read the deposited results files in
``manuscript/supplement/benchmark`` and assert every number the paper prints from them, at
the paper's rounding; derived numbers (sums, ratios) are computed from the files.  The
``reproduce`` claims rerun each script into a temporary directory and assert that the
regenerated results equal the deposited ones, ignoring only Python warnings (which some
deposited files captured from stderr) and the lines or fields that report wall-clock time,
as each docstring says.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from fractions import Fraction

from sellingproofs import paths
from sellingproofs.registry import claim

B = paths.BENCH


# ---------------------------------------------------------------------------- helpers

def _text(name: str) -> str:
    p = B / name
    assert p.is_file(), f"deposited results file missing: {p}"
    return p.read_text()


def _rows(name: str, pattern: str):
    """all regex matches (groups) over the lines of a deposited file."""
    out = [m.groups() for m in (re.match(pattern, line) for line in _text(name).splitlines()) if m]
    assert out, f"no line of {name} matches {pattern!r}"
    return out


def _frac(s: str):
    a, b = s.split("/")
    return int(a), int(b)


def _found(name: str):
    """{(family, method): (found, n)} from a compare_methods results file."""
    out = {}
    for fam, meth, k, n in _rows(name, r"^(\S+)\s+(closure|small1|small2|niggli1|cctbx_sim1|cctbx_sim2|pipeline)\s+(\d+)/(\d+)\s"):
        out[fam, meth] = (int(k), int(n))
    return out


def _cheapest(name: str):
    """{(family, method): (true cheapest, found)} from a compare_methods results file."""
    out = {}
    for fam, meth, a, b in _rows(name, r"^(\S+)\s+(\S+)\s+\d+/\d+\s+\d+/\d+\s+(\d+)/(\d+)\s*$"):
        out[fam, meth] = (int(a), int(b))
    return out


def _window():
    """{(family, Niggli row): {method: count, 'n': n}} from results_window_failures.txt."""
    lines = _text("results_window_failures.txt").splitlines()
    head = lines[1].split()
    assert head[:3] == ["family", "Niggli", "n"], head
    out = {}
    for line in lines[2:]:
        f = line.split()
        if len(f) == len(head):
            out[f[0], f[1]] = dict(zip(head[2:], map(int, f[2:])))
    return out


# ---------------------------------------------------------------------------- the benchmark

@claim("z64", "main §4.4", "SI §S2", kind="benchmark")
def test_benchmark_composition():
    """The benchmark has 780 primitive cells with a random metric error (median strain 0.1%)
    and a random proper reindexing: 75 measured in another shape of the same lattice, 660 near
    a specialisation, 20 whose only reduced superbase is a difference-wall member, 16 needles
    and plates and 9 of the boundary sorts.

    Counts from benchmark.jsonl and results_strain.txt; the median strain
    ||G^-1/2 E G^-1/2||_F of the error E (in the reference's frame, from the true operator)
    is computed from benchmark.jsonl and agrees with the per-family medians of
    results_strain.txt."""
    import numpy as np

    from agentsg.cell.metric import UnitCell
    rows = [json.loads(l) for l in _text("benchmark.jsonl").splitlines()]
    fam = {}
    for r in rows:
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    assert len(rows) == 780
    assert fam == {"shapes": 75, "specialisation": 660, "hidden": 20, "anisotropy": 16, "boundary": 9}
    n = {f: int(k) for f, k in _rows("results_strain.txt", r"^(\w+)\s+(\d+) \|")}
    assert n == {**fam, "all": 780}
    strain, by = [], {}
    for r in rows:
        G = np.array(UnitCell(*r["reference_cell"]).metric_tensor())
        Gm = np.array(UnitCell(*r["measured_cell"]).metric_tensor())
        Mi = np.linalg.inv(np.array(r["true_operator"], float))
        w, V = np.linalg.eigh(G)
        Gi = V @ np.diag(w ** -0.5) @ V.T
        s = float(np.linalg.norm(Gi @ (Mi.T @ Gm @ Mi - G) @ Gi))
        strain.append(s)
        by.setdefault(r["family"], []).append(s)
    assert round(100 * float(np.median(strain)), 1) == 0.1
    for f, med in _rows("results_strain.txt", r"^(\w+)\s+\d+ \|.*\s(\d\.\d\de-\d\d)\s*$"):
        assert abs(float(np.median(by[f])) - float(med)) <= 0.006 * float(med), (f, med)


@claim("z64", "SI §S2", kind="benchmark")
def test_measured_niggli_cell_against_the_reference():
    """The measured Niggli cell is a different cell from the reference's in 45 cases (O1B 23,
    O3 12, M1B 6, one each at M2B, M3, R3 and T2; 14 of them a change between type I and
    type II), and the same cell in another basis in 125 (O1B 25, M1A 15, O1A 15, T1 15, M1B
    13, O4 9, O2 7, R1 7, M2A 6, T2 6, O3 5, A2 1, A3 1).

    From results_niggli.txt (evaluate.py --niggli): the family rows and the sort rows each
    sum to the totals."""
    fam, sort = {}, {}
    for key, n, same, other, flip in _rows("results_niggli.txt", r"^(\w+|sort \w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"):
        (sort if key.startswith("sort ") else fam)[key.removeprefix("sort ")] = (int(same), int(other), int(flip))
    assert set(fam) == {"shapes", "specialisation", "hidden", "anisotropy", "boundary"}
    for d in (fam, sort):
        assert sum(v[1] for v in d.values()) == 45
        assert sum(v[2] for v in d.values()) == 14
        assert sum(v[0] for v in d.values()) == 125
    other = {k: v[1] for k, v in sort.items() if v[1]}
    assert other == {"O1B": 23, "O3": 12, "M1B": 6, "M2B": 1, "M3": 1, "R3": 1, "T2": 1}
    same = {k: v[0] for k, v in sort.items() if v[0]}
    assert same == {"O1B": 25, "M1A": 15, "O1A": 15, "T1": 15, "M1B": 13, "O4": 9, "O2": 7, "R1": 7,
                    "M2A": 6, "T2": 6, "O3": 5, "A2": 1, "A3": 1}


@claim("z64", "SI §S2", "main §5.2", kind="benchmark")
def test_strain_procedure_at_the_band_and_at_three_errors():
    """evaluate_strain.py scores the closure and the {-1,0,1} search at the band and at three
    times each case's error: the closure has the true operator in 780 of 780 cases at the
    band and the {-1,0,1} search in 778; at three times the error, 778 of 778 and 776 of 778.

    From the 'all' row of results_strain.txt; each family row sums to it."""
    cols = {}
    for f, n, a, b, c, d in _rows("results_strain.txt", r"^(\w+)\s+(\d+) \|\s+(\d+)/\d+\s+(\d+)/\d+\s+\|\s+(\d+)/\d+\s+(\d+)/\d+"):
        cols[f] = tuple(map(int, (a, b, c, d)))
    assert cols.pop("all") == (780, 778, 778, 776)
    assert tuple(map(sum, zip(*cols.values()))) == (780, 778, 778, 776)


@claim("z64", "SI §S4", kind="benchmark")
def test_uniform_band_declines_half_the_needles_and_plates():
    """On the 16 needles and plates of the benchmark the construction with a uniform conorm
    band declined 8.

    From the 'anisotropy closure' row of results.txt (evaluate.py): 16 cases, 8 flagged."""
    (n, ok, flagged), = [tuple(map(int, g)) for g in
                         _rows("results.txt", r"^anisotropy\s+closure\s+(\d+)\s+(\d+)/\d+\s+\S+\s+(\d+)\s*$")]
    assert (n, flagged) == (16, 8) and ok == n - flagged


# ---------------------------------------------------------------------------- the stress and window families

@claim("z64", "main §4.4", "SI §S2", kind="benchmark")
def test_stress_family_design():
    """The stress family puts independent errors on both cells of 60 references, scaled so
    that the true operator's price is 0.5, 0.9, 1.1 and 1.5 times the band; at 1.1 and 1.5 no
    method found the true operator, as by construction none should.

    From results_stress.txt: 60 cases at each t, and 0 of 60 found by every method at
    t = 1.1 and 1.5."""
    found = _found("results_stress.txt")
    for t in ("0.5", "0.9", "1.1", "1.5"):
        assert {n for (f, m), (k, n) in found.items() if f == f"t={t}"} == {60}
    for t in ("1.1", "1.5"):
        assert {k for (f, m), (k, n) in found.items() if f == f"t={t}"} == {0}


@claim("z64", "main §4.4", "SI §S2", kind="benchmark")
def test_window_family_design():
    """A further family has 300 lattices with two right angles sharing a vector or three
    right angles, each measured twice with an error of 0.1 to 2%, the reference given as its
    Niggli cell (171 of Niggli type I, 129 of type II) or in a random basis (150 with two
    zeros, 150 with three), and 60 lattices measured in a member that needs an entry of 3.

    From the n column of results_window_failures.txt."""
    w = _window()
    assert w["typeI", "all"]["n"] == 300 and w["random", "all"]["n"] == 300 and w["entry3", "all"]["n"] == 60
    assert (w["typeI", "I"]["n"], w["typeI", "II"]["n"]) == (171, 129)
    assert w["random", "I/2z"]["n"] + w["random", "II/2z"]["n"] == 150
    assert w["random", "I/3z"]["n"] + w["random", "II/3z"]["n"] == 150


# ---------------------------------------------------------------------------- the guarantee and the searches

def _table_s13():
    """Table S13 as regenerated from the deposited results files, columns: benchmark,
    stress 0.5, stress 0.9, entry 3, Niggli type I, type II, random 2 zeros, 3 zeros."""
    bench, stress, w = _found("results_compare.txt"), _found("results_stress.txt"), _window()
    look = {f: (int(k), int(n)) for f, k, n in _rows("results_lookup.txt", r"^\s+(\S+)\s+found (\d+)/(\d+)\s")}
    # results_lookup.txt holds the benchmark block first, then the stress block; 'all' is the benchmark's
    lk_all = _rows("results_lookup.txt", r"^\s+all\s+found (\d+)/(\d+)\s")[0]

    def col(method, wkey):
        def wf(fam, rows):
            return sum(w[fam, r][wkey] for r in rows), sum(w[fam, r]["n"] for r in rows)
        return [wf("entry3", ["all"]), wf("typeI", ["I"]), wf("typeI", ["II"]),
                wf("random", ["I/2z", "II/2z"]), wf("random", ["I/3z", "II/3z"])]
    table = {}
    for name, m in (("closure", "closure"), ("{-1,0,1} Selling", "small1"), ("{-2..2} Selling", "small2"),
                    ("{-1,0,1} Niggli", "niggli1"), ("cctbx 1", "cctbx_sim1"), ("cctbx 2", "cctbx_sim2"),
                    ("pipeline", "pipeline")):
        table[name] = [bench["all", m], stress["t=0.5", m], stress["t=0.9", m]] + col(name, m)
    table["lookup"] = [tuple(map(int, lk_all)), look["t=0.5"], look["t=0.9"]] + col("lookup", "lookup")
    return table


TABLE_S13 = {   # as printed in SI Table S13
    "closure": "780/780 60/60 60/60 60/60 171/171 129/129 150/150 150/150",
    "lookup": "780/780 60/60 60/60 60/60 171/171 129/129 150/150 150/150",
    "{-1,0,1} Selling": "778/780 58/60 52/60 24/60 142/171 129/129 120/150 99/150",
    "{-2..2} Selling": "780/780 59/60 60/60 52/60 170/171 129/129 150/150 149/150",
    "{-1,0,1} Niggli": "780/780 60/60 60/60 60/60 171/171 129/129 150/150 150/150",
    "cctbx 1": "780/780 60/60 60/60 60/60 171/171 129/129 150/150 150/150",
    "cctbx 2": "780/780 60/60 60/60 60/60 171/171 129/129 150/150 150/150",
    "pipeline": "779/780 56/60 54/60 60/60 171/171 126/129 147/150 148/150",
}


@claim("z64", "SI Table S13", kind="benchmark")
def test_existing_searches_table_s13():
    """SI Table S13: the true operator among the candidates, by method and family, for the
    closure, the lookup of U, the {-1,0,1} and {-2,...,2} windows between Selling-reduced
    cells, the {-1,0,1} window between Niggli cells, the cctbx similarity search with ranges 1
    and 2, and the lattice-symmetry pipeline.

    Every cell of the table, from results_compare.txt, results_stress.txt, results_lookup.txt
    and results_window_failures.txt (the window columns are sums of the Niggli-type and
    face-kind rows)."""
    got = {k: " ".join(f"{a}/{b}" for a, b in v) for k, v in _table_s13().items()}
    assert got == TABLE_S13


@claim("z64", "main §5.2", "main abstract", "main §6", kind="benchmark")
def test_closure_found_the_true_operator_in_every_case():
    """The closure found the true operator in every case of every family, as Lemma A7
    guarantees: 780 of 780 on the benchmark, 120 of 120 in the stress cases, 600 of 600 near
    two- and three-zero faces and 60 of 60 in members that need an entry of 3."""
    bench, stress, w = _found("results_compare.txt"), _found("results_stress.txt"), _window()
    assert bench["all", "closure"] == (780, 780)
    assert tuple(map(sum, zip(stress["t=0.5", "closure"], stress["t=0.9", "closure"]))) == (120, 120)
    assert (w["typeI", "all"]["closure"] + w["random", "all"]["closure"],
            w["typeI", "all"]["n"] + w["random", "all"]["n"]) == (600, 600)
    assert (w["entry3", "all"]["closure"], w["entry3", "all"]["n"]) == (60, 60)
    s = _rows("results_strain.txt", r"^all\s+(\d+) \|\s+(\d+)/(\d+)")[0]
    assert s == ("780", "780", "780")


@claim("z64", "main §5.2", kind="benchmark")
def test_lookup_found_the_true_operator_in_every_case():
    """The lookup of U found it in every case too, including 12 benchmark cases on three
    needle and plate references whose r_1 lies below the 7.7% that would certify it.

    From results_lookup.txt (benchmark and stress blocks; no miss with a certified
    reference) and the lookup column of results_window_failures.txt."""
    lines = _text("results_lookup.txt").splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith("stress:"))
    bench = [re.match(r"^\s+(\S+)\s+found (\d+)/(\d+)\s+references with r_1 <= 7\.7 %: (\d+)\s+missed.*: (\d+)$", l)
             for l in lines[:k]]
    bench = {m.group(1): tuple(map(int, m.groups()[1:])) for m in bench if m}
    assert bench["all"] == (780, 780, 12, 0)
    assert bench["anisotropy"][2] == 12
    stress = {m.group(1): tuple(map(int, m.groups()[1:])) for m in
              (re.match(r"^\s+(\S+)\s+found (\d+)/(\d+)\s+references with r_1 <= 7\.7 %: (\d+)\s+missed.*: (\d+)$", l)
               for l in lines[k:]) if m}
    assert stress["t=0.5"][:2] == (60, 60) and stress["t=0.9"][:2] == (60, 60)
    assert all(v[3] == 0 for v in stress.values())
    w = _window()
    assert all(v["lookup"] == v["n"] for v in w.values())


@claim("z64", "main §5.2", kind="benchmark", needs=("slow",))
def test_the_uncertified_lookup_cases_lie_on_three_references():
    """The 12 benchmark cases whose reference has r_1 below 7.7% lie on three needle and
    plate references.

    The adjacency radius of every anisotropy reference of benchmark.jsonl, recomputed."""
    from agentsg.cell.strain_closure import adjacency_radius, reduce_cell
    rows = [json.loads(l) for l in _text("benchmark.jsonl").splitlines() if '"anisotropy"' in l]
    low = {}
    for r in rows:
        key = tuple(round(x, 6) for x in r["reference_cell"])
        if key not in low:
            low[key] = adjacency_radius(reduce_cell(r["reference_cell"]).G)
    cases = [r for r in rows if low[tuple(round(x, 6) for x in r["reference_cell"])] <= 0.077]
    assert len(cases) == 12
    assert len({tuple(round(x, 6) for x in r["reference_cell"]) for r in cases}) == 3


@claim("z64", "main §5.2", kind="benchmark")
def test_lookup_misses_beyond_the_radius_on_a_plate():
    """Against the 100 x 100.1 x 5 A plate of Table 8 (r_1 = 5%), a new cell whose true
    operator costs 6.1% is found by the enumeration and missed by the lookup.

    From results_lookup_plate_case.txt: the true operator's price, its row (marked TRUE)
    among the enumerated candidates, and the lookup's verdict."""
    t = _text("results_lookup_plate_case.txt")
    price = float(re.search(r"^true P .* price (\S+)$", t, re.M).group(1))
    assert round(100 * price, 1) == 6.1 and price <= 0.074
    assert re.search(r"^\d+ \S+ .* TRUE$", t, re.M)
    m = re.search(r"^lookup finds true: (\w+)\s+lookup n: \d+\s+r1 of reference: (\S+)\s+layer (\d)$", t, re.M)
    assert m.group(1) == "False" and round(100 * float(m.group(2))) == 5
    # the radius the script quotes is the certified r_1 of that plate (Table 8)
    cert = {r["cell"]: r for r in json.loads(_text("certified_radii.json"))}
    (plate,) = [r for k, r in cert.items() if k.startswith("plate 100x100.1x5")]
    lo, hi = Fraction(plate["r1"]["lower"]), Fraction(plate["r1"]["upper"])
    assert round(float(lo), 4) == round(float(hi), 4) == float(m.group(2))     # strain, not %


@claim("z64", "main §5.2", "main §6", kind="benchmark")
def test_existing_searches_in_the_main_text():
    """The {-1,0,1} window between the two Niggli cells and the cctbx similarity search found
    the true operator in every case of every family; the lattice-symmetry pipeline missed 1 of
    780 benchmark cases, 10 of 120 stress cases and 8 of 600 near two- and three-zero faces
    (19 of 1560 in all, Discussion). A {-1,0,1} window between the two Selling-reduced cells
    found 24 of the 60 entry-3 cases and 219 of 300 references given in a random basis."""
    bench, stress, w = _found("results_compare.txt"), _found("results_stress.txt"), _window()
    fams = [("typeI", "all"), ("random", "all"), ("entry3", "all")]
    for m in ("niggli1", "cctbx_sim1", "cctbx_sim2"):
        assert bench["all", m] == (780, 780)
        assert stress["t=0.5", m] == (60, 60) and stress["t=0.9", m] == (60, 60)
        assert all(w[f][m] == w[f]["n"] for f in fams)
    miss = [780 - bench["all", "pipeline"][0],
            120 - stress["t=0.5", "pipeline"][0] - stress["t=0.9", "pipeline"][0],
            600 - w["typeI", "all"]["pipeline"] - w["random", "all"]["pipeline"],
            60 - w["entry3", "all"]["pipeline"]]
    assert miss == [1, 10, 8, 0]
    assert (sum(miss), 780 + 120 + 600 + 60) == (19, 1560)
    assert w["entry3", "all"]["small1"] == 24 and w["random", "all"]["small1"] == 219


@claim("z64", "SI §S8", kind="benchmark")
def test_benchmark_misses():
    """The pipeline missed a needle whose reference and measured cell idealise at 3 deg to
    different lattices; the {-1,0,1} window between Selling-reduced cells missed two A2 cases
    near a specialisation, whose operator between the reduced cells has an entry of 2.

    From results_benchmark_misses.txt: one pipeline line, an M4 needle 5 x 5 x 100 A, and two
    small1 lines, both A2 with largest entry 2."""
    rows = _rows("results_benchmark_misses.txt", r"^(\w+): (\w+) (\w+) reference \[([^\]]*)\].*largest entry between reduced cells (\d)$")
    pipe = [r for r in rows if r[0] == "pipeline"]
    small = [r for r in rows if r[0] == "small1"]
    assert len(pipe) == 1 and len(small) == 2
    name, fam, sort, cell, e = pipe[0]
    c = [float(x) for x in cell.split(",")]
    assert fam == "anisotropy" and sort == "M4" and c[:3] == [5.0, 5.0, 100.0]
    assert all(r[1] == "specialisation" and r[2] == "A2" and r[4] == "2" for r in small)


@claim("z64", "main §5.3", kind="benchmark")
def test_what_the_metric_does_not_decide():
    """The cheapest candidate lies in the true operator's coset of the exact lattice
    symmetries in 742 of the 780 benchmark cases, and is the true operator itself in 741. The
    38 that fail are the 20 difference-wall cases and 18 of the 660 cases near a
    specialisation. In the stress cases the cheapest candidate is in the true coset in 68 of
    120, and is the true operator itself in 40.

    From results_coset.txt (coset) and the 'true cheapest' column of results_compare.txt and
    results_stress.txt (the operator itself)."""
    cos = {f: (int(n), int(found), int(c)) for f, n, found, c in
           _rows("results_coset.txt", r"^(\w+ [\w.=]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+\S+\s*$")}
    assert cos["benchmark all"] == (780, 780, 742)
    assert cos["benchmark hidden"][2] == 0 and cos["benchmark hidden"][0] == 20
    assert cos["benchmark specialisation"][0] - cos["benchmark specialisation"][2] == 18
    assert 780 - 742 == 20 + 18
    assert sum(v[0] - v[2] for k, v in cos.items() if k.startswith("benchmark ") and k != "benchmark all") == 38
    assert cos["stress all"][0] == 120 and cos["stress all"][2] == 68
    assert _cheapest("results_compare.txt")["all", "closure"] == (741, 780)
    assert _cheapest("results_stress.txt")["all", "closure"] == (40, 120)


# ---------------------------------------------------------------------------- Niggli

@claim("z64", "main §5.2", "SI §S9", kind="benchmark")
def test_why_the_niggli_window_works():
    """The true operator between the two Niggli cells, for 100 lattices on each of the 38
    faces and on each of the 24 sorts measured twice with errors of total strain 0.5 to 7.7%,
    had its entries in {-1,0,1} in all 24,800 cases, strictly and not only up to the point
    group, and in 3,040 further cases on strongly anisotropic lattices (the square root of the
    metric's eigenvalue ratio up to 98, median 5). Between the two Selling-reduced cells an
    entry of 2 or 3 was needed in 3,765 cases (on two zeros sharing a vector, three zeros, and
    the sorts cP, tP and oP) and in none once composed with the point group.

    From results_niggli_window.txt (the four 'total' lines and the per-row counts) and
    results_niggli_window_thin.txt."""
    t = _text("results_niggli_window.txt")
    blocks = re.split(r"^(Niggli|Niggli mod point group|Selling|Selling mod point group)$", t, flags=re.M)
    tot = {}
    big = {}
    for name, body in zip(blocks[1::2], blocks[2::2]):
        tot[name] = tuple(map(int, re.search(r"total: (\d+) of (\d+) cases", body).groups()))
        big[name] = {m.group(1).strip(): sum(int(x) for x in re.findall(r"(\d+)/\s*\d+ \(\d\)", m.group(2)))
                     for m in re.finditer(r"^  (face \S+(?: \S+)?|sort .*?\))\s+(.*)$", body, re.M)}
    assert tot == {"Niggli": (0, 24800), "Niggli mod point group": (0, 24800),
                   "Selling": (3765, 24800), "Selling mod point group": (0, 24800)}
    assert (38 + 24) * 100 * 4 == 24800
    where = {k for k, v in big["Selling"].items() if v}
    assert where and all(k in ("face 2 sharing", "face 3 triangle", "face 3 path")
                         or re.match(r"sort \w+ \((cP|tP|oP)\)", k) for k in where), where
    assert {re.match(r"sort \w+ \((\w+)\)", k).group(1) for k in where if k.startswith("sort")} == {"cP", "tP", "oP"}
    thin = _text("results_niggli_window_thin.txt")
    m = re.search(r"(\d+) cases; largest entry between Niggli cells: \{1: (\d+)\};.*median (\d+), max (\d+)", thin)
    assert m.groups() == ("3040", "3040", "5", "98")
    assert 38 * 40 * 2 == 3040


@claim("z64", "main §5.1", kind="benchmark")
def test_niggli_type_and_presentation_near_op():
    """Near a lattice with three right angles, every reduction of a type I measurement ended
    with a zero on d: 402 of 402 reductions of measured near-oP cells from random bases, and 67
    of 67 from the Niggli cell. A type II Niggli cell is itself obtuse with its small conorms
    off d, so reduction started from it stays there (83 of 83), but reduction started from
    another basis put a near-zero conorm on d in 370 of 498 cases.

    From results_niggli_presentation.txt."""
    got = {(t, p): (int(k), int(n)) for t, p, k, n in
           _rows("results_niggli_presentation.txt", r"^\s+type (I|II)\s+(Niggli form|random basis)\s+(\d+) / (\d+)$")}
    assert got == {("I", "Niggli form"): (67, 67), ("I", "random basis"): (402, 402),
                   ("II", "Niggli form"): (0, 83), ("II", "random basis"): (370, 498)}


@claim("z64", "main §5.1", kind="benchmark")
def test_niggli_type_near_mp():
    """Near a lattice with two right angles a type II Niggli cell kept its right angles on ab,
    ac, bc in 519 of 519 measured near-mP cells; from a type I Niggli cell reduction put at
    least one near-zero conorm on d in 614 of 681, and both near-zero conorms in 316; in that
    sample 57% of near-mP measurements were type I.

    From results_niggli_types.txt: the counts by (type, on d), the flip words with the slots
    of the two smallest conorms, and the same counts with agentsg's reduce_cell."""
    t = _text("results_niggli_types.txt")
    first, rest = t.split("flip words", 1)
    lib = rest.split("the same with agentsg reduce_cell", 1)[1]
    for block in (first, lib):
        c = {(a, b == "True"): int(n) for a, b, n in re.findall(r"\('(I+)', (True|False)\) (\d+)", block)}
        assert c == {("I", False): 67, ("I", True): 614, ("II", False): 519}
    words = [(a, w, s1, s2, int(n)) for a, w, s1, s2, n in
             re.findall(r"\('(I+)', '([\w.]+)', \('(\w\w)', '(\w\w)'\)\) (\d+)", rest.split("the same with")[0])]
    assert sum(n for a, *_, n in words if a == "II") == 519
    assert all(not (s1.startswith("d") or s2.startswith("d")) for a, w, s1, s2, n in words if a == "II")
    both = sum(n for a, w, s1, s2, n in words if a == "I" and s1.startswith("d") and s2.startswith("d"))
    some = sum(n for a, w, s1, s2, n in words if a == "I" and (s1.startswith("d") or s2.startswith("d")))
    assert (some, both, sum(n for a, *_, n in words if a == "I")) == (614, 316, 681)
    assert round(100 * 681 / (681 + 519)) == 57


@claim("z64", "main Lemma A2", "main §3.3", kind="benchmark")
def test_best_labelling_of_the_reduced_cell():
    """Relabelling F0 once for all members, the largest entry is 1 on every face but the path
    of three zeros, where it is 2.

    From results_face_labellings.txt: all 38 faces; a three-zero face is a path when its zero
    slots touch all four vectors (a triangle touches three)."""
    rows = _rows("results_face_labellings.txt",
                 r"^(\d) zeros\s+(\S+)\s+members\s+(\d+)\s+largest entry: as reduced (\d), best labelling (\d)$")
    assert len(rows) == 38
    for k, zs, members, as_red, best in rows:
        verts = set("".join(zs.split(","))) if zs != "-" else set()
        path = k == "3" and len(verts) == 4
        assert int(best) == (2 if path else 1), (zs, best)
    assert sum(1 for r in rows if r[0] == "3" and len(set("".join(r[1].split(",")))) == 4) == 12


# ---------------------------------------------------------------------------- radii and timings

TABLE_8 = {   # certified_radii.json cell name prefix -> (r1, r2) as printed in Table 8
    "worked cell": (">25", ">25"),
    "needle 10x10x30": (">25", ">25"),
    "needle 10x10x100": ("13.9", ">25"),
    "needle 5.58x5.21x132": ("5.6", "11.1"),
    "plate 100x100.1x5": ("5.0", "14.1"),
    "plate 150x120x4.8": ("4.3", "10.0"),
}


@claim("z64", "main Table 8", "main §5.2", "SI §S7", kind="benchmark")
def test_table_8_certified_radii():
    """Table 8: adjacency radii in strain (%), r_1 and r_2, searched up to 25%, each
    certified in exact rational arithmetic to the digits shown: worked cell >25, >25; needle
    10 x 10 x 30 A >25, >25; needle 10 x 10 x 100 A 13.9, >25; needle 5.58 x 5.21 x 131.97 A
    5.6, 11.1; plate 100 x 100.1 x 5 A 5.0, 14.1; plate 150 x 120 x 4.8 A 4.3, 10.0.

    From certified_radii.json: both proved bounds (exact fractions) round to the printed
    value, and '>25' is recorded where no superbase outside the layer lies within the cap;
    results_certified_radii.txt states the same."""
    cert = json.loads(_text("certified_radii.json"))
    txt = _text("results_certified_radii.txt")
    assert len(cert) == len(TABLE_8)
    for r in cert:
        (want,) = [v for k, v in TABLE_8.items() if r["cell"].startswith(k)]
        for layer, printed in zip(("r1", "r2"), want):
            rr = r[layer]
            if printed == ">25":
                assert rr["text"] == ">25", (r["cell"], layer, rr)
                continue
            lo, hi = Fraction(rr["lower"]) * 100, Fraction(rr["upper"]) * 100
            assert lo <= hi and hi - lo <= Fraction(1, 1000)
            for x in (lo, hi):
                assert f"{float(x):.1f}" == printed, (r["cell"], layer, float(x), printed)
            assert f"{layer} {rr['text']}" in txt


@claim("z64", "main §5.2", "main §6", kind="benchmark")
def test_adjacency_radii_on_the_archive():
    """r_1 reaches the cap for 99.4% of the 697 benchmark references and 98.6% of 2000 PDB
    cells drawn at random; every one of those 2000 has r_1 above the 7.7% that certifies the
    default band. 3HR2 and 3HQV (40 x 27 x 678 A, r_1 = 6.0%) and 6BZP (13.8 x 4.9 x 101.9 A,
    6.8%) lie below 7.7%, and a placeholder cell of 1 x 1 x 70.8 A (1CGM) is refused by the
    cap on the ball; the implementation refuses balls above 5000 vectors with a message.

    From results_adjacency_radii.txt, and the library's cap MAX_BALL."""
    t = _text("results_adjacency_radii.txt")
    m = re.search(r"^benchmark references: (\d+) cells \(0 refused\).*at the 25 % cap: (\S+) %; above 7\.7 %: (\S+) %", t, re.M)
    assert m.group(1) == "697" and m.group(2) == "99.4"
    m = re.search(r"^PDB sample: (\d+) cells \(0 refused\).*at the 25 % cap: (\S+) %; above 7\.7 %: (\S+) %", t, re.M)
    assert m.groups() == ("2000", "98.6", "100.0")
    low = {pid: (cell, float(r)) for pid, cell, r in re.findall(r"^(\w{4}) \(([^)]*)\) .*: r_1 (\S+) %$", t, re.M)}
    assert set(low) == {"3HR2", "3HQV", "6BZP"}
    assert round(low["3HR2"][1], 1) == round(low["3HQV"][1], 1) == 6.0 and round(low["6BZP"][1], 1) == 6.8
    c = [float(x) for x in low["3HR2"][0].split(",")]
    assert [round(x) for x in c[:3]] == [40, 27, 678]
    c = [float(x) for x in low["6BZP"][0].split(",")]
    assert [round(x, 1) for x in c[:3]] == [13.8, 4.9, 101.9]
    m = re.search(r"^1CGM \(([^)]*)\) .*: refused \(the ball of Lemma A6 holds (\d+) lattice vectors .*\(limit (\d+)\)", t, re.M)
    assert [float(x) for x in m.group(1).split(",")][:3] == [1.0, 1.0, 70.8]
    assert int(m.group(2)) > 5000 and m.group(3) == "5000"
    from agentsg.cell.strain_closure import MAX_BALL
    assert MAX_BALL == 5000


@claim("z64", "main §5.2", "SI §S7", kind="benchmark")
def test_timings():
    """On a tenth of the benchmark, widening a reference took a median of 0.02 s and
    reindexing one new cell 0.03 s (90th percentiles 0.03 and 0.12 s); a 150 x 120 x 4.8 A
    plate took 0.9 s to widen. A 300 x 300 x 3 A plate takes about a minute to widen and its
    r_1 several minutes, and thinner cells are refused by the cap on the ball. A vectorised
    lookup of U for one new cell, reductions included, takes about 0.04 s.

    From results_adjacency_radii.txt, as measured when it was deposited ('about a minute':
    between 30 and 90 s; 'several minutes': between 3 and 15 min)."""
    t = _text("results_adjacency_radii.txt")
    m = re.search(r"every 10th case \((\d+)\): widening the reference median (\S+) s \(90th (\S+)\); "
                  r"reindexing one new cell median (\S+) s \(90th (\S+)\)", t)
    assert m.groups()[1:] == ("0.02", "0.03", "0.03", "0.12") and int(m.group(1)) == 78
    m = re.search(r"thin cell \(150\.0, 120\.0, 4\.8.*?\): widening (\S+) s", t)
    assert round(float(m.group(1)), 1) == 0.9
    m = re.search(r"thin cell \(300\.0, 300\.0, 3\.0.*?\): widening (\S+) s .* in (\S+) s", t)
    assert 30 <= float(m.group(1)) <= 90 and 180 <= float(m.group(2)) <= 900
    assert re.search(r"thin cell \(5\.0, 5\.0, 200\.0.*\): .*r_1 refused \(ball above the cap", t)
    m = re.search(r"vectorised lookup of U .*: (\S+) s on average", t)
    assert round(float(m.group(1)), 2) == 0.04


# ---------------------------------------------------------------------------- Le Page by sort, SELLA

TABLE_S11 = {   # sort: (type, n, median distance, 10th, 90th, twofold price, other orders) as printed
    "C1": ("cI", 25, "8.3", "7.6", "9.0", "14.8", "3: 14.1, 4: 15.0"),
    "C3": ("cF", 25, "8.3", "7.7", "9.2", "14.8", "3: 14.3, 4: 14.5"),
    "C5": ("cP", 25, "8.5", "7.6", "9.3", "14.8", "3: 14.2, 4: 14.7"),
    "H4": ("hP", 25, "7.9", "7.5", "8.5", "14.8", "3: 13.7, 6: 11.3"),
    "T1": ("tI", 25, "7.9", "7.5", "9.1", "14.8", "4: 14.0"),
    "T2": ("tI", 25, "8.0", "7.6", "8.8", "14.8", "4: 14.4"),
    "T5": ("tP", 25, "8.0", "7.4", "8.9", "14.8", "4: 14.1"),
    "R1": ("hR", 25, "8.0", "7.5", "8.6", "14.8", "3: 14.4"),
    "R3": ("hR", 25, "8.1", "7.5", "8.8", "14.8", "3: 14.5"),
    "O1A": ("oF", 24, "7.5", "7.4", "7.8", "14.8", ""),
    "O1B": ("oI", 25, "7.6", "7.4", "8.1", "14.8", ""),
    "O2": ("oI", 25, "7.6", "7.4", "8.2", "14.8", ""),
    "O3": ("oI", 25, "8.6", "7.6", "13.6", "14.8", ""),
    "O4": ("oS", 25, "7.7", "7.4", "8.2", "14.8", ""),
    "O5": ("oP", 25, "7.5", "7.4", "8.3", "14.8", ""),
    "M1A": ("mC", 20, "7.4", "7.4", "9.1", "14.8", ""),
    "M1B": ("mC", 15, "7.4", "7.4", "7.4", "14.8", ""),
    "M2A": ("mC", 18, "7.4", "7.4", "8.5", "14.8", ""),
    "M2B": ("mC", 17, "9.3", "7.5", "19.9", "14.8", ""),
    "M3": ("mC", 23, "8.9", "7.5", "16.1", "14.8", ""),
    "M4": ("mP", 23, "7.4", "7.4", "7.4", "14.8", ""),
}


@claim("z64", "SI Table S11", "SI §S6", "main Proposition A9", kind="benchmark")
def test_lepage_three_degrees_by_sort():
    """SI Table S11 and its text: at Le Page's 3 deg the median largest twofold price is 14.8%
    in every sort (larger only in the monoclinic tails, up to 40-55%), while the median
    distance to the sort is 7.4-7.7% for the orthorhombic and monoclinic sorts other than O3,
    M2B and M3, 7.9-8.1% for tP, tI, hP and hR, and 8.3-8.5% for the cubic sorts; the
    fourfolds of a body-centred cubic lattice cost a median 15.0%, more than its twofolds.

    Every row of Table S11 from results_lepage_calibration.txt."""
    rows = _rows("results_lepage_calibration.txt",
                 r"^(\w+)\s+(\w\w)\s+(\d+)\s+\d+\s+(\S+)/(\S+)/(\S+)\s+(\S+)/(\S+)/(\S+)\s+2: (\S+?)(?:, (.*))?$")
    got = {}
    for s, typ, n, d10, d50, d90, p10, p50, p90, two, other in rows:
        got[s] = (typ, int(n), d50, d10, d90, two.rstrip(","), other or "")
        if typ not in ("mC", "mP"):
            assert float(p90) <= 20.0                     # the long tails are monoclinic
    assert got == TABLE_S11
    mono_tail = max(float(r[8]) for r in rows if r[1] == "mC")
    assert 39.5 <= mono_tail <= 55
    ranges = {"om": [float(v[2]) for k, v in TABLE_S11.items() if v[0][0] in "om" and k not in ("O3", "M2B", "M3")],
              "tthr": [float(v[2]) for v in TABLE_S11.values() if v[0] in ("tP", "tI", "hP", "hR")],
              "c": [float(v[2]) for v in TABLE_S11.values() if v[0][0] == "c"]}
    assert (min(ranges["om"]), max(ranges["om"])) == (7.4, 7.7)
    assert (min(ranges["tthr"]), max(ranges["tthr"])) == (7.9, 8.1)
    assert (min(ranges["c"]), max(ranges["c"])) == (8.3, 8.5)
    assert "4: 15.0" in TABLE_S11["C1"][6] and 15.0 > float(TABLE_S11["C1"][5])


@claim("z64", "SI §S6", kind="benchmark")
def test_against_sella():
    """Against SELLA: Spearman's rank correlation 0.77 to 0.97 over all 24 sorts; SELLA's
    distance divided by the mean squared edge is a median 0.71 times the strain distance, with
    a 10th to 90th percentile range of 0.05 to 26; the Bravais types within the band agree in
    11 of the 14 cells and differ on the needle and the two plates, SELLA admitting hP, hR,
    oF, tI and tP for the needle and the strain band oF (and tI) for the plates; for these
    thin cells r_1 is 4 to 6%.

    From results_sella.txt (compare_sella.py)."""
    t = _text("results_sella.txt")
    m = re.search(r"median (\S+) \(10-90 % (\S+)-(\S+)\)", t)
    assert (round(float(m.group(1)), 2), round(float(m.group(2)), 2), round(float(m.group(3)))) == (0.71, 0.05, 26)
    cells = re.findall(r"^(\S.{25})\s+(\d\.\d\d)\s+(\d\.\d{3})\s+.*\|", t, re.M)
    assert len(cells) == 14
    rho = [float(c[1]) for c in cells]
    assert (min(rho), max(rho)) == (0.77, 0.97)
    assert re.search(r"same types within band in 11 of 14 cells", t)
    differ = re.findall(r"^\s+types within band: strain (\[.*?\])\s+SELLA (\[.*?\])\s+differ: (\[.*?\])$", t, re.M)
    assert len(differ) == 3
    names = [c[0].strip() for c in cells]
    blocks = t.split("\n")
    thin = {}
    for i, l in enumerate(blocks):
        for nm in names:
            if l.startswith(nm) and "differ" in blocks[i + 1]:
                thin[nm] = (float(re.search(r"\s(\d\.\d{3})\s", l).group(1)), blocks[i + 1])
    assert set(thin) == {"needle 5.58 x 5.21 x 132", "plate 100 x 100.1 x 5", "plate 150 x 120 x 4.8"}
    assert "differ: ['hP', 'hR', 'oF', 'tI', 'tP']" in thin["needle 5.58 x 5.21 x 132"][1]
    assert "differ: ['oF', 'tI']" in thin["plate 100 x 100.1 x 5"][1]
    assert "differ: ['oF']" in thin["plate 150 x 120 x 4.8"][1]
    for nm in ("plate 100 x 100.1 x 5", "plate 150 x 120 x 4.8"):      # the strain band admits them
        s, e = re.search(r"strain (\[.*?\])\s+SELLA (\[.*?\])", thin[nm][1]).groups()
        assert "'oF'" in s and "'oF'" not in e
    assert all(0.04 <= v[0] <= 0.06 for v in thin.values())


# ---------------------------------------------------------------------------- reproduction

def _normalise(text: str, drop=()) -> list[str]:
    """lines without Python warnings (a 'Warning:' line and the source line echoed after it)
    and with the patterns in ``drop`` replaced by '<t>'."""
    out, skip = [], False
    for line in text.splitlines():
        if skip:
            skip = False
            continue
        if re.search(r"\w+Warning: ", line):
            skip = True
            continue
        for pat in drop:
            line = re.sub(pat, "<t>", line)
        out.append(line.rstrip())
    while out and not out[-1]:
        out.pop()
    return out


def _run(script: str, *args, timeout=7200):
    """run a benchmark script with its output directory a fresh temporary directory; returns
    (stdout, directory)."""
    import tempfile
    d = tempfile.mkdtemp(prefix="z64-")
    env = dict(os.environ)
    env["SELLINGPROOFS_OUT"] = d
    env["PYTHONPATH"] = os.pathsep.join([str(paths.REPO / "agentsg" / "src"), str(paths.PACKAGE.parent),
                                         env.get("PYTHONPATH", "")])
    r = subprocess.run([sys.executable, str(B / script), *args], cwd=paths.REPO, env=env,
                       capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout, d


def _same(regenerated: str, deposited: str, drop=()):
    a, b = _normalise(regenerated, drop), _normalise(_text(deposited), drop)
    assert a == b, "\n".join(f"- {x}\n+ {y}" for x, y in zip(b, a) if x != y)[:3000] or (len(a), len(b))


@claim("z64", "main §4.4", "SI §S2", kind="reproduce", needs=("slow",))
def test_reproduce_benchmark_cases():
    """make_benchmark.py regenerates benchmark.jsonl (seed 2026) identically."""
    out, d = _run("make_benchmark.py")
    assert (B / "benchmark.jsonl").read_text() == open(os.path.join(d, "benchmark.jsonl")).read()


@claim("z64", "main §3.3", "main Remark A8", kind="reproduce")
def test_reproduce_universal_operator_table():
    """export_universal_operators.py regenerates universal_operators_columns.json identically
    (6096 column operators, 3048 of each determinant)."""
    out, d = _run("export_universal_operators.py")
    assert out.strip() == "6096 operators; 3048 of determinant +1, 3048 of -1"
    assert json.load(open(os.path.join(d, "universal_operators_columns.json"))) == json.loads(_text("universal_operators_columns.json"))


@claim("z64", "SI §S4", kind="reproduce")
def test_reproduce_evaluate():
    """evaluate.py regenerates the method table of results.txt (the block before the first
    blank line; the Niggli block that follows it is regenerated by evaluate.py --niggli)."""
    out, _ = _run("evaluate.py")
    dep = _text("results.txt").split("\n\n")[0]
    assert _normalise(out) == _normalise(dep)


@claim("z64", "SI §S2", kind="reproduce")
def test_reproduce_evaluate_niggli():
    """evaluate.py --niggli regenerates results_niggli.txt identically."""
    out, _ = _run("evaluate.py", "--niggli")
    _same(out, "results_niggli.txt")


@claim("z64", "SI §S2", kind="reproduce", needs=("slow",))
def test_reproduce_evaluate_strain():
    """evaluate_strain.py regenerates results_strain.txt, apart from the wall-clock time at
    the end of the 'all' row ('(960 s)')."""
    out, _ = _run("evaluate_strain.py")
    _same(out, "results_strain.txt", drop=(r"\(\d+ s\)",))


@claim("z64", "SI Table S13", "main §5.2", kind="reproduce", needs=("cctbx", "slow"))
def test_reproduce_compare_methods():
    """compare_methods.py regenerates results_compare.txt, apart from the line 'seconds per
    case' (wall-clock time)."""
    out, _ = _run("compare_methods.py")
    _same(out, "results_compare.txt", drop=(r"^seconds per case: .*$",))


@claim("z64", "SI Table S13", "main §5.2", kind="reproduce", needs=("cctbx", "slow"))
def test_reproduce_compare_methods_stress():
    """compare_methods.py --stress regenerates results_stress.txt, apart from the line
    'seconds per case' (wall-clock time)."""
    out, _ = _run("compare_methods.py", "--stress")
    _same(out, "results_stress.txt", drop=(r"^seconds per case: .*$",))


@claim("z64", "SI Table S13", "main §5.2", kind="reproduce", needs=("cctbx", "slow"))
def test_reproduce_window_failures():
    """window_failures.py regenerates results_window_failures.txt. The deposited file has a
    last column from LatticeRepLib's CmdLM (LM_BIN); when CmdLM is not configured that column
    is absent from the regenerated file and is left out of the comparison."""
    out, _ = _run("window_failures.py")
    dep = _normalise(_text("results_window_failures.txt"))
    if "CmdLM" not in out.splitlines()[1]:
        dep = dep[:1] + [l.rsplit(None, 1)[0] for l in dep[1:]]
    a = [" ".join(x.split()) for x in _normalise(out)]
    assert a == [" ".join(x.split()) for x in dep], list(zip(dep, a))


@claim("z64", "main §5.2", "SI Table S13", kind="reproduce", needs=("cctbx", "slow"))
def test_reproduce_lookup_benchmark():
    """lookup_benchmark.py regenerates results_lookup.txt (the deposited file also holds the
    Python warnings printed to stderr, which are left out)."""
    out, _ = _run("lookup_benchmark.py")
    _same(out, "results_lookup.txt")


@claim("z64", "main §5.2", kind="reproduce", needs=("cctbx",))
def test_reproduce_lookup_plate_case():
    """lookup_plate_case.py regenerates results_lookup_plate_case.txt identically."""
    out, _ = _run("lookup_plate_case.py")
    _same(out, "results_lookup_plate_case.txt")


@claim("z64", "main §5.3", kind="reproduce", needs=("cctbx", "slow"))
def test_reproduce_coset_rank():
    """coset_rank.py regenerates results_coset.txt identically."""
    out, _ = _run("coset_rank.py")
    _same(out, "results_coset.txt")


@claim("z64", "SI §S8", kind="reproduce", needs=("cctbx", "slow"))
def test_reproduce_benchmark_misses():
    """benchmark_misses.py regenerates results_benchmark_misses.txt identically."""
    out, _ = _run("benchmark_misses.py")
    _same(out, "results_benchmark_misses.txt")


@claim("z64", "main §5.2", "SI §S9", kind="reproduce", needs=("cctbx", "slow"))
def test_reproduce_niggli_window():
    """niggli_window.py 100 regenerates results_niggli_window.txt identically."""
    out, d = _run("niggli_window.py", "100")
    _same(open(os.path.join(d, "results_niggli_window.txt")).read(), "results_niggli_window.txt")


@claim("z64", "main §5.2", "SI §S9", kind="reproduce", needs=("cctbx",))
def test_reproduce_niggli_window_thin():
    """niggli_window.py thin regenerates results_niggli_window_thin.txt identically."""
    out, d = _run("niggli_window.py", "thin")
    _same(open(os.path.join(d, "results_niggli_window_thin.txt")).read(), "results_niggli_window_thin.txt")


@claim("z64", "main §5.1", kind="reproduce")
def test_reproduce_niggli_presentation():
    """niggli_presentation.py regenerates results_niggli_presentation.txt identically."""
    out, _ = _run("niggli_presentation.py")
    _same(out, "results_niggli_presentation.txt")


@claim("z64", "main §5.1", kind="reproduce", needs=("cctbx",))
def test_reproduce_niggli_types():
    """niggli_types.py regenerates results_niggli_types.txt identically."""
    out, _ = _run("niggli_types.py")
    _same(out, "results_niggli_types.txt")


@claim("z64", "main Lemma A2", kind="reproduce")
def test_reproduce_face_labellings():
    """face_labellings.py regenerates results_face_labellings.txt identically."""
    out, _ = _run("face_labellings.py")
    _same(out, "results_face_labellings.txt")


@claim("z64", "main Table 8", "SI §S7", kind="reproduce", needs=("slow",))
def test_reproduce_certified_radii():
    """certify_radii.py regenerates results_certified_radii.txt and certified_radii.json
    identically: the exact balls, the dual lower bounds and the feasible points."""
    out, d = _run("certify_radii.py")
    _same(open(os.path.join(d, "results_certified_radii.txt")).read(), "results_certified_radii.txt")
    assert json.load(open(os.path.join(d, "certified_radii.json"))) == json.loads(_text("certified_radii.json"))


@claim("z64", "main §5.2", "SI §S7", kind="reproduce", needs=("pdb", "slow"))
def test_reproduce_adjacency_radii():
    """adjacency_radii.py regenerates results_adjacency_radii.txt, apart from wall-clock
    times (every number followed by 's' or 'ms', and the 90th percentiles of the timings)."""
    out, _ = _run("adjacency_radii.py")
    _same(out, "results_adjacency_radii.txt", drop=(r"\d+(\.\d+)? (s|ms)\b", r"\(90th [\d.]+\)"))


@claim("z64", "SI Table S11", kind="reproduce", needs=("cctbx", "slow"))
def test_reproduce_lepage_calibration():
    """lepage_calibration.py 25 regenerates results_lepage_calibration.txt and
    lepage_calibration.jsonl identically."""
    out, d = _run("lepage_calibration.py", "25")
    _same(out, "results_lepage_calibration.txt")
    assert [json.loads(l) for l in open(os.path.join(d, "lepage_calibration.jsonl"))] == \
        [json.loads(l) for l in _text("lepage_calibration.jsonl").splitlines()]


@claim("z64", "SI §S6", kind="reproduce", needs=("sella",))
def test_reproduce_compare_sella():
    """compare_sella.py (SELLA_BIN = LatticeRepLib's CmdSella) regenerates results_sella.txt."""
    out, _ = _run("compare_sella.py")
    _same(out, "results_sella.txt")
