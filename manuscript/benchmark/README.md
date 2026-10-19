# Conditions-for-trouble benchmark

Measured unit cells constructed from the places where the Selling closure
says two correct descriptions of one lattice can disagree (the paper
*The Selling closure*, Sections 4.1 and 4.3). Each case is a JSON line in
`benchmark.jsonl` with an exact reference lattice, a measured cell (the
reference plus a metric error of Frobenius norm `eta`, reindexed by a random
unimodular matrix), the true operator, the suggested band `tau = 3 eta`,
`lambda1` and whether `lambda1 > 32 tau`, and the expected answers.

| family | cases | what is hard |
|---|---|---|
| shapes | 75 | the measured cell is a different cell shape of the same lattice (the three monoclinic cell choices, the monoclinic-looking cells of oP, tP, cP, ...); no symmetry relates it to the reference's reduced cell |
| specialisation | 660 | the lattice is a distance `delta` (0.5 to 8 `eta`) from a codimension-one condition that would make it a higher sort (Table 6 of the paper): pseudo-symmetry priced at about `delta` |
| hidden | 20 | the measured cell's only reduced superbase is a member of the reference's widened closure reachable only through a non-member (two small conorms with `z <= tau < y <= z + tau`) |
| anisotropy | 16 | needles and plates with an angular error of 0.02 to 0.5 degrees; eight cases violate `lambda1 > 32 tau` with a uniform band |
| boundary | 9 | the boundary sorts O3, M2B, M3 |

`true_operator` M satisfies `G_measured = M^T (G_reference + E) M` with the
error E in the reference's reduced frame. `measured_niggli_cell` is the
Niggli-reduced measured cell (Grosse-Kunstleve, Sauter & Adams 2004) and
`niggli_from_measured` the matrix N with `G_niggli = N^T G_measured N`, so the
true operator from the Niggli cell to the reference frame is `M N`.

## Scoring

`python evaluate.py` scores, for each case, whether a method proposes the
true operator from a single Selling reduction of the measured cell paired
with the method's stored members of the reference (`reindex ok`), and
whether the method's member set at `tau` has the brute-force size
(`set ok`). `flagged` counts cases a method refuses because its hypothesis
is violated. Methods: `symmetry` (reference reduced cell in its 48
labellings), `exact` (the exact closure), `walk-tau` and `walk-2tau`
(flip wherever `|p| <= tau` or `2 tau`, filter), `closure` (idealise at
`4 tau`, enumerate, filter) and `slot` (the same with a band per slot where
the case carries an angular error). Results are in `results.txt`.

Reproduce: `PYTHONPATH=agentsg/src python manuscript/benchmark/make_benchmark.py`
(seed 2026), then `evaluate.py`. The generator imports the check files in
`manuscript/appendix_z`.

## The Niggli cell across the benchmark

`python evaluate.py --niggli` compares the Niggli cell of each measured cell
with the Niggli cell of its reference. In 170 of the 780 cases the measured
Niggli basis is a different basis of the lattice (not a signed permutation of
the reference's); in 125 of those the cell parameters agree up to relabelling
(a tie between Niggli's special conditions decided the other way), and in 45
the Niggli cell is a different cell: a type I / type II flip or a different
lattice character. Those 45 are concentrated on the body-centred orthorhombic
boundary sort O3 and the lattices one condition away from it (oI O1B, mC M2B
and M3, cF), and on the mC M1B/M3 boundary --- the known instability of the
Niggli cell (Gruber 1973; Andrews & Bernstein 1988), here with the sorts that
carry it. A conventional cell derived from the measured Niggli cell through
its lattice character then starts from a cell the reference's route never
produced; the two Niggli cells are two shapes of one closure, related by a
flip and by no symmetry, and the operator between them is in the closure's
candidate set in every one of these cases (`reindex ok` above).
