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
error E in the reference's reduced frame.

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
