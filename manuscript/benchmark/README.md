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
lattice character (14 of the 45 are a type I / type II change). By sort, the 45
are O1B 23, O3 12, M1B 6 and one each at M2B, M3, R3 and T2; the 125 ties are O1B 25,
M1A 15, O1A 15, T1 15, M1B 13, O4 9, O2 7, R1 7, M2A 6, T2 6, O3 5, A2 1 and A3 1
(`results_niggli.txt`). The benchmark was built near Selling zero conorms, so
Niggli-specific boundaries are under-sampled. A tie changes the reindexing operator as a
different cell does; in every one of these cases the operator between the two Niggli
cells is in the closure's candidate set.

## The strain-norm procedure

`evaluate_strain.py` scores `agentsg.cell.strain_closure` (the procedure of the
paper) and a small-matrix baseline. The reference is widened at `band_for(accept)`
in the strain norm and the measured cell reduced once; the baseline Selling-reduces
both cells and tries the 6960 unimodular matrices with entries in {-1, 0, 1} between
the reduced bases, priced in the same norm. At `accept` = 0.074 (one angle changed by
3 degrees) the closure has the true operator among the candidates in 780 of 780
cases and the baseline in 778 (two triclinic cells near mP whose operator between
the reduced bases has an entry of 2); at three times each case's own error strain,
778 of 778 and 776 of 778 (two cases exceed the proved range at that multiple).
The 16 anisotropic cases fall in layer 2 (12) and layer 1 (4); none is refused.
Results, with candidate-count percentiles, are in `results_strain.txt`.

    PYTHONPATH=agentsg/src python manuscript/benchmark/evaluate_strain.py

Since the paper's round-2 revision only proper operators are scored: the true operator
is made proper (-M when det M = -1) and must itself be among the proper candidates.

## Comparisons

* `compare_methods.py` scores six methods on the benchmark (`results_compare.txt`) and,
  with `--stress`, on 240 cases with independent errors on both cells at 0.5, 0.9, 1.1
  and 1.5 times the band (`results_stress.txt`): the closure, the {-1,0,1} and
  {-2,...,2} searches between Selling-reduced cells, cctbx
  `unit_cell.similarity_transformations` (generator range 1 and 2) between Niggli
  cells, and a cctbx `lattice_symmetry` pipeline (highest lattice group of each cell,
  every operation between the two standard settings). Needs cctbx.
* `coset_rank.py`: how often the closure's cheapest candidate lies in the true
  operator's coset of the exact lattice symmetries (`results_coset.txt`).
* `cod_real_cells.py`: the 210 COD census cells with deposited esds, reindexed under
  their own error model at a band of 2.5 times their rms strain, and the reported
  Bravais types against the sort spectrum (`results_cod.txt`).
* `compare_sella.py` (SELLA_BIN=CmdSella from LatticeRepLib) and `compare_s6.py`
  (NCDIST_DIR with libcs6dist.so and libncdist.so): sort rankings against SELLA, and
  nearest-reference matching with CS6Dist and NCDist (`results_sella.txt`,
  `results_s6.txt`).

## Window failures, Niggli types and the PDB

* `window_failures.py` (LM_BIN = LatticeRepLib CmdLM, optional): lattices whose closure
  needs an operator entry of 3, and near-mP / near-orthogonal lattices measured with a
  type I or type II Niggli cell; the closure, the lookup of the 6096 matrices, the
  small-matrix and cctbx searches, the pipeline and CmdLM (`results_window_failures.txt`).
* `niggli_types.py`: Selling reduction from the Niggli cell of near-monoclinic cells and
  where the near-zero conorms land (`results_niggli_types.txt`).
* `pdb_nonisomorphism.py`, `pdb_lepage.py`, `pdb_pricing_summary.py` need the Zenodo
  snapshot in `data/pdb_cells.duckdb`: the price between entries of one crystal form, and
  Le Page's 3 degrees on every entry with the prices of the pseudo-symmetries
  (`results_pdb_*.txt`, `pdb_lepage.jsonl`).
* `lepage_calibration.py`: the strain distance to each sort and the operator prices at
  exactly 3 degrees of obliquity (`results_lepage_calibration.txt`).

Since the geodesic price (agentsg CHANGELOG), every script prices candidates by the
affine-invariant distance after the best scale.
