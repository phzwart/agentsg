# Machine-checked proofs (Lean 4 + Mathlib)

This directory formalises the parts of the manuscript's appendix that do not need matrix
analysis.  Build with

```
lake build
```

(Lean and Mathlib versions are pinned in `lean-toolchain` and `lake-manifest.json`.)
`SellingClosure/Data.lean` is generated from
`manuscript/benchmark/universal_operators_columns.json` by `gen_data.py`.

## Setting

A lattice is given by the six conorms `p : Fin 6 → ℝ` of a reduced superbase
`(d, a, b, c)`, `d = -a-b-c`, in slot order `da, db, dc, ab, ac, bc`.  A lattice vector is
an integer triple `(x₁, x₂, x₃)` in the basis `(a, b, c)`, and the dot product is
Selling's identity (`Analytic.dot`).  An operator `W` is given by its three columns, the
new `a, b, c` in the old basis, as in the deposited table.

## What is proved

| Paper | Lean | Notes |
|---|---|---|
| 𝒰 has 6096 distinct unimodular elements, entries in {−3,…,3}, closed under inversion | `Finite.U_length`, `U_nodup`, `U_unimodular`, `U_entries`, `U_inv` | `native_decide` |
| 38 faces (connected supports) | `Finite.faces_count` | `native_decide` |
| Lemma A2 (class excess identity) | `Analytic.excess_identity`, `gam_nonneg` | |
| Selling's identity in the frame of any superbase; partial sums of an obtuse superbase are shortest in their parity class | `Analytic.selling_identity`, `partial_shortest` | |
| Lemma A1 (window) | `Finite.box_of_tree`, `box_of_conn` | via the 16 spanning trees |
| **Completeness of 𝒰**: every unimodular matrix between two obtuse superbases of any lattice is in the deposited table | `Analytic.arrow_mem_U` | the finite step `Finite.finite_check` uses `native_decide` |
| Every element of 𝒰 is an arrow of every lattice on some face | `Analytic.U_arrow` | `native_decide` |
| Arrows depend only on the zero pattern | contained in `arrow_mem_U` and `U_arrow` | |
| Proposition A10 (i)–(v): prices subadditive, candidates between two cosets, a group in a gap, one coset in a gap, bounds from the nearest symmetric lattice | `Displacement.price_mul`, `price_inv`, `coset_lower`, `coset_upper`, `Hgroup`, `candidates_eq_coset`, `stab_sub_H`, `H_sub_H` | for any isometric group action on a metric space |
| Proposition A9 (i): price ≤ 2 × distance to the fixed metrics; equality for a twofold given unique midpoints | `Displacement.price_le_two_dist_fixed`, `twofold_midpoint_fixed` | uniqueness of midpoints (non-positive curvature) is a hypothesis |
| Proposition A11 (i), (ii) | `Niggli.typeII_obtuse`, `typeII_d_conorms`, `typeII_separation`, `typeI_no_orthogonal_obtuse` | |

## What is not formalised

- Proposition A4, Lemmas A6 and A8, Proposition A9 (ii)–(iii): these use the
  affine-invariant distance on positive definite matrices (matrix square roots, logarithms
  and eigenvalues), which Mathlib does not yet provide in a usable form.  The facts used
  about it (a metric, changes of basis are isometries, unique midpoints) enter the
  formalised statements as hypotheses of an isometric action on a metric space.
- Proposition A11 (iii), Remark A7 and the layer counts (127, 2755), the adjacency radii,
  and the conorm band of the supporting information.
- Lemma A3 (generation by flips) and the termination of Selling reduction: not needed for
  `arrow_mem_U`, which is proved directly.

## Trust

`native_decide` evaluates the finite checks with compiled code, so these theorems also
trust the Lean compiler (axiom `Lean.ofReduceBool`); `#print axioms` lists it.  Everything
else is checked by the Lean kernel.
