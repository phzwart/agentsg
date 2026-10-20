# Census revision 1 — fixes to `manuscript/census/make_census.py`

Rerun the full table after these changes, regenerate every output, and commit
as "Fix units, esd selection and sort closure in the census".

1. **Units of the ambiguity test.** `path["t"]` is in cosine units; `tau` is in
   conorm units (Å²). `ambiguous_esd` (`path["t"][0] < 4*tau`) and the `4tau`
   column of `sorts_by_source.csv` compare the two. Convert the band per slot
   to cosine units, `tau_cos_ij = 4*tau / (|u_i||u_j|)` for zero-type edges and
   `4*tau / sqrt(|u_i||u_j||u_k||u_l|)` for tie-type edges, and test each
   transition against the band of the edge it crosses. Better still, use the
   per-slot propagated sigmas: `3*sig_p[ij] / (|u_i||u_j|)`.

2. **Esd selection.** "All six esds present" keeps only triclinic rows
   (124,500 of 127,782), because COD gives no esd for parameters fixed by
   symmetry. Use the reported crystal system: a parameter fixed by the lattice
   (90° and 120° angles, equal lengths) gets σ = 0 and the propagated σ is
   taken from the free parameters, which must all have esds. Rows whose esds
   are all zero, or whose free parameters lack esds, stay nominal. Report the
   new esd/nominal split in the inventory.

3. **Sort closure in the filtration.** Closing the imposed ties under
   transitivity is not the sort closure; it produces 2,184 "unlisted" sets.
   Close with the operators, as in `test_z53_sorts_as_fixed_subspaces.py`:
   the closure of a constraint set V is Fix(Sym(V)) (symmetries = operators of
   the zero pattern fixing V pointwise; Fix = their common fixed subspace).
   After this every set is one of the 24 sorts. Constraints crossed at the
   same `t` (to 1e-12) are imposed together; record those as one step, and
   then report as `non-codimension-one-jump` only steps that cross more than
   one Table C2 edge at distinct `t`.

4. **Niggli.** Every closure member describes the same lattice under the same
   metric, so its Niggli cell is identical by uniqueness of Niggli reduction —
   the zero in `niggli.csv` is a tautology (the prompt was wrong). Replace it
   with the question the benchmark answered: does the Niggli cell of the
   measured cell differ (as an integer change of basis, i.e. a different
   Niggli setting of the same lattice) from the Niggli cell of the cell
   idealised at the band? Count per sort and per source.

5. **Zero bands.** `tau == 0` makes the walk compare against an exact zero
   with no numerical tolerance, while the closure uses `1e-9 max|G|`; this
   produces spurious walk misses (COD1507213, COD3000168). Floor `tau` at
   `1e-9 * max|G|`, and treat all-zero esds as missing (item 2).

6. **Keep** everything else. Add to `conorms.npz`: the per-slot `sig_p`
   (already there), the band per transition in cosine units, and
   `ambiguous_esd` from item 1. Do not commit it if it stays over 10 MB.

7. **Per-slot bands as the primary construction.** The uniform band
   `tau = 3*max sig_p` is set by the longest axis (σ(a²) = 2a·σ_a) and compared
   with λ₁, which is set by the shortest; `beyond-theorem` then fires whenever
   (long/short)² ≳ 5 under the nominal errors. Run the census a second time
   with the per-slot construction (`members_by('slot', p0, tau, sigma)` style,
   but with the band of each slot taken from the propagated `3*sig_p[ij]` of
   that slot rather than a common angular sigma), and its own hypothesis test
   from `test_z49_anisotropy_and_shapes.py::_construct_rel` (singular Λ*, or
   32 × the largest band on the idealised slots ≥ λ₁ over all four superbase
   vectors). Report both constructions side by side in tables C and the
   flags, and for every `beyond-theorem` / `step2-singular` cell under either
   construction check the closure against the brute-force enumeration
   (`test_z44_widened_complete._brute` over superbases with entries in
   [-2, 2]; for per-slot bands, the per-slot filter of
   `test_z49._brute_rel`). Report the counts exact / closure-misses / refused.

8. **Tolerance calibration tables.** (a) For every Table C2 edge and per
   reported system: the fraction of cells whose path *passes through* the
   source sort (denominator) and crosses the edge below δ, for zero-type edges
   at angular δ ∈ {0.03, 0.1, 0.3, 1, 3}° (cosine units sin δ), and for
   tie-type edges at relative squared-length differences 2ε with
   ε ∈ {0.5, 1, 2, 5}% (i.e. |p_ij − p_kl| ≤ 2ε·max(p_ij, p_kl)). Also the
   per-cell union: the fraction of cells with *any* edge below the tolerance.
   Write `tables/tolerance_calibration.csv`. (b) The same with δ replaced by
   k·σ for the esd-band cells, k ∈ {1, 2, 3, 5, 10}. (c) Filtration gaps: for
   each cell, the log-gaps between consecutive transitions
   `log10 t_{k+1} − log10 t_k` (with t_0 = the cell's 1σ cosine band where
   esds exist, else 1e-6); record the widest gap below sin 3° and the
   tolerance at its midpoint, and tabulate the distribution of that midpoint
   and of the gap width per exact sort. Write `tables/filtration_gaps.csv`
   and add the midpoint and gap width to `conorms.npz`.
