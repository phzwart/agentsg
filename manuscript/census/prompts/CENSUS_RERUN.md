# Census revision 2 — per-slot closure, calibration, path length

Fixes 1–6 of `manuscript/census/prompts/CENSUS_FIXES.md` are in (`b63783b`); this file supersedes its items 7 and 8. Do the following on `main`
(which now has `manuscript/appendix_z/test_z57_perslot_bands.py`), rerun the
full table, regenerate the outputs, and commit as
"Add the per-slot closure and the tolerance calibration to the census".

1. **Per-slot closure with consistent idealisation (replaces item 7).** Do not
   use `members_by('slot')` or `test_z49._construct_rel`: their idealisation
   in the frame of F0 alone is wrong on some needles (see z57). Use
   `test_z57_perslot_bands.consistent_construct(G, sigma)`, with sigma the
   angular band of the cell: where esds exist, `sigma = 3 * max_ij sig_p[ij] /
   (|u_i||u_j|)` (the propagated per-slot sigma in cosine units); nominal rows
   `sigma = sin(0.03 deg)`. It returns `(set | None, Z, status, rounds)`;
   `None` is a refusal (singular idealisation), `status = 'ok'` means the
   condition of Conjecture A6'' holds. Report per source x reported system x
   tau_kind: exact closure sizes, `ok` / `hypothesis` / refused, and the
   walk-misses rate measured against this closure, side by side with the
   uniform construction already in table C.

2. **Brute-force checks use [-3, 3].** For every refused cell, and for a
   stratified sample of 2,000 `hypothesis` cells, enumerate the widened set
   with `test_z57_perslot_bands.delta_obtuse(G, sigma, 3)` and compare with
   the consistent construction where it returned a set. Report exact / wrong /
   refused, and the distribution of the largest basis coordinate needed (1, 2,
   3). For 200 of them also run K = 4 and report whether anything is added.
   Any `wrong` is an issue with five example ids and their cells.

3. **The 3° rule on the archive.** Run the consistent construction a second
   time at `sigma = sin(3 deg) / 4` for every cell (idealise at 3°, filter at
   about 0.75°). Report per reported system and per axis-ratio bin (longest
   / shortest edge of the reduced cell: <1.5, 1.5–2.4, 2.4–3, 3–5, 5–10, >10):
   condition holds (`ok`), exact beyond it (`hypothesis`), refused; and the
   closure-size distribution. This is the table the paper quotes for "how
   often does a 3° enumeration work on real cells".

4. **Calibration and filtration gaps.** Two new tables. Their purpose: the
   first prices a tolerance rule (what fraction of real cells a given δ, ε or
   kσ offers a higher-symmetry candidate), the second proposes a per-cell
   tolerance from the cell's own filtration.

   Distances used below. For an edge of Table C2 leaving a sort S (as named in
   `test_z50_specialisations.py`), read at the moment the cell's path is in S:
   * zero-type edge (a class r -> 0): `d_cos = max over the slots ij of the
     class of p_ij / (|u_i||u_j|)` (cosine units), `d_rel` undefined,
     `d_sig = max p_ij / sig_p[ij]`;
   * tie-type edge (classes r, s merge): with `p_r`, `p_s` the class means,
     `d_rel = |p_r - p_s| / max(p_r, p_s)`, `d_cos` as in the path,
     `d_sig = |p_r - p_s| / sqrt(sig_r^2 + sig_s^2)` with `sig_r` the largest
     `sig_p` in the class (covariances ignored; say so).
   `d_sig` only for `tau_kind = esd` rows.

   **`tables/tolerance_calibration.csv`**, one row per
   (source, reported_system, edge, tol_kind, tol_value), columns:
   `source, reported_system, edge, edge_type, tol_kind, tol_value,
   n_through, n_within, frac`.
   * `edge` is the Table C2 condition name (e.g. `M4>O5: r = 0`), or one of
     `any_zero`, `any_tie`, `any`.
   * `tol_kind = angle_deg` with `tol_value` in {0.03, 0.1, 0.3, 1, 3}: a
     zero-type edge is within when `d_cos <= sin(tol_value)`;
     `tol_kind = length_pct` with `tol_value` in {0.5, 1, 2, 5}: a tie-type
     edge is within when `d_rel <= 2 * tol_value / 100`;
     `tol_kind = k_sigma` with `tol_value` in {1, 2, 3, 5, 10}: either type is
     within when `d_sig <= tol_value` (esd rows only);
     `tol_kind = rule` with `tol_value` one of `0.1deg/0.5pct`,
     `0.3deg/1pct`, `1deg/2pct`, `3deg/5pct`: used only on the `any` rows,
     a cell is within when any zero-type edge passes the angle and any tie-type
     edge passes the length.
   * `n_through` is the number of cells whose path passes through the source
     sort of the edge (including cells whose exact sort is that sort); for
     `any_*` rows it is every cell of the group. `n_within` counts cells with
     that edge (or, for `any_*`, at least one edge of that type) within the
     tolerance; `frac = n_within / n_through`.
   * Also write the same with `reported_system = all`.

   **`tables/filtration_gaps.csv`**. Per cell: the merged transition
   tolerances of its path `t_1 < t_2 < ...` (cosine units), with
   `t_0` = the cell's 1σ band in cosine units (`max_ij sig_p[ij] /
   (|u_i||u_j|)`) for esd rows and `1e-6` otherwise, and `t_top = sin 3°`;
   keep the transitions in `(t_0, t_top)`, and the log-gaps
   `log10 t_k - log10 t_(k-1)` between consecutive members of
   `t_0, t_1, ..., t_top`. The widest gap gives `gap_lo, gap_hi, gap_width`
   (decades), `gap_mid = sqrt(gap_lo * gap_hi)` and `gap_sort` (the sort the
   cell is in throughout the gap). Rows, one per (source, sort_exact,
   tau_kind), columns: `source, sort_exact, tau_kind, n, width_q10, width_q25,
   width_q50, width_q75, width_q90, frac_width_ge_1, frac_width_ge_2,
   mid_deg_q10, mid_deg_q50, mid_deg_q90, top_gap_sort, frac_top_gap_sort`,
   with `mid_deg = degrees(asin(gap_mid))`. A second file
   `tables/filtration_gap_sorts.csv` has one row per (source, gap_sort):
   `n, mid_deg_q50, width_q50`. Add `gap_lo, gap_hi, gap_width, gap_mid,
   gap_sort` to `conorms.npz`.

5. **Path length.** 11,948 cells report a sort path longer than six steps.
   A path that only adds constraints climbs at most five codimension-one steps
   from A1 to C5 (from six free conorm values to one), so more than six is a
   bug: the closure under Fix(Sym) is not monotone, or simultaneous crossings
   are recorded as separate steps. Find which, fix it, and assert in the
   script that every path is monotone (each sort's subspace contains the next)
   and has at most five steps after merging crossings at equal t.

Commit `conorms.npz` only if under 10 MB; otherwise say where it is.
