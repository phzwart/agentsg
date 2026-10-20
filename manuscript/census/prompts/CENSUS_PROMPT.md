# Wall-incidence census of real unit cells (COD / AMCSD / PDB)

You are working in the `agentsg` repository (branch `main`). A DuckDB cell
database built with `agentsg.cell.celldb.CellDatabase` sits at

    DB=<absolute path to the .duckdb>          # fill in before running

It holds PDB cells (`source = 'pdb'`) and small-molecule cells ingested with
`add_cod` (`source = 'cod'`, possibly `'amcsd'`), each with the deposited cell
`a..gamma`, `sg_number`, `sg_hm`, one Selling-reduced cell `red_a..red_gamma`,
the change of basis `cob00..cob22`, and for COD/AMCSD rows the esds
`sig_a..sig_gamma` (NULL where the file gave none).

Goal: measure, on real cells, the quantities that the paper
`manuscript/selling_reduction/zwart_selling_reduction.tex` currently supports
only with the constructed benchmark in `manuscript/benchmark/`: how often a
measured cell sits near a wall of the Selling orthant, how large its widened
closure is at the precision of the measurement, how often a flip walk misses a
member, and how often the metric symmetry at that precision disagrees with the
reported space group. Produce aggregate tables and a short list of flagged
cases that a reviewer can re-check by id. Do not change library code; if a
function misbehaves, work around it in the census script and record the issue
in the README.

## Deliverables (all under `manuscript/census/`, total size under 5 MB)

1. `make_census.py` — one script, run as
   `python manuscript/census/make_census.py DB [--sample N] [--seed 0] [--out manuscript/census]`.
   Deterministic for a given seed. `--sample N` draws N rows per source (use
   2000 for development, then run the full table). Reuse the helpers that the
   benchmark already uses rather than re-deriving them:
   `manuscript/appendix_z/test_z44_widened_complete.py` (`_gram`, `_cn`,
   `_flip`, `_key`, `_lam1`, `_generate`, `S0`, `SL`) and
   `manuscript/benchmark/evaluate.py` (`walk`, `members_by` with methods
   `exact`, `closure`, `walk-tau`, `walk-2tau`, `slot`). Add
   `manuscript/appendix_z` and `manuscript/benchmark` to `sys.path` the way
   `evaluate.py` does.
2. `census_summary.json` — every number quoted in the README, machine-readable.
3. `tables/*.csv` — the aggregate tables below (aggregates only, never one row
   per cell).
4. `flagged.jsonl` — at most 500 lines, one per flagged cell: `id, source,
   sg_number, sg_hm, cell, esd, conorms, lam1, tau, n_closure, n_walk_tau,
   n_walk_2tau, visible_order, reported_system, flags`. Flags are the strings
   defined in sections C–E. Choose the 500 by stratifying over flags and
   sources, not the first 500 found.
5. `README.md` — method, parameters, runtime, the tables rendered as markdown,
   and a section "Issues" for anything that did not behave as expected.
6. `env.txt` — `git rev-parse HEAD`, agentsg version, duckdb and numpy
   versions, row counts per source, `schema_meta` contents, wall-clock time.

Never commit the `.duckdb`. Commit the six items above on `main` in one
commit titled "Add the wall-incidence census of real cells".

## Definitions (use these exactly; they match the paper)

- Reduced superbase: from the stored reduced cell `red_*`, build the basis
  vectors `a, b, c` and `d = -(a+b+c)`; the six conorms are
  `p_ij = -u_i . u_j` in slot order `da, db, dc, ab, ac, bc` (this is the
  order `SL = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]` with index 0 = d). All six
  must be >= -1e-9 * max|G|; count and report any row where they are not (that
  is a stored-cell problem, not a result).
- `lam1`: the smallest squared length among `a, b, c, d, a+b, a+c, b+c`
  (`_lam1(p)` in test_z44 computes it from the conorms).
- Relative wall proximity: `r = min_ij p_ij / lam1`.
- Band `tau` of a cell. Where all six esds are present, propagate them to the
  six conorms of the reduced superbase by a numerical Jacobian of the map
  (a,b,c,alpha,beta,gamma) -> G -> P^T G P -> conorms (P = the stored cob
  matrix, `parse_cob_columns`), treating the six parameter errors as
  independent, and set `tau = 3 * max_ij sigma(p_ij)` (one uniform band, as in
  the paper's main construction). Where esds are missing (all PDB rows, some
  COD rows) use the nominal errors `sigma_len = 0.001 * length` and
  `sigma_angle = 0.03 degrees` and record in the output that the band is
  nominal (`tau_kind = 'esd' | 'nominal'`).
- Closure at `tau`: `members_by('closure', p0, tau)` — idealise at `4*tau`,
  enumerate, filter at `tau`. It returns `None` when the idealised lattice is
  singular; count those as `flag = 'step2-singular'`. Separately test the
  theorem's hypothesis `lam1 > 32*tau` and set `flag = 'beyond-theorem'` when
  it fails (the construction is still run; the paper says it stays exact up to
  about `lam1/6`).
- Walks: `members_by('walk-tau', ...)` and `members_by('walk-2tau', ...)`.
- Visible symmetry: the number of the 48 signed slot permutations that leave
  the idealised conorm vector (zeros at `<= 4*tau`, equalities at `<= 4*tau`)
  invariant. Map its order to a crystal system by the holohedry orders in Table
  4 of the paper (48 cubic, 24 hexagonal, 16 tetragonal, 12 rhombohedral, 8
  orthorhombic, 4 monoclinic, 2 triclinic); note that the visible order is a
  lower bound on the lattice symmetry (Table 4, column "visible"), so compare
  reported vs visible only in the direction that is meaningful (section D).
- Reported crystal system: from `sg_number` (1–2 triclinic, 3–15 monoclinic,
  16–74 orthorhombic, 75–142 tetragonal, 143–167 trigonal, 168–194 hexagonal,
  195–230 cubic); rows with NULL `sg_number` form their own class "unknown".

## Tables

A. Inventory — rows per source; per source x reported system; per source the
   number with esds, with NULL sg_number, with a non-P lattice letter; cells
   whose stored reduced cell fails the obtuse test.

B. Wall proximity — per source, the histogram of `log10 r` in half-decade bins
   from -6 to 0 (with a separate bin for exact zeros, r = 0), and the fraction
   of cells with `r < 1e-2, 1e-3, 1e-4, 1e-5`. Also, at relative tolerance
   `1e-3 * lam1`, the number of near-zero conorms (0, 1, 2, 3, more) cross-tabbed
   against the reported system — cubic, hexagonal, tetragonal and orthorhombic
   lattices should populate the 2- and 3-zero columns; say whether they do.

C. Closure census at `tau` — per source and per reported system: the
   distribution of `n_closure` over {1, 2, 3, 6, 16, other}; the fraction of
   cells where `n_walk_tau < n_closure` (`flag = 'walk-misses'`) and where
   `n_walk_2tau < n_closure` (`flag = 'walk2-misses'`); the fraction
   `step2-singular` and `beyond-theorem`. Give the same table twice: for
   `tau_kind = 'esd'` rows and for nominal rows. If the esd-band rows show a
   markedly different picture from the nominal ones, say so in the README.

D. Symmetry at the measurement precision — cross-tab of reported system vs the
   system implied by the visible order. Flag `metric-above-reported` when the
   visible holohedry order exceeds the reported system's, and
   `reported-above-visible` when the reported system's holohedry order exceeds
   the visible order AND the closure has only one member (the second condition
   avoids flagging the genuine Table-4 cases where symmetry is hidden across
   members). Count both per source, and give the ten most common (reported,
   visible) off-diagonal pairs.

E. Classic traps — among monoclinic rows (sg 3–15): the number whose closure
   has two or more members of the same visible order (the cell-choice
   ambiguity), broken down by whether the reported symbol is a C/I/A setting.
   Among orthorhombic rows: the number with `n_closure > 1` (the oP/oC
   families). Report what fraction of all `walk-misses` cases fall in these
   two groups.

F. Niggli — for every cell with `n_closure > 1`, the Niggli cell
   (`agentsg.cell.reduction.niggli_gk`) of the measured reduced cell versus the
   Niggli cell of each other closure member under the measured metric; count
   the cells where any member gives a different Niggli cell (the benchmark's
   45/780 number, now on real cells), per source.

G. Spot checks — print, with full numbers, the rows for these ids when
   present: `COD9000001`-style alpha-quartz, NaCl, Si, albite, diopside, any
   AMCSD feldspar, and PDB `193L`, `1LYZ`, `2HHB`. If an id does not exist,
   pick the nearest formula/name match and say which.

## Practicalities

- The closure construction is the expensive step. Time it on the 2000-row
  sample first and extrapolate; if the full table would take more than two
  hours, run the full table for B and the inventory, and a stratified sample
  (equal counts per reported system, up to 20 000 per source) for C–F, and
  say so in the README.
- Use `db.sql(...)` from `CellDatabase` (open read-only) for the pulls; do not
  load the whole table into pandas at once if it is large — stream in chunks.
- Keep the numeric tolerance `1e-9 * max|G|` of the appendix code everywhere;
  do not introduce new tolerances.
- When the output disagrees with the paper's expectations (for example, many
  triclinic-reported cells with three near-zeros, or a large `walk-misses`
  rate among cells with esds), do not smooth it over: report it under
  "Issues" with five example ids so it can be looked at.

H. Raw dimensionless conorms for the distributional analysis — write
   `conorms.parquet` (or `.npz` if pyarrow is not installed) with one row per
   cell: `id, source, sg_number, reported_system, V, lam1, tau, tau_kind`, the
   six conorms `p_da..p_bc` of the reduced superbase in slot order, the six
   cosines `cos_da..cos_bc = p_ij / (|u_i||u_j|)`, the six esd-propagated
   `sig_da..sig_bc` (NaN where nominal), and for each slot a boolean
   `forced_da..forced_bc` that is true when the reported space group's lattice
   forces that conorm to zero (derive it from the metric symmetry of the
   reported crystal system applied to the reduced superbase: a slot is forced
   when it is zero and remains zero under every signed slot permutation that
   fixes the idealised conorm vector — use `4*tau` for the idealisation — and
   the reported holohedry order is at least the visible order). Use float32;
   commit the file if it is under 10 MB, otherwise do not commit it and say in
   the README where it was written so it can be attached separately. Do no
   fitting here; this file feeds a separate analysis.

I. The specialisation filtration — assign sorts as a function of tolerance,
   as one monotone path per cell, not as independent snapshots.

   The object. For a tolerance t, idealise the reduced conorms by imposing
   every zero `p_ij <= t` and every tie `|p_ij - p_kl| <= t`, and close the
   constraint set (ties are transitive; a closed set of equalities is one of
   the 24 sorts of Table 4, identified from its (zeros, equality classes)
   placement exactly as `manuscript/appendix_z/test_z54_table_from_operators.py`
   does with `_sella_partition` / `vis_by_pattern`). As t grows the constraint
   set only grows, so each cell moves monotonically up the Hasse diagram of
   Table C2 (`test_z50_specialisations.py`), from its exact sort at t = 0 to
   C5. Compute that path exactly: list the candidate transitions of the cell
   in cosine units (zero-type `d = p_ij / (|u_i||u_j|)`, equality-type
   `d = |p_ij - p_kl| / sqrt(|u_i||u_j||u_k||u_l|)`), sort them, accumulate
   them one at a time, re-close after each (a newly imposed constraint can
   make earlier near-ties closed, and transitivity can pull in a tie up to
   2t), and record `(t_k, sort_k)` whenever the sort changes. Do NOT compute
   the sort independently at several tolerances and compare: that is not
   monotone and shows spurious back-and-forth.

   Per cell (into `conorms.parquet`): `sort_exact` (zeros at 1e-9 relative),
   the path as `t_1..t_K` and `sort_1..sort_K` (pad to K = 6 with NaN), the
   edge taken at each step (named by the Table C2 condition), `t_1` as the
   persistence of the exact sort, and `ambiguous_esd = (t_1 < 4*tau)` in the
   cell's own band.

   Aggregates: (a) `tables/edge_transitions.csv` — for each edge of the Hasse
   diagram (each Table C2 condition), per source: number of cells crossing it
   and the histogram of `log10 t` at the crossing (half-decade bins, -6..0).
   This is the per-sort histogram organised by the condition that is being
   approached; it is what the lower-tail analysis will be fitted to. (b)
   `tables/sort_population.csv` — for each sort, per source, the number of
   cells sitting in it at t = 10^-6, 10^-5.5, ..., 10^0 (the population
   curves; read them off the paths). (c) `tables/sorts_by_source.csv` — the
   same at the four readable tolerances `4*tau` (esd band), `1e-2, 1e-3,
   1e-4` in cosine units, as a table. (d) the fraction `ambiguous_esd` per
   `sort_exact` and per source. (e) in the README, render (c), the ten most
   crossed edges of (a) with their histograms, and the population curves of
   the ten most populated sorts.

   Do not fit anything; the fits and the invariant-measure reference per edge
   are done separately from the parquet.
