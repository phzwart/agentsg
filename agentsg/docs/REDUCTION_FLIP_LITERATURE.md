# The reduction flip: how well documented, and how software handles it

> Sourcing note. Bibliographic details (authors, year, journal, volume, pages,
> titles) were verified against retrieved reference/abstract pages (IUCr, PubMed,
> arXiv) in the session that produced this note. The prose is the author's
> synthesis of this literature stated in plain terms — it contains no verbatim
> quotations; every characterisation of a paper's content is paraphrase. Consult
> the primary sources for exact statements.


The "reduction flip" — the fact that a small change in a nearly-degenerate unit
cell can make the Niggli/Buerger reduction land on a *different but equivalent*
reduced cell (edges swap, an angle jumps from one axis to another) — is a
well-known and explicitly documented phenomenon, not a folklore edge case. It is
usually discussed under the headings **numerical instability of cell reduction**
and **boundary discontinuities of the reduced-cell cone**.

## What the literature says

**It is intrinsic, not a bug.** The Andrews–Bernstein series makes this explicit:
the space of reduced cells is bounded by *boundary polytopes* (the title of their
first paper is literally "The Geometry of Niggli Reduction I: The Boundary
Polytopes of the Niggli Cone", arXiv:1203.5146), and near such a boundary a small
change in the lattice can move the reduced cell discontinuously. Their 1988 paper
introduced the idea of representing a cell together with its near-alternate
(nearly Buerger-reduced) cells and using distances among these representations as
a way to reason about lattice similarity and the stability of the reduction
(Andrews & Bernstein, Acta Cryst. A44, 1009–1018, 1988; paraphrased). A probe
cell sitting on such a boundary is genuinely ambiguous: which reduced cell you get
depends on which side of the boundary rounding error puts you.

**The cause is edge/length degeneracy.** The relevant boundaries are where two
cell edges become equal (e.g. b = c in the metric tensor); Niggli's conditions
then impose a secondary tie-break on the associated angles to order the edges.
When that tie-break is itself near-degenerate, the ordering — and hence the
reduced cell — flips. This is precisely the `a ≈ b` case demonstrated in agentsg.
(Paraphrased; see the BGAOL papers for the exact boundary conditions.)

**The standard mitigation for the numerics is a consistent tolerance.**
Grosse-Kunstleve, Sauter & Adams (*Numerically stable algorithms for the
computation of reduced unit cells*, Acta Cryst. A60, 1–6, 2004) showed that a
conventional implementation of the Křivý–Gruber algorithm is numerically
unstable and gave a stabilised version based on consistent use of a tolerance in
the floating-point comparisons — the algorithm agentsg implements (paraphrased).
But a tolerance only removes the *rounding-error* flips; it cannot
remove the *genuine* ambiguity of a cell sitting exactly on a boundary. That
requires a different tool.

**The genuine ambiguity is handled by an embedding / distance metric, not by
reduction alone.** The Andrews–Bernstein programme (BGAOL, SAUC, NCDist; built on
their 1988 G6 embedding) represents a cell as a vector in a Euclidean 6-space —
the Niggli matrix elements with the off-diagonal terms doubled — in which the
non-triclinic Bravais types form linear subspaces, so choosing the best Bravais
lattice becomes a matter of Euclidean distances from the cell to those subspaces
(Andrews & Bernstein 1988). Because a single reduced representation is not
adequate near a boundary, the distance is taken over the reduced cell *and its
boundary-related alternates* — Andrews & Bernstein (1988) use the nearly
Buerger-reduced cells; later work searches a set of Gruber (1973) boundary
transforms. A companion S6 / Selling-scalar embedding was introduced later
(Andrews, Bernstein & Sauter, Acta Cryst. A75, 115–120, 2019). This is the
literature form of exactly the point at issue here: the primitive setting is not
unique under metric distortion, so a robust distance must consider the
boundary-related alternates. (The specific attributions to Zimmermann & Burzlaff,
Oishi-Tomiyasu, Kabsch and Macíček & Yordanov in earlier drafts have been removed
as they were not verified against source in this session.)

## How processing software handles it

- **DIALS `dials.refine_bravais_settings`** does NOT trust a single reduced cell.
  It refines the model in *all* Bravais settings consistent with the primitive
  cell and prints a table with the **metric fit** (deviation from triclinic),
  RMSD to spot centroids, refined cell, and the change-of-basis operator for each.
  The user (or a downstream tool) picks. This is a **combinatorial** treatment of
  the ambiguity — enumerate the candidates, score them — rather than a claim that
  reduction gives one answer.

- **DIALS indexing (`basis_vector_search.combinations`,
  `candidate_orientation_matrices`)** likewise enumerates unique *combinations* of
  candidate basis vectors and tests them, rather than committing to one reduced
  basis; `find_matching_symmetry` then matches against target symmetries within
  tolerance.

- **`dials.cosym` / `reindex_to_reference`** map each dataset to a common setting
  (`change_of_basis_op_to_primitive_setting` + `map_to_asu`) and then resolve the
  residual, setting-independent indexing ambiguity by **intensity correlation**
  (Brehm & Diederichs 2014; Gildea & Winter 2018). The "best cell" selection
  upstream (`change_of_basis_op_to_best_cell`) is itself tolerance-based
  (`max_delta`, relative length / absolute angle tolerances) — i.e. DIALS knows
  the setting choice is tolerance-dependent and treats it as such.

- **cctbx `reduction_base` / `bravais_types`** carries an `iteration_limit_exceeded`
  guard — a direct acknowledgement that reduction can fail to settle near a
  boundary.

## Bottom line

The reduction flip is **thoroughly documented** (Grosse-Kunstleve et al. 2004 for
the numerics; Andrews & Bernstein 1988 and the BGAOL papers for the boundary
geometry) and
**well handled** in production software — but *not* by making reduction unique.
The field's answer is one of two strategies, both combinatorial in spirit:

  1. **Enumerate candidates + score geometrically** (refine_bravais_settings:
     metric fit + RMSD; BGAOL/SAUC: boundary-aware G6 distance). This is the
     regime where the metric itself discriminates — pseudo-merohedral /
     cell-choice.
  2. **Enumerate candidates + score by intensity** (cosym / reindex_to_reference).
     This is the only thing that resolves *true* merohedral ambiguity, where every
     candidate has an identical metric.

agentsg's design matches this exactly: `surface_geometric_operators` supplies the
complete candidate set (the coset, reduction flips included), annotated with a
metric residual and an `is_metric_symmetry` flag that marks which branches fall
into regime (2); `ReindexingReference.resolve` covers regime (1) by geometry, and
`resolve_intensities` covers regime (2) by intensity CC. What agentsg does NOT
implement is the Andrews–Bernstein *embedding* — it uses the exact
metric-automorphism coset instead of a continuous boundary-aware distance. The
embedding would be the natural extension for cell *database search* (finding
near-neighbours across boundaries), which is a different task from resolving a
per-frame reindexing choice.

## Key references

(Bibliographic details below verified against retrieved IUCr / PubMed / arXiv
reference pages in this session.)

- Křivý, I. & Gruber, B. (1976). Acta Cryst. A32, 297–298. Unique Niggli
  reduction algorithm.
- Gruber, B. (1973). Acta Cryst. A29, 433–440. The reduction boundary transforms.
- Grosse-Kunstleve, R. W., Sauter, N. K. & Adams, P. D. (2004). Acta Cryst. A60,
  1–6. Numerically stable reduction (the tolerance fix).
- Andrews, L. C. & Bernstein, H. J. (1988). Acta Cryst. A44, 1009–1018. "Lattices
  and reduced cells as points in 6-space and selection of Bravais lattice type by
  projections" — the G6 embedding.
- Andrews, L. C. & Bernstein, H. J. (2014). J. Appl. Cryst. 47, 346–359. BGAOL —
  embedding Niggli reduction (Geometry of Niggli Reduction II). (Part I: boundary
  polytopes, arXiv:1203.5146.)
- McGill, K. J., Asadi, M., Karakasheva, M. T., Andrews, L. C. & Bernstein, H. J.
  (2014). J. Appl. Cryst. 47, 360–364. SAUC — search of alternate unit cells.
- Andrews, L. C., Bernstein, H. J. & Sauter, N. K. (2019). Acta Cryst. A75,
  115–120. Selling reduction versus Niggli reduction (the S6 space).
- Brehm, W. & Diederichs, K. (2014). Acta Cryst. D70, 101–109. Indexing-ambiguity
  resolution by intensity correlation.
- Gildea, R. J. & Winter, G. (2018). Acta Cryst. D74, 405–410. dials.cosym.
