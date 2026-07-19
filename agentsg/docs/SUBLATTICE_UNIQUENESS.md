# Sublattice / supercell non-uniqueness and the root invariant

## The question

When we relate two cells by a volume change (index r), the supercell / sublattice
is **not unique**. Does the Kurlin (2022) root invariant address this?

## Answer: no — it is out of scope

Kurlin (2022), *A complete isometry classification of 3-dimensional lattices*
(arXiv:2201.10543), solves exactly one problem: the complete, continuous
**isometry** classification of a *single* lattice. A text search of the paper
finds zero occurrences of "sublattice", "superlattice", "supercell", "subgroup",
"coincidence", or "Hermite"; all 33 occurrences of "index" mean *index-
permutation* (relabelling the four superbase vectors), not sublattice index.

The root invariant answers **"are these two lattices the same lattice?"** It does
not answer **"is this lattice a supercell of that one?"** — that is classical
sublattice theory (Delaunay / Hermite normal form), independent of the root form.

## Two kinds of multiplicity — only one is redundant

1. **Many HNF matrices -> the same lattice (redundant).** For index r there are
   A001001(r) Hermite-normal-form matrices (7 for r=2, 13 for r=3, ...). Several
   can describe the *same* physical sublattice in different bases. Niggli
   reduction + the root invariant collapse these to one point — this is where the
   root form *does* help: it deduplicates the enumerated list canonically.

2. **Many genuinely inequivalent sublattices of the same index (irreducible).**
   After removing (1), a lattice still has several *truly different* index-r
   sublattices. Example (orthorhombic 40,50,60,90,90,90): **7 distinct index-2
   supercells**, each a different lattice with a different root invariant
   (doubling a, b, or c, plus four centred variants). This multiplicity is NOT
   redundant and CANNOT be collapsed — they are different lattices. It is why
   "the supercell of a cell" is ill-defined without naming *which* sublattice.

## Consequence for cell-similarity search

`CellDatabase.nearest_with_supercells` enumerates *all* index-r sublattices of
the query, reduces each (removing redundancy type 1), looks up each distinct one
in the fast root-invariant index, and reports every database supercell match
against the SPECIFIC integer sublattice matrix M it corresponds to. The
non-uniqueness is surfaced, not hidden: a hit is reported as "db cell = query x M"
for a named M, and different database cells may match different index-r
sublattices of the same query.

Generating and counting the sublattices themselves is `generate_sublattices` /
`sublattice_count` (the A001001 count); classifying the special high-symmetry
coincidences is coincidence-site-lattice (CSL) theory — both are separate from,
and complementary to, the root-form identity test.
