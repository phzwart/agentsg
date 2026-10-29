# Crystal-form ledger

Open questions about the crystal-form clusters and their records (branch `crystal-form-records`) that still need an answer. Each item says what is known, what has to be decided, and how it could be done. The issue ontology (`agentsg/src/agentsg/cell/data/alignment_issues.yaml`) points here for terms whose status is `deferred`.

## L1. Entries in another space-group type than their cluster reference

Status: deferred (2026-10-29).

1,257 entries in 321 clusters carry a space-group type different from the reference of their cluster and are left unplaced (`status = different_space_group` in `frames.tsv`). The reference is the highest-resolution entry of the cluster's most common IT number, so these entries disagree with the majority.

| relation to the reference group | entries |
|---|---|
| fewer operators (a subgroup, e.g. P1 in a P2₁ cluster) | 762 |
| more operators (e.g. I222 entries in an I2 cluster) | 323 |
| same number of operators, other type (e.g. P2₁2₁2 against P2₁2₁2₁) | 172 |

Most common pairs (entry group in a cluster of reference group): P2₁ in P2₁2₁2₁ (174), P2₁ in C222₁ (97), P1 in C2 (63), I222 in I2 (58), P3₁21 in P3₁ (50), C2 in I222 (44), P2₁ in P2₁2₁2 (43), P2₁2₁2 in P2₁2₁2₁ (36), P1 in P2₁ (35), P6₁22 in P6₁ (29).

To answer:

- For each entry, is the difference real (a genuinely lower or higher symmetry, a different crystal form on the same lattice), a choice of the depositor (data processed in a subgroup, twinning), or a mis-assignment (a screw axis read as a rotation axis)?
- When the entries of a cluster split into two groups, which should be the reference group? The 323 entries with more operators than the reference suggest that in some clusters the majority group is the lower one.

How it could be done: place each entry on the reference in the maximal common subgroup of the two groups. The reference is expanded from its asymmetric unit to the subgroup's, the reindexing coset and allowed origins are taken in the subgroup, and the Cα RMSD decides as now. A placement whose extra copies coincide with the reference's symmetry mates within the noise says the higher group holds; copies that do not coincide say the lower group is real. agentsg has the subgroup graph (`subgroup_graph`) and the normalizer for any group, so no new crystallographic machinery is needed.

## L2. Residue-level comparison within a crystal form

Status: deferred (2026-10-29). Ontology terms `sequence_variant`, `altloc_label_permutation`, `resolution_dependent_modelling`.

Whole-chain Cα placement is not affected, but comparing entries residue by residue mixes three things into the differences:

- point variants filed under one accession (carbonic anhydrase II, cluster 1: Phe131 is Val in 116 entries, also Ala, Tyr, Leu, Cys; His64 is Ala, Cys, Tyr or Trp in 28; Thr200 and Val143 variants);
- alternate-location letters, which are arbitrary between entries (His64 has altlocs in 487 entries, and the letters agree with the geometry 53 % of the time; 50-71 % at Glu14, Asp175, Ser217, Ser220);
- whether a minor conformer is modelled at all, which depends on resolution (His64: both states modelled in about half the entries below 1.6 Å and in 9 % above 2 Å).

To answer: what a residue-level record needs so that these are separated, and how the statistics of each position should treat variant entries, altlocs and low-resolution single-state models.

How it could be done: extend the coordinate cache from first-altloc Cα to all atoms with residue names, altloc labels and occupancies; match conformers between entries by geometry (optimal assignment on RMSD), never by letter; leave variant entries out of a position's statistics; treat single-state models at low resolution as censored, or stratify by resolution.

## L3. Conformer populations and the thermal state

Status: deferred (2026-10-29). Ontology terms `discrete_conformer_population`, `thermal_state`.

Some residues switch between the same few discrete states in every crystal of a form, and the crystals differ in how the states are populated (carbonic anhydrase II, cluster 1: His64 in/out, 3.0 Å apart, pair RMSD quartiles 0.43 and 2.97 Å; also Glu14, Asp52, Asp165, Asp175, Asn253, Glu239, Thr37, Asn67). Room-temperature and cryogenic crystals form separate cell populations (median c 72.91 Å at room temperature, 72.10 Å cryogenic), and the populations shift with temperature (His64 'in' fraction 0.45 at room temperature and 0.26 cryogenic, entries at 1.6-2.0 Å).

To answer: for each multi-state position, the states and the population of each in each entry, and how the populations depend on temperature, resolution and ligand.

How it could be done: classify each entry's conformers into the states found across the form (needs the L2 record), model the populations within resolution strata, and stratify by `_diffrn.ambient_temp`, which the coordinate fetch does not yet keep.

## L4. Coherent sub-ångström deformation

Status: deferred (2026-10-29). Ontology term `coherent_subangstrom_deformation`.

Between crystals of one form, displacements of a few tenths of an Å are coherent over the molecule: fitted on any five-residue window, residues away from the window deviate by two to five times what rigid attachment to that window would give (carbonic anhydrase II, cluster 1; the His64 backbone and the 170s and 230s loops move together). No segment moves as a rigid body at the precision of the models.

To answer: whether the deformation is a few collective modes, and whether those modes follow the cell (lattice strain), the temperature, the ligand, or the conformer populations of L3.

How it could be done: decompose the residue displacements of all entries in the common frame (no fitting, from the existing frames) into modes, and regress the mode amplitudes on cell parameters, temperature, resolution and state populations.

## L5. Renumbering that no constant shift fixes

Status: deferred (2026-10-29). Ontology term `nonuniform_renumbering`.

`far_different_structure` (1,444 entries) mixes real conformational differences with numbering that differs by insertions or deletions along the chain, which the register search (a constant shift) cannot fix.

To answer: how many `far_different_structure` entries are renumbering, and how many are real conformational differences.

How it could be done: align residue sequences (needs residue names, as in L2) and repeat the pairing; entries whose superposed RMSD then falls below 1.5 Å were renumbering.

## Not pursued

- Entries that cannot be paired with their reference (no shared UniProt accession, 343; too few paired Cα, 143): dropped 2026-10-29, too few entries to matter. Ontology term `missing_mapping`, status `dropped`.
