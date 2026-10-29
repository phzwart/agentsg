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

## Not pursued

- Entries that cannot be paired with their reference (no shared UniProt accession, 343; too few paired Cα, 143): dropped 2026-10-29, too few entries to matter. Ontology term `missing_mapping`, status `dropped`.
