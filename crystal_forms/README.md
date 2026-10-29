# Crystal-form records

One directory per protein identity for every crystal-form cluster of more than 10 PDB entries. Each record holds, for every entry, the transformation that puts it on the setting and origin of one reference entry, and the all-pairs Cα RMSD within the cluster. No coordinates are stored: an entry in the reference frame is rebuilt from its deposited model and its row of `frames.tsv`.

Produced with agentsg (branch `crystal-form-clusters`, modules `cell.clusters`, `cell.pdb_coords`, `cell.cluster_frames`, `cell.cluster_records`) from the PDB cell snapshot on Zenodo (record 23270215, 208,509 cells, UniProt from SIFTS 2026-10-04) and the deposited mmCIF files from files.rcsb.org (October 2026).

## How the clusters were made

Two entries are joined when they carry the same identity (the exact set of UniProt accessions, or the deposition group of an entry with none), their strain price over the widened Selling closure is at most 14.8 % (Le Page's 3°), and their primitive volumes agree within 7 %. A cluster is a connected component of these joins (single linkage), so it can chain: two members of one cluster may differ by more than 14.8 %. Because edges need the same identity, a wider band only merges crystal forms of the same protein; the volume gate keeps most of those apart.

## How the frames were made

The reference of a cluster is its highest-resolution entry among those whose space group has the cluster's most common IT number. Every other entry with that IT number is placed on it:

- the reindexing operators within Le Page's 3° (strain price 14.8 %), one per class of the reference's point group; every class is scored on the Cα, so an indexing near a higher lattice symmetry is not missed. When none is found, or the best placement is still more than 3 Å from the reference, the classes up to 20 % are scored as well;
- the origin shift that turns the entry's operators into the reference's, solved exactly;
- among the allowed origins, the proper Euclidean normalizer elements and the group operators, the placement with the lowest Cα RMSD, with Cα paired by UniProt residue number (a chain mapped to several accessions is paired one accession at a time, named `chain:accession`).

An entry with no reindexing within 20 % of the reference is placed on an already-placed neighbour and the two frames are composed (`via`). Entries with another IT number are listed and not placed.

## Files

```
index.tsv                          pdb_id, identity_dir, cluster_id, status, reference
run_info.json                      clustering parameters and counts
<identity>/record.json             identity, accessions or deposition group, name, and per cluster:
                                   reference, its resolution, space group and cell, status counts,
                                   RMSD summaries (median, quartiles, 5th/95th percentile, max)
<identity>/cluster_<id>/frames.tsv one row per entry (columns below)
<identity>/cluster_<id>/pairwise.tsv.gz   pdb_a, pdb_b, frame_rmsd, fit_rmsd, n_common
```

Identity directories are the UniProt accession(s) joined by `+`; a long list is shortened to the first two accessions, the count of the rest and a hash. Entries without UniProt use `DEPOSITION_GROUP_<id>`.

## The frame

`L` and `t` in `frames.tsv` map the entry's deposited fractional coordinates onto the reference's fractional coordinates:

$$
x_{\mathrm{ref}} = L\,x + t
$$

`L` is exact: rationals, rows separated by `;`, entries by `,`. `t` is three numbers separated by `,`; components along floating (polar) directions are fitted, the rest are exact allowed-origin values plus the group's translation parts. A frame is defined up to the lattice translations of the reference, including its centring translations.

## frames.tsv columns

| column | meaning |
|---|---|
| pdb_id | entry |
| status | `reference`, `placed`, or why it was not placed: `different_space_group`, `no_shared_accession`, `too_few_paired_atoms`, `no_reindexing_within_band`, `no_reference`, `unreadable_space_group` |
| resolution | high-resolution limit from the mmCIF (Å) |
| sg_hm, a … gamma | space group and cell as deposited |
| L, t | the frame (above) |
| rmsd_to_ref | Cα RMSD to the reference after the frame, no fitting (Å) |
| n_paired | Cα pairs used for `rmsd_to_ref` |
| margin | RMSD of the next distinct placement minus the best (Å); below about 1 Å the coordinates barely decide; a lower bound when above 3 Å |
| asu_rmsd | worst chain of the entry against its nearest reference chain under the group's operators, no fitting (Å) |
| accept | band whose classes produced the chosen placement (0.148 or 0.20) |
| n_classes, class_price | number of reindexing classes tried, strain price of the chosen one |
| via | anchor entry when placed through a neighbour |
| member_chain, ref_chain | chains used for the placement |
| unp_offset | residue-number shift applied to the entry's UniProt numbering |
| acc_fallback | 1 when the entry maps to another accession of the same protein and was paired on residue numbers |
| ambiguous_snap | 1 when the mean shift fell halfway between two allowed origins |

## pairwise.tsv.gz

Every pair of placed entries (and the reference) sharing at least 20 Cα by UniProt residue number. `frame_rmsd` compares the two after their frames with no fitting, each orthogonalised with its own cell written in the reference setting, so it contains the shift of the molecule in the lattice as well as its internal change. `fit_rmsd` is the same residue pairs after a Kabsch superposition. Values in Å, three decimals.

## flags.tsv

One quality flag per placed entry: `pdb_id`, `identity_dir`, `cluster_id`, `rmsd_to_ref` (no fitting), `fit_to_ref` (the superposed RMSD to the reference, from the pairwise table), `margin`, `via`, `flag`.

| flag | rule | count |
|---|---|---|
| `ok` | otherwise | 53,233 |
| `low_margin` | margin below 1 Å: the coordinates barely decide between two placements | 2,219 |
| `far_same_structure` | more than 3 Å from the reference without fitting, below 1.5 Å after superposition: the same structure in another packing. Example: TTK (P33981), I222 with a pseudo-tetragonal cell; after the best indexing (b↔c, 9.4 %) the molecule is still rotated 14.6° about a relative to the reference, which no lattice operator removes | 1,666 |
| `far_mispaired` | more than 3 Å from the reference without fitting and at least 1.5 Å after superposition: the Cα atoms are probably paired wrongly, most likely a residue-numbering mismatch between depositions | 2,839 |

The two `far_` flags are open problems, not settled results; use `ok` entries for anything quantitative.

## Caveats

- The RMSDs use the chain placed on the reference's probe chain (the reference chain with the most mapped residues); other copies in the asymmetric unit enter only through `asu_rmsd`.
- `different_space_group` entries (for example P1 or P2₁ entries inside a P2₁2₁2₁ cluster) are not placed; matching them needs the common subgroup.
- Clustering is single linkage; check `rmsd_to_ref` and the pairwise table before treating a cluster as one crystal form.
