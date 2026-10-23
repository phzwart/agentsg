# Changelog

## Unreleased (library)

- `agentsg.cell.strain_closure`: `reindex` prices candidates at the new cell after a common scale fitted in the strain norm (not the Frobenius norm), takes `accept` (the price band) with the reference widened at `band_for(accept)` = accept/(1−accept), and `proper=True` to drop improper candidates; `price_spectrum(ref)` lists the prices of the reference's self-maps; `indexing_classes` takes `anomalous=True` to group by the point group instead of the Laue group, and `proper`; `LEPAGE_PRICE` = 2 × `LEPAGE_STRAIN`. `tier` is renamed `layer` (`tier`, `laue_order`, `laue_in_symmetry` and the `eps` argument of `reindex`/`indexing_classes` stay as deprecated aliases for one release). No HTTP or MCP change.
- `agentsg.cell.strain_closure` computes the widened Selling closure of a measured cell in the strain norm ‖G^-1/2 ΔG G^-1/2‖_F, which is the same in every basis. `widened_closure(cell, sg_hm, eps)` enumerates it exactly (the finite ball of Lemma A6) and reports the tier that holds it: 1 for the 127 superbases of the first adjacency layer, 2 for the 2755 of the second. The default band is Le Page-style, one cell angle changed by 3°, a strain of 0.074. `reindex(reference, new_cell)` returns every candidate operator within the band, exact and ranked by strain price. `sort_spectrum(cell)` gives the strain distance to each of Delone's 24 sorts. `indexing_classes(reference, new_cell, space_group)` groups the candidates into the classes of the reference's Laue group inside the band's (pseudo-)symmetry group, one priced representative per physically distinct indexing (4 for P3, 2 for P3₁21 and P2₁3, 2 for a pseudo-tetragonal P2₁2₁2₁ cell, 1 for P4₃2₁2). Needs numpy and scipy. No HTTP or MCP change.

## 0.3.8

- `pdb_app build-cod` writes a second cell database that holds only Crystallography Open Database entries. That file has no UniProt or deposition-group columns. `add-cod` still inserts those rows into an existing database.
- `python -m agentsg.cell.cod_explorer` opens that archive as three principal-component views of the volume-normalised sorted linear key, with a rotatable 3D panel. Each key is divided by the cube root of the primitive-cell volume. A click shows the deposited cell, the Selling cell, and the chemical formula. The explorer does not read UniProt or deposition-group columns.

## 0.3.7

- The cell database takes Crystallography Open Database entries next to the PDB ones: `CellDatabase.add_cod(path)` reads a COD table export (CSV/TSV) or a tree of COD CIFs and stores each cell as `COD<id>` with `source = 'cod'`, its esds (`sig_a` … `sig_gamma`) and chemical formula; rhombohedral-axes cells (`:R`) are treated as primitive. Existing rows are marked `source = 'pdb'`.
- `pdb_search` hits and `pdb_lookup` report `source` (and `formula` when there is one).

## 0.3.6

- The MX Agents landing page tells people to install the space-group agents locally and point a client at `127.0.0.1`, instead of linking to the public Muse and MCP hosts.

## 0.3.5

- The public sg-muse server starts with no bearer token, the same way sg-mcp does. Compute routes are open, and the rate limit is per client address.
- The MX Agents landing page describes the space-group agents and the unit-cell embedding.

## 0.3.4

- A rebuilt PDB cell database stores two independent annotations: each entry's UniProt accessions, short names, and residue counts from the SIFTS chain table, and its PDB deposition-group id.

## 0.3.3

- The tunnel front door serves the MX Agents landing page on `mxagents.org` and `www.mxagents.org`.

## 0.3.2

- The tunnel front door sends `cell-embed.mxagents.org` to the linear-key explorer on `127.0.0.1:8051`.

## 0.3.1

- Short monoclinic names (`P21/n`, `C2/n`, …), parenthesised screws (`P2(1)/c`), and ITA `number:setting` codes (`14:b2`, `14:c1`, `68:1`) resolve to the tabulated setting. An ambiguous short name assumes unique axis b and reports `assumed`.
- `ita_plate` accepts a bare or parenthesised change of basis together with `sg`, a full `symbol (cob)` setting, the same string in `sg`, and `R` on an R group.
- A symbol in `setting` must be the same IT number as `sg`. `setting="P"` against `C 2 2 21` is an error. A bare lattice letter is not a change of basis.
- `setting` reports `centring_from`, `centring_to`, and `centring_changed` from the translation coset. `added_centering` remains for one release, true only when the new cell has more centring vectors, and is marked deprecated.
- Orthorhombic `projection="all"` panel titles are the ITA symbol of the panel's change of basis, with that cob in the subtitle. An along-b view of C222₁ is A2₁22, not C22₁2.
- Concept-graph anchors were refreshed from this commit.
