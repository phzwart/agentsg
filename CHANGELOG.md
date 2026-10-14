# Changelog

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
