# Changelog

## 0.3.1

- Short monoclinic names (`P21/n`, `C2/n`, …), parenthesised screws (`P2(1)/c`), and ITA `number:setting` codes (`14:b2`, `14:c1`, `68:1`) resolve to the tabulated setting. An ambiguous short name assumes unique axis b and reports `assumed`.
- `ita_plate` accepts a bare or parenthesised change of basis together with `sg`, a full `symbol (cob)` setting, the same string in `sg`, and `R` on an R group.
- A symbol in `setting` must be the same IT number as `sg`. `setting="P"` against `C 2 2 21` is an error. A bare lattice letter is not a change of basis.
- `setting` reports `centring_from`, `centring_to`, and `centring_changed` from the translation coset. `added_centering` remains for one release, true only when the new cell has more centring vectors, and is marked deprecated.
- Orthorhombic `projection="all"` panel titles are the ITA symbol of the panel's change of basis, with that cob in the subtitle. An along-b view of C222₁ is A2₁22, not C22₁2.
- Concept-graph anchors were refreshed from this commit.
