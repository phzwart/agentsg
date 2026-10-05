# Space-group diagrams

ITA-style plates are drawn from the derived operations. There is no tabulated
diagram. This needs the `plot` extra (Matplotlib and NumPy). The drawing
modules load only when called, so the rest of the package stays free of that
dependency.

```python
from agentsg.cell import general_position_diagram, ita_plate, symmetry_element_diagram
from agentsg.setting import SpaceGroupSetting

fig = ita_plate(96)
fig = ita_plate(96, legend=True)
fig.savefig("P43212.png", dpi=200)

general_position_diagram(14, ax=ax1)
symmetry_element_diagram(14, ax=ax2, projection="c")

st = SpaceGroupSetting.parse("P 21 21 2 (2a,b-a,c)")
ita_plate(st, legend=True, show_centring=True)
```

`projection` is `"c"` (default), `"a"`, or `"b"`. For a monoclinic group the
standard plate is unique-axis *b* (`projection="b"`).

The general-position diagram replicates one general point through every
operation. The default point is the centre of the largest sphere inscribed in
the asymmetric unit. The symmetry-element diagram classifies each `(W, w)`
and draws the corresponding glyph: rotation axes, screws, rotoinversions,
mirrors, and glides. `full_cell=True` (the default) tiles every copy across
the cell. `symbol_legend` draws the glyph alphabet; `element_legend` draws
only the glyphs present in one group.
