# Space groups

All 230 groups are reproduced from their Hall symbols. The table in
`space_groups.py` stores the number, Hermann–Mauguin symbol, Hall symbol, and
crystal system. Generators are parsed and closed on demand. No operation lists
are stored.

```python
from fractions import Fraction as Fr

from agentsg import space_group, ops_from_hall
from agentsg.linalg import Vector3
from agentsg.reflections import reflection_conditions
from agentsg.wyckoff import multiplicity, site_symmetry_order

sg = space_group(225)            # or "Fm-3m", or "F 4 2 3"
ops = sg.operations()            # closed, exact operation set
print(sg, "order", sg.order())   # 192

print(reflection_conditions(list(ops)))

x = Vector3((Fr(1, 4), Fr(1, 4), Fr(1, 4)))
print(multiplicity(x, list(ops)), site_symmetry_order(x, list(ops)))

print(len(ops_from_hall("-P 2ybc")))   # P2_1/c, order 4
```

Reflection conditions are the stabiliser strata of reciprocal space. The
condition on each class is a sublattice. They are not read from a table of
systematic absences.

## What is derived

Derived at runtime from the operator list: general and special positions,
multiplicities, site symmetry, the point group, Bravais centring, reflection
conditions, and, through the metric bridge, crystal-system cell restrictions.

The 230 standard-setting Hall and Hermann–Mauguin symbols are embedded literal
data. Wyckoff letters are not reproduced.

## Settings and change of basis

A setting is a base symbol plus an attached change of basis. The parenthesised
part lists the three new basis vectors as combinations of the old ones (the
columns of `P`). Letters may be spelled `x,y,z` or `a,b,c`.

```python
from agentsg import SpaceGroupSetting

s = SpaceGroupSetting.parse("Hall: I 4 2 3 (y+z,x+z,x+y)")
s.change_of_basis_matrix().det()   # 2: the transform doubles the cell
s.order()                          # 96: base I432 gains I-centring
```

When `det(P)` is not 1, lattice translations that were integral in the base
setting appear as fractional centring translations in the new setting.

## Identify, semi-invariants, Harker sections

`identify_space_group` recovers the International Tables type, including when
the input differs from the standard setting by an origin shift only. Origin
recovery solves `(W − I) p = Δw` exactly over the rationals.

```python
from agentsg import (
    equivalent_reflections,
    harker_sections,
    identify_space_group,
    phase_restriction,
    semi_invariants,
    space_group,
)
from agentsg.linalg import Vector3

ops = space_group(19).operations()
hit = identify_space_group(ops)
hit.number, hit.hall, hit.floating_origin

semi_invariants(ops)
phase_restriction(Vector3((1, 2, 3)), ops)
equivalent_reflections(Vector3((1, 2, 3)), ops)
harker_sections(ops)
```

A semi-invariant modulus of 0 is a continuous floating origin along that
direction. Recovered origin shifts are pinned so the free components are zero.
Harker sections are the left-nullspace constraints of `(I − W)`.

## Asymmetric units

```python
from agentsg import DirectAsuBrick, ReciprocalAsu, laue_class, optimize_asu, space_group
from agentsg.cell.metric import UnitCell

rasu = ReciprocalAsu.from_space_group(19)
hkl, isym = rasu.to_asu((3, 2, 1), space_group(19).operations())

brick = DirectAsuBrick.from_space_group(19)
brick.contains((0.1, 0.2, 0.3))

laue_class(225)   # 'm-3m'

opt = optimize_asu(space_group(4).operations(), UnitCell(5, 5, 40, 90, 90, 90))
opt.score, opt.origin_shift
```

`ReciprocalAsu` and `DirectAsuBrick` follow the CCP4 conventional bricks.
`optimize_asu` searches origin gauges of a metric Dirichlet domain. Full
International Tables polyhedral inequality galleries are out of scope.
