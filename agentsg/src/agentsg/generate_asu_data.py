"""Write ``asu_data.py`` from the space-group operators.

Nothing here imports gemmi. For each standard setting 1..230:

* The Laue class is the centrosymmetric closure of the point group, named
  from its order and the crystal system.
* The reciprocal ASU is the CCP4 inequality for that Laue class. The two
  settings of ``-3m`` are distinguished by which twofold the Laue group
  contains.
* The direct-space brick is the smallest origin-anchored box on the 1/24
  grid whose images under the group cover the unit cell. Each closed face
  is then opened when the cell is still covered. A face of width 1/6 is
  left closed: on this grid ``x<=1/8`` and ``x<1/6`` are the same points.

    python -m agentsg.generate_asu_data
"""
from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path

from agentsg.group import point_group
from agentsg.linalg import IDENTITY3, ZERO3, Matrix3
from agentsg.space_groups import space_group

_N = 24
_ALLOWED = (3, 4, 6, 8, 12, 16, 18, 24)
_NEG = Matrix3([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
# Twofold of the hexagonal "1 2" / "1 m" setting (P312, P31m, ...).
_TWO_312 = Matrix3([[1, 0, 0], [1, -1, 0], [0, 0, -1]])

# Order of the Laue group within each crystal system.
_LAUE_SYMBOL = {
    ("triclinic", 2): "-1",
    ("monoclinic", 4): "2/m",
    ("orthorhombic", 8): "mmm",
    ("tetragonal", 8): "4/m",
    ("tetragonal", 16): "4/mmm",
    ("trigonal", 6): "-3",
    ("trigonal", 12): "-3m",
    ("hexagonal", 12): "6/m",
    ("hexagonal", 24): "6/mmm",
    ("cubic", 24): "m-3",
    ("cubic", 48): "m-3m",
}

# CCP4 reciprocal-ASU inequality for each Laue class. ``-3m`` has the
# two hexagonal settings.
_RECIPROCAL = {
    "-1": "l>0 or (l=0 and (h>0 or (h=0 and k>=0)))",
    "2/m": "k>=0 and (l>0 or (l=0 and h>=0))",
    "mmm": "h>=0 and k>=0 and l>=0",
    "4/m": "l>=0 and ((h>=0 and k>0) or (h=0 and k=0))",
    "6/m": "l>=0 and ((h>=0 and k>0) or (h=0 and k=0))",
    "4/mmm": "h>=k and k>=0 and l>=0",
    "6/mmm": "h>=k and k>=0 and l>=0",
    "-3": "(h>=0 and k>0) or (h=0 and k=0 and l>=0)",
    "-3m-312": "h>=k and k>=0 and (k>0 or l>=0)",
    "-3m-321": "h>=k and k>=0 and (h>k or l>=0)",
    "m-3": "h>=0 and ((l>=h and k>h) or (l=h and k=h))",
    "m-3m": "k>=l and l>=h and h>=0",
}


def _laue_key(ops, crystal_system: str) -> str:
    rotations = point_group(ops)
    laue = rotations | {_NEG @ W for W in rotations}
    symbol = _LAUE_SYMBOL[(crystal_system, len(laue))]
    if symbol == "-3m":
        return "-3m-312" if _TWO_312 in laue else "-3m-321"
    return symbol


def _scaled_ops(ops):
    """Non-identity operators as (W flattened, translation in 1/24)."""
    scaled = []
    for op in ops:
        if op.W == IDENTITY3 and op.w == ZERO3:
            continue
        rows = op.W.rows
        flat = []
        for i in range(3):
            for j in range(3):
                entry = rows[i][j]
                if entry.denominator != 1:
                    raise ValueError(f"non-integer rotation entry {entry}")
                flat.append(int(entry))
        trans = []
        for component in op.w.v:
            grid = component * _N
            if grid.denominator != 1:
                raise ValueError(f"translation {component} is not a multiple of 1/24")
            trans.append(int(grid))
        scaled.append((flat, trans))
    return scaled


def _covers(size, closed, scaled) -> bool:
    """True when images of the brick cover every point of the 24^3 grid."""
    limits = [size[i] + (1 if closed[i] else 0) for i in range(3)]
    seen = bytearray(_N * _N * _N)
    for w in range(limits[2]):
        for v in range(limits[1]):
            base = (w * _N + v) * _N
            for u in range(limits[0]):
                idx = base + u
                if seen[idx]:
                    continue
                seen[idx] = 1
                for W, t in scaled:
                    x = (W[0] * u + W[1] * v + W[2] * w + t[0]) % _N
                    y = (W[3] * u + W[4] * v + W[5] * w + t[1]) % _N
                    z = (W[6] * u + W[7] * v + W[8] * w + t[2]) % _N
                    seen[(z * _N + y) * _N + x] = 1
    return all(seen)


def direct_brick(ops) -> tuple[tuple[int, int, int], list[bool]]:
    """Smallest covering origin brick, then open any face that still covers.

    Sizes are numerators over 24, chosen from
    ``1/8, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 1``. Among equal volumes the more
    nearly cubic box (smaller sum of edge numerators) wins.
    """
    order = len(ops)
    scaled = _scaled_ops(ops)
    candidates = []
    for a in _ALLOWED:
        for b in _ALLOWED:
            for c in _ALLOWED:
                if a * b * c * order >= _N ** 3:
                    candidates.append((a * b * c, a + b + c, a, b, c))
    candidates.pop()  # the whole cell is the fallback, not a candidate
    candidates.sort(key=lambda item: (item[0], item[1]))
    for _volume, _span, a, b, c in candidates:
        size = (a, b, c)
        closed = [a < _N, b < _N, c < _N]
        if not _covers(size, closed, scaled):
            continue
        for axis, edge in enumerate(size):
            if closed[axis] and edge != 4:
                closed[axis] = False
                if not _covers(size, closed, scaled):
                    closed[axis] = True
        return size, closed
    return (_N, _N, _N), [False, False, False]


def _fraction_text(numerator: int, denominator: int = _N) -> str:
    g = gcd(numerator, denominator)
    numerator //= g
    denominator //= g
    if denominator == 1:
        return str(numerator)
    return f"{numerator}/{denominator}"


def _brick_string(size, closed) -> str:
    parts = []
    for axis, name in enumerate("xyz"):
        relation = "<=" if closed[axis] else "<"
        parts.append(f"0<={name}{relation}{_fraction_text(size[axis])}")
    return "; ".join(parts)


def _fr(numerator: int, denominator: int = _N) -> str:
    text = _fraction_text(numerator, denominator)
    if "/" not in text:
        return f"Fr({text})"
    num, den = text.split("/")
    return f"Fr({num}, {den})"


def collect():
    conditions: list[str] = []
    condition_index: list[int] = []
    laue = [""]
    bricks: list[str] = []
    brick_index: list[int] = []
    bounds: list[tuple[bool, int, bool, int, bool, int]] = []
    for number in range(1, 231):
        group = space_group(number)
        ops = group.operations()
        condition = _RECIPROCAL[_laue_key(ops, group.crystal_system)]
        if condition not in conditions:
            conditions.append(condition)
        condition_index.append(conditions.index(condition))
        laue.append(_laue_symbol(ops, group.crystal_system))
        size, closed = direct_brick(ops)
        text = _brick_string(size, closed)
        if text not in bricks:
            bricks.append(text)
            bounds.append(
                (
                    closed[0], size[0],
                    closed[1], size[1],
                    closed[2], size[2],
                )
            )
        brick_index.append(bricks.index(text))
        if number % 20 == 0:
            print(f"{number}/230", flush=True)
    return conditions, condition_index, laue, bricks, bounds, brick_index


def _laue_symbol(ops, crystal_system: str) -> str:
    key = _laue_key(ops, crystal_system)
    if key.startswith("-3m"):
        return "-3m"
    return key


def _string_tuple(name: str, values: list[str], *, first_comment: str = "") -> str:
    lines = [f"{name}: tuple[str, ...] = ("]
    for i, value in enumerate(values):
        comment = f"  # {first_comment}" if i == 0 and first_comment else ""
        lines.append(f"    {value!r},{comment}")
    lines.append(")")
    return "\n".join(lines)


def _int_tuple(name: str, values: list[int]) -> str:
    body = ", ".join(str(v) for v in values)
    return f"{name}: tuple[int, ...] = ({body})"


def _bounds_tuple(rows) -> str:
    lines = ["ASU_BRICK_BOUNDS: tuple[tuple, ...] = ("]
    for xc, xs, yc, ys, zc, zs in rows:
        lines.append(
            "    ("
            f"{xc}, {_fr(xs)}, {yc}, {_fr(ys)}, {zc}, {_fr(zs)}"
            "),"
        )
    lines.append(")")
    return "\n".join(lines)


def render() -> str:
    conditions, condition_index, laue, bricks, bounds, brick_index = collect()
    parts = [
        '"""Tabulated ASU data (CCP4 convention).',
        "",
        "Generated by ``python -m agentsg.generate_asu_data``. Do not hand-edit.",
        "The generator imports no third-party library. For each standard",
        "setting 1..230 it uses the closed operator list of that group:",
        "",
        "* Laue class. The point group is closed under inversion. The Laue",
        "  symbol is the name of that centrosymmetric group, read off from",
        "  its order and the crystal system (``4/m`` versus ``4/mmm``,",
        "  ``-3`` versus ``-3m``, and so on).",
        "* Reciprocal ASU. Each Laue class has one CCP4 inequality. The two",
        "  settings of ``-3m`` are split by which twofold the Laue group",
        "  contains: the ``P312`` / ``P31m`` twofold selects",
        "  ``k>0 or l>=0``, and the ``P321`` / ``P3m1`` twofold selects",
        "  ``h>k or l>=0``.",
        "* Direct-space brick. Edges are chosen from",
        "  1/8, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 1 on a grid of 24. The box",
        "  starts at the origin. The smallest box whose images under the",
        "  group cover every grid point is kept; equal volumes prefer the",
        "  more nearly cubic box. Each closed face is then opened when the",
        "  cell is still covered. A face of width 1/6 stays closed, because",
        "  on this grid ``x<=1/8`` and ``x<1/6`` are the same points.",
        "",
        "Conditions and bricks are stored once, in the order groups 1..230",
        "first produce them. ``RECIPROCAL_CONDITION_INDEX`` and",
        "``ASU_BRICK_INDEX`` point each group at its row. ``ASU_BRICK_BOUNDS``",
        "is the same brick with ``<=`` closed and ``<`` open.",
        '"""',
        "from __future__ import annotations",
        "from fractions import Fraction as Fr",
        "",
        _string_tuple("RECIPROCAL_CONDITIONS", conditions),
        "",
        "# Per IT number 1..230: index into RECIPROCAL_CONDITIONS",
        _int_tuple("RECIPROCAL_CONDITION_INDEX", condition_index),
        "",
        _string_tuple("LAUE_CLASS", laue, first_comment="1-based"),
        "",
        _string_tuple("ASU_BRICK_STRINGS", bricks),
        "",
        "# (x_closed, x_hi, y_closed, y_hi, z_closed, z_hi) with Fr highs",
        _bounds_tuple(bounds),
        "",
        _int_tuple("ASU_BRICK_INDEX", brick_index),
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name("asu_data.py"),
        help="path to write (default: asu_data.py next to this file)",
    )
    args = parser.parse_args(argv)
    args.output.write_text(render())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
