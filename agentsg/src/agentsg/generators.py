"""
Seed generator table.

This is deliberately NOT a full table of all 230 (or 531 with alternate
settings) space groups -- that's phase-2 data-entry work, ideally sourced
from a verified Hall-symbol table rather than transcribed by hand. This is
a representative starter set chosen to exercise the interesting cases:
  - P1        trivial group
  - P-1       inversion
  - P21       screw axis (translation-bearing generator)
  - C2/c      inversion + glide + centering (order 4 x 2 = 8)
  - R3 (hex)  rhombohedral (1/3, 2/3) centering, 3-fold generator
  - Fm-3m     large point group (48) x F-centering (4) = 192, from 3 generators

Every entry is exercised by tests/test_group_closure.py against its known
group order and (for a couple of groups) known reflection conditions.
"""
from __future__ import annotations
from fractions import Fraction as Fr
from .linalg import Vector3
from .symmetry_op import SymmetryOp

_HALF = Fr(1, 2)
_THIRD = Fr(1, 3)
_TWOTHIRD = Fr(2, 3)


def _ops(*xyz_strings: str) -> list[SymmetryOp]:
    return [SymmetryOp.from_xyz(s) for s in xyz_strings]


def _vecs(*triples) -> list[Vector3]:
    return [Vector3(t) for t in triples]


GENERATOR_TABLE = {
    "P1": dict(
        generators=_ops("x,y,z"),
        centering=_vecs((0, 0, 0)),
        expected_order=1,
    ),
    "P-1": dict(
        generators=_ops("-x,-y,-z"),
        centering=_vecs((0, 0, 0)),
        expected_order=2,
    ),
    "P21": dict(
        generators=_ops("-x,y+1/2,-z"),
        centering=_vecs((0, 0, 0)),
        expected_order=2,
    ),
    "C2/c": dict(
        # inversion at origin + c-glide (mirror normal to b, translation c/2);
        # their product reproduces the 2-fold axis displaced off the origin.
        generators=_ops("-x,-y,-z", "x,-y,z+1/2"),
        centering=_vecs((0, 0, 0), (_HALF, _HALF, 0)),
        expected_order=8,
    ),
    "R3_hex": dict(
        # 3-fold about c, hexagonal axes, obverse-setting R-centering
        generators=_ops("-y,x-y,z"),
        centering=_vecs((0, 0, 0), (_TWOTHIRD, _THIRD, _THIRD), (_THIRD, _TWOTHIRD, _TWOTHIRD)),
        expected_order=9,
    ),
    "Fm-3m": dict(
        # 4-fold about c, 3-fold about [111], inversion; F-centering
        generators=_ops("-y,x,z", "z,x,y", "-x,-y,-z"),
        centering=_vecs((0, 0, 0), (0, _HALF, _HALF), (_HALF, 0, _HALF), (_HALF, _HALF, 0)),
        expected_order=192,
    ),
}
