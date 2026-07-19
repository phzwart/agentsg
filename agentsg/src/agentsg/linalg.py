"""
Exact-rational linear algebra primitives.

Everything here is built on ``fractions.Fraction`` — arbitrary-precision,
exact, self-reducing. There is no fixed denominator anywhere: a 1/2, a 1/3,
a 1/24, and a 1/60 all just work, and combining them never requires picking
a common base ahead of time. This is the direct fix for cctbx's fixed
translation-base-factor design (``tr_vec`` defaults to a denominator of 12).
"""
from __future__ import annotations
from fractions import Fraction
from typing import Iterable, Tuple

Number = Fraction | int


def F(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


def frac_mod1(x: Fraction) -> Fraction:
    """Reduce a Fraction into [0, 1), exactly."""
    x = F(x)
    return Fraction(x.numerator % x.denominator, x.denominator)


class Vector3:
    __slots__ = ("v",)

    def __init__(self, v: Iterable):
        v = tuple(v)
        assert len(v) == 3
        self.v = tuple(F(x) for x in v)

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(a + b for a, b in zip(self.v, other.v))

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(a - b for a, b in zip(self.v, other.v))

    def __neg__(self) -> "Vector3":
        return Vector3(-a for a in self.v)

    def mod1(self) -> "Vector3":
        return Vector3(frac_mod1(a) for a in self.v)

    def dot(self, other: "Vector3") -> Fraction:
        return sum((a * b for a, b in zip(self.v, other.v)), Fraction(0))

    def __eq__(self, other) -> bool:
        return isinstance(other, Vector3) and self.v == other.v

    def __hash__(self) -> int:
        return hash(self.v)

    def __repr__(self) -> str:
        return f"Vector3{tuple(str(x) for x in self.v)}"


class Matrix3:
    __slots__ = ("rows",)

    def __init__(self, rows: Iterable[Iterable]):
        rows = tuple(tuple(F(x) for x in row) for row in rows)
        assert len(rows) == 3 and all(len(r) == 3 for r in rows)
        self.rows = rows

    def __matmul__(self, other):
        if isinstance(other, Matrix3):
            cols = list(zip(*other.rows))
            return Matrix3(
                [[sum((a * b for a, b in zip(row, col)), Fraction(0)) for col in cols]
                 for row in self.rows]
            )
        if isinstance(other, Vector3):
            return Vector3(
                sum((a * b for a, b in zip(row, other.v)), Fraction(0))
                for row in self.rows
            )
        return NotImplemented

    def transpose(self) -> "Matrix3":
        return Matrix3(zip(*self.rows))

    def det(self) -> Fraction:
        (a, b, c), (d, e, f), (g, h, i) = self.rows
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    def inverse(self) -> "Matrix3":
        (a, b, c), (d, e, f), (g, h, i) = self.rows
        det = self.det()
        if det == 0:
            raise ValueError("Matrix3 is singular")
        cof = Matrix3([
            [(e * i - f * h), -(d * i - f * g), (d * h - e * g)],
            [-(b * i - c * h), (a * i - c * g), -(a * h - b * g)],
            [(b * f - c * e), -(a * f - c * d), (a * e - b * d)],
        ])
        adj = cof.transpose()  # adjugate = transpose of the cofactor matrix
        return Matrix3([[x / det for x in row] for row in adj.rows])

    def __eq__(self, other) -> bool:
        return isinstance(other, Matrix3) and self.rows == other.rows

    def __hash__(self) -> int:
        return hash(self.rows)

    def __repr__(self) -> str:
        return "Matrix3(" + ", ".join(str(r) for r in self.rows) + ")"


IDENTITY3 = Matrix3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
ZERO3 = Vector3([0, 0, 0])
