"""
A crystallographic symmetry operation (W, w): x' = W x + w (mod 1).

W is an integer 3x3 matrix (det = +/-1) in a conventional crystallographic
basis, w is an exact rational translation reduced into [0, 1). Composition
and inversion are exact -- no floating point anywhere in this module.
"""
from __future__ import annotations
import re
from fractions import Fraction
from .linalg import Matrix3, Vector3, IDENTITY3, ZERO3, F

_TERM_RE = re.compile(r'([+-]?)(\d*)([xyz])|([+-]?)(\d+)(?:/(\d+))?')


def _parse_component(s: str) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    s = s.replace(' ', '')
    coeffs = {'x': Fraction(0), 'y': Fraction(0), 'z': Fraction(0)}
    const = Fraction(0)
    pos = 0
    for m in _TERM_RE.finditer(s):
        if m.start() != pos:
            raise ValueError(f"could not parse symmetry component {s!r} at {pos}")
        pos = m.end()
        if m.group(3):  # variable term: sign, coef, var
            sign = -1 if m.group(1) == '-' else 1
            coef = int(m.group(2)) if m.group(2) else 1
            coeffs[m.group(3)] += sign * coef
        else:  # constant term: sign, num[/den]
            sign = -1 if m.group(4) == '-' else 1
            num = int(m.group(5))
            den = int(m.group(6)) if m.group(6) else 1
            const += sign * Fraction(num, den)
    if pos != len(s):
        raise ValueError(f"could not parse symmetry component {s!r} (trailing {s[pos:]!r})")
    return coeffs['x'], coeffs['y'], coeffs['z'], const


class SymmetryOp:
    __slots__ = ("W", "w")

    def __init__(self, W: Matrix3, w: Vector3):
        self.W = W
        self.w = w.mod1()

    def __mul__(self, other: "SymmetryOp") -> "SymmetryOp":
        # (W1,w1)*(W2,w2) applied to x: W1(W2 x + w2) + w1 = (W1 W2) x + (W1 w2 + w1)
        return SymmetryOp(self.W @ other.W, (self.W @ other.w) + self.w)

    def inverse(self) -> "SymmetryOp":
        Winv = self.W.inverse()
        return SymmetryOp(Winv, -(Winv @ self.w))

    def __eq__(self, other) -> bool:
        return isinstance(other, SymmetryOp) and self.W == other.W and self.w == other.w

    def __hash__(self) -> int:
        return hash((self.W, self.w))

    def __repr__(self) -> str:
        return self.as_xyz()

    @classmethod
    def identity(cls) -> "SymmetryOp":
        return cls(IDENTITY3, ZERO3)

    @classmethod
    def from_xyz(cls, triplet: str) -> "SymmetryOp":
        comps = triplet.split(',')
        if len(comps) != 3:
            raise ValueError(f"expected 3 comma-separated components, got {triplet!r}")
        rows, consts = [], []
        for c in comps:
            x, y, z, k = _parse_component(c)
            rows.append((x, y, z))
            consts.append(k)
        return cls(Matrix3(rows), Vector3(consts))

    def as_xyz(self) -> str:
        names = ('x', 'y', 'z')
        parts = []
        for row, t in zip(self.W.rows, self.w.v):
            s = ""
            for coef, v in zip(row, names):
                if coef == 0:
                    continue
                if coef == 1:
                    s += f"+{v}"
                elif coef == -1:
                    s += f"-{v}"
                else:
                    s += f"{'+' if coef > 0 else ''}{coef}{v}"
            if t != 0:
                s += f"{'+' if t > 0 else ''}{t}"
            if s.startswith('+'):
                s = s[1:]
            parts.append(s or "0")
        return ",".join(parts)
