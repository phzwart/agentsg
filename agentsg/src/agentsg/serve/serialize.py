"""JSON-safe conversions for exact-rational types."""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Sequence

from ..linalg import Matrix3, Vector3
from ..space_groups import space_group
from ..symmetry_op import SymmetryOp


def frac_to_json(x) -> str:
    """Render a rational (or int/float) as an exact string."""
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        return f"{x.numerator}/{x.denominator}"
    if isinstance(x, int):
        return str(x)
    try:
        f = Fraction(x).limit_denominator(10_000)
        return frac_to_json(f)
    except (TypeError, ValueError):
        return str(x)


def parse_frac(raw) -> Fraction:
    """Parse a JSON number or ``'1/4'`` string into a Fraction."""
    if isinstance(raw, Fraction):
        return raw
    if isinstance(raw, (int, float)):
        return Fraction(raw).limit_denominator(10_000)
    s = str(raw).strip()
    if "/" in s:
        n, d = s.split("/", 1)
        return Fraction(int(n), int(d))
    return Fraction(s).limit_denominator(10_000)


def vec_to_json(v: Vector3) -> list[str]:
    """Three rational components as strings."""
    return [frac_to_json(c) for c in v.v]


def matrix_to_json(M: Matrix3) -> list[list[str]]:
    """3×3 rational matrix as nested strings."""
    return [[frac_to_json(x) for x in row] for row in M.rows]


def op_to_xyz(op: SymmetryOp) -> str:
    """Seitz operator as an xyz triplet."""
    return op.as_xyz()


def xyz_to_op(triplet: str) -> SymmetryOp:
    """Parse one xyz triplet."""
    return SymmetryOp.from_xyz(str(triplet).strip())


def parse_xyz_point(raw) -> Vector3:
    """Accept ``[x,y,z]``, ``'x,y,z'``, or a 3-list of rationals."""
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(" ", "").split(",")]
        if len(parts) != 3:
            raise ValueError("xyz: must have three components")
        return Vector3(tuple(parse_frac(p) for p in parts))
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("xyz: must be a list of three numbers or a triplet string")
    return Vector3(tuple(parse_frac(x) for x in raw))


def parse_cell(raw) -> tuple[float, float, float, float, float, float]:
    """Validate six cell parameters (Å, degrees)."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        raise ValueError("cell: must be a list of six numbers (a,b,c,alpha,beta,gamma)")
    return tuple(float(x) for x in raw)


def parse_hkl(raw) -> tuple[int, int, int]:
    """Parse a Miller index triple."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("hkl: must be a list of three integers")
    return int(raw[0]), int(raw[1]), int(raw[2])


def resolve_sg(sg: Any):
    """Resolve number / HM / Hall to a SpaceGroup.

    Missing ``sg`` is 400; an unknown number or symbol is 404.
    """
    from .http import HttpError
    if sg is None:
        raise HttpError(400, "sg: is required (IT number, Hermann-Mauguin, or Hall symbol)")
    if isinstance(sg, str) and sg.strip().isdigit():
        sg = int(sg.strip())
    try:
        return space_group(sg)
    except KeyError as exc:
        raise HttpError(404, str(exc)) from exc


def numpy_vec_to_json(v) -> list[float] | None:
    """Finite float list, or None."""
    if v is None:
        return None
    try:
        return [float(x) for x in v]
    except TypeError:
        return [float(v)]
