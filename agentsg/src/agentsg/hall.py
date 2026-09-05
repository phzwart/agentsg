"""
Hall space-group symbol parser (Hall, *Acta Cryst.* A37, 517 (1981); ITA Vol. B
1.4 / Vol. A Table A1.4.2.7).

A Hall symbol is the field's compact, unambiguous encoding of a space group's
*generators* -- exactly the input `close_group()` wants. It is the standard
"verified source" alternative to hand-transcribed xyz-triplet tables that the
package's design notes call for.

    <lattice-symbol> <generator>... [ (<origin-shift>) ]

  * lattice symbol: P A B C I R F, optionally prefixed with '-' to add an
    inversion centre at the origin (a centrosymmetric lattice). It fixes the
    centring translations.
  * each generator:  [-] N [axis] [translation-chars] [screw-digit]
        -N           improper (rotoinversion): the rotation matrix is negated.
        N            proper rotation order (1,2,3,4,6).
        axis         x|y|z (principal), ' or " (in-plane 2-folds referred to the
                     preceding axis), * (body diagonal [111]).  If omitted, a
                     default is chosen from position + preceding order (see
                     _default_axis): 1st -> z; 2nd order-2 -> x (after 2/4) or '
                     (after 3/6); 3rd order-3 -> *.
        translation  any of a b c n u v w d (glide/centring fractions), each
                     adding a fixed fraction to the intrinsic translation.
        screw-digit  a digit t for an N_t screw axis: intrinsic translation
                     t/N along the axis (only meaningful for 3- and 6-fold; 2-
                     and 4-fold screws are spelled with translation letters).
  * origin shift (v1 v2 v3) in TWELFTHS: conjugate every generator by the
    change of basis that shifts the origin by (v1,v2,v3)/12.

Everything is exact (fractions.Fraction); there is no fixed denominator.
The reference rotation matrices below were cross-checked, entry for entry,
against an independent Hall-symbol implementation -- not recalled from memory.
"""
from __future__ import annotations
from fractions import Fraction as Fr
from .linalg import Matrix3, Vector3, IDENTITY3
from .symmetry_op import SymmetryOp

# --- centring translations per lattice symbol (fractions of the cell) ---
_H = Fr(1, 2)
_T = Fr(1, 3)
_TT = Fr(2, 3)
LATTICE_CENTERING: dict[str, list[tuple]] = {
    "P": [(0, 0, 0)],
    "A": [(0, 0, 0), (0, _H, _H)],
    "B": [(0, 0, 0), (_H, 0, _H)],
    "C": [(0, 0, 0), (_H, _H, 0)],
    "I": [(0, 0, 0), (_H, _H, _H)],
    "R": [(0, 0, 0), (_TT, _T, _T), (_T, _TT, _TT)],  # obverse (hexagonal axes)
    "F": [(0, 0, 0), (0, _H, _H), (_H, 0, _H), (_H, _H, 0)],
}

# --- reference proper-rotation matrices, by (order, axis) ---
# Verified against gemmi's Hall implementation (test-only oracle).
_REF: dict[tuple[int, str], Matrix3] = {
    (1, "z"): IDENTITY3,
    (2, "x"): Matrix3([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    (2, "y"): Matrix3([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    (2, "z"): Matrix3([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    (3, "x"): Matrix3([[1, 0, 0], [0, 0, -1], [0, 1, -1]]),
    (3, "y"): Matrix3([[-1, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    (3, "z"): Matrix3([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
    (4, "x"): Matrix3([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    (4, "y"): Matrix3([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    (4, "z"): Matrix3([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    (6, "z"): Matrix3([[1, -1, 0], [1, 0, 0], [0, 0, 1]]),
    (3, "*"): Matrix3([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),   # 3-fold about [111]
    # in-plane 2-folds referred to the preceding principal axis:
    (2, "'"): Matrix3([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),  # ' after z
    (2, '"'): Matrix3([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),    # " after z
}
# ' and " referred to a preceding x- or y-axis (used only in cubic 2-fold nets):
_REF[(2, "'x")] = Matrix3([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
_REF[(2, '"x')] = Matrix3([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
_REF[(2, "'y")] = Matrix3([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
_REF[(2, '"y')] = Matrix3([[0, 0, 1], [0, -1, 0], [1, 0, 0]])

# --- translation letters -> fraction vectors ---
_TRANSLATIONS: dict[str, tuple] = {
    "a": (_H, 0, 0), "b": (0, _H, 0), "c": (0, 0, _H),
    "n": (_H, _H, _H),
    "u": (Fr(1, 4), 0, 0), "v": (0, Fr(1, 4), 0), "w": (0, 0, Fr(1, 4)),
    "d": (Fr(1, 4), Fr(1, 4), Fr(1, 4)),
}

# axis unit vectors, for placing screw translations along the axis
_AXIS_VEC = {
    "x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1),
    "*": (1, 1, 1),
}


def _default_axis(position: int, order: int, prev_order: int | None) -> str:
    """Implied axis when a generator omits it (Hall 1981, sec. defaults)."""
    if position == 0:
        return "z"
    if order == 2:
        if prev_order in (2, 4):
            return "x"
        if prev_order in (3, 6):
            return "'"
    if order == 3:
        return "*"       # third generator, body diagonal (cubic)
    return "z"


def _parse_generator(tok: str, position: int, prev_order: int | None):
    """Parse one Hall generator token -> (Matrix3 W, Vector3 w, order)."""
    s = tok
    improper = s.startswith("-")
    if improper:
        s = s[1:]
    if not s or not s[0].isdigit():
        raise ValueError(f"generator token {tok!r} must start with rotation order 1–6")
    order = int(s[0])
    if order not in (1, 2, 3, 4, 6):
        raise ValueError(f"Hall rotation order must be 1,2,3,4, or 6; got {order} in {tok!r}")
    s = s[1:]
    # axis, translation chars, and screw digit may follow, in any order in
    # practice (' " * x y z are axes; a b c n u v w d are translations; a lone
    # digit is a screw). Scan left to right.
    axis = None
    screw = None
    trans_chars: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "xyz*'\"":
            axis = ch
        elif ch in _TRANSLATIONS:
            trans_chars.append(ch)
        elif ch.isdigit():
            screw = int(ch)
        else:
            raise ValueError(f"unrecognised char {ch!r} in Hall generator {tok!r}")
        i += 1

    if axis is None:
        axis = _default_axis(position, order, prev_order)

    # look up the reference matrix; ' and " may be referred to a non-z prev axis
    key: tuple[int, str]
    if axis in ("'", '"') and prev_order is not None and position >= 1:
        # decide reference axis of the preceding generator: default is z, but a
        # preceding generator about x or y changes the referral plane.
        key = (order, axis)  # default (referred to z); cubic x/y handled by caller
    else:
        key = (order, axis)
    try:
        W = _REF[key]
    except KeyError as exc:
        raise ValueError(f"no Hall matrix for order={order} axis={axis!r} in {tok!r}") from exc
    if improper:
        W = Matrix3([[-x for x in row] for row in W.rows])

    # intrinsic translation: sum of translation-letter vectors + screw component
    tx = ty = tz = Fr(0)
    for ch in trans_chars:
        a, b, c = _TRANSLATIONS[ch]
        tx += a; ty += b; tz += c
    if screw is not None:
        av = _AXIS_VEC.get(axis, (0, 0, 1))
        frac = Fr(screw, order)
        tx += frac * av[0]; ty += frac * av[1]; tz += frac * av[2]
    return W, Vector3((tx, ty, tz)), order


def parse_hall(symbol: str) -> tuple[list[SymmetryOp], list[Vector3]]:
    """Parse a Hall symbol into (generators, centering_vectors).

    The returned generators + centering are exactly what `close_group` expects.
    Malformed symbols raise :class:`ValueError` (never raw IndexError/KeyError).
    """
    if not isinstance(symbol, str):
        raise ValueError(f"Hall symbol must be a string, got {type(symbol).__name__}")
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("empty Hall symbol")

    try:
        return _parse_hall_body(symbol)
    except (IndexError, KeyError, AssertionError) as exc:
        raise ValueError(f"malformed Hall symbol {symbol!r}: {exc}") from exc


def _parse_hall_body(symbol: str) -> tuple[list[SymmetryOp], list[Vector3]]:
    """Parse a verified non-empty Hall symbol string into generators and centrings."""
    origin = None
    if "(" in symbol:
        body, _, rest = symbol.partition("(")
        shift_str = rest.rstrip(")").split()
        if len(shift_str) != 3:
            raise ValueError(
                f"origin shift must have 3 integers in twelfths, got {shift_str!r}"
            )
        try:
            origin = tuple(Fr(int(v), 12) for v in shift_str)
        except ValueError as exc:
            raise ValueError(f"origin shift values must be integers, got {shift_str!r}") from exc
        symbol = body.strip()

    parts = symbol.split()
    if not parts:
        raise ValueError("Hall symbol has no lattice letter")
    lat = parts[0]
    centrosymmetric = lat.startswith("-")
    lat_letter = lat.lstrip("-").upper()
    if lat_letter not in LATTICE_CENTERING:
        raise ValueError(f"unknown lattice symbol {lat!r}")
    centering = [Vector3(v) for v in LATTICE_CENTERING[lat_letter]]

    generators: list[SymmetryOp] = []
    prev_order: int | None = None
    prev_axis: str | None = None
    for position, tok in enumerate(parts[1:]):
        # Determine referral for ' and " : if a preceding generator was about
        # x or y, the in-plane 2-fold is referred to that axis.
        improper = tok.startswith("-")
        core = tok[1:] if improper else tok
        if not core or not core[0].isdigit():
            raise ValueError(f"generator token {tok!r} must start with rotation order 1–6")
        axis_hint = None
        for ch in core[1:]:
            if ch in "xyz*'\"":
                axis_hint = ch
                break
        if axis_hint in ("'", '"') and prev_axis in ("x", "y"):
            # rebuild with the referred key
            order = int(core[0])
            try:
                W = _REF[(order, axis_hint + prev_axis)]
            except KeyError as exc:
                raise ValueError(
                    f"no Hall matrix for order={order} axis={axis_hint!r}+{prev_axis!r}"
                ) from exc
            if improper:
                W = Matrix3([[-x for x in r] for r in W.rows])
            # translation chars / screw
            tx = ty = tz = Fr(0)
            for ch in core[1:]:
                if ch in _TRANSLATIONS:
                    a, b, c = _TRANSLATIONS[ch]
                    tx += a; ty += b; tz += c
                elif ch.isdigit():
                    pass  # screw along the ' /" axis: rare
            generators.append(SymmetryOp(W, Vector3((tx, ty, tz))))
            prev_order, prev_axis = order, axis_hint
            continue

        W, w, order = _parse_generator(tok, position, prev_order)
        generators.append(SymmetryOp(W, w))
        prev_order = order
        # record the resolved axis for the next generator's referral logic
        resolved_axis = axis_hint or _default_axis(position, order, prev_order)
        prev_axis = resolved_axis

    if centrosymmetric:
        generators.append(SymmetryOp(Matrix3([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                                     Vector3((0, 0, 0))))

    # apply origin shift by conjugation: (W,w) -> (W, w + (I-W) p)
    if origin is not None:
        p = Vector3(origin)
        shifted = []
        for op in generators:
            # w' = w + p - W p   (origin shift of a Seitz operator)
            Wp = op.W @ p
            neww = op.w + p - Wp
            shifted.append(SymmetryOp(op.W, neww))
        generators = shifted
        # centering vectors are lattice translations: unaffected by origin shift
    return generators, centering


def ops_from_hall(symbol: str) -> "frozenset[SymmetryOp]":
    """Convenience: parse a Hall symbol and return the closed operation set."""
    from .group import close_group
    gens, cent = parse_hall(symbol)
    return close_group(gens, cent)
