"""
Extended setting notation: a base space group with an *attached change of basis*.

This is the notation used in Zwart, Grosse-Kunstleve & Adams, "Exploring Metric
Symmetry" (IUCr Comp. Comm. Newsletter, 2006): a Hall (or Hermann-Mauguin)
symbol followed by a parenthesised change-of-basis, e.g.

    Hall: I 4 2 3 (y+z,x+z,x+y)
    P21 (2a,a+b,c-a)
    C 2y (x+y,z,x-y)

The parenthesised part lists, comma-separated, the THREE new basis vectors as
linear combinations of the old ones -- i.e. the *columns* of the change-of-basis
matrix P (see agentsg.change_of_basis for the full convention). The linear
part may be spelled with x,y,z or a,b,c; coefficients may be written 2a, 2*x,
-x, 2*x-y, or with fractions: ``a/2``, the grouped half-sum ``(y+z)/2``, or
``1/2*(y+z)``. A constant term is an origin shift and is written in x,y,z
(``x+1/4``) or after a semicolon as in the ITA ``(P, p)`` form
``(a,b,c;1/4,0,0)``. A constant written on a, b, or c is rejected.

Crucially, when det(P) != 1 the transform rescales the lattice, and lattice
translations that were integral in the base setting become fractional
centring translations in the new setting. This module surfaces them: the
operation list of a setting is obtained by transforming the base operators AND
adding the images of the integer lattice under P^-1, then closing the group.
That is how "P1 (2a,2b,2c)"-type notations *introduce* added lattice symmetry.
"""
from __future__ import annotations
import re
from fractions import Fraction as Fr
from itertools import product

from .linalg import Matrix3, Vector3, IDENTITY3
from .symmetry_op import SymmetryOp
from .change_of_basis import ChangeOfBasis
from .space_groups import space_group, SpaceGroup
from .group import close_group


def _is_crystallographic_W(W: Matrix3) -> bool:
    """True if W is an integer matrix with det ±1 (crystallographic rotation)."""
    for row in W.rows:
        for x in row:
            if x.denominator != 1:
                return False
    return W.det() in (1, -1, Fr(1), Fr(-1))


# --- parse one linear-combination field into (coeff_a, coeff_b, coeff_c, const) ---
_LETTER = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}
_NUM_RE = re.compile(r"(\d+)(?:/(\d+))?")
_LETTER_RE = re.compile(r"[abcxyz]")


def _parse_field(field: str) -> tuple[Fr, Fr, Fr, Fr]:
    """Parse one linear combination field into (c_a, c_b, c_c, const).

    Accepts ``2a``, ``2*x``, ``a/2``, ``z+1/2``, grouped ``(y+z)/2``,
    and prefixed ``1/2*(y+z)`` / ``1/2(y+z)``.
    """
    s = field.replace(" ", "")
    if not s:
        raise ValueError("empty change-of-basis field")
    coeffs, const, pos = _parse_sum(s, 0)
    if pos != len(s):
        raise ValueError(f"cannot parse change-of-basis field {field!r} at {s[pos:]!r}")
    return coeffs[0], coeffs[1], coeffs[2], const


def _parse_sum(s: str, pos: int) -> tuple[list[Fr], Fr, int]:
    """Parse a signed sum of atoms until end-of-string or ``)``."""
    coeffs = [Fr(0), Fr(0), Fr(0)]
    const = Fr(0)
    if pos >= len(s) or s[pos] == ")":
        raise ValueError("empty change-of-basis field")
    first = True
    while pos < len(s) and s[pos] != ")":
        sign = 1
        if s[pos] == "+":
            pos += 1
            first = False
        elif s[pos] == "-":
            sign = -1
            pos += 1
            first = False
        elif not first:
            raise ValueError(f"cannot parse change-of-basis field at {s[pos:]!r}")
        else:
            first = False
        if pos >= len(s) or s[pos] == ")":
            raise ValueError("dangling sign in change-of-basis field")
        term_c, term_k, pos = _parse_atom(s, pos)
        for i in range(3):
            coeffs[i] += sign * term_c[i]
        const += sign * term_k
    return coeffs, const, pos


def _parse_atom(s: str, pos: int) -> tuple[list[Fr], Fr, int]:
    """Unsigned atom: number, letter[/n], (sum)[/n], or number[*](sum)[/n]."""
    scale = Fr(1)
    m = _NUM_RE.match(s, pos)
    if m:
        scale = Fr(int(m.group(1)), int(m.group(2)) if m.group(2) else 1)
        pos = m.end()
        if pos < len(s) and s[pos] == "*":
            pos += 1

    if pos < len(s) and s[pos] == "(":
        inner_c, inner_k, pos = _parse_sum(s, pos + 1)
        if pos >= len(s) or s[pos] != ")":
            raise ValueError("unbalanced '(' in change-of-basis field")
        pos += 1
        mden = re.match(r"/(\d+)", s[pos:])
        if mden:
            scale /= int(mden.group(1))
            pos += mden.end()
        return [scale * x for x in inner_c], scale * inner_k, pos

    letter = _LETTER_RE.match(s, pos)
    if letter:
        idx = _LETTER[letter.group(0)]
        pos = letter.end()
        mden = re.match(r"/(\d+)", s[pos:])
        if mden:
            scale /= int(mden.group(1))
            pos += mden.end()
        c = [Fr(0), Fr(0), Fr(0)]
        c[idx] = scale
        return c, Fr(0), pos

    if m:
        return [Fr(0), Fr(0), Fr(0)], scale, pos

    raise ValueError(f"cannot parse change-of-basis field at {s[pos:]!r}")


def _reject_abc_constant(field: str, const: Fr) -> None:
    """Origin shifts stay in x,y,z. a,b,c name the basis, not the origin."""
    if const == 0:
        return
    if re.search(r"[abc]", field.replace(" ", "")):
        raise ValueError(
            f"origin shift in {field!r} must be written in x, y, z "
            f"(or after ';' as in (a,b,c;p1,p2,p3)), not with a, b, c"
        )


def parse_cob(cob: str) -> ChangeOfBasis:
    """Parse a parenthesised change-of-basis string into a ChangeOfBasis(P, p).

    Columns of P are the new basis vectors in old coordinates. Constant terms
    written in x,y,z, or the three numbers after a semicolon, are the origin
    shift p in old fractional coordinates. ``(a,b,c;p1,p2,p3)`` is the ITA
    ``(P, p)`` form.
    """
    inner = cob.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    origin_fields = None
    if ";" in inner:
        basis, origin = inner.split(";", 1)
        origin_fields = [f for f in origin.split(",")]
        if len(origin_fields) != 3:
            raise ValueError(
                f"origin shift after ';' needs 3 comma-separated fields, "
                f"got {len(origin_fields)}: {cob!r}"
            )
        inner = basis
    fields = [f for f in inner.split(",")]
    if len(fields) != 3:
        raise ValueError(f"change-of-basis needs 3 comma-separated fields, got {len(fields)}: {cob!r}")
    P_cols = []
    p = [Fr(0), Fr(0), Fr(0)]
    for j, field in enumerate(fields):
        ca, cb, cc, const = _parse_field(field)
        _reject_abc_constant(field, const)
        P_cols.append((ca, cb, cc))     # column j = (coef of old-a, old-b, old-c)
        p[j] = const
    if origin_fields is not None:
        if any(p):
            raise ValueError(
                "write the origin shift either in the basis fields or after ';', not both"
            )
        for j, field in enumerate(origin_fields):
            ca, cb, cc, const = _parse_field(field)
            _reject_abc_constant(field, const)
            if (ca, cb, cc) != (0, 0, 0):
                raise ValueError(
                    f"origin component {field!r} must be a number, not a basis vector"
                )
            p[j] = const
    # P[i][j] = coefficient of old vector i in new vector j = column j, row i
    P = Matrix3([[P_cols[j][i] for j in range(3)] for i in range(3)])
    return ChangeOfBasis(P, Vector3(tuple(p)))


def _strip_hall(text: str) -> str:
    return re.sub(r"^\s*Hall:\s*", "", text, flags=re.IGNORECASE).strip()


def _split_setting(text: str) -> tuple[str, str | None]:
    """Split ``'F 4 3 2 ((y+z)/2,...)'`` into (base, cob) allowing nested parens."""
    text = _strip_hall(text)
    if not text.endswith(")"):
        return text, None
    depth = 0
    for i in range(len(text) - 1, -1, -1):
        if text[i] == ")":
            depth += 1
        elif text[i] == "(":
            depth -= 1
            if depth == 0:
                cob = text[i:]
                # ``P2(1)`` is a screw index, not a change of basis.
                if "," not in cob:
                    return text, None
                base = text[:i].strip()
                if not base:
                    raise ValueError(
                        "setting is missing a space-group symbol before the change of basis"
                    )
                return base, cob
    raise ValueError(f"unbalanced parentheses in setting {text!r}")


def parse_setting(text: str):
    """Parse '<base> (<cob>)' into (base_key, ChangeOfBasis).

    ``base_key`` is the leading Hall or Hermann-Mauguin string (an optional
    'Hall:' prefix is stripped). A short monoclinic name, a parenthesised
    screw, or an ITA ``number:setting`` code is rewritten to the Hall symbol
    of that setting before lookup. If there is no parenthesised part, the
    whole string is the base and the change of basis is the identity. Nested
    parentheses in the CoB (``(y+z)/2``) are allowed.
    """
    base, cob_txt = _split_setting(text)
    from .ita_settings import canonical_lookup_key
    base = canonical_lookup_key(base)
    if cob_txt is None:
        return base, ChangeOfBasis(IDENTITY3, Vector3((0, 0, 0)))
    return base, parse_cob(cob_txt)


def _lattice_coset_ops(cob: ChangeOfBasis) -> list[SymmetryOp]:
    """Images of the integer lattice under P^-1, reduced mod 1 -- the centring
    translations introduced (or removed) by a det != 1 change of basis."""
    Pinv = cob.P.inverse()
    det = abs(cob.P.det())
    if det == 1:
        return []
    # enumerate integer lattice points in a box large enough to cover one new cell
    seen = set()
    ops = []
    rng = int(det) + 1
    for t in product(range(-rng, rng + 1), repeat=3):
        img = (Pinv @ Vector3((Fr(t[0]), Fr(t[1]), Fr(t[2])))).mod1()
        key = img.v
        if key in seen:
            continue
        seen.add(key)
        ops.append(SymmetryOp(IDENTITY3, img))
    return ops


class SpaceGroupSetting:
    """A base space group viewed through an attached change of basis.

    ``operations()`` returns the closed operation set in the NEW setting,
    including any centring translations surfaced by a det != 1 transform.
    """
    __slots__ = ("base", "cob", "_base_key")

    def __init__(self, base, cob: ChangeOfBasis | None = None):
        if isinstance(base, str):
            self._base_key = base
            self.base = space_group(base)
        elif isinstance(base, SpaceGroup):
            self._base_key = base.hermann_mauguin
            self.base = base
        else:
            raise TypeError("base must be a symbol string or a SpaceGroup")
        self.cob = cob if cob is not None else ChangeOfBasis(IDENTITY3, Vector3((0, 0, 0)))

    @classmethod
    def parse(cls, text: str) -> "SpaceGroupSetting":
        """Parse a setting specification like ``'P 21 21 21 (b,c,a)'`` into a SpaceGroupSetting."""
        base, cob = parse_setting(text)
        return cls(base, cob)

    def operations(self) -> frozenset[SymmetryOp]:
        """Closed operation set in the new setting.

        Requires every transformed rotation to be crystallographic (integer,
        det ±1). Non-unimodular axis scalings such as ``P 4 (2a,b,c)`` do not
        yield a space group on the new basis and raise ``ValueError``.
        Supercells like ``F m -3 m (2a,2b,2c)`` are allowed: ``max_order`` scales
        with ``|det(P)|`` so newly surfaced centring can close.
        """
        base_ops = self.base.operations()
        transformed = [self.cob.apply_to_op(op) for op in base_ops]
        transformed.extend(_lattice_coset_ops(self.cob))
        bad = [op for op in transformed if not _is_crystallographic_W(op.W)]
        if bad:
            raise ValueError(
                f"change of basis {format_cob(self.cob)} produces "
                f"non-crystallographic rotation(s) (non-integer or |det|≠1); "
                f"example W={bad[0].W.rows}. Use a unimodular P, or a pure "
                f"lattice scaling such as (2a,2b,2c) that preserves integer W."
            )
        det = abs(self.cob.P.det())
        # |det| integer ⇒ supercell multiplicity; allow up to 192 × multiplicity.
        mult = int(det) if det.denominator == 1 and det >= 1 else max(1, int(det.numerator) or 1)
        max_order = max(192, 192 * mult)
        return close_group(transformed, max_order=max_order)

    def order(self) -> int:
        """Total number of operations in this setting."""
        return len(self.operations())

    def change_of_basis_matrix(self) -> Matrix3:
        """Matrix P relating new basis vectors to the standard basis: (a',b',c') = (a,b,c) P."""
        return self.cob.P

    def __str__(self) -> str:
        return f"{self.base.hermann_mauguin} {format_cob(self.cob)}"

    def __repr__(self) -> str:
        return f"SpaceGroupSetting(base={self.base.hermann_mauguin!r}, cob={format_cob(self.cob)})"


def _fmt_frac(f: Fr) -> str:
    """Format a Fraction as an exact string without trailing .0."""
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def format_cob(cob: ChangeOfBasis, letters: str = "xyz") -> str:
    """Render a ChangeOfBasis back into the parenthesised column notation."""
    L = letters
    P = cob.P.rows
    p = cob.p.v
    fields = []
    for j in range(3):
        terms = []
        for i in range(3):
            c = P[i][j]
            if c == 0:
                continue
            if c == 1:
                terms.append(("+", L[i]))
            elif c == -1:
                terms.append(("-", L[i]))
            else:
                terms.append(("+" if c > 0 else "-", f"{_fmt_frac(abs(c))}*{L[i]}"))
        if p[j] != 0:
            terms.append(("+" if p[j] > 0 else "-", _fmt_frac(abs(p[j]))))
        if not terms:
            s = "0"
        else:
            s = ""
            for k, (sign, tok) in enumerate(terms):
                if k == 0:
                    s += ("-" if sign == "-" else "") + tok
                else:
                    s += sign + tok
        fields.append(s)
    return "(" + ",".join(fields) + ")"
