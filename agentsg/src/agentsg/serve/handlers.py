"""Request handlers: wrap the public agentsg API as JSON dicts."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from .. import (
    ReciprocalAsu,
    equivalent_reflections,
    discrete_allowed_origins,
    floating_origin_basis,
    harker_sections,
    identify_space_group,
    is_systematically_absent,
    lattice_symmetry,
    laue_class,
    multiplicity,
    phase_restriction,
    point_group,
    reflection_conditions,
    site_symmetry_order,
    orbit,
)
from ..cell import (
    UnitCell,
    compare_cells,
    niggli_reduce,
    primitive_cell,
    root_distance,
    root_invariant,
    root_volume_decomposition,
    sorted_linear_key,
    similarity_distance,
    similarity_invariant,
    surface_geometric_operators,
)
from ..cell.pdb_server import search_compatible
from ..tolerances import (
    BOUNDARY_REL,
    COB_ANGLE_SIGMA_DEG,
    COB_ANGLE_TOL_DEG,
    COB_LENGTH_TOL_PCT,
    COMPARE_ANGLE_TOL_DEG,
    COMPARE_LENGTH_TOL_PCT,
    LE_PAGE_MAX_DELTA_DEG,
    METRIC_ANGLE_TOL_DEG,
    METRIC_LENGTH_TOL_PCT,
)
from ..cell.primitive import lattice_letter
from ..group import centering_translations, close_group
from ..linalg import ZERO3
from ..setting import SpaceGroupSetting, format_cob
from .http import HttpError
from .serialize import (
    frac_to_json,
    matrix_to_json,
    numpy_vec_to_json,
    op_to_xyz,
    parse_cell,
    parse_hkl,
    parse_xyz_point,
    resolve_sg,
    vec_to_json,
    xyz_to_op,
)


# Concept ids each tool exercises. Every id must exist in the concept payload.
TOOL_CONCEPTS: dict[str, list[str]] = {
    "space_group_info": [
        "space_group", "hall_symbol", "hermann_mauguin_symbol", "point_group",
        "reflection_conditions",
    ],
    "reflections_info": [
        "reflection_conditions", "systematic_absences", "centric_reflection",
        "epsilon_factor", "equivalent_reflections",
    ],
    "harker_info": ["harker_section", "symmetry_operation"],
    "allowed_origins_info": ["allowed_origins", "floating_origin"],
    "normalizer_info": [
        "euclidean_normalizer", "euclidean_normalizer_linear", "allowed_origins",
        "floating_origin", "numeric_gate",
    ],
    "match_models_info": [
        "model_matching", "euclidean_normalizer", "allowed_origins",
        "floating_origin", "torus_distance",
    ],
    "site_info": ["site_symmetry", "wyckoff_position", "crystallographic_orbit"],
    "subgroups_info": ["subgroup", "t_subgroup", "k_subgroup"],
    "lattice_symmetry_info": [
        "lattice_symmetry_determination", "holohedry", "bravais_lattice", "metric_tensor",
        "numeric_gate",
    ],
    "compare_info": [
        "kurlin_root_form", "selling_closure", "tolerance_gated_matching", "sublattice",
        "numeric_gate",
    ],
    "reindex_info": [
        "reindexing", "indexing_ambiguity", "twinning_merohedry", "reference_orbit",
        "numeric_gate",
    ],
    "pdb_search": [
        "pdb_lattice_search", "sorted_linear_key", "kurlin_root_form", "kd_tree", "primitive_cell",
        "reference_orbit", "selling_closure", "tolerance_gated_matching", "numeric_gate",
    ],
    "ita_plate_json": [
        "ita_diagrams", "ita_graphical_symbols", "projection_convention", "general_position",
    ],
}


def _with_concepts(tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["concepts"] = list(TOOL_CONCEPTS[tool])
    return out


def _sg_payload(rec) -> dict[str, Any]:
    ops = list(rec.operations())
    pg = point_group(ops)
    return {
        "sg_number": rec.number,
        "sg_hm": rec.hermann_mauguin,
        "hall": rec.hall,
        "crystal_system": rec.crystal_system,
        "centering": lattice_letter(rec.hermann_mauguin),
        "order": rec.order(),
        "point_group_order": len(pg),
        "laue_class": laue_class(rec.number),
        "ops": [op_to_xyz(op) for op in sorted(ops, key=lambda o: o.as_xyz())],
        "reflection_conditions": reflection_conditions(ops),
    }


def space_group_info(data: dict[str, Any]) -> dict[str, Any]:
    rec = resolve_sg(data.get("sg"))
    return _with_concepts("space_group_info", _sg_payload(rec))


def setting_info(data: dict[str, Any]) -> dict[str, Any]:
    text = data.get("setting")
    if not text:
        raise ValueError("setting is required (e.g. 'P 21 21 2 (2a,b-a,c)')")
    st = SpaceGroupSetting.parse(str(text))
    P = st.change_of_basis_matrix()
    det = P.det()
    ops = list(st.operations())
    return {
        "setting": str(st),
        "base_sg_number": st.base.number,
        "base_sg_hm": st.base.hermann_mauguin,
        "cob": format_cob(st.cob),
        "P": matrix_to_json(P),
        "det": frac_to_json(det),
        "added_centering": abs(int(det)) != 1 if det.denominator == 1 else True,
        "order": st.order(),
        "ops": [op_to_xyz(op) for op in sorted(ops, key=lambda o: o.as_xyz())],
    }


def identify_ops(data: dict[str, Any]) -> dict[str, Any]:
    raw = data.get("ops")
    if not isinstance(raw, list) or not raw:
        raise ValueError("ops must be a non-empty list of xyz triplets")
    ops = [xyz_to_op(s) for s in raw]
    hit = identify_space_group(ops)
    if hit is None:
        raise HttpError(404, "could not identify a standard space group from these operators")
    ops_set = frozenset(ops)
    closed = close_group(list(ops_set), list(centering_translations(ops_set)) or [ZERO3])
    det = hit.change_of_basis.P.det()
    out = {
        "sg_number": hit.number,
        "sg_hm": hit.hermann_mauguin,
        "hall": hit.hall,
        "cob": format_cob(hit.change_of_basis),
        "P": matrix_to_json(hit.change_of_basis.P),
        "origin": vec_to_json(hit.change_of_basis.p),
        "floating_origin": [vec_to_json(v) for v in hit.floating_origin],
        "input_order": len(closed),
        "matched_order": hit.space_group.order(),
        "det": frac_to_json(det),
    }
    if det.denominator == 1 and abs(int(det)) > 1:
        out["note"] = (
            f"Input closes to {len(closed)} operators; matched "
            f"{hit.hermann_mauguin} (#{hit.number}, order "
            f"{hit.space_group.order()}) after a det-{abs(int(det))} change of "
            "basis to the conventional cell. That is the space-group type, "
            "not the primitive IT number with the same operator count."
        )
    return out


def site_info(data: dict[str, Any]) -> dict[str, Any]:
    rec = resolve_sg(data.get("sg"))
    xyz = parse_xyz_point(data.get("xyz"))
    ops = list(rec.operations())
    pts = orbit(xyz, ops)
    return _with_concepts("site_info", {
        "sg_number": rec.number,
        "sg_hm": rec.hermann_mauguin,
        "xyz": vec_to_json(xyz),
        "multiplicity": multiplicity(xyz, ops),
        "site_symmetry_order": site_symmetry_order(xyz, ops),
        "orbit": sorted(vec_to_json(p) for p in pts),
        "wyckoff_letter": None,
        "note": "ITA Wyckoff letters are not assigned; numeric orbit content only.",
    })


def reflections_info(data: dict[str, Any]) -> dict[str, Any]:
    rec = resolve_sg(data.get("sg"))
    ops = list(rec.operations())
    out: dict[str, Any] = {
        "sg_number": rec.number,
        "sg_hm": rec.hermann_mauguin,
        "conditions": reflection_conditions(ops),
    }
    if data.get("hkl") is not None:
        hkl = parse_hkl(data["hkl"])
        from ..linalg import Vector3
        hv = Vector3(hkl)
        pr = phase_restriction(hv, ops)
        eq = equivalent_reflections(hv, ops)
        rasu = ReciprocalAsu.from_space_group(rec.number)
        out["hkl"] = list(hkl)
        out["absent"] = bool(pr.absent) or is_systematically_absent(hv, ops)
        out["centric"] = bool(pr.centric)
        out["phase"] = None if pr.phase is None else frac_to_json(pr.phase)
        out["multiplicity"] = eq.multiplicity
        out["epsilon"] = eq.epsilon
        out["laue_multiplicity"] = eq.laue_multiplicity
        out["equivalent_hkls"] = [list(t) for t in eq.hkls]
        out["in_reciprocal_asu"] = rasu.is_in(hkl)
        out["asu_condition"] = rasu.condition_str
    return _with_concepts("reflections_info", out)


def harker_info(data: dict[str, Any]) -> dict[str, Any]:
    rec = resolve_sg(data.get("sg"))
    loci = harker_sections(rec.operations())
    return _with_concepts("harker_info", {
        "sg_number": rec.number,
        "sg_hm": rec.hermann_mauguin,
        "loci": [
            {
                "kind": loc.kind,
                "rank": loc.rank,
                "constraints": [str(c) for c in loc.constraints],
            }
            for loc in loci
        ],
    })


def allowed_origins_info(data: dict[str, Any]) -> dict[str, Any]:
    """Discrete alternative origins, with floating directions pinned to zero."""
    rec = resolve_sg(data.get("sg"))
    ops = list(rec.operations())
    origins = discrete_allowed_origins(ops)
    floating = floating_origin_basis(ops)
    return _with_concepts("allowed_origins_info", {
        "sg_number": rec.number,
        "sg_hm": rec.hermann_mauguin,
        "n_origins": len(origins),
        "origins": [vec_to_json(o) for o in origins],
        "floating_origin": [vec_to_json(v) for v in floating],
    })


def _candidate_json(item) -> dict[str, Any]:
    return {
        "reindex": matrix_to_json(item.reindex),
        "operator": item.operator.as_xyz(),
        "discrete_origin": vec_to_json(item.discrete_origin),
        "floating_shift": [frac_to_json(component) for component in item.floating_shift.v],
        "rmsd": item.rmsd,
        "snap_residual": list(item.snap_residual),
        "ambiguous_snap": item.ambiguous_snap,
        "det": item.det,
    }


def normalizer_info(data: dict[str, Any]) -> dict[str, Any]:
    """Euclidean normalizer. JSON follows the allowed-origins vectors."""
    from ..normalizer import euclidean_normalizer
    rec = resolve_sg(data.get("sg"))
    cell = None if data.get("cell") is None else parse_cell(data.get("cell"))
    length_tol = float(data.get("length_tol_pct", METRIC_LENGTH_TOL_PCT))
    angle_tol = float(data.get("angle_tol_deg", METRIC_ANGLE_TOL_DEG))
    norm = euclidean_normalizer(
        rec, cell, length_tol_pct=length_tol, angle_tol_deg=angle_tol,
    )
    return _with_concepts("normalizer_info", {
        "sg_number": rec.number,
        "sg_hm": rec.hermann_mauguin,
        "metric_specialized": norm.metric_specialized,
        "index": norm.index(),
        "n_linear": len(norm.linear_reps),
        "n_origins": norm.origin_lattice.n_alternative_origins,
        "origins": [vec_to_json(origin) for origin in norm.origin_lattice.discrete_origins()],
        "floating_origin": [vec_to_json(vector) for vector in norm.origin_lattice.floating],
        "linear_reps": [
            {"M": matrix_to_json(matrix), "m": vec_to_json(shift), "det": det}
            for matrix, shift, det in norm.linear_reps
        ],
        "rejected": [
            {"M": matrix_to_json(matrix), "reason": reason}
            for matrix, reason in norm.rejected
        ],
        "gates": norm.gates,
        "note": (
            "det -1 maps a chiral molecule to its mirror image. "
            "The affine normalizer is not computed, and Wyckoff letters are not assigned."
        ),
    })


def match_models_info(data: dict[str, Any]) -> dict[str, Any]:
    """Match two corresponding models. Arrays, or two PDB paths when gemmi is installed."""
    from ..match_models import match_models, match_models_pdb
    allow = data.get("allow_improper", False)
    if isinstance(allow, str):
        allow = allow.lower() in ("1", "true", "yes")
    cartesian = data.get("cartesian", False)
    if isinstance(cartesian, str):
        cartesian = cartesian.lower() in ("1", "true", "yes")
    chain_map = None
    if data.get("pdb_a") or data.get("pdb_b"):
        if not data.get("pdb_a") or not data.get("pdb_b"):
            raise ValueError("pdb_a and pdb_b are both required")
        result, chain_map = match_models_pdb(
            data["pdb_a"], data["pdb_b"], data.get("sg"),
            cell_a=data.get("cell_a"), cell_b=data.get("cell_b"),
            allow_improper=bool(allow),
        )
        rec = resolve_sg(data.get("sg"))
    else:
        if data.get("xyz_a") is None or data.get("xyz_b") is None:
            raise ValueError("xyz_a and xyz_b are required when pdb paths are omitted")
        rec = resolve_sg(data.get("sg"))
        result = match_models(
            data["xyz_a"], data["xyz_b"], rec, data.get("cell_a"),
            cell_b=data.get("cell_b"), allow_improper=bool(allow),
            cartesian=bool(cartesian),
        )
    payload = {
        "sg_number": rec.number,
        "sg_hm": rec.hermann_mauguin,
        "n_atoms": result.n_atoms,
        "enantiomorph_flag": result.enantiomorph_flag,
        "best": _candidate_json(result.best),
        "ranked": [_candidate_json(item) for item in result.ranked],
        "note": (
            "det -1 maps a chiral molecule to its mirror image. "
            "enantiomorph_flag is true when such a representative fits better "
            "than every proper one."
        ),
    }
    if chain_map is not None:
        payload["chain_map"] = [[left, right] for left, right in chain_map]
    return _with_concepts("match_models_info", payload)


def subgroups_info(data: dict[str, Any]) -> dict[str, Any]:
    from ..subgroups import subgroup_graph
    rec = resolve_sg(data.get("sg"))
    kind = str(data.get("kind") or "both").lower()
    if kind not in ("t", "k", "both"):
        raise ValueError("kind: must be t, k, or both")
    maximal = data.get("maximal", True)
    if isinstance(maximal, str):
        maximal = maximal.lower() in ("1", "true", "yes")
    return _with_concepts("subgroups_info", subgroup_graph(rec, kind=kind, maximal=bool(maximal)))


def default_projection(crystal_system: str | None, sg=None) -> str:
    """Project a monoclinic group along its unique axis, else along c.

    An explicit ``projection`` argument still wins. A monoclinic setting whose
    2-fold has moved off b is drawn down that axis.
    """
    if crystal_system and str(crystal_system).lower().startswith("monoclinic"):
        axis = _unique_axis(sg) if sg is not None else None
        return axis or "b"
    return "c"


def _unique_axis(sg) -> str | None:
    """Cell axis of the monoclinic 2-fold, or None when it is not along a, b, c."""
    try:
        from ..cell.diagrams import _element_copies
        els = _element_copies(sg)
    except Exception:
        return None
    for el in els:
        if el.get("type") not in ("rotation", "screw") or el.get("order") != 2:
            continue
        ax = el.get("axis_exact")
        if not ax:
            continue
        nz = [i for i, c in enumerate(ax) if c]
        if len(nz) == 1 and abs(ax[nz[0]]) == 1:
            return "abc"[nz[0]]
    return None


# Hexagonal-to-rhombohedral change of basis (obverse). Columns are the
# rhombohedral axes in the hexagonal basis.
_RHOMBO_GROUPS = {146, 148, 155, 160, 161, 166, 167}
_RHOMBO_COB = "((2a+b+c)/3,(-a+b+c)/3,(-a-2b+c)/3)"


def resolve_plate_sg(data: dict[str, Any]):
    """SpaceGroup or SpaceGroupSetting for plate / classify."""
    setting = data.get("setting")
    if isinstance(setting, str) and setting.strip().lower() in ("r", "rhombohedral"):
        from ..setting import parse_cob
        sg = resolve_sg(data.get("sg"))
        if getattr(sg, "number", None) not in _RHOMBO_GROUPS:
            raise ValueError(
                "setting R applies to the rhombohedral groups "
                "146, 148, 155, 160, 161, 166 and 167"
            )
        return SpaceGroupSetting(sg, parse_cob(_RHOMBO_COB))
    if setting:
        return SpaceGroupSetting.parse(str(setting))
    return resolve_sg(data.get("sg"))


def _sg_number(sg):
    num = getattr(sg, "number", None)
    if num is None and hasattr(sg, "base"):
        num = getattr(sg.base, "number", None)
    return num


def _plate_note(n_elements, symbols, projection: str) -> str:
    shown = ", ".join(symbols) if symbols else "none"
    return (
        f"Projection along {projection}. {n_elements} symmetry elements "
        f"are drawn ({shown}). A height is printed when it is not 0. "
        f"The partners 1/2 with 0, and 3/4 with 1/4, are left implicit. "
        f"Wyckoff letters are not assigned."
    )


def ita_plate_json(data: dict[str, Any], *, png_query: str) -> dict[str, Any]:
    sg = resolve_plate_sg(data)
    system = getattr(sg, "crystal_system", None)
    if system is None and hasattr(sg, "base"):
        system = getattr(sg.base, "crystal_system", None)
    requested = str(data.get("projection") or default_projection(system, sg))
    if requested == "all":
        # The element list stays the default single projection. The PNG is
        # the multi-panel figure.
        projection = default_projection(system, sg)
    elif requested not in ("a", "b", "c", "111"):
        raise ValueError("projection must be a, b, c, 111, or all")
    else:
        projection = requested
    legend = _as_bool(data.get("legend", False))
    show_centring = _as_bool(data.get("show_centring", False))
    compact = _as_bool(data.get("compact", False))
    try:
        from ..cell.diagrams import _element_copies
        raw = _element_copies(sg)
    except ImportError as exc:
        raise HttpError(503, "ITA plates require matplotlib+numpy (pip install agentsg[plot])") from exc
    elements = _plate_copies(raw)
    n_total = len(elements)
    counts = {}
    for el in elements:
        counts[el["symbol"]] = counts.get(el["symbol"], 0) + 1
    if compact:
        seen = set()
        reps = []
        for el in elements:
            if el["symbol"] in seen:
                continue
            seen.add(el["symbol"])
            reps.append(el)
        elements = reps
    from ..cell.diagrams import _ita_names
    from ..setting import format_cob
    num = _sg_number(sg)
    classic, modern = _ita_names(sg)
    symbols = sorted(counts)
    out = {
        "sg_number": num,
        "sg_hm": classic,
        "sg_hm_2016": modern,
        "crystal_system": system,
        "projection": projection,
        "legend": legend,
        "show_centring": show_centring,
        "n_total": n_total,
        "counts": counts,
        "elements": elements,
        "png_url": f"/v1/ita-plate.png?{png_query}",
        "note": _plate_note(n_total, symbols, projection),
    }
    if compact:
        out["compact"] = True
    cob = getattr(sg, "cob", None)
    if cob is not None:
        out["setting"] = format_cob(cob)
        out["cob"] = out["setting"]
    return _with_concepts("ita_plate_json", out)


def _plate_copies(raw) -> list[dict[str, Any]]:
    """Distinct in-cell copies, folding an edge at 1 back onto 0.

    ``_element_copies`` can emit the same axis twice with opposite direction
    and can leave a boundary location at 1 rather than 0. The plate draws
    that symbol once per cell face, repeated on the opposite edge.
    """
    import numpy as np
    seen = set()
    elements = []
    for el in raw:
        loc = el.get("location")
        if loc is None:
            continue
        exact_loc = el.get("location_exact")
        exact_axis = el.get("axis_exact")
        if exact_loc is not None:
            loc_a = [float(c) % 1.0 for c in exact_loc]
            loc_a = [0.0 if c > 1.0 - 1e-9 else c for c in loc_a]
            lk = tuple(exact_loc)
        else:
            loc_a = np.mod(np.asarray(loc, float), 1.0)
            loc_a = np.where(loc_a > 1.0 - 1e-6, 0.0, loc_a)
            lk = tuple(round(float(c), 3) for c in loc_a)
        axis = exact_axis if exact_axis is not None else el.get("axis")
        if axis is None:
            ak = None
        else:
            axv = [float(c) for c in axis]
            for component in axv:
                if abs(component) > 1e-9:
                    if component < 0:
                        axv = [-c for c in axv]
                    break
            ak = tuple(axv)
        key = (el.get("type"), el.get("symbol"), ak, lk)
        if key in seen:
            continue
        seen.add(key)
        item = {
            "type": el.get("type"),
            "symbol": el.get("symbol"),
            "order": el.get("order"),
            "axis": None if ak is None else list(ak),
            "location": [float(c) for c in loc_a],
        }
        if el.get("contained_in"):
            item["contained_in"] = el["contained_in"]
        if el.get("type") == "glide":
            raws = el.get("_glide_raws")
            if el.get("symbol") == "e" and raws and len(raws) >= 2:
                item["glide"] = [
                    [float(c) for c in raw] for raw in raws[:2]]
            elif el.get("intrinsic_exact") is not None:
                item["glide"] = [float(c) for c in el["intrinsic_exact"]]
        elements.append(item)
    return elements


def _as_bool(raw) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in ("1", "true", "yes")


def _cob_angle_sigma(data: dict[str, Any]) -> float:
    raw = data.get("angle_sigma")
    if raw is None or raw == "":
        return COB_ANGLE_SIGMA_DEG
    return float(raw)


def _cob_boundary_rel(data: dict[str, Any]) -> float:
    raw = data.get("boundary_rel")
    if raw is None or raw == "":
        return BOUNDARY_REL
    return float(raw)


def plate_png_args(data: dict[str, Any]) -> tuple[Any, str, bool, bool]:
    sg = resolve_plate_sg(data)
    system = getattr(sg, "crystal_system", None)
    if system is None and hasattr(sg, "base"):
        system = getattr(sg.base, "crystal_system", None)
    projection = str(data.get("projection") or default_projection(system))
    if projection not in ("a", "b", "c", "all"):
        raise ValueError("projection must be a, b, c, or all")
    return sg, projection, _as_bool(data.get("legend", False)), _as_bool(data.get("show_centring", False))


def cell_info(data: dict[str, Any]) -> dict[str, Any]:
    cell = parse_cell(data.get("cell"))
    uc = UnitCell(*cell)
    rec = uc.reciprocal()
    reduced, cob = niggli_reduce(*cell)
    out: dict[str, Any] = {
        "cell": list(cell),
        "volume": uc.volume(),
        "reciprocal": [rec.a, rec.b, rec.c, rec.alpha, rec.beta, rec.gamma],
        "niggli": list(reduced),
        "niggli_cob": [[int(x) for x in row] for row in cob],
        "root_invariant": list(root_invariant(cell)),
        "similarity_invariant": list(similarity_invariant(cell)),
    }
    if data.get("sg") is not None:
        rec_sg = resolve_sg(data["sg"])
        prim = primitive_cell(cell, rec_sg.hermann_mauguin)
        out["sg_number"] = rec_sg.number
        out["sg_hm"] = rec_sg.hermann_mauguin
        out["centering"] = lattice_letter(rec_sg.hermann_mauguin)
        out["primitive_cell"] = list(prim)
        out["root_invariant"] = list(root_invariant(prim))
        out["similarity_invariant"] = list(similarity_invariant(prim))
    return out


def lattice_symmetry_info(data: dict[str, Any]) -> dict[str, Any]:
    cell = parse_cell(data.get("cell"))
    max_delta = float(data.get("max_delta", LE_PAGE_MAX_DELTA_DEG))
    ls = lattice_symmetry(cell, max_delta=max_delta)
    scores = []
    for s in (ls.two_fold_scores or [])[:12]:
        scores.append({
            "le_page_delta": s.le_page_delta,
            "kurlin_distance": s.kurlin_distance,
            "direct_axis": list(s.direct_axis) if s.direct_axis is not None else None,
        })
    out: dict[str, Any] = {
        "cell": list(cell),
        "order": ls.order,
        "crystal_system": ls.crystal_system,
        "two_fold_scores": scores,
        "max_delta_deg": max_delta,
        "units": {"le_page_delta": "degrees", "kurlin_distance": "angstrom"},
    }
    if _as_bool(data.get("include_g6", False)):
        from ..cell import distance_to_symmetry
        from .. import space_group
        from ..group import point_group as pg
        # diagnostic only — cubic reference if requested without sg
        out["g6_note"] = "G6 deficiency is diagnostic (Å²); prefer kurlin_distance."
        if data.get("sg") is not None:
            rec = resolve_sg(data["sg"])
            out["g6_distance_to_symmetry"] = distance_to_symmetry(
                cell, pg(rec.operations()))
    return _with_concepts("lattice_symmetry_info", out)


def compare_info(data: dict[str, Any]) -> dict[str, Any]:
    a = parse_cell(data.get("cell_a"))
    b = parse_cell(data.get("cell_b"))
    sg_a = data.get("sg_a")
    sg_b = data.get("sg_b")
    cell_a, cell_b = a, b
    if sg_a is not None:
        cell_a = primitive_cell(a, resolve_sg(sg_a).hermann_mauguin)
    if sg_b is not None:
        cell_b = primitive_cell(b, resolve_sg(sg_b).hermann_mauguin)
    dec = root_volume_decomposition(cell_a, cell_b)
    out: dict[str, Any] = {
        "cell_a": list(a),
        "cell_b": list(b),
        "primitive_a": list(cell_a),
        "primitive_b": list(cell_b),
        "root_distance": root_distance(cell_a, cell_b),
        "similarity_distance": similarity_distance(cell_a, cell_b),
        "volume_decomposition": {
            "total": dec["total"],
            "volume_component": dec["volume_component"],
            "shape_residual": dec["shape_residual"],
            "coupling_angle_deg": dec["coupling_angle_deg"],
        },
        "units": "angstrom (root); similarity is dimensionless",
    }
    if _as_bool(data.get("include_sublattices", False)):
        res = compare_cells(a, b)
        out["sublattices"] = {
            "length_tol_pct": COMPARE_LENGTH_TOL_PCT,
            "angle_tol_deg": COMPARE_ANGLE_TOL_DEG,
            "volume_ratio": res.get("volume_ratio"),
            "solutions": [
                {
                    "index": m.index,
                    "M": [list(row) for row in m.M],
                    "resulting_cell": list(m.resulting_cell),
                    "max_length_dev_pct": m.max_length_dev,
                    "max_angle_dev_deg": m.max_angle_dev,
                }
                for m in res.get("solutions", [])[:20]
            ],
        }
    return _with_concepts("compare_info", out)


def reindex_info(data: dict[str, Any]) -> dict[str, Any]:
    rec = resolve_sg(data.get("sg"))
    cell = parse_cell(data.get("cell"))
    length_tol = float(data.get("length_tol_pct", METRIC_LENGTH_TOL_PCT))
    angle_tol = float(data.get("angle_tol_deg", METRIC_ANGLE_TOL_DEG))
    ops = surface_geometric_operators(
        rec.number, cell, length_tol_pct=length_tol, angle_tol_deg=angle_tol)
    return _with_concepts("reindex_info", {
        "sg_number": rec.number,
        "sg_hm": rec.hermann_mauguin,
        "cell": list(cell),
        "length_tol_pct": length_tol,
        "angle_tol_deg": angle_tol,
        "note": (
            "Geometry surfaces branches only. Residual 0 (is_metric_symmetry) "
            "is true merohedry — intensities are required to decide; v1 cannot."
        ),
        "operators": [
            {
                "xyz": op_to_xyz(g.op),
                "residual": g.residual,
                "is_identity": g.is_identity,
                "is_metric_symmetry": g.is_metric_symmetry,
            }
            for g in ops
        ],
    })


def pdb_search(state, data: dict[str, Any]) -> dict[str, Any]:
    if state is None or state.db is None or state.index is None:
        raise HttpError(503, "PDB database is not loaded")
    rec = resolve_sg(data.get("sg") or data.get("sg_hm") or data.get("sg_number"))
    cell = parse_cell(data.get("cell")) if data.get("cell") is not None else parse_cell([
        data["a"], data["b"], data["c"],
        data["alpha"], data["beta"], data["gamma"],
    ])
    k = data.get("k")
    cutoff = data.get("cutoff")
    same_hm = _as_bool(data.get("same_hm", data.get("same_sg", False)))
    return_cob = _as_bool(data.get("return_cob", False))
    if k is not None:
        k = int(k)
        if k < 1:
            raise ValueError("k must be a positive integer")
        hits = state.index.k_nearest(cell, k=k, sg_hm=rec.hermann_mauguin)
        if same_hm:
            hits = [(pid, d) for pid, d in hits]
            meta = state.db.lookup_cells([pid for pid, _ in hits])
            hits = [(pid, d) for pid, d in hits
                    if meta.get(pid, {}).get("sg_hm") == rec.hermann_mauguin]
        if cutoff is not None:
            cutoff = float(cutoff)
            hits = [(pid, d) for pid, d in hits if d <= cutoff]
        meta = state.db.lookup_cells([pid for pid, _ in hits])
        enriched = []
        for pid, dist in hits:
            rec_h = {"pdb_id": pid, "distance": dist}
            info = meta.get(pid)
            if info is not None:
                rec_h.update({"sg_number": info["sg_number"],
                              "sg_hm": info["sg_hm"], "cell": info["cell"]})
            enriched.append(rec_h)
        prim = primitive_cell(cell, rec.hermann_mauguin)
        result = {
            "cell": list(cell),
            "sg_number": rec.number,
            "sg_hm": rec.hermann_mauguin,
            "centering": lattice_letter(rec.hermann_mauguin),
            "primitive_cell": list(prim),
            "k": k,
            "cutoff": None if cutoff is None else float(cutoff),
            "same_hm": same_hm,
            "count": len(enriched),
            "hits": enriched,
            "pipeline": {
                "centering_to_primitive": "ITA Table 5.1.3.1",
                "reduction": "selling_delaunay",
                "invariant": "sorted_linear_key",
            },
        }
        if return_cob:
            from ..cell.selling_cob import annotate_search_hits
            annotate_search_hits(
                state.db, cell, rec.hermann_mauguin, enriched,
                angle_sigma=_cob_angle_sigma(data),
                boundary_rel=_cob_boundary_rel(data),
            )
    else:
        if cutoff is None:
            raise ValueError("provide cutoff (Å) and/or k")
        result = search_compatible(
            state.db, state.index,
            cell=cell,
            cutoff=float(cutoff),
            sg_number=rec.number,
            sg_hm=rec.hermann_mauguin,
            same_hm=same_hm,
            return_cob=return_cob,
            angle_sigma=_cob_angle_sigma(data),
            boundary_rel=_cob_boundary_rel(data),
        )
    if return_cob:
        result["gates"] = {
            "length_tol_pct": COB_LENGTH_TOL_PCT,
            "angle_tol_deg": COB_ANGLE_TOL_DEG,
            "angle_sigma_deg": _cob_angle_sigma(data),
            "boundary_rel": _cob_boundary_rel(data),
        }
    if _as_bool(data.get("plot", False)):
        _attach_root_plot(state, result)
    return _with_concepts("pdb_search", result)


def _attach_root_plot(state, result: dict[str, Any]) -> None:
    """SVD the hit-set linear keys and attach a PC1–PC2 scatter."""
    from .scatter import scatter_payload

    hits = result.get("hits") or []
    roots_by_id = state.db.lookup_roots([h["pdb_id"] for h in hits])
    kept = []
    rows = []
    distances = []
    for hit in hits:
        root = roots_by_id.get(hit["pdb_id"])
        if root is None:
            continue
        kept.append(hit)
        rows.append(root)
        distances.append(float(hit.get("distance") or 0.0))
    if len(rows) < 2:
        result["plot"] = {
            "n": len(rows),
            "note": "need at least 2 hits with linear keys",
        }
        return
    payload = scatter_payload(
        rows,
        [h["pdb_id"] for h in kept],
        distances,
        sorted_linear_key(result["primitive_cell"]),
    )
    for hit, xy in zip(kept, payload["xy"]):
        hit["xy"] = xy
    result["svd"] = payload["svd"]
    result["query_xy"] = payload["query_xy"]
    result["plot_png_base64"] = payload["png_base64"]
    if payload.get("png_error"):
        result["plot_error"] = payload["png_error"]


def pdb_lookup(state, pdb_id: str) -> dict[str, Any]:
    if state is None or state.db is None:
        raise HttpError(503, "PDB database is not loaded")
    pid = pdb_id.strip().upper()
    meta = state.db.lookup_cells([pid])
    info = meta.get(pid)
    if info is None:
        raise HttpError(404, f"unknown pdb_id {pid}")
    return {"pdb_id": pid, **info}


def help_catalog() -> dict[str, Any]:
    """Short index derived from the same catalog as GET /api."""
    from .manifest import ENDPOINTS, apply_site, public_url
    compute = [
        e for e in ENDPOINTS
        if e["path"].startswith("/v1/") or e["path"] in ("/search", "/plates", "/api")
    ]
    return {
        "service": "agentsg",
        "base_url": public_url(),
        "auth": "Authorization: Bearer (stored in connector credentials)",
        "rule": "GET /api for the full catalog, then call an endpoint before answering.",
        "skill": "/skill.md",
        "discover": "/api",
        "calls": [
            {"when": e["description"], "method": e["methods"][0], "path": e["path"],
             "example": apply_site(e["example"])}
            for e in compute
        ],
    }


def plates_png(qs: dict[str, Any]) -> bytes:
    """GET /plates — ITA plate PNG. Default projection is c. Unknown sg → 404."""
    from .http import HttpError
    from .plates import render_ita_png
    raw_sg = qs.get("sg")
    if raw_sg is None or str(raw_sg).strip() == "":
        raise HttpError(400, "sg: is required (IT number or Hermann-Mauguin symbol)")
    projection = str(qs.get("projection") or "c")
    if projection not in ("a", "b", "c", "all"):
        raise HttpError(400, "projection: must be a, b, c, or all")
    sg = resolve_sg(raw_sg)
    return render_ita_png(sg, projection=projection, legend=False, show_centring=False)


def plate_query_string(data: dict[str, Any]) -> str:
    q = {}
    if data.get("sg") is not None:
        q["sg"] = str(data["sg"])
    if data.get("setting"):
        q["setting"] = str(data["setting"])
    if data.get("projection"):
        q["projection"] = str(data["projection"])
    if _as_bool(data.get("legend", False)):
        q["legend"] = "true"
    if _as_bool(data.get("show_centring", False)):
        q["show_centring"] = "true"
    return urlencode(q)
