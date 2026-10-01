"""FastMCP surface over the same handlers as the HTTP API.

No bearer token. Clients connect to the streamable HTTP endpoint ``/mcp``.
Install the optional dependency with ``pip install 'agentsg[mcp]'``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from ..tolerances import (
    BOUNDARY_REL,
    COB_ANGLE_SIGMA_DEG,
    LE_PAGE_MAX_DELTA_DEG,
    METRIC_ANGLE_TOL_DEG,
    METRIC_LENGTH_TOL_PCT,
)
from . import handlers
from .app import ServerState
from .concepts import (
    concept_count,
    concept_info,
    concept_neighbors as concept_neighbors_query,
    concept_module as concept_module_query,
    concept_search as concept_search_query,
    concept_receipt as concept_receipt_query,
    concept_used_by as concept_used_by_query,
    concept_uses as concept_uses_query,
)
from .http import HttpError
from .manifest import API_VERSION

_HERE = Path(__file__).resolve().parent
_PUBLIC_DEFAULT = "https://sg-mcp.mxagents.org"
_READONLY = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


def _result(fn, data: dict[str, Any]):
    try:
        return fn(data)
    except HttpError as exc:
        from fastmcp.exceptions import ToolError
        raise ToolError(str(exc.payload.get("error") or exc)) from exc


def _render_plate(data: dict[str, Any]) -> bytes:
    """ITA plate PNG. ImportError becomes an HTTP-style 503 the tools already map."""
    from .plates import render_ita_png
    try:
        group, proj, show_legend, cent = handlers.plate_png_args(data)
        return render_ita_png(
            group, projection=proj, legend=show_legend, show_centring=cent,
        )
    except ImportError as exc:
        raise HttpError(
            503, "ITA plates require matplotlib+numpy (pip install 'agentsg[plot]')",
        ) from exc


def _clean(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _image_result(png: bytes, data: dict[str, Any]):
    """Return a PNG as image content plus JSON that does not repeat the bytes."""
    from fastmcp.tools.base import ToolResult
    from fastmcp.utilities.types import Image

    structured = dict(data)
    structured.pop("plot_png_base64", None)
    structured.pop("png_url", None)
    structured["png"] = "included"
    return ToolResult(
        content=Image(data=png, format="png"),
        structured_content=structured,
    )


def _mcp_playbook(base: str) -> str:
    """Scientific playbook with MCP tools and no API key.

    The HTTP ``skill.md`` still documents bearer auth. This copy keeps the
    LIMITATIONS and keeps the call table, and replaces the connector contract.
    """
    raw = (_HERE / "skill.md").read_text(encoding="utf-8")
    body = raw.split("# LIMITATIONS", 1)[1]
    body = body.replace("{{BASE_URL}}", base).replace("{{API_NAME}}", "sg-mcp")
    body = body.replace(
        "Then GET `png_url` (same bearer) and display the PNG.",
        "The tool result includes the plate PNG. Display that image.",
    )
    body = body.replace(
        "Then GET `png_url` and display it.",
        "The tool result includes the plate PNG. Display that image.",
    )
    body = body.replace(
        "Display `plot_png_base64`.",
        "The `pdb_search` tool result includes the scatter PNG when `plot` is true. Display that image.",
    )
    body = body.replace(
        "| Draw / show the ITA plate | `GET /plates?sg=19` (PNG) or `POST /v1/ita-plate` `{\"sg\":96,\"legend\":true}` then **GET the returned `png_url`** |",
        "| Draw / show the ITA plate | `ita_plate` with `sg`. The result includes the PNG and every in-cell copy. |",
    )
    body = body.replace(
        '| Find similar PDB cells | `POST /v1/pdb/search` with `sg` + `cutoff` or `k`. Add `"plot": true` for an SVD scatter of those hits. Add `"return_cob": true` for the change of basis from the query cell onto each deposited hit |',
        "| Find similar PDB cells | `pdb_search` with `sg` and `cutoff` or `k`. Set `plot` true for the scatter PNG. Set `return_cob` true for the change of basis onto each deposited cell. |",
    )
    body = body.replace(
        "| Remind yourself of this playbook | `GET /api` (full catalog) or `GET /skill.md` |",
        "| Remind yourself of this playbook | call the `playbook` tool |",
    )
    body = body.replace(
        "| What does this word mean here, and where is it in the code? | `GET /v1/concept?q=Smith+normal+form`, then `GET /v1/concept?id=` the top hit |",
        "| What does this word mean here, and where is it in the code? | `concept` with `q`, then `concept` with `id` of the top hit |",
    )
    body = body.replace(
        "| A result lists `concepts` | Call `GET /v1/concept?id=` on each id, then explain the term in ordinary sentences: name the file and the function in the sentence, and put links at the end |",
        "| A result lists `concepts` | Call `concept` with each id, then explain the term in ordinary sentences: name the file and the function in the sentence, and put links at the end |",
    )
    body = body.replace(
        "| Allowed origins of a space group | `GET /v1/allowed-origins?sg=225` |",
        "| Allowed origins of a space group | `allowed_origins` with `sg` |",
    )
    body = body.replace(
        "| What does a concept rest on? | `GET /v1/concept/uses?id=reflection_conditions&depth=3` |",
        "| What does a concept rest on? | `concept_uses` with `id` and `depth` (1–3) |",
    )
    body = body.replace(
        "| What uses this concept? | `GET /v1/concept/used-by?id=smith_normal_form&depth=2` |",
        "| What uses this concept? | `concept_used_by` with `id` and `depth` (1–3) |",
    )
    body = body.replace(
        "| Which concepts does this file implement? | `GET /v1/concept/module?module=agentsg/semi_invariants.py` |",
        "| Which concepts does this file implement? | `concept_module` with `module` |",
    )
    body = body.replace(
        "| Show the receipt for a claim | `GET /v1/concept/receipt?id=` the `receipt` field from the card |",
        "| Show the receipt for a claim | `concept_receipt` with the `receipt` id from the card |",
    )
    for http_call, tool_call in (
        ("GET /v1/concept?q=Smith+normal+form", 'concept with q="Smith normal form"'),
        ("GET /v1/concept?id=smith_normal_form", 'concept with id="smith_normal_form"'),
        ("GET /v1/concept?q=allowed+origin", 'concept with q="allowed origin"'),
        ("GET /v1/concept?id=allowed_origins", 'concept with id="allowed_origins"'),
        (
            "GET /v1/concept/uses?id=reflection_conditions&depth=3",
            'concept_uses with id="reflection_conditions" and depth=3',
        ),
        (
            "GET /v1/concept/used-by?id=smith_normal_form&depth=2",
            'concept_used_by with id="smith_normal_form" and depth=2',
        ),
        (
            "GET /v1/concept/module?module=agentsg/semi_invariants.py",
            'concept_module with module="agentsg/semi_invariants.py"',
        ),
        ("GET /v1/concept?id=t_subgroup", 'concept with id="t_subgroup"'),
        ("GET /v1/concept?id=k_subgroup", 'concept with id="k_subgroup"'),
        ("GET /v1/concept?id=reciprocal_lattice", 'concept with id="reciprocal_lattice"'),
        (
            "GET /v1/concept/receipt?id=ent:concept:allowed_origins",
            'concept_receipt with receipt="ent:concept:allowed_origins"',
        ),
        ("`/v1/concept/uses`", "`concept_uses`"),
        ("`/v1/concept/used-by`", "`concept_used_by`"),
        ("/v1/concept/used-by", "`concept_used_by`"),
        ("/v1/concept/uses", "`concept_uses`"),
        ("`/v1/concept/module`", "`concept_module`"),
        ("`/v1/concept/receipt`", "`concept_receipt`"),
        ("Fetch `/v1/concept/receipt`", "Call `concept_receipt`"),
        ("`GET /v1/concept?id=`", "`concept` with `id`="),
        ("use those endpoints", "use those tools"),
        ("still call `GET /v1/subgroups`", "still call `subgroups`"),
        ("**Concept answers come from `/v1/concept`.**", "**Concept answers come from `concept`.**"),
    ):
        body = body.replace(http_call, tool_call)
    header = f"""---
name: agentsg
description: Call the agentsg MCP tools for every space-group, reflection, site, ITA plate, unit-cell, PDB-lattice, or concept question. Never answer crystallography from memory.
---

# Standing rule

You have MCP tools for the agentsg crystallography engine.

- Endpoint: `{base}/mcp`
- Auth: none. There is no API key.
- Read-only. Rate limit 60/min (`429` + `Retry-After`)

**Call a tool before you answer.** Do not recite International Tables from memory. Do not invent systematic absences or Wyckoff letters. If you are unsure which tool, call `space_group`.

After a tool error, read the message and retry with corrected arguments. Do not invent a result.

HTTP paths named below are the same operations. Call the tool instead: `space_group`, `reflections`, `site`, `harker`, `allowed_origins`, `subgroups`, `ita_plate`, `setting`, `identify`, `cell`, `lattice_symmetry`, `compare_cells`, `reindex`, `pdb_search`, `pdb_lookup`, `concept`, `concept_uses`, `concept_used_by`, `concept_module`, `concept_receipt`, `playbook`.

**Concepts.** For “what does this word mean?”, “where is this implemented?”, “what does this rest on?”, “what uses this?”, “what does this file implement?”, or “where did that sentence come from?”, call `concept`, `concept_uses`, `concept_used_by`, `concept_module`, or `concept_receipt`. Answer in ordinary sentences, as at a blackboard: no heading and no list. Name the file and the function in the sentence. Put links at the end. Skip hashes and receipt ids unless they ask. When they ask what you can explain about the code, offer the list under Concept questions and wait for one.

**Pictures.** `ita_plate` returns the ITA plate PNG in the tool result, together with every in-cell copy of each element. `pdb_search` with `plot` true returns the PC1–PC2 scatter of those cells the same way. There is no image URL to fetch.

# LIMITATIONS"""
    return header + body


def build_mcp(state: ServerState, *, public_url: str = _PUBLIC_DEFAULT):
    """Build an unauthenticated FastMCP server bound to ``state``."""
    try:
        from fastmcp import FastMCP
        from starlette.requests import Request
        from starlette.responses import JSONResponse
    except ImportError as exc:
        raise SystemExit(
            "FastMCP is not installed. pip install 'agentsg[mcp]'"
        ) from exc

    base = public_url.rstrip("/")
    mcp = FastMCP(
        "agentsg",
        instructions=(
            "agentsg crystallography engine, exposed as MCP tools. "
            "No API key. Call a tool before answering a space-group, reflection, "
            "site, plate, unit-cell, PDB-lattice, or concept question. Never invent "
            "systematic absences or ITA Wyckoff letters (wyckoff_letter is null). "
            "Subgroup edges are t or k derived from operators, not the ITA A1 table. "
            "concept looks up what a term means in this code (q) or one term (id): "
            "definition, dictionary quote, code anchor. Answer in ordinary sentences, "
            "with the file and function in the sentence and links at the end. "
            "concept_uses, concept_used_by, "
            "concept_module, and concept_receipt are the dependency walk, the reverse "
            "walk, the file lookup, and one ledger node. Offer the Concept questions from playbook when the user "
            "asks what you can explain about the code. "
            f"Call playbook for the full LIMITATIONS. Endpoint: {base}/mcp"
        ),
        version=API_VERSION,
        website_url=base,
        auth=None,
    )

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "api_version": API_VERSION,
            "transport": "mcp",
            "auth_required": False,
            "endpoint": "/mcp",
            "public_url": base,
            "cells": state.n_cells,
            "concepts": concept_count(),
            "db": state.db_path,
        })

    @mcp.custom_route("/", methods=["GET"])
    async def landing(_request: Request) -> JSONResponse:
        return JSONResponse({
            "service": "agentsg",
            "transport": "mcp",
            "auth_required": False,
            "mcp": "/mcp",
            "health": "/health",
            "public_url": f"{base}/mcp",
        })

    @mcp.tool(annotations=_READONLY)
    def playbook() -> str:
        """Standing playbook: LIMITATIONS, t versus k, and which tool to call.

        No API key. Call the named tools; do not send Authorization.
        """
        return _mcp_playbook(base)

    @mcp.tool(annotations=_READONLY)
    def space_group(sg: str) -> dict[str, Any]:
        """Look up a space group: Hermann–Mauguin, system, order, operators, derived absences.

        sg: IT number 1–230, Hermann–Mauguin, or Hall symbol.
        """
        return _result(handlers.space_group_info, {"sg": sg})

    @mcp.tool(annotations=_READONLY)
    def setting(setting: str) -> dict[str, Any]:
        """Parse a non-standard setting and its change-of-basis.

        setting: HM symbol plus optional ITA change-of-basis, e.g. 'P 21 21 2 (2a,b-a,c)'.
        """
        return _result(handlers.setting_info, {"setting": setting})

    @mcp.tool(annotations=_READONLY)
    def identify(ops: list[str]) -> dict[str, Any]:
        """Identify the ITA space-group type from xyz operators.

        ops: non-empty list of triplets such as 'x,y,z' or '-x,-y,z+1/2'.
        |det P|>1 is a conventional centred cell, not a mis-count of the operators.
        """
        return _result(handlers.identify_ops, {"ops": ops})

    @mcp.tool(annotations=_READONLY)
    def site(sg: str, xyz: str) -> dict[str, Any]:
        """Orbit, multiplicity, and site-symmetry order at xyz. No ITA Wyckoff letter.

        xyz: fractional coordinates, e.g. '1/4,1/4,1/4'.
        """
        return _result(handlers.site_info, {"sg": sg, "xyz": xyz})

    @mcp.tool(annotations=_READONLY)
    def reflections(sg: str, hkl: list[int] | None = None) -> dict[str, Any]:
        """Systematic absences. With hkl, whether that reflection is allowed and its phase.

        hkl: optional Miller index, e.g. [0, 0, 1]. Omit for condition strings only.
        """
        return _result(handlers.reflections_info, _clean({"sg": sg, "hkl": hkl}))

    @mcp.tool(annotations=_READONLY)
    def harker(sg: str) -> dict[str, Any]:
        """Harker sections and lines for Patterson interpretation."""
        return _result(handlers.harker_info, {"sg": sg})

    @mcp.tool(annotations=_READONLY)
    def allowed_origins(sg: str) -> dict[str, Any]:
        """Discrete alternative origins, with floating directions pinned to zero.

        sg: IT number 1–230, Hermann–Mauguin, or Hall symbol.
        origins are the distinct gauges. floating_origin is the continuous freedom
        (empty when the origin is unique, one vector for a polar axis, three for P1).
        """
        return _result(handlers.allowed_origins_info, {"sg": sg})

    @mcp.tool(annotations=_READONLY)
    def subgroups(sg: str, kind: str = "both", maximal: bool = True) -> dict[str, Any]:
        """Derived translationengleiche (t) and klassengleiche (k) subgroups.

        kind: 't', 'k', or 'both'. Operator-derived edges, not the ITA A1 table.
        """
        return _result(
            handlers.subgroups_info,
            {"sg": sg, "kind": kind, "maximal": maximal},
        )

    @mcp.tool(annotations=_READONLY)
    def ita_plate(
        sg: str | None = None,
        setting: str | None = None,
        projection: str | None = None,
        legend: bool = False,
        show_centring: bool = False,
        compact: bool = False,
    ) -> dict[str, Any]:
        """ITA plate element inventory. The tool result includes the plate PNG.

        projection: 'a', 'b', 'c', or 'all'. Monoclinic defaults to b; otherwise c.
        'all' draws the three element projections (one panel for a cubic group).
        compact: counts by symbol plus one representative, for the large cubic groups.
        setting: a change-of-basis string, or 'R' for rhombohedral axes of an R group.
        """
        data = _clean({
            "sg": sg,
            "setting": setting,
            "projection": projection,
            "legend": legend,
            "show_centring": show_centring,
            "compact": compact,
        })
        try:
            out = handlers.ita_plate_json(data, png_query=handlers.plate_query_string(data))
            png = _render_plate(data)
        except HttpError as exc:
            from fastmcp.exceptions import ToolError
            raise ToolError(str(exc.payload.get("error") or exc)) from exc
        out.pop("png_url", None)
        return _image_result(png, out)

    @mcp.tool(annotations=_READONLY)
    def cell(cell: list[float], sg: str | None = None) -> dict[str, Any]:
        """Volume, reciprocal cell, Niggli reduction, and Kurlin root invariant.

        cell: a, b, c in Å and alpha, beta, gamma in degrees.
        sg: required for C/I/F/R conventional cells so the root is taken on the primitive cell.
        """
        return _result(handlers.cell_info, _clean({"cell": cell, "sg": sg}))

    @mcp.tool(annotations=_READONLY)
    def lattice_symmetry(
        cell: list[float],
        max_delta: float = LE_PAGE_MAX_DELTA_DEG,
        include_g6: bool = False,
        sg: str | None = None,
    ) -> dict[str, Any]:
        """Le Page holohedry and Kurlin two-fold scores for a cell.

        max_delta: Le Page angle cutoff in degrees.
        """
        return _result(handlers.lattice_symmetry_info, _clean({
            "cell": cell,
            "max_delta": max_delta,
            "include_g6": include_g6,
            "sg": sg,
        }))

    @mcp.tool(annotations=_READONLY)
    def compare_cells(
        cell_a: list[float],
        cell_b: list[float],
        sg_a: str | None = None,
        sg_b: str | None = None,
        include_sublattices: bool = False,
    ) -> dict[str, Any]:
        """Root-invariant distance between two cells. Optional integer sublattice search.

        Pass sg_a / sg_b when a cell is centred so the comparison uses the primitive cell.
        """
        return _result(handlers.compare_info, _clean({
            "cell_a": cell_a,
            "cell_b": cell_b,
            "sg_a": sg_a,
            "sg_b": sg_b,
            "include_sublattices": include_sublattices,
        }))

    @mcp.tool(annotations=_READONLY)
    def reindex(
        sg: str,
        cell: list[float],
        length_tol_pct: float = METRIC_LENGTH_TOL_PCT,
        angle_tol_deg: float = METRIC_ANGLE_TOL_DEG,
    ) -> dict[str, Any]:
        """Geometric reindexing branches. Intensities are required to pick a branch."""
        return _result(handlers.reindex_info, {
            "sg": sg,
            "cell": cell,
            "length_tol_pct": length_tol_pct,
            "angle_tol_deg": angle_tol_deg,
        })

    @mcp.tool(annotations=_READONLY)
    def pdb_search(
        sg: str,
        cell: list[float],
        cutoff: float | None = None,
        k: int | None = None,
        same_hm: bool = False,
        plot: bool = False,
        return_cob: bool = False,
        angle_sigma: float = COB_ANGLE_SIGMA_DEG,
        boundary_rel: float = BOUNDARY_REL,
    ) -> dict[str, Any]:
        """PDB lattices near a cell on the sorted Kurlin root invariant.

        Provide cutoff (Å) and/or k nearest neighbours. sg is required so centred
        cells are reduced to primitive before the search.
        plot: when true, the tool result includes a PC1–PC2 scatter PNG of the hits.
        return_cob: when true, Selling-reduce the query and attach cob from the
        query cell onto each deposited hit whose reduced cell matches within
        0.75% in length and 0.5° in angle. cob_residual is that leftover
        mismatch. cob is null when no proper operator is within tolerance.
        Only determinant +1 settings are listed. cob is the one with the fewest
        minus signs, then the spelling closest to a,b,c; cob_coset lists every
        proper match in that order.
        angle_sigma (degrees, default 0.05) and boundary_rel (default 1e-3)
        widen zero-conorm detection on the query closure so a reindexed
        high-symmetry cell is not classified as a generic lattice.
        """
        import base64
        out = _result(lambda data: handlers.pdb_search(state, data), _clean({
            "sg": sg,
            "cell": cell,
            "cutoff": cutoff,
            "k": k,
            "same_hm": same_hm,
            "plot": plot,
            "return_cob": return_cob,
            "angle_sigma": angle_sigma,
            "boundary_rel": boundary_rel,
        }))
        encoded = out.get("plot_png_base64") if isinstance(out, dict) else None
        if not encoded:
            return out
        return _image_result(base64.b64decode(encoded), out)

    @mcp.tool(annotations=_READONLY)
    def concept(q: str = "", id: str = "", limit: int = 8) -> dict[str, Any]:
        """Look up what a term means in this code, or which function implements it.

        q: words such as "Smith normal form" or "allowed origin". Returns hits
        with id, label, and definition. id: concept id such as smith_normal_form.
        The result has a definition, a dictionary or Wikipedia sentence with a
        url, the file and function, and relations. Answer in ordinary sentences.
        Do not mention hashes unless asked.
        Sample: where is the Smith normal form implemented?
        """
        return _result(concept_info, _clean({"q": q or None, "id": id or None, "limit": limit}))

    @mcp.tool(annotations=_READONLY)
    def concept_search(q: str, limit: int = 8) -> dict[str, Any]:
        """Token-overlap search of concept labels, aliases, and definitions.

        q: words such as "Smith normal form". Returns hits with id and definition.
        """
        return _result(concept_search_query, {"q": q, "limit": limit})

    @mcp.tool(annotations=_READONLY)
    def concept_neighbors(id: str, relation: str = "", depth: int = 1) -> dict[str, Any]:
        """Outgoing neighbours of a concept, optional relation filter, depth 1–3.

        relation: USES, SPECIALIZES, IS_A, PART_OF, DUAL_OF, EQUIVALENT_TO,
        RELATED_TO, DEFINED_BY, INSTANCE_OF, or CONTRASTS_WITH. Empty means all.
        """
        return _result(concept_neighbors_query, _clean({
            "id": id, "relation": relation or None, "depth": depth,
        }))

    @mcp.tool(annotations=_READONLY)
    def concept_uses(id: str, depth: int = 3) -> dict[str, Any]:
        """Concepts this one USES, out to depth 1–3, each with its definition.

        Sample: what do the reflection conditions rest on? id=reflection_conditions.
        """
        return _result(concept_uses_query, {"id": id, "depth": depth})

    @mcp.tool(annotations=_READONLY)
    def concept_used_by(id: str, depth: int = 3) -> dict[str, Any]:
        """Concepts whose USES edges point at this one, nearest first, depth 1–3.

        Depth 2 is things that use things that use id. USES edges only.
        Sample: what uses the Smith normal form? id=smith_normal_form, depth=2.
        """
        return _result(concept_used_by_query, {"id": id, "depth": depth})

    @mcp.tool(annotations=_READONLY)
    def concept_module(module: str) -> dict[str, Any]:
        """Concepts anchored in one source file.

        module: agentsg/semi_invariants.py or a bare filename.
        Sample: which concepts does semi_invariants.py implement?
        """
        return _result(concept_module_query, {"module": module})

    @mcp.tool(annotations=_READONLY)
    def concept_receipt(receipt: str) -> dict[str, Any]:
        """One ledger node for a receipt id copied from a concept card.

        receipt: definition_receipt, code_evidence[].receipt, references[].receipt,
        or relations[].receipt. Example: ent:concept:allowed_origins.
        Sample: show the receipt for the allowed-origins definition.
        """
        return _result(concept_receipt_query, {"receipt": receipt})

    @mcp.tool(annotations=_READONLY)
    def pdb_lookup(pdb_id: str) -> dict[str, Any]:
        """Look up one stored PDB conventional cell and space group."""
        try:
            return handlers.pdb_lookup(state, pdb_id)
        except HttpError as exc:
            from fastmcp.exceptions import ToolError
            raise ToolError(str(exc.payload.get("error") or exc)) from exc

    return mcp


def run_mcp(
    db_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 9877,
    public_url: str | None = None,
):
    """Load the optional PDB index and serve MCP with no API key."""
    public = (public_url or _PUBLIC_DEFAULT).rstrip("/")
    os.environ["AGENTSG_PUBLIC_URL"] = public
    os.environ["AGENTSG_API_NAME"] = "sg-mcp"
    state = ServerState(db_path, token=None)
    mcp = build_mcp(state, public_url=public)
    print(f"agentsg mcp on http://{host}:{port}/mcp")
    print(f"  concepts: {concept_count()}")
    print(f"  public: {public}/mcp")
    print("  auth: open (no API key)")
    if state.db_path:
        print(f"  database: {state.db_path} ({state.n_cells} cells)")
    try:
        mcp.run(
            transport="http",
            host=host,
            port=port,
            path="/mcp",
            stateless_http=True,
            json_response=True,
            show_banner=False,
        )
    finally:
        if state.db is not None:
            state.db.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="agentsg FastMCP server (no API key)")
    parser.add_argument("--db", default=os.environ.get("AGENTSG_DB", ""),
                        help="path to pdb_cells.duckdb (optional)")
    parser.add_argument("--host", default=os.environ.get("AGENTSG_MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("AGENTSG_MCP_PORT", "9877")))
    parser.add_argument("--public-url",
                        default=os.environ.get("AGENTSG_MCP_PUBLIC_URL", _PUBLIC_DEFAULT),
                        help="Public base URL advertised to clients")
    args = parser.parse_args(argv)
    try:
        run_mcp(
            args.db or None,
            host=args.host,
            port=args.port,
            public_url=args.public_url or None,
        )
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
