"""Single endpoint catalog. ``GET /api`` and OpenAPI are generated from this."""
from __future__ import annotations

import os
from typing import Any

API_VERSION = "0.3.0"
_BASE_TOKEN = "{{BASE_URL}}"
_NAME_TOKEN = "{{API_NAME}}"
_AUTH = '-H "Authorization: Bearer $AGENTSG_TOKEN"'


def api_name() -> str:
    """Connector name from ``AGENTSG_API_NAME`` (default ``agentsg``)."""
    return os.environ.get("AGENTSG_API_NAME", "agentsg") or "agentsg"


def public_url() -> str:
    """Public base URL from ``AGENTSG_PUBLIC_URL`` (default localhost)."""
    return (os.environ.get("AGENTSG_PUBLIC_URL") or "http://127.0.0.1:8765").rstrip("/")


def apply_site(text: str) -> str:
    """Fill ``{{BASE_URL}}`` and ``{{API_NAME}}`` from the process environment."""
    return text.replace(_BASE_TOKEN, public_url()).replace(_NAME_TOKEN, api_name())


def _p(name: str, typ: str, required: bool, description: str) -> dict[str, Any]:
    return {"name": name, "type": typ, "required": required, "description": description}


def _curl(path: str, *, data: str | None = None, auth: bool = True) -> str:
    if data is not None:
        hdr = f"{_AUTH} " if auth else ""
        return (
            f"curl -s {hdr}-H \"Content-Type: application/json\" "
            f"-d '{data}' {_BASE_TOKEN}{path}"
        )
    hdr = f"{_AUTH} " if auth else ""
    return f"curl -s {hdr}{_BASE_TOKEN}{path}"


# Public fields: path, methods, description, params, example, side_effects.
# Underscore keys are stripped from GET /api (OpenAPI / auth metadata).
ENDPOINTS: list[dict[str, Any]] = [
    {
        "path": "/",
        "methods": ["GET"],
        "description": "Landing links to health, docs, skill, OpenAPI, and GET /api",
        "params": [],
        "example": _curl("/", auth=False),
        "side_effects": "none",
        "_auth": False,
        "_op_ids": {"GET": "landing"},
    },
    {
        "path": "/health",
        "methods": ["GET"],
        "description": "Liveness, PDB cell count, whether ITA plates can be drawn",
        "params": [],
        "example": _curl("/health", auth=False),
        "side_effects": "none",
        "_auth": False,
        "_op_ids": {"GET": "health"},
    },
    {
        "path": "/api",
        "methods": ["GET"],
        "description": "Machine-readable catalog of every endpoint; call this before guessing paths",
        "params": [],
        "example": _curl("/api"),
        "side_effects": "none",
        "_auth": True,
        "_op_ids": {"GET": "apiManifest"},
    },
    {
        "path": "/openapi.json",
        "methods": ["GET"],
        "description": "OpenAPI 3.1 document generated from the same catalog as GET /api",
        "params": [],
        "example": _curl("/openapi.json", auth=False),
        "side_effects": "none",
        "_auth": False,
        "_op_ids": {"GET": "openapi"},
    },
    {
        "path": "/docs/muse.md",
        "methods": ["GET"],
        "description": "Muse connector contract and paste-in prompts",
        "params": [],
        "example": _curl("/docs/muse.md", auth=False),
        "side_effects": "none",
        "_auth": False,
        "_op_ids": {"GET": "museDocs"},
    },
    {
        "path": "/skill.md",
        "methods": ["GET"],
        "description": "Standing playbook: when to call which symmetry endpoint",
        "params": [],
        "example": _curl("/skill.md", auth=False),
        "side_effects": "none",
        "_auth": False,
        "_op_ids": {"GET": "skill"},
    },
    {
        "path": "/v1/help",
        "methods": ["GET"],
        "description": "Short JSON index of compute calls; prefer GET /api for the full catalog",
        "params": [],
        "example": _curl("/v1/help", auth=False),
        "side_effects": "none",
        "_auth": False,
        "_op_ids": {"GET": "help"},
    },
    {
        "path": "/v1/space-group",
        "methods": ["GET", "POST"],
        "description": "Look up a space group: Hermann–Mauguin, system, order, operators, derived absences",
        "params": [
            _p("sg", "string", True,
               "IT number 1–230, Hermann–Mauguin, or Hall. GET query or POST JSON."),
        ],
        "example": _curl("/v1/space-group?sg=96"),
        "side_effects": "none",
        "_op_ids": {"GET": "spaceGroupGet", "POST": "spaceGroup"},
    },
    {
        "path": "/v1/setting",
        "methods": ["POST"],
        "description": "Parse a non-standard setting and change-of-basis (e.g. P 21 21 2 (2a,b-a,c))",
        "params": [
            _p("setting", "string", True,
               "HM symbol plus optional ITA change-of-basis in parentheses"),
        ],
        "example": _curl("/v1/setting", data='{"setting":"P 21 21 2 (2a,b-a,c)"}'),
        "side_effects": "none",
        "_op_ids": {"POST": "setting"},
    },
    {
        "path": "/v1/identify",
        "methods": ["POST"],
        "description": (
            "Identify the ITA space-group type from xyz operators. "
            "|det P|>1 is a conventional centred cell (F222 primitive → #22, "
            "not P222 / #16), not a mis-count of the input operators"
        ),
        "params": [
            _p("ops", "array", True,
               "Non-empty list of xyz triplets such as x,y,z or -x,-y,z+1/2"),
        ],
        "example": _curl("/v1/identify", data='{"ops":["x,y,z","-x,-y,z+1/2"]}'),
        "side_effects": "none",
        "_op_ids": {"POST": "identify"},
    },
    {
        "path": "/v1/site",
        "methods": ["GET", "POST"],
        "description": "Orbit, multiplicity, and site-symmetry order at xyz (no ITA Wyckoff letter)",
        "params": [
            _p("sg", "string", True, "IT number, Hermann–Mauguin, or Hall"),
            _p("xyz", "string", True,
               "Fractional coordinates: 1/4,1/4,1/4 or JSON [\"1/4\",\"1/4\",\"1/4\"]"),
        ],
        "example": _curl("/v1/site?sg=225&xyz=1/4,1/4,1/4"),
        "side_effects": "none",
        "_op_ids": {"GET": "siteGet", "POST": "site"},
    },
    {
        "path": "/v1/reflections",
        "methods": ["GET", "POST"],
        "description": "Systematic absences; with hkl, whether that reflection is allowed and its phase",
        "params": [
            _p("sg", "string", True, "IT number, Hermann–Mauguin, or Hall"),
            _p("hkl", "string", False,
               "Optional Miller index as 0,0,1 (GET) or [0,0,1] (POST). Omit for condition strings only."),
        ],
        "example": _curl("/v1/reflections?sg=96&hkl=0,0,1"),
        "side_effects": "none",
        "_op_ids": {"GET": "reflectionsGet", "POST": "reflections"},
    },
    {
        "path": "/v1/harker",
        "methods": ["GET", "POST"],
        "description": "Harker sections and lines for Patterson interpretation",
        "params": [
            _p("sg", "string", True, "IT number, Hermann–Mauguin, or Hall"),
        ],
        "example": _curl("/v1/harker?sg=19"),
        "side_effects": "none",
        "_op_ids": {"GET": "harkerGet", "POST": "harker"},
    },
    {
        "path": "/v1/subgroups",
        "methods": ["GET", "POST"],
        "description": (
            "Derived maximal translationengleiche (t) and klassengleiche (k) "
            "subgroups from operators — not the ITA A1 table"
        ),
        "params": [
            _p("sg", "string", True, "IT number, Hermann–Mauguin, or Hall"),
            _p("kind", "string", False, "t, k, or both (default both)"),
            _p("maximal", "bool", False,
               "If true (default), only maximal t-subgroups among those derived"),
        ],
        "example": _curl("/v1/subgroups?sg=96"),
        "side_effects": "none",
        "_op_ids": {"GET": "subgroupsGet", "POST": "subgroups"},
    },
    {
        "path": "/v1/ita-plate",
        "methods": ["GET", "POST"],
        "description": "ITA plate metadata and element inventory plus png_url to fetch the drawing",
        "params": [
            _p("sg", "string", False, "IT number, Hermann–Mauguin, or Hall (or use setting)"),
            _p("setting", "string", False, "Non-standard setting string instead of sg"),
            _p("projection", "string", False, "a, b, c, or all. Monoclinic defaults to b; else c"),
            _p("legend", "bool", False, "If true, include the element legend panel"),
            _p("show_centring", "bool", False, "If true, draw centring translations"),
        ],
        "example": _curl("/v1/ita-plate", data='{"sg":96,"legend":true}'),
        "side_effects": "none",
        "_op_ids": {"GET": "itaPlateGet", "POST": "itaPlate"},
    },
    {
        "path": "/v1/ita-plate.png",
        "methods": ["GET"],
        "description": "Raw ITA plate PNG (same arguments as /v1/ita-plate)",
        "params": [
            _p("sg", "string", False, "IT number or Hermann–Mauguin"),
            _p("setting", "string", False, "Non-standard setting string"),
            _p("projection", "string", False, "a, b, c, or all"),
            _p("legend", "string", False, "true to draw the legend"),
            _p("show_centring", "string", False, "true to draw centring"),
        ],
        "example": _curl("/v1/ita-plate.png?sg=96&legend=true"),
        "side_effects": "none",
        "_op_ids": {"GET": "itaPlatePng"},
        "_png": True,
    },
    {
        "path": "/plates",
        "methods": ["GET"],
        "description": "ITA-style symmetry plate as PNG for a space group",
        "params": [
            _p("sg", "string", True,
               "IT number or Hermann–Mauguin. Unknown group → 404."),
            _p("projection", "string", False,
               "View axis a, b, c, or all. Default c (unlike monoclinic ITA default b)."),
        ],
        "example": _curl("/plates?sg=19"),
        "side_effects": "none",
        "_op_ids": {"GET": "platesPng"},
        "_png": True,
    },
    {
        "path": "/v1/cell",
        "methods": ["GET", "POST"],
        "description": "Volume, reciprocal cell, Niggli reduction, Kurlin root invariant",
        "params": [
            _p("cell", "array", True,
               "a,b,c Å and alpha,beta,gamma degrees. GET: cell=79,79,38,90,90,90"),
            _p("sg", "string", False,
               "Required for C/I/F/R conventional cells so the server reduces to primitive first"),
        ],
        "example": _curl("/v1/cell?cell=79,79,38,90,90,90&sg=96"),
        "side_effects": "none",
        "_op_ids": {"GET": "cellGet", "POST": "cell"},
    },
    {
        "path": "/v1/lattice-symmetry",
        "methods": ["GET", "POST"],
        "description": "Le Page holohedry and Kurlin two-fold scores for a noisy cell",
        "params": [
            _p("cell", "array", True, "Six cell parameters (Å, degrees)"),
            _p("max_delta", "float", False, "Le Page angle cutoff in degrees (default 3)"),
            _p("include_g6", "bool", False, "If true, add diagnostic G6 deficiency (Å²)"),
        ],
        "example": _curl("/v1/lattice-symmetry",
                         data='{"cell":[50,50,51,90,90,90]}'),
        "side_effects": "none",
        "_op_ids": {"GET": "latticeSymmetryGet", "POST": "latticeSymmetry"},
    },
    {
        "path": "/v1/compare",
        "methods": ["POST"],
        "description": "Root-invariant distance between two cells; optional integer sublattice M",
        "params": [
            _p("cell_a", "array", True, "First cell, six parameters"),
            _p("cell_b", "array", True, "Second cell, six parameters"),
            _p("sg_a", "string", False, "Space group of cell_a if centred"),
            _p("sg_b", "string", False, "Space group of cell_b if centred"),
            _p("include_sublattices", "bool", False,
               "If true, also search integer transforms M (native vs SeMet style)"),
        ],
        "example": _curl(
            "/v1/compare",
            data='{"cell_a":[50,50,50,90,90,90],"cell_b":[51,51,51,90,90,90]}',
        ),
        "side_effects": "none",
        "_op_ids": {"POST": "compare"},
    },
    {
        "path": "/v1/reindex",
        "methods": ["POST"],
        "description": "Geometric reindexing / ambiguity branches; cannot pick the intensity branch",
        "params": [
            _p("sg", "string", True, "IT number or Hermann–Mauguin of the indexed group"),
            _p("cell", "array", True, "Six cell parameters"),
            _p("length_tol_pct", "float", False, "Length tolerance percent (default 2)"),
            _p("angle_tol_deg", "float", False, "Angle tolerance degrees (default 2)"),
        ],
        "example": _curl(
            "/v1/reindex",
            data='{"sg":75,"cell":[50,50,80,90,90,90]}',
        ),
        "side_effects": "none",
        "_op_ids": {"POST": "reindex"},
    },
    {
        "path": "/v1/concept",
        "methods": ["GET", "POST"],
        "description": (
            "Concept graph. q searches labels and aliases "
            "(Smith normal form, allowed origin). id returns the card: "
            "builder definition, IUCr/Wikipedia quote with url and status, "
            "code anchors, and typed relations. Hashes are the graph snapshot"
        ),
        "params": [
            _p("q", "string", False,
               "Words to search. Example: Smith normal form"),
            _p("id", "string", False,
               "Concept id from a search hit, e.g. smith_normal_form. If set, q is ignored"),
            _p("limit", "int", False, "Search hit cap, 1–20, default 8"),
        ],
        "example": _curl("/v1/concept?q=Smith+normal+form"),
        "side_effects": "none",
        "_op_ids": {"GET": "conceptGet", "POST": "concept"},
    },
    {
        "path": "/v1/concept/uses",
        "methods": ["GET", "POST"],
        "description": "Concepts reached by USES in 1–3 steps, with each definition",
        "params": [
            _p("id", "string", True, "Start concept id, e.g. reflection_conditions"),
            _p("depth", "int", False, "1, 2, or 3 (default 3)"),
        ],
        "example": _curl("/v1/concept/uses?id=reflection_conditions&depth=3"),
        "side_effects": "none",
        "_op_ids": {"GET": "conceptUsesGet", "POST": "conceptUses"},
    },
    {
        "path": "/v1/concept/used-by",
        "methods": ["GET", "POST"],
        "description": "Concepts whose USES edges reach this id, in 1–3 steps, nearest first",
        "params": [
            _p("id", "string", True, "Target concept id, e.g. smith_normal_form"),
            _p("depth", "int", False, "1, 2, or 3 (default 3). 2 means things that use things that use id"),
        ],
        "example": _curl("/v1/concept/used-by?id=smith_normal_form&depth=2"),
        "side_effects": "none",
        "_op_ids": {"GET": "conceptUsedByGet", "POST": "conceptUsedBy"},
    },
    {
        "path": "/v1/concept/module",
        "methods": ["GET", "POST"],
        "description": "Concepts anchored in one source file",
        "params": [
            _p("module", "string", True,
               "agentsg/semi_invariants.py or a bare filename such as semi_invariants.py"),
        ],
        "example": _curl("/v1/concept/module?module=agentsg/semi_invariants.py"),
        "side_effects": "none",
        "_op_ids": {"GET": "conceptModuleGet", "POST": "conceptModule"},
    },
    {
        "path": "/v1/concept/receipt",
        "methods": ["GET", "POST"],
        "description": (
            "One ledger node for a receipt id from a concept card "
            "(definition_receipt, code_evidence[].receipt, references[].receipt, relations[].receipt)"
        ),
        "params": [
            _p("id", "string", True,
               "Receipt id, e.g. ent:concept:allowed_origins"),
        ],
        "example": _curl("/v1/concept/receipt?id=ent:concept:allowed_origins"),
        "side_effects": "none",
        "_op_ids": {"GET": "conceptReceiptGet", "POST": "conceptReceipt"},
    },
    {
        "path": "/search",
        "methods": ["GET", "POST"],
        "description": "PDB lattices near a cell in Kurlin root space (legacy alias of /v1/pdb/search)",
        "params": [
            _p("a", "float", True, "Cell edge a in Å (GET). POST may send cell:[6] instead"),
            _p("b", "float", True, "Cell edge b in Å"),
            _p("c", "float", True, "Cell edge c in Å"),
            _p("alpha", "float", True, "Angle α in degrees"),
            _p("beta", "float", True, "Angle β in degrees"),
            _p("gamma", "float", True, "Angle γ in degrees"),
            _p("sg", "string", True,
               "IT number or HM. Required so centred cells are reduced to primitive before the root"),
            _p("cutoff", "float", False, "Radius in Å on the root invariant. Provide cutoff and/or k"),
            _p("k", "int", False, "If set, return this many nearest neighbours"),
            _p("same_hm", "bool", False, "If true, restrict hits to the same Hermann–Mauguin setting"),
            _p("plot", "bool", False,
               "If true, SVD the hit Kurlin roots and return a PC1–PC2 scatter PNG"),
            _p("return_cob", "bool", False,
               "If true, include cob onto each deposited cell when a det +1 operator matches the reduced cell within 0.75% and 0.5°, plus cob_residual (null if none)"),
        ],
        "example": _curl(
            "/search?a=79&b=79&c=38&alpha=90&beta=90&gamma=90&sg=P212121&cutoff=1.0"
        ),
        "side_effects": "reads database",
        "_op_ids": {"GET": "legacySearchGet", "POST": "legacySearchPost"},
    },
    {
        "path": "/v1/pdb/search",
        "methods": ["GET", "POST"],
        "description": "PDB lattices within cutoff Å and/or k nearest neighbours on the root invariant",
        "params": [
            _p("cell", "array", True, "Six cell parameters; GET may use a,b,c,alpha,beta,gamma instead"),
            _p("sg", "string", True, "IT number or HM; required for centring → primitive"),
            _p("cutoff", "float", False, "Radius in Å. Provide cutoff and/or k"),
            _p("k", "int", False, "Nearest-neighbour count"),
            _p("same_hm", "bool", False, "If true, keep only the same Hermann–Mauguin setting"),
            _p("plot", "bool", False,
               "If true, mean-centred SVD of the hit Kurlin roots; response adds xy, svd, and plot_png_base64"),
            _p("return_cob", "bool", False,
               "If true, include cob onto each deposited cell when a det +1 operator matches the reduced cell within 0.75% and 0.5°, plus cob_residual (null if none)"),
        ],
        "example": _curl(
            "/v1/pdb/search",
            data='{"cell":[79,79,38,90,90,90],"sg":"P 21 21 21","cutoff":1.0}',
        ),
        "side_effects": "reads database",
        "_op_ids": {"GET": "pdbSearchGet", "POST": "pdbSearch"},
    },
    {
        "path": "/v1/pdb/{pdb_id}",
        "methods": ["GET"],
        "description": "Look up one stored PDB conventional cell and space group",
        "params": [
            _p("pdb_id", "string", True, "Four-character PDB id in the path, e.g. 1LYZ"),
        ],
        "example": _curl("/v1/pdb/1LYZ"),
        "side_effects": "reads database",
        "_op_ids": {"GET": "pdbLookup"},
    },
]


def _public_endpoint(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": entry["path"],
        "methods": list(entry["methods"]),
        "description": entry["description"],
        "params": [dict(p) for p in entry["params"]],
        "example": apply_site(entry["example"]),
        "side_effects": entry["side_effects"],
    }


def build_api_manifest() -> dict[str, Any]:
    """Machine-readable catalog served at GET /api."""
    return {
        "name": api_name(),
        "api_version": API_VERSION,
        "auth": {"scheme": "bearer", "header": "Authorization"},
        "endpoints": [_public_endpoint(e) for e in ENDPOINTS],
    }


def routed_paths() -> set[str]:
    """Path templates declared in the catalog."""
    return {e["path"] for e in ENDPOINTS}


def open_paths() -> set[str]:
    """Paths that skip Bearer auth."""
    return {e["path"] for e in ENDPOINTS if not e.get("_auth", True)}
