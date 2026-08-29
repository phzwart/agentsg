#!/usr/bin/env python3
"""Interactive PDB similarity-embedding explorer (Dash + Plotly).

Lasso / box-select points on the learned 2-D manifold, inspect them, and save
a selection JSON the agent (or CLI) can analyse further.

Usage (leanmap venv)::

    cd /Users/phzwart/Projects/agentsg
    KMP_DUPLICATE_LIB_OK=TRUE ../leanmap/.venv/bin/python \\
        analysis/embedding_explorer.py --port 8050

Then open http://127.0.0.1:8050

Selections are written to::

    analysis/data/selections/selection_YYYYMMDD_HHMMSS.json
    analysis/data/embedding_selection_latest.json   # always the last save

Analyse a saved selection::

    ../leanmap/.venv/bin/python analysis/analyze_selection.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "analysis" / "data"
META = DATA / "pdb_embedding_explorer_meta.npz"
Z_FALLBACK = DATA / "pdb_embedding_similarity_full.npy"
SEL_DIR = DATA / "selections"
LATEST = DATA / "embedding_selection_latest.json"
TRAJ = DATA / "deformation_trajectories.npz"

SYSTEMS = (
    "triclinic", "monoclinic", "orthorhombic", "tetragonal",
    "trigonal", "hexagonal", "cubic",
)
# plotly qualitative matching matplotlib tab10-ish
SYS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]
# Bravais centring for display (A/B/C collapsed to C; H → R)
CENTERINGS = ("P", "C", "I", "F", "R")
CENT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _ensure_path():
    src = REPO / "agentsg" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    analysis = REPO / "analysis"
    if str(analysis) not in sys.path:
        sys.path.insert(0, str(analysis))


def crystal_system_ids(sg_number: np.ndarray) -> np.ndarray:
    _ensure_path()
    from agentsg.space_groups import SPACE_GROUPS

    sys_by_num = np.empty(230, dtype=np.int32)
    sys_to_id = {s: i for i, s in enumerate(SYSTEMS)}
    for n, _hm, _hall, cs in SPACE_GROUPS:
        sys_by_num[n - 1] = sys_to_id.get(cs, 0)
    return sys_by_num[np.clip(sg_number, 1, 230) - 1]


def centering_ids(sg_hm: np.ndarray, sg_number: np.ndarray) -> np.ndarray:
    """P/C/I/F/R codes (A/B/C → C, H → R)."""
    _ensure_path()
    from agentsg.cell.primitive import lattice_letter
    from agentsg.space_groups import SPACE_GROUPS

    cent_to_id = {c: i for i, c in enumerate(CENTERINGS)}
    merge = {"A": "C", "B": "C", "C": "C", "H": "R"}
    hm_by_num = {n: hm for n, hm, *_ in SPACE_GROUPS}
    out = np.zeros(len(sg_number), dtype=np.int32)
    for i, (hm, num) in enumerate(zip(sg_hm, sg_number)):
        sym = str(hm) if hm is not None and str(hm) not in ("", "None", "nan") else None
        if not sym:
            sym = hm_by_num.get(int(num), "P 1")
        try:
            L = lattice_letter(sym)
        except Exception:
            L = "P"
        L = merge.get(L, L)
        out[i] = cent_to_id.get(L, 0)
    return out


def load_bundle(meta_path: Path = META):
    if not meta_path.is_file():
        raise SystemExit(
            f"missing {meta_path}\n"
            "Rebuild with: copy DuckDB → export (see script docstring) "
            "or re-run the explorer once with --rebuild-meta."
        )
    d = np.load(meta_path, allow_pickle=True)
    Z = d["Z"].astype(np.float32, copy=False)
    sg = d["sg_number"].astype(np.int32, copy=False)
    sg_hm = d["sg_hm"]
    return {
        "pdb_id": d["pdb_id"],
        "cell": d["cell"].astype(np.float64, copy=False),
        "volume": d["volume"].astype(np.float64, copy=False),
        "sg_number": sg,
        "sg_hm": sg_hm,
        "S": d["S"].astype(np.float32, copy=False),
        "Z": Z,
        "sys_id": crystal_system_ids(sg),
        "cent_id": centering_ids(sg_hm, sg),
    }


def rebuild_meta(db_path: Path, out: Path = META) -> None:
    """Export aligned metadata from a DuckDB file (uses a temp copy if locked)."""
    import shutil
    import tempfile

    import duckdb

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        copy = Path(td) / "pdb.duckdb"
        shutil.copy2(db_path, copy)
        con = duckdb.connect(str(copy), read_only=True)
        rows = con.execute(
            "SELECT pdb_id, a,b,c,alpha,beta,gamma, volume, sg_number, sg_hm, "
            "s0,s1,s2,s3,s4,s5 FROM cells WHERE s0 IS NOT NULL ORDER BY pdb_id"
        ).fetchall()
        con.close()

    pdb_id = np.array([r[0] for r in rows], dtype=object)
    cell = np.array([r[1:7] for r in rows], dtype=np.float64)
    volume = np.array([r[7] for r in rows], dtype=np.float64)
    sg_number = np.array([r[8] for r in rows], dtype=np.int32)
    sg_hm = np.array([r[9] for r in rows], dtype=object)
    S = np.array([r[10:16] for r in rows], dtype=np.float32)
    Z = np.load(Z_FALLBACK)
    if len(Z) != len(pdb_id):
        raise SystemExit(f"Z ({len(Z)}) vs meta ({len(pdb_id)}) length mismatch")
    np.savez_compressed(
        out,
        pdb_id=pdb_id, cell=cell, volume=volume,
        sg_number=sg_number, sg_hm=sg_hm, S=S, Z=Z,
    )
    print(f"wrote {out} ({len(pdb_id)} rows)")


def stratified_idx(sys_id: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    labels = np.unique(sys_id)
    per = max(1, n // max(1, len(labels)))
    for lab in labels:
        idx = np.flatnonzero(sys_id == lab)
        take = min(len(idx), per)
        parts.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(parts)
    if len(out) > n:
        out = rng.choice(out, size=n, replace=False)
    return np.sort(out)


def entropy_effective_rank(sig: np.ndarray) -> float:
    """Roy–Vetterli effective rank: exp(H), H = -Σ p ln p, p = σ / Σσ."""
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig[sig > 0]
    if sig.size == 0:
        return 0.0
    p = sig / sig.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def svd_root_spectrum(S: np.ndarray) -> dict | None:
    """Mean-centred SVD of Kurlin similarity roots (n × 6).

    Returns singular values σ, variance fractions σ²/Σσ², and cumulative.
    Needs at least 2 rows.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] < 2 or S.shape[1] < 1:
        return None
    X = S - S.mean(axis=0, keepdims=True)
    # economy SVD; for n < 6, rank ≤ n-1 after centering
    _, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    # pad to 6 for a stable report shape
    d = S.shape[1]
    sig = np.zeros(d, dtype=np.float64)
    sig[: len(sigma)] = sigma
    ss = float(np.sum(sig ** 2))
    frac = (sig ** 2 / ss).tolist() if ss > 0 else [0.0] * d
    cum = np.cumsum(frac).tolist()
    # leading right-singular vectors (loadings on s0..s5)
    loadings = []
    for k in range(min(3, Vt.shape[0])):
        loadings.append({
            "component": k + 1,
            "sigma": float(sig[k]),
            "frac": float(frac[k]),
            "v": Vt[k].tolist(),  # coefficients on s0..s5
        })
    return {
        "n": int(S.shape[0]),
        "dim": int(d),
        "feature": "similarity_invariant s=RI/V^(1/3)",
        "centered": True,
        "singular_values": sig.tolist(),
        "variance_frac": frac,
        "variance_cum": cum,
        "effective_rank": entropy_effective_rank(sig),
        "top_loadings": loadings,
    }


def selection_payload(bundle: dict, indices: list[int]) -> dict:
    idx = np.asarray(indices, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(bundle["Z"]))]
    sys_id = bundle["sys_id"][idx]
    sg = bundle["sg_number"][idx]
    # top space groups
    uniq, counts = np.unique(sg, return_counts=True)
    order = np.argsort(-counts)
    top_sg = [
        {
            "sg_number": int(uniq[i]),
            "sg_hm": str(bundle["sg_hm"][idx][np.flatnonzero(sg == uniq[i])[0]]),
            "n": int(counts[i]),
        }
        for i in order[:15]
    ]
    sys_counts = {
        SYSTEMS[i]: int((sys_id == i).sum())
        for i in range(len(SYSTEMS))
        if (sys_id == i).sum()
    }
    cent_id = bundle["cent_id"][idx]
    cent_counts = {
        CENTERINGS[i]: int((cent_id == i).sum())
        for i in range(len(CENTERINGS))
        if (cent_id == i).sum()
    }
    points = []
    for j in idx.tolist():
        a, b, c, al, be, ga = bundle["cell"][j].tolist()
        points.append({
            "index": int(j),
            "pdb_id": str(bundle["pdb_id"][j]),
            "z": bundle["Z"][j].tolist(),
            "S": bundle["S"][j].tolist(),
            "cell": [a, b, c, al, be, ga],
            "volume": float(bundle["volume"][j]),
            "sg_number": int(bundle["sg_number"][j]),
            "sg_hm": str(bundle["sg_hm"][j]),
            "crystal_system": SYSTEMS[int(bundle["sys_id"][j])],
            "centering": CENTERINGS[int(bundle["cent_id"][j])],
        })
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(points),
        "indices": idx.tolist(),
        "pdb_ids": [p["pdb_id"] for p in points],
        "summary": {
            "crystal_systems": sys_counts,
            "centerings": cent_counts,
            "top_space_groups": top_sg,
            "volume_median": float(np.median(bundle["volume"][idx])) if len(idx) else None,
            "volume_iqr": (
                np.percentile(bundle["volume"][idx], [25, 75]).tolist() if len(idx) else None
            ),
            "z_centroid": bundle["Z"][idx].mean(axis=0).tolist() if len(idx) else None,
            "S_mean": bundle["S"][idx].mean(axis=0).tolist() if len(idx) else None,
            "svd_roots": svd_root_spectrum(bundle["S"][idx]) if len(idx) else None,
        },
        "points": points,
    }


def save_selection(payload: dict) -> tuple[Path, Path]:
    SEL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SEL_DIR / f"selection_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    LATEST.write_text(json.dumps(payload, indent=2))
    return path, LATEST



def build_app(bundle: dict, *, display_n: int, seed: int):
    from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html
    import plotly.graph_objects as go

    N = len(bundle["Z"])
    if display_n >= N:
        base_show = np.arange(N)
    else:
        base_show = stratified_idx(bundle["sys_id"], display_n, seed=seed)

    Z = bundle["Z"]
    sys_id = bundle["sys_id"]
    cent_id = bundle["cent_id"]
    sel_state: dict = {"trace_full_idx": [], "n_data": 0, "visible": base_show.copy()}

    def filtered_show(sys_sel, cent_sel) -> np.ndarray:
        """AND filter: crystal system ∈ sys_sel AND centring ∈ cent_sel."""
        if not sys_sel or not cent_sel:
            return np.array([], dtype=np.int64)
        sys_codes = {SYSTEMS.index(s) for s in sys_sel if s in SYSTEMS}
        cent_codes = {CENTERINGS.index(c) for c in cent_sel if c in CENTERINGS}
        if not sys_codes or not cent_codes:
            return np.array([], dtype=np.int64)
        m = base_show[
            np.isin(sys_id[base_show], list(sys_codes))
            & np.isin(cent_id[base_show], list(cent_codes))
        ]
        return m

    def make_figure(sys_sel, cent_sel, color_by: str):
        fig = go.Figure()
        trace_full_idx: list[np.ndarray] = []
        show = filtered_show(sys_sel, cent_sel)
        sel_state["visible"] = show

        if color_by == "centering":
            labels, colors, codes = CENTERINGS, CENT_COLORS, cent_id
            title_mode = "colour=centring"
        else:
            labels, colors, codes = SYSTEMS, SYS_COLORS, sys_id
            title_mode = "colour=system"

        for i, name in enumerate(labels):
            m = show[codes[show] == i] if len(show) else np.array([], dtype=np.int64)
            if len(m) == 0:
                continue
            full_idx = np.asarray(m, dtype=np.int64)
            trace_full_idx.append(full_idx)
            fig.add_trace(go.Scattergl(
                x=Z[full_idx, 0].tolist(),
                y=Z[full_idx, 1].tolist(),
                mode="markers",
                name=name,
                ids=[str(int(j)) for j in full_idx],
                customdata=[[int(j), name,
                             SYSTEMS[int(sys_id[j])],
                             CENTERINGS[int(cent_id[j])]] for j in full_idx],
                marker=dict(size=5, color=colors[i], opacity=0.65, line=dict(width=0)),
                selected=dict(marker=dict(opacity=1.0, size=7)),
                unselected=dict(marker=dict(opacity=0.15)),
                hovertemplate=(
                    "idx=%{customdata[0]}<br>"
                    "system=%{customdata[2]}  centring=%{customdata[3]}<br>"
                    "z=(%{x:.3f}, %{y:.3f})<extra></extra>"
                ),
            ))

        if TRAJ.is_file():
            t = np.load(TRAJ)
            if "nn_Z" in t.files:
                fig.add_trace(go.Scatter(
                    x=t["nn_Z"][:, 0].tolist(), y=t["nn_Z"][:, 1].tolist(),
                    mode="lines+markers", name="NN path P2→trigonal",
                    line=dict(color="#145a32", width=2),
                    marker=dict(size=5, color="#145a32"), hoverinfo="name",
                ))
            if "p2_Z" in t.files:
                fig.add_trace(go.Scatter(
                    x=t["p2_Z"][:, 0].tolist(), y=t["p2_Z"][:, 1].tolist(),
                    mode="lines", name="param morph P2→trigonal",
                    line=dict(color="#666666", width=1.5, dash="dash"), hoverinfo="name",
                ))
            if "cubic_Z" in t.files:
                fig.add_trace(go.Scatter(
                    x=t["cubic_Z"][:, 0].tolist(), y=t["cubic_Z"][:, 1].tolist(),
                    mode="lines", name="cubic a stretch",
                    line=dict(color="#1b7a3d", width=2), hoverinfo="name",
                ))

        filt = (
            f"filter: system∈{{{','.join(sys_sel or [])}}} "
            f"AND centring∈{{{','.join(cent_sel or [])}}}"
        )
        fig.update_layout(
            template="plotly_white",
            title=f"PDB embedding — {title_mode} · {len(show):,} shown / {N:,}<br><sup>{filt}</sup>",
            xaxis_title="z₀", yaxis_title="z₁",
            yaxis_scaleanchor="x",
            legend=dict(
                orientation="v",
                yanchor="top", y=1,
                xanchor="left", x=1.02,
                bgcolor="rgba(255,255,255,0.85)",
                borderwidth=0,
            ),
            margin=dict(l=40, r=120, t=90, b=40),
            dragmode="lasso", clickmode="event+select",
            height=640, uirevision="embed", selectdirection="any",
        )
        sel_state["trace_full_idx"] = trace_full_idx
        sel_state["n_data"] = len(trace_full_idx)
        return fig

    def indices_from_selected(selected) -> list[int]:
        if not selected:
            return []
        trace_full_idx = sel_state["trace_full_idx"]
        n_data_traces = sel_state["n_data"]
        out: set[int] = set()
        for p in selected.get("points") or []:
            cd = p.get("customdata")
            if isinstance(cd, (list, tuple)) and cd:
                try:
                    out.add(int(cd[0])); continue
                except (TypeError, ValueError):
                    pass
            if cd is not None and not isinstance(cd, (list, tuple)):
                try:
                    out.add(int(cd)); continue
                except (TypeError, ValueError):
                    pass
            pid = p.get("id")
            if pid is not None:
                try:
                    out.add(int(pid)); continue
                except (TypeError, ValueError):
                    pass
            cnum = p.get("curveNumber")
            pidx = p.get("pointIndex", p.get("pointNumber"))
            if cnum is None or pidx is None:
                continue
            cnum, pidx = int(cnum), int(pidx)
            if 0 <= cnum < n_data_traces and 0 <= pidx < len(trace_full_idx[cnum]):
                out.add(int(trace_full_idx[cnum][pidx]))
        if out:
            return sorted(out)

        shown = sel_state.get("visible", base_show)
        if len(shown) == 0:
            return []
        Zs = Z[shown]
        rng = selected.get("range")
        if rng and "x" in rng and "y" in rng:
            xmin, xmax = rng["x"]; ymin, ymax = rng["y"]
            hit = shown[
                (Zs[:, 0] >= min(xmin, xmax)) & (Zs[:, 0] <= max(xmin, xmax)) &
                (Zs[:, 1] >= min(ymin, ymax)) & (Zs[:, 1] <= max(ymin, ymax))
            ]
            return sorted(int(i) for i in hit.tolist())
        lasso = selected.get("lassoPoints")
        if lasso and lasso.get("x") and lasso.get("y"):
            from matplotlib.path import Path as MplPath
            poly = np.column_stack([lasso["x"], lasso["y"]])
            if len(poly) >= 3:
                inside = MplPath(poly).contains_points(Zs)
                return sorted(int(i) for i in shown[inside].tolist())
        return []

    app = Dash(__name__)
    app.title = "PDB embedding explorer"
    fig0 = make_figure(list(SYSTEMS), list(CENTERINGS), "system")

    app.layout = html.Div(
        style={"fontFamily": "IBM Plex Sans, Helvetica, sans-serif", "margin": "12px 18px"},
        children=[
            html.H2("PDB similarity embedding explorer"),
            html.P(
                "Filter by crystal system AND centring (both must match), then lasso/box-select. "
                "Saved JSON → analysis/data/embedding_selection_latest.json",
                style={"maxWidth": "960px", "color": "#444"},
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr 220px",
                       "gap": "16px", "marginBottom": "10px", "maxWidth": "1100px"},
                children=[
                    html.Div([
                        html.Label("Crystal system (AND…)", style={"fontWeight": 600}),
                        dcc.Checklist(
                            id="sys-filter",
                            options=[{"label": f" {s}", "value": s} for s in SYSTEMS],
                            value=list(SYSTEMS),
                            inline=True,
                            style={"fontSize": 13},
                        ),
                    ]),
                    html.Div([
                        html.Label("Centring (…AND)", style={"fontWeight": 600}),
                        dcc.Checklist(
                            id="cent-filter",
                            options=[{"label": f" {c}", "value": c} for c in CENTERINGS],
                            value=list(CENTERINGS),
                            inline=True,
                            style={"fontSize": 13},
                        ),
                        html.Div(
                            style={"marginTop": "6px", "display": "flex", "gap": "8px"},
                            children=[
                                html.Button("All", id="cent-all", n_clicks=0, style={"fontSize": 12}),
                                html.Button("None", id="cent-none", n_clicks=0, style={"fontSize": 12}),
                                html.Button("P only", id="cent-p", n_clicks=0, style={"fontSize": 12}),
                            ],
                        ),
                    ]),
                    html.Div([
                        html.Label("Colour traces by", style={"fontWeight": 600}),
                        dcc.RadioItems(
                            id="color-by",
                            options=[
                                {"label": " system", "value": "system"},
                                {"label": " centring", "value": "centering"},
                            ],
                            value="system",
                        ),
                        html.Div(
                            style={"marginTop": "8px", "display": "flex", "gap": "8px", "flexWrap": "wrap"},
                            children=[
                                html.Button("All systems", id="sys-all", n_clicks=0, style={"fontSize": 12}),
                                html.Button("No systems", id="sys-none", n_clicks=0, style={"fontSize": 12}),
                            ],
                        ),
                    ]),
                ],
            ),
            dcc.Graph(id="scatter", figure=fig0, config={
                "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                "displaylogo": False,
            }),
            html.Div(
                style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"},
                children=[
                    html.Button("Save selection", id="btn-save", n_clicks=0,
                                style={"padding": "8px 14px", "fontWeight": 600}),
                    html.Button("Clear selection", id="btn-clear", n_clicks=0,
                                style={"padding": "8px 14px"}),
                    html.Button("Expand box to all cells", id="btn-expand", n_clicks=0,
                                title="Re-query full 206k in bbox, still applying system∧centring filter",
                                style={"padding": "8px 14px"}),
                    html.Span(id="sel-status", style={"color": "#145a32", "fontWeight": 600}),
                ],
            ),
            html.Div(id="summary", style={"marginTop": "12px", "maxWidth": "960px"}),
            dash_table.DataTable(
                id="table",
                columns=[
                    {"name": c, "id": c}
                    for c in (
                        "pdb_id", "centering", "crystal_system", "sg_hm", "sg_number",
                        "a", "b", "c", "alpha", "beta", "gamma", "volume", "index",
                    )
                ],
                page_size=20,
                style_table={"overflowX": "auto", "marginTop": "10px"},
                style_cell={"fontSize": 12, "padding": "4px 8px"},
                style_header={"fontWeight": "600"},
                row_selectable=False,
            ),
            dcc.Store(id="sel-indices", data=[]),
            dcc.Store(id="active-sys", data=list(SYSTEMS)),
            dcc.Store(id="active-cent", data=list(CENTERINGS)),
        ],
    )

    def _rows_from_indices(indices: list[int]) -> list[dict]:
        rows = []
        for j in indices[:5000]:
            a, b, c, al, be, ga = bundle["cell"][j]
            rows.append({
                "pdb_id": str(bundle["pdb_id"][j]),
                "centering": CENTERINGS[int(bundle["cent_id"][j])],
                "crystal_system": SYSTEMS[int(bundle["sys_id"][j])],
                "sg_hm": str(bundle["sg_hm"][j]),
                "sg_number": int(bundle["sg_number"][j]),
                "a": round(float(a), 3),
                "b": round(float(b), 3),
                "c": round(float(c), 3),
                "alpha": round(float(al), 2),
                "beta": round(float(be), 2),
                "gamma": round(float(ga), 2),
                "volume": round(float(bundle["volume"][j]), 1),
                "index": int(j),
            })
        return rows

    def _summary_div(indices: list[int]):
        if not indices:
            return html.Div("No selection.")
        payload = selection_payload(bundle, indices)
        s = payload["summary"]
        sys_bits = ", ".join(f"{k}: {v}" for k, v in s["crystal_systems"].items())
        cent_bits = ", ".join(f"{k}: {v}" for k, v in s["centerings"].items())
        sg_bits = ", ".join(
            f"{t['sg_hm']} ({t['n']})" for t in s["top_space_groups"][:8]
        )
        kids = [
            html.Strong(f"{payload['n']} cells selected"),
            html.Br(),
            html.Span(f"Centring — {cent_bits}"),
            html.Br(),
            html.Span(f"Systems — {sys_bits}"),
            html.Br(),
            html.Span(f"Top SG — {sg_bits}"),
            html.Br(),
            html.Span(
                f"Volume median {s['volume_median']:.0f} Å³ "
                f"(IQR {s['volume_iqr'][0]:.0f}–{s['volume_iqr'][1]:.0f})"
            ),
        ]
        svd = s.get("svd_roots")
        if svd:
            sig = svd["singular_values"]
            frac = svd["variance_frac"]
            cum = svd["variance_cum"]
            sig_s = "  ".join(f"σ{i+1}={sig[i]:.4g}" for i in range(len(sig)))
            frac_s = "  ".join(f"{100*frac[i]:.1f}%" for i in range(len(frac)))
            cum_s = "  ".join(f"{100*cum[i]:.1f}%" for i in range(len(cum)))
            kids.extend([
                html.Br(), html.Br(),
                html.Strong("SVD of Kurlin similarity roots s=RI/V⅓ (mean-centred)"),
                html.Br(),
                html.Span(sig_s),
                html.Br(),
                html.Span(f"var frac  {frac_s}"),
                html.Br(),
                html.Span(f"var cum   {cum_s}"),
                html.Br(),
                html.Span(f"effective rank (entropy) = {svd['effective_rank']:.2f}"),
            ])
            for ld in svd.get("top_loadings") or []:
                v = ld["v"]
                # highlight largest |loading|
                kmax = int(np.argmax(np.abs(v)))
                kids.extend([
                    html.Br(),
                    html.Span(
                        f"PC{ld['component']}: σ={ld['sigma']:.4g} "
                        f"({100*ld['frac']:.1f}%)  "
                        f"loadings s0..s5 = [{', '.join(f'{x:+.3f}' for x in v)}]  "
                        f"(max |·| on s{kmax})"
                    ),
                ])
        return html.Div(kids)

    @app.callback(
        Output("sys-filter", "value"),
        Input("sys-all", "n_clicks"),
        Input("sys-none", "n_clicks"),
        State("sys-filter", "value"),
        prevent_initial_call=True,
    )
    def sys_presets(n_all, n_none, current):
        trig = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trig == "sys-all":
            return list(SYSTEMS)
        if trig == "sys-none":
            return []
        return current or []

    @app.callback(
        Output("cent-filter", "value"),
        Input("cent-all", "n_clicks"),
        Input("cent-none", "n_clicks"),
        Input("cent-p", "n_clicks"),
        State("cent-filter", "value"),
        prevent_initial_call=True,
    )
    def cent_presets(n_all, n_none, n_p, current):
        trig = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trig == "cent-all":
            return list(CENTERINGS)
        if trig == "cent-none":
            return []
        if trig == "cent-p":
            return ["P"]
        return current or []

    @app.callback(
        Output("scatter", "figure"),
        Output("active-sys", "data"),
        Output("active-cent", "data"),
        Input("sys-filter", "value"),
        Input("cent-filter", "value"),
        Input("color-by", "value"),
    )
    def on_filter(sys_sel, cent_sel, color_by):
        sys_sel = sys_sel or []
        cent_sel = cent_sel or []
        return make_figure(sys_sel, cent_sel, color_by or "system"), sys_sel, cent_sel

    @app.callback(
        Output("sel-indices", "data"),
        Output("table", "data"),
        Output("summary", "children"),
        Output("sel-status", "children"),
        Input("scatter", "selectedData"),
        Input("btn-clear", "n_clicks"),
        Input("btn-expand", "n_clicks"),
        Input("btn-save", "n_clicks"),
        State("sel-indices", "data"),
        State("active-sys", "data"),
        State("active-cent", "data"),
        prevent_initial_call=True,
    )
    def on_select(selected, n_clear, n_expand, n_save, current, sys_sel, cent_sel):
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0]

        if triggered == "btn-clear":
            return [], [], "No selection.", "cleared"

        indices = list(current or [])

        if triggered == "scatter":
            if selected is None:
                return [], [], "No selection.", "deselected"
            indices = indices_from_selected(selected)
            if not indices and not (selected.get("points") or selected.get("range")
                                    or selected.get("lassoPoints")):
                indices = list(current or [])
            elif not indices:
                return [], [], html.Div([
                    html.Strong("Selection event had no resolvable points."),
                    html.Br(),
                    html.Span("Use the box tool (□), or zoom in first."),
                ]), "0 resolved — see note"

        if triggered == "btn-expand" and indices:
            Zs = Z[np.asarray(indices)]
            pad = 0.02 * max(float(Zs[:, 0].ptp()), float(Zs[:, 1].ptp()), 0.1)
            xmin, xmax = float(Zs[:, 0].min() - pad), float(Zs[:, 0].max() + pad)
            ymin, ymax = float(Zs[:, 1].min() - pad), float(Zs[:, 1].max() + pad)
            hit = np.flatnonzero(
                (Z[:, 0] >= xmin) & (Z[:, 0] <= xmax) &
                (Z[:, 1] >= ymin) & (Z[:, 1] <= ymax)
            )
            # keep AND filter on expand-to-full
            sys_codes = {SYSTEMS.index(s) for s in (sys_sel or []) if s in SYSTEMS}
            cent_codes = {CENTERINGS.index(c) for c in (cent_sel or []) if c in CENTERINGS}
            if sys_codes and cent_codes:
                hit = hit[np.isin(sys_id[hit], list(sys_codes)) & np.isin(cent_id[hit], list(cent_codes))]
            indices = hit.tolist()

        status = f"{len(indices)} selected"
        if triggered == "btn-save":
            if not indices:
                return indices, _rows_from_indices(indices), _summary_div(indices), "nothing to save"
            payload = selection_payload(bundle, indices)
            path, latest = save_selection(payload)
            status = f"saved {payload['n']} → {path.name}  (and {latest.name})"
            return indices, _rows_from_indices(indices), _summary_div(indices), status

        return indices, _rows_from_indices(indices), _summary_div(indices), status

    return app



def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--display-n", type=int, default=60000,
                    help="stratified points drawn (lasso selects among these; "
                         "use Expand box to hit full PDB)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild-meta", action="store_true")
    ap.add_argument("--db", type=Path, default=REPO / "data" / "pdb_cells.duckdb")
    args = ap.parse_args(argv)

    if args.rebuild_meta or not META.is_file():
        if not args.db.is_file():
            raise SystemExit(f"DuckDB not found: {args.db}")
        rebuild_meta(args.db)

    bundle = load_bundle()
    app = build_app(bundle, display_n=args.display_n, seed=args.seed)
    print(f"open http://{args.host}:{args.port}")
    print(f"selections → {SEL_DIR}/  and {LATEST}")
    u, c = np.unique(bundle["cent_id"], return_counts=True)
    print("centring counts:", {CENTERINGS[int(i)]: int(n) for i, n in zip(u, c)})
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
