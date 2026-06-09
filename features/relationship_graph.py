"""features/relationship_graph.py
==================================

ParselyFi "Relationship Graph" tab — an INTERACTIVE corporate relationship /
lineage graph.

Public entry point (the ONLY thing the main app calls, inside an
already-created tab):

    render_relationship_graph_tab() -> None

What it does
------------
1. INPUT — pick a seed either from free text ("From a company") or from the
   companies already produced by the List Intelligence tab ("From my last List
   run", reusing the ``li_results`` session-state list if present; honest hint
   to run List Intelligence first when absent).
2. EXTRACT — for the seed (depth 1, bounded) run ONE LinkUp ``sourcedAnswer``
   query (``company_research.company_search_with_linkup``) asking for parent,
   subsidiaries, acquisitions, investors, partnerships, competitors. The
   source-backed answer text + the REAL LinkUp source URLs are then fed to a
   structured ``common.gemini_generate_json`` call that returns
   ``{edges: [{source, target, relation, year?, evidence_url}]}``.
3. HONEST FILTER — every ``evidence_url`` must be one of the real LinkUp source
   URLs (set-membership) or it is blanked to "". Edges are de-duped, self-loops
   dropped, and NODES<=40 / EDGES<=80 enforced.
4. RENDER — a ``networkx.DiGraph`` -> ``pyvis.Network`` (in-line CDN, barnes_hut
   physics), embedded as a self-contained HTML string via
   ``net.generate_html()`` (never ``net.show()``, which writes a temp file) and
   shown with ``st.components.v1.html``. pyvis/networkx are imported under
   try/except with an honest fallback (``st.graphviz_chart`` then an edge table).
5. BELOW — an ``ui.section`` + a ``st.dataframe`` of the edges (Evidence as a
   ``LinkColumn``) and a ``ui.kpi_row`` of #entities / #relationships /
   #with-evidence.

Design rules (project CLAUDE.md) baked in:
- HONEST STATUS: missing keys/deps -> warning + early return, never fabricated
  data; an evidence URL that is not a real LinkUp source is blanked, not faked.
- BOUNDED MEMORY: hard NODE/EDGE caps; session history via
  ``common.push_history`` (already FIFO-capped); session keys all ``rg_``.
- TIMEOUTS: all network/LLM work goes through the timeout-guarded ``common.*``
  / ``company_research.*`` primitives — nothing new is duplicated here.
- All client/scraper/LLM logic is REUSED from ``common.*`` /
  ``company_research.*``; this module owns only the graph extraction + render.
"""

from __future__ import annotations

import html
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components

from features import common, ui
from features import company_research as cr
from features.common import (
    GEMINI_MODEL,
    feature_available,
    gemini_generate_json,
    get_history,
    push_history,
    render_gemini_health,
    render_token_usage,
    run_async,
)

logger = logging.getLogger("parselyfi.features.relationship_graph")

# ---------------------------------------------------------------------------
# Optional graph deps (imported under try/except so importing this module — and
# the whole app — never crashes if pyvis/networkx are unavailable). Availability
# is reflected through GRAPH_LIBS_AVAILABLE and an honest in-UI fallback.
# ---------------------------------------------------------------------------
try:
    import networkx as nx  # type: ignore
    from pyvis.network import Network  # type: ignore
    GRAPH_LIBS_AVAILABLE = True
    _GRAPH_IMPORT_ERROR = ""
except Exception as _e:  # pragma: no cover - exercised only when deps missing
    nx = None  # type: ignore
    Network = None  # type: ignore
    GRAPH_LIBS_AVAILABLE = False
    _GRAPH_IMPORT_ERROR = str(_e)

# ===========================================================================
# Constants / bounded-memory caps (all rg_-prefixed session keys)
# ===========================================================================
_PFX = "rg_"
SS_SEED = _PFX + "seed"                  # the resolved seed company name
SS_EDGES = _PFX + "edges"                # list[dict] — last extracted edges
SS_SOURCES = _PFX + "sources"            # list[str] — real LinkUp source URLs
SS_RUNNING = _PFX + "running"            # guard against double-submit
SS_HISTORY = _PFX + "graph_history"      # bounded via common.push_history
SS_DROPPED = _PFX + "dropped"            # int — edges dropped by the node/edge caps

# Hard caps so nothing the user / LLM supplies can blow up memory or the canvas.
MAX_NODES: int = 40
MAX_EDGES: int = 80
MAX_SEED_CHARS: int = 120                # cap a seed name fed downstream

REQUIRED_KEYS = ["GEMINI_API_KEY", "LINKUP_API_KEY"]

# Relation vocabulary (the ONLY relations we keep; anything else is dropped).
RELATIONS = ("parent_of", "subsidiary_of", "acquired",
             "invested_in", "partner_of", "competitor_of")

# Node-type -> color. Seed is ParselyFi green; the rest are role-coded.
TYPE_COLORS: Dict[str, str] = {
    "seed": "#10b981",        # ParselyFi green
    "subsidiary": "#0a7cff",  # blue
    "investor": "#8e75b2",    # violet
    "competitor": "#f59e0b",  # amber
    "partner": "#14b8a6",     # teal
    "parent": "#64748b",      # slate
    "other": "#94a3b8",       # default grey
}

# How a relation (pointing source -> target) assigns the TARGET node's type.
# (The seed always wins its own color regardless of these.)
_RELATION_TARGET_TYPE: Dict[str, str] = {
    "parent_of": "subsidiary",     # seed is parent of target -> target is a subsidiary
    "subsidiary_of": "parent",     # seed is subsidiary of target -> target is a parent
    "acquired": "subsidiary",      # acquirer acquired target -> target absorbed
    "invested_in": "subsidiary",   # investor invested in target -> target is portfolio
    "partner_of": "partner",
    "competitor_of": "competitor",
}
# How a relation assigns the SOURCE node's type (when source is not the seed).
_RELATION_SOURCE_TYPE: Dict[str, str] = {
    "parent_of": "parent",
    "subsidiary_of": "subsidiary",
    "acquired": "parent",          # acquirer behaves like a parent
    "invested_in": "investor",
    "partner_of": "partner",
    "competitor_of": "competitor",
}


# ===========================================================================
# Small helpers
# ===========================================================================
def _clean(name: Any) -> str:
    """Trim + bound a free-text/LLM-supplied string to a safe display length."""
    s = str(name or "").strip()
    return s[:MAX_SEED_CHARS]


def _seed_candidates_from_list_run() -> List[str]:
    """Company names from the last List Intelligence run (``li_results``).

    HONEST: reads the SAME session key the List Intelligence tab writes
    (``li_results``, a list of row dicts) without duplicating any of its logic.
    Returns a de-duped (case-insensitive), order-preserving list of names, or an
    empty list when no run is present.
    """
    out: List[str] = []
    seen: set = set()
    try:
        results = st.session_state.get("li_results")
    except Exception:  # pragma: no cover - defensive
        results = None
    if not isinstance(results, list):
        return out
    for row in results:
        if not isinstance(row, dict):
            continue
        # HONEST: only offer companies that actually RESOLVED to a real entity in
        # the List run. A No-Match row has an empty canonical_name — skip it so the
        # picker never presents an unresolved typed name as a confirmed company.
        canonical = _clean(row.get("canonical_name"))
        if not canonical:
            continue
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical)
    return out


def _source_urls(sources: List[Any]) -> List[str]:
    """Extract the REAL source URLs from a LinkUp sources list (bounded, deduped)."""
    urls: List[str] = []
    seen: set = set()
    for src in sources or []:
        url = cr.safe_get_attribute(src, "url", None)
        if not url:
            continue
        url = str(url).strip()
        if not url or url in seen:
            continue
        # Only keep http(s) URLs — anything else can't be a clickable evidence link.
        try:
            scheme = urlparse(url).scheme.lower()
        except Exception:
            continue
        if scheme not in ("http", "https"):
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= 200:  # bounded read on the source list
            break
    return urls


# ===========================================================================
# Edge extraction (ONE LinkUp sourcedAnswer + ONE Gemini JSON pass)
# ===========================================================================
async def extract_relationship_edges(seed: str) -> Dict[str, Any]:
    """Depth-1 relationship extraction for ``seed``.

    Runs ONE LinkUp ``sourcedAnswer`` query (reusing
    ``company_research.company_search_with_linkup``) then ONE structured
    ``common.gemini_generate_json`` pass to turn the source-backed answer +
    real source URLs into a bounded, de-duped, evidence-filtered edge list.

    Returns ``{"success": bool, "seed": str, "edges": [...], "sources": [...],
    "error": Optional[str]}``. HONEST STATUS: on any failure ``success`` is
    False with a populated ``error`` and empty ``edges`` — never fabricated.
    """
    seed = _clean(seed)
    if not seed:
        return {"success": False, "seed": seed, "edges": [], "sources": [],
                "error": "Empty seed company."}

    # --- 1) ONE LinkUp sourcedAnswer query (depth 1, bounded) ---------------
    linkup_prompt = (
        f"Map the corporate relationships of the company \"{seed}\". "
        "Report, with sources: (1) its PARENT company or owner (if any); "
        "(2) its SUBSIDIARIES and business units; (3) ACQUISITIONS — who "
        "acquired whom and in what YEAR (acquirer and acquired company); "
        "(4) its notable INVESTORS / backers; (5) key PARTNERSHIPS and "
        "alliances; and (6) its top direct COMPETITORS. For each, name the "
        "specific other company and, where known, the year. Use only "
        "verifiable sources."
    )
    search = await cr.company_search_with_linkup(seed, additional_context=linkup_prompt)
    if not search.get("success"):
        return {
            "success": False, "seed": seed, "edges": [], "sources": [],
            "error": search.get("error") or "LinkUp returned no usable answer.",
        }

    answer_text = (search.get("answer") or "").strip()
    source_urls = _source_urls(search.get("sources", []) or [])

    if not answer_text:
        return {
            "success": False, "seed": seed, "edges": [], "sources": source_urls,
            "error": "LinkUp returned no source-backed answer for this company.",
        }

    # --- 2) ONE Gemini JSON pass: answer + real source URLs -> edges --------
    sources_block = "\n".join(f"- {u}" for u in source_urls) or "N/A"
    relations_list = ", ".join(RELATIONS)
    extraction_prompt = f"""
You are extracting a corporate RELATIONSHIP GRAPH for the seed company
"{seed}" STRICTLY from the source-backed answer text below. Do NOT invent or
infer any relationship that is not supported by the text.

--- SOURCE-BACKED ANSWER TEXT ---
{answer_text}

--- ALLOWED EVIDENCE URLS (an edge's evidence_url MUST be EXACTLY one of these,
    copied verbatim, or an empty string ""; never invent a URL) ---
{sources_block}

Return ONLY a single JSON object of this exact shape:
{{
  "edges": [
    {{"source": "<company A>", "target": "<company B>", "relation": "<one of: {relations_list}>", "year": "<YYYY or empty>", "evidence_url": "<one of the allowed URLs or empty>"}}
  ]
}}

Rules:
- "relation" MUST be exactly one of: {relations_list}.
- Direction matters and is "source <relation> target":
  - parent_of: source is the PARENT of target.
  - subsidiary_of: source is a SUBSIDIARY of target.
  - acquired: source ACQUIRED target (acquirer -> acquired).
  - invested_in: source (an investor) INVESTED IN target.
  - partner_of: source is a PARTNER of target.
  - competitor_of: source is a COMPETITOR of target.
- Include the seed company "{seed}" as the source or target of most edges.
- "year" is the 4-digit year if the text states one, else "".
- "evidence_url" MUST be copied verbatim from the ALLOWED EVIDENCE URLS list, or
  "" if no listed source supports the edge. Do NOT output any other URL.
- Omit any relationship the text does not support. Output JSON only, no prose.
"""
    result = await gemini_generate_json(
        extraction_prompt, model=GEMINI_MODEL, agent_name=f"Relationship Graph: {seed}"
    )

    raw_edges = []
    if isinstance(result, dict):
        raw_edges = result.get("edges") if isinstance(result.get("edges"), list) else []

    edges, dropped = _normalize_edges(seed, raw_edges, source_urls)

    # HONEST_STATUS: distinguish an LLM failure (timeout / blocked / parse-fail ->
    # gemini_generate_json returns {}) from a genuine "no relationships found".
    if not isinstance(result, dict) or not result:
        return {
            "success": False,
            "seed": seed,
            "edges": [],
            "sources": source_urls,
            "dropped": 0,
            "error": ("Relationship extraction failed — the model returned no data "
                      "(timeout, blocked, or unparseable). Please try again."),
        }

    return {
        "success": True,
        "seed": seed,
        "edges": edges,
        "sources": source_urls,
        "dropped": dropped,
        "error": None if edges else "No relationships were found in the available sources.",
    }


# Relation importance — when a seed has more neighbors than the node cap allows,
# structural edges (parent/subsidiary/acquired) survive before trivia (competitors).
_RELATION_PRIORITY = {
    "parent_of": 0,
    "subsidiary_of": 0,
    "acquired": 1,
    "invested_in": 2,
    "partner_of": 3,
    "competitor_of": 4,
}


def _valid_year(raw_year: Any) -> str:
    """Return a 4-digit year in a sane range, else '' — validate, never slice-first."""
    y = str(raw_year or "").strip()
    if len(y) == 4 and y.isdigit() and 1850 <= int(y) <= 2100:
        return y
    return ""


def _normalize_edges(
    seed: str,
    raw_edges: List[Any],
    allowed_urls: List[str],
) -> Tuple[List[Dict[str, str]], int]:
    """Validate, evidence-filter, dedupe, drop self-loops, canonicalize, and bound.

    Returns ``(edges, dropped)`` where ``dropped`` counts otherwise-valid edges
    discarded because the MAX_NODES / MAX_EDGES caps were hit, so the caller can
    disclose the truncation instead of presenting a capped graph as the whole set.

    HONEST: any ``evidence_url`` not in ``allowed_urls`` (the real LinkUp source
    set) is blanked to "". A single company is ONE node — labels are canonicalized
    to a first-seen casing so 'Anthropic'/'anthropic' never split into two dots.
    """
    allowed = set(allowed_urls)
    out: List[Dict[str, str]] = []
    seen_edges: set = set()
    node_set: set = set()
    dropped = 0

    # lower() -> first-seen casing, so the same entity is one consistently-cased node.
    canon: Dict[str, str] = {}

    def _canon(name: str) -> str:
        key = name.lower()
        if key not in canon:
            canon[key] = name
        return canon[key]

    seed_c = _canon(_clean(seed))
    node_set.add(seed_c.lower())

    # Stable, importance-ordered pass so the cap truncates trivia before structure.
    ordered = sorted(
        [r for r in raw_edges if isinstance(r, dict)],
        key=lambda r: _RELATION_PRIORITY.get(
            str(r.get("relation") or "").strip().lower(), 99
        ),
    )

    for raw in ordered:
        src_raw = _clean(raw.get("source"))
        tgt_raw = _clean(raw.get("target"))
        relation = str(raw.get("relation") or "").strip().lower()
        if not src_raw or not tgt_raw:
            continue
        if relation not in RELATIONS:
            continue
        if src_raw.lower() == tgt_raw.lower():   # self-loop
            continue
        source = _canon(src_raw)
        target = _canon(tgt_raw)

        # Dedupe on (source, relation, target), case-insensitive (not a "drop").
        dedupe_key = (source.lower(), relation, target.lower())
        if dedupe_key in seen_edges:
            continue

        # Evidence-URL set-membership filter (HONEST): blank anything not real.
        evidence_url = str(raw.get("evidence_url") or "").strip()
        if evidence_url and evidence_url not in allowed:
            evidence_url = ""
        year = _valid_year(raw.get("year"))

        # Capacity caps — count what we drop so the UI can disclose truncation.
        if len(out) >= MAX_EDGES:
            dropped += 1
            continue
        new_nodes = {n for n in (source.lower(), target.lower()) if n not in node_set}
        if len(node_set) + len(new_nodes) > MAX_NODES:
            dropped += 1
            continue

        seen_edges.add(dedupe_key)
        node_set.update(new_nodes)
        out.append({
            "source": source,
            "target": target,
            "relation": relation,
            "year": year,
            "evidence_url": evidence_url,
        })

    return out, dropped


# ===========================================================================
# Node typing + graph building
# ===========================================================================
def _assign_node_types(seed: str, edges: List[Dict[str, str]]) -> Dict[str, str]:
    """Map each node name -> a type key in TYPE_COLORS.

    The seed always gets type "seed". Every other node's type is inferred from
    the FIRST relation that touches it (source/target role aware). Ties go to
    the first edge seen, which keeps coloring deterministic.
    """
    types: Dict[str, str] = {}
    seed_l = seed.lower()
    types[seed] = "seed"

    for e in edges:
        src, tgt, rel = e["source"], e["target"], e["relation"]
        if src.lower() != seed_l and src not in types:
            types[src] = _RELATION_SOURCE_TYPE.get(rel, "other")
        if tgt.lower() != seed_l and tgt not in types:
            types[tgt] = _RELATION_TARGET_TYPE.get(rel, "other")
        # Never let an edge override the seed's own color.
        if src.lower() == seed_l:
            types[src] = "seed"
        if tgt.lower() == seed_l:
            types[tgt] = "seed"
    return types


def _edge_label(relation: str, year: str) -> str:
    """Human edge label: relation (+ year)."""
    label = relation.replace("_", " ")
    if year:
        label += f" ({year})"
    return label


def build_pyvis_html(seed: str, edges: List[Dict[str, str]]) -> Optional[str]:
    """Build a self-contained pyvis HTML string for the relationship graph.

    Returns the HTML string, or ``None`` if the graph libs are unavailable (the
    caller then falls back to graphviz / a table). Uses
    ``net.generate_html()`` — NOT ``net.show()`` — so nothing is written to a
    temp file; CDN resources are in-lined for a fully self-contained embed.
    """
    if not GRAPH_LIBS_AVAILABLE or nx is None or Network is None:
        return None

    types = _assign_node_types(seed, edges)

    # networkx.DiGraph first (per spec), then hand its nodes/edges to pyvis.
    g = nx.DiGraph()
    for node, ntype in types.items():
        g.add_node(node, group=ntype)
    for e in edges:
        g.add_edge(
            e["source"], e["target"],
            relation=e["relation"], year=e["year"],
        )

    net = Network(
        height="620px",
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
        bgcolor="#ffffff",
        font_color="#0f172a",
    )

    for node in g.nodes():
        ntype = types.get(node, "other")
        color = TYPE_COLORS.get(ntype, TYPE_COLORS["other"])
        net.add_node(
            node,
            label=node,
            color=color,
            title=f"{node} — {ntype}",
            shape="dot",
            size=24 if ntype == "seed" else 16,
        )
    for e in edges:
        lbl = _edge_label(e["relation"], e["year"])
        has_ev = bool(e.get("evidence_url"))
        edge_kwargs = {
            "title": lbl if has_ev else f"{lbl} — no cited source",
            "label": lbl,
            "arrows": "to",
        }
        if not has_ev:
            # Uncited relationships are drawn dashed + grey so the GRAPH itself
            # (the primary surface), not just the table, shows what is source-backed.
            edge_kwargs["dashes"] = True
            edge_kwargs["color"] = "#cbd5e1"
        net.add_edge(e["source"], e["target"], **edge_kwargs)

    # Light physics so the layout settles quickly without thrashing.
    try:
        net.barnes_hut(
            gravity=-8000, central_gravity=0.3,
            spring_length=160, spring_strength=0.02, damping=0.09,
        )
    except Exception as e:  # pragma: no cover - never let physics tuning break render
        logger.debug("barnes_hut tuning failed: %s", e)

    try:
        return net.generate_html()
    except Exception as e:
        logger.error("pyvis generate_html failed: %s", e)
        return None


def _legend_html() -> str:
    """A small inline color legend matching the node-type palette."""
    order = ["seed", "subsidiary", "parent", "investor", "partner", "competitor"]
    chips = []
    for t in order:
        chips.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'margin-right:14px;font-size:.82rem;color:#475569;">'
            f'<span style="width:12px;height:12px;border-radius:50%;'
            f'background:{TYPE_COLORS[t]};display:inline-block;"></span>{t}</span>'
        )
    return (
        '<div style="margin:2px 0 10px;display:flex;flex-wrap:wrap;'
        'align-items:center;">' + "".join(chips) + "</div>"
    )


# ===========================================================================
# Fallback render (pyvis/networkx missing)
# ===========================================================================
def _render_graphviz_fallback(seed: str, edges: List[Dict[str, str]]) -> None:
    """Honest fallback when pyvis/networkx are unavailable.

    Tries ``st.graphviz_chart`` (DOT string built by hand — no networkx needed);
    if even that fails, shows a plain edge table so the data is never hidden.
    """
    st.error(
        "Interactive graph libraries (pyvis/networkx) are unavailable "
        f"({html.escape(_GRAPH_IMPORT_ERROR) or 'import failed'}). "
        "Showing a static fallback instead."
    )
    if not edges:
        st.info("No relationships to display.")
        return

    # Build a DOT string by hand (does not import networkx/pyvis).
    def _q(s: str) -> str:
        return '"' + str(s).replace('"', "'") + '"'

    try:
        lines = ["digraph G {", '  rankdir=LR;', '  node [shape=box, style=rounded];']
        types = _assign_node_types(seed, edges) if GRAPH_LIBS_AVAILABLE else {seed: "seed"}
        for e in edges:
            label = _edge_label(e["relation"], e["year"])
            lines.append(f'  {_q(e["source"])} -> {_q(e["target"])} [label={_q(label)}];')
        lines.append("}")
        st.graphviz_chart("\n".join(lines), use_container_width=True)
    except Exception as e:  # pragma: no cover - last-resort table
        logger.warning("graphviz fallback failed: %s", e)
        st.warning("Graphviz unavailable too — showing the edge table below.")


# ===========================================================================
# Relationships section (table + KPIs) — shared by all render paths
# ===========================================================================
def _render_relationships_section(seed: str, edges: List[Dict[str, str]],
                                  sources: List[str], dropped: int = 0) -> None:
    ui.section("Relationships", f"Depth-1 corporate lineage for {seed}")
    if dropped:
        st.caption(
            f"⚠️ Graph capped at {MAX_NODES} entities / {MAX_EDGES} relationships — "
            f"{dropped} additional extracted relationship(s) are not shown."
        )

    n_entities = len({seed.lower()} | {e["source"].lower() for e in edges}
                     | {e["target"].lower() for e in edges})
    n_rel = len(edges)
    n_evidence = sum(1 for e in edges if e.get("evidence_url"))

    ui.kpi_row([
        ("Entities", f"{n_entities}"),
        ("Relationships", f"{n_rel}"),
        ("With evidence", f"{n_evidence}/{n_rel}" if n_rel else "0"),
    ])

    if not edges:
        st.info("No relationships extracted for this company from the available sources.")
        return

    rows = [
        {
            "Source": e["source"],
            "Relation": e["relation"].replace("_", " "),
            "Target": e["target"],
            "Year": e["year"] or "",
            "Evidence": e["evidence_url"] or "",
        }
        for e in edges
    ]
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Source": st.column_config.TextColumn("Source", width="medium"),
            "Relation": st.column_config.TextColumn("Relation", width="small"),
            "Target": st.column_config.TextColumn("Target", width="medium"),
            "Year": st.column_config.TextColumn("Year", width="small"),
            "Evidence": st.column_config.LinkColumn(
                "Evidence", display_text="source", width="small"
            ),
        },
    )


# ===========================================================================
# Public entry point
# ===========================================================================
def render_relationship_graph_tab() -> None:
    """Render the Relationship Graph tab. The ONLY public symbol of this module."""
    ui.inject_css()
    ui.hero(
        "Relationship Graph",
        "Map a company's corporate lineage — parents, subsidiaries, "
        "acquisitions, investors, partners, and competitors — from "
        "source-backed evidence.",
        chips=[
            "<b>LinkUp</b> sourcedAnswer",
            "<b>Gemini</b> edge extraction",
            "depth 1 · bounded",
        ],
    )

    # --- HONEST-STATUS gate: required keys must be present ------------------
    ok, missing = feature_available(REQUIRED_KEYS)
    if not ok:
        render_gemini_health()
        if "LINKUP_API_KEY" in missing:
            st.warning(
                "⚠️ LinkUp is not configured — set `LINKUP_API_KEY` in secrets. "
                "Relationship extraction needs source-backed search and is "
                "disabled until then."
            )
        st.caption(f"Missing secrets: {', '.join(missing)}")
        return

    # --- INPUT -------------------------------------------------------------
    mode = st.segmented_control(
        "Seed source",
        options=["From a company", "From my last List run"],
        default="From a company",
        key=_PFX + "mode",
    )

    seed = ""
    if mode == "From my last List run":
        candidates = _seed_candidates_from_list_run()
        if not candidates:
            st.info(
                "No List Intelligence run found in this session. Run the "
                "**List Intelligence** tab first, then come back here to graph "
                "any company from that run."
            )
        else:
            seed = st.selectbox(
                "Company (from your last List Intelligence run)",
                options=candidates,
                key=_PFX + "list_pick",
            )
    else:
        seed = st.text_input(
            "Seed company",
            value=st.session_state.get(SS_SEED, ""),
            placeholder="e.g. Anthropic, Stripe, NVIDIA",
            key=_PFX + "seed_input",
        )
    seed = _clean(seed)

    run_clicked = st.button(
        "Build relationship graph",
        type="primary",
        disabled=(not seed) or bool(st.session_state.get(SS_RUNNING)),
        key=_PFX + "run_btn",
    )

    # --- EXECUTION ---------------------------------------------------------
    if run_clicked and seed:
        st.session_state[SS_RUNNING] = True
        try:
            with st.status(f"Mapping relationships for {seed}…", expanded=True) as status:
                st.write("Querying LinkUp for source-backed relationships…")
                result = run_async(extract_relationship_edges(seed))
                edges = result.get("edges", [])
                sources = result.get("sources", [])
                st.write(
                    f"Extracted {len(edges)} relationship(s) from "
                    f"{len(sources)} source(s)."
                )
                if not result.get("success"):
                    status.update(label="Extraction incomplete", state="error")
                    st.error(result.get("error") or "Extraction failed.")
                else:
                    status.update(
                        label=f"Built graph: {len(edges)} relationships", state="complete"
                    )
            # Persist the latest result (bounded session history).
            st.session_state[SS_SEED] = seed
            st.session_state[SS_EDGES] = edges
            st.session_state[SS_SOURCES] = sources
            st.session_state[SS_DROPPED] = int(result.get("dropped") or 0)
            push_history(SS_HISTORY, {"seed": seed, "n_edges": len(edges)})
            if result.get("success"):
                st.toast(f"Relationship graph ready for {seed} "
                         f"({len(edges)} relationships).", icon="🕸️")
        finally:
            st.session_state[SS_RUNNING] = False

    # --- RENDER (latest result in session) ---------------------------------
    edges: List[Dict[str, str]] = st.session_state.get(SS_EDGES) or []
    render_seed: str = st.session_state.get(SS_SEED) or seed
    sources: List[str] = st.session_state.get(SS_SOURCES) or []
    dropped: int = int(st.session_state.get(SS_DROPPED) or 0)

    if render_seed and edges:
        if GRAPH_LIBS_AVAILABLE:
            graph_html = build_pyvis_html(render_seed, edges)
            if graph_html:
                st.html(_legend_html())
                # Iframe taller than the 620px canvas (+ body margins) so nothing
                # clips; scrolling off avoids the wheel-zoom-vs-page-scroll fight.
                components.html(graph_html, height=700, scrolling=False)
            else:
                _render_graphviz_fallback(render_seed, edges)
        else:
            _render_graphviz_fallback(render_seed, edges)

        _render_relationships_section(render_seed, edges, sources, dropped)
    elif render_seed:
        st.info(
            f"No relationships are stored for {render_seed} yet. "
            "Click **Build relationship graph** to extract them."
        )

    # --- Token usage (bottom) ----------------------------------------------
    with st.expander("Token usage & cost", expanded=False):
        render_token_usage(inside_expander=True)


__all__ = ["render_relationship_graph_tab"]
