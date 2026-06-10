"""features/financials.py
=========================

ParselyFi "Adjusted-EBITDA Bridge" tab — a finance-grade reconciliation where
the LLM ONLY proposes line items + adjustments and a HARD-CODED Python function
does ALL the arithmetic (auditable, never hallucinated).

Public entry point (the ONLY thing the main app calls, inside an
already-created tab)::

    render_financials_tab() -> None

CORE PRINCIPLE (the whole point)
--------------------------------
The model NEVER computes a total. It extracts/normalizes raw inputs and proposes
an adjustments list; a pure Python function (``compute_bridge``) computes
NI -> EBIT -> EBITDA -> Adjusted EBITDA. Every displayed number is either:
  (a) a model-extracted (or user-entered) INPUT — clearly labelled, or
  (b) a Python-COMPUTED subtotal.
The bridge function is the single source of truth: pure, deterministic, and
unit-testable with no LLM in the loop.

Design rules (project CLAUDE.md) baked in
------------------------------------------
- HONEST DATA: the LLM proposes components only; Python computes every subtotal.
  Missing figures -> ``None`` (never guessed). On extraction failure
  ``gemini_generate_json`` returns ``{}`` and we surface an honest
  "extraction unavailable" state — no fabricated numbers, no fake EBITDA margin.
- DEGRADES GRACEFULLY: the compute is pure Python, so MANUAL ENTRY works with NO
  API key. The Gemini gate only guards the EXTRACT step. Every optional viz lib
  (plotly / pyvis / networkx) is imported under try/except behind a ``*_AVAILABLE``
  flag so ``import features.financials`` ALWAYS succeeds; each has an honest
  in-UI fallback.
- BOUNDED: adjustments capped (MAX_ADJUSTMENTS), pasted text capped before it
  reaches the model (MAX_INPUT_CHARS / MAX_MODEL_CHARS), uploaded files read with
  a byte cap.
- ERROR_BOUNDARY: extraction is wrapped; a failure shows an honest message, never
  a crash and never a faked figure.
- All client/LLM logic is REUSED from ``common.*`` / ``ui.*``; nothing duplicated.

Session prefix: ``fin_``.
"""

from __future__ import annotations

import csv
import html
import io
import logging
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from features import common, ui
from features.common import (
    GEMINI_MODEL,
    feature_available,
    gemini_generate_json,
    push_history,
    render_gemini_health,
    render_token_usage,
    run_async,
)

logger = logging.getLogger("parselyfi.features.financials")

# ---------------------------------------------------------------------------
# Optional viz/data deps (imported under try/except so importing this module —
# and the whole app — never crashes if a lib is missing). Availability is
# reflected through *_AVAILABLE flags + honest in-UI fallbacks.
# ---------------------------------------------------------------------------
try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - pandas is a hard dep in this venv
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

try:
    import plotly.graph_objects as go  # type: ignore
    PLOTLY_AVAILABLE = True
    _PLOTLY_IMPORT_ERROR = ""
except Exception as _e:  # pragma: no cover - exercised only when plotly missing
    go = None  # type: ignore
    PLOTLY_AVAILABLE = False
    _PLOTLY_IMPORT_ERROR = str(_e)

try:
    import networkx as nx  # type: ignore
    from pyvis.network import Network  # type: ignore
    PYVIS_AVAILABLE = True
    _PYVIS_IMPORT_ERROR = ""
except Exception as _e:  # pragma: no cover - exercised only when deps missing
    nx = None  # type: ignore
    Network = None  # type: ignore
    PYVIS_AVAILABLE = False
    _PYVIS_IMPORT_ERROR = str(_e)

import streamlit.components.v1 as components

# ===========================================================================
# Constants / bounded-memory caps (all fin_-prefixed session keys)
# ===========================================================================
_PFX = "fin_"
SS_RAW_TEXT = _PFX + "raw_text"            # last pasted/uploaded source text
SS_FIGURES = _PFX + "figures"              # dict — last extracted/edited core figures
SS_ADJ = _PFX + "adjustments"              # list[dict] — last adjustments table rows
SS_EXTRACTED = _PFX + "extracted"          # bool — has an extraction been attempted
SS_EXTRACT_OK = _PFX + "extract_ok"        # bool — did the last extraction return data
SS_EXTRACT_ERR = _PFX + "extract_err"      # str — honest error/empty reason
SS_RUNNING = _PFX + "running"              # guard against double-submit
SS_HISTORY = _PFX + "history"              # bounded via common.push_history

# Hard caps so nothing the user / LLM / a file can supply blows up memory.
MAX_ADJUSTMENTS: int = 30          # cap rows in the adjustments table
MAX_INPUT_CHARS: int = 60_000      # cap the text_area / file text we hold in session
MAX_MODEL_CHARS: int = 40_000      # cap the slice of text we feed the model
MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024   # 5MB bounded read on an uploaded file
MAX_LABEL_CHARS: int = 80          # cap an adjustment label length

REQUIRED_KEYS = ["GEMINI_API_KEY"]   # the EXTRACT step only — manual entry needs no key

# The core income-statement lines the model extracts (never computes). Order is
# the canonical display order. ``revenue`` is optional and ONLY used to compute
# an EBITDA margin when present (else the margin KPI is omitted — never faked).
CORE_FIELDS: List[Tuple[str, str]] = [
    ("net_income", "Net income"),
    ("interest_expense", "Interest expense"),
    ("income_taxes", "Income taxes"),
    ("depreciation", "Depreciation"),
    ("amortization", "Amortization"),
    ("revenue", "Revenue (optional — for margin only)"),
]
_CORE_KEYS = {k for k, _ in CORE_FIELDS}

# Rounding precision for displayed money. Inputs are extracted as-is; subtotals
# are computed in Decimal then quantized to cents for display.
_CENTS = Decimal("0.01")


# ===========================================================================
# Numeric helpers (pure, defensive — used by the compute + the UI)
# ===========================================================================
# A strictly-numeric token: optional sign, optional thousands grouping, optional
# single decimal part. Used to REJECT (not silently mangle) ambiguous figures.
_NUM_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^[+-]?\d+(?:\.\d+)?$")


def _to_decimal(value: Any) -> Optional[Decimal]:
    """Coerce a model/user-supplied value to a FINITE ``Decimal`` (or ``None``).

    Returns ``None`` for missing/blank/unparseable/non-finite values (HONEST: a
    missing figure stays missing — never silently becomes 0; the *compute*
    decides how to treat a None and flags it "not provided"). For strings it
    tolerates a leading currency symbol, thousands separators, and parenthesized
    negatives ("(1,234)" -> -1234), but REJECTS anything ambiguous — unit
    suffixes ("$1.2M"), scientific notation ("1e9"), percents ("50%"), or weird
    grouping ("1,2,3") become ``None`` rather than a fabricated, wrong number,
    because in an auditable bridge a silently mis-scaled figure is worse than a
    rejected one.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # guard: bool is an int subclass
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None  # reject NaN / +Inf / -Inf
        try:
            d = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
        return d if d.is_finite() else None
    s = str(value).strip()
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    # Drop a leading currency symbol and any internal spaces, then validate
    # strictly. Letters (unit suffixes / sci-notation 'e' / "USD") or a failed
    # numeric match -> reject, so we never coerce "$1.2M" into 1.2.
    s = s.lstrip("$€£¥").strip().replace(" ", "")
    if not s or any(ch.isalpha() for ch in s) or not _NUM_RE.match(s):
        return None
    try:
        d = Decimal(s.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None
    return -d if neg else d


def _fmt_money(value: Optional[Decimal], currency: str = "") -> str:
    """Format a Decimal as a thousands-separated money string with a sign.

    ``None`` renders as an em-dash so "not provided" is visible, not faked as 0.
    """
    if value is None:
        return "—"
    # Non-finite (NaN/Inf) must never render as an authoritative money string.
    try:
        if not value.is_finite():
            return "—"
    except (AttributeError, InvalidOperation):
        return "—"
    try:
        q = value.quantize(_CENTS)
    except (InvalidOperation, ValueError):
        q = value
    sign = "-" if q < 0 else ""
    q_abs = -q if q < 0 else q
    body = f"{q_abs:,.2f}"
    prefix = (currency + " ") if currency else ""
    return f"{sign}{prefix}{body}"


def _clean_currency(raw: Any) -> str:
    """Bound + sanitize a currency code/symbol-or-scale label for display."""
    s = str(raw or "").strip()
    return s[:24]   # roomy enough for "USD thousands" etc. (still BOUNDED)


def _clean_label(raw: Any) -> str:
    s = str(raw or "").strip()
    return s[:MAX_LABEL_CHARS]


# ===========================================================================
# THE SOURCE OF TRUTH: pure, deterministic bridge computation (NO LLM)
# ===========================================================================
def compute_bridge(
    figures: Dict[str, Any],
    adjustments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute the Adjusted-EBITDA bridge — the auditable source of truth.

    PURE FUNCTION. No LLM, no I/O, no Streamlit. Given model-extracted (or
    user-entered) raw ``figures`` and a list of proposed ``adjustments``, it
    computes, in order::

        NI
        + Interest expense
        + Income taxes        => EBIT
        + Depreciation
        + Amortization        => EBITDA
        +/- each adjustment   => Adjusted EBITDA

    Arithmetic is done in ``Decimal`` for exactness. A ``None`` input is treated
    as ``0`` for the math but flagged (``provided=False``) so the UI can show
    "not provided" rather than implying the company reported a 0.

    Parameters
    ----------
    figures:
        Mapping with optional keys ``net_income``, ``interest_expense``,
        ``income_taxes``, ``depreciation``, ``amortization``, ``revenue``
        (each a number / numeric-string / ``None``).
    adjustments:
        List of dicts ``{"label", "amount", "add_back": bool, "rationale"}``.
        ``add_back=True`` ADDS the (abs) amount; ``add_back=False`` SUBTRACTS it.
        A signed amount is respected too: the sign convention is
        ``signed = +abs(amount) if add_back else -abs(amount)`` so the UI toggle
        is unambiguous and the displayed +/- always matches the running total.

    Returns
    -------
    dict with::

        {
          "steps": [ {kind, key, component, sign, amount(Decimal|None),
                      signed(Decimal), running(Decimal), provided(bool),
                      is_subtotal(bool), rationale(str)} , ... ],
          "ebit": Decimal, "ebitda": Decimal, "adjusted_ebitda": Decimal,
          "revenue": Decimal|None, "ebitda_margin": Decimal|None,  # None => omit, never fake
        }
    """
    def d(key: str) -> Tuple[Decimal, bool]:
        """Return (value_for_math, provided_flag) for a core figure."""
        v = _to_decimal(figures.get(key))
        if v is None:
            return Decimal("0"), False
        return v, True

    ni, ni_p = d("net_income")
    interest, interest_p = d("interest_expense")
    taxes, taxes_p = d("income_taxes")
    dep, dep_p = d("depreciation")
    amort, amort_p = d("amortization")
    revenue = _to_decimal(figures.get("revenue"))

    steps: List[Dict[str, Any]] = []
    running = Decimal("0")

    def add_step(kind: str, key: str, component: str, signed: Optional[Decimal],
                 provided: bool, *, is_subtotal: bool = False,
                 rationale: str = "") -> None:
        nonlocal running
        if signed is not None:
            running = running + signed
        steps.append({
            "kind": kind,                  # "start" | "addback" | "subtotal" | "adjustment"
            "key": key,
            "component": component,
            "sign": "" if signed is None else ("+" if signed > 0 else ("-" if signed < 0 else "")),
            "amount": (abs(signed) if signed is not None else None),
            "signed": signed if signed is not None else Decimal("0"),
            "running": running,
            "provided": provided,
            "is_subtotal": is_subtotal,
            "rationale": rationale,
        })

    # --- NI -> EBIT ---------------------------------------------------------
    add_step("start", "net_income", "Net income", ni, ni_p)
    add_step("addback", "interest_expense", "+ Interest expense", interest, interest_p)
    add_step("addback", "income_taxes", "+ Income taxes", taxes, taxes_p)
    ebit = running
    add_step("subtotal", "ebit", "= EBIT", None, True, is_subtotal=True)
    # Patch the subtotal's running to the live value (signed=None left it as the
    # prior running, which already equals EBIT — keep it explicit/honest).
    steps[-1]["running"] = ebit

    # --- EBIT -> EBITDA -----------------------------------------------------
    add_step("addback", "depreciation", "+ Depreciation", dep, dep_p)
    add_step("addback", "amortization", "+ Amortization", amort, amort_p)
    ebitda = running
    add_step("subtotal", "ebitda", "= EBITDA", None, True, is_subtotal=True)
    steps[-1]["running"] = ebitda

    # --- EBITDA -> Adjusted EBITDA (apply each proposed adjustment) ---------
    for adj in (adjustments or [])[:MAX_ADJUSTMENTS]:
        if not isinstance(adj, dict):
            continue
        label = _clean_label(adj.get("label")) or "Adjustment"
        amt = _to_decimal(adj.get("amount"))
        if amt is None:
            # An adjustment with no parseable amount contributes nothing but is
            # still shown (provided=False) so the user sees it needs a number.
            add_step("adjustment", "adj", label, None, False,
                     rationale=_clean_label(adj.get("rationale")))
            continue
        add_back = bool(adj.get("add_back", True))
        signed = abs(amt) if add_back else -abs(amt)
        add_step("adjustment", "adj", label, signed, True,
                 rationale=str(adj.get("rationale") or "")[:240])

    adjusted = running
    add_step("subtotal", "adjusted_ebitda", "= Adjusted EBITDA", None, True,
             is_subtotal=True)
    steps[-1]["running"] = adjusted

    # --- EBITDA margin: ONLY for a finite, strictly-positive revenue (never faked) -
    ebitda_margin: Optional[Decimal] = None
    if (revenue is not None and revenue.is_finite() and revenue > 0
            and ebitda is not None and ebitda.is_finite()):
        ebitda_margin = (ebitda / revenue) * Decimal("100")

    return {
        "steps": steps,
        "ebit": ebit,
        "ebitda": ebitda,
        "adjusted_ebitda": adjusted,
        "revenue": revenue,
        "ebitda_margin": ebitda_margin,
    }


# ===========================================================================
# Extraction (LLM PROPOSES raw figures + adjustments — does NOT compute totals)
# ===========================================================================
def _build_extraction_prompt(text: str) -> str:
    """Prompt the model to return ONLY raw components + proposed adjustments.

    The prompt EXPLICITLY forbids any computed EBIT/EBITDA/total — the model
    supplies inputs; Python computes the bridge.
    """
    snippet = text[:MAX_MODEL_CHARS]
    return f"""
You are a financial-data EXTRACTION tool. From the income-statement / financial
text below, extract ONLY the raw reported figures and PROPOSE add-back
adjustments. You MUST NOT compute or return any total, subtotal, EBIT, EBITDA,
operating income, or "adjusted" figure — a separate auditable program computes
those. You provide COMPONENTS ONLY.

--- FINANCIAL TEXT ---
{snippet}
--- END TEXT ---

Return ONLY a single JSON object of EXACTLY this shape:
{{
  "currency": "<ISO code or symbol if stated, else empty string>",
  "period": "<reporting period if stated, e.g. 'FY2024' or 'Q3 2024', else empty>",
  "net_income": <number or null>,
  "interest_expense": <number or null>,
  "income_taxes": <number or null>,
  "depreciation": <number or null>,
  "amortization": <number or null>,
  "revenue": <number or null>,
  "adjustments": [
    {{"label": "<short name>", "amount": <number>, "add_back": true, "rationale": "<why>"}}
  ]
}}

STRICT RULES:
- Output JSON ONLY, no prose, no markdown fences.
- Every figure is the RAW reported number (positive expense values, e.g.
  interest_expense and income_taxes are positive add-back magnitudes). If a
  figure is NOT stated in the text, use null — NEVER guess or infer it.
- DO NOT include EBIT, EBITDA, operating income, or any total/subtotal field.
- "adjustments" are PROPOSED non-recurring / non-cash add-backs you can justify
  from the text (e.g. stock-based comp, restructuring, impairment, litigation,
  one-time gains). "add_back": true means ADD it to EBITDA; false means SUBTRACT
  it (e.g. a one-time GAIN). "amount" is a positive magnitude. If none are
  evident, return an empty list. Do not invent adjustments not supported by the
  text.
- "revenue" only if total revenue / net sales is explicitly stated, else null.
"""


async def extract_figures(text: str) -> Dict[str, Any]:
    """Run ONE Gemini JSON pass to extract raw figures + proposed adjustments.

    HONEST STATUS: ``common.gemini_generate_json`` returns ``{}`` on any failure
    (missing client, timeout, parse error) and never raises. We mirror that:
    returns ``{"ok": bool, "figures": {...}, "adjustments": [...],
    "currency": str, "period": str, "error": Optional[str]}``. The model is
    forbidden from returning computed totals; we additionally STRIP any such key
    defensively so a hallucinated total can never reach the UI.
    """
    text = (text or "").strip()
    if not text:
        return {"ok": False, "figures": {}, "adjustments": [], "currency": "",
                "period": "", "error": "No financial text provided."}

    prompt = _build_extraction_prompt(text)
    result = await gemini_generate_json(
        prompt, model=GEMINI_MODEL, agent_name="Financials: EBITDA bridge",
        temperature=0.0,
    )

    # HONEST: {} => extraction unavailable (timeout / blocked / no key / parse-fail).
    if not isinstance(result, dict) or not result:
        return {
            "ok": False, "figures": {}, "adjustments": [], "currency": "",
            "period": "",
            "error": ("Extraction unavailable — the model returned no data "
                      "(missing key, timeout, blocked, or unparseable). You can "
                      "still enter the figures manually below."),
        }

    figures: Dict[str, Any] = {}
    for key in _CORE_KEYS:
        # Keep raw value; _to_decimal() normalizes at compute/display time. A
        # missing/None key stays missing (never coerced to a guessed number).
        figures[key] = result.get(key, None)

    raw_adj = result.get("adjustments")
    adjustments = _normalize_adjustments(raw_adj)

    return {
        "ok": True,
        "figures": figures,
        "adjustments": adjustments,
        "currency": _clean_currency(result.get("currency")),
        "period": str(result.get("period") or "")[:60],
        "error": None,
    }


def _normalize_adjustments(raw: Any) -> List[Dict[str, Any]]:
    """Validate + bound a proposed adjustments list (cap MAX_ADJUSTMENTS)."""
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = _clean_label(item.get("label"))
        if not label:
            continue
        amt = _to_decimal(item.get("amount"))
        out.append({
            "label": label,
            "amount": float(amt) if amt is not None else None,
            "add_back": bool(item.get("add_back", True)),
            "rationale": str(item.get("rationale") or "")[:240],
        })
        if len(out) >= MAX_ADJUSTMENTS:
            break
    return out


# ===========================================================================
# File parsing (csv / xlsx / txt) -> plain text, bounded
# ===========================================================================
def _read_uploaded_file(uploaded) -> Tuple[str, Optional[str]]:
    """Read an uploaded csv/xlsx/txt into bounded plain text.

    Returns ``(text, error)``. HONEST: on an unreadable/oversized file returns
    ``("", reason)`` — never fabricates content.
    """
    try:
        raw = uploaded.getvalue()
    except Exception as e:  # pragma: no cover - defensive
        return "", f"Could not read upload: {e}"

    if raw is None:
        return "", "Empty upload."
    if len(raw) > MAX_UPLOAD_BYTES:
        return "", (f"File too large ({len(raw):,} bytes) — cap is "
                    f"{MAX_UPLOAD_BYTES:,} bytes.")

    name = (getattr(uploaded, "name", "") or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            if not PANDAS_AVAILABLE:
                return "", "pandas/openpyxl unavailable — cannot read spreadsheet."
            df = pd.read_excel(io.BytesIO(raw))  # type: ignore[union-attr]
            return df.to_csv(index=False)[:MAX_INPUT_CHARS], None
        if name.endswith(".csv"):
            if PANDAS_AVAILABLE:
                df = pd.read_csv(io.BytesIO(raw))  # type: ignore[union-attr]
                return df.to_csv(index=False)[:MAX_INPUT_CHARS], None
            # Fallback: decode + re-serialize via csv (no pandas needed).
            text = raw.decode("utf-8", errors="replace")
            return text[:MAX_INPUT_CHARS], None
        # txt / anything else -> decode as text.
        return raw.decode("utf-8", errors="replace")[:MAX_INPUT_CHARS], None
    except Exception as e:
        return "", f"Failed to parse file: {e}"


# ===========================================================================
# Visualizations (all guarded; honest fallbacks)
# ===========================================================================
def _waterfall_figure(bridge: Dict[str, Any], currency: str):
    """Build a plotly go.Waterfall for the bridge, or None if plotly missing."""
    if not PLOTLY_AVAILABLE or go is None:
        return None
    measures: List[str] = []
    labels: List[str] = []
    values: List[float] = []
    text: List[str] = []
    for s in bridge["steps"]:
        if s["kind"] == "start":
            measures.append("absolute")
            labels.append("Net income")
            values.append(float(s["signed"]))
            text.append(_fmt_money(s["running"], currency))
        elif s["is_subtotal"]:
            measures.append("total")
            labels.append(s["component"].replace("= ", ""))
            values.append(float(s["running"]))
            text.append(_fmt_money(s["running"], currency))
        else:
            measures.append("relative")
            labels.append(s["component"].lstrip("+ ").strip())
            values.append(float(s["signed"]))
            text.append(_fmt_money(s["signed"], currency))
    try:
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            text=text,
            textposition="outside",
            connector={"line": {"color": ui.LINE}},
            increasing={"marker": {"color": ui.GREEN}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": ui.BLUE}},
        ))
        fig.update_layout(
            title="Adjusted-EBITDA Bridge (Python-computed subtotals)",
            showlegend=False,
            margin=dict(t=50, b=40, l=10, r=10),
            height=460,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color=ui.INK),
        )
        return fig
    except Exception as e:  # pragma: no cover - never let chart tuning crash render
        logger.warning("plotly waterfall build failed: %s", e)
        return None


def _build_dag_html(bridge: Dict[str, Any], currency: str) -> Optional[str]:
    """Build a self-contained pyvis HTML DAG of the bridge, or None if missing.

    Nodes: NI, EBIT, EBITDA, Adjusted EBITDA (+ one node per adjustment).
    Directed edges are labelled with the SIGNED amount that produced each
    subtotal. Mirrors relationship_graph: Network(notebook=False,
    cdn_resources="in_line") + net.generate_html() (NEVER net.show()).
    """
    if not PYVIS_AVAILABLE or nx is None or Network is None:
        return None

    try:
        g = nx.DiGraph()  # noqa: F841 - parallels relationship_graph's build order

        net = Network(
            height="420px", width="100%", directed=True, notebook=False,
            cdn_resources="in_line", bgcolor="#ffffff", font_color="#0f172a",
        )

        subtotal_color = {"ebit": ui.MUTE, "ebitda": ui.BLUE,
                          "adjusted_ebitda": ui.GREEN}
        # Core subtotal chain nodes.
        chain = [
            ("net_income", "Net income"),
            ("ebit", "EBIT"),
            ("ebitda", "EBITDA"),
            ("adjusted_ebitda", "Adjusted EBITDA"),
        ]
        running_by_key = {s["key"]: s["running"] for s in bridge["steps"]
                          if s.get("is_subtotal") or s["kind"] == "start"}
        for key, label in chain:
            running = running_by_key.get(key)
            val = _fmt_money(running, currency) if running is not None else ""
            net.add_node(
                key, label=f"{label}\n{val}", shape="box",
                color=subtotal_color.get(key, ui.GREEN), font={"color": "#0f172a"},
                title=html.escape(f"{label}: {val}"),   # currency is user-text -> escape
            )

        # Edge NI->EBIT labelled with (interest+taxes), EBIT->EBITDA with (dep+amort).
        def _sum_signed(keys: set) -> Decimal:
            tot = Decimal("0")
            for s in bridge["steps"]:
                if s["key"] in keys:
                    tot += s["signed"]
            return tot

        ni_ebit = _sum_signed({"interest_expense", "income_taxes"})
        ebit_ebitda = _sum_signed({"depreciation", "amortization"})
        net.add_edge("net_income", "ebit", label=_fmt_money(ni_ebit, currency),
                     arrows="to", title="+ interest + taxes")
        net.add_edge("ebit", "ebitda", label=_fmt_money(ebit_ebitda, currency),
                     arrows="to", title="+ depreciation + amortization")

        # One node per adjustment, edge into Adjusted EBITDA with the signed amt.
        adj_idx = 0
        for s in bridge["steps"]:
            if s["kind"] != "adjustment":
                continue
            adj_idx += 1
            node_id = f"adj_{adj_idx}"
            signed = s["signed"]
            net.add_node(
                # label/title carry LLM/user text -> escape before it reaches the
                # pyvis tooltip innerHTML sink (prevents markup/script injection).
                node_id, label=html.escape(str(s["component"]))[:60], shape="dot", size=14,
                color=(ui.GREEN if signed >= 0 else "#ef4444"),
                title=html.escape(str(s.get("rationale") or s["component"])),
            )
            net.add_edge(node_id, "adjusted_ebitda",
                         label=_fmt_money(signed, currency), arrows="to")
        # Also connect EBITDA -> Adjusted EBITDA as the spine.
        net.add_edge("ebitda", "adjusted_ebitda", arrows="to", color="#cbd5e1",
                     dashes=True)

        try:
            net.barnes_hut(gravity=-6000, central_gravity=0.3, spring_length=140,
                           spring_strength=0.02, damping=0.09)
        except Exception:  # pragma: no cover
            pass
        return net.generate_html()
    except Exception as e:
        logger.error("pyvis DAG build failed: %s", e)
        return None


def _render_dag_graphviz_fallback(bridge: Dict[str, Any], currency: str) -> None:
    """graphviz DOT fallback for the DAG (no pyvis/networkx needed)."""
    def _q(s: str) -> str:
        return '"' + str(s).replace('"', "'") + '"'

    try:
        lines = ["digraph G {", "  rankdir=LR;",
                 "  node [shape=box, style=rounded];"]
        running_by_key = {s["key"]: s["running"] for s in bridge["steps"]
                          if s.get("is_subtotal") or s["kind"] == "start"}
        for key, label in [("net_income", "Net income"), ("ebit", "EBIT"),
                           ("ebitda", "EBITDA"),
                           ("adjusted_ebitda", "Adjusted EBITDA")]:
            r = running_by_key.get(key)
            lines.append(
                f'  {key} [label={_q(label + " = " + _fmt_money(r, currency))}];')
        lines.append(f'  net_income -> ebit;')
        lines.append(f'  ebit -> ebitda;')
        lines.append(f'  ebitda -> adjusted_ebitda;')
        adj_idx = 0
        for s in bridge["steps"]:
            if s["kind"] != "adjustment":
                continue
            adj_idx += 1
            nid = f"adj_{adj_idx}"
            lines.append(f'  {nid} [label={_q(s["component"])}, shape=ellipse];')
            lines.append(
                f'  {nid} -> adjusted_ebitda '
                f'[label={_q(_fmt_money(s["signed"], currency))}];')
        lines.append("}")
        st.graphviz_chart("\n".join(lines), use_container_width=True)
    except Exception as e:  # pragma: no cover - last resort handled by caller
        logger.warning("graphviz DAG fallback failed: %s", e)
        st.info("Graph view unavailable — see the step table above.")


# ===========================================================================
# CSV export
# ===========================================================================
def _bridge_to_csv(bridge: Dict[str, Any], currency: str, period: str) -> str:
    """Serialize the bridge steps to CSV text for download."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Adjusted-EBITDA Bridge"])
    if period:
        writer.writerow(["Period", period])
    if currency:
        writer.writerow(["Currency", currency])
    writer.writerow([])
    writer.writerow(["Step", "Component", "Provided", "Sign",
                     "Amount", "Running subtotal"])
    for s in bridge["steps"]:
        amt = "" if s["amount"] is None else f"{s['amount']:.2f}"
        provided = "computed" if s["is_subtotal"] else (
            "yes" if s["provided"] else "not provided")
        sign = "" if s["is_subtotal"] else s["sign"]
        writer.writerow([
            "SUBTOTAL" if s["is_subtotal"] else s["kind"],
            s["component"], provided, sign, amt, f"{s['running']:.2f}",
        ])
    writer.writerow([])
    writer.writerow(["Note", "Figures are model-extracted/user-entered inputs; "
                     "all subtotals are computed in Python (no LLM arithmetic)."])
    return buf.getvalue()


# ===========================================================================
# Step-table rendering (the auditable core surface)
# ===========================================================================
def _render_step_table(bridge: Dict[str, Any], currency: str) -> None:
    """Render the Step / Component / +/- Amount / Running subtotal table."""
    rows = []
    for s in bridge["steps"]:
        if s["is_subtotal"]:
            provenance = "Python-computed"
            amount_disp = ""
        else:
            provenance = "Input" if s["provided"] else "Input (not provided)"
            amount_disp = (f"{s['sign']}{_fmt_money(s['amount'], currency)}"
                           if s["amount"] is not None else "—")
        rows.append({
            "Step": ("SUBTOTAL" if s["is_subtotal"]
                     else {"start": "Start", "addback": "Add-back",
                           "adjustment": "Adjustment"}.get(s["kind"], s["kind"])),
            "Component": s["component"],
            "Provenance": provenance,
            "+/- Amount": amount_disp,
            "Running subtotal": _fmt_money(s["running"], currency),
        })
    st.dataframe(
        rows, use_container_width=True, hide_index=True,
        column_config={
            "Step": st.column_config.TextColumn("Step", width="small"),
            "Component": st.column_config.TextColumn("Component", width="medium"),
            "Provenance": st.column_config.TextColumn("Provenance", width="small"),
            "+/- Amount": st.column_config.TextColumn("+/- Amount", width="small"),
            "Running subtotal": st.column_config.TextColumn(
                "Running subtotal", width="medium"),
        },
    )


def _render_barchart_fallback(bridge: Dict[str, Any]) -> None:
    """Plotly-missing fallback: a clean running-subtotal bar chart."""
    if PANDAS_AVAILABLE:
        data = {s["component"].replace("= ", ""): float(s["running"])
                for s in bridge["steps"]
                if s["is_subtotal"] or s["kind"] == "start"}
        df = pd.DataFrame(  # type: ignore[union-attr]
            {"Running subtotal": list(data.values())}, index=list(data.keys()))
        st.bar_chart(df)
    else:  # pragma: no cover - pandas is present in this venv
        st.info("Install plotly or pandas to see the visual bridge; "
                "the step table above is the auditable source.")


# ===========================================================================
# Public entry point
# ===========================================================================
def render_financials_tab() -> None:
    """Render the Adjusted-EBITDA Bridge tab. The ONLY public symbol here."""
    ui.inject_css()
    ui.hero(
        "Adjusted-EBITDA Bridge",
        "Reconcile Net Income → EBIT → EBITDA → Adjusted EBITDA. The model only "
        "proposes inputs and add-backs; a pure Python function computes every "
        "subtotal — auditable, never hallucinated.",
        chips=[
            "<b>LLM</b> proposes inputs",
            "<b>Python</b> computes totals",
            "manual entry works <b>without a key</b>",
        ],
    )

    # --- INPUT -------------------------------------------------------------
    ui.section("1 · Source", "Paste an income statement or upload a file, "
                             "then extract — or skip straight to manual entry.")

    src_mode = st.radio(
        "Input method",
        options=["Paste text", "Upload file (csv / xlsx / txt)"],
        horizontal=True,
        key=_PFX + "src_mode",
    )

    pasted = ""
    if src_mode == "Paste text":
        pasted = st.text_area(
            "Income-statement / financial text",
            value=st.session_state.get(SS_RAW_TEXT, ""),
            height=180,
            max_chars=MAX_INPUT_CHARS,
            placeholder=("Paste the income statement here… e.g.\n"
                         "Net income 100\nInterest expense 20\nIncome taxes 30\n"
                         "Depreciation 15\nAmortization 5\n"
                         "Stock-based compensation 10 (add-back)"),
            key=_PFX + "text_input",
        )
    else:
        uploaded = st.file_uploader(
            "Upload csv / xlsx / txt",
            type=["csv", "xlsx", "xls", "txt"],
            key=_PFX + "uploader",
        )
        if uploaded is not None:
            text, err = _read_uploaded_file(uploaded)
            if err:
                st.warning(f"⚠️ {err}")
            else:
                pasted = text
                with st.expander("Preview parsed file text", expanded=False):
                    st.text(text[:4000] + ("…" if len(text) > 4000 else ""))

    # Persist source text (bounded).
    if pasted:
        st.session_state[SS_RAW_TEXT] = pasted[:MAX_INPUT_CHARS]

    # HONEST-STATUS: the EXTRACT step needs a Gemini key; manual entry does not.
    ok, missing = feature_available(REQUIRED_KEYS)
    render_gemini_health()

    c_extract, c_clear = st.columns([3, 1])
    extract_clicked = c_extract.button(
        "Extract figures (Gemini proposes inputs only)",
        type="primary",
        disabled=(not ok) or (not pasted) or bool(st.session_state.get(SS_RUNNING)),
        help=("Set GEMINI_API_KEY in secrets to enable extraction."
              if not ok else "The model returns raw figures + proposed add-backs; "
                             "it never computes a total."),
        key=_PFX + "extract_btn",
    )
    if c_clear.button("Reset", key=_PFX + "reset_btn"):
        for k in (SS_FIGURES, SS_ADJ, SS_EXTRACTED, SS_EXTRACT_OK, SS_EXTRACT_ERR, SS_RAW_TEXT):
            st.session_state.pop(k, None)
        # Bump the nonce so the input widgets get fresh keys and actually clear
        # (keyed widgets otherwise restore their old value from session_state).
        st.session_state[_PFX + "nonce"] = st.session_state.get(_PFX + "nonce", 0) + 1
        st.rerun()

    if not ok:
        st.caption(
            "ℹ️ Gemini extraction is disabled (missing "
            f"{', '.join(missing)}). You can still enter every figure manually "
            "below — the bridge is computed in pure Python with no API key.")

    # --- EXTRACTION (LLM proposes; ERROR_BOUNDARY) -------------------------
    if extract_clicked and pasted:
        st.session_state[SS_RUNNING] = True
        try:
            with st.status("Extracting figures (model proposes inputs only)…",
                           expanded=False) as status:
                try:
                    result = run_async(extract_figures(pasted))
                except Exception as e:  # ERROR_BOUNDARY: never crash the tab
                    logger.error("extract_figures crashed: %s", e)
                    result = {"ok": False, "figures": {}, "adjustments": [],
                              "currency": "", "period": "",
                              "error": f"Extraction error: {e}"}
                st.session_state[SS_EXTRACTED] = True
                st.session_state[SS_EXTRACT_OK] = bool(result.get("ok"))
                st.session_state[SS_EXTRACT_ERR] = result.get("error") or ""
                if result.get("ok"):
                    st.session_state[SS_FIGURES] = dict(result.get("figures") or {})
                    st.session_state[SS_FIGURES]["currency"] = result.get("currency", "")
                    st.session_state[SS_FIGURES]["period"] = result.get("period", "")
                    st.session_state[SS_ADJ] = list(result.get("adjustments") or [])
                    # Bump the nonce so the freshly extracted figures re-seed the
                    # input widgets (value= is ignored once a keyed widget has state).
                    st.session_state[_PFX + "nonce"] = (
                        st.session_state.get(_PFX + "nonce", 0) + 1)
                    status.update(label="Figures extracted (you can edit them)",
                                  state="complete")
                    push_history(SS_HISTORY,
                                 {"period": result.get("period", ""),
                                  "n_adj": len(result.get("adjustments") or [])})
                    st.toast("Figures extracted — review & edit, the bridge "
                             "recomputes live.", icon="🧮")
                else:
                    status.update(label="Extraction unavailable", state="error")
        finally:
            st.session_state[SS_RUNNING] = False

    # Honest "extraction unavailable" surface (persists across reruns).
    if st.session_state.get(SS_EXTRACTED) and not st.session_state.get(SS_EXTRACT_OK):
        st.info(st.session_state.get(SS_EXTRACT_ERR)
                or "Extraction unavailable — enter the figures manually below.")

    # --- EDITABLE FIGURES + ADJUSTMENTS (deterministic recompute on edit) --
    ui.section("2 · Inputs (editable)",
               "Model-extracted or hand-entered. These are INPUTS — every total "
               "below is computed in Python.")

    figures: Dict[str, Any] = dict(st.session_state.get(SS_FIGURES) or {})
    # Widget-key nonce: bumped on Extract/Reset so a fresh extraction (or reset)
    # re-seeds the input widgets via value= (Streamlit ignores value= once a keyed
    # widget already holds session state — a fresh key lets value= win again).
    _nonce = str(st.session_state.get(_PFX + "nonce", 0))

    meta_cols = st.columns(2)
    currency = meta_cols[0].text_input(
        "Currency / scale (label only)", value=_clean_currency(figures.get("currency")),
        max_chars=24, key=_PFX + "currency_" + _nonce)
    period = meta_cols[1].text_input(
        "Period (label only)", value=str(figures.get("period") or "")[:60],
        max_chars=60, key=_PFX + "period_" + _nonce)

    st.caption("Core income-statement lines — leave a line BLANK if it was not "
               "reported (a blank stays 'not provided', NOT a reported 0):")
    num_cols = st.columns(3)
    edited_figures: Dict[str, Any] = {}
    for i, (key, label) in enumerate(CORE_FIELDS):
        col = num_cols[i % 3]
        existing = _to_decimal(figures.get(key))
        edited_figures[key] = col.number_input(
            label,
            value=(float(existing) if existing is not None else None),
            step=1.0, format="%.2f",
            key=_PFX + "num_" + key + "_" + _nonce,
        )
    edited_figures["currency"] = currency
    edited_figures["period"] = period

    st.caption(f"Proposed adjustments (add-backs to EBITDA) — capped at "
               f"{MAX_ADJUSTMENTS} rows. Toggle Add-back off to SUBTRACT "
               f"(e.g. a one-time gain). Pre-filled rows are MODEL PROPOSALS — "
               f"review / edit them before trusting the Adjusted-EBITDA total.")

    adj_rows = list(st.session_state.get(SS_ADJ) or [])
    if PANDAS_AVAILABLE:
        adj_df = pd.DataFrame(
            adj_rows if adj_rows else [],
            columns=["label", "amount", "add_back", "rationale"],
        )
        if adj_df.empty:
            adj_df = pd.DataFrame(
                [{"label": "", "amount": 0.0, "add_back": True, "rationale": ""}])
        edited = st.data_editor(
            adj_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key=_PFX + "adj_editor_" + _nonce,
            column_config={
                "label": st.column_config.TextColumn(
                    "Label", required=False, max_chars=MAX_LABEL_CHARS),
                "amount": st.column_config.NumberColumn(
                    "Amount", format="%.2f", help="Positive magnitude."),
                "add_back": st.column_config.CheckboxColumn(
                    "Add-back?", help="On = ADD to EBITDA; off = SUBTRACT."),
                "rationale": st.column_config.TextColumn(
                    "Rationale", width="large", max_chars=240),
            },
        )
        # Normalize back to a bounded list[dict] (drop blank-label rows).
        adjustments: List[Dict[str, Any]] = []
        try:
            records = edited.to_dict("records")
        except Exception:
            records = []
        for rec in records[:MAX_ADJUSTMENTS]:
            label = _clean_label(rec.get("label"))
            if not label:
                continue
            adjustments.append({
                "label": label,
                "amount": rec.get("amount"),
                "add_back": bool(rec.get("add_back", True)),
                "rationale": str(rec.get("rationale") or "")[:240],
            })
        _valid = sum(1 for r in records if _clean_label(r.get("label")))
        if _valid > MAX_ADJUSTMENTS:
            st.warning(
                f"Only the first {MAX_ADJUSTMENTS} adjustments are included in the "
                f"bridge; {_valid - MAX_ADJUSTMENTS} additional row(s) were dropped.")
    else:  # pragma: no cover - pandas present in this venv
        adjustments = adj_rows
        st.info("pandas unavailable — adjustments shown read-only.")

    # Persist edited state (bounded) so it survives reruns.
    st.session_state[SS_FIGURES] = edited_figures
    st.session_state[SS_ADJ] = adjustments[:MAX_ADJUSTMENTS]

    # --- DETERMINISTIC COMPUTE (the source of truth) -----------------------
    try:
        bridge = compute_bridge(edited_figures, adjustments)
    except Exception as e:  # ERROR_BOUNDARY: a malformed input must never crash the tab
        logger.error("compute_bridge failed: %s", e)
        st.error("Could not compute the bridge for the current inputs — please "
                 "check the figures and adjustment amounts for a malformed value.")
        with st.expander("Token usage & cost", expanded=False):
            common.render_token_usage(inside_expander=True)
        return

    # --- VISUAL OUTPUT -----------------------------------------------------
    ui.section("3 · Bridge", "All subtotals below are computed in Python "
                             "from the inputs above — no LLM arithmetic.")

    kpis: List[Tuple] = [
        ("EBIT", _fmt_money(bridge["ebit"], currency), None,
         "Net income + interest + taxes (Python-computed)."),
        ("EBITDA", _fmt_money(bridge["ebitda"], currency), None,
         "EBIT + depreciation + amortization (Python-computed)."),
        ("Adjusted EBITDA", _fmt_money(bridge["adjusted_ebitda"], currency), None,
         "EBITDA +/- proposed adjustments (Python-computed)."),
    ]
    # HONEST: EBITDA margin ONLY when revenue is provided & non-zero — else omit.
    if bridge["ebitda_margin"] is not None:
        try:
            margin_disp = f"{bridge['ebitda_margin'].quantize(_CENTS)}%"
        except Exception:
            margin_disp = f"{bridge['ebitda_margin']:.2f}%"
        kpis.append(("EBITDA margin", margin_disp, None,
                     "EBITDA / revenue (revenue was provided)."))
    ui.kpi_row(kpis)

    # (2) Bridge: step table + visual waterfall (plotly) or fallback.
    st.markdown("**Bridge step table** (Step / Component / +/- Amount / Running subtotal)")
    _render_step_table(bridge, currency)

    fig = _waterfall_figure(bridge, currency)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        if not PLOTLY_AVAILABLE:
            st.caption("ℹ️ plotly unavailable — showing a running-subtotal bar "
                       "chart instead.")
        _render_barchart_fallback(bridge)

    # (3) Auditable DAG via pyvis (guarded) -> graphviz -> step table.
    with st.expander("Auditable computation graph (DAG)", expanded=False):
        dag_html = _build_dag_html(bridge, currency)
        if dag_html:
            components.html(dag_html, height=460, scrolling=False)
        else:
            st.caption(
                "ℹ️ Interactive DAG (pyvis/networkx) unavailable "
                f"({_PYVIS_IMPORT_ERROR or 'import failed'}); showing a "
                "static graph.")
            _render_dag_graphviz_fallback(bridge, currency)

    # --- EXPORT ------------------------------------------------------------
    ui.section("4 · Export")
    csv_text = _bridge_to_csv(bridge, currency, period)
    fname = "adjusted_ebitda_bridge"
    if period:
        safe = "".join(ch for ch in period if ch.isalnum() or ch in "-_")[:40]
        if safe:
            fname += f"_{safe}"
    st.download_button(
        "⬇️ Download bridge as CSV",
        data=csv_text,
        file_name=f"{fname}.csv",
        mime="text/csv",
        key=_PFX + "download",
    )

    # --- FOOTER NOTE + token usage -----------------------------------------
    st.caption(
        "Figures are model-extracted inputs; all subtotals are computed in "
        "Python (no LLM arithmetic).")

    with st.expander("Token usage & cost", expanded=False):
        render_token_usage(inside_expander=True)


__all__ = ["render_financials_tab", "compute_bridge"]
