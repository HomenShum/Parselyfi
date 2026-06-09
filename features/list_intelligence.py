"""
features/list_intelligence.py
=============================

ParselyFi FLAGSHIP "List Intelligence" tab.

This is the company-LIST pipeline the production artifacts proved is the real
job: take a list of company names (typed or uploaded) and run them through

    MATCH -> ENRICH -> CLASSIFY -> SCORE -> EXPORT

Public entry point (the ONLY thing the main app calls):

    def render_list_intelligence_tab() -> None

It is invoked inside an already-open ``st.tab`` context, so this module does
NOT call ``st.set_page_config`` and has no ``__main__`` run block.

Stages
------
1. MATCH    — for each typed name, LinkUp ``sourcedAnswer`` gives a source-backed
              answer; a structured Gemini call resolves it to a canonical company:
              {canonical_name, website, match_tier in [High, Medium, No Match],
               confidence 0-1, reason}. HONEST: "No Match" when the sources are
              weak / the answer does not actually describe a real, identifiable
              company (no fabricated canonical name).
2. ENRICH   — from the SAME LinkUp sourcedAnswer, reuse
              ``company_research._extract_structured_data_from_answer`` to pull
              the ``LINKUP_COMPANY_SCHEMA`` fields (industry, HQ, founded_year,
              funding_stage, investors, partnerships, key_people, competitors).
              Unsupported fields stay None — never invented.
3. CLASSIFY — the user pastes a sector/taxonomy DEFINITION; one structured Gemini
              call per company decides {match: bool, reasoning, keywords[]} of the
              company vs that definition, grounded in the enrich/answer evidence.
4. SCORE    — an EDITABLE VC rubric (Dimension/Description/Weight) via
              ``st.data_editor``. Per company, per dimension Gemini returns
              {tier in [Very High, High, Medium, Low], evidence (1 sentence),
               evidence_url} grounded in the LinkUp sources. We compute a weighted
              TotalScore (0-100), Coverage (fraction of dims with cited evidence)
              and Confidence (avg tier strength). HONEST: only real source URLs
              from the LinkUp result are accepted as evidence_url; an empty
              evidence lowers coverage rather than inventing a citation.

Execution
---------
The whole list runs CONCURRENTLY via ``common.run_async`` + an
``asyncio.Semaphore`` (capped at <= 5) so a long list never fans out unbounded
LinkUp/Gemini calls. Progress is streamed inside ``st.status(...)`` and a
``st.toast`` fires on completion. Every Gemini call records tokens centrally
through the ``common.*`` helpers, so ``common.render_token_usage`` shows the
real cost at the bottom.

Honest status / safety (per project CLAUDE.md)
----------------------------------------------
- Missing GEMINI / LINKUP key (or SDK) -> ``ui.inject_css()`` +
  ``render_gemini_health`` / warning + ``return``. Nothing fabricated.
- BOUNDED memory: the input list is capped at ``MAX_ROWS`` (40) with an honest
  notice when truncated; run history is capped via ``common.push_history``
  (FIFO, ``common.MAX_HISTORY``); the rubric is capped at ``MAX_DIMENSIONS``.
- BOUNDED concurrency: ``asyncio.Semaphore(min(user_choice, MAX_CONCURRENCY))``.
- TIMEOUTS: every LinkUp/Gemini call is the timeout-guarded ``common.*`` /
  ``company_research.*`` primitive; the whole batch is wrapped in an
  ``asyncio.wait_for`` wall-clock budget.
- Session-state keys are all prefixed ``li_`` so they never collide with the
  main app ("s3_file_manager" / "selected_files") or the sibling feature tabs
  ("cr_" / "ny_" / transcription).
- All client/scraper/LLM logic is REUSED from ``common.*`` and
  ``company_research.*`` — nothing is duplicated here.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import streamlit as st

from features import common, ui
from features import company_research as cr

try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - pandas is a hard dep of the grid UI
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

logger = logging.getLogger("parselyfi.features.list_intelligence")

# ===========================================================================
# Constants / bounded-memory caps (all li_-prefixed session keys)
# ===========================================================================
_PFX = "li_"
SS_RESULTS = _PFX + "results"            # list[dict] — the last completed run
SS_STAGES = _PFX + "stages"              # which stages last ran
SS_RUNNING = _PFX + "running"            # guard against double-submit
SS_RUBRIC = _PFX + "rubric_df"           # editable scoring rubric (DataFrame)
SS_HISTORY = _PFX + "run_history"        # bounded via common.push_history
SS_DRILL = _PFX + "drill_company"        # canonical name for the open dialog
SS_TAXONOMY = _PFX + "taxonomy_def"      # the pasted classify definition
SS_RAW_INPUT = _PFX + "raw_text_input"   # the names text_area

# Hard caps so nothing the user / LLM supplies can blow up memory or fan-out.
MAX_ROWS: int = 40                       # cap companies processed per run
MAX_CONCURRENCY: int = 5                 # absolute ceiling on parallel companies
DEFAULT_CONCURRENCY: int = 5
MAX_DIMENSIONS: int = 12                 # cap rubric rows
MAX_EVIDENCE_CHARS: int = 240            # cap a single evidence sentence
MAX_ANSWER_CHARS: int = 12_000           # cap LinkUp answer text fed to Gemini
BATCH_WALL_CLOCK_S: float = 1200.0       # overall budget for one pipeline run

# Stage identifiers.
STAGE_MATCH = "Match"
STAGE_ENRICH = "Enrich"
STAGE_CLASSIFY = "Classify"
STAGE_SCORE = "Score"
ALL_STAGES = [STAGE_MATCH, STAGE_ENRICH, STAGE_CLASSIFY, STAGE_SCORE]

REQUIRED_KEYS = ["GEMINI_API_KEY", "LINKUP_API_KEY"]

# Tier -> numeric strength for weighted scoring (0-100 scale per dimension).
TIER_POINTS: Dict[str, float] = {
    "very high": 100.0,
    "high": 75.0,
    "medium": 50.0,
    "low": 25.0,
}
SCORE_TIERS = ["Very High", "High", "Medium", "Low"]
MATCH_TIERS = ["High", "Medium", "No Match"]

# Starter VC rubric (Dimension, Description, Weight). Weights are relative; the
# scorer normalizes them so they always sum to 1.0 regardless of user edits.
STARTER_RUBRIC: List[Dict[str, Any]] = [
    {"Dimension": "Founders/Backers",
     "Description": "Quality of founding team and notable investors backing the company.",
     "Weight": 25},
    {"Dimension": "Funding",
     "Description": "Funding stage, amount raised, and capital efficiency.",
     "Weight": 20},
    {"Dimension": "Traction",
     "Description": "Customers, revenue, usage, or other evidence of market pull.",
     "Weight": 20},
    {"Dimension": "Moat",
     "Description": "Defensibility: technology, data, network effects, or brand.",
     "Weight": 15},
    {"Dimension": "Team",
     "Description": "Team depth, key hires, advisors, and relevant domain expertise.",
     "Weight": 10},
    {"Dimension": "Momentum",
     "Description": "Recent news, partnerships, growth trajectory in the last 6 months.",
     "Weight": 10},
]


# ===========================================================================
# Small helpers
# ===========================================================================
def _clean_name(name: str) -> str:
    return (name or "").strip()


def _parse_text_names(raw: str) -> List[str]:
    """One company per line; de-duped (case-insensitive), order-preserving."""
    out: List[str] = []
    seen = set()
    for line in (raw or "").splitlines():
        nm = _clean_name(line)
        if not nm:
            continue
        key = nm.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(nm)
    return out


def _names_from_uploaded(uploaded_file) -> Tuple[List[str], Optional[str]]:
    """Pull a company-name column from an uploaded csv/xlsx.

    Reuses ``company_research.parse_excel_file`` (which returns a DataFrame with
    a ``company_name`` column, detecting company-like headers and falling back
    to row-wise extraction for messy sheets). Returns ``(names, error)``;
    ``error`` is a human string when parsing fails (honest — no silent empty).
    """
    if uploaded_file is None:
        return [], None
    if not PANDAS_AVAILABLE:
        return [], "pandas is not installed; spreadsheet upload unavailable."
    try:
        df = common.run_async(cr.parse_excel_file(uploaded_file))
    except Exception as e:  # noqa: BLE001 - surface honestly
        logger.error("List Intelligence: failed to parse upload: %s", e)
        return [], f"Could not parse the uploaded file: {e}"
    if df is None or "company_name" not in getattr(df, "columns", []):
        return [], "No company-name column could be detected in the file."
    names: List[str] = []
    seen = set()
    for v in df["company_name"].tolist():
        nm = _clean_name(str(v))
        if not nm or nm.lower() in ("nan", "none"):
            continue
        key = nm.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(nm)
    return names, None


def _source_url_set(sources: List[Any]) -> List[str]:
    """Return the de-duped list of real source URLs from a LinkUp result."""
    urls: List[str] = []
    seen = set()
    for src in sources or []:
        url = cr.safe_get_attribute(src, "url", None)
        if not url:
            continue
        url = str(url).strip()
        if not url or not url.lower().startswith(("http://", "https://")):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url


def _clamp01(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


def _norm_tier(t: Any, allowed: List[str]) -> str:
    """Normalize a model-returned tier string to one of ``allowed`` (or '')."""
    s = str(t or "").strip().lower()
    for a in allowed:
        if s == a.lower():
            return a
    # tolerate minor variants
    if s in ("veryhigh", "very-high"):
        return "Very High"
    return ""


# ===========================================================================
# STAGE 1 — MATCH (LinkUp sourcedAnswer + structured Gemini resolution)
# ===========================================================================
async def _match_company(name: str) -> Dict[str, Any]:
    """Resolve a typed name to a canonical company via LinkUp + Gemini.

    Returns a dict carrying both the decision AND the raw LinkUp payload
    (answer + sources) so the later ENRICH/CLASSIFY/SCORE stages reuse the SAME
    source-backed answer rather than re-querying LinkUp.

    HONEST: ``match_tier == "No Match"`` (confidence low) when LinkUp returned
    no usable source-backed answer, or when Gemini judges the answer does not
    identify a real, specific company. No canonical name is fabricated.
    """
    search = await cr.company_search_with_linkup(name)
    answer = (search.get("answer") or "")[:MAX_ANSWER_CHARS]
    sources = search.get("sources", []) or []
    source_urls = _source_url_set(sources)

    base = {
        "input_name": name,
        "answer": answer,
        "sources": sources,
        "source_urls": source_urls,
        "search_ok": bool(search.get("success")),
        "search_error": search.get("error"),
    }

    # Honest short-circuit: no source-backed answer -> No Match, do not call Gemini.
    if not search.get("success") or not answer.strip() or not source_urls:
        base.update({
            "canonical_name": "",
            "website": "",
            "match_tier": "No Match",
            "match_confidence": 0.0,
            "match_reason": (
                search.get("error")
                or "LinkUp returned no source-backed answer for this name."
            ),
        })
        return base

    src_block = "\n".join(
        f"- {cr.safe_get_attribute(s, 'name', _domain(u))}: {u}"
        for s, u in zip(sources, source_urls)
    ) or "N/A"

    prompt = f"""
You are matching a user-typed company name to a real, specific company using ONLY
the source-backed answer and source list below. Do NOT invent a company.

USER TYPED: "{name}"

--- SOURCE-BACKED ANSWER ---
{answer}

--- SOURCES ---
{src_block}

Decide:
- canonical_name: the company's full/canonical name IF the answer clearly identifies
  one specific real company; otherwise "".
- website: the company's official website URL if present in the answer/sources; else "".
  It MUST be one of the source URLs above or clearly stated in the answer — never guessed.
- match_tier: "High" (sources clearly and specifically identify this exact company),
  "Medium" (likely but with some ambiguity, e.g. a common name or thin sources),
  or "No Match" (the answer does not identify a real specific company, or is about a
  different/ambiguous entity).
- confidence: float 0.0-1.0.
- reason: one short sentence citing what in the sources drove the decision.

Be HONEST: prefer "No Match" with low confidence over guessing. Return ONLY this JSON:
{{"canonical_name": "...", "website": "...", "match_tier": "High|Medium|No Match",
  "confidence": 0.0, "reason": "..."}}
"""
    res = await common.gemini_generate_json(
        prompt, model=common.GEMINI_MODEL, agent_name=f"Match: {name}"
    )

    tier = _norm_tier(res.get("match_tier"), MATCH_TIERS) or "No Match"
    canonical = _clean_name(str(res.get("canonical_name") or ""))
    website = str(res.get("website") or "").strip()
    # Honest: only accept a website that is an http(s) URL.
    if website and not website.lower().startswith(("http://", "https://")):
        website = ""
    confidence = _clamp01(res.get("confidence"))
    reason = str(res.get("reason") or "").strip()

    # If Gemini gave us nothing usable, degrade to No Match honestly.
    if not canonical and tier != "No Match":
        tier = "No Match"
        confidence = min(confidence, 0.2)
        reason = reason or "No canonical company could be identified from the sources."

    base.update({
        "canonical_name": canonical,
        "website": website,
        "match_tier": tier,
        "match_confidence": confidence,
        "match_reason": reason or "No reasoning returned.",
    })
    return base


# ===========================================================================
# STAGE 2 — ENRICH (reuse company_research schema extractor on the SAME answer)
# ===========================================================================
# Fields the spec asks us to surface in the drill-down.
ENRICH_FIELDS = [
    "industry", "headquarters", "founded_year", "funding_stage",
    "investors", "partnerships", "key_people", "competitors",
]


async def _enrich_company(match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract LINKUP_COMPANY_SCHEMA fields from the match's sourcedAnswer.

    Reuses ``company_research._extract_structured_data_from_answer`` so there is
    a single schema-extraction implementation. HONEST: returns ``None`` when
    there is no answer to extract from or Gemini is unavailable; individual
    unsupported fields stay None inside the returned dict.
    """
    answer = match.get("answer") or ""
    if not answer.strip():
        return None
    name = match.get("canonical_name") or match.get("input_name") or "the company"
    structured = await cr._extract_structured_data_from_answer(
        company_name=name,
        answer_text=answer,
        sources=match.get("sources", []) or [],
        agent_name=f"Enrich: {name}",
    )
    if not structured:
        return None
    # Project a stable subset (the spec's fields) onto the result; keep None when
    # unsupported rather than coercing to empty.
    out: Dict[str, Any] = {}
    for f in ENRICH_FIELDS:
        val = structured.get(f)
        if val in (None, "", [], {}):
            out[f] = None
        else:
            out[f] = val
    out["_full"] = structured  # keep the whole schema dict for the drill-down
    return out


# ===========================================================================
# STAGE 3 — CLASSIFY (company vs a user-supplied taxonomy DEFINITION)
# ===========================================================================
async def _classify_company(
    match: Dict[str, Any],
    enrich: Optional[Dict[str, Any]],
    taxonomy_def: str,
) -> Optional[Dict[str, Any]]:
    """Decide {match, reasoning, keywords[]} of a company vs a sector definition.

    HONEST: returns ``None`` (classify unavailable) when there is no taxonomy
    definition or no source-backed answer to judge against.
    """
    taxonomy_def = (taxonomy_def or "").strip()
    if not taxonomy_def:
        return None
    answer = match.get("answer") or ""
    if not answer.strip():
        return None
    name = match.get("canonical_name") or match.get("input_name") or "the company"

    enrich_hint = ""
    if enrich:
        bits = []
        for f in ("industry", "competitors", "funding_stage"):
            v = enrich.get(f)
            if v:
                bits.append(f"{f}: {v}")
        if bits:
            enrich_hint = "Structured signals: " + "; ".join(str(b) for b in bits)

    prompt = f"""
Decide whether the company below fits the following SECTOR/TAXONOMY DEFINITION,
based ONLY on the source-backed evidence. Do not rely on outside assumptions.

--- TAXONOMY DEFINITION ---
{taxonomy_def}

--- COMPANY: {name} ---
{answer}
{enrich_hint}

Return ONLY this JSON:
{{"match": true or false,
  "reasoning": "one or two sentences citing what in the evidence supports the verdict",
  "keywords": ["up to 6 short phrases from the evidence that drove the decision"]}}

If the evidence is too thin to decide, return match=false and say so in reasoning.
"""
    res = await common.gemini_generate_json(
        prompt, model=common.GEMINI_MODEL, agent_name=f"Classify: {name}"
    )
    if not res:
        return None
    kws = res.get("keywords")
    if not isinstance(kws, list):
        kws = []
    kws = [str(k).strip() for k in kws if str(k).strip()][:6]
    return {
        "match": bool(res.get("match", False)),
        "reasoning": str(res.get("reasoning") or "").strip() or "No reasoning returned.",
        "keywords": kws,
    }


# ===========================================================================
# STAGE 4 — SCORE (per-dimension tier + cited evidence, weighted total)
# ===========================================================================
def _normalize_rubric(rubric_df) -> List[Dict[str, Any]]:
    """Sanitize the editable rubric into a bounded list of dimension dicts.

    Drops blank/duplicate dimensions, coerces weights to non-negative numbers,
    and caps the count at ``MAX_DIMENSIONS``. Always returns at least the
    starter rubric if the user emptied the table (honest default, never zero
    dimensions to score against).
    """
    dims: List[Dict[str, Any]] = []
    seen = set()
    rows: List[Dict[str, Any]] = []
    if rubric_df is not None and PANDAS_AVAILABLE:
        try:
            rows = rubric_df.fillna("").to_dict("records")
        except Exception:
            rows = []
    elif isinstance(rubric_df, list):
        rows = rubric_df
    for r in rows:
        dim = _clean_name(str(r.get("Dimension", "")))
        if not dim or dim.lower() in seen:
            continue
        seen.add(dim.lower())
        try:
            w = float(r.get("Weight", 0) or 0)
        except (TypeError, ValueError):
            w = 0.0
        w = max(0.0, w)
        dims.append({
            "Dimension": dim,
            "Description": _clean_name(str(r.get("Description", ""))),
            "Weight": w,
        })
        if len(dims) >= MAX_DIMENSIONS:
            break
    if not dims:
        dims = [dict(d) for d in STARTER_RUBRIC]
    # If every weight is zero, distribute evenly (avoid div-by-zero downstream).
    if sum(d["Weight"] for d in dims) <= 0:
        for d in dims:
            d["Weight"] = 1.0
    return dims


async def _score_company(
    match: Dict[str, Any],
    rubric: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Score one company against the rubric, grounded in real LinkUp sources.

    For each dimension Gemini returns {tier, evidence, evidence_url}. We accept
    ``evidence_url`` ONLY if it is one of the company's real LinkUp source URLs;
    any other URL (or a missing one) yields empty evidence_url. Coverage is the
    fraction of dimensions with BOTH a tier and a real cited URL; Confidence is
    the average tier strength (0-1). TotalScore is the weight-normalized sum of
    per-dimension tier points (0-100).

    HONEST: returns ``None`` when there is no answer to score against; a
    dimension with no support yields tier "Low"/empty evidence and lowers
    coverage rather than inventing a citation.
    """
    answer = match.get("answer") or ""
    source_urls = match.get("source_urls") or []
    if not answer.strip() or not source_urls:
        return None
    name = match.get("canonical_name") or match.get("input_name") or "the company"

    dims_block = "\n".join(
        f"{i+1}. {d['Dimension']}: {d['Description']}" for i, d in enumerate(rubric)
    )
    src_block = "\n".join(f"- {u}" for u in source_urls)

    prompt = f"""
Score the company "{name}" against each scoring dimension below, using ONLY the
source-backed answer and the allowed source URLs. Ground every dimension in the
evidence — do NOT invent facts or citations.

--- SOURCE-BACKED ANSWER ---
{answer}

--- ALLOWED SOURCE URLs (evidence_url MUST be EXACTLY one of these, or "") ---
{src_block}

--- SCORING DIMENSIONS ---
{dims_block}

For EACH dimension return:
- tier: one of "Very High", "High", "Medium", "Low" (your honest read of the evidence;
  use "Low" when the evidence is weak or silent for that dimension).
- evidence: ONE sentence (<= 240 chars) quoting/paraphrasing the supporting evidence,
  or "" if there is no support.
- evidence_url: the single most relevant URL from the ALLOWED list that backs the
  evidence, or "" if none applies. NEVER output a URL not in the allowed list.

Return ONLY this JSON (one object per dimension, in the same order):
{{"dimensions": [
  {{"dimension": "<exact dimension name>", "tier": "...", "evidence": "...", "evidence_url": "..."}}
]}}
"""
    res = await common.gemini_generate_json(
        prompt, model=common.GEMINI_MODEL, agent_name=f"Score: {name}"
    )
    items = res.get("dimensions") if isinstance(res, dict) else None
    # Map returned items by dimension name for robust alignment.
    by_name: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                dn = _clean_name(str(it.get("dimension", ""))).lower()
                if dn:
                    by_name[dn] = it

    allowed = set(source_urls)
    scored_dims: List[Dict[str, Any]] = []
    weighted_sum = 0.0
    total_weight = sum(d["Weight"] for d in rubric) or 1.0
    covered = 0
    strength_acc = 0.0

    for d in rubric:
        it = by_name.get(d["Dimension"].lower(), {})
        tier = _norm_tier(it.get("tier"), SCORE_TIERS) or "Low"
        evidence = str(it.get("evidence") or "").strip()[:MAX_EVIDENCE_CHARS]
        ev_url = str(it.get("evidence_url") or "").strip()
        # HONEST: only a real LinkUp source URL is accepted as a citation.
        if ev_url not in allowed:
            ev_url = ""
        points = TIER_POINTS.get(tier.lower(), 25.0)
        weighted_sum += points * d["Weight"]
        strength_acc += points / 100.0
        has_evidence = bool(ev_url) and bool(evidence)
        if has_evidence:
            covered += 1
        scored_dims.append({
            "dimension": d["Dimension"],
            "weight": d["Weight"],
            "tier": tier,
            "points": points,
            "evidence": evidence,
            "evidence_url": ev_url,
            "has_evidence": has_evidence,
        })

    total_score = round(weighted_sum / total_weight, 1)  # 0-100
    coverage = round(covered / len(rubric), 3) if rubric else 0.0
    confidence = round(strength_acc / len(rubric), 3) if rubric else 0.0
    return {
        "dimensions": scored_dims,
        "total_score": total_score,
        "coverage": coverage,
        "confidence": confidence,
    }


# ===========================================================================
# Per-company pipeline (runs the selected stages in order)
# ===========================================================================
async def _process_one(
    name: str,
    stages: List[str],
    taxonomy_def: str,
    rubric: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the selected stages for a single company and assemble a result row."""
    row: Dict[str, Any] = {"input_name": name}

    # MATCH is the foundation: it produces the source-backed answer every other
    # stage reuses. If the user deselected Match we still need the LinkUp answer,
    # so we always run the match query but only surface its decision when Match
    # was requested.
    match = await _match_company(name)
    row.update({
        "canonical_name": match.get("canonical_name") or "",
        "website": match.get("website") or "",
        "match_tier": match.get("match_tier") or "No Match",
        "match_confidence": match.get("match_confidence", 0.0),
        "match_reason": match.get("match_reason") or "",
        "source_urls": match.get("source_urls", []),
        "_sources": match.get("sources", []),
        "search_ok": match.get("search_ok", False),
    })
    if STAGE_MATCH not in stages:
        # Match not requested: keep the answer for downstream stages but blank the
        # surfaced decision fields so the UI does not imply a match was computed.
        row["match_tier"] = ""
        row["match_reason"] = "Match stage not run."

    display_name = row["canonical_name"] or name

    if STAGE_ENRICH in stages:
        enrich = await _enrich_company(match)
        row["enrich"] = enrich
        row["enrich_available"] = enrich is not None
    else:
        enrich = None
        row["enrich"] = None
        row["enrich_available"] = False

    if STAGE_CLASSIFY in stages:
        classify = await _classify_company(match, enrich, taxonomy_def)
        row["classify"] = classify
        if classify is None:
            row["classify_label"] = "—"
        else:
            row["classify_label"] = "Match" if classify.get("match") else "No"
    else:
        row["classify"] = None
        row["classify_label"] = ""

    if STAGE_SCORE in stages:
        score = await _score_company(match, rubric)
        row["score"] = score
        if score is None:
            row["total_score"] = None
            row["coverage"] = None
            row["confidence"] = None
        else:
            row["total_score"] = score.get("total_score")
            row["coverage"] = score.get("coverage")
            row["confidence"] = score.get("confidence")
    else:
        row["score"] = None
        row["total_score"] = None
        row["coverage"] = None
        row["confidence"] = None

    row["_display_name"] = display_name
    return row


async def _run_pipeline_async(
    names: List[str],
    stages: List[str],
    taxonomy_def: str,
    rubric: List[Dict[str, Any]],
    concurrency: int,
    progress_cb,
) -> List[Dict[str, Any]]:
    """Run every company concurrently under a bounded semaphore.

    ``progress_cb(done, total, name)`` is invoked from the worker loop as each
    company completes; it is a plain callable that appends to a thread-safe list
    (the Streamlit st.status update happens on the main thread after the run).
    """
    sem = asyncio.Semaphore(max(1, min(int(concurrency or 1), MAX_CONCURRENCY)))
    total = len(names)
    counter = {"done": 0}

    async def _guarded(nm: str) -> Dict[str, Any]:
        async with sem:
            try:
                res = await _process_one(nm, stages, taxonomy_def, rubric)
            except Exception as e:  # noqa: BLE001 - one bad company never kills the run
                logger.error("List Intelligence: company %r failed: %s", nm, e)
                res = {
                    "input_name": nm, "canonical_name": "", "website": "",
                    "match_tier": "No Match", "match_confidence": 0.0,
                    "match_reason": f"Processing error: {e}", "source_urls": [],
                    "_sources": [], "search_ok": False, "enrich": None,
                    "enrich_available": False, "classify": None,
                    "classify_label": "—", "score": None, "total_score": None,
                    "coverage": None, "confidence": None, "_display_name": nm,
                }
            counter["done"] += 1
            if progress_cb:
                try:
                    progress_cb(counter["done"], total, res.get("_display_name", nm))
                except Exception:
                    pass
            return res

    tasks = [_guarded(nm) for nm in names]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=BATCH_WALL_CLOCK_S,
        )
    except asyncio.TimeoutError:
        logger.error("List Intelligence: batch wall-clock %ss exceeded.", BATCH_WALL_CLOCK_S)
        return []
    out: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("List Intelligence: task exception: %s", r)
            continue
        out.append(r)
    return out


# ===========================================================================
# Results assembly + export
# ===========================================================================
def _results_to_grid_df(results: List[Dict[str, Any]], stages: List[str]):
    """Build the flat grid DataFrame shown in st.dataframe."""
    if not PANDAS_AVAILABLE:
        return None
    rows = []
    for r in results:
        enrich = r.get("enrich") or {}
        rows.append({
            "Company": r.get("_display_name") or r.get("input_name"),
            "Input": r.get("input_name"),
            "Match": r.get("match_tier") or "",
            "Confidence": round(float(r.get("match_confidence") or 0.0), 2),
            "Website": r.get("website") or "",
            "Industry": (enrich.get("industry") if isinstance(enrich, dict) else None) or "",
            "Classify": r.get("classify_label") or "",
            "Score": r.get("total_score") if r.get("total_score") is not None else None,
            "Coverage": (round(float(r["coverage"]) * 100, 0)
                         if r.get("coverage") is not None else None),
        })
    return pd.DataFrame(rows)


def _results_to_export_df(results: List[Dict[str, Any]]):
    """Flatten results to a wide export DataFrame (CSV/Excel friendly)."""
    if not PANDAS_AVAILABLE:
        return None
    rows = []
    for r in results:
        enrich = r.get("enrich") or {}
        classify = r.get("classify") or {}
        score = r.get("score") or {}
        row = {
            "input_name": r.get("input_name"),
            "canonical_name": r.get("canonical_name"),
            "match_tier": r.get("match_tier"),
            "match_confidence": r.get("match_confidence"),
            "match_reason": r.get("match_reason"),
            "website": r.get("website"),
        }
        if isinstance(enrich, dict):
            for f in ENRICH_FIELDS:
                v = enrich.get(f)
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, default=str)
                row[f"enrich_{f}"] = v
        row["classify_match"] = classify.get("match") if classify else None
        row["classify_reasoning"] = classify.get("reasoning") if classify else None
        row["classify_keywords"] = (
            ", ".join(classify.get("keywords", [])) if classify else None
        )
        row["total_score"] = r.get("total_score")
        row["coverage"] = r.get("coverage")
        row["confidence"] = r.get("confidence")
        if score and score.get("dimensions"):
            for d in score["dimensions"]:
                dim = d.get("dimension", "dim")
                row[f"score_{dim}_tier"] = d.get("tier")
                row[f"score_{dim}_evidence"] = d.get("evidence")
                row[f"score_{dim}_url"] = d.get("evidence_url")
        rows.append(row)
    return pd.DataFrame(rows)


def _excel_bytes(df) -> Optional[bytes]:
    """Serialize a DataFrame to xlsx bytes; None if no engine available."""
    if df is None or not PANDAS_AVAILABLE:
        return None
    buf = io.BytesIO()
    for engine in ("openpyxl", "xlsxwriter"):
        try:
            with pd.ExcelWriter(buf, engine=engine) as writer:
                df.to_excel(writer, index=False, sheet_name="List Intelligence")
            return buf.getvalue()
        except Exception:
            buf.seek(0)
            buf.truncate(0)
            continue
    return None


# ===========================================================================
# Drill-down dialog (per-company)
# ===========================================================================
def _render_drilldown(result: Dict[str, Any]) -> None:
    """Render the enrichment + classify + per-dimension scoring for one company."""
    name = result.get("_display_name") or result.get("input_name")
    st.markdown(f"### {name}")
    tier_html = ui.tier_badge(result.get("match_tier") or "No Match")
    conf = float(result.get("match_confidence") or 0.0)
    st.markdown(
        f"{tier_html} &nbsp; **Confidence:** {conf:.0%} &nbsp; "
        f"_{result.get('match_reason') or ''}_",
        unsafe_allow_html=True,
    )
    if result.get("website"):
        st.link_button("Visit website", result["website"], use_container_width=False)

    # ---- ENRICH ----
    enrich = result.get("enrich")
    ui.section("Enrichment", "Source-backed company facts")
    if not enrich:
        st.caption("Enrichment unavailable (stage not run or no supported facts).")
    else:
        cols = st.columns(2)
        simple = [("Industry", "industry"), ("Headquarters", "headquarters"),
                  ("Founded", "founded_year"), ("Funding stage", "funding_stage")]
        for i, (label, key) in enumerate(simple):
            val = enrich.get(key)
            cols[i % 2].metric(label, val if val else "—")
        for label, key in [("Investors", "investors"), ("Partnerships", "partnerships"),
                            ("Key people", "key_people"), ("Competitors", "competitors")]:
            val = enrich.get(key)
            if not val:
                continue
            with st.expander(f"{label}"):
                if isinstance(val, list):
                    for item in val[:20]:
                        if isinstance(item, dict):
                            nm = item.get("name") or item.get("partner") or ""
                            role = item.get("role") or item.get("type") or item.get("description") or ""
                            st.markdown(f"- **{nm}**" + (f" — {role}" if role else ""))
                        else:
                            st.markdown(f"- {item}")
                else:
                    st.write(val)

    # ---- CLASSIFY ----
    classify = result.get("classify")
    ui.section("Classification", "Vs your taxonomy definition")
    if not classify:
        st.caption("Classification not run.")
    else:
        verdict = "✅ Match" if classify.get("match") else "❌ Not a match"
        st.markdown(f"**Verdict:** {verdict}")
        st.markdown(f"_{classify.get('reasoning', '')}_")
        kws = classify.get("keywords") or []
        if kws:
            st.markdown(" ".join(f"`{k}`" for k in kws))

    # ---- SCORE ----
    score = result.get("score")
    ui.section("Scoring", "Per-dimension tier with cited evidence")
    if not score:
        st.caption("Scoring not run (or no source-backed answer to ground it).")
        return
    st.markdown(
        f"**Total:** {score.get('total_score')}/100 &nbsp;|&nbsp; "
        f"**Coverage:** {float(score.get('coverage') or 0):.0%} &nbsp;|&nbsp; "
        f"**Confidence:** {float(score.get('confidence') or 0):.0%}"
    )
    for d in score.get("dimensions", []):
        badge = ui.tier_badge(d.get("tier"))
        wtxt = f"weight {d.get('weight')}"
        st.markdown(
            f"{badge} &nbsp; **{d.get('dimension')}** "
            f"<span style='color:{ui.MUTE};font-size:.82rem'>({wtxt})</span>",
            unsafe_allow_html=True,
        )
        ev = d.get("evidence")
        if ev:
            st.markdown(f"&nbsp;&nbsp;{ev}")
        url = d.get("evidence_url")
        if url:
            st.markdown(f"&nbsp;&nbsp;[{_domain(url)}]({url})")
        elif not ev:
            st.caption("&nbsp;&nbsp;No cited evidence (lowers coverage).")


if hasattr(st, "dialog"):
    @st.dialog("Company detail", width="large")
    def _drilldown_dialog(result: Dict[str, Any]) -> None:
        _render_drilldown(result)
else:  # pragma: no cover - older Streamlit fallback
    def _drilldown_dialog(result: Dict[str, Any]) -> None:
        with st.expander(f"Detail — {result.get('_display_name')}", expanded=True):
            _render_drilldown(result)


# ===========================================================================
# Results UI
# ===========================================================================
def _render_results(results: List[Dict[str, Any]], stages: List[str]) -> None:
    if not results:
        st.info("No results yet. Add companies and run the pipeline.")
        return

    # ---- KPI row ----
    total = len(results)
    matched = sum(1 for r in results if (r.get("match_tier") in ("High", "Medium")))
    scored = [r for r in results if r.get("total_score") is not None]
    avg_score = round(sum(r["total_score"] for r in scored) / len(scored), 1) if scored else 0
    cov_vals = [r["coverage"] for r in results if r.get("coverage") is not None]
    avg_cov = (sum(cov_vals) / len(cov_vals)) if cov_vals else 0.0

    ui.kpi_row([
        ("Companies", str(total)),
        ("Matched", f"{matched}/{total}", None, "High or Medium match tier"),
        ("Avg score", f"{avg_score:g}" if scored else "—", None, "Mean weighted score (0-100)"),
        ("Coverage", f"{avg_cov:.0%}" if cov_vals else "—", None,
         "Mean fraction of dimensions with cited evidence"),
    ])

    # ---- main grid ----
    ui.section("Results grid", "Click a company below to open its detail")
    grid_df = _results_to_grid_df(results, stages)
    if grid_df is not None:
        col_cfg = {
            "Company": st.column_config.TextColumn("Company", width="medium"),
            "Input": st.column_config.TextColumn("Typed as", width="small"),
            "Match": st.column_config.TextColumn("Match", width="small"),
            "Confidence": st.column_config.NumberColumn("Conf.", format="%.2f", width="small"),
            "Website": st.column_config.LinkColumn("Website", display_text="Open", width="small"),
            "Industry": st.column_config.TextColumn("Industry", width="medium"),
            "Classify": st.column_config.TextColumn("Class", width="small"),
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%d", width="medium"),
            "Coverage": st.column_config.NumberColumn("Cov %", format="%d%%", width="small"),
        }
        st.dataframe(
            grid_df, use_container_width=True, hide_index=True, column_config=col_cfg
        )

    # ---- per-company drill-down picker ----
    options = [r.get("_display_name") or r.get("input_name") for r in results]
    picked = st.selectbox(
        "Open detail for", options, index=None,
        placeholder="Select a company to inspect enrichment, classification & scoring…",
        key=_PFX + "drill_select",
    )
    if picked:
        match_row = next(
            (r for r in results if (r.get("_display_name") or r.get("input_name")) == picked),
            None,
        )
        if match_row is not None:
            _drilldown_dialog(match_row)

    # ---- exports ----
    ui.section("Export", "Download the full enriched/scored list")
    export_df = _results_to_export_df(results)
    if export_df is not None:
        c1, c2 = st.columns(2)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        c1.download_button(
            "Download CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"list_intelligence_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        xls = _excel_bytes(export_df)
        if xls is not None:
            c2.download_button(
                "Download Excel",
                data=xls,
                file_name=f"list_intelligence_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            c2.caption("Excel export needs `openpyxl` or `xlsxwriter`.")


# ===========================================================================
# Public entry point
# ===========================================================================
def render_list_intelligence_tab() -> None:
    """Render the flagship List Intelligence tab (called inside a tab context).

    Honest status: if a required key/SDK is missing we inject the CSS, surface
    the Gemini health banner / a warning, and return without fabricating data.
    """
    ui.inject_css()

    # ---- HONEST STATUS gate ----
    ok, missing = common.feature_available(REQUIRED_KEYS)
    if not ok:
        ui.hero(
            "List Intelligence",
            "MATCH → ENRICH → CLASSIFY → SCORE → EXPORT for a list of companies.",
        )
        common.render_gemini_health()
        if "LINKUP_API_KEY" in missing:
            st.warning(
                "⚠️ LinkUp is not configured — set `LINKUP_API_KEY` in secrets. "
                "List Intelligence needs LinkUp for source-backed matching, "
                "enrichment and scoring. The pipeline is disabled until then."
            )
        st.caption(f"Missing secrets: {', '.join(missing)}")
        return

    # Key present but possibly rejected (leaked/invalid) — surface honestly.
    common.render_gemini_health()

    if not PANDAS_AVAILABLE:
        st.error("pandas is required for the List Intelligence grid and exports.")
        return

    ui.hero(
        "List Intelligence",
        "Turn a raw company list into a matched, enriched, classified and scored "
        "pipeline — every fact source-backed.",
        chips=[
            "<b>MATCH</b> → ENRICH → CLASSIFY → SCORE → EXPORT",
            f"Up to <b>{MAX_ROWS}</b> companies / run",
            "Source-cited evidence only",
        ],
    )

    # ---- INPUT ----
    ui.section("1 · Input", "One company per line, or upload a CSV/XLSX")
    in_col, up_col = st.columns([2, 1])
    with in_col:
        raw_text = st.text_area(
            "Company names (one per line)",
            key=SS_RAW_INPUT,
            height=180,
            placeholder="Anthropic\nOpenAI\nMistral AI\nCohere",
        )
    with up_col:
        uploaded = st.file_uploader(
            "…or upload a list", type=["csv", "xlsx", "xls"], key=_PFX + "uploader"
        )

    names = _parse_text_names(raw_text)
    if uploaded is not None:
        up_names, up_err = _names_from_uploaded(uploaded)
        if up_err:
            st.warning(up_err)
        # Merge (text first), de-dup case-insensitively.
        seen = {n.lower() for n in names}
        for n in up_names:
            if n.lower() not in seen:
                seen.add(n.lower())
                names.append(n)

    truncated = False
    if len(names) > MAX_ROWS:
        st.info(
            f"You provided {len(names)} companies; processing the first "
            f"{MAX_ROWS} (bounded for reliability). Split the rest into another run."
        )
        names = names[:MAX_ROWS]
        truncated = True

    if names:
        st.caption(f"**{len(names)}** companies ready" + (" (truncated)" if truncated else ""))

    # ---- STAGE PICKER ----
    ui.section("2 · Stages", "Choose which stages to run")
    picker = getattr(st, "segmented_control", None) or getattr(st, "pills", None)
    if picker is not None:
        stages = picker(
            "Pipeline stages",
            options=ALL_STAGES,
            selection_mode="multi",
            default=ALL_STAGES,
            key=_PFX + "stage_picker",
        ) or []
    else:  # pragma: no cover - very old Streamlit
        stages = st.multiselect(
            "Pipeline stages", ALL_STAGES, default=ALL_STAGES, key=_PFX + "stage_picker"
        )
    if not stages:
        stages = [STAGE_MATCH]
        st.caption("No stage selected — defaulting to Match only.")

    concurrency = st.slider(
        "Max concurrent companies", min_value=1, max_value=MAX_CONCURRENCY,
        value=DEFAULT_CONCURRENCY, key=_PFX + "concurrency",
        help="Bounded fan-out across LinkUp + Gemini calls.",
    )

    # ---- CLASSIFY definition (only relevant if Classify selected) ----
    taxonomy_def = ""
    if STAGE_CLASSIFY in stages:
        ui.section("3 · Classify definition", "Paste the sector/taxonomy to test each company against")
        taxonomy_def = st.text_area(
            "Taxonomy / sector definition",
            key=SS_TAXONOMY,
            height=120,
            placeholder=(
                "e.g. 'AI infrastructure companies that sell developer-facing "
                "tooling (APIs, SDKs, model-serving) to other software teams, "
                "as opposed to consumer apps or pure research labs.'"
            ),
        )
        if not taxonomy_def.strip():
            st.caption("Classify will be skipped per-company until a definition is provided.")

    # ---- SCORE rubric (only relevant if Score selected) ----
    rubric_list: List[Dict[str, Any]] = [dict(d) for d in STARTER_RUBRIC]
    if STAGE_SCORE in stages:
        ui.section("4 · Scoring rubric", "Edit dimensions / weights (weights are normalized)")
        if SS_RUBRIC not in st.session_state:
            st.session_state[SS_RUBRIC] = pd.DataFrame(STARTER_RUBRIC)
        edited = st.data_editor(
            st.session_state[SS_RUBRIC],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key=_PFX + "rubric_editor",
            column_config={
                "Dimension": st.column_config.TextColumn("Dimension", required=True),
                "Description": st.column_config.TextColumn("Description", width="large"),
                "Weight": st.column_config.NumberColumn(
                    "Weight", min_value=0, max_value=100, step=1, format="%d"),
            },
        )
        st.session_state[SS_RUBRIC] = edited
        rubric_list = _normalize_rubric(edited)
        if len(rubric_list) >= MAX_DIMENSIONS:
            st.caption(f"Rubric capped at {MAX_DIMENSIONS} dimensions.")

    # ---- RUN ----
    ui.section("5 · Run", "")
    run = st.button(
        "🚀 Run pipeline", type="primary", use_container_width=True,
        disabled=(len(names) == 0),
        key=_PFX + "run_btn",
    )
    if len(names) == 0:
        st.caption("Add at least one company to enable the run.")

    if run:
        progress_log: List[Tuple[int, int, str]] = []

        def _cb(done: int, total: int, nm: str) -> None:
            progress_log.append((done, total, nm))

        with st.status("Running pipeline…", expanded=True) as status:
            status.write(
                f"Processing **{len(names)}** companies through: "
                f"{' → '.join(stages)} (max {concurrency} concurrent)."
            )
            start = time.time()
            results = common.run_async(
                _run_pipeline_async(
                    names=names,
                    stages=stages,
                    taxonomy_def=taxonomy_def,
                    rubric=rubric_list,
                    concurrency=concurrency,
                    progress_cb=_cb,
                )
            )
            elapsed = time.time() - start
            # Replay the worker-thread progress on the main thread.
            for done, total, nm in progress_log:
                status.write(f"✓ [{done}/{total}] {nm}")
            if results:
                status.update(
                    label=f"Done — {len(results)} companies in {elapsed:.1f}s",
                    state="complete", expanded=False,
                )
            else:
                status.update(
                    label="Pipeline returned no results (timeout or all failed)",
                    state="error", expanded=True,
                )

        st.session_state[SS_RESULTS] = results
        st.session_state[SS_STAGES] = stages
        common.push_history(SS_HISTORY, {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "n_companies": len(names),
            "stages": list(stages),
            "matched": sum(1 for r in results if r.get("match_tier") in ("High", "Medium")),
        })
        if results:
            try:
                st.toast(f"Pipeline complete — {len(results)} companies processed.", icon="✅")
            except Exception:
                pass

    # ---- RESULTS ----
    results = st.session_state.get(SS_RESULTS)
    last_stages = st.session_state.get(SS_STAGES, stages)
    if results:
        st.divider()
        _render_results(results, last_stages)

    # ---- TOKEN USAGE ----
    st.divider()
    with st.expander("Token usage & cost", expanded=False):
        common.render_token_usage(inside_expander=True)


__all__ = ["render_list_intelligence_tab"]
