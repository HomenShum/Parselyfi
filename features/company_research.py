"""
features/company_research.py
============================

ParselyFi flagship "Company Research" tab.

This is a faithful port of the reference Streamlit app
``reference_codes/test7_create_company_report_LINKUP_v040725.py`` into an
importable module that exposes exactly one public entry point:

    render_company_research_tab() -> None

The main app (prod_parselyfi_v031525.py) owns ``st.set_page_config`` and the
auth/sidebar, and calls this function *inside an already-created tab*. This
module therefore:

- never calls ``st.set_page_config`` and has no ``if __name__ == '__main__'``
  block,
- reads secrets ONLY through ``features.common`` helpers,
- uses the session-state key prefix ``"cr_"`` for every key it owns,
- degrades honestly: if a required key/dep is missing it shows a warning and
  returns early instead of fabricating results,
- bounds every accumulating collection (search history, observability log)
  with a MAX + oldest-eviction,
- pushes all network/LLM/scrape work through the shared, timeout-guarded,
  SSRF-hardened ``common.*`` primitives.

Workflow ported (the full flagship feature):

1. Free-text input AND spreadsheet upload (xlsx/xls/csv) -> a
   ``company_name`` + ``additional_context`` dataframe (``parse_excel_file``:
   detect company-like columns; row-wise Gemini entity extraction fallback
   for messy sheets).
2. Entity resolution: candidate table with Select-Target / Mark-Irrelevant +
   source domains/snippets preview
   (``first_pass_entity_selection`` / ``first_pass_ui_with_data_editor``).
3. Editable review dataframe (``st.data_editor``) with Pending/Complete/Skip
   status; batch-enrich only Pending rows with concurrency
   (``batch_process_companies`` / ``process_companies_with_progress``).
4. 3-pass research: entity selection -> structured search (sourcedAnswer +
   schema) -> analyze missing fields + targeted backfill
   (``second_pass_structured_search`` / ``analyze_missing_fields`` /
   ``third_pass_fill_missing_info``).
5. LinkUp ``sourcedAnswer`` for source-backed answers; "no answer / no
   source" surfaced honestly.
6. Optional web-scraping enhancement of URLs found in row context
   (``common.scrape_urls_from_context``).
7. Token/cost tracking via ``common.render_token_usage``; bounded search
   history save/export/import via ``common.push_history`` / ``get_history``.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import streamlit as st

from . import common
from .common import (
    GEMINI_MODEL,
    feature_available,
    gemini_generate_json,
    get_async_runner,
    get_history,
    get_linkup_client,
    push_history,
    record_tokens,
    render_token_usage,
    run_async,
    scrape_urls_from_context,
)

try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover - pandas is a hard dep of the data editor UI
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

logger = logging.getLogger("parselyfi.features.company_research")

# ===========================================================================
# Session-state keys (ALL prefixed "cr_" so they never collide with the main
# app's "s3_file_manager"/"selected_files" or the other feature tabs).
# ===========================================================================
SS_PHASE = "cr_research_phase"
SS_MODE = "cr_research_mode"
SS_FIRST_PASS = "cr_first_pass_result"
SS_SELECTED_ENTITY = "cr_selected_entity"
SS_NEGATIVE_ENTITIES = "cr_negative_entities"
SS_ENTITY_FULL_DATA = "cr_entity_full_data"
SS_SECOND_PASS = "cr_second_pass_result"
SS_THIRD_PASS = "cr_third_pass_result"
SS_ADDITIONAL_CONTEXT = "cr_additional_context"
SS_VIEWING_HISTORY_ID = "cr_viewing_history_id"
SS_HISTORY = "cr_search_history"            # bounded via common.push_history
SS_STRUCT_VISIBILITY = "cr_structured_data_visibility"
SS_OBSERVABILITY = "cr_observability_log"   # bounded manually below
SS_UPLOAD_DF = "cr_upload_editor_df"
SS_BATCH_RESULTS = "cr_batch_results"

# Bounded-memory caps for this module's own collections.
MAX_OBSERVABILITY = 200       # bounded observability log
MAX_VISIBILITY_KEYS = 500     # bounded structured-data toggle map
DEFAULT_MAX_CONCURRENT = 5    # LinkUp/Gemini fan-out cap
HARD_MAX_CONCURRENT = 10
MAX_UPLOAD_ROWS = 200         # cap how many spreadsheet rows we will process
LINKUP_TIMEOUT_S = 90.0       # per-LinkUp-call total budget
BATCH_POLL_TIMEOUT_S = 900.0  # overall wall-clock cap for a batch run

# ===========================================================================
# LinkUp company schema (ported verbatim from the reference).
# ===========================================================================
LINKUP_COMPANY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string", "description": "Full legal name of the company."},
        "company_website": {"type": "string", "format": "uri", "description": "URL of the company's official website, if found."},
        "company_description": {"type": "string", "description": "Comprehensive description of the company's business, mission, and offerings."},
        "industry": {"type": "string", "description": "Primary industry or sector the company operates in."},
        "headquarters": {"type": "string", "description": "Location of the company's headquarters, including city and country."},
        "founded_year": {"type": "string", "description": "Year the company was founded."},
        "company_size": {"type": "string", "description": "Approximate number of employees or size category."},
        "business_model": {"type": "string", "description": "Description of how the company generates revenue (B2B, B2C, SaaS, etc.)."},
        "leadership_team": {
            "type": "array",
            "description": "Key leadership personnel at the company.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the leader."},
                    "role": {"type": "string", "description": "Position or title."},
                    "linkedin_url": {"type": "string", "description": "LinkedIn profile URL, if available."},
                    "is_founder": {"type": "boolean", "description": "Whether the person is a founder."},
                    "background": {"type": "string", "description": "Brief professional background."},
                },
                "required": ["name", "role"],
            },
        },
        "funding": {
            "type": "array",
            "description": "Funding rounds and investments, from most recent to oldest.",
            "items": {
                "type": "object",
                "properties": {
                    "round": {"type": "string", "description": "Funding round type (Seed, Series A, etc.)."},
                    "date": {"type": "string", "description": "Date of the funding round."},
                    "amount": {"type": "string", "description": "Amount raised."},
                    "investors": {"type": "array", "items": {"type": "string"}, "description": "List of investors."},
                    "lead_investor": {"type": "string", "description": "Lead investor of the round."},
                },
                "required": ["round", "date", "amount"],
            },
        },
        "products": {
            "type": "array",
            "description": "Main products or services offered by the company.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the product or service."},
                    "description": {"type": "string", "description": "Comprehensive description."},
                    "launch_date": {"type": "string", "description": "When the product was launched."},
                    "key_features": {"type": "array", "items": {"type": "string"}, "description": "Main features or capabilities."},
                },
                "required": ["name", "description"],
            },
        },
        "tech_stack": {"type": "array", "description": "Technologies and tools used by the company.", "items": {"type": "string"}},
        "competitors": {"type": "array", "description": "Main competitors in the market.", "items": {"type": "string"}},
        # --- High-value finance fields (additive; ported from component2 schema) ---
        "funding_stage": {"type": "string", "description": "Current overall funding stage of the company (e.g. Bootstrapped, Pre-seed, Seed, Series A/B/C/D/E/F, Public, Acquired, Subsidiary, Donations/Grant). Use the most recent/highest stage supported by the sources; empty string if unknown."},
        "investors": {
            "type": "array",
            "description": "Notable investors / investment firms backing the company.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the investor or investment firm."},
                    "type": {"type": "string", "description": "Type of investor (e.g. Venture Capital, Angel, Private Equity, Corporate)."},
                    "stage": {"type": "string", "description": "Round/stage this investor participated in (e.g. Seed, Series A)."},
                    "lead": {"type": "boolean", "description": "Whether this investor led a round."},
                },
                "required": ["name"],
            },
        },
        "partnerships": {
            "type": "array",
            "description": "Publicly announced partnerships, alliances, or notable customers.",
            "items": {
                "type": "object",
                "properties": {
                    "partner": {"type": "string", "description": "Name of the partner entity."},
                    "description": {"type": "string", "description": "Summary of the partnership and what it involves."},
                    "year": {"type": "string", "description": "Year the partnership was announced/happened."},
                    "url": {"type": "string", "description": "Source URL for the partnership, if available."},
                },
                "required": ["partner"],
            },
        },
        "key_people": {
            "type": "array",
            "description": "Key people beyond the core leadership team: advisors, board members, notable hires.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the person."},
                    "role": {"type": "string", "description": "Specific role or title (e.g. Advisor, Board Member, Head of Eng)."},
                    "linkedin_url": {"type": "string", "description": "LinkedIn profile URL, if available."},
                    "background": {"type": "string", "description": "Short background/biography."},
                },
                "required": ["name", "role"],
            },
        },
        "recent_news": {
            "type": "array",
            "description": "Recent news or announcements about the company from the past 3 months.",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title of the news item."},
                    "date": {"type": "string", "description": "Date of the news."},
                    "summary": {"type": "string", "description": "Brief summary of the news."},
                    "url": {"type": "string", "description": "URL to the full news article."},
                },
                "required": ["title", "date", "summary"],
            },
        },
        "market_position": {
            "type": "object",
            "description": "Company's position in the market.",
            "properties": {
                "strengths": {"type": "array", "items": {"type": "string"}, "description": "Key competitive advantages."},
                "challenges": {"type": "array", "items": {"type": "string"}, "description": "Major challenges or weaknesses."},
                "market_share": {"type": "string", "description": "Estimated market share, if available."},
                "growth_trajectory": {"type": "string", "description": "Recent growth pattern (rapid, steady, declining)."},
            },
        },
        "sources": {
            "type": "array",
            "description": "Source URLs used to compile this company information",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title of the source page"},
                    "url": {"type": "string", "description": "URL of the source"},
                    "domain": {"type": "string", "description": "Domain of the source"},
                },
                "required": ["url"],
            },
        },
    },
    "required": ["company_name", "company_description", "industry", "headquarters", "founded_year"],
}

# LinkUp concurrency limiter (lazy, bound to the active event loop).
_LINKUP_SEM: Optional[asyncio.Semaphore] = None


def _linkup_sem() -> asyncio.Semaphore:
    """Return a process-wide LinkUp safety-ceiling semaphore.

    This is only a HARD safety ceiling so a runaway never exceeds the absolute
    max; it is set to ``HARD_MAX_CONCURRENT`` (the same value the UI slider
    allows) so it never silently caps the user's chosen "Maximum concurrent
    requests" below what the UI offered. Real per-run throttling is done by the
    per-call ``max_concurrent`` semaphores plumbed through each fan-out.
    """
    global _LINKUP_SEM
    if _LINKUP_SEM is None:
        _LINKUP_SEM = asyncio.Semaphore(HARD_MAX_CONCURRENT)
    return _LINKUP_SEM


# ===========================================================================
# Small helpers (ported)
# ===========================================================================
def safe_get_attribute(obj, attribute_name, default=None):
    """Get an attribute whether ``obj`` is a dict or an object; else default."""
    if hasattr(obj, attribute_name):
        return getattr(obj, attribute_name)
    if isinstance(obj, dict) and attribute_name in obj:
        return obj[attribute_name]
    return default


def format_time(seconds: float) -> str:
    """Human-readable duration."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "0.0 seconds"
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    return f"{seconds / 3600:.1f} hours"


def _append_observability(entry: Dict[str, Any]) -> None:
    """Append to the bounded observability log in session state."""
    log = st.session_state.get(SS_OBSERVABILITY)
    if not isinstance(log, list):
        log = []
    log.append(entry)
    if len(log) > MAX_OBSERVABILITY:
        log = log[-MAX_OBSERVABILITY:]
    st.session_state[SS_OBSERVABILITY] = log


def _set_visibility(key: str, value: bool) -> None:
    """Set a structured-data toggle in the bounded visibility map."""
    vis = st.session_state.get(SS_STRUCT_VISIBILITY)
    if not isinstance(vis, dict):
        vis = {}
    vis[key] = value
    # BOUNDED: keep only the most-recent N keys.
    if len(vis) > MAX_VISIBILITY_KEYS:
        for k in list(vis.keys())[: len(vis) - MAX_VISIBILITY_KEYS]:
            vis.pop(k, None)
    st.session_state[SS_STRUCT_VISIBILITY] = vis


def _get_visibility(key: str) -> bool:
    vis = st.session_state.get(SS_STRUCT_VISIBILITY)
    if isinstance(vis, dict):
        return bool(vis.get(key, False))
    return False


# ===========================================================================
# Context extraction helpers (ported)
# ===========================================================================
def extract_context_category(context_text: str, keywords: List[str]) -> str:
    """Pull a value associated with any of ``keywords`` out of free text."""
    if not context_text:
        return ""
    context_lower = context_text.lower()
    for keyword in keywords:
        for pattern in (
            rf"{keyword}\s*:\s*([^,;]+)",
            rf"{keyword}\s*-\s*([^,;]+)",
            rf"{keyword}[=]\s*([^,;]+)",
        ):
            matches = re.search(pattern, context_lower)
            if matches:
                return matches.group(1).strip()
    for keyword in keywords:
        if keyword in context_lower:
            sentences = re.split(r"[.!?]\s+", context_text)
            for sentence in sentences:
                if keyword.lower() in sentence.lower():
                    parts = sentence.split(",")
                    for part in parts:
                        if keyword.lower() in part.lower():
                            return part.strip()
                    return sentence.strip()
    return ""


def extract_disambiguated_company_info(company_name: str, additional_context: str) -> Dict[str, str]:
    """Best-effort structured disambiguation fields from free-text context."""
    info = {
        "industry": extract_context_category(additional_context, ["industry", "sector", "field", "domain", "market"]),
        "location": extract_context_category(additional_context, ["located", "location", "headquarters", "hq", "based in", "region", "city", "country"]),
        "founded": extract_context_category(additional_context, ["founded", "established", "started", "creation", "year", "since"]),
        "founders": extract_context_category(additional_context, ["founder", "creator", "started by", "founded by", "ceo", "owner"]),
        "product": extract_context_category(additional_context, ["product", "service", "offering", "solution", "platform", "app", "software"]),
        "size": extract_context_category(additional_context, ["employees", "team size", "company size", "headcount", "staff"]),
        "website": extract_context_category(additional_context, ["website", "site", "url", "web", "domain", "online at"]),
    }
    for key in info:
        if info[key]:
            for prefix in (f"{key}:", f"{key} -", f"{key}="):
                if info[key].lower().startswith(prefix.lower()):
                    info[key] = info[key][len(prefix):].strip()
    return info


def extract_context_for_company(general_context: str, company_name: str) -> str:
    """Slice the part of a multi-company context that mentions ``company_name``."""
    if not general_context:
        return ""
    if company_name.lower() not in general_context.lower():
        return general_context
    sentences = re.split(r"[.!?]\s+", general_context)
    relevant_sentences: List[str] = []
    for sentence in sentences:
        if company_name.lower() in sentence.lower():
            relevant_sentences.append(sentence)
        elif relevant_sentences and len(relevant_sentences) < 5:
            relevant_sentences.append(sentence)
    if relevant_sentences:
        company_context = ". ".join(relevant_sentences)
        if not company_context.endswith("."):
            company_context += "."
        return company_context
    return general_context


# ===========================================================================
# LinkUp search (the source-backed answer engine).  Honest status: surfaces
# "no answer / no source" rather than inventing data, and never blocks the
# event loop without a timeout.
# ===========================================================================
async def company_search_with_linkup(
    company_name: str,
    additional_context: str = "",
) -> Dict[str, Any]:
    """Query LinkUp (``sourcedAnswer``) for source-backed company info.

    Returns a dict with ``success``, ``answer``, ``sources``,
    ``search_text_formatted`` and ``raw_response``. On any failure (no client,
    timeout, exception) ``success`` is False and the failure is reported
    honestly -- no fabricated answer/sources.
    """
    client = get_linkup_client()
    if client is None:
        logger.warning("LinkUp client not available for %s.", company_name)
        return {
            "success": False,
            "error": "LinkUp client not initialized (missing LINKUP_API_KEY or SDK)",
            "answer": "",
            "sources": [],
            "structured_data": None,
            "search_text_formatted": f"No LinkUp client available; cannot research {company_name}.",
            "parsing_error": None,
            "raw_response": None,
        }

    query = additional_context or company_name
    logger.info("Querying LinkUp (sourcedAnswer) for: %s", company_name)

    def _do_search():
        # Synchronous LinkUp call run off the event loop (bounded by wait_for).
        return client.search(
            query=query,
            depth="standard",
            include_images=True,
            output_type="sourcedAnswer",
        )

    try:
        async with _linkup_sem():
            response = await asyncio.wait_for(
                asyncio.to_thread(_do_search),
                timeout=LINKUP_TIMEOUT_S,
            )
    except asyncio.TimeoutError:
        logger.error("LinkUp search timed out for %s after %ss.", company_name, LINKUP_TIMEOUT_S)
        return {
            "success": False,
            "error": f"LinkUp search timed out after {LINKUP_TIMEOUT_S:.0f}s",
            "answer": "",
            "sources": [],
            "structured_data": None,
            "search_text_formatted": f"Error: LinkUp search timed out for {company_name}.",
            "parsing_error": None,
            "raw_response": None,
        }
    except Exception as e:  # noqa: BLE001 - report honestly, never fabricate
        logger.error("Error during LinkUp search for %s: %s", company_name, e)
        return {
            "success": False,
            "error": str(e),
            "answer": "",
            "sources": [],
            "structured_data": None,
            "search_text_formatted": f"Error searching for {company_name}: {e}",
            "parsing_error": None,
            "raw_response": None,
        }

    if not response:
        logger.error("No response received from LinkUp for %s.", company_name)
        return {
            "success": False,
            "error": "No response from LinkUp",
            "answer": "",
            "sources": [],
            "structured_data": None,
            "search_text_formatted": f"Error: No response from LinkUp for {company_name}.",
            "parsing_error": None,
            "raw_response": None,
        }

    answer_text = safe_get_attribute(response, "answer", "") or ""
    sources = safe_get_attribute(response, "sources", []) or []

    if not answer_text:
        logger.warning("No answer found in LinkUp response for %s.", company_name)
    if not sources:
        logger.warning("No sources found in LinkUp response for %s.", company_name)

    # Honest formatting: explicitly say when there is no answer / no source.
    search_text_formatted = f"# Information for {company_name}\n\n"
    search_text_formatted += answer_text if answer_text else (
        f"No answer was returned by LinkUp for {company_name} (no source-backed information found)."
    )
    if sources:
        search_text_formatted += "\n\n## Sources (from LinkUp)"
        for i, src in enumerate(sources):
            src_name = safe_get_attribute(src, "name", f"Source {i + 1}")
            src_url = safe_get_attribute(src, "url", "N/A")
            src_snippet = safe_get_attribute(src, "snippet", "")
            search_text_formatted += f"\n{i + 1}. **{src_name}**"
            search_text_formatted += f"\n   - URL: {src_url}"
            if src_snippet:
                snip = src_snippet[:300] + "..." if len(src_snippet) > 300 else src_snippet
                search_text_formatted += f"\n   - Snippet: {snip}"
            else:
                search_text_formatted += "\n   - Snippet: N/A"
    else:
        search_text_formatted += "\n\n_No sources were returned by LinkUp for this query._"

    return {
        "success": True,
        "answer": answer_text,
        "sources": sources,
        "structured_data": None,
        "search_text_formatted": search_text_formatted,
        "parsing_error": None,
        "raw_response": response,
    }


# ===========================================================================
# Formatting structured LinkUp data into markdown (ported)
# ===========================================================================
def format_linkup_results_as_search_text(structured_data: Dict[str, Any], company_name: str) -> str:
    """Render the company schema dict as readable markdown."""
    if not structured_data:
        return f"# {company_name}\n\n_No structured data available._"
    lines: List[str] = []
    lines.append(f"# {structured_data.get('company_name', company_name)}")
    lines.append("")
    lines.append("## 1. Company Overview")
    if website := structured_data.get("company_website"):
        lines.append(f"**Official Website:** {website}")
    if desc := structured_data.get("company_description"):
        lines.append(f"**Description:** {desc}")
    basic_info = []
    if industry := structured_data.get("industry"):
        basic_info.append(f"Industry: {industry}")
    if hq := structured_data.get("headquarters"):
        basic_info.append(f"Headquarters: {hq}")
    if founded := structured_data.get("founded_year"):
        basic_info.append(f"Founded: {founded}")
    if size := structured_data.get("company_size"):
        basic_info.append(f"Company Size: {size}")
    if biz_model := structured_data.get("business_model"):
        basic_info.append(f"Business Model: {biz_model}")
    if basic_info:
        lines.append("**Company Details:**")
        for info in basic_info:
            lines.append(f"- {info}")
    lines.append("")

    if leaders := structured_data.get("leadership_team"):
        lines.append("## 2. Leadership Team")
        for leader in leaders:
            name = leader.get("name", "")
            role = leader.get("role", "")
            is_founder = leader.get("is_founder", False)
            linkedin = leader.get("linkedin_url", "")
            bg = leader.get("background", "")
            leader_title = f"**{name}**"
            if role:
                leader_title += f", {role}"
            if is_founder:
                leader_title += " (Founder)"
            lines.append(leader_title)
            if bg:
                lines.append(f"- Background: {bg}")
            if linkedin:
                lines.append(f"- LinkedIn: {linkedin}")
            lines.append("")

    if key_people := structured_data.get("key_people"):
        lines.append("## 2b. Key People (Advisors / Board / Notable Hires)")
        for person in key_people:
            if not isinstance(person, dict):
                lines.append(f"- {person}")
                continue
            kp_line = f"**{person.get('name', '')}**"
            if role := person.get("role"):
                kp_line += f", {role}"
            lines.append(kp_line)
            if bg := person.get("background"):
                lines.append(f"- Background: {bg}")
            if li := person.get("linkedin_url"):
                lines.append(f"- LinkedIn: {li}")
            lines.append("")

    if (funding := structured_data.get("funding")) or structured_data.get("funding_stage") or structured_data.get("investors"):
        lines.append("## 3. Funding History")
        if stage := structured_data.get("funding_stage"):
            lines.append(f"**Funding Stage:** {stage}")
            lines.append("")
        for round_data in (funding or []):
            round_name = round_data.get("round", "")
            date = round_data.get("date", "")
            amount = round_data.get("amount", "")
            round_line = f"**{round_name}**"
            if date:
                round_line += f" ({date})"
            if amount:
                round_line += f": {amount}"
            lines.append(round_line)
            if lead := round_data.get("lead_investor"):
                lines.append(f"- Lead Investor: {lead}")
            if investors := round_data.get("investors"):
                if len(investors) == 1:
                    lines.append(f"- Investor: {investors[0]}")
                elif len(investors) > 1:
                    extra = f" and {len(investors) - 3} more" if len(investors) > 3 else ""
                    lines.append(f"- Investors: {', '.join(investors[:3])}{extra}")
        if company_investors := structured_data.get("investors"):
            lines.append("**Notable Investors:**")
            for inv in company_investors:
                if not isinstance(inv, dict):
                    lines.append(f"- {inv}")
                    continue
                inv_line = f"- {inv.get('name', '')}"
                meta = [v for v in (inv.get("type"), inv.get("stage")) if v]
                if inv.get("lead"):
                    meta.append("lead")
                if meta:
                    inv_line += f" ({', '.join(str(m) for m in meta)})"
                lines.append(inv_line)
        lines.append("")

    if partnerships := structured_data.get("partnerships"):
        lines.append("## 3b. Partnerships")
        for pship in partnerships:
            if not isinstance(pship, dict):
                lines.append(f"- {pship}")
                continue
            p_line = f"**{pship.get('partner', '')}**"
            if year := pship.get("year"):
                p_line += f" ({year})"
            lines.append(p_line)
            if pdesc := pship.get("description"):
                lines.append(f"- {pdesc}")
            if purl := pship.get("url"):
                lines.append(f"- Source: {purl}")
        lines.append("")

    products_section_added = False
    if products := structured_data.get("products"):
        products_section_added = True
        lines.append("## 4. Products & Technology")
        for product in products:
            name = product.get("name", "")
            desc = product.get("description", "")
            launch = product.get("launch_date", "")
            features = product.get("key_features", [])
            if name:
                product_line = f"**{name}**"
                if launch:
                    product_line += f" (Launched: {launch})"
                lines.append(product_line)
                if desc:
                    lines.append(f"- {desc}")
                if features:
                    lines.append("- Key Features:")
                    for feature in features:
                        lines.append(f"  - {feature}")
                lines.append("")
    if tech_stack := structured_data.get("tech_stack"):
        if not products_section_added:
            lines.append("## 4. Products & Technology")
            products_section_added = True
        lines.append("**Technology Stack:**")
        for i in range(0, len(tech_stack), 5):
            lines.append(f"- {', '.join(tech_stack[i:i + 5])}")
        lines.append("")

    market_section_added = False
    if market_pos := structured_data.get("market_position"):
        market_section_added = True
        lines.append("## 5. Market Position & Competitors")
        if strengths := market_pos.get("strengths"):
            lines.append("**Strengths:**")
            for strength in strengths:
                lines.append(f"- {strength}")
            lines.append("")
        if challenges := market_pos.get("challenges"):
            lines.append("**Challenges:**")
            for challenge in challenges:
                lines.append(f"- {challenge}")
            lines.append("")
        if market_share := market_pos.get("market_share"):
            lines.append(f"**Market Share:** {market_share}")
        if growth := market_pos.get("growth_trajectory"):
            lines.append(f"**Growth Trajectory:** {growth}")
        lines.append("")
    if competitors := structured_data.get("competitors"):
        if not market_section_added:
            lines.append("## 5. Market Position & Competitors")
            market_section_added = True
        lines.append("**Competitors:**")
        for competitor in competitors:
            lines.append(f"- {competitor}")
        lines.append("")

    if news := structured_data.get("recent_news"):
        lines.append("## 6. Recent News & Developments")
        for news_item in news:
            title = news_item.get("title", "")
            date = news_item.get("date", "")
            summary = news_item.get("summary", "")
            url = news_item.get("url", "")
            if title:
                news_title = f"**{title}**"
                if date:
                    news_title += f" ({date})"
                lines.append(news_title)
                if summary:
                    lines.append(f"- {summary}")
                if url:
                    lines.append(f"- Source: {url}")
                lines.append("")

    if sources := structured_data.get("sources", []):
        lines.append("\n## Sources")
        for idx, source in enumerate(sources):
            title = source.get("title", source.get("url", f"Source {idx + 1}"))
            url = source.get("url", "")
            if url:
                lines.append(f"{idx + 1}. [{title}]({url})")

    return "\n".join(lines)


# ===========================================================================
# Gemini-backed agents (now routed through common.gemini_generate_json so all
# token usage is recorded centrally and every call is timeout-bounded).
# ===========================================================================
async def entity_context_extraction_agent(text: str) -> Dict[str, Any]:
    """Extract company names + additional context from free text (JSON mode)."""
    parsing_prompt = f"""
    Extract company names and additional context from the following information:

    {text}

    Return your response in JSON format with the following structure:
    {{
        "entity_names": ["company1", "company2", ...],
        "additional_context": "Any additional context about the companies"
    }}

    When extracting entity names:
    1. Focus on full, official company names when possible
    2. If industry is mentioned, include it in additional_context
    3. If location, founding date, or size is mentioned, include it in additional_context
    4. If product names are mentioned, include them in additional_context

    If no company names are mentioned, return an empty array for entity_names.
    """
    result = await gemini_generate_json(
        parsing_prompt, model=GEMINI_MODEL, agent_name="Entity Extraction Agent"
    )
    # Honest fallback: gemini_generate_json returns {} on any failure.
    if not isinstance(result, dict):
        return {"entity_names": [], "additional_context": ""}
    result.setdefault("entity_names", [])
    result.setdefault("additional_context", "")
    if not isinstance(result["entity_names"], list):
        result["entity_names"] = []
    return result


async def disambiguate_entities(
    user_input: str,
    intended_context: str,
    entity_info: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """ADDITIVE entity disambiguation: score each candidate as SAME vs DIFFERENT.

    Given the user's original input (the company they *mean*), any extra context
    they supplied, and the gathered candidates (name + source-backed snippet),
    ask Gemini to return, per candidate index, a ``confidence`` (0.0-1.0) that
    the candidate is the SAME company the user means, plus a short ``reason``.

    Negative-example aware (ported from component3's differentiation framework):
    the prompt explicitly tells the model to LOWER confidence on founder
    mismatch, founding-year mismatch, and similarly-named-but-different entities
    (e.g. "X" vs "X AI" / "X Bio" suffix collisions).

    HONEST STATUS: returns ``{}`` (callers degrade to current behavior, no
    confidence column shown) when Gemini is unavailable or returns nothing. This
    never auto-selects or auto-removes — it only surfaces a hint for the user.

    Returns ``{"<index>": {"confidence": float, "reason": str}, ...}`` keyed by
    the candidate's position in ``entity_info``.
    """
    if not entity_info:
        return {}

    candidate_blocks: List[str] = []
    for i, ent in enumerate(entity_info):
        name = safe_get_attribute(ent, "name", f"Candidate {i}")
        snippet = (safe_get_attribute(ent, "snippet", "") or "")[:1500]
        domains: List[str] = []
        for src in (safe_get_attribute(ent, "sources", []) or [])[:3]:
            url = safe_get_attribute(src, "url", None)
            if url:
                try:
                    domains.append(urlparse(url).netloc)
                except Exception:  # noqa: BLE001
                    pass
        candidate_blocks.append(
            f"--- CANDIDATE {i} ---\n"
            f"Name: {name}\n"
            f"Source domains: {', '.join(domains) if domains else 'none'}\n"
            f"Snippet: {snippet or 'No description available.'}"
        )
    candidates_text = "\n\n".join(candidate_blocks)

    prompt = f"""
    You are disambiguating which candidate company matches the entity the user
    actually means. Companies with similar names are frequently confused (for
    example "Anthropic" vs an unrelated "Anthropic AI", or "X" vs "X Bio").

    THE USER IS LOOKING FOR (original request):
    {user_input}

    EXTRA CONTEXT THE USER PROVIDED (may be empty):
    {intended_context or 'N/A'}

    CANDIDATES (each is a separate, possibly-different real-world entity):
    {candidates_text}

    For EACH candidate index, decide how confident you are that it is the SAME
    company the user means (NOT a similarly-named different one). Use this
    differentiation framework and LOWER confidence when you see conflicts:
    - FOUNDER MISMATCH: candidate's founders differ from what the user implies.
    - FOUNDING-YEAR MISMATCH: candidate founded in a different year.
    - NAME-SUFFIX COLLISION: candidate is "<Name> AI" / "<Name> Bio" / "<Name>
      Labs" etc. while the user means the bare name (or vice-versa).
    - INDUSTRY / LOCATION / PRODUCT mismatch vs the user's context.
    Raise confidence when founders, founding year, industry, location, product,
    or website align with the user's request and context.

    Return ONLY a JSON object of this exact shape (confidence is a float 0.0-1.0,
    reason is one short sentence). Include an entry for every candidate index:
    {{
        "scores": [
            {{"index": 0, "confidence": 0.92, "reason": "Founder and 2014 founding year match the user's context."}},
            {{"index": 1, "confidence": 0.15, "reason": "Different founder; 'X AI' is a separate entity from the bare 'X' the user means."}}
        ]
    }}
    """
    result = await gemini_generate_json(
        prompt, model=GEMINI_MODEL, agent_name="Entity Disambiguation Agent"
    )
    if not isinstance(result, dict):
        return {}
    scores = result.get("scores")
    if not isinstance(scores, list):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for item in scores:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(entity_info):
            continue
        try:
            conf = float(item.get("confidence"))
        except (TypeError, ValueError):
            continue
        # Clamp to [0, 1] so a bad model value can never display as >100%.
        conf = max(0.0, min(1.0, conf))
        reason = str(item.get("reason", "") or "").strip()
        out[str(idx)] = {"confidence": conf, "reason": reason}
    return out


async def analyze_missing_fields(structured_data: Dict[str, Any]) -> Dict[str, Any]:
    """Identify important empty fields in the schema and suggest queries."""
    analysis_prompt = f"""
    Analyze the following company data and identify important missing fields:

    ```json
    {json.dumps(structured_data, indent=2, default=str)}
    ```

    For each missing or empty field that's important, provide:
    1. The field name (use exact dot notation for nested fields)
    2. Importance level (high, medium, low)
    3. A specific targeted query to find that information

    Return ONLY a JSON object with this structure:
    {{
        "missing_fields": [
            {{
                "field_name": "products",
                "importance": "high",
                "query_suggestion": "What are the main products and services of [company]?"
            }}
        ]
    }}

    Prioritize missing high-importance fields like company_description, headquarters, industry,
    leadership_team, products, and recent_news if they're empty.
    """
    result = await gemini_generate_json(
        analysis_prompt, model=GEMINI_MODEL, agent_name="Missing Fields Analyzer"
    )
    if not isinstance(result, dict) or "missing_fields" not in result:
        return {"missing_fields": []}
    if not isinstance(result["missing_fields"], list):
        return {"missing_fields": []}
    return result


# ===========================================================================
# Nested-update helper (ported)
# ===========================================================================
def apply_nested_update(data_dict, path_str, value) -> None:
    """Apply ``value`` at a dot-separated ``path_str`` inside ``data_dict``."""
    keys = path_str.split(".")
    current = data_dict
    for key in keys[:-1]:
        if key.isdigit() and isinstance(current, list):
            key = int(key)
            while len(current) <= key:
                current.append({})
            current = current[key]
        elif "[" in key and key.endswith("]"):
            base_key, idx_str = key.split("[", 1)
            idx = int(idx_str.rstrip("]"))
            if base_key not in current:
                current[base_key] = []
            while len(current[base_key]) <= idx:
                current[base_key].append({})
            current = current[base_key][idx]
        else:
            if key not in current:
                current[key] = {}
            current = current[key]
    final_key = keys[-1]
    if final_key.isdigit() and isinstance(current, list):
        final_key = int(final_key)
        while len(current) <= final_key:
            current.append(None)
        current[final_key] = value
    elif "[" in final_key and final_key.endswith("]"):
        base_key, idx_str = final_key.split("[", 1)
        idx = int(idx_str.rstrip("]"))
        if base_key not in current:
            current[base_key] = []
        while len(current[base_key]) <= idx:
            current[base_key].append(None)
        current[base_key][idx] = value
    else:
        current[final_key] = value


# ===========================================================================
# THREE-PASS RESEARCH (ported; LinkUp/Gemini routed through common.*)
# ===========================================================================
async def first_pass_entity_selection(user_input: str, observability_log: List[Dict]) -> Dict[str, Any]:
    """Pass 1: extract candidate entities + a sourcedAnswer snippet for each."""
    start_time = time.time()
    log_entry: Dict[str, Any] = {"step": "Pass 1", "input_query": user_input}
    try:
        extraction_result = await entity_context_extraction_agent(f"User input: {user_input}")
        potential_entities = extraction_result.get("entity_names", [])
        initial_context = extraction_result.get("additional_context", "")
        log_entry["gemini_entities"] = potential_entities
        log_entry["gemini_context"] = initial_context

        if not potential_entities:
            log_entry["status"] = "Failed - No entities identified"
            log_entry["duration"] = time.time() - start_time
            observability_log.append(log_entry)
            return {
                "success": False,
                "error": "No entities identified in input text",
                "entities": [],
                "selected_entity": None,
                "negative_entities": [],
                "additional_context": initial_context,
            }

        disambiguation_prompt = """
        Provide a brief description and key identifying information for the company: {entity_name}

        Focus on:
        1. What the company does (main business/products)
        2. When it was founded and where it's headquartered
        3. Industry sector
        4. Company size if available
        5. Official website URL

        Use reliable sources and verify the information relates to the correct company.
        """
        linkup_tasks = [
            company_search_with_linkup(entity, additional_context=disambiguation_prompt.format(entity_name=entity))
            for entity in potential_entities
        ]
        search_results = await asyncio.gather(*linkup_tasks, return_exceptions=True)

        entity_info: List[Dict[str, Any]] = []
        log_entry["linkup_calls"] = []
        for i, result in enumerate(search_results):
            entity_name = potential_entities[i]
            linkup_log: Dict[str, Any] = {"entity": entity_name}
            if isinstance(result, Exception):
                linkup_log["status"] = "Error"
                linkup_log["error"] = str(result)
                entity_info.append({"name": entity_name, "snippet": "Error fetching info", "sources": []})
            elif result.get("success", False):
                linkup_log["status"] = "Success"
                linkup_log["num_sources"] = len(result.get("sources", []))
                entity_info.append({
                    "name": entity_name,
                    "snippet": result.get("answer", "No description available."),
                    "sources": result.get("sources", []),
                    "search_text_formatted": result.get("search_text_formatted", ""),
                })
            else:
                linkup_log["status"] = "Failed"
                linkup_log["error"] = result.get("error", "Unknown LinkUp error")
                entity_info.append({"name": entity_name, "snippet": f"Failed: {result.get('error')}", "sources": []})
            log_entry["linkup_calls"].append(linkup_log)

        # ADDITIVE: entity disambiguation. Attach a SAME-vs-DIFFERENT confidence
        # (0-1) + short reason to each candidate so the resolution UI can guide
        # the user. HONEST STATUS / SAFE: if Gemini is unavailable (or errors)
        # this returns {} and we leave the candidates untouched — no auto-select,
        # no auto-remove, exact prior behavior preserved.
        try:
            disambig = await disambiguate_entities(
                user_input=user_input,
                intended_context=initial_context,
                entity_info=entity_info,
            )
        except Exception as dis_err:  # noqa: BLE001 - best-effort enhancement only
            logger.warning("Entity disambiguation skipped: %s", dis_err)
            disambig = {}
        if disambig:
            for i, ent in enumerate(entity_info):
                score = disambig.get(str(i))
                if score:
                    ent["confidence"] = score.get("confidence")
                    ent["confidence_reason"] = score.get("reason", "")
            log_entry["disambiguation"] = "ok"
        else:
            log_entry["disambiguation"] = "unavailable"

        log_entry["status"] = "Success"
        log_entry["duration"] = time.time() - start_time
        observability_log.append(log_entry)
        return {
            "success": True,
            "entities": entity_info,
            "selected_entity": None,
            "negative_entities": [],
            "additional_context": initial_context,
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Error in first_pass_entity_selection: %s", e)
        log_entry["status"] = "Failed - Exception"
        log_entry["error"] = str(e)
        log_entry["duration"] = time.time() - start_time
        observability_log.append(log_entry)
        return {
            "success": False,
            "error": str(e),
            "entities": [],
            "selected_entity": None,
            "negative_entities": [],
            "additional_context": "",
        }


async def _extract_structured_data_from_answer(
    company_name: str,
    answer_text: str,
    sources: List[Any],
    agent_name: str = "Schema Extractor",
) -> Optional[Dict[str, Any]]:
    """Convert a LinkUp ``sourcedAnswer`` TEXT into the company schema via Gemini.

    LinkUp returns a source-backed natural-language answer, not schema JSON.
    This step asks Gemini to extract the ``LINKUP_COMPANY_SCHEMA`` fields from
    that answer text (plus the source list) and return a JSON object.

    HONEST STATUS: returns ``None`` (never ``{}`` and never fabricated data)
    when there is no answer to extract from, when Gemini is unavailable, or
    when Gemini returns an empty/invalid result. Callers must treat ``None``
    as "structured enhancement unavailable" and show the source-backed answer
    only, rather than rendering an empty ``{}`` as if it were a result.
    """
    if not answer_text or not answer_text.strip():
        return None

    sources_lines: List[str] = []
    for i, src in enumerate(sources or []):
        src_name = safe_get_attribute(src, "name", f"Source {i + 1}")
        src_url = safe_get_attribute(src, "url", "N/A")
        sources_lines.append(f"- {src_name}: {src_url}")
    sources_block = "\n".join(sources_lines) if sources_lines else "N/A"

    schema_json_string = json.dumps(LINKUP_COMPANY_SCHEMA, indent=2)
    extraction_prompt = f"""
    Extract structured company information for "{company_name}" STRICTLY from the
    source-backed answer text and source list below. Do NOT invent or infer any
    fact that is not supported by the provided text. If a field is not supported
    by the text, use an empty/null JSON value (`[]`, `""`, or `null`).

    --- SOURCE-BACKED ANSWER TEXT ---
    {answer_text}

    --- SOURCES ---
    {sources_block}

    Return ONLY a single JSON object that conforms to this schema (same keys and
    nesting). Output JSON only, no prose.

    --- JSON SCHEMA ---
    ```json
    {schema_json_string}
    ```
    """
    result = await gemini_generate_json(
        extraction_prompt, model=GEMINI_MODEL, agent_name=agent_name
    )
    # HONEST STATUS: {} / non-dict / "result"-wrapped non-dict -> unavailable.
    if not isinstance(result, dict) or not result or "result" in result:
        return None
    return result


async def second_pass_structured_search(
    selected_entity: Dict[str, Any],
    negative_entities: List[Dict[str, Any]],
    additional_context: str,
    observability_log: List[Dict],
) -> Dict[str, Any]:
    """Pass 2: detailed sourcedAnswer research, schema-guided."""
    start_time = time.time()
    log_entry: Dict[str, Any] = {"step": "Pass 2", "target_entity": selected_entity.get("name")}
    try:
        positive_context = f"**TARGET ENTITY:** {selected_entity['name']}\n"
        if selected_entity.get("sources"):
            source_urls = [
                safe_get_attribute(s, "url", None)
                for s in selected_entity["sources"]
                if safe_get_attribute(s, "url", None)
            ][:5]
            if source_urls:
                positive_context += "Key URLs (verify against these):\n" + "\n".join(f"- {u}" for u in source_urls) + "\n"
        if additional_context:
            positive_context += f"\n**Additional Context:** {additional_context}"

        negative_context_str = "**ENTITIES TO IGNORE:**\n"
        if negative_entities:
            negative_context_str += "\n".join(
                f"- {safe_get_attribute(neg, 'name', 'Unknown')}" for neg in negative_entities
            )
        else:
            negative_context_str += "N/A"

        schema_json_string = json.dumps(LINKUP_COMPANY_SCHEMA, indent=2)
        structured_prompt_pass2 = f"""
        Perform comprehensive research analysis for the TARGET ENTITY, generating JSON output matching the schema. Ensure information pertains ONLY to the TARGET.

        {positive_context}

        {negative_context_str}

        **RESEARCH SCOPE & JSON SCHEMA:**
        - Adhere strictly to the JSON schema.
        - Fill fields based *only* on the TARGET ENTITY. Use info from its key URLs if possible.
        - Prioritize official sources, recent news (last 6 months).
        - Use JSON null/empty values (`[]`, `""`, `null`) if info for TARGET is not found. Do NOT use data from IGNORED entities.
        - Output the result as a JSON block: ```json ... ```

        **JSON Schema Definition:**
        ```json
        {schema_json_string}
        ```
        """
        log_entry["input_prompt"] = structured_prompt_pass2[:500] + "..."

        search_result = await company_search_with_linkup(
            selected_entity["name"],
            additional_context=structured_prompt_pass2,
        )

        # Schema-guided structured output: LinkUp returns a source-backed TEXT
        # answer, not schema JSON. Convert that answer (+ its sources) into the
        # company schema with a Gemini JSON-extraction step. HONEST STATUS: if
        # Gemini is unavailable or returns {} (e.g. no/invalid key), leave
        # structured_data=None so the UI shows the source-backed answer only and
        # never renders an empty {} as if it were a result.
        structured_data = await _extract_structured_data_from_answer(
            company_name=selected_entity["name"],
            answer_text=search_result.get("answer", "") or "",
            sources=search_result.get("sources", []) or [],
            agent_name="Pass 2 Schema Extractor",
        ) if search_result.get("success", False) else None

        log_entry["linkup_output_type"] = "sourcedAnswer"
        log_entry["linkup_num_sources"] = len(search_result.get("sources", []))
        log_entry["structured_extraction"] = "ok" if structured_data else "unavailable"
        log_entry["status"] = "Success" if search_result.get("success", False) else "Failed"
        log_entry["error"] = search_result.get("error")
        log_entry["duration"] = time.time() - start_time
        observability_log.append(log_entry)

        return {
            "success": search_result.get("success", False),
            "answer": search_result.get("answer", ""),
            "sources": search_result.get("sources", []),
            "structured_data": structured_data,
            "search_results": search_result.get("search_text_formatted", ""),
            "search_text_formatted": search_result.get("search_text_formatted", ""),
            "error": search_result.get("error"),
            "parsing_error": search_result.get("parsing_error"),
            "raw_response": search_result.get("raw_response"),
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Error in second_pass_structured_search: %s", e)
        log_entry["status"] = "Failed - Exception"
        log_entry["error"] = str(e)
        log_entry["duration"] = time.time() - start_time
        observability_log.append(log_entry)
        return {
            "success": False,
            "error": str(e),
            "answer": "",
            "sources": [],
            "structured_data": None,
            "search_results": f"Error in detailed search: {e}",
            "search_text_formatted": f"Error in detailed search: {e}",
            "parsing_error": None,
            "raw_response": None,
        }


async def third_pass_fill_missing_info(
    structured_data: Optional[Dict[str, Any]],
    selected_entity: Dict[str, Any],
    negative_entities: List[Dict[str, Any]],
    pass2_sources: List[Dict],
    observability_log: List[Dict],
) -> Dict[str, Any]:
    """Pass 3: find important missing fields and backfill from targeted searches."""
    start_time = time.time()
    log_entry: Dict[str, Any] = {"step": "Pass 3", "target_entity": selected_entity.get("name")}

    if not structured_data:
        # HONEST STATUS: Pass 2 produced no structured_data (e.g. Gemini schema
        # extraction was unavailable). Return None (not {}) so the UI can show a
        # "structured enhancement unavailable" notice instead of an empty {}.
        log_entry["status"] = "Skipped - No structured data from Pass 2"
        log_entry["duration"] = 0
        observability_log.append(log_entry)
        return {"success": True, "structured_data": None, "enhanced": False, "enhanced_fields": []}

    try:
        enhanced_data = copy.deepcopy(structured_data)
        enhanced_fields: List[str] = []
        all_pass3_linkup_calls: List[Dict[str, Any]] = []

        missing_fields_analysis = await analyze_missing_fields(structured_data)
        missing_fields = missing_fields_analysis.get("missing_fields", [])
        high_importance_fields = [f for f in missing_fields if f.get("importance") in ("high", "medium")][:3]
        log_entry["gemini_missing_fields"] = high_importance_fields

        if not high_importance_fields:
            log_entry["status"] = "Success - No important fields missing"
            log_entry["duration"] = time.time() - start_time
            observability_log.append(log_entry)
            return {"success": True, "structured_data": structured_data, "enhanced": False, "enhanced_fields": []}

        context_str = f"""
        **TARGET ENTITY (Focus ONLY on this one):**
        {selected_entity['name']} (Key URLs: {', '.join([safe_get_attribute(s, 'url', 'N/A') for s in selected_entity.get('sources', [])[:3]])})

        **ENTITIES TO IGNORE:**
        {', '.join([safe_get_attribute(neg, 'name', 'N/A') for neg in negative_entities]) if negative_entities else 'N/A'}
        """

        linkup_tasks = []
        for field_info in high_importance_fields:
            field_path = field_info.get("field_name", "")
            query_suggestion = field_info.get("query_suggestion", "")
            targeted_query = f"""
            Regarding *only* the TARGET ENTITY, find information for: '{query_suggestion}'?

            {context_str}

            Provide the answer based *only* on verifiable sources related to the TARGET ENTITY.
            State the answer clearly. Include supporting source snippets and URLs.
            If unknown, state 'Unknown based on available sources for the target'.
            """
            linkup_tasks.append(
                (field_path, company_search_with_linkup(selected_entity.get("name", "Unknown"), additional_context=targeted_query))
            )

        targeted_results = await asyncio.gather(*[t[1] for t in linkup_tasks], return_exceptions=True)

        combined_pass3_evidence = ""
        for i, result in enumerate(targeted_results):
            field_path = linkup_tasks[i][0]
            linkup_log: Dict[str, Any] = {"field": field_path}
            if isinstance(result, Exception):
                linkup_log["status"] = "Error"
                linkup_log["error"] = str(result)
            elif result.get("success", False) and result.get("answer"):
                linkup_log["status"] = "Success"
                linkup_log["num_sources"] = len(result.get("sources", []))
                combined_pass3_evidence += f"\n\n--- Evidence for field: {field_path} ---\n"
                combined_pass3_evidence += f"Answer Found: {result.get('answer', '')}\n"
                if result.get("sources"):
                    combined_pass3_evidence += "Supporting Sources:\n" + "\n".join(
                        f"- {safe_get_attribute(s, 'url', 'N/A')} ({safe_get_attribute(s, 'snippet', '')[:80]}...)"
                        for s in result.get("sources", [])
                    )
                else:
                    combined_pass3_evidence += "No specific sources returned for this answer.\n"
            else:
                linkup_log["status"] = "Failed / No Answer"
                linkup_log["error"] = result.get("error", "Unknown error")
            all_pass3_linkup_calls.append(linkup_log)

        log_entry["pass3_linkup_calls"] = all_pass3_linkup_calls

        if combined_pass3_evidence:
            integration_prompt = f"""
            Existing company data (potentially incomplete):
            ```json
            {json.dumps(structured_data, indent=2, default=str)}
            ```

            New evidence found for specific fields:
            {combined_pass3_evidence}

            Based *only* on the new evidence provided above, identify updates for the existing JSON data.
            *   Only update fields if the evidence provides a clear, verifiable answer for the TARGET company.
            *   Verify the source URLs seem relevant to the target entity if possible.
            *   Format the updates strictly as a JSON object: {{ "field_path1": "new_value", "field_path2": [...] }}
            *   If evidence confirms a field is empty/null for the target, represent that appropriately (e.g., `[]`, `null`, `""`).
            *   If evidence is insufficient or contradictory for a field, do not include it in the update JSON.

            Return ONLY the JSON object containing the updates.
            """
            updates = await gemini_generate_json(
                integration_prompt, model=GEMINI_MODEL, agent_name="Pass 3 Integrator"
            )
            log_entry["gemini_parsed_updates"] = updates
            if isinstance(updates, dict) and updates and "result" not in updates:
                for path, value in updates.items():
                    try:
                        apply_nested_update(enhanced_data, path, value)
                        enhanced_fields.append(path)
                    except Exception as apply_err:  # noqa: BLE001
                        logger.error("Pass 3: failed to apply update for %s: %s", path, apply_err)
                        log_entry.setdefault("integration_errors", []).append(f"Apply error for {path}: {apply_err}")

        log_entry["enhanced_fields"] = enhanced_fields
        log_entry["status"] = "Success"
        log_entry["duration"] = time.time() - start_time
        observability_log.append(log_entry)
        return {
            "success": True,
            "structured_data": enhanced_data,
            "enhanced": len(enhanced_fields) > 0,
            "enhanced_fields": enhanced_fields,
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Error in third_pass_fill_missing_info: %s", e)
        log_entry["status"] = "Failed - Exception"
        log_entry["error"] = str(e)
        log_entry["duration"] = time.time() - start_time
        observability_log.append(log_entry)
        return {
            "success": False,
            "error": str(e),
            "structured_data": structured_data,
            "enhanced": False,
            "enhanced_fields": [],
        }


# ===========================================================================
# STANDARD (single-shot, parallel) SEARCH (ported; common.* primitives)
# ===========================================================================
async def gemini_company_search_with_verification(
    user_input: str = "",
    company_names: Optional[List[str]] = None,
    additional_context: str = "",
    status_container=None,
    max_concurrent: int = 5,
    enable_web_scraping: bool = True,
) -> Dict[str, Any]:
    """Parse -> (optionally scrape context) -> parallel LinkUp search -> verify."""
    start_time = time.time()
    try:
        semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

        if status_container:
            status_container.update(label="Step 1: Parsing company names and additional context...")

        if not company_names:
            extraction_result = await entity_context_extraction_agent(f"User input: {user_input}")
            company_names = extraction_result.get("entity_names", [])
            if not additional_context and extraction_result.get("additional_context"):
                additional_context = extraction_result.get("additional_context", "").strip()

        if not company_names:
            if status_container:
                status_container.update(
                    label="No company names could be identified in your query. Please try again with clearer company information.",
                    state="error",
                )
            return {
                "success": False,
                "error": "No company names could be identified in your query",
                "parsed_entities": [],
                "search_results": [],
                "verified_results": [],
            }

        step1_time = time.time() - start_time

        # Step 1.5: optional web-scraping context enhancement (common.*).
        step15_time = 0.0
        if enable_web_scraping and additional_context:
            if status_container:
                status_container.update(label="Step 1.5: Enhancing context by scraping URLs...")
            step15_start = time.time()
            try:
                additional_context = await scrape_urls_from_context(
                    additional_context, max_concurrent=max(1, max_concurrent // 2)
                )
            except Exception as e:  # noqa: BLE001 - scraping is best-effort
                logger.warning("Context enhancement failed: %s", e)
            step15_time = time.time() - step15_start

        if status_container:
            status_container.update(
                label=f"Step 2: Searching for information on {len(company_names)} companies in parallel..."
            )
        step2_start = time.time()

        async def rate_limited_search(company_name: str) -> Dict[str, Any]:
            async with semaphore:
                company_specific_context = extract_context_for_company(additional_context, company_name)
                # Build a research prompt so LinkUp returns rich, sourced text.
                prompt = company_specific_context or company_name
                search_result = await company_search_with_linkup(company_name, additional_context=prompt)
                return {
                    "company_name": company_name,
                    "search_results": search_result.get("search_text_formatted", ""),
                    "structured_data": search_result.get("structured_data"),
                    "success": search_result.get("success", False),
                }

        search_tasks = [asyncio.create_task(rate_limited_search(c)) for c in company_names]
        gathered = await asyncio.gather(*search_tasks, return_exceptions=True)

        all_search_results = []
        for result in gathered:
            if isinstance(result, Exception):
                logger.error("Search task failed: %s", result)
            elif result.get("search_results"):
                all_search_results.append(result)
            if status_container:
                status_container.update(label=f"Step 2: Completed {len(all_search_results)}/{len(company_names)} searches")

        step2_time = time.time() - step2_start

        # Step 3: verification (Gemini) when extra context is available.
        verified_results: List[Dict[str, Any]] = []
        step3_time = 0.0
        if additional_context and all_search_results:
            if status_container:
                status_container.update(label="Step 3: Verifying results against additional context...")
            step3_start = time.time()

            async def rate_limited_verification(company_name: str, search_results_text: str) -> Dict[str, Any]:
                async with semaphore:
                    company_specific_context = extract_context_for_company(additional_context, company_name)
                    verification_text = await _verify_results(company_name, company_specific_context, search_results_text)
                    return {"company_name": company_name, "verification": verification_text}

            verification_tasks = [
                asyncio.create_task(rate_limited_verification(r["company_name"], r["search_results"]))
                for r in all_search_results
            ]
            verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
            for result in verification_results:
                if isinstance(result, Exception):
                    logger.error("Verification task failed: %s", result)
                elif result.get("verification"):
                    verified_results.append(result)
            step3_time = time.time() - step3_start

        total_time = time.time() - start_time
        if status_container:
            status_container.update(
                label=f"Search complete! Processed {len(company_names)} companies in {format_time(total_time)}",
                state="complete",
            )

        return {
            "success": True,
            "parsed_entities": company_names,
            "additional_context": additional_context,
            "search_results": all_search_results,
            "verified_results": verified_results if additional_context else [],
            "timing": {
                "step1_time": step1_time,
                "step15_time": step15_time,
                "step2_time": step2_time,
                "step3_time": step3_time,
                "total_time": total_time,
            },
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Error in gemini_company_search_with_verification: %s", e)
        if status_container:
            status_container.update(label=f"Error: {e}", state="error")
        return {
            "success": False,
            "error": str(e),
            "parsed_entities": company_names or [],
            "search_results": [],
            "verified_results": [],
        }


async def _verify_results(company_name: str, additional_context: str, search_results: str) -> str:
    """Gemini verification of search results vs. known context (free-form text)."""
    context_info = extract_disambiguated_company_info(company_name, additional_context)
    verification_prompt = f"""
    # Company Verification Task

    ## COMPANY TO VERIFY
    Company name: {company_name}

    ## KNOWN INFORMATION
    """
    if any(context_info.values()):
        for label, field in (
            ("Industry/Sector", "industry"),
            ("Location/HQ", "location"),
            ("Founded", "founded"),
            ("Founder(s)/CEO", "founders"),
            ("Product/Service", "product"),
            ("Company Size", "size"),
            ("Website", "website"),
        ):
            if context_info[field]:
                verification_prompt += f"- {label}: {context_info[field]}\n"
    else:
        verification_prompt += "- Limited additional context available\n"

    verification_prompt += f"""
    ## SEARCH RESULTS TO VERIFY
    {search_results}

    ## VERIFICATION TASK
    1. Compare the search results with the known information
    2. Determine if the search results refer to the same company as described
    3. Identify any discrepancies or conflicting information
    4. Highlight which information is confirmed by multiple sources

    ## RESPONSE FORMAT

    VERIFICATION RESULT:
    [State whether search results match the expected company]

    CONFIRMED INFORMATION:
    [List information points confirmed by multiple sources]

    DISCREPANCIES OR CONFLICTS:
    [List any discrepancies or conflicting information]

    MISSING INFORMATION:
    [List any important information that is missing]

    RECOMMENDATION:
    [Suggest if additional research is needed and on which aspects]
    """
    return await common.gemini_generate_text(
        verification_prompt, model=GEMINI_MODEL, agent_name=f"Verification Agent ({company_name})"
    )


# ===========================================================================
# Spreadsheet parsing (ported parse_excel_file with row-wise Gemini fallback)
# ===========================================================================
def _read_uploaded_dataframe(uploaded_file):
    """Read an uploaded xlsx/xls/csv into a DataFrame."""
    raw = uploaded_file.read()
    buf = io.BytesIO(raw)
    buf.seek(0)
    name = (uploaded_file.name or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(buf)
    return pd.read_excel(buf)


async def parse_excel_file(uploaded_file) -> "pd.DataFrame":
    """Parse uploaded spreadsheet -> company_name + additional_context df.

    Detects company-like columns; for messy sheets with no obvious company
    column, runs row-wise Gemini entity extraction as a fallback. Bounded to
    ``MAX_UPLOAD_ROWS`` rows so a huge sheet can't fan out unbounded LLM calls.
    """
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas is not installed; spreadsheet upload unavailable.")

    df = _read_uploaded_dataframe(uploaded_file)
    if len(df) > MAX_UPLOAD_ROWS:
        logger.warning("Spreadsheet has %d rows; capping at %d.", len(df), MAX_UPLOAD_ROWS)
        df = df.head(MAX_UPLOAD_ROWS)

    company_cols = [
        col for col in df.columns
        if any(re.search(rf"\b{name}\b", str(col).lower()) for name in ("company", "organization", "entity", "name"))
    ]

    result_df = pd.DataFrame()

    if company_cols:
        company_col = company_cols[0]
        result_df["company_name"] = df[company_col].astype(str)
        cols_to_combine = [c for c in df.columns if c != company_col]
        if cols_to_combine:
            context_data = []
            for _, row in df.iterrows():
                row_context = ", ".join(f"{c}: {row[c]}" for c in cols_to_combine if pd.notna(row[c]))
                context_data.append(row_context)
            result_df["additional_context"] = context_data
        else:
            result_df["additional_context"] = ""
    else:
        # Messy sheet: row-wise Gemini entity extraction fallback.
        company_names: List[str] = []
        contexts: List[str] = []
        raw_rows: List[str] = []
        for idx, row in df.iterrows():
            row_info = ", ".join(f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c]))
            raw_rows.append(row_info)
            extraction_result = await entity_context_extraction_agent(row_info)
            names = extraction_result.get("entity_names", [])
            company_name = names[0] if names else f"Row {idx + 1}"
            additional_context = extraction_result.get("additional_context", "")
            company_names.append(company_name)
            contexts.append(additional_context if additional_context else row_info)

        if not company_names or all(str(n).startswith("Row ") for n in company_names):
            raise ValueError(
                "Could not identify any company names in the uploaded file. "
                "Please try a file with clearer company information."
            )
        result_df["company_name"] = company_names
        result_df["additional_context"] = contexts
        result_df["raw_data"] = raw_rows

    result_df["status"] = "Pending"
    return result_df


async def process_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """Wrap parse_excel_file with honest success/error reporting."""
    try:
        editor_df = await parse_excel_file(uploaded_file)
        return {"success": True, "data": editor_df, "error": None}
    except Exception as e:  # noqa: BLE001
        logger.error("Error processing uploaded file: %s", e)
        return {"success": False, "data": None, "error": str(e)}


# ===========================================================================
# BATCH PROCESSING (ported; concurrency-limited; only Pending rows)
# ===========================================================================
async def batch_process_companies(
    search_df, max_concurrent=5, enable_web_scraping=True, progress_state=None
) -> Dict[str, Any]:
    """Process every row of ``search_df`` concurrently (bounded).

    ``progress_state``: optional shared dict (lives on the main thread, mutated
    here from the background loop) used to surface per-row progress. We only do
    plain dict writes of integers/strings — atomic under CPython's GIL — so the
    main thread polling it never needs a lock. Concurrency stays bounded by the
    ``max_concurrent`` semaphore regardless of progress reporting.
    """
    all_results: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

    def _mark(idx, status: str, name: str) -> None:
        if progress_state is None:
            return
        try:
            rows = progress_state.setdefault("rows", {})
            rows[idx] = {"status": status, "name": name}
            if status in ("complete", "error"):
                progress_state["done"] = progress_state.get("done", 0) + 1
        except Exception:  # never let progress reporting break the batch
            pass

    async def process_company(idx, row) -> Dict[str, Any]:
        async with semaphore:
            company_name = str(row["company_name"])
            additional_context = str(row.get("additional_context", "") or "")
            _mark(idx, "running", company_name)
            try:
                response = await gemini_company_search_with_verification(
                    company_names=[company_name],
                    additional_context=additional_context,
                    status_container=None,
                    max_concurrent=1,
                    enable_web_scraping=enable_web_scraping,
                )
                _mark(idx, "complete" if response.get("success", False) else "error", company_name)
                return {
                    "idx": idx,
                    "company_name": company_name,
                    "additional_context": additional_context,
                    "result": response,
                    "success": response.get("success", False),
                }
            except Exception as e:  # noqa: BLE001
                logger.error("Error processing company %s: %s", company_name, e)
                _mark(idx, "error", company_name)
                return {
                    "idx": idx,
                    "company_name": company_name,
                    "additional_context": additional_context,
                    "result": {"success": False, "error": str(e)},
                    "success": False,
                }

    start_time = time.time()
    tasks = [asyncio.create_task(process_company(idx, row)) for idx, row in search_df.iterrows()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error("Batch task failed: %s", result)
            all_results.append({
                "idx": -1,
                "company_name": "Unknown",
                "additional_context": "",
                "result": {"success": False, "error": str(result)},
                "success": False,
            })
        else:
            all_results.append(result)

    all_results = sorted([r for r in all_results if r["idx"] != -1], key=lambda x: str(x["idx"]))
    total_time = time.time() - start_time
    return {"results": all_results, "total_time": total_time}


def process_companies_with_progress(search_df, async_runner, max_concurrent=5, enable_web_scraping=True):
    """Run the batch on the background loop and poll with a Streamlit status.

    Shows VISIBLE per-row progress: an ``st.progress`` bar (done/total), a
    completed-count + simple ETA line, and a per-company status list. The
    background coroutine mutates a shared ``progress_state`` dict that this main
    thread polls once per second; concurrency stays bounded by ``max_concurrent``
    inside ``batch_process_companies`` (this only reads, never adds parallelism).
    """
    total_companies = len(search_df)
    status = st.status(f"Preparing to search for {total_companies} companies...", expanded=True)
    progress_bar = st.progress(0.0, text=f"0/{total_companies} companies done")
    progress_placeholder = st.empty()

    company_statuses = {idx: {"status": "pending", "name": str(row["company_name"])}
                        for idx, row in search_df.iterrows()}
    progress_placeholder.text("\n".join(f"[wait] {cs['name']}" for cs in company_statuses.values()))

    # Shared progress dict mutated by the background coroutine (atomic dict
    # writes under the GIL); polled here on the main thread.
    progress_state: Dict[str, Any] = {"done": 0, "rows": {}}

    batch_future = async_runner.run_task_async(
        batch_process_companies(search_df, max_concurrent, enable_web_scraping, progress_state)
    )

    start_time = time.time()

    def _render_progress(done: int, rows: Dict[Any, Dict[str, str]]) -> None:
        # Merge live row statuses over the initial pending snapshot.
        merged = dict(company_statuses)
        for idx, cs in rows.items():
            merged[idx] = cs
        done = min(done, total_companies)
        frac = (done / total_companies) if total_companies else 1.0
        elapsed = time.time() - start_time
        eta_txt = ""
        if 0 < done < total_companies:
            per = elapsed / done
            remaining = per * (total_companies - done)
            eta_txt = f" | ETA ~{format_time(remaining)}"
        progress_bar.progress(
            min(1.0, max(0.0, frac)),
            text=f"{done}/{total_companies} companies done"
                 f" | Elapsed {format_time(elapsed)}{eta_txt}",
        )
        icon = {"complete": "[done]", "error": "[fail]", "running": "[....]"}
        progress_placeholder.text("\n".join(
            f"{icon.get(cs['status'], '[wait]')} {cs['name']}" for cs in merged.values()
        ))

    while True:
        if batch_future.done():
            try:
                result = batch_future.result()
                all_results = result["results"]
                total_time = result["total_time"]
                for company_result in all_results:
                    idx = company_result["idx"]
                    company_statuses[idx] = {
                        "status": "complete" if company_result["success"] else "error",
                        "name": company_result["company_name"],
                    }
                _render_progress(total_companies, progress_state.get("rows", {}))
                progress_bar.progress(
                    1.0,
                    text=f"{total_companies}/{total_companies} companies done"
                         f" | {format_time(total_time)}",
                )
                status.update(
                    label=f"All searches complete! Processed {total_companies} companies in {format_time(total_time)}",
                    state="complete",
                    expanded=False,
                )
                return all_results
            except Exception as e:  # noqa: BLE001
                st.error(f"Error in batch processing: {e}")
                status.update(label=f"Error: {e}", state="error")
                return []

        elapsed = time.time() - start_time
        if elapsed > BATCH_POLL_TIMEOUT_S:
            # BOUNDED wall clock: don't poll forever if the loop wedges.
            batch_future.cancel()
            status.update(label="Batch processing timed out.", state="error")
            st.error(f"Batch processing exceeded {format_time(BATCH_POLL_TIMEOUT_S)} and was cancelled.")
            return []

        done = int(progress_state.get("done", 0))
        _render_progress(done, progress_state.get("rows", {}))
        status.update(
            label=f"Processing {total_companies} companies in parallel... "
                  f"({done}/{total_companies} done, Elapsed: {format_time(elapsed)})"
        )
        time.sleep(1)


# ===========================================================================
# History helpers (bounded via common.push_history / get_history)
# ===========================================================================
def save_to_search_history(company_names, search_results, verified_results=None,
                           scraped_urls=None, failed_urls=None) -> None:
    """Append a (bounded) normalized history entry to session state."""
    # Microsecond-resolution ISO timestamp so two entries in the same second
    # don't collide, and a DETERMINISTIC id (sha1 over the timestamp + the
    # sorted-key JSON of the company names) instead of Python's salted hash().
    timestamp = datetime.now().isoformat()
    normalized_results = []
    for result in search_results:
        company_name = safe_get_attribute(result, "company_name", "Unknown")
        result_obj = safe_get_attribute(result, "result", result)
        search_results_text = safe_get_attribute(
            result_obj, "search_results",
            safe_get_attribute(result_obj, "search_text_formatted", "No search results available"),
        )
        # Normalize search_results into a list of text blocks for display.
        if isinstance(search_results_text, list):
            blocks = []
            for item in search_results_text:
                if isinstance(item, dict):
                    blocks.append(item.get("search_results", item.get("search_text_formatted", str(item))))
                else:
                    blocks.append(str(item))
        else:
            blocks = [str(search_results_text)]
        normalized = {"company_name": company_name, "result": {"search_results": blocks}}
        sd = safe_get_attribute(result_obj, "structured_data", None)
        if sd is not None:
            normalized["result"]["structured_data"] = sd
        normalized_results.append(normalized)

    entry = {
        "timestamp": timestamp,
        "companies": list(company_names) if company_names else [],
        "search_results": normalized_results,
        "verified_results": verified_results or [],
        "scraped_urls": scraped_urls or [],
        "failed_urls": failed_urls or [],
        "id": hashlib.sha1(
            (timestamp + json.dumps(list(company_names) if company_names else [], sort_keys=True)).encode()
        ).hexdigest(),
    }
    push_history(SS_HISTORY, entry)


def export_search_history() -> None:
    """Offer the bounded history as a JSON download."""
    history = get_history(SS_HISTORY)
    if not history:
        st.warning("No search history to export.")
        return
    history_json = json.dumps(history, default=str)
    st.download_button(
        label="Download Search History",
        data=history_json,
        file_name="company_research_history.json",
        mime="application/json",
        key="cr_export_history_btn",
    )


def import_search_history(uploaded_file) -> bool:
    """Import history JSON, de-duped by id, into the bounded history list."""
    try:
        imported = json.loads(uploaded_file.getvalue().decode())
        if not isinstance(imported, list):
            st.error("Invalid history file format.")
            return False
        existing_ids = {e.get("id") for e in get_history(SS_HISTORY)}
        added = 0
        for entry in imported:
            if isinstance(entry, dict) and entry.get("id") not in existing_ids:
                push_history(SS_HISTORY, entry)
                existing_ids.add(entry.get("id"))
                added += 1
        st.success(f"Successfully imported {added} new search entries.")
        return True
    except Exception as e:  # noqa: BLE001
        st.error(f"Error importing history: {e}")
        return False


# ===========================================================================
# Session-state initialization (cr_-prefixed)
# ===========================================================================
def _initialize_session_state() -> None:
    defaults = {
        SS_PHASE: "input",
        SS_FIRST_PASS: None,
        SS_SELECTED_ENTITY: None,
        SS_NEGATIVE_ENTITIES: [],
        SS_ENTITY_FULL_DATA: [],
        SS_SECOND_PASS: None,
        SS_THIRD_PASS: None,
        SS_ADDITIONAL_CONTEXT: "",
        SS_VIEWING_HISTORY_ID: None,
        SS_HISTORY: [],
        SS_STRUCT_VISIBILITY: {},
        SS_OBSERVABILITY: [],
        SS_UPLOAD_DF: None,
        SS_BATCH_RESULTS: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ===========================================================================
# UI: first-pass entity selection (data editor)
# ===========================================================================
def first_pass_ui_with_data_editor(async_runner) -> None:
    st.subheader("First Pass: Entity Selection")
    st.write("Select the correct entity and mark any irrelevant entities:")

    first_pass_result = st.session_state.get(SS_FIRST_PASS)
    entity_options = (first_pass_result or {}).get("entities", []) if first_pass_result else []

    if not entity_options:
        st.warning("No entities were found. Please try again with more specific information.")
        if st.button("Start Over", key="cr_fp_startover_empty"):
            _reset_three_pass()
            st.rerun()
        return

    st.session_state[SS_ENTITY_FULL_DATA] = entity_options

    with st.expander("Debug Entity Data Structure"):
        for i, entity in enumerate(entity_options):
            name = safe_get_attribute(entity, "name", "Unknown")
            sources = safe_get_attribute(entity, "sources", []) or []
            st.write(f"**Entity {i + 1}: {name}** - {len(sources)} source(s)")

    # ADDITIVE: did the disambiguation step attach confidence to any candidate?
    # Only show the confidence/reason columns when present so the UI degrades to
    # the prior layout when Gemini disambiguation was unavailable.
    has_confidence = any(
        safe_get_attribute(entity, "confidence", None) is not None
        for entity in entity_options
    )
    if has_confidence:
        st.caption(
            "Match Confidence is an AI hint (0-100%) for whether each candidate "
            "is the company you mean vs. a similarly-named different one. It does "
            "NOT auto-select or remove anything — you still choose."
        )

    data = []
    for i, entity in enumerate(entity_options):
        snippet_preview = safe_get_attribute(entity, "snippet", "No description available.") or ""
        if len(snippet_preview) > 5000:
            snippet_preview = snippet_preview[:5000] + "..."
        source_domains = []
        for source in (safe_get_attribute(entity, "sources", []) or [])[:3]:
            url = safe_get_attribute(source, "url", None)
            if url:
                source_domains.append(urlparse(url).netloc)
        row = {
            "Select Target": False,
            "Mark Irrelevant": False,
            "Company Name": safe_get_attribute(entity, "name", f"Entity {i + 1}"),
            "Description": snippet_preview,
            "Sources": ", ".join(source_domains) if source_domains else "No sources",
            "Index": i,
        }
        if has_confidence:
            conf = safe_get_attribute(entity, "confidence", None)
            row["Match Confidence"] = float(conf) if conf is not None else None
            row["Why"] = safe_get_attribute(entity, "confidence_reason", "") or ""
        data.append(row)

    df = pd.DataFrame(data)
    column_config = {
        "Select Target": st.column_config.CheckboxColumn("Select as Target", default=False, width="small"),
        "Mark Irrelevant": st.column_config.CheckboxColumn("Mark as Irrelevant", default=False, width="small"),
        "Company Name": st.column_config.TextColumn("Company Name", width="small"),
        "Description": st.column_config.TextColumn("Description", width="large"),
        "Sources": st.column_config.TextColumn("Source Domains", width="medium"),
        "Index": st.column_config.NumberColumn("Index", width="small"),
    }
    disabled_cols = ["Index", "Company Name", "Description", "Sources"]
    if has_confidence:
        column_config["Match Confidence"] = st.column_config.ProgressColumn(
            "Match Confidence", min_value=0.0, max_value=1.0, format="%.0f%%",
            help="AI confidence this is the SAME company you mean (hint only).",
        )
        column_config["Why"] = st.column_config.TextColumn(
            "Why", width="large", help="Why the AI assigned that confidence.",
        )
        disabled_cols += ["Match Confidence", "Why"]
    edited_df = st.data_editor(
        df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        key="cr_entity_selection_editor",
        disabled=disabled_cols,
    )

    st.divider()
    st.subheader("Entity Preview")
    preview_options = [
        f"{i + 1}: {safe_get_attribute(entity, 'name', f'Entity {i + 1}')}"
        for i, entity in enumerate(entity_options)
    ]
    selected_preview = st.selectbox("Select entity to preview:", preview_options, key="cr_entity_preview_select")
    selected_preview_idx = int(selected_preview.split(":")[0]) - 1
    preview_entity = entity_options[selected_preview_idx]
    st.write(f"### {safe_get_attribute(preview_entity, 'name', 'Unknown Entity')}")
    preview_conf = safe_get_attribute(preview_entity, "confidence", None)
    if preview_conf is not None:
        preview_reason = safe_get_attribute(preview_entity, "confidence_reason", "") or ""
        st.progress(
            float(preview_conf),
            text=f"Match confidence: {float(preview_conf) * 100:.0f}%"
            + (f" — {preview_reason}" if preview_reason else ""),
        )
    st.markdown(safe_get_attribute(preview_entity, "snippet", "No description available."))

    sources = safe_get_attribute(preview_entity, "sources", []) or []
    if sources:
        st.write("#### Sources:")
        for source in sources[:5]:
            url = safe_get_attribute(source, "url", None)
            if url:
                domain = urlparse(url).netloc
                st.write(f"- [{domain}]({url})")
                snippet = safe_get_attribute(source, "snippet", "")
                if snippet:
                    with st.expander(f"Preview from {domain}"):
                        st.write(snippet[:300] + "..." if len(snippet) > 300 else snippet)
    else:
        st.info("No sources available for this entity.")

    st.divider()
    st.subheader("Selection Summary")
    selected_entity = None
    negative_entities = []
    for _, row in edited_df.iterrows():
        idx = int(row["Index"])
        entity = entity_options[idx]
        if row["Select Target"]:
            selected_entity = entity
        if row["Mark Irrelevant"]:
            negative_entities.append(entity)

    if selected_entity is not None and selected_entity in negative_entities:
        st.warning("You've marked the same entity as both target and irrelevant. Treating it as the target.")
        negative_entities = [e for e in negative_entities if e is not selected_entity]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Selected Target Entity:**")
        if selected_entity:
            st.success(f"**{safe_get_attribute(selected_entity, 'name', 'Unknown Entity')}**")
            st.session_state[SS_SELECTED_ENTITY] = selected_entity
        else:
            st.info("No entity selected as target yet")
    with col2:
        st.write("**Marked as Irrelevant:**")
        if negative_entities:
            for neg_entity in negative_entities:
                st.warning(f"**{safe_get_attribute(neg_entity, 'name', 'Unknown Entity')}**")
            st.session_state[SS_NEGATIVE_ENTITIES] = negative_entities
        else:
            st.info("No entities marked as irrelevant")

    st.divider()
    additional_context = st.text_area(
        "Additional context about the selected entity (optional):",
        value=safe_get_attribute(first_pass_result, "additional_context", "") or "",
        placeholder="E.g., Founded in 2020, AI-based company in healthcare sector, etc.",
        key="cr_fp_additional_context",
    )

    col1, col2 = st.columns(2)
    with col1:
        continue_disabled = selected_entity is None
        if st.button("Continue to Second Pass", disabled=continue_disabled, key="cr_fp_continue"):
            st.session_state[SS_ADDITIONAL_CONTEXT] = additional_context
            st.session_state[SS_PHASE] = "detailed_search"
            st.rerun()
        if continue_disabled:
            st.warning("Please select a target entity to continue")
    with col2:
        if st.button("Start Over", key="cr_fp_startover"):
            _reset_three_pass()
            st.rerun()


def _reset_three_pass() -> None:
    st.session_state[SS_PHASE] = "input"
    st.session_state[SS_FIRST_PASS] = None
    st.session_state[SS_SELECTED_ENTITY] = None
    st.session_state[SS_NEGATIVE_ENTITIES] = []
    st.session_state[SS_SECOND_PASS] = None
    st.session_state[SS_THIRD_PASS] = None


# ===========================================================================
# UI: second / third pass
# ===========================================================================
def display_second_pass_ui(async_runner) -> None:
    st.subheader("Second Pass: Detailed Company Information")
    selected_entity = st.session_state.get(SS_SELECTED_ENTITY)
    if not selected_entity:
        st.warning("No target entity selected. Returning to entity selection.")
        st.session_state[SS_PHASE] = "entity_selection"
        st.rerun()
        return

    entity_name = safe_get_attribute(selected_entity, "name", "Unknown Entity")
    st.write(f"Researching: **{entity_name}**")

    if not st.session_state.get(SS_SECOND_PASS):
        with st.status("Running detailed search...") as status:
            second_pass_result = async_runner.run_task(
                second_pass_structured_search(
                    selected_entity,
                    st.session_state.get(SS_NEGATIVE_ENTITIES, []),
                    st.session_state.get(SS_ADDITIONAL_CONTEXT, ""),
                    st.session_state.get(SS_OBSERVABILITY, []),
                )
            )
            if not second_pass_result["success"]:
                status.update(label=f"Error: {second_pass_result.get('error', 'Unknown error')}", state="error")
                st.error(second_pass_result.get("search_text_formatted", "Detailed search failed."))
                if st.button("Start Over", key="cr_sp_startover_err"):
                    _reset_three_pass()
                    st.rerun()
                return
            st.session_state[SS_SECOND_PASS] = second_pass_result
            status.update(label="Information gathered successfully!", state="complete")

    second = st.session_state.get(SS_SECOND_PASS, {})
    search_results = second.get("search_results", second.get("search_text_formatted", "No search results available."))
    st.markdown(search_results)

    structured_data = second.get("structured_data")
    if structured_data:
        with st.expander("Show structured data (JSON)"):
            st.json(structured_data)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Enhance with Third Pass", key="cr_sp_enhance"):
            st.session_state[SS_PHASE] = "enhancement"
            st.rerun()
    with col2:
        if st.button("Start Over", key="cr_sp_startover"):
            _reset_three_pass()
            st.rerun()


def display_third_pass_ui(async_runner) -> None:
    st.subheader("Third Pass: Enhanced Company Information")
    selected_entity = st.session_state.get(SS_SELECTED_ENTITY)
    if not selected_entity:
        st.warning("No target entity selected. Returning to entity selection.")
        st.session_state[SS_PHASE] = "entity_selection"
        st.rerun()
        return
    entity_name = safe_get_attribute(selected_entity, "name", "Unknown Entity")
    st.write(f"Enhancing information for: **{entity_name}**")

    second = st.session_state.get(SS_SECOND_PASS, {}) or {}

    if not st.session_state.get(SS_THIRD_PASS):
        with st.status("Finding missing information...") as status:
            third_pass_result = async_runner.run_task(
                third_pass_fill_missing_info(
                    second.get("structured_data", {}),
                    selected_entity,
                    st.session_state.get(SS_NEGATIVE_ENTITIES, []),
                    second.get("sources", []),
                    st.session_state.get(SS_OBSERVABILITY, []),
                )
            )
            if not third_pass_result["success"]:
                status.update(label=f"Error: {third_pass_result.get('error', 'Unknown error')}", state="error")
            elif third_pass_result["enhanced"]:
                status.update(label=f"Enhanced {len(third_pass_result['enhanced_fields'])} fields!", state="complete")
            else:
                status.update(label="No significant enhancements needed or found.", state="complete")
            st.session_state[SS_THIRD_PASS] = third_pass_result

    third = st.session_state.get(SS_THIRD_PASS, {}) or {}
    # structured_data is None when the Gemini schema-extraction step was
    # unavailable (no/invalid key) in Pass 2 and never produced; only treat a
    # real dict as structured data so we never render an empty {} as a result.
    enhanced_data = third.get("structured_data") or second.get("structured_data")
    if not isinstance(enhanced_data, dict) or not enhanced_data:
        enhanced_data = None

    if third.get("enhanced"):
        st.success(f"Successfully enhanced {len(third.get('enhanced_fields', []))} fields")
        st.write("**Enhanced fields:**")
        for field in third.get("enhanced_fields", []):
            st.write(f"- `{field}`")
    else:
        st.info("No significant enhancements were needed or found.")

    if enhanced_data is not None:
        # HONEST STATUS: real structured data -> render the schema view.
        enhanced_text = format_linkup_results_as_search_text(enhanced_data, entity_name)
        st.markdown(enhanced_text)
        with st.expander("Show enhanced structured data (JSON)"):
            st.json(enhanced_data)
    else:
        # HONEST STATUS: no structured data. Show the source-backed answer text
        # from Pass 2 (never an empty {}), and explain why enhancement is off.
        st.warning(
            "Structured enhancement unavailable (needs a valid Gemini key) — "
            "showing source-backed answer only."
        )
        enhanced_text = second.get(
            "search_text_formatted",
            second.get("search_results", "No source-backed answer available."),
        )
        st.markdown(enhanced_text)

    # Save to (bounded) search history EXACTLY ONCE per completed third pass.
    # display_third_pass_ui() re-runs on every Streamlit interaction, so guard
    # with a flag persisted back into SS_THIRD_PASS to avoid duplicate entries.
    if not third.get("_saved_to_history"):
        save_to_search_history(
            [entity_name],
            [{"company_name": entity_name, "result": {"search_results": [enhanced_text]}}],
            None,
        )
        third["_saved_to_history"] = True
        st.session_state[SS_THIRD_PASS] = third

    if st.button("Start New Research", key="cr_tp_startover"):
        _reset_three_pass()
        st.session_state[SS_FIRST_PASS] = None
        st.rerun()


# ===========================================================================
# UI: standard search results / file upload results / history
# ===========================================================================
def display_standard_search_results(response: Dict[str, Any]) -> None:
    if not response.get("success"):
        st.error(f"Error: {response.get('error', 'An unknown error occurred')}")
        return

    companies_searched = response["parsed_entities"]
    save_to_search_history(
        companies_searched,
        [{"company_name": r["company_name"], "result": {"search_results": [r["search_results"]],
          "structured_data": r.get("structured_data")}} for r in response["search_results"]],
        response.get("verified_results"),
    )

    timing = response.get("timing", {})
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            f"**Search completed in {format_time(timing.get('total_time', 0))}**\n\n"
            f"- Step 1 (Parsing): {format_time(timing.get('step1_time', 0))}\n"
            f"- Step 1.5 (Scraping): {format_time(timing.get('step15_time', 0))}\n"
            f"- Step 2 (Searching): {format_time(timing.get('step2_time', 0))}\n"
            f"- Step 3 (Verification): {format_time(timing.get('step3_time', 0))}"
        )
    with col2:
        render_token_usage(inside_expander=False)

    st.subheader("Companies Identified")
    for company in response["parsed_entities"]:
        st.write(f"- {company}")

    st.subheader("Search Results")
    for idx, result in enumerate(response["search_results"]):
        company_name = result["company_name"]
        st.markdown(f"### {company_name}")
        st.markdown(result["search_results"])
        if structured_data := result.get("structured_data"):
            visibility_key = f"cr_manual_data_{company_name}_{idx}"
            show_data = st.checkbox(f"Show structured data for {company_name}", key=visibility_key,
                                    value=_get_visibility(visibility_key))
            _set_visibility(visibility_key, show_data)
            if show_data:
                st.json(structured_data)

    if response.get("verified_results"):
        st.subheader("Verified Results")
        for verified in response["verified_results"]:
            st.markdown(f"### {verified['company_name']}")
            st.markdown(verified["verification"])


def display_file_upload_results(result, async_runner, max_concurrent, enable_batch_scraping) -> None:
    if not result["success"]:
        st.error(f"Error processing uploaded file: {result['error']}")
        return
    if result["data"] is None:
        st.error("Could not process the uploaded file. Please try another file.")
        return

    editor_df = result["data"]
    st.success(f"Successfully processed file and identified {len(editor_df)} potential companies.")
    st.subheader("Review and Edit Company Information")
    st.write("Edit the company names / context, set each row's Status, then enrich only the Pending rows.")

    columns_to_display = ["company_name", "additional_context", "status"]
    column_config = {
        "company_name": st.column_config.TextColumn("Company Name", help="Name of the company to research"),
        "additional_context": st.column_config.TextColumn("Additional Context", help="Industry, founding year, location, URLs, etc."),
        "status": st.column_config.SelectboxColumn("Status", options=["Pending", "Complete", "Skip"], required=True,
                                                   help="Only 'Pending' rows are enriched"),
    }
    has_raw_data = "raw_data" in editor_df.columns and not editor_df["raw_data"].isna().all()
    if has_raw_data and st.checkbox("Show original row data from spreadsheet", value=False, key="cr_show_raw"):
        columns_to_display.append("raw_data")
        column_config["raw_data"] = st.column_config.TextColumn("Original Data", width="large")

    edited_df = st.data_editor(
        editor_df[columns_to_display],
        column_config=column_config,
        use_container_width=True,
        num_rows="dynamic",
        key="cr_upload_editor",
    )

    if st.button("Enrich Pending Companies", key="cr_batch_search") and not edited_df.empty:
        search_df = edited_df[edited_df["status"] == "Pending"]
        if search_df.empty:
            st.warning("No companies with 'Pending' status to search for.")
            return
        all_results = process_companies_with_progress(
            search_df, async_runner, max_concurrent=max_concurrent, enable_web_scraping=enable_batch_scraping
        )
        if not all_results:
            return

        companies_searched = [r["company_name"] for r in all_results if r["success"]]
        save_to_search_history(companies_searched, all_results, None)

        st.subheader("Resource Usage Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Companies Processed", len(search_df))
            successful = [r for r in all_results if r["success"]]
            avg_time = 0.0
            if successful:
                avg_time = sum(r["result"].get("timing", {}).get("total_time", 0) for r in successful) / len(successful)
            st.metric("Average Processing Time", format_time(avg_time))
        with col2:
            render_token_usage(inside_expander=False)

        st.subheader("Search Results")
        successful = [r for r in all_results if r["success"]]
        failed = [r for r in all_results if not r["success"]]
        st.write(f"Successfully searched: {len(successful)}/{len(all_results)} companies")

        if failed:
            with st.expander(f"Failed Searches ({len(failed)})", expanded=False):
                for r in failed:
                    st.error(f"{r['company_name']}: {r['result'].get('error', 'Unknown error')}")

        for company_result in successful:
            company_name = company_result["company_name"]
            response = company_result["result"]
            with st.expander(f"Results for {company_name}"):
                for result_idx, res in enumerate(response.get("search_results", [])):
                    st.markdown(res["search_results"])
                    if structured_data := res.get("structured_data"):
                        visibility_key = f"cr_data_{company_name}_{result_idx}"
                        show_data = st.checkbox("Show structured data", key=f"cr_cb_{visibility_key}",
                                                value=_get_visibility(visibility_key))
                        _set_visibility(visibility_key, show_data)
                        if show_data:
                            st.json(structured_data)
                            st.divider()
                if response.get("verified_results"):
                    st.subheader("Verified Results")
                    for verified in response["verified_results"]:
                        st.markdown(verified["verification"])


def display_search_history() -> None:
    st.header("Previous Search Results")
    history = get_history(SS_HISTORY)
    if not history:
        st.info("No search history available. Run a search to see results here.")
        return
    for idx, entry in enumerate(history):
        timestamp = safe_get_attribute(entry, "timestamp", "Unknown date")
        companies = ", ".join(safe_get_attribute(entry, "companies", ["Unknown"]) or ["Unknown"])
        with st.expander(f"{companies} - {timestamp}"):
            st.markdown(f"**Companies:** {companies}")
            st.markdown(f"**Time:** {timestamp}")
            search_results = safe_get_attribute(entry, "search_results", []) or []
            if search_results:
                st.markdown("### Results Preview")
                st.divider()
                for result in search_results[:2]:
                    company_name = safe_get_attribute(result, "company_name", "Unknown")
                    st.markdown(f"#### {company_name}")
                    result_obj = safe_get_attribute(result, "result", {})
                    result_text = None
                    if isinstance(result_obj, dict):
                        items = result_obj.get("search_results", [])
                        if items:
                            first = items[0]
                            result_text = safe_get_attribute(first, "search_results",
                                          safe_get_attribute(first, "search_text_formatted",
                                          first if isinstance(first, str) else None))
                    if not result_text:
                        result_text = "No results available"
                    result_text = str(result_text)
                    st.markdown(result_text[:500] + "..." if len(result_text) > 500 else result_text)
                    st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View Full Details", key=f"cr_view_{safe_get_attribute(entry, 'id', idx)}"):
                    st.session_state[SS_VIEWING_HISTORY_ID] = safe_get_attribute(entry, "id", idx)
                    st.rerun()
            with col2:
                if st.button("Remove from History", key=f"cr_remove_{safe_get_attribute(entry, 'id', idx)}"):
                    new_hist = [e for j, e in enumerate(history) if j != idx]
                    st.session_state[SS_HISTORY] = new_hist
                    st.rerun()


def display_historical_search_view() -> None:
    history = get_history(SS_HISTORY)
    viewing_id = st.session_state.get(SS_VIEWING_HISTORY_ID)
    history_entry = next((e for e in history if e.get("id") == viewing_id), None)
    if not history_entry:
        st.warning("That history entry could not be found.")
        if st.button("Return to Current Search", key="cr_hist_return_missing"):
            st.session_state[SS_VIEWING_HISTORY_ID] = None
            st.rerun()
        return

    st.success(f"Viewing historical search from {history_entry['timestamp']}")
    if st.button("Return to Current Search", key="cr_hist_return"):
        st.session_state[SS_VIEWING_HISTORY_ID] = None
        st.rerun()

    st.subheader(f"Search Results for: {', '.join(history_entry.get('companies', []))}")
    for result in history_entry.get("search_results", []):
        company_name = result.get("company_name", "Unknown")
        with st.expander(f"Results for {company_name}", expanded=True):
            for result_idx, search_result in enumerate(result.get("result", {}).get("search_results", [])):
                st.markdown(str(search_result))
                if structured_data := result.get("result", {}).get("structured_data"):
                    visibility_key = f"cr_history_{history_entry['id']}_{company_name}_{result_idx}"
                    show_data = st.checkbox("Show structured data", key=visibility_key,
                                            value=_get_visibility(visibility_key))
                    _set_visibility(visibility_key, show_data)
                    if show_data:
                        st.json(structured_data)
                        st.divider()


# ===========================================================================
# Integrated input UI (manual / spreadsheet / history tabs)
# ===========================================================================
def integrated_search_ui(async_runner, app_mode: str) -> None:
    tab1, tab2, tab3 = st.tabs(["Enter Manually", "Upload Spreadsheet", "Search History"])

    with tab1:
        with st.form("cr_company_search_form"):
            user_input = st.text_area(
                "Enter company names or describe what you're looking for:",
                placeholder="Example: Tell me about Ideaflow, a company founded in 2014 in the Bay Area",
                key="cr_manual_user_input",
            )
            additional_context = st.text_area(
                "Additional context (optional):",
                placeholder="Example: Founded in 2014, located in Palo Alto, funded by 8VC (you can paste URLs)",
                key="cr_manual_context",
            )
            col1, col2 = st.columns(2)
            with col1:
                max_concurrent = st.slider("Maximum concurrent requests:", 1, HARD_MAX_CONCURRENT,
                                           DEFAULT_MAX_CONCURRENT, key="cr_manual_concurrency",
                                           help="Higher values may improve speed but could hit API rate limits")
            with col2:
                enable_scraping = st.checkbox("Enhance context with web scraping", value=True,
                                              key="cr_manual_scraping",
                                              help="Scrape URLs in the additional context to improve search quality")
            submit_button = st.form_submit_button("Search")

        if submit_button and user_input:
            if get_linkup_client() is None:
                st.error("LinkUp API is not configured. Please add LINKUP_API_KEY to your secrets.")
                return
            status = st.status("Initializing search...", expanded=True)
            if app_mode == "Three-Pass Research":
                with status:
                    status.update(label="First Pass: Finding possible companies...", state="running")
                    first_pass_result = async_runner.run_task(
                        first_pass_entity_selection(user_input, st.session_state.get(SS_OBSERVABILITY, []))
                    )
                    # Persist the bounded observability log after the run.
                    _trim_observability()
                    if not first_pass_result["success"]:
                        status.update(label=f"Error: {first_pass_result.get('error', 'Unknown error')}", state="error")
                        return
                    st.session_state[SS_ADDITIONAL_CONTEXT] = additional_context
                    st.session_state[SS_FIRST_PASS] = first_pass_result
                    st.session_state[SS_PHASE] = "entity_selection"
                    status.update(label="Found potential entities! Please select the correct one.", state="complete")
                    st.rerun()
            else:
                response = async_runner.run_task(
                    gemini_company_search_with_verification(
                        user_input=user_input,
                        additional_context=additional_context,
                        status_container=status,
                        max_concurrent=max_concurrent,
                        enable_web_scraping=enable_scraping,
                    )
                )
                display_standard_search_results(response)

    with tab2:
        uploaded_file = st.file_uploader(
            "Upload a spreadsheet with company information", type=["xlsx", "xls", "csv"], key="cr_uploader"
        )
        col1, col2 = st.columns(2)
        with col1:
            max_concurrent = st.slider("Maximum concurrent requests:", 1, HARD_MAX_CONCURRENT,
                                       DEFAULT_MAX_CONCURRENT, key="cr_batch_concurrency",
                                       help="Higher values may improve speed but could hit API rate limits")
        with col2:
            enable_batch_scraping = st.checkbox("Enable web scraping for context enhancement", value=True,
                                                key="cr_batch_scraping",
                                                help="Scrape URLs found in row context to improve search quality")

        if uploaded_file is not None:
            if get_linkup_client() is None:
                st.error("LinkUp API is not configured. Please add LINKUP_API_KEY to your secrets.")
                return
            with st.spinner("Processing uploaded file (parsing + entity extraction)..."):
                result = async_runner.run_task(process_uploaded_file(uploaded_file))
            display_file_upload_results(result, async_runner, max_concurrent, enable_batch_scraping)

    with tab3:
        display_search_history()


def _trim_observability() -> None:
    log = st.session_state.get(SS_OBSERVABILITY)
    if isinstance(log, list) and len(log) > MAX_OBSERVABILITY:
        st.session_state[SS_OBSERVABILITY] = log[-MAX_OBSERVABILITY:]


# ===========================================================================
# PUBLIC ENTRY POINT
# ===========================================================================
def render_company_research_tab() -> None:
    """Render the Company Research tab (called inside a tab by the main app)."""
    _initialize_session_state()

    st.header("Company Research")
    st.caption(
        "Source-backed company research using LinkUp sourcedAnswer + Gemini. "
        "Enter a company or upload a spreadsheet, resolve the right entity, then enrich."
    )

    # HONEST STATUS: check required keys up-front and degrade gracefully.
    gemini_ok, gemini_missing = feature_available(["GEMINI_API_KEY"])
    linkup_ok, linkup_missing = feature_available(["LINKUP_API_KEY"])

    if not gemini_ok or not linkup_ok:
        missing = []
        if not gemini_ok:
            missing.append("GEMINI_API_KEY (or legacy GOOGLE_AI_STUDIO)")
        if not linkup_ok:
            missing.append("LINKUP_API_KEY")
        st.warning(
            "Company Research needs API keys that are not configured: "
            + ", ".join(missing)
            + ". Add them to st.secrets (top level) to enable this tab. "
            "No research can run until then."
        )
        # Still allow viewing/exporting any existing (bounded) history honestly.
        with st.expander("Search History (read-only)"):
            display_search_history()
        return

    # Keys present but possibly rejected (leaked/invalid) — surface that
    # honestly so a 403 does not masquerade as "no results".
    common.render_gemini_health()

    # Controls row: mode + maintenance.
    with st.expander("Settings & Diagnostics", expanded=False):
        app_mode = st.radio(
            "Research mode",
            options=["Three-Pass Research", "Standard Search"],
            index=0,
            key="cr_app_mode",
            help="Three-Pass = entity resolution + structured search + targeted backfill. "
                 "Standard = single-shot parallel search.",
        )
        st.session_state[SS_MODE] = app_mode

        col1, col2 = st.columns(2)
        with col1:
            if get_linkup_client() is not None:
                st.success("LinkUp API connected")
            else:
                st.error("LinkUp API not configured")
        with col2:
            if common.get_gemini_client() is not None:
                st.success("Gemini API connected")
            else:
                st.error("Gemini API not configured")

        st.subheader("API Usage")
        render_token_usage(inside_expander=True)

        st.subheader("Search History Tools")
        export_search_history()
        up = st.file_uploader("Import history (JSON)", type=["json"], key="cr_history_import")
        if up is not None and st.button("Import History", key="cr_history_import_btn"):
            import_search_history(up)

        # Observability log is shown without nesting an expander (Streamlit
        # forbids expander-in-expander).
        if st.session_state.get(SS_OBSERVABILITY):
            st.subheader("Observability Log (bounded, last 25)")
            st.json(st.session_state.get(SS_OBSERVABILITY, [])[-25:])

    # app_mode is always defined: the radio above runs every rerun and also
    # persists to SS_MODE. Fall back to the stored value defensively.
    app_mode = st.session_state.get(SS_MODE, "Three-Pass Research")

    async_runner = get_async_runner()

    # Viewing a historical entry overrides everything else.
    if st.session_state.get(SS_VIEWING_HISTORY_ID):
        display_historical_search_view()
        return

    # Three-pass phase routing.
    if app_mode == "Three-Pass Research":
        phase = st.session_state.get(SS_PHASE, "input")
        if phase == "entity_selection":
            first_pass_ui_with_data_editor(async_runner)
            return
        if phase == "detailed_search":
            display_second_pass_ui(async_runner)
            return
        if phase == "enhancement":
            display_third_pass_ui(async_runner)
            return

    integrated_search_ui(async_runner, app_mode)
