"""
features/news_youtube.py
========================

ParselyFi "News & YouTube" feature tab, adapted from the reference Streamlit
app ``test2_crawl4ai_google_search_v032525_report_article_video_editable_search.py``
(article/news search + structured report) and its editable sibling
``..._editable.py`` (editable-results / widget UI patterns).

Public entry point (the ONLY thing the main app calls):

    def render_news_youtube_tab() -> None

It is invoked inside an already-open ``st.tab`` context, so this module does
NOT call ``st.set_page_config`` and has no ``__main__`` run block.

What was ported and how it differs from the reference
-----------------------------------------------------
The reference performed "search" by calling Gemini with the built-in
``google_search`` tool and a Helicone gateway. The shared ``features.common``
Gemini helpers (``gemini_generate_json`` / ``gemini_generate_text``) do not
expose the google_search tool, and per project rules we must use ONLY those
helpers + ``common.scrape_urls`` for network access. So the pipeline is:

  News Alerts (LinkUp-primary; reliable, real current sources):
    query -> LinkUp sourcedAnswer returns a source-backed answer + real source
             URLs/snippets (no dependence on scraping bot-blocked news sites)
          -> Gemini structures the answer into a cited briefing (summary, key
             points, tags, sentiment)
          -> editable source list with a Keep/Skip column (st.data_editor)
          -> source citations (the actual LinkUp source URLs).
    Fallback (LinkUp absent): Gemini proposes article URLs -> common.scrape_urls
    (SSRF-guarded, bounded) -> per-article Gemini report. Less reliable because
    reputable news sites commonly block bots / paywall / 404 stale URLs.

  YouTube Daily Reports:
    YouTube URL (page text scraped) OR pasted transcript
          -> Gemini structured video report (metadata, summary, key insights
             with timestamps, key topics, "make it useful")
          -> editable summary fields + JSON download.

Honest status / safety
----------------------
- If no Gemini key (or SDK) is available we ``st.warning`` and return early.
  Nothing is fabricated; a missing client yields an empty Gemini result and we
  show that honestly rather than inventing a report.
- All accumulated state (search history, last results) is bounded via
  ``common.push_history`` (FIFO cap = ``common.MAX_HISTORY``) and explicit caps
  on candidate counts.
- All scraping goes through ``common.scrape_urls`` which is SSRF-guarded,
  timeout-bounded and read-size-capped.
- Session-state keys are all prefixed ``ny_`` so they never collide with the
  main app ("s3_file_manager" / "selected_files") or sibling tabs.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import streamlit as st

from features import common

logger = logging.getLogger("parselyfi.features.news_youtube")

# ---------------------------------------------------------------------------
# Constants (bounded-memory caps + session keys, all ny_ prefixed)
# ---------------------------------------------------------------------------
_AGENT_NEWS_CANDIDATES = "news_candidates"
_AGENT_NEWS_REPORT = "news_report"
_AGENT_YT_REPORT = "youtube_report"

# Hard caps so nothing the LLM/user supplies can blow up memory or fan-out.
_MAX_CANDIDATES = 8          # max candidate article URLs we will scrape
_MAX_SCRAPE_CONCURRENCY = 3  # passed to common.scrape_urls
_MAX_SCRAPED_CHARS = 12_000  # chars of scraped text fed to the summarizer per article
_MAX_TRANSCRIPT_CHARS = 40_000  # cap pasted transcript / scraped video text

# Session-state keys (ny_ prefix everywhere).
_K_NEWS_QUERY = "ny_news_query"
_K_NEWS_RESULTS = "ny_news_results"          # list[dict] current report rows
_K_NEWS_EDITED = "ny_news_edited_df"         # last edited dataframe snapshot
_K_NEWS_HISTORY = "ny_news_history"          # bounded via push_history
_K_NEWS_RUNNING = "ny_news_running"

_K_YT_INPUT = "ny_yt_input"                  # URL or pasted transcript
_K_YT_MODE = "ny_yt_mode"                    # "url" | "transcript"
_K_YT_RESULT = "ny_yt_result"                # dict current video report
_K_YT_HISTORY = "ny_yt_history"              # bounded via push_history
_K_YT_EDIT_MODE = "ny_yt_edit_mode"

_REQUIRED_KEYS = ["GEMINI_API_KEY"]


# ===========================================================================
# Small utilities
# ===========================================================================

def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + " [truncated]"


def _safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


# ===========================================================================
# News Alerts: candidate discovery -> scrape -> structured report
# ===========================================================================

async def _propose_candidate_urls(query: str) -> Dict[str, Any]:
    """Ask Gemini for likely article URLs + refined search terms for ``query``.

    Returns the parsed dict from the model. Honest: returns ``{}`` if the model
    is unavailable or produces nothing (common.gemini_generate_json handles that).
    We only ever USE real scraped content downstream, so a bad/empty URL just
    drops out when scraping reports no content.
    """
    prompt = (
        "You are a financial-news research assistant. For the user's topic, "
        "propose up to "
        f"{_MAX_CANDIDATES} likely, real, publicly-reachable news article URLs "
        "(prefer reputable finance/news outlets) and a few refined search "
        "phrases. Only return URLs you are confident are real article pages "
        "(no paywalls, no logins, no homepages). "
        "Respond ONLY as JSON with this shape:\n"
        '{ "refined_queries": ["...", "..."], '
        '"candidates": [ {"title": "...", "url": "https://...", '
        '"source": "...", "why_relevant": "..."} ] }\n\n'
        f"Topic: {query}"
    )
    data = await common.gemini_generate_json(
        prompt, model=common.GEMINI_MODEL, agent_name=_AGENT_NEWS_CANDIDATES
    )
    return data if isinstance(data, dict) else {}


async def _summarize_article(url: str, title: str, source: str, content: str) -> Dict[str, Any]:
    """Produce a structured per-article report from REAL scraped text."""
    prompt = (
        "Analyze the following news article content and produce a concise, "
        "factual structured report. Do not invent facts beyond the provided "
        "text. If the text is insufficient, say so in the summary.\n\n"
        "Respond ONLY as JSON with this shape:\n"
        "{\n"
        '  "title": "best article title",\n'
        '  "source": "publication/source",\n'
        '  "summary": "3-5 sentence summary",\n'
        '  "key_points": ["...", "..."],\n'
        '  "tags": ["...", "..."],\n'
        '  "sentiment": "positive|neutral|negative",\n'
        '  "make_it_useful": {\n'
        '     "target_audience": "...",\n'
        '     "actionable_takeaways": ["...", "..."]\n'
        "  }\n"
        "}\n\n"
        f"Known title (may be empty): {title}\n"
        f"Known source (may be empty): {source}\n"
        f"URL: {url}\n\n"
        f"Article content:\n{_truncate(content, _MAX_SCRAPED_CHARS)}"
    )
    data = await common.gemini_generate_json(
        prompt, model=common.GEMINI_MODEL, agent_name=_AGENT_NEWS_REPORT
    )
    return data if isinstance(data, dict) else {}


def _src_attr(src: Any, name: str, default: str = "") -> str:
    """Read a field from a LinkUp source (object or dict), never raising."""
    try:
        val = src.get(name, default) if isinstance(src, dict) else getattr(src, name, default)
    except Exception:
        val = default
    return _coerce_str(val) if val is not None else default


async def _summarize_news_overview(query: str, answer_text: str) -> Optional[Dict[str, Any]]:
    """Turn a LinkUp source-backed answer into a structured briefing via Gemini.

    HONEST STATUS: returns None (never a fabricated/empty ``{}``) when Gemini is
    unavailable or returns nothing; callers then show the raw sourced answer.
    """
    if not answer_text or not answer_text.strip():
        return None
    prompt = (
        "You are a financial news analyst. Using ONLY the source-backed summary "
        f"below about '{query}', produce a concise briefing. Do not invent facts "
        "not present in the text.\n\n"
        "Respond ONLY as JSON:\n"
        '{ "summary": "3-5 sentence briefing", "key_points": ["...", "..."], '
        '"tags": ["...", "..."], "sentiment": "positive|neutral|negative" }\n\n'
        f"SOURCE-BACKED SUMMARY:\n{_truncate(answer_text, _MAX_TRANSCRIPT_CHARS)}"
    )
    data = await common.gemini_generate_json(
        prompt, model=common.GEMINI_MODEL, agent_name=_AGENT_NEWS_REPORT
    )
    if not isinstance(data, dict) or not data or "result" in data:
        return None
    return data


async def _run_news_via_linkup(query: str) -> Dict[str, Any]:
    """Primary News pipeline: LinkUp ``sourcedAnswer`` -> real current sources +
    a Gemini briefing. Reliable: no dependence on scraping bot-blocked sites.
    """
    out: Dict[str, Any] = {
        "query": query, "refined_queries": [], "rows": [],
        "overview": None, "mode": "linkup", "errors": [],
    }
    client = common.get_linkup_client()
    if client is None:
        out["errors"].append("LinkUp client unavailable.")
        return out

    import asyncio

    def _do():
        return client.search(
            query=f"Latest news and key developments: {query}",
            depth="standard", output_type="sourcedAnswer",
        )

    try:
        resp = await asyncio.to_thread(_do)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"LinkUp search failed: {e}")
        return out

    answer = getattr(resp, "answer", None)
    if answer is None and isinstance(resp, dict):
        answer = resp.get("answer")
    answer = _coerce_str(answer or "")
    sources = getattr(resp, "sources", None)
    if sources is None and isinstance(resp, dict):
        sources = resp.get("sources")
    sources = sources or []

    if not answer and not sources:
        out["errors"].append("LinkUp returned no answer or sources.")
        return out

    if answer:
        out["overview"] = await _summarize_news_overview(query, answer)
        if out["overview"] is None:
            # Gemini unavailable -> still show the raw sourced answer honestly.
            out["overview"] = {"summary": _truncate(answer, 1200),
                               "key_points": [], "tags": [], "sentiment": ""}

    seen = set()
    for s in sources:
        url = _src_attr(s, "url").strip()
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        name = _src_attr(s, "name") or url
        snippet = _src_attr(s, "snippet")
        out["rows"].append({
            "keep": True,
            "title": (name[:160] or url),
            "source": urlparse(url).netloc,
            "url": url,
            "summary": _truncate(snippet, 600),
            "key_points": "", "tags": "", "sentiment": "",
            "audience": "", "takeaways": "",
            "scrape_status": "linkup",
        })
        if len(out["rows"]) >= _MAX_CANDIDATES:
            break

    if not out["rows"] and not out["overview"]:
        out["errors"].append("LinkUp produced no usable sources or overview.")
    return out


async def _run_news_pipeline(query: str) -> Dict[str, Any]:
    """Dispatch: prefer LinkUp (real current sources); fall back to scraping."""
    if common.get_linkup_client() is not None:
        res = await _run_news_via_linkup(query)
        if res.get("rows") or res.get("overview"):
            return res
        fb = await _run_news_via_scrape(query)
        fb["errors"] = (res.get("errors") or []) + (fb.get("errors") or [])
        return fb
    return await _run_news_via_scrape(query)


async def _run_news_via_scrape(query: str) -> Dict[str, Any]:
    """Fallback News pipeline: LLM proposes URLs, we scrape the reachable ones.

    Used only when LinkUp is unavailable. NOTE: reputable news sites frequently
    block bots / paywall / 404 stale URLs, so this path is unreliable — LinkUp
    (``_run_news_via_linkup``) is preferred.

    Shape:
      {
        "query": str,
        "refined_queries": [str, ...],
        "rows": [ {title, source, url, summary, key_points, tags,
                   sentiment, audience, takeaways, scrape_status, keep} ],
        "overview": dict | None,
        "mode": str,
        "errors": [str, ...],   # honest reporting of what failed
      }
    """
    out: Dict[str, Any] = {
        "query": query,
        "refined_queries": [],
        "rows": [],
        "overview": None,
        "mode": "scrape",
        "errors": [],
    }

    # 1) Candidate discovery (LLM).
    proposal = await _propose_candidate_urls(query)
    out["refined_queries"] = [
        _coerce_str(q) for q in _safe_list(proposal.get("refined_queries"))
    ][:5]
    candidates = _safe_list(proposal.get("candidates"))[:_MAX_CANDIDATES]

    if not candidates:
        out["errors"].append(
            "No candidate articles were proposed (Gemini unavailable or empty)."
        )
        return out

    # Build url -> meta map (only well-formed http(s) URLs).
    url_meta: Dict[str, Dict[str, str]] = {}
    urls: List[str] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        url = _coerce_str(c.get("url")).strip()
        if not url.startswith(("http://", "https://")):
            continue
        if url in url_meta:
            continue
        url_meta[url] = {
            "title": _coerce_str(c.get("title")),
            "source": _coerce_str(c.get("source")),
        }
        urls.append(url)
        if len(urls) >= _MAX_CANDIDATES:
            break

    if not urls:
        out["errors"].append("No valid http(s) candidate URLs to scrape.")
        return out

    # 2) Scrape (SSRF-guarded, bounded, timed) via common.scrape_urls.
    scraped = await common.scrape_urls(urls, max_concurrent=_MAX_SCRAPE_CONCURRENCY)

    # 3) Summarize each successfully-scraped article (honest about failures).
    for url in urls:
        meta = url_meta.get(url, {})
        payload = scraped.get(url) or {}
        content = payload.get("content")
        err = payload.get("error")

        if not content:
            out["rows"].append(
                {
                    "keep": False,
                    "title": meta.get("title") or url,
                    "source": meta.get("source", ""),
                    "url": url,
                    "summary": "",
                    "key_points": "",
                    "tags": "",
                    "sentiment": "",
                    "audience": "",
                    "takeaways": "",
                    "scrape_status": f"no content: {err or 'unknown'}",
                }
            )
            out["errors"].append(f"{url}: {err or 'no content extracted'}")
            continue

        report = await _summarize_article(
            url, meta.get("title", ""), meta.get("source", ""), content
        )
        useful = report.get("make_it_useful") if isinstance(report, dict) else {}
        useful = useful if isinstance(useful, dict) else {}

        out["rows"].append(
            {
                "keep": True,
                "title": _coerce_str(report.get("title") or meta.get("title") or url),
                "source": _coerce_str(report.get("source") or meta.get("source")),
                "url": url,
                "summary": _coerce_str(report.get("summary")),
                "key_points": "; ".join(
                    _coerce_str(p) for p in _safe_list(report.get("key_points"))
                ),
                "tags": ", ".join(
                    _coerce_str(t) for t in _safe_list(report.get("tags"))
                ),
                "sentiment": _coerce_str(report.get("sentiment")),
                "audience": _coerce_str(useful.get("target_audience")),
                "takeaways": "; ".join(
                    _coerce_str(t) for t in _safe_list(useful.get("actionable_takeaways"))
                ),
                "scrape_status": "ok",
            }
        )

    if not any(r.get("scrape_status") == "ok" for r in out["rows"]):
        out["errors"].append(
            "None of the candidate URLs yielded usable content to summarize."
        )

    return out


def _render_news_alerts() -> None:
    st.markdown(
        "Enter a topic (e.g. *Fed interest rate decision*, *NVIDIA earnings*). "
        "ParselyFi runs a **source-backed search** (LinkUp) and generates a cited "
        "briefing plus an editable list of real sources you can keep/skip and export."
    )

    default_q = st.session_state.get(_K_NEWS_QUERY, "")
    with st.form("ny_news_form"):
        query = st.text_input(
            "News topic / query",
            value=default_q,
            placeholder="e.g. Federal Reserve interest rate outlook 2025",
            key="ny_news_query_input",
        )
        submitted = st.form_submit_button("Generate News Report", type="primary")

    if submitted:
        if not query or not query.strip():
            st.warning("Please enter a topic to search for.")
        else:
            st.session_state[_K_NEWS_QUERY] = query.strip()
            with st.spinner("Discovering candidates, scraping, and summarizing..."):
                try:
                    result = common.run_async(_run_news_pipeline(query.strip()))
                except Exception as e:  # never let the tab crash the app
                    logger.error("News pipeline failed: %s", e, exc_info=True)
                    st.error(f"News pipeline error: {e}")
                    result = None

            if result is not None:
                st.session_state[_K_NEWS_RESULTS] = result
                # BOUNDED history (FIFO cap = common.MAX_HISTORY).
                common.push_history(
                    _K_NEWS_HISTORY,
                    {
                        "query": result.get("query", ""),
                        "n_rows": len(result.get("rows", [])),
                        "n_ok": sum(
                            1 for r in result.get("rows", [])
                            if r.get("scrape_status") == "ok"
                        ),
                    },
                )

    result = st.session_state.get(_K_NEWS_RESULTS)
    if not result:
        st.info("No news report yet. Enter a topic above and generate one.")
        _render_news_history()
        return

    _render_news_results(result)
    _render_news_history()


def _render_news_results(result: Dict[str, Any]) -> None:
    rows = result.get("rows", [])
    errors = result.get("errors", [])
    refined = result.get("refined_queries", [])

    overview = result.get("overview")
    ok_count = sum(
        1 for r in rows
        if r.get("scrape_status") in ("ok", "linkup") or r.get("summary")
    )
    mode = result.get("mode", "")
    st.subheader(f"Report for: {result.get('query', '')}")
    st.caption(
        f"{len(rows)} source(s), {ok_count} with content"
        + (f" · via {mode}" if mode else "")
    )

    # Source-backed briefing (LinkUp answer, structured by Gemini).
    if isinstance(overview, dict) and overview:
        st.markdown("#### 📋 Briefing")
        if overview.get("summary"):
            st.write(overview["summary"])
        kps = _safe_list(overview.get("key_points"))
        if kps:
            st.markdown("**Key points:**")
            for kp in kps[:8]:
                st.markdown(f"- {_coerce_str(kp)}")
        meta_bits = []
        tags = _safe_list(overview.get("tags"))
        if tags:
            meta_bits.append("**Tags:** " + ", ".join(_coerce_str(t) for t in tags[:8]))
        if overview.get("sentiment"):
            meta_bits.append(f"**Sentiment:** {_coerce_str(overview.get('sentiment'))}")
        if meta_bits:
            st.caption(" · ".join(meta_bits))
        st.divider()

    if refined:
        st.markdown("**Refined search angles:** " + " · ".join(refined))

    if not rows:
        if not (isinstance(overview, dict) and overview):
            st.warning("No sources or briefing were produced. Try a different topic.")
        if errors:
            with st.expander("Diagnostics"):
                for e in errors:
                    st.write(f"- {e}")
        return

    st.markdown(
        "Edit any cell below. Toggle **keep** to include an item in the citations "
        "and export. Skipped rows are excluded."
    )

    edited_rows = rows
    if common.PANDAS_AVAILABLE:
        import pandas as pd  # local import; availability already confirmed

        df = pd.DataFrame(rows)
        # Stable, friendly column order.
        preferred = [
            "keep", "title", "source", "summary", "key_points", "tags",
            "sentiment", "audience", "takeaways", "url", "scrape_status",
        ]
        cols = [c for c in preferred if c in df.columns] + [
            c for c in df.columns if c not in preferred
        ]
        df = df[cols]

        column_config = {
            "keep": st.column_config.CheckboxColumn(
                "Keep", help="Include this article in citations & export."
            ),
            "url": st.column_config.LinkColumn("URL"),
            "scrape_status": st.column_config.TextColumn("Scrape", disabled=True),
            "summary": st.column_config.TextColumn("Summary", width="large"),
        }
        try:
            edited_df = st.data_editor(
                df,
                key="ny_news_data_editor",
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                column_config=column_config,
            )
            st.session_state[_K_NEWS_EDITED] = edited_df
            edited_rows = edited_df.to_dict("records")
        except Exception as e:
            logger.warning("data_editor failed, falling back to dataframe: %s", e)
            st.dataframe(df, use_container_width=True, hide_index=True)
            edited_rows = rows
    else:
        st.info("pandas not available; showing a read-only list.")
        for r in rows:
            st.write(
                f"- **{r.get('title')}** ({r.get('source')}) — "
                f"{r.get('scrape_status')}"
            )

    kept = [r for r in edited_rows if r.get("keep")]

    # Source citations (only kept items that actually scraped).
    st.subheader("Source Citations")
    cited = [r for r in kept if r.get("url")]
    if cited:
        for i, r in enumerate(cited, 1):
            title = r.get("title") or r.get("url")
            source = r.get("source") or ""
            status = r.get("scrape_status", "")
            note = "" if status in ("ok", "linkup") else f" _(note: {status})_"
            st.markdown(f"{i}. [{title}]({r.get('url')}) — {source}{note}")
    else:
        st.caption("No kept items with a URL to cite yet.")

    # Export kept rows.
    if kept:
        try:
            export = json.dumps(kept, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                "Download kept report (JSON)",
                data=export,
                file_name="news_report.json",
                mime="application/json",
                key="ny_news_download",
            )
        except Exception as e:
            logger.debug("news export failed: %s", e)

    if errors:
        with st.expander("Diagnostics (scrape/LLM failures)"):
            for e in errors:
                st.write(f"- {e}")


def _render_news_history() -> None:
    history = common.get_history(_K_NEWS_HISTORY)
    if not history:
        return
    with st.expander(f"Recent news queries ({len(history)})"):
        for h in reversed(history):
            st.write(
                f"- *{h.get('query', '')}* — "
                f"{h.get('n_ok', 0)}/{h.get('n_rows', 0)} usable"
            )


# ===========================================================================
# YouTube Daily Reports: URL (scrape) or pasted transcript -> Gemini report
# ===========================================================================

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _extract_video_id(url: str) -> Optional[str]:
    """Parse a YouTube video id with urllib (robust to param ordering).

    Returns None unless the candidate id matches the canonical 11-char
    [A-Za-z0-9_-] pattern, so a malformed/foreign URL never yields a bogus
    thumbnail id.
    """
    if not url:
        return None
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        vid: Optional[str] = None
        if host.endswith("youtu.be"):
            vid = parsed.path.lstrip("/").split("/")[0]
        elif "youtube.com" in host:
            vid = parse_qs(parsed.query).get("v", [None])[0]
            if not vid:
                # /embed/<id>, /shorts/<id>, /v/<id> path forms.
                parts = [p for p in parsed.path.split("/") if p]
                if len(parts) >= 2 and parts[0] in ("embed", "shorts", "v"):
                    vid = parts[1]
    except Exception:
        return None
    if vid and _VIDEO_ID_RE.match(vid):
        return vid
    return None


def _youtube_thumbnail(url: str) -> str:
    vid = _extract_video_id(url)
    if vid:
        return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
    return ""


async def _summarize_video(source_text: str, url: str) -> Dict[str, Any]:
    """Structured video report from scraped page text or pasted transcript."""
    prompt = (
        "Analyze the following YouTube video material (page text or transcript) "
        "and produce a structured daily-report style summary. Do not fabricate "
        "metadata you cannot infer from the text; leave unknown fields blank.\n\n"
        "Respond ONLY as JSON with this shape:\n"
        "{\n"
        '  "title": "...",\n'
        '  "channel": "...",\n'
        '  "date": "...",\n'
        '  "summary": "5-8 sentence summary",\n'
        '  "key_insights": [ {"insight": "...", "timestamp": "mm:ss", '
        '"significance": "..."} ],\n'
        '  "key_topics": [ {"name": "...", "description": "..."} ],\n'
        '  "tags": ["...", "..."],\n'
        '  "make_it_useful": {\n'
        '     "target_audience": "...",\n'
        '     "actionable_takeaways": ["...", "..."]\n'
        "  }\n"
        "}\n\n"
        f"Video URL (may be empty): {url}\n\n"
        f"Material:\n{_truncate(source_text, _MAX_TRANSCRIPT_CHARS)}"
    )
    data = await common.gemini_generate_json(
        prompt, model=common.GEMINI_MODEL, agent_name=_AGENT_YT_REPORT
    )
    return data if isinstance(data, dict) else {}


async def _run_youtube_pipeline(mode: str, raw_input: str) -> Dict[str, Any]:
    """Fetch/prepare material then summarize. Honest about scrape failures."""
    out: Dict[str, Any] = {
        "mode": mode,
        "url": "",
        "thumbnail": "",
        "report": {},
        "errors": [],
        "material_source": "",
    }

    source_text = ""
    url = ""

    if mode == "transcript":
        source_text = raw_input.strip()
        out["material_source"] = "pasted transcript"
        if not source_text:
            out["errors"].append("Empty transcript provided.")
            return out
    else:  # url mode
        url = raw_input.strip()
        out["url"] = url
        out["thumbnail"] = _youtube_thumbnail(url)
        if not url.startswith(("http://", "https://")):
            out["errors"].append("Please provide a full http(s) YouTube URL.")
            return out
        # Scrape the page text (SSRF-guarded, bounded) via common.scrape_urls.
        scraped = await common.scrape_urls([url], max_concurrent=1)
        payload = scraped.get(url) or {}
        content = payload.get("content")
        if content:
            source_text = content
            out["material_source"] = "scraped video page"
        else:
            err = payload.get("error") or "no content extracted"
            out["errors"].append(
                f"Could not scrape video page ({err}). "
                "Paste the transcript instead for a full report."
            )
            return out

    report = await _summarize_video(source_text, url)
    if not report:
        out["errors"].append(
            "Gemini produced no report (key missing or empty response)."
        )
    out["report"] = report
    return out


def _render_youtube_reports() -> None:
    st.markdown(
        "Generate a daily-report summary from a YouTube video. Provide a video "
        "URL (the page is scraped) or paste a transcript for the richest result."
    )

    mode_label = st.radio(
        "Input type",
        ["YouTube URL", "Paste transcript"],
        horizontal=True,
        key="ny_yt_mode_radio",
    )
    mode = "url" if mode_label == "YouTube URL" else "transcript"
    st.session_state[_K_YT_MODE] = mode

    with st.form("ny_yt_form"):
        if mode == "url":
            raw_input = st.text_input(
                "YouTube video URL",
                value=st.session_state.get(_K_YT_INPUT, ""),
                placeholder="https://www.youtube.com/watch?v=...",
                key="ny_yt_url_input",
            )
        else:
            raw_input = st.text_area(
                "Transcript / notes",
                value=st.session_state.get(_K_YT_INPUT, ""),
                height=220,
                placeholder="Paste the video transcript here...",
                key="ny_yt_transcript_input",
            )
        submitted = st.form_submit_button("Generate Video Report", type="primary")

    if submitted:
        if not raw_input or not raw_input.strip():
            st.warning("Please provide a URL or transcript.")
        else:
            st.session_state[_K_YT_INPUT] = raw_input.strip()
            with st.spinner("Preparing material and summarizing..."):
                try:
                    result = common.run_async(
                        _run_youtube_pipeline(mode, raw_input.strip())
                    )
                except Exception as e:
                    logger.error("YouTube pipeline failed: %s", e, exc_info=True)
                    st.error(f"YouTube pipeline error: {e}")
                    result = None

            if result is not None:
                st.session_state[_K_YT_RESULT] = result
                rep = result.get("report") or {}
                common.push_history(
                    _K_YT_HISTORY,
                    {
                        "url": result.get("url", ""),
                        "title": _coerce_str(rep.get("title")) or "(no title)",
                        "ok": bool(rep),
                    },
                )

    result = st.session_state.get(_K_YT_RESULT)
    if not result:
        st.info("No video report yet. Provide input above and generate one.")
        _render_youtube_history()
        return

    _render_youtube_result(result)
    _render_youtube_history()


def _render_youtube_result(result: Dict[str, Any]) -> None:
    errors = result.get("errors", [])
    # Copy before mutating: `result["report"]` is the session-state
    # source-of-truth (st.session_state[_K_YT_RESULT]["report"]). Render-time
    # edits (e.g. the editable summary) must not silently overwrite it without
    # an explicit save action; we mutate this local copy and export from it.
    report = dict(result.get("report") or {})
    url = result.get("url", "")
    thumb = result.get("thumbnail", "")

    if not report:
        st.warning("No report could be generated.")
        if errors:
            with st.expander("Diagnostics"):
                for e in errors:
                    st.write(f"- {e}")
        return

    st.caption(f"Material source: {result.get('material_source', 'unknown')}")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(_coerce_str(report.get("title")) or "Untitled video")
        meta_bits = []
        if report.get("channel"):
            meta_bits.append(f"Channel: {_coerce_str(report.get('channel'))}")
        if report.get("date"):
            meta_bits.append(f"Date: {_coerce_str(report.get('date'))}")
        if meta_bits:
            st.caption(" · ".join(meta_bits))
        if url:
            st.markdown(f"[Watch on YouTube]({url})")
    with col2:
        if thumb:
            try:
                st.image(thumb, use_container_width=True)
            except Exception:
                pass

    # Editable summary (edit-mode toggle, ported from the editable reference).
    edit_mode = st.toggle(
        "Enable editing mode",
        value=st.session_state.get(_K_YT_EDIT_MODE, False),
        key="ny_yt_edit_toggle",
    )
    st.session_state[_K_YT_EDIT_MODE] = edit_mode

    st.subheader("Summary")
    if edit_mode:
        new_summary = st.text_area(
            "Summary (editable)",
            value=_coerce_str(report.get("summary")),
            key="ny_yt_edit_summary",
            height=160,
        )
        # Cap edited summary length so a paste cannot blow up the export/state,
        # and write only to the local copy (not session-state source-of-truth).
        report["summary"] = _truncate(new_summary, _MAX_SCRAPED_CHARS)
    else:
        st.write(_coerce_str(report.get("summary")) or "No summary available.")

    # Key insights (with timestamps).
    insights = _safe_list(report.get("key_insights"))
    if insights:
        st.subheader("Key Insights")
        for i, ins in enumerate(insights, 1):
            if isinstance(ins, dict):
                ts = _coerce_str(ins.get("timestamp"))
                text = _coerce_str(ins.get("insight"))
                sig = _coerce_str(ins.get("significance"))
                line = f"{i}. {text}"
                if ts:
                    line += f"  _( {ts} )_"
                st.markdown(line)
                if sig:
                    st.caption(sig)
            else:
                st.markdown(f"{i}. {_coerce_str(ins)}")

    # Key topics.
    topics = _safe_list(report.get("key_topics"))
    if topics:
        st.subheader("Key Topics")
        for t in topics:
            if isinstance(t, dict):
                name = _coerce_str(t.get("name"))
                desc = _coerce_str(t.get("description"))
                if name or desc:
                    with st.expander(name or "Topic"):
                        st.write(desc)
            else:
                st.markdown(f"- {_coerce_str(t)}")

    # Tags.
    tags = [_coerce_str(t) for t in _safe_list(report.get("tags")) if _coerce_str(t)]
    if tags:
        st.subheader("Tags")
        st.write(", ".join(tags))

    # Make it useful.
    useful = report.get("make_it_useful")
    if isinstance(useful, dict) and useful:
        st.subheader("Make It Useful")
        audience = _coerce_str(useful.get("target_audience"))
        if audience:
            st.markdown(f"**Target audience:** {audience}")
        takeaways = [
            _coerce_str(t) for t in _safe_list(useful.get("actionable_takeaways"))
            if _coerce_str(t)
        ]
        if takeaways:
            st.markdown("**Actionable takeaways:**")
            for t in takeaways:
                st.markdown(f"- {t}")

    # Export (reflects any in-session edits to the summary).
    try:
        export = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "Download video report (JSON)",
            data=export,
            file_name="youtube_report.json",
            mime="application/json",
            key="ny_yt_download",
        )
    except Exception as e:
        logger.debug("youtube export failed: %s", e)

    if errors:
        with st.expander("Diagnostics"):
            for e in errors:
                st.write(f"- {e}")


def _render_youtube_history() -> None:
    history = common.get_history(_K_YT_HISTORY)
    if not history:
        return
    with st.expander(f"Recent video reports ({len(history)})"):
        for h in reversed(history):
            label = h.get("title") or h.get("url") or "(report)"
            mark = "ok" if h.get("ok") else "failed"
            st.write(f"- {label} — {mark}")


# ===========================================================================
# Public entry point
# ===========================================================================

def render_news_youtube_tab() -> None:
    """Render the News & YouTube feature tab (called inside a tab context).

    Honest status: if the Gemini key/SDK is missing we warn and return without
    fabricating any output.
    """
    st.subheader("News & YouTube")

    # HONEST STATUS gate: require a Gemini key. No key -> warn + return.
    ok, missing = common.feature_available(_REQUIRED_KEYS)
    if not ok:
        st.warning(
            "News & YouTube needs a Gemini API key to run. "
            f"Missing secret(s): {', '.join(missing)}. "
            "Add GEMINI_API_KEY (or legacy GOOGLE_AI_STUDIO) to your secrets, "
            "then reload."
        )
        return
    if common.get_gemini_client() is None:
        st.warning(
            "Gemini client unavailable (SDK not installed or key invalid). "
            "Cannot generate reports."
        )
        return

    # Key present but possibly rejected (leaked/invalid) — surface honestly.
    common.render_gemini_health()

    if not common.AIOHTTP_AVAILABLE:
        st.info(
            "Web scraping is disabled (aiohttp not installed). News candidate "
            "scraping and YouTube URL fetch will return no content; you can "
            "still paste a transcript for the YouTube report."
        )

    news_tab, yt_tab = st.tabs(["News Alerts", "YouTube Daily Reports"])
    with news_tab:
        _render_news_alerts()
    with yt_tab:
        _render_youtube_reports()

    # Shared, bounded token-usage panel (honest cost/usage accounting).
    with st.expander("Token usage & cost"):
        common.render_token_usage(inside_expander=True)
