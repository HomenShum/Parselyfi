"""
features/common.py
==================

Shared support module for the three ParselyFi feature tabs
(company_research, news_youtube, transcription).

Everything network/LLM/scrape related is centralized here so the feature
tabs stay thin. Hard contract: the public API names + behaviors below are
relied on by the feature modules. Do not rename without updating callers.

Design rules baked in (see project CLAUDE.md):
- HONEST STATUS: never fabricate data. If a key/dep is missing we return
  None / {} / "" / early, and surface a warning in the UI helpers. No
  success path on a failure path.
- BOUNDED MEMORY: the TokenLedger and the history helpers are capped with a
  MAX + oldest-eviction so nothing grows without bound across reruns.
- TIMEOUTS + bounded reads on every network/LLM/scrape call.
- SSRF guard: every user/LLM-supplied URL is resolved and checked against
  loopback / private / link-local / cloud-metadata ranges before any fetch.
- Optional libs (linkup, trafilatura, google.genai, bs4) are imported under
  try/except so importing this module never crashes; availability is
  reflected through feature_available() / get_*_client() returning None.

This module is import-safe in a non-Streamlit context too: ``streamlit`` is
imported under try/except and all session_state access is guarded.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("parselyfi.features.common")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

# ---------------------------------------------------------------------------
# Optional dependencies (wrapped so import never crashes)
# ---------------------------------------------------------------------------
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:  # pragma: no cover - streamlit always present in app, defensive here
    st = None  # type: ignore
    STREAMLIT_AVAILABLE = False

try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

try:
    import aiohttp  # type: ignore
    AIOHTTP_AVAILABLE = True
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore
    AIOHTTP_AVAILABLE = False

try:
    from bs4 import BeautifulSoup  # type: ignore
    BS4_AVAILABLE = True
except Exception:
    BeautifulSoup = None  # type: ignore
    BS4_AVAILABLE = False

try:
    import trafilatura  # type: ignore
    TRAFILATURA_AVAILABLE = True
except Exception:
    trafilatura = None  # type: ignore
    TRAFILATURA_AVAILABLE = False

try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
    GENAI_AVAILABLE = True
except Exception:
    genai = None  # type: ignore
    genai_types = None  # type: ignore
    GENAI_AVAILABLE = False

try:
    from linkup import LinkupClient  # type: ignore
    LINKUP_AVAILABLE = True
except Exception:
    LinkupClient = None  # type: ignore
    LINKUP_AVAILABLE = False

try:
    import nest_asyncio  # type: ignore
    NEST_ASYNCIO_AVAILABLE = True
except Exception:  # pragma: no cover
    nest_asyncio = None  # type: ignore
    NEST_ASYNCIO_AVAILABLE = False


# ===========================================================================
# Constants / config
# ===========================================================================

# The Gemini model the reference code uses (test7_create_company_report_LINKUP).
GEMINI_MODEL: str = "gemini-3.5-flash"  # latest flash (released 2026-05-19, 1M ctx)

# Per-1M-token prices (USD) for a rough cost estimate. These are estimates only
# and used for display; not a billing source of truth.
GEMINI_PRICING: Dict[str, Dict[str, float]] = {
    # model name -> {"input": $/1M input tokens, "output": $/1M output tokens}
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},  # released 2026-05-19
    "gemini-flash-latest": {"input": 1.50, "output": 9.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}
# Fallback pricing for unknown models so cost estimates never KeyError.
_DEFAULT_PRICING = {"input": 0.10, "output": 0.40}

# Secret names the contract reads at the TOP LEVEL of st.secrets.
GEMINI_KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_AI_STUDIO")  # primary, then legacy fallback
LINKUP_KEY_NAMES = ("LINKUP_API_KEY",)

# Bounded-memory caps.
MAX_HISTORY: int = 50
_MAX_LEDGER_ROWS: int = 500  # bounded token ledger

# Network / LLM safety budgets.
_LLM_TIMEOUT_S: float = 90.0          # total budget for a single Gemini call
_LLM_MAX_RESPONSE_CHARS: int = 2_000_000  # bounded read on model text output
_SCRAPE_TOTAL_TIMEOUT_S: float = 120.0    # total budget for a scrape batch


# ===========================================================================
# Secrets access (the ONLY way feature modules read secrets)
# ===========================================================================

def get_secret(name: str, section: str | None = None, default=None) -> str | None:
    """Safely read a secret from st.secrets (or env), never raising.

    Looks in ``st.secrets[section][name]`` when ``section`` is given, otherwise
    ``st.secrets[name]`` (top level), then falls back to ``os.environ[name]``.
    Returns ``default`` (None) if absent or if streamlit/secrets unavailable.
    """
    # 1) Streamlit secrets
    if STREAMLIT_AVAILABLE and st is not None:
        try:
            if section is not None:
                sect = st.secrets[section]  # may raise
                if name in sect:
                    val = sect[name]
                    if val not in (None, ""):
                        return val
            else:
                # st.secrets supports .get on the top level mapping
                try:
                    val = st.secrets.get(name)  # type: ignore[attr-defined]
                except Exception:
                    val = st.secrets[name] if name in st.secrets else None
                if val not in (None, ""):
                    return val
        except Exception:
            # Missing secrets.toml or missing key -> fall through to env
            pass

    # 2) Environment fallback
    env_val = os.getenv(name)
    if env_val not in (None, ""):
        return env_val

    return default


def _first_present_secret(names: Tuple[str, ...]) -> str | None:
    """Return the first non-empty secret among ``names``."""
    for n in names:
        v = get_secret(n)
        if v:
            return v
    return None


def feature_available(required_keys: List[str]) -> Tuple[bool, List[str]]:
    """Check that every required secret is present.

    Returns ``(ok, missing)`` where ``missing`` is the list of secret names
    that are absent. ``ok`` is True only when ``missing`` is empty.

    Special-cased aliases: requesting ``"GEMINI_API_KEY"`` is satisfied by
    either GEMINI_API_KEY or the legacy GOOGLE_AI_STUDIO secret.
    """
    missing: List[str] = []
    for key in required_keys:
        if key == "GEMINI_API_KEY":
            present = _first_present_secret(GEMINI_KEY_NAMES) is not None
        elif key == "LINKUP_API_KEY":
            present = _first_present_secret(LINKUP_KEY_NAMES) is not None
        else:
            present = get_secret(key) is not None
        if not present:
            missing.append(key)
    return (len(missing) == 0, missing)


# ===========================================================================
# Cached clients (genai + linkup)
# ===========================================================================
# We avoid importing st.cache_resource directly at module top so that a
# non-streamlit import still works. When streamlit is present we use a simple
# module-level singleton keyed by api key fingerprint.

_GEMINI_CLIENT_CACHE: Dict[str, Any] = {}
_LINKUP_CLIENT_CACHE: Dict[str, Any] = {}


def _key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def get_gemini_client():
    """Return a cached ``genai.Client`` or ``None`` if unavailable.

    Returns None (no raise) when the google-genai SDK is missing or when no
    Gemini API key secret is present. Uses the same init style as the
    reference: ``genai.Client(api_key=...)``.
    """
    if not GENAI_AVAILABLE or genai is None:
        logger.warning("google-genai SDK not installed; Gemini features disabled.")
        return None

    api_key = _first_present_secret(GEMINI_KEY_NAMES)
    if not api_key:
        logger.warning("No Gemini API key found (GEMINI_API_KEY / GOOGLE_AI_STUDIO).")
        return None

    fp = _key_fingerprint(api_key)
    client = _GEMINI_CLIENT_CACHE.get(fp)
    if client is None:
        try:
            client = genai.Client(api_key=api_key)
            _GEMINI_CLIENT_CACHE.clear()  # keep cache to a single live client
            _GEMINI_CLIENT_CACHE[fp] = client
            logger.info("Gemini client initialized.")
        except Exception as e:
            logger.error("Failed to init Gemini client: %s", e)
            return None
    return client


def get_linkup_client():
    """Return a cached ``LinkupClient`` or ``None`` if unavailable.

    Returns None (no raise) when the linkup SDK is missing or when
    LINKUP_API_KEY is absent.
    """
    if not LINKUP_AVAILABLE or LinkupClient is None:
        logger.warning("linkup SDK not installed; LinkUp features disabled.")
        return None

    api_key = _first_present_secret(LINKUP_KEY_NAMES)
    if not api_key:
        logger.warning("No LINKUP_API_KEY found; LinkUp features disabled.")
        return None

    fp = _key_fingerprint(api_key)
    client = _LINKUP_CLIENT_CACHE.get(fp)
    if client is None:
        try:
            client = LinkupClient(api_key=api_key)
            _LINKUP_CLIENT_CACHE.clear()
            _LINKUP_CLIENT_CACHE[fp] = client
            logger.info("LinkUp client initialized.")
        except Exception as e:
            logger.error("Failed to init LinkUp client: %s", e)
            return None
    return client


# ===========================================================================
# Async runner (port of AsyncTaskRunner) + run_async via nest_asyncio
# ===========================================================================

class AsyncTaskRunner:
    """Runs coroutines on a dedicated background event loop thread.

    Ported from the reference. Lets Streamlit (sync, rerun-driven) submit
    async work and block for the result without colliding with any loop the
    main thread might have.
    """

    def __init__(self) -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[Thread] = None
        self.initialized = False

    def initialize(self) -> None:
        if not self.initialized:
            self.loop = asyncio.new_event_loop()
            self.thread = Thread(target=self._run_event_loop_forever, daemon=True)
            self.thread.start()
            self.initialized = True

    def _run_event_loop_forever(self) -> None:
        assert self.loop is not None
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_task(self, coro):
        """Run a coroutine on the worker loop and block for its result."""
        if not self.initialized:
            self.initialize()
        assert self.loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def run_task_async(self, coro):
        """Submit a coroutine to the worker loop, returning a concurrent.futures.Future."""
        if not self.initialized:
            self.initialize()
        assert self.loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


_ASYNC_RUNNER_SINGLETON: Optional[AsyncTaskRunner] = None


def get_async_runner() -> AsyncTaskRunner:
    """Return the process-wide shared ``AsyncTaskRunner`` (initialized)."""
    global _ASYNC_RUNNER_SINGLETON
    # Prefer st.cache_resource when streamlit is available for proper reuse.
    if STREAMLIT_AVAILABLE and st is not None:
        try:
            @st.cache_resource(show_spinner=False)
            def _cached_runner() -> AsyncTaskRunner:
                r = AsyncTaskRunner()
                r.initialize()
                return r

            return _cached_runner()
        except Exception:
            pass  # fall through to module singleton

    if _ASYNC_RUNNER_SINGLETON is None:
        _ASYNC_RUNNER_SINGLETON = AsyncTaskRunner()
        _ASYNC_RUNNER_SINGLETON.initialize()
    return _ASYNC_RUNNER_SINGLETON


_NEST_APPLIED = False
_RUN_ASYNC_LOOP = None  # one persistent loop reused across run_async calls


def run_async(coro):
    """Run a coroutine to completion from a synchronous Streamlit context.

    Uses ``nest_asyncio`` so it works even when an event loop is already
    running (common in Streamlit). Avoids the deprecated
    ``asyncio.get_event_loop()`` and reuses a single owned loop instead of
    creating (and leaking) a new one per call; offloads to the shared
    AsyncTaskRunner if a loop is already running without nest_asyncio.
    """
    global _NEST_APPLIED, _RUN_ASYNC_LOOP
    if NEST_ASYNCIO_AVAILABLE and not _NEST_APPLIED:
        try:
            nest_asyncio.apply()
            _NEST_APPLIED = True
        except Exception as e:  # pragma: no cover
            logger.debug("nest_asyncio.apply() failed: %s", e)

    # Is a loop already running on THIS thread? (get_running_loop is the
    # non-deprecated check and only succeeds inside a running loop.)
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None:
        if NEST_ASYNCIO_AVAILABLE and _NEST_APPLIED:
            return running_loop.run_until_complete(coro)
        # Nested loop without nest_asyncio -> offload to the background runner.
        return get_async_runner().run_task(coro)

    # No running loop here: reuse one persistent loop we own (no per-call
    # creation/leak, no get_event_loop() deprecation warning).
    loop = _RUN_ASYNC_LOOP
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _RUN_ASYNC_LOOP = loop
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ===========================================================================
# Token ledger (bounded) + UI
# ===========================================================================

_LEDGER_SESSION_KEY = "common_token_ledger"


class TokenLedger:
    """Bounded record of LLM token usage with cost estimation.

    Bounded memory: keeps at most ``max_rows`` most-recent records; oldest are
    dropped (FIFO eviction). Aggregated totals are tracked separately so the
    running cost/usage summary survives eviction of detail rows.
    """

    _OTHER_KEY = "(other agents)"

    def __init__(self, max_rows: int = _MAX_LEDGER_ROWS,
                 max_agg_keys: int = 200) -> None:
        self.max_rows = max(1, int(max_rows))
        # Cap distinct aggregate keys too: callers may embed dynamic values
        # (e.g. a company name) in agent_name, so without a cap the aggregate
        # dicts grow unbounded across reruns. Evicted agents fold into a single
        # "(other agents)" bucket so totals stay honest.
        self.max_agg_keys = max(1, int(max_agg_keys))
        # Detail rows (bounded).
        self._rows: List[Dict[str, Any]] = []
        # Aggregates that persist even after detail rows are evicted.
        self._agg_in: Dict[str, int] = {}   # agent -> total input tokens
        self._agg_out: Dict[str, int] = {}  # agent -> total output tokens
        self._agg_model: Dict[str, str] = {}  # agent -> last model seen
        self._agg_order: List[str] = []     # insertion order for FIFO eviction

    def record(
        self,
        agent_name: str,
        model: str,
        in_tokens: int,
        out_tokens: int = 0,
    ) -> None:
        """Record one call's token usage (input + output) for an agent."""
        try:
            in_tokens = max(0, int(in_tokens or 0))
            out_tokens = max(0, int(out_tokens or 0))
        except (TypeError, ValueError):
            in_tokens, out_tokens = 0, 0

        agent_name = str(agent_name or "agent")
        model = str(model or GEMINI_MODEL)

        self._rows.append(
            {
                "timestamp": datetime.now(),
                "agent_name": agent_name,
                "model": model,
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
            }
        )
        # BOUNDED: evict oldest detail rows beyond cap.
        if len(self._rows) > self.max_rows:
            self._rows = self._rows[-self.max_rows:]

        # Aggregates (bounded by max_agg_keys; evicted agents fold into a
        # single "(other agents)" bucket so the running totals stay honest).
        if agent_name not in self._agg_in:
            self._agg_order.append(agent_name)
        self._agg_in[agent_name] = self._agg_in.get(agent_name, 0) + in_tokens
        self._agg_out[agent_name] = self._agg_out.get(agent_name, 0) + out_tokens
        self._agg_model[agent_name] = model
        self._evict_aggregates()

    def _evict_aggregates(self) -> None:
        """Fold the oldest distinct agents into '(other agents)' over the cap."""
        while len(self._agg_in) > self.max_agg_keys:
            victim = next((k for k in self._agg_order if k != self._OTHER_KEY), None)
            if victim is None:
                break
            self._agg_order.remove(victim)
            vi = self._agg_in.pop(victim, 0)
            vo = self._agg_out.pop(victim, 0)
            self._agg_model.pop(victim, None)
            if self._OTHER_KEY not in self._agg_in:
                self._agg_order.append(self._OTHER_KEY)
            self._agg_in[self._OTHER_KEY] = self._agg_in.get(self._OTHER_KEY, 0) + vi
            self._agg_out[self._OTHER_KEY] = self._agg_out.get(self._OTHER_KEY, 0) + vo
            self._agg_model.setdefault(self._OTHER_KEY, GEMINI_MODEL)

    @staticmethod
    def _price(model: str) -> Dict[str, float]:
        return GEMINI_PRICING.get(model, _DEFAULT_PRICING)

    def total_cost(self) -> float:
        """Estimated total USD cost across all recorded usage."""
        cost = 0.0
        for agent, in_tok in self._agg_in.items():
            out_tok = self._agg_out.get(agent, 0)
            price = self._price(self._agg_model.get(agent, GEMINI_MODEL))
            cost += (in_tok / 1_000_000.0) * price["input"]
            cost += (out_tok / 1_000_000.0) * price["output"]
        return round(cost, 6)

    def total_tokens(self) -> int:
        """Total input+output tokens across all agents."""
        return int(sum(self._agg_in.values()) + sum(self._agg_out.values()))

    def summary_df(self):
        """Per-agent aggregate DataFrame (input, output, total, est cost)."""
        if not PANDAS_AVAILABLE:
            return None
        rows = []
        for agent in sorted(set(self._agg_in) | set(self._agg_out)):
            in_tok = self._agg_in.get(agent, 0)
            out_tok = self._agg_out.get(agent, 0)
            model = self._agg_model.get(agent, GEMINI_MODEL)
            price = self._price(model)
            cost = (in_tok / 1e6) * price["input"] + (out_tok / 1e6) * price["output"]
            rows.append(
                {
                    "Agent": agent,
                    "Model": model,
                    "Input Tokens": in_tok,
                    "Output Tokens": out_tok,
                    "Total Tokens": in_tok + out_tok,
                    "Est. Cost ($)": round(cost, 6),
                }
            )
        if rows:
            total_row = {
                "Agent": "TOTAL",
                "Model": "",
                "Input Tokens": sum(r["Input Tokens"] for r in rows),
                "Output Tokens": sum(r["Output Tokens"] for r in rows),
                "Total Tokens": sum(r["Total Tokens"] for r in rows),
                "Est. Cost ($)": round(sum(r["Est. Cost ($)"] for r in rows), 6),
            }
            rows.append(total_row)
        return pd.DataFrame(rows)

    def df(self):
        """Detailed per-call DataFrame (bounded to the most-recent rows).

        Returns a pandas.DataFrame. If pandas is unavailable, returns the raw
        list of row dicts so callers still get the data (honest, no crash).
        """
        if not PANDAS_AVAILABLE:
            return list(self._rows)
        if not self._rows:
            return pd.DataFrame(
                columns=["Timestamp", "Agent", "Model", "Input Tokens",
                         "Output Tokens", "Total Tokens"]
            )
        return pd.DataFrame(
            [
                {
                    "Timestamp": r["timestamp"].strftime("%H:%M:%S"),
                    "Agent": r["agent_name"],
                    "Model": r["model"],
                    "Input Tokens": r["in_tokens"],
                    "Output Tokens": r["out_tokens"],
                    "Total Tokens": r["in_tokens"] + r["out_tokens"],
                }
                for r in self._rows
            ]
        )


def _get_ledger() -> TokenLedger:
    """Return the shared TokenLedger from session_state (or a module fallback)."""
    if STREAMLIT_AVAILABLE and st is not None:
        try:
            if _LEDGER_SESSION_KEY not in st.session_state:
                st.session_state[_LEDGER_SESSION_KEY] = TokenLedger()
            return st.session_state[_LEDGER_SESSION_KEY]
        except Exception:
            pass
    # Fallback module-level singleton (no streamlit / no session).
    global _MODULE_LEDGER
    try:
        return _MODULE_LEDGER  # type: ignore[name-defined]
    except NameError:
        globals()["_MODULE_LEDGER"] = TokenLedger()
        return globals()["_MODULE_LEDGER"]


def record_tokens(agent_name: str, model: str, in_tokens: int, out_tokens: int = 0) -> None:
    """Convenience: record token usage into the shared session-state ledger."""
    _get_ledger().record(agent_name, model, in_tokens, out_tokens)


def render_token_usage(inside_expander: bool = False) -> None:
    """Render the token ledger summary + detail in Streamlit.

    No-op-with-info when there is no usage yet. ``inside_expander`` avoids
    nesting an expander inside another (Streamlit forbids that).
    """
    if not STREAMLIT_AVAILABLE or st is None:
        return

    ledger = _get_ledger()
    if ledger.total_tokens() == 0:
        st.info("No token usage data available yet.")
        return

    st.subheader("Token Usage Summary")
    summary = ledger.summary_df()
    if summary is not None:
        st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption(
        f"Total tokens: {ledger.total_tokens():,} | "
        f"Estimated cost: ${ledger.total_cost():.4f}"
    )

    detail = ledger.df()
    if inside_expander:
        st.subheader("Detailed Token Log")
        if detail is not None:
            st.dataframe(detail, use_container_width=True, hide_index=True)
    else:
        with st.expander("View Detailed Token Log"):
            if detail is not None:
                st.dataframe(detail, use_container_width=True, hide_index=True)


# ===========================================================================
# Gemini generation helpers (JSON + text), with token recording + timeout
# ===========================================================================

def _extract_text_from_response(response) -> str:
    """Pull text out of a genai response defensively (bounded)."""
    text = ""
    try:
        # Newer SDKs expose .text directly.
        t = getattr(response, "text", None)
        if t:
            text = t
        else:
            cand = response.candidates[0]
            parts = cand.content.parts
            text = "".join(getattr(p, "text", "") or "" for p in parts)
    except Exception:
        try:
            cand = response.candidates[0]
            text = cand.content.parts[0].text
        except Exception:
            text = ""
    if text and len(text) > _LLM_MAX_RESPONSE_CHARS:
        logger.warning("Gemini response exceeded %d chars; truncating.",
                       _LLM_MAX_RESPONSE_CHARS)
        text = text[:_LLM_MAX_RESPONSE_CHARS]
    return text or ""


def _record_usage_from_response(response, agent_name: str, model: str) -> None:
    """Record token usage from a genai response's usage_metadata if present."""
    in_tok, out_tok = 0, 0
    try:
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            in_tok = int(getattr(um, "prompt_token_count", 0) or 0)
            out_tok = int(getattr(um, "candidates_token_count", 0) or 0)
    except Exception:
        in_tok, out_tok = 0, 0
    record_tokens(agent_name, model, in_tok, out_tok)


# ---------------------------------------------------------------------------
# LLM health tracking
#
# Gemini calls run inside background threads (run_async / AsyncTaskRunner), so
# we cannot touch st.session_state from there. Record the last LLM error in a
# process-global guarded by a lock; the main thread reads it at render time to
# surface an actionable banner (e.g. a leaked/invalid key) instead of letting
# the failure look like an empty result.
# ---------------------------------------------------------------------------
_llm_error_lock = Lock()
_last_llm_error: Optional[Dict[str, str]] = None


def _looks_like_auth_error(message: str) -> bool:
    m = (message or "").lower()
    return any(
        s in m
        for s in (
            "permission_denied", "permission denied", "api key not valid",
            "api_key_invalid", "leaked", "unauthorized", "unauthenticated",
            "invalid api key", " 401", " 403",
        )
    )


def _record_llm_error(agent_name: str, exc: Exception) -> None:
    global _last_llm_error
    try:
        with _llm_error_lock:
            _last_llm_error = {"agent": str(agent_name), "error": str(exc)}
    except Exception:  # never let diagnostics break the call path
        pass


def gemini_last_error() -> Optional[Dict[str, str]]:
    """Return the most recent Gemini call error, or None."""
    with _llm_error_lock:
        return dict(_last_llm_error) if _last_llm_error else None


def render_gemini_health() -> None:
    """Surface an actionable banner when Gemini is unusable.

    Shows nothing when Gemini is configured and no auth error has been seen.
    Distinguishes a missing key from a present-but-rejected (leaked/invalid)
    key so an empty result isn't mistaken for "no data".
    """
    if not STREAMLIT_AVAILABLE:
        return
    ok, _missing = feature_available(["GEMINI_API_KEY"])
    if not ok:
        st.warning(
            "⚠️ Gemini is not configured — set `GEMINI_API_KEY` in secrets. "
            "AI extraction and summaries are disabled until then."
        )
        return
    err = gemini_last_error()
    if err and _looks_like_auth_error(err.get("error", "")):
        st.error(
            "⚠️ Gemini rejected the API key (invalid or reported as leaked). "
            "Rotate `GEMINI_API_KEY` in your secrets — AI extraction and "
            "summaries will return empty until a valid key is set."
        )


async def gemini_generate_json(
    prompt: str,
    *,
    model: str = GEMINI_MODEL,
    agent_name: str = "agent",
) -> dict:
    """Generate JSON with Gemini and parse to a dict.

    Uses ``response_mime_type='application/json'``. Records token usage.
    Returns ``{}`` on any failure (missing client, timeout, parse error) and
    logs the reason. Never raises, never fabricates.
    """
    client = get_gemini_client()
    if client is None or genai_types is None:
        logger.warning("gemini_generate_json: no Gemini client; returning {}.")
        return {}

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model,
                contents=[prompt],
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            ),
            timeout=_LLM_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.error("gemini_generate_json[%s]: timeout after %ss.", agent_name, _LLM_TIMEOUT_S)
        return {}
    except Exception as e:
        logger.error("gemini_generate_json[%s]: request failed: %s", agent_name, e)
        _record_llm_error(agent_name, e)
        return {}

    _record_usage_from_response(response, agent_name, model)

    text = _extract_text_from_response(response)
    if not text:
        logger.warning("gemini_generate_json[%s]: empty response.", agent_name)
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to salvage a JSON object embedded in the text.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception as e:
                logger.error("gemini_generate_json[%s]: JSON parse failed: %s", agent_name, e)
                return {}
        else:
            logger.error("gemini_generate_json[%s]: no JSON object in response.", agent_name)
            return {}

    if isinstance(parsed, dict):
        return parsed
    # Honest: the contract is dict. Wrap non-dict JSON rather than lie about shape.
    logger.warning("gemini_generate_json[%s]: parsed JSON was %s, wrapping.",
                   agent_name, type(parsed).__name__)
    return {"result": parsed}


async def gemini_generate_text(
    prompt: str,
    *,
    model: str = GEMINI_MODEL,
    agent_name: str = "agent",
) -> str:
    """Generate free-form text with Gemini.

    Records token usage. Returns ``""`` on any failure and logs it.
    """
    client = get_gemini_client()
    if client is None or genai_types is None:
        logger.warning("gemini_generate_text: no Gemini client; returning ''.")
        return ""

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model,
                contents=[prompt],
                config=genai_types.GenerateContentConfig(),
            ),
            timeout=_LLM_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.error("gemini_generate_text[%s]: timeout after %ss.", agent_name, _LLM_TIMEOUT_S)
        return ""
    except Exception as e:
        logger.error("gemini_generate_text[%s]: request failed: %s", agent_name, e)
        _record_llm_error(agent_name, e)
        return ""

    _record_usage_from_response(response, agent_name, model)
    return _extract_text_from_response(response)


# ===========================================================================
# SSRF guard
# ===========================================================================

_BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",  # GCP metadata
}


def _ip_is_blocked(ip_str: str) -> bool:
    """True if an IP is loopback / private / link-local / reserved / metadata."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable -> block to be safe
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True
    # Cloud metadata endpoint (AWS/GCP/Azure) lives at 169.254.169.254 which is
    # link-local (covered above), but block the exact address explicitly too.
    if ip_str == "169.254.169.254":
        return True
    return False


def _is_url_safe(url: str) -> Tuple[bool, str]:
    """SSRF guard: resolve the host and reject internal/metadata targets.

    Returns ``(ok, reason)``. ``ok`` is False for non-http(s) schemes,
    blocked hostnames, or any resolved IP in a private/loopback/link-local/
    reserved range (including the 169.254.169.254 cloud-metadata address).
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "unparseable URL"

    if parsed.scheme not in ("http", "https"):
        return False, f"disallowed scheme: {parsed.scheme!r}"

    host = parsed.hostname
    if not host:
        return False, "missing host"

    if host.lower() in _BLOCKED_HOSTNAMES:
        return False, f"blocked hostname: {host}"

    # If the host is a literal IP, check it directly.
    try:
        ipaddress.ip_address(host)
        if _ip_is_blocked(host):
            return False, f"blocked IP literal: {host}"
        return True, "ok"
    except ValueError:
        pass  # not a literal IP -> resolve it

    # Resolve ALL addresses and block if ANY is internal. This is a check-time
    # filter; the actual connect-time DNS resolution is independently guarded by
    # _make_guarded_connector (which refuses internal IPs at connect), so a
    # DNS-rebind between this check and the fetch is also rejected.
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception as e:
        return False, f"DNS resolution failed: {e}"

    if not infos:
        return False, "no DNS records"

    for info in infos:
        ip_str = info[4][0]
        if _ip_is_blocked(ip_str):
            return False, f"resolves to blocked IP {ip_str}"

    return True, "ok"


# ===========================================================================
# Ethical web scraper (ported + SSRF-hardened)
# ===========================================================================

@dataclass
class ScraperConfig:
    """Configuration for the ethical web scraper."""
    request_delay: float = 2.0            # base per-domain delay (seconds)
    timeout: int = 30                      # per-request timeout (seconds)
    max_retries: int = 3                   # max retry attempts
    user_agent: str = (
        "Mozilla/5.0 (compatible; ParselyFiScraper/1.0; "
        "+https://github.com/parselyfi/ethical-scraper)"
    )
    max_concurrent: int = 3                # max concurrent requests
    respect_robots: bool = True            # honor robots.txt
    cache_enabled: bool = True             # cache responses in-memory (bounded)
    content_size_limit: int = 5 * 1024 * 1024  # 5MB bounded read cap
    follow_redirects: bool = True          # follow HTTP redirects
    max_redirect_depth: int = 5            # max redirect chain depth
    total_timeout: float = _SCRAPE_TOTAL_TIMEOUT_S  # total budget for a batch
    cache_max_entries: int = 256           # BOUNDED cache size (entries)


def _make_guarded_connector(limit: int):
    """Build a TCPConnector whose DNS resolution rejects internal IPs at
    connect time, closing the DNS-rebind TOCTOU window (a hostname that
    resolves public at check time but private at connect time is refused).

    Falls back to a plain connector (per-hop _is_url_safe checks still apply)
    if the guarded resolver can't be constructed on this aiohttp version.
    """
    try:
        base_resolver = aiohttp.ThreadedResolver()

        class _GuardResolver(aiohttp.abc.AbstractResolver):
            async def resolve(self, host, port=0, family=socket.AF_INET):
                res = await base_resolver.resolve(host, port, family)
                for r in res:
                    if _ip_is_blocked(r["host"]):
                        raise OSError(
                            f"SSRF: {host} resolves to blocked {r['host']}"
                        )
                return res

            async def close(self):
                await base_resolver.close()

        return aiohttp.TCPConnector(
            limit=limit, force_close=True, enable_cleanup_closed=True,
            ttl_dns_cache=0,  # re-resolve through the guard every connect
            resolver=_GuardResolver(),
        )
    except Exception as e:  # noqa: BLE001 - never let hardening break scraping
        logger.warning(
            "Guarded DNS connector unavailable (%s); using plain connector "
            "with per-hop SSRF checks only.", e)
        return aiohttp.TCPConnector(
            limit=limit, force_close=True, enable_cleanup_closed=True,
            ttl_dns_cache=300,
        )


class EthicalWebScraper:
    """Async ethical scraper: robots.txt, per-domain delay, SSRF guard,
    content-size cap, bounded cache, and a total-timeout budget.

    Hardened vs. the reference: every fetch goes through an SSRF guard that
    resolves the host and rejects loopback/private/link-local/reserved/
    metadata (169.254.169.254) targets before any network read.
    """

    def __init__(self, config: Optional[ScraperConfig] = None) -> None:
        self.config = config or ScraperConfig()
        self.domain_delays: Dict[str, float] = {}
        self.last_request_times: Dict[str, float] = {}
        self.robots_parsers: Dict[str, RobotFileParser] = {}
        self.session = None  # aiohttp.ClientSession
        self.cache: Dict[str, Tuple[float, str]] = {}  # url -> (ts, content)
        self.cache_ttl = 3600
        # Known link-shortener / redirect services worth resolving.
        self.redirect_services = [
            "vertexaisearch.cloud.google.com",
            "bit.ly", "tinyurl.com", "t.co", "goo.gl",
            "ow.ly", "is.gd", "buff.ly",
        ]

    # ---- lifecycle ----
    async def __aenter__(self):
        if not AIOHTTP_AVAILABLE or aiohttp is None:
            raise RuntimeError("aiohttp not installed; scraping unavailable.")
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "DNT": "1",
        }
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent,
            force_close=True,
            enable_cleanup_closed=True,
            ttl_dns_cache=300,
        )
        self.session = aiohttp.ClientSession(
            headers=headers,
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            trust_env=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    # ---- helpers ----
    def _get_domain(self, url: str) -> str:
        return urlparse(url).netloc

    def _cache_put(self, key: str, content: str) -> None:
        # BOUNDED cache: evict oldest if over cap.
        if len(self.cache) >= self.config.cache_max_entries:
            oldest_key = min(self.cache, key=lambda k: self.cache[k][0])
            self.cache.pop(oldest_key, None)
        self.cache[key] = (time.time(), content)

    def _is_redirect_service(self, url: str) -> bool:
        domain = self._get_domain(url)
        return any(svc in domain for svc in self.redirect_services)

    async def _resolve_redirects(self, url: str, depth: int = 0) -> str:
        if depth >= self.config.max_redirect_depth:
            return url
        if not self._is_redirect_service(url):
            return url
        # SSRF: validate the URL we are about to HEAD (a shortener's Location
        # can point anywhere) before any network egress.
        ok, reason = _is_url_safe(url)
        if not ok:
            logger.warning("SSRF guard blocked redirect probe %s (%s)", url, reason)
            return url
        try:
            async with self.session.head(
                url, allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if location:
                        if not location.startswith(("http://", "https://")):
                            location = urljoin(url, location)
                        # Validate each resolved hop before following it.
                        ok2, reason2 = _is_url_safe(location)
                        if not ok2:
                            logger.warning(
                                "SSRF guard blocked redirect target %s (%s)",
                                location, reason2)
                            return url
                        return await self._resolve_redirects(location, depth + 1)
        except Exception as e:
            logger.debug("Redirect resolve failed for %s: %s", url, e)
        return url

    async def _get_robots_parser(self, url: str) -> Optional[RobotFileParser]:
        if not self.config.respect_robots:
            return None
        domain = self._get_domain(url)
        if domain in self.robots_parsers:
            return self.robots_parsers[domain]
        robots_url = f"{urlparse(url).scheme}://{domain}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            # allow_redirects=False: do NOT follow a robots.txt 3xx into a
            # possibly-internal host (SSRF). A redirected robots.txt is treated
            # as "no robots" (allow_all) rather than chased.
            async with self.session.get(
                robots_url, allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    content = await response.text(errors="replace")
                    rp.parse(content.splitlines())
                    crawl_delay = rp.crawl_delay(self.config.user_agent)
                    if crawl_delay:
                        self.domain_delays[domain] = max(
                            float(crawl_delay), self.config.request_delay
                        )
                else:
                    rp.allow_all = True
        except Exception as e:
            logger.debug("robots.txt fetch failed for %s: %s", domain, e)
            rp.allow_all = True
        self.robots_parsers[domain] = rp
        return rp

    async def _enforce_delay(self, domain: str) -> None:
        now = time.time()
        last = self.last_request_times.get(domain)
        required = self.domain_delays.get(domain, self.config.request_delay)
        if last is not None:
            elapsed = now - last
            if elapsed < required:
                await asyncio.sleep(required - elapsed)
        self.last_request_times[domain] = time.time()

    async def _fetch_url(self, url: str, retry_count: int = 0) -> Optional[str]:
        # Resolve known redirect services first.
        if self._is_redirect_service(url):
            resolved = await self._resolve_redirects(url)
            if resolved != url:
                logger.info("Resolved redirect %s -> %s", url, resolved)
                url = resolved

        # SSRF guard BEFORE any fetch.
        ok, reason = _is_url_safe(url)
        if not ok:
            logger.warning("SSRF guard blocked %s (%s)", url, reason)
            return None

        domain = self._get_domain(url)

        # Cache check.
        if self.config.cache_enabled and url in self.cache:
            ts, content = self.cache[url]
            if time.time() - ts < self.cache_ttl:
                return content
            self.cache.pop(url, None)

        # robots.txt.
        if self.config.respect_robots:
            rp = await self._get_robots_parser(url)
            if rp and not getattr(rp, "allow_all", False):
                try:
                    if not rp.can_fetch(self.config.user_agent, url):
                        logger.warning("robots.txt disallows %s", url)
                        return None
                except Exception:
                    pass

        await self._enforce_delay(domain)

        try:
            current = url
            redirects_left = (
                self.config.max_redirect_depth if self.config.follow_redirects else 0
            )
            while True:
                # allow_redirects=False: follow redirects MANUALLY so every hop
                # is SSRF-validated. aiohttp's auto-redirect would chase a 3xx to
                # an internal/metadata host (incl. literal-IP targets) unchecked.
                async with self.session.get(current, allow_redirects=False) as response:
                    if response.status in (301, 302, 303, 307, 308):
                        if redirects_left <= 0:
                            logger.warning("Redirect limit reached for %s", url)
                            return None
                        location = response.headers.get("Location")
                        if not location:
                            logger.warning("Redirect with no Location for %s", current)
                            return None
                        nxt = (location if location.startswith(("http://", "https://"))
                               else urljoin(current, location))
                        ok2, reason2 = _is_url_safe(nxt)
                        if not ok2:
                            logger.warning("SSRF guard blocked redirect %s -> %s (%s)",
                                           current, nxt, reason2)
                            return None
                        redirects_left -= 1
                        current = nxt
                        await self._enforce_delay(self._get_domain(current))
                        continue

                    if response.status >= 400:
                        if response.status == 429 and retry_count < self.config.max_retries:
                            retry_after = response.headers.get(
                                "Retry-After", self.config.request_delay * 2
                            )
                            try:
                                await asyncio.sleep(float(retry_after))
                            except (TypeError, ValueError):
                                await asyncio.sleep(self.config.request_delay * 2)
                            return await self._fetch_url(url, retry_count + 1)
                        logger.warning("HTTP %s for %s", response.status, current)
                        return None

                    content_type = response.headers.get("Content-Type", "")
                    if "html" not in content_type.lower() and "text" not in content_type.lower():
                        logger.warning("Non-HTML/text content type %s for %s", content_type, current)
                        return None

                    # Bounded read: cap declared length and actual read.
                    declared = int(response.headers.get("Content-Length", 0) or 0)
                    if declared and declared > self.config.content_size_limit:
                        logger.warning("Content too large (%d bytes) for %s", declared, current)
                        return None

                    raw = await response.content.read(self.config.content_size_limit + 1)
                    if len(raw) > self.config.content_size_limit:
                        logger.warning("Content exceeded size cap for %s", current)
                        return None
                    # Use the Content-Type charset (header-only); get_encoding()'s
                    # chardet fallback raises on a streamed/partial body. Default utf-8.
                    encoding = response.charset or "utf-8"
                    try:
                        content = raw.decode(encoding, errors="replace")
                    except (LookupError, TypeError):
                        content = raw.decode("utf-8", errors="replace")

                    if self.config.cache_enabled:
                        self._cache_put(url, content)
                    return content

        except asyncio.TimeoutError:
            logger.warning("Timeout fetching %s", url)
            if retry_count < self.config.max_retries:
                await asyncio.sleep(min(2 ** retry_count, 10))
                return await self._fetch_url(url, retry_count + 1)
            return None
        except Exception as e:
            if AIOHTTP_AVAILABLE and isinstance(e, aiohttp.ClientError) \
                    and retry_count < self.config.max_retries:
                await asyncio.sleep(min(2 ** retry_count, 10))
                return await self._fetch_url(url, retry_count + 1)
            logger.error("Error fetching %s: %s", url, e)
            return None

    async def _extract_content(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        try:
            if TRAFILATURA_AVAILABLE and trafilatura is not None:
                extracted = await asyncio.to_thread(
                    trafilatura.extract,
                    html,
                    url=url,
                    include_comments=False,
                    include_tables=True,
                    output_format="json",
                )
                if extracted:
                    result = json.loads(extracted)
                    if not result.get("text") or len(result.get("text", "")) < 100:
                        result = self._extract_with_bs4(html)
                    result["url"] = url
                    result["timestamp"] = time.time()
                    result["content_hash"] = hashlib.sha256(
                        (result.get("text") or "").encode()
                    ).hexdigest()
                    return result
            # Fallback to BeautifulSoup (or minimal regex if bs4 missing).
            result = self._extract_with_bs4(html)
            result["url"] = url
            result["timestamp"] = time.time()
            result["content_hash"] = hashlib.sha256(
                (result.get("text") or "").encode()
            ).hexdigest()
            return result
        except Exception as e:
            logger.error("Content extraction failed for %s: %s", url, e)
            return None

    def _extract_with_bs4(self, html: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"text": "", "title": "", "source": "fallback"}
        if not BS4_AVAILABLE or BeautifulSoup is None:
            # Crude tag strip so we still return *some* honest text.
            text = re.sub(r"<script.*?</script>", " ", html, flags=re.DOTALL | re.I)
            text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.I)
            text = re.sub(r"<[^>]+>", " ", text)
            result["text"] = " ".join(text.split())
            result["source"] = "regex-fallback"
            return result
        try:
            soup = BeautifulSoup(html, "html.parser")
            if soup.title and soup.title.string:
                result["title"] = soup.title.string
            for el in soup(["script", "style", "nav", "footer", "iframe", "img", "svg"]):
                el.decompose()
            article = (
                soup.find("article")
                or soup.find("main")
                or soup.find("div", role="main")
            )
            text = (article or soup).get_text(separator=" ", strip=True)
            result["text"] = " ".join(text.split())
            result["source"] = "BeautifulSoup"
        except Exception as e:
            logger.error("BeautifulSoup extraction error: %s", e)
        return result

    async def scrape_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single page; returns extracted-content dict or None."""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        html = await self._fetch_url(url)
        if not html:
            return None
        return await self._extract_content(html, url)


# ===========================================================================
# Scrape helpers (module-level public API)
# ===========================================================================

# URL pattern shared by the context-scraping helpers.
_URL_PATTERN = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:[^\s<>"\')\]]*)')


async def scrape_urls(urls: List[str], max_concurrent: int = 3) -> dict:
    """Scrape a list of URLs concurrently (bounded, SSRF-guarded, timed).

    Returns ``{url: {"content": str|None, "error": str|None}}``. On any
    per-URL failure ``content`` is None and ``error`` explains why. Honest:
    blocked/unreachable URLs are reported, not faked.
    """
    results: Dict[str, Dict[str, Optional[str]]] = {}

    if not urls:
        return results
    if not AIOHTTP_AVAILABLE:
        for u in urls:
            results[u] = {"content": None, "error": "aiohttp not installed"}
        return results

    # De-dup while preserving order.
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    config = ScraperConfig(max_concurrent=max(1, int(max_concurrent)))
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def _one(scraper: EthicalWebScraper, url: str):
        async with semaphore:
            try:
                data = await scraper.scrape_page(url)
                if data and data.get("text"):
                    return url, {"content": data["text"], "error": None}
                return url, {"content": None, "error": "no content extracted"}
            except Exception as e:
                return url, {"content": None, "error": str(e)}

    try:
        async with EthicalWebScraper(config) as scraper:
            tasks = [_one(scraper, u) for u in unique_urls]
            gathered = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=config.total_timeout,
            )
        for item in gathered:
            if isinstance(item, Exception):
                logger.warning("scrape task exception: %s", item)
                continue
            url, payload = item
            results[url] = payload
    except asyncio.TimeoutError:
        logger.error("scrape_urls: total timeout (%ss) exceeded.", config.total_timeout)
    except Exception as e:
        logger.error("scrape_urls failed: %s", e)

    # Ensure every requested URL has an entry (honest reporting).
    for u in unique_urls:
        results.setdefault(u, {"content": None, "error": "not processed (timeout/abort)"})
    return results


async def scrape_urls_from_context(additional_context: str, max_concurrent: int = 3) -> str:
    """Find URLs in free text, scrape them, and inline the content.

    Each URL in ``additional_context`` is replaced with a
    ``[Content from <url>]: ... [End of content]`` block (truncated to ~1000
    words). On total failure the original context is returned unchanged.
    """
    if not additional_context:
        return additional_context

    urls = _URL_PATTERN.findall(additional_context)
    if not urls:
        return additional_context

    scraped = await scrape_urls(urls, max_concurrent=max_concurrent)

    enhanced = additional_context
    for url in urls:
        payload = scraped.get(url)
        if not payload:
            continue
        if payload.get("content"):
            text = payload["content"]
            words = text.split()
            if len(words) > 1000:
                text = " ".join(words[:1000]) + " [truncated due to length]"
            marker = f"[Content from {url}]:\n{text}\n[End of content]"
        else:
            err = payload.get("error") or "no content"
            marker = f"[No content available from {url}: {err}]"
        enhanced = enhanced.replace(url, marker)

    return enhanced


# ===========================================================================
# Bounded history helpers
# ===========================================================================

def push_history(key: str, item) -> None:
    """Append ``item`` to the session-state list at ``key``, capped at MAX_HISTORY.

    Oldest entries are dropped first (FIFO eviction). No-op if streamlit/
    session_state is unavailable.
    """
    if not STREAMLIT_AVAILABLE or st is None:
        return
    try:
        lst = st.session_state.get(key)
        if not isinstance(lst, list):
            lst = []
        lst.append(item)
        if len(lst) > MAX_HISTORY:
            lst = lst[-MAX_HISTORY:]
        st.session_state[key] = lst
    except Exception as e:
        logger.debug("push_history failed for %s: %s", key, e)


def get_history(key: str) -> list:
    """Return the session-state history list at ``key`` (empty list if absent)."""
    if not STREAMLIT_AVAILABLE or st is None:
        return []
    try:
        lst = st.session_state.get(key)
        return lst if isinstance(lst, list) else []
    except Exception:
        return []


__all__ = [
    "get_secret",
    "feature_available",
    "GEMINI_MODEL",
    "GEMINI_PRICING",
    "get_gemini_client",
    "get_linkup_client",
    "gemini_generate_json",
    "gemini_generate_text",
    "run_async",
    "get_async_runner",
    "AsyncTaskRunner",
    "TokenLedger",
    "record_tokens",
    "render_token_usage",
    "ScraperConfig",
    "EthicalWebScraper",
    "scrape_urls_from_context",
    "scrape_urls",
    "MAX_HISTORY",
    "push_history",
    "get_history",
]
