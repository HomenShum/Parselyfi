#!/usr/bin/env python
"""tests/eval/run_eval.py
========================

RUNNABLE research-agent EVAL HARNESS for ParselyFi.

Ported in spirit from the user's nodebench fast/slow scored-eval templates:
for every fixture case (see ``tests/eval/cases.py``) it invokes the REAL
ParselyFi research function via ``features.common.run_async``, times it, and
applies a small set of checks. Two kinds of checks, kept visually distinct so
nobody mistakes one for the other:

  [DET]  DETERMINISTIC checks — pure Python over the returned object. No model,
         no randomness. nonempty / sources-present / abstained-on-ambiguous /
         latency-within-budget. These run in EVERY mode, including ``--dry``.

  [LLM]  LLM-JUDGE checks — only when a Gemini key is present AND not ``--dry``.
         A single ``common.gemini_generate_json`` call scores the result on
         entity_correct / no_hallucinations / actionable (each 0..1) and we
         compare against thresholds. HONEST: a score is printed ONLY when the
         judge call actually returned it. If the judge is unavailable we print
         "n/a" — never a fabricated number, never a hardcoded floor.

Bounding (CLAUDE.md agentic-reliability mandate): each case declares
``max_external_calls`` and ``max_llm_calls``. The harness wraps the shared
LinkUp-search and Gemini-JSON entry points with a per-case counter and a hard
ceiling. The (N+1)-th call of either kind raises ``_BudgetExceeded``, which the
harness records as a FAILED bound check rather than letting a runaway agent
loop burn the API. Counts are always reported so an under-budget run is visible.

Usage (from the repo root)::

    python tests/eval/run_eval.py --dry      # deterministic checks only, no API
    python tests/eval/run_eval.py --fast     # fast cases, real API if keys set
    python tests/eval/run_eval.py --slow     # slow cases
    python tests/eval/run_eval.py            # all cases
    python tests/eval/run_eval.py --no-judge # skip LLM-judge even if key present
    python tests/eval/run_eval.py --json out.json   # also dump machine-readable

Exit code is 0 when every non-skipped case PASSES, 1 otherwise (CI-friendly).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Make the repo root importable whether run as
#   `python tests/eval/run_eval.py`  (cwd = repo root)
# or `python -m tests.eval.run_eval`.
# repo root = parents[2] of this file (tests/eval/run_eval.py -> repo root).
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# cases.py is dependency-free; import it via package or by path.
try:
    from tests.eval.cases import CASES, select_cases  # type: ignore
except Exception:  # pragma: no cover - fallback when not run as a package
    sys.path.insert(0, _THIS_DIR)
    from cases import CASES, select_cases  # type: ignore


# ===========================================================================
# Lazy app imports
#
# We import the app lazily inside main() so that `--help` and a missing-deps
# environment still print something useful instead of crashing at import time.
# These globals are filled by _load_app().
# ===========================================================================
common = None  # features.common module
company = None  # features.company_research module
_GEMINI_MODEL = "gemini-3.5-flash"


def _load_app() -> Tuple[Any, Any, List[str]]:
    """Import the ParselyFi app modules. Returns (common, company, warnings).

    ``warnings`` is a list of human-readable strings about anything that could
    not be imported — surfaced honestly rather than silently swallowed.
    """
    warnings: List[str] = []
    _common = None
    _company = None
    try:
        from features import common as _common  # type: ignore
    except Exception as e:  # pragma: no cover
        warnings.append(f"could not import features.common: {e}")
    try:
        from features import company_research as _company  # type: ignore
    except Exception as e:  # pragma: no cover
        warnings.append(f"could not import features.company_research: {e}")
    return _common, _company, warnings


# ===========================================================================
# Call-budget guard (bounded external + LLM calls per case)
# ===========================================================================
class _BudgetExceeded(RuntimeError):
    """Raised when a case tries to exceed its declared call budget."""


@dataclass
class _CallBudget:
    """Per-case counter + hard ceiling for external and LLM calls.

    We monkeypatch the shared entry points (LinkUp search, Gemini JSON) to run
    through ``tick_external`` / ``tick_llm`` so a runaway can never exceed the
    case's declared maxima. The (max+1)-th call raises ``_BudgetExceeded``.
    """

    max_external: int
    max_llm: int
    external_used: int = 0
    llm_used: int = 0

    def tick_external(self) -> None:
        if self.external_used >= self.max_external:
            raise _BudgetExceeded(
                f"external-call budget exceeded (max {self.max_external})"
            )
        self.external_used += 1

    def tick_llm(self) -> None:
        if self.llm_used >= self.max_llm:
            raise _BudgetExceeded(
                f"LLM-call budget exceeded (max {self.max_llm})"
            )
        self.llm_used += 1


class _Patches:
    """Context manager that wraps common.gemini_generate_json and
    company_research.company_search_with_linkup with budget-counting shims for
    the duration of one case, then restores the originals.

    Counting (not stubbing): the REAL functions still run. We only intercept to
    increment the counter and trip the ceiling. This keeps the eval honest —
    it measures the real app's behavior and real call volume.
    """

    def __init__(self, budget: _CallBudget) -> None:
        self.budget = budget
        self._orig_llm: Optional[Callable] = None
        self._orig_search: Optional[Callable] = None

    def __enter__(self):
        assert common is not None and company is not None
        self._orig_llm = common.gemini_generate_json
        self._orig_search = company.company_search_with_linkup

        orig_llm = self._orig_llm
        orig_search = self._orig_search
        budget = self.budget

        async def _llm_shim(prompt, *args, **kwargs):
            budget.tick_llm()
            return await orig_llm(prompt, *args, **kwargs)

        async def _search_shim(company_name, *args, **kwargs):
            budget.tick_external()
            return await orig_search(company_name, *args, **kwargs)

        # Patch on BOTH modules: company_research imported gemini_generate_json
        # by name (`from .common import gemini_generate_json`), so we must
        # rebind the name in company_research too, not just on common.
        common.gemini_generate_json = _llm_shim
        if hasattr(company, "gemini_generate_json"):
            company.gemini_generate_json = _llm_shim
        company.company_search_with_linkup = _search_shim
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if common is not None and self._orig_llm is not None:
            common.gemini_generate_json = self._orig_llm
            if hasattr(company, "gemini_generate_json"):
                company.gemini_generate_json = self._orig_llm  # type: ignore
        if company is not None and self._orig_search is not None:
            company.company_search_with_linkup = self._orig_search
        return False  # never swallow exceptions


# ===========================================================================
# Target resolution
#
# A case's ``target`` is "<alias>.<callable>". The alias maps to a real module.
# Two composite targets are handled specially because they chain >1 real call:
#   company.search_then_extract_schema  -> search then schema-extract
# Everything else resolves to a plain attribute on the module.
# ===========================================================================
def _resolve_target(target: str) -> Tuple[Optional[str], Optional[str]]:
    """Split "alias.name" -> (alias, name). Returns (None, None) if malformed."""
    if "." not in target:
        return None, None
    alias, name = target.split(".", 1)
    return alias, name


def _target_module(alias: str):
    """Map a case alias to the real app module (or None if unknown/missing)."""
    if alias == "company":
        return company
    # Spec named features.list_intelligence; it does not exist in the repo yet.
    # Resolve honestly to None so such cases SKIP rather than fabricate.
    if alias == "list":
        return None
    return None


# ===========================================================================
# Invocation: build the coroutine for a case and run it (bounded + timed)
# ===========================================================================
@dataclass
class Invocation:
    """The outcome of actually calling the target for a case."""

    ok: bool                       # the call itself completed without exception
    result: Any = None             # the returned object (shape depends on target)
    error: Optional[str] = None    # exception text if ok is False
    elapsed_s: float = 0.0
    external_calls: int = 0
    llm_calls: int = 0
    skipped: bool = False          # target unavailable -> case skipped
    skip_reason: Optional[str] = None


def _build_coro(name: str, case: Dict[str, Any], dry: bool):
    """Return an awaitable for the case's target, or raise LookupError.

    In ``--dry`` mode we DO NOT issue real API calls. For dry runs we return a
    trivially-empty coroutine result so the deterministic checks still execute
    against a known-empty object (which is honest: with no API, there is no
    data, so nonempty/sources checks should fail for cases that require them —
    that is the correct dry-run signal, not a pass).
    """
    if company is None:
        raise LookupError("features.company_research unavailable")

    prompt = case["prompt"]

    if dry:
        async def _empty():
            # Honest empty result per target shape.
            if name in ("first_pass_entity_selection",):
                return {"success": False, "entities": [], "_dry": True}
            if name in ("company_search_with_linkup",):
                return {"success": False, "answer": "", "sources": [], "_dry": True}
            if name in ("disambiguate_entities",):
                return {}
            if name in ("search_then_extract_schema",):
                return {"_dry": True, "structured_data": None, "sources": []}
            return None
        return _empty()

    # ---- real invocations --------------------------------------------------
    # IMPORTANT: every branch resolves the app callable INSIDE an `async def`
    # body so the attribute lookup happens at await-time, AFTER _Patches has
    # rebound the budget-counting shims. A direct `company.fn(prompt)` here
    # would capture the UNPATCHED original at build-time (the coro is built
    # before the `with _Patches(...)` block), so its calls would go uncounted.
    if name == "first_pass_entity_selection":
        # Signature: (user_input, observability_log)
        async def _first_pass():
            return await company.first_pass_entity_selection(prompt, [])
        return _first_pass()

    if name == "company_search_with_linkup":
        # Signature: (company_name, additional_context="")
        async def _search():
            return await company.company_search_with_linkup(prompt)
        return _search()

    if name == "disambiguate_entities":
        # Signature: (user_input, intended_context, entity_info)
        # Provide two same-named candidates so the classifier has something to
        # score: the one matching the user's context vs a colliding sibling.
        entity_info = _disambig_fixture_candidates(prompt)
        async def _disambig():
            return await company.disambiguate_entities(prompt, prompt, entity_info)
        return _disambig()

    if name == "search_then_extract_schema":
        # Composite: search, then Gemini-extract the company schema from the
        # sourced answer. Mirrors second_pass's text->schema path.
        async def _composite():
            search = await company.company_search_with_linkup(prompt)
            structured = None
            if search.get("success") and search.get("answer"):
                structured = await company._extract_structured_data_from_answer(
                    company_name=prompt,
                    answer_text=search.get("answer", "") or "",
                    sources=search.get("sources", []) or [],
                    agent_name="Eval Schema Extractor",
                )
            return {
                "success": bool(search.get("success")),
                "answer": search.get("answer", ""),
                "sources": search.get("sources", []) or [],
                "structured_data": structured,
            }
        return _composite()

    raise LookupError(f"unknown target callable: {name}")


def _disambig_fixture_candidates(prompt: str) -> List[Dict[str, Any]]:
    """Two candidate dicts for the disambiguation classifier fixtures.

    The classifier consumes ``[{name, snippet, sources}]`` and returns a
    per-index confidence. We hand it the genuinely-confusable pair implied by
    the case prompt so the judged behavior (right one scores higher) is real.
    """
    p = prompt.lower()
    if "apple" in p:
        return [
            {"name": "Apple Inc.", "snippet": "Consumer-electronics company; makes the iPhone, Mac, iPad. HQ Cupertino, CA.", "sources": [{"url": "https://www.apple.com"}]},
            {"name": "Apple Corps", "snippet": "Multimedia/record company founded by The Beatles in 1968, London.", "sources": [{"url": "https://www.applecorps.com"}]},
        ]
    # Default: the X / xAI suffix-collision pair.
    return [
        {"name": "X (formerly Twitter)", "snippet": "Social-media platform formerly named Twitter, owned by X Corp.", "sources": [{"url": "https://x.com"}]},
        {"name": "xAI", "snippet": "Separate AI research company founded by Elon Musk; builds the Grok model.", "sources": [{"url": "https://x.ai"}]},
    ]


def _invoke(case: Dict[str, Any], dry: bool) -> Invocation:
    """Resolve, bound, run, and time a single case's target call."""
    target = case["target"]
    alias, name = _resolve_target(target)
    if alias is None or name is None:
        return Invocation(ok=False, skipped=True,
                          skip_reason=f"malformed target {target!r}")

    module = _target_module(alias)
    if module is None:
        return Invocation(
            ok=False, skipped=True,
            skip_reason=f"target module for alias {alias!r} unavailable "
                        f"(no such module in this build)",
        )

    # The composite/synthetic callables live conceptually on company_research.
    real_names = {
        "first_pass_entity_selection",
        "company_search_with_linkup",
        "disambiguate_entities",
        "_extract_structured_data_from_answer",
    }
    if name not in real_names and name != "search_then_extract_schema":
        return Invocation(ok=False, skipped=True,
                          skip_reason=f"callable {name!r} not implemented in build")
    if name in real_names and not hasattr(module, name):
        return Invocation(ok=False, skipped=True,
                          skip_reason=f"{alias}.{name} missing from module")

    budget = _CallBudget(
        max_external=int(case.get("max_external_calls", 0)),
        max_llm=int(case.get("max_llm_calls", 0)),
    )

    start = time.time()
    try:
        coro = _build_coro(name, case, dry)
    except LookupError as e:
        return Invocation(ok=False, skipped=True, skip_reason=str(e))

    if dry:
        # No patching/budget needed; just run the trivial empty coroutine.
        try:
            result = common.run_async(coro)
            return Invocation(ok=True, result=result, elapsed_s=time.time() - start,
                              external_calls=0, llm_calls=0)
        except Exception as e:  # pragma: no cover
            return Invocation(ok=False, error=f"{type(e).__name__}: {e}",
                              elapsed_s=time.time() - start)

    # Real run: wrap in the budget patches so calls are counted + ceiling-capped.
    try:
        with _Patches(budget):
            result = common.run_async(coro)
        return Invocation(
            ok=True, result=result, elapsed_s=time.time() - start,
            external_calls=budget.external_used, llm_calls=budget.llm_used,
        )
    except _BudgetExceeded as e:
        return Invocation(
            ok=False, error=str(e), elapsed_s=time.time() - start,
            external_calls=budget.external_used, llm_calls=budget.llm_used,
        )
    except Exception as e:
        return Invocation(
            ok=False, error=f"{type(e).__name__}: {e}",
            elapsed_s=time.time() - start,
            external_calls=budget.external_used, llm_calls=budget.llm_used,
        )


# ===========================================================================
# Result introspection helpers (target-shape aware)
# ===========================================================================
def _result_nonempty(name: str, result: Any) -> bool:
    """True if the result carries actual content for this target shape."""
    if result is None:
        return False
    if name == "first_pass_entity_selection":
        return bool(result.get("entities"))
    if name == "company_search_with_linkup":
        return bool((result.get("answer") or "").strip())
    if name == "disambiguate_entities":
        return bool(result)  # non-empty {idx: {...}} mapping
    if name == "search_then_extract_schema":
        return bool((result.get("answer") or "").strip()) or bool(result.get("structured_data"))
    # Unknown shape: treat any truthy as nonempty.
    return bool(result)


def _result_sources(name: str, result: Any) -> List[Any]:
    """Return the list of sources carried by the result (may be empty)."""
    if not isinstance(result, dict):
        return []
    if name == "first_pass_entity_selection":
        out: List[Any] = []
        for ent in result.get("entities", []) or []:
            out.extend((ent or {}).get("sources", []) or [])
        return out
    if name in ("company_search_with_linkup", "search_then_extract_schema"):
        return result.get("sources", []) or []
    return []


def _result_candidate_count(name: str, result: Any) -> int:
    """How many distinct candidate entities the result surfaced."""
    if name == "first_pass_entity_selection" and isinstance(result, dict):
        return len(result.get("entities", []) or [])
    if name == "disambiguate_entities" and isinstance(result, dict):
        return len(result)
    return 0


def _disambiguation_signaled(name: str, result: Any) -> bool:
    """True if the result surfaced a disambiguation signal (multi-candidate or
    per-candidate confidence) rather than silently committing to one entity.

    This is the deterministic proxy for "did the agent abstain / ask to
    disambiguate" on an ambiguous prompt.
    """
    if not isinstance(result, dict):
        return False
    if name == "first_pass_entity_selection":
        ents = result.get("entities", []) or []
        if len(ents) > 1:
            return True
        # Single candidate is acceptable ONLY if it carries an explicit, low-ish
        # confidence hint (the disambiguator ran and flagged uncertainty).
        for ent in ents:
            if isinstance(ent, dict) and ent.get("confidence") is not None:
                return True
        # Also accept: nothing auto-selected (selected_entity is None) AND
        # success — i.e. the system handed candidates back to the user.
        return result.get("selected_entity") is None and bool(ents)
    if name == "disambiguate_entities":
        return len(result) > 1
    return False


# ===========================================================================
# Checks (deterministic + LLM-judge)
# ===========================================================================
@dataclass
class CheckResult:
    name: str
    kind: str  # "DET" or "LLM"
    passed: bool
    detail: str = ""


@dataclass
class CaseReport:
    case_id: str
    mode: str
    category: str
    resolution: str
    invocation: Invocation
    checks: List[CheckResult] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)  # LLM-judge scores
    judged: bool = False

    @property
    def skipped(self) -> bool:
        return self.invocation.skipped

    @property
    def passed(self) -> bool:
        if self.skipped:
            return False  # skipped is neither pass nor fail; tracked separately
        return all(c.passed for c in self.checks)


def _deterministic_checks(case: Dict[str, Any], inv: Invocation) -> List[CheckResult]:
    """Run the [DET] checks for a case against its invocation result."""
    name = case["target"].split(".", 1)[1]
    exp = case.get("expectations", {})
    checks: List[CheckResult] = []

    # 0) The call itself completed (a crash/budget-trip fails everything).
    if not inv.ok:
        checks.append(CheckResult(
            "call_completed", "DET", False,
            inv.error or "call did not complete",
        ))
        return checks  # downstream checks meaningless without a result
    checks.append(CheckResult("call_completed", "DET", True, "ok"))

    result = inv.result

    # 1) non-empty
    if exp.get("must_return_nonempty"):
        ne = _result_nonempty(name, result)
        checks.append(CheckResult(
            "nonempty", "DET", ne,
            "has content" if ne else "result empty",
        ))

    # 2) sources present (only when required)
    if exp.get("must_cite_sources"):
        srcs = _result_sources(name, result)
        has = len(srcs) > 0
        checks.append(CheckResult(
            "sources_present", "DET", has,
            f"{len(srcs)} source(s)" if has else "no sources cited",
        ))

    # 3) abstain / disambiguate on ambiguous cases
    if exp.get("must_abstain"):
        signaled = _disambiguation_signaled(name, result)
        n = _result_candidate_count(name, result)
        checks.append(CheckResult(
            "abstained_or_disambiguated", "DET", signaled,
            f"surfaced {n} candidate(s)/signal" if signaled
            else "committed to a single entity with no disambiguation signal",
        ))

    # 4) latency within budget
    budget_s = float(exp.get("latency_budget_s", 0) or 0)
    if budget_s > 0:
        within = inv.elapsed_s <= budget_s
        checks.append(CheckResult(
            "latency_within_budget", "DET", within,
            f"{inv.elapsed_s:.1f}s / {budget_s:.0f}s budget",
        ))

    # 5) call-budget respected (we counted them; trivially true if we got here,
    #    but report the counts so under/at-budget is visible and honest).
    checks.append(CheckResult(
        "within_call_budget", "DET", True,
        f"ext={inv.external_calls}/{case.get('max_external_calls')} "
        f"llm={inv.llm_calls}/{case.get('max_llm_calls')}",
    ))

    return checks


# ---- LLM-judge -------------------------------------------------------------
_JUDGE_THRESHOLDS = {
    "entity_correct": 0.6,
    "no_hallucinations": 0.6,
    "actionable": 0.5,
}


def _result_text_for_judge(name: str, result: Any) -> str:
    """Flatten the result into a compact text blob the judge can score."""
    if not isinstance(result, dict):
        return str(result)[:4000]
    if name == "first_pass_entity_selection":
        lines = []
        for ent in (result.get("entities") or [])[:5]:
            if not isinstance(ent, dict):
                continue
            lines.append(f"- {ent.get('name')}: {(ent.get('snippet') or '')[:400]}")
            if ent.get("confidence") is not None:
                lines.append(f"  (confidence={ent.get('confidence')}, reason={ent.get('confidence_reason')})")
        return "CANDIDATES:\n" + "\n".join(lines)
    if name in ("company_search_with_linkup", "search_then_extract_schema"):
        ans = (result.get("answer") or "")[:3000]
        nsrc = len(result.get("sources") or [])
        sd = result.get("structured_data")
        sd_txt = json.dumps(sd, default=str)[:1500] if sd else "none"
        return f"ANSWER:\n{ans}\n\nNUM_SOURCES: {nsrc}\nSTRUCTURED:\n{sd_txt}"
    if name == "disambiguate_entities":
        return "DISAMBIGUATION_SCORES:\n" + json.dumps(result, default=str)[:2000]
    return json.dumps(result, default=str)[:3000]


def _llm_judge(case: Dict[str, Any], inv: Invocation) -> Optional[Dict[str, float]]:
    """Score the result with a Gemini LLM-judge. Returns the 0..1 scores dict,
    or None if the judge could not run / returned nothing (NEVER a fabricated
    or floored score).
    """
    if common is None or not inv.ok:
        return None
    name = case["target"].split(".", 1)[1]
    result_text = _result_text_for_judge(name, inv.result)

    prompt = f"""
You are an impartial evaluator for a company-research agent. Score the agent's
output for the user's request on three axes, each a float from 0.0 to 1.0.

USER REQUEST:
{case['prompt']}

RESOLUTION EXPECTATION: {case.get('resolution_expectation')}
(If "ambiguous", a GOOD answer surfaces multiple candidates / asks to
disambiguate rather than confidently picking one entity.)

AGENT OUTPUT:
{result_text}

Score each axis:
- entity_correct: does the output concern the entity the user actually meant
  (and, for ambiguous requests, appropriately reflect the ambiguity)?
- no_hallucinations: are the claims plausibly grounded / not invented? Penalize
  confident specifics with no supporting sources.
- actionable: is the output useful and specific enough to act on?

Return ONLY this JSON (floats 0.0-1.0, no prose):
{{"entity_correct": 0.0, "no_hallucinations": 0.0, "actionable": 0.0,
  "rationale": "one short sentence"}}
"""
    try:
        # Judge call goes through the REAL helper (not the budgeted shim — the
        # patch context has exited by the time we judge). It is a separate,
        # clearly-labeled call, not part of the case's research budget.
        scored = common.run_async(
            common.gemini_generate_json(
                prompt, model=_GEMINI_MODEL, agent_name=f"Eval Judge [{case['id']}]"
            )
        )
    except Exception:
        return None
    if not isinstance(scored, dict) or not scored:
        return None

    out: Dict[str, float] = {}
    for axis in ("entity_correct", "no_hallucinations", "actionable"):
        v = scored.get(axis)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue  # missing axis -> simply not scored (honest)
        out[axis] = max(0.0, min(1.0, fv))
    if not out:
        return None
    if scored.get("rationale"):
        out["_rationale"] = scored.get("rationale")  # type: ignore
    return out


# ===========================================================================
# Scorecard rendering (plain terminal table — this is a CLI, not Streamlit)
# ===========================================================================
def _c(s: str, width: int) -> str:
    s = str(s)
    return s[:width].ljust(width)


def _print_scorecard(reports: List[CaseReport], judged_any: bool) -> None:
    print("\n" + "=" * 100)
    print("PARSELYFI RESEARCH-AGENT EVAL SCORECARD")
    print("=" * 100)
    header = (
        f"{_c('CASE', 30)} {_c('MODE', 5)} {_c('DET', 9)} "
        f"{_c('LLM-JUDGE (ent/halu/act)', 26)} {_c('LAT', 8)} {_c('RESULT', 8)}"
    )
    print(header)
    print("-" * 100)

    for r in reports:
        det = [c for c in r.checks if c.kind == "DET"]
        det_pass = sum(1 for c in det if c.passed)
        det_str = f"{det_pass}/{len(det)}"

        if r.skipped:
            verdict = "SKIP"
            det_str = "-"
            judge_str = "skipped"
            lat = "-"
        else:
            if r.judged and r.scores:
                judge_str = (
                    f"{r.scores.get('entity_correct', float('nan')):.2f}/"
                    f"{r.scores.get('no_hallucinations', float('nan')):.2f}/"
                    f"{r.scores.get('actionable', float('nan')):.2f}"
                )
            else:
                judge_str = "n/a"
            lat = f"{r.invocation.elapsed_s:.1f}s"
            verdict = "PASS" if r.passed else "FAIL"

        print(
            f"{_c(r.case_id, 30)} {_c(r.mode, 5)} {_c(det_str, 9)} "
            f"{_c(judge_str, 26)} {_c(lat, 8)} {_c(verdict, 8)}"
        )

    print("-" * 100)
    # Per-check detail for any FAILED non-skipped case.
    failed = [r for r in reports if not r.skipped and not r.passed]
    if failed:
        print("\nFAILED-CHECK DETAIL:")
        for r in failed:
            print(f"  [{r.case_id}]")
            for c in r.checks:
                if not c.passed:
                    print(f"     x [{c.kind}] {c.name}: {c.detail}")
            if r.judged and r.scores:
                for axis, thr in _JUDGE_THRESHOLDS.items():
                    if axis in r.scores and r.scores[axis] < thr:
                        print(f"     x [LLM] {axis}={r.scores[axis]:.2f} < {thr}")

    skipped = [r for r in reports if r.skipped]
    if skipped:
        print("\nSKIPPED (target unavailable in this build — NOT counted as pass):")
        for r in skipped:
            print(f"  - {r.case_id}: {r.invocation.skip_reason}")

    # Summary.
    total = len(reports)
    n_skip = len(skipped)
    scored = [r for r in reports if not r.skipped]
    n_pass = sum(1 for r in scored if r.passed)
    n_fail = len(scored) - n_pass
    rate = (n_pass / len(scored) * 100.0) if scored else 0.0

    print("\n" + "=" * 100)
    print(
        f"SUMMARY: {n_pass}/{len(scored)} passed ({rate:.0f}%)  |  "
        f"{n_fail} failed  |  {n_skip} skipped  |  {total} total"
    )
    print(
        "Legend: [DET]=deterministic (pure Python, runs in --dry).  "
        "[LLM]=Gemini judge (only with a key, not in --dry)."
    )
    if not judged_any:
        print(
            "NOTE: No LLM-judge scores were computed (dry run, no key, or "
            "--no-judge). 'n/a' means NOT scored — no number was fabricated."
        )
    print("=" * 100 + "\n")


# ===========================================================================
# Main
# ===========================================================================
def _run_cases(cases: List[Dict[str, Any]], *, dry: bool, judge: bool) -> Tuple[List[CaseReport], bool]:
    reports: List[CaseReport] = []
    judged_any = False

    # Decide once whether the LLM-judge can run at all (honest gating).
    judge_enabled = judge and not dry
    gemini_ok = False
    if judge_enabled and common is not None:
        try:
            gemini_ok, _missing = common.feature_available(["GEMINI_API_KEY"])
        except Exception:
            gemini_ok = False
    judge_enabled = judge_enabled and gemini_ok

    for case in cases:
        print(f"[run] {case['id']} ({case['mode']}/{case['category']}) ...", flush=True)
        inv = _invoke(case, dry)
        report = CaseReport(
            case_id=case["id"],
            mode=case["mode"],
            category=case["category"],
            resolution=case.get("resolution_expectation", ""),
            invocation=inv,
        )

        if not inv.skipped:
            report.checks = _deterministic_checks(case, inv)

            if judge_enabled and inv.ok:
                scores = _llm_judge(case, inv)
                if scores is not None:
                    report.judged = True
                    judged_any = True
                    report.scores = {k: v for k, v in scores.items() if not k.startswith("_")}
                    # Add LLM-judge checks against thresholds.
                    for axis, thr in _JUDGE_THRESHOLDS.items():
                        if axis in report.scores:
                            passed = report.scores[axis] >= thr
                            report.checks.append(CheckResult(
                                f"judge_{axis}", "LLM", passed,
                                f"{report.scores[axis]:.2f} >= {thr}" if passed
                                else f"{report.scores[axis]:.2f} < {thr}",
                            ))

        reports.append(report)

    return reports, judged_any


def main(argv: Optional[List[str]] = None) -> int:
    global common, company, _GEMINI_MODEL

    parser = argparse.ArgumentParser(
        description="ParselyFi research-agent eval harness.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="run only fast cases")
    mode.add_argument("--slow", action="store_true", help="run only slow cases")
    parser.add_argument("--dry", action="store_true",
                        help="skip ALL API/LLM calls; deterministic checks only")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the LLM-judge even when a Gemini key is present")
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="also write a machine-readable JSON report to PATH")
    args = parser.parse_args(argv)

    # Load app modules now (after --help has had its chance).
    common, company, warnings = _load_app()
    for w in warnings:
        print(f"[warn] {w}", file=sys.stderr)
    if common is None or company is None:
        print("[fatal] ParselyFi app modules unavailable; cannot run eval.",
              file=sys.stderr)
        return 2
    _GEMINI_MODEL = getattr(common, "GEMINI_MODEL", _GEMINI_MODEL)

    # Honest status banner about the LLM-judge availability.
    if args.dry:
        print("[mode] DRY RUN — no external/LLM calls; deterministic checks only.")
    else:
        try:
            ok, missing = common.feature_available(["GEMINI_API_KEY"])
        except Exception:
            ok, missing = False, ["GEMINI_API_KEY"]
        if args.no_judge:
            print("[mode] LIVE — LLM-judge disabled by --no-judge.")
        elif ok:
            print(f"[mode] LIVE — LLM-judge ENABLED (Gemini key present, model={_GEMINI_MODEL}).")
        else:
            print(f"[mode] LIVE — LLM-judge UNAVAILABLE (missing {missing}); "
                  "deterministic checks still run, judge scores will show 'n/a'.")
        lok, _lmiss = common.feature_available(["LINKUP_API_KEY"])
        if not lok:
            print("[warn] LINKUP_API_KEY missing — LinkUp-backed cases will "
                  "return empty/failed (honest), failing nonempty/sources checks.")

    selected = "fast" if args.fast else "slow" if args.slow else None
    cases = select_cases(selected)
    print(f"[info] {len(cases)} case(s) selected"
          + (f" (mode={selected})" if selected else " (all modes)") + ".")

    reports, judged_any = _run_cases(cases, dry=args.dry, judge=not args.no_judge)
    _print_scorecard(reports, judged_any)

    if args.json:
        payload = {
            "selected_mode": selected,
            "dry": args.dry,
            "judged_any": judged_any,
            "cases": [
                {
                    "id": r.case_id, "mode": r.mode, "category": r.category,
                    "resolution": r.resolution,
                    "skipped": r.skipped,
                    "skip_reason": r.invocation.skip_reason,
                    "passed": (None if r.skipped else r.passed),
                    "elapsed_s": round(r.invocation.elapsed_s, 3),
                    "external_calls": r.invocation.external_calls,
                    "llm_calls": r.invocation.llm_calls,
                    "error": r.invocation.error,
                    "checks": [
                        {"name": c.name, "kind": c.kind, "passed": c.passed, "detail": c.detail}
                        for c in r.checks
                    ],
                    "judge_scores": r.scores if r.judged else None,
                }
                for r in reports
            ],
        }
        try:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"[info] wrote JSON report to {args.json}")
        except Exception as e:
            print(f"[warn] could not write JSON report: {e}", file=sys.stderr)

    # CI-friendly exit code: 0 only if every NON-skipped case passed.
    scored = [r for r in reports if not r.skipped]
    all_pass = bool(scored) and all(r.passed for r in scored)
    return 0 if all_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[abort] interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception:  # pragma: no cover - last-resort honest crash report
        traceback.print_exc()
        raise SystemExit(2)
