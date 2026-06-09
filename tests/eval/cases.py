"""tests/eval/cases.py
=====================

FIXTURE cases for the ParselyFi research-agent eval harness.

Each case is a plain ``dict`` (kept dependency-free on purpose so this file
imports instantly and can be edited by non-engineers). The runner
(``tests/eval/run_eval.py``) reads these and invokes the REAL ParselyFi
research function named by ``target`` against the live app code.

Ported in spirit from the user's nodebench fast/slow scored-eval templates:
a mix of cheap "fast" probes and heavier "slow" multi-call research runs,
spread across categories, each with an explicit resolution expectation and a
small set of checks.

------------------------------------------------------------------------------
CASE SCHEMA (every key is required unless marked optional)
------------------------------------------------------------------------------
id                      str   stable, unique slug for the case
mode                    str   "fast" | "slow"  (selects with --fast / --slow)
category                str   "entity" | "people" | "product" | "classify" | "ambiguous"
prompt                  str   the user input handed to the target function
target                  str   "<module_alias>.<callable>" to invoke in the app.
                              module_alias is one of the keys in TARGETS below
                              (e.g. "company.first_pass_entity_selection").
max_external_calls      int   hard cap on external (LinkUp/scrape) calls this
                              case may make. The runner BOUNDS the run to this.
max_llm_calls           int   hard cap on Gemini calls this case may make.
resolution_expectation  str   "exact" | "probable" | "ambiguous"
expectations            dict  the deterministic check switches:
    must_return_nonempty bool  result must be non-empty (entities/answer/dict)
    must_cite_sources    bool  result must carry >=1 source when required
    must_abstain         bool  ambiguous cases: must surface >1 candidate OR a
                               disambiguation signal rather than silently
                               committing to one entity
    latency_budget_s     float wall-clock budget; over budget fails the latency
                               check (still runs, just flagged)

The ``target`` strings reference ONLY callables that actually exist in the
repo today. The spec also named a ``features.list_intelligence`` module; that
module does not exist yet, so we do NOT invent cases against it. When/if it
lands, add its alias to ``TARGETS`` in run_eval.py and append cases here.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# The fixture cases (~12). Ordered fast-first so a --fast run reads top-down.
# ---------------------------------------------------------------------------
CASES: List[Dict[str, Any]] = [
    # ---- FAST: cheap, single-purpose probes -------------------------------
    {
        "id": "fast_entity_anthropic",
        "mode": "fast",
        "category": "entity",
        "prompt": "Anthropic, the AI safety company founded by the Amodei siblings.",
        "target": "company.first_pass_entity_selection",
        "max_external_calls": 2,
        "max_llm_calls": 3,
        "resolution_expectation": "exact",
        "expectations": {
            "must_return_nonempty": True,
            "must_cite_sources": True,
            "must_abstain": False,
            "latency_budget_s": 45.0,
        },
    },
    {
        "id": "fast_entity_stripe",
        "mode": "fast",
        "category": "entity",
        "prompt": "Stripe (the payments infrastructure company, HQ in San Francisco).",
        "target": "company.first_pass_entity_selection",
        "max_external_calls": 2,
        "max_llm_calls": 3,
        "resolution_expectation": "exact",
        "expectations": {
            "must_return_nonempty": True,
            "must_cite_sources": True,
            "must_abstain": False,
            "latency_budget_s": 45.0,
        },
    },
    {
        "id": "fast_search_databricks",
        "mode": "fast",
        "category": "entity",
        "prompt": "Databricks",
        "target": "company.company_search_with_linkup",
        "max_external_calls": 1,
        "max_llm_calls": 0,
        "resolution_expectation": "exact",
        "expectations": {
            "must_return_nonempty": True,
            "must_cite_sources": True,
            "must_abstain": False,
            "latency_budget_s": 40.0,
        },
    },
    {
        "id": "fast_ambiguous_jaguar",
        "mode": "fast",
        "category": "ambiguous",
        # "Jaguar" -> the car maker vs the animal vs Atari Jaguar vs Fender
        # Jaguar guitar. A correct system must NOT silently commit to one.
        "prompt": "Tell me about Jaguar.",
        "target": "company.first_pass_entity_selection",
        "max_external_calls": 3,
        "max_llm_calls": 3,
        "resolution_expectation": "ambiguous",
        "expectations": {
            "must_return_nonempty": True,
            "must_cite_sources": False,
            "must_abstain": True,
            "latency_budget_s": 50.0,
        },
    },
    {
        "id": "fast_ambiguous_x",
        "mode": "fast",
        "category": "ambiguous",
        # "X" -> X (formerly Twitter) vs xAI vs X Development (Alphabet) vs
        # SpaceX nickname. Classic name-suffix collision the disambiguator
        # is explicitly built to flag.
        "prompt": "Research the company X.",
        "target": "company.first_pass_entity_selection",
        "max_external_calls": 3,
        "max_llm_calls": 3,
        "resolution_expectation": "ambiguous",
        "expectations": {
            "must_return_nonempty": True,
            "must_cite_sources": False,
            "must_abstain": True,
            "latency_budget_s": 50.0,
        },
    },
    {
        "id": "fast_classify_disambig_apple",
        "mode": "fast",
        "category": "classify",
        # Direct unit of the disambiguation classifier: given two same-named
        # candidates (the tech company vs a record label), it must score the
        # one matching the user's context higher.
        "prompt": "Apple, the consumer-electronics company that makes the iPhone.",
        "target": "company.disambiguate_entities",
        "max_external_calls": 0,
        "max_llm_calls": 1,
        "resolution_expectation": "probable",
        "expectations": {
            "must_return_nonempty": True,
            "must_cite_sources": False,
            "must_abstain": False,
            "latency_budget_s": 30.0,
        },
    },
    # ---- SLOW: heavier, multi-call research runs --------------------------
    {
        "id": "slow_people_openai_leadership",
        "mode": "slow",
        "category": "people",
        "prompt": (
            "OpenAI — the company behind ChatGPT. Focus on leadership and "
            "key people (CEO, CTO, founders)."
        ),
        "target": "company.first_pass_entity_selection",
        "max_external_calls": 3,
        "max_llm_calls": 3,
        "resolution_expectation": "exact",
        "expectations": {
            "must_return_nonempty": True,
            "must_cite_sources": True,
            "must_abstain": False,
            "latency_budget_s": 90.0,
        },
    },
    {
        "id": "slow_product_nvidia",
        "mode": "slow",
        "category": "product",
        "prompt": (
            "NVIDIA — focus on its core products (GPUs, CUDA, data-center "
            "accelerators) and what they do."
        ),
        "target": "company.company_search_with_linkup",
        "max_external_calls": 1,
        "max_llm_calls": 0,
        "resolution_expectation": "exact",
        "expectations": {
            "must_return_nonempty": True,
            "must_cite_sources": True,
            "must_abstain": False,
            "latency_budget_s": 90.0,
        },
    },
    {
        "id": "slow_entity_schema_snowflake",
        "mode": "slow",
        "category": "entity",
        # Exercises the full text->schema extraction path: search, then
        # Gemini-extract the LINKUP_COMPANY_SCHEMA from the sourced answer.
        "prompt": "Snowflake, the cloud data-warehouse company.",
        "target": "company.search_then_extract_schema",  # composite, see run_eval
        "max_external_calls": 1,
        "max_llm_calls": 1,
        "resolution_expectation": "exact",
        "expectations": {
            "must_return_nonempty": True,
            "must_cite_sources": True,
            "must_abstain": False,
            "latency_budget_s": 120.0,
        },
    },
    {
        "id": "slow_people_advisors_figure",
        "mode": "slow",
        "category": "people",
        "prompt": (
            "Figure AI, the humanoid-robotics startup. Who leads it and who "
            "are its notable backers/advisors?"
        ),
        "target": "company.first_pass_entity_selection",
        "max_external_calls": 3,
        "max_llm_calls": 3,
        "resolution_expectation": "probable",
        "expectations": {
            "must_return_nonempty": True,
            "must_cite_sources": True,
            "must_abstain": False,
            "latency_budget_s": 90.0,
        },
    },
    {
        "id": "slow_ambiguous_phoenix",
        "mode": "slow",
        "category": "ambiguous",
        # "Phoenix" -> Phoenix (Arizona) / Phoenix Suns / Phoenix the
        # observability tool / many startups. Should disambiguate, not commit.
        "prompt": "Look up Phoenix.",
        "target": "company.first_pass_entity_selection",
        "max_external_calls": 4,
        "max_llm_calls": 3,
        "resolution_expectation": "ambiguous",
        "expectations": {
            "must_return_nonempty": True,
            "must_cite_sources": False,
            "must_abstain": True,
            "latency_budget_s": 100.0,
        },
    },
    {
        "id": "slow_classify_disambig_x_collision",
        "mode": "slow",
        "category": "classify",
        # Negative-example aware classify: the bare "X" vs "X AI" suffix
        # collision. The judged behavior is that the candidate matching the
        # user's stated context scores higher than the colliding sibling.
        "prompt": "X, the social platform formerly known as Twitter (NOT xAI).",
        "target": "company.disambiguate_entities",
        "max_external_calls": 0,
        "max_llm_calls": 1,
        "resolution_expectation": "probable",
        "expectations": {
            "must_return_nonempty": True,
            "must_cite_sources": False,
            "must_abstain": False,
            "latency_budget_s": 40.0,
        },
    },
]


def select_cases(mode: str | None = None) -> List[Dict[str, Any]]:
    """Return the cases, optionally filtered to one ``mode`` ("fast"/"slow")."""
    if mode in ("fast", "slow"):
        return [c for c in CASES if c.get("mode") == mode]
    return list(CASES)


__all__ = ["CASES", "select_cases"]
