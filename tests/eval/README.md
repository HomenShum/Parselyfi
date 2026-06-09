# ParselyFi research-agent eval harness

A runnable harness that exercises the **real** ParselyFi company-research code
against a set of fixture cases and produces a scorecard. Ported in spirit from
the nodebench fast/slow scored-eval templates.

## How to run

From the **repo root** (`Parselyfi/`):

```bash
# Deterministic checks only — NO external/LLM calls. Fast, no keys needed.
python tests/eval/run_eval.py --dry

# Fast cases against the live app (uses LinkUp/Gemini if keys are configured).
python tests/eval/run_eval.py --fast

# Slow (heavier, multi-call) cases.
python tests/eval/run_eval.py --slow

# Everything.
python tests/eval/run_eval.py

# Skip the LLM-judge even when a Gemini key is present.
python tests/eval/run_eval.py --fast --no-judge

# Also emit a machine-readable report.
python tests/eval/run_eval.py --fast --json eval_report.json
```

With the project venv explicitly:

```bash
"D:\VSCode Projects\parselyfi\parselyfi_venv\Scripts\python.exe" tests/eval/run_eval.py --dry
```

Exit code is **0** only when every non-skipped case passes, **1** otherwise
(CI-friendly). Skipped cases (target unavailable in this build) are reported but
are **not** counted as passes.

## What it checks

Each case (`tests/eval/cases.py`) declares a `target` (a real callable in
`features.company_research`), call budgets, a resolution expectation, and a set
of expectations. The runner invokes the target via `common.run_async`, times
it, bounds its external + LLM calls, and applies two clearly-labeled kinds of
checks:

### `[DET]` — deterministic checks (pure Python, run in every mode incl. `--dry`)
- **call_completed** — the target call returned without exception / budget trip.
- **nonempty** — the result carries actual content (entities / answer / dict).
- **sources_present** — when `must_cite_sources`, the result carries ≥1 source.
- **abstained_or_disambiguated** — on `ambiguous` cases, the result surfaces
  multiple candidates or a per-candidate confidence signal instead of silently
  committing to one entity.
- **latency_within_budget** — wall-clock ≤ the case's `latency_budget_s`.
- **within_call_budget** — external/LLM call counts are reported and capped at
  the case maxima (the (max+1)-th call trips `_BudgetExceeded` → fail).

### `[LLM]` — Gemini LLM-judge (only with a key, and never in `--dry`)
One `common.gemini_generate_json` call scores the result 0..1 on:
- **entity_correct** (threshold 0.6)
- **no_hallucinations** (threshold 0.6)
- **actionable** (threshold 0.5)

Each axis becomes a `[LLM]` check against its threshold.

## Honest-status / no-fabrication mandate

This harness enforces the project's honest-status rules (see `CLAUDE.md`):

- **Scores are never fabricated.** An LLM-judge number is printed **only** when
  the judge call actually returned it. When the judge can't run (dry run, no
  key, `--no-judge`, or an empty model response) the scorecard shows `n/a` —
  not a hardcoded floor, not a guess. The summary explicitly flags when no judge
  scores were computed.
- **Deterministic vs. judged is always labeled** — `[DET]` vs `[LLM]` in the
  scorecard, so nobody mistakes a pure-Python check for a model opinion.
- **Bounded calls.** Every case caps `max_external_calls` and `max_llm_calls`;
  the harness counts real calls through monkeypatched shims around the shared
  LinkUp-search and Gemini-JSON entry points and trips a hard ceiling on excess.
- **Bounded inputs.** The fixture list is small and finite; the runner allocates
  no unbounded in-memory collections per case.
- **Timeouts.** All network/LLM work flows through `features.common` helpers,
  which already impose per-call timeouts; the latency check surfaces overruns.
- **Skips are honest.** The spec also named a `features.list_intelligence`
  module, which does not exist in this build. Rather than fabricate an import,
  the runner resolves targets dynamically and marks any case whose target module
  or callable is missing as **SKIP (target unavailable)** — not as a pass.
- **`--dry` tells the truth.** With no API, LinkUp/Gemini-backed cases return
  empty results, so their `nonempty`/`sources_present` checks correctly fail.
  That is the intended dry-run signal (the harness wiring works; there is just
  no live data), not a bug.

## Adding cases

Append a dict to `CASES` in `tests/eval/cases.py` (schema documented at the top
of that file). To target a new module, add its alias to `_target_module()` /
`TARGETS` handling in `run_eval.py`. Keep budgets tight — agents amplify every
unbounded call.
