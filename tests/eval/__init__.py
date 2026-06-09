"""ParselyFi research-agent eval harness.

Two public surfaces:

- ``tests.eval.cases``  — the FIXTURE cases (a list of plain dicts).
- ``tests.eval.run_eval`` — the runner (CLI + importable ``main``).

The runner invokes the REAL ParselyFi research functions (from
``features.company_research``) through ``features.common.run_async`` and applies
deterministic checks, optionally augmented by a Gemini LLM-judge when a key is
present. It never fabricates a score: a score only appears when it was actually
computed by a judge call.
"""
