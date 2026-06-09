# ParselyFi — Tribute & Convergence Roadmap

*A deep-dive across five past projects → how to fold their best ideas into the shipped ParselyFi. Generated from a parallel multi-agent review of each codebase.*

## The throughline

Across five projects the same loop kept recurring from different angles: **turn messy, multi-source inputs into structured, trustworthy, decision-ready research for finance.** Shipped ParselyFi already absorbed the easy throughline (source-backed search, a 3-pass profile, transcription, an honest-status reliability core) — but it stops at *ephemeral, single-source* output. The convergence: **one source-backed research workspace** where (1) every entity is disambiguated before it's trusted, (2) every uploaded filing and recorded call becomes durable, per-user, searchable knowledge, (3) every video and document is grounded evidence with timestamps and citations, and (4) the numbers that matter are computed by deterministic tools, never hallucinated — then optionally narrated and exported.

Each past project contributes exactly **one pillar**.

## Tributes — what each project pioneered

| Project | Contributed | Signature feature |
| --- | --- | --- |
| **parsely_Jan25** (early ParselyFi) | The company-research DNA: per-user corpus mgmt, schema-driven Excel enrichment, multi-source fan-out (→ today's 3-pass) | **Entity disambiguation engine** — negative-example mining + transparent weighted-conflict scoring (founder −5.0, founding-year −4.0, `X` vs `X Bio`/`X AI` suffix −3.5) + LLM same/different verdict; plus regex/LLM corporate-relationship edges (acquired-by / subsidiary / rebranded / spun-off) |
| **FinFlow** (NVIDIA GTC 2025) | The research-to-deliverable loop: fuse live web + fresh video into a narrative and a spoken briefing | **Recency-filtered YouTube scrape + TF-IDF rerank → native Gemini multimodal video analysis** (`Part.from_uri`, `[MM:SS]` timestamps) → speech-script + **ElevenLabs TTS narration** |
| **project1-governmentforms** | Finance-grade rigor: deterministic math-as-a-tool, cost-aware schema-first ingestion, hybrid RAG, smart query routing | **Adjusted-EBITDA reconciliation** where the LLM only proposes line items + an adjustments dict and a **hard-coded Python function does the arithmetic** — auditable NI→EBITDA→AdjEBITDA bridge with a Pyvis DAG of every tool call |
| **associate_assistant_vhs** | The persistence + privacy layer for spoken research | **Fernet-encrypted, username+session-scoped transcript store** in Supabase/Postgres + a user-editable **banking/diligence summary template** + DOCX export |
| **parsely_tool** | The reusable RAG/scraping toolkit + domain agents | **Agentic hybrid RAG** — Qdrant dense (BGE/FastEmbed) + BM25 sparse, **Cohere-reranked**, per-user collections, SHA-256-deduped multi-format ingestion (PDF/DOCX/XLSX/OCR via LlamaParse/unstructured/PyMuPDF), LlamaIndex planner orchestration |

## Key finding

`requirements.txt` **already stages** `qdrant_client`, `fastembed`, `llama-index*`, `pymupdf`, `python-docx`, `neo4j`/`graphiti`, and `pyvis` — but **none are wired into `features/`**. So RAG, document ingestion, the relationship graph, and EBITDA math are **absent-but-pre-provisioned**: the biggest "staged but unused" opportunity.

## Prioritized adoption (value ÷ effort)

| # | Feature | From | Value | Effort | Fits in |
| --- | --- | --- | --- | --- | --- |
| 1 | Entity disambiguation scoring + LLM verdict in Pass-1 | parsely_Jan25 | high | M | Company Search (`first_pass_entity_selection`) |
| 2 | Native Gemini multimodal YouTube analysis (`[MM:SS]`) | FinFlow | high | S | News & YouTube (`_summarize_video`) |
| 3 | Editable banking/diligence summary template + DOCX export | associate_vhs | med | S | Transcription |
| 4 | Encrypted, session-scoped transcript persistence (Supabase) | associate_vhs | high | M | Transcription + Supabase |
| 5 | Per-user auth + storage namespacing (tenant isolation) | jan25 / vhs / tool | high | M | App-level + every per-user store |
| 6 | Multi-format doc ingestion pipeline (PyMuPDF/llama-index, dedupe) | govforms / tool | high | L | new `features/ingestion.py` ← File Manager |
| 7 | Qdrant hybrid RAG → grounded chat-over-your-files | parsely_tool | high | L | new `features/rag.py` → AI Assistant |
| 8 | Corporate relationship / lineage graph | parsely_Jan25 | high | M | Company Search + `features/relationship_graph.py` |
| 9 | Adjusted-EBITDA reconciliation (deterministic Python + DAG) | govforms | high | L | new `features/financials.py` (new tab) |
| 10 | Schema-first per-statement extraction (cheap classifier gate) | govforms | high | L | `features/financials.py` + ingestion |

Also worth it (medium): domain schema library (funding/investors/partnerships/people), adaptive rate-limit + batch progress, recency-filtered YouTube discovery, briefing TTS narration, smart query routing, agentic Excel gap-fill, Pydantic-AI tool-calling orchestrator.

## Roadmap

1. **Phase 1 — Harden what ships** (quick, existing infra): disambiguation scoring; multimodal video grounding; editable summary template + DOCX; richer finance schema fields; batch progress + adaptive rate limiting. *All ride on infra already present (Gemini, LinkUp, ElevenLabs, Pydantic, async batch, python-docx).*
2. **Phase 2 — Durable + tenant-safe**: per-user auth + storage namespacing → encrypted session-scoped transcript persistence → corporate relationship graph (reuses Phase-1 disambiguation edges).
3. **Phase 3 — The document brain (RAG)**: multi-format ingestion (PyMuPDF/llama-index, SHA-256 dedupe) wired into File Manager → Qdrant hybrid dense+sparse RAG → grounded chat-over-your-files → smart query routing (LinkUp + RAG). *Activates the staged-but-unused stack.*
4. **Phase 4 — Finance differentiation**: schema-first statement extraction (income/balance/cash-flow/990) gated by a cheap classifier → deterministic Adjusted-EBITDA reconciliation + DAG trace (new Financials tab) → Pydantic-AI orchestrator that self-sequences web/video/RAG/EBITDA tools.

## Deliberately skipped (off-thesis)

- Reference-video **style-template narration** (cute, low finance value)
- **Real-time WebRTC mic** capture (heavy on Streamlit Cloud)
- Operator **usage-telemetry dashboard** (premature)
- **Selenium/Spider** scraping (the existing SSRF-guarded ethical scraper is strictly safer)
- **Government/benefits form auto-fill** (off-thesis for finance research)
- **Code-review / program-evaluation agent** (developer tooling, orthogonal)
