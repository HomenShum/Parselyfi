<div align="center">

# 🌱 ParselyFi

### Source-backed financial research, in one Streamlit app.

**Company intelligence · News & YouTube briefings · Audio transcription & summaries** — powered by LinkUp `sourcedAnswer`, Google **Gemini 3.5 Flash**, and ElevenLabs.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://parselyfi.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
![Gemini 3.5 Flash](https://img.shields.io/badge/LLM-Gemini%203.5%20Flash-8E75B2)
![Search](https://img.shields.io/badge/search-LinkUp%20sourcedAnswer-0A7CFF)
![License](https://img.shields.io/badge/license-All%20rights%20reserved-lightgrey)

<a href="assets/parselyfi-demo.mp4">
  <img src="assets/parselyfi-demo.gif" alt="ParselyFi feature demo" width="820">
</a>

<sub>▶︎ Click for the full demo video · feature-by-feature walkthrough</sub>

</div>

---

## What is ParselyFi?

ParselyFi is a Streamlit workspace for **venture-capital, middle-market, and public-company research**. It turns a company name, a news topic, or an audio file into **cited, source-backed output** — never fabricated. Three research tabs sit alongside a Supabase/S3 file manager, an AI assistant, and a public financial-data dashboard.

> **Design rule everywhere:** if a key, source, or result is missing, the app says so and degrades gracefully. It never invents companies, sources, transcripts, or summaries.

---

## ✨ Features

Three source-backed research workflows — each previewed below — plus a file manager, AI assistant, and public data dashboard.

### 🔍 Company Search & Analysis

<p align="center"><img src="assets/feature-company.gif" alt="Company Search & Analysis preview" width="680"></p>

Source-backed company research with a **3-pass workflow** — entity resolution → structured profile → missing-field backfill. Free-text *or* spreadsheet upload, an editable Pending/Complete/Skip review grid, batch enrichment with concurrency, per-agent token/cost tracking, and exportable search history.

### 📰 News & YouTube

<p align="center"><img src="assets/feature-news.gif" alt="News & YouTube preview" width="680"></p>

**News Alerts:** a topic → LinkUp source-backed search → a cited Gemini briefing (summary, key points, sentiment) + an editable list of real sources. **YouTube Daily Reports:** a video URL / pasted transcript → a structured key-insight summary.

### 🎙️ Transcription & Summaries

<p align="center"><img src="assets/feature-transcription.gif" alt="Transcription & Summaries preview" width="680"></p>

Upload audio → **ElevenLabs** speech-to-text → a synced AnyWidget player that highlights the transcript during playback → one-click **AI summary** (summary + key points + action items). Paste/upload an existing transcript if you have no audio key.

### Plus

| Tab | What it does |
| --- | --- |
| 🗂️ **File Manager** | Browse, upload, download, and organize files in S3-compatible **Supabase Storage**, with pagination and folder navigation. |
| 🤖 **Parsely AI Assistant** | Chat assistant for workflow help and Q&A. |
| 📊 **Public Dashboard** | Curated financial data (companies, YouTube transcriptions, news, forums) with interactive tables — Master DB, Products, Partnerships, Investors. |

> The full ~40s reel above (hero) stitches these three workflows end-to-end with narration.

---

## 🏗️ Architecture

```
streamlit_app.py            # entry point (sidebar + 6 tabs + auth)
features/
  common.py                 # shared core: secret guard, lazy Gemini/LinkUp clients,
                            #   async runner, bounded token ledger, hardened web scraper
  company_research.py       # render_company_research_tab()  — 3-pass LinkUp + Gemini
  news_youtube.py           # render_news_youtube_tab()      — LinkUp-backed news + video
  transcription.py          # render_transcription_tab()     — ElevenLabs STT + Gemini summary
dev_preview_tabs.py         # no-auth dev/QA harness that renders the 3 tabs directly
demo/                       # TestReel + Playwright recording scripts + Remotion video project
assets/                     # rendered demo video + preview GIF
legacy/                     # archived prototypes & versioned experiments (see legacy/README.md)
```

**Reliability built into `common.py`** (it amplifies across agent loops, so it's enforced):

- **SSRF guard** — every fetch validates the URL and re-checks redirects per-hop; a connect-time DNS resolver blocks private/loopback/link-local/metadata IPs (DNS-rebind safe).
- **Bounded memory** — token ledger, scraper cache, and search history all have hard caps with eviction.
- **Timeouts + bounded reads** on every LLM / network / scrape call.
- **Honest status** — failed calls return empty results and surface an actionable banner (e.g. "rotate your Gemini key"), never fabricated data.

**Stack:** Streamlit · Supabase (Postgres + S3 Storage) · `google-genai` (Gemini 3.5 Flash) · `linkup-sdk` · `trafilatura`/BeautifulSoup (ethical scraping) · ElevenLabs + `streamlit-anywidget` · pandas.

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/HomenShum/Parselyfi.git
cd Parselyfi

# 2. Install (use a virtualenv)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure secrets (see below), then run
streamlit run streamlit_app.py
```

> The Streamlit Community Cloud entry point is **`streamlit_app.py`**.

### 🔑 Secrets

Create `.streamlit/secrets.toml` (gitignored). Storage/auth go under `[supabase]`; **API keys are top-level**:

```toml
[supabase]
SUPABASE_URL = "..."
SUPABASE_KEY = "..."
SUPABASE_S3_BUCKET_NAME = "..."
SUPABASE_S3_ENDPOINT_URL = "..."
SUPABASE_S3_BUCKET_REGION = "..."
SUPABASE_S3_BUCKET_ACCESS_KEY = "..."
SUPABASE_S3_BUCKET_SECRET_KEY = "..."

# Feature API keys (top level)
GEMINI_API_KEY      = "..."   # or legacy GOOGLE_AI_STUDIO   — Company / News / summaries
LINKUP_API_KEY      = "..."   # https://www.linkup.so        — source-backed search
ELEVEN_LABS_API_KEY = "..."   # speech-to-text for Transcription
```

Each feature **degrades gracefully**: a missing key shows a notice instead of failing, and never fabricates results.

---

## 🧪 Dev / QA

`dev_preview_tabs.py` renders the three feature tabs directly (no Google-login gate) for fast iteration and headless QA:

```bash
streamlit run dev_preview_tabs.py
```

The demo video is produced from this harness — see [`demo/README.md`](demo/README.md).

---

## 🗂️ Legacy

Earlier prototypes, the versioned `prod_*` experiments, ingestion/Qdrant/social-profile spikes, and the original `reference_codes/` research scripts live in [`legacy/`](legacy/). They are kept for provenance and are **not** part of the running app.

---

## 🧬 Lineage & tribute

ParselyFi is where five earlier experiments converge. Each attacked the same loop — *turn messy, multi-source inputs into trustworthy, decision-ready finance research* — from a different angle, and each contributes one pillar:

| Project | Pillar | Signature idea |
| --- | --- | --- |
| **parsely_Jan25** | Correctness | Entity disambiguation (negative-example mining + weighted-conflict scoring + LLM verdict) and corporate-lineage edges |
| **FinFlow** *(NVIDIA GTC 2025)* | Freshness + delivery | Recency-filtered YouTube → native Gemini multimodal video analysis with `[MM:SS]` timestamps → ElevenLabs spoken briefing |
| **project1-governmentforms** | Rigor | Adjusted-EBITDA where **Python does the math**, with an auditable tool-call DAG |
| **associate_assistant_vhs** | Durability + privacy | Encrypted, session-scoped transcript persistence + editable diligence templates + DOCX export |
| **parsely_tool** | Grounding | Agentic hybrid RAG (Qdrant dense + BM25 sparse + Cohere rerank) over multi-format, OCR'd documents |

The full deep-dive and the phased convergence roadmap are in **[`docs/TRIBUTE.md`](docs/TRIBUTE.md)**.

---

## 👨‍💻 About the creator

**Homen Shum** — data-driven builder across AI/ML, data analytics, and workflow automation, with a startup-banking background (JPMC) and a technical co-founder track record.

- 🌐 [homenshum.com](https://homenshum.com/) · [LinkedIn](https://linkedin.com/in/homen-shum) · [GitHub](https://github.com/HomenShum)

## 📜 License

Homen Shum reserves all rights not expressly granted by the license. See [LICENSE](LICENSE).
