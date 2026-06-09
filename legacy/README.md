# legacy/

Archived prototypes and experiments, kept for provenance. **None of this is part of the running app** (the app is `../streamlit_app.py` + the `../features/` package).

| Folder / pattern | What it was |
| --- | --- |
| `reference_codes/` | Original research scripts the production features were ported from — LinkUp company report (`test7_*`), article/video & document reports (`test2_*`), ethical scraper (`test8_*`), audio transcription (`test6_*`), Graphiti/Neo4j spikes (`test9_*`). |
| `prod_parselyfi_v*.py` | Earlier monolithic versions of the main app (superseded by `streamlit_app.py`). |
| `prod_ingestion_pipeline_v*.py` | Document/multimodal ingestion pipeline experiments. |
| `prod_social_media_profile_v*.py` | Social-media profile analysis spikes. |
| `prod_qdrant_*.py`, `qdrant_hybrid_retrieval_v*.py` | Qdrant vector-store retrieval experiments. |
| `Concept_prod_ui_interface_*.py`, `custom_real_time_chat_*.py` | UI/chat concept prototypes. |
| `*.html`, `cafe_corner.css` | Static UI prototypes. |
| `*.json`, `*.txt` | Sample analysis data and onboarding prompt templates. |
| `DESIGN_OVERVIEW.md`, `SELF_ADAPTIVE_IMPLEMENTATION_ROADMAP.md` | Aspirational design docs for a separate self-adaptive multi-agent system (not implemented here). |
