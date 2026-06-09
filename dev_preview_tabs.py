"""Dev/QA harness — renders the three finished ParselyFi feature tabs directly.

The production app (prod_parselyfi_v031525.py) gates all content behind a
Google-login wall (st.stop() when not logged in), which blocks headless QA.
This harness calls the exact render_*_tab() entry points the production tabs
use, with the same per-tab error boundary, so the feature code can be
dogfooded in a real browser without auth. Not part of the shipped app.

Run from the Parselyfi/ directory (so st.secrets and the features package resolve):
    streamlit run dev_preview_tabs.py
"""
import os

import streamlit as st

# DEMO_CLEAN=1 hides the QA chrome (harness title + Streamlit toolbar) so the
# demo-video capture frames look like the production app, not a QA harness.
DEMO = os.environ.get("DEMO_CLEAN") == "1"

st.set_page_config(
    page_title="ParselyFi" if DEMO else "ParselyFi · Feature Tab QA",
    page_icon="🌱" if DEMO else "🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Shared design system (custom CSS) — applied app-wide.
try:
    from features import ui
    ui.inject_css()
except Exception:
    pass

if DEMO:
    st.markdown(
        """
        <style>
          header[data-testid="stHeader"] {display: none !important;}
          [data-testid="stToolbar"] {display: none !important;}
          #MainMenu, footer {visibility: hidden;}
          .block-container {padding-top: 2.2rem !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("🧪 ParselyFi — Feature Tab Dogfood Harness")
    st.caption(
        "Renders the 3 finished tabs via their render_*_tab() entry points, "
        "bypassing the Google-login gate for QA only."
    )

li_tab, tab3, tab4, tab5 = st.tabs([
    "📋 List Intelligence",
    "🔍 Company Search & Analysis",
    "📰 News & YouTube",
    "🎙️ Transcription & Summaries",
])

with li_tab:
    try:
        from features.list_intelligence import render_list_intelligence_tab
        render_list_intelligence_tab()
    except Exception as e:  # noqa: BLE001
        st.error("⚠️ List Intelligence could not be loaded.")
        with st.expander("Error details"):
            st.exception(e)

with tab3:
    try:
        from features.company_research import render_company_research_tab
        render_company_research_tab()
    except Exception as e:  # noqa: BLE001 - surface any load error for QA
        st.error("⚠️ Company Search & Analysis could not be loaded.")
        with st.expander("Error details"):
            st.exception(e)

with tab4:
    try:
        from features.news_youtube import render_news_youtube_tab
        render_news_youtube_tab()
    except Exception as e:  # noqa: BLE001
        st.error("⚠️ News & YouTube could not be loaded.")
        with st.expander("Error details"):
            st.exception(e)

with tab5:
    try:
        from features.transcription import render_transcription_tab
        render_transcription_tab()
    except Exception as e:  # noqa: BLE001
        st.error("⚠️ Transcription & Summaries could not be loaded.")
        with st.expander("Error details"):
            st.exception(e)
