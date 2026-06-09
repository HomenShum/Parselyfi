import os, json, tempfile, time
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

# ─────────────────────────── 0 · Load Environment ──────────────────────────
load_dotenv()  # loads GEMINI_API_KEY if stored in .env

# Initialize session state for analysis JSON
if 'analysis' not in st.session_state:
    st.session_state['analysis'] = None

# ─────────────────────────── 1 · App Config ───────────────────────────────
st.set_page_config(page_title="Gemini Social-Media On-Boarding", page_icon="🧩", layout="centered")

# Default Model (update from official docs)
MODEL_OPTIONS = [
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-exp-03-25",
]
MODEL_NAME = st.sidebar.selectbox("Select Gemini Model", MODEL_OPTIONS, index=3)

# Supported MIME types (from official docs)
SUPPORTED_MIME = {
    # video
    ".mp4":  "video/mp4",
    ".webm": "video/webm",
    ".mkv":  "video/x-matroska",
    ".mov":  "video/quicktime",
    ".flv":  "video/x-flv",
    ".wmv":  "video/wmv",
    ".mpeg": "video/mpeg",
    ".mpg":  "video/mpg",
    ".3gp":  "video/3gpp",
    # audio
    ".mp3":  "audio/mpeg",
    ".wav":  "audio/wav",
    # images
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    # documents
    ".pdf":  "application/pdf",
    ".txt":  "text/plain",
}

# ─────────────────────────── 2 · Sidebar ───────────────────────────
def sidebar():
    st.sidebar.title("🔑 Gemini API Key")
    default = os.getenv("GEMINI_API_KEY", "")
    api_key = st.sidebar.text_input("Enter API Key", default, type="password")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built with [Streamlit](https://streamlit.io/) + [Google GenAI SDK](https://ai.google.dev/gemini-api/docs/sdks)"
    )

sidebar()

# ─────────────────────────── Helper: wait until file ACTIVE ───────────────
def wait_until_active(client: genai.Client, file):
    """Polls the uploaded File resource until its state becomes ACTIVE or FAILED."""
    start = time.time()
    with st.spinner("Processing uploaded file …"):
        while file.state == "PROCESSING":
            if time.time() - start > 600:  # 10‑minute safety timeout
                raise TimeoutError("File processing timed out. Try a smaller file or retry later.")
            time.sleep(5)
            file = client.files.get(name=file.name)
    if file.state != "ACTIVE":
        raise RuntimeError(f"File failed processing with state: {file.state}")
    return file

# ─────────────────────────── 3 · Tabs ───────────────────────────
tabs = st.tabs(["On-Boarding", "Official Video Example", "Dashboard"])

# --- Tab: On-Boarding ---
with tabs[0]:
    st.title("📈 Social-Media Analysis On-Boarding")

    # File uploader
    uploaded = st.file_uploader(
        "Step 1 · Upload an *optional* media file (video / audio / image / pdf)",
        type=list(SUPPORTED_MIME.keys()),
    )

    # Prompt editor (lazy load from file if available)
    prompt_path = Path("onboarding_prompt_template.txt")
    DEFAULT_PROMPT = prompt_path.read_text() if prompt_path.exists() else ""
    prompt_text = st.text_area(
        "Step 2 · Review / edit the analysis prompt", DEFAULT_PROMPT, height=300
    )

    # Generate button
    if st.button("🚀 Generate JSON Analysis", disabled=not prompt_text.strip()):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            st.error("🔑 Please provide a Gemini API key.")
            st.stop()
        client = genai.Client(api_key=key)

        parts = [types.Part.from_text(text=prompt_text)]
        if uploaded:
            ext = Path(uploaded.name).suffix.lower()
            mime = SUPPORTED_MIME.get(ext)
            if not mime:
                st.error(f"Unsupported file type: {uploaded.name}")
                st.stop()

            # ─── Small files inline ───
            if uploaded.size < 20_000_000:
                parts.insert(0, types.Part(inline_data=types.Blob(data=uploaded.read(), mime_type=mime)))
            else:
                # ─── Large files via upload API ───
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                tmp.write(uploaded.getbuffer())
                cfg = types.UploadFileConfig(display_name=uploaded.name, mime_type=mime)
                with st.spinner("⬆️ Uploading to Gemini…"):
                    file_ref = client.files.upload(file=tmp.name, config=cfg)
                try:
                    file_ref = wait_until_active(client, file_ref)
                except Exception as e:
                    st.error(str(e))
                    st.stop()
                parts.insert(0, types.Part.from_uri(file_uri=file_ref.uri, mime_type=mime))

        contents = [types.Content(role="user", parts=parts)]
        placeholder = st.empty()
        full_text = ""

        try:
            stream = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            for chunk in stream:
                if chunk.text:
                    full_text += chunk.text
                    placeholder.code(full_text, language="json")
            st.success("Generation complete ✨")
            try:
                parsed = json.loads(full_text)
                # Store in session state and save to disk
                st.session_state['analysis'] = parsed
                with open("social_media_analysis.json", "w") as f:
                    json.dump(parsed, f, indent=2)

                st.json(parsed, expanded=False)
                st.download_button(
                    "💾 Download JSON", json.dumps(parsed, indent=2),
                    "social_media_analysis.json", "application/json"
                )
            except json.JSONDecodeError:
                st.warning("Returned text isn’t valid JSON. Inspect manually.")
        except Exception as e:
            st.error(f"Error while generating content: {e}")

# --- Tab: Official Video Example ---
with tabs[1]:
    st.header("🎬 Official Video Understanding Quickstart")
    st.markdown("Use the example below to explore Gemini's video analysis capabilities.")
    st.markdown("[Open Colab Notebook](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Video_understanding.ipynb)")
    st.code(
        '''python
from google import genai
from google.genai import types
import time, os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_name = "gemini-2.5-pro-exp-03-25"

video = client.files.upload("Pottery.mp4")
while video.state == "PROCESSING":
    time.sleep(5)
    video = client.files.get(name=video.name)

prompt = "For each scene, generate captions with timecodes."
response = client.models.generate_content(model=model_name, contents=[video, prompt])
print(response.text)
''',
        language="python",
    )
    st.video("https://www.youtube.com/embed/Mot-JEU26GQ?si=pcb7-_MZTSi_1Zkw")

# --- Tab: Dashboard ---
with tabs[2]:
    st.title("📊 Analysis Dashboard")
    analysis = st.session_state.get('analysis')
    # Try loading from file if not in session
    if analysis is None and os.path.exists("social_media_analysis.json"):
        with open("social_media_analysis.json") as f:
            analysis = json.load(f)
            st.session_state['analysis'] = analysis

    if analysis:
        # Final summary
        final = analysis.get('detailedUserProfileAnalysis', {}).get('finalSummary')
        if final:
            st.subheader("📝 Final Summary")
            st.write(final)

        # Display key metadata
        meta = analysis.get('detailedUserProfileAnalysis', {}).get('analysisMetadata')
        if meta:
            st.subheader("📆 Analysis Metadata")
            st.json(meta)

        # List platforms
        profiles = analysis.get('detailedUserProfileAnalysis', {}).get('platformProfiles', [])
        if profiles:
            st.subheader("🌐 Platforms Analyzed")
            for p in profiles:
                st.markdown(f"**{p.get('platformName')}**")

        # Full JSON viewer
        st.subheader("🔍 Full JSON Structure")
        st.json(analysis)
    else:
        st.info("No analysis JSON found. Run On-Boarding to generate analysis.")

# ─────────────────────────── Footer ──────────────────────────
st.markdown("---")
st.caption("© 2025 ParselyFi — prototype built with Streamlit + Gemini SDK")
