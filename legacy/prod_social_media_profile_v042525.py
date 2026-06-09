# Required imports for the Dashboard tab enhancements
import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64 # Potentially needed for embedding images if not saving to disk
import datetime

# ─────────────────────────── Helper Functions for Dashboard ───────────────────

@st.cache_data
def get_platform_data(analysis_data):
    """Extracts platform profiles safely."""
    # Ensure the structure exists before trying to access nested keys
    detailed_analysis = analysis_data.get('detailedUserProfileAnalysis', {})
    if detailed_analysis:
        return detailed_analysis.get('platformProfiles', [])
    return [] # Return empty list if structure is missing

@st.cache_data
def create_content_type_pie(platform_data, selected_platform):
    """Generates a COMPACT pie chart of content types."""
    all_content_types = []
    profiles_to_analyze = []

    if selected_platform == "All Platforms":
        profiles_to_analyze = platform_data
    else:
        profiles_to_analyze = [p for p in platform_data if p.get('platformName') == selected_platform]

    if not profiles_to_analyze: return None

    for profile in profiles_to_analyze:
        types = profile.get('contentActivity', {}).get('primaryContentTypes', [])
        all_content_types.extend(types)

    if not all_content_types: return None

    type_counts = pd.Series(all_content_types).value_counts()

    # --- COMPACT Changes ---
    fig, ax = plt.subplots(figsize=(3.5, 2.2)) # Smaller figure size
    pie_wedges, texts, autotexts = ax.pie(type_counts, autopct='%1.0f%%', startangle=90, radius=1.2, textprops={'fontsize': 7})
    ax.axis('equal')
    # Add legend instead of labels on wedges if too crowded
    ax.legend(pie_wedges, type_counts.index, title="Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
    plt.setp(autotexts, size=7, weight="bold", color="white") # Make percentages visible
    #plt.title(f"Content Types ({selected_platform})", fontsize=9) # Title might make it too big
    plt.tight_layout(rect=[0, 0, 0.75, 1]) # Adjust layout to make space for legend
    # --- End COMPACT Changes ---
    return fig

@st.cache_data
def create_themes_wordcloud(platform_data, selected_platform):
    """Generates a COMPACT word cloud of recurring themes."""
    all_themes = []
    profiles_to_analyze = []

    if selected_platform == "All Platforms":
        profiles_to_analyze = platform_data
    else:
        profiles_to_analyze = [p for p in platform_data if p.get('platformName') == selected_platform]

    if not profiles_to_analyze: return None

    for profile in profiles_to_analyze:
        themes = profile.get('contentActivity', {}).get('recurringThemes', [])
        all_themes.extend(themes)

    if not all_themes: return None

    text = ' '.join(all_themes)
    # Handle cases where text might be empty or only contains stop words
    if not text.strip(): return None

    try:
        # --- COMPACT Changes ---
        wordcloud = WordCloud(width=250, height=150, background_color='white', max_words=30).generate(text)
        fig, ax = plt.subplots(figsize=(4, 2.5)) # Smaller figure
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        # plt.title(f"Themes ({selected_platform})", fontsize=9) # Optional Title
        plt.tight_layout(pad=0)
        # --- End COMPACT Changes ---
        return fig
    except ValueError:
        return None
    except ImportError:
        st.error("WordCloud library not installed. `pip install wordcloud`")
        return None

@st.cache_data
def create_tone_comparison_chart(platform_data):
    """Generates a COMPACT bar chart comparing tones across platforms."""
    tone_data = []
    platforms = [p.get('platformName') for p in platform_data if p.get('platformName')]
    if not platforms or len(platforms) < 1: # Can show even for 1 platform
         return None

    for profile in platform_data:
        platform_name = profile.get('platformName')
        tone_desc = profile.get('contentActivity', {}).get('toneAndVoice', '')
        primary_tone = tone_desc.split(',')[0].split(' ')[0].capitalize() if tone_desc else 'Unknown'
        if platform_name:
             tone_data.append({'Platform': platform_name, 'Primary Tone': primary_tone})

    if not tone_data: return None

    df = pd.DataFrame(tone_data)
    tone_counts = df.groupby(['Platform', 'Primary Tone']).size().unstack(fill_value=0)

    if tone_counts.empty: return None

    # --- COMPACT Changes ---
    fig, ax = plt.subplots(figsize=(4, 2.5)) # Smaller figure
    tone_counts.plot(kind='bar', ax=ax, width=0.8, fontsize=8) # Adjust bar width and font size
    #plt.title("Primary Tone Comparison", fontsize=9) # Optional title
    plt.ylabel("Count", fontsize=8)
    plt.xlabel(None) # Remove X label to save space
    plt.xticks(rotation=0, ha='center', fontsize=8)
    plt.legend(title='Tone', fontsize=7, title_fontsize=8) # Smaller legend
    plt.tight_layout()
    # --- End COMPACT Changes ---
    return fig


# --- (Keep existing imports and code from the original script) ---
import os, json, tempfile, time
from pathlib import Path
# import streamlit as st # already imported
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

# ─────────────────────────── 0 · Load Environment ──────────────────────────
load_dotenv()

if 'analysis' not in st.session_state:
    st.session_state['analysis'] = None
if 'selected_platform' not in st.session_state:
    st.session_state['selected_platform'] = "All Platforms"
if 'show_profile_fundamentals' not in st.session_state:
    st.session_state['show_profile_fundamentals'] = True
if 'show_content_activity' not in st.session_state:
    st.session_state['show_content_activity'] = True
if 'show_network_community' not in st.session_state:
    st.session_state['show_network_community'] = True
if 'show_algorithmic_insights' not in st.session_state:
    st.session_state['show_algorithmic_insights'] = True

# ─────────────────────────── 1 · App Config ───────────────────────────────
st.set_page_config(page_title="Gemini Social-Media On-Boarding", page_icon="🧩", layout="wide") # Use wide layout for dashboard

MODEL_OPTIONS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-exp-03-25", # Might not be available publicly
]
# Find a reasonable default, like 2.5 Flash if available, otherwise the first one
try:
    default_index = MODEL_OPTIONS.index("gemini-2.5-flash-preview-04-17")
except ValueError:
    default_index = 0
MODEL_NAME = st.sidebar.selectbox("Select Gemini Model", MODEL_OPTIONS, index=default_index)

SUPPORTED_MIME = {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska", ".mov": "video/quicktime", ".flv": "video/x-flv", ".wmv": "video/wmv", ".mpeg": "video/mpeg", ".mpg": "video/mpg", ".3gp": "video/3gpp", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".pdf": "application/pdf", ".txt": "text/plain"}

# ─────────────────────────── 2 · Sidebar ───────────────────────────
def sidebar():
    st.sidebar.title("🔑 Gemini API Key")
    default = os.getenv("GEMINI_API_KEY", "")
    api_key = st.sidebar.text_input("Enter API Key", value=default, type="password")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with [Streamlit](https://streamlit.io/) + [Google GenAI SDK](https://ai.google.dev/gemini-api/docs/sdks)")

sidebar()

# ─────────────────────────── Helper: wait until file ACTIVE ───────────────
def wait_until_active(client: genai.Client, file):
    start = time.time()
    with st.spinner("Processing uploaded file..."):
        while file.state.name == "PROCESSING":
            if time.time() - start > 600:
                raise TimeoutError("File processing timed out.")
            time.sleep(5)
            file = client.get_file(name=file.name) # Use get_file
    if file.state.name != "ACTIVE":
        raise RuntimeError(f"File failed processing with state: {file.state.name}")
    return file

# ─────────────────────────── 3 · Tabs ───────────────────────────
tabs = st.tabs(["On-Boarding", "Official Video Example", "Dashboard"])

# --- Tab: On-Boarding ---
with tabs[0]:
    st.title("📈 Social-Media Analysis On-Boarding")
    uploaded = st.file_uploader("Step 1 · Upload an *optional* media file (video / audio / image / pdf)", type=list(SUPPORTED_MIME.keys()))
    prompt_path = Path("onboarding_prompt_template.txt")
    DEFAULT_PROMPT = prompt_path.read_text() if prompt_path.exists() else ""
    prompt_text = st.text_area("Step 2 · Review / edit the analysis prompt", DEFAULT_PROMPT, height=300)

    if st.button("🚀 Generate JSON Analysis", disabled=not prompt_text.strip()):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            st.error("🔑 Please provide a Gemini API key.")
            st.stop()

        # Configure the GenAI client (use configure instead of Client directly for API key)
        try:
            genai.configure(api_key=key)
        except Exception as e:
            st.error(f"Failed to configure Gemini client: {e}")
            st.stop()

        # Create the generative model instance
        model = genai.GenerativeModel(MODEL_NAME)

        parts = [types.Part.from_text(text=prompt_text)]
        file_ref = None # Keep track of uploaded file for potential deletion later
        tmp_file_path = None # Keep track of temp file path

        if uploaded:
            ext = Path(uploaded.name).suffix.lower()
            mime = SUPPORTED_MIME.get(ext)
            if not mime:
                st.error(f"Unsupported file type: {uploaded.name}")
                st.stop()

            # Use File API for all uploads for consistency and robustness
            with st.spinner(f"⬆️ Uploading '{uploaded.name}' to Gemini..."):
                # Save uploaded file temporarily to disk for the API
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_file_path = tmp.name

                # Upload using genai.upload_file
                try:
                    # Note: display_name is not a parameter in genai.upload_file directly
                    # It might be inferred from the path or set later if needed.
                    file_ref = genai.upload_file(path=tmp_file_path, mime_type=mime)
                    st.write(f"Uploaded file: {file_ref.name} ({file_ref.display_name}), URI: {file_ref.uri}")

                    # Wait for the file to be active
                    file_ref = wait_until_active(genai, file_ref) # Pass the genai module/client
                    st.write(f"File '{file_ref.display_name}' is ACTIVE.")
                    parts.insert(0, file_ref) # Add the file reference directly

                except Exception as e:
                    st.error(f"File upload or processing failed: {e}")
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path) # Clean up temp file on error
                    st.stop()
                finally:
                    # Clean up the temporary file after processing
                    if tmp_file_path and os.path.exists(tmp_file_path):
                         os.remove(tmp_file_path)


        contents = parts # Pass the list of parts directly to generate_content
        placeholder = st.empty()
        full_text = ""

        try:
            with st.spinner("🧠 Gemini is thinking..."):
                 stream = model.generate_content(
                     contents=contents,
                     generation_config=genai.types.GenerationConfig(
                         response_mime_type="application/json"
                     ),
                     stream=True
                 )
                 for chunk in stream:
                     if chunk.text:
                         full_text += chunk.text
                         placeholder.code(full_text, language="json")

            st.success("Generation complete ✨")
            try:
                # Clean potential markdown fences if the model added them
                if full_text.strip().startswith("```json"):
                    full_text = full_text.strip()[7:-3].strip()
                elif full_text.strip().startswith("```"):
                     full_text = full_text.strip()[3:-3].strip()

                parsed = json.loads(full_text)
                st.session_state['analysis'] = parsed
                with open("social_media_analysis.json", "w") as f:
                    json.dump(parsed, f, indent=2)
                st.success("Analysis saved to social_media_analysis.json")

                # Expander removed as full JSON is now in Dashboard
                # st.json(parsed, expanded=False)
                st.download_button(
                    "💾 Download JSON", json.dumps(parsed, indent=2),
                    "social_media_analysis.json", "application/json"
                )
                # Automatically switch to dashboard? Maybe too abrupt. Let user click.
                st.info("✅ Analysis generated. View the results in the Dashboard tab.")

            except json.JSONDecodeError as e:
                st.error(f"Returned text isn’t valid JSON. Error: {e}")
                st.text_area("Raw Output", full_text, height=200)
            except Exception as e: # Catch other errors during parsing/saving
                st.error(f"An error occurred after generation: {e}")
                st.text_area("Raw Output", full_text, height=200)

        except errors.InternalServerError as e:
             st.error(f"An internal server error occurred: {e}. The model might be temporarily unavailable or the request too complex.")
        except errors.ResourceExhaustedError as e:
             st.error(f"Resource exhausted: {e}. You might have hit API rate limits.")
        except Exception as e:
            st.error(f"Error during content generation: {e}")
            st.text_area("Raw Output", full_text, height=200) # Show partial output if any

        # # Clean up uploaded file from Gemini storage if it exists
        # # Consider if you want to keep it for regeneration or debugging
        # if file_ref:
        #      try:
        #          with st.spinner(f"🧹 Cleaning up uploaded file '{file_ref.display_name}'..."):
        #               genai.delete_file(name=file_ref.name)
        #          st.write(f"Cleaned up file: {file_ref.name}")
        #      except Exception as e:
        #          st.warning(f"Could not delete uploaded file {file_ref.name}: {e}")


# --- Tab: Official Video Example ---
with tabs[1]:
    # (Content remains the same as provided)
    st.header("🎬 Official Video Understanding Quickstart")
    st.markdown("Use the example below to explore Gemini's video analysis capabilities.")
    st.markdown("[Open Colab Notebook](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Video_understanding.ipynb)")
    st.code(
        '''python
# Requires google-generativeai library
# pip install google-generativeai

import google.generativeai as genai
import time
import os

# Configure with your API key
# genai.configure(api_key="YOUR_API_KEY") # Or load from environment

# Upload the video file
# Note: Files uploaded this way persist for 2 days.
# Consider deleting them if not needed after analysis.
video_file = genai.upload_file(path="Pottery.mp4")
print(f"Uploaded file '{video_file.display_name}' as: {video_file.name}")

# Wait for the video processing to complete.
# File processing can take several minutes.
print("Processing video...")
while video_file.state.name == "PROCESSING":
    print('.', end='', flush=True)
    time.sleep(10)
    # Check the file processing state.
    video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
  raise ValueError(video_file.state.name)
print(f"Video processing complete.")

# Create the prompt.
prompt = "Provide a description of the video."

# Set the model to Gemini 1.5 Pro.
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

# Make the LLM request.
print("Generating content...")
response = model.generate_content([prompt, video_file],
                                   request_options={"timeout": 600}) # Increase timeout for video
print(response.text)

# Optional: Delete the file to clean up storage
# genai.delete_file(video_file.name)
# print(f"Deleted file {video_file.name}")
''',
        language="python",
    )
    # Embed the YouTube video using st.video
    st.video("https://www.youtube.com/watch?v=Mot-JEU26GQ") # Direct YouTube URL works


# --- Function to extract content pillars from the schema ---
@st.cache_data
def extract_content_pillars(analysis_data):
    """Extract content pillars from the social media analysis schema."""
    content_pillars = []
    
    # Check if we have crossPlatformSynthesis data
    cross_platform = analysis_data.get('crossPlatformSynthesis', {})
    if cross_platform:
        # Extract professional skills
        if cross_platform.get('synthesizedExpertiseInterests'):
            skills = cross_platform.get('synthesizedExpertiseInterests', {}).get('coreProfessionalSkills', [])
            for skill in skills:
                if skill:  # Ensure the skill is not empty
                    if isinstance(skill, dict) and 'name' in skill:
                        content_pillars.append(skill)
                    else:
                        # Convert string skills to dictionary format
                        content_pillars.append({
                            'name': str(skill),
                            'description': 'Professional skill identified across platforms'
                        })
        
        # Extract personal interests
        if cross_platform.get('synthesizedExpertiseInterests'):
            interests = cross_platform.get('synthesizedExpertiseInterests', {}).get('corePersonalInterests', [])
            for interest in interests:
                if interest:  # Ensure the interest is not empty
                    if isinstance(interest, dict) and 'name' in interest:
                        content_pillars.append(interest)
                    else:
                        # Convert string interests to dictionary format
                        content_pillars.append({
                            'name': str(interest),
                            'description': 'Personal interest identified across platforms'
                        })
    
    # If we have account optimization data with content pillars, use it
    account_opt = analysis_data.get('detailedUserProfileAnalysis', {}).get('accountSetupOptimization', {})
    if account_opt and 'contentPillars' in account_opt:
        account_pillars = account_opt.get('contentPillars', [])
        if account_pillars:
            content_pillars.extend(account_pillars)
    
    return content_pillars

# --- Extract recommendation steps from schema ---
@st.cache_data
def extract_recommendations(analysis_data):
    """Extract recommendations from the social media analysis schema."""
    recommendations = []
    
    # Check account optimization data
    account_opt = analysis_data.get('detailedUserProfileAnalysis', {}).get('accountSetupOptimization', {})
    if account_opt and 'recommendedNextSteps' in account_opt:
        recommendations.extend(account_opt.get('recommendedNextSteps', []))
    
    # Check cross-platform synthesis for professional evaluation
    cross_platform = analysis_data.get('crossPlatformSynthesis', {})
    prof_eval = cross_platform.get('professionalEvaluation', {})
    if prof_eval:
        strengths = prof_eval.get('strengthsSkillsMatch')
        if strengths:
            recommendations.append(f"Leverage strengths: {strengths}")
        
        red_flags = prof_eval.get('potentialRedFlagsClarifications')
        if red_flags:
            recommendations.append(f"Address potential concerns: {red_flags}")
    
    return recommendations

# --- Display the content pillars section with proper type checking ---
def display_content_pillars(pillars):
    """Safely display content pillars with proper type checking."""
    if not pillars:
        st.info("No content pillars identified in the analysis data.")
        return
    
    # Create a visual representation of content pillars
    num_cols = min(3, len(pillars))
    if num_cols > 0:
        pillar_cols = st.columns(num_cols)
        
        for i, pillar in enumerate(pillars):
            with pillar_cols[i % num_cols]:
                # Handle different pillar formats with type checking
                if isinstance(pillar, dict):
                    pillar_name = pillar.get('name', f'Pillar {i+1}')
                    pillar_desc = pillar.get('description', 'No description available')
                elif isinstance(pillar, str):
                    pillar_name = f'Pillar {i+1}'
                    pillar_desc = pillar
                else:
                    pillar_name = f'Pillar {i+1}'
                    pillar_desc = str(pillar) if pillar is not None else 'No description available'
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-left: 4px solid #4B8BBE; 
                           padding: 10px; margin-bottom: 10px; border-radius: 4px;">
                    <h4>{pillar_name}</h4>
                    <p>{pillar_desc}</p>
                </div>
                """, unsafe_allow_html=True)

# --- Main dashboard tab code ---
with tabs[2]:
    st.title("📊 Interactive Analysis Dashboard")
    
    # Add a custom CSS to improve the overall look
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
    .section-title {
        border-left: 4px solid #4B8BBE;
        padding-left: 10px;
        margin-bottom: 20px;
    }
    .highlight-text {
        background-color: #f0f7ff;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
    }
    .platform-card {
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease-in-out;
    }
    .platform-card:hover {
        box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4B8BBE;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Load Analysis Data ---
    analysis = st.session_state.get('analysis')
    if analysis is None and os.path.exists("social_media_analysis.json"):
        try:
            with open("social_media_analysis.json") as f:
                analysis = json.load(f)
                st.session_state['analysis'] = analysis
            st.success("✅ Loaded analysis from `social_media_analysis.json`")
        except Exception as e:
            st.error(f"Error loading analysis JSON: {e}")
            analysis = None # Ensure analysis is None if loading fails

    if not analysis:
        st.warning("📉 No analysis data found. Please run the analysis on the 'On-Boarding' tab first.")
        # Add a helpful button to navigate to the On-Boarding tab
        if st.button("Go to On-Boarding Tab"):
            st.switch_page("Your_App_Name.py")  # This will reload the app
        st.stop() # Stop execution for this tab if no data

    # --- Extract Key Data with schema compatibility ---
    try:
        # Extract basic information
        target_individual = analysis.get('targetIndividual', 'Unknown User')
        analyzed_platforms = analysis.get('analyzedPlatforms', [])
        platform_profiles = analysis.get('platformSpecificAnalysis', [])
        
        # Extract metadata if available in the old format
        analysis_meta = analysis.get('detailedUserProfileAnalysis', {}).get('analysisMetadata', {})
        if not analysis_meta:
            # Create metadata from available information in the new schema
            analysis_meta = {
                'analysisDate': datetime.datetime.now().strftime("%Y-%m-%d"),
                'targetName': target_individual
            }
        
        # Get content pillars and recommendations
        content_pillars = extract_content_pillars(analysis)
        recommendations = extract_recommendations(analysis)
        
        # Extract summary
        final_summary = analysis.get('finalComprehensiveSummary', 
                                   analysis.get('detailedUserProfileAnalysis', {}).get('finalSummary', 'Not Available'))
        
        # Extract cross platform synthesis
        cross_platform_synth = analysis.get('crossPlatformSynthesis', 
                                           analysis.get('detailedUserProfileAnalysis', {}).get('crossPlatformSynthesis', {}))
        
        # Extract algorithmic insights
        algo_insight = analysis.get('detailedUserProfileAnalysis', {}).get('algorithmicInsight', {})
        if not algo_insight and cross_platform_synth:
            # Try to get algorithmic insights from the new schema
            algo_perceptions = cross_platform_synth.get('inferredAlgorithmicPerception', [])
            if algo_perceptions:
                algo_insight = {
                    'feedContentObservations': 'See platform-specific algorithmic perceptions below',
                    'algorithmInterpretation': '\n'.join([
                        f"{ap.get('platformName', 'Platform')}: {ap.get('categorizationHypothesis', 'No data')}" 
                        for ap in algo_perceptions if isinstance(ap, dict)
                    ])
                }

        # Calculate key metrics for display
        content_pillars_count = len(content_pillars)
        platform_count = len(platform_profiles) if platform_profiles else 0
        recommendation_count = len(recommendations)
        
    except Exception as e:
        st.error(f"Error parsing analysis data structure: {e}")
        st.json(analysis) # Show the structure that caused the error
        st.stop()

    # --- Add a progress indicator for the dashboard ---
    st.markdown('<div class="section-title"><h3>📊 Dashboard Status</h3></div>', unsafe_allow_html=True)
    
    progress_cols = st.columns([3, 1])
    with progress_cols[0]:
        # Create a progress bar based on data completeness
        progress_value = min(1.0, (platform_count * 0.2) + (content_pillars_count * 0.1) + (recommendation_count * 0.05))
        st.progress(progress_value)
    with progress_cols[1]:
        st.markdown(f"<span class='highlight-text'>Score: {int(progress_value * 100)}%</span>", unsafe_allow_html=True)
    
    # ────────────────── 1. Top-Level Metrics Panel ──────────────────
    st.markdown('<div class="section-title"><h3>📈 Key Metrics Overview</h3></div>', unsafe_allow_html=True)
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <p>👤 Target Profile</p>
            <p class="stat-value">{}</p>
        </div>
        """.format(target_individual), unsafe_allow_html=True)
        
    with metric_cols[1]:
        st.markdown("""
        <div class="metric-card">
            <p>🌐 Platforms Analyzed</p>
            <p class="stat-value">{}</p>
        </div>
        """.format(platform_count), unsafe_allow_html=True)
        
    with metric_cols[2]:
        st.markdown("""
        <div class="metric-card">
            <p>📊 Content Pillars</p>
            <p class="stat-value">{}</p>
        </div>
        """.format(content_pillars_count), unsafe_allow_html=True)
        
    with metric_cols[3]:
        st.markdown("""
        <div class="metric-card">
            <p>✨ Recommendations</p>
            <p class="stat-value">{}</p>
        </div>
        """.format(recommendation_count), unsafe_allow_html=True)

    # --- Executive Summary Section ---
    with st.expander("📝 Executive Summary", expanded=True):
        # Add null check for final_summary
        summary_text = final_summary[:300] + "..." if final_summary and len(final_summary) > 300 else final_summary
        st.markdown(f"**Analysis Overview**: {summary_text}")
        
        # Add key recommendations in an easy-to-scan format
        st.markdown("### 🚀 Key Recommendations")
        if recommendations:
            for i, rec in enumerate(recommendations[:3]):  # Show top 3 recommendations
                st.markdown(f"**{i+1}.** {rec}")
            
            # Use a checkbox instead of nested expander
            if len(recommendations) > 3:
                show_more = st.checkbox("Show all recommendations")
                if show_more:
                    for i, rec in enumerate(recommendations[3:]):
                        st.markdown(f"**{i+4}.** {rec}")
        else:
            st.info("No specific recommendations found in the analysis data.")

    st.markdown("---")

    # ────────────────── Layout for Sidebar & Main Content ──────────────────
    dash_cols = st.columns([1, 3]) # Ratio for sidebar-like column and main content

    with dash_cols[0]: # Mini-Sidebar Column
        st.markdown('<div class="section-title"><h3>⚙️ Dashboard Controls</h3></div>', unsafe_allow_html=True)

        # --- 2. Sidebar Filters ---
        platform_names = ["All Platforms"]
        if platform_profiles:
            platform_names += [p.get('platformName', f'Platform {i+1}') for i, p in enumerate(platform_profiles)]
            
        selected_platform = st.selectbox(
            "Select Platform to View",
            options=platform_names,
            key='selected_platform' # Use session state key
        )

        st.markdown("### Display Options")
        
        # Use toggles with icons for better visual organization
        show_profile = st.toggle("👤 Profile Fundamentals", key='show_profile_fundamentals', value=True)
        show_content = st.toggle("📝 Content Activity", key='show_content_activity', value=True)
        show_network = st.toggle("🔄 Network & Community", key='show_network_community', value=True)
        show_algo = st.toggle("⚙️ Algorithmic Insights", key='show_algorithmic_insights', value=True)
        
        # Add export options to the sidebar
        st.markdown("### 💾 Export Options")
        
        export_format = st.radio("Format:", ["JSON", "PDF", "CSV"], horizontal=True)
        
        if st.button("Export Dashboard", type="primary"):
            st.success("Export functionality would be implemented here!")
            # This would be implemented in a full version
        
        # Add date filter (for a more complete dashboard)
        st.markdown("### 📅 Date Range")
        date_range = st.date_input("Filter data by date:", 
                                  [datetime.datetime.now() - datetime.timedelta(days=30),
                                   datetime.datetime.now()])

    # ────────────────── Main Content Area ──────────────────
    with dash_cols[1]:
        # Filter data based on selection
        if selected_platform == "All Platforms":
            profiles_to_display = platform_profiles if platform_profiles else []
            st.markdown('<div class="section-title"><h2>🌎 All Platforms View</h2></div>', unsafe_allow_html=True)
        else:
            profiles_to_display = [p for p in platform_profiles if p.get('platformName') == selected_platform] if platform_profiles else []
            if profiles_to_display:
                st.markdown(f'<div class="section-title"><h2>{selected_platform} Analysis</h2></div>', unsafe_allow_html=True)
            else:
                st.warning(f"No data found for platform: {selected_platform}")
                st.stop()

        # Add a quick stats bar for the selected view
        stats_cols = st.columns(3)
        with stats_cols[0]:
            # Extract engagement rate if available from platform data
            engagement_rate = "N/A"
            if profiles_to_display and len(profiles_to_display) == 1:
                # Try to calculate from profile data
                engagement_patterns = profiles_to_display[0].get('engagementPatterns', {})
                if engagement_patterns:
                    engagement_rate = "Active"  # Simple placeholder, would calculate from real data
            
            st.metric("Engagement Level", engagement_rate)
            
        with stats_cols[1]:
            # Get posting frequency from data when available
            frequency = "N/A"
            if profiles_to_display and len(profiles_to_display) == 1:
                content_act = profiles_to_display[0].get('contentGenerationActivity', {})
                frequency = content_act.get('postingFrequency', 'Unknown') if content_act else 'Unknown'
            elif len(profiles_to_display) > 1:
                frequency = "Various (see details)"
                
            st.metric("Posting Frequency", frequency)
            
        with stats_cols[2]:
            # Follower count from data
            follower_count = "N/A"
            if profiles_to_display and len(profiles_to_display) == 1:
                network = profiles_to_display[0].get('networkCommunity', {})
                if network:
                    follower_count = network.get('followerCount', 'N/A')
                    if follower_count is None:
                        follower_count = "N/A"
                        
            st.metric("Audience Size", follower_count)

        # --- 3. Platform Overview Card(s) ---
        if profiles_to_display:
            st.markdown('<div class="section-title"><h3>👤 Profile Overview</h3></div>', unsafe_allow_html=True)
            
            for profile in profiles_to_display:
                platform_name = profile.get('platformName', 'Unknown Platform')
                fundamentals = profile.get('profileFundamentals', {}) if profile else {}
                bio_analysis = profile.get('bioAnalysis', {}) if profile else {}
                content_act = profile.get('contentGenerationActivity', {}) if profile else {}
                network_comm = profile.get('networkCommunity', {}) if profile else {}

                # Improved card design with HTML/CSS - add null checks
                st.markdown(f"""
                <div class="platform-card">
                    <h4>{platform_name}</h4>
                    <p><em>@{fundamentals.get('username', 'N/A') if fundamentals else 'N/A'}</em></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create three columns for profile stats
                profile_cols = st.columns(3)
                
                with profile_cols[0]:
                    # Get bio text from bioAnalysis or fundamentals
                    bio_text = bio_analysis.get('fullText', 'N/A') if bio_analysis else 'N/A'
                    bio_snippet = bio_text[:100] + "..." if bio_text and len(bio_text) > 100 else bio_text
                    st.markdown("**Bio:**")
                    st.info(bio_snippet)
                    
                with profile_cols[1]:
                    # Engagement metrics
                    st.markdown("**Engagement:**")
                    
                    # Create engagement metrics visualization based on available data
                    engagement_data = profile.get('engagementPatterns', {})
                    
                    # Create a placeholder visualization if no detailed data
                    fig_engagement, ax_engagement = plt.subplots(figsize=(3, 2))
                    
                    # Determine metrics to display based on available data
                    if engagement_data:
                        metrics = []
                        values = []
                        
                        # Extract metrics that have values
                        if engagement_data.get('outgoingInteractionStyle'):
                            metrics.append('Outgoing')
                            values.append(0.7)  # Placeholder value
                            
                        if engagement_data.get('typesOfContentEngagedWith'):
                            metrics.append('Content Types')
                            values.append(0.5)  # Placeholder value
                            
                        if engagement_data.get('incomingEngagementHighlights'):
                            metrics.append('Incoming')
                            values.append(0.8)  # Placeholder value
                    else:
                        # Default placeholder
                        metrics = ['Outgoing', 'Incoming', 'Overall']
                        values = [0.5, 0.7, 0.6]  # Placeholder values
                    
                    # Create the visualization
                    bars = ax_engagement.barh(metrics, values, color=['#4B8BBE', '#306998', '#FFD43B'])
                    
                    # Add value labels to the bars
                    for bar in bars:
                        width = bar.get_width()
                        ax_engagement.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                                ha='left', va='center', fontsize=8)
                    
                    ax_engagement.set_xlim(0, 1)
                    ax_engagement.set_title("Engagement Metrics", fontsize=9)
                    ax_engagement.tick_params(axis='y', labelsize=8)
                    ax_engagement.tick_params(axis='x', labelsize=6)
                    
                    st.pyplot(fig_engagement)
                    plt.close(fig_engagement)
                    
                with profile_cols[2]:
                    # Posting data visualization
                    st.markdown("**Content Activity:**")
                    
                    # Create a posting activity chart based on available data
                    content_activity = profile.get('contentGenerationActivity', {})
                    
                    # Create a placeholder visualization if no detailed data
                    fig_activity, ax_activity = plt.subplots(figsize=(3, 2))
                    
                    # Use content types to create a visualization
                    content_types = content_activity.get('dominantContentTypes', []) if content_activity else []
                    
                    if content_types:
                        # Create a simplified bar chart showing content type distribution
                        content_counts = {}
                        for content_type in content_types[:4]:  # Limit to top 4
                            content_counts[content_type] = content_counts.get(content_type, 0) + 1
                            
                        # Sort by counts
                        labels = list(content_counts.keys())
                        values = list(content_counts.values())
                        
                        ax_activity.bar(labels, values, color='#4B8BBE')
                        ax_activity.set_title("Content Types", fontsize=9)
                        ax_activity.tick_params(axis='x', rotation=45, labelsize=7)
                        ax_activity.tick_params(axis='y', labelsize=8)
                    else:
                        # Default placeholder - posts by day
                        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                        posts = [3, 2, 4, 1, 5, 2, 0]  # Placeholder values
                        
                        ax_activity.plot(days, posts, marker='o', linestyle='-', color='#4B8BBE')
                        ax_activity.set_ylim(0, max(posts) + 1)
                        ax_activity.set_title("Posts by Day", fontsize=9)
                        ax_activity.tick_params(axis='both', labelsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig_activity)
                    plt.close(fig_activity)

                if show_profile:
                    with st.expander("View Profile Details"):
                        # A more organized display of profile details
                        st.markdown("#### Basic Information")
                        profile_detail_cols = st.columns(2)
                        
                        with profile_detail_cols[0]:
                            # Fix the NoneType error by ensuring fundamentals exists
                            if fundamentals:
                                st.markdown("**Username:** " + str(fundamentals.get('username', 'N/A')))
                                st.markdown("**Full Name:** " + str(fundamentals.get('fullName', 'N/A')))
                                st.markdown("**Location:** " + str(fundamentals.get('location', 'N/A')))
                            else:
                                st.markdown("**Username:** N/A")
                                st.markdown("**Full Name:** N/A")
                                st.markdown("**Location:** N/A")
                            
                        with profile_detail_cols[1]:
                            if fundamentals:
                                profile_url = fundamentals.get('profileURL', fundamentals.get('profileUrl', 'N/A'))
                                st.markdown("**Profile URL:** " + str(profile_url))
                                
                                # Handle linked websites
                                linked_sites = fundamentals.get('linkedWebsites', [])
                                if linked_sites:
                                    sites_text = ", ".join(linked_sites[:3])
                                    if len(linked_sites) > 3:
                                        sites_text += f" (+{len(linked_sites) - 3} more)"
                                    st.markdown("**Linked Sites:** " + sites_text)
                                else:
                                    st.markdown("**Linked Sites:** None")
                            else:
                                st.markdown("**Profile URL:** N/A")
                                st.markdown("**Linked Sites:** N/A")
                            
                        st.markdown("#### Bio Analysis")
                        if bio_analysis:
                            # Display full bio text
                            st.info(bio_analysis.get('fullText', 'No bio available'))
                            
                            # Display keywords if available
                            keywords = bio_analysis.get('identifiedKeywords', [])
                            if keywords:
                                st.markdown("**Keywords:** " + ", ".join(keywords))
                                
                            # Display tone if available
                            tone = bio_analysis.get('tone')
                            if tone:
                                st.markdown("**Tone:** " + tone)
                        else:
                            st.info("No bio analysis available")
                        
                        # Skills and Interests section
                        st.markdown("#### Skills & Interests")
                        skills = profile.get('skillsInterestsExpertise', [])
                        if skills:
                            # Create a more visual representation of skills
                            skill_cols = st.columns(3)
                            for i, skill in enumerate(skills[:9]):  # Limit to prevent overwhelming
                                with skill_cols[i % 3]:
                                    st.markdown(f"🔹 {skill}")
                                    
                            if len(skills) > 9:
                                show_more_skills = st.checkbox(f"Show all {len(skills)} skills", key=f"skills_{platform_name}")
                                if show_more_skills:
                                    skill_text = ", ".join(skills[9:])
                                    st.markdown(f"**Additional skills:** {skill_text}")
                        else:
                            st.info("No skills or interests listed")
                        
                        # Use checkbox instead of nested expander for raw data
                        show_raw_fundamentals = st.checkbox(f"Show Raw Data for {platform_name} Profile", key=f"raw_fund_{platform_name}")
                        if show_raw_fundamentals:
                            st.json(profile)

            st.markdown("---") # Separator after overview cards


        # --- 4. Visual Analytics Panels ---
        st.markdown('<div class="section-title"><h3>🎨 Visual Analytics</h3></div>', unsafe_allow_html=True)
        
        # Improved visual analytics with tabs for organization
        viz_tabs = st.tabs(["Content Types", "Themes", "Tone Analysis"])
        
        with viz_tabs[0]:  # Content Types Tab
            # Create content type visualization
            if profiles_to_display:
                content_types = []
                for profile in profiles_to_display:
                    content_act = profile.get('contentGenerationActivity', {})
                    if content_act:
                        types = content_act.get('dominantContentTypes', [])
                        if types:
                            content_types.extend(types)
                
                if content_types:
                    # Create a pie chart of content types
                    type_counts = {}
                    for content_type in content_types:
                        type_counts[content_type] = type_counts.get(content_type, 0) + 1
                    
                    # Create the visualization
                    fig_pie, ax_pie = plt.subplots(figsize=(8, 5))
                    wedges, texts, autotexts = ax_pie.pie(
                        type_counts.values(), 
                        labels=type_counts.keys(), 
                        autopct='%1.1f%%',
                        textprops={'fontsize': 9}
                    )
                    ax_pie.axis('equal')
                    plt.setp(autotexts, size=9, weight="bold")
                    
                    st.pyplot(fig_pie)
                    plt.close(fig_pie)
                    
                    # Add explanatory text
                    st.markdown("### Key Observations")
                    st.markdown("This chart shows the distribution of content types across the selected platform(s). " +
                              "The dominant content type can help guide content strategy.")
                else:
                    st.info("No content type data available for the selected platform.")
            else:
                st.info("No profile data available to analyze content types.")
                
        with viz_tabs[1]:  # Themes Tab
            # Create themes visualization
            if profiles_to_display:
                themes = []
                for profile in profiles_to_display:
                    content_act = profile.get('contentGenerationActivity', {})
                    if content_act:
                        profile_themes = content_act.get('recurringThemesTopics', [])
                        if profile_themes:
                            themes.extend(profile_themes)
                
                if themes:
                    # Create a word cloud of themes
                    try:
                        # Generate a word cloud
                        text = ' '.join(themes)
                        wordcloud = WordCloud(width=600, height=300, background_color='white', max_words=100).generate(text)
                        
                        # Display the word cloud
                        fig_cloud, ax_cloud = plt.subplots(figsize=(10, 5))
                        ax_cloud.imshow(wordcloud, interpolation='bilinear')
                        ax_cloud.axis('off')
                        
                        st.pyplot(fig_cloud)
                        plt.close(fig_cloud)
                        
                        # Add explanatory text
                        st.markdown("### Theme Analysis")
                        st.markdown("The word cloud highlights recurring themes in the content. " +
                                  "Larger words appear more frequently across posts.")
                    except Exception as e:
                        st.error(f"Error generating word cloud: {e}")
                        st.markdown("**Recurring Themes:**")
                        for theme in themes[:10]:  # Show top 10 themes
                            st.markdown(f"- {theme}")
                else:
                    st.info("No theme data available for the selected platform.")
            else:
                st.info("No profile data available to analyze themes.")
                
        with viz_tabs[2]:  # Tone Analysis Tab
            # Tone analysis visualization
            if profiles_to_display:
                tones = {}
                for profile in profiles_to_display:
                    platform_name = profile.get('platformName', 'Unknown')
                    content_act = profile.get('contentGenerationActivity', {})
                    if content_act:
                        tone = content_act.get('overallToneVoice')
                        if tone:
                            tones[platform_name] = tone
                    
                    # Also check bio analysis for tone
                    bio = profile.get('bioAnalysis', {})
                    if bio and not tones.get(platform_name):
                        tone = bio.get('tone')
                        if tone:
                            tones[platform_name] = tone
                
                if tones:
                    # Display tone comparison
                    if len(tones) > 1:
                        st.markdown("### Cross-Platform Tone Comparison")
                        for platform, tone in tones.items():
                            st.markdown(f"**{platform}:** {tone}")
                        
                        # Add insights
                        st.markdown("### Tone Consistency")
                        if cross_platform_synth:
                            consistency = cross_platform_synth.get('consistencyVsVariation', {})
                            if consistency:
                                tone_consistency = consistency.get('contentTonePersonaConsistency')
                                if tone_consistency:
                                    st.info(tone_consistency)
                    else:
                        # Single platform tone analysis
                        platform, tone = list(tones.items())[0]
                        st.markdown(f"### Tone Analysis: {platform}")
                        st.info(f"**Primary Tone:** {tone}")
                        
                        # Create a simple visualization of the tone attributes
                        # This is a placeholder - would need real tone attributes
                        tone_attributes = {
                            "Professional": 0.8,
                            "Casual": 0.3,
                            "Informative": 0.9,
                            "Persuasive": 0.6,
                            "Friendly": 0.7
                        }
                        
                        # Create the visualization
                        fig_tone, ax_tone = plt.subplots(figsize=(8, 4))
                        ax_tone.barh(
                            list(tone_attributes.keys()),
                            list(tone_attributes.values()),
                            color='#4B8BBE'
                        )
                        ax_tone.set_xlim(0, 1)
                        ax_tone.set_title(f"Tone Attributes: {platform}", fontsize=12)
                        
                        st.pyplot(fig_tone)
                        plt.close(fig_tone)
                else:
                    st.info("No tone data available for the selected platform.")
            else:
                st.info("No profile data available to analyze tone.")

        st.markdown("---")


        # --- 5. Deep-Dive Sections ---
        st.markdown('<div class="section-title"><h3>🔍 In-Depth Analysis</h3></div>', unsafe_allow_html=True)
        
        # Use tabs instead of multiple expanders for better organization
        if profiles_to_display:
            deep_dive_tabs = st.tabs(["Content Strategy", "Audience Insights", "Algorithm Analysis"])
            
            with deep_dive_tabs[0]:  # Content Strategy
                if show_content:
                    for profile in profiles_to_display:
                        platform_name = profile.get('platformName', 'Unknown Platform')
                        content_act = profile.get('contentGenerationActivity', {}) if profile else {}
                        
                        st.subheader(f"{platform_name} Content Strategy")
                        
                        # Create a more visual representation of content data
                        content_cols = st.columns(2)
                        
                        with content_cols[0]:
                            st.markdown("#### Posting Frequency")
                            st.info(content_act.get('postingFrequency', 'N/A') if content_act else 'N/A')
                            
                            st.markdown("#### Content Types")
                            types = content_act.get('dominantContentTypes', []) if content_act else []
                            if types:
                                for type in types:
                                    st.markdown(f"- {type}")
                            else:
                                st.markdown("No content types identified.")
                                
                        with content_cols[1]:
                            st.markdown("#### Content Examples")
                            examples = content_act.get('contentExamples', []) if content_act else []
                            if isinstance(examples, list) and examples:
                                for ex in examples[:3]:  # Limit to 3 examples
                                    st.markdown(f"- {ex}")
                                    
                                if len(examples) > 3:
                                    show_more_examples = st.checkbox(f"Show all {len(examples)} examples", key=f"examples_{platform_name}")
                                    if show_more_examples:
                                        for ex in examples[3:]:
                                            st.markdown(f"- {ex}")
                            else:
                                st.markdown("No content examples available.")
                        
                        # Check for platform-specific features
                        platform_features = profile.get('platformFeatureUsage', [])
                        if platform_features:
                            st.markdown("#### Platform Features Used")
                            feature_cols = st.columns(2)
                            for i, feature in enumerate(platform_features):
                                with feature_cols[i % 2]:
                                    if isinstance(feature, dict):
                                        feature_name = feature.get('featureName', 'Feature')
                                        usage = feature.get('usageDescription', 'N/A')
                                        st.markdown(f"**{feature_name}:** {usage}")
                                    else:
                                        st.markdown(f"- {feature}")
                        
                        # Use checkbox instead of nested expander
                        show_raw_content = st.checkbox(f"Show Raw Data for {platform_name} Content", key=f"raw_content_{platform_name}")
                        if show_raw_content and content_act:
                            st.json(content_act)
                        elif show_raw_content:
                            st.info("No content activity data available")
                else:
                    st.info("Enable 'Content Activity' in the dashboard controls to view content strategy insights.")
            
            with deep_dive_tabs[1]:  # Audience Insights
                if show_network:
                    for profile in profiles_to_display:
                        platform_name = profile.get('platformName', 'Unknown Platform')
                        network_comm = profile.get('networkCommunity', {}) if profile else {}
                        
                        st.subheader(f"{platform_name} Audience Insights")
                        
                        # Create a more visual audience representation
                        audience_cols = st.columns(2)
                        
                        with audience_cols[0]:
                            # Network size metrics
                            follower_count = network_comm.get('followerCount', 'N/A')
                            following_count = network_comm.get('followingCount', 'N/A')
                            
                            st.metric("Followers", follower_count)
                            st.metric("Following", following_count)
                            
                        with audience_cols[1]:
                            st.markdown("#### Audience Description")
                            audience_desc = network_comm.get('audienceDescription', network_comm.get('followerAudienceDescription', 'N/A'))
                            st.info(audience_desc if audience_desc else 'N/A')
                        
                        st.markdown("#### Groups & Communities")
                        groups = network_comm.get('groupCommunityMemberships', []) if network_comm else []
                        if groups:
                            # Create a table for better organization
                            group_data = []
                            for group in groups:
                                if isinstance(group, dict):
                                    group_data.append({
                                        "Group Name": group.get('groupName', 'Unnamed Group'),
                                        "Topic": group.get('topic', 'N/A'),
                                        "Activity Level": group.get('activityLevel', 'N/A')
                                    })
                                else:
                                    group_data.append({
                                        "Group Name": str(group),
                                        "Topic": "N/A",
                                        "Activity Level": "N/A"
                                    })
                            
                            if group_data:
                                st.dataframe(group_data, use_container_width=True)
                            else:
                                st.markdown("No detailed group data available.")
                        else:
                            st.markdown("No groups or communities identified.")
                        
                        # Use checkbox instead of nested expander
                        show_raw_network = st.checkbox(f"Show Raw Data for {platform_name} Network", key=f"raw_network_{platform_name}")
                        if show_raw_network and network_comm:
                            st.json(network_comm)
                        elif show_raw_network:
                            st.info("No network and community data available")
                else:
                    st.info("Enable 'Network & Community' in the dashboard controls to view audience insights.")
            
            with deep_dive_tabs[2]:  # Algorithm Analysis
                if show_algo:
                    st.subheader("Platform Algorithm Analysis")
                    
                    # Check for algorithmic insights in either format
                    algo_perceptions = cross_platform_synth.get('inferredAlgorithmicPerception', []) if cross_platform_synth else []
                    
                    if algo_perceptions:
                        # Display algorithm perceptions from the new schema
                        for perception in algo_perceptions:
                            if isinstance(perception, dict):
                                platform_name = perception.get('platformName', 'Platform')
                                hypothesis = perception.get('categorizationHypothesis', 'No data')
                                
                                st.markdown(f"#### {platform_name} Algorithm")
                                st.info(hypothesis)
                    elif algo_insight:
                        # Display algorithm insights from the old schema
                        algo_cols = st.columns(2)
                        
                        with algo_cols[0]:
                            st.markdown("#### Feed Content Observations")
                            st.info(algo_insight.get('feedContentObservations', 'N/A'))
                            
                            st.markdown("#### Suggestion Observations")
                            st.info(algo_insight.get('suggestionObservations', 'N/A'))
                        
                        with algo_cols[1]:
                            st.markdown("#### Advertisement Observations")
                            st.info(algo_insight.get('advertisementObservations', 'N/A'))
                            
                            st.markdown("#### Algorithm Interpretation")
                            st.info(algo_insight.get('algorithmInterpretation', 'N/A'))
                    else:
                        st.info("No algorithmic insight data available")
                    
                    # Show raw data if requested
                    show_raw_algo = st.checkbox("Show Raw Algorithmic Insight Data")
                    if show_raw_algo:
                        if algo_perceptions:
                            st.json(algo_perceptions)
                        elif algo_insight:
                            st.json(algo_insight)
                        else:
                            st.info("No algorithm data available")
                else:
                    st.info("Enable 'Algorithmic Insights' in the dashboard controls to view algorithm analysis.")

        st.markdown("---")


        # ────────────────── 6. Action Plan Section ──────────────────
        st.markdown('<div class="section-title"><h3>🎯 Action Plan</h3></div>', unsafe_allow_html=True)
        
        # Extract content pillars and create a more visual representation
        if content_pillars:
            st.markdown("### Content Pillars")
            
            # Use the safe display function
            display_content_pillars(content_pillars)
        else:
            st.info("No content pillars identified in the analysis.")
        
        # Extract recommended next steps
        if recommendations:
            st.markdown("### Next Steps")
            
            # Create a timeline visualization for next steps
            for i, step in enumerate(recommendations):
                if step:  # Ensure step is not None or empty
                    step_str = str(step)  # Convert to string to handle any type
                    st.markdown(f"""
                    <div style="display: flex; margin-bottom: 10px;">
                        <div style="background-color: #4B8BBE; color: white; border-radius: 50%; 
                                width: 30px; height: 30px; display: flex; justify-content: center; 
                                align-items: center; margin-right: 10px;">
                            {i+1}
                        </div>
                        <div style="flex-grow: 1; background-color: #f8f9fa; padding: 10px; border-radius: 4px;">
                            {step_str}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Action buttons in a single row
        action_cols = st.columns(3)
        
        with action_cols[0]:
            if st.button("🗓️ Create Content Calendar", use_container_width=True):
                st.success("This would generate a content calendar based on the analysis!")
        
        with action_cols[1]:
            if st.button("📊 Generate Performance Report", use_container_width=True):
                st.success("This would generate a detailed performance report!")
        
        with action_cols[2]:
            if st.button("📱 Create Platform Strategy", use_container_width=True):
                st.success("This would create a customized platform strategy!")

        # Final call to action
        st.markdown("---")
        st.markdown("### 🚀 Ready to implement your strategy?")
        if st.button("Begin Implementation", type="primary", use_container_width=True):
            st.balloons()
            st.success("Implementation mode activated! In a full version, this would guide you through implementing your strategy step-by-step.")


# ─────────────────────────── Footer ──────────────────────────
st.markdown("---")
st.caption("© 2025 ParselyFi — Enterprise Social Media Analytics powered by Streamlit + Gemini")