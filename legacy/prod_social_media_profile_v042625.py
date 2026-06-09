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

# --- Tab: Dashboard ---
with tabs[2]:
    st.title("📊 Interactive Analysis Dashboard")
    
    # Add a custom CSS to improve the overall look
    st.markdown("""
    <style>
    .dashboard-container {
        max-width: 100%;
        margin: 0 auto;
    }
    .upload-section {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
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
            analysis = None

    # HTML Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📄 HTML Report Display")
    
    # Option 1: Upload HTML file
    uploaded_html = st.file_uploader("Upload an HTML report file", type=["html"], key="html_uploader")
    
    # Option 2: Use the generated report (if available from analysis)
    use_generated = st.checkbox("Use HTML template with current analysis data", value=True)
    
    if uploaded_html is not None:
        # Display uploaded HTML
        html_content = uploaded_html.getvalue().decode("utf-8")
        st.html(html_content)
        
        # Download option for the uploaded file
        st.download_button(
            "💾 Download HTML Report", 
            html_content,
            "social_media_report.html", 
            "text/html"
        )
        
    elif use_generated and analysis:
        # Generate a simplified HTML report from the analysis data
        try:
            # Extract key data from analysis
            target_individual = analysis.get('targetIndividual', 'Unknown User')
            analyzed_platforms = analysis.get('analyzedPlatforms', [])
            platform_profiles = analysis.get('platformSpecificAnalysis', [])
            
            # Create a basic HTML report
            html_report = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Social Media Analysis: {target_individual}</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        line-height: 1.6;
                        margin: 0;
                        padding: 20px;
                        background-color: #f8f9fa;
                        color: #333;
                    }}
                    .report-container {{
                        max-width: 950px;
                        margin: 20px auto;
                        background-color: #fff;
                        padding: 30px;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    h1, h2, h3, h4 {{
                        color: #0056b3;
                        margin-top: 1.5em;
                        margin-bottom: 0.5em;
                    }}
                    h1 {{
                        text-align: center;
                        border-bottom: 2px solid #e9ecef;
                        padding-bottom: 0.5em;
                        margin-top: 0;
                    }}
                    .platform-card {{
                        border: 1px solid #e9ecef;
                        padding: 20px;
                        margin-top: 20px;
                        border-radius: 6px;
                        background-color: #ffffff;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                    }}
                    .platform-card h3 {{
                        margin-top: 0;
                        color: #28a745;
                        border-bottom: 1px dashed #ccc;
                        padding-bottom: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="report-container">
                    <h1>Social Media Analysis Report</h1>
                    
                    <h2>Target Individual</h2>
                    <p><strong>Name:</strong> {target_individual}</p>
                    
                    <h2>Analyzed Platforms</h2>
                    <ul>
                        {"".join([f"<li>{platform}</li>" for platform in analyzed_platforms])}
                    </ul>
                    
                    <h2>Platform Analysis</h2>
            """
            
            # Add platform cards
            for platform in platform_profiles:
                platform_name = platform.get('platformName', 'Unknown Platform')
                
                # Get platform-specific data
                fundamentals = platform.get('profileFundamentals', {})
                bio = platform.get('bioAnalysis', {}).get('fullText', 'No bio available')
                
                html_report += f"""
                    <div class="platform-card">
                        <h3>{platform_name}</h3>
                        <p><strong>Username:</strong> {fundamentals.get('username', 'N/A')}</p>
                        <p><strong>Bio:</strong> {bio}</p>
                    </div>
                """
            
            # Close the HTML
            html_report += """
                </div>
            </body>
            </html>
            """
            
            # Display the generated HTML
            st.html(html_report)
            
            # Download option
            st.download_button(
                "💾 Download HTML Report", 
                html_report,
                "social_media_report.html", 
                "text/html"
            )
            
        except Exception as e:
            st.error(f"Error generating HTML report: {e}")
            # Show a JSON viewer as fallback
            st.json(analysis)
    
    elif not analysis:
        st.warning("📉 No analysis data found. Please run the analysis on the 'On-Boarding' tab first.")
        # Add a helpful button to navigate to the On-Boarding tab
        if st.button("Go to On-Boarding Tab"):
            st.switch_page("Your_App_Name.py")  # This will reload the app
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add an option to view the raw JSON data
    if analysis:
        with st.expander("View Raw JSON Data"):
            st.json(analysis)

# ─────────────────────────── Footer ──────────────────────────
st.markdown("---")
st.caption("© 2025 ParselyFi — Enterprise Social Media Analytics powered by Streamlit + Gemini")