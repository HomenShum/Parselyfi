import streamlit as st
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import re # Import regex for parsing tags

# Set page config
st.set_page_config(page_title="Coffee Card Demo", layout="wide")

# --- CSS Styling (Reverted and adjusted) ---
def load_css():
    st.markdown("""
    <style>
    /* Main coffee card styling */
    .coffee-card {
        width: 350px; /* Adjust width as needed */
        border: 2px solid #6b4f4f;
        border-radius: 12px;
        padding: 16px;
        background: #fff8f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin: 16px auto; /* Center card if in a single column */
        display: flex;
        flex-direction: column;
        min-height: 450px; /* Maintain minimum height */
        justify-content: space-between; /* Push actions/summary down */
    }

     .card-content {
        flex-grow: 1; /* Allow content to take available space */
    }

    /* Header section */
    .card-header {
        display: flex;
        align-items: flex-start;
        margin-bottom: 16px;
    }
    .card-icon { font-size: 24px; margin-right: 10px; line-height: 1.2; }
    .card-title-container { flex-grow: 1; }
    .card-name { font-size: 1.2em; font-weight: bold; margin: 0; color: #4a3f3f; }
    .card-title { font-size: 0.9em; color: #555; margin-top: 2px; }

    /* Tag section styling */
    .tag-section { margin-bottom: 10px; }
    .tag-header { font-weight: bold; font-size: 0.9em; color: #6b4f4f; margin-bottom: 4px; }
    .tag-container { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; min-height: 20px; }
    .tag { background: #ffe8d6; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; display: inline-block; color: #6b4f4f; border: 1px solid #e6ccb2; }
    .tag-placeholder { font-style: italic; color: #999; font-size: 0.8em; }

    /* Experiences section */
    .exp-section { margin: 16px 0; }
    .experiences-display { border: 1px solid #e6ccb2; border-radius: 4px; min-height: 80px; padding: 8px; background: #fffaf5; white-space: pre-wrap; font-size: 0.9em; color: #333; line-height: 1.5; max-height: 150px; overflow-y: auto; }
    .experiences-placeholder { font-style: italic; color: #999; padding: 8px; text-align: center; }

    /* Progress bar and summary section */
    .progress-summary-container { margin-top: auto; padding-top: 12px; border-top: 1px solid #e6ccb2; } /* margin-top: auto pushes it down */
    .missing-fields-summary { font-size: 0.8em; color: #666; margin-bottom: 8px; }
    .missing-fields-summary h6 { margin-bottom: 4px; font-weight: bold; font-size: 0.9em; color: #6b4f4f; }
    .missing-fields-summary ul { margin: 0; padding-left: 20px; }
    .missing-fields-summary li { margin-bottom: 2px; }
    .progress-label { font-size: 0.8em; color: #555; margin-bottom: 4px; text-align: right; }
    .progress-bar-bg { width: 100%; background-color: #eee; border-radius: 4px; height: 8px; overflow: hidden; margin-bottom: 10px; }
    .progress-bar-fill { height: 100%; background-color: #b36b00; border-radius: 4px; }

    /* Action buttons container */
    .card-actions { display: flex; justify-content: flex-end; gap: 8px; padding-top: 10px; border-top: 1px dashed #e6ccb2; }

    /* Override Streamlit button defaults for card actions */
    .card-actions .stButton button {
        background-color: transparent;
        color: #b36b00; border: 1px solid #b36b00;
        padding: 2px 8px; font-size: 0.9em; border-radius: 4px;
    }
    .card-actions .stButton button:hover { background-color: #fff0e0; color: #8a5a00; border-color: #8a5a00; }
    /* Specific style for delete button */
    .card-actions .stButton[kind="secondary"] button { color: #d9534f; border-color: #d9534f; }
    .card-actions .stButton[kind="secondary"] button:hover { background-color: #fdf7f7; color: #c9302c; border-color: #ac2925; }

    /* Ensure dialog form elements have some space */
    .stDialog form > div { margin-bottom: 10px; }

    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization (Modified) ---
def init_session_state():
    if 'profiles' not in st.session_state:
        st.session_state.profiles = [{
            "id": str(uuid.uuid4()), "name": "Homen Shum",
            "title": "AI Workflow Engineer • SF Bay Area",
            "interests": ["Reinforcement Learning", "Cycling", "Meme-Lab"],
            "hobbies": ["Sourdough Baking", "DeepRacer Racing"],
            "skills": ["Python", "Streamlit", "Qdrant", "Gemini API"],
            "experiences": "• LLM RAG developer @ ParselyFi ('24–)\n• Workflow automation @ JPMC ('23–'24)\n• Top-5 ML finalist – SB Hack VII ('21)",
        }, {
            "id": str(uuid.uuid4()), "name": "New Profile",
            "title": "Job Title • Location", "interests": ["Placeholder Interest"],
            "hobbies": [], "skills": [], "experiences": "",
        }]
    # State variable to control the edit dialog
    if 'editing_profile_id_dialog' not in st.session_state:
        st.session_state.editing_profile_id_dialog = None

# --- Profile Completion Calculation (No Changes Needed) ---
def calculate_profile_completion(profile: Dict[str, Any]) -> Tuple[int, List[str]]:
    DEFAULT_NAME = "New Profile"
    DEFAULT_TITLE = "Job Title • Location"
    DEFAULT_EXPERIENCES = ""
    total_sections = 6
    completed_sections = 0
    missing_fields_summary: List[str] = []

    name = profile.get("name", "").strip()
    if name and name != DEFAULT_NAME: completed_sections += 1
    else: missing_fields_summary.append("Update 'Name'")

    title = profile.get("title", "").strip()
    if title and title != DEFAULT_TITLE: completed_sections += 1
    else: missing_fields_summary.append("Update 'Title'")

    if profile.get("interests"): completed_sections += 1
    else: missing_fields_summary.append("Add 'Interests'")

    if profile.get("hobbies"): completed_sections += 1
    else: missing_fields_summary.append("Add 'Hobbies'")

    if profile.get("skills"): completed_sections += 1
    else: missing_fields_summary.append("Add 'Skills'")

    if profile.get("experiences", "").strip() != DEFAULT_EXPERIENCES: completed_sections += 1
    else: missing_fields_summary.append("Add 'Experiences'")

    percentage = int((completed_sections / total_sections) * 100) if total_sections > 0 else 0
    return percentage, missing_fields_summary

# --- Profile/Tag Action Handlers (Modified/Simplified) ---
def add_profile():
    new_profile = { "id": str(uuid.uuid4()), "name": "New Profile", "title": "Job Title • Location", "interests": [], "hobbies": [], "skills": [], "experiences": "" }
    st.session_state.profiles.append(new_profile)
    st.rerun()

def delete_profile(profile_id):
    st.session_state.profiles = [p for p in st.session_state.profiles if p["id"] != profile_id]
    # If deleting the profile being edited in dialog, close dialog
    if st.session_state.editing_profile_id_dialog == profile_id:
        st.session_state.editing_profile_id_dialog = None
    st.rerun()

def update_profile(profile_id, updates: Dict[str, Any]):
    """Updates multiple fields of a profile."""
    profile_updated = False
    for i, profile in enumerate(st.session_state.profiles):
        if profile["id"] == profile_id:
            # Check each field in updates dict
            for field, value in updates.items():
                 if profile.get(field) != value:
                     st.session_state.profiles[i][field] = value
                     profile_updated = True
            break # Found profile
    # Rerun will be handled by the calling function (dialog form submission/cancel)

# --- Edit Dialog Handlers ---
def open_edit_dialog(profile_id: str):
    """Sets the state to open the dialog for the given profile."""
    st.session_state.editing_profile_id_dialog = profile_id
    # No rerun here, the button click handles it

def close_edit_dialog():
    """Closes the edit dialog."""
    st.session_state.editing_profile_id_dialog = None
    # Rerun needed to make dialog disappear
    st.rerun()

def parse_tags(tag_string: str) -> List[str]:
    """Parses a comma or newline separated string into a list of unique, stripped tags."""
    # Split by comma or newline, handle potential multiple separators
    tags = re.split(r'[,\n]+', tag_string)
    # Strip whitespace and filter out empty strings, ensure uniqueness
    unique_tags = list(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))
    return unique_tags

# --- RENDER COFFEE CARD (Complete function with fixes) ---
def render_coffee_card(profile: Dict):
    """
    Renders a single profile card using st.markdown for HTML structure and CSS.
    Includes profile info, tags, experiences, completion progress, and summary.
    Streamlit action buttons (Edit, Delete) are rendered *after* the markdown block.

    Args:
        profile: A dictionary representing the user profile data.
    """
    profile_id = profile["id"]
    # Assumes calculate_profile_completion is defined elsewhere
    completion_percentage, missing_fields = calculate_profile_completion(profile)

    # Helper function to generate HTML for tags or a placeholder, includes html.escape
    def generate_tags_html(tags: List[str], placeholder: str) -> str:
        if not tags:
            return f'<span class="tag-placeholder">{placeholder}</span>'
        # Ensure tags themselves are HTML-safe before inserting into HTML
        safe_tags = [tag for tag in tags]
        return "".join(f'<span class="tag">{tag}</span>' for tag in safe_tags)

    # Build the HTML list for the missing fields summary
    summary_list_html = ""
    if missing_fields:
        # Field names ('Name', 'Title', etc.) are controlled by us, generally safe
        for field in missing_fields:
            summary_list_html += f"<li>{field}</li>"
    else:
        summary_list_html = "<li>All fields complete! ✔️</li>"

    # Prepare experiences content for display
    experiences_content = profile.get("experiences", "").strip()
    # Rely on 'white-space: pre-wrap' CSS for formatting (like bullet points).
    # No html.escape() here to allow basic formatting to render.
    # If input source isn't trusted or could contain malicious HTML, escaping is crucial.
    safe_experiences_content = experiences_content if experiences_content else '<p class="experiences-placeholder">No experiences added</p>'


    # Construct the entire card HTML string.
    # Key fixes:
    # 1. Escape literal '%' characters as '%%' in f-string formatting.
    # 2. Place variables directly adjacent to tags (e.g., >{var}<) in sections
    #    using 'white-space: pre-wrap' to avoid unwanted leading whitespace.
    # 3. Use html.escape() for potentially user-controlled text fields like name/title/tags.
    card_html = f"""
    <div class="coffee-card" id="card-{profile_id}">
        <div class="card-content"> <!-- Main content area -->
            <div class="card-header">
                <div class="card-icon">☕</div>
                <div class="card-title-container">
                    <h3 class="card-name">{profile.get('name', 'N/A')}</h3>
                    <p class="card-title">{profile.get('title', 'N/A')}</p>
                </div>
            </div>
            <div class="tag-section">
                <div class="tag-header">Interests:</div>
                <div class="tag-container">
                    {generate_tags_html(profile.get("interests", []), "No interests added")}
                </div>
            </div>
            <div class="tag-section">
                <div class="tag-header">Hobbies:</div>
                <div class="tag-container">
                    {generate_tags_html(profile.get("hobbies", []), "No hobbies added")}
                </div>
            </div>
            <div class="tag-section">
                <div class="tag-header">Skills:</div>
                <div class="tag-container">
                    {generate_tags_html(profile.get("skills", []), "No skills added")}
                </div>
            </div>
            <div class="exp-section">
                <div class="tag-header">Experiences:</div>
                <div class="experiences-display">{safe_experiences_content}</div>
            </div>
        </div> <!-- End card-content -->

        <div class="progress-summary-container">
             <div class="progress-label">{completion_percentage}%% Complete</div>
             <div class="progress-bar-bg">
                 <div class="progress-bar-fill" style="width: {completion_percentage}%%;"></div>
             </div>
             <div class="missing-fields-summary">
                 <h6>Profile Checklist:</h6>
                 <ul>{summary_list_html}</ul>
             </div>
        </div>

        <div class="card-actions">
        </div>
    </div> <!-- End coffee-card -->
    """
    # Render the generated HTML structure using st.markdown
    st.markdown(card_html, unsafe_allow_html=True)

    # Render Streamlit action buttons separately, positioned visually near the card actions area
    # They are not direct DOM children of the .card-actions div created above.
    action_cols = st.columns([0.75, 0.125, 0.125]) # Ratios: Spacer | Edit Button | Delete Button
    with action_cols[1]:
         # Assumes open_edit_dialog function is defined elsewhere
         st.button("✏️", key=f"edit_{profile_id}", help="Edit Profile",
                   on_click=open_edit_dialog, args=(profile_id,),
                   use_container_width=True)
    with action_cols[2]:
         # Assumes delete_profile function is defined elsewhere
         st.button("🗑️", key=f"delete_{profile_id}", help="Delete Profile",
                   on_click=delete_profile, args=(profile_id,),
                   use_container_width=True, type="secondary")


# --- RENDER EDIT DIALOG ---
def render_edit_dialog(profile: Dict[str, Any]):
    profile_id = profile["id"]

    @st.dialog("Edit Profile") # Use decorator form
    def edit_profile_dialog():
        st.subheader(f"Editing: {profile.get('name', 'Profile')}")

        with st.form(key=f"edit_profile_form_{profile_id}"):
            # --- Form Fields ---
            name = st.text_input("Name", value=profile.get("name", ""))
            title = st.text_input("Title", value=profile.get("title", ""))

            interests_str = st.text_area(
                "Interests",
                value=", ".join(profile.get("interests", [])), # Join tags for editing
                help="Enter interests separated by commas or newlines."
            )
            hobbies_str = st.text_area(
                "Hobbies",
                value=", ".join(profile.get("hobbies", [])), # Join tags for editing
                help="Enter hobbies separated by commas or newlines."
            )
            skills_str = st.text_area(
                "Skills",
                value=", ".join(profile.get("skills", [])), # Join tags for editing
                help="Enter skills separated by commas or newlines."
            )

            experiences = st.text_area(
                "Experiences",
                value=profile.get("experiences", ""),
                height=200,
                help="Enter work experience, projects, etc. Use bullet points (*) or dashes (-) for lists."
            )

            # --- Form Submission ---
            submitted = st.form_submit_button("💾 Save Changes")
            if submitted:
                # Parse tags from text areas
                updated_interests = parse_tags(interests_str)
                updated_hobbies = parse_tags(hobbies_str)
                updated_skills = parse_tags(skills_str)

                # Prepare updates dictionary
                updates = {
                    "name": name.strip(),
                    "title": title.strip(),
                    "interests": updated_interests,
                    "hobbies": updated_hobbies,
                    "skills": updated_skills,
                    "experiences": experiences.strip()
                }

                # Apply updates
                update_profile(profile_id, updates)

                # Close dialog and rerun automatically via state change
                st.session_state.editing_profile_id_dialog = None
                st.success("Profile updated!")
                st.rerun() # Explicit rerun after successful save might be needed

        # Cancel button outside the form
        if st.button("✖️ Cancel", key=f"cancel_dialog_{profile_id}", type="secondary"):
             close_edit_dialog() # Closes dialog and reruns

    # This line actually triggers the dialog display if the state is correct
    # (The function decorated with @st.dialog runs when called)
    edit_profile_dialog()


# --- Main App ---
def main():
    load_css()
    init_session_state()

    st.title("☕ Coffee Card Profiles")
    st.caption("Create, view, and edit simple profile cards.")

    if st.button("➕ Add New Profile Card"):
        add_profile() # Handles rerun

    st.markdown("---")

    # Display all profiles
    if not st.session_state.profiles:
        st.info("No profiles yet. Click 'Add New Profile Card' to start.")
    else:
        num_columns = 2 # Adjust as needed
        cols = st.columns(num_columns)
        col_idx = 0
        profile_to_edit = None # Variable to hold the profile if dialog needs to be shown

        sorted_profiles = st.session_state.profiles # Or sort if needed

        for profile in sorted_profiles:
             with cols[col_idx % num_columns]:
                 # Render the card display (HTML via markdown)
                 render_coffee_card(profile) # This now includes the Streamlit action buttons too

                 # Check if this profile is the one to be edited in the dialog
                 if st.session_state.editing_profile_id_dialog == profile['id']:
                     profile_to_edit = profile

                 # Add vertical space between cards in the same column
                 st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

             col_idx += 1

        # --- Render Edit Dialog (if applicable) ---
        # This needs to be called *outside* the column rendering loop
        # to ensure the dialog appears correctly over the whole page.
        if profile_to_edit:
            render_edit_dialog(profile_to_edit)


if __name__ == "__main__":
    main()