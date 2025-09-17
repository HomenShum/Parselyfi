# -*- coding: utf-8 -*-
import streamlit as st
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import re # Import regex for parsing tags
import html # Import html for escaping
import base64

# --- CSS Styling (Copied from your provided code in app_new.py) ---
# This CSS defines the target look and feel for the generated HTML
CSS_STYLES = """
<style>
:root {
    /* --- Coffee Card Theme Palette --- */
    --cc-bg-main: #fff8f0;             /* Main card background */
    --cc-bg-exp: #fffaf5;              /* Experiences box background */
    --cc-bg-tag: #ffe8d6;              /* Tag background */
    --cc-bg-progress-track: #eee;      /* Progress bar track (light gray) */
    --cc-bg-btn-delete-hover: #fdf7f7; /* Delete button hover background */
    --cc-bg-btn-default-hover: #fff0e0; /* Default button hover background */

    --cc-accent-dark-brown: #6b4f4f;   /* Main border, tag headers/text */
    --cc-accent-theme-brown: #b36b00;  /* Progress fill, default button */
    --cc-accent-light-tan: #e6ccb2;   /* Tag/Exp border, separators */

    --cc-text-name: #4a3f3f;           /* Card name text */
    --cc-text-title: #555;             /* Card title/subtitle text */
    --cc-text-exp: #333;               /* Experiences text */
    --cc-text-tag: var(--cc-accent-dark-brown); /* Tag text (same as dark brown accent) */
    --cc-text-progress: #555;         /* Progress label text */
    --cc-text-placeholder: #999;       /* Placeholder text */
    --cc-text-missing-summary: #666;   /* Missing fields summary text */
    --cc-text-general: #37352F;        /* General body text (fallback) */
    --cc-text-secondary: #6B6B6B;      /* Secondary text (fallback) */

    --cc-btn-delete-text: #d9534f;     /* Delete button text/border */
    --cc-btn-delete-hover-text: #c9302c; /* Delete button hover text */
    --cc-btn-delete-hover-border: #ac2925; /* Delete button hover border */
    --cc-btn-default-text: var(--cc-accent-theme-brown); /* Default button text/border */
    --cc-btn-default-hover-text: #8a5a00; /* Default button hover text */
    --cc-btn-default-hover-border: #8a5a00; /* Default button hover border */
}

/* Card styling - Apply Coffee Card Theme to our generated card */
.coffee-card-generated { /* Using a specific class for the generated card */
    border: 2px solid var(--cc-accent-dark-brown);
    border-radius: 12px;
    background: var(--cc-bg-main);
    padding: 1.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 3px, rgba(0, 0, 0, 0.05) 0px 20px 25px -5px, rgba(0, 0, 0, 0.04) 0px 10px 10px -5px;
    transition: transform 0.2s ease-in-out, box-shadow 0.3s ease-in-out;
    display: flex;
    flex-direction: column;
    min-height: 450px; /* Or adjust based on content */
    justify-content: space-between; /* Push progress/summary down */
    width: 100%; /* Make it take container width by default */
    max-width: 400px; /* Limit max width */
    margin-left: auto;
    margin-right: auto; /* Center in column */
}

.coffee-card-generated:hover {
    transform: translateY(-3px);
    box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
}

/* Top accent bar */
.coffee-card-generated::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 8px;
    background: var(--cc-accent-dark-brown);
    opacity: 0.9;
    border-top-left-radius: 10px; /* Match card radius */
    border-top-right-radius: 10px;
}

.cc-card-content { /* Container for main content above progress/summary */
    padding-top: 8px; /* Space below the accent bar */
    flex-grow: 1;
}

/* Header styling */
.cc-header-content {
    display: flex;
    align-items: flex-start;
    gap: 1rem; 
    margin-bottom: 1rem;
}

img.cc-avatar {
    width: 80px; 
    height: 80px;
    border-radius: 50%;
    border: 2px solid var(--cc-accent-light-tan);
    object-fit: cover;
    flex-shrink: 0;
    box-shadow: rgba(0, 0, 0, 0.05) 0px 2px 4px;
}

.cc-header-text h1.cc-name { 
    color: var(--cc-text-name);
    margin: 0 0 0.2rem 0;
    font-size: 1.4em; 
    font-weight: bold;
}

.cc-header-text p.cc-title { 
    font-size: 1em;
    color: var(--cc-text-title);
    margin: 0.2rem 0 0.5rem 0;
    line-height: 1.4;
}
.cc-header-text p.cc-tagline {
    font-size: 0.9em;
    color: var(--cc-text-secondary);
    margin: 0.3rem 0 0.5rem 0;
    font-style: italic;
    line-height: 1.3;
}
.cc-header-text p.cc-location {
    font-size: 0.85em;
    color: var(--cc-text-secondary);
    margin: 0.1rem 0 0.3rem 0;
    line-height: 1.3;
}
.cc-header-text p.cc-profile-url a {
    font-size: 0.8em;
    color: var(--cc-accent-theme-brown);
    text-decoration: none;
}
.cc-header-text p.cc-profile-url a:hover {
    text-decoration: underline;
}


/* Section styling */
.cc-section {
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--cc-accent-light-tan);
}
.cc-section:last-of-type { /* Ensure last section doesn't have bottom border/padding */
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}

/* Pill/Tag Styling */
h5.cc-section-header { 
    color: var(--cc-accent-dark-brown);
    font-weight: bold;
    font-size: 0.95em;
    margin-bottom: 0.5rem;
    margin-top: 0; 
    display: flex;
    align-items: center;
}
.cc-icon { margin-right: 0.4rem; opacity: 0.9; font-size: 1.1em; }

.cc-pill-container {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
    display: flex;
    flex-wrap: wrap;
    gap: 6px; 
    min-height: 24px; 
}

/* Expandable Pill Container */
.cc-pill-container.collapsed {
    max-height: 58px; /* Approx 2 rows, adjust based on pill height and gap */
    overflow: hidden;
    position: relative;
}
.cc-pill-container.collapsed::after {
    content: '... expand to see more';
    position: absolute;
    bottom: 0;
    right: 0;
    background: linear-gradient(to right, transparent, var(--cc-bg-main) 50%);
    padding: 0.3rem 0.5rem 0.3rem 1.5rem;
    font-size: 0.75em;
    color: var(--cc-text-secondary);
    cursor: pointer;
}
/* We won't implement JS click here, but this sets up the visual cue */


.cc-pill {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    margin: 0; 
    border-radius: 12px;
    background: var(--cc-bg-tag);
    font-size: 0.8em;
    color: var(--cc-text-tag);
    font-weight: 500;
    line-height: 1.3;
    white-space: nowrap;
    border: 1px solid var(--cc-accent-light-tan);
}
.cc-pill:hover { 
     background: #f7e1ce;
}

.cc-pill-placeholder {
    font-style: italic;
    color: var(--cc-text-placeholder);
    font-size: 0.8em;
    padding: 0.3rem 0; 
}

/* Experiences / Education / Projects Item Styling */
.cc-item-list {
    margin-top: 0.5rem;
}
.cc-item {
    background: var(--cc-bg-exp);
    padding: 0.8rem;
    border-radius: 4px;
    border: 1px solid var(--cc-accent-light-tan);
    margin-bottom: 0.8rem;
    font-size: 0.9em;
}
.cc-item:last-child {
    margin-bottom: 0;
}
.cc-item-header {
    font-weight: bold;
    color: var(--cc-text-exp);
    font-size: 1em; /* Relative to .cc-item font-size */
}
.cc-item-subheader {
    font-size: 0.9em;
    color: var(--cc-text-secondary);
    margin-bottom: 0.3rem;
}
.cc-item-dates {
    font-size: 0.8em;
    color: var(--cc-text-secondary);
    font-style: italic;
    margin-bottom: 0.5rem;
}
.cc-item-description {
    font-size: 0.95em;
    line-height: 1.5;
    color: var(--cc-text-exp);
    white-space: pre-wrap; /* Preserve newlines */
    /* For expandability (conceptual) */
    max-height: 70px; /* Initial collapsed height */
    overflow: hidden;
    position: relative;
    cursor: pointer; /* Indicate it's clickable */
    transition: max-height 0.3s ease-out;
}
.cc-item-description.expanded {
    max-height: 1000px; /* Large enough for full content */
}
.cc-item-description:not(.expanded)::after {
    content: '... expand to read more';
    position: absolute;
    bottom: 0;
    right: 0;
    background: linear-gradient(to right, transparent, var(--cc-bg-exp) 50%);
    padding: 0.2rem 0.5rem 0.2rem 1.5rem;
    font-size: 0.8em;
    color: var(--cc-text-secondary);
}
.cc-item-skills-header {
    font-size: 0.85em;
    font-weight: bold;
    color: var(--cc-accent-dark-brown);
    margin-top: 0.6rem;
    margin-bottom: 0.3rem;
}
.cc-item-skill-pill {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    margin-right: 4px;
    margin-bottom: 4px;
    border-radius: 10px;
    background: var(--cc-bg-tag);
    font-size: 0.75em;
    color: var(--cc-text-tag);
    border: 1px solid var(--cc-accent-light-tan);
    cursor: help; /* Indicate popup capability */
}
.cc-item-skill-pill:hover {
    background: #f7e1ce;
}
.cc-item-project-url a {
    font-size: 0.85em;
    color: var(--cc-accent-theme-brown);
    text-decoration: none;
}
.cc-item-project-url a:hover {
    text-decoration: underline;
}

.cc-list-placeholder {
    font-style: italic;
    color: var(--cc-text-placeholder);
    padding: 0.8rem;
    text-align: center;
    background: var(--cc-bg-exp);
    border: 1px dashed var(--cc-accent-light-tan); 
    border-radius: 4px;
    min-height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 0.5rem;
}

/* Progress bar and summary section */
.cc-progress-summary-container {
    margin-top: auto; 
    padding-top: 1rem;
    border-top: 1px solid var(--cc-accent-light-tan);
}
.cc-missing-fields-summary { font-size: 0.8em; color: var(--cc-text-missing-summary); margin-bottom: 0.5rem; }
.cc-missing-fields-summary h6 { margin-bottom: 4px; font-weight: bold; font-size: 0.9em; color: var(--cc-accent-dark-brown); }
.cc-missing-fields-summary ul { margin: 0; padding-left: 18px; list-style-type: '☕ '; } 
.cc-missing-fields-summary li { margin-bottom: 2px; }

.cc-progress-label { font-size: 0.8em; color: var(--cc-text-progress); margin-bottom: 4px; text-align: right; }
.cc-progress-bar-bg { width: 100%; background-color: var(--cc-bg-progress-track); border-radius: 4px; height: 8px; overflow: hidden; margin-bottom: 8px; }
.cc-progress-bar-fill { height: 100%; background-color: var(--cc-accent-theme-brown); border-radius: 4px; transition: width 0.5s ease-in-out; }

/* Action buttons container */
.cc-card-actions { display: flex; justify-content: flex-end; gap: 8px; padding-top: 10px; border-top: 1px dashed var(--cc-accent-light-tan); }

/* --- Streamlit Button Overrides (Keep these) --- */
div[data-testid="stForm"] .stButton button:not([kind="secondary"]),
.stButton button:not([kind="secondary"]) { 
    border: 1px solid var(--cc-btn-default-text) !important;
    color: var(--cc-btn-default-text) !important;
    background-color: transparent !important;
    border-radius: 4px !important;
    padding: 0.3rem 0.7rem !important; 
    font-size: 0.9em !important; 
}

div[data-testid="stForm"] .stButton button:not([kind="secondary"]):hover,
.stButton button:not([kind="secondary"]):hover {
    border-color: var(--cc-btn-default-hover-border) !important;
    color: var(--cc-btn-default-hover-text) !important;
    background-color: var(--cc-bg-btn-default-hover) !important;
}

div[data-testid="stForm"] .stButton button[kind="secondary"],
.stButton button[kind="secondary"] {
    border: 1px solid var(--cc-btn-delete-text) !important;
    color: var(--cc-btn-delete-text) !important;
    background-color: transparent !important;
    border-radius: 4px !important;
    padding: 0.3rem 0.7rem !important;
    font-size: 0.9em !important;
}

div[data-testid="stForm"] .stButton button[kind="secondary"]:hover,
.stButton button[kind="secondary"]:hover {
    border-color: var(--cc-btn-delete-hover-border) !important;
    color: var(--cc-btn-delete-hover-text) !important;
    background-color: var(--cc-bg-btn-delete-hover) !important;
}

.stDialog form > div { margin-bottom: 10px; }

</style>
"""
def load_css():
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

# --- Helper Functions (Assume these exist and work as intended) ---
def calculate_profile_completion_new(profile_data: Dict[str, Any]) -> Tuple[int, List[str]]:
    # This needs to be adapted for the new 'yourCoffeeCard' structure
    if not profile_data:
        return 0, ["Profile data missing"]

    total_fields = 10 # Estimate: name, title, tagline, skills, experiences, education, projects, interests, hobbies, profile_url
    completed_fields = 0
    missing_summary = []

    # Mini-card essentials
    if profile_data.get("name"): completed_fields +=1
    else: missing_summary.append("Name")
    if profile_data.get("title"): completed_fields +=1
    else: missing_summary.append("Title")
    if profile_data.get("taglineOrBriefSummary"): completed_fields +=1
    else: missing_summary.append("Tagline/Summary")
    if profile_data.get("primaryProfileUrlForCard"): completed_fields +=1
    # else: missing_summary.append("Primary Profile URL") # Optional for completion score

    # Core content sections
    if profile_data.get("skills"): completed_fields +=1
    else: missing_summary.append("Skills")
    if profile_data.get("experiences"): completed_fields +=1
    else: missing_summary.append("Experiences")
    if profile_data.get("education"): completed_fields +=1
    # else: missing_summary.append("Education") # Can be optional
    if profile_data.get("projects"): completed_fields +=1
    # else: missing_summary.append("Projects") # Can be optional

    # Less critical for "completion" score but good to have
    if profile_data.get("interests"): completed_fields +=1
    # else: missing_summary.append("Interests")
    if profile_data.get("hobbies"): completed_fields +=1
    # else: missing_summary.append("Hobbies")
    
    # Adjust total_fields based on what's considered "essential" for completion
    # For now, let's say 6 are essential: name, title, tagline, skills, experiences, one of (edu/proj)
    essential_count = 0
    if profile_data.get("name"): essential_count +=1
    if profile_data.get("title"): essential_count +=1
    if profile_data.get("taglineOrBriefSummary"): essential_count +=1
    if profile_data.get("skills"): essential_count +=1
    if profile_data.get("experiences"): essential_count +=1
    if profile_data.get("education") or profile_data.get("projects"): essential_count +=1
    
    percentage = int((essential_count / 6) * 100) if missing_summary else 100 # If no missing essential, 100%
    
    if not missing_summary and essential_count < 6: # If still not 100% based on essential count
        if not profile_data.get("education") and not profile_data.get("projects"):
            missing_summary.append("Education or Projects")

    return percentage, missing_summary


def make_initials_svg_avatar(name: str, size: int = 80,
                            bg: str = "#6b4f4f", 
                            fg: str = "#fff8f0") -> str: 
    display_name = name if name and name.strip() else "?"
    initials = "".join([w[0].upper() for w in display_name.split()][:2]) or "?"
    svg = f'''
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">
<circle cx="{size/2}" cy="{size/2}" r="{size/2}" fill="{bg}"/>
<text x="50%" y="50%" fill="{fg}" font-size="{int(size/2.2)}"
        text-anchor="middle" dominant-baseline="central"
        font-family="sans-serif" font-weight="500">{initials}</text>
</svg>'''
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"

def open_edit_dialog(profile_id): st.session_state.editing_profile_id_dialog = profile_id
def delete_profile(profile_id): st.warning(f"Delete action called for {profile_id}")

# --- REVAMPED RENDER COFFEE CARD FUNCTION ---
def render_coffee_card(profile_data: Dict, profile_id: str): # profile_data is now the yourCoffeeCard object
    """
    Renders a single Coffee Card based on the new `yourCoffeeCard` structure.
    """
    if not profile_data or not isinstance(profile_data, dict):
        st.warning("Invalid Coffee Card data provided.")
        return

    # Get completion percentage for the new structure
    completion_percentage, missing_fields = calculate_profile_completion_new(profile_data)

    # --- Prepare data with safe defaults and escaping ---
    name = html.escape(profile_data.get("name", "N/A"))
    title = html.escape(profile_data.get("title", "N/A"))
    tagline = html.escape(profile_data.get("taglineOrBriefSummary", ""))
    location = html.escape(profile_data.get("location", ""))
    avatar_url = profile_data.get("profilePictureUrlForCard") or make_initials_svg_avatar(name if name != 'N/A' else '??')
    profile_url = profile_data.get("primaryProfileUrlForCard", "")
    call_to_action = html.escape(profile_data.get("callToActionForCard", ""))

    interests: List[str] = profile_data.get("interests", [])
    hobbies: List[str] = profile_data.get("hobbies", [])
    skills: List[str] = profile_data.get("skills", [])
    
    experiences_data: List[Dict] = profile_data.get("experiences", [])
    education_data: List[Dict] = profile_data.get("education", [])
    projects_data: List[Dict] = profile_data.get("projects", [])
    key_achievements: List[str] = profile_data.get("keyAchievementsOverall", [])


    # --- Helper to generate HTML for pill tags ---
    def generate_pills_html(items: List[str], placeholder: str, max_rows_collapsed: int = 2) -> str:
        if not items:
            return f'<span class="cc-pill-placeholder">{html.escape(placeholder)}</span>'
        safe_items = [html.escape(item) for item in items if item and item.strip()]
        if not safe_items:
            return f'<span class="cc-pill-placeholder">{html.escape(placeholder)}</span>'
        
        pills_html = "".join(f'<span class="cc-pill">{item}</span>' for item in safe_items)
        
        # For simplicity in this Streamlit-only version, we won't fully implement JS-based expand
        # but we can use the `collapsed` class if there are many items.
        # A true row calculation is complex without JS, so we'll use item count as a proxy.
        container_class = "cc-pill-container"
        # Estimate pills per row, e.g., 4-5. If more than max_rows_collapsed * 4, add 'collapsed'.
        if len(safe_items) > max_rows_collapsed * 4: # Heuristic
             container_class += " collapsed"
             
        return f'<div class="{container_class}">{pills_html}</div>'

    # --- Build HTML for missing fields summary ---
    summary_list_html = ""
    if missing_fields:
        for field in missing_fields:
            summary_list_html += f"<li>Update '{html.escape(field)}'</li>" 
    else:
        summary_list_html = "<li>All essential fields complete! ✔️</li>"

    # --- Build HTML for list-based sections (Experiences, Education, Projects) ---
    def generate_items_list_html(items_data: List[Dict], item_type: str) -> str:
        if not items_data:
            return f'<div class="cc-list-placeholder">No {item_type.lower()} added</div>'

        items_html_list = []
        for idx, item in enumerate(items_data):
            desc_id = f"desc-{profile_id}-{item_type}-{idx}"
            item_html = '<div class="cc-item">'
            
            if item_type == "Experiences":
                role = html.escape(item.get("role", "N/A"))
                company = html.escape(item.get("company", "N/A"))
                dates = html.escape(item.get("dates", ""))
                description = html.escape(item.get("description", "")) # Escape here
                skill_details = item.get("skillDetails", [])
                
                item_html += f'<div class="cc-item-header">{role} at {company}</div>'
                if dates: item_html += f'<div class="cc-item-dates">{dates}</div>'
                # For Streamlit, expand/collapse is hard. We'll show full desc or a snippet.
                # To simulate expandability, you could use st.expander PER item later.
                # For now, this HTML structure is a simplified display.
                # Adding a unique ID for potential JS interaction if this HTML is used elsewhere
                item_html += (
                    '<div class="cc-item-description" id="{did}">{desc}</div>'
                    .format(
                        did=desc_id,
                        desc=description.replace("\n", "<br>")
                    )
                )

                if skill_details:
                    item_html += '<div class="cc-item-skills-header">Skills Used:</div>'
                    item_html += '<div class="cc-pill-container">' # Re-use pill container for item skills
                    for skill_info in skill_details:
                        skill_name = html.escape(skill_info.get("skillName",""))
                        # skill_context = html.escape(skill_info.get("contextualSnippet",""))
                        # title_for_pill = f"{skill_name}: {skill_context}" if skill_context else skill_name
                        item_html += f'<span class="cc-item-skill-pill" title="Click for details (not implemented in pure HTML)">{skill_name}</span>'
                    item_html += '</div>'

            elif item_type == "Education":
                institution = html.escape(item.get("institution", "N/A"))
                degree = html.escape(item.get("degree", ""))
                field = html.escape(item.get("fieldOfStudy", ""))
                dates = html.escape(item.get("dates", ""))
                description = html.escape(item.get("description", ""))

                item_html += f'<div class="cc-item-header">{degree}</div>'
                item_html += f'<div class="cc-item-subheader">{institution}{" - " + field if field else ""}</div>'
                if dates: item_html += f'<div class="cc-item-dates">{dates}</div>'
                if description: item_html += (
                    '<div class="cc-item-description" id="{did}">{desc}</div>'
                    .format(
                        did=desc_id,
                        desc=description.replace("\n", "<br>")
                    )
                )


            elif item_type == "Projects":
                project_name = html.escape(item.get("projectName", "N/A"))
                dates = html.escape(item.get("datesOrDuration", ""))
                description = html.escape(item.get("description", ""))
                skills_used = item.get("skillsUsed", [])
                project_url = item.get("projectUrl", "")

                item_html += f'<div class="cc-item-header">{project_name}</div>'
                if dates: item_html += f'<div class="cc-item-dates">{dates}</div>'
                if description: item_html += (
                    '<div class="cc-item-description" id="{did}">{desc}</div>'
                    .format(
                        did=desc_id,
                        desc=description.replace("\n", "<br>")
                    )
                )
                if skills_used:
                    item_html += '<div class="cc-item-skills-header">Skills/Tech:</div>'
                    item_html += '<div class="cc-pill-container">'
                    for skill in skills_used:
                        item_html += f'<span class="cc-item-skill-pill">{html.escape(skill)}</span>'
                    item_html += '</div>'
                if project_url:
                    item_html += f'<p class="cc-item-project-url"><a href="{html.escape(project_url)}" target="_blank">View Project</a></p>'

            item_html += '</div>' # End cc-item
            items_html_list.append(item_html)
        
        return f'<div class="cc-item-list">{"".join(items_html_list)}</div>'

    # --- Construct the card HTML ---
    # This will be a simplified representation. True expandability needs st.expander or JS.
    # Mini-card elements are shown directly. Others can be conceptually "expandable sections".
    
    card_html = f"""
<div class="coffee-card-generated" id="card-{profile_id}">
    <div class="cc-card-content"> <!-- Main content area -->
        <div class="cc-header-content">
            <img src="{avatar_url}" alt="Avatar" class="cc-avatar">
            <div class="cc-header-text">
                <h1 class="cc-name">{name}</h1>
                <p class="cc-title">{title}</p>
                """
    if tagline:
        card_html += f'<p class="cc-tagline">{tagline}</p>'
    if location:
        card_html += f'<p class="cc-location">📍 {location}</p>'
    if profile_url:
        card_html += f'<p class="cc-profile-url"><a href="{html.escape(profile_url)}" target="_blank">🔗 View Profile</a></p>'
    if call_to_action: # Optional CTA
        card_html += f'<p class="cc-tagline" style="font-weight:bold;">{call_to_action}</p>'
    
    card_html += """
            </div>
        </div>""" # End cc-header-text and cc-header-content

    # --- Key Achievements (if any) ---
    if key_achievements:
        card_html += """
        <div class="cc-section">
            <h5 class="cc-section-header"><span class="cc-icon">🏆</span>Key Achievements</h5>
            <div class="cc-item-list">""" # Use item-list for consistent spacing
        for ach in key_achievements:
            card_html += f'<div class="cc-item"><p>{html.escape(ach)}</p></div>' # Simple paragraph for each
        card_html += """
            </div>
        </div>"""

    # --- Skills Section (Top N for mini, then expandable) ---
    card_html += f"""
        <div class="cc-section">
            <h5 class="cc-section-header"><span class="cc-icon">🛠️</span>Skills</h5>
            {generate_pills_html(skills, "No skills added", max_rows_collapsed=2)}
        </div>"""

    # --- Experiences Section ---
    # In pure HTML/CSS, full expandability per item is tricky without JS.
    # We'll list them. Streamlit `st.expander` would be used for true expandability per item.
    card_html += f"""
        <div class="cc-section">
            <h5 class="cc-section-header"><span class="cc-icon">💼</span>Experiences</h5>
            {generate_items_list_html(experiences_data, "Experiences")}
        </div>"""

    # --- Education Section ---
    card_html += f"""
        <div class="cc-section">
            <h5 class="cc-section-header"><span class="cc-icon">🎓</span>Education</h5>
            {generate_items_list_html(education_data, "Education")}
        </div>"""

    # --- Projects Section ---
    card_html += f"""
        <div class="cc-section">
            <h5 class="cc-section-header"><span class="cc-icon">🚀</span>Projects</h5>
            {generate_items_list_html(projects_data, "Projects")}
        </div>"""

    # --- Interests and Hobbies (potentially collapsed) ---
    card_html += f"""
        <div class="cc-section">
            <h5 class="cc-section-header"><span class="cc-icon">💡</span>Interests</h5>
            {generate_pills_html(interests, "No interests added", max_rows_collapsed=1)}
        </div>"""
    card_html += f"""
        <div class="cc-section">
            <h5 class="cc-section-header"><span class="cc-icon">🎨</span>Hobbies</h5>
            {generate_pills_html(hobbies, "No hobbies added", max_rows_collapsed=1)}
        </div>"""
    # right before your st.markdown call:

    progress_html = (
        f'</div>'  # close whatever came before
        f'<div class="cc-progress-summary-container">'
            f'<div class="cc-progress-label">{completion_percentage}% Complete</div>'
            f'<div class="cc-progress-bar-bg">'
                f'<div class="cc-progress-bar-fill" style="width: {completion_percentage}%;"></div>'
            f'</div>'
            f'<div class="cc-missing-fields-summary">'
                f'<h6>Profile Checklist:</h6>'
                f'<ul>{summary_list_html}</ul>'
            f'</div>'
        f'</div>'
        f'<div class="cc-card-actions-placeholder" style="height: 0;"></div>'
        f'</div>'  # final close of your card
    )

    card_html += progress_html
    st.markdown(card_html, unsafe_allow_html=True)

    # Render the actual Streamlit buttons
    action_cols = st.columns([0.75, 0.125, 0.125]) 
    with action_cols[1]:
         st.button("✏️", key=f"edit_{profile_id}", help="Edit Profile",
                   on_click=open_edit_dialog, args=(profile_id,),
                   use_container_width=True)
    with action_cols[2]:
         st.button("🗑️", key=f"delete_{profile_id}", help="Delete Profile",
                   on_click=delete_profile, args=(profile_id,),
                   use_container_width=True, type="secondary")


# --- Example Usage (Updated for new 'yourCoffeeCard' structure) ---
def init_session_state(): 
    if 'profiles' not in st.session_state:
        # This EXAMPLE_YOUR_COFFEE_CARD_DATA should match the new structure
        EXAMPLE_YOUR_COFFEE_CARD_DATA_HOMEN = {
            "id": str(uuid.uuid4()), # Keep ID for Streamlit list management
            "name": "Homen Shum",
            "title": "AI Workflow Engineer • LLM/Data Expert",
            "profilePictureUrlForCard": None, # Will use initials
            "taglineOrBriefSummary": "Ex-JPM Startup Banking Associate transforming financial research and workflows with LLMs and Generative AI.",
            "primaryProfileUrlForCard": "https://www.linkedin.com/in/Homenshum",
            "callToActionForCard": "Connect on LinkedIn",
            "location": "Fremont, CA",
            "interests": ["Large Language Models (LLM)", "Generative AI", "FinTech", "Healthcare AI", "Startups"],
            "hobbies": ["Memes", "Anime", "Sourdough Baking", "DeepRacer"],
            "skills": ["Python", "LLMOps", "RAG", "GCP", "Azure", "Data Analysis", "Automation", "Financial Modeling"],
            "keyAchievementsOverall": [
                "Reduced financial data processing time by over 95% at JPM using AI.",
                "Co-founded CerebroLeap Inc. focusing on medical AI applications.",
                "Achieved 2nd place in Nonecon Cybersecurity challenge."
            ],
            "experiences": [
                {
                    "role": "Founder", "company": "CafeCorner LLC", "dates": "Dec 2022-Present",
                    "isCurrentOrPrimary": True,
                    "briefSummaryForMiniCard": "Developing AI-driven sales recommendation and workflow automation tools.",
                    "description": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS, and Vertex using Docker. Focused on leveraging Large Language Models for intelligent system design.",
                    "skillDetails": [
                        {"skillName": "LLMOps", "contextualSnippet": "Deployed applications across GCP, Azure, AWS, and Vertex using Docker.", "relatedSkillsInThisExperience": ["Large Scale Data Analytics", "Docker"]},
                        {"skillName": "Large Scale Data Analytics", "contextualSnippet": "Built and deployed sales recommendation...applications.", "relatedSkillsInThisExperience": ["LLMOps", "Docker"]},
                        {"skillName": "Docker", "contextualSnippet": "Deployed ... using Docker.", "relatedSkillsInThisExperience": ["LLMOps", "Large Scale Data Analytics"]}
                    ]
                },
                {
                    "role": "Startup Banking Associate", "company": "JPMorgan Chase & Co.", "dates": "Jan 2021-Dec 2022",
                    "isCurrentOrPrimary": False, "briefSummaryForMiniCard": None,
                    "description": "Worked with Bay Area startups, focusing on Healthcare and Life Science. Automated classification systems using GPT and LlamaIndex for physician instructions, reducing processing time for 2,000+ companies from two weeks to under 30 seconds.",
                    "skillDetails": [
                        {"skillName": "Data Analysis", "contextualSnippet": "Automated classification systems...reducing processing time...", "relatedSkillsInThisExperience": ["Automation", "GPT", "LlamaIndex"]},
                        {"skillName": "Automation", "contextualSnippet": "Automated classification systems...", "relatedSkillsInThisExperience": ["Data Analysis", "GPT"]},
                        {"skillName": "GPT", "contextualSnippet": "Automated classification system...using GPT and LlamaIndex", "relatedSkillsInThisExperience": ["LlamaIndex", "Data Analysis"]}
                    ]
                }
            ],
            "education": [
                {
                    "institution": "UC Santa Barbara", "degree": "Certificate, Business Administration", "fieldOfStudy": "Management, General",
                    "dates": "2020-2021", "description": "Cooperated with professors on personal project, recommended by 2 professors."
                },
                {
                    "institution": "Financial Edge Training", "degree": "Commercial Banking Program", "fieldOfStudy": "Accounting and Finance",
                    "dates": "Nov 2021", "description": "Completed budget training and financial analysis modules."
                }
            ],
            "projects": [
                {
                    "projectName": "Patentyogi Screening Tool", "datesOrDuration": "Nov 2018-Present",
                    "description": "Integrating multi-agent architecture for comprehensive financial research, featuring cross-validation with web data and structured output processing for report generation.",
                    "skillsUsed": ["Multi-agent Systems", "Financial Research", "Streamlit", "LLMs"],
                    "projectUrl": "https://patentyogi-chat.streamlit.app/"
                },
                {
                    "projectName": "Medical Code Search & Report Generation", "datesOrDuration": "Mar 2023 (Concept)",
                    "description": "Developed MVP for real-time transcription, OFD/CID matching, and physician portal. Deployed via Azure with HIPAA compliant security.",
                    "skillsUsed": ["AI Transcription", "Azure", "HIPAA Compliance", "RAG"],
                    "projectUrl": "https://www.youtube.com/watch?v=kqVNt8pJOQI"
                }
            ]
        }
        EXAMPLE_YOUR_COFFEE_CARD_DATA_JANE = { # Simplified second profile
            "id": str(uuid.uuid4()),
            "name": "Jane Doe",
            "title": "Senior UX Designer • Creative Solutions",
             "profilePictureUrlForCard": "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1", # Example real image
            "taglineOrBriefSummary": "Crafting intuitive and accessible user experiences for innovative tech products.",
            "primaryProfileUrlForCard": "https://example.com/janedoe",
            "location": "Remote",
            "skills": ["Figma", "User Research", "Wireframing", "Prototyping", "Accessibility"],
            "experiences": [
                {
                    "role": "Senior UX Designer", "company": "TechSolutions Inc.", "dates": "2020-Present",
                    "isCurrentOrPrimary": True, "briefSummaryForMiniCard": "Leading UX design for flagship products.",
                    "description": "Lead designer for several key product lines, focusing on user-centered design principles and accessibility standards. Mentor junior designers.",
                    "skillDetails": [
                        {"skillName": "User Research", "contextualSnippet": "Focusing on user-centered design principles.", "relatedSkillsInThisExperience": ["Figma", "Accessibility"]},
                        {"skillName": "Figma", "contextualSnippet": "Lead designer for several key product lines...", "relatedSkillsInThisExperience": ["User Research", "Prototyping"]}
                    ]
                }
            ],
            "education": [{"institution": "Design University", "degree": "MFA in Interaction Design", "dates": "2016-2018"}],
            "interests": ["Minimalist Design", "Ethical AI", "Sustainable Tech"],
            "hobbies": ["Pottery", "Urban Sketching"],
        }
        st.session_state.profiles_new_structure = [
            EXAMPLE_YOUR_COFFEE_CARD_DATA_HOMEN,
            EXAMPLE_YOUR_COFFEE_CARD_DATA_JANE
            # Add more example profiles with the new structure if needed
        ]
    if 'editing_profile_id_dialog' not in st.session_state:
        st.session_state.editing_profile_id_dialog = None

def main():
    init_session_state()
    load_css()

    st.title("☕ Coffee Card Profiles (New Structure)")
    st.caption("Displaying profiles using the new `yourCoffeeCard` data structure.")

    st.markdown("---")

    if not st.session_state.get('profiles_new_structure'):
        st.info("No profiles yet.")
    else:
        num_columns = 2 
        cols = st.columns(num_columns)
        col_idx = 0
        
        for profile_data in st.session_state.profiles_new_structure:
             profile_id = profile_data.get("id", str(uuid.uuid4())) # Ensure an ID
             with cols[col_idx % num_columns]:
                 render_coffee_card(profile_data, profile_id) # Pass the whole profile_data object

                 if st.session_state.editing_profile_id_dialog == profile_id:
                     # Dialog logic would need to be adapted for this new structure if editing is implemented
                     st.warning(f"Edit Dialog (not fully implemented for new structure) for {profile_data.get('name')}") 
                 
                 st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
             col_idx += 1

if __name__ == "__main__":
    main()