# -*- coding: utf-8 -*-
import streamlit as st
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import re # Import regex for parsing tags
import html # Import html for escaping
import base64

# --- Constants for Card View Conciseness ---
MAX_PILLS_ON_CARD = 5
MAX_ITEMS_PER_SECTION_ON_CARD = 2 # For experiences, education, projects, achievements
MAX_DESC_LINES_ON_CARD = 2


# --- CSS Styling ---
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
.coffee-card-generated {
    border: 2px solid var(--cc-accent-dark-brown);
    border-radius: 12px;
    background: var(--cc-bg-main);
    padding: 1.8rem;
    margin-bottom: 1rem; 
    position: relative;
    box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 3px, rgba(0, 0, 0, 0.05) 0px 20px 25px -5px, rgba(0, 0, 0, 0.04) 0px 10px 10px -5px;
    transition: transform 0.2s ease-in-out, box-shadow 0.3s ease-in-out;
    display: flex;
    flex-direction: column;
    min-height: 400px; 
    justify-content: space-between;
    width: 100%; 
    max-width: 400px; 
    margin-left: auto;
    margin-right: auto;
}

.coffee-card-generated:hover {
    transform: translateY(-3px);
    box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
}

.coffee-card-generated::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 8px;
    background: var(--cc-accent-dark-brown);
    opacity: 0.9;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
}

.cc-card-content {
    padding-top: 8px;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}

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
    display: -webkit-box;
    -webkit-line-clamp: 2; 
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
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

.cc-section {
    margin-bottom: 0.8rem; 
    padding-bottom: 0.6rem; 
    border-bottom: 1px solid var(--cc-accent-light-tan);
}
.cc-card-content > .cc-section:last-of-type { 
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}

h5.cc-section-header { 
    color: var(--cc-accent-dark-brown);
    font-weight: bold;
    font-size: 0.9em; 
    margin-bottom: 0.4rem;
    margin-top: 0; 
    display: flex;
    align-items: center;
}
.cc-icon { margin-right: 0.4rem; opacity: 0.9; font-size: 1em; }

.cc-pill-container {
    margin-top: 0.3rem;
    margin-bottom: 0.3rem;
    display: flex;
    flex-wrap: wrap;
    gap: 5px; 
    min-height: auto; 
}

.cc-pill {
    display: inline-block;
    padding: 0.25rem 0.7rem; 
    margin: 0; 
    border-radius: 10px;
    background: var(--cc-bg-tag);
    font-size: 0.75em; 
    color: var(--cc-text-tag);
    font-weight: 500;
    line-height: 1.2;
    white-space: nowrap;
    border: 1px solid var(--cc-accent-light-tan);
}
.cc-pill:hover { 
     background: #f7e1ce;
}
.cc-pill-placeholder {
    font-style: italic;
    color: var(--cc-text-placeholder);
    font-size: 0.75em;
    padding: 0.3rem 0; 
}

.cc-item-list {
    margin-top: 0.4rem;
}
.cc-item {
    background: var(--cc-bg-exp);
    padding: 0.7rem; 
    border-radius: 4px;
    border: 1px solid var(--cc-accent-light-tan);
    margin-bottom: 0.6rem;
    font-size: 0.85em; 
}
.cc-item:last-child {
    margin-bottom: 0;
}
.cc-item-header {
    font-weight: bold;
    color: var(--cc-text-exp);
    font-size: 1em;
}
.cc-item-subheader {
    font-size: 0.9em;
    color: var(--cc-text-secondary);
    margin-bottom: 0.2rem;
}
.cc-item-dates {
    font-size: 0.8em;
    color: var(--cc-text-secondary);
    font-style: italic;
    margin-bottom: 0.3rem;
}
.cc-item-description { 
    font-size: 0.95em;
    line-height: 1.4;
    color: var(--cc-text-exp);
    white-space: pre-wrap; 
    max-height: calc(1.4em * ${MAX_DESC_LINES_ON_CARD}); 
    overflow: hidden;
    position: relative;
}

.cc-item-skills-header {
    font-size: 0.85em;
    font-weight: bold;
    color: var(--cc-accent-dark-brown);
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
}
.cc-item-skill-pill { 
    display: inline-block;
    padding: 0.15rem 0.5rem;
    margin-right: 3px;
    margin-bottom: 3px;
    border-radius: 8px;
    background: var(--cc-bg-tag);
    font-size: 0.7em;
    color: var(--cc-text-tag);
    border: 1px solid var(--cc-accent-light-tan);
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
    padding: 0.6rem;
    text-align: center;
    background: var(--cc-bg-exp);
    border: 1px dashed var(--cc-accent-light-tan); 
    border-radius: 4px;
    min-height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 0.4rem;
}

.cc-progress-summary-container {
    margin-top: auto; 
    padding-top: 0.8rem;
    border-top: 1px solid var(--cc-accent-light-tan);
}
.cc-missing-fields-summary { font-size: 0.8em; color: var(--cc-text-missing-summary); margin-bottom: 0.5rem; }
.cc-missing-fields-summary h6 { margin-bottom: 4px; font-weight: bold; font-size: 0.9em; color: var(--cc-accent-dark-brown); }
.cc-missing-fields-summary ul { margin: 0; padding-left: 18px; list-style-type: '☕ '; } 
.cc-missing-fields-summary li { margin-bottom: 2px; }

.cc-progress-label { font-size: 0.8em; color: var(--cc-text-progress); margin-bottom: 4px; text-align: right; }
.cc-progress-bar-bg { width: 100%; background-color: var(--cc-bg-progress-track); border-radius: 4px; height: 8px; overflow: hidden; margin-bottom: 8px; }
.cc-progress-bar-fill { height: 100%; background-color: var(--cc-accent-theme-brown); border-radius: 4px; transition: width 0.5s ease-in-out; }

/* Streamlit Button Overrides */
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

/* Style for popover trigger buttons in full details (these are default Streamlit buttons) */
div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button {
    /* This is a broad selector, be careful. It targets all buttons in expanders. 
       Ideally, wrap skill buttons in a specific container if more granularity is needed */
    font-size: 0.85em !important;
    padding: 0.25rem 0.6rem !important; /* Make them a bit pill-like */
    margin: 3px 2px !important;
    border-radius: 12px !important; /* Rounded corners for pill feel */
    /* Use theme colors for consistency, but slightly different from card's HTML pills */
    background-color: var(--cc-bg-tag) !important; 
    color: var(--cc-text-tag) !important;
    border: 1px solid var(--cc-accent-light-tan) !important;
}
div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button:hover {
    background-color: #f7e1ce !important; /* Darker tan on hover */
    border-color: var(--cc-accent-theme-brown) !important;
}

/* Popover content styling */
div[data-testid="stPopover"] h3 { /* Skill name in popover */
    color: var(--cc-accent-dark-brown);
    margin-top: 0;
    margin-bottom: 0.5rem;
    font-size: 1.1em;
}
div[data-testid="stPopover"] p {
    font-size: 0.9em;
    margin-bottom: 0.3rem;
}
div[data-testid="stPopover"] strong {
    color: var(--cc-text-exp);
}
div[data-testid="stPopover"] hr {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}


</style>
""".replace("${MAX_DESC_LINES_ON_CARD}", str(MAX_DESC_LINES_ON_CARD))

def load_css():
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

def calculate_profile_completion_new(profile_data: Dict[str, Any]) -> Tuple[int, List[str]]:
    if not profile_data: return 0, ["Profile data missing"]
    missing_summary = []
    essential_fields_map = {
        "name": "Name", "title": "Title", "taglineOrBriefSummary": "Tagline/Summary",
        "skills": "Skills", "experiences": "Experiences",
    }
    completed_essential_count = 0
    total_essential_fields = len(essential_fields_map)
    for key, display_name in essential_fields_map.items():
        if profile_data.get(key): completed_essential_count += 1
        else: missing_summary.append(display_name)
    has_edu_or_proj = bool(profile_data.get("education") or profile_data.get("projects"))
    total_essential_fields_adjusted = total_essential_fields + 1
    if has_edu_or_proj: completed_essential_count +=1
    else: missing_summary.append("Education or Projects")
    percentage = int((completed_essential_count / total_essential_fields_adjusted) * 100)
    if not missing_summary and completed_essential_count == total_essential_fields_adjusted: percentage = 100
    return percentage, missing_summary

def make_initials_svg_avatar(name: str, size: int = 80, bg: str = "#6b4f4f", fg: str = "#fff8f0") -> str: 
    display_name = name if name and name.strip() else "?"
    initials = "".join([w[0].upper() for w in display_name.split()][:2]) or "?"
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"><circle cx="{size/2}" cy="{size/2}" r="{size/2}" fill="{bg}"/><text x="50%" y="50%" fill="{fg}" font-size="{int(size/2.2)}" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-weight="500">{initials}</text></svg>'
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"

def open_edit_dialog(profile_id): st.session_state.editing_profile_id_dialog = profile_id
def delete_profile(profile_id): st.warning(f"Delete action called for {profile_id}")


def truncate_description(text: str, max_lines: int) -> str:
    if not text: return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + "..."
    return text

# --- CONCISE CARD RENDERING FUNCTION ---
def render_coffee_card_concise(profile_data: Dict, profile_id: str):
    if not profile_data or not isinstance(profile_data, dict):
        st.warning("Invalid Coffee Card data provided for concise card.")
        return

    completion_percentage, missing_fields = calculate_profile_completion_new(profile_data)

    name = html.escape(profile_data.get("name", "N/A"))
    title_text = html.escape(profile_data.get("title", "N/A"))
    tagline = html.escape(profile_data.get("taglineOrBriefSummary", ""))
    location = html.escape(profile_data.get("location", ""))
    avatar_url = profile_data.get("profilePictureUrlForCard") or make_initials_svg_avatar(name if name != 'N/A' else '??')
    profile_url = profile_data.get("primaryProfileUrlForCard", "")
    call_to_action = html.escape(profile_data.get("callToActionForCard", ""))

    skills_on_card: List[str] = profile_data.get("skills", [])[:MAX_PILLS_ON_CARD]
    interests_on_card: List[str] = profile_data.get("interests", [])[:MAX_PILLS_ON_CARD]
    hobbies_on_card: List[str] = profile_data.get("hobbies", [])[:MAX_PILLS_ON_CARD]
    
    key_achievements_on_card: List[str] = profile_data.get("keyAchievementsOverall", [])[:MAX_ITEMS_PER_SECTION_ON_CARD]
    experiences_on_card: List[Dict] = profile_data.get("experiences", [])[:MAX_ITEMS_PER_SECTION_ON_CARD]
    education_on_card: List[Dict] = profile_data.get("education", [])[:MAX_ITEMS_PER_SECTION_ON_CARD]
    projects_on_card: List[Dict] = profile_data.get("projects", [])[:MAX_ITEMS_PER_SECTION_ON_CARD]

    def generate_pills_html_for_card(items: List[str], placeholder: str) -> str:
        if not items: return f'<span class="cc-pill-placeholder">{html.escape(placeholder)}</span>'
        safe_items = [html.escape(item) for item in items if item and item.strip()]
        if not safe_items: return f'<span class="cc-pill-placeholder">{html.escape(placeholder)}</span>'
        pills_html_list = [f'<span class="cc-pill">{item}</span>' for item in safe_items]
        return f'<div class="cc-pill-container">{"".join(pills_html_list)}</div>'

    summary_list_html = ""
    if missing_fields:
        for field in missing_fields: summary_list_html += f"<li>Update '{html.escape(field)}'</li>" 
    else: summary_list_html = "<li>All essential fields complete! ✔️</li>"

    def generate_items_list_html_for_card(items_data: List[Dict], item_type: str) -> str:
        if not items_data: return f'<div class="cc-list-placeholder">No {item_type.lower()} on card.</div>'
        items_html_list = []
        for idx, item in enumerate(items_data):
            item_html = '<div class="cc-item">'
            desc_id = f"card-desc-{profile_id}-{item_type}-{idx}"
            
            if item_type == "Experiences":
                role = html.escape(item.get("role", "N/A"))
                company = html.escape(item.get("company", "N/A"))
                dates = html.escape(item.get("dates", ""))
                description = truncate_description(item.get("description", ""), MAX_DESC_LINES_ON_CARD)
                skill_details_on_card = item.get("skillDetails", [])[:MAX_PILLS_ON_CARD] 

                item_html += f'<div class="cc-item-header">{role} at {company}</div>'
                if dates: item_html += f'<div class="cc-item-dates">{dates}</div>'
                if description: item_html += f'<div class="cc-item-description" id="{desc_id}">{html.escape(description).replace(chr(10), "<br>")}</div>'
                if skill_details_on_card:
                    item_html += '<div class="cc-item-skills-header">Key Skills:</div><div class="cc-pill-container">'
                    for skill_info in skill_details_on_card:
                        item_html += f'<span class="cc-item-skill-pill">{html.escape(skill_info.get("skillName",""))}</span>'
                    item_html += '</div>'
            elif item_type == "Education": 
                institution = html.escape(item.get("institution", "N/A"))
                degree = html.escape(item.get("degree", ""))
                field = html.escape(item.get("fieldOfStudy", ""))
                dates = html.escape(item.get("dates", ""))
                description = truncate_description(item.get("description", ""), MAX_DESC_LINES_ON_CARD)
                item_html += f'<div class="cc-item-header">{degree}</div>'
                item_html += f'<div class="cc-item-subheader">{institution}{" - " + field if field else ""}</div>'
                if dates: item_html += f'<div class="cc-item-dates">{dates}</div>'
                if description: item_html += f'<div class="cc-item-description" id="{desc_id}">{html.escape(description).replace(chr(10), "<br>")}</div>'
            elif item_type == "Projects": 
                project_name = html.escape(item.get("projectName", "N/A"))
                dates = html.escape(item.get("datesOrDuration", ""))
                description = truncate_description(item.get("description", ""), MAX_DESC_LINES_ON_CARD)
                skills_used_on_card = item.get("skillsUsed", [])[:MAX_PILLS_ON_CARD]
                project_url_val = item.get("projectUrl", "")
                item_html += f'<div class="cc-item-header">{project_name}</div>'
                if dates: item_html += f'<div class="cc-item-dates">{dates}</div>'
                if description: item_html += f'<div class="cc-item-description" id="{desc_id}">{html.escape(description).replace(chr(10), "<br>")}</div>'
                if skills_used_on_card:
                    item_html += '<div class="cc-item-skills-header">Tech Used:</div><div class="cc-pill-container">'
                    for skill in skills_used_on_card: item_html += f'<span class="cc-item-skill-pill">{html.escape(skill)}</span>'
                    item_html += '</div>'
                if project_url_val: item_html += f'<p class="cc-item-project-url"><a href="{html.escape(project_url_val)}" target="_blank" rel="noopener noreferrer">View Project</a></p>'
            item_html += '</div>'
            items_html_list.append(item_html)
        return f'<div class="cc-item-list">{"".join(items_html_list)}</div>'

    card_html = f"""
    <div class="coffee-card-generated" id="card-{profile_id}">
        <div class="cc-card-content">
            <div class="cc-header-content">
                <img src="{avatar_url}" alt="{name}'s Avatar" class="cc-avatar">
                <div class="cc-header-text">
                    <h1 class="cc-name">{name}</h1>
                    <p class="cc-title">{title_text}</p>"""
    if tagline: card_html += f'<p class="cc-tagline">{tagline}</p>' 
    if location: card_html += f'<p class="cc-location">📍 {location}</p>'
    if profile_url: card_html += f'<p class="cc-profile-url"><a href="{html.escape(profile_url)}" target="_blank" rel="noopener noreferrer">🔗 View Profile</a></p>'
    if call_to_action: card_html += f'<p class="cc-tagline" style="font-weight:bold; color: var(--cc-accent-theme-brown);">{call_to_action}</p>'
    card_html += "</div></div>"

    if skills_on_card: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🛠️</span>Top Skills</h5>{generate_pills_html_for_card(skills_on_card, "")}</div>'
    # Only show placeholder if field exists but is empty, not if field is missing
    elif profile_data.get("skills") is not None:  card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🛠️</span>Top Skills</h5>{generate_pills_html_for_card([], "No skills added.")}</div>'


    if interests_on_card: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">💡</span>Interests</h5>{generate_pills_html_for_card(interests_on_card, "")}</div>'
    elif profile_data.get("interests") is not None: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">💡</span>Interests</h5>{generate_pills_html_for_card([], "No interests added.")}</div>'

    if hobbies_on_card: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🎨</span>Hobbies</h5>{generate_pills_html_for_card(hobbies_on_card, "")}</div>'
    elif profile_data.get("hobbies") is not None: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🎨</span>Hobbies</h5>{generate_pills_html_for_card([], "No hobbies added.")}</div>'
    
    if key_achievements_on_card:
        card_html += '<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🏆</span>Key Achievements</h5><div class="cc-item-list">'
        for ach in key_achievements_on_card: card_html += f'<div class="cc-item" style="padding: 0.5rem; background: var(--cc-bg-main);"><p style="margin:0; font-size:0.9em;">{truncate_description(html.escape(ach), MAX_DESC_LINES_ON_CARD)}</p></div>'
        card_html += '</div></div>'
    elif profile_data.get("keyAchievementsOverall") is not None : card_html += '<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🏆</span>Key Achievements</h5><div class="cc-list-placeholder">No achievements on card.</div></div>'


    if experiences_on_card: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">💼</span>Recent Experience</h5>{generate_items_list_html_for_card(experiences_on_card, "Experiences")}</div>'
    elif profile_data.get("experiences") is not None: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">💼</span>Recent Experience</h5>{generate_items_list_html_for_card([], "Experiences")}</div>'

    if education_on_card: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🎓</span>Education</h5>{generate_items_list_html_for_card(education_on_card, "Education")}</div>'
    elif profile_data.get("education") is not None: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🎓</span>Education</h5>{generate_items_list_html_for_card([], "Education")}</div>'
    
    if projects_on_card: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🚀</span>Featured Projects</h5>{generate_items_list_html_for_card(projects_on_card, "Projects")}</div>'
    elif profile_data.get("projects") is not None: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🚀</span>Featured Projects</h5>{generate_items_list_html_for_card([], "Projects")}</div>'
    
    card_html += '</div>' 
    card_html += f"""
        <div class="cc-progress-summary-container">
            <div class="cc-progress-label">{completion_percentage}% Complete</div>
            <div class="cc-progress-bar-bg"><div class="cc-progress-bar-fill" style="width: {completion_percentage}%;"></div></div>"""
    if missing_fields or completion_percentage < 100: card_html += f'<div class="cc-missing-fields-summary"><h6>Profile Checklist:</h6><ul>{summary_list_html}</ul></div>'
    else: card_html += '<div class="cc-missing-fields-summary"><h6>Profile Checklist:</h6><ul><li>All essential fields complete! ✔️</li></ul></div>'
    card_html += "</div></div>" 
    
    st.markdown(card_html, unsafe_allow_html=True)

    action_cols = st.columns([0.75, 0.125, 0.125]) 
    with action_cols[1]: st.button("✏️", key=f"edit_{profile_id}", help="Edit Profile", on_click=open_edit_dialog, args=(profile_id,), use_container_width=True)
    with action_cols[2]: st.button("🗑️", key=f"delete_{profile_id}", help="Delete Profile", on_click=delete_profile, args=(profile_id,), use_container_width=True, type="secondary")


# --- FULL PROFILE DETAILS RENDERING FUNCTION (Streamlit Native with Enhanced Skill Popovers) ---
def render_full_profile_details(profile_data: Dict, profile_id: str):
    if not profile_data: return

    expander_title = f"View Full Profile Details for {profile_data.get('name', 'N/A')}"
    with st.expander(expander_title, expanded=False):
        
        if profile_data.get("taglineOrBriefSummary"):
            st.markdown(f"**Tagline/Summary:** {profile_data['taglineOrBriefSummary']}")
        if profile_data.get("primaryProfileUrlForCard"):
            st.markdown(f"🔗 [View LinkedIn/Profile]({profile_data['primaryProfileUrlForCard']})")
        st.markdown("---")

        # Full Skills with Enhanced Popovers
        skills_all = profile_data.get("skills", [])
        if skills_all:
            st.subheader("🛠️ Skills")
            # Determine number of columns for skill popover triggers
            num_skill_cols = min(len(skills_all), 4) if skills_all else 1
            skill_cols = st.columns(num_skill_cols)
            
            for idx, skill_name in enumerate(skills_all):
                with skill_cols[idx % num_skill_cols]:
                    with st.popover(skill_name, use_container_width=True):
                        st.markdown(f"### {skill_name}")
                        
                        experiences_with_skill = []
                        for exp in profile_data.get("experiences", []):
                            for sd in exp.get("skillDetails", []):
                                if sd.get("skillName") == skill_name:
                                    experiences_with_skill.append({
                                        "role": exp.get("role", "N/A"),
                                        "company": exp.get("company", "N/A"),
                                        "context": sd.get("contextualSnippet", "No specific context.")
                                    })
                                    break # Found skill in this experience
                        
                        projects_with_skill = []
                        for proj in profile_data.get("projects", []):
                            if skill_name in proj.get("skillsUsed", []):
                                projects_with_skill.append(proj.get("projectName", "N/A"))

                        if experiences_with_skill:
                            st.markdown("**Applied in Experiences:**")
                            for exp_info in experiences_with_skill:
                                st.markdown(f"- **{exp_info['role']} at {exp_info['company']}**: _{exp_info['context']}_")
                            st.markdown("---")
                        
                        if projects_with_skill:
                            st.markdown("**Used in Projects:**")
                            for proj_name in projects_with_skill:
                                st.markdown(f"- {proj_name}")
                        
                        if not experiences_with_skill and not projects_with_skill:
                            st.caption("No specific context provided in experiences or projects for this skill.")
            st.markdown("---")


        # Full Key Achievements
        key_achievements_all = profile_data.get("keyAchievementsOverall", [])
        if key_achievements_all:
            st.subheader("🏆 Key Achievements")
            for ach in key_achievements_all: st.markdown(f"- {ach}")
            st.markdown("---")

        # Full Experiences with Skill Popovers
        experiences_all = profile_data.get("experiences", [])
        if experiences_all:
            st.subheader("💼 Experience")
            for exp_idx, exp in enumerate(experiences_all):
                st.markdown(f"#### {exp.get('role', 'N/A')} at {exp.get('company', 'N/A')}")
                st.caption(f"_{exp.get('dates', '')}_")
                if exp.get('description'): st.markdown(exp.get('description').replace("\n", "\n\n")) # Preserve newlines as paragraphs
                
                skill_details = exp.get("skillDetails", [])
                if skill_details:
                    st.markdown("**Skills/Tools Used in this Role:**")
                    
                    num_exp_skill_cols = min(len(skill_details), 4) if skill_details else 1
                    exp_skill_cols = st.columns(num_exp_skill_cols)

                    for s_idx, skill_info in enumerate(skill_details):
                        skill_name_exp = skill_info.get("skillName")
                        context_exp = skill_info.get("contextualSnippet", "No specific context provided.")
                        related_skills_exp = skill_info.get("relatedSkillsInThisExperience", [])
                        
                        with exp_skill_cols[s_idx % num_exp_skill_cols]:
                            with st.popover(skill_name_exp, use_container_width=True):
                                st.markdown(f"### {skill_name_exp}")
                                st.markdown(f"**Context:** {context_exp}")
                                if related_skills_exp:
                                    st.markdown(f"**Related skills in this role:** {', '.join(related_skills_exp)}")
                    st.markdown("<br>", unsafe_allow_html=True) 
                st.markdown("---") 
            st.markdown("---")

        # Full Education
        education_all = profile_data.get("education", [])
        if education_all:
            st.subheader("🎓 Education")
            for edu in education_all:
                st.markdown(f"#### {edu.get('degree', 'N/A')} - _{edu.get('fieldOfStudy', '')}_")
                st.markdown(f"{edu.get('institution', 'N/A')} ({edu.get('dates', '')})")
                if edu.get('description'): st.write(edu.get('description').replace("\n", "\n\n"))
                st.markdown("---")
            st.markdown("---")

        # Full Projects with Skill Popovers
        projects_all = profile_data.get("projects", [])
        if projects_all:
            st.subheader("🚀 Projects")
            for proj_idx, proj in enumerate(projects_all):
                st.markdown(f"#### {proj.get('projectName', 'N/A')}")
                st.caption(f"_{proj.get('datesOrDuration', '')}_")
                if proj.get('projectUrl'): st.markdown(f"🔗 [View Project]({proj.get('projectUrl')})")
                if proj.get('description'): st.write(proj.get('description').replace("\n", "\n\n"))
                
                skills_used_proj = proj.get("skillsUsed", [])
                if skills_used_proj:
                    st.markdown("**Skills/Tech Used:**")
                    num_proj_skill_cols = min(len(skills_used_proj), 4) if skills_used_proj else 1
                    proj_skill_cols = st.columns(num_proj_skill_cols)
                    for s_idx, skill_name_proj in enumerate(skills_used_proj):
                        with proj_skill_cols[s_idx % num_proj_skill_cols]:
                            with st.popover(skill_name_proj, use_container_width=True):
                                st.markdown(f"### {skill_name_proj}")
                                st.caption("Used in this project.") 
                    st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("---")
            st.markdown("---")

        interests_all = profile_data.get("interests", [])
        if interests_all:
            st.subheader("💡 Interests")
            st.markdown("- " + "\n- ".join(interests_all))
        
        hobbies_all = profile_data.get("hobbies", [])
        if hobbies_all:
            st.subheader("🎨 Hobbies")
            st.markdown("- " + "\n- ".join(hobbies_all))


# --- Example Usage Data ---
def init_session_state(): 
    if 'profiles_new_structure' not in st.session_state:
        EXAMPLE_YOUR_COFFEE_CARD_DATA_HOMEN = {
            "id": str(uuid.uuid4()), "name": "Homen Shum", "title": "AI Workflow Engineer • LLM/Data Expert",
            "profilePictureUrlForCard": None, 
            "taglineOrBriefSummary": "Ex-JPM Startup Banking Associate transforming financial research and workflows with LLMs and Generative AI. This is a slightly longer tagline that might be subject to CSS truncation on the card.",
            "primaryProfileUrlForCard": "https://www.linkedin.com/in/Homenshum", "callToActionForCard": "Connect on LinkedIn!",
            "location": "Fremont, CA",
            "interests": ["LLMs", "GenAI", "FinTech", "Healthcare AI", "Startups", "AI Ethics", "Decentralized Systems", "Quantum Computing"],
            "hobbies": ["Memes", "Anime", "Sourdough Baking", "DeepRacer", "Sci-Fi Novels", "Japanese Language", "Board Games"],
            "skills": ["Python", "LLMOps", "RAG", "GCP", "Azure", "Data Analysis", "Automation", "Financial Modeling", "Streamlit", "Docker", "Kubernetes", "Langchain", "Vector DBs", "FastAPI"],
            "keyAchievementsOverall": [
                "Reduced financial data processing time by >95% at JPM using AI.",
                "Co-founded CerebroLeap Inc. focusing on medical AI applications for diagnostics.",
                "Achieved 2nd place in Nonecon Cybersecurity challenge (secure coding).",
                "Developed a novel algorithm for anomaly detection in time-series data, patented."
            ],
            "experiences": [
                {
                    "role": "Founder", "company": "CafeCorner LLC", "dates": "Dec 2022-Present", "isCurrentOrPrimary": True,
                    "description": """Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS, and Vertex using Docker. 
- Focused on leveraging Large Language Models for intelligent system design.
- Developed a multi-tenant architecture for scalability and deployed it successfully.
- Integrated advanced RAG techniques for contextual understanding in customer support bots.
This is a longer description for Homen's first experience. It has multiple lines and will be truncated on the card view but fully visible in the details section. We need enough text to ensure the truncation is visible and that the full detail view is necessary.""",
                    "skillDetails": [
                        {"skillName": "LLMOps", "contextualSnippet": "Deployed LLM applications across GCP, Azure, AWS, and Vertex using Docker and K8s.", "relatedSkillsInThisExperience": ["Docker", "Kubernetes", "GCP", "Azure"]},
                        {"skillName": "RAG", "contextualSnippet": "Integrated advanced RAG techniques for contextual understanding."},
                        {"skillName": "Docker", "contextualSnippet": "Utilized Docker for containerizing all microservices.", "relatedSkillsInThisExperience": ["Kubernetes", "LLMOps"]},
                        {"skillName": "GCP", "contextualSnippet": "Leveraged Vertex AI and BigQuery on Google Cloud Platform."}
                    ]
                },
                {
                    "role": "Startup Banking Associate", "company": "JPMorgan Chase & Co.", "dates": "Jan 2021-Dec 2022",
                    "description": "Worked with Bay Area startups (Healthcare & Life Science). Automated classification systems (GPT & LlamaIndex) for physician instructions, cutting processing time for 2,000+ companies from weeks to <30s. Onboarded 50+ new clients.",
                    "skillDetails": [
                        {"skillName": "Data Analysis", "contextualSnippet": "Automated classification systems, reducing processing time with data insights."},
                        {"skillName": "Automation", "contextualSnippet": "Scripted and deployed automation for financial document processing."}
                    ]
                }
            ],
            "education": [
                {
                    "institution": "UC Santa Barbara", "degree": "Certificate, Business Administration", "fieldOfStudy": "Management, General",
                    "dates": "2020-2021", "description": "Cooperated with professors on a personal project involving market analysis for tech startups. Recommended by 2 professors for innovative approach. Focused on strategic management and entrepreneurial finance. This description is also made a bit longer."
                },
            ],
            "projects": [
                {
                    "projectName": "Patentyogi Screening Tool", "datesOrDuration": "Nov 2018-Present",
                    "description": "Integrating multi-agent architecture for comprehensive financial research. Features cross-validation with web data and structured output processing for report generation. This project aims to revolutionize how patent prior art search is conducted using modern AI, specifically focusing on NLP techniques for semantic matching.",
                    "skillsUsed": ["Multi-agent Systems", "Financial Research", "Streamlit", "LLMs", "Python"], # Added Python to test top-level skill
                    "projectUrl": "https://patentyogi-chat.streamlit.app/"
                },
                 {
                    "projectName": "AI Tutoring Platform (Concept)", "datesOrDuration": "2023 Q идеи",
                    "description": "Conceptualized an AI-powered tutoring system adapting to individual learning styles. Designed architecture for personalized feedback and content generation using LLMs.",
                    "skillsUsed": ["AI EdTech", "LLMs", "UX Design", "System Architecture"],
                }
            ]
        }
        EXAMPLE_YOUR_COFFEE_CARD_DATA_JANE = { 
            "id": str(uuid.uuid4()), "name": "Jane Doe", "title": "Senior UX Designer • Creative Solutions",
            "profilePictureUrlForCard": "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1", 
            "taglineOrBriefSummary": "Crafting intuitive and accessible user experiences for innovative tech products. Passionate about user-centered design and ethical AI.",
            "primaryProfileUrlForCard": "https://example.com/janedoe", "location": "Remote",
            "skills": ["Figma", "User Research", "Wireframing", "Prototyping", "Accessibility", "UI Design", "Interaction Design", "Design Systems", "A/B Testing"],
            "keyAchievementsOverall": ["Led a team to win the 'Best UX' award at TechCrunch Disrupt 2022.", "Successfully launched 3 major products, achieving an average NPS of 75+."],
            "experiences": [
                {
                    "role": "Senior UX Designer", "company": "TechSolutions Inc.", "dates": "2020-Present",
                    "description": """Lead designer for several key product lines, focusing on user-centered design principles and accessibility standards. 
- Mentor junior designers and contribute to the internal design system. 
- Conducted extensive user research to inform design decisions, resulting in a 20% uplift in user satisfaction scores for the flagship product.""",
                    "skillDetails": [
                        {"skillName": "User Research", "contextualSnippet": "Conducted heuristic evaluations, usability testing, and user interviews.", "relatedSkillsInThisExperience":["A/B Testing"]},
                        {"skillName": "Figma", "contextualSnippet": "Created high-fidelity prototypes and design specifications in Figma."},
                        {"skillName": "Accessibility", "contextualSnippet": "Ensured WCAG AA compliance for all new features."}
                    ]
                },
                 {
                    "role": "UX Designer", "company": "Innovatech", "dates": "2018-2020",
                    "description": "Designed user interfaces for web and mobile applications. Collaborated with product managers and developers to deliver user-friendly solutions. Participated in all phases of the design process from concept to launch.",
                    "skillDetails": [
                        {"skillName": "Wireframing", "contextualSnippet": "Developed detailed wireframes and user flows for new features."},
                        {"skillName": "Prototyping", "contextualSnippet": "Built interactive prototypes for user testing and stakeholder presentations."}
                    ]
                }
            ],
            "education": [{"institution": "Design University", "degree": "MFA in Interaction Design", "fieldOfStudy": "Human-Computer Interaction", "dates": "2016-2018", "description":"Thesis on ethical AI interfaces and their impact on user trust. Explored design patterns for transparent AI."}],
            "interests": ["Minimalist Design", "Ethical AI", "Sustainable Tech", "Photography", "Data Visualization"],
            "hobbies": ["Pottery", "Urban Sketching", "Yoga", "Reading Psychology Books"],
            "projects": [
                {
                    "projectName": "Accessible Mobile Banking App", "datesOrDuration": "6 Months (2022)",
                    "description": "Led the redesign of a mobile banking app to meet WCAG 2.1 AAA standards. This involved user testing with individuals with disabilities and iterating on designs based on feedback.",
                    "skillsUsed": ["Accessibility", "Figma", "User Testing", "Mobile UI/UX"], # Added Accessibility to test top-level
                }
            ] 
        }
        st.session_state.profiles_new_structure = [
            EXAMPLE_YOUR_COFFEE_CARD_DATA_HOMEN,
            EXAMPLE_YOUR_COFFEE_CARD_DATA_JANE
        ]
    if 'editing_profile_id_dialog' not in st.session_state:
        st.session_state.editing_profile_id_dialog = None

def main():
    st.set_page_config(layout="wide")
    init_session_state()
    load_css()

    st.title("☕ Coffee Card Profiles (Concise + Full Details)")
    st.caption("Displaying concise cards with an option to expand for full details using Streamlit native elements.")
    st.markdown("---")

    if not st.session_state.get('profiles_new_structure'):
        st.info("No profiles yet.")
    else:
        num_columns = 2 
        cols = st.columns(num_columns)
        current_col = 0
        for profile_data in st.session_state.profiles_new_structure:
            profile_id = profile_data.get("id", str(uuid.uuid4())) 
            with cols[current_col % num_columns]:
                render_coffee_card_concise(profile_data, profile_id) 
                render_full_profile_details(profile_data, profile_id) 
                
                if st.session_state.editing_profile_id_dialog == profile_id:
                    with st.dialog("Edit Profile", width="large"):
                        st.warning(f"Edit Dialog (not fully implemented) for {profile_data.get('name')}")
                        if st.button("Close", key=f"close_edit_{profile_id}"):
                            st.session_state.editing_profile_id_dialog = None
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            current_col += 1

if __name__ == "__main__":
    main()