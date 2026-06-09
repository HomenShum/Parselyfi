# -*- coding: utf-8 -*-
import streamlit as st
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import re # Import regex for parsing tags
import html # Import html for escaping
import base64
import qrcode # For QR code generation
import io # For BytesIO

# --- Constants for Card View Conciseness (Mainly for HTML card and Previews now) ---
MAX_PILLS_ON_CARD = 5
MAX_ITEMS_PER_SECTION_ON_CARD = 2
MAX_DESC_LINES_ON_CARD = 2
MAX_PILLS_FOR_WALLET_PASS = 10
MAX_PILLS_FOR_SOCIAL_PNG = 10
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

/* Style for popover trigger buttons in full details */
div[data-testid="stExpander"] div[data-testid="stButton"] > button {
    font-size: 0.85em !important;
    padding: 0.25rem 0.6rem !important;
    margin: 3px 2px !important;
    border-radius: 12px !important;
    background-color: var(--cc-bg-tag) !important;
    color: var(--cc-text-tag) !important;
    border: 1px solid var(--cc-accent-light-tan) !important;
}
div[data-testid="stExpander"] div[data-testid="stButton"] > button:hover {
    background-color: #f7e1ce !important;
    border-color: var(--cc-accent-theme-brown) !important;
}

/* Popover content styling */
div[data-testid="stPopover"] h3 {
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

/* --- Apple Wallet Preview Styles --- */
.wallet-pass-preview-container {
    display: flex;
    justify-content: center;
    margin-bottom: 15px;
}
.wallet-pass-preview {
    background-color: #2c2c2e; color: #fff; border-radius: 12px; padding: 15px;
    width: 100%; max-width: 340px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2); font-size: 14px;
}
.wallet-pass-header {
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #4a4a4e;
}
.wallet-pass-header .logo { font-size: 1.6em; color: var(--cc-accent-theme-brown); margin-top: 2px; }
.wallet-pass-header .pass-type-stack { display: flex; flex-direction: column; align-items: flex-end; text-align: right; }
.wallet-pass-header .pass-type { font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.8px; color: #c7c7cc; }
.wallet-pass-header .pass-location {
    font-size: 0.65em; color: #8e8e93; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; max-width: 100px;
}
.wallet-pass-body .name {
    font-size: 1.2em; font-weight: 500; margin-bottom: 2px; color: #fff;
    line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.wallet-pass-body .title {
    font-size: 0.85em; color: #e5e5ea; margin-bottom: 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.wallet-pass-body .summary {
    font-size: 0.75em; color: #aeaeb2; line-height: 1.2;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 5px;
}
.wallet-pass-body .key-skills-list {
    font-size: 0.7em; color: #c7c7cc; margin-top: 4px; line-height: 1.3;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 90%;
}
.wallet-pass-body .key-skills-list .skills-label { font-weight: 500; color: #e5e5ea; }
.wallet-pass-qr-section { margin-top: 10px; padding-top: 10px; border-top: 1px solid #4a4a4e; text-align: center; }
.wallet-pass-qr-section img {
    background-color: white; padding: 3px; border-radius: 3px;
    max-width: 75px; display: block; margin: 0 auto;
}
.wallet-pass-qr-section .qr-label { font-size: 0.7em; color: #8e8e93; margin-top: 5px; }

/* --- Social PNG Preview Styles --- */
.social-png-preview-container { display: flex; justify-content: center; margin-bottom: 15px; }
.social-png-preview {
    border: 1px solid var(--cc-accent-light-tan); border-radius: 8px; padding: 18px;
    background-color: var(--cc-bg-exp); width: 100%; max-width: 380px; font-family: sans-serif;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; flex-direction: column;
}
.social-png-header { display: flex; align-items: center; margin-bottom: 12px; gap: 12px; }
.social-png-avatar {
    width: 65px; height: 65px; border-radius: 50%; object-fit: cover;
    border: 2px solid var(--cc-accent-theme-brown); flex-shrink: 0;
}
.social-png-text-info { flex-grow: 1; min-width: 0; }
.social-png-text-info .name {
    font-size: 1.25em; font-weight: bold; color: var(--cc-text-name); margin: 0 0 2px 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.social-png-text-info .title {
    font-size: 0.8em; color: var(--cc-text-title); margin: 0; line-height: 1.3;
    white-space: normal; overflow: hidden; display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.social-png-tagline {
    font-size: 0.85em; color: var(--cc-text-secondary); margin-bottom: 8px; line-height: 1.4;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden; text-overflow: ellipsis;
}
.social-png-location {
    font-size: 0.75em; color: var(--cc-text-secondary); margin-bottom: 10px;
    display: flex; align-items: center;
}
.social-png-location .icon { margin-right: 5px; opacity: 0.7; }
.social-png-pills-section { margin-bottom: 10px; }
.social-png-pills-section .pills-label {
    font-size: 0.7em; color: var(--cc-accent-dark-brown); font-weight: bold;
    margin-bottom: 4px; text-transform: uppercase;
}
.social-png-pills-container .cc-pill { font-size: 0.7em; padding: 0.2rem 0.6rem; margin: 2px; }
.social-png-footer {
    margin-top: auto; padding-top: 8px; border-top: 1px dashed var(--cc-accent-light-tan);
    font-size: 0.7em; color: var(--cc-text-placeholder); text-align: center;
}
.social-png-footer .cta { font-weight: bold; color: var(--cc-accent-theme-brown); }

/* Additional styling for dialog form elements if needed */
div[data-testid="stDialog"] .stTextArea textarea,
div[data-testid="stDialog"] .stTextInput input { font-size: 0.95em; }
div[data-testid="stDialog"] h3, div[data-testid="stDialog"] h4 {
    color: var(--cc-accent-dark-brown); margin-top: 0.8rem; margin-bottom: 0.3rem;
}
div[data-testid="stDialog"] hr { margin-top: 0.8rem; margin-bottom: 0.8rem; }

/* Styling for native card popover buttons to make them look like pills */
/* Targets buttons that are direct children of stPopover, within stVerticalBlock (common layout parent) */
div[data-testid="stVerticalBlock"] div[data-testid="stPopover"] > button,
div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button { /* Added stHorizontalBlock for popovers in columns */
    background-color: var(--cc-bg-tag) !important;
    color: var(--cc-text-tag) !important;
    border: 1px solid var(--cc-accent-light-tan) !important;
    border-radius: 10px !important;
    padding: 0.25rem 0.7rem !important;
    font-size: 0.75em !important;
    margin: 2px 3px !important; /* small gap */
    line-height: 1.3 !important; /* ensure text fits */
    display: inline-block !important; /* Better for pill-like behavior */
    text-align: center !important;
}
div[data-testid="stVerticalBlock"] div[data-testid="stPopover"] > button:hover,
div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button:hover {
    background-color: #f7e1ce !important;
    border-color: var(--cc-accent-theme-brown) !important;
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

def open_edit_dialog(profile_id_to_edit: str):
    st.session_state.editing_profile_id_dialog = profile_id_to_edit

def delete_profile(profile_id):
    st.warning(f"Delete action called for {profile_id}. (Actual deletion not implemented in this example)")

def truncate_description(text: str, max_lines: int, for_markdown: bool = False) -> str:
    if not text: return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        truncated_text = "\n".join(lines[:max_lines]) + "..."
    else:
        truncated_text = text
    if for_markdown:
        return html.escape(truncated_text).replace("\n", "  \n")
    return truncated_text

def truncate_to_one_line(text: str, max_length: int = 50) -> str:
    if not text: return ""
    first_line = text.splitlines()[0]
    if len(first_line) > max_length:
        return first_line[:max_length-3] + "..."
    return first_line

def parse_text_area_to_list(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]

def join_list_to_text_area(items: Optional[List[str]]) -> str:
    return "\n".join(items) if items else ""

def parse_comma_separated_to_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(',') if item.strip()]

def join_list_to_comma_separated(items: Optional[List[str]]) -> str:
    return ", ".join(items) if items else ""

# --- HTML CONCISE CARD RENDERING FUNCTION (Existing) ---
def render_coffee_card_concise(profile_data: Dict, profile_id: str):
    # ... (This function remains unchanged from your previous version)
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
                <img src="{avatar_url}" alt="{html.escape(name)}'s Avatar" class="cc-avatar">
                <div class="cc-header-text">
                    <h1 class="cc-name">{name}</h1>
                    <p class="cc-title">{title_text}</p>"""
    if tagline: card_html += f'<p class="cc-tagline">{tagline}</p>'
    if location: card_html += f'<p class="cc-location">📍 {location}</p>'
    if profile_url: card_html += f'<p class="cc-profile-url"><a href="{html.escape(profile_url)}" target="_blank" rel="noopener noreferrer">🔗 View Profile</a></p>'
    if call_to_action: card_html += f'<p class="cc-tagline" style="font-weight:bold; color: var(--cc-accent-theme-brown);">{call_to_action}</p>'
    card_html += "</div></div>"
    if skills_on_card: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🛠️</span>Top Skills</h5>{generate_pills_html_for_card(skills_on_card, "")}</div>'
    elif profile_data.get("skills") is not None:  card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🛠️</span>Top Skills</h5>{generate_pills_html_for_card([], "No skills added.")}</div>'
    if interests_on_card: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">💡</span>Interests</h5>{generate_pills_html_for_card(interests_on_card, "")}</div>'
    elif profile_data.get("interests") is not None: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">💡</span>Interests</h5>{generate_pills_html_for_card([], "No interests added.")}</div>'
    if hobbies_on_card: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🎨</span>Hobbies</h5>{generate_pills_html_for_card(hobbies_on_card, "")}</div>'
    elif profile_data.get("hobbies") is not None: card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🎨</span>Hobbies</h5>{generate_pills_html_for_card([], "No hobbies added.")}</div>'
    if key_achievements_on_card:
        card_html += '<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🏆</span>Key Achievements</h5><div class="cc-item-list">'
        for ach_idx, ach in enumerate(key_achievements_on_card): card_html += f'<div class="cc-item" style="padding: 0.5rem; background: var(--cc-bg-main);"><p style="margin:0; font-size:0.9em;">{truncate_description(html.escape(ach), MAX_DESC_LINES_ON_CARD)}</p></div>'
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
    with action_cols[1]: st.button("✏️", key=f"edit_html_{profile_id}", help="Edit Profile (HTML Card)", on_click=open_edit_dialog, args=(profile_id,), use_container_width=True)
    with action_cols[2]: st.button("🗑️", key=f"delete_html_{profile_id}", help="Delete Profile (HTML Card)", on_click=delete_profile, args=(profile_id,), use_container_width=True, type="secondary")


# --- NATIVE STREAMLIT COMPREHENSIVE CARD RENDERING FUNCTION ---
def render_coffee_card_native_comprehensive(profile_data: Dict, profile_id: str):
    if not profile_data or not isinstance(profile_data, dict):
        st.warning("Invalid Coffee Card data provided for native comprehensive card.")
        return

    with st.container(border=True):
        # --- Header ---
        name = profile_data.get("name", "N/A")
        title_text = profile_data.get("title", "N/A")
        tagline = profile_data.get("taglineOrBriefSummary", "")
        location = profile_data.get("location", "")
        avatar_url = profile_data.get("profilePictureUrlForCard") or make_initials_svg_avatar(name if name != 'N/A' else '??')
        profile_url = profile_data.get("primaryProfileUrlForCard", "")
        call_to_action = profile_data.get("callToActionForCard", "")

        header_cols = st.columns([1, 3])
        with header_cols[0]:
            st.image(avatar_url, width=100, caption="" if avatar_url and "pexels.com" in avatar_url else "Avatar")
        with header_cols[1]:
            st.subheader(name)
            if title_text and title_text != "N/A":
                st.markdown(f"**{html.escape(title_text)}**")
            if tagline:
                st.caption(html.escape(tagline)) # Full tagline
            if location:
                st.caption(f"📍 {html.escape(location)}")
            if profile_url:
                st.markdown(f"🔗 [View Profile]({html.escape(profile_url)})")
            if call_to_action:
                 st.markdown(f"**<font color='var(--cc-accent-theme-brown)'>{html.escape(call_to_action)}</font>**", unsafe_allow_html=True)
        st.markdown("---")

        # --- Skills (Full list with Popovers) ---
        skills_all = profile_data.get("skills", [])
        if skills_all:
            st.markdown(f"**🛠️ Skills**")
            num_skill_cols = min(len(skills_all), 5) # Show more skills per row
            skill_cols = st.columns(num_skill_cols)
            for idx, skill_name in enumerate(skills_all):
                with skill_cols[idx % num_skill_cols]:
                    with st.popover(skill_name, use_container_width=True):
                        st.markdown(f"### {html.escape(skill_name)}")
                        experiences_with_skill = []
                        for exp in profile_data.get("experiences", []):
                            for sd in exp.get("skillDetails", []):
                                if sd.get("skillName") == skill_name:
                                    experiences_with_skill.append(f"_{exp.get('role', 'N/A')} at {exp.get('company', 'N/A')}_: {sd.get('contextualSnippet', '')}") # Full context
                        projects_with_skill = []
                        for proj in profile_data.get("projects", []):
                            if skill_name in proj.get("skillsUsed", []):
                                projects_with_skill.append(proj.get("projectName", "N/A"))
                        if experiences_with_skill:
                            st.markdown("**Applied in Experiences:**\n" + "\n".join([f"- {e}" for e in experiences_with_skill]))
                        if projects_with_skill:
                            st.markdown("**Used in Projects:**\n" + "\n".join([f"- {p}" for p in projects_with_skill]))
                        if not experiences_with_skill and not projects_with_skill:
                            st.caption("General skill proficiency.")
            st.markdown("---")
        elif skills_all is not None: # Field exists but is empty
            st.markdown(f"**🛠️ Skills**")
            st.caption("No skills added.")
            st.markdown("---")


        # --- Key Achievements (Full list) ---
        key_achievements_all = profile_data.get("keyAchievementsOverall", [])
        if key_achievements_all:
            st.markdown("**🏆 Key Achievements**")
            for ach in key_achievements_all:
                st.markdown(f"- {html.escape(ach)}")
            st.markdown("---")
        elif key_achievements_all is not None:
            st.markdown("**🏆 Key Achievements**")
            st.caption("No key achievements added.")
            st.markdown("---")

        # --- Experiences (Full list) ---
        experiences_all = profile_data.get("experiences", [])
        if experiences_all:
            st.markdown("**💼 Experience**")
            for exp_idx, exp in enumerate(experiences_all):
                with st.container(border=True):
                    st.markdown(f"#### {html.escape(exp.get('role', 'N/A'))} at {html.escape(exp.get('company', 'N/A'))}")
                    st.caption(f"_{html.escape(exp.get('dates', ''))}_")
                    if exp.get('description'):
                        st.markdown(html.escape(exp.get('description')).replace("\n", "  \n"), unsafe_allow_html=True) # Markdown line breaks
                    skill_details = exp.get("skillDetails", [])
                    if skill_details:
                        st.markdown("**Skills/Tools Used:**")
                        num_exp_skill_cols = min(len(skill_details), 4)
                        exp_skill_cols = st.columns(num_exp_skill_cols)
                        for s_idx, skill_info in enumerate(skill_details):
                            skill_name_exp = skill_info.get("skillName","")
                            with exp_skill_cols[s_idx % num_exp_skill_cols]:
                                with st.popover(html.escape(skill_name_exp), use_container_width=True):
                                    st.markdown(f"### {html.escape(skill_name_exp)}")
                                    st.markdown(f"**Context:** {html.escape(skill_info.get('contextualSnippet', 'No specific context provided.'))}")
                                    related_skills_exp = skill_info.get("relatedSkillsInThisExperience", [])
                                    if related_skills_exp:
                                        st.markdown(f"**Related:** {', '.join(map(html.escape, related_skills_exp))}")
                st.markdown("---" if exp_idx < len(experiences_all) -1 else "") # Divider between experiences
            # st.markdown("---") # Final divider for section moved to after loop
        elif experiences_all is not None:
            st.markdown("**💼 Experience**")
            st.caption("No experience added.")
        if experiences_all is not None: st.markdown("---") # Divider if section was shown


        # --- Education (Full list) ---
        education_all = profile_data.get("education", [])
        if education_all:
            st.markdown("**🎓 Education**")
            for edu_idx, edu in enumerate(education_all):
                with st.container(border=True):
                    st.markdown(f"#### {html.escape(edu.get('degree', 'N/A'))} - _{html.escape(edu.get('fieldOfStudy', ''))}_")
                    st.markdown(f"{html.escape(edu.get('institution', 'N/A'))} ({html.escape(edu.get('dates', ''))})")
                    if edu.get('description'):
                        st.markdown(html.escape(edu.get('description')).replace("\n", "  \n"), unsafe_allow_html=True)
                st.markdown("---" if edu_idx < len(education_all) -1 else "")
            # st.markdown("---")
        elif education_all is not None:
            st.markdown("**🎓 Education**")
            st.caption("No education added.")
        if education_all is not None: st.markdown("---")


        # --- Projects (Full list) ---
        projects_all = profile_data.get("projects", [])
        if projects_all:
            st.markdown("**🚀 Projects**")
            for proj_idx, proj in enumerate(projects_all):
                with st.container(border=True):
                    st.markdown(f"#### {html.escape(proj.get('projectName', 'N/A'))}")
                    st.caption(f"_{html.escape(proj.get('datesOrDuration', ''))}_")
                    if proj.get('projectUrl'):
                        st.markdown(f"🔗 [View Project]({html.escape(proj.get('projectUrl'))})")
                    if proj.get('description'):
                        st.markdown(html.escape(proj.get('description')).replace("\n", "  \n"), unsafe_allow_html=True)
                    skills_used_proj = proj.get("skillsUsed", [])
                    if skills_used_proj:
                        st.markdown("**Skills/Tech Used:** " + ", ".join(map(html.escape, skills_used_proj)))
                st.markdown("---" if proj_idx < len(projects_all)-1 else "")
            # st.markdown("---")
        elif projects_all is not None:
            st.markdown("**🚀 Projects**")
            st.caption("No projects added.")
        if projects_all is not None: st.markdown("---")


        # --- Interests and Hobbies (Full lists, simpler display) ---
        interests_all = profile_data.get("interests", [])
        if interests_all:
            st.markdown(f"**💡 Interests**")
            st.markdown("- " + "\n- ".join(map(html.escape, interests_all)))
            st.markdown("---")
        elif interests_all is not None:
            st.markdown(f"**💡 Interests**")
            st.caption("No interests added.")
            st.markdown("---")

        hobbies_all = profile_data.get("hobbies", [])
        if hobbies_all:
            st.markdown(f"**🎨 Hobbies**")
            st.markdown("- " + "\n- ".join(map(html.escape, hobbies_all)))
            st.markdown("---")
        elif hobbies_all is not None:
            st.markdown(f"**🎨 Hobbies**")
            st.caption("No hobbies added.")
            st.markdown("---")


        # --- Progress and Summary ---
        completion_percentage, missing_fields = calculate_profile_completion_new(profile_data)
        st.caption(f"Profile Completion: {completion_percentage}%")
        st.progress(completion_percentage / 100)

        if missing_fields or completion_percentage < 100:
            # Using st.expander for checklist to save space on the comprehensive card
            with st.expander("Profile Checklist", expanded=False):
                if missing_fields:
                    for field in missing_fields: st.markdown(f"- Update '{html.escape(field)}'")
                else: st.markdown("- All essential fields complete! ✔️")
        else:
             st.markdown("All essential fields complete! ✔️")

        # --- Action Buttons ---
        st.markdown("---")
        action_cols_native = st.columns([0.75, 0.125, 0.125])
        with action_cols_native[1]:
            st.button("✏️", key=f"edit_native_comp_{profile_id}", help="Edit Profile", on_click=open_edit_dialog, args=(profile_id,), use_container_width=True)
        with action_cols_native[2]:
            st.button("🗑️", key=f"delete_native_comp_{profile_id}", help="Delete Profile", on_click=delete_profile, args=(profile_id,), use_container_width=True, type="secondary")


# --- PREVIEW RENDERING FUNCTIONS (Existing) ---
# render_apple_wallet_preview and render_social_png_preview remain unchanged

def render_apple_wallet_preview(profile_data: Dict, profile_id: str):
    name = html.escape(profile_data.get("name", "N/A"))
    title = html.escape(profile_data.get("title", "No Title"))
    summary_raw = profile_data.get("taglineOrBriefSummary", profile_data.get("title", "Professional Profile"))
    summary_one_line = truncate_to_one_line(summary_raw, max_length=35)
    location_raw = profile_data.get("location", "")
    location_short = truncate_to_one_line(location_raw, max_length=15)
    skills_all = profile_data.get("skills", [])
    top_skills_for_wallet = skills_all[:MAX_PILLS_FOR_WALLET_PASS]
    skills_display_str = ", ".join([html.escape(s) for s in top_skills_for_wallet])
    qr_data = profile_data.get("primaryProfileUrlForCard", f"https://example.com/profile/{profile_id}")
    qr_img_html_embed = ""
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=2)
        qr.add_data(qr_data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        qr_img_base64 = base64.b64encode(buffered.getvalue()).decode()
        qr_img_html_embed = f'<img src="data:image/png;base64,{qr_img_base64}" alt="QR Code" style="background-color: white; padding: 3px; border-radius: 3px; max-width: 75px; display: block; margin: 0 auto;">'
    except Exception:
        qr_img_html_embed = '<div class="qr-label" style="text-align:center;">QR (Profile URL error or not set)</div>'
    wallet_html_parts = [
        f'<div class="wallet-pass-preview-container">',
        f'  <div class="wallet-pass-preview" id="wallet-{profile_id}">',
        '    <div class="wallet-pass-header">',
        '        <span class="logo">☕</span>',
        '        <div class="pass-type-stack">',
        '           <span class="pass-type">Coffee Card</span>'
    ]
    if location_short:
        wallet_html_parts.append(f'<span class="pass-location">{html.escape(location_short)}</span>')
    wallet_html_parts.extend([
        '        </div>', '    </div>', '    <div class="wallet-pass-body">',
        f'        <div class="name">{name}</div>', f'        <div class="title">{title}</div>',
        f'        <div class="summary">{html.escape(summary_one_line)}</div>'
    ])
    if skills_display_str:
        wallet_html_parts.append(f'<div class="key-skills-list"><span class="skills-label">Key Skills:</span> {skills_display_str}</div>')
    wallet_html_parts.extend(['    </div>', '    <div class="wallet-pass-qr-section">'])
    wallet_html_parts.append(qr_img_html_embed)
    if 'src="data:image/png;base64,' in qr_img_html_embed:
         wallet_html_parts.append('       <div class="qr-label">Scan for Profile</div>')
    wallet_html_parts.extend(['    </div>', '  </div>', '</div>'])
    final_wallet_html = "\n".join(wallet_html_parts)
    st.markdown(final_wallet_html, unsafe_allow_html=True)

def render_social_png_preview(profile_data: Dict, profile_id: str):
    name = html.escape(profile_data.get("name", "N/A"))
    title_text = html.escape(profile_data.get("title", "No Title"))
    tagline = html.escape(profile_data.get("taglineOrBriefSummary", ""))
    avatar_url = profile_data.get("profilePictureUrlForCard") or make_initials_svg_avatar(name if name != 'N/A' else '??', size=65)
    location = html.escape(profile_data.get("location", ""))
    call_to_action_short = html.escape(truncate_to_one_line(profile_data.get("callToActionForCard", "View Full Profile"), 30))
    skills_all = profile_data.get("skills", [])
    interests_all = profile_data.get("interests", [])
    pills_for_social = []
    pills_for_social.extend(skills_all[:MAX_PILLS_FOR_SOCIAL_PNG])
    remaining_slots = MAX_PILLS_FOR_SOCIAL_PNG - len(pills_for_social)
    if remaining_slots > 0 and interests_all:
        pills_for_social.extend(interests_all[:remaining_slots])
    if not pills_for_social and (skills_all or interests_all):
        pills_html = '<span class="cc-pill-placeholder">Key areas...</span>'
    elif not pills_for_social:
         pills_html = '<span class="cc-pill-placeholder">No skills/interests to show</span>'
    else:
        pills_html = "".join([f'<span class="cc-pill">{html.escape(item)}</span>' for item in pills_for_social])
    social_html = f"""
    <div class="social-png-preview-container">
        <div class="social-png-preview" id="social-{profile_id}">
            <div class="social-png-header">
                <img src="{avatar_url}" alt="{html.escape(name)}'s Avatar" class="social-png-avatar">
                <div class="social-png-text-info">
                    <div class="name">{name}</div>
                    <div class="title">{title_text}</div>
                </div>
            </div>"""
    if tagline: social_html += f'<div class="social-png-tagline">{tagline}</div>'
    if location: social_html += f'<div class="social-png-location"><span class="icon">📍</span>{location}</div>'
    social_html += f"""
            <div class="social-png-pills-section">
                <div class="pills-label">Skills & Interests</div>
                <div class="social-png-pills-container cc-pill-container">{pills_html}</div>
            </div>
            <div class="social-png-footer">"""
    if call_to_action_short: social_html += f'<span class="cta">{call_to_action_short}</span> | '
    social_html += """Coffee Card by CafeCorner</div></div></div>"""
    st.markdown(social_html, unsafe_allow_html=True)

# --- FULL PROFILE DETAILS EXPANDER (Only for HTML card now) ---
def render_full_profile_details_expander(profile_data: Dict, profile_id: str):
    # This is essentially the old render_full_profile_details function,
    # to be used ONLY when the HTML/CSS card is displayed.
    if not profile_data: return

    expander_title = f"View Full Profile Details for {profile_data.get('name', 'N/A')}"
    with st.expander(expander_title, expanded=False):
        # ... (Contents from your original render_full_profile_details)
        if profile_data.get("taglineOrBriefSummary"):
            st.markdown(f"**Tagline/Summary:** {profile_data['taglineOrBriefSummary']}")
        if profile_data.get("primaryProfileUrlForCard"):
            st.markdown(f"🔗 [View LinkedIn/Profile]({profile_data['primaryProfileUrlForCard']})")
        st.markdown("---")
        skills_all = profile_data.get("skills", [])
        if skills_all:
            st.subheader("🛠️ Skills")
            num_skill_cols = min(len(skills_all), 4) if skills_all else 1
            skill_cols = st.columns(num_skill_cols)
            for idx, skill_name in enumerate(skills_all):
                with skill_cols[idx % num_skill_cols]:
                    with st.popover(skill_name, use_container_width=True):
                        st.markdown(f"### {skill_name}")
                        experiences_with_skill = []
                        for exp_item in profile_data.get("experiences", []): # Renamed exp to exp_item
                            for sd in exp_item.get("skillDetails", []):
                                if sd.get("skillName") == skill_name:
                                    experiences_with_skill.append({"role": exp_item.get("role", "N/A"), "company": exp_item.get("company", "N/A"), "context": sd.get("contextualSnippet", "No specific context.")}); break
                        projects_with_skill = []
                        for proj_item in profile_data.get("projects", []): # Renamed proj to proj_item
                            if skill_name in proj_item.get("skillsUsed", []): projects_with_skill.append(proj_item.get("projectName", "N/A"))
                        if experiences_with_skill:
                            st.markdown("**Applied in Experiences:**")
                            for e in experiences_with_skill:
                                st.markdown(f"- **{e['role']} at {e['company']}**: _{e['context']}_")
                            st.markdown("---")
                        if projects_with_skill:
                            st.markdown("**Used in Projects:**")
                            for p in projects_with_skill:
                                st.markdown(f"- {html.escape(p)}") # Also added html.escape for safety
                        if not experiences_with_skill and not projects_with_skill: st.caption("General skill.")
            st.markdown("---")
        key_achievements_all = profile_data.get("keyAchievementsOverall", [])
        if key_achievements_all: 
            st.subheader("🏆 Key Achievements")
            for ach in key_achievements_all:
                st.markdown(f"- {html.escape(ach)}")
            st.markdown("---")
        experiences_all = profile_data.get("experiences", [])
        if experiences_all:
            st.subheader("💼 Experience")
            for exp_idx, exp_item in enumerate(experiences_all): # Renamed exp to exp_item
                st.markdown(f"#### {exp_item.get('role', 'N/A')} at {exp_item.get('company', 'N/A')}"); st.caption(f"_{exp_item.get('dates', '')}_")
                if exp_item.get('description'): st.markdown(exp_item.get('description').replace("\n", "  \n"))
                skill_details = exp_item.get("skillDetails", [])
                if skill_details:
                    st.markdown("**Skills/Tools Used in this Role:**")
                    num_exp_skill_cols = min(len(skill_details), 4) if skill_details else 1
                    exp_skill_cols = st.columns(num_exp_skill_cols)
                    for s_idx, skill_info in enumerate(skill_details):
                        skill_name_exp = skill_info.get("skillName"); context_exp = skill_info.get("contextualSnippet", "No context."); related_skills_exp = skill_info.get("relatedSkillsInThisExperience", [])
                        with exp_skill_cols[s_idx % num_exp_skill_cols]:
                            with st.popover(skill_name_exp, use_container_width=True):
                                st.markdown(f"### {skill_name_exp}"); st.markdown(f"**Context:** {context_exp}")
                                if related_skills_exp: st.markdown(f"**Related:** {', '.join(related_skills_exp)}")
                    st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("---")
            st.markdown("---")
        education_all = profile_data.get("education", [])
        if education_all:
            st.subheader("🎓 Education")
            for edu_item in education_all: # Renamed edu to edu_item
                st.markdown(f"#### {edu_item.get('degree', 'N/A')} - _{edu_item.get('fieldOfStudy', '')}_"); st.markdown(f"{edu_item.get('institution', 'N/A')} ({edu_item.get('dates', '')})")
                if edu_item.get('description'): st.write(edu_item.get('description').replace("\n", "  \n")); st.markdown("---")
            st.markdown("---")
        projects_all = profile_data.get("projects", [])
        if projects_all:
            st.subheader("🚀 Projects")
            for proj_idx, proj_item in enumerate(projects_all): # Renamed proj to proj_item
                st.markdown(f"#### {proj_item.get('projectName', 'N/A')}"); st.caption(f"_{proj_item.get('datesOrDuration', '')}_")
                if proj_item.get('projectUrl'): st.markdown(f"🔗 [View Project]({proj_item.get('projectUrl')})")
                if proj_item.get('description'): st.write(proj_item.get('description').replace("\n", "  \n"))
                skills_used_proj = proj_item.get("skillsUsed", [])
                if skills_used_proj:
                    st.markdown("**Skills/Tech Used:**")
                    num_proj_skill_cols = min(len(skills_used_proj), 4) if skills_used_proj else 1
                    proj_skill_cols = st.columns(num_proj_skill_cols)
                    for s_idx, skill_name_proj in enumerate(skills_used_proj):
                        with proj_skill_cols[s_idx % num_proj_skill_cols]:
                            with st.popover(skill_name_proj, use_container_width=True):
                                st.markdown(f"### {skill_name_proj}"); st.caption("Used in this project.")
                    st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("---")
            st.markdown("---")
        interests_all = profile_data.get("interests", []); hobbies_all = profile_data.get("hobbies", [])
        if interests_all: st.subheader("💡 Interests"); st.markdown("- " + "\n- ".join(interests_all))
        if hobbies_all: st.subheader("🎨 Hobbies"); st.markdown("- " + "\n- ".join(hobbies_all))


# --- Example Usage Data (init_session_state) ---
# Remains the same
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
                    "skillsUsed": ["Multi-agent Systems", "Financial Research", "Streamlit", "LLMs", "Python"],
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
                    "skillsUsed": ["Accessibility", "Figma", "User Testing", "Mobile UI/UX"],
                }
            ]
        }
        st.session_state.profiles_new_structure = [
            EXAMPLE_YOUR_COFFEE_CARD_DATA_HOMEN,
            EXAMPLE_YOUR_COFFEE_CARD_DATA_JANE
        ]
    if 'editing_profile_id_dialog' not in st.session_state:
        st.session_state.editing_profile_id_dialog = None


# --- EDIT DIALOG (Existing, with full form) ---
# edit_profile_modal remains unchanged
@st.dialog("Edit Profile", width="large")
def edit_profile_modal(profile_id_to_edit: str):
    profile_index = -1
    profile_data_for_dialog = None
    for i, p in enumerate(st.session_state.profiles_new_structure):
        if p.get("id") == profile_id_to_edit:
            profile_data_for_dialog = p
            profile_index = i
            break
    if not profile_data_for_dialog:
        st.error("Profile data not found. Cannot edit.")
        if st.button("Close Dialog", key=f"close_dialog_notfound_{profile_id_to_edit}"):
            st.session_state.editing_profile_id_dialog = None; st.rerun()
        return
    st.subheader(f"Editing: {profile_data_for_dialog.get('name', 'N/A')}")
    st.markdown(f"_Profile ID: {profile_id_to_edit}_"); st.markdown("---")
    with st.form(key=f"edit_form_{profile_id_to_edit}"):
        updated_data = {}
        st.markdown("### 👤 Basic Information")
        updated_data["name"] = st.text_input("Name", value=profile_data_for_dialog.get("name", ""), key=f"edit_name_{profile_id_to_edit}")
        updated_data["title"] = st.text_input("Title", value=profile_data_for_dialog.get("title", ""), key=f"edit_title_{profile_id_to_edit}")
        updated_data["profilePictureUrlForCard"] = st.text_input("Profile Picture URL", value=profile_data_for_dialog.get("profilePictureUrlForCard", ""), key=f"edit_avatar_{profile_id_to_edit}")
        updated_data["taglineOrBriefSummary"] = st.text_area("Tagline/Summary", value=profile_data_for_dialog.get("taglineOrBriefSummary", ""), height=100, key=f"edit_tagline_{profile_id_to_edit}")
        updated_data["primaryProfileUrlForCard"] = st.text_input("Primary Profile URL (e.g., LinkedIn)", value=profile_data_for_dialog.get("primaryProfileUrlForCard", ""), key=f"edit_profileurl_{profile_id_to_edit}")
        updated_data["callToActionForCard"] = st.text_input("Call to Action (for card)", value=profile_data_for_dialog.get("callToActionForCard", ""), key=f"edit_cta_{profile_id_to_edit}")
        updated_data["location"] = st.text_input("Location", value=profile_data_for_dialog.get("location", ""), key=f"edit_location_{profile_id_to_edit}")
        st.markdown("---"); st.markdown("### 🛠️ Skills")
        skills_text = join_list_to_text_area(profile_data_for_dialog.get("skills"))
        updated_data["skills_text_area"] = st.text_area("Skills (one per line)", value=skills_text, height=150, key=f"edit_skills_{profile_id_to_edit}", help="Enter each skill on a new line.")
        st.markdown("---"); st.markdown("### 💡 Interests")
        interests_text = join_list_to_text_area(profile_data_for_dialog.get("interests"))
        updated_data["interests_text_area"] = st.text_area("Interests (one per line)", value=interests_text, height=100, key=f"edit_interests_{profile_id_to_edit}")
        st.markdown("---"); st.markdown("### 🎨 Hobbies")
        hobbies_text = join_list_to_text_area(profile_data_for_dialog.get("hobbies"))
        updated_data["hobbies_text_area"] = st.text_area("Hobbies (one per line)", value=hobbies_text, height=100, key=f"edit_hobbies_{profile_id_to_edit}")
        st.markdown("---"); st.markdown("### 🏆 Key Achievements (Overall)")
        achievements_text = join_list_to_text_area(profile_data_for_dialog.get("keyAchievementsOverall"))
        updated_data["achievements_text_area"] = st.text_area("Key Achievements (one per line)", value=achievements_text, height=150, key=f"edit_achievements_{profile_id_to_edit}")
        st.markdown("---"); st.markdown("### 💼 Experiences")
        experiences_data = profile_data_for_dialog.get("experiences", [])
        updated_experiences = []
        for i, exp in enumerate(experiences_data):
            st.markdown(f"#### Experience {i+1}")
            role = st.text_input(f"Role##exp{i}", value=exp.get("role", ""), key=f"edit_exp_{i}_role_{profile_id_to_edit}")
            company = st.text_input(f"Company##exp{i}", value=exp.get("company", ""), key=f"edit_exp_{i}_company_{profile_id_to_edit}")
            dates = st.text_input(f"Dates##exp{i}", value=exp.get("dates", ""), key=f"edit_exp_{i}_dates_{profile_id_to_edit}")
            is_current = st.checkbox(f"Current/Primary Role##exp{i}", value=exp.get("isCurrentOrPrimary", False), key=f"edit_exp_{i}_current_{profile_id_to_edit}")
            description = st.text_area(f"Description##exp{i}", value=exp.get("description", ""), height=120, key=f"edit_exp_{i}_desc_{profile_id_to_edit}")
            st.markdown(f"##### Skill Details for Experience {i+1}")
            current_skill_details = exp.get("skillDetails", [])
            updated_skill_details_for_exp = []
            for sd_idx, sd in enumerate(current_skill_details):
                st.markdown(f"###### Skill Detail {sd_idx+1}")
                sd_name = st.text_input(f"Skill Name##exp{i}sd{sd_idx}", value=sd.get("skillName", ""), key=f"edit_exp_{i}_sd_{sd_idx}_name_{profile_id_to_edit}")
                sd_context = st.text_area(f"Contextual Snippet##exp{i}sd{sd_idx}", value=sd.get("contextualSnippet", ""), height=70, key=f"edit_exp_{i}_sd_{sd_idx}_context_{profile_id_to_edit}")
                sd_related_text = join_list_to_comma_separated(sd.get("relatedSkillsInThisExperience", []))
                sd_related_input = st.text_input(f"Related Skills (comma-separated)##exp{i}sd{sd_idx}", value=sd_related_text, key=f"edit_exp_{i}_sd_{sd_idx}_related_{profile_id_to_edit}")
                if sd_name: updated_skill_details_for_exp.append({"skillName": sd_name, "contextualSnippet": sd_context, "relatedSkillsInThisExperience": parse_comma_separated_to_list(sd_related_input)})
            updated_experiences.append({"role": role, "company": company, "dates": dates, "isCurrentOrPrimary": is_current, "description": description, "skillDetails": updated_skill_details_for_exp})
            st.markdown("---")
        updated_data["experiences"] = updated_experiences
        st.markdown("---"); st.markdown("### 🎓 Education")
        education_data = profile_data_for_dialog.get("education", [])
        updated_education = []
        for i, edu in enumerate(education_data):
            st.markdown(f"#### Education {i+1}")
            institution = st.text_input(f"Institution##edu{i}", value=edu.get("institution", ""), key=f"edit_edu_{i}_inst_{profile_id_to_edit}")
            degree = st.text_input(f"Degree##edu{i}", value=edu.get("degree", ""), key=f"edit_edu_{i}_degree_{profile_id_to_edit}")
            fieldOfStudy = st.text_input(f"Field of Study##edu{i}", value=edu.get("fieldOfStudy", ""), key=f"edit_edu_{i}_field_{profile_id_to_edit}")
            dates_edu = st.text_input(f"Dates##edu{i}", value=edu.get("dates", ""), key=f"edit_edu_{i}_dates_edu_{profile_id_to_edit}")
            description_edu = st.text_area(f"Description##edu{i}", value=edu.get("description", ""), height=100, key=f"edit_edu_{i}_desc_{profile_id_to_edit}")
            updated_education.append({"institution": institution, "degree": degree, "fieldOfStudy": fieldOfStudy, "dates": dates_edu, "description": description_edu})
            st.markdown("---")
        updated_data["education"] = updated_education
        st.markdown("---"); st.markdown("### 🚀 Projects")
        projects_data = profile_data_for_dialog.get("projects", [])
        updated_projects = []
        for i, proj in enumerate(projects_data):
            st.markdown(f"#### Project {i+1}")
            projectName = st.text_input(f"Project Name##proj{i}", value=proj.get("projectName", ""), key=f"edit_proj_{i}_name_{profile_id_to_edit}")
            datesOrDuration = st.text_input(f"Dates/Duration##proj{i}", value=proj.get("datesOrDuration", ""), key=f"edit_proj_{i}_dates_{profile_id_to_edit}")
            projectUrl = st.text_input(f"Project URL##proj{i}", value=proj.get("projectUrl", ""), key=f"edit_proj_{i}_url_{profile_id_to_edit}")
            description_proj = st.text_area(f"Description##proj{i}", value=proj.get("description", ""), height=120, key=f"edit_proj_{i}_desc_{profile_id_to_edit}")
            skillsUsed_text = join_list_to_text_area(proj.get("skillsUsed"))
            skillsUsed_input = st.text_area(f"Skills Used (one per line)##proj{i}", value=skillsUsed_text, height=100, key=f"edit_proj_{i}_skills_{profile_id_to_edit}")
            updated_projects.append({"projectName": projectName, "datesOrDuration": datesOrDuration, "projectUrl": projectUrl, "description": description_proj, "skillsUsed": parse_text_area_to_list(skillsUsed_input)})
            st.markdown("---")
        updated_data["projects"] = updated_projects
        submitted = st.form_submit_button("Save Changes")
        if submitted:
            final_profile_update = {
                "id": profile_id_to_edit, "name": updated_data["name"], "title": updated_data["title"],
                "profilePictureUrlForCard": updated_data["profilePictureUrlForCard"], "taglineOrBriefSummary": updated_data["taglineOrBriefSummary"],
                "primaryProfileUrlForCard": updated_data["primaryProfileUrlForCard"], "callToActionForCard": updated_data["callToActionForCard"],
                "location": updated_data["location"], "skills": parse_text_area_to_list(updated_data["skills_text_area"]),
                "interests": parse_text_area_to_list(updated_data["interests_text_area"]), "hobbies": parse_text_area_to_list(updated_data["hobbies_text_area"]),
                "keyAchievementsOverall": parse_text_area_to_list(updated_data["achievements_text_area"]),
                "experiences": updated_data["experiences"], "education": updated_data["education"], "projects": updated_data["projects"],
            }
            if profile_index != -1: st.session_state.profiles_new_structure[profile_index] = final_profile_update; st.toast("Profile updated successfully!", icon="✔️")
            else: st.toast("Error: Profile not found for update.", icon="❌")
            st.session_state.editing_profile_id_dialog = None; st.rerun()
    if st.button("Cancel", key=f"cancel_edit_dialog_{profile_id_to_edit}"):
        st.session_state.editing_profile_id_dialog = None; st.rerun()


def main():
    st.set_page_config(layout="wide")
    init_session_state()
    load_css()

    st.title("☕ Coffee Card Profiles Showcase")
    st.caption("Displaying profiles with HTML/CSS, Comprehensive Native Streamlit Card, Shareable Previews.")
    st.markdown("---")

    if not st.session_state.get('profiles_new_structure'):
        st.info("No profiles yet. Add some to see the magic!")
    else:
        card_type_to_show = st.radio(
            "Select Card Display Method:",
            ("HTML/CSS Card (Concise + Full Details Expander)", "Native Streamlit Card (Comprehensive)"), # Simplified options
            index=1, # Default to Native Comprehensive
            horizontal=True
        )
        st.markdown("---")

        num_columns = 1 # Each profile takes a full column width for comprehensive view
        if card_type_to_show == "HTML/CSS Card (Concise + Full Details Expander)":
            num_columns = 2 # Can fit two concise HTML cards side-by-side

        cols = st.columns(num_columns)
        current_col_idx = 0

        for profile_data_loop in st.session_state.profiles_new_structure:
            profile_id_loop = profile_data_loop.get("id", str(uuid.uuid4()))
            target_col = cols[current_col_idx % num_columns]

            with target_col:
                st.header(f"Profile: {profile_data_loop.get('name')}", divider="rainbow")

                if card_type_to_show == "HTML/CSS Card (Concise + Full Details Expander)":
                    st.subheader("Method 1: HTML/CSS Card ✨ (Concise)")
                    render_coffee_card_concise(profile_data_loop, profile_id_loop)
                    render_full_profile_details_expander(profile_data_loop, profile_id_loop) # Show expander
                    st.markdown("<br>", unsafe_allow_html=True)

                elif card_type_to_show == "Native Streamlit Card (Comprehensive)":
                    st.subheader("Method 2: Native Streamlit Card 🎈 (Comprehensive)")
                    render_coffee_card_native_comprehensive(profile_data_loop, profile_id_loop)
                    # No separate expander needed here as the card is comprehensive
                    st.markdown("<br>", unsafe_allow_html=True)

                # Previews (Common to both card display methods)
                st.markdown("<h5 style='text-align: center; color: var(--cc-accent-dark-brown); margin-top:15px; margin-bottom:5px;'>🔗 Shareable Previews</h5>", unsafe_allow_html=True)
                preview_cols = st.columns(2)
                with preview_cols[0]:
                    render_apple_wallet_preview(profile_data_loop, profile_id_loop)
                with preview_cols[1]:
                    render_social_png_preview(profile_data_loop, profile_id_loop)
                st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

            current_col_idx += 1

        if st.session_state.editing_profile_id_dialog:
            profile_id_to_show_dialog_for = st.session_state.editing_profile_id_dialog
            edit_profile_modal(profile_id_to_show_dialog_for)

if __name__ == "__main__":
    main()