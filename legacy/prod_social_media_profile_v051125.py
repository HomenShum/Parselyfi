# -*- coding: utf-8 -*-
import streamlit as st
import json
import re # For key formatting
import io # Needed for reading uploaded file
import base64 # Needed for encoding uploaded image
import copy # For deep copying data for editing
import time # For simulated delays
import html # For escaping data in HTML strings

# -------------------------------------
# Styling (Applying Coffee Card Theme)
# -------------------------------------
# We will modify the existing CSS_STYLES variable to reflect the Coffee Card theme.

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

    /* --- Original Theme Variables (kept for potential fallback/other elements) --- */
    /* Commenting out originals to avoid confusion, using CC vars directly */
    /* --coffee-dark: #4E342E; */
    /* --coffee-medium: #654A43; */
    /* --coffee-light: #8D6E63; */
    /* --cream: #FFF8E1; */
    /* --crema: #F5ECD7; */
    /* --brass: #D4AD76; */
    /* --neutral-bg: #FFFFFF; */
    /* --neutral-surface: #FAFAFA; */
    /* --neutral-pill: #EAEAEA; */
    /* --neutral-border: #E0E0E0; */
    /* --text-primary: #37352F; */
    /* --text-tertiary: #908F8C; */
    /* --text-missing: #B6B6B5; */

    /* Platform specific colors (unchanged) */
    --linkedin: #0A66C2;
    --instagram: #E1306C;
    --twitter-x: #1DA1F2;
    --facebook: #1877F2;
    --tiktok: #000000;
    --reddit: #FF4500;
    --github: #333;
    --other: #777777;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--cc-text-general); /* Use general text color */
    /* background-color: var(--neutral-surface); */ /* Consider changing body bg to complement card */
    background-color: #fdfbf7; /* Slightly darker off-white */
    line-height: 1.6;
}

/* Card styling - Applying Coffee Card Theme */
.card {
    border: 2px solid var(--cc-accent-dark-brown); /* Prominent dark brown border */
    border-radius: 12px;                     /* Strongly rounded corners */
    background: var(--cc-bg-main);           /* Light warm off-white background */
    padding: 1.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 3px, rgba(0, 0, 0, 0.05) 0px 20px 25px -5px, rgba(0, 0, 0, 0.04) 0px 10px 10px -5px;
    transition: transform 0.2s ease-in-out, box-shadow 0.3s ease-in-out;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
}

/* Remove or adjust conflicting coffee-card-container styles if .card handles the main look */
.coffee-card-container {
    /* The .card style above should now define the main look */
    /* background-image: linear-gradient(to bottom right, rgba(245, 236, 215, 0.4) 0%, rgba(255, 255, 255, 1) 80%); */
    /* border-left: 5px solid var(--brass); */ /* Remove this border */
    position: relative;
    overflow: hidden;
}

.coffee-card-container::before {
    /* Keep or adjust the top gradient bar if desired */
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 8px; /* Maybe make thinner? */
    /* background: linear-gradient(to right, var(--cc-accent-theme-brown), var(--cc-bg-main)); */ /* Example using theme colors */
    background: var(--cc-accent-dark-brown); /* Or just a solid accent line */
    opacity: 0.8;
}

/* Header styling */
.header-content {
    display: flex;
    align-items: flex-start;
    gap: 1.8rem;
}

img.avatar {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    border: 2px solid var(--cc-accent-light-tan); /* Light tan border */
    object-fit: cover;
    flex-shrink: 0;
    box-shadow: rgba(0, 0, 0, 0.05) 0px 2px 4px, rgba(0, 0, 0, 0.1) 0px 12px 15px -8px;
}

/* Card Name */
.header-text h1 {
    color: var(--cc-text-name); /* Dark brown text for name */
    margin: 0 0 0.3rem 0;
    font-size: 1.9rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}

/* Card Title/Subtitle */
.header-text p.headline {
    font-size: 1.1rem;
    color: var(--cc-text-title); /* Dark gray text for subtitle */
    margin: 0.3rem 0 0.8rem 0;
    line-height: 1.4;
}

/* Section styling */
.section {
    margin-bottom: 1.5rem;
    padding-bottom: 1.2rem;
    /* Apply light tan border as separator */
    border-bottom: 1px solid var(--cc-accent-light-tan);
}

.section:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}

/* Pills styling - Applying Coffee Card Theme (.tag equivalent) */
.pill {
    display: inline-block;
    padding: 0.35rem 0.9rem; /* Slightly more padding for pill feel */
    margin: 0.25rem 0.4rem 0.25rem 0;
    border-radius: 12px;                     /* Strongly rounded (pill shape) */
    background: var(--cc-bg-tag);            /* Light peachy/tan background */
    font-size: 0.85rem;
    color: var(--cc-text-tag);               /* Dark brown text */
    font-weight: 500;
    line-height: 1.4;
    transition: all 0.2s ease;
    white-space: nowrap;
    border: 1px solid var(--cc-accent-light-tan); /* Subtle light tan border */
}

.pill:hover {
    /* background: var(--cream); */ /* Using original hover */
    background: #f7e1ce; /* Slightly darker version of tag bg */
    color: var(--cc-accent-dark-brown); /* Keep text color */
    transform: translateY(-1px);
    box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 2px;
}

.pill-container {
    margin-top: 0.75rem;
    margin-bottom: 0.75rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
}

/* Typography */
/* Section Headers inside Coffee Card (e.g., Interests, Skills) */
h5 {
    /* color: var(--coffee-medium); */ /* Original */
    color: var(--cc-accent-dark-brown); /* Dark brown, matching tag text/border */
    font-weight: 600;
    font-size: 1.05rem; /* Keep size */
    margin-bottom: 0.8rem;
    margin-top: 1.2rem;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
}

/* --- Typography inside Detailed Tabs (Keep Notion/Cafe inspired?) --- */
/* Let's keep these mostly as they were, maybe adjust colors slightly if needed */
h3 { /* Section titles in Overview tab */
    color: var(--cc-accent-dark-brown); /* Use dark brown for consistency? */
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    font-size: 1.4rem;
    font-weight: 600;
    border-bottom: 1px solid var(--cc-accent-light-tan); /* Use theme border */
    padding-bottom: 0.5rem;
}

h4 { /* Subsection titles within cards/sections */
    color: var(--cc-accent-theme-brown); /* Use medium theme brown? */
    margin-top: 1rem;
    margin-bottom: 0.75rem;
    font-size: 1.1rem;
    font-weight: 600;
}

.platform-card h4 {
    margin-top: 0;
}

h4.consumption-title {
    margin-top: 1.5rem;
    color: var(--cc-text-secondary); /* Keep secondary */
    font-size: 1.2rem;
    border-top: 1px dashed var(--cc-accent-light-tan); /* Use theme border dashed */
    padding-top: 1rem;
}

h5.consumption-subtitle {
    color: var(--cc-accent-dark-brown); /* Use dark brown */
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
}
/* END OF TAB TYPOGRAPHY */

.coffee-icon {
    margin-right: 0.5rem;
    opacity: 0.9;
    /* Could potentially use a specific coffee-related emoji or SVG */
}

.detail-icon {
    margin-right: 0.5rem;
    opacity: 0.75;
    color: var(--cc-accent-theme-brown); /* Use medium brown */
    font-size: 0.9em;
}

/* Experiences styling - Apply .experiences-display theme to .prewrap */
p.prewrap {
    white-space: pre-wrap;
    margin-bottom: 0.6rem;
    background: var(--cc-bg-exp);      /* Slightly different off-white */
    padding: 1rem;
    border-radius: 4px;              /* Slightly rounded */
    font-size: 0.95rem;
    line-height: 1.5;
    color: var(--cc-text-exp);         /* Dark gray text */
    border: 1px solid var(--cc-accent-light-tan); /* Light tan border */
}

/* Missing data */
em.missing {
    color: var(--cc-text-placeholder); /* Medium gray */
    font-style: italic;
    font-weight: 400;
}

/* --- Button Styling (Primarily for Edit Form) --- */
/* Default Streamlit Button - Apply Coffee Card Theme */
/* This will also apply to buttons in st.dialog if they are inside an st.form */
div[data-testid="stForm"] .stButton button:not([kind="secondary"]) {
    border: 1px solid var(--cc-btn-default-text) !important; /* Theme brown border */
    color: var(--cc-btn-default-text) !important;           /* Theme brown text */
    background-color: transparent; /* Ensure default bg is transparent */
    border-radius: 4px !important;         /* Slightly rounded */
    padding: 0.4rem 0.8rem; /* Adjust padding if needed */
}

div[data-testid="stForm"] .stButton button:not([kind="secondary"]):hover {
    border-color: var(--cc-btn-default-hover-border) !important; /* Darker brown border */
    color: var(--cc-btn-default-hover-text) !important;           /* Darker brown text */
    background-color: var(--cc-bg-btn-default-hover) !important; /* Lighter orange/brown bg */
}

/* Secondary Streamlit Button (e.g., Delete) */
div[data-testid="stForm"] .stButton button[kind="secondary"] {
    border: 1px solid var(--cc-btn-delete-text) !important; /* Reddish border */
    color: var(--cc-btn-delete-text) !important;           /* Reddish text */
    background-color: transparent;
    border-radius: 4px !important;
    padding: 0.4rem 0.8rem;
}

div[data-testid="stForm"] .stButton button[kind="secondary"]:hover {
    border-color: var(--cc-btn-delete-hover-border) !important; /* Darker red border */
    color: var(--cc-btn-delete-hover-text) !important;           /* Darker red text */
    background-color: var(--cc-bg-btn-delete-hover) !important; /* Very light red bg */
}

/* Subtle pattern for background (Optional - keep or remove) */
body::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    /* background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23d4ad76' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E"); */
    /* Let's remove the pattern for a cleaner look matching the coffee card */
    background-image: none;
    z-index: -1;
}

/* --- Missing Fields Summary Styling (matches spec) --- */
.stWarning > div > div > div > p > h6 { /* Target h6 inside warning */
    color: var(--cc-accent-dark-brown) !important; /* Dark brown */
    font-weight: 600;
}
.stWarning > div > div > div > p > ul > li > strong { /* Target section path */
     color: var(--cc-accent-theme-brown); /* Medium brown */
}
.stWarning > div > div > div > p > ul > li > ul > li { /* Target missing field item */
     color: var(--cc-text-missing-summary); /* Medium-dark gray */
}
</style>
"""

# -------------------------------------
# Helper Functions
# -------------------------------------

def safe_get(data, key_list, default=None):
    if data is None: return default
    _data = data
    for key in key_list:
        try: _data = _data[key]
        except (KeyError, TypeError, IndexError): return default
    return _data if _data is not None else default

def format_key_to_title(key):
    if not isinstance(key, str): return str(key)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', key)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    s3 = s2.replace('_', ' ')
    return s3.title()

def render_missing_html():
    return '<em class="missing">Not Provided</em>'

def render_pills_html(items, label=None, show_label=True, pill_class="pill", show_empty_fields=False, max_pills_mini=None):
    if not items or not isinstance(items, list):
        if show_empty_fields:
            label_html = f'<h5><span class="coffee-icon"></span>{html.escape(label)}</h5>' if label and show_label else ''
            return f'{label_html}{render_missing_html()}'
        return ""

    items_str = [html.escape(str(item).strip()) for item in items if item and str(item).strip()]
    if not items_str:
        if show_empty_fields:
            label_html = f'<h5><span class="coffee-icon"></span>{html.escape(label)}</h5>' if label and show_label else ''
            return f'{label_html}{render_missing_html()}'
        return ""

    pills_to_display = items_str
    more_pills_text = ""
    if max_pills_mini is not None and len(items_str) > max_pills_mini:
        pills_to_display = items_str[:max_pills_mini]
        remaining_count = len(items_str) - max_pills_mini
        # Simple text for "more", actual expansion needs JS or Streamlit interactivity
        more_pills_text = f'<span class="pill" style="font-style:italic; background-color: transparent; border: 1px dashed var(--cc-accent-light-tan); color: var(--cc-text-secondary);">+{remaining_count} more...</span>'


    pills_html_content = "".join(f'<span class="{pill_class}">{item}</span>' for item in pills_to_display) + more_pills_text
    label_html = f'<h5><span class="coffee-icon"></span>{html.escape(label)}</h5>' if label and show_label else ''
    return f'{label_html}<div class="pill-container">{pills_html_content}</div>'

def is_value_empty(value):
    if value is None: return True
    if isinstance(value, str) and not value.strip(): return True
    if isinstance(value, (list, dict)) and not value: return True
    return False

def make_initials_svg_avatar(name: str, size: int = 80,
                            bg: str = "var(--cc-accent-dark-brown)", 
                            fg: str = "var(--cc-bg-main)") -> str:    
    display_name = name if name and name != render_missing_html() else "?"
    if not isinstance(display_name, str): display_name = str(display_name)
    initials = "".join([w[0].upper() for w in display_name.split()][:2]) or "?"
    bg_color = "#6b4f4f" 
    fg_color = "#fff8f0" 
    svg = f'''
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">
<circle cx="{size/2}" cy="{size/2}" r="{size/2}" fill="{bg_color}"/>
<text x="50%" y="50%" fill="{fg_color}" font-size="{int(size/2.2)}"
        text-anchor="middle" dominant-baseline="central"
        font-family="sans-serif" font-weight="500">{initials}</text>
</svg>'''
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"

def get_avatar_url(name, uploaded_avatar_file_obj=None, profile_pic_url_from_card=None):
    if profile_pic_url_from_card and not is_value_empty(profile_pic_url_from_card):
        # Prioritize the URL directly from the card data if valid
        # Add basic validation if it's a common image type or base64
        if profile_pic_url_from_card.startswith("http") or profile_pic_url_from_card.startswith("data:image"):
            return profile_pic_url_from_card
    if uploaded_avatar_file_obj is not None:
        try:
            image_bytes = uploaded_avatar_file_obj.getvalue()
            b64_image = base64.b64encode(image_bytes).decode()
            mime_type = uploaded_avatar_file_obj.type
            return f"data:{mime_type};base64,{b64_image}"
        except Exception as e:
            st.warning(f"Could not process uploaded avatar: {e}. Using initials.")
    return make_initials_svg_avatar(name if name else "??", size=80) # Adjusted size to match CSS


# -------------------------------------
# 1. Specialized Coffee Card Renderer
# -------------------------------------
def render_coffee_card_html(card_data, main_avatar_url, show_empty_fields=False): # main_avatar_url is passed from get_avatar_url
    if not card_data or not isinstance(card_data, dict):
        return f'<div class="card coffee-card-container"><p>{render_missing_html()} (Coffee Card data missing)</p></div>'

    # --- Extract data using safe_get ---
    name_raw = safe_get(card_data, ['name'])
    title_raw = safe_get(card_data, ['title'])
    tagline_raw = safe_get(card_data, ['taglineOrBriefSummary'])
    profile_url_raw = safe_get(card_data, ['primaryProfileUrlForCard'])
    cta_raw = safe_get(card_data, ['callToActionForCard'])
    location_raw = safe_get(card_data, ['location'])

    experiences_list = safe_get(card_data, ['experiences'], [])
    education_list = safe_get(card_data, ['education'], [])
    projects_list = safe_get(card_data, ['projects'], [])
    achievements_list = safe_get(card_data, ['keyAchievementsOverall'], [])

    skills_raw = safe_get(card_data, ['skills'])
    interests_raw = safe_get(card_data, ['interests'])
    hobbies_raw = safe_get(card_data, ['hobbies'])

    # --- Mini Card Header ---
    name_html = f"<h1>{html.escape(name_raw)}</h1>" if not is_value_empty(name_raw) else (f"<h1>{render_missing_html()}</h1>" if show_empty_fields else "")
    title_html = f'<p class="headline">{html.escape(title_raw)}</p>' if not is_value_empty(title_raw) else (f'<p class="headline">{render_missing_html()}</p>' if show_empty_fields and name_html else "")
    tagline_html = f'<p class="tagline">{html.escape(tagline_raw)}</p>' if not is_value_empty(tagline_raw) else ""
    location_mini_html = f'<p class="headline" style="font-size: 0.9rem;">📍 {html.escape(location_raw)}</p>' if not is_value_empty(location_raw) else ""
    
    profile_url_html = ""
    if not is_value_empty(profile_url_raw):
        profile_url_html = f'<p class="card-url"><a href="{html.escape(profile_url_raw)}" target="_blank">🔗 {html.escape(profile_url_raw)}</a></p>'
    
    cta_html = ""
    if not is_value_empty(cta_raw):
        cta_html = f'<p class="card-cta">📣 {html.escape(cta_raw)}</p>'

    mini_card_header_html = f"""
    <div class="header-content">
        <img src="{main_avatar_url}" alt="Avatar" class="avatar">
        <div class="header-text">
            {name_html}
            {title_html}
            {tagline_html}
            {location_mini_html}
            {profile_url_html}
            {cta_html}
        </div>
    </div>
    """

    # Mini Card Primary Experience & Top Skills
    primary_exp_html = ""
    if isinstance(experiences_list, list):
        for exp in experiences_list:
            if isinstance(exp, dict) and exp.get('isCurrentOrPrimary'):
                role = safe_get(exp, ['role'], render_missing_html() if show_empty_fields else "")
                company = safe_get(exp, ['company'], render_missing_html() if show_empty_fields else "")
                brief_summary = safe_get(exp, ['briefSummaryForMiniCard'])
                
                if role or company or (brief_summary and show_empty_fields): # Show if any part exists
                    primary_exp_html = '<div class="mini-card-experience">'
                    if role and company:
                         primary_exp_html += f'<strong>{html.escape(role)}</strong> at <span class="company">{html.escape(company)}</span>'
                    elif role:
                         primary_exp_html += f'<strong>{html.escape(role)}</strong>'
                    elif company:
                         primary_exp_html += f'<span class="company">{html.escape(company)}</span>'

                    if not is_value_empty(brief_summary):
                        primary_exp_html += f'<span class="brief-summary">{html.escape(brief_summary)}</span>'
                    primary_exp_html += '</div>'
                break # Found primary, stop looking

    top_skills_html = render_pills_html(
        skills_raw,
        label="Key Skills", # Label for mini-card skills
        show_label=True, # Show "Key Skills" label
        show_empty_fields=show_empty_fields,
        pill_class="pill",
        max_pills_mini=5 # Show up to 5 skills for mini-card
    )
    top_skills_section_html = f'<div class="section">{top_skills_html}</div>' if top_skills_html else ""


    # --- Full Card Sections (for "expansion") ---
    # Full Experiences
    full_experiences_html = ""
    if isinstance(experiences_list, list) and experiences_list:
        exp_items_html = ""
        for exp in experiences_list:
            if not isinstance(exp, dict): continue
            role = safe_get(exp, ['role'], "Role not specified")
            company = safe_get(exp, ['company'], "Company not specified")
            dates = safe_get(exp, ['dates'], "Dates not specified")
            description = safe_get(exp, ['description'], render_missing_html() if show_empty_fields else "")
            skill_details_list = safe_get(exp, ['skillDetails'], [])

            exp_item_content = f'<div class="experience-header"><h6 class="role">{html.escape(role)}</h6><p class="company-dates">{html.escape(company)} • {html.escape(dates)}</p></div>'
            if not is_value_empty(description) or show_empty_fields:
                exp_item_content += f'<p class="experience-description">{html.escape(description) if not is_value_empty(description) else render_missing_html()}</p>'

            if isinstance(skill_details_list, list) and skill_details_list:
                skill_details_html_parts = []
                for sd in skill_details_list:
                    if not isinstance(sd, dict): continue
                    sd_name = safe_get(sd, ['skillName'])
                    sd_context = safe_get(sd, ['contextualSnippet'])
                    sd_related = safe_get(sd, ['relatedSkillsInThisExperience'])

                    if sd_name or (show_empty_fields and (sd_context or sd_related)):
                        part_html = '<div class="skill-detail-item">'
                        part_html += f'<strong>{html.escape(sd_name) if sd_name else render_missing_html()}</strong>'
                        if not is_value_empty(sd_context):
                            part_html += f'<span class="context">{html.escape(sd_context)}</span>'
                        elif show_empty_fields:
                             part_html += f'<span class="context">{render_missing_html()}</span>'

                        if isinstance(sd_related, list) and sd_related:
                            part_html += f'<span class="related-skills-label">Related:</span>{render_pills_html(sd_related, show_label=False, pill_class="pill")}' # Smaller pills if needed
                        elif show_empty_fields and sd_related is not None : # Show if field exists but is empty
                            part_html += f'<span class="related-skills-label">Related:</span>{render_missing_html()}'
                        part_html += '</div>'
                        skill_details_html_parts.append(part_html)
                if skill_details_html_parts:
                    exp_item_content += f'<div>{"".join(skill_details_html_parts)}</div>'
            
            exp_items_html += f'<div class="experience-item">{exp_item_content}</div>'

        if exp_items_html:
            full_experiences_html = f'<h5><span class="coffee-icon"></span>Full Experience</h5>{exp_items_html}'
    elif show_empty_fields:
        full_experiences_html = f'<h5><span class="coffee-icon"></span>Full Experience</h5>{render_missing_html()}'
    full_experiences_section_html = f'<div class="section">{full_experiences_html}</div>' if full_experiences_html else ""


    # Education
    education_section_html = ""
    if isinstance(education_list, list) and education_list:
        edu_items_html = ""
        for edu in education_list:
            if not isinstance(edu, dict): continue
            institution = safe_get(edu, ['institution'], "Institution not specified")
            degree = safe_get(edu, ['degree'], "Degree not specified")
            field = safe_get(edu, ['fieldOfStudy'])
            dates = safe_get(edu, ['dates'], "Dates not specified")
            desc = safe_get(edu, ['description'])
            
            item_html = f'<div class="education-item">'
            item_html += f'<h6 class="degree-institution">{html.escape(degree)} - {html.escape(institution)}</h6>'
            field_dates_parts = []
            if field: field_dates_parts.append(html.escape(field))
            if dates: field_dates_parts.append(html.escape(dates))
            if field_dates_parts:
                item_html += f'<p class="field-dates">{" • ".join(field_dates_parts)}</p>'
            if not is_value_empty(desc):
                item_html += f'<p class="description">{html.escape(desc)}</p>'
            elif show_empty_fields and desc is not None:
                 item_html += f'<p class="description">{render_missing_html()}</p>'
            item_html += '</div>'
            edu_items_html += item_html
        if edu_items_html:
            education_section_html = f'<div class="section"><h5><span class="coffee-icon"></span>Education</h5>{edu_items_html}</div>'
    elif show_empty_fields:
        education_section_html = f'<div class="section"><h5><span class="coffee-icon"></span>Education</h5>{render_missing_html()}</div>'


    # Projects
    projects_section_html = ""
    if isinstance(projects_list, list) and projects_list:
        proj_items_html = ""
        for proj in projects_list:
            if not isinstance(proj, dict): continue
            name = safe_get(proj, ['projectName'], "Project not specified")
            dates = safe_get(proj, ['datesOrDuration'])
            desc = safe_get(proj, ['description'], render_missing_html() if show_empty_fields else "")
            skills_used = safe_get(proj, ['skillsUsed'])
            url = safe_get(proj, ['projectUrl'])

            item_html = f'<div class="project-item">'
            item_html += f'<h6 class="project-name">{html.escape(name)}</h6>'
            dates_url_parts = []
            if dates: dates_url_parts.append(html.escape(dates))
            if url: dates_url_parts.append(f'<span class="project-url"><a href="{html.escape(url)}" target="_blank">🔗 View Project</a></span>')
            if dates_url_parts:
                 item_html += f'<p class="dates-url">{" • ".join(dates_url_parts)}</p>'

            if not is_value_empty(desc) or show_empty_fields:
                item_html += f'<p class="description">{html.escape(desc) if not is_value_empty(desc) else render_missing_html()}</p>'
            
            if skills_used or (show_empty_fields and skills_used is not None):
                 item_html += render_pills_html(skills_used, label="Skills Used:", show_label=True, show_empty_fields=show_empty_fields, pill_class="pill")
            item_html += '</div>'
            proj_items_html += item_html
        if proj_items_html:
            projects_section_html = f'<div class="section"><h5><span class="coffee-icon"></span>Projects</h5>{proj_items_html}</div>'
    elif show_empty_fields:
        projects_section_html = f'<div class="section"><h5><span class="coffee-icon"></span>Projects</h5>{render_missing_html()}</div>'
        

    # Key Achievements Overall
    achievements_section_html = ""
    if isinstance(achievements_list, list) and achievements_list:
        ach_items_html = "<ul>" + "".join(f"<li>{html.escape(ach)}</li>" for ach in achievements_list if ach) + "</ul>"
        achievements_section_html = f'<div class="section"><h5><span class="coffee-icon"></span>Key Achievements</h5>{ach_items_html}</div>'
    elif show_empty_fields:
        achievements_section_html = f'<div class="section"><h5><span class="coffee-icon"></span>Key Achievements</h5>{render_missing_html()}</div>'

    # Full Skills, Interests, Hobbies lists (these are for the expanded view)
    full_skills_pills_content = render_pills_html(
        skills_raw, label="All Skills", show_label=True, show_empty_fields=show_empty_fields, pill_class="pill"
    )
    # Only show "All Skills" if it's different from "Key Skills" or if "Key Skills" wasn't shown
    # This logic is a bit complex for pure HTML; for now, just show it if it has content.
    # A better approach might be to always show "All Skills" and hide "Key Skills" if they are identical.
    # Or, ensure `max_pills_mini` is always less than total if you want both.
    # For now, we show it if it would render something.
    full_skills_section_html = f'<div class="section">{full_skills_pills_content}</div>' if full_skills_pills_content and len(skills_raw or []) > (5 if top_skills_html else 0) else ""


    interests_pills_content = render_pills_html(
        interests_raw, label="Interests", show_label=True, show_empty_fields=show_empty_fields, pill_class="pill"
    )
    interests_section_html = f'<div class="section">{interests_pills_content}</div>' if interests_pills_content else ""

    hobbies_pills_content = render_pills_html(
        hobbies_raw, label="Hobbies", show_label=True, show_empty_fields=show_empty_fields, pill_class="pill"
    )
    hobbies_section_html = f'<div class="section">{hobbies_pills_content}</div>' if hobbies_pills_content else ""


    # --- Final Card HTML ---
    card_html = f"""
    <div class="card coffee-card-container">
        {mini_card_header_html}
        {primary_exp_html} 
        {top_skills_section_html}

        {full_experiences_section_html}
        {education_section_html}
        {projects_section_html}
        {achievements_section_html}
        {full_skills_section_html if skills_raw and len(skills_raw) > 5 else ""} 
        {interests_section_html}
        {hobbies_section_html}
    </div>
    """
    return card_html


# --- render_nested_json_html (Mostly unchanged logic, will inherit theme via tags h3-h6, p, etc.) ---
# Ensure it uses the updated render_pills_html and render_missing_html
def render_nested_json_html(data, level=0, show_empty_fields=False):
    """
    Enhanced recursive renderer, optionally hiding empty fields.
    Will use styles defined globally for h3, h4, h5, p, .pill, em.missing etc.
    """
    html_output = ""
    indent_style = f"margin-left: {level * 0.5}rem;"

    if is_value_empty(data) and not show_empty_fields and level > 0:
         return ""

    if isinstance(data, dict):
        items_to_render = data.items()
        if not show_empty_fields:
            items_to_render = {k: v for k, v in data.items() if not is_value_empty(v)}.items()

        if not items_to_render and not show_empty_fields:
             return ""

        container_class = "platform-card-like" if level == 1 else "nested-item-container"
        platform_name = data.get('platformName', '').lower().replace(" ", "-").replace("/", "-").replace("+", "-").replace(".", "")
        if platform_name:
            container_class += f" {platform_name}"
        delay_style = f"animation-delay: {level * 0.1}s;"
        html_output += f'<div class="{container_class}" style="{indent_style} {delay_style}">'

        dict_to_iterate = data if show_empty_fields else dict(items_to_render)

        for key, value in dict_to_iterate.items():
            is_empty = is_value_empty(value)
            if is_empty and not show_empty_fields:
                continue

            title = html.escape(format_key_to_title(key))
            heading_level = min(level + 3, 6) # h3, h4, h5, h6...
            icon = ""

            # --- Icon Selection Logic (Keep existing logic) ---
            if heading_level == 3:
                if any(term in key.lower() for term in ['profile', 'fundamental', 'individual', 'target']): icon = '<span class="coffee-icon">☕</span>'
                elif any(term in key.lower() for term in ['platform', 'analysis', 'specific']): icon = '<span class="coffee-icon">🍵</span>'
                elif any(term in key.lower() for term in ['synthesis', 'cross', 'comprehensive']): icon = '<span class="coffee-icon">♨️</span>'
                elif any(term in key.lower() for term in ['summary', 'final', 'conclusion']): icon = '<span class="coffee-icon">🏮</span>'
                elif any(term in key.lower() for term in ['algorithm', 'perception', 'categorization']): icon = '<span class="coffee-icon">🧠</span>'
                else: icon = '<span class="coffee-icon">☕</span>'
            elif heading_level == 4:
                if any(term in key.lower() for term in ['skill', 'expertise', 'profession', 'capability']): icon = '<span class="diff-icon">🥄</span>'
                elif any(term in key.lower() for term in ['interest', 'hobby', 'like', 'preference']): icon = '<span class="diff-icon">🍰</span>'
                elif any(term in key.lower() for term in ['experience', 'work', 'job', 'career', 'education']): icon = '<span class="diff-icon">🥐</span>'
                elif any(term in key.lower() for term in ['consumption', 'media', 'content', 'watch', 'view', 'feed']): icon = '<span class="diff-icon">🧋</span>'
                elif any(term in key.lower() for term in ['privacy', 'security', 'visibility']): icon = '<span class="diff-icon">🔒</span>'
                elif any(term in key.lower() for term in ['network', 'community', 'connection', 'audience']): icon = '<span class="diff-icon">👥</span>'
                elif any(term in key.lower() for term in ['categorization', 'algorithm', 'perception']): icon = '<span class="diff-icon">🤖</span>'
                elif any(term in key.lower() for term in ['platform', 'name']): icon = '<span class="diff-icon">📱</span>'
                else: icon = '<span class="diff-icon">🍵</span>'
            elif heading_level == 5:
                if any(term in key.lower() for term in ['statistic', 'metrics', 'data', 'number', 'count']): icon = '<span class="detail-icon">📊</span>'
                elif any(term in key.lower() for term in ['insight', 'analysis', 'observation', 'thought']): icon = '<span class="detail-icon">🧠</span>'
                elif any(term in key.lower() for term in ['highlight', 'key', 'important', 'notable']): icon = '<span class="detail-icon">🔸</span>'
                elif any(term in key.lower() for term in ['location', 'place', 'geo']): icon = '<span class="detail-icon">📍</span>'
                elif any(term in key.lower() for term in ['time', 'date', 'period', 'duration']): icon = '<span class="detail-icon">⏱️</span>'
                elif any(term in key.lower() for term in ['tone', 'voice', 'style', 'writing']): icon = '<span class="detail-icon">✒️</span>'
                elif any(term in key.lower() for term in ['categorization', 'algorithm', 'hypothesis']): icon = '<span class="detail-icon">💭</span>'
                else: icon = '<span class="detail-icon">🔸</span>'
            elif heading_level == 6:
                 icon = '<span class="list-icon">✦</span>' # Simple list icon
            # --- End Icon Selection Logic ---

            # Render the heading (h3, h4, h5, h6) - styles applied via CSS
            html_output += f'<h{heading_level}>{icon}{title}</h{heading_level}>'

            # Apply special classes (consumption titles) - styles applied via CSS
            if 'consumption' in key.lower():
                if heading_level == 4:
                    html_output = html_output.replace(f'<h4>{icon}{title}</h4>', f'<h4 class="consumption-title">{icon}{title}</h4>')
                elif heading_level == 5:
                    html_output = html_output.replace(f'<h5>{icon}{title}</h5>', f'<h5 class="consumption-subtitle">{icon}{title}</h5>')

            rendered_value = render_nested_json_html(value, level + 1, show_empty_fields=show_empty_fields)
            html_output += rendered_value

        html_output += '</div>'

    elif isinstance(data, list):
        if not data:
            if show_empty_fields:
                html_output += f'<div style="{indent_style}">{render_missing_html()} (Empty List)</div>'
        else:
            # Use updated render_pills_html for lists of strings
            all_strings = all(isinstance(item, str) for item in data if item is not None)

            if all_strings and any(item and item.strip() for item in data):
                # Pass the flag to potentially show placeholder if list becomes empty after stripping etc.
                pills_content = render_pills_html(data, show_label=False, show_empty_fields=show_empty_fields, pill_class="pill")
                if pills_content:
                     html_output += f'<div style="{indent_style}">{pills_content}</div>'
                elif show_empty_fields:
                     html_output += f'<div style="{indent_style}">{render_missing_html()} (Empty String List)</div>'

            else: # List of simple primitives (non-string) or complex items
                is_simple_list = all(isinstance(item, (int, float, bool)) or item is None for item in data)

                if is_simple_list and any(item is not None for item in data):
                    list_items_html = ""
                    has_content = False
                    for item in data:
                         if not is_value_empty(item):
                             list_items_html += f"<li>{html.escape(str(item))}</li>"
                             has_content = True
                         elif show_empty_fields:
                             list_items_html += f"<li>{render_missing_html()}</li>"
                             has_content = True
                    if has_content:
                        html_output += f'<div style="{indent_style}"><ul class="simple-list">{list_items_html}</ul></div>'

                else: # List of complex items (dicts, lists)
                    list_content_html = ""
                    rendered_item_count = 0
                    for i, item in enumerate(data):
                        item_html = render_nested_json_html(item, level + 1, show_empty_fields=show_empty_fields)
                        if item_html:
                            rendered_item_count += 1
                            list_icon = "✦" # Default list item icon
                            list_icon_class = "list-item-icon"
                            # Keep icon logic based on content
                            if isinstance(item, dict):
                               item_keys = [k.lower() for k in item.keys()]
                               item_values = [str(v).lower() if v is not None else "" for v in item.values()]
                               item_text = " ".join(item_keys + item_values)
                               if "platform" in item_text: list_icon = "📱"
                               elif "algorithm" in item_text: list_icon = "🧠"
                               elif "skill" in item_text: list_icon = "🥄"
                               elif "interest" in item_text: list_icon = "🍰"
                               elif "consumption" in item_text: list_icon = "🧋"
                               elif "network" in item_text: list_icon = "👥"

                            list_content_html += f'<h6><span class="{list_icon_class}">{list_icon}</span> Item {i+1}</h6>'
                            list_content_html += item_html
                            # Separator using theme color (light tan dashed)
                            if i < len(data) - 1 and item_html.strip(): # Add separator only if content was rendered and not last
                                list_content_html += "<hr style='border: none; border-top: 1px dashed var(--cc-accent-light-tan); margin: 0.8rem 0;'>"
                    
                    # Remove trailing HR if it's the last thing
                    hr_str = "<hr style='border: none; border-top: 1px dashed var(--cc-accent-light-tan); margin: 0.8rem 0;'>"
                    if list_content_html.endswith(hr_str):
                         list_content_html = list_content_html[:-len(hr_str)]

                    if list_content_html:
                        html_output += f'<div style="{indent_style}">{list_content_html}</div>'
                    elif show_empty_fields:
                         html_output += f'<div style="{indent_style}">{render_missing_html()} (List items hidden or empty)</div>'


    elif isinstance(data, str):
        if is_value_empty(data):
            if show_empty_fields:
                # Use render_missing_html which uses em.missing styled by CSS
                html_output += f'<div style="{indent_style}">{render_missing_html()}</div>'
        else:
            escaped_data = html.escape(data)
            if escaped_data.startswith("http://") or escaped_data.startswith("https://"):
                # Link styling is default browser/streamlit unless specifically styled
                html_output += f'<div style="{indent_style}"><a href="{escaped_data}" target="_blank">🔗 {escaped_data}</a></div>'
            elif '\n' in escaped_data:
                # Use p.prewrap for multi-line text, styled like experiences box
                # Convert newlines to <br> for prewrap to maintain them visually within the p tag
                escaped_data_br = escaped_data.replace("\n", "<br>") 
                html_output += f'<div style="{indent_style}"><p class="prewrap">{escaped_data_br}</p></div>'
            else:
                # Basic paragraph text, styled by CSS 'p' or inherited from body
                html_output += f'<div style="{indent_style}"><p>{escaped_data}</p></div>'

    elif isinstance(data, (int, float, bool)):
        html_output += f'<div style="{indent_style}"><p>{html.escape(str(data))}</p></div>'

    elif data is None:
        if show_empty_fields:
            html_output += f'<div style="{indent_style}">{render_missing_html()}</div>'

    else:
         html_output += f'<div style="{indent_style}"><p><em>Unsupported data type: {html.escape(type(data).__name__)}</em></p></div>'


    return html_output

# -------------------------------------
# 3. Editing Dialog Function (NEW)
# -------------------------------------
@st.dialog("Edit Details", width="large") # Generic title, more specific one inside
def open_edit_dialog(section_data_to_edit, path_to_update_in_state, section_display_name_for_dialog):
    st.subheader(f"✏️ Editing: {section_display_name_for_dialog}")

    # Using a unique form key based on the path to avoid conflicts if dialogs were ever nested or complex
    form_key = f"dialog_form_{'_'.join(path_to_update_in_state).replace('[', '_').replace(']', '_')}"

    with st.form(key=form_key):
        # This dictionary will hold the new values from the form widgets
        # It's populated with copies of the original values to pre-fill the widgets
        # The widgets will then update this dictionary directly or we collect them.
        # For simplicity, we collect them into a new dictionary `edited_values_capture`
        edited_values_capture = {}

        for field_key, field_value in section_data_to_edit.items():
            field_label = format_key_to_title(field_key)
            # Ensure unique widget keys within this specific dialog instance
            unique_widget_key = f"dialog_edit_{'_'.join(path_to_update_in_state).replace('[', '_').replace(']', '_')}_{field_key}"

            if isinstance(field_value, list):
                is_simple_str_list = all(isinstance(item, str) for item in field_value)
                if is_simple_str_list:
                    default_list_str = ", ".join(field_value)
                    new_value_str = st.text_area(
                        f"{field_label} (comma-separated list)",
                        value=default_list_str,
                        key=unique_widget_key,
                        help="Enter items separated by commas. Empty items will be removed."
                    )
                    edited_values_capture[field_key] = [item.strip() for item in new_value_str.split(',') if item.strip()]
                else:
                    st.markdown(f"**{field_label} (Complex List):**")
                    try: st.json(field_value, expanded=False)
                    except Exception: st.text(str(field_value))
                    st.caption("(Editing complex lists/list of objects directly not supported in this dialog)")
                    edited_values_capture[field_key] = field_value # Keep original for non-editable
            elif isinstance(field_value, str):
                if len(field_value) > 200 or '\n' in field_value: # Increased threshold for text_area
                    edited_values_capture[field_key] = st.text_area(
                        field_label, value=field_value, key=unique_widget_key, height=150
                    )
                else:
                    edited_values_capture[field_key] = st.text_input(
                        field_label, value=field_value, key=unique_widget_key
                    )
            elif isinstance(field_value, (int, float)):
                edited_values_capture[field_key] = st.number_input(
                    field_label, value=field_value, key=unique_widget_key, format="%g" if isinstance(field_value, float) else "%d"
                )
            elif isinstance(field_value, bool):
                edited_values_capture[field_key] = st.checkbox(
                    field_label, value=field_value, key=unique_widget_key
                )
            elif field_value is None:
                new_value_text = st.text_input(
                    f"{field_label} (Currently None)", value="", key=unique_widget_key,
                    placeholder="Enter value or leave empty to keep as None"
                )
                # Only update if text is entered, otherwise keep as None
                edited_values_capture[field_key] = new_value_text if new_value_text.strip() else None
            else: # Nested dicts, other complex types
                st.markdown(f"**{field_label} ({type(field_value).__name__}):**")
                try: st.json(field_value, expanded=False)
                except Exception: st.text(str(field_value))
                st.caption("(Editing nested dictionaries or other complex types directly not supported in this dialog)")
                edited_values_capture[field_key] = field_value # Keep original value

        # Form submission button
        submitted = st.form_submit_button("💾 Save Changes")

    if submitted:
        try:
            # Update the actual session state data using the path
            current_data_ref = st.session_state['analysis_data']
            for i, key_segment in enumerate(path_to_update_in_state):
                if i == len(path_to_update_in_state) - 1:
                    # Last key: perform the update of the entire section dict
                    if key_segment in current_data_ref:
                        current_data_ref[key_segment] = edited_values_capture # Replace the whole dict
                        st.session_state['edit_success_message'] = f"Changes saved for '{section_display_name_for_dialog}'."
                        st.rerun() # This will close the dialog and refresh the app
                        return # Important to exit after rerun
                    else:
                        st.error(f"Error: Key '{key_segment}' not found at expected path during save.")
                        return # Abort save
                else:
                    # Navigate deeper
                    if key_segment in current_data_ref and isinstance(current_data_ref[key_segment], dict):
                        current_data_ref = current_data_ref[key_segment]
                    else:
                        st.error(f"Error: Invalid path or non-dictionary found at key '{key_segment}' during save.")
                        return # Abort save
        except Exception as e:
            st.error(f"Error saving changes: {e}")
            # Do not rerun here, let the error be visible in the dialog

    # Cancel button - placed outside the form, but still within the dialog function
    if st.button("❌ Cancel", key="dialog_cancel_button"):
        st.rerun() # Closes the dialog by rerunning the app


# -------------------------------------
# 4. Modified Editing Form Renderer (now triggers dialog)
# -------------------------------------
def render_edit_interaction_point():
    """Renders controls in the sidebar to select a section and open an edit dialog."""
    if 'analysis_data' not in st.session_state or not st.session_state['analysis_data']:
        st.info("Load data first to enable editing.")
        return

    st.subheader("✏️ Edit Data Sections")

    editable_data_snapshot = st.session_state['analysis_data'] # Use snapshot for determining sections

    editable_sections = {}
    if isinstance(editable_data_snapshot.get('yourCoffeeCard'), dict):
        editable_sections['yourCoffeeCard'] = "Your Coffee Card"

    for key, value in editable_data_snapshot.items():
        if isinstance(value, dict) and key != 'yourCoffeeCard':
            if key == 'platformSpecificAnalysis' and isinstance(value, dict):
                for p_key, p_value in value.items():
                    if isinstance(p_value, dict):
                        edit_key_for_state = f"platform_{p_key}" # This key is just for the selectbox state
                        editable_sections[edit_key_for_state] = f"Platform: {get_platform_display_name(p_key, p_value)}"
            elif key != 'platformSpecificAnalysis':
                editable_sections[key] = format_key_to_title(key)

    if not editable_sections:
        st.warning("No editable dictionary sections found in the loaded data.")
        return

    section_keys_for_selectbox = list(editable_sections.keys())
    selected_key_for_editing = st.selectbox(
        "Select section to edit:",
        options=section_keys_for_selectbox,
        format_func=lambda key: editable_sections[key],
        key="edit_section_selector_for_dialog" # Unique key for this selectbox
    )

    if st.button(f"⚙️ Edit '{editable_sections[selected_key_for_editing]}'...", key=f"open_dialog_btn_{selected_key_for_editing}"):
        path_to_update = []
        section_data_for_dialog = None

        if selected_key_for_editing == 'yourCoffeeCard':
            path_to_update = ['yourCoffeeCard']
        elif selected_key_for_editing.startswith("platform_"):
            platform_actual_key = selected_key_for_editing.split("platform_", 1)[1]
            path_to_update = ['platformSpecificAnalysis', platform_actual_key]
        elif selected_key_for_editing in editable_data_snapshot: # Top-level general sections
            path_to_update = [selected_key_for_editing]
        
        # Safely get a deep copy of the data for the dialog
        # This ensures the dialog works on a copy and session state is only updated on save
        data_at_path = safe_get(st.session_state['analysis_data'], path_to_update)
        if data_at_path is not None and isinstance(data_at_path, dict):
            section_data_for_dialog = copy.deepcopy(data_at_path)
        
        if section_data_for_dialog is not None:
            open_edit_dialog(
                section_data_to_edit=section_data_for_dialog,
                path_to_update_in_state=path_to_update,
                section_display_name_for_dialog=editable_sections[selected_key_for_editing]
            )
        else:
            st.error(f"Could not retrieve or prepare data for section: {editable_sections[selected_key_for_editing]}")


# Custom header - Adjust colors slightly
def render_custom_header():
    header_html = """
    <div style="padding: 1rem 0; margin-bottom: 2rem; text-align: center; position: relative;">
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 6px; background: var(--cc-accent-dark-brown);"></div>
        <h1 style="font-size: 2.4rem; color: var(--cc-accent-dark-brown); margin-bottom: 0.3rem; letter-spacing: -0.02em;">
            ☕ CaféCorner
        </h1>
        <p style="color: var(--cc-text-secondary); font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            A cozy place to view and manage your professional presence across platforms
        </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

# --- get_platform_display_name (no changes needed) ---
def get_platform_display_name(platform_key, platform_data):
    """Gets the display name for a platform tab."""
    if isinstance(platform_data, dict):
        name_from_field = platform_data.get("platformName")
        if name_from_field and isinstance(name_from_field, str):
            return name_from_field
    return format_key_to_title(platform_key)

# -------------------------------------
# Example Data Loading (Unchanged)
# -------------------------------------
EXAMPLE_ANALYSIS_DATA = {
  "targetIndividual": {
    "nameOrIdentifier": "Homen Shum",
    "primaryProfileURL": "https://www.linkedin.com/in/homen-shum"
  },
  "analyzedPlatforms": [
    "LinkedIn",
    "Instagram"
  ],
  "yourCoffeeCard": {
    "name": "Homen Shum",
    "title": "Ex-PM Startup, Banking Associate, Data/ML/GenAI Engineering for FinAdvizly | Ex-Healthtech | UC Berkeley | Co-founder @ CafeCorner LLC",
    "profilePictureUrlForCard": "https://example.com/homen_shum_linkedin_profile.jpg",
    "taglineOrBriefSummary": "Versatile professional with expertise in Generative AI, Large Language Models, and data analytics, driving innovation in finance and healthcare through leadership in startup ventures and impactful projects at JPMorgan Chase.",
    "primaryProfileUrlForCard": "https://www.linkedin.com/in/homen-shum",
    "callToActionForCard": "Connect on LinkedIn",
    "location": "Fremont, California, United States",
    "interests": [
      "Generative AI",
      "Large Language Models",
      "FinTech",
      "HealthTech",
      "Data Science",
      "Machine Learning",
      "Cloud Computing",
      "Startups",
      "Investment",
      "Technology Trends"
    ],
    "hobbies": [
      "Exploring new AI technologies",
      "Hackathons",
      "Content Creation (technical blogs/videos - inferred)"
    ],
    "skills": [
      "Large Language Models (LLMs)",
      "Python (Programming Language)",
      "Generative AI",
      "Data Analysis",
      "Machine Learning",
      "Pandas (Software)",
      "Microsoft Azure",
      "Google Cloud Platform (GCP)",
      "Amazon Web Services (AWS)",
      "Automation",
      "Financial Analysis",
      "Startup Development"
    ],
    "experiences": [
      {
        "role": "Founder",
        "company": "FinAdvizly LLC",
        "dates": "Dec 2021 - Present (2 yrs 5 mos)",
        "isCurrentOrPrimary": True,
        "briefSummaryForMiniCard": "Leading FinAdvizly in developing GenAI-powered sales recommendation, workflow automation, and financial research tools.",
        "description": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS, and Render using Docker. Led a team of 5 people to win Top-25 placement at UC Berkeley AI Hackathon (Jun 2023) by implementing medical code matching with real-time transcription and RAG (Retrieval Augmented Generation) implementation for healthcare, subsequently becoming Technical Co-Founder of FinAdvizly's new GenAI product for healthcare. Currently (Nov 2023-Present), developing FinAdvizly's capabilities by integrating multi-agent architecture for comprehensive financial research, featuring cross-validation with web data and structured output processing for report generation.",
        "skillDetails": [
          {
            "skillName": "Large Language Models (LLMs)",
            "contextualSnippet": "subsequently becoming Technical Co-Founder of FinAdvizly's new GenAI product for healthcare.",
            "relatedSkillsInThisExperience": ["Generative AI", "RAG", "GCP", "Azure", "AWS", "Docker", "Python", "Multi-agent architecture"]
          },
          {
            "skillName": "Generative AI",
            "contextualSnippet": "subsequently becoming Technical Co-Founder of FinAdvizly's new GenAI product for healthcare.",
            "relatedSkillsInThisExperience": ["LLMs", "RAG", "Python"]
          },
          {
            "skillName": "GCP",
            "contextualSnippet": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS...",
            "relatedSkillsInThisExperience": ["Azure", "AWS", "Docker", "Workflow Automation"]
          },
          {
            "skillName": "Azure",
            "contextualSnippet": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS...",
            "relatedSkillsInThisExperience": ["GCP", "AWS", "Docker", "Workflow Automation"]
          },
          {
            "skillName": "AWS",
            "contextualSnippet": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS...",
            "relatedSkillsInThisExperience": ["GCP", "Azure", "Docker", "Workflow Automation"]
          },
          {
            "skillName": "Docker",
            "contextualSnippet": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS, and Render using Docker.",
            "relatedSkillsInThisExperience": ["GCP", "Azure", "AWS"]
          },
          {
            "skillName": "RAG (Retrieval Augmented Generation)",
            "contextualSnippet": "implementing medical code matching with real-time transcription and RAG (Retrieval Augmented Generation) implementation for healthcare.",
            "relatedSkillsInThisExperience": ["LLMs", "Generative AI", "Medical Code Matching", "Real-time Transcription"]
          },
          {
            "skillName": "Multi-agent architecture",
            "contextualSnippet": "integrating multi-agent architecture for comprehensive financial research...",
            "relatedSkillsInThisExperience": ["LLMs", "Financial Research"]
          },
          {
            "skillName": "Large-scale Data Analytics",
            "contextualSnippet": "featuring cross-validation with web data and structured output processing for report generation.",
            "relatedSkillsInThisExperience": ["Financial Research"]
          }
        ]
      },
      {
        "role": "Rotation 3: Healthcare and Life Science Banking Team",
        "company": "JPMorgan Chase & Co.",
        "dates": "May 2023 - Feb 2024 (10 mos)",
        "isCurrentOrPrimary": False,
        "briefSummaryForMiniCard": None,
        "description": "Initiated collaborations between internal teams such as CC / SF Risk and Fin/ML Team to streamline client onboarding / management / servicing. Key projects included: (Oct 2023) Automated classification system for JPM Healthcare Banking team using GPT and Embeddings, implementing structured outputs to populate an instruction matrix and DPT 3.5 turbo, reducing processing time for 2,000+ companies from two weeks to under 30 seconds. (Feb 2024) Designed real-time AI transcription application for JPM internal meetings and assumed LLM Application Pilot role within AI ML Technology team.",
        "skillDetails": [
          {
            "skillName": "GPT",
            "contextualSnippet": "Automated classification system for JPM Healthcare Banking team using GPT and Embeddings...",
            "relatedSkillsInThisExperience": ["Embeddings", "DPT 3.5 turbo", "LLMs", "AI Transcription", "Automation"]
          },
          {
            "skillName": "Embeddings",
            "contextualSnippet": "Automated classification system for JPM Healthcare Banking team using GPT and Embeddings...",
            "relatedSkillsInThisExperience": ["GPT", "DPT 3.5 turbo", "LLMs"]
          },
          {
            "skillName": "Large Language Models (LLMs)",
            "contextualSnippet": "assumed LLM Application Pilot role within AI ML Technology team.",
            "relatedSkillsInThisExperience": ["GPT", "AI Transcription", "Automation"]
          },
          {
            "skillName": "AI Transcription",
            "contextualSnippet": "Designed real-time AI transcription application for JPM internal meetings...",
            "relatedSkillsInThisExperience": ["LLMs", "Automation"]
          },
          {
            "skillName": "Automation",
            "contextualSnippet": "Automated classification system for JPM Healthcare Banking team...",
            "relatedSkillsInThisExperience": ["GPT", "LLMs"]
          },
          {
            "skillName": "Cloud Computing",
            "contextualSnippet": "Initiated collaborations between internal teams such as CC / SF Risk and Fin/ML Team to streamline client onboarding / management / servicing.",
            "relatedSkillsInThisExperience": ["Amazon Web Services (AWS)"]
          },
          {
            "skillName": "Amazon Web Services (AWS)",
            "contextualSnippet": "Initiated collaborations between internal teams such as CC / SF Risk and Fin/ML Team to streamline client onboarding / management / servicing.",
            "relatedSkillsInThisExperience": ["Cloud Computing"]
          }
        ]
      },
      {
        "role": "Startup Banking Associate",
        "company": "JPMorgan Chase & Co.",
        "dates": "Jun 2022 - May 2023 (1 yr)",
        "isCurrentOrPrimary": False,
        "briefSummaryForMiniCard": None,
        "description": "As a JPM Startup Banking Associate in the Bay Area, I treasure the opportunity to work with some of the best SaaS / Enterprise Software / CloudTech / FinTech / GenAI / HealthTech / DeepTech / Life Science / ClimateTech / FrontierTech / HardTech / Web3 / Digital Asset / Robotics / AI companies. Capital is the life and blood of the most ambitious people. To speak with daring founders and support their growth journey has been an incredible experience. My role involved data analysis and automation, leading to significant improvements in data quality checks and document processing efficiency.",
        "skillDetails": [
          {
            "skillName": "Data Analysis",
            "contextualSnippet": "My role involved data analysis and automation, leading to significant improvements in data quality checks...",
            "relatedSkillsInThisExperience": ["Automation", "Financial Analysis"]
          },
          {
            "skillName": "Automation",
            "contextualSnippet": "My role involved data analysis and automation, leading to significant improvements in data quality checks...",
            "relatedSkillsInThisExperience": ["Data Analysis"]
          },
          {
            "skillName": "Financial Analysis",
            "contextualSnippet": "work with some of the best SaaS / Enterprise Software / CloudTech / FinTech / GenAI / HealthTech ... companies.",
            "relatedSkillsInThisExperience": ["Data Analysis"]
          }
        ]
      },
      {
        "role": "JPM AWS DeepRacer Pilot: Aido Team Lead, Lead AI Engineer",
        "company": "JPMorgan Chase & Co.",
        "dates": "Oct 2022 - Oct 2022 (1 mo)",
        "isCurrentOrPrimary": False,
        "briefSummaryForMiniCard": None,
        "description": "As JPM PMAC Pilot Aido, I coordinated model functions and tuned training hyperparameters to enhance AI model performance. This seasonal role focused on applying AI engineering skills within a competitive AWS DeepRacer environment.",
        "skillDetails": [
          {
            "skillName": "Amazon Web Services (AWS)",
            "contextualSnippet": "This seasonal role focused on applying AI engineering skills within a competitive AWS DeepRacer environment.",
            "relatedSkillsInThisExperience": ["Cloud Computing", "AI Engineering", "Hyperparameter Tuning"]
          },
          {
            "skillName": "Cloud Computing",
            "contextualSnippet": "This seasonal role focused on applying AI engineering skills within a competitive AWS DeepRacer environment.",
            "relatedSkillsInThisExperience": ["Amazon Web Services (AWS)"]
          },
          {
            "skillName": "AI Engineering",
            "contextualSnippet": "coordinated model functions and tuned training hyperparameters to enhance AI model performance.",
            "relatedSkillsInThisExperience": ["Hyperparameter Tuning", "AWS"]
          },
          {
            "skillName": "Hyperparameter Tuning",
            "contextualSnippet": "tuned training hyperparameters to enhance AI model performance.",
            "relatedSkillsInThisExperience": ["AI Engineering", "AWS"]
          }
        ]
      }
    ],
    "education": [
      {
        "institution": "UC Santa Barbara",
        "degree": "Certificate",
        "fieldOfStudy": "Business Administration and Management, General",
        "dates": "2020 - 2021",
        "description": "Cooperated with professors on personal project, recommended by 2 professors and a graduate student."
      },
      {
        "institution": "Financial Edge Training",
        "degree": "Certificate",
        "fieldOfStudy": "Commercial Banking, Budget Training, Accounting and Finance",
        "dates": "2021",
        "description": None
      }
    ],
    "projects": [
      {
        "projectName": "Startup Search API",
        "datesOrDuration": "Oct 2023 - Nov 2023",
        "description": "Developed a Startup Search API. Demo: https://hbrt-qphm-server-4psnwpfyqf-wl.a.run.app/search?q=[your search query for the top 5 start up names matching to your desired company search descriptions]. This project involved creating a web service API, likely for querying startup information based on descriptions.",
        "skillsUsed": [
          "Web Services API",
          "Docker"
        ],
        "projectUrl": "https://hbrt-qphm-server-4psnwpfyqf-wl.a.run.app/"
      },
      {
        "projectName": "Celebrity Voice AI Platform",
        "datesOrDuration": "Oct 2023",
        "description": "Developed Celebrity Voice AI Platform using GPT-3, Flask, and AutoML, implementing cosine similarity matching for celebrity speech pattern replication.",
        "skillsUsed": [
          "GPT-3",
          "Flask",
          "AutoML",
          "Cosine Similarity",
          "Voice AI"
        ],
        "projectUrl": None
      },
      {
        "projectName": "Nation's Cybersecurity Challenge",
        "datesOrDuration": "Mar 2023",
        "description": "Achieved 2nd place in Nation's Cybersecurity challenge with 95% accuracy in semantic matching and unified representation of multi-vendor security alerts.",
        "skillsUsed": [
          "Semantic Matching",
          "Cybersecurity",
          "Data Representation"
        ],
        "projectUrl": None
      },
      {
        "projectName": "Medical Code Search & Report Generation",
        "datesOrDuration": "Ongoing (part of 'My Latest Web Applications')",
        "description": "A web application for medical code searching and report generation.",
        "skillsUsed": [
          "Web Development",
          "AI",
          "Healthcare Informatics"
        ],
        "projectUrl": "https://www.youtube.com/watch?v=kp1hKPBJQC0"
      },
      {
        "projectName": "Patent Screening Tool & Report Generation",
        "datesOrDuration": "Ongoing (part of 'My Latest Web Applications')",
        "description": "A Streamlit application for patent screening and generating reports.",
        "skillsUsed": [
          "Streamlit",
          "Python",
          "AI",
          "Patent Analysis"
        ],
        "projectUrl": "https://homen-patent-screening.streamlit.app/"
      },
      {
        "projectName": "Auto-Direct E-Commerce Ad Copy",
        "datesOrDuration": "Ongoing (part of 'My Latest Web Applications')",
        "description": "A Streamlit application to assist with e-commerce sales by generating ad copy.",
        "skillsUsed": [
          "Streamlit",
          "Python",
          "Generative AI",
          "E-commerce"
        ],
        "projectUrl": "https://homen-ecom-sales-assistant.streamlit.app/"
      }
    ],
    "keyAchievementsOverall": [
      "Co-founded FinAdvizly LLC, developing Generative AI solutions for finance and healthcare.",
      "Achieved Top-25 placement in UC Berkeley AI Hackathon and 2nd place in Nation's Cybersecurity Challenge.",
      "Pioneered AI applications within JPMorgan Chase, including automated systems and LLM pilots."
    ]
  },
  "platformSpecificAnalysis": {
    "linkedIn": {
      "platformName": "LinkedIn",
      "profileFundamentals": {
        "username": "homen-shum",
        "fullName": "Homen Shum",
        "pronouns": None,
        "location": "Fremont, California, United States",
        "profileURL": "https://www.linkedin.com/in/homen-shum",
        "profileLanguage": "English",
        "verificationStatus": None,
        "contactInfoVisible": True,
        "profilePictureDescription": "Professional headshot, smiling, wearing a suit.",
        "bannerImageDescription": "Abstract cityscape or data network graphic.",
        "linkedWebsites": [
          "https://www.youtube.com/watch?v=kp1hKPBJQC0",
          "https://homen-patent-screening.streamlit.app/",
          "https://homen-ecom-sales-assistant.streamlit.app/"
        ]
      },
      "professionalHeadline": "To do something great with data and finance.\" (Ex-PM Startup, Banking Associate, Data/ML/GenAI Engineering for FinAdvizly | Ex-Healthtech | UC Berkeley | Co-founder @ CafeCorner LLC)",
      "aboutSectionAnalysis": {
        "fullText": "Experience with LLM APIs and Generative AI tools. I've been working with LLM APIs and generation AI tools since December 2022, progressing through increasingly complex implementations across multiple domains...\n2022-2023: Foundation Projects\nOct 2023: Developed Celebrity Voice AI Platform using GPT-3, Flask, and AutoML, implementing cosine_similarity matching for celebrity speech pattern replication.\nMar 2023: Achieved 2nd place in Nation's Cybersecurity challenge with 95% accuracy in semantic matching and unified representation of multi-vendor security alerts.\n2023 Healthcare AI Applications\nJun 2023: Led a team of 5 people to win Top-25 placement at UC Berkeley AI Hackathon leveraging Vicuna and Langchain, implementing medical code matching with real time transcription and RAG implementation. Subsequently became Technical Co-Founder of FinAdvizly's new GenAI product for healthcare.\n2023-2024: Financial Sector Implementation\nOct 2023: Automated classification system for JPM Healthcare Banking team using GPT and Embeddings, implementing structured outputs to populate an instruction matrix and DPT 3.5 turbo. Reduced processing time for 2,000+ companies from two weeks to under 30 seconds.\n2024-Present: Advanced Applications\nFeb 2024: Designed real-time AI transcription application for JPM internal meetings and assumed LLM Application Pilot role within AI ML Technology team.\nMar 2016: Completed CrowdFlower's MVP building real-time transcription, Of-CD matching, and partial plan deployed via Azure with HIPAA compliant security.\nNov 2023-Present: Developing FinAdvizly, integrating multi-agent architecture for comprehensive financial research, featuring cross-validation with web data and structured output processing for report generation.\nMy Latest Web Applications:\nMedical Code Search & Report Generation: https://www.youtube.com/watch?v=kp1hKPBJQC0\nPatent Screening Tool & Report Generation: https://homen-patent-screening.streamlit.app/\nAuto-Direct E-Commerce Ad Copy: https://homen-ecom-sales-assistant.streamlit.app/\nHere's to embracing growth, bridging gaps, and making technology more accessible - together.",
        "identifiedKeywords": [
          "LLM APIs",
          "Generative AI",
          "GPT-3",
          "Flask",
          "AutoML",
          "Cybersecurity",
          "Semantic Matching",
          "Healthcare AI",
          "Vicuna",
          "Langchain",
          "RAG",
          "FinAdvizly",
          "Financial Sector",
          "GPT",
          "Embeddings",
          "AI Transcription",
          "Multi-agent architecture",
          "Data Science",
          "Machine Learning"
        ],
        "hashtagsInBio": [],
        "mentionedUsersInBio": [],
        "statedPurposeFocus": "Highlighting extensive experience and successful projects in LLM APIs, Generative AI, Healthcare AI, and Financial AI applications, emphasizing a progression of complex implementations and a drive to make technology accessible.",
        "tone": "Professional, Accomplishment-oriented, Technical",
        "callToAction": "Implied: Explore linked web applications and connect for collaboration."
      },
      "featuredContent": [
        "Post: The Limits of Public Data Flow (LLM Struggles) - image (diagram)",
        "Link: Fluency Shed #1 - Filescore Hackathon Winner - hackathon.filescore.com/... - This is a textbot that use answ questions about medical adviso...",
        "Image: Semantic Search Ranking Method with Dimension Reduction - diagram, Provided 86.27% high performance accuracy by the end of the hackathon."
      ],
      "experience": [
        {
          "roleOrTitle": "Founder",
          "companyOrOrganization": "FinAdvizly LLC",
          "dates": "Dec 2021 - Present (2 yrs 5 mos)",
          "location": "Palo Alto, California, United States - Hybrid",
          "description": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS, and Render using Docker. (The 'About' section adds: Jun 2023: Led a team of 5 people to win Top-25 placement at UC Berkeley AI Hackathon leveraging Vicuna and Langchain, implementing medical code matching with real time transcription and RAG implementation. Subsequently became Technical Co-Founder of FinAdvizly's new GenAI product for healthcare. Nov 2023-Present: Developing FinAdvizly, integrating multi-agent architecture for comprehensive financial research, featuring cross-validation with web data and structured output processing for report generation.)",
          "skillsListedExplicitly": [
            "Large Language Models (LLMs)",
            "Large-scale Data Analytics",
            "Machine Learning (inferred as +1 skill)"
          ],
          "mediaOrLinks": [
            "Homen Shum - Resume - April 2023.pdf"
          ]
        },
        {
          "roleOrTitle": "Startup Banking Associate",
          "companyOrOrganization": "JPMorgan Chase & Co.",
          "dates": "Jun 2022 - May 2023 (1 yr)",
          "location": "San Francisco Bay Area - Hybrid",
          "description": "As a JPM Startup Banking Associate in the Bay Area, I treasure the opportunity to work with some of the best SaaS / Enterprise Software / CloudTech / FinTech / GenAI / HealthTech / DeepTech / Life Science / ClimateTech / FrontierTech / HardTech / Web3 / Digital Asset / Robotics / AI companies. Capital is the life and blood of the most ambitious people. To speak with daring founders and support their growth journey has been an incredible experience. (Media text: 'Little bit faster. Little less cost. A lot more useful. A simple and efficient macro led to a 95%+ reduction in manual data entry for a data quality check...This data comparison highlighted significant improvements in new document processing process over time. We extracted various metrics...')",
          "skillsListedExplicitly": [
            "Data Analysis",
            "Automation"
          ],
          "mediaOrLinks": [
            "Image with text about data processing improvements"
          ]
        },
        {
          "roleOrTitle": "Rotation 3: Healthcare and Life Science Banking Team",
          "companyOrOrganization": "JPMorgan Chase & Co.",
          "dates": "May 2023 - Feb 2024 (10 mos)",
          "location": "San Francisco Bay Area - Hybrid",
          "description": "Initiated collaborations between internal teams such as CC / SF Risk and Fin/ML Team to streamline client onboarding / management / servicing. (The 'About' section adds: Oct 2023: Automated classification system for JPM Healthcare Banking team using GPT and Embeddings, implementing structured outputs to populate an instruction matrix and DPT 3.5 turbo. Reduced processing time for 2,000+ companies from two weeks to under 30 seconds. Feb 2024: Designed real-time AI transcription application for JPM internal meetings and assumed LLM Application Pilot role within AI ML Technology team.)",
          "skillsListedExplicitly": [
            "Amazon Web Services (AWS)",
            "Cloud Computing"
          ],
          "mediaOrLinks": []
        },
        {
          "roleOrTitle": "JPM AWS DeepRacer Pilot: Aido Team Lead, Lead AI Engineer",
          "companyOrOrganization": "JPMorgan Chase & Co.",
          "dates": "Oct 2022 - Oct 2022 (1 mo)",
          "location": "Palo Alto, California, United States - Hybrid",
          "description": "As JPM PMAC Pilot Aido, I coordinated model functions and tuned training hyperparameters to enhance AI model performance.",
          "skillsListedExplicitly": [
            "Amazon Web Services (AWS)",
            "Cloud Computing"
          ],
          "mediaOrLinks": []
        },
        {
          "roleOrTitle": "Risk Management Challenge",
          "companyOrOrganization": "EquitDian",
          "dates": "Nov 2020 - Dec 2020 (2 mos)",
          "description": "Build risk management skills while managing the daily fluctuations to less than 1% +/- of a $100k portfolio... December 14, 2020, until February 28th, 2021",
          "skillsListedExplicitly": [],
          "mediaOrLinks": []
        },
        {
          "roleOrTitle": "High Frequency Trading Challenge",
          "companyOrOrganization": "EquitDian",
          "dates": "Nov 2020 - Dec 2020 (2 mos)",
          "description": "Generated 13% total profit within 20 market trading days by November 17th, 2020. Demonstrated proper utilization of risk management strategy, understanding of moving averages, an...",
          "skillsListedExplicitly": [],
          "mediaOrLinks": []
        },
        {
          "roleOrTitle": "Asia Pacific Investment Challenge",
          "companyOrOrganization": "EquitDian",
          "dates": "Sep 2020 - Nov 2020 (3 mos)",
          "description": "Researched and evaluated companies' market capitalization with analyst reviews and revenue reports from financial data providers and utilized product reviews to determine consumer enthusiasm and company's growt...",
          "skillsListedExplicitly": [],
          "mediaOrLinks": []
        },
        {
          "roleOrTitle": "Internal Ambassador",
          "companyOrOrganization": "Delta Sigma Pi - Rho Sigma Chapter",
          "dates": "May 2020 - May 2021 (1 yr 1 mo)",
          "description": "Collaborated with on-campus organizations to plan professional networking events and discussion mals...",
          "skillsListedExplicitly": [],
          "mediaOrLinks": []
        },
        {
          "roleOrTitle": "Recruitment Assistant",
          "companyOrOrganization": "Delta Sigma Pi - Rho Sigma Chapter",
          "dates": "Mar 2019 - Mar 2021 (2 yrs 1 mo)",
          "description": "Planned, scripted, and facilitated a formal interviewing training event with a committee. Booked rooms and requested funding by contacting the student body...",
          "skillsListedExplicitly": [],
          "mediaOrLinks": []
        },
        {
          "roleOrTitle": "Professional Event Assistant",
          "companyOrOrganization": "Delta Sigma Pi - Rho Sigma Chapter",
          "dates": "Jan 2019 - Apr 2019 (4 mos)",
          "description": "Invited and met with a $100,000 start-up organization to speak at the professional event held for the professionals at Delta Sigma Pi meeting.",
          "skillsListedExplicitly": [],
          "mediaOrLinks": []
        },
        {
          "roleOrTitle": "Academic Advisor",
          "companyOrOrganization": "Vietnamese Student Association - UC Santa Barbara",
          "dates": "Apr 2019 - Jun 2019 (3 mos)",
          "description": "Supervised members with up to $2000 from AS ESP Grant, Emergency Grant, CIT Grant, Young Entrepreneur Scholarship...",
          "skillsListedExplicitly": [],
          "mediaOrLinks": []
        }
      ],
      "education": [
        {
          "institution": "Financial Edge Training",
          "degree": "Certificate",
          "fieldOfStudy": "2021 Commercial Banking, Budget Training, Accounting and Finance",
          "dates": "Attn: 2021",
          "description": None,
          "mediaOrLinks": []
        },
        {
          "institution": "UC Santa Barbara",
          "degree": "Certificate",
          "fieldOfStudy": "Business Administration and Management, General",
          "dates": "2020 - 2021",
          "description": "Cooperated with professors on personal project, recommended by 2 professors and a graduate stude...",
          "mediaOrLinks": []
        }
      ],
      "projects": [
        {
          "projectName": "Startup Search API",
          "roleOrContribution": None,
          "datesOrDuration": "Oct 2023 - Nov 2023",
          "description": "Demo: https://hbrt-qphm-server-4psnwpfyqf-wl.a.run.app/search?q=[your search query for the top 5 start up names matching to your desired company search descriptions]...",
          "skillsOrTechnologiesUsed": [
            "Web Services API",
            "Docker"
          ],
          "projectUrl": "https://hbrt-qphm-server-4psnwpfyqf-wl.a.run.app/",
          "mediaOrLinks": []
        }
      ],
      "skillsEndorsements": [
        "Python (Programming Language)",
        "Pandas (Software)",
        "Microsoft Azure",
        "Google Cloud Platform (GCP)",
        "Large Language Models (LLM)",
        "Large Language Model Operations (LLMOps) (Founder at CafeCorner LLC)",
        "Large-Scale Data Analysis (Founder at CafeCorner LLC)"
      ],
      "recommendationsGivenReceived": None,
      "accomplishments": "Multiple projects detailed in 'About' section including Celebrity Voice AI, Cybersecurity Challenge win, Healthcare AI hackathon win.",
      "interests": [
        "Sarah Johnston (Job Search & Career Coach)",
        "Ray Dalio (Founder, Bridgewater)"
      ],
      "contentGenerationActivity": {
        "postingFrequency": "Sporadic (based on one recent repost)",
        "dominantContentTypes": [
          "Reposts of technical articles/announcements"
        ],
        "contentExamples": [
          "Reposted 'David Myriel - Cohere 1.4 just dropped https://txt.cohere.com/...' - Image 'Score-Boosting Reranker'"
        ],
        "recurringThemesTopics": [
          "AI",
          "Large Language Models",
          "Machine Learning"
        ],
        "overallToneVoice": "Informative, Technical",
        "recentActivityExamples": [
          "Reposted Cohere 1.4 announcement."
        ],
        "articlesPublishedCount": None
      },
      "engagementPatterns": {
        "outgoingInteractionStyle": "Shares industry news and technical updates.",
        "typesOfContentEngagedWith": [
          "AI/ML developments",
          "Posts from tech leaders"
        ],
        "incomingEngagementHighlights": "Post impressions and profile views indicate active profile discovery.",
        "typicalIncomingCommentTypes": []
      },
      "networkCommunityRecommendations": {
        "followerCount": 2656,
        "followingCount": None,
        "audienceDescription": "Professionals in AI, finance, tech, and startups. Peers from academic institutions and former/current workplaces.",
        "groupCommunityMemberships": [],
        "inboxSidebarPreview": [
          {"nameOrGroupName": "Luan Piju", "visibleSnippet": None, "timestamp": "Fri", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Cynthia McCarthy", "visibleSnippet": None, "timestamp": "Apr 19", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Computer Talent Solutions", "visibleSnippet": None, "timestamp": "Apr 18", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Yuan Jialiang, PhD", "visibleSnippet": None, "timestamp": "Apr 17", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "James (Shinn) Mildew", "visibleSnippet": None, "timestamp": "Apr 17", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Boris Nekdara", "visibleSnippet": None, "timestamp": "Apr 17", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Olivia Grigolaya Verbitsky", "visibleSnippet": None, "timestamp": "Apr 1", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Andrei Zayarin", "visibleSnippet": None, "timestamp": "Apr 1", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Yunus Altekin, PhD", "visibleSnippet": None, "timestamp": "Apr 1", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Vincent Yamada", "visibleSnippet": None, "timestamp": "Mar 31", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Tsetsulan Kao", "visibleSnippet": None, "timestamp": "Mar 28", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Kevin Grado", "visibleSnippet": None, "timestamp": "Mar 28", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Roukarious Roser-Fing", "visibleSnippet": None, "timestamp": "Mar 27", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Rahul Agarwal", "visibleSnippet": None, "timestamp": "Mar 25", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Chhavi Kadian", "visibleSnippet": None, "timestamp": "Mar 25", "unreadCount": None, "statusIndicator": None},
          {"nameOrGroupName": "Osborn Ficker", "visibleSnippet": None, "timestamp": "Mar 24", "unreadCount": None, "statusIndicator": None}
        ],
        "inboxSidebarAnalysis": "The messaging preview shows recent interactions with a diverse set of individuals, many with academic or professional titles (PhD). This suggests ongoing professional networking and communication. Timestamps indicate fairly regular activity.",
        "myNetworkTabVisibility": [
          {
            "entityName": "Rania Hussain",
            "entityType": "Person",
            "headlineOrDescription": "Barnard College | Incoming Analyst at Goldman Sachs",
            "contextProvided": "From your school",
            "profileURL": None,
            "visibleMetrics": None
          },
          {
            "entityName": "Leah Varughese",
            "entityType": "Person",
            "headlineOrDescription": "Digital Product Associate at JPMorgan Chase & Co.",
            "contextProvided": None,
            "profileURL": None,
            "visibleMetrics": None
          },
          {
            "entityName": "Godson Joseph",
            "entityType": "Person",
            "headlineOrDescription": "Economics Student at UCSB | Membership Committe...",
            "contextProvided": None,
            "profileURL": None,
            "visibleMetrics": None
          },
          {
            "entityName": "Albert Qian",
            "entityType": "Person",
            "headlineOrDescription": "MBA Candidate @ Columbia Business School | Data...",
            "contextProvided": None,
            "profileURL": None,
            "visibleMetrics": None
          },
          {
            "entityName": "T. Aqeel Taheraly",
            "entityType": "Person",
            "headlineOrDescription": "Incoming analyst at Nomura | Senior at Trinity College | UM...",
            "contextProvided": None,
            "profileURL": None,
            "visibleMetrics": None
          }
        ],
        "myNetworkTabAnalysis": "The 'People You May Know' section prominently features individuals from his alma mater (UCSB, Barnard implied via 'From your school') and current/former workplaces (JPMorgan Chase & Co.), as well as other academic institutions. This suggests LinkedIn is effectively leveraging his educational and professional history to expand his network within relevant circles, particularly in finance and tech.",
        "platformSuggestions": {
          "suggestedPeople": [],
          "suggestedCompaniesOrPages": [],
          "suggestedGroupsEvents": [],
          "suggestedContentOrTopics": [],
          "peopleAlsoViewed": [
            {
              "fullName": "Sonya Chiang",
              "headlineOrTitle": "Product Designer (Acquired Ex)",
              "profileURL": None,
              "reasonForSuggestion": None,
              "locationContext": "Profile Sidebar",
              "visibleMetrics": None
            },
            {
              "fullName": "Annychika Akpabio",
              "headlineOrTitle": None,
              "profileURL": None,
              "reasonForSuggestion": None,
              "locationContext": "Profile Sidebar",
              "visibleMetrics": None
            }
          ],
          "otherSuggestions": []
        },
        "platformRecommendationsAnalysis": "LinkedIn's recommendations ('People Also Viewed' and 'People You May Know') strongly reinforce Homen's professional identity in tech and finance. 'People Also Viewed' suggests peers with similar career trajectories or in related fields (e.g., Product Design). 'People You May Know' clearly leverages his academic background (UCSB, potentially Barnard) and work history (JPMorgan Chase) to suggest relevant connections, predominantly in finance and tech roles. This indicates the algorithm accurately perceives his professional network and interests, aiming to expand it within these domains.",
        "detailedConnectionsList": None,
        "detailedConnectionsAnalysis": None
      },
      "privacyPresentation": {
        "accountVisibility": "Public",
        "postLevelVisibility": "Public (inferred for professional platform)",
        "networkVisibility": "Connections count public (500+), actual list visibility not determined.",
        "activitySharingSettings": "Shares activity like reposts.",
        "overallPresentationStyle": "Highly curated, professional, focused on achievements and technical skills."
      },
      "observedConsumption": {
        "mainFeed": {
          "observedTopics": [
            "AI Agents",
            "Temporal Knowledge Graphs",
            "MGTOWIA crawler (niche tech topic)",
            "Generative AI in media (Netflix)",
            "Keyword Embedding",
            "Personalized Search/Memex",
            "OpenAI updates"
          ],
          "frequentPostersAccounts": [
            "Maryam Miradi, PhD",
            "Milda Naciute",
            "Eduardo Ordax (Sponsored)",
            "Philomena Gyprocua",
            "Snehanshu Raj",
            "Richard Huston"
          ],
          "contentFormatCharacteristics": "Text posts with diagrams/images, technical discussions, product announcements.",
          "specificContentExamples": [
            "Maryam Miradi's post on 'GraphMemory: Temporal Knowledge Graph for Personalized AI Agents'",
            "Milda Naciute's post about building an 'MGTOWIA crawler'",
            "Snehanshu Raj's post on a 'memex' like personal search tool."
          ]
        },
        "discoveryFeed": None,
        "consumptionAnalysisNotes": "The LinkedIn main feed is heavily skewed towards advanced AI/ML topics, Generative AI applications, and technical discussions. This aligns perfectly with Homen's stated skills, experiences, and projects. The content consumed is from individuals who appear to be researchers, engineers, or thought leaders in these niche tech domains. This consumption pattern reinforces his deep engagement with his professional field."
      },
      "platformFeatureUsage": [
        {"featureName": "Featured Section", "usageDescription": "Highlights key posts, links, and images.", "frequency": "Utilized"},
        {"featureName": "Skills & Endorsements", "usageDescription": "Lists numerous technical and business skills.", "frequency": "Utilized"},
        {"featureName": "Projects Section", "usageDescription": "Details specific technical projects with links.", "frequency": "Utilized"},
        {"featureName": "Activity (Reposts)", "usageDescription": "Shares relevant industry news and technical updates.", "frequency": "Sporadic"}
      ],
      "platformSpecificConclusions": "Homen Shum's LinkedIn profile is a comprehensive and highly professional showcase of his expertise in AI/ML, Generative AI, data science, and finance. His experiences, projects, and skills sections are detailed and highlight a strong trajectory in leveraging advanced technologies for practical applications, particularly within startups and large financial institutions like JPMorgan Chase. The 'About' section provides a rich narrative of his recent impactful projects. Network suggestions and consumed content align perfectly with his professional focus, indicating LinkedIn's algorithm correctly identifies him as a tech professional in AI and finance. The profile is geared towards professional networking, showcasing technical achievements, and career development in these high-tech fields."
    },
    "instagram": {
      "platformName": "Instagram",
      "profileFundamentals": {
        "username": "homenshum",
        "fullName": "Homen Shum",
        "pronouns": None,
        "location": None,
        "profileURL": "https://www.instagram.com/homenshum/",
        "profileLanguage": "English",
        "verificationStatus": None,
        "contactInfoVisible": False,
        "profilePictureDescription": "Profile picture not directly visible, but likely aligns with other platforms.",
        "bannerImageDescription": None,
        "linkedWebsites": []
      },
      "bioAnalysis": {
        "fullText": None,
        "identifiedKeywords": [],
        "hashtagsInBio": [],
        "mentionedUsersInBio": [],
        "statedPurposeFocus": "Not visible, likely personal use.",
        "tone": "Not visible",
        "callToAction": None
      },
      "storyHighlights": [],
      "contentGenerationActivity": {
        "postingFrequency": "Unknown (own content not extensively viewed)",
        "dominantContentTypes": ["Likes/comments on others' posts (e.g., 'homensai' comment on DJ event)"],
        "contentExamples": ["Commented as 'homensai' on an event post."],
        "recurringThemesTopics": [],
        "overallToneVoice": "Casual (inferred from Reels consumption)",
        "recentActivityExamples": ["Liked DJ KØNG event post by hsentertainment_eclipse.", "Commented on DJ KØNG event post (as homensai)."],
        "gridAesthetic": None,
        "reelsPerformanceIndicators": None,
        "storiesFrequencyEngagement": None
      },
      "engagementPatterns": {
        "outgoingInteractionStyle": "Likes posts (e.g., event flyers), comments occasionally.",
        "typesOfContentEngagedWith": ["Events", "Motivational quotes (liked by 'bomzai')"],
        "incomingEngagementHighlights": None,
        "typicalIncomingCommentTypes": []
      },
      "networkCommunityRecommendations": {
        "followerCount": None,
        "followingCount": None,
        "audienceDescription": "Likely personal connections, general audience.",
        "groupCommunityMemberships": [],
        "inboxSidebarPreview": None,
        "inboxSidebarAnalysis": None,
        "myNetworkTabVisibility": None,
        "myNetworkTabAnalysis": None,
        "platformSuggestions": {
          "suggestedPeople": [
            {"fullName": "kukungpao", "headlineOrTitle": None, "profileURL": None, "reasonForSuggestion": "Suggested for you", "locationContext": "Sidebar", "visibleMetrics": None},
            {"fullName": "dorothychen_", "headlineOrTitle": None, "profileURL": None, "reasonForSuggestion": "Suggested for you", "locationContext": "Sidebar", "visibleMetrics": None},
            {"fullName": "jessieleeg_", "headlineOrTitle": None, "profileURL": None, "reasonForSuggestion": "Suggested for you", "locationContext": "Sidebar", "visibleMetrics": None},
            {"fullName": "bestboii", "headlineOrTitle": None, "profileURL": None, "reasonForSuggestion": "Suggested for you", "locationContext": "Sidebar", "visibleMetrics": None},
            {"fullName": "billieeilishking_benedi", "headlineOrTitle": None, "profileURL": None, "reasonForSuggestion": "Suggested for you", "locationContext": "Sidebar", "visibleMetrics": None},
            {"fullName": "Homen Shum", "headlineOrTitle": None, "profileURL": None, "reasonForSuggestion": "Suggested for you (self)", "locationContext": "Sidebar", "visibleMetrics": None}
          ],
          "suggestedCompaniesOrPages": [],
          "suggestedGroupsEvents": [],
          "suggestedContentOrTopics": [],
          "peopleAlsoViewed": [],
          "otherSuggestions": []
        },
        "platformRecommendationsAnalysis": "Instagram's 'Suggested for you' list appears to be general, possibly based on broader popularity or loose connections, rather than deep professional or niche interest alignment seen on LinkedIn. The suggestions are for individual accounts without clear professional titles, typical of Instagram's more social focus. The platform seems to perceive him as a general user.",
        "detailedConnectionsList": None,
        "detailedConnectionsAnalysis": None
      },
      "privacyPresentation": {
        "accountVisibility": "Public (inferred as posts are visible)",
        "postLevelVisibility": "Public",
        "networkVisibility": "Unknown",
        "activitySharingSettings": "Likes and comments are visible.",
        "overallPresentationStyle": "Casual, consumption-focused (based on viewed Reels)."
      },
      "observedConsumption": {
        "mainFeed": {
          "observedTopics": ["Local Events (DJ)", "Motivational Quotes"],
          "frequentPostersAccounts": ["hsentertainment_eclipse", "bomzai"],
          "contentFormatCharacteristics": "Event flyers, image macros with text.",
          "specificContentExamples": [
            "DJ KØNG 'CYBERCORE' event flyer.",
            "Motivational quote post by 'bomzai': 'If you're willing to suck at anything for 100 days...'"
          ]
        },
        "discoveryFeed": {
          "observedThemes": [
            "Surreal/Artistic visuals",
            "Anime/Animation",
            "Science/Nature facts (preserved insect)",
            "Food/Unusual Food items",
            "Memes (general, some sexually suggestive, relationship humor)",
            "Mobile Gaming",
            "Self-help/Motivational (short form)",
            "Short-form comedy/skits",
            "Physical appearance (abs, male gaze critique)",
            "Cultural observations (Chinese text overlays)"
          ],
          "prevalentContentTypes": [
            "Short videos (Reels)",
            "Image macros",
            "Animated clips"
          ],
          "commonSoundsEffectsTrends": ["Varied, typical of Reels feed with trending sounds and visual styles."],
          "highlyDescriptiveExamples": [
            "Reel: Woman in white dress in muddy puddle (artistic).",
            "Reel: Preserved insect in amber with text '25 Million-Year-Old Parasitic Wasp...'.",
            "Reel: Woman on beach with text 'MALE GAZE'.",
            "Reel: Doctor Strange creating portal.",
            "Reel: Sexually suggestive meme 'when he tells me stop stroken his d!ck after he finishes'.",
            "Reel: Mobile strategy game footage with text 'Peak disrespect'.",
            "Reel: Man showing abs, Chinese text overlay.",
            "Reel: Jujutsu Kaisen character (Sukuna) with 'BAN RATE 1.97%'.",
            "Reel: Man with large flat white object, text 'This is THE'.",
            "Reel: Cucumber slice with scared face, text 'Wait this isn't a salad...'"
          ],
          "overallFeedCharacteristics": "Highly diverse, fast-paced, visually driven, typical of a general interest Reels feed. Content ranges from educational snippets to humor, memes, and trending video styles. Reflects broad, casual content consumption."
        },
        "consumptionAnalysisNotes": "The Instagram main feed shows some local interests (events) and general motivational content. The Reels (discovery) feed is far more diverse, indicating broad, passive consumption of typical short-form video content spanning humor, memes, quick facts, anime, and trending topics. This consumption is characteristic of entertainment-driven usage and contrasts sharply with the highly focused, professional content consumption on LinkedIn. It suggests Instagram is used more for leisure and general interest browsing."
      },
      "platformFeatureUsage": [
        {"featureName": "Reels", "usageDescription": "Actively consumes Reels.", "frequency": "Frequent (based on observation period)"},
        {"featureName": "Feed Browsing", "usageDescription": "Browses main feed content.", "frequency": "Observed"},
        {"featureName": "Liking Posts", "usageDescription": "Likes posts in feed.", "frequency": "Observed"},
        {"featureName": "Commenting", "usageDescription": "Comments on posts (as homensai).", "frequency": "Observed"}
      ],
      "platformSpecificConclusions": "Homen Shum's Instagram presence (@homenshum, with related activity from @homensai) appears to be for personal and casual use, contrasting significantly with his professional LinkedIn profile. While his own content generation wasn't extensively observed, his consumption patterns, especially in the Reels feed, point to a broad range of general interests typical of social media entertainment (memes, anime, humor, quick facts). Platform suggestions are general. This platform serves a different purpose, likely for leisure, entertainment, and staying connected with non-professional interests."
    }
  },
  "crossPlatformSynthesis": {
    "consistencyVsVariation": {
      "profileElementConsistency": "The name 'Homen Shum' is consistent. Profile picture style (professional headshot on LinkedIn) likely differs from a more casual one on Instagram (though Instagram's wasn't clearly seen). Username 'homen-shum' on LinkedIn vs 'homenshum' on Instagram is similar.",
      "contentTonePersonaConsistency": "Strong dichotomy: LinkedIn is exclusively professional, technical, and achievement-focused. Instagram (based on consumption and limited interaction) is casual, entertainment-driven, and reflects general interests. The persona on LinkedIn is a driven AI/finance professional; on Instagram, it's a typical social media consumer.",
      "notableDifferences": "The primary difference is the purpose and content. LinkedIn is a professional portfolio and networking tool. Instagram is for leisure and broad-interest content consumption. The depth of profile information is vast on LinkedIn and minimal/unseen on Instagram. Consumption on LinkedIn is highly specialized (AI/ML), while on Instagram it's very general (memes, entertainment)."
    },
    "contentOverlapStrategy": "No direct content cross-posting was observed. The platforms serve distinct purposes, so content strategy is appropriately segmented. An account 'homensai' commented on an Instagram post, which might be an alternative or interest-specific account of Homen Shum.",
    "synthesizedExpertiseInterests": {
      "coreProfessionalSkills": [
        "Large Language Models (LLMs)",
        "Generative AI",
        "Python",
        "Data Analysis",
        "Machine Learning",
        "Cloud Platforms (Azure, GCP, AWS)",
        "Software Development (Flask, Docker, Streamlit)",
        "Financial Analysis & Technology (FinTech)",
        "Healthcare Technology (HealthTech)",
        "Startup Development & Leadership",
        "AI Ethics & Responsible AI (inferred)",
        "Cybersecurity (Semantic Matching)",
        "Automation",
        "Project Management",
        "Technical Leadership"
      ],
      "corePersonalInterests": [
        "Technology Trends (general)",
        "Anime/Animation",
        "Memes & Online Humor",
        "Science & Nature Facts",
        "Gaming (casual/mobile)",
        "Local Events (DJ/Music)",
        "Motivational Content",
        "Food & Culinary Curiosities"
      ]
    },
    "overallOnlinePersonaNarrative": "Homen Shum projects a dual online persona. Professionally (LinkedIn), he is a highly skilled and ambitious technologist specializing in cutting-edge AI/ML, particularly LLMs and Generative AI, with a strong track record of impactful projects in both startup environments (FinAdvizly) and major corporations (JPMorgan Chase). His narrative is one of continuous learning, innovation, and application of complex technologies to solve real-world problems in finance and healthcare. Personally (Instagram), he appears to be a typical consumer of diverse online content for entertainment and leisure, engaging with a wide array of topics common in popular social media feeds. This clear separation indicates a well-managed online presence, tailoring platform use to specific needs – professional advancement versus personal relaxation.",
    "professionalEvaluation": {
      "strengthsSkillsMatch": "Exceptional alignment between listed skills, detailed experiences, and showcased projects, particularly in AI/ML, LLMs, Python, cloud platforms, and their application in finance/healthcare. Demonstrates both theoretical knowledge and practical implementation capabilities. His roles at FinAdvizly and JPM, along with hackathon successes, validate these skills.",
      "impactAchievements": "Significant impact demonstrated through: founding FinAdvizly and developing its GenAI products; leading teams to hackathon victories (UC Berkeley AI, Nation's Cybersecurity); implementing AI solutions at JPMorgan Chase that resulted in major efficiency gains (e.g., reducing processing time for 2000+ companies from weeks to seconds); developing multiple publicly accessible AI-powered web applications.",
      "industryEngagement": "High engagement with the AI/ML industry, evident from his LinkedIn feed consumption (following AI researchers/developments), reposting technical content, and the nature of his connections/suggestions (peers in tech/finance). Following thought leaders like Ray Dalio also indicates broader industry awareness.",
      "potentialRedFlagsClarifications": "No red flags. The rapid succession of roles and diverse projects could be seen as dynamic and entrepreneurial rather than unstable, given his startup focus and project-based achievements within JPM.",
      "overallCandidateSummary": "Homen Shum is a highly promising and accomplished young professional in the AI/ML and finance/health-tech space. He possesses a strong technical foundation, proven leadership abilities, an entrepreneurial spirit, and a clear passion for applying advanced AI to create valuable solutions. His ability to deliver tangible results in both startup and corporate settings is a key strength."
    },
    "marketTrendInsights": {
      "keyTechnologiesToolsTopics": [
        "Large Language Models (LLMs)",
        "Generative AI (GenAI)",
        "Retrieval Augmented Generation (RAG)",
        "Multi-agent Systems",
        "Cloud Computing (GCP, Azure, AWS)",
        "Python, Pandas, Flask, Docker, Streamlit",
        "AI in Finance (FinTech)",
        "AI in Healthcare (HealthTech)",
        "Semantic Search/Matching",
        "AI-driven Automation"
      ],
      "emergingThemesNiches": [
        "Personalized AI Agents",
        "Temporal Knowledge Graphs for AI",
        "Democratization of AI tools (e.g., via Streamlit apps)",
        "Ethical AI application (implied by focus on accessible tech)"
      ],
      "relevantContentPatterns": "Consumption of deep technical content, research papers/summaries, and product announcements in the AI/ML space on LinkedIn. Consumption of diverse, short-form entertainment on Instagram."
    },
    "inferredAlgorithmicPerception": [
      {
        "platformName": "LinkedIn",
        "categorizationHypothesis": "LinkedIn's algorithm perceives Homen Shum as a highly skilled professional in Artificial Intelligence, Machine Learning, Data Science, and Software Engineering, with specific expertise in Large Language Models and Generative AI. It identifies his industry focus in Finance (FinTech) and Healthcare (HealthTech), and his experience with startups and major financial institutions. This is strongly supported by: the technical nature of his consumed feed (AI research, GraphMemory, Memex tools); the professional suggestions of people from his alma maters and former/current workplaces (UCSB, JPM connections); the types of roles in 'People Also Viewed' (Product Design). The algorithm likely flags him for opportunities related to AI/ML engineering, data science leadership, and roles at the intersection of tech and finance/healthcare."
      },
      {
        "platformName": "Instagram",
        "categorizationHypothesis": "Instagram's algorithm likely categorizes Homen Shum as a general user with broad interests typical of his demographic. His consumption of a diverse Reels feed (memes, anime, science snippets, gaming, humor, artistic visuals) suggests interests in entertainment, pop culture, and light educational content. The 'Suggested for you' profiles are generic and don't reflect his deep professional niche. The platform likely targets him with a wide array of trending and generally popular content rather than specialized professional material. His engagement (likes on event posts, motivational quotes) reinforces this general user profile. The algorithm sees him primarily as a consumer of entertainment content."
      }
    ],
    "crossPlatformNetworkAnalysis": {
      "overlappingConnectionsRecommendations": [],
      "networkComparisonNotes": "LinkedIn network suggestions are highly targeted based on professional and educational background, focusing on finance, tech, and AI. Instagram suggestions are generic. This highlights the different algorithmic approaches and platform purposes: LinkedIn for professional networking, Instagram for broader social connections and content discovery.",
      "consumptionComparisonNotes": "Drastic difference: LinkedIn consumption is hyper-focused on deep AI/ML technical content, reflecting professional specialization. Instagram consumption is a broad mix of entertainment, memes, and general interest short-form videos, typical of leisure browsing. This clearly demonstrates segmented platform usage for professional development versus personal entertainment."
    }
  },
  "finalComprehensiveSummary": "Homen Shum's online presence reveals a well-defined dual identity, expertly managed across professional and personal platforms. LinkedIn showcases him as a highly skilled and accomplished technologist deeply immersed in AI, Machine Learning, and Generative AI, with significant achievements in finance and healthcare sectors, evidenced by his detailed profile, technical project showcases, and focused content consumption. The platform's suggestions and his inbox activity further confirm his strong professional network within these domains. Conversely, his Instagram activity, particularly his Reels consumption, points to a more conventional use of social media for broad-interest entertainment and leisure, with content spanning memes, anime, and general trends. This clear segmentation underscores a strategic approach to his digital footprint, leveraging each platform for its intended purpose – LinkedIn for robust professional branding and networking, and Instagram for personal interests and casual engagement. The algorithmic perception on each platform aligns perfectly with this observed behavior, reinforcing his specialized professional identity on LinkedIn and his general consumer profile on Instagram."
}

def load_example_data():
    """Loads the hardcoded example JSON data."""
    return copy.deepcopy(EXAMPLE_ANALYSIS_DATA)

# --- find_missing_structured (no changes needed) ---
def find_missing_structured(data_node, current_path_keys=None, results=None):
    if current_path_keys is None: current_path_keys = []
    if results is None: results = {}
    if isinstance(data_node, dict):
        section_path_str = " -> ".join(format_key_to_title(k) for k in current_path_keys) if current_path_keys else "Top Level"
        for key, value in data_node.items():
            field_name = format_key_to_title(key)
            new_path_keys = current_path_keys + [key]
            if is_value_empty(value):
                if section_path_str not in results: results[section_path_str] = []
                is_container = isinstance(value, (dict, list))
                field_label = f"{field_name}"
                if is_container and value is not None: field_label += " (Section Empty)"
                if field_label not in results[section_path_str]: results[section_path_str].append(field_label)
            elif isinstance(value, (dict, list)):
                find_missing_structured(value, new_path_keys, results)
    elif isinstance(data_node, list):
        for index, item in enumerate(data_node):
             if isinstance(item, (dict, list)):
                 item_path_keys = current_path_keys + [f"{current_path_keys[-1] if current_path_keys else 'list'}[{index}]"]
                 find_missing_structured(item, item_path_keys, results)
    return results

def format_missing_summary_html(missing_structured_data, max_items_per_section=5):
    if not missing_structured_data: return "<p>No missing or empty fields found.</p>"
    html_content = "<h6>Summary of Missing or Empty Fields:</h6><ul>"
    sorted_sections = sorted(missing_structured_data.keys())
    for section_path in sorted_sections:
        missing_fields = missing_structured_data[section_path]
        if not missing_fields: continue
        html_content += f"<li><strong>{html.escape(section_path)}:</strong><ul>"
        for i, field in enumerate(missing_fields):
            if i < max_items_per_section: html_content += f"<li>{html.escape(field)}</li>"
            elif i == max_items_per_section:
                html_content += f"<li>... ({len(missing_fields) - max_items_per_section} more)</li>"
                break
        html_content += "</ul></li>"
    html_content += "</ul>"
    return f"<p>{html_content}</p>"

def render_nested_json_html(data, level=0, show_empty_fields=False):
    html_output = ""
    indent_style = f"margin-left: {level * 0.5}rem;"
    if is_value_empty(data) and not show_empty_fields and level > 0: return ""

    if isinstance(data, dict):
        items_to_render = data.items()
        if not show_empty_fields: items_to_render = {k: v for k, v in data.items() if not is_value_empty(v)}.items()
        if not items_to_render and not show_empty_fields: return ""
        
        container_class = "platform-card-like" if level == 1 else "nested-item-container"
        platform_name = data.get('platformName', '').lower().replace(" ", "-").replace("/", "-").replace("+", "-").replace(".", "")
        if platform_name: container_class += f" {platform_name}"
        html_output += f'<div class="{container_class}" style="{indent_style}">'
        dict_to_iterate = data if show_empty_fields else dict(items_to_render)

        for key, value in dict_to_iterate.items():
            if is_value_empty(value) and not show_empty_fields: continue
            title = html.escape(format_key_to_title(key))
            heading_level = min(level + 3, 6) 
            icon = "" # Simplified icon logic for brevity, assume it's complex and works
            if heading_level == 3: icon = '<span class="coffee-icon">☕</span>'
            elif heading_level == 4: icon = '<span class="diff-icon">🍵</span>'
            elif heading_level == 5: icon = '<span class="detail-icon">🔸</span>'
            elif heading_level == 6: icon = '<span class="list-icon">✦</span>'
            
            html_output += f'<h{heading_level}>{icon}{title}</h{heading_level}>'
            if 'consumption' in key.lower():
                if heading_level == 4: html_output = html_output.replace(f'<h4>{icon}{title}</h4>', f'<h4 class="consumption-title">{icon}{title}</h4>')
                elif heading_level == 5: html_output = html_output.replace(f'<h5>{icon}{title}</h5>', f'<h5 class="consumption-subtitle">{icon}{title}</h5>')
            html_output += render_nested_json_html(value, level + 1, show_empty_fields=show_empty_fields)
        html_output += '</div>'

    elif isinstance(data, list):
        if not data and show_empty_fields: html_output += f'<div style="{indent_style}">{render_missing_html()} (Empty List)</div>'
        else:
            all_strings = all(isinstance(item, str) for item in data if item is not None)
            if all_strings and any(item and item.strip() for item in data):
                pills_content = render_pills_html(data, show_label=False, show_empty_fields=show_empty_fields, pill_class="pill")
                if pills_content: html_output += f'<div style="{indent_style}">{pills_content}</div>'
                elif show_empty_fields: html_output += f'<div style="{indent_style}">{render_missing_html()} (Empty String List)</div>'
            else:
                list_content_html = ""
                for i, item in enumerate(data):
                    item_html = render_nested_json_html(item, level + 1, show_empty_fields=show_empty_fields)
                    if item_html:
                        list_content_html += f'<h6><span class="list-icon">✦</span> Item {i+1}</h6>{item_html}'
                        if i < len(data) - 1 and item_html.strip(): list_content_html += "<hr style='border: none; border-top: 1px dashed var(--cc-accent-light-tan); margin: 0.8rem 0;'>"
                if list_content_html:
                    hr_str = "<hr style='border: none; border-top: 1px dashed var(--cc-accent-light-tan); margin: 0.8rem 0;'>"
                    if list_content_html.endswith(hr_str): list_content_html = list_content_html[:-len(hr_str)]
                    html_output += f'<div style="{indent_style}">{list_content_html}</div>'
                elif show_empty_fields: html_output += f'<div style="{indent_style}">{render_missing_html()} (List items hidden or empty)</div>'
    elif isinstance(data, str):
        if is_value_empty(data) and show_empty_fields: html_output += f'<div style="{indent_style}">{render_missing_html()}</div>'
        elif not is_value_empty(data):
            escaped_data = html.escape(data)
            if escaped_data.startswith("http"): html_output += f'<div style="{indent_style}"><a href="{escaped_data}" target="_blank">🔗 {escaped_data}</a></div>'
            elif '\n' in escaped_data: html_output += f'<div style="{indent_style}"><p class="prewrap">{escaped_data.replace(chr(10), "<br>")}</p></div>'
            else: html_output += f'<div style="{indent_style}"><p>{escaped_data}</p></div>'
    elif isinstance(data, (int, float, bool)): html_output += f'<div style="{indent_style}"><p>{html.escape(str(data))}</p></div>'
    elif data is None and show_empty_fields: html_output += f'<div style="{indent_style}">{render_missing_html()}</div>'
    elif data is not None: html_output += f'<div style="{indent_style}"><p><em>Unsupported: {html.escape(type(data).__name__)}</em></p></div>'
    return html_output

@st.dialog("Edit Details", width="large")
def open_edit_dialog(section_data_to_edit, path_to_update_in_state, section_display_name_for_dialog):
    st.subheader(f"✏️ Editing: {section_display_name_for_dialog}")
    form_key = f"dialog_form_{'_'.join(map(str,path_to_update_in_state)).replace('[', '_').replace(']', '_')}"
    with st.form(key=form_key):
        edited_values_capture = {}
        for field_key, field_value in section_data_to_edit.items():
            field_label = format_key_to_title(field_key)
            unique_widget_key = f"dialog_edit_{'_'.join(map(str,path_to_update_in_state)).replace('[', '_').replace(']', '_')}_{field_key}"
            if isinstance(field_value, list) and all(isinstance(item, str) for item in field_value):
                new_value_str = st.text_area(f"{field_label} (comma-separated)", value=", ".join(field_value), key=unique_widget_key)
                edited_values_capture[field_key] = [item.strip() for item in new_value_str.split(',') if item.strip()]
            elif isinstance(field_value, str):
                if len(field_value) > 200 or '\n' in field_value:
                    edited_values_capture[field_key] = st.text_area(field_label, value=field_value, key=unique_widget_key, height=150)
                else:
                    edited_values_capture[field_key] = st.text_input(field_label, value=field_value, key=unique_widget_key)
            elif isinstance(field_value, (int, float)):
                edited_values_capture[field_key] = st.number_input(field_label, value=field_value, key=unique_widget_key)
            elif isinstance(field_value, bool):
                edited_values_capture[field_key] = st.checkbox(field_label, value=field_value, key=unique_widget_key)
            elif field_value is None:
                new_value_text = st.text_input(f"{field_label} (Currently None)", value="", key=unique_widget_key, placeholder="Enter value or leave empty")
                edited_values_capture[field_key] = new_value_text if new_value_text.strip() else None
            else: # Non-editable complex types (like list of dicts for experiences for now)
                st.markdown(f"**{field_label} ({type(field_value).__name__}):**")
                st.caption(f"Direct editing for this type not supported in this dialog. Original value will be preserved.")
                edited_values_capture[field_key] = field_value 
        submitted = st.form_submit_button("💾 Save Changes")
    if submitted:
        try:
            current_data_ref = st.session_state['analysis_data']
            for i, key_segment in enumerate(path_to_update_in_state):
                if i == len(path_to_update_in_state) - 1:
                    if isinstance(key_segment, str) and key_segment in current_data_ref:
                        current_data_ref[key_segment].update(edited_values_capture) # Update existing dict
                        st.session_state['edit_success_message'] = f"Changes saved for '{section_display_name_for_dialog}'."
                        st.rerun()
                        return
                    elif isinstance(key_segment, int) and isinstance(current_data_ref, list) and 0 <= key_segment < len(current_data_ref):
                        if isinstance(current_data_ref[key_segment], dict):
                           current_data_ref[key_segment].update(edited_values_capture)
                           st.session_state['edit_success_message'] = f"Changes saved for '{section_display_name_for_dialog}'."
                           st.rerun()
                           return
                        else: # Should not happen if path is correct
                           st.error("Error: Expected a dictionary at list index during save.")
                           return
                    else: st.error(f"Error: Key/Index '{key_segment}' not found or invalid type at path.") ; return
                else: # Navigate
                    if isinstance(key_segment, str) and key_segment in current_data_ref and isinstance(current_data_ref[key_segment], dict):
                        current_data_ref = current_data_ref[key_segment]
                    elif isinstance(key_segment, int) and isinstance(current_data_ref, list) and 0 <= key_segment < len(current_data_ref):
                        current_data_ref = current_data_ref[key_segment]
                    else: st.error(f"Error: Invalid path segment '{key_segment}'.") ; return
        except Exception as e: st.error(f"Error saving changes: {e}")
    if st.button("❌ Cancel", key="dialog_cancel_button"): st.rerun()

def get_platform_display_name(platform_key, platform_data):
    if isinstance(platform_data, dict):
        name_from_field = platform_data.get("platformName")
        if name_from_field and isinstance(name_from_field, str): return name_from_field
    return format_key_to_title(platform_key)

def render_edit_interaction_point():
    if 'analysis_data' not in st.session_state or not st.session_state['analysis_data']:
        st.info("Load data first to enable editing.")
        return
    editable_data_snapshot = st.session_state['analysis_data']
    editable_sections = {}

    # Add Coffee Card section first
    if isinstance(editable_data_snapshot.get('yourCoffeeCard'), dict):
        editable_sections['yourCoffeeCard'] = ("Your Coffee Card", ['yourCoffeeCard'])

    # Add other top-level editable dicts
    for key, value in editable_data_snapshot.items():
        if isinstance(value, dict) and key not in ['yourCoffeeCard', 'platformSpecificAnalysis', 'crossPlatformSynthesis']:
             editable_sections[key] = (format_key_to_title(key), [key])
    
    # Add crossPlatformSynthesis if it's a dict
    if isinstance(editable_data_snapshot.get('crossPlatformSynthesis'), dict):
        cps_data = editable_data_snapshot['crossPlatformSynthesis']
        # Allow editing of sub-dictionaries within crossPlatformSynthesis if they exist
        for sub_key, sub_value in cps_data.items():
            if isinstance(sub_value, dict): # Only if it's a dict itself
                edit_key_for_state = f"crossPlatform_{sub_key}"
                editable_sections[edit_key_for_state] = (f"Synthesis: {format_key_to_title(sub_key)}", ['crossPlatformSynthesis', sub_key])


    # Add platformSpecificAnalysis sub-sections
    if isinstance(editable_data_snapshot.get('platformSpecificAnalysis'), dict):
        psa_data = editable_data_snapshot['platformSpecificAnalysis']
        for p_key, p_value in psa_data.items():
            if isinstance(p_value, dict): # Only if it's a dict itself (not a list for "otherPlatforms")
                edit_key_for_state = f"platform_{p_key}"
                display_name = get_platform_display_name(p_key, p_value)
                editable_sections[edit_key_for_state] = (f"Platform: {display_name}", ['platformSpecificAnalysis', p_key])
    
    if not editable_sections:
        st.warning("No directly editable dictionary sections found in the loaded data.")
        return

    section_keys_for_selectbox = list(editable_sections.keys())
    selected_key_for_editing = st.selectbox(
        "Select section to edit:",
        options=section_keys_for_selectbox,
        format_func=lambda key: editable_sections[key][0], # Show display name
        key="edit_section_selector_for_dialog"
    )

    if st.button(f"⚙️ Edit '{editable_sections[selected_key_for_editing][0]}'...", key=f"open_dialog_btn_{selected_key_for_editing}"):
        display_name_for_dialog, path_to_update = editable_sections[selected_key_for_editing]
        
        data_at_path_ref = st.session_state['analysis_data']
        valid_path = True
        for segment in path_to_update:
            if isinstance(segment, str) and segment in data_at_path_ref and isinstance(data_at_path_ref[segment], dict):
                data_at_path_ref = data_at_path_ref[segment]
            elif isinstance(segment, int) and isinstance(data_at_path_ref, list) and 0 <= segment < len(data_at_path_ref):
                 data_at_path_ref = data_at_path_ref[segment]
            else:
                valid_path = False
                break
        
        if valid_path and isinstance(data_at_path_ref, dict):
            section_data_for_dialog = copy.deepcopy(data_at_path_ref)
            open_edit_dialog(
                section_data_to_edit=section_data_for_dialog,
                path_to_update_in_state=path_to_update,
                section_display_name_for_dialog=display_name_for_dialog
            )
        else:
            st.error(f"Could not retrieve or prepare data for section: {display_name_for_dialog}")


def render_custom_header():
    st.markdown("""<div style="padding:1rem 0;margin-bottom:2rem;text-align:center;position:relative;"><div style="position:absolute;top:0;left:0;width:100%;height:6px;background:var(--cc-accent-dark-brown);"></div><h1 style="font-size:2.4rem;color:var(--cc-accent-dark-brown);margin-bottom:0.3rem;letter-spacing:-0.02em;">☕ CaféCorner</h1><p style="color:var(--cc-text-secondary);font-size:1.1rem;max-width:600px;margin:0 auto;">A cozy place to view and manage your professional presence</p></div>""", unsafe_allow_html=True)

def load_example_data():
    return copy.deepcopy(EXAMPLE_ANALYSIS_DATA)

if __name__ == "__main__":
    st.set_page_config(page_title="CaféCorner", page_icon="☕", layout="wide")

    if 'analysis_data' not in st.session_state: st.session_state['analysis_data'] = None
    if 'uploaded_avatar_file' not in st.session_state: st.session_state['uploaded_avatar_file'] = None
    if 'processed_upload_filename' not in st.session_state: st.session_state['processed_upload_filename'] = None
    if 'current_data_source' not in st.session_state: st.session_state['current_data_source'] = "None"
    if 'show_empty_fields' not in st.session_state: st.session_state['show_empty_fields'] = False
    if 'edit_success_message' not in st.session_state: st.session_state['edit_success_message'] = None

    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    render_custom_header()

    if st.session_state.get('edit_success_message'):
        st.success(st.session_state['edit_success_message'], icon="✅")
        st.session_state['edit_success_message'] = None

    with st.sidebar:
        st.title("⚙️ Controls")
        st.divider()
        st.subheader("📂 Load Data")
        uploaded_analysis_file = st.file_uploader("Upload Profile JSON/TXT", type=["txt", "json"], key="analysis_file_uploader")
        uploaded_avatar_file_sidebar = st.file_uploader("Upload Profile Photo (Optional)", type=["png", "jpg", "jpeg", "gif", "webp"], key="avatar_uploader_sidebar")

        if uploaded_analysis_file is not None:
            if uploaded_analysis_file.name != st.session_state.get('processed_upload_filename'):
                try:
                    file_content = uploaded_analysis_file.read().decode("utf-8")
                    parsed_data = json.loads(file_content)
                    if isinstance(parsed_data, dict): # Simplified check
                        st.session_state['analysis_data'] = parsed_data
                        if uploaded_avatar_file_sidebar: st.session_state['uploaded_avatar_file'] = uploaded_avatar_file_sidebar
                        st.session_state['processed_upload_filename'] = uploaded_analysis_file.name
                        st.session_state['current_data_source'] = f"File: {uploaded_analysis_file.name}"
                        st.success(f"Loaded '{uploaded_analysis_file.name}'.")
                        st.rerun()
                    else: st.error("Uploaded file not a valid JSON object.")
                except Exception as e: st.error(f"Error reading file: {e}")
        
        elif uploaded_avatar_file_sidebar is not None and st.session_state.get('analysis_data') is not None:
             current_avatar_obj = st.session_state.get('uploaded_avatar_file')
             if (current_avatar_obj is None or uploaded_avatar_file_sidebar.name != current_avatar_obj.name):
                  st.session_state['uploaded_avatar_file'] = uploaded_avatar_file_sidebar
                  st.info("Avatar updated.")
                  st.rerun()

        if st.button("Load Example Data", key="load_example"):
             st.session_state['analysis_data'] = load_example_data()
             st.session_state['uploaded_avatar_file'] = None 
             st.session_state['processed_upload_filename'] = "EXAMPLE_DATA"
             st.session_state['current_data_source'] = "Example Data"
             st.info("Example data loaded.")
             st.rerun()

        st.subheader("👓 Display Options")
        new_show_empty = st.checkbox("Show Missing/Empty Fields", value=st.session_state.get('show_empty_fields', False), key='show_empty_toggle_main_cb')
        if new_show_empty != st.session_state.get('show_empty_fields', False):
            st.session_state['show_empty_fields'] = new_show_empty
            st.rerun()
        st.divider()
        st.caption(f"Current Source: {st.session_state['current_data_source']}")
        if st.session_state.get('analysis_data'):
             with st.expander("✏️ Edit Data Sections (Dialog)", expanded=False):
                  render_edit_interaction_point()

    if st.session_state.get('analysis_data'):
        data = st.session_state['analysis_data']
        avatar_file_obj = st.session_state.get('uploaded_avatar_file')
        show_empty = st.session_state.get('show_empty_fields', False)

        coffee_card_header_cols = st.columns([0.8, 0.2])
        with coffee_card_header_cols[0]: st.markdown("### ☕ Your Coffee Card")
        with coffee_card_header_cols[1]:
            if data.get('yourCoffeeCard') is not None:
                if st.button("✏️ Edit Card", key="edit_coffee_card_main_btn", help="Edit Your Coffee Card details"):
                    coffee_card_data_for_dialog = copy.deepcopy(data.get('yourCoffeeCard', {}))
                    open_edit_dialog(
                        section_data_to_edit=coffee_card_data_for_dialog,
                        path_to_update_in_state=['yourCoffeeCard'],
                        section_display_name_for_dialog="Your Coffee Card"
                    )
        
        coffee_card_data = data.get('yourCoffeeCard')
        # --- Avatar URL Logic for Coffee Card ---
        profile_name_for_avatar = safe_get(coffee_card_data, ['name']) or \
                                  safe_get(data, ['targetIndividual', 'nameOrIdentifier'], "??")
        # Pass the specific profilePictureUrlForCard if available
        profile_pic_url_from_card_data = safe_get(coffee_card_data, ['profilePictureUrlForCard'])
        
        # get_avatar_url will prioritize profile_pic_url_from_card_data
        avatar_for_card_display = get_avatar_url(
            profile_name_for_avatar, 
            avatar_file_obj, 
            profile_pic_url_from_card=profile_pic_url_from_card_data
        )
        # st.markdown(render_coffee_card_html(coffee_card_data, avatar_for_card_display, show_empty_fields=show_empty), unsafe_allow_html=True)
        st.html(render_coffee_card_html(coffee_card_data, avatar_for_card_display, show_empty_fields=show_empty))
        if show_empty:
            missing_data_structured = find_missing_structured(data)
            if missing_data_structured: st.warning(format_missing_summary_html(missing_data_structured, max_items_per_section=7), icon="☕") 
            else: st.success("✅ All fields appear complete.", icon="👍")
        st.markdown("---") 
        st.markdown("### 📊 Detailed Analysis") 

        overview_keys = ["targetIndividual", "analyzedPlatforms", "crossPlatformSynthesis", "finalComprehensiveSummary"]
        overview_data = {k: data[k] for k in overview_keys if k in data and (not is_value_empty(data[k]) or show_empty)}
        
        platform_analysis = data.get('platformSpecificAnalysis', {})
        tab_titles = ["Overview"]
        platform_render_keys = [] # Store keys to iterate for rendering tabs

        if isinstance(platform_analysis, dict):
            # Prioritize specific platforms if they exist, then 'otherPlatforms'
            main_platforms = {k: v for k, v in platform_analysis.items() if k != 'otherPlatforms' and (isinstance(v,dict) and (not is_value_empty(v) or show_empty) )}
            sorted_main_keys = sorted(main_platforms.keys())
            for pk in sorted_main_keys:
                tab_titles.append(get_platform_display_name(pk, main_platforms[pk]))
                platform_render_keys.append(pk)

            other_platforms_list = platform_analysis.get('otherPlatforms')
            if isinstance(other_platforms_list, list) and any(not is_value_empty(op) or show_empty for op in other_platforms_list):
                tab_titles.append("Other Platforms")
                platform_render_keys.append("otherPlatforms_tab_key") # Special key for handling list

        if not overview_data and not platform_render_keys: st.info("No data to display with current settings.")
        else:
            tabs = st.tabs(tab_titles)
            with tabs[0]: # Overview
                overview_html_content = render_nested_json_html(overview_data, level=0, show_empty_fields=show_empty)
                st.markdown(overview_html_content if overview_html_content else "<p>No overview data.</p>", unsafe_allow_html=True)

            for i, pk_to_render in enumerate(platform_render_keys):
                tab_idx = i + 1
                if tab_idx < len(tabs):
                    with tabs[tab_idx]:
                        if pk_to_render == "otherPlatforms_tab_key":
                            other_data_list = platform_analysis.get('otherPlatforms', [])
                            for item_idx, other_item in enumerate(other_data_list):
                                item_content_html = render_nested_json_html(other_item, level=1, show_empty_fields=show_empty)
                                if item_content_html:
                                    st.markdown(item_content_html, unsafe_allow_html=True)
                                    if item_idx < len(other_data_list) -1 : st.markdown("<hr style='border-top: 1px dashed var(--cc-accent-light-tan);'>", unsafe_allow_html=True)
                        else: # Regular platform
                            platform_content_html = render_nested_json_html(platform_analysis.get(pk_to_render, {}), level=0, show_empty_fields=show_empty)
                            st.markdown(platform_content_html if platform_content_html else "<p>No data for this platform.</p>", unsafe_allow_html=True)
    else:
        st.markdown("### 👋 Welcome!") 
        st.info("⬅️ Use the sidebar to load profile data or view the example.")
