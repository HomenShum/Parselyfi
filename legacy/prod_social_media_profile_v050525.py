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
# Helper Functions - Minor adjustments (NO significant changes needed based on theme)
# -------------------------------------

def safe_get(data, key_list, default=None):
    """Safely access nested dictionary keys."""
    if data is None:
        return default
    _data = data
    for key in key_list:
        try:
            _data = _data[key]
        except (KeyError, TypeError, IndexError):
            return default
    # Handle case where the final value is None explicitly
    return _data if _data is not None else default

def format_key_to_title(key):
    """Converts a camelCase or snake_case key to Title Case Heading."""
    if not isinstance(key, str):
        return str(key)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', key)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    s3 = s2.replace('_', ' ')
    return s3.title()

def render_missing_html():
    """Returns the HTML string for missing data."""
    # Style will be controlled by em.missing CSS rule
    return '<em class="missing">Not Provided</em>'

# Adjusted render_pills_html to use html.escape on label
def render_pills_html(items, label=None, show_label=True, pill_class="pill", show_empty_fields=False):
    """
    Returns an HTML string of pills. Returns empty string if no valid items
    or if items is empty and show_empty_fields is False.
    """
    if not items or not isinstance(items, list):
        if show_empty_fields:
            label_html = f'<h5><span class="coffee-icon"></span>{html.escape(label)}</h5>' if label and show_label else '' # Added icon placeholder
            return f'{label_html}{render_missing_html()}'
        else:
            return "" # Hide if empty and flag is off

    items_str = [html.escape(str(item).strip()) for item in items if item and str(item).strip()]
    if not items_str:
        if show_empty_fields:
            label_html = f'<h5><span class="coffee-icon"></span>{html.escape(label)}</h5>' if label and show_label else '' # Added icon placeholder
            return f'{label_html}{render_missing_html()}'
        else:
            return "" # Hide if effectively empty and flag is off

    pills_html_content = "".join(f'<span class="{pill_class}">{item}</span>' for item in items_str)
    # Always show label with icon if there are pills
    label_html = f'<h5><span class="coffee-icon"></span>{html.escape(label)}</h5>' if label and show_label else '' # Added icon placeholder
    return f'{label_html}<div class="pill-container">{pills_html_content}</div>'


def is_value_empty(value):
    """Checks if a value is None, an empty string/list/dict, or just whitespace."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False


# --- Avatar Helper - Use Coffee Card Colors ---
def make_initials_svg_avatar(name: str, size: int = 80,
                            bg: str = "var(--cc-accent-dark-brown)", # Use dark brown bg
                            fg: str = "var(--cc-bg-main)") -> str:    # Use light card bg for text
    """Generates a Base64 encoded SVG avatar with initials."""
    display_name = name if name and name != render_missing_html() else "?"
    if not isinstance(display_name, str):
        display_name = str(display_name)
    initials = "".join([w[0].upper() for w in display_name.split()][:2]) or "?"

    # Cannot use CSS variables directly in SVG easily, so insert hex codes
    bg_color = "#6b4f4f" # --cc-accent-dark-brown
    fg_color = "#fff8f0" # --cc-bg-main

    svg = f'''
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">
<circle cx="{size/2}" cy="{size/2}" r="{size/2}" fill="{bg_color}"/>
<text x="50%" y="50%" fill="{fg_color}" font-size="{int(size/2.2)}"
        text-anchor="middle" dominant-baseline="central"
        font-family="sans-serif" font-weight="500">{initials}</text>
</svg>'''
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"

# --- get_avatar_url (no changes needed) ---
def get_avatar_url(name, uploaded_avatar_file_obj=None):
    """Gets the avatar URL, preferring uploaded file, falling back to initials."""
    if uploaded_avatar_file_obj is not None:
        try:
            image_bytes = uploaded_avatar_file_obj.getvalue()
            b64_image = base64.b64encode(image_bytes).decode()
            mime_type = uploaded_avatar_file_obj.type
            return f"data:{mime_type};base64,{b64_image}"
        except Exception as e:
            st.warning(f"Could not process uploaded avatar: {e}. Using initials.")
    return make_initials_svg_avatar(name if name else "??", size=90) # Ensure size matches CSS

# --- find_missing_structured (no changes needed) ---
def find_missing_structured(data_node, current_path_keys=None, results=None):
    """
    Recursively finds empty fields and returns them structured by section.
    Returns a dictionary where keys are section paths and values are lists of
    missing field names within that section.
    """
    if current_path_keys is None:
        current_path_keys = []
    if results is None:
        results = {}

    if isinstance(data_node, dict):
        all_children_empty = all(is_value_empty(v) for v in data_node.values())
        section_path_str = " -> ".join(format_key_to_title(k) for k in current_path_keys) if current_path_keys else "Top Level"

        for key, value in data_node.items():
            field_name = format_key_to_title(key)
            new_path_keys = current_path_keys + [key]

            if is_value_empty(value):
                if section_path_str not in results:
                    results[section_path_str] = []
                is_container = isinstance(value, (dict, list))
                field_label = f"{field_name}"
                if is_container and value is not None:
                     field_label += " (Section Empty)"

                if field_label not in results[section_path_str]:
                     results[section_path_str].append(field_label)

            elif isinstance(value, (dict, list)):
                find_missing_structured(value, new_path_keys, results)

    elif isinstance(data_node, list):
        section_path_str = " -> ".join(format_key_to_title(k) for k in current_path_keys) if current_path_keys else "Top Level"
        for index, item in enumerate(data_node):
             if isinstance(item, (dict, list)):
                 item_path_keys = current_path_keys + [f"{current_path_keys[-1]}[{index}]"]
                 find_missing_structured(item, item_path_keys, results)

    return results

# --- format_missing_summary_html (no changes needed) ---
def format_missing_summary_html(missing_structured_data, max_items_per_section=5):
    """Formats the structured missing data into cleaner HTML."""
    if not missing_structured_data:
        return "<p>No missing or empty fields found.</p>"

    # Styling is now handled by CSS targeting elements within st.warning
    html_content = "<h6>Summary of Missing or Empty Fields:</h6>" # Use h6 as specified in CSS
    html_content += "<ul>"

    sorted_sections = sorted(missing_structured_data.keys())

    for section_path in sorted_sections:
        missing_fields = missing_structured_data[section_path]
        if not missing_fields:
            continue

        html_content += f"<li><strong>{html.escape(section_path)}:</strong>" # Strong tag for section path
        html_content += "<ul>" # Nested list for fields
        for i, field in enumerate(missing_fields):
            if i < max_items_per_section:
                html_content += f"<li>{html.escape(field)}</li>" # List item for field
            elif i == max_items_per_section:
                html_content += f"<li>... ({len(missing_fields) - max_items_per_section} more)</li>"
                break
        html_content += "</ul></li>"

    html_content += "</ul>"
    # Wrap in a paragraph tag to work better with Streamlit markdown/HTML rendering inside components like st.warning
    return f"<p>{html_content}</p>"

# -------------------------------------
# 1. Specialized Coffee Card Renderer - Returns HTML String (Ensure correct classes/structure)
# -------------------------------------
def render_coffee_card_html(card_data, avatar_url, show_empty_fields=False):
    """
    Renders the specialized Coffee Card UI, optionally hiding empty fields.
    Uses the class names targeted by the updated CSS_STYLES.
    """
    if not card_data or not isinstance(card_data, dict):
        # Use the main card class structure for consistency
        return f'<div class="card coffee-card-container"><p>{render_missing_html()} (Coffee Card data missing or invalid)</p></div>'

    # --- Extract data using safe_get ---
    name_raw = safe_get(card_data, ['name'])
    title_raw = safe_get(card_data, ['title'])
    location_raw = safe_get(card_data, ['location'])
    experiences_raw = safe_get(card_data, ['experiences'])
    interests_raw = safe_get(card_data, ['interests'])
    hobbies_raw = safe_get(card_data, ['hobbies'])
    skills_raw = safe_get(card_data, ['skills'])

    # --- Prepare HTML snippets (Conditionally Rendered) ---

    # Name (uses h1 inside .header-text)
    name_html = ""
    if not is_value_empty(name_raw):
        name_html = f"<h1>{html.escape(name_raw)}</h1>"
    elif show_empty_fields:
        name_html = f"<h1>{render_missing_html()}</h1>"

    # Title (uses p.headline inside .header-text)
    title_html = ""
    if not is_value_empty(title_raw):
        title_html = f'<p class="headline">{html.escape(title_raw)}</p>'
    elif show_empty_fields:
         if name_html: title_html = f'<p class="headline">{render_missing_html()}</p>'

    # Location (uses p.headline - consider a dedicated class if style differs)
    location_html = ""
    if not is_value_empty(location_raw):
        location_html = f'<p class="headline" style="font-size: 0.95rem;">📍 {html.escape(location_raw)}</p>'
    elif show_empty_fields:
         if name_html: location_html = f'<p class="headline" style="font-size: 0.95rem;">📍 {render_missing_html()}</p>'

    # Experiences section (uses h5 and p.prewrap)
    experiences_html_content = ""
    if not is_value_empty(experiences_raw):
        # Use h5 for the header and p.prewrap for the content
        experiences_html_content = f'<h5><span class="coffee-icon"></span>Experiences</h5>\n<p class="prewrap">{html.escape(experiences_raw)}</p>'
    elif show_empty_fields:
        experiences_html_content = f'<h5><span class="coffee-icon"></span>Experiences</h5>\n{render_missing_html()}'
    # Wrap the content in a section div only if there is content
    experiences_section_html = f'<div class="section">{experiences_html_content}</div>' if experiences_html_content else ""


    # --- Pills sections (Uses h5 and render_pills_html which creates .pill spans) ---

    # Skills
    skills_pills_content = render_pills_html(
        skills_raw,
        label="Skills",
        show_label=True,  # Let the helper render the H5 now
        show_empty_fields=show_empty_fields,
        pill_class="pill" # Ensure correct class
    )
    # Wrap section content in a div if it's not empty
    skills_section_html = f'<div class="section">{skills_pills_content}</div>' if skills_pills_content else ""

    # Interests
    interests_pills_content = render_pills_html(
        interests_raw,
        label="Interests",
        show_label=True,
        show_empty_fields=show_empty_fields,
        pill_class="pill"
    )
    interests_section_html = f'<div class="section">{interests_pills_content}</div>' if interests_pills_content else ""

    # Hobbies
    hobbies_pills_content = render_pills_html(
        hobbies_raw,
        label="Hobbies",
        show_label=True,
        show_empty_fields=show_empty_fields,
        pill_class="pill"
    )
    hobbies_section_html = f'<div class="section">{hobbies_pills_content}</div>' if hobbies_pills_content else ""


    # --- Final Card HTML ---
    # Assemble using the defined structure and classes
    card_html = f"""
    <div class="card coffee-card-container"> 
        <div class="header-content">
            <img src="{avatar_url}" alt="Avatar" class="avatar">
            <div class="header-text">
                {name_html}
                {title_html}
                {location_html}
            </div>
        </div>
        {skills_section_html}
        {interests_section_html}
        {hobbies_section_html}
        {experiences_section_html}
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
                            if i < len(data) - 1:
                                list_content_html += "<hr style='border: none; border-top: 1px dashed var(--cc-accent-light-tan); margin: 0.8rem 0;'>"

                    if list_content_html:
                        if list_content_html.endswith("</hr>"):
                             list_content_html = list_content_html[:-len("<hr style='border: none; border-top: 1px dashed var(--cc-accent-light-tan); margin: 0.8rem 0;'>")]
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
# 3. Editing Form Renderer - Use Coffee Card Button Styles
# -------------------------------------
# No structural changes needed here, the CSS targets the buttons inside the form
def render_edit_form():
    """Renders a form to edit selected parts of the session state data."""
    if 'analysis_data' not in st.session_state or not st.session_state['analysis_data']:
        st.info("Load data first to enable editing.")
        return

    # st.markdown("---") # Removed extra separator
    st.subheader("✏️ Edit Profile Data")

    editable_data = copy.deepcopy(st.session_state['analysis_data'])

    # Allow editing top-level dicts and the Coffee Card specifically
    editable_sections = {}
    if isinstance(editable_data.get('yourCoffeeCard'), dict):
         editable_sections['yourCoffeeCard'] = "Your Coffee Card"

    # Add other top-level dictionary sections
    for key, value in editable_data.items():
        if isinstance(value, dict) and key != 'yourCoffeeCard': # Avoid duplicating coffee card
             # Add platform sections if they are dicts
             if key == 'platformSpecificAnalysis' and isinstance(value, dict):
                 for p_key, p_value in value.items():
                      if isinstance(p_value, dict): # Only add if it's a dictionary
                           # Create a unique key for editing state
                           edit_key = f"platform_{p_key}"
                           editable_sections[edit_key] = f"Platform: {get_platform_display_name(p_key, p_value)}"
             elif key != 'platformSpecificAnalysis': # Add other top-level dicts
                 editable_sections[key] = format_key_to_title(key)


    if not editable_sections:
        st.warning("No editable dictionary sections found in the loaded data.")
        return

    section_keys = list(editable_sections.keys()) # These are the keys used for selection/state
    selected_key_to_edit = st.selectbox(
        "Select section to edit:",
        options=section_keys,
        format_func=lambda key: editable_sections[key], # Show formatted name
        key="edit_section_selector"
    )

    if not selected_key_to_edit:
        return

    # --- Determine the actual data path based on selected_key_to_edit ---
    section_data = None
    path_to_update = []
    if selected_key_to_edit == 'yourCoffeeCard':
        path_to_update = ['yourCoffeeCard']
        section_data = safe_get(editable_data, path_to_update)
    elif selected_key_to_edit.startswith("platform_"):
        platform_key = selected_key_to_edit.split("platform_", 1)[1]
        path_to_update = ['platformSpecificAnalysis', platform_key]
        section_data = safe_get(editable_data, path_to_update)
    elif selected_key_to_edit in editable_data: # Other top-level dicts
        path_to_update = [selected_key_to_edit]
        section_data = safe_get(editable_data, path_to_update)

    if section_data is None or not isinstance(section_data, dict):
        st.warning(f"Selected section '{editable_sections[selected_key_to_edit]}' could not be accessed or is not an editable dictionary.")
        return

    with st.form(key=f"edit_form_{selected_key_to_edit}"):
        st.markdown(f"**Editing:** {editable_sections[selected_key_to_edit]}")

        new_section_data = {}

        for field_key, field_value in section_data.items():
            field_label = format_key_to_title(field_key)
            unique_widget_key = f"edit_{selected_key_to_edit}_{field_key}" # Ensure unique key

            # --- Input field rendering (logic unchanged) ---
            if isinstance(field_value, list):
                is_simple_str_list = all(isinstance(item, str) for item in field_value)
                if is_simple_str_list:
                     default_list_str = ", ".join(field_value)
                     new_value_str = st.text_area(
                         f"{field_label} (comma-separated)",
                         value=default_list_str,
                         key=unique_widget_key
                     )
                     new_section_data[field_key] = [item.strip() for item in new_value_str.split(',') if item.strip()]
                else:
                     st.markdown(f"**{field_label} (List):**")
                     try: st.json(field_value, expanded=False)
                     except Exception: st.text(str(field_value))
                     st.caption("(Editing complex lists directly not supported)")
                     new_section_data[field_key] = field_value # Keep original

            elif isinstance(field_value, str):
                 if len(field_value) > 100 or '\n' in field_value:
                     new_section_data[field_key] = st.text_area(
                         field_label, value=field_value, key=unique_widget_key, height=150
                     )
                 else:
                     new_section_data[field_key] = st.text_input(
                         field_label, value=field_value, key=unique_widget_key
                     )
            elif isinstance(field_value, (int, float)):
                 new_section_data[field_key] = st.number_input(
                     field_label, value=field_value, key=unique_widget_key
                 )
            elif isinstance(field_value, bool):
                  new_section_data[field_key] = st.checkbox(
                      field_label, value=field_value, key=unique_widget_key
                  )
            elif field_value is None:
                 new_value_text = st.text_input(
                     f"{field_label} (Currently None)", value="", key=unique_widget_key
                 )
                 # Only update if text is entered, otherwise keep as None
                 new_section_data[field_key] = new_value_text if new_value_text else None

            else: # Nested dicts, etc.
                 st.markdown(f"**{field_label} ({type(field_value).__name__}):**")
                 try: st.json(field_value, expanded=False)
                 except Exception: st.text(str(field_value))
                 st.caption("(Editing nested structures directly not supported)")
                 new_section_data[field_key] = field_value # Keep original value

        submitted = st.form_submit_button("Save Changes")
        if submitted:
            try:
                # Update the actual session state data using the path
                current_data_ref = st.session_state['analysis_data']
                for i, key in enumerate(path_to_update):
                    if i == len(path_to_update) - 1:
                        # Last key: perform the update
                        if key in current_data_ref:
                            current_data_ref[key] = new_section_data
                            st.success(f"Changes saved for '{editable_sections[selected_key_to_edit]}'. Rerunning...")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"Error: Key '{key}' not found at expected path.")
                            break # Stop update process
                    else:
                        # Navigate deeper
                        if key in current_data_ref and isinstance(current_data_ref[key], dict):
                             current_data_ref = current_data_ref[key]
                        else:
                             st.error(f"Error: Invalid path or non-dictionary found at key '{key}'.")
                             break # Stop update process
            except Exception as e:
                st.error(f"Error saving changes: {e}")


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
    "primaryProfileURL": "https://www.linkedin.com/in/Homenshum"
  },
  "analyzedPlatforms": [
    "LinkedIn",
    "Instagram"
  ],
  "yourCoffeeCard": {
    "name": "Homen Shum",
    "title": "\"Do something great with data and finance\" | Ex-JPM Startup Banking Associate | LLM/Data Engineer",
    "location": "Fremont, California, United States",
    "interests": [
      "Large Language Models (LLM)",
      "Generative AI",
      "Data Analysis",
      "Finance",
      "Healthcare Technology",
      "Software Development",
      "Startups",
      "AI Engineering"
    ],
    "hobbies": [
      "Memes",
      "Anime",
      "Gaming",
      "Short-form video content"
    ],
    "skills": [
      "Python (Programming Language)",
      "Pandas (Software)",
      "Microsoft Azure",
      "Google Cloud Platform (GCP)",
      "Large Language Models (LLM)",
      "Data Analysis",
      "Automation",
      "LLMOps",
      "Software Development",
      "AI Engineering",
      "Startup Development",
      "Financial Analysis"
    ],
    "experiences": "Founder at CafeCorner LLC (Dec 2022-Present), building sales recommendation and workflow automation applications. Formerly Startup Banking Associate at JPMorgan Chase & Co. (Jan 2021-Dec 2022), with rotations in Healthcare and Life Science Banking and experience as a Lead AI Engineer for an AWS DesignRacer. Previous experience includes roles in risk management, high-frequency trading challenges, and various assistant/advisor roles during university."
  },
  "platformSpecificAnalysis": {
    "linkedIn": {
      "platformName": "LinkedIn",
      "profileFundamentals": {
        "username": "Homenshum",
        "fullName": "Homen Shum",
        "pronouns": None,
        "location": "Fremont, California, United States",
        "profileURL": "https://www.linkedin.com/in/Homenshum",
        "profileLanguage": "English",
        "verificationStatus": None,
        "contactInfoVisible": True,
        "profilePictureDescription": "Professional headshot of a young man.",
        "bannerImageDescription": "Image of a cityscape.",
        "linkedWebsites": [
          "https://Homenshum.com/software/fluency-bled-1-filescore-hackathon",
          "https://www.youtube.com/watch?v=kqVNt8pJOQI",
          "https://patentyogi-chat.streamlit.app/",
          "https://cafecorner-lexica-sales-assistant.streamlit.app/"
        ]
      },
      "professionalHeadline": "\"Do something great with data and finance\" | Ex-JPM Startup Banking Associate | LLM/Data Engineer | Ex-Founder (AI FinAnlytix) | Ex-Healthcare Tech StartUp (Cofounder, CTO) | CafeCorner LLC.",
      "aboutSectionAnalysis": {
        "fullText": "Experience with LLM APIs and Generative AI Tools. I've been working with LLM APIs and generative AI tools since December 2022, progressing through increasingly complex implementations across multiple domains. 2022-2023: Foundation Projects. Oct 2023: Developed Celebrity Voice AI platform using DFL 2 platform, and AutoRPE, implementing course_similarity matching for unified speech pattern replication. Mar 2023: Achieved 2nd place in Nonecon Cybersecurity challenge with 96% accuracy in semantic matching and unified segmentation of multi-vendor security alerts. 2023: Healthcare AI Applications. Jun 2023: Led a team of 3 persons to train >25 placement at UC Berkeley AI hackathon with FinAnlytix, addressing OPTUM medical code matching with real time transcription and RAG implementation. Subsequently became technical co-founder of CerebroLeap Inc. 2023: Financial Sector Implementation. Oct 2023: Automated classification system for JPM Healthcare Banking team using GPT and LlamaIndex, implementing structured outputs to physician's instruction memory and OPT-3.5 turbo; Reduced processing time for 2,000+ companies from two weeks to under 30 seconds. 2024: Prescient Advanced Applications. Feb 2024: Designed real-time AI transcription application for JPM internal meetings and assumed LLM Application Pilot role with All In AI, Technology team. Mar 2016: Completed CourseHero's MVP building real-time transcription, OFD/CID matching, and physician portal. Deployed via Azure with HIPAA compliant security. Nov 2018-Present: Developing Patentyogi, integrating multi-agent architecture for comprehensive financial research, featuring cross-validation with web data and structured output processing for report generation. My Latest Web Applications: Medical Code Search & Report Generation: https://www.youtube.com/watch?v=kqVNt8pJOQI Patentyogi Screening Tool & Report Generation: https://patentyogi-chat.streamlit.app/ Auto-Stock Screener & Commercial RAG Chatbot Sales Assistant: https://cafecorner-lexica-sales-assistant.streamlit.app/ Here's to embracing growth, bridging gaps, and making technology more accessible - together.",
        "identifiedKeywords": [
          "LLM APIs",
          "Generative AI",
          "Celebrity Voice AI",
          "DFL 2",
          "AutoRPE",
          "Cybersecurity",
          "Semantic Matching",
          "Healthcare AI",
          "FinAnlytix",
          "OPTUM",
          "RAG implementation",
          "CerebroLeap Inc.",
          "Financial Sector",
          "JPM Healthcare Banking",
          "GPT",
          "LlamaIndex",
          "AI transcription",
          "CourseHero",
          "Azure",
          "HIPAA",
          "Patentyogi",
          "Multi-agent architecture",
          "Financial research"
        ],
        "hashtagsInBio": [],
        "mentionedUsersInBio": [],
        "statedPurposeFocus": "Demonstrating extensive experience and project work in LLMs, Generative AI, data engineering, and AI applications across finance, healthcare, and cybersecurity.",
        "tone": "Professional, Technical, Accomplishment-focused",
        "callToAction": None
      },
      "featuredContent": [
        "Post: The Limits of Public Data Flow (LLM Struggles) - Link to substack.",
        "Link: Fluency Bled #1 - Filescore Hackathon Link: https://Homenshum.com/software/fluency-bled-1-filescore-hackathon",
        "Image: Semantic Search Ranking Method with Dimension Reduction. Provided 96.17% high performance accuracy...",
        "Link: 5. Location Presentation with RAG"
      ],
      "experience": [
        {
          "titleOrDegree": "Founder",
          "organizationOrSchool": "CafeCorner LLC",
          "dates": "Dec 2022 - Present (1 yr 6 mos)",
          "location": "San Francisco Bay Area",
          "description": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS, and Vertex using Docker. Skills: Large Language Model Operations (LLMOps), Large Scale Data Analytics. Attached resume: Homen Shum - Resume - April 2023.pdf",
          "mediaOrLinks": ["Homen Shum - Resume - April 2023.pdf"]
        },
        {
          "titleOrDegree": "Startup Banking Associate",
          "organizationOrSchool": "JPMorgan Chase & Co.",
          "dates": "Jan 2021 - Dec 2022 (2 yrs)",
          "location": "San Francisco Bay Area, Hybrid",
          "description": "As a JPM Startup Banking Associate in the Bay Area, I treasure the opportunity to work with some of the best human capital here and some of the most ambitious people, to speak with daring founders and to... Skills: Data Analysis, Automation.",
          "mediaOrLinks": []
        },
        {
          "titleOrDegree": "Rotation 1: Healthcare and Life Science Banking Team",
          "organizationOrSchool": "JPMorgan Chase & Co.",
          "dates": "May 2021 - Feb 2022 (10 mos)",
          "location": "San Francisco Bay Area, Hybrid",
          "description": "Initiated collaborations between internal teams such as TCF R&D Team and ML Team to streamline client... Skills: Amazon Web Services (AWS), Cloud Computing.",
          "mediaOrLinks": []
        },
        {
          "titleOrDegree": "JPM AWS DesignRacer Palo Alto Team Lead, Lead AI Engineer",
          "organizationOrSchool": "JPMorgan Chase & Co.",
          "dates": "Oct 2021 - Oct 2021 (1 mo)",
          "location": "Palo Alto, California, United States, Hybrid",
          "description": "Innovated cloud functions and trained tuning hyperparameters to enhance AI model performance... Skills: Amazon Web Services (AWS), Cloud Computing.",
          "mediaOrLinks": []
        },
        {
          "titleOrDegree": "Risk Management Challenge",
          "organizationOrSchool": "EquitDiem",
          "dates": "Dec 2020 - Jan 2021 (2 mos)",
          "description": "Built risk management skills while managing the daily fluctuations to less than 1% +/- of a $100k portfolio...",
          "mediaOrLinks": []
        },
        {
          "titleOrDegree": "High Frequency Trading Challenge",
          "organizationOrSchool": None,
          "dates": "Nov 2020 - Dec 2020 (2 mos)",
          "description": "Generated 15% total profit within 2E market trading days by November 17th, 2020. Demonstrated proficient utilization of risk management strategy, understanding of moving averages, an...",
          "mediaOrLinks": []
        },
        {"titleOrDegree": "Asia Pacific Investment Challenge", "organizationOrSchool": None, "dates": "Sep 2020 - Nov 2020 (3 mos)", "description": "Researched global market capitalization with analyst views and consumer reports from financial news sources and online product reviews to determine consumer enthusiasm and company's grow...", "mediaOrLinks": []},
        {"titleOrDegree": "Internal Al Auditor", "organizationOrSchool": "Delta Sigma Pi - Rho Sigma Chapter", "dates": "May 2020 - Jul 2020 (3 mos)", "description": "Collaborated with on-campus organizations to plan professional networking events and discussion mak... see more", "mediaOrLinks": []},
        {"titleOrDegree": "Recruitment Assistant", "organizationOrSchool": None, "dates": "Mar 2018 - May 2020 (2 yrs 3 mos)", "description": "Planned, scripted, and facilitated a fraternal mentoring opening event with a committee. Booked rooms and requested funding by structuring the student body's...", "mediaOrLinks": []},
        {"titleOrDegree": "Professional Event Assistant", "organizationOrSchool": None, "dates": "Jan 2019 - Apr 2019 (4 mos)", "description": "Invited and work with $100,000 start-up organization to speak at the professional event held for the professionals at Delta Sigma Pi meeting.", "mediaOrLinks": []},
        {"titleOrDegree": "Academic Advisor", "organizationOrSchool": "Vietnamese Student Association - UC Santa Barbara", "dates": "Apr 2018 - Jun 2019 (1 yr 3 mos)", "description": "Supervised members with up to $2000 from AS ESP Grant, Emergency Grant, EIF Grant, Young Entrepreneur Scholarship...", "mediaOrLinks": []}
      ],
      "education": [
        {
          "titleOrDegree": "Financial Edge Training",
          "organizationOrSchool": "2021 Commercial Banking: Budget Training, Accounting and Finance",
          "dates": "Nov 2021",
          "location": None,
          "description": None,
          "mediaOrLinks": []
        },
        {
          "titleOrDegree": "Certificate, Business Administration and Management, General",
          "organizationOrSchool": "UC Santa Barbara",
          "dates": "2020 - 2021",
          "location": None,
          "description": "Cooperated with professors on personal project, recommended by 2 professors and a graduate stude... see more",
          "mediaOrLinks": []
        }
      ],
      "skillsEndorsements": [
        "Python (Programming Language)",
        "Pandas (Software)",
        "Microsoft Azure",
        "Google Cloud Platform (GCP)",
        "Large Language Models (LLM)",
        "Large Language Model Operations (LLMOps)",
        "Large-scale Data Analysis"
      ],
      "recommendationsGivenReceived": None,
      "accomplishments": "Projects section includes: Startup Search API (Oct 2021 - Nov 2021, Demo: https://hshum-gptsearch-server-4pvovgqz6a-uc.a.run.app/search?q=[your search query for the top 5 start-up names aims to your desired company search descriptions]..., Skills: Web Services API, Docker, +11 skills).",
      "interests": [
        "Sarah Johnston (Recruiter, LinkedIn Branding & Interview Coach...)",
        "Ray Dalio (Founder, CIO Mentor, and Member of the Bridgewater...)"
      ],
      "contentGenerationActivity": {
        "postingFrequency": "Sporadic (reposts observed recently).",
        "dominantContentTypes": ["Reposts of industry content (LLMs, AI)", "Links to personal projects/articles"],
        "contentExamples": [
          "Featured Post: The Limits of Public Data Flow (LLM Struggles) - Link to substack.",
          "Featured Link: Fluency Bled #1 - Filescore Hackathon",
          "Featured Image: Semantic Search Ranking Method with Dimension Reduction."
        ],
        "recurringThemesTopics": ["LLMs", "AI", "Data Engineering", "Software Development Projects"],
        "overallToneVoice": "Informative, Technical",
        "recentActivityExamples": [
          "Reposted LN Pandey's post about LLM systems.",
          "Reposted David Mlynd's post about Qdrant 1.9."
        ],
        "articlesPublishedCount": None
      },
      "engagementPatterns": {
        "outgoingInteractionStyle": "Reposts relevant industry content; likely engages with content related to AI, LLMs, and data engineering based on feed consumption.",
        "typesOfContentEngagedWith": ["AI/LLM developments", "Data engineering tools", "Technical articles", "Startup news"],
        "incomingEngagementHighlights": "Analytics show 1,727 post impressions and 105 search appearances, indicating visibility.",
        "typicalIncomingCommentTypes": ["Not directly observed, but comments on his own posts (if any) would likely be technical or project-related."]
      },
      "networkCommunityRecommendations": {
        "followerCount": 2856,
        "followingCount": None,
        "audienceDescription": "Primarily industry peers, individuals interested in AI/LLMs, data science, and software development.",
        "groupCommunityMemberships": [],
        "inboxSidebarPreview": None,
        "inboxSidebarAnalysis": "Not visible in provided data.",
        "myNetworkTabVisibility": None,
        "myNetworkTabAnalysis": "Not visible in provided data.",
        "platformSuggestions": {
          "suggestedPeople": [
            {
              "fullName": "Ramia Hussain",
              "headlineOrTitle": "Barnard College | Incoming Analyst at Goldman Sachs",
              "profileURL": None,
              "reasonForSuggestion": "From your school",
              "locationContext": "Sidebar - People you may know",
              "visibleMetrics": None
            },
            {
              "fullName": "Leah Varughese",
              "headlineOrTitle": "Digital Product Associate at JPMorgan Chase & Co.",
              "profileURL": None,
              "reasonForSuggestion": None,
              "locationContext": "Sidebar - People you may know",
              "visibleMetrics": None
            },
            {
              "fullName": "Goulios Joseph",
              "headlineOrTitle": "Economics Student @ UCSB | Membership Committe...",
              "profileURL": None,
              "reasonForSuggestion": None,
              "locationContext": "Sidebar - People you may know",
              "visibleMetrics": None
            },
            {
              "fullName": "Albert Qian",
              "headlineOrTitle": "MBA Candidate @ Columbia Business School | Data...",
              "profileURL": None,
              "reasonForSuggestion": None,
              "locationContext": "Sidebar - People you may know",
              "visibleMetrics": None
            },
            {
              "fullName": "T. Aqeel Taheraly",
              "headlineOrTitle": "Incoming analyst at Nomura | Senior at Trinity College | UM...",
              "profileURL": None,
              "reasonForSuggestion": None,
              "locationContext": "Sidebar - People you may know",
              "visibleMetrics": None
            }
          ],
          "suggestedCompaniesOrPages": [],
          "suggestedGroupsEvents": [],
          "suggestedContentOrTopics": [
            {
                "itemName": "Add case studies that showcase your skills",
                "itemType": "Other",
                "description": "Add projects to your profile.",
                "itemURLOrIdentifier": None,
                "reasonForSuggestion": "Suggested for you (profile enhancement)",
                "locationContext": "Profile page - Suggested for you section",
                "visibleMetrics": None
            }
          ],
          "peopleAlsoViewed": [
            {
              "fullName": "Sonya Chiang",
              "headlineOrTitle": "Product Designer @ Acquired.io | Product Designer (Digital an...",
              "profileURL": None,
              "reasonForSuggestion": "People also viewed",
              "locationContext": "Profile Sidebar - People also viewed",
              "visibleMetrics": None
            },
            {
              "fullName": "Anyshika Bajada",
              "headlineOrTitle": "Student at UC Berkeley",
              "profileURL": None,
              "reasonForSuggestion": "People also viewed",
              "locationContext": "Profile Sidebar - People also viewed",
              "visibleMetrics": None
            }
          ],
          "otherSuggestions": []
        },
        "platformRecommendationsAnalysis": "LinkedIn suggests connecting with individuals from his educational background (UC Santa Barbara, indicated by 'From your school' for Ramia Hussain) and potentially former colleagues or industry peers (Leah Varughese at JPMC, Albert Qian with data focus). The 'People Also Viewed' section suggests that individuals looking at Homen's profile are also interested in other tech/product-focused individuals, some also with UC Berkeley connections. The platform also nudges towards profile completion ('Add case studies'). This indicates LinkedIn views him as an active professional in tech/AI and aims to strengthen his network within this domain and educational alumni.",
        "detailedConnectionsList": None,
        "detailedConnectionsAnalysis": "Not provided."
      },
      "privacyPresentation": {
        "accountVisibility": "Public",
        "postLevelVisibility": "Public (for reposts and featured content)",
        "networkVisibility": "Connections count public (500+), follower count public.",
        "activitySharingSettings": "Shares reposts publicly.",
        "overallPresentationStyle": "Highly curated, professional, and focused on technical expertise and accomplishments."
      },
      "observedConsumption": {
        "mainFeed": {
          "observedTopics": ["Temporal Knowledge Graphs", "Personalized AI Agents", "AI Model Development", "Browser Extensions for Productivity", "OpenAI developments", "LLMs in media (Netflix)"],
          "frequentPostersAccounts": ["Maryam Mazaheri, PhD", "Milda Naciute", "Philomena Kyrosmou", "Snehanshu Ray", "Richard Huston", "Eduardo Ordax"],
          "contentFormatCharacteristics": "Professional posts, technical deep dives, infographics, code snippets, product demos, news about AI.",
          "specificContentExamples": [
            "Maryam Mazaheri's post on 'GraphMemory: Temporal Knowledge Graph for Personalized AI Agents' with infographic.",
            "Milda Naciute's post about an AR converter project with an image of code.",
            "Snehanshu Ray's video demo of 'Overlook' browser extension.",
            "Richard Huston's post about a call with Sam Altman."
          ]
        },
        "discoveryFeed": None,
        "consumptionAnalysisNotes": "The LinkedIn main feed consumption is highly aligned with Homen Shum's professional profile, focusing on advanced AI/LLM topics, data engineering, and software development tools and news. He consumes content from peers and experts in these fields. This reinforces his image as an active participant and learner in the AI/tech community."
      },
      "platformFeatureUsage": [
        {"featureName": "Featured Section", "usageDescription": "Highlights key projects, posts, and links.", "frequency": "Static until updated"},
        {"featureName": "Resume Attachment", "usageDescription": "Attached PDF resume to Founder experience.", "frequency": "Static"},
        {"featureName": "Skills Section", "usageDescription": "Lists numerous technical skills.", "frequency": "Static until updated"},
        {"featureName": "Reposting", "usageDescription": "Recently reposted content related to LLMs and data.", "frequency": "Sporadic"},
        {"featureName": "Creator Mode (implied)", "usageDescription": "Access to 'Resources', follower count prominently displayed, focus on content sharing.", "frequency": "Ongoing"}
      ],
      "platformSpecificConclusions": "Homen Shum's LinkedIn profile is a detailed and professional showcase of his expertise in LLMs, Generative AI, data engineering, and his entrepreneurial ventures. His 'About' section, 'Experience', and 'Featured' content all highlight significant projects and accomplishments. The platform's recommendations and his feed consumption data strongly align with his stated skills and interests, positioning him as a technical expert in the AI field. The profile is well-maintained and geared towards professional networking and showcasing capabilities."
    },
    "instagram": {
      "platformName": "Instagram",
      "profileFundamentals": {
        "username": "Homen.shum",
        "fullName": "Homen Shum",
        "pronouns": None,
        "location": None,
        "profileURL": "https://www.instagram.com/Homen.shum/",
        "profileLanguage": "English (UI)",
        "verificationStatus": None,
        "contactInfoVisible": None,
        "profilePictureDescription": "Same headshot as LinkedIn, professional.",
        "bannerImageDescription": None,
        "linkedWebsites": []
      },
      "bioAnalysis": {
        "fullText": "Not directly observed in the provided video footage (profile page not visited).",
        "identifiedKeywords": [],
        "hashtagsInBio": [],
        "mentionedUsersInBio": [],
        "statedPurposeFocus": None,
        "tone": None,
        "callToAction": None
      },
      "storyHighlights": [],
      "contentGenerationActivity": {
        "postingFrequency": "Low (only one comment observed).",
        "dominantContentTypes": ["Comments"],
        "contentExamples": [
          "Commented '🙌' on hkentertainment's post about a DJ KANG event."
        ],
        "recurringThemesTopics": ["Event appreciation (based on single comment)"],
        "overallToneVoice": "Positive, brief",
        "recentActivityExamples": [
          "Commented '🙌' on hkentertainment's post."
        ],
        "gridAesthetic": "Not observed.",
        "reelsPerformanceIndicators": "Not observed for own content.",
        "storiesFrequencyEngagement": "Not observed."
      },
      "engagementPatterns": {
        "outgoingInteractionStyle": "Likes posts (e.g., hkentertainment's event), comments briefly.",
        "typesOfContentEngagedWith": ["Events (DJ/Music)", "Motivational quotes (liked `bornreal` post)"],
        "incomingEngagementHighlights": "Not observed.",
        "typicalIncomingCommentTypes": ["Not observed."]
      },
      "networkCommunityRecommendations": {
        "followerCount": None,
        "followingCount": None,
        "audienceDescription": "Likely a mix of personal connections and accounts related to interests observed in Reels consumption (memes, anime, etc.).",
        "groupCommunityMemberships": [],
        "inboxSidebarPreview": None,
        "inboxSidebarAnalysis": "Not visible in provided data.",
        "myNetworkTabVisibility": None,
        "myNetworkTabAnalysis": "Not visible in provided data.",
        "platformSuggestions": {
          "suggestedPeople": [
            {"fullName": "lulukuchen", "headlineOrTitle": None, "profileURL": None, "reasonForSuggestion": None, "locationContext": "Main Feed Sidebar - Suggested for you", "visibleMetrics": None},
            {"fullName": "dimchryspyros", "headlineOrTitle": None, "profileURL": None, "reasonForSuggestion": None, "locationContext": "Main Feed Sidebar - Suggested for you", "visibleMetrics": None},
            {"fullName": "joannhtgs_", "headlineOrTitle": None, "profileURL": None, "reasonForSuggestion": None, "locationContext": "Main Feed Sidebar - Suggested for you", "visibleMetrics": None},
            {"fullName": "bestvines", "headlineOrTitle": None, "profileURL": None, "reasonForSuggestion": "Suggested for you", "locationContext": "Main Feed Sidebar - Suggested for you", "visibleMetrics": None},
            {"fullName": "billieeilishking_benedl", "headlineOrTitle": None, "profileURL": None, "reasonForSuggestion": "Suggested for you", "locationContext": "Main Feed Sidebar - Suggested for you", "visibleMetrics": None}
          ],
          "suggestedCompaniesOrPages": [],
          "suggestedGroupsEvents": [],
          "suggestedContentOrTopics": [],
          "peopleAlsoViewed": [],
          "otherSuggestions": []
        },
        "platformRecommendationsAnalysis": "Instagram's 'Suggested for you' list includes a mix of individual accounts and larger theme/meme accounts (like `bestvines`). This suggests the algorithm identifies Homen as someone interested in general popular content and potentially specific individuals based on network overlaps or activity not fully visible. The suggestions point towards a more entertainment-focused usage pattern, aligning with the Reels consumption.",
        "detailedConnectionsList": None,
        "detailedConnectionsAnalysis": "Not provided."
      },
      "privacyPresentation": {
        "accountVisibility": "Assumed Public (as feed content is visible).",
        "postLevelVisibility": "Not observed for own posts.",
        "networkVisibility": "Not observed.",
        "activitySharingSettings": "Likes and comments are visible to others.",
        "overallPresentationStyle": "Not enough data on own profile to determine; interaction suggests casual usage."
      },
      "observedConsumption": {
        "mainFeed": {
          "observedTopics": ["Local Events (DJ/Music)", "Motivational Quotes"],
          "frequentPostersAccounts": ["hkentertainment", "bornreal"],
          "contentFormatCharacteristics": "Event posters, image macros with text.",
          "specificContentExamples": [
            "DJ KANG event poster by hkentertainment.",
            "Motivational quote post by bornreal: \"If you're willing to suck at anything for 100 days...\""
          ]
        },
        "discoveryFeed": {
          "observedThemes": ["Memes", "Anime clips/references", "Gaming clips/memes (Jujutsu Kaisen, Call of Duty)", "Short comedy skits", "Relationship humor/advice (often with sexual innuendo)", "Interesting facts/science (parasitic wasp in amber)", "Food/Cooking", "Satisfying videos (crafting, food prep)", "Cultural observations (East Asian content, Chinese language skits)", "Fitness", "Animal videos (cat)"],
          "prevalentContentTypes": ["Short-form videos (Reels)", "Animated clips", "Screen recordings (gaming, text messages)", "User-generated comedy", "Text overlays on videos"],
          "commonSoundsEffectsTrends": ["Varied, typical of trending Reels audio, often with humorous or dramatic voiceovers or music accompanying visuals."],
          "highlyDescriptiveExamples": [
            "Animated character in pink puddle, 'Humanity this simply not possible'.",
            "Person holding a geoduck-like creature.",
            "Woman on beach, 'MALE GAZE'.",
            "Insect in amber, '3 Million-Year-Old Parasitic Wasp...'.",
            "Woman crying, 'GHETTO STONESTOWN?'.",
            "Man looking at Dr. Strange portal.",
            "Asian woman surprised, 'when he tells me stop stroken his d!ck after he finishes'.",
            "Demon Slayer animated fight scene.",
            "Chinese skit about newlywed's room.",
            "Mobile game 'Peak disrespect'.",
            "Anime 'HE KISSED THE PRINCESS ON HER LIPS...'.",
            "Hand revealing pearl in sand.",
            "Man showing abs with Chinese fitness text.",
            "Sukuna 'BAN RATE 1.97%'.",
            "Woman talking, 'The F_ck First Rule'.",
            "Man receiving flowers, 'You got catfished from temu'.",
            "Sad cat, Chinese text '男生恋爱前 男生恋爱后'.",
            "Food 'What the Puff?!'.",
            "Large flat white fish, 'This is the...'.",
            "Thor 'Can Thor fly without his hammer?'.",
            "Cucumber with scared face, 'Wait this isn't a sablen...'.",
            "Man hijacking plane in MW2 game footage.",
            "Nose shapes 'Unselfish' vs 'Strong Minded'.",
            "Text message 'MORNING BABY... FAVORITE PERSON'.",
            "Craft video of cutting wood/leather.",
            "'BEST NOODLES AT COSTCO'."
          ],
          "overallFeedCharacteristics": "Highly diverse, fast-paced, driven by humor, relatability, current trends, and visually engaging short clips. Strong presence of anime, gaming, and East Asian cultural content alongside general internet humor."
        },
        "consumptionAnalysisNotes": "Instagram consumption shows a stark contrast to LinkedIn. The main feed shows some local/social interests (events, motivational content). The Reels (discovery) feed is dominated by entertainment: memes, anime, gaming, humorous skits, and viral short-form videos. This indicates a clear separation between professional (LinkedIn) and personal/entertainment (Instagram) online activity. The Instagram algorithm is feeding a wide array of general interest and niche hobby (anime, gaming, specific types of humor) content.",
        "platformFeatureUsage": [
          {"featureName": "Reels", "usageDescription": "Extensive consumption of Reels in the discovery feed.", "frequency": "High (during observation period)"},
          {"featureName": "Liking Posts", "usageDescription": "Liked posts on main feed.", "frequency": "Observed"},
          {"featureName": "Commenting", "usageDescription": "Made a brief comment on a post.", "frequency": "Observed (low)"}
        ],
        "platformSpecificConclusions": "Homen Shum's Instagram presence, as observed, is primarily for content consumption and light engagement, heavily skewed towards entertainment, memes, anime, and gaming, particularly through the Reels discovery feed. There's minimal evidence of content creation beyond a single comment. The platform suggestions and observed consumption patterns paint a picture of a user engaging with popular internet culture and specific hobbyist content, distinct from his professional LinkedIn persona."
      }
    },
    "crossPlatformSynthesis": {
      "consistencyVsVariation": {
        "profileElementConsistency": "The profile picture is consistent (professional headshot). The username `Homen.shum` is used. Full name is assumed consistent.",
        "contentTonePersonaConsistency": "Significant variation. LinkedIn is strictly professional, technical, and focused on career accomplishments and industry news. Instagram activity observed is casual, entertainment-focused, and related to personal interests/hobbies like anime, gaming, and memes. The tone on LinkedIn is formal and informative; on Instagram (based on Reels consumption and one comment) it's aligned with general internet culture.",
        "notableDifferences": "The primary difference is the persona projected: LinkedIn is the curated professional, while Instagram (consumption) reflects a more relaxed individual with common entertainment interests. Content generation is high on LinkedIn (profile building, featured content, reposts), very low on Instagram (one comment). Feed consumption is also starkly different: LinkedIn for professional development/news, Instagram for entertainment."
      },
      "contentOverlapStrategy": "No direct content cross-posting was observed. The platforms serve entirely different purposes for this user. LinkedIn is for professional branding and networking; Instagram is for personal entertainment consumption.",
      "synthesizedExpertiseInterests": {
        "coreProfessionalSkills": [
          "Large Language Models (LLM)",
          "Generative AI",
          "Python",
          "Pandas",
          "Microsoft Azure",
          "Google Cloud Platform (GCP)",
          "Data Analysis",
          "Automation",
          "Software Development",
          "AI Engineering",
          "LLMOps",
          "Startup Development",
          "Financial Analysis",
          "Project Management",
          "Cybersecurity (Semantic Matching)"
        ],
        "corePersonalInterests": [
          "Anime",
          "Gaming",
          "Memes/Internet Humor",
          "Short-form Video Content",
          "Technology (general interest)",
          "Food",
          "Music/Events (DJ)"
        ]
      },
      "overallOnlinePersonaNarrative": "Homen Shum projects a dual online persona. Professionally (LinkedIn), he is a highly skilled and accomplished AI engineer and entrepreneur with deep expertise in LLMs, data science, and building AI applications, particularly in finance and healthcare. His LinkedIn is a testament to his technical abilities and career progression. Personally (inferred from Instagram consumption), he engages with popular culture, including anime, gaming, and memes, suggesting a typical range of hobbies for someone in his demographic. There's a clear delineation between his professional showcase and his private entertainment consumption.",
      "professionalEvaluation": {
        "strengthsSkillsMatch": "Strong match between listed skills (Python, LLMs, Azure, GCP, data analysis) and detailed project/experience descriptions on LinkedIn. Demonstrates practical application of these skills in various roles and personal projects.",
        "impactAchievements": "Significant achievements include founding CafeCorner LLC, co-founding CerebroLeap Inc., placing in a cybersecurity challenge, leading AI projects at JPMC, and developing multiple web applications/tools showcased. Quantifiable results mentioned (e.g., reducing processing time for JPM project).",
        "industryEngagement": "Engages with industry content on LinkedIn by reposting relevant articles and consuming technical posts. His projects (Patentyogi, Fluency Bled) and 'About' section show a proactive approach to applying AI in various sectors. Connections and suggestions point to a network within tech and finance.",
        "potentialRedFlagsClarifications": "No major red flags. The resume attached is dated April 2023, so more recent accomplishments might not be on that specific document but are likely updated on the profile itself.",
        "overallCandidateSummary": "A highly capable and driven AI/Data professional with a strong entrepreneurial spirit and a proven track record of developing and deploying AI solutions. His LinkedIn profile effectively communicates his technical depth and breadth of experience across multiple domains."
      },
      "marketTrendInsights": {
        "keyTechnologiesToolsTopics": ["Large Language Models (LLMs)", "Generative AI", "RAG (Retrieval Augmented Generation)", "Vector Databases (Qdrant mentioned in repost)", "Cloud Platforms (Azure, GCP, AWS)", "Python", "Pandas", "Streamlit", "Docker", "AI in Finance", "AI in Healthcare", "Cybersecurity AI"],
        "emergingThemesNiches": ["Personalized AI Agents", "Temporal Knowledge Graphs", "AI for code generation/assistance", "Multi-agent AI systems", "Democratization of AI tools"],
        "relevantContentPatterns": "Focus on practical applications, project demos, technical deep-dives, and discussions around the capabilities and limitations of current AI models. Sharing of open-source tools and insights from industry leaders."
      },
      "inferredAlgorithmicPerception": [
        {
          "platformName": "LinkedIn",
          "categorizationHypothesis": "LinkedIn's algorithm perceives Homen Shum as an active, high-value professional in the AI/ML, data engineering, and software development space, likely with an interest in entrepreneurship and finance. This is based on his detailed profile, skills, experience in these areas, the technical nature of his 'About' section and featured projects, his consumption of AI-related content (e.g., Maryam Mazaheri's post on GraphMemory), and the platform suggestions aiming to connect him with others in tech/finance and from his alma mater (e.g., Ramia Hussain 'From your school', Leah Varughese from JPMC). The suggestion to 'Add case studies' reinforces this professional focus."
        },
        {
          "platformName": "Instagram",
          "categorizationHypothesis": "Instagram's algorithm likely categorizes Homen Shum as a general consumer interested in a broad range of popular entertainment content, with specific leanings towards anime, gaming, memes, and East Asian cultural content. This is strongly indicated by the highly diverse Reels discovery feed (e.g., anime clips like Demon Slayer/Sukuna, gaming content like MW2, Chinese skits, general internet humor) and suggestions for accounts like 'bestvines'. His minimal content creation (one comment on an event post) and likes on general motivational content suggest a passive consumer profile focused on entertainment rather than niche creation or influence on this platform."
        }
      ],
      "crossPlatformNetworkAnalysis": {
        "overlappingConnectionsRecommendations": [],
        "networkComparisonNotes": "LinkedIn recommendations are heavily career and education-focused (e.g., people from his university or past company, industry peers). Instagram suggestions are for general entertainment accounts and individuals without clear professional overlap. This highlights the different ways the platforms are used and how their algorithms profile the user based on distinct interaction patterns.",
        "consumptionComparisonNotes": "Vast difference. LinkedIn consumption is for professional development and staying updated on AI/tech trends (e.g., posts on temporal knowledge graphs, AI tools). Instagram Reels consumption is purely for entertainment, covering a wide array of memes, anime, gaming, and short humorous videos. The main feed on Instagram shows some interest in local events and general motivational content, still distinct from LinkedIn's professional focus."
      }
    },
    "finalComprehensiveSummary": "Homen Shum maintains a distinct dual online presence. His LinkedIn profile meticulously crafts an image of a highly skilled AI engineer, data scientist, and entrepreneur, showcasing extensive technical projects, a strong academic background, and active engagement with professional AI/tech content. Platform suggestions and his LinkedIn feed consumption reinforce this professional persona, highlighting his expertise in LLMs, generative AI, and data-driven solutions across finance and healthcare. In contrast, his observed Instagram activity, particularly the Reels discovery feed, reveals a consumer of mainstream and niche entertainment, including anime, gaming, memes, and diverse short-form video content. Instagram's suggestions align with this entertainment-focused usage. This clear separation indicates a deliberate use of LinkedIn for professional branding and networking, and Instagram for personal leisure and entertainment, with algorithms on each platform tailoring content and suggestions accordingly."
  }
}
def load_example_data():
    """Loads the hardcoded example JSON data."""
    return copy.deepcopy(EXAMPLE_ANALYSIS_DATA)

# -------------------------------------
# Main Execution Block - WITH TABS (Theme applied via CSS)
# -------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="CaféCorner", page_icon="☕", layout="wide")

    # --- Initialize Session State ---
    if 'analysis_data' not in st.session_state:
        st.session_state['analysis_data'] = None
    if 'uploaded_avatar_file' not in st.session_state:
        st.session_state['uploaded_avatar_file'] = None
    if 'processed_upload_filename' not in st.session_state:
         st.session_state['processed_upload_filename'] = None
    if 'current_data_source' not in st.session_state:
         st.session_state['current_data_source'] = "None"
    if 'show_empty_fields' not in st.session_state:
         st.session_state['show_empty_fields'] = False

    # Apply CSS globally
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

    render_custom_header()

    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ Controls")
        st.divider()

        st.subheader("📂 Load Data")
        uploaded_analysis_file = st.file_uploader(
            "Upload Profile JSON/TXT", type=["txt", "json"], key="analysis_file_uploader"
        )
        uploaded_avatar_file_sidebar = st.file_uploader(
            "Upload Profile Photo (Optional)", type=["png", "jpg", "jpeg", "gif", "webp"], key="avatar_uploader_sidebar"
        )
        # Removed video uploader as it wasn't used
        # uploaded_video_file = st.file_uploader(
        #     "Upload Video (Optional)", type=["mp4", "mov", "avi", "mkv", "webm"], key="video_uploader"
        #  )

        # Process uploaded JSON/TXT file
        if uploaded_analysis_file is not None:
            if uploaded_analysis_file.name != st.session_state.get('processed_upload_filename'):
                try:
                    file_content = uploaded_analysis_file.read().decode("utf-8")
                    parsed_data = json.loads(file_content)
                    if isinstance(parsed_data, dict) and \
                       isinstance(parsed_data.get("platformSpecificAnalysis"), (dict, type(None))):
                        st.session_state['analysis_data'] = parsed_data
                        st.session_state['uploaded_avatar_file'] = uploaded_avatar_file_sidebar
                        st.session_state['processed_upload_filename'] = uploaded_analysis_file.name
                        st.session_state['current_data_source'] = f"File: {uploaded_analysis_file.name}"
                        st.success(f"Loaded '{uploaded_analysis_file.name}'.")
                        st.rerun()
                    elif not isinstance(parsed_data.get("platformSpecificAnalysis"), (dict, type(None))):
                         st.error("Invalid format: 'platformSpecificAnalysis' must be an object (dictionary) in the JSON.")
                         st.session_state.update({'processed_upload_filename': None, 'analysis_data': None, 'uploaded_avatar_file': None})
                    else:
                        st.error("Uploaded file does not contain a valid JSON object.")
                        st.session_state.update({'processed_upload_filename': None, 'analysis_data': None, 'uploaded_avatar_file': None})
                except json.JSONDecodeError:
                    st.error(f"Invalid JSON in '{uploaded_analysis_file.name}'.")
                    st.session_state.update({'processed_upload_filename': None, 'analysis_data': None, 'uploaded_avatar_file': None})
                except Exception as e:
                    st.error(f"Error reading file '{uploaded_analysis_file.name}': {e}")
                    st.session_state.update({'processed_upload_filename': None, 'analysis_data': None, 'uploaded_avatar_file': None})

        # Update avatar if only avatar is uploaded/changed
        elif uploaded_avatar_file_sidebar is not None and st.session_state.get('analysis_data') is not None:
             current_avatar_obj = st.session_state.get('uploaded_avatar_file')
             if (current_avatar_obj is None or
                 uploaded_avatar_file_sidebar.name != current_avatar_obj.name or
                 uploaded_avatar_file_sidebar.size != current_avatar_obj.size):
                  st.session_state['uploaded_avatar_file'] = uploaded_avatar_file_sidebar
                  st.info("Avatar updated.")
                  st.rerun()

        if st.button("Load Example Data", key="load_example"):
             example_data = load_example_data()
             if isinstance(example_data, dict):
                 st.session_state['analysis_data'] = example_data
                 st.session_state['uploaded_avatar_file'] = None
                 st.session_state['processed_upload_filename'] = "EXAMPLE_DATA"
                 st.session_state['current_data_source'] = "Example Data"
                 st.info("Example data loaded.")
                 st.rerun()
             else:
                 st.error("Failed to load example data.")

        # Display Options
        st.subheader("👓 Display Options")
        st.session_state['show_empty_fields'] = st.checkbox(
            "Show Missing/Empty Fields",
            value=st.session_state.get('show_empty_fields', False),
            key='show_empty_toggle'
        )
        # Rerun happens automatically on checkbox change

        st.divider()
        st.caption(f"Current Source: {st.session_state['current_data_source']}")

        # Edit Form Expander
        if st.session_state.get('analysis_data'):
             with st.expander("✏️ Edit Data Sections", expanded=False):
                  render_edit_form()

    # --- Main Page Content Area ---
    # Removed redundant title, using custom header now
    # st.title("👤 CaféCorner Profile")

    if st.session_state.get('analysis_data'):
        data = st.session_state['analysis_data']
        avatar_file_obj = st.session_state.get('uploaded_avatar_file')
        show_empty = st.session_state.get('show_empty_fields', False)

        # --- 1. Render Specialized Coffee Card ---
        st.markdown("### ☕ Your Coffee Card") # Use markdown header
        coffee_card_data = data.get('yourCoffeeCard')
        # Determine name for avatar generation (fallback logic)
        profile_name_for_avatar = safe_get(coffee_card_data, ['name']) or \
                                  safe_get(data, ['targetIndividual', 'nameOrIdentifier'], "??")
        avatar_url = get_avatar_url(profile_name_for_avatar, avatar_file_obj)
        # Render the card HTML
        st.markdown(render_coffee_card_html(coffee_card_data, avatar_url, show_empty_fields=show_empty), unsafe_allow_html=True)

        # --- 2. Missing Fields Summary Callout ---
        if show_empty:
            missing_data_structured = find_missing_structured(data)
            if missing_data_structured:
                summary_html = format_missing_summary_html(missing_data_structured, max_items_per_section=7)
                # Use st.warning for visibility, icon="⚠️"
                # The HTML structure is designed to be styled by the CSS rules targeting .stWarning elements
                st.warning(summary_html, icon="☕") # Using coffee icon for theme
            else:
                # Use st.success if no missing fields found when show_empty is true
                st.success("✅ All fields appear complete.", icon="👍")

        st.markdown("---") # Separator

        # --- 3. Prepare and Render Tabs ---
        st.markdown("### 📊 Detailed Analysis") # Use markdown header

        overview_keys_to_render = [
            "targetIndividual", "analyzedPlatforms", "crossPlatformSynthesis", "finalComprehensiveSummary"
        ]
        overview_data_subset = {}
        for k in overview_keys_to_render:
            if k in data:
                value = data[k]
                if not is_value_empty(value) or show_empty:
                     overview_data_subset[k] = value

        platform_analysis_obj = data.get('platformSpecificAnalysis', {})
        platform_tabs_available = isinstance(platform_analysis_obj, dict) and platform_analysis_obj

        tab_titles = ["Overview"]
        platform_keys_in_order = []
        other_platforms_data = None

        if platform_tabs_available:
            # Separate 'otherPlatforms' first
            other_platforms_data_raw = platform_analysis_obj.pop('otherPlatforms', None)
            if isinstance(other_platforms_data_raw, list) and other_platforms_data_raw:
                # Check if any item has content or if we're showing empty fields
                should_show_other_tab = any(not is_value_empty(item) for item in other_platforms_data_raw) or show_empty
                if should_show_other_tab:
                    other_platforms_data = other_platforms_data_raw # Store for rendering
                    tab_titles.append("Other")
                    platform_keys_in_order.append("otherPlatforms") # Use special key


            # Process remaining regular platforms
            sorted_platform_keys = sorted(platform_analysis_obj.keys()) # Sort alphabetically
            for platform_key in sorted_platform_keys:
                platform_data = platform_analysis_obj.get(platform_key)
                if isinstance(platform_data, dict): # Ensure it's a dictionary
                    should_show_tab = not is_value_empty(platform_data) or show_empty
                    if should_show_tab:
                        display_name = get_platform_display_name(platform_key, platform_data)
                        tab_titles.append(display_name)
                        platform_keys_in_order.append(platform_key)

        # Create tabs
        if not overview_data_subset and len(tab_titles) == 1:
             # Case: Only overview potentially, but it's empty and we're hiding
              st.warning("No data available to display with current settings.")
        else:
            try:
                tabs = st.tabs(tab_titles)

                # --- Render Overview Tab ---
                with tabs[0]:
                    # st.subheader("Synthesis & Summary") # Use h3 from renderer
                    overview_html = render_nested_json_html(overview_data_subset, level=0, show_empty_fields=show_empty)
                    if overview_html:
                        st.markdown(overview_html, unsafe_allow_html=True)
                    else:
                        st.info("No overview data to display.")

                # --- Render Platform Tabs ---
                tab_index_offset = 1
                for i, platform_key in enumerate(platform_keys_in_order):
                    current_tab_index = i + tab_index_offset
                    if current_tab_index < len(tabs): # Ensure tab index is valid
                        with tabs[current_tab_index]:
                            if platform_key == "otherPlatforms":
                                # st.subheader("Other Platform Analyses") # Use h3 from renderer
                                rendered_any_other = False
                                if other_platforms_data:
                                    for idx, other_platform_item in enumerate(other_platforms_data):
                                         # Pass show_empty flag to renderer for each item
                                         item_html = render_nested_json_html(other_platform_item, level=1, show_empty_fields=show_empty)
                                         if item_html: # Only render if content exists or showing empty placeholders
                                             rendered_any_other = True
                                             # Item name handled by renderer's H4 now
                                             st.markdown(item_html, unsafe_allow_html=True)
                                             if idx < len(other_platforms_data) - 1: st.markdown("---") # Separator between items
                                if not rendered_any_other:
                                    st.info("No data found or displayed for 'Other Platforms' items with current settings.")
                            else:
                                 # Get data for regular platform tab
                                 platform_data_to_render = platform_analysis_obj.get(platform_key)
                                 if platform_data_to_render: # Should exist if key is in list
                                     # Pass show_empty flag to renderer
                                     platform_html = render_nested_json_html(platform_data_to_render, level=0, show_empty_fields=show_empty)
                                     if platform_html: # Only render if content exists
                                          # Platform title handled by renderer's H3 now
                                          st.markdown(platform_html, unsafe_allow_html=True)
                                     else:
                                          # This case implies the platform data was empty AND show_empty=False
                                          st.info(f"No data to display for {get_platform_display_name(platform_key, platform_data_to_render)} with current settings.")
                                 else:
                                      st.error(f"Data inconsistency: Platform key '{platform_key}' found but data missing.")
                    else:
                        st.error(f"Tab index mismatch error for platform key: {platform_key}")

            except Exception as e:
                 st.error(f"Error creating or rendering tabs: {e}. Displaying overview only.")
                 # Fallback to just overview if tabs fail
                 overview_html = render_nested_json_html(overview_data_subset, level=0, show_empty_fields=show_empty)
                 st.markdown(overview_html if overview_html else "<p>No overview data.</p>", unsafe_allow_html=True)

    else:
        # Display instructions if no data is loaded
        st.markdown("### 👋 Welcome!") # Markdown header
        st.info("⬅️ Use the sidebar to load profile data from a JSON/TXT file or view the example.")
        st.markdown("""
        This application displays profile information using the **CaféCorner** theme:
        1.  A specialized **Coffee Card** view summarizing key details.
        2.  An **Overview** tab for cross-platform synthesis.
        3.  **Platform-specific tabs** showing detailed analysis.

        Use the **Display Options** in the sidebar to toggle visibility of missing fields and the **Edit Data Sections** expander to modify loaded data.
        """)