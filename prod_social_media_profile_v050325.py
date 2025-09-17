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
# Styling (Using the provided CSS) - NO CHANGES
# -------------------------------------
CSS_STYLES = """
<style>
:root {
    --coffee: #4E342E;
    --crema: #F8F4E6;
    --brass: #CBA35B; /* Default accent */
    --neutral-bg: #FAF9F7;
    --neutral-pill: #F2EFEA;
    --accent: #0B66C2; /* LinkedIn Blue - Secondary Accent */
    --alert-border: #D43F3A;
    --alert-bg: #FFF5F5;
    --border-light: #E5E0DB;
    --text-primary: #222;
    --text-secondary: #555;
    --text-missing: #777;
    --white: #fff;

    /* Platform specific variations */
    --brass-linkedin: #0A66C2;
    --brass-instagram: #E1306C;
    --brass-twitter-x: #1DA1F2; /* Use for Twitter/X */
    --brass-facebook: #1877F2;
    --brass-tiktok: #000000; /* Black */
    --brass-reddit: #FF4500; /* Orange */
    --brass-github: #333;
    --brass-other: #777777; /* Grey for 'Other' */
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
    color: var(--text-primary);
    background-color: #FDFDFB;
}

/* --- Coffee Card Header --- */
.card {
    border: 1px solid var(--border-light);
    border-radius: 12px;
    background: var(--white);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.08);
}

.coffee-card-container { /* Specific container for the specialized coffee card */
    /* Inherits .card styling */
}

.coffee-card-header {
    padding-top: 2.5rem; /* Space for progress bar if added */
}

/* Optional Progress Bar styling if you want to reuse it */
.progress-bar-container {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 10px;
    background-color: var(--border-light);
    border-radius: 12px 12px 0 0;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    width: 85%; /* Example Width */
    background: linear-gradient(to right, var(--crema), #d1c5a8);
    border-radius: 12px 0 0 0;
}

.header-content {
    display: flex;
    align-items: flex-start;
    gap: 1.5rem;
}

img.avatar {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    border: 2px solid var(--border-light);
    object-fit: cover; /* Ensures image covers the area */
    flex-shrink: 0;
}

.header-text h1 {
    color: var(--coffee);
    margin: 0 0 0.25rem 0;
    font-size: 1.8rem;
    font-weight: 600;
}


.header-text p.headline { /* Reusing for title/location in coffee card */
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin: 0.25rem 0 0.75rem 0;
    line-height: 1.4;
}

/* --- Pills --- */
.pill {
    display: inline-block;
    padding: 3px 12px;
    margin: 3px 5px 3px 0;
    border-radius: 15px;
    background: var(--neutral-pill);
    font-size: 0.8rem;
    color: var(--coffee);
    font-weight: 500;
    line-height: 1.4;
    transition: filter 0.2s ease;
    white-space: nowrap;
}

.pill:hover {
    filter: brightness(95%);
}

.pill-container {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

/* --- Section Containers & Platform Cards (General Styling) --- */
/* Adjusted: Platform card style applied within the recursive renderer */
.platform-card-like { /* Renamed to avoid conflict if used elsewhere */
    background: var(--neutral-bg);
    border: 1px solid var(--border-light);
    border-left: 5px solid var(--brass); /* Default accent */
    border-radius: 0 8px 8px 0;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Platform specific accents (can be applied dynamically if needed) */
.platform-card-like.linkedin { border-left-color: var(--brass-linkedin); }
.platform-card-like.instagram { border-left-color: var(--brass-instagram); }
.platform-card-like.twitter-x { border-left-color: var(--brass-twitter-x); }
.platform-card-like.facebook { border-left-color: var(--brass-facebook); }
.platform-card-like.tiktok { border-left-color: var(--brass-tiktok); }
.platform-card-like.reddit { border-left-color: var(--brass-reddit); }
.platform-card-like.other { border-left-color: var(--brass-other); }


.section { /* General section separator */
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border-light);
}
.section:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}

/* --- Typography (General) --- */
h3 { /* Top-level keys in general viewer / Section titles */
    color: var(--coffee);
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    font-size: 1.4rem;
    font-weight: 600;
    border-bottom: 1px solid var(--border-light);
    padding-bottom: 0.5rem;
}

h4 { /* Nested keys in general viewer / Subsections */
    color: var(--coffee);
    margin-top: 1rem;
    margin-bottom: 0.75rem;
    font-size: 1.1rem;
    font-weight: 600;
}
.platform-card-like h4 {
    margin-top: 0;
}

h5 { /* Deeper nested keys / Labels */
    color: var(--coffee);
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
}

h6 { /* Even deeper or item indicators */
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 0.9rem;
    margin-bottom: 0.3rem;
    margin-top: 0.8rem;
}


p, ul, ol {
    color: var(--text-primary);
    line-height: 1.6;
    margin-bottom: 1rem;
}
ul, ol {
    padding-left: 1.5rem;
}
li {
    margin-bottom: 0.5rem;
}

a {
    color: var(--accent);
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}

/* --- Missing Data --- */
em.missing {
    color: var(--text-missing);
    font-style: italic;
    font-weight: 400; /* Normal weight */
}

/* Styling for pre-wrap paragraph */
p.prewrap {
    white-space: pre-wrap;
    margin-bottom: 0.5rem; /* Adjust spacing */
}

/* Simple list styling */
ul.simple-list {
    list-style-type: disc;
    padding-left: 20px;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}
ul.simple-list li {
    margin-bottom: 0.3rem;
    font-size: 0.95rem;
}

/* Container for recursively rendered items */
.nested-item-container {
    margin-left: 1rem; /* Indent nested items */
    padding-left: 1rem;
    border-left: 2px solid var(--border-light);
    margin-bottom: 1rem;
    margin-top: 0.5rem; /* Space above nested block */
}
.nested-item-container:last-child {
     margin-bottom: 0;
}

/* --- Edit Form Specific Styling --- */
.edit-form-container {
    background-color: var(--neutral-bg);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 2rem;
}

/* --- Tab Navigation (Targeting Streamlit Defaults) --- */
div[data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 2px solid var(--border-light);
    padding-bottom: 0;
    margin-bottom: 1.5rem;
}

button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px;
    padding: 0.75rem 1rem !important;
    font-size: 1rem;
    font-weight: 500;
    transition: color 0.2s ease, border-color 0.2s ease;
    border-radius: 6px 6px 0 0 !important;
}

button[data-baseweb="tab"]:hover {
    color: var(--coffee) !important;
    background-color: var(--neutral-pill) !important;
}

button[aria-selected="true"] {
    color: var(--coffee) !important;
    font-weight: 600 !important;
    border-bottom-color: var(--brass) !important;
}

</style>
"""

# -------------------------------------
# Helper Functions - Minor adjustments
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
    return '<em class="missing">Not Provided</em>'

def render_pills_html(items, label=None, show_label=True, pill_class="pill"):
    """Returns an HTML string of pills. Returns empty string if no valid items."""
    if not items or not isinstance(items, list):
        return ""
    items_str = [html.escape(str(item).strip()) for item in items if item and str(item).strip()]
    if not items_str:
        return ""
    pills_html_content = "".join(f'<span class="{pill_class}">{item}</span>' for item in items_str)
    label_html = f'<h5>{html.escape(label)}</h5>' if label and show_label else ''
    return f'{label_html}<div class="pill-container">{pills_html_content}</div>'

# --- Avatar Helper - NO CHANGES ---
def make_initials_svg_avatar(name: str, size: int = 80,
                            bg: str = "#4E342E", fg: str = "#F8F4E6") -> str:
    """Generates a Base64 encoded SVG avatar with initials."""
    display_name = name if name and name != render_missing_html() else "?"
    if not isinstance(display_name, str):
        display_name = str(display_name)
    initials = "".join([w[0].upper() for w in display_name.split()][:2]) or "?"
    svg = f'''
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">
<circle cx="{size/2}" cy="{size/2}" r="{size/2}" fill="{bg}"/>
<text x="50%" y="50%" fill="{fg}" font-size="{int(size/2)}"
        text-anchor="middle" dominant-baseline="central"
        font-family="sans-serif">{initials}</text>
</svg>'''
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"

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
    return make_initials_svg_avatar(name if name else "??", size=80)

# -------------------------------------
# 1. Specialized Coffee Card Renderer - Returns HTML String
# -------------------------------------
def render_coffee_card_html(card_data, avatar_url):
    """
    Renders the specialized Coffee Card UI for the 'yourCoffeeCard' section.
    RETURNS: An HTML string for the card.
    """
    if not card_data or not isinstance(card_data, dict):
        return f'<div class="card coffee-card-container"><p>{render_missing_html()} (Coffee Card data missing or invalid)</p></div>'

    # Extract data using safe_get and html.escape for safety
    name = html.escape(safe_get(card_data, ['name'], 'Unknown'))
    title = html.escape(safe_get(card_data, ['title'], ''))
    location = html.escape(safe_get(card_data, ['location'], ''))
    experiences_raw = safe_get(card_data, ['experiences'], '') # Assuming string for now
    experiences = html.escape(experiences_raw) if experiences_raw else render_missing_html()
    interests = safe_get(card_data, ['interests'], [])
    hobbies = safe_get(card_data, ['hobbies'], [])
    skills = safe_get(card_data, ['skills'], [])

    # --- Prepare HTML snippets ---
    name_html = f"<h1>{name}</h1>" if name != 'Unknown' else f"<h1>{render_missing_html()}</h1>"
    title_html = f'<p class="headline">{title}</p>' if title else ""
    location_html = f'<p class="headline" style="font-size: 0.9rem;">📍 {location}</p>' if location else ""
    experiences_html = f'<div><h5>Experiences:</h5><p class="prewrap">{experiences}</p></div>' if experiences_raw else f'<div><h5>Experiences:</h5> {render_missing_html()}</div>'

    # Use render_pills_html which already escapes
    interests_pills = render_pills_html(interests, label="Interests", show_label=True) if interests else f"<h5>Interests</h5>{render_missing_html()}"
    hobbies_pills = render_pills_html(hobbies, label="Hobbies", show_label=True) if hobbies else f"<h5>Hobbies</h5>{render_missing_html()}"
    skills_pills = render_pills_html(skills, label="Skills", show_label=True) if skills else f"<h5>Skills</h5>{render_missing_html()}"

    # --- Final Card HTML ---
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
        <div class="section">
            {skills_pills}
        </div>
        <div class="section">
            {interests_pills}
        </div>
        <div class="section">
            {hobbies_pills}
        </div>
         <div class="section">
            {experiences_html}
        </div>
    </div>
    """
    return card_html

# -------------------------------------
# 2. General Nested JSON Renderer - Refactored to Return HTML String
# -------------------------------------
def render_nested_json_html(data, level=0):
    """
    Recursively renders nested Python data into an HTML string
    using the defined styles and formatting keys as titles.
    """
    html_output = ""
    indent_style = f"margin-left: {level * 0.5}rem;" # Adjusted indentation

    if isinstance(data, dict):
        # Determine container class based on level or specific keys if needed
        container_class = "platform-card-like" if level == 1 else "nested-item-container" # Apply card style one level down
        # Special check for platform name to add CSS class
        platform_name = data.get('platformName', '').lower().replace(" ", "-").replace("/", "-").replace("+", "-").replace(".", "")
        if platform_name:
            container_class += f" {platform_name}"

        # Start the container div
        html_output += f'<div class="{container_class}" style="{indent_style}">'
        for key, value in data.items():
            title = html.escape(format_key_to_title(key))
            heading_level = min(level + 3, 6) # h3, h4, h5, h6
            html_output += f'<h{heading_level}>{title}</h{heading_level}>'
            # Recursively call for the value and append the returned HTML
            html_output += render_nested_json_html(value, level + 1)
        html_output += '</div>' # Close the container div

    elif isinstance(data, list):
        if not data:
            html_output += f'<div style="{indent_style}">{render_missing_html()} (Empty List)</div>'
        else:
            is_simple_list = all(isinstance(item, (str, int, float, bool)) for item in data)
            all_strings = all(isinstance(item, str) for item in data)

            if all_strings:
                # Render list of strings as pills
                pills_content = render_pills_html(data, show_label=False)
                if pills_content:
                    html_output += f'<div style="{indent_style}">{pills_content}</div>'
                else:
                    html_output += f'<div style="{indent_style}">{render_missing_html()} (Empty List)</div>'
            elif is_simple_list:
                # Render list of simple primitives as bullets
                list_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in data)
                html_output += f'<div style="{indent_style}"><ul class="simple-list">{list_items}</ul></div>'
            else:
                # Render list of complex items
                html_output += f'<div style="{indent_style}">' # Container for complex list items
                for i, item in enumerate(data):
                    html_output += f"<h6>Item {i+1}</h6>"
                    # Recursively call for each item and append
                    html_output += render_nested_json_html(item, level + 1)
                    if i < len(data) - 1: # Add separator between items
                         html_output += "<hr style='border-top: 1px dashed var(--border-light); margin: 0.5rem 0;'>"
                html_output += '</div>'

    elif isinstance(data, str):
        # Escape the string content
        escaped_data = html.escape(data)
        if escaped_data.startswith("http://") or escaped_data.startswith("https://"):
            # For escaped URLs, render as link but display the escaped version
            html_output += f'<div style="{indent_style}"><a href="{escaped_data}" target="_blank">{escaped_data}</a></div>'
        elif '\n' in escaped_data:
             # Replace escaped newlines with <br> for prewrap effect in HTML
             escaped_data_br = escaped_data.replace("\n", "<br>")
             html_output += f'<div style="{indent_style}"><p class="prewrap">{escaped_data_br}</p></div>'
        elif not escaped_data.strip():
             html_output += f'<div style="{indent_style}">{render_missing_html()} (Empty String)</div>'
        else:
            html_output += f'<div style="{indent_style}"><p>{escaped_data}</p></div>'

    elif isinstance(data, (int, float, bool)):
        html_output += f'<div style="{indent_style}"><p>{html.escape(str(data))}</p></div>'

    elif data is None:
        html_output += f'<div style="{indent_style}">{render_missing_html()}</div>'

    else:
        html_output += f'<div style="{indent_style}"><p><em>Unsupported data type: {html.escape(type(data).__name__)}</em></p></div>'

    return html_output # Return the accumulated HTML string


# -------------------------------------
# 3. Editing Form Renderer - NO CHANGES Needed
# -------------------------------------
def render_edit_form():
    """Renders a form to edit selected parts of the session state data."""
    if 'analysis_data' not in st.session_state or not st.session_state['analysis_data']:
        st.info("Load data first to enable editing.")
        return

    st.markdown("---")
    st.subheader("✏️ Edit Profile Data")

    editable_data = copy.deepcopy(st.session_state['analysis_data'])

    editable_sections = {
        key: format_key_to_title(key)
        for key, value in editable_data.items()
        if isinstance(value, dict) # Only allow editing dicts for now
    }

    if not editable_sections:
        st.warning("No editable dictionary sections found in the loaded data.")
        return

    section_keys = list(editable_sections.keys())
    selected_key_to_edit = st.selectbox( # Renamed variable
        "Select section to edit:",
        options=section_keys,
        format_func=lambda key: editable_sections[key],
        key="edit_section_selector"
    )

    if not selected_key_to_edit:
        return

    # Make sure selected_key_to_edit actually exists in editable_data before accessing
    if selected_key_to_edit not in editable_data:
        st.error("Selected key no longer exists in data.")
        return

    section_data = editable_data[selected_key_to_edit] # Safe now

    if not isinstance(section_data, dict):
        st.warning(f"Selected section '{editable_sections[selected_key_to_edit]}' is not an editable dictionary.")
        return


    with st.form(key=f"edit_form_{selected_key_to_edit}"):
        st.markdown(f"**Editing:** {editable_sections[selected_key_to_edit]}")

        new_section_data = {}

        for field_key, field_value in section_data.items():
            field_label = format_key_to_title(field_key)
            unique_widget_key = f"edit_{selected_key_to_edit}_{field_key}" # Ensure unique key

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
                     try:
                        st.json(field_value, expanded=False)
                     except Exception as json_e:
                        st.text(f"Cannot display as JSON: {json_e}")
                        st.text(str(field_value))
                     st.caption("(Editing complex lists directly in form not yet supported)")
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
                 new_section_data[field_key] = st.text_input(
                     f"{field_label} (Currently None)", value="", key=unique_widget_key
                 )
                 if new_section_data[field_key] == "":
                     new_section_data[field_key] = None

            else: # Fallback for other types (e.g., nested dicts) - show raw JSON
                 st.markdown(f"**{field_label} ({type(field_value).__name__}):**")
                 try:
                     st.json(field_value, expanded=False)
                 except Exception as json_e:
                     st.text(f"Cannot display as JSON: {json_e}")
                     st.text(str(field_value))
                 st.caption("(Editing nested dictionaries directly in form not yet supported)")
                 new_section_data[field_key] = field_value # Keep original value

        submitted = st.form_submit_button("Save Changes")
        if submitted:
            try:
                # Update the actual session state data
                # Ensure the key still exists before assigning
                if selected_key_to_edit in st.session_state['analysis_data']:
                     st.session_state['analysis_data'][selected_key_to_edit] = new_section_data
                     st.success(f"Changes saved for '{editable_sections[selected_key_to_edit]}'. Rerunning...")
                     # Short delay before rerun can sometimes help UI updates
                     time.sleep(0.5)
                     st.rerun()
                else:
                     st.error("Error: The section key being edited seems to have disappeared.")

            except Exception as e:
                st.error(f"Error saving changes: {e}")

# --- Helper function specifically for platform name extraction ---
def get_platform_display_name(platform_key, platform_data):
    """Gets the display name for a platform tab."""
    if isinstance(platform_data, dict):
        # Prioritize platformName field if it exists
        name_from_field = platform_data.get("platformName")
        if name_from_field and isinstance(name_from_field, str):
            return name_from_field
    # Fallback: format the key if platformName is not found or data isn't a dict
    return format_key_to_title(platform_key)

# -------------------------------------
# Example Data Loading (Using placeholder structure) - NO CHANGES
# -------------------------------------
EXAMPLE_ANALYSIS_DATA = {
    "targetIndividual": {
        "nameOrIdentifier": "Homen Shum",
        "primaryProfileURL": "https://www.linkedin.com/in/thomashshum"
    },
    "analyzedPlatforms": [
        "LinkedIn",
        "Instagram"
    ],
    "yourCoffeeCard": {
        "name": "Homen Shum",
        "title": "Ex-PM Startup Banking Associate | USC LUM AI/GenAI Engineering",
        "location": "Fremont, California, United States",
        "interests": [
            "AI/Machine Learning",
            "Finance/Banking",
            "Technology Development",
            "Personal Projects",
            "Humor",
            "Meme Culture"
        ],
        "hobbies": [
            "Browsing social media for entertainment"
        ],
        "skills": [
            "Data",
            "Finance",
            "Startup Banking",
            "AI",
            "GenAI Engineering",
            "Project Management (implied)",
            "Founder skills (implied)"
        ],
        "experiences": "\"Do something great with data and finance\" | Ex-PM Startup Banking Associate | USC LUM AI/GenAI Engineering | Previously JPMM | CalCareer LLC Founder."
    },
    "platformSpecificAnalysis": {
        "linkedIn": {
            "platformName": "LinkedIn",
            "profileFundamentals": {
                "username": "Homen Shum",
                "fullName": "Homen Shum",
                "pronouns": None,
                "location": "Fremont, California, United States",
                "profileURL": "https://www.linkedin.com/in/thomashshum",
                "profileLanguage": "English (inferred)",
                "verificationStatus": False,
                "contactInfoVisible": True,
                "profilePictureDescription": "Professional headshot of a young Asian man in business attire.",
                "bannerImageDescription": "Panoramic photo of a city skyline (likely San Francisco based on location listed).",
                "linkedWebsites": [
                    "https://calcareer.llc"
                ]
            },
            "professionalHeadline": "\"Do something great with data and finance\" | Ex-PM Startup Banking Associate | USC LUM AI/GenAI Engineering | Previously JPMM | CalCareer LLC Founder.",
            "aboutSectionAnalysis": None,
            "featuredContent": [
                "GraphMemory post (162 reactions, 8 comments, 8 reposts)",
                "Fluency Reef project (2 comments)"
            ],
            "experience": [
                {
                    "titleOrDegree": "Ex-PM Startup Banking Associate",
                    "organizationOrSchool": "Unknown (implied from headline)",
                    "dates": None,
                    "location": None,
                    "description": None
                },
                {
                    "titleOrDegree": "Associate (implied)",
                    "organizationOrSchool": "JPMM",
                    "dates": None,
                    "location": None,
                    "description": None
                },
                {
                    "titleOrDegree": "Founder",
                    "organizationOrSchool": "CalCareer LLC",
                    "dates": None,
                    "location": None,
                    "description": None
                }
            ],
            "education": [
                {
                    "titleOrDegree": "LUM AI/GenAI Engineering",
                    "organizationOrSchool": "USC",
                    "dates": None,
                    "location": None,
                    "description": None
                }
            ],
            "skillsEndorsements": [
                "Data",
                "Finance",
                "AI",
                "GenAI Engineering",
                "Startup Banking"
            ],
            "recommendationsGivenReceived": None,
            "accomplishments": "Showcased projects: GraphMemory, Fluency Reef, Startup Search AI.",
            "interests": [
                "Sarah Johnstone (Top Voice/Person)",
                "Ray Dalio (Top Voice/Person)",
                "AI/Machine Learning",
                "Finance/Banking",
                "Technology Development"
            ],
            "contentGenerationActivity": {
                "postingFrequency": "Somewhat active",
                "dominantContentTypes": [
                    "Text updates",
                    "Article shares",
                    "Images",
                    "Videos",
                    "Project showcases",
                    "Posts",
                    "Activity feed updates",
                    "Featured section content"
                ],
                "contentExamples": [
                    "Posts related to AI agents",
                    "Sharing articles on engineering/ML",
                    "Showcasing personal projects (Startup Search AI, GraphMemory, Fluency Reef)"
                ],
                "recurringThemesTopics": [
                    "AI/Machine Learning",
                    "Finance/Banking",
                    "Technology Development",
                    "Personal Projects",
                    "Professional Insights"
                ],
                "overallToneVoice": "Professional, informative, focuses on technical details and achievements.",
                "recentActivityExamples": [
                    "GraphMemory post",
                    "Fluency Reef project post",
                    "Startup Search AI showcase (implied)",
                    "Shares of AI/ML articles (implied)"
                ],
                "articlesPublishedCount": None
            },
            "engagementPatterns": {
                "outgoingInteractionStyle": "Moderate engagement (visible likes and comments), appears supportive and topic-relevant (e.g., comments on AI/ML posts).",
                "typesOfContentEngagedWith": [
                    "Industry news",
                    "AI/ML posts",
                    "Posts from connections (implied)"
                ],
                "incomingEngagementHighlights": "Posts showcasing technical projects or achievements receive significant engagement (e.g., GraphMemory).",
                "typicalIncomingCommentTypes": [
                    "Affirmations (implied)",
                    "Topic-relevant questions/comments (implied)",
                    "Supportive messages from contacts (implied)"
                ]
            },
            "networkCommunityRecommendations": {
                "followerCount": None,
                "followingCount": None,
                "audienceDescription": "Likely a mix of professional contacts, recruiters, and others interested in AI, finance, and technology.",
                "groupCommunityMemberships": [],
                "inboxSidebarPreview": None,
                "inboxSidebarAnalysis": "Inbox sidebar data not available in the source material.",
                "myNetworkTabVisibility": None,
                "myNetworkTabAnalysis": "'My Network' tab data not available in the source material.",
                "platformSuggestions": {
                    "suggestedPeople": [],
                    "suggestedCompaniesOrPages": [],
                    "suggestedGroupsEvents": [],
                    "suggestedContentOrTopics": [],
                    "peopleAlsoViewed": [],
                    "otherSuggestions": []
                },
                "platformRecommendationsAnalysis": "No specific platform recommendations (people, content, etc.) were captured in the source material. Analysis cannot be performed.",
                "detailedConnectionsList": None,
                "detailedConnectionsAnalysis": "Detailed connection list not available in the source material."
            },
            "privacyPresentation": {
                "accountVisibility": "Mostly public",
                "postLevelVisibility": "Public (implied)",
                "networkVisibility": None,
                "activitySharingSettings": "Activity (likes, comments, posts) appears publicly visible.",
                "overallPresentationStyle": "Highly curated, Professional"
            },
            "observedConsumption": {
                "mainFeed": {
                    "observedTopics": [
                        "Professional updates",
                        "Industry news",
                        "Technical content (AI/ML, Engineering)"
                    ],
                    "frequentPostersAccounts": None,
                    "contentFormatCharacteristics": "Mix of text, articles, images, videos.",
                    "specificContentExamples": [
                        "Posts from connections (implied)",
                        "Industry news articles (implied)",
                        "Project showcases from network (implied)"
                    ]
                },
                "discoveryFeed": None,
                "consumptionAnalysisNotes": "Consumption observed primarily reflects professional interests aligned with profile. Data on discovery feed content is unavailable."
            },
            "platformFeatureUsage": [
                {
                    "featureName": "Featured section",
                    "usageDescription": "Used to highlight key projects/posts.",
                    "frequency": "Actively Used"
                },
                {
                    "featureName": "Activity feed",
                    "usageDescription": "Used for posting updates, sharing articles, and engaging.",
                    "frequency": "Actively Used"
                },
                {
                    "featureName": "Posts",
                    "usageDescription": "Primary method for content generation.",
                    "frequency": "Actively Used"
                },
                {
                    "featureName": "Article Sharing",
                    "usageDescription": "Shares relevant industry articles.",
                    "frequency": "Occasionally"
                }
            ],
            "platformSpecificConclusions": "LinkedIn profile is well-maintained, professional, and clearly communicates expertise in AI, finance, and tech project development. Content generation focuses on showcasing technical work and industry knowledge. Engagement is professional. The profile effectively serves career advancement and networking goals within the tech/finance sectors. Lack of UI element data (suggestions, detailed network interactions) limits deeper algorithmic insight."
        },
        "instagram": {
            "platformName": "Instagram",
            "profileFundamentals": {
                "username": None,
                "fullName": None,
                "pronouns": None,
                "location": None,
                "profileURL": None,
                "profileLanguage": None,
                "verificationStatus": None,
                "contactInfoVisible": None,
                "profilePictureDescription": None,
                "bannerImageDescription": None,
                "linkedWebsites": []
            },
            "bioAnalysis": {
                "fullText": None,
                "identifiedKeywords": [],
                "hashtagsInBio": [],
                "mentionedUsersInBio": [],
                "statedPurposeFocus": None,
                "tone": None,
                "callToAction": None
            },
            "storyHighlights": [],
            "contentGenerationActivity": {
                "postingFrequency": "Unknown (User observed in consumption mode)",
                "dominantContentTypes": [],
                "contentExamples": [],
                "recurringThemesTopics": [],
                "overallToneVoice": None,
                "recentActivityExamples": [],
                "gridAesthetic": None,
                "reelsPerformanceIndicators": None,
                "storiesFrequencyEngagement": None
            },
            "engagementPatterns": {
                "outgoingInteractionStyle": "Unknown (User observed viewing content only)",
                "typesOfContentEngagedWith": [
                    "Humor",
                    "Absurdity",
                    "Memes",
                    "Short videos (Reels)",
                    "Random facts/visuals"
                ],
                "incomingEngagementHighlights": None,
                "typicalIncomingCommentTypes": []
            },
            "networkCommunityRecommendations": {
                "followerCount": None,
                "followingCount": None,
                "audienceDescription": None,
                "groupCommunityMemberships": [],
                "inboxSidebarPreview": None,
                "inboxSidebarAnalysis": "Inbox sidebar data not available in the source material.",
                "myNetworkTabVisibility": None,
                "myNetworkTabAnalysis": "'My Network' tab data not available in the source material.",
                "platformSuggestions": {
                    "suggestedPeople": [],
                    "suggestedCompaniesOrPages": [],
                    "suggestedGroupsEvents": [],
                    "suggestedContentOrTopics": [],
                    "peopleAlsoViewed": [],
                    "otherSuggestions": []
                },
                "platformRecommendationsAnalysis": "No specific platform recommendations (people, content, etc.) were captured in the source material. Analysis cannot be performed.",
                "detailedConnectionsList": None,
                "detailedConnectionsAnalysis": "Detailed connection list not available in the source material."
            },
            "privacyPresentation": {
                "accountVisibility": None,
                "postLevelVisibility": None,
                "networkVisibility": None,
                "activitySharingSettings": None,
                "overallPresentationStyle": None
            },
            "observedConsumption": {
                "mainFeed": None,
                "discoveryFeed": {
                    "observedThemes": [
                        "Humor",
                        "Absurdity",
                        "Meme Culture",
                        "Pop Culture",
                        "Random facts/visuals"
                    ],
                    "prevalentContentTypes": [
                        "Images",
                        "Short videos (Reels)",
                        "Memes"
                    ],
                    "commonSoundsEffectsTrends": None,
                    "highlyDescriptiveExamples": [
                        "Meme format 'God: Were you happy with your life? Me: Yes' followed by an absurd list (Early life, Controversial thoughts on the Antichrist, Schizophrenia diagnosis, Meme page admin, CIA assassination) from mr.tom.foolery."
                    ],
                    "overallFeedCharacteristics": "Feed appears tailored towards humor, memes, and visually engaging short-form content."
                },
                "consumptionAnalysisNotes": "Observed Instagram consumption focuses heavily on entertainment, humor, and meme culture via the discovery feed. This contrasts sharply with the professional focus on LinkedIn. It suggests Instagram is used for leisure and personal interests. Lack of main feed data prevents comparison between followed accounts and algorithmic suggestions."
            },
            "platformFeatureUsage": [
                {
                    "featureName": "Browsing Feed",
                    "usageDescription": "Primary observed activity is scrolling and viewing content.",
                    "frequency": "Actively Used (during observation)"
                },
                {
                    "featureName": "Viewing Reels",
                    "usageDescription": "Consumes short-form video content.",
                    "frequency": "Implied Frequent (based on feed content)"
                }
            ],
            "platformSpecificConclusions": "Instagram usage, based solely on observed consumption, is centered around entertainment, specifically humor and meme culture delivered via short-form visual content (Reels, image posts). It serves a purpose distinct from LinkedIn, likely personal relaxation or staying current with online trends. The user's own profile, activity, and network remain unknown from the provided data, limiting conclusions about their content generation or specific community engagement on this platform."
        },
        "twitter": None,
        "facebook": None,
        "tiktok": None,
        "reddit": None,
        "otherPlatforms": []
    },
    "crossPlatformSynthesis": {
        "consistencyVsVariation": {
            "profileElementConsistency": "Profile picture, banner, and username consistency cannot be assessed due to limited Instagram data. LinkedIn profile is detailed and professional; Instagram profile details are unknown.",
            "contentTonePersonaConsistency": "Significant difference. LinkedIn is professional, technical, career-focused. Instagram consumption is informal, entertainment-focused (humor, memes).",
            "notableDifferences": "Deliberate separation between professional (LinkedIn) and personal/entertainment (Instagram consumption) online activity and likely persona."
        },
        "contentOverlapStrategy": "No evidence of cross-posting between LinkedIn and observed Instagram consumption. Suggests distinct content strategies for each platform.",
        "synthesizedExpertiseInterests": {
            "coreProfessionalSkills": [
                "AI/Machine Learning",
                "GenAI Engineering",
                "Finance",
                "Startup Banking",
                "Data Analysis",
                "Project Management",
                "Technical Project Development"
            ],
            "corePersonalInterests": [
                "Humor/Absurdity",
                "Meme Culture",
                "Pop Culture (implied)",
                "Short-form Video Content",
                "Technology Trends (professional overlap)"
            ]
        },
        "overallOnlinePersonaNarrative": "Projects a dual online persona: a highly skilled and ambitious professional in the AI/finance/tech space on LinkedIn, actively showcasing projects and engaging with industry content; contrasted with a consumer of informal, humorous, and meme-based content on platforms like Instagram for entertainment.",
        "professionalEvaluation": {
            "strengthsSkillsMatch": "LinkedIn profile strongly highlights in-demand skills in AI, GenAI, finance, and project execution.",
            "impactAchievements": "Featured projects (GraphMemory, Fluency Reef) demonstrate practical application of skills and initiative, garnering positive engagement.",
            "industryEngagement": "Active posting, sharing, and engagement on LinkedIn suggest awareness of industry trends and active networking.",
            "potentialRedFlagsClarifications": "None apparent based on LinkedIn data. The distinct separation between professional and personal platforms is common and not inherently a red flag.",
            "overallCandidateSummary": "Appears to be a strong, technically proficient candidate with relevant experience and demonstrated project work in AI and finance. Presents professionally and seems engaged in their field. (Contingent on LinkedIn data only)."
        },
        "marketTrendInsights": {
            "keyTechnologiesToolsTopics": [
                "AI/Machine Learning",
                "Generative AI (GenAI)",
                "Finance Technology (FinTech)",
                "Startup Ecosystem",
                "Data Science"
            ],
            "emergingThemesNiches": [
                "AI Agents",
                "Specific AI projects (GraphMemory, Fluency Reef)"
            ],
            "relevantContentPatterns": "Focus on project showcases, technical insights, and sharing relevant articles within the AI/ML/Finance domains on LinkedIn."
        },
        "inferredAlgorithmicPerception": [
            {
                "platformName": "LinkedIn",
                "categorizationHypothesis": "Algorithm likely categorizes user as: Professional in AI/ML, Finance, Tech; actively job seeking or career-building; interested in specific technologies (GenAI, AI Agents), industry news, networking with peers and recruiters. Hypothesis based on detailed profile information (headline, skills, experience), content posted/shared (project showcases, tech articles), engagement patterns (likes/comments on professional topics), and stated interests (following industry figures). Lack of specific recommendation/suggestion data limits confirmation."
            },
            {
                "platformName": "Instagram",
                "categorizationHypothesis": "Algorithm likely categorizes user as: Interested in Humor, Memes, Absurdist Content, Short-form Video (Reels), Pop Culture trends. Hypothesis strongly based on observed consumption patterns in the discovery feed (viewing meme accounts like mr.tom.foolery, types of content surfaced). Assumes user engagement (likes, shares, follows - though not observed) aligns with consumption. Lack of profile data, user posts, main feed content, or specific recommendations prevents a more detailed hypothesis about demographics or specific sub-interests."
            }
        ],
        "crossPlatformNetworkAnalysis": None
    },
    # Renamed the final key as per the structure request, keeping the original content
    "finalComprehensiveSummary": "Analysis reveals a bifurcated online presence. Homen Shum maintains a highly professional, detailed LinkedIn profile focused on showcasing expertise and projects in AI, finance, and technology. Activity on LinkedIn aligns with career goals, involving technical posts and professional engagement. Conversely, observed Instagram usage is purely consumption-based, centered on informal entertainment like humor and memes via the discovery feed. This indicates a clear separation between professional branding and personal leisure online. Algorithmic perception likely differs significantly across platforms, with LinkedIn identifying a tech/finance professional and Instagram identifying a consumer of humor/meme content. This dual approach seems intentional. Key strengths lie in the well-articulated professional profile on LinkedIn. Future strategy should leverage this strong professional base while maintaining the desired separation for personal platforms. Insights are limited by the lack of user interface element data (suggestions, ads, detailed network interactions) across platforms."
}

# You can now use the EXAMPLE_ANALYSIS_DATA dictionary in your Python code.
# For example:
# print(EXAMPLE_ANALYSIS_DATA["targetIndividual"]["nameOrIdentifier"])
def load_example_data():
    """Loads the hardcoded example JSON data."""
    # No conversion needed, just return a deep copy
    return copy.deepcopy(EXAMPLE_ANALYSIS_DATA)

# -------------------------------------
# Main Execution Block - WITH TABS
# -------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Unified Profile Viewer", page_icon="👤", layout="wide")

    # --- Initialize Session State ---
    if 'analysis_data' not in st.session_state:
        st.session_state['analysis_data'] = None
    if 'uploaded_avatar_file' not in st.session_state:
        st.session_state['uploaded_avatar_file'] = None
    if 'processed_upload_filename' not in st.session_state:
         st.session_state['processed_upload_filename'] = None
    if 'current_data_source' not in st.session_state:
         st.session_state['current_data_source'] = "None"

    # Apply CSS globally
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

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
        uploaded_video_file = st.file_uploader(
            "Upload Video (Optional)", type=["mp4", "mov", "avi", "mkv", "webm"], key="video_uploader"
         )

        # Process uploaded JSON/TXT file
        # ***** TARGETED CHANGE: Ensure platformSpecificAnalysis is treated as object *****
        if uploaded_analysis_file is not None:
            if uploaded_analysis_file.name != st.session_state.get('processed_upload_filename'):
                try:
                    file_content = uploaded_analysis_file.read().decode("utf-8")
                    parsed_data = json.loads(file_content)
                    # Basic validation + check platformSpecificAnalysis type
                    if isinstance(parsed_data, dict) and \
                       isinstance(parsed_data.get("platformSpecificAnalysis"), (dict, type(None))): # Allow dict or missing

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
             example_data = load_example_data() # Example now keeps platformSpecificAnalysis as object
             if isinstance(example_data, dict):
                 st.session_state['analysis_data'] = example_data
                 st.session_state['uploaded_avatar_file'] = None
                 st.session_state['processed_upload_filename'] = "EXAMPLE_DATA"
                 st.session_state['current_data_source'] = "Example Data" # Update if needed
                 st.info("Example data loaded.")
                 st.rerun()
             else:
                 st.error("Failed to load example data.")

        st.divider()
        st.caption(f"Current Source: {st.session_state['current_data_source']}")

        if st.session_state.get('analysis_data'):
             with st.expander("✏️ Edit Data Sections", expanded=False):
                  render_edit_form() # Edit form remains in sidebar


    # --- Main Page Content Area ---
    st.title("👤 Unified Profile Viewer")

    if st.session_state.get('analysis_data'):
        data = st.session_state['analysis_data']
        avatar_file_obj = st.session_state.get('uploaded_avatar_file')

        # --- 1. Render Specialized Coffee Card (Displayed Above Tabs) ---
        # [ ... No changes needed in Coffee Card rendering call ... ]
        st.header("☕ Coffee Card")
        coffee_card_data = data.get('yourCoffeeCard')
        profile_name = safe_get(coffee_card_data, ['name']) or \
                       safe_get(data, ['targetIndividual', 'nameOrIdentifier'], "Unknown")
        avatar_url = get_avatar_url(profile_name, avatar_file_obj)
        st.markdown(render_coffee_card_html(coffee_card_data, avatar_url), unsafe_allow_html=True)
        st.markdown("---")

        # --- 2. Prepare and Render Tabs ---
        # ***** TARGETED CHANGE: Tab logic based on OBJECT keys *****
        st.header("📊 Profile Details")

        overview_keys_to_render = [
            "targetIndividual", "analyzedPlatforms", "crossPlatformSynthesis", "finalComprehensiveSummary"
        ]
        overview_data_subset = {k: data.get(k) for k in overview_keys_to_render if k in data}

        # Get platform data OBJECT
        platform_analysis_obj = data.get('platformSpecificAnalysis', {}) # Default to empty dict
        platform_tabs_available = isinstance(platform_analysis_obj, dict) and platform_analysis_obj

        # Create tab titles from the object keys
        tab_titles = ["Overview"]
        platform_keys_in_order = [] # Keep track of the order for mapping tabs to data
        other_platforms_data = None # To store the array for the 'Other' tab

        if platform_tabs_available:
            for platform_key, platform_data in platform_analysis_obj.items():
                if platform_key == "otherPlatforms":
                     # Handle 'otherPlatforms' array separately
                     if isinstance(platform_data, list) and platform_data:
                          other_platforms_data = platform_data
                          tab_titles.append("Other") # Add tab title for 'Other'
                          platform_keys_in_order.append(platform_key) # Still need a key placeholder
                elif isinstance(platform_data, dict): # Ensure it's a dictionary
                    # Get display name using helper
                    display_name = get_platform_display_name(platform_key, platform_data)
                    tab_titles.append(display_name)
                    platform_keys_in_order.append(platform_key) # Store the original key

        # Create tabs
        if not platform_tabs_available or len(tab_titles) == 1:
             # Case: No platform data or only 'Overview'
             st.warning("No specific platform analysis data found to display in tabs.")
             tabs = st.tabs(["Overview"])
             with tabs[0]:
                 st.subheader("Synthesis & Summary")
                 overview_html = render_nested_json_html(overview_data_subset, level=0)
                 st.markdown(overview_html, unsafe_allow_html=True)
        else:
             # Create tabs with Overview + Platforms
             try:
                 tabs = st.tabs(tab_titles)

                 # Render Overview Tab
                 with tabs[0]:
                     st.subheader("Synthesis & Summary")
                     overview_html = render_nested_json_html(overview_data_subset, level=0)
                     st.markdown(overview_html, unsafe_allow_html=True)

                 # Render Platform Tabs
                 tab_index_offset = 1 # Start platform tabs from index 1
                 for i, platform_key in enumerate(platform_keys_in_order):
                    current_tab_index = i + tab_index_offset
                    if current_tab_index < len(tabs): # Safety check
                        with tabs[current_tab_index]:
                            if platform_key == "otherPlatforms":
                                 # Special rendering for the 'Other' tab
                                 st.subheader("Other Platform Analyses")
                                 if other_platforms_data:
                                     for idx, other_platform_item in enumerate(other_platforms_data):
                                         # Try to get a name for a sub-heading
                                         item_name = get_platform_display_name(f"other_{idx}", other_platform_item)
                                         st.markdown(f"<h4>{item_name}</h4>", unsafe_allow_html=True)
                                         item_html = render_nested_json_html(other_platform_item, level=1) # Start nesting
                                         st.markdown(item_html, unsafe_allow_html=True)
                                         if idx < len(other_platforms_data) - 1:
                                              st.markdown("---") # Separator between 'other' items
                                 else:
                                     st.info("No data found for 'Other Platforms'.")
                            else:
                                 # Render standard platform data
                                 platform_data_to_render = platform_analysis_obj.get(platform_key)
                                 if platform_data_to_render and isinstance(platform_data_to_render, dict):
                                     # Use display name for subheader inside tab
                                     display_name = get_platform_display_name(platform_key, platform_data_to_render)
                                     st.subheader(f"{display_name} Details")
                                     platform_html = render_nested_json_html(platform_data_to_render, level=0) # Render the whole object
                                     st.markdown(platform_html, unsafe_allow_html=True)
                                 else:
                                     st.warning(f"Could not retrieve valid dictionary data for platform key: {platform_key}")
                    else:
                        st.error(f"Tab index mismatch for platform key: {platform_key}")

             except Exception as e:
                 st.error(f"Error creating or rendering tabs: {e}. Displaying overview only.")
                 # Fallback: just show overview in a single tab
                 tabs = st.tabs(["Overview"])
                 with tabs[0]:
                     st.subheader("Synthesis & Summary")
                     overview_html = render_nested_json_html(overview_data_subset, level=0)
                     st.markdown(overview_html, unsafe_allow_html=True)

    else:
        # Display instructions if no data is loaded
        st.header("👋 Welcome!")
        st.info("⬅️ Use the sidebar to load profile data from a JSON/TXT file or view the example.")
        st.markdown("""
        This application displays profile information using:
        1.  A specialized **Coffee Card** view.
        2.  An **Overview** tab summarizing cross-platform synthesis.
        3.  **Platform-specific tabs** showing detailed data for each analyzed platform.

        All details are rendered dynamically from the loaded JSON structure using consistent styling.
        Use the **Edit Data Sections** expander in the sidebar to modify loaded data.
        """)