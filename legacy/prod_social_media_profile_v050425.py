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
    /* Cafe-inspired color palette */
    --coffee-dark: #362417;
    --coffee-medium: #4E342E;
    --coffee-light: #6D4C41;
    --crema: #F5ECD7;
    --cream: #FFF8E1;
    --brass: #D4AD76;
    
    /* Notion-inspired neutrals */
    --neutral-bg: #FFFFFF;
    --neutral-surface: #FAFAFA;
    --neutral-pill: #EAEAEA;
    --neutral-border: #E0E0E0;
    
    /* Text colors */
    --text-primary: #37352F;
    --text-secondary: #5A5A58;
    --text-tertiary: #908F8C;
    --text-missing: #B6B6B5;
    
    /* Accent colors */
    --accent-blue: #0B66C2;
    --accent-green: #0F7B6C;
    --accent-purple: #6940A5;
    --accent-orange: #D4732A;
    
    /* Platform specific accents */
    --brass-linkedin: #0A66C2;
    --brass-instagram: #E1306C;
    --brass-twitter-x: #1DA1F2;
    --brass-facebook: #1877F2;
    --brass-tiktok: #000000;
    --brass-reddit: #FF4500;
    --brass-github: #333;
    --brass-other: #777777;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--text-primary);
    background-color: var(--neutral-surface);
    line-height: 1.6;
}

/* --- Coffee Card --- */
.card {
    border: 1px solid var(--neutral-border);
    border-radius: 12px;
    background: var(--neutral-bg);
    padding: 1.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 3px, rgba(0, 0, 0, 0.05) 0px 20px 25px -5px, rgba(0, 0, 0, 0.04) 0px 10px 10px -5px;
    transition: transform 0.2s ease-in-out, box-shadow 0.3s ease-in-out;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: rgba(0, 0, 0, 0.05) 0px 5px 10px, rgba(0, 0, 0, 0.05) 0px 15px 30px -8px, rgba(0, 0, 0, 0.04) 0px 10px 15px -3px;
}

.coffee-card-container {
    background-image: linear-gradient(to bottom right, rgba(245, 236, 215, 0.4) 0%, rgba(255, 255, 255, 1) 80%);
    border-left: 5px solid var(--brass);
    position: relative;
    overflow: hidden;
}

.coffee-card-container::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 8px;
    background: linear-gradient(to right, var(--brass), var(--brass-light));
    opacity: 0.8;
}

.coffee-card-header {
    position: relative;
    padding-top: 2.5rem;
}

.header-content {
    display: flex;
    align-items: flex-start;
    gap: 1.8rem;
}

img.avatar {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    border: 2px solid var(--cream);
    object-fit: cover;
    flex-shrink: 0;
    box-shadow: rgba(0, 0, 0, 0.05) 0px 2px 4px, rgba(0, 0, 0, 0.1) 0px 12px 15px -8px;
}

.header-text h1 {
    color: var(--coffee-dark);
    margin: 0 0 0.3rem 0;
    font-size: 1.9rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}

.header-text p.headline {
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin: 0.3rem 0 0.8rem 0;
    line-height: 1.4;
    letter-spacing: -0.01em;
}

/* --- Pills --- */
.pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    margin: 0.25rem 0.5rem 0.25rem 0;
    border-radius: 15px;
    background: var(--neutral-pill);
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
    line-height: 1.4;
    transition: all 0.2s ease;
    white-space: nowrap;
    border: 1px solid rgba(0, 0, 0, 0.04);
}

.pill:hover {
    background: var(--cream);
    color: var(--coffee-medium);
    box-shadow: rgba(0, 0, 0, 0.05) 0px 1px 2px;
}

.pill-container {
    margin-top: 0.75rem;
    margin-bottom: 0.75rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
}

/* --- Platform Cards --- */
.platform-card-like {
    background: var(--neutral-bg);
    border: 1px solid var(--neutral-border);
    border-left: 5px solid var(--brass);
    border-radius: 0 8px 8px 0;
    padding: 1.8rem;
    margin-bottom: 1.8rem;
    box-shadow: rgba(0, 0, 0, 0.03) 0px 1px 3px, rgba(0, 0, 0, 0.02) 0px 15px 20px -5px;
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}

.platform-card-like:hover {
    transform: translateY(-2px);
    box-shadow: rgba(0, 0, 0, 0.05) 0px 2px 4px, rgba(0, 0, 0, 0.05) 0px 15px 20px -5px;
}

/* Platform specific accents with improved colors */
.platform-card-like.linkedin { border-left-color: var(--brass-linkedin); }
.platform-card-like.instagram { border-left-color: var(--brass-instagram); }
.platform-card-like.twitter-x { border-left-color: var(--brass-twitter-x); }
.platform-card-like.facebook { border-left-color: var(--brass-facebook); }
.platform-card-like.tiktok { border-left-color: var(--brass-tiktok); }
.platform-card-like.reddit { border-left-color: var(--brass-reddit); }
.platform-card-like.github { border-left-color: var(--brass-github); }
.platform-card-like.other { border-left-color: var(--brass-other); }

.section {
    margin-bottom: 1.8rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--neutral-border);
}

.section:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}

/* --- Typography --- */
h3 {
    color: var(--coffee-dark);
    margin-top: 1.8rem;
    margin-bottom: 1.2rem;
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    border-bottom: 1px solid var(--neutral-border);
    padding-bottom: 0.6rem;
    position: relative;
}

h3::after {
    content: "";
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 4rem;
    height: 3px;
    background: var(--brass);
    border-radius: 3px;
}

h4 {
    color: var(--coffee-medium);
    margin-top: 1.2rem;
    margin-bottom: 0.9rem;
    font-size: 1.2rem;
    font-weight: 600;
    letter-spacing: -0.01em;
}

.platform-card-like h4 {
    margin-top: 0.3rem;
    display: flex;
    align-items: center;
}

.platform-card-like h4::before {
    content: "☕";
    margin-right: 0.5rem;
    font-size: 1.1rem;
    opacity: 0.8;
}

h5 {
    color: var(--coffee-medium);
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 0.6rem;
    margin-top: 1.2rem;
    letter-spacing: -0.01em;
}

h6 {
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
    margin-top: 0.9rem;
    letter-spacing: -0.01em;
}

p, ul, ol {
    color: var(--text-primary);
    line-height: 1.65;
    margin-bottom: 1.2rem;
}

ul, ol {
    padding-left: 1.8rem;
}

li {
    margin-bottom: 0.6rem;
}

a {
    color: var(--accent-blue);
    text-decoration: none;
    transition: all 0.2s ease;
    border-bottom: 1px dotted transparent;
}

a:hover {
    border-bottom: 1px dotted var(--accent-blue);
}

/* --- Missing Data --- */
em.missing {
    color: var(--text-missing);
    font-style: italic;
    font-weight: 400;
}

/* Styling for pre-wrap paragraph */
p.prewrap {
    white-space: pre-wrap;
    margin-bottom: 0.6rem;
    background: var(--neutral-surface);
    padding: 0.8rem;
    border-radius: 6px;
    font-size: 0.95rem;
    border: 1px solid var(--neutral-border);
}

/* Simple list styling */
ul.simple-list {
    list-style-type: none;
    padding-left: 0.3rem;
    margin-top: 0.6rem;
    margin-bottom: 1.2rem;
}

ul.simple-list li {
    margin-bottom: 0.4rem;
    font-size: 0.95rem;
    padding-left: 1.5rem;
    position: relative;
}

ul.simple-list li::before {
    content: "•";
    color: var(--brass);
    font-size: 1.2rem;
    position: absolute;
    left: 0;
    top: -0.1rem;
}

/* Container for recursively rendered items */
.nested-item-container {
    margin-left: 1.2rem;
    padding-left: 1.2rem;
    border-left: 2px solid var(--neutral-border);
    margin-bottom: 1.2rem;
    margin-top: 0.6rem;
}

.nested-item-container:hover {
    border-left: 2px solid var(--brass);
}

.nested-item-container:last-child {
    margin-bottom: 0;
}

/* --- Edit Form Specific Styling --- */
.edit-form-container {
    background-color: var(--neutral-surface);
    border: 1px solid var(--neutral-border);
    border-radius: 8px;
    padding: 1.8rem;
    margin-top: 2rem;
    box-shadow: rgba(0, 0, 0, 0.04) 0px 1px 3px;
}

/* --- Tab Navigation --- */
div[data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 2px solid var(--neutral-border);
    padding-bottom: 0;
    margin-bottom: 1.8rem;
}

button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px;
    padding: 0.8rem 1.2rem !important;
    font-size: 1rem;
    font-weight: 500;
    transition: all 0.2s ease;
    border-radius: 6px 6px 0 0 !important;
}

button[data-baseweb="tab"]:hover {
    color: var(--coffee-medium) !important;
    background-color: rgba(212, 173, 118, 0.1) !important;
}

button[aria-selected="true"] {
    color: var(--coffee-dark) !important;
    font-weight: 600 !important;
    border-bottom-color: var(--brass) !important;
}

/* Additional Cafe-themed elements */
.coffee-icon {
    display: inline-block;
    margin-right: 0.5rem;
    opacity: 0.9;
}

/* Subtle pattern for background */
body::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23d4ad76' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    z-index: -1;
}

/* Animation for cards */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.card, .platform-card-like {
    animation: fadeIn 0.5s ease-out;
}

/* Streamlit Overrides */
.stApp {
    background-color: transparent;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1rem;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    letter-spacing: -0.02em;
}

/* Decorative elements */
.coffee-card-container::after {
    content: "☕";
    position: absolute;
    top: 1rem;
    right: 1.5rem;
    font-size: 1.8rem;
    opacity: 0.15;
    color: var(--coffee-dark);
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
    Renders the specialized Coffee Card UI with improved cafe-themed styling.
    """
    if not card_data or not isinstance(card_data, dict):
        return f'<div class="card coffee-card-container"><p>{render_missing_html()} (Coffee Card data missing or invalid)</p></div>'

    # Extract data using safe_get and html.escape for safety
    name = html.escape(safe_get(card_data, ['name'], 'Unknown'))
    title = html.escape(safe_get(card_data, ['title'], ''))
    location = html.escape(safe_get(card_data, ['location'], ''))
    experiences_raw = safe_get(card_data, ['experiences'], '')
    experiences = html.escape(experiences_raw) if experiences_raw else render_missing_html()
    interests = safe_get(card_data, ['interests'], [])
    hobbies = safe_get(card_data, ['hobbies'], [])
    skills = safe_get(card_data, ['skills'], [])

    # --- Prepare HTML snippets ---
    name_html = f"<h1>{name}</h1>" if name != 'Unknown' else f"<h1>{render_missing_html()}</h1>"
    title_html = f'<p class="headline">{title}</p>' if title else ""
    location_html = f'<p class="headline" style="font-size: 0.95rem;">📍 {location}</p>' if location else ""
    
    # Fix for experiences section - simplified to match other sections
    experiences_html = f'<h5><span class="coffee-icon">☕</span>Experiences</h5>'
    if experiences_raw:
        experiences_html += f'<p class="prewrap">{experiences}</p>'
    else:
        experiences_html += render_missing_html()

    # Use render_pills_html which already escapes
    interests_pills = render_pills_html(interests, label="<span class='coffee-icon'>🔎</span>Interests", show_label=True) if interests else f"<h5><span class='coffee-icon'>🔎</span>Interests</h5>{render_missing_html()}"
    hobbies_pills = render_pills_html(hobbies, label="<span class='coffee-icon'>🎯</span>Hobbies", show_label=True) if hobbies else f"<h5><span class='coffee-icon'>🎯</span>Hobbies</h5>{render_missing_html()}"
    skills_pills = render_pills_html(skills, label="<span class='coffee-icon'>✨</span>Skills", show_label=True) if skills else f"<h5><span class='coffee-icon'>✨</span>Skills</h5>{render_missing_html()}"

    # --- Final Card HTML with enhanced styling ---
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

def render_pills_html(items, label=None, show_label=True, pill_class="pill"):
    """Enhanced pill rendering with better styling"""
    if not items or not isinstance(items, list):
        return ""
    items_str = [html.escape(str(item).strip()) for item in items if item and str(item).strip()]
    if not items_str:
        return ""
    
    # Create pills with a subtle shadow effect
    pills_html_content = "".join(f'<span class="{pill_class}">{item}</span>' for item in items_str)
    label_html = f'<h5>{label}</h5>' if label and show_label else ''
    return f'{label_html}<div class="pill-container">{pills_html_content}</div>'

def render_nested_json_html(data, level=0):
    """
    Enhanced recursive renderer with cafe-themed styling touches
    """
    html_output = ""
    indent_style = f"margin-left: {level * 0.5}rem;"

    if isinstance(data, dict):
        # Determine container class based on level or specific keys
        container_class = "platform-card-like" if level == 1 else "nested-item-container"
        # Special check for platform name to add CSS class
        platform_name = data.get('platformName', '').lower().replace(" ", "-").replace("/", "-").replace("+", "-").replace(".", "")
        if platform_name:
            container_class += f" {platform_name}"

        # Add a subtle animation delay based on nesting level
        delay_style = f"animation-delay: {level * 0.1}s;"
        
        # Start the container div with enhanced styling
        html_output += f'<div class="{container_class}" style="{indent_style} {delay_style}">'
        
        for key, value in data.items():
            title = html.escape(format_key_to_title(key))
            heading_level = min(level + 3, 6)  # h3, h4, h5, h6
            
            # Add coffee or diff icon to headings for a cafe feel
            icon = ""
            if heading_level == 3:
                icon = '<span class="coffee-icon">☕</span>'
            elif heading_level == 4:
                icon = '<span class="diff-icon">(●' + '◡' + '●)</span>'
            
            html_output += f'<h{heading_level}>{icon}{title}</h{heading_level}>'
            # Recursively call for the value and append the returned HTML
            html_output += render_nested_json_html(value, level + 1)
        
        html_output += '</div>'  # Close the container div

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
                # Render list of simple primitives as bullets with enhanced styling
                list_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in data)
                html_output += f'<div style="{indent_style}"><ul class="simple-list">{list_items}</ul></div>'
            else:
                # Render list of complex items with dividers
                html_output += f'<div style="{indent_style}">'  # Container for complex list items
                for i, item in enumerate(data):
                    html_output += f"<h6>✦ Item {i+1}</h6>"
                    # Recursively call for each item and append
                    html_output += render_nested_json_html(item, level + 1)
                    if i < len(data) - 1:  # Add separator between items
                        html_output += "<hr style='border: none; border-top: 1px dashed var(--neutral-border); margin: 0.8rem 0;'>"
                html_output += '</div>'

    elif isinstance(data, str):
        # Escape the string content
        escaped_data = html.escape(data)
        if escaped_data.startswith("http://") or escaped_data.startswith("https://"):
            # For escaped URLs, render as link with icon
            html_output += f'<div style="{indent_style}"><a href="{escaped_data}" target="_blank">🔗 {escaped_data}</a></div>'
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

    return html_output

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

# Custom header with cafe theme
def render_custom_header():
    header_html = """
    <div style="padding: 1rem 0; margin-bottom: 2rem; text-align: center; position: relative;">
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 8px; background: linear-gradient(to right, var(--brass), var(--cream));"></div>
        <h1 style="font-size: 2.4rem; color: var(--coffee-dark); margin-bottom: 0.3rem; letter-spacing: -0.02em;">
            ☕ CaféCorner
        </h1>
        <p style="color: var(--text-secondary); font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            A cozy place to view and manage your professional presence across platforms
        </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

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
    "nameOrIdentifier": "Homin Shum",
    "primaryProfileURL": "https://www.linkedin.com/in/thomashshum"
  },
  "analyzedPlatforms": [
    "LinkedIn",
    "Instagram"
  ],
  "yourCoffeeCard": {
    "name": "Homin Shum",
    "title": "Ex-PM Startup Banking Associate | USC LUM AI/GenAI Engineering", # Derived from headline/bio
    "location": "Fremont, California, United States",
    "interests": [ # Inferred from LinkedIn themes and Instagram consumption
      "AI/Machine Learning",
      "Finance/Banking",
      "Technology Development",
      "Humor/Memes"
    ],
    "hobbies": [ # Inferred, Placeholder
        "Personal Project Development",
        "Following Tech Trends"
    ],
    "skills": [ # From LinkedIn bioKeywords
      "Data",
      "Finance",
      "Startup Banking",
      "AI",
      "GenAI Engineering",
      "Project Management (implied)"
    ],
    "experiences": "\"Do something great with data and finance\" | Ex-PM Startup Banking Associate | USC LUM AI/GenAI Engineering | Previously JPMM | CalCareer LLC Founder." # Short summary from Bio
  },
  "platformSpecificAnalysis": {
    "linkedIn": {
      "platformName": "LinkedIn",
      "profileFundamentals": {
        "username": "thomashshum", # Extracted from URL
        "fullName": "Homin Shum",
        "pronouns": None, # Not visible in provided data
        "location": "Fremont, California, United States",
        "profileURL": "https://www.linkedin.com/in/thomashshum",
        "profileLanguage": None, # Not discernible from provided data
        "verificationStatus": None, # Not visible in provided data
        "contactInfoVisible": True, # Explicit "Contact info" link mentioned
        "profilePictureDescription": "Professional headshot of a young Asian man in business attire.",
        "bannerImageDescription": "Panoramic photo of a city skyline (likely San Francisco based on location listed).",
        "linkedWebsites": [
          "https://calcareer.llc" # From profile links
        ]
      },
      "professionalHeadline": "\"Do something great with data and finance\" | Ex-PM Startup Banking Associate | USC LUM AI/GenAI Engineering | Previously JPMM | CalCareer LLC Founder.", # From bio text
      "aboutSectionAnalysis": { # Mapping bio details
        "fullText": None, # Full "About" section text not provided, only headline/bio line
        "identifiedKeywords": [
          "Data",
          "Finance",
          "Startup Banking",
          "USC LUM",
          "AI",
          "GenAI Engineering",
          "JPMM",
          "CalCareer LLC",
          "Founder"
        ],
        "hashtagsInBio": [], # None observed
        "mentionedUsersInBio": [], # None observed
        "statedPurposeFocus": "Highlighting expertise and experience at the intersection of data, finance, AI, and entrepreneurship.",
        "tone": "Professional, concise, achievement-oriented.",
        "callToAction": None # bioHasCTA was false
      },
      "featuredContent": [ # From analysis of 'Featured' section
        "Showcasing technical projects (Startup Search AI)",
        "GraphMemory post",
        "Fluency Reef project"
      ],
      "experience": [ # Assumed structure based on bio mentions
        {
          "titleOrDegree": "Ex-PM Startup Banking Associate",
          "organizationOrSchool": "Unknown (Startup Banking)", # Placeholder
          "dates": None,
          "location": None,
          "description": None
        },
        {
          "titleOrDegree": "Previously",
          "organizationOrSchool": "JPMM", # Assumed JPMorgan job title abbreviation
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
      "education": [ # Assumed structure based on bio mention
        {
          "titleOrDegree": "AI/GenAI Engineering",
          "organizationOrSchool": "USC LUM", # Assumed USC Viterbi School of Engineering program
          "dates": None,
          "location": None,
          "description": None
        }
      ],
      "skillsEndorsements": [ # Primarily from bioKeywords, endorsements not detailed
        "Data",
        "Finance",
        "Startup Banking",
        "AI",
        "GenAI Engineering"
      ],
      "recommendationsGivenReceived": None, # Not visible in provided data
      "accomplishments": "Showcased projects: Startup Search AI, GraphMemory, Fluency Reef.", # From featured/activity
      "interests": [ # From 'Interests' section analysis
        "Sarah Johnstone",
        "Ray Dalio",
        "AI/Machine Learning Topics",
        "Finance/Banking Topics",
        "Technology Development Topics"
      ],
      "contentGenerationActivity": {
        "postingFrequency": "Appears somewhat active, posts and shares visible.",
        "dominantContentTypes": [
          "Text updates",
          "Article shares",
          "Images",
          "Videos",
          "Project showcases",
          "Featured section updates"
        ],
        "contentExamples": [
          "Posts related to AI agents",
          "Sharing articles on engineering/ML",
          "Showcasing personal projects (Startup Search AI)",
          "GraphMemory project showcase post (162 reactions, 8 comments, 8 reposts)",
          "Fluency Reef project showcase post (2 comments)"
        ],
        "recurringThemesTopics": [
          "AI/Machine Learning",
          "Finance/Banking",
          "Technology Development",
          "Personal Projects",
          "Professional Insights"
        ],
        "overallToneVoice": "Professional, informative, focuses on technical details and achievements.",
        "recentActivityExamples": [ # Specific examples from analysis
            "GraphMemory post",
            "Fluency Reef project post"
        ],
        "articlesPublishedCount": None # Not specified if user authors articles vs sharing
      },
      "engagementPatterns": {
        "outgoingInteractionStyle": "Moderate engagement; appears supportive and topic-relevant (e.g., comments on AI/ML posts, likes).",
        "typesOfContentEngagedWith": [ # Inferred from context
          "Posts from connections",
          "Industry news/articles",
          "AI/ML related content",
          "Technology posts"
        ],
        "incomingEngagementHighlights": "Posts showcasing technical projects or achievements receive significant engagement (e.g., GraphMemory post).",
        "typicalIncomingCommentTypes": [ # Inferred from example counts
          "Affirmations",
          "Questions",
          "Supportive messages"
        ]
      },
      "networkCommunityRecommendations": {
        "followerCount": None, # Not provided
        "followingCount": None, # Not provided
        "audienceDescription": "Likely a mix of professional contacts, recruiters, and others interested in AI, finance, and technology.",
        "groupCommunityMemberships": [], # None specified
        "inboxSidebarPreview": None, # Data not available from provided analysis
        "inboxSidebarAnalysis": "Data not available to analyze inbox patterns.",
        "myNetworkTabVisibility": None, # Data not available from provided analysis
        "myNetworkTabAnalysis": "Data not available to analyze network tab patterns.",
        "platformSuggestions": { # Data not available, structure included per schema requirement
            "suggestedPeople": None,
            "suggestedCompaniesOrPages": None,
            "suggestedGroupsEvents": None,
            "suggestedContentOrTopics": None,
            "peopleAlsoViewed": None,
            "otherSuggestions": None
        },
        "platformRecommendationsAnalysis": "No specific platform suggestion data (e.g., suggested connections, content, people also viewed) was provided in the source analysis. Therefore, a detailed analysis of algorithmic recommendations or inferred network gaps based on suggestions is not possible.",
        "detailedConnectionsList": None, # Not provided
        "detailedConnectionsAnalysis": "Detailed connection list not available for analysis.",
        "interestsSectionBasedFollowing": [ # Mapping 'Interests' section
            {"entityName": "Sarah Johnstone", "entityType": "Person", "contextProvided": "Followed Influencer/Top Voice"},
            {"entityName": "Ray Dalio", "entityType": "Person", "contextProvided": "Followed Influencer/Top Voice"}
            # Potentially add companies/schools followed if listed under Interests
        ]
      },
      "privacyPresentation": {
        "accountVisibility": "Mostly public (profile details, activity, featured content visible).",
        "postLevelVisibility": "Assumed Public based on profile visibility.",
        "networkVisibility": None, # Not specified
        "activitySharingSettings": "Likely shares activity (likes, comments) publicly based on visibility.",
        "overallPresentationStyle": "Highly curated, professional."
      },
      "observedConsumption": { # Limited view, focused on profile activity
        "mainFeed": {
            "observedTopics": ["AI/Machine Learning", "Finance/Banking", "Technology Development", "Professional Insights"],
            "frequentPostersAccounts": None, # Not observable from profile view alone
            "contentFormatCharacteristics": "Text updates, Article shares, Images, Videos, Project showcases.",
            "specificContentExamples": [
                "Posts related to AI agents",
                "Articles on engineering/ML"
             ]
        },
        "discoveryFeed": None, # Not applicable/observed for LinkedIn profile view
        "consumptionAnalysisNotes": "Consumption analysis based primarily on user's own activity feed and 'Interests' section. Feed shows professional updates, industry news, and technical content, aligning with generated content. No 'discovery' feed data available for comparison."
      },
      "platformFeatureUsage": [ # Based on analysis
        {"featureName": "Featured Section", "usageDescription": "Used to highlight key projects like Startup Search AI, GraphMemory, Fluency Reef.", "frequency": "Actively Maintained"},
        {"featureName": "Activity Feed (Posting)", "usageDescription": "Shares updates, articles, and project details.", "frequency": "Somewhat Active"},
        {"featureName": "Liking/Commenting", "usageDescription": "Engages with others' posts, particularly on relevant topics.", "frequency": "Moderate"},
        {"featureName": "Following Influencers/Topics", "usageDescription": "Follows key figures (Sarah Johnstone, Ray Dalio) and topics via 'Interests'.", "frequency": "Active"}
      ],
      "platformSpecificConclusions": "LinkedIn presence is highly professional, well-structured, and focused on showcasing technical expertise in AI/ML and finance, alongside entrepreneurial activity. The profile effectively highlights key projects and skills. Engagement seems targeted towards professional networking and knowledge sharing within relevant industries. Privacy is generally public, supporting professional visibility goals. Consumption aligns with professional interests shown in activity. Lack of UI element data (suggestions, network tab) limits deeper algorithmic/network analysis."
    },
    "instagram": {
      "platformName": "Instagram",
      "profileFundamentals": {
        "username": None, # Not provided in analysis
        "fullName": None, # Not provided
        "pronouns": None,
        "location": None,
        "profileURL": None, # Not provided
        "profileLanguage": None,
        "verificationStatus": None,
        "contactInfoVisible": None,
        "profilePictureDescription": None, # Not provided
        "bannerImageDescription": None, # N/A for Instagram profile
        "linkedWebsites": []
      },
      "bioAnalysis": { # No profile details provided, only feed consumption
        "fullText": None,
        "identifiedKeywords": [],
        "hashtagsInBio": [],
        "mentionedUsersInBio": [],
        "statedPurposeFocus": None,
        "tone": None,
        "callToAction": None
      },
      "storyHighlights": [], # Not observed
      "contentGenerationActivity": { # User was consuming, not generating
        "postingFrequency": "Cannot be determined from browsing feed; user is in consumption mode.",
        "dominantContentTypes": [], # Not applicable (observing consumption)
        "contentExamples": [], # Not applicable
        "recurringThemesTopics": [], # Not applicable
        "overallToneVoice": None, # Not applicable
        "recentActivityExamples": [], # Not applicable
        "gridAesthetic": None, # Not applicable
        "reelsPerformanceIndicators": None, # Not applicable
        "storiesFrequencyEngagement": None # Not applicable
      },
      "engagementPatterns": { # User was consuming, not engaging actively in clip
        "outgoingInteractionStyle": "Cannot be determined; not shown interacting in the clip.",
        "typesOfContentEngagedWith": [ # Inferred from consumption
          "Memes",
          "Short humorous videos (Reels)",
          "Absurdist content",
          "Random facts/visuals",
          "Pop culture content"
        ],
        "incomingEngagementHighlights": "Cannot be determined.",
        "typicalIncomingCommentTypes": [] # Cannot be determined
      },
      "networkCommunityRecommendations": {
        "followerCount": None, # Not provided
        "followingCount": None, # Not provided
        "audienceDescription": None, # Not provided
        "groupCommunityMemberships": [], # N/A for standard Instagram
        "inboxSidebarPreview": None, # Data not available
        "inboxSidebarAnalysis": "Data not available.",
        "myNetworkTabVisibility": None, # Data not available
        "myNetworkTabAnalysis": "Data not available.",
        "platformSuggestions": { # Data not available, structure included
            "suggestedPeople": None,
            "suggestedCompaniesOrPages": None,
            "suggestedGroupsEvents": None,
            "suggestedContentOrTopics": None,
            "peopleAlsoViewed": None,
            "otherSuggestions": None
        },
        "platformRecommendationsAnalysis": "No specific platform suggestion data (e.g., suggested accounts, Explore tab themes beyond consumed content) was provided. Analysis is limited to inferring interests based on consumed feed content.",
        "detailedConnectionsList": None,
        "detailedConnectionsAnalysis": "Data not available.",
        "followingDescriptionInfered": "Likely follows a mix of accounts including those sharing memes, short videos, and general entertainment, based on observed feed." # Added based on original analysis text
      },
      "privacyPresentation": {
        "accountVisibility": "Cannot be determined from browsing public feed.",
        "postLevelVisibility": None,
        "networkVisibility": None,
        "activitySharingSettings": None,
        "overallPresentationStyle": None # Cannot assess profile style
      },
      "observedConsumption": { # Primary data available for Instagram
        "mainFeed": None, # Analysis focused on general feed browsing, not distinguishing main/discovery explicitly
        "discoveryFeed": { # Assuming observed feed is akin to discovery/mixed feed
            "observedThemes": [
              "Humor",
              "Absurdity",
              "Meme Culture",
              "Pop Culture (implied)",
              "Random facts/visuals"
            ],
            "prevalentContentTypes": [
              "Images (Memes)",
              "Short videos (Reels)"
            ],
            "commonSoundsEffectsTrends": None, # Not detailed in analysis
            "highlyDescriptiveExamples": [ # From original file's focus example
              "Meme format 'God: Were you happy with your life? Me: Yes' followed by an absurd list (Early life, Controversial thoughts on the Antichrist, Schizophrenia diagnosis, Meme page admin, CIA assassination) from account mr.tom.foolery."
              # Add more if other examples were provided
            ],
            "overallFeedCharacteristics": "Mix of memes, short humorous videos, random facts/visuals, potentially trending content."
        },
        "consumptionAnalysisNotes": "Instagram usage appears heavily focused on consumption of entertainment content, particularly absurdist humor and memes delivered via images and short videos (Reels). This contrasts sharply with the professional focus on LinkedIn. The feed content suggests interests distinct from the professional persona."
      },
      "platformFeatureUsage": [ # Based on observed activity
          {"featureName": "Feed Browsing", "usageDescription": "Actively consuming content from the main/discovery feed.", "frequency": "Observed"},
          {"featureName": "Reels Viewing", "usageDescription": "Consuming short-form video content.", "frequency": "Observed"}
          # No generation or other features observed
      ],
      "platformSpecificConclusions": "Instagram analysis is based solely on observed feed consumption, revealing a preference for informal, humorous, and visually driven content (memes, Reels). This suggests the platform serves primarily an entertainment purpose, distinct from the professional persona on LinkedIn. Algorithmic categorization likely focuses on these entertainment interests. No data is available on the user's own profile, content generation, or network on this platform."
    },
    "twitter": None, # No data provided
    "facebook": None, # No data provided
    "tiktok": None, # No data provided
    "reddit": None, # No data provided
    "otherPlatforms": [] # No other platforms analyzed
  },
  "crossPlatformSynthesis": {
    "consistencyVsVariation": {
      "profileElementConsistency": "Insufficient data for comparison. LinkedIn profile is professional (headshot, relevant banner, detailed bio). Instagram profile elements were not observed.",
      "contentTonePersonaConsistency": "Significant variation observed. LinkedIn features professional, technical, and career-focused content with an informative tone. Observed Instagram activity involves consuming informal, humorous, and entertainment-focused content. This points to a deliberate segmentation of online personas.",
      "notableDifferences": "The primary difference is the clear separation between a professional persona (LinkedIn) focused on career, skills (AI, Finance), and projects, and a personal consumption pattern (Instagram) geared towards humor, memes, and general entertainment."
    },
    "contentOverlapStrategy": "No evidence of cross-posting between LinkedIn and the observed Instagram consumption. Suggests distinct content strategies for each platform, with LinkedIn for professional branding/sharing and Instagram for personal consumption/entertainment.",
    "synthesizedExpertiseInterests": {
      "coreProfessionalSkills": [ # Primarily from LinkedIn
        "AI/Machine Learning",
        "GenAI Engineering",
        "Data Analysis",
        "Finance",
        "Startup Banking",
        "Project Management",
        "Entrepreneurship (Founder)"
      ],
      "corePersonalInterests": [ # From LinkedIn + Instagram consumption
        "Technology Trends",
        "AI Developments",
        "Finance News",
        "Humor/Meme Culture",
        "Absurdist Content",
        "Short-form Video Entertainment"
      ]
    },
    "overallOnlinePersonaNarrative": "The analysis suggests a dual online presence. On professional platforms like LinkedIn, Homin Shum presents as a highly skilled and driven individual focused on the intersection of AI, finance, and technology, actively showcasing projects and engaging with industry topics. On platforms geared towards personal use like Instagram (based on consumption), the persona shifts to a consumer of informal entertainment, particularly humor and memes. This indicates a conscious separation between professional branding and personal leisure interests online.",
    "professionalEvaluation": { # Based *only* on LinkedIn data provided
        "strengthsSkillsMatch": "Strong alignment between stated skills/experience (AI, GenAI, Finance, Startup Banking, Founder) and showcased projects/activity on LinkedIn.",
        "impactAchievements": "Demonstrates initiative and technical capability through featured projects (Startup Search AI, GraphMemory, Fluency Reef) which show engagement.",
        "industryEngagement": "Actively shares and engages with content related to AI/ML and Finance; follows relevant industry figures.",
        "potentialRedFlagsClarifications": "None evident from the limited LinkedIn data provided.",
        "overallCandidateSummary": "Based solely on the LinkedIn profile, presents as a strong candidate with relevant, in-demand skills in AI/ML and Finance, demonstrated project experience, and entrepreneurial initiative. Appears professional and engaged in their field."
    },
    "marketTrendInsights": { # Based *only* on LinkedIn data provided
        "keyTechnologiesToolsTopics": ["AI Agents", "Machine Learning", "Generative AI", "Graph Databases (implied by GraphMemory)", "Financial Technology (FinTech)"],
        "emergingThemesNiches": ["AI application in finance/startups", "Personal AI project development"],
        "relevantContentPatterns": "Focus on technical project showcases, sharing articles/insights on AI/ML advancements, professional networking updates."
    },
    "inferredAlgorithmicPerception": [
      {
        "platformName": "LinkedIn",
        "categorizationHypothesis": "Algorithm likely categorizes user as a professional in the AI/ML, Finance, and Technology sectors. Interests heavily weighted towards: GenAI Engineering, Startup Ecosystem, Financial Services, Data Science, Project Development. Network suggestions likely focus on individuals/companies in these fields, potentially targeting recruiters or peers at similar tech/finance companies. Consumption of professional content reinforces this categorization. (Basis: Profile keywords, headline, experience, education, featured projects, content posted/shared, 'Interests' section following)."
      },
      {
        "platformName": "Instagram",
        "categorizationHypothesis": "Algorithm likely categorizes user based on observed consumption patterns. Interests heavily weighted towards: Humor, Memes, Absurdist Content, Short-form Video (Reels), potentially specific pop culture niches reflected in memes. Feed curation (discovery/reels tab) likely optimized to deliver more content similar to the highly descriptive 'mr.tom.foolery' meme example provided. Network/account suggestions (if shown) would likely include similar meme accounts, comedy creators, or visual entertainment pages. (Basis: Observed consumption in feed, specific meme example provided)."
      }
    ],
    "crossPlatformNetworkAnalysis": { # Insufficient data for deep analysis
        "overlappingConnectionsRecommendations": None, # Cannot compare network/suggestions across platforms
        "networkComparisonNotes": "LinkedIn network appears professional (based on audience description and interests). Instagram network composition is unknown, but following likely geared towards entertainment based on consumption.",
        "consumptionComparisonNotes": "Clear contrast: LinkedIn consumption (inferred from activity/interests) is professional and industry-focused. Instagram consumption (observed) is personal, entertainment-focused (humor, memes, short videos)."
    }
  },
  "finalComprehensiveSummary": "This analysis reveals a distinct dual online presence for Homin Shum. LinkedIn serves as a robust professional platform, meticulously curated to showcase expertise and achievements in AI/ML, Finance, and technology development, supported by detailed profile information and project examples. Engagement here is professional and industry-focused. Conversely, observed activity on Instagram indicates usage primarily for personal entertainment, characterized by the consumption of informal content like humor, memes, and short videos. This clear segmentation suggests an intentional strategy to separate professional branding from personal interests. Algorithmic perceptions likely differ significantly across platforms, with LinkedIn identifying a tech/finance professional and Instagram identifying a consumer of humor/entertainment content. Key strengths lie in the well-defined LinkedIn profile and demonstrated technical projects."
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
    st.title("👤 CaféCorner Profile")

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