# -*- coding: utf-8 -*-
import streamlit as st
import json
from datetime import datetime
import os
import io # Needed for reading uploaded file
import base64 # Needed for encoding uploaded image

# -------------------------------------
# Styling (Based on Design Prompt) - NO CHANGES HERE
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

.coffee-card-header {
    padding-top: 2.5rem;
}

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

.header-text span.pronouns {
    font-size: 0.9rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin-left: 0.5rem;
}
.header-text span.verification-badge {
    font-size: 0.8rem;
    color: var(--accent); /* Default blue, can be overridden */
    margin-left: 0.5rem;
    font-weight: 600;
    vertical-align: middle;
}
.header-text span.verification-badge.twitter-x { color: var(--brass-twitter-x); }
.header-text span.verification-badge.instagram { color: var(--brass-instagram); }
.header-text span.verification-badge.facebook { color: var(--brass-facebook); }
/* Add others if needed */


.header-text p.headline {
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin: 0.25rem 0 0.75rem 0;
    line-height: 1.4;
}

.header-text p.location-links {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin: 0 0 0.5rem 0;
}

.header-text p.location-links a {
    color: var(--accent);
    text-decoration: none;
    margin-left: 0.75rem;
    font-weight: 500;
}
.header-text p.location-links a:hover {
    text-decoration: underline;
}

.analyzed-platforms-container {
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px dashed var(--border-light);
}
.analyzed-platforms-container h5 {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
    font-weight: 500;
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
}
.analyzed-platforms-container .pill-container {
     margin-top: 0;
}


/* --- Section Containers & Platform Cards --- */
.platform-card {
    background: var(--neutral-bg);
    border: 1px solid var(--border-light);
    border-left: 5px solid var(--brass); /* Default accent */
    border-radius: 0 8px 8px 0;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}
.platform-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.06);
}

/* Platform specific accents */
.platform-card.linkedin { border-left-color: var(--brass-linkedin); }
.platform-card.instagram { border-left-color: var(--brass-instagram); }
.platform-card.twitter-x { border-left-color: var(--brass-twitter-x); }
.platform-card.facebook { border-left-color: var(--brass-facebook); }
.platform-card.tiktok { border-left-color: var(--brass-tiktok); }
.platform-card.reddit { border-left-color: var(--brass-reddit); }
.platform-card.github { border-left-color: var(--brass-github); }
.platform-card.other { border-left-color: var(--brass-other); }


.section {
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border-light);
}
.section:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}

/* --- Typography inside sections --- */
h3 { /* Section titles in Overview tab */
    color: var(--coffee);
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    font-size: 1.4rem;
    font-weight: 600;
    border-bottom: 1px solid var(--border-light);
    padding-bottom: 0.5rem;
}

h4 { /* Subsection titles within cards/sections */
    color: var(--coffee);
    margin-top: 1rem;
    margin-bottom: 0.75rem;
    font-size: 1.1rem;
    font-weight: 600;
}
.platform-card h4 {
    margin-top: 0; /* First h4 in card has no top margin */
}
/* Specific title for consumption */
h4.consumption-title {
    margin-top: 1.5rem; /* More space before consumption */
    color: var(--text-secondary); /* Slightly less emphasis */
    font-size: 1.2rem;
    border-top: 1px dashed var(--border-light);
    padding-top: 1rem;
}
h5.consumption-subtitle {
    color: var(--coffee);
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
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

/* --- Alerts & Caveats --- */
.alert {
    border-left: 4px solid var(--alert-border);
    background: var(--alert-bg);
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 4px;
    font-size: 0.9rem;
}
.alert p {
    margin-bottom: 0;
    color: #a94442; /* Darker red for text */
}


/* --- Missing Data --- */
em.missing {
    color: var(--text-missing);
    font-style: italic;
    font-weight: 400; /* Normal weight */
}

/* --- Specific List Styling for UI Elements & Consumption --- */
.ui-element-list, .consumption-list {
    background-color: #ffffff;
    border: 1px solid var(--border-light);
    border-radius: 6px;
    padding: 1rem;
    margin-top: 0.5rem;
}
.ui-element-list li, .consumption-list li {
    padding: 0.5rem 0;
    border-bottom: 1px dashed var(--border-light);
    font-size: 0.9rem;
}
.ui-element-list li:last-child, .consumption-list li:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.ui-element-list strong, .consumption-list strong { /* e.g., Name in inbox preview, Poster name */
    color: var(--coffee);
}
.ui-element-list span, .consumption-list span { /* e.g., Snippet, context, content summary */
    display: block;
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: 0.2rem;
}
.consumption-list.examples li { /* Style for specific examples */
    font-style: italic;
    color: var(--text-secondary);
    border-bottom-style: dotted; /* Differentiate example list */
}

.ui-element-list code { /* Representing data snippets */
    background-color: var(--neutral-pill);
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 0.8em;
}

/* Style for analysis notes */
.analysis-notes {
    margin-top: 1rem;
    font-style: italic;
    color: var(--text-secondary);
    font-size: 0.9rem;
    padding: 0.5rem;
    background-color: #fdfcfb; /* Very subtle background */
    border-radius: 4px;
    border: 1px dashed var(--border-light);
}


/* --- Edit Simulation (Optional Styling) --- */
.steam-and-save-button {
    position: fixed;
    top: 1rem;
    right: 1rem;
    background-color: var(--coffee);
    color: var(--white);
    border: none;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    z-index: 1000; /* Ensure it's on top */
    transition: background-color 0.2s ease;
}
.steam-and-save-button:hover {
    background-color: #3E2723; /* Darker coffee */
}

.editable-text:hover {
    outline: 1px dashed #ccc;
    cursor: text;
    background-color: #f9f9f9;
}

/* Add contenteditable styling if needed for true editing */
[contenteditable="true"] {
    outline: 1px dashed var(--accent);
    background-color: #f0f8ff;
    padding: 2px 4px;
    border-radius: 3px;
}

</style>
"""

# -------------------------------------
# Helper Functions - NO CHANGES HERE
# -------------------------------------

def safe_get(data_dict, key_list, default=None):
    """Safely access nested dictionary keys."""
    current = data_dict
    for key in key_list:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
             current = current[key]
        else:
            return default
    # Ensure the final result isn't None if default wasn't None
    return current if current is not None else default

def render_missing():
    """Returns the HTML for missing data."""
    return '<em class="missing">Not specified</em>'

def render_pills(items, label="Items", show_label=True, pill_class="pill"):
    """Renders a list of strings as HTML pills with optional class."""
    if not items:
        return render_missing()
    # Ensure items are strings and filter out empty ones
    pills_html = "".join(f'<span class="{pill_class}">{str(item)}</span>' for item in items if item and str(item).strip())
    if not pills_html: # Handle case where items might be list of empty strings/None
        return render_missing()
    label_html = f'<h4>{label}</h4>' if show_label else ''
    # Add pill-container class for consistent styling
    return f'{label_html}<div class="pill-container">{pills_html}</div>'


def render_list(items, list_class="", empty_message="No items listed."):
    """Renders a simple HTML unordered list with optional class."""
    if not items:
        return f"<p>{render_missing()}</p>"
    # Filter out empty/None items before joining
    valid_items = [item for item in items if item and str(item).strip()]
    if not valid_items:
         return f"<p>{render_missing()}</p>"
    list_html = f"<ul class='{list_class}'>" + "".join(f"<li>{item}</li>" for item in valid_items) + "</ul>"
    return list_html

def render_posters(posters_list):
    """Renders the observed posters list."""
    if not posters_list:
        return f"<p>{render_missing()}</p>"
    html = '<ul class="consumption-list">'
    for item in posters_list:
        name = safe_get(item, ['posterName'], 'N/A')
        summary = safe_get(item, ['exampleContentSummary'], '')
        html += f'<li><strong>{name}</strong>'
        if summary:
            html += f'<span>Example Content: {summary}</span>'
        html += '</li>'
    html += '</ul>'
    return html

def render_dict_list_html(items, title_key, desc_key, link_key=None, extra_key=None, extra_label=None):
    """Renders a list of dictionaries into a structured HTML list."""
    if not items:
        return f"<p>{render_missing()}</p>"

    html = '<ul class="ui-element-list">'
    for item in items:
        title = safe_get(item, [title_key], 'N/A')
        description = safe_get(item, [desc_key], '')
        link = safe_get(item, [link_key], '') if link_key else ''
        extra = safe_get(item, [extra_key], '') if extra_key else ''

        html += f'<li><strong>{title}</strong>'
        if description:
            html += f'<span>{description}</span>'
        if link:
             # Basic check for valid looking URL before making it clickable
             if isinstance(link, str) and (link.startswith('http://') or link.startswith('https://')):
                 html += f'<span><a href="{link}" target="_blank">Link</a></span>'
             else:
                 html += f'<span>Link: {link}</span>' # Display non-URL link as text
        if extra and extra_label:
             # Special handling for list-type extras (like associatedSkills)
             if isinstance(extra, list):
                 if extra: # Only show if list is not empty
                      html += f'<span>{extra_label}: {", ".join(str(e) for e in extra)}</span>'
             else:
                 html += f'<span>{extra_label}: {extra}</span>'
        html += '</li>'
    html += '</ul>'
    return html

def render_platform_suggestions(suggestions):
    """Renders the complex platform suggestions object."""
    if not suggestions:
        return f"<p>{render_missing()}</p>"

    html = ""
    # Added usageCount and followerCount handling
    suggestion_map = {
        "suggestedPeople": ("Suggested People", "name", "headlineOrDetail", "profileURL", "reasonForSuggestion", "Reason"),
        "suggestedCompaniesOrPages": ("Suggested Companies/Pages", "name", "descriptionOrIndustry", "pageURL", "followerCount", "Followers"),
        "suggestedGroups": ("Suggested Groups", "name", "topicOrMemberCount", "groupURL", "reasonForSuggestion", "Reason"),
        "peopleAlsoViewed": ("People Also Viewed", "name", "headlineOrDetail", "profileURL", "reasonForSuggestion", "Reason"),
        "otherSuggestions": ("Other Suggestions", "nameOrTitle", "description", "link", "usageCount", "Usage Count") # Assuming usageCount is the extra field
    }

    has_suggestions = False
    for key, (title, title_key, desc_key, link_key, extra_key, extra_label) in suggestion_map.items():
        items = safe_get(suggestions, [key], [])
        if items:
            has_suggestions = True
            html += f"<h4>{title}</h4>"
            html += render_dict_list_html(items, title_key, desc_key, link_key, extra_key, extra_label)

    return html if has_suggestions else f"<p>{render_missing()}</p>"

def render_professional_history(history):
    """Renders Experience, Education, Projects, Licenses & Certs."""
    if not history:
        return f"<p>{render_missing()}</p>"
    html = ""
    has_content = False
    # Experience
    experience = safe_get(history, ['experience'], [])
    if experience:
        has_content = True
        html += "<h4>Experience</h4>"
        html += '<ul class="ui-element-list">'
        for item in experience:
            title = safe_get(item, ['title'], 'N/A')
            org = safe_get(item, ['organization'], 'N/A')
            duration = safe_get(item, ['duration'], '')
            dates = safe_get(item, ['dates'], '')
            loc = safe_get(item, ['location'], '')
            desc = safe_get(item, ['description'], '')
            skills = safe_get(item, ['associatedSkills'], [])

            timeframe = f"{dates}" if dates else f"{duration}" if duration else ""
            html += f'<li><strong>{title} at {org}</strong>'
            if timeframe: html += f'<span>{timeframe}</span>'
            if loc: html += f'<span>{loc}</span>'
            if desc: html += f'<span style="white-space: pre-wrap;">{desc}</span>' # Preserve whitespace
            if skills: html += f'<span>Skills: {", ".join(skills)}</span>'
            html += '</li>'
        html += '</ul>'

    # Education
    education = safe_get(history, ['education'], [])
    if education:
        has_content = True
        html += "<h4>Education</h4>"
        # Add 'activities' rendering
        html += render_dict_list_html(education, 'institution', 'degreeField', None, 'activities', 'Activities')


    # Projects
    projects = safe_get(history, ['projects'], [])
    if projects:
        has_content = True
        html += "<h4>Projects</h4>"
        # Add 'contributors' and 'associatedSkills' rendering
        html += '<ul class="ui-element-list">'
        for item in projects:
            name = safe_get(item, ['name'], 'N/A')
            desc = safe_get(item, ['description'], '')
            link = safe_get(item, ['link'], '')
            dates = safe_get(item, ['dates'], '')
            skills = safe_get(item, ['associatedSkills'], [])
            contrib = safe_get(item, ['contributors'], [])

            html += f'<li><strong>{name}</strong>'
            if dates: html += f'<span>{dates}</span>'
            if desc: html += f'<span>{desc}</span>'
            if link:
                if isinstance(link, str) and (link.startswith('http://') or link.startswith('https://')):
                     html += f'<span><a href="{link}" target="_blank">Link</a></span>'
                else:
                     html += f'<span>Link: {link}</span>'
            if skills: html += f'<span>Skills: {", ".join(skills)}</span>'
            if contrib: html += f'<span>Contributors: {", ".join(contrib)}</span>'
            html += '</li>'
        html += '</ul>'


    # --- Handle LinkedIn Specific Licenses & Certs ---
    licenses = safe_get(history, ['licensesCertifications'], [])
    if licenses:
        has_content = True
        html += "<h4>Licenses & Certifications</h4>"
        # Assuming licenses have 'name' and 'issuer', potentially 'date', 'url'
        html += render_dict_list_html(licenses, 'name', 'issuer', 'url', 'date', 'Date')

    return html if has_content else f"<p>{render_missing()}</p>"

# -------------------------------------
# Rendering Functions for Page Sections - MODIFIED HEADER
# -------------------------------------

# --- Modified render_coffee_card_header to accept uploaded_avatar_data ---
def render_coffee_card_header(data, uploaded_avatar_data=None):
    """
    Renders the main header card, including analyzed platforms.
    Accepts uploaded_avatar_data (BytesIO or similar) for custom avatar.
    """
    target_name = safe_get(data, ['targetIndividual'], "Individual")
    analyzed_platforms = safe_get(data, ['analyzedPlatforms'], [])

    # --- Logic for determining profile info (headline, location, etc.) ---
    headline = render_missing()
    pronouns = None
    location = f'<span>📍 {render_missing()}</span>' # Default to missing HTML span
    linked_websites = []
    verification_status = None
    verification_platform = None # Track which platform provided verification

    for platform_data in safe_get(data, ['platformSpecificAnalysis'], []):
        profile_fund = safe_get(platform_data, ['profileFundamentals'], {})
        bio_analysis = safe_get(platform_data, ['bioAnalysis'], {})

        current_headline = safe_get(bio_analysis, ['statedPurposeFocus']) or safe_get(bio_analysis, ['fullText'])
        current_pronouns = safe_get(profile_fund, ['pronouns'])
        current_location = safe_get(profile_fund, ['location'])
        current_websites = safe_get(profile_fund, ['linkedWebsites'], [])
        current_verification = safe_get(profile_fund, ['verificationStatus']) # Can be bool, string, null

        if current_headline and headline == render_missing():
            headline = current_headline
        if current_pronouns and not pronouns:
            pronouns = current_pronouns
        if current_location and location == f'<span>📍 {render_missing()}</span>':
            location = f'<span>📍 {current_location}</span>'
        if current_websites: # Aggregate websites from all platforms
            linked_websites.extend(w for w in current_websites if w not in linked_websites and w) # ensure unique and not None/empty
        # Take the first non-null verification status found
        if current_verification is not None and verification_status is None:
             verification_status = current_verification
             verification_platform = safe_get(platform_data, ['platformName'], '').lower().replace(" ", "-").replace("/", "-").replace("+", "-").replace(".", "")

    # --- Prepare HTML snippets ---
    pronouns_html = f'<span class="pronouns">({pronouns})</span>' if pronouns else ''
    headline_html = f'<p class="headline">{headline}</p>' # Will show 'Not specified' if never found

    # Verification Badge
    verification_html = ''
    if verification_status: # Check if True or a non-empty string indicating verified
        badge_text = "✔️ Verified"
        if isinstance(verification_status, str) and verification_status.strip():
            badge_text = f"✔️ {verification_status}" # Use platform's text if provided
        platform_class = f"{verification_platform}" if verification_platform else ""
        verification_html = f'<span class="verification-badge {platform_class}">{badge_text}</span>'

    links_html = ""
    if linked_websites:
        links_html = " | ".join(f'<a href="{url}" target="_blank">{url.replace("https://","").replace("http://","").split("/")[0]}</a>' for url in linked_websites)
        links_html = f' | {links_html}' # Add separator

    # --- Avatar Logic: Use uploaded image or placeholder ---
    avatar_url = "https://via.placeholder.com/80x80.png?text=?" # Default placeholder
    if uploaded_avatar_data is not None:
        try:
            # Get image bytes and encode to Base64
            image_bytes = uploaded_avatar_data.getvalue() # Use getvalue() for BytesIO from upload
            b64_image = base64.b64encode(image_bytes).decode()
            # Get MIME type from uploaded file object
            mime_type = uploaded_avatar_data.type
            avatar_url = f"data:{mime_type};base64,{b64_image}"
        except Exception as e:
            st.warning(f"Could not process uploaded avatar image: {e}. Using placeholder.")
            # avatar_url remains the placeholder

    # Render Analyzed Platforms Pills
    platforms_pills_html = render_pills(analyzed_platforms, show_label=False)

    # --- Final Header HTML ---
    st.session_state['avatar_url'] = avatar_url
    st.session_state['target_name'] = target_name
    st.session_state['pronouns'] = pronouns
    st.session_state['location'] = location
    st.session_state['linked_websites'] = linked_websites
    st.session_state['verification_status'] = verification_status
    st.session_state['verification_platform'] = verification_platform
    st.session_state['pronouns_html'] = pronouns_html
    st.session_state['verification_html'] = verification_html
    st.session_state['headline_html'] = headline_html
    st.session_state['links_html'] = links_html
    st.session_state['platforms_pills_html'] = platforms_pills_html

    # --- Render Header ---
    st.markdown(f"""
    <div class="card coffee-card-header">
        <div class="progress-bar-container">
            <div class="progress-bar-fill"></div>
        </div>
        <div class="header-content">
            <img src="{st.session_state['avatar_url']}" alt="Avatar" class="avatar">
            <div class="header-text">
                <h1>{st.session_state['target_name']} {st.session_state['pronouns_html']} {st.session_state['verification_html']}</h1>
                {st.session_state['headline_html']}
                <p class="location-links">
                    {st.session_state['location']}
                    {st.session_state['links_html']}
                </p>
                <div class="analyzed-platforms-container">
                <h5>Platforms Analyzed:</h5>
                {st.session_state['platforms_pills_html']}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- render_overview_tab - NO CHANGES NEEDED HERE ---
def render_overview_tab(data):
    """Renders the content for the Overview/Synthesis tab."""
    st.markdown("<h2>🔄 Cross-Platform Synthesis</h2>", unsafe_allow_html=True)
    st.caption("Your cross-platform synthesis analysis.")
    synthesis = safe_get(data, ['crossPlatformSynthesis'], {})

    if not synthesis:
        st.markdown(render_missing(), unsafe_allow_html=True)
        st.markdown("<h2>Final Summary</h2>", unsafe_allow_html=True)
        st.markdown(safe_get(data, ['finalComprehensiveSummary'], render_missing()), unsafe_allow_html=True)
        return

    # Consistency vs Variation
    consistency = safe_get(synthesis, ['consistencyVsVariation'], {})
    if consistency:
        st.markdown("<h3>Consistency vs. Variation</h3>", unsafe_allow_html=True)
        st.markdown(f"**Profile Elements:** {safe_get(consistency, ['profileElementConsistency'], render_missing())}", unsafe_allow_html=True)
        st.markdown(f"**Content/Tone/Persona:** {safe_get(consistency, ['contentTonePersonaConsistency'], render_missing())}", unsafe_allow_html=True)
        st.markdown(f"**Notable Differences:** {safe_get(consistency, ['notableDifferences'], render_missing())}", unsafe_allow_html=True)

    # Content Overlap Strategy
    st.markdown("<h3>Content Strategy</h3>", unsafe_allow_html=True)
    st.markdown(safe_get(synthesis, ['contentOverlapStrategy'], render_missing()), unsafe_allow_html=True)

    # Synthesized Expertise & Interests
    expertise = safe_get(synthesis, ['synthesizedExpertiseInterests'], {})
    if expertise:
        st.markdown(render_pills(safe_get(expertise, ['coreProfessionalSkills'], []), label="Synthesized Professional Skills"), unsafe_allow_html=True)
        st.markdown(render_pills(safe_get(expertise, ['corePersonalInterests'], []), label="Synthesized Personal Interests"), unsafe_allow_html=True)

    # Overall Persona Narrative
    st.markdown("<h3>Overall Online Persona Narrative</h3>", unsafe_allow_html=True)
    st.markdown(safe_get(synthesis, ['overallOnlinePersonaNarrative'], render_missing()), unsafe_allow_html=True)

    # Professional Evaluation (if present)
    prof_eval = safe_get(synthesis, ['professionalEvaluation'])
    if prof_eval:
        st.markdown("<h3>Professional Evaluation</h3>", unsafe_allow_html=True)
        st.markdown(f"**Strengths/Skills Match:** {safe_get(prof_eval, ['strengthsSkillsMatch'], render_missing())}", unsafe_allow_html=True)
        st.markdown(f"**Impact/Achievements:** {safe_get(prof_eval, ['impactAchievements'], render_missing())}", unsafe_allow_html=True)
        st.markdown(f"**Industry Engagement:** {safe_get(prof_eval, ['industryEngagement'], render_missing())}", unsafe_allow_html=True)
        st.markdown(f"**Potential Red Flags/Clarifications:** {safe_get(prof_eval, ['potentialRedFlagsClarifications'], render_missing())}", unsafe_allow_html=True)
        st.markdown(f"**Overall Candidate Summary:** {safe_get(prof_eval, ['overallCandidateSummary'], render_missing())}", unsafe_allow_html=True)

    # Market Trend Insights (if present)
    market_insights = safe_get(synthesis, ['marketTrendInsights'])
    if market_insights:
        st.markdown("<h3>Market Trend Insights</h3>", unsafe_allow_html=True)
        st.markdown(render_pills(safe_get(market_insights, ['keyTechnologiesToolsTopics'], []), label="Observed Technologies/Tools/Topics"), unsafe_allow_html=True)
        st.markdown(render_pills(safe_get(market_insights, ['emergingThemesNiches'], []), label="Emerging Themes/Niches"), unsafe_allow_html=True)
        st.markdown(f"**Relevant Content Patterns:** {safe_get(market_insights, ['relevantContentPatterns'], render_missing())}", unsafe_allow_html=True)

    # Inferred Algorithmic Perception
    algo_perception = safe_get(synthesis, ['inferredAlgorithmicPerception'], [])
    if algo_perception:
        st.markdown("<h3>Inferred Algorithmic Perception</h3>", unsafe_allow_html=True)
        html = '<ul class="ui-element-list">'
        for item in algo_perception:
            platform = safe_get(item, ['platformName'], 'N/A')
            hypothesis = safe_get(item, ['categorizationHypothesis'], render_missing())
            html += f'<li><strong>{platform}:</strong> <span>{hypothesis}</span></li>'
        html += '</ul>'
        st.markdown(html, unsafe_allow_html=True)

    # Cross-Platform Network Analysis (if present) - ADDED consumptionComparisonNotes
    cross_network = safe_get(synthesis, ['crossPlatformNetworkAnalysis'])
    if cross_network:
        st.markdown("<h3>Cross-Platform Network & Consumption Analysis</h3>", unsafe_allow_html=True)
        overlaps = safe_get(cross_network, ['overlappingConnectionsRecommendations'], [])
        if overlaps:
            st.markdown("<h5>Overlapping Connections/Recommendations</h5>", unsafe_allow_html=True)
            html = '<ul class="ui-element-list">'
            for item in overlaps:
                 name = safe_get(item, ['entityName'], 'N/A')
                 platforms = ", ".join(safe_get(item, ['appearingOnPlatforms'], []))
                 html += f'<li><strong>{name}</strong> <span>Observed on: {platforms}</span></li>'
            html += '</ul>'
            st.markdown(html, unsafe_allow_html=True)
        st.markdown(f"**Network Comparison Notes:** {safe_get(cross_network, ['networkComparisonNotes'], render_missing())}", unsafe_allow_html=True)
        # Render consumption notes
        st.markdown(f"**Consumption Comparison Notes:** {safe_get(cross_network, ['consumptionComparisonNotes'], render_missing())}", unsafe_allow_html=True)


    # Final Comprehensive Summary
    st.markdown("<h3>Final Summary</h3>", unsafe_allow_html=True)
    summary = safe_get(data, ['finalComprehensiveSummary'])
    if summary:
        st.markdown(summary, unsafe_allow_html=True)
    else:
        st.markdown(render_missing(), unsafe_allow_html=True)

# --- render_platform_tab - NO CHANGES NEEDED HERE ---
def render_platform_tab(platform_data):
    """
    Renders the content for a single platform tab using common structures.
    Handles LinkedIn specific skills/interests object.
    Renders the detailed observedConsumption structure.
    """
    platform_name = safe_get(platform_data, ['platformName'], 'Unknown Platform')   # Get platform name
    platform_class = platform_name.lower().replace(" ", "-").replace("/", "-").replace("+", "-").replace(".", "")
    if not platform_class: platform_class="other" # Use 'other' if class generation fails

    # Use platform_class in the outer div for styling
    # st.markdown(f'<div class="platform-card {platform_class}">', unsafe_allow_html=True)

    # --- Render COMMON STRUCTURES ---

    # Profile Fundamentals (from commonStructures)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown(f"<h4>🧑‍💻 {platform_name} Profile Fundamentals</h4>", unsafe_allow_html=True)
    st.caption(f"Your {platform_class} profile analysis.")
    profile_fund = safe_get(platform_data, ['profileFundamentals'], {})
    if profile_fund:
        username = safe_get(profile_fund, ['username'])
        fullname = safe_get(profile_fund, ['fullName'])
        pronouns = safe_get(profile_fund, ['pronouns'])
        location = safe_get(profile_fund, ['location'])
        profile_url = safe_get(profile_fund, ['profileURL'])
        language = safe_get(profile_fund, ['profileLanguage'])
        verification = safe_get(profile_fund, ['verificationStatus'])
        contact_vis = safe_get(profile_fund, ['contactInfoVisible'])
        pic_desc = safe_get(profile_fund, ['profilePictureDescription'])
        banner_desc = safe_get(profile_fund, ['bannerImageDescription'])
        websites = safe_get(profile_fund, ['linkedWebsites'], [])

        st.markdown(f"**Username:** {username if username else render_missing()}", unsafe_allow_html=True)
        if fullname and fullname != username: st.markdown(f"**Full Name:** {fullname}", unsafe_allow_html=True)
        if pronouns: st.markdown(f"**Pronouns:** {pronouns}", unsafe_allow_html=True)
        if location: st.markdown(f"**Location:** {location}", unsafe_allow_html=True)
        if verification is not None:
             badge_text = "Yes" if verification else "No"
             if isinstance(verification, str) and verification.strip():
                 badge_text = f"Yes ({verification})" # Include platform specific string if available
             st.markdown(f"**Verified:** {badge_text}", unsafe_allow_html=True)
        if profile_url: st.markdown(f'**Profile URL:** <a href="{profile_url}" target="_blank">{profile_url}</a>', unsafe_allow_html=True)
        if language: st.markdown(f"**Profile Language:** {language}", unsafe_allow_html=True)
        if contact_vis is not None: st.markdown(f"**Contact Info Visible:** {'Yes' if contact_vis else 'No'}", unsafe_allow_html=True)
        if pic_desc: st.markdown(f"**Profile Picture:** {pic_desc}", unsafe_allow_html=True)
        if banner_desc: st.markdown(f"**Banner Image:** {banner_desc}", unsafe_allow_html=True)
        if websites: st.markdown(f"**Linked Websites:** {render_list(websites)}", unsafe_allow_html=True)
    else:
         st.markdown(render_missing(), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


    # Bio Analysis (from commonStructures)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("<h4>Bio / Headline / About</h4>", unsafe_allow_html=True)
    bio_analysis = safe_get(platform_data, ['bioAnalysis'], {})
    if bio_analysis:
        full_text = safe_get(bio_analysis, ['fullText'])
        keywords = safe_get(bio_analysis, ['identifiedKeywords'], [])
        hashtags = safe_get(bio_analysis, ['hashtagsInBio'], [])
        mentions = safe_get(bio_analysis, ['mentionedUsersInBio'], []) # Added
        purpose = safe_get(bio_analysis, ['statedPurposeFocus'])
        tone = safe_get(bio_analysis, ['tone'])
        cta = safe_get(bio_analysis, ['callToAction'])

        if full_text:
            with st.expander("View Full Text", expanded=False):
                 st.markdown(f'<p style="white-space: pre-wrap;">{full_text}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p>{render_missing()}</p>', unsafe_allow_html=True) # Show missing if no full text

        if keywords: st.markdown(render_pills(keywords, label="Identified Keywords", show_label=False), unsafe_allow_html=True)
        if hashtags: st.markdown(render_pills(hashtags, label="Bio Hashtags", show_label=False), unsafe_allow_html=True)
        if mentions: st.markdown(f"**Mentioned Users:** {', '.join(mentions)}", unsafe_allow_html=True)
        if purpose: st.markdown(f"**Stated Purpose/Focus:** {purpose}", unsafe_allow_html=True)
        if tone: st.markdown(f"**Tone:** {tone}", unsafe_allow_html=True)
        if cta: st.markdown(f"**Call to Action:** {cta}", unsafe_allow_html=True)
        # Only show 'missing' if NO bio data exists at all
        if not any([full_text, keywords, hashtags, mentions, purpose, tone, cta]):
             st.markdown(render_missing(), unsafe_allow_html=True)
    else:
         st.markdown(render_missing(), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- PLATFORM SPECIFIC STRUCTURES (Render conditionally) ---

    # Skills/Interests/Expertise (Handle LinkedIn object vs. array)
    skills_data = safe_get(platform_data, ['skillsInterestsExpertise']) # Could be object or array
    if skills_data:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("<h4>Skills, Interests & Expertise</h4>", unsafe_allow_html=True)
        if platform_name == "LinkedIn" and isinstance(skills_data, dict):
            # LinkedIn Specific Rendering
            skills_list = safe_get(skills_data, ['skillsList'], [])
            if skills_list:
                st.markdown("<h5>Skills</h5>", unsafe_allow_html=True)
                html = '<ul class="ui-element-list">'
                for skill_item in skills_list:
                    name = safe_get(skill_item, ['skillName'])
                    count = safe_get(skill_item, ['endorsementCount'])
                    count_str = f" ({count} endorsements)" if count is not None else ""
                    if name: html += f'<li><strong>{name}</strong>{count_str}</li>'
                html += '</ul>'
                st.markdown(html, unsafe_allow_html=True)

            licenses = safe_get(skills_data, ['licensesCertifications'], [])
            if licenses:
                 st.markdown("<h5>Licenses & Certifications</h5>", unsafe_allow_html=True)
                 st.markdown(render_dict_list_html(licenses, 'name', 'issuer', None, 'date', 'Date'), unsafe_allow_html=True)

            courses = safe_get(skills_data, ['courses'], [])
            if courses:
                 st.markdown("<h5>Courses</h5>", unsafe_allow_html=True)
                 st.markdown(render_list(courses), unsafe_allow_html=True)

            st.markdown("<h5>Interests</h5>", unsafe_allow_html=True)
            influencers = safe_get(skills_data, ['followedInfluencers'], [])
            companies = safe_get(skills_data, ['followedCompanies'], [])
            groups = safe_get(skills_data, ['followedGroups'], [])
            schools = safe_get(skills_data, ['followedSchools'], [])
            interests_found = False
            if influencers:
                st.markdown(f"**Followed Influencers:** {', '.join(influencers)}", unsafe_allow_html=True)
                interests_found = True
            if companies:
                st.markdown(f"**Followed Companies:** {', '.join(companies)}", unsafe_allow_html=True)
                interests_found = True
            if groups:
                st.markdown(f"**Followed Groups:** {', '.join(groups)}", unsafe_allow_html=True)
                interests_found = True
            if schools:
                st.markdown(f"**Followed Schools:** {', '.join(schools)}", unsafe_allow_html=True)
                interests_found = True
            if not interests_found: st.markdown(render_missing(), unsafe_allow_html=True)

        elif isinstance(skills_data, list):
            # Generic rendering for simple arrays
            st.markdown(render_pills(skills_data, show_label=False), unsafe_allow_html=True)
        else:
            # Fallback if it's neither list nor expected object type
            st.markdown(render_missing(), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


    # Featured/Pinned Content (LinkedIn specific, but render if found)
    featured = safe_get(platform_data, ['featuredPinnedContent'], [])
    if featured:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("<h4>Featured/Pinned Content</h4>", unsafe_allow_html=True)
        st.markdown(render_dict_list_html(featured, 'title', 'description', 'link', 'contentType', 'Type'), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Professional History (LinkedIn/Facebook specific, but render if found)
    prof_hist = safe_get(platform_data, ['professionalHistory'])
    if prof_hist and any(safe_get(prof_hist, [key], []) for key in ['experience', 'education', 'projects', 'licensesCertifications']):
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("<h4>Professional History</h4>", unsafe_allow_html=True)
        st.markdown(render_professional_history(prof_hist), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- RENDER REMAINING COMMON STRUCTURES ---

    # Content Generation Activity (from commonStructures)
    content_gen = safe_get(platform_data, ['contentGenerationActivity'], {})
    renderable_gen_keys = ['postingFrequency', 'dominantContentTypes', 'recurringThemesTopics', 'overallToneVoice', 'contentExamples', 'recentActivityExamples']
    if content_gen and any(safe_get(content_gen, [key]) for key in renderable_gen_keys):
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("<h4>Content Generation & Activity</h4>", unsafe_allow_html=True)
        freq = safe_get(content_gen, ['postingFrequency'])
        dom_types = safe_get(content_gen, ['dominantContentTypes'], [])
        themes = safe_get(content_gen, ['recurringThemesTopics'], [])
        tone_voice = safe_get(content_gen, ['overallToneVoice'])
        examples = safe_get(content_gen, ['contentExamples'], [])
        recent_examples = safe_get(content_gen, ['recentActivityExamples'], [])

        if freq: st.markdown(f"**Posting Frequency:** {freq}", unsafe_allow_html=True)
        if dom_types: st.markdown(f"**Dominant Content Types:** {', '.join(dom_types)}", unsafe_allow_html=True)
        if themes: st.markdown(f"**Recurring Themes/Topics:** {render_pills(themes, show_label=False)}", unsafe_allow_html=True)
        if tone_voice: st.markdown(f"**Overall Tone/Voice:** {tone_voice}", unsafe_allow_html=True)
        if examples:
            with st.expander("View Content Examples"):
                 st.markdown(render_list(examples), unsafe_allow_html=True)
        if recent_examples:
             with st.expander("View Recent Activity Examples"):
                 st.markdown(render_list(recent_examples), unsafe_allow_html=True)

        # NOTE: Platform-specific content fields (e.g., gridAesthetic, soundsUsed) are ignored here

        if not any([freq, dom_types, themes, tone_voice, examples, recent_examples]):
            st.markdown(render_missing(), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


    # Engagement Patterns (from commonStructures)
    engagement = safe_get(platform_data, ['engagementPatterns'], {})
    renderable_eng_keys = ['outgoingInteractionStyle', 'typesOfContentEngagedWith', 'incomingEngagementHighlights', 'typicalIncomingCommentTypes']
    if engagement and any(safe_get(engagement, [key]) for key in renderable_eng_keys):
         st.markdown('<div class="section">', unsafe_allow_html=True)
         st.markdown("<h4>Engagement Patterns</h4>", unsafe_allow_html=True)
         outgoing = safe_get(engagement, ['outgoingInteractionStyle'])
         engaged_with = safe_get(engagement, ['typesOfContentEngagedWith'], [])
         incoming_hl = safe_get(engagement, ['incomingEngagementHighlights'])
         incoming_comments = safe_get(engagement, ['typicalIncomingCommentTypes'], [])

         if outgoing: st.markdown(f"**Outgoing Interaction Style:** {outgoing}", unsafe_allow_html=True)
         if engaged_with: st.markdown(f"**Engages With:** {render_pills(engaged_with, show_label=False)}", unsafe_allow_html=True)
         if incoming_hl: st.markdown(f"**Incoming Engagement Highlights:** {incoming_hl}", unsafe_allow_html=True)
         if incoming_comments: st.markdown(f"**Typical Incoming Comment Types:** {render_pills(incoming_comments, show_label=False)}", unsafe_allow_html=True)
         if not any([outgoing, engaged_with, incoming_hl, incoming_comments]):
              st.markdown(render_missing(), unsafe_allow_html=True)
         st.markdown('</div>', unsafe_allow_html=True)

    # Network, Community, Recommendations (from commonStructures)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("<h4>Network, Community & Recommendations</h4>", unsafe_allow_html=True)
    network_data = safe_get(platform_data, ['networkCommunityRecommendations'], {})
    if network_data: # Check if the object exists
        followers = safe_get(network_data, ['followerCount'])
        following = safe_get(network_data, ['followingCount'])
        audience = safe_get(network_data, ['audienceDescription'])
        groups = safe_get(network_data, ['groupCommunityMemberships'], []) # Common structure now

        # Basic Network Stats
        network_stats_html = ""
        if followers is not None: network_stats_html += f"**Followers/Connections:** {followers} "
        if following is not None: network_stats_html += f"**Following:** {following}"
        if network_stats_html: st.markdown(network_stats_html.strip(), unsafe_allow_html=True)

        if audience: st.markdown(f"**Audience Description:** {audience}", unsafe_allow_html=True)

        # Render groups/communities/subscribed subreddits
        group_title = "Group/Community Memberships"
        if platform_name == "Reddit": group_title = "Subscribed Subreddits"
        elif platform_name == "Twitter/X": group_title = "List Memberships" # Example adaptation
        if groups:
            st.markdown(f"<h5>{group_title}</h5>", unsafe_allow_html=True)
            # Render using the groupMembershipItem structure
            st.markdown(render_dict_list_html(groups, 'groupName', 'topic', 'groupURL', 'activityLevel', 'Activity'), unsafe_allow_html=True)

        # --- MANDATORY UI Element Sections ---
        st.markdown("<h5>Platform Interface Observations</h5>", unsafe_allow_html=True)

        # Inbox Sidebar Preview (common)
        inbox_preview = safe_get(network_data, ['inboxSidebarPreview']) # Don't default list yet
        inbox_analysis = safe_get(network_data, ['inboxSidebarAnalysis'])
        inbox_title = "Inbox/DM Sidebar Preview"
        # Platform-specific title variations
        if platform_name == "LinkedIn": inbox_title = "Messaging Inbox Preview"
        elif platform_name == "Reddit": inbox_title = "Chat/Messages Preview"
        st.markdown(f"<h6>{inbox_title}</h6>", unsafe_allow_html=True)
        if inbox_preview is not None: # Check if the key exists
            if len(inbox_preview) > 0:
                 st.markdown(render_dict_list_html(inbox_preview, 'name', 'detailSnippet', 'conversationURL', 'timestamp', 'Timestamp'), unsafe_allow_html=True)
                 if inbox_analysis: st.markdown(f'<div class="analysis-notes">Analysis: {inbox_analysis}</div>', unsafe_allow_html=True)
            else: # Empty list means observed but empty
                 st.markdown(f"<p>{render_missing()} (Observed but empty)</p>", unsafe_allow_html=True)
        else: # Key was missing entirely
             st.markdown(f"<p>{render_missing()} (UI element potentially not visible or captured)</p>", unsafe_allow_html=True)


        # My Network Tab Visibility (common)
        network_tab = safe_get(network_data, ['myNetworkTabVisibility'])
        network_tab_analysis = safe_get(network_data, ['myNetworkTabAnalysis'])
        network_title = "'My Network' / Connections Tab"
        # Platform-specific title variations
        if platform_name == "LinkedIn": network_title = "'My Network' Tab"
        elif platform_name == "Facebook": network_title = "Friends Tab / Suggestions"
        st.markdown(f"<h6>{network_title}</h6>", unsafe_allow_html=True)
        if network_tab is not None:
            if len(network_tab) > 0:
                 st.markdown(render_dict_list_html(network_tab, 'entityName', 'headlineOrDetail', 'entityURL', 'context', 'Context'), unsafe_allow_html=True)
                 if network_tab_analysis: st.markdown(f'<div class="analysis-notes">Analysis: {network_tab_analysis}</div>', unsafe_allow_html=True)
            else:
                 st.markdown(f"<p>{render_missing()} (Observed but empty)</p>", unsafe_allow_html=True)
        else:
             st.markdown(f"<p>{render_missing()} (UI element potentially not visible or captured)</p>", unsafe_allow_html=True)


        # Platform Suggestions / Recommendations (common)
        suggestions = safe_get(network_data, ['platformSuggestions'])
        recommendations_analysis = safe_get(network_data, ['platformRecommendationsAnalysis'])
        st.markdown("<h6>Platform Suggestions & Recommendations</h6>", unsafe_allow_html=True)
        if suggestions is not None: # Check if suggestions object exists
            suggestions_html = render_platform_suggestions(suggestions) # Handles empty sub-lists
            if suggestions_html != f"<p>{render_missing()}</p>":
                 st.markdown(suggestions_html, unsafe_allow_html=True)
                 if recommendations_analysis: st.markdown(f'<div class="analysis-notes">Overall Analysis: {recommendations_analysis}</div>', unsafe_allow_html=True)
            else: # Suggestions object exists but contains no actual suggestions
                 st.markdown(f"<p>{render_missing()} (No suggestions observed)</p>", unsafe_allow_html=True)
        else: # suggestions key itself was missing
             st.markdown(f"<p>{render_missing()} (UI elements potentially not visible or captured)</p>", unsafe_allow_html=True)


        # Detailed Connections (common)
        connections = safe_get(network_data, ['detailedConnectionsList']) # Can be null
        connections_analysis = safe_get(network_data, ['detailedConnectionsAnalysis'])
        st.markdown("<h6>Detailed Connections List</h6>", unsafe_allow_html=True)
        if connections:
            st.markdown(render_dict_list_html(connections, 'connectionName', 'connectionHeadline', 'connectionProfileURL', 'connectionDate', 'Connected'), unsafe_allow_html=True)
            if connections_analysis: st.markdown(f'<div class="analysis-notes">Analysis: {connections_analysis}</div>', unsafe_allow_html=True)
        elif connections is None:
             st.markdown(f"<p><em>{render_missing()} (Detailed connection data not provided in input)</em></p>", unsafe_allow_html=True)
        else: # connections is an empty list []
             st.markdown(f"<p>{render_missing()} (No detailed connections listed)</p>", unsafe_allow_html=True)

    else: # networkCommunityRecommendations object itself was missing
         st.markdown(f"<p>{render_missing()} (Section data not provided)</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


    # Observed Consumption (Render the new structure)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("<h4 class='consumption-title'>Observed Content Consumption</h4>", unsafe_allow_html=True)
    consumption_data = safe_get(platform_data, ['observedConsumption'])
    if consumption_data:
        main_feed_data = safe_get(consumption_data, ['mainFeed'])
        discovery_feed_data = safe_get(consumption_data, ['discoveryFeed'])
        analysis_notes = safe_get(consumption_data, ['consumptionAnalysisNotes'])

        has_main_feed_content = main_feed_data and any(safe_get(main_feed_data, [key]) for key in ['observedTopics', 'observedPosters', 'otherFeedObservations'])
        has_discovery_feed_content = discovery_feed_data and any(safe_get(discovery_feed_data, [key]) for key in ['observedThemes', 'contentTypes', 'specificExamples', 'otherDiscoveryObservations'])

        if not has_main_feed_content and not has_discovery_feed_content:
             st.markdown(f"<p>{render_missing()} (No consumption data provided for this platform)</p>", unsafe_allow_html=True)
        else:
            if has_main_feed_content:
                st.markdown("<h5 class='consumption-subtitle'>Main Feed Observations (Home/Following)</h5>", unsafe_allow_html=True)
                main_topics = safe_get(main_feed_data, ['observedTopics'], [])
                main_posters = safe_get(main_feed_data, ['observedPosters'], [])
                main_other = safe_get(main_feed_data, ['otherFeedObservations'])

                if main_topics: st.markdown(render_pills(main_topics, label="Observed Topics"), unsafe_allow_html=True)
                if main_posters:
                    st.markdown("<h6>Observed Posters</h6>", unsafe_allow_html=True)
                    st.markdown(render_posters(main_posters), unsafe_allow_html=True)
                if main_other: st.markdown(f"<p><strong>Other Notes:</strong> {main_other}</p>", unsafe_allow_html=True)
                if not main_topics and not main_posters and not main_other: st.markdown(render_missing(), unsafe_allow_html=True)


            if has_discovery_feed_content:
                # Adapt subtitle for specific platforms
                discovery_title = "Discovery Feed Observations (Explore/FYP/Recommendations)"
                if platform_name == "Reddit": discovery_title = "Feed Observations (r/All, Popular, Home)"
                elif platform_name == "Twitter/X": discovery_title = "Feed Observations (For You, Following)"

                st.markdown(f"<h5 class='consumption-subtitle'>{discovery_title}</h5>", unsafe_allow_html=True)
                disc_themes = safe_get(discovery_feed_data, ['observedThemes'], [])
                disc_types = safe_get(discovery_feed_data, ['contentTypes'], [])
                disc_examples = safe_get(discovery_feed_data, ['specificExamples'], [])
                disc_other = safe_get(discovery_feed_data, ['otherDiscoveryObservations'])

                if disc_themes: st.markdown(render_pills(disc_themes, label="Observed Themes"), unsafe_allow_html=True)
                if disc_types: st.markdown(render_pills(disc_types, label="Content Types"), unsafe_allow_html=True)
                if disc_examples:
                     st.markdown("<h6>Specific Examples Seen</h6>", unsafe_allow_html=True)
                     st.markdown(render_list(disc_examples, list_class="consumption-list examples"), unsafe_allow_html=True)
                if disc_other: st.markdown(f"<p><strong>Other Notes:</strong> {disc_other}</p>", unsafe_allow_html=True)
                if not disc_themes and not disc_types and not disc_examples and not disc_other: st.markdown(render_missing(), unsafe_allow_html=True)

        if analysis_notes:
            st.markdown("<h5 class='consumption-subtitle'>Consumption Analysis Notes</h5>", unsafe_allow_html=True)
            st.markdown(f'<div class="analysis-notes">{analysis_notes}</div>', unsafe_allow_html=True)

    else: # observedConsumption object itself was missing
         st.markdown(f"<p>{render_missing()} (Section data not provided)</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


    # Privacy & Presentation (from commonStructures)
    privacy = safe_get(platform_data, ['privacyPresentation'])
    if privacy:
         st.markdown('<div class="section">', unsafe_allow_html=True)
         st.markdown("<h4>Privacy & Presentation Notes</h4>", unsafe_allow_html=True)
         st.markdown(privacy, unsafe_allow_html=True)
         st.markdown('</div>', unsafe_allow_html=True)

     # Platform Specific Conclusions
    conclusions = safe_get(platform_data, ['platformSpecificConclusions'])
    if conclusions:
         st.markdown('<div class="section">', unsafe_allow_html=True)
         st.markdown("<h4>Platform Specific Conclusions</h4>", unsafe_allow_html=True)
         st.markdown(conclusions, unsafe_allow_html=True)
         st.markdown('</div>', unsafe_allow_html=True)


    # --- Placeholder for other platform-specific fields ---
    # Add rendering logic here for fields not covered by common structures
    # Example: storyHighlights for Instagram
    if platform_name == "Instagram":
       highlights = safe_get(platform_data, ['storyHighlights'], [])
       if highlights:
           st.markdown('<div class="section">', unsafe_allow_html=True)
           st.markdown("<h4>Story Highlights</h4>", unsafe_allow_html=True)
           st.markdown(render_dict_list_html(highlights, 'highlightName', 'coverDescription', 'contentSummary', None, None), unsafe_allow_html=True)
           st.markdown('</div>', unsafe_allow_html=True)
    # Example: karmaScores for Reddit
    if platform_name == "Reddit":
        karma = safe_get(platform_data, ['karmaScores'])
        if karma:
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.markdown("<h4>Karma Scores</h4>", unsafe_allow_html=True)
            post_karma = safe_get(karma, ['postKarma'])
            comment_karma = safe_get(karma, ['commentKarma'])
            award_karma = safe_get(karma, ['awardKarma']) # If available
            karma_html = ""
            if post_karma is not None: karma_html += f"**Post Karma:** {post_karma} "
            if comment_karma is not None: karma_html += f"**Comment Karma:** {comment_karma} "
            if award_karma is not None: karma_html += f"**Award Karma:** {award_karma}"
            if karma_html:
                st.markdown(karma_html.strip(), unsafe_allow_html=True)
            else:
                st.markdown(render_missing(), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


    st.markdown('</div>', unsafe_allow_html=True) # Close platform-card

# -------------------------------------
# Main Page Function - MODIFIED TO ACCEPT AVATAR DATA
# -------------------------------------

# --- Modified render_coffee_card_page to accept uploaded_avatar_data ---
def render_coffee_card_page(data, uploaded_avatar_data=None):
    """
    Main function to render the entire Coffee Card page.
    Accepts uploaded_avatar_data to pass to the header.
    """

    st.title("☕ Your Coffee Card")
    st.caption("Your Personalized Social Media Analysis")
    # Apply the custom CSS first
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

    # Render the main header card, passing the avatar data
    render_coffee_card_header(data, uploaded_avatar_data) # Pass avatar here

    # Prepare tabs
    platform_analyses = safe_get(data, ['platformSpecificAnalysis'], [])
    tab_titles = ["Overview"] + [f"{p['platformName']}" for p in platform_analyses if p.get('platformName')]

    if not platform_analyses:
        st.warning("No platform-specific analysis found in the data.")
        tabs = st.tabs(["Overview"])
        overview_tab = tabs[0]
        with overview_tab:
             render_overview_tab(data)
    else:
        tabs = st.tabs(tab_titles)

        # Overview Tab
        with tabs[0]:
             render_overview_tab(data)

        # Platform Tabs
        for i, platform_data in enumerate(platform_analyses):
            # Check if platform_data is a dictionary before proceeding
            if isinstance(platform_data, dict):
                with tabs[i+1]:
                    render_platform_tab(platform_data)
            else:
                # Handle cases where an item in platformSpecificAnalysis might not be a dict
                with tabs[i+1]:
                    st.error(f"Invalid data format encountered for platform analysis item {i+1}. Expected a dictionary.")
                    st.write(platform_data)

# -------------------------------------
# Data Extraction Function - NO CHANGES HERE
# -------------------------------------
def extract_data_from_schema_like_json(schema_like_data):
    """
    Extracts the actual data values from a JSON structure that resembles
    a schema definition (with data often under 'value' or 'items' keys).
    """
    extracted_data = {}
    properties = safe_get(schema_like_data, ['properties'], {})

    if not properties:
        # Maybe it's already flat? Return as is.
        # Or raise an error if properties are expected.
        st.warning("Input JSON doesn't seem to have the expected 'properties' structure. Attempting to use as is.")
        return schema_like_data # Try using it directly

    # --- Extract Top-Level Values ---
    # Check for 'value' key first, then direct access if 'value' isn't present
    extracted_data['targetIndividual'] = safe_get(properties, ['targetIndividual', 'value'], safe_get(properties, ['targetIndividual']))
    extracted_data['analyzedPlatforms'] = safe_get(properties, ['analyzedPlatforms', 'value'], safe_get(properties, ['analyzedPlatforms']))
    # platformSpecificAnalysis seems to store data directly in 'items' array within properties
    extracted_data['platformSpecificAnalysis'] = safe_get(properties, ['platformSpecificAnalysis', 'items'], [])
    extracted_data['crossPlatformSynthesis'] = safe_get(properties, ['crossPlatformSynthesis', 'value'], safe_get(properties, ['crossPlatformSynthesis'], {})) # Default to empty dict if missing
    # Handle final summary potentially being under 'value' or directly
    extracted_data['finalComprehensiveSummary'] = safe_get(properties, ['finalComprehensiveSummary', 'value'], None)


    # --- Basic Validation of Extracted Data ---
    # Check if the essential extracted keys actually got populated
    required_keys = ["targetIndividual", "analyzedPlatforms", "platformSpecificAnalysis", "crossPlatformSynthesis", "finalComprehensiveSummary"]
    all_present = True
    missing_keys = []
    for key in required_keys:
        if key not in extracted_data or extracted_data[key] is None:
             # platformSpecificAnalysis can be empty, crossPlatformSynthesis can be {}, summary can be None.
             if key not in ['platformSpecificAnalysis', 'finalComprehensiveSummary', 'crossPlatformSynthesis']:
                 all_present = False
                 missing_keys.append(key)
             # Special check for crossPlatformSynthesis potentially being empty {}
             elif key == 'crossPlatformSynthesis' and not extracted_data[key]:
                  # Allow empty dict for crossPlatformSynthesis, but maybe warn if expected?
                  pass # Or add a warning?

    if not all_present:
        st.error(f"Failed to extract required data from the input JSON structure. Missing/Null values for: {', '.join(missing_keys)}. Please check the input file's format.")
        st.write("Expected structure within 'properties':")
        st.code("""
 'properties': {
    'targetIndividual': { 'value': '...' } OR 'targetIndividual': '...',
    'analyzedPlatforms': { 'value': [...] } OR 'analyzedPlatforms': [...],
    'platformSpecificAnalysis': { 'items': [...] },
    'crossPlatformSynthesis': { 'value': {...} } OR 'crossPlatformSynthesis': {...},
    'finalComprehensiveSummary': { 'value': '...' } OR 'finalComprehensiveSummary': '...'
 }
        """, language="json")
        st.write("Input 'properties' structure found:")
        st.json(properties) # Show what was actually found
        return None # Indicate extraction failure

    # If crossPlatformSynthesis wasn't under 'value', try getting it directly from properties
    # Check if it's None or {} before trying direct access
    if not extracted_data['crossPlatformSynthesis'] and 'crossPlatformSynthesis' in properties:
         extracted_data['crossPlatformSynthesis'] = properties['crossPlatformSynthesis']

    # If finalComprehensiveSummary wasn't under 'value', try getting it directly
    if extracted_data['finalComprehensiveSummary'] is None and 'finalComprehensiveSummary' in properties:
         extracted_data['finalComprehensiveSummary'] = properties['finalComprehensiveSummary'] # Could still be None or the direct string

    return extracted_data

# -------------------------------------
# Example Data Loading - NO CHANGES HERE
# -------------------------------------
def load_example_data():
    """Loads example JSON data matching the SCHEMA-LIKE structure provided by the user."""
    # This dictionary now holds the structure provided in the prompt,
    # WITHOUT the invalid "definitions" block.
    schema_like_placeholder_data = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "title": "Comprehensive Social Media Profile Analysis Report (Platform-Specific Schemas & MANDATORY UI/Consumption Data)",
      "description": "Detailed analysis using dedicated schemas per platform, mandating inclusion/analysis of specific UI element data (Inbox/DM Preview, Network Tabs, Recommendations), detailed content consumption observations, and individual connections.",
      "type": "object",
      "properties": {
        "targetIndividual": {
          "description": "Name or identifier of the individual being analyzed.",
          "type": "string",
          "value": "Homen Shum"
        },
        "analyzedPlatforms": {
          "description": "List of social media platforms included in this analysis.",
          "type": "array",
          "items": {
            "type": "string",
            "enum": ["LinkedIn","Instagram","Twitter/X","Facebook","TikTok","Reddit","Other"]
          },
          "value": ["LinkedIn", "Instagram"]
        },
        "platformSpecificAnalysis": {
          "description": "Detailed analysis for each individual platform, using the specific schema defined for that platform type found in definitions/platformSchemas.",
          "type": "array",
          "items": [
            # --- LinkedIn Data ---
            {
              "platformName": "LinkedIn",
              "profileFundamentals": {
                "username": "homenshum", "fullName": "Homen Shum", "pronouns": "(He/Him)",
                "location": "Fremont, California, United States", "profileURL": "https://www.linkedin.com/in/homenshum",
                "profileLanguage": "English", "verificationStatus": None, "contactInfoVisible": True,
                "profilePictureDescription": "Professional headshot of Homen Shum, smiling, wearing a dark top, blurred cityscape background.",
                "bannerImageDescription": "Stylized photo of a modern cityscape, possibly San Francisco, during daytime/sunset.",
                "linkedWebsites": ["CafeCorner LLC", "Financial Edge Training"]
              },
              "bioAnalysis": {
                "fullText": "Headline: “Do something great with data and finance” | Ex-PM Startup Banking Associate | LLM/Data/Backend Engineering for FinTech/AdTech | Ex-HealthTech Founder (Data/ML focused) | CafeCorner LLC. About: Experience with LLM APIs and Generative AI tools. I've been working with LLM APIs and generation AI tools since December 2022, progressing through increasingly complex implementations across multiple domains... Mar 2023: Achieved 2nd place in Hakation Cybersecurity challenge with 96% accuracy vs semantic matching and unified generation of multi-vendor security alerts... 2023 Healthcare AI Application... 2023-Ongoing Financial Sector Implementation... 2024-Present: Advanced Applications... My Latest Web Applications... Here’s to embracing growth, bridging gaps, and making technology more accessible - together.",
                "identifiedKeywords": ["LLM","Generative AI","Data","Finance","FinTech","AdTech","HealthTech","Backend Engineering","Startup Banking","Founder","Cybersecurity","Semantic Matching","RAG","NLP","Python","SQL","Azure","GCP","Large Language Models","AWS","Risk Management","High-Frequency Trading"],
                "hashtagsInBio": [], "mentionedUsersInBio": [],
                "statedPurposeFocus": "Focus on applying data science, LLM/AI, and engineering skills within Finance, Technology (FinTech, AdTech, HealthTech), and startup environments. Emphasizes project execution, growth, and accessibility.",
                "tone": "Professional, technical, results-oriented, forward-looking",
                "callToAction": "Implicit: Connect, view projects, consider for roles ('Show recruiters you're open to work'). Explicit: 'Get started' buttons for profile sections."
              },
              "skillsInterestsExpertise": { # LinkedIn specific object structure
                "skillsList": [{"skillName": "Python (Programming Language)", "endorsementCount": None}, {"skillName": "Pandas (Software)", "endorsementCount": None}, {"skillName": "Microsoft Azure", "endorsementCount": None}, {"skillName": "Google Cloud Platform (GCP)", "endorsementCount": None}, {"skillName": "Large Language Models (LLM)", "endorsementCount": None}],
                "licensesCertifications": [], "courses": [], "recommendationsReceivedCount": None, "recommendationsGivenCount": None,
                "followedInfluencers": [], "followedCompanies": [], "followedGroups": [], "followedSchools": []
              },
              "featuredPinnedContent": [
                {"contentType": "Link", "title": "The Limits of Public Data Flow (LLM Struggle)", "description": None, "link": "(partially visible GitHub link)", "imageURL": "(Screenshot/diagram visible)"},
                {"contentType": "Link", "title": "Fluency Blast #1 - Filescore Hackathon Entry", "description": "Hackathon link: https://devpost.com/software/fluency-blast... This is a chatbot that can answer questions about medical advice...", "link": "https://devpost.com/software/fluency-blast...", "imageURL": "(Screenshot of UI visible)"},
                {"contentType": "Link", "title": "Semantic Search Pairing Method with Dimension Reduction", "description": "Provided 86.17% high performance accuracy by the end of the hackathon... Find out more at... 1. Solution Presentation with...", "link": "(Not fully visible)", "imageURL": "(Diagram/Flowchart visible)"}
              ],
              "professionalHistory": {
                "experience": [
                  {"title": "Founder", "organization": "CafeCorner LLC", "duration": "2 yrs 5 mos", "dates": "Dec 2021 - Present", "description": "Built and deployed sales recommendation...", "associatedSkills": ["Large Language Model Operations (LLMOps)", "+1 skills"]},
                  {"title": "Startup Banking Associate", "organization": "JPMorgan Chase & Co.", "dates": "Fall 2021 - Nov 2024", "description": "As a JPM Startup Banking Associate...", "associatedSkills": ["Data Analysis, Automation Lead", "+6 skills"]},
                  {"title": "Rotation 1: Healthcare and Life Science Banking Team", "organization": "JPMorgan Chase & Co.", "dates": "May 2021 - Feb 2024", "description": "Initiated collaboration...", "associatedSkills": ["Amazon Web Services (AWS)", "+1 skills"]},
                   {"title": "Academic Advisor", "organization": "Vietnamese Student Association - UC Santa Barbara", "duration": "1 yr 1 mo", "dates": "Apr 2019 - Apr 2020", "description": "Supervised members..."}
                ],
                "education": [
                  {"institution": "Financial Edge Training", "degreeField": "Commercial Banking Analyst Training, Accounting and Finance", "dates": "2021", "activities": None},
                  {"institution": "UC Santa Barbara", "degreeField": "Certificate, Business Administration and Management, General", "dates": "2020 - 2021", "description": "Cooperated with professors...", "activities": None}
                ],
                "projects": [
                  {"name": "Startup Search API", "dates": "Oct 2023 - Nov 2023", "description": "Demo: https://hzn-qdrant-server-4ppnayg6xa-uc.a.run.app/applications/search?q=(your search query...)", "link": "https://hzn-qdrant-server-4ppnayg6xa-uc.a.run.app/applications/search?...", "associatedSkills": ["Web Services API", "Docker", "+1 skills"], "contributors": []}
                ]
              },
              "contentGenerationActivity": {
                "postingFrequency": "Infrequent based on limited view.", "dominantContentTypes": ["Reposts", "Comments"],
                "contentExamples": ["Reposted David Myriel's post...", "Commented on LIN Pendley's post..."],
                "recurringThemesTopics": ["AI", "LLMs", "Vector Databases (Qdrant)"], "overallToneVoice": "Technical, engaged",
                "recentActivityExamples": ["Homen Shum reposted this", "Homen Shum commented on this"], "articlesPublishedCount": None
              },
              "engagementPatterns": {
                "outgoingInteractionStyle": "Comments on relevant technical posts (LLMs, Vector DBs), Reposts technical news (Qdrant).",
                "typesOfContentEngagedWith": ["Technical posts related to AI/LLMs", "Specific technologies (Qdrant)"],
                "incomingEngagementHighlights": "Limited view - Reposted post had 4 comments.", "typicalIncomingCommentTypes": []
              },
              "platformFeatureUsage": [
                {"featureName": "Featured Section", "usageDescription": "Actively used."},
                {"featureName": "Projects Section", "usageDescription": "Used to detail specific project."},
                {"featureName": "Skills Section", "usageDescription": "Populated."},
                {"featureName": "Experience Section", "usageDescription": "Extensively detailed."},
                {"featureName": "Education Section", "usageDescription": "Lists relevant education."}
              ],
              "networkCommunityRecommendations": {
                "followerCount": "500+", "followingCount": None, "audienceDescription": "Likely technology professionals, recruiters.",
                "groupCommunityMemberships": [],
                "inboxSidebarPreview": [
                   {"name": "Chao Riu", "detailSnippet": "Hi Homen, MATCHA is hiring...", "timestamp": "Apr 19"},
                   {"name": "Cristina McCarthy", "detailSnippet": "Who's Hiring in Ann Arbor", "timestamp": "Apr 19"},
                   {"name": "Otnam Kader", "detailSnippet": "Hi Homen. There is an exciting...", "timestamp": "Mar 26"}
                ],
                "inboxSidebarAnalysis": "The inbox preview is heavily dominated by messages indicative of professional outreach, particularly recruitment efforts ('hiring', 'great fit', 'role is aligned') and networking. This strongly suggests Homen Shum is perceived as possessing in-demand skills.",
                "myNetworkTabVisibility": [],
                "myNetworkTabAnalysis": "The 'My Network' tab was not visited.",
                "platformSuggestions": {
                  "suggestedPeople": [
                    {"name": "Rania Hussain", "headlineOrDetail": "Incoming Investment Banking Analyst at Goldman Sachs", "reasonForSuggestion": "From your school"},
                    {"name": "Leah Varughese", "headlineOrDetail": "Digital Product Associate at JPMorgan Chase & Co."},
                  ],
                  "suggestedCompaniesOrPages": [], "suggestedGroups": [],
                  "peopleAlsoViewed": [
                    {"name": "Sonya Chiang", "headlineOrDetail": "Product @ Frame.io..."},
                    {"name": "Anyichika Achufusi Jr.", "headlineOrDetail": "[Headline obscured/short]"}
                  ],
                  "otherSuggestions": []
                },
                "platformRecommendationsAnalysis": "LinkedIn's recommendations reinforce connections to finance (JPMC, Goldman Sachs) and universities. 'People Also Viewed' hints at interest in product roles alongside engineering.",
                "detailedConnectionsList": None, "detailedConnectionsAnalysis": "Detailed connections list was not available."
              },
              "privacyPresentation": "Profile appears public. Professionally structured.",
              "observedConsumption": {
                "mainFeed": {
                  "observedTopics": ["AI", "Machine Learning", "Startups", "Knowledge Graphs", "Vector Databases (Qdrant)", "OpenAI"],
                  "observedPosters": [
                    {"posterName": "Maryam Miosdi, PhD", "exampleContentSummary": "Post about GraphMemory..."},
                    {"posterName": "Snehanshu Raj", "exampleContentSummary": "Post about website tracking/privacy."}
                  ],
                  "otherFeedObservations": "Feed heavily focused on technical AI/ML concepts and industry news."
                },
                "discoveryFeed": None,
                "consumptionAnalysisNotes": "Main feed consumption directly aligns with the professional profile's focus."
              },
              "platformSpecificConclusions": "Strong LinkedIn presence projecting a technically adept professional in AI/ML, data science within FinTech/HealthTech. High inbound message volume and focused feed consumption confirm active involvement."
            },
             # --- Instagram Data ---
            {
              "platformName": "Instagram",
              "profileFundamentals": {
                "username": "homenshum", "fullName": "Homen Shum", "pronouns": None, "location": None,
                "profileURL": "https://www.instagram.com/homenshum/", "profileLanguage": None, "verificationStatus": None,
                "contactInfoVisible": None, "profilePictureDescription": "Small icon suggests same professional headshot.", "bannerImageDescription": None, "linkedWebsites": []
              },
              "bioAnalysis": None,
              "skillsInterestsExpertise": [],
              "storyHighlights": [],
              "contentGenerationActivity": {
                "postingFrequency": None, "dominantContentTypes": None, "contentExamples": [], "recurringThemesTopics": [],
                "overallToneVoice": None,
                "recentActivityExamples": ["Liked a post from 'konrad' approx 1 hour prior. Post text: \"If you're willing to suck at anything...\""],
                "gridAesthetic": None, "reelsPerformanceIndicators": None, "storiesFrequencyEngagement": None
              },
              "engagementPatterns": {
                "outgoingInteractionStyle": "Likes content related to perseverance/motivational themes.",
                "typesOfContentEngagedWith": ["Motivational/philosophical graphic posts"],
                "incomingEngagementHighlights": None, "typicalIncomingCommentTypes": []
              },
              "platformFeatureUsage": [
                {"featureName": "Main Feed Browsing", "usageDescription": "Observed scrolling main feed."},
                {"featureName": "Reels/Explore Tab Browsing", "usageDescription": "Observed actively scrolling discovery feed."}
              ],
              "networkCommunityRecommendations": {
                "followerCount": None, "followingCount": None, "audienceDescription": None, "groupCommunityMemberships": [],
                "inboxSidebarPreview": [], "inboxSidebarAnalysis": "Instagram DM preview was not visible.",
                "myNetworkTabVisibility": [], "myNetworkTabAnalysis": "Instagram network tabs were not visited.",
                "platformSuggestions": {
                  "suggestedPeople": [
                    {"name": "homenshum", "headlineOrDetail": "Homen Shum"},
                    {"name": "kokopark", "reasonForSuggestion": "Suggested for you"},
                     {"name": "billlie_kim_suyeon", "headlineOrDetail": "Billlie Kim Suyeon"}
                  ],
                  "suggestedCompaniesOrPages": [{"name": "besticecream", "descriptionOrIndustry": "Assumed Food/Dessert Page"}],
                  "suggestedGroups": [], "peopleAlsoViewed": [], "otherSuggestions": []
                },
                "platformRecommendationsAnalysis": "Instagram suggestions mix personal-appearing accounts, food interest, and K-Pop, contrasting sharply with LinkedIn and suggesting perception based on general engagement/demographics.",
                "detailedConnectionsList": None, "detailedConnectionsAnalysis": "Detailed connections list was not available."
              },
              "privacyPresentation": "Likely public account, as discovery feed was accessible.",
              "observedConsumption": {
                "mainFeed": {
                  "observedTopics": ["Music Events (DJ)", "Motivational Quotes"],
                  "observedPosters": [
                    {"posterName": "hiventertainment", "exampleContentSummary": "Flyer for DJ Kang event..."},
                    {"posterName": "konrad", "exampleContentSummary": "Graphic with text... (Liked by homenshum)."}
                  ],
                  "otherFeedObservations": "Brief view showed content from followed accounts."
                },
                "discoveryFeed": {
                  "observedThemes": ["Memes","Viral Videos","Pop Culture","Food","Science/Nature","Asian Cultural Content","Relationship Commentary","Fitness","News","Humorous Clips"],
                  "contentTypes": ["Short-form video","Memes","News clips","Informational clips","Personal vlogs","Anime clips","Food videos"],
                  "specificExamples": [
                    "Meme showing Dr. Strange creating a portal", "Clip about parasitic wasp", "Video reviewing Costco noodles",
                    "Graphic 'The F*ck First Rule'", "Clip 'when he tells me stop...'", "Video 'MALE GAZE'", "Video 'GHETTO STONESTOWN?'",
                    "Anime clip 'He kissed the princess...'", "Fitness clip with Chinese text", "Clip about Columbia student cheating",
                    "Meme comparing noses", "Gaming clip (MW2) 'BRO HIJACKS A PLANE'", "Crying cat meme with Chinese text",
                    "Video of man handling large white fish/sheet", "Video asking 'Can Thor fly without hammer?'", "Hip Hop dance tutorial",
                    "Man cutting leather/shoe material", "Clip 'you got catfished from Temu'"
                  ],
                  "otherDiscoveryObservations": "High frequency of content with Asian language overlays. Strong presence of meme formats, short humorous clips, food, gaming, anime/pop culture."
                },
                "consumptionAnalysisNotes": "Significant difference between Main Feed (events, motivation) and Discovery Feed (broad, meme-heavy, entertainment). Suggests typical passive consumption of viral content, possibly with affinity for Asian cultural content and memes."
              },
              "platformSpecificConclusions": "Instagram activity points towards personal use focused on general entertainment and social consumption, distinct from the professional sphere shown on LinkedIn."
            }
          ]
        },
        "crossPlatformSynthesis": {
          "consistencyVsVariation": {
            "profileElementConsistency": "Profile picture appears consistent. Username ('homenshum') consistent.",
            "contentTonePersonaConsistency": "High variation. LinkedIn: technical, professional. Instagram: personal, entertainment.",
            "notableDifferences": "Focus and content consumed/engaged with are entirely different."
          },
          "contentOverlapStrategy": "No content overlap observed. Deliberate platform differentiation.",
          "synthesizedExpertiseInterests": {
            "coreProfessionalSkills": ["AI/ML", "LLMs", "Generative AI", "Data Science", "Backend Engineering", "Python", "Pandas", "SQL", "Cloud (Azure, GCP, AWS)", "Finance (Startup Banking, Risk, Trading)", "FinTech", "AdTech", "HealthTech", "Project Management", "Automation", "Cybersecurity", "RAG", "Vector DBs", "Semantic Search"],
            "corePersonalInterests": ["Memes", "Viral Content", "Food", "Pop Culture (Marvel, Anime, Gaming)", "Asian Culture", "Music Events", "Motivational Quotes", "Humor"]
          },
          "overallOnlinePersonaNarrative": "Distinct dual persona. Professionally (LinkedIn): skilled technologist in AI/ML, finance, tech startups. Personally (Instagram): engages with mainstream digital entertainment, memes, food, pop culture.",
          "professionalEvaluation": {
            "strengthsSkillsMatch": "Strong evidence on LinkedIn of applied AI/ML skills via projects/hackathons.",
            "impactAchievements": "Quantifiable achievements highlighted. Founder/lead experience shows initiative.",
            "industryEngagement": "Active consumption/engagement on LinkedIn. High inbound message volume suggests visibility/demand.",
            "potentialRedFlagsClarifications": "Timeline of JPMC vs founder role. Short duration in Lead AI Engineer role. Series of short-term challenges pre-JPMC.",
            "overallCandidateSummary": "Technically strong, motivated AI/ML expert with project leadership/entrepreneurial experience. Compelling LinkedIn profile despite minor timeline queries."
          },
          "marketTrendInsights": {
            "keyTechnologiesToolsTopics": ["LLMs", "Generative AI", "RAG", "Vector Databases (Qdrant)", "Semantic Search", "Cloud (Azure, GCP, AWS)", "Python", "Pandas", "FinTech AI", "HealthTech AI", "Cybersecurity AI", "Personalized AI Agents", "Knowledge Graphs"],
            "emergingThemesNiches": ["Practical deployment of AI/LLMs", "AI in specific verticals", "Performance/automation via AI", "Vector DB importance", "Hackathons for skill demos"],
            "relevantContentPatterns": "Sharing/discussing technical articles, project showcases, tool updates on LinkedIn."
          },
          "inferredAlgorithmicPerception": [
            {"platformName": "LinkedIn", "categorizationHypothesis": "Algorithm sees Homen as Software Engineer/Data Scientist (AI/ML/LLMs) in FinTech/HealthTech/Startups. Signals high technical proficiency and job market value (recruitment messages). Recommendations and feed support this."},
            {"platformName": "Instagram", "categorizationHypothesis": "Algorithm sees Homen based on broad engagement/demographics. Consumption (memes, viral clips, food, pop culture) and suggestions (personal, food, K-Pop) indicate interest in Entertainment, Memes, Food, Pop Culture (incl. Asian media). Minimal influence from professional profile."}
          ],
          "crossPlatformNetworkAnalysis": {
            "overlappingConnectionsRecommendations": [],
            "networkComparisonNotes": "LinkedIn network professional (tech, finance, recruitment). Instagram network social/general interest.",
            "consumptionComparisonNotes": "LinkedIn consumption focused on technical/professional. Instagram consumption broad entertainment/leisure."
          }
        },
        "finalComprehensiveSummary": {
          "description": "High-level summary of the entire analysis, specifically highlighting insights derived from mandatory UI element and consumption data across platforms.",
          "type": "string",
          "value": "This analysis reveals a distinct dual online presence for Homen Shum. His LinkedIn profile meticulously crafts a persona of a highly skilled AI/ML engineer and data scientist with significant experience in finance and tech startups, backed by detailed project descriptions and quantifiable achievements. Critically, the LinkedIn inbox preview data confirms this perception, showing substantial inbound interest from recruiters and professional contacts, while feed consumption and recommendations center firmly on advanced AI/tech topics. In stark contrast, his Instagram usage, inferred from feed/discovery consumption and platform suggestions, points to a personal leisure profile. The mandatory observation of the Instagram discovery feed revealed consumption dominated by memes, viral entertainment, food, and pop culture, with potential affinity for Asian cultural content, completely separate from his professional focus. Instagram's recommendations (personal accounts, food, K-Pop) further reinforce this non-professional categorization. The mandatory inclusion of UI elements (inbox, recommendations) and detailed consumption analysis across both platforms was essential in clearly delineating these separate professional and personal digital spheres."
        }
      },
      "required": ["targetIndividual","analyzedPlatforms","platformSpecificAnalysis","crossPlatformSynthesis","finalComprehensiveSummary"]
      # Removed the definitions block from the data instance
    }

    return schema_like_placeholder_data


# -------------------------------------
# Main Execution Block - MODIFIED TO ADD AVATAR UPLOAD & PASS DATA
# -------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Your Coffee Card", page_icon="☕", layout="wide")
    st.title("🌱 Café Corner")
    st.caption("💡 A place where ideas brew and connections pour. 🧑‍💻")

    raw_report_data = None # Store the initially loaded data
    report_data = None     # Store the extracted, flat data
    uploaded_avatar = None # Store the uploaded avatar file object
    data_source = "Example Data" # Default source

    # --- Initialize Session State ---
    if 'avatar_url' not in st.session_state:
        st.session_state['avatar_url'] = "https://via.placeholder.com/80x80.png?text=?"
    if 'target_name' not in st.session_state:
        st.session_state['target_name'] = "Unknown"
    if 'pronouns_html' not in st.session_state:
        st.session_state['pronouns_html'] = ""
    if 'verification_html' not in st.session_state:
        st.session_state['verification_html'] = ""
    if 'headline_html' not in st.session_state:
        st.session_state['headline_html'] = ""
    if 'links_html' not in st.session_state:
        st.session_state['links_html'] = ""
    if 'platforms_pills_html' not in st.session_state:
        st.session_state['platforms_pills_html'] = ""
    if 'location' not in st.session_state:
        st.session_state['location'] = ""

    # --- Render Header ---
    with st.sidebar:
        st.markdown(f"""
        <div class="card coffee-card-header">
            <div class="progress-bar-container">
                <div class="progress-bar-fill"></div>
            </div>
            <div class="header-content">
                <img src="{st.session_state['avatar_url']}" alt="Avatar" class="avatar">
                <div class="header-text">
                    <h1>{st.session_state['target_name']} {st.session_state['pronouns_html']} {st.session_state['verification_html']}</h1>
                    {st.session_state['headline_html']}
                    <p class="location-links">
                        {st.session_state['location']}
                        {st.session_state['links_html']}
                    </p>
                    <div class="analyzed-platforms-container">
                    <h5>Platforms Analyzed:</h5>
                    {st.session_state['platforms_pills_html']}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.sidebar:
        with st.expander("Upload Data"):
            st.header("Upload Data")
            # --- File Uploader for JSON Analysis ---
            uploaded_file = st.file_uploader(
                "1. Upload Profile Analysis JSON",
                type=["json", "txt"], # Accept both extensions
                help="Upload the JSON file (or a .txt file containing valid JSON) for the profile analysis."
            )

            # --- File Uploader for Avatar Image ---
            uploaded_avatar = st.file_uploader(
                "2. (Optional) Upload Avatar Image",
                type=["png", "jpg", "jpeg", "gif"],
                help="Upload a profile picture to display in the header."
            )

            # --- Load and Process JSON Data ---
            if uploaded_file is not None:
                try:
                    # Read content as bytes, then decode (assuming utf-8), then parse
                    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                    raw_report_data = json.load(stringio) # Load the potentially nested structure
                    data_source = f"Uploaded: {uploaded_file.name}"
                    st.success(f"Successfully loaded data from {uploaded_file.name}")
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing file '{uploaded_file.name}': Invalid JSON format. Please check the file content. Details: {e}")
                    raw_report_data = load_example_data() # Fallback to example (schema-like structure)
                    data_source = "Example Data (Upload Failed)"
                except UnicodeDecodeError:
                    st.error(f"Error decoding file '{uploaded_file.name}'. Please ensure it's UTF-8 encoded.")
                    raw_report_data = load_example_data() # Fallback to example
                    data_source = "Example Data (Encoding Error)"
                except Exception as e:
                    st.error(f"An error occurred while processing the uploaded file '{uploaded_file.name}': {e}")
                    raw_report_data = load_example_data() # Fallback to example
                    data_source = "Example Data (Upload Error)"
            else:
                # Load example data (schema-like structure) if no file is uploaded
                raw_report_data = load_example_data()

    # --- Attempt to extract the flat data structure ---
    if raw_report_data:
        report_data = extract_data_from_schema_like_json(raw_report_data)
    else:
        st.error("Could not load raw report data.")
        report_data = None # Ensure report_data is None if raw loading failed

    # Render the main page content *only if* extraction was successful
    tabs = st.tabs(["Profile"])
    with tabs[0]:
        if report_data:
            # Pass both the extracted data and the uploaded avatar file object
            render_coffee_card_page(report_data, uploaded_avatar)
        else:
            # Error messages are handled within extract_data_from_schema_like_json
            # or if raw_report_data itself was None.
            st.warning("Report rendering halted due to data loading or extraction errors.")