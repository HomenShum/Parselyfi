import streamlit as st
import pandas as pd
import re
from datetime import datetime, timedelta
from github import Github
import altair as alt
from collections import defaultdict
import io
import json
import base64
import os
import time

st.set_page_config(
    page_title="Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------
# State Management Functions
# --------------------------------

def initialize_edit_mode():
    """Initialize editing mode in session state if not already present"""
    if 'editing_mode' not in st.session_state:
        st.session_state.editing_mode = False

def initialize_document_data(report_data):
    """Initialize document data in session state with the current report data"""
    if 'document_data' not in st.session_state:
        st.session_state.document_data = report_data.copy()
    elif st.session_state.get('refresh_data', False):
        # Only update data when explicitly refreshing
        st.session_state.document_data = report_data.copy()
        st.session_state.refresh_data = False
    
    # Always ensure we're using document_data as our source of truth
    # This is critical - never use the original report_data after initialization
    return st.session_state.document_data

def apply_editable_css():
    """Apply custom CSS styling for editable content"""
    st.markdown("""
    <style>
        /* Styles for editable content */
        .editable-text {
            cursor: text;
            transition: all 0.2s ease;
            border: 1px dashed transparent;
            border-radius: 4px;
            padding: 4px;
        }
        
        .editable-text:hover {
            background-color: #F3F4F6;
            border-color: #D1D5DB;
        }
        
        .editing-active {
            border: 1px dashed #3B82F6 !important;
            background-color: #EFF6FF !important;
        }
        
        /* Add a faint edit icon on hover */
        .editable-container:hover::after {
            content: "✏️";
            position: absolute;
            right: 8px;
            top: 8px;
            font-size: 14px;
            opacity: 0.5;
        }

        /* Edit mode indicator */
        .edit-mode-banner {
            background-color: #EFF6FF;
            color: #1E40AF;
            padding: 0.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            text-align: center;
            font-weight: 500;
        }
        
        /* Edited content indicator */
        .edited-content {
            border-left: 3px solid #3B82F6;
        }
        
        /* Dashboard styling */
        .report-header-wrapper {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .company-title {
            font-size: 1.8rem;
            font-weight: bold;
            color: #1F2937;
        }
        .report-subtitle {
            font-size: 1.1rem;
            color: #4B5563;
        }
        .tag {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            margin-right: 0.5rem;
            margin-top: 0.5rem;
        }
        .blue-tag {
            background-color: #EFF6FF;
            color: #1E40AF;
        }
        .green-tag {
            background-color: #ECFDF5;
            color: #065F46;
        }
        .purple-tag {
            background-color: #F5F3FF;
            color: #5B21B6;
        }
        .yellow-tag {
            background-color: #FFFBEB;
            color: #92400E;
        }
        .metadata {
            text-align: right;
            font-size: 0.875rem;
            color: #6B7280;
        }
        .high-confidence {
            color: #059669;
            font-weight: 600;
        }
        .card {
            background-color: white;
            border: 1px solid #E5E7EB;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .blue-card {
            border-left: 4px solid #3B82F6;
            background-color: #EFF6FF;
        }
        .green-card {
            border-left: 4px solid #10B981;
            background-color: #ECFDF5;
        }
        .yellow-card {
            border-left: 4px solid #F59E0B;
            background-color: #FFFBEB;
        }
        .purple-card {
            border-left: 4px solid #8B5CF6;
            background-color: #F5F3FF;
        }
        .footer {
            margin-top: 2rem;
            padding-top: 1rem;
            text-align: center;
            font-size: 0.875rem;
            color: #6B7280;
            border-top: 1px solid #E5E7EB;
        }
        .tab-content {
            padding: 1.5rem;
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .section-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1F2937;
            margin-bottom: 1rem;
        }
        .subsection-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1F2937;
            margin: 1rem 0 0.5rem 0;
        }
        .data-table {
            margin-bottom: 1rem;
        }

        /* Tabs styling */        
        .stTabs {
            background-color: white;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            font-size: 0.5rem;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------
# Update Functions for Document Data
# --------------------------------

def update_simple_field(field_path, value):
    """
    Update a field in the document data using a path notation.
    
    Args:
        field_path (str): Dot-notation path to the field (e.g., "company_data.name")
        value: New value to set
    """
    parts = field_path.split('.')
    data = st.session_state.document_data
    
    # Navigate to the parent object
    for part in parts[:-1]:
        if part not in data:
            data[part] = {}
        data = data[part]
    
    # Set the value
    data[parts[-1]] = value

def update_array_item(array_path, index, value):
    """
    Update an item in an array within the document data.
    
    Args:
        array_path (str): Dot-notation path to the array (e.g., "leadership_team")
        index (int): Index of the item to update
        value: New value to set
    """
    parts = array_path.split('.')
    data = st.session_state.document_data
    
    # Navigate to the array
    for part in parts:
        if part not in data:
            data[part] = []
        data = data[part]
    
    # Update the item
    if isinstance(data, list) and index < len(data):
        data[index] = value

def add_array_item(array_path, default_value=None):
    """
    Add a new item to an array within the document data.
    
    Args:
        array_path (str): Dot-notation path to the array
        default_value: Default value for the new item
    """
    parts = array_path.split('.')
    data = st.session_state.document_data
    
    # Navigate to the array
    for part in parts:
        if part not in data:
            data[part] = []
        data = data[part]
    
    # Add the new item
    if isinstance(data, list):
        data.append(default_value)

def remove_array_item(array_path, index):
    """
    Remove an item from an array within the document data.
    
    Args:
        array_path (str): Dot-notation path to the array
        index (int): Index of the item to remove
    """
    parts = array_path.split('.')
    data = st.session_state.document_data
    
    # Navigate to the array
    for part in parts:
        if part not in data:
            return
        data = data[part]
    
    # Remove the item
    if isinstance(data, list) and index < len(data):
        data.pop(index)

def update_dict_in_array(array_path, index, key, value):
    """
    Update a key in a dictionary within an array.
    
    Args:
        array_path (str): Path to the array
        index (int): Index of the dictionary in the array
        key (str): Dictionary key to update
        value: New value to set
    """
    parts = array_path.split('.')
    data = st.session_state.document_data
    
    # Navigate to the array
    for part in parts:
        if part not in data:
            return
        data = data[part]
    
    # Update the dictionary key
    if isinstance(data, list) and index < len(data) and isinstance(data[index], dict):
        data[index][key] = value

# --------------------------------
# Editable Components
# --------------------------------

def editable_text_input(label, field_path, key_prefix, width=None, placeholder=None, unique_id=None):
    """
    Render an editable text input field with guaranteed unique keys.
    
    Args:
        label (str): Field label
        field_path (str): Path to the field in document_data
        key_prefix (str): Prefix for the input key
        width (str, optional): CSS width value
        placeholder (str, optional): Placeholder text
        unique_id (str, optional): Additional identifier to ensure uniqueness
    
    Returns:
        str: The current value of the field
    """
    # Get the current value from the document data
    parts = field_path.split('.')
    data = st.session_state.document_data
    
    # Navigate to the field
    for part in parts[:-1]:
        if part not in data:
            data[part] = {}
        data = data.get(part, {})
    
    # Get the value, with fallback to empty string
    current_value = data.get(parts[-1], "")
    
    # Create a truly unique key using combination of prefix, path, and a random component
    # Add unique_id if provided or generate a random string
    unique_suffix = unique_id if unique_id else f"id_{hash(label)}_{hash(field_path) % 10000}"
    input_key = f"edit_{key_prefix}_{field_path.replace('.', '_')}_{unique_suffix}"
    
    # If in editing mode, show editable input
    if st.session_state.editing_mode:
        new_value = st.text_input(
            label,
            value=current_value,
            placeholder=placeholder if placeholder else "",
            key=input_key,
            on_change=lambda: update_simple_field(field_path, st.session_state[input_key])
        )
        return new_value
    else:
        # In read-only mode, just return the value from session state
        return current_value

def editable_text_area(label, field_path, key_prefix, height=None, placeholder=None):
    """
    Render an editable text area for longer text.
    
    Args:
        label (str): Field label
        field_path (str): Path to the field in document_data
        key_prefix (str): Prefix for the input key
        height (str, optional): CSS height value
        placeholder (str, optional): Placeholder text
    
    Returns:
        str: The current value of the field
    """
    # Get the current value from the document data
    parts = field_path.split('.')
    data = st.session_state.document_data
    
    # Navigate to the field
    for part in parts[:-1]:
        if part not in data:
            data[part] = {}
        data = data.get(part, {})
    
    # Get the value, with fallback to empty string
    current_value = data.get(parts[-1], "")
    
    # Create a unique key for this input
    input_key = f"edit_{key_prefix}_{field_path.replace('.', '_')}"
    
    # If in editing mode, show editable textarea
    if st.session_state.editing_mode:
        new_value = st.text_area(
            label,
            value=current_value,
            placeholder=placeholder if placeholder else "",
            key=input_key,
            on_change=lambda: update_simple_field(field_path, st.session_state[input_key]),
            height=height
        )
        return new_value
    else:
        # In read-only mode, just return the value
        return current_value

def editable_array_field(label, array_path, key_prefix, item_type="text", default_item=None):
    """
    Render an editable array of items.
    
    Args:
        label (str): Field label
        array_path (str): Path to the array in document_data
        key_prefix (str): Prefix for the input keys
        item_type (str): Type of items ("text", "dict", etc.)
        default_item: Default value for new items
    
    Returns:
        list: The current array of items
    """
    # Get the current array from the document data
    parts = array_path.split('.')
    data = st.session_state.document_data
    
    # Navigate to the array
    for part in parts[:-1]:
        if part not in data:
            data[part] = {}
        data = data.get(part, {})
    
    # Get the array or create empty one
    current_array = data.get(parts[-1], [])
    if not isinstance(current_array, list):
        current_array = []
        data[parts[-1]] = current_array
    
    # Display the array label
    st.markdown(f"**{label}**")
    
    # If in editing mode, show editable array
    if st.session_state.editing_mode:
        # Display each item with an edit field and delete button
        for i, item in enumerate(current_array):
            cols = st.columns([5, 1])
            
            with cols[0]:
                if item_type == "text":
                    item_key = f"edit_{key_prefix}_{array_path.replace('.', '_')}_{i}"
                    new_item = st.text_input(
                        f"Item {i+1}",
                        value=item,
                        key=item_key,
                        on_change=lambda idx=i, key=item_key: update_array_item(
                            array_path, idx, st.session_state[key]
                        )
                    )
                elif item_type == "dict" and isinstance(item, dict):
                    # For dictionary items, create a sub-form
                    st.markdown(f"**Item {i+1}**")
                    new_item = {}
                    for key, value in item.items():
                        field_key = f"edit_{key_prefix}_{array_path.replace('.', '_')}_{i}_{key}"
                        new_value = st.text_input(
                            key,
                            value=value,
                            key=field_key,
                            on_change=lambda idx=i, k=key, fk=field_key: update_dict_in_array(
                                array_path, idx, k, st.session_state[fk]
                            )
                        )
            
            with cols[1]:
                if st.button("🗑️", key=f"delete_{key_prefix}_{array_path.replace('.', '_')}_{i}"):
                    remove_array_item(array_path, i)
                    st.rerun()
        
        # Add new item button
        if st.button(f"+ Add {label.rstrip('s')}", key=f"add_{key_prefix}_{array_path.replace('.', '_')}"):
            add_array_item(array_path, default_item)
            st.rerun()
    
    return current_array

def editable_dict_array(label, array_path, dict_keys, key_prefix, default_item=None):
    """
    Render an editable array of dictionaries with specific keys.
    
    Args:
        label (str): Field label
        array_path (str): Path to the array in document_data
        dict_keys (dict): Dictionary of {key: display_name} pairs to edit
        key_prefix (str): Prefix for the input keys
        default_item (dict, optional): Default dictionary for new items
    
    Returns:
        list: The current array of dictionaries
    """
    # Get the current array from the document data
    parts = array_path.split('.')
    data = st.session_state.document_data
    
    # Navigate to the array
    for part in parts[:-1]:
        if part not in data:
            data[part] = {}
        data = data.get(part, {})
    
    # Get the array or create empty one
    current_array = data.get(parts[-1], [])
    if not isinstance(current_array, list):
        current_array = []
        data[parts[-1]] = current_array
    
    # Display the array label
    st.markdown(f"### {label}")
    
    # If in editing mode, show editable array
    if st.session_state.editing_mode:
        # Create columns for each field
        cols_width = [3] * len(dict_keys) + [1]  # Equal width for all fields + delete button
        
        # Header row
        header_cols = st.columns(cols_width)
        for i, (key, display_name) in enumerate(dict_keys.items()):
            with header_cols[i]:
                st.markdown(f"**{display_name}**")
        
        # Divider
        st.markdown("---")
        
        # Display each item with editable fields and delete button
        for i, item in enumerate(current_array):
            row_cols = st.columns(cols_width)
            
            # Edit fields
            for j, (key, display_name) in enumerate(dict_keys.items()):
                with row_cols[j]:
                    field_key = f"edit_{key_prefix}_{array_path.replace('.', '_')}_{i}_{key}"
                    new_value = st.text_input(
                        f"{display_name} #{i+1}",
                        value=item.get(key, ""),
                        key=field_key,
                        label_visibility="collapsed",
                        on_change=lambda idx=i, k=key, fk=field_key: update_dict_in_array(
                            array_path, idx, k, st.session_state[fk]
                        )
                    )
            
            # Delete button
            with row_cols[-1]:
                st.write("")  # Add some space
                if st.button("🗑️", key=f"delete_{key_prefix}_{array_path.replace('.', '_')}_{i}"):
                    remove_array_item(array_path, i)
                    st.rerun()
        
        # Add new item button
        if st.button(f"+ Add {label.rstrip('s')}", key=f"add_{key_prefix}_{array_path.replace('.', '_')}"):
            # Create default item if not provided
            if default_item is None:
                default_item = {key: "" for key in dict_keys.keys()}
            add_array_item(array_path, default_item)
            st.rerun()
    else:
        # Display as a table in read mode
        if current_array:
            # Create a DataFrame from the array
            df = pd.DataFrame(current_array)
            
            # Rename columns based on dict_keys
            renamed_cols = {}
            for key, display_name in dict_keys.items():
                if key in df.columns:
                    renamed_cols[key] = display_name
            
            if renamed_cols:
                df = df.rename(columns=renamed_cols)
            
            # Display the table
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(f"No {label.lower()} data available")
    
    return current_array

# --------------------------------
# GitHub API Integration
# --------------------------------

def get_github_stats(repo_name, access_token=None):
    """
    Fetch GitHub repository statistics using the GitHub API.
    
    Args:
        repo_name (str): Repository name in the format 'owner/repo'
        access_token (str, optional): GitHub API access token
        
    Returns:
        dict: Repository statistics including stars, forks, etc.
    """
    g = Github(access_token) if access_token else Github()
    
    try:
        repo = g.get_repo(repo_name)
        return {
            "name": repo.name,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "created_at": repo.created_at,
            "updated_at": repo.updated_at,
            "open_issues": repo.open_issues_count,
            "subscribers": repo.subscribers_count,
            "html_url": repo.html_url,
            "full_name": repo.full_name
        }
    except Exception as e:
        st.warning(f"Error fetching GitHub metrics: {str(e)}. Using test data instead.")
        return {
            "name": repo_name.split('/')[-1],
            "stars": 40114,
            "forks": 5715,
            "open_issues": 756,
            "subscribers": 1024,
            "html_url": f"https://github.com/{repo_name}",
            "full_name": repo_name
        }

def get_star_history(repo_name, access_token=None):
    """
    Fetch star history for a GitHub repository.
    
    Args:
        repo_name (str): Repository name in the format 'owner/repo'
        access_token (str, optional): GitHub API access token
        
    Returns:
        list: Star history data with dates and cumulative star counts
    """
    g = Github(access_token) if access_token else Github()
    
    try:
        repo = g.get_repo(repo_name)
        
        # Get stargazers with dates (this requires a token with proper permissions)
        stargazers = list(repo.get_stargazers_with_dates())
        
        # Process star history
        star_history = []
        for idx, stargazer in enumerate(stargazers):
            star_history.append({
                "date": stargazer.starred_at.date(),
                "stars": idx + 1
            })
        
        # If there are too many data points, aggregate by month
        if len(star_history) > 100:
            monthly_data = defaultdict(int)
            for item in star_history:
                # Format as YYYY-MM for monthly aggregation
                month_key = item["date"].strftime("%Y-%m")
                monthly_data[month_key] = max(monthly_data[month_key], item["stars"])
            
            # Convert back to list format
            star_history = [
                {"date": datetime.strptime(month, "%Y-%m").date(), "stars": stars}
                for month, stars in sorted(monthly_data.items())
            ]
        
        return star_history
    except Exception as e:
        st.warning(f"Error fetching GitHub star history: {str(e)}. Using test data instead.")
        
        # Generate fake star history for demo purposes
        fake_history = []
        today = datetime.now().date()
        for i in range(12):
            date = today - timedelta(days=30*i)
            stars = 40114 - (1000 * i)
            if stars < 0:
                break
            fake_history.append({
                "date": date,
                "stars": stars
            })
        
        return sorted(fake_history, key=lambda x: x["date"])

def display_github_metrics_ui(stats, compact=False):
    """
    Display GitHub metrics UI in the Streamlit application.
    
    Args:
        stats (dict): GitHub statistics
        compact (bool): If True, display a more compact version for the Executive Summary
    """
    if stats:
        # Display current metrics
        if compact:
            # For Executive Summary - more compact display
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("GitHub Stars", f"{stats['stars']:,}")
            with metrics_col2:
                st.metric("Forks", f"{stats['forks']:,}")
            with metrics_col3:
                st.metric("Open Issues", f"{stats['open_issues']:,}")
        else:
            # Full display for GitHub Analytics tab
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("GitHub Stars", f"{stats['stars']:,}")
            with col2:
                st.metric("Forks", f"{stats['forks']:,}")
            with col3:
                st.metric("Open Issues", f"{stats['open_issues']:,}")
        
        # Add last updated info
        st.markdown(f"""
        <div style="font-size: 0.8rem; color: #6B7280; text-align: right;">
            Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        """, unsafe_allow_html=True)

# --------------------------------
# Excel Export Functionality
# --------------------------------

def create_excel_download_button(report_data):
    """
    Create a download button for exporting all company data to Excel.
    Shows each item (competitor, product, etc.) individually in the preview.
    
    Args:
        report_data (dict): Comprehensive data structure containing all report information
    
    Returns:
        None: Displays a download button in the Streamlit UI
    """
    company_data = report_data["company_data"]
    github_stats = report_data["github_stats"] if "github_stats" in report_data else None
    
    # Create a DataFrame with one row containing all basic company data
    company_df_data = {
        "Company Name": company_data["name"],
        "Founded": company_data.get("founded", ""),
        "Headquarters": company_data.get("headquarters", ""),
        "Website": company_data.get("website", ""),
        "Total Funding": company_data.get("total_funding", ""),
        "Latest Funding Round": company_data.get("latest_funding_round", ""),
        "Monthly Downloads": company_data.get("monthly_downloads", ""),
        "Organizations Using": company_data.get("organizations_using", ""),
        "Fortune 500 Clients": company_data.get("fortune_500_clients", "")
    }
    
    # Add GitHub stats if available
    if github_stats:
        company_df_data["GitHub Stars"] = github_stats.get("stars", "")
        company_df_data["GitHub Forks"] = github_stats.get("forks", "")
        company_df_data["GitHub Open Issues"] = github_stats.get("open_issues", "")
        company_df_data["GitHub Subscribers"] = github_stats.get("subscribers", "")
    
    # Create DataFrame with a single row
    company_df = pd.DataFrame([company_df_data])
    
    try:
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write company overview to first sheet
            company_df.to_excel(writer, sheet_name='Company Overview', index=False)
            
            # Write executive summary
            pd.DataFrame({"Content": [report_data["executive_summary"]["company_summary"]]}).to_excel(
                writer, sheet_name='Executive Summary', index=False)
            
            # Write company description
            pd.DataFrame({"Content": [report_data["company_profile"]["description"]]}).to_excel(
                writer, sheet_name='Company Description', index=False)
            
            # Write leadership team info
            if "leadership_team" in report_data and report_data["leadership_team"]:
                pd.DataFrame(report_data["leadership_team"]).to_excel(
                    writer, sheet_name='Leadership Team', index=False)
            
            # Write product offerings
            if "product_offerings" in report_data and report_data["product_offerings"]:
                products_df = pd.DataFrame([
                    {
                        "Name": p["name"],
                        "Description": p["description"],
                        "Features": ", ".join(p["features"]) if isinstance(p["features"], list) else p["features"],
                        "Launch Date": p.get("launch_date", ""),
                        "User Base": p.get("user_base", "")
                    }
                    for p in report_data["product_offerings"]
                ])
                products_df.to_excel(writer, sheet_name='Product Offerings', index=False)
            
            # Write funding data
            if "funding_data" in report_data and report_data["funding_data"]:
                pd.DataFrame(report_data["funding_data"]).to_excel(
                    writer, sheet_name='Funding History', index=False)
            
            # Write funding analysis
            if "funding_analysis" in report_data:
                pd.DataFrame({"Content": [report_data["funding_analysis"]]}).to_excel(
                    writer, sheet_name='Funding Analysis', index=False)
            
            # Write key technologies
            if "key_technologies" in report_data and report_data["key_technologies"]:
                pd.DataFrame(report_data["key_technologies"]).to_excel(
                    writer, sheet_name='Key Technologies', index=False)
            
            # Write technical differentiation
            if "technical_differentiation" in report_data and report_data["technical_differentiation"]:
                pd.DataFrame([
                    {"Title": t["title"], "Description": t["description"]}
                    for t in report_data["technical_differentiation"]
                ]).to_excel(writer, sheet_name='Technical Differentiation', index=False)
            
            # Write competitors
            if "competitors" in report_data and report_data["competitors"]:
                pd.DataFrame(report_data["competitors"]).to_excel(
                    writer, sheet_name='Competitors', index=False)
            
            # Write notable customers
            if "notable_customers" in report_data and report_data["notable_customers"]:
                pd.DataFrame(report_data["notable_customers"]).to_excel(
                    writer, sheet_name='Notable Customers', index=False)
            
            # Write market advantages
            if "market_advantages" in report_data and report_data["market_advantages"]:
                pd.DataFrame(report_data["market_advantages"]).to_excel(
                    writer, sheet_name='Market Advantages', index=False)
            
            # Write market trends
            if "market_trends" in report_data and report_data["market_trends"]:
                pd.DataFrame(report_data["market_trends"]).to_excel(
                    writer, sheet_name='Market Trends', index=False)
            
            # Write recent news
            if "recent_news" in report_data and report_data["recent_news"]:
                pd.DataFrame(report_data["recent_news"]).to_excel(
                    writer, sheet_name='Recent News', index=False)
            
            # Write strategic direction
            if "strategic_direction" in report_data and report_data["strategic_direction"]:
                pd.DataFrame(report_data["strategic_direction"]).to_excel(
                    writer, sheet_name='Strategic Direction', index=False)
            
            # Write information gaps
            if "information_gaps" in report_data:
                pd.DataFrame({"Information Gap": report_data["information_gaps"]}).to_excel(
                    writer, sheet_name='Information Gaps', index=False)
            
            # Write use cases
            if "company_profile" in report_data and "use_cases" in report_data["company_profile"]:
                pd.DataFrame([
                    {"Title": uc["title"], "Description": uc["description"]}
                    for uc in report_data["company_profile"]["use_cases"]
                ]).to_excel(writer, sheet_name='Use Cases', index=False)
            
            # Format each worksheet for better readability
            workbook = writer.book
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                # Default column width for all sheets
                for i in range(10):  # Assuming max 10 columns
                    worksheet.set_column(i, i, 25)
        
        # Get the Excel data
        excel_data = output.getvalue()
        
        # Get current time for filename
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{company_data['name']}_Report_{current_time}.xlsx"
        
        # Create a preview table of what's included
        st.subheader("Report Contents")
        
        # Create a simple preview with expandable sections
        sections = {
            "Company Overview": "Basic company information and metadata",
            "Executive Summary": "Comprehensive summary of the company",
            "Company Description": "Detailed company description and profile",
            "Leadership Team": f"{len(report_data.get('leadership_team', []))} team members",
            "Product Offerings": f"{len(report_data.get('product_offerings', []))} products",
            "Funding History": f"{len(report_data.get('funding_data', []))} funding rounds",
            "Competitors": f"{len(report_data.get('competitors', []))} competitors",
            "Market Trends": f"{len(report_data.get('market_trends', []))} market trends",
            "Recent News": f"{len(report_data.get('recent_news', []))} news items"
        }
        
        # Display sections as expandable
        for section, description in sections.items():
            with st.expander(f"{section}: {description}"):
                if section == "Company Overview":
                    st.dataframe(company_df, use_container_width=True)
                elif section == "Executive Summary":
                    st.markdown(report_data["executive_summary"]["company_summary"][:500] + "...")
                elif section == "Company Description":
                    st.markdown(report_data["company_profile"]["description"][:500] + "...")
                elif section == "Leadership Team" and report_data.get("leadership_team"):
                    st.dataframe(pd.DataFrame(report_data["leadership_team"]), use_container_width=True)
                elif section == "Product Offerings" and report_data.get("product_offerings"):
                    st.dataframe(pd.DataFrame([{"Name": p["name"], "Description": p["description"][:100] + "..."} 
                                              for p in report_data["product_offerings"]]), use_container_width=True)
                elif section == "Funding History" and report_data.get("funding_data"):
                    st.dataframe(pd.DataFrame([{"Round": f["round"], "Amount": f["amount"], "Date": f["date"]} 
                                              for f in report_data["funding_data"]]), use_container_width=True)
                elif section == "Competitors" and report_data.get("competitors"):
                    st.dataframe(pd.DataFrame([{"Name": c["name"], "Description": c.get("description", c.get("analysis", ""))[:100] + "..."} 
                                              for c in report_data["competitors"]]), use_container_width=True)
                elif section == "Market Trends" and report_data.get("market_trends"):
                    st.dataframe(pd.DataFrame([{"Title": t["title"], "Description": t["description"][:100] + "..."} 
                                              for t in report_data["market_trends"]]), use_container_width=True)
                elif section == "Recent News" and report_data.get("recent_news"):
                    st.dataframe(pd.DataFrame([{"Date": n["date"], "Title": n["title"], "Summary": n["summary"][:100] + "..."} 
                                              for n in report_data["recent_news"]]), use_container_width=True)
        
        # Create download button
        return st.download_button(
            label=f"Download Excel Report",
            data=excel_data,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

# --------------------------------
# Data Preparation Functions
# --------------------------------

def create_default_report_data():
    """
    Create a comprehensive data structure containing all information for the company report.
    This is the demo/default data for LlamaIndex.
    
    Returns:
        dict: Complete data structure for the company report
    """
    # Fetch GitHub stats first as they'll be used in other parts of the report
    github_repo = "jerryjliu/llama_index"
    
    try:
        # Try to get the github token from secrets
        github_token = st.secrets["GITHUB_TOKEN"]
    except:
        # If not available, use None
        github_token = None
    
    github_stats = get_github_stats(github_repo, github_token)
    
    # Format GitHub stars for readable display
    stars_formatted = f"{github_stats['stars'] / 1000:.1f}K" if github_stats['stars'] >= 1000 else f"{github_stats['stars']}"
    
    return {
        "company_data": {
            "name": "LlamaIndex",
            "founded": "2023",
            "headquarters": "San Francisco, CA",
            "website": "https://www.llamaindex.ai",
            "total_funding": "$27.5 million",
            "latest_funding_round": "Series A - February 2024",
            "monthly_downloads": "2.5 million+",
            "organizations_using": "10,000+",
            "fortune_500_clients": "90",
            "industry_tags": ["AI Industry", "Data Infrastructure", "Series A", "RAG Framework"]
        },
        
        "github_stats": github_stats,
        
        "executive_summary": {
            "company_summary": "LlamaIndex is a data framework for Large Language Models (LLMs) that facilitates the development of AI-powered applications by enabling LLMs to connect with and leverage private data from various sources. Founded in 2023, it has quickly established itself as a leading solution for building context-augmented AI agents and Retrieval-Augmented Generation (RAG) pipelines. The company has raised a total of $27.5 million in funding, including an $8.5 million seed round led by Greylock and a $19 million Series A round led by Norwest Venture Partners in February 2024.\n\nThe company offers both open-source and commercial products: LlamaIndex (core open-source framework), LlamaCloud (enterprise knowledge management platform), and LlamaParse (document parsing tool). Its technology is utilized by over 10,000 organizations, including 90 Fortune 500 companies, with particularly strong adoption in the Technology, Financial Services, and Legal sectors. Key customers include Salesforce, Rakuten, Carlyle, and KPMG.\n\nLed by founder and CEO Jerry Liu and CTO Simon Suo, LlamaIndex differentiates itself through RAG pipeline efficiency, document processing capabilities, advanced workflows architecture, and enterprise-grade security (SOC 2 Type 2 certified). The company operates in a competitive landscape that includes LangChain, Lyzr, Dify.AI, and Cohere, but maintains strong market positioning through its balance of simplicity, performance, and enterprise features.\n\nRecent developments include the February 2024 Series A funding round, integration with Salesforce's Agentforce product line, the launch of LlamaCloud with enterprise features, achievement of SOC 2 Type 2 certification, and educational partnerships with DeepLearning.AI. These milestones indicate a strategic focus on enterprise expansion while maintaining a strong open-source community foundation.",
            "strengths": [
                f"Strong open-source community with {stars_formatted} GitHub stars",
                "Recent Series A funding ($19M) to fuel growth",
                "SOC 2 Type 2 certification for enterprise security",
                "Notable enterprise customers including Salesforce",
                "Technical differentiation in document processing"
            ],
            "information_gaps": [
                "Revenue figures and business performance metrics",
                "Detailed pricing model for enterprise offerings",
                "Comprehensive list of all integrations and partnerships",
                "Performance benchmarks for LlamaCloud and LlamaParse",
                "Current employee count",
                "Future product roadmap details"
            ]
        },
        
        "company_profile": {
            "description": "LlamaIndex is a data framework for Large Language Models (LLMs) that facilitates the development of AI-powered applications by enabling LLMs to connect with and leverage private data from various sources. It operates as an open-source framework in both Python and TypeScript, providing tools and templates for building context-augmented AI agents and Retrieval-Augmented Generation (RAG) pipelines.\n\nFounded to address the context size limitations of large language models, LlamaIndex has garnered significant traction within the AI community. It streamlines data ingestion, indexing, and querying, allowing developers and enterprises to build production-ready RAG applications, knowledge assistants, and agentic generative AI applications.\n\nThe company operates with both an open-source framework and commercial offerings, positioning itself at the intersection of developer tools and enterprise AI solutions. The core technology focuses on connecting LLMs with private data sources, enabling more accurate and contextual AI applications.",
            "use_cases": [
                {
                    "title": "Retrieval-Augmented Generation",
                    "description": "Enabling LLMs to access and utilize private data sources for more accurate responses. RAG pipelines connect custom data to language models, improving accuracy and reducing hallucinations.",
                    "style": "blue"
                },
                {
                    "title": "Knowledge Assistants",
                    "description": "Building AI assistants that can answer questions based on domain-specific knowledge bases. These assistants can handle complex queries across large document collections.",
                    "style": "green"
                },
                {
                    "title": "Agentic Applications",
                    "description": "Creating AI agents capable of reasoning and performing complex tasks across enterprise data, automating workflows and decision-making processes.",
                    "style": "purple"
                }
            ]
        },
        
        "leadership_team": [
            {
                "name": "Jerry Liu",
                "position": "Founder & CEO",
                "background": "Prior experience at Robust Intelligence, Uber, Quora, Two Sigma, Apple"
            },
            {
                "name": "Simon Suo",
                "position": "CTO",
                "background": "Technical leadership experience in AI and data infrastructure"
            }
        ],
        
        "product_offerings": [
            {
                "name": "LlamaIndex (Open Source)",
                "description": "Data framework for LLMs available in Python and TypeScript, enabling RAG pipelines and agent building.",
                "features": [
                    "Support for 160+ data formats",
                    "Vector store, summary, and knowledge graph indices",
                    "High & low-level APIs",
                    "Open-source community with 38k GitHub stars"
                ],
                "launch_date": "2023",
                "user_base": "Large open-source community"
            },
            {
                "name": "LlamaCloud",
                "description": "Commercial knowledge management platform for agent development with enterprise features like SOC 2 Type 2 certification.",
                "features": [
                    "SaaS or on-premise deployment",
                    "Role-based access control",
                    "GDPR-compliant EU data residency",
                    "Enterprise-grade security",
                    "Advanced document processing"
                ],
                "launch_date": "December 2023",
                "user_base": "Enterprise customers"
            },
            {
                "name": "LlamaParse",
                "description": "Document parsing tool for complex, unstructured document formats such as PDFs, PowerPoints, and images.",
                "features": [
                    "Advanced parsing capabilities",
                    "Integration with LlamaCloud",
                    "Extraction from unstructured documents",
                    "Support for images, charts, and complex layouts"
                ],
                "launch_date": "2023",
                "user_base": "Integrated with other LlamaIndex products"
            }
        ],
        
        "funding_data": [
            {
                "round": "Seed Round",
                "amount": "$8.5M",
                "investor": "Greylock",
                "date": "Early 2023",
                "notes": "",
                "source_url": "https://www.crunchbase.com/organization/llamaindex/company_financials"
            },
            {
                "round": "Series A",
                "amount": "$19M",
                "investor": "Norwest Venture Partners",
                "date": "February 2024",
                "notes": "Existing investor Greylock also participated",
                "source_url": "https://www.crunchbase.com/organization/llamaindex/company_financials"
            }
        ],
        
        "funding_analysis": "The February 2024 Series A round represents a significant validation of LlamaIndex's market position and growth trajectory. With $27.5M in total funding, the company has secured substantial financial resources to:\n\n- Expand its enterprise offerings with LlamaCloud and LlamaParse\n- Grow its team to accelerate product development\n- Enhance enterprise security features and compliance capabilities\n- Strengthen its market position against competitors like LangChain\n- Develop strategic partnerships with enterprise customers\n\nThe involvement of established venture capital firms like Norwest Venture Partners and Greylock indicates strong investor confidence in LlamaIndex's business model and growth potential. The relatively quick progression from seed to Series A (approximately one year) suggests strong market traction and business momentum.",
        
        "key_technologies": [
            {"category": "Data Ingestion", "items": "160+ data formats including APIs, PDFs, images, SQL databases"},
            {"category": "Indexing Types", "items": "Vector store index, summary index, knowledge graph index"},
            {"category": "Agent Types", "items": "OpenAI Function agent, ReAct agent, Workflows architecture for complex tasks"},
            {"category": "LLM Support", "items": "OpenAI, IBM Granite series, Llama2, and integration with other frameworks"},
            {"category": "Infrastructure", "items": "SaaS offering or on-premise solution with EU data residency option"},
            {"category": "Security", "items": "SOC 2 Type 2 certified, role-based access control, SSO"}
        ],
        
        "technical_differentiation": [
            {
                "title": "RAG Pipeline Efficiency",
                "description": "LlamaIndex provides efficient document-based RAG systems with hierarchical structures, offering a balance of simplicity and performance that outperforms competitors for many use cases.",
                "style": "blue"
            },
            {
                "title": "Document Processing Superiority",
                "description": "LlamaParse excels at parsing complex, unstructured document formats, with particular strength in table reconstruction and handling multi-modal content.",
                "style": "green"
            },
            {
                "title": "Workflows Architecture",
                "description": "The event-driven Workflows architecture enables complex, customizable agent workflows with advanced orchestration capabilities for enterprise scenarios.",
                "style": "purple"
            },
            {
                "title": "Enterprise Security Focus",
                "description": "SOC 2 Type 2 certification, GDPR compliance, and EU data residency options demonstrate a commitment to enterprise-grade security that surpasses many competitors.",
                "style": "yellow"
            }
        ],
        
        "competitors": [
            {
                "name": "LangChain",
                "description": "A framework for developing applications powered by language models. It offers more granular control over different components in the RAG pipeline and broader integration support, but can be more complex for simpler use cases.",
                "founded": "2022"
            },
            {
                "name": "Lyzr",
                "description": "An AI agent framework with pre-built agents like Jazon (AI sales) and Skott (content marketing), but with less focus on data framework capabilities.",
                "founded": ""
            },
            {
                "name": "Dify.AI",
                "description": "A generative AI platform providing tools for creating, orchestrating, and managing AI workflows and agents. Founded in 2023 and based in Middletown, Delaware.",
                "founded": "2023"
            },
            {
                "name": "Cohere",
                "description": "An enterprise AI solution provider with an established presence in enterprise AI solutions but a different primary focus than LlamaIndex's data framework approach. Founded in 2019 and based in Toronto, Canada.",
                "founded": "2019"
            }
        ],
        
        "notable_customers": [
            {"name": "Salesforce", "industry": "Enterprise Software", "useCase": "Agentforce development"},
            {"name": "Rakuten", "industry": "E-commerce", "useCase": "RAG performance enhancements"},
            {"name": "Carlyle", "industry": "Private Equity", "useCase": "Financial data analysis"},
            {"name": "KPMG", "industry": "Professional Services", "useCase": "Knowledge management solutions"}
        ],
        
        "market_advantages": [
            {
                "title": "Document Processing Excellence", 
                "description": "Superior handling of complex, unstructured documents with LlamaParse outperforms competitors in multi-modal content processing."
            },
            {
                "title": "Simplicity-Performance Balance", 
                "description": "Balanced approach providing both developer-friendly APIs and deep customization options."
            },
            {
                "title": "Enterprise Security Credentials", 
                "description": "SOC 2 Type 2 certification and EU data residency options position the company well for enterprise adoption in regulated industries."
            },
            {
                "title": "Open Source Community", 
                "description": "Strong open-source foundation with 38,000+ GitHub stars provides wide developer adoption and community contributions."
            }
        ],
        
        "market_trends": [
            {
                "title": "Rising Demand for RAG Solutions", 
                "description": "Organizations increasingly recognize the need to connect private data sources to LLMs, driving demand for robust RAG solutions."
            },
            {
                "title": "Growth in Agentic AI Applications", 
                "description": "Enterprise interest in agentic AI applications is accelerating, with companies seeking tools that can automate complex workflows and decision processes."
            },
            {
                "title": "Enterprise Security Requirements", 
                "description": "As AI adoption grows in regulated industries, demand for enterprise-grade security and compliance features is becoming a key differentiator."
            },
            {
                "title": "Multi-Modal Data Processing", 
                "description": "The ability to handle and extract insights from diverse data formats (text, tables, images) is increasingly important for comprehensive AI solutions."
            }
        ],
        
        "recent_news": [
            {
                "date": "February 2024",
                "title": "LlamaIndex Raises $19 Million Series A Funding",
                "summary": "LlamaIndex announced a $19 million Series A funding round led by Norwest Venture Partners with participation from Greylock. This funding will accelerate the development of its generative AI agent platform and expand its enterprise offerings.",
                "url": "https://www.prnewswire.com/news-releases/llamaindex-raises-19-million-series-a-funding-to-accelerate-generative-ai-agent-platform-and-enterprise-offerings-301734111.html"
            },
            {
                "date": "January 2024",
                "title": "Salesforce Integration with LlamaIndex for Agentforce",
                "summary": "Salesforce announced integration of LlamaIndex technology to accelerate development of its Agentforce product line, allowing for improved data retrieval and agent capabilities.",
                "url": "https://www.salesforce.com/news/announcements/2024/01/llamaindex-agentforce/"
            },
            {
                "date": "December 2023",
                "title": "LlamaCloud Launch with Enterprise Features",
                "summary": "LlamaIndex launched LlamaCloud, a commercial knowledge management platform to power generative AI stacks with improved accuracy for agent workflows over unstructured data.",
                "url": "https://www.llamaindex.ai/blog/llamacloud-launch"
            },
            {
                "date": "November 2023",
                "title": "SOC 2 Type 2 Certification Achievement",
                "summary": "LlamaIndex announced its LlamaCloud platform had achieved SOC 2 Type 2 certification, highlighting the company's commitment to enterprise-grade security and compliance.",
                "url": "https://www.llamaindex.ai/blog/soc-2-type-2-certification"
            },
            {
                "date": "October 2023",
                "title": "Partnership with DeepLearning.AI for Educational Courses",
                "summary": "LlamaIndex partnered with Andrew Ng's DeepLearning.AI to create three educational courses focused on retrieval-augmented generation and building AI applications with private data.",
                "url": "https://www.deeplearning.ai/courses/retrieval-augmented-generation/"
            }
        ],
        
        "strategic_direction": [
            {
                "title": "Enterprise Expansion", 
                "description": "The launch of LlamaCloud and SOC 2 Type 2 certification demonstrate a clear pivot toward enterprise customers and regulated industries."
            },
            {
                "title": "Strategic Partnerships", 
                "description": "Collaboration with Salesforce for Agentforce and educational partnerships with DeepLearning.AI show a dual focus on enterprise integration and developer education."
            },
            {
                "title": "Open Source Foundation", 
                "description": "Maintaining a strong open-source community while building commercial offerings follows a proven business model in the developer tools space."
            }
        ],
        
        "information_gaps": [
            "Specific revenue figures for LlamaIndex",
            "Detailed pricing model for enterprise offerings",
            "Comprehensive list of all integrations and partnerships",
            "Performance benchmarks for LlamaCloud and LlamaParse",
            "Current employee count",
            "Future product roadmap details"
        ],
                
        "github_analysis": {
            "importance": "GitHub metrics indicate community engagement and project activity."
        },
        
        "report_metadata": {
            "report_id": "BI-2025-03-18-LI-01",
            "report_date": datetime.now().strftime("%B %d, %Y"),
            "data_confidence": "High",
            "disclaimer": "This report is based on publicly available information gathered through automated business intelligence processes. Information should be verified through additional research for critical business decisions."
        }
    }

def ensure_report_compatibility(report_data):
    """
    Ensure report data is compatible with the dashboard requirements.
    
    Args:
        report_data (dict): The loaded report data
        
    Returns:
        dict: Updated report data with all required fields
    """
    # Handle missing report_metadata
    if "report_metadata" not in report_data:
        if "metadata" in report_data:
            report_data["report_metadata"] = report_data["metadata"].copy()
            
            # Add additional required metadata fields
            if "report_date" not in report_data["report_metadata"]:
                report_data["report_metadata"]["report_date"] = report_data["report_metadata"].get("generated_on", datetime.now().strftime("%B %d, %Y"))
            
            if "report_id" not in report_data["report_metadata"]:
                company_initial = report_data['company_data']['name'][0].upper() if report_data['company_data']['name'] else "X"
                report_data["report_metadata"]["report_id"] = f"BI-LEGACY-{company_initial}-01"
            
            if "data_confidence" not in report_data["report_metadata"]:
                report_data["report_metadata"]["data_confidence"] = "Medium"
            
            if "disclaimer" not in report_data["report_metadata"]:
                report_data["report_metadata"]["disclaimer"] = "This report is based on publicly available information."
        else:
            # Create basic report metadata
            report_data["report_metadata"] = {
                "report_date": datetime.now().strftime("%B %d, %Y"),
                "report_id": f"BI-LEGACY-{report_data['company_data']['name'][0].upper()}-01",
                "data_confidence": "Medium",
                "generated_on": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "report_version": "1.0",
                "disclaimer": "This report is based on publicly available information."
            }
    
    # Ensure company_data has all required fields
    if "company_data" in report_data:
        required_company_fields = {
            "industry_tags": ["Technology"],
            "founded": "",
            "headquarters": "",
            "website": "",
            "total_funding": "",
            "latest_funding_round": "",
            "monthly_downloads": "",
            "organizations_using": "",
            "fortune_500_clients": "",
            "business_model": "",
            "company_size": "",
            "description": ""
        }
        
        for field, default_value in required_company_fields.items():
            if field not in report_data["company_data"]:
                report_data["company_data"][field] = default_value
    
    # Fix executive_summary structure
    if "executive_summary" in report_data:
        # Handle company_summary vs overview field mismatch
        if "overview" in report_data["executive_summary"] and "company_summary" not in report_data["executive_summary"]:
            report_data["executive_summary"]["company_summary"] = report_data["executive_summary"]["overview"]
        elif "company_summary" in report_data["executive_summary"] and "overview" not in report_data["executive_summary"]:
            report_data["executive_summary"]["overview"] = report_data["executive_summary"]["company_summary"]
        
        # Ensure strengths field exists
        if "strengths" not in report_data["executive_summary"]:
            report_data["executive_summary"]["strengths"] = [
                "Technology provider",
                "Market participant",
                "Industry solution"
            ]
    else:
        # Create basic executive summary
        report_data["executive_summary"] = {
            "overview": f"Information about {report_data['company_data']['name']}",
            "company_summary": f"Information about {report_data['company_data']['name']}",
            "information_gaps": [],
            "confidence_level": "Medium",
            "additional_research_needed": True,
            "strengths": ["Technology provider"]
        }
    
    # Fix product_offerings structure with additional fields
    if "product_offerings" in report_data:
        for product in report_data["product_offerings"]:
            # Add required fields if missing
            if "launch_date" not in product:
                product["launch_date"] = ""
            if "user_base" not in product:
                product["user_base"] = ""
            
            # Critical fix: Ensure features is always a list, never None
            if "features" not in product or product["features"] is None:
                product["features"] = []
            elif not isinstance(product["features"], list):
                # Convert to list if it's not already
                product["features"] = [str(product["features"])]
            
            # Ensure no None values in features list
            if product["features"]:
                product["features"] = [f or "" for f in product["features"]]
                
            if "target_market" not in product or product["target_market"] is None:
                product["target_market"] = "General market"
            
            # Ensure description is never None
            if "description" not in product or product["description"] is None:
                product["description"] = f"{product.get('name', 'Product')} offering"
    
    # Fix funding_data structure
    if "funding_data" in report_data:
        for funding in report_data["funding_data"]:
            if "round" not in funding or funding["round"] is None:
                funding["round"] = ""
            if "date" not in funding or funding["date"] is None:
                funding["date"] = ""
            if "amount" not in funding or funding["amount"] is None:
                funding["amount"] = ""
            if "source_url" not in funding or funding["source_url"] is None:
                funding["source_url"] = ""
            if "investors" not in funding or funding["investors"] is None:
                funding["investors"] = []
            elif isinstance(funding["investors"], str):
                # Split string of investors into a list
                funding["investors"] = [inv.strip() for inv in funding["investors"].split(",") if inv.strip()]
                
            if "lead_investor" not in funding or funding["lead_investor"] is None:
                funding["lead_investor"] = ""
    
    # Fix competitors structure
    if "competitors" in report_data:
        for competitor in report_data["competitors"]:
            if "strengths" not in competitor or competitor["strengths"] is None:
                competitor["strengths"] = []
            if "weaknesses" not in competitor or competitor["weaknesses"] is None:
                competitor["weaknesses"] = []
            if "market_share" not in competitor or competitor["market_share"] is None:
                competitor["market_share"] = ""
                
            # Critical fix: Handle description/analysis field mismatch
            if "analysis" in competitor and "description" not in competitor:
                competitor["description"] = competitor["analysis"]
            elif "description" in competitor and "analysis" not in competitor:
                competitor["analysis"] = competitor["description"]
            elif "description" not in competitor and "analysis" not in competitor:
                competitor["description"] = f"Competitor in the {report_data['company_data']['name']} market space"
                competitor["analysis"] = competitor["description"]
                
            if "name" not in competitor or competitor["name"] is None:
                competitor["name"] = "Unnamed Competitor"
                
            # Ensure description is not None
            if competitor.get("description") is None:
                competitor["description"] = competitor.get("analysis", "Competitor information")
            if competitor.get("analysis") is None:
                competitor["analysis"] = competitor.get("description", "Competitor information")
                
            # Add founded field if missing
            if "founded" not in competitor:
                competitor["founded"] = ""
    
    # Fix market_trends structure
    if "market_trends" in report_data:
        for i, trend in enumerate(report_data["market_trends"]):
            # Critical fix: Handle name/title field mismatch
            if "name" in trend and "title" not in trend:
                trend["title"] = trend["name"]
            elif "title" in trend and "name" not in trend:
                trend["name"] = trend["title"]
            elif "name" not in trend and "title" not in trend:
                trend["name"] = f"Market Trend {i+1}"
                trend["title"] = trend["name"]
            
            # Ensure description is not None
            if "description" not in trend or trend["description"] is None:
                trend["description"] = "Industry trend affecting the market"
                
            # Ensure other required fields
            if "impact_level" not in trend or trend["impact_level"] is None:
                trend["impact_level"] = "Medium"
            if "timeline" not in trend or trend["timeline"] is None:
                trend["timeline"] = "Current"
            if "supporting_data" not in trend or trend["supporting_data"] is None:
                trend["supporting_data"] = ""
    
    # Fix recent_news structure
    if "recent_news" in report_data:
        for news in report_data["recent_news"]:
            if "date" not in news or news["date"] is None:
                news["date"] = "Recent"
            if "title" not in news or news["title"] is None:
                news["title"] = "Company News"
            if "summary" not in news or news["summary"] is None:
                news["summary"] = "Recent company update"
            if "url" not in news or news["url"] is None:
                news["url"] = ""
    
    # Ensure GitHub stats have all required fields
    if "github_stats" in report_data:
        github_stats = report_data["github_stats"]
        
        if "html_url" not in github_stats:
            # Don't construct fake URLs, just set a flag we can check later
            github_stats["needs_search"] = True
            
            # Store company name and repo name for later search
            github_stats["search_company"] = report_data['company_data']['name']
            
            # Still need full_name for other references
            if "full_name" not in github_stats:
                if "name" in github_stats:
                    # Just store the name component, not a fake path
                    github_stats["name_component"] = github_stats["name"]
                github_stats["full_name"] = github_stats.get("name", "")
    
    # Ensure all required sections exist
    required_sections = {
        "company_profile": {
            "description": report_data["company_data"].get("description", f"Company profile for {report_data['company_data']['name']}"),
            "use_cases": [
                {"title": "Primary Use Case", "description": "Main product application", "style": "blue"},
                {"title": "Secondary Use Case", "description": "Additional application", "style": "green"},
                {"title": "Industry Application", "description": "Specific industry use", "style": "purple"}
            ]
        },
        "leadership_team": [],
        "product_offerings": [],
        "funding_data": [],
        "key_technologies": [
            {"category": "Primary Technology", "items": "Core technology offering"},
            {"category": "Infrastructure", "items": "Technology infrastructure"},
            {"category": "Integration", "items": "Integration capabilities"}
        ],
        "technical_differentiation": [
            {"title": "Technology Advantage", "description": "Key technological advantage", "style": "blue"},
            {"title": "Architectural Approach", "description": "Architectural approach", "style": "green"}
        ],
        "market_advantages": [
            {"title": "Market Position", "description": "Position in the market"}
        ],
        "competitors": [],
        "notable_customers": [
            {"name": "Example Customer", "industry": "Various", "useCase": "Example use case"}
        ],
        "market_trends": [],
        "recent_news": [],
        "strategic_direction": [
            {"title": "Business Strategy", "description": "Company's business strategy"}
        ],
        "information_gaps": [],
        "github_stats": {
            "name": report_data['company_data']['name'].lower().replace(' ', ''),
            "stars": 0,
            "forks": 0,
            "open_issues": 0,
            "subscribers": 0
        },
        "github_analysis": {
            "importance": "GitHub metrics indicate community engagement and project activity."
        },
        "funding_analysis": "Funding information for the company."
    }
    
    for section, default_value in required_sections.items():
        if section not in report_data:
            report_data[section] = default_value
    
    return report_data

# --------------------------------
# Editable Report Header & Footer
# --------------------------------

def render_editable_report_header(original_data, report_metadata):
    """Render an enhanced report header with editing capabilities"""
    # First, check if editing mode has been initialized
    initialize_edit_mode()
    
    # Always use session state data instead of original data
    company_data = st.session_state.document_data["company_data"]
    report_metadata = st.session_state.document_data["report_metadata"]
    
    # Apply CSS for editable content
    apply_editable_css()
    
    # Add editing mode toggle
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        previous_mode = st.session_state.get('editing_mode', False)
        current_mode = st.toggle(
            "Enable Editing Mode", 
            value=previous_mode
        )
        
        # Only update the mode if it changed (prevents reinitialization)
        if current_mode != previous_mode:
            st.session_state.editing_mode = current_mode
            # No data refresh needed, just display changes
        
        if st.session_state.editing_mode:
            st.info("📝 Editing mode is enabled. Click on any text to edit. Changes are saved automatically.")
    
    # Create the full header HTML with proper wrapper
    st.markdown('<div class="report-header-wrapper">', unsafe_allow_html=True)
    
    # Title and subtitle section
    st.markdown('<div style="display: flex; flex-direction: row;">', unsafe_allow_html=True)
    
    # Left column (title and tags) - approximately 60% width
    st.markdown('<div style="flex: 3;">', unsafe_allow_html=True)
    
    # Editable company name
    if st.session_state.editing_mode:
        company_name = editable_text_input("Company Name", "company_data.name", "header", unique_id="main_header")
    else:
        company_name = company_data["name"]  # Using session state data
    
    st.markdown(f'<div class="company-title">{company_name}</div>', unsafe_allow_html=True)
    st.markdown('<div class="report-subtitle">Business Intelligence Report</div>', unsafe_allow_html=True)
    
    # Display tag container with all types of tags
    st.markdown('<div class="tag-container">', unsafe_allow_html=True)
    
    # Editable industry tags
    if st.session_state.editing_mode:
        industry_tags = editable_array_field("Industry Tags", "company_data.industry_tags", "header", "text", "New Tag")
    else:
        # Display industry tags - using session state data
        for tag in company_data["industry_tags"]:
            if "Series" in tag:
                tag_class = "green-tag"
            elif "AI" in tag:
                tag_class = "purple-tag"
            else:
                tag_class = "blue-tag"
            st.markdown(f'<span class="tag {tag_class}">{tag}</span>', unsafe_allow_html=True)
    
    # Add key company characteristic tags (in view mode)
    if not st.session_state.editing_mode:
        if "founded" in company_data and company_data["founded"]:
            st.markdown(f'<span class="tag yellow-tag">Founded {company_data["founded"]}</span>', unsafe_allow_html=True)
        
        if "headquarters" in company_data and company_data["headquarters"]:
            st.markdown(f'<span class="tag green-tag">{company_data["headquarters"]}</span>', unsafe_allow_html=True)
        
        if "business_model" in company_data and company_data["business_model"]:
            st.markdown(f'<span class="tag purple-tag">{company_data["business_model"]}</span>', unsafe_allow_html=True)
        
        if "company_size" in company_data and company_data["company_size"]:
            st.markdown(f'<span class="tag blue-tag">{company_data["company_size"]}</span>', unsafe_allow_html=True)
        
        # Add total funding if available
        if "total_funding" in company_data and company_data["total_funding"]:
            st.markdown(f'<span class="tag green-tag">Funding: {company_data["total_funding"]}</span>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close tag container
    
    # In edit mode, add fields for key company info
    if st.session_state.editing_mode:
        edit_col1, edit_col2 = st.columns(2)
        
        with edit_col1:
            editable_text_input("Founded", "company_data.founded", "header", unique_id="founded_field")
            editable_text_input("Headquarters", "company_data.headquarters", "header", unique_id="hq_field")
            editable_text_input("Total Funding", "company_data.total_funding", "header", unique_id="funding_field")
        
        with edit_col2:
            editable_text_input("Business Model", "company_data.business_model", "header", unique_id="model_field")
            editable_text_input("Company Size", "company_data.company_size", "header", unique_id="size_field")
            editable_text_input("Latest Funding Round", "company_data.latest_funding_round", "header", unique_id="round_field")
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close left column
    
    # Right column (metadata) - approximately 40% width
    st.markdown('<div style="flex: 2.5;">', unsafe_allow_html=True)
    st.markdown('<div class="metadata">', unsafe_allow_html=True)
    
    # Editable report metadata
    if st.session_state.editing_mode:
        report_date = editable_text_input("Report Date", "report_metadata.report_date", "header", unique_id="date_field")
        data_confidence = editable_text_input("Data Confidence", "report_metadata.data_confidence", "header", unique_id="confidence_field")
        report_id = editable_text_input("Report ID", "report_metadata.report_id", "header", unique_id="id_field")
        report_version = editable_text_input("Version", "report_metadata.report_version", "header", unique_id="version_field")
    else:
        # Use session state data
        report_date = report_metadata["report_date"]
        data_confidence = report_metadata["data_confidence"]
        report_id = report_metadata["report_id"]
        report_version = report_metadata.get("report_version", "1.0")
    
    st.markdown(f'<div class="metadata-row">Report Generated: <span class="metadata-highlight">{report_date}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metadata-row">Data Confidence: <span class="high-confidence">{data_confidence}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metadata-row">Report ID: <span>{report_id}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metadata-row">Version: <span>{report_version}</span></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close metadata div
    st.markdown('</div>', unsafe_allow_html=True)  # Close right column
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close flex container
    st.markdown('</div>', unsafe_allow_html=True)  # Close report-header-wrapper
    

def render_report_footer(report_data):
    """Render the report footer"""
    company_name = report_data["company_data"]["name"]
    report_date = report_data["report_metadata"]["report_date"]
    
    # Editable disclaimer
    if st.session_state.editing_mode:
        disclaimer = editable_text_area("Disclaimer", "report_metadata.disclaimer", "footer")
    else:
        disclaimer = report_data["report_metadata"]["disclaimer"]
    
    st.markdown(f"""
    <div class="footer">
        <p>Business Intelligence Report for {company_name} | Generated on {report_date}</p>
        <p style="margin-top: 0.5rem;">Data sources include company websites, news articles, and public information</p>
        <p style="margin-top: 0.5rem;">{disclaimer}</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------
# Has GitHub Info Function
# --------------------------------

def has_valid_github_info(report_data):
    """
    Check if the report has valid GitHub information.
    
    Args:
        report_data (dict): The report data dictionary
        
    Returns:
        bool: True if valid GitHub info exists, False otherwise
    """
    # Check if github_stats exists and has required fields
    if "github_stats" not in report_data:
        return False
        
    github_stats = report_data["github_stats"]
    
    # Check if stats has non-zero values (indicating real data)
    has_stars = github_stats.get("stars", 0) > 0
    has_forks = github_stats.get("forks", 0) > 0
    
    # Consider valid if it has some stats
    return (has_stars or has_forks)

def create_navigation_tabs(report_data):
    """Create and return the navigation tabs, conditionally including GitHub tab"""
    
    # Define base tabs that are always shown
    tabs = [
        "Executive Summary", 
        "Company Profile", 
        "Products & Technology",
        "Leadership & Funding",
        "Market & Competition",
        "News & Developments"
    ]
    
    # Add GitHub tab conditionally
    if has_valid_github_info(report_data):
        tabs.append("GitHub Analytics")
    
    return st.tabs(tabs)

# --------------------------------
# Editable Tab Content Functions
# --------------------------------

def render_editable_executive_summary_tab(report_data):
    """Render the Executive Summary tab with editing capabilities"""
    st.markdown('<div class="tab-content" style="padding: 0.2rem; text-align: center;"><h2 class="section-title">Executive Summary</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-title">Company Summary</h3>', unsafe_allow_html=True)
    
    # Editable company summary
    if st.session_state.editing_mode:
        company_summary = editable_text_area(
            "Company Summary", 
            "executive_summary.company_summary", 
            "summary", 
            height=300
        )
        st.markdown(company_summary)
    else:
        st.markdown(report_data["executive_summary"]["company_summary"])
    
    # Only show GitHub metrics if valid GitHub info exists
    if has_valid_github_info(report_data):
        st.markdown('<h3 class="subsection-title">Real-Time GitHub Metrics</h3>', unsafe_allow_html=True)
        st.markdown("""
        <p style="margin-bottom: 1rem;">
            The following metrics are pulled in real-time from GitHub's API, showing current community engagement with the company's open-source repository:
        </p>
        """, unsafe_allow_html=True)
        
        # Display GitHub metrics UI
        display_github_metrics_ui(report_data["github_stats"], compact=True)
    
    # Strengths and Information Gaps
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4 style="font-weight: 600; color: #1E40AF; margin-bottom: 0.5rem;">Strengths</h4>', unsafe_allow_html=True)
        
        # Editable strengths
        if st.session_state.editing_mode:
            strengths = editable_array_field(
                "Strengths", 
                "executive_summary.strengths", 
                "summary", 
                "text", 
                "New strength"
            )
            
            # Display strengths in read-only format for consistency
            strengths_html = ""
            for strength in strengths:
                strengths_html += f"<li>{strength}</li>"
            
            st.markdown(f"""
            <div style="background-color: #EFF6FF; padding: 1rem; border-radius: 0.5rem;">
                <ul style="margin-left: 1.5rem;">
                    {strengths_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            strengths_html = ""
            for strength in report_data["executive_summary"]["strengths"]:
                strengths_html += f"<li>{strength}</li>"
            
            st.markdown(f"""
            <div style="background-color: #EFF6FF; padding: 1rem; border-radius: 0.5rem;">
                <ul style="margin-left: 1.5rem;">
                    {strengths_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h4 style="font-weight: 600; color: #92400E; margin-bottom: 0.5rem;">Information Gaps</h4>', unsafe_allow_html=True)
        
        # Editable information gaps
        if st.session_state.editing_mode:
            gaps = editable_array_field(
                "Information Gaps", 
                "executive_summary.information_gaps", 
                "summary", 
                "text", 
                "New information gap"
            )
            
            # Display gaps in read-only format for consistency
            if gaps and len(gaps) > 0:
                gaps_html = ""
                for gap in gaps:
                    gaps_html += f"<li>{gap}</li>"
                
                st.markdown(f"""
                <div style="background-color: #FFFBEB; padding: 1rem; border-radius: 0.5rem;">
                    <ul style="margin-left: 1.5rem;">
                        {gaps_html}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #FFFBEB; padding: 1rem; border-radius: 0.5rem;">
                    <p>No significant information gaps identified for this company.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            if report_data["executive_summary"]["information_gaps"] and len(report_data["executive_summary"]["information_gaps"]) > 0:
                gaps_html = ""
                for gap in report_data["executive_summary"]["information_gaps"]:
                    gaps_html += f"<li>{gap}</li>"
                
                st.markdown(f"""
                <div style="background-color: #FFFBEB; padding: 1rem; border-radius: 0.5rem;">
                    <ul style="margin-left: 1.5rem;">
                        {gaps_html}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #FFFBEB; padding: 1rem; border-radius: 0.5rem;">
                    <p>No significant information gaps identified for this company.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Add Excel download button
    st.markdown('<h3 class="subsection-title">Export Report Data</h3>', unsafe_allow_html=True)
    st.markdown("""
    <p style="margin-bottom: 1rem;">
        Download all company data as an Excel file for offline analysis and reporting:
    </p>
    """, unsafe_allow_html=True)
    
    # Create and display the download button
    create_excel_download_button(st.session_state.document_data if st.session_state.editing_mode else report_data)
    
    # Add export edited data JSON button
    if st.session_state.editing_mode:
        st.markdown("### Export Edited JSON Data")
        if st.button("Export Edited Report Data (JSON)"):
            # Get the current document data
            export_data = st.session_state.document_data
            
            # Add a timestamp to the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            company_name = export_data["company_data"]["name"]
            filename = f"{company_name}_report_edited_{timestamp}.json"
            
            # Convert to JSON with proper formatting
            json_data = json.dumps(export_data, indent=2, default=str)
            
            # Create a download button
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=filename,
                mime="application/json"
            )
            
            st.success(f"Data prepared for export as {filename}")
    
    st.markdown(f"""
    <div style="background-color: #F9FAFB; padding: 1rem; border-radius: 0.5rem; margin-top: 1.5rem; font-size: 0.875rem;">
        <p style="color: #4B5563;">
            <strong>Note:</strong> {report_data["report_metadata"]["disclaimer"]}
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_editable_company_profile_tab(report_data):
    """Render the Company Profile tab with editing capabilities"""
    st.markdown('<div class="tab-content" style="padding: 0.2rem; text-align: center;"><h2 class="section-title">Company Profile</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="subsection-title">Company Overview</h3>', unsafe_allow_html=True)
        
        # Create editable company metrics
        if st.session_state.editing_mode:
            # Direct editable fields
            company_metrics = []
            
            name = editable_text_input("Company Name", "company_data.name", "profile")
            company_metrics.append({"Metric": "Company Name", "Value": name})
            
            founded = editable_text_input("Founded", "company_data.founded", "profile")
            company_metrics.append({"Metric": "Founded", "Value": founded})
            
            headquarters = editable_text_input("Headquarters", "company_data.headquarters", "profile")
            company_metrics.append({"Metric": "Headquarters", "Value": headquarters})
            
            website = editable_text_input("Website", "company_data.website", "profile")
            company_metrics.append({"Metric": "Website", "Value": website})
            
            total_funding = editable_text_input("Total Funding", "company_data.total_funding", "profile")
            company_metrics.append({"Metric": "Total Funding", "Value": total_funding})
            
            latest_funding = editable_text_input("Latest Funding", "company_data.latest_funding_round", "profile")
            company_metrics.append({"Metric": "Latest Funding", "Value": latest_funding})
            
            # Additional fields if available
            monthly_downloads = editable_text_input("Monthly Downloads", "company_data.monthly_downloads", "profile")
            if monthly_downloads:
                company_metrics.append({"Metric": "Monthly Downloads", "Value": monthly_downloads})
            
            orgs_using = editable_text_input("Organizations Using", "company_data.organizations_using", "profile")
            if orgs_using:
                company_metrics.append({"Metric": "Organizations Using", "Value": orgs_using})
            
            fortune_clients = editable_text_input("Fortune 500 Clients", "company_data.fortune_500_clients", "profile")
            if fortune_clients:
                company_metrics.append({"Metric": "Fortune 500 Clients", "Value": fortune_clients})
            
            company_size = editable_text_input("Company Size", "company_data.company_size", "profile")
            if company_size:
                company_metrics.append({"Metric": "Company Size", "Value": company_size})
            
            business_model = editable_text_input("Business Model", "company_data.business_model", "profile")
            if business_model:
                company_metrics.append({"Metric": "Business Model", "Value": business_model})
        else:
            # Regular display using the original data
            company_data = report_data["company_data"]
            
            # Create key-value pairs for company metrics
            company_metrics = [
                {"Metric": "Company Name", "Value": company_data["name"]},
                {"Metric": "Founded", "Value": company_data["founded"]},
                {"Metric": "Headquarters", "Value": company_data["headquarters"]},
                {"Metric": "Website", "Value": company_data["website"]},
                {"Metric": "Total Funding", "Value": company_data["total_funding"]},
                {"Metric": "Latest Funding", "Value": company_data["latest_funding_round"]},
            ]
            
            # Add new fields if available
            if company_data.get("monthly_downloads"):
                company_metrics.append({"Metric": "Monthly Downloads", "Value": company_data["monthly_downloads"]})
            
            if company_data.get("organizations_using"):
                company_metrics.append({"Metric": "Organizations Using", "Value": company_data["organizations_using"]})
            
            if company_data.get("fortune_500_clients"):
                company_metrics.append({"Metric": "Fortune 500 Clients", "Value": company_data["fortune_500_clients"]})
            
            if company_data.get("company_size"):
                company_metrics.append({"Metric": "Company Size", "Value": company_data["company_size"]})
            
            if company_data.get("business_model"):
                company_metrics.append({"Metric": "Business Model", "Value": company_data["business_model"]})
        
        # Company metrics table
        company_df = pd.DataFrame(company_metrics)
        
        st.markdown('<div style="background-color: #F9FAFB; padding: 1rem; border-radius: 0.5rem;">', unsafe_allow_html=True)
        st.table(company_df)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 class="subsection-title">Company Description</h3>', unsafe_allow_html=True)
        
        # Editable company description
        if st.session_state.editing_mode:
            description = editable_text_area(
                "Description", 
                "company_profile.description", 
                "profile", 
                height=300
            )
            st.markdown(description)
        else:
            # Display description if available, otherwise display executive summary
            if report_data["company_profile"]["description"]:
                st.markdown(report_data["company_profile"]["description"])
            else:
                st.markdown(report_data["executive_summary"]["company_summary"])
        
        # Display industry tags in read mode
        if not st.session_state.editing_mode and report_data["company_data"]["industry_tags"]:
            industry_tags_html = ""
            for tag in report_data["company_data"]["industry_tags"]:
                if "Series" in tag:
                    tag_class = "green-tag"
                elif "AI" in tag:
                    tag_class = "purple-tag"
                else:
                    tag_class = "blue-tag"
                industry_tags_html += f'<span class="tag {tag_class}">{tag}</span>'
            
            st.markdown(f"""
            <div style="margin-top: 1rem;">
                <h4 style="font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem;">Industry Tags:</h4>
                <div>{industry_tags_html}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Editable Notable Customers section
    st.markdown('<h3 class="subsection-title">Notable Customers</h3>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Using the editable_dict_array for a cleaner interface
        editable_dict_array(
            "Notable Customers",
            "notable_customers",
            {"name": "Company", "industry": "Industry", "useCase": "Use Case"},
            "customer",
            {"name": "New Customer", "industry": "Industry", "useCase": "Use Case"}
        )
    else:
        # Regular display
        has_real_customers = (report_data.get("notable_customers") and 
                          len(report_data["notable_customers"]) > 0 and 
                          report_data["notable_customers"][0]["name"] != "Example Customer" and
                          report_data["notable_customers"][0]["name"] != "Customer Example")

        if has_real_customers:
            # Create customer dataframe
            customers_df = pd.DataFrame(report_data["notable_customers"])
            st.dataframe(customers_df, hide_index=True, use_container_width=True)
            
            st.markdown(f"""
            <p style="font-size: 0.875rem; color: #6B7280; margin-top: 0.5rem;">
                <strong>Note:</strong> This represents a subset of {report_data["company_data"]["name"]}'s reported customers. Full customer list is not publicly available.
            </p>
            """, unsafe_allow_html=True)
        else:
            st.info("Detailed customer information is not available for this company.")
    
    # Display use cases
    st.markdown('<h3 class="subsection-title">Primary Use Cases</h3>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Use editable_dict_array for use cases
        editable_dict_array(
            "Use Cases",
            "company_profile.use_cases",
            {"title": "Title", "description": "Description", "style": "Style (blue/green/purple)"},
            "usecase",
            {"title": "New Use Case", "description": "Description of the use case", "style": "blue"}
        )
    else:
        use_cases = report_data["company_profile"]["use_cases"]
        
        # Check if we have real use cases (not just placeholders)
        has_real_use_cases = any(
            uc.get("title") != "Primary Use Case" and 
            uc.get("description") != "The company's main product offering and application."
            for uc in use_cases
        )
        
        if has_real_use_cases:
            use_cases_col1, use_cases_col2, use_cases_col3 = st.columns(3)
            
            cols = [use_cases_col1, use_cases_col2, use_cases_col3]
            for i, use_case in enumerate(use_cases):
                with cols[i % 3]:
                    style = use_case["style"]
                    bg_color = "#EFF6FF" if style == "blue" else "#ECFDF5" if style == "green" else "#F5F3FF"
                    text_color = "#1E40AF" if style == "blue" else "#065F46" if style == "green" else "#5B21B6"
                    
                    st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 1rem; border-radius: 0.5rem;">
                        <h4 style="font-weight: 600; color: {text_color}; margin-bottom: 0.5rem;">{use_case["title"]}</h4>
                        <p style="font-size: 0.875rem;">
                            {use_case["description"]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Detailed use case information is not available for this company.")

def render_editable_products_tab(report_data):
    """Render the Products & Technology tab with editing capabilities"""
    st.markdown('<div class="tab-content" style="padding: 0.2rem; text-align: center;"><h2 class="section-title">Products & Technology</h2></div>', unsafe_allow_html=True)
    
    # Add count indicator for total products
    product_count = len(report_data["product_offerings"])
    st.markdown(f'<h3 class="subsection-title">Product Offerings <span style="font-size: 0.9rem; color: #6B7280; font-weight: normal;">({product_count} products identified)</span></h3>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Complex editing for product offerings
        for i, product in enumerate(st.session_state.document_data["product_offerings"]):
            with st.expander(f"Product {i+1}: {product.get('name', 'New Product')}", expanded=True):
                # Product name and description
                col1, col2 = st.columns([1, 3])
                with col1:
                    name_key = f"edit_product_{i}_name"
                    new_name = st.text_input(
                        "Product Name",
                        value=product.get("name", ""),
                        key=name_key,
                        on_change=lambda idx=i, key=name_key: update_dict_in_array(
                            "product_offerings", idx, "name", st.session_state[key]
                        )
                    )
                
                with col2:
                    desc_key = f"edit_product_{i}_description"
                    new_desc = st.text_area(
                        "Description",
                        value=product.get("description", ""),
                        key=desc_key,
                        on_change=lambda idx=i, key=desc_key: update_dict_in_array(
                            "product_offerings", idx, "description", st.session_state[key]
                        )
                    )
                
                # Features (as an array)
                st.markdown("#### Features")
                features = product.get("features", [])
                if not isinstance(features, list):
                    features = [str(features)]
                
                for j, feature in enumerate(features):
                    feat_cols = st.columns([5, 1])
                    with feat_cols[0]:
                        feat_key = f"edit_product_{i}_feature_{j}"
                        new_feat = st.text_input(
                            f"Feature {j+1}",
                            value=feature,
                            key=feat_key,
                            on_change=lambda prod_idx=i, feat_idx=j, key=feat_key: update_feature(
                                prod_idx, feat_idx, st.session_state[key]
                            )
                        )
                    
                    with feat_cols[1]:
                        if st.button("🗑️", key=f"delete_product_{i}_feature_{j}"):
                            # Remove this feature
                            product_features = st.session_state.document_data["product_offerings"][i].get("features", [])
                            if isinstance(product_features, list) and j < len(product_features):
                                product_features.pop(j)
                                st.rerun()
                
                # Add feature button
                if st.button("+ Add Feature", key=f"add_product_{i}_feature"):
                    # Add a new feature
                    if "features" not in st.session_state.document_data["product_offerings"][i]:
                        st.session_state.document_data["product_offerings"][i]["features"] = []
                    
                    if isinstance(st.session_state.document_data["product_offerings"][i]["features"], list):
                        st.session_state.document_data["product_offerings"][i]["features"].append("New feature")
                        st.rerun()
                
                # Additional product details
                col1, col2 = st.columns(2)
                with col1:
                    launch_key = f"edit_product_{i}_launch_date"
                    new_launch = st.text_input(
                        "Launch Date",
                        value=product.get("launch_date", ""),
                        key=launch_key,
                        on_change=lambda idx=i, key=launch_key: update_dict_in_array(
                            "product_offerings", idx, "launch_date", st.session_state[key]
                        )
                    )
                
                with col2:
                    user_base_key = f"edit_product_{i}_user_base"
                    new_user_base = st.text_input(
                        "User Base",
                        value=product.get("user_base", ""),
                        key=user_base_key,
                        on_change=lambda idx=i, key=user_base_key: update_dict_in_array(
                            "product_offerings", idx, "user_base", st.session_state[key]
                        )
                    )
                
                # Delete product button
                if st.button("🗑️ Delete Product", key=f"delete_product_{i}"):
                    remove_array_item("product_offerings", i)
                    st.rerun()
        
        # Add product button
        if st.button("+ Add New Product"):
            add_array_item("product_offerings", {
                "name": "New Product",
                "description": "Product description",
                "features": ["Feature 1"],
                "launch_date": "",
                "user_base": ""
            })
            st.rerun()
    else:
        # Display products in a grid layout
        products = report_data["product_offerings"]
        
        # Process products in pairs for the two-column layout
        for i in range(0, len(products), 2):
            # Create a row with two columns
            col1, col2 = st.columns(2)
            
            # First product in the pair - left column
            with col1:
                if i < len(products):
                    product = products[i]
                    with st.expander(f"{product['name']}", expanded=True, icon="🎁"):
                        with st.container(height=320, border=False):
                            # Header section with product name and launch date
                            st.markdown(f"""
                            <div>
                                <h4 style="font-size: 1.1rem; font-weight: 600; color: #1E40AF; margin-bottom: 0.2rem;">{product["name"]}</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Description
                            st.markdown(f"<p style='margin-top: 0.5rem;'>{product.get('description', '')}</p>", unsafe_allow_html=True)
                            
                            # Features section - use native Streamlit for list rendering
                            st.markdown("<h5 style='font-weight: 600; font-size: 0.875rem; color: #4B5563; margin-bottom: 0.5rem; margin-top: 0.75rem;'>Key Features:</h5>", unsafe_allow_html=True)
                            
                            if product["features"] and len(product["features"]) > 0:
                                # Filter out empty features
                                valid_features = [feature for feature in product["features"] if feature and str(feature).strip()]
                                
                                if valid_features:
                                    # Use Streamlit's native bullet points for better rendering
                                    for feature in valid_features:
                                        st.markdown(f"• {feature}")
                                else:
                                    st.markdown("<p style='font-style: italic; color: #6B7280;'>Feature information not available</p>", unsafe_allow_html=True)
                            else:
                                st.markdown("<p style='font-style: italic; color: #6B7280;'>Feature information not available</p>", unsafe_allow_html=True)
                            
                            # User base section
                            user_base = product.get("user_base", "")
                            if user_base:
                                st.markdown(f"""
                                <div style="margin-top: 0.75rem; font-size: 0.875rem; color: #4B5563;">
                                    <span style="font-weight: 600;">User Base:</span> {user_base}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Launch date section if available
                            launch_date = product.get("launch_date", "")
                            if launch_date:
                                st.markdown(f"""
                                <div style="font-size: 0.875rem; color: #4B5563;">
                                    <span style="font-weight: 600;">Launch Date:</span> {launch_date}
                                </div>
                                """, unsafe_allow_html=True)
            
            # Second product in the pair - right column
            with col2:
                if i + 1 < len(products):
                    product = products[i + 1]
                    with st.expander(f"{product['name']}", expanded=True, icon="🎁"):
                        with st.container(height=320, border=False):
                            # Header section with product name and launch date
                            st.markdown(f"""
                            <div>
                                <h4 style="font-size: 1.1rem; font-weight: 600; color: #1E40AF; margin-bottom: 0.2rem;">{product["name"]}</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Description
                            st.markdown(f"<p style='margin-top: 0.5rem;'>{product.get('description', '')}</p>", unsafe_allow_html=True)
                            
                            # Features section - use native Streamlit for list rendering
                            st.markdown("<h5 style='font-weight: 600; font-size: 0.875rem; color: #4B5563; margin-bottom: 0.5rem; margin-top: 0.75rem;'>Key Features:</h5>", unsafe_allow_html=True)
                            
                            if product["features"] and len(product["features"]) > 0:
                                # Filter out empty features
                                valid_features = [feature for feature in product["features"] if feature and str(feature).strip()]
                                
                                if valid_features:
                                    # Use Streamlit's native bullet points for better rendering
                                    for feature in valid_features:
                                        st.markdown(f"• {feature}")
                                else:
                                    st.markdown("<p style='font-style: italic; color: #6B7280;'>Feature information not available</p>", unsafe_allow_html=True)
                            else:
                                st.markdown("<p style='font-style: italic; color: #6B7280;'>Feature information not available</p>", unsafe_allow_html=True)
                            
                            # User base section
                            user_base = product.get("user_base", "")
                            if user_base:
                                st.markdown(f"""
                                <div style="margin-top: 0.75rem; font-size: 0.875rem; color: #4B5563;">
                                    <span style="font-weight: 600;">User Base:</span> {user_base}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Launch date section if available
                            launch_date = product.get("launch_date", "")
                            if launch_date:
                                st.markdown(f"""
                                <div style="font-size: 0.875rem; color: #4B5563;">
                                    <span style="font-weight: 600;">Launch Date:</span> {launch_date}
                                </div>
                                """, unsafe_allow_html=True)
    
    # Add Key Technologies section after the products grid
    st.markdown('<h3 class="subsection-title" style="margin-top: 2rem;">Technology Stack</h3>', unsafe_allow_html=True)

    if st.session_state.editing_mode:
        # Editable key technologies
        editable_dict_array(
            "Key Technologies",
            "key_technologies",
            {"category": "Category", "items": "Technologies"},
            "tech",
            {"category": "New Category", "items": "Associated technologies"}
        )
    else:
        has_real_tech = (report_data.get("key_technologies") and 
                        len(report_data["key_technologies"]) > 0 and 
                        report_data["key_technologies"][0]["category"] != "Primary Technology" and
                        report_data["key_technologies"][0]["items"] != "Core technology offering")

        if has_real_tech:
            # Create technology dataframe
            tech_df = pd.DataFrame(report_data["key_technologies"])
            st.dataframe(tech_df, hide_index=True, use_container_width=True)
        else:
            st.info("Detailed technology stack information is not available for this company.")
    
    # Technical Differentiation section
    st.markdown('<h3 class="subsection-title">Technical Differentiation</h3>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Editable technical differentiation
        editable_dict_array(
            "Technical Differentiation",
            "technical_differentiation",
            {"title": "Title", "description": "Description", "style": "Style (blue/green/purple/yellow)"},
            "diff",
            {"title": "New Differentiation", "description": "Description of technical advantage", "style": "blue"}
        )
    else:
        diff_col1, diff_col2 = st.columns(2)
        
        # Split technical_differentiation into two columns
        differentiations = report_data["technical_differentiation"]
        half = len(differentiations) // 2
        
        for i, diff in enumerate(differentiations[:half]):
            with diff_col1:
                st.markdown(f"""
                <div class="{diff['style']}-card" style="padding: 1rem; margin-bottom: 1rem;">
                    <h4 style="font-weight: 600; margin-bottom: 0.5rem;">{diff["title"]}</h4>
                    <p style="font-size: 0.875rem;">
                        {diff["description"]}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        for i, diff in enumerate(differentiations[half:]):
            with diff_col2:
                st.markdown(f"""
                <div class="{diff['style']}-card" style="padding: 1rem; margin-bottom: 1rem;">
                    <h4 style="font-weight: 600; margin-bottom: 0.5rem;">{diff["title"]}</h4>
                    <p style="font-size: 0.875rem;">
                        {diff["description"]}
                    </p>
                </div>
                """, unsafe_allow_html=True)

# Helper function for updating product features
def update_feature(product_idx, feature_idx, new_value):
    """Update a specific feature in a product"""
    if (product_idx < len(st.session_state.document_data["product_offerings"]) and
        "features" in st.session_state.document_data["product_offerings"][product_idx] and
        isinstance(st.session_state.document_data["product_offerings"][product_idx]["features"], list) and
        feature_idx < len(st.session_state.document_data["product_offerings"][product_idx]["features"])):
        st.session_state.document_data["product_offerings"][product_idx]["features"][feature_idx] = new_value

def render_editable_leadership_tab(report_data):
    """Render the Leadership & Funding tab with editing capabilities"""
    st.markdown('<div class="tab-content" style="padding: 0.2rem; text-align: center;"><h2 class="section-title">Leadership & Funding</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-title">Leadership Team</h3>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Editable leadership team
        editable_dict_array(
            "Leadership Team",
            "leadership_team",
            {"name": "Name", "position": "Position", "background": "Background"},
            "leader",
            {"name": "New Leader", "position": "Position", "background": "Professional background"}
        )
    else:
        leadership_col1, leadership_col2 = st.columns(2)
        
        for i, leader in enumerate(report_data["leadership_team"]):
            col = leadership_col1 if i % 2 == 0 else leadership_col2
            with col:
                # Add LinkedIn URL if available - using simplified markdown approach
                linkedin_link = ""
                if leader.get("linkedin_url") and leader["linkedin_url"].strip():
                    linkedin_link = leader["linkedin_url"]

                # Use Streamlit container instead of HTML
                with st.container(border=True):
                    # Header with name and role tags
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"#### {leader.get('name', 'Unnamed')}")
                    with col2:
                        if leader.get("is_founder"):
                            st.markdown("<span style='background: #EFF6FF; color: #1E40AF; font-size: 0.75rem; padding: 0.1rem 0.5rem; border-radius: 9999px;'>Founder</span>", unsafe_allow_html=True)
                        if leader.get("is_executive"):
                            st.markdown("<span style='background: #ECFDF5; color: #065F46; font-size: 0.75rem; padding: 0.1rem 0.5rem; border-radius: 9999px;'>Executive</span>", unsafe_allow_html=True)
                    
                    # Position
                    st.markdown(f"<p style='color: #3B82F6;'>{leader.get('position', leader.get('role', 'Unknown position'))}</p>", unsafe_allow_html=True)
                    
                    # Background
                    st.markdown(f"**Background:** {leader.get('background', 'Unknown background') or 'Not available'}")
                    
                    # LinkedIn link using Streamlit elements
                    if linkedin_link:
                        st.markdown(f"[LinkedIn Profile]({linkedin_link})")
    
    st.markdown("""
    <div style="background-color: #FFFBEB; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
        <p style="font-size: 0.875rem; color: #92400E;">
            <strong>Note:</strong> Limited information available on the complete executive team composition. The full organizational structure and board of directors information is not publicly available.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-title">Funding History</h3>', unsafe_allow_html=True)

    if st.session_state.editing_mode:
        # Editable funding data
        editable_dict_array(
            "Funding Rounds",
            "funding_data",
            {"round": "Round", "date": "Date", "amount": "Amount", "investor": "Lead Investor", "notes": "Notes", "source_url": "Source URL"},
            "funding",
            {"round": "New Round", "date": "", "amount": "", "investor": "", "notes": "", "source_url": ""}
        )
        
        # Add funding analysis editor
        st.markdown('<h4 style="font-weight: 600; margin-top: 1.5rem;">Funding Analysis</h4>', unsafe_allow_html=True)
        funding_analysis = editable_text_area(
            "Analysis", 
            "funding_analysis", 
            "funding", 
            height=200,
            placeholder="Analysis of the company's funding history and implications"
        )
        st.markdown(funding_analysis)
    else:
        has_real_funding = (report_data.get("funding_data") and len(report_data["funding_data"]) > 0)

        if has_real_funding:
            # Create a more structured funding DataFrame with source URLs
            funding_data = []
            for fund in report_data["funding_data"]:
                # Format investors list - ensure we always have a string
                investors_str = ""
                if fund.get("investors"):
                    if isinstance(fund.get("investors"), list):
                        # Join list of investors into a comma-separated string
                        investors_str = ", ".join([str(inv) for inv in fund["investors"] if inv])
                    elif isinstance(fund.get("investors"), str):
                        # Already a string
                        investors_str = fund["investors"]
                else:
                    # Convert any other type to string
                    investors_str = str(fund.get("investors", ""))
            
                # Add lead investor indication
                lead_investor = str(fund.get("lead_investor", "")) if fund.get("lead_investor") else ""
                if lead_investor:
                    if investors_str:
                        if lead_investor not in investors_str:
                            investors_str = f"{lead_investor} (lead), {investors_str}"
                        else:
                            # Highlight the lead investor within the investors list
                            investors_str = investors_str.replace(
                                lead_investor, 
                                f"{lead_investor} (lead)"
                            )
                    else:
                        investors_str = f"{lead_investor} (lead)"
                
                # Prepare source URL
                source_url = ""
                if fund.get("source_url"):
                    source_url = fund["source_url"]
                elif fund.get("url"):  # Try alternate field names
                    source_url = fund["url"]
                elif fund.get("funding_news_url"):
                    source_url = fund["funding_news_url"]
                    
                # Create a formatted row - ensure all values are strings
                row_data = {
                    "Round": str(fund.get("round", "")),
                    "Date": str(fund.get("date", "")),
                    "Amount": str(fund.get("amount", "")),
                    "Investors": investors_str,  # Now guaranteed to be a string
                    "Source": source_url
                }
                
                funding_data.append(row_data)
            
            # Create and display funding DataFrame
            funding_df = pd.DataFrame(funding_data)

            # Convert source URLs to clickable links if they exist
            if "Source" in funding_df.columns and len(funding_df) > 0:
                # Create a copy to avoid SettingWithCopyWarning
                funding_df = funding_df.copy()
                
                # Convert URLs to clickable links - improve to handle invalid URLs
                def make_clickable(url):
                    if not url or url.strip() == "":
                        return ""
                    try:
                        # Add http:// prefix if missing
                        if not url.startswith(("http://", "https://")):
                            url = "https://" + url
                        return f'<a href="{url}" target="_blank">Source</a>'
                    except:
                        return url
                
                funding_df["Source"] = funding_df["Source"].apply(make_clickable)
                
                # Apply the formatting to the dataframe
                st.dataframe(
                    funding_df, 
                    hide_index=True, 
                    use_container_width=True,
                    column_config={
                        "Source": st.column_config.Column("Source", width="small"),
                        "Round": st.column_config.Column("Round", width="medium"),
                        "Date": st.column_config.Column("Date", width="medium"),
                        "Amount": st.column_config.Column("Amount", width="medium"),
                        "Investors": st.column_config.Column("Investors", width="large")
                    }
                )
            else:
                st.dataframe(funding_df, hide_index=True, use_container_width=True)

        else:
            st.info("Detailed funding history information is not available for this company.")
        
        # Still show funding analysis if available
        if report_data.get("funding_analysis") and report_data["funding_analysis"] not in ["", "Funding information for the company."]:
            st.markdown('<h4 style="font-weight: 600; margin-top: 1.5rem;">Funding Analysis</h4>', unsafe_allow_html=True)
            st.markdown(report_data["funding_analysis"])

def render_editable_market_tab(report_data):
    """Render the Market & Competition tab with editing capabilities"""
    st.markdown('<div class="tab-content" style="padding: 0.2rem; text-align: center;"><h2 class="section-title">Market & Competition</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-title">Competitive Landscape</h3>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Editable competitors
        editable_dict_array(
            "Competitors",
            "competitors",
            {"name": "Name", "description": "Description", "founded": "Founded", "market_share": "Market Share"},
            "competitor",
            {"name": "New Competitor", "description": "Competitor description", "founded": "", "market_share": ""}
        )
        
        # Add strengths and weaknesses separately
        st.markdown("### Competitor Strengths & Weaknesses")
        
        for i, competitor in enumerate(st.session_state.document_data["competitors"]):
            with st.expander(f"Edit {competitor.get('name', 'Competitor')} Strengths & Weaknesses"):
                # Strengths
                st.markdown(f"#### Strengths for {competitor.get('name', 'Competitor')}")
                strengths = competitor.get("strengths", [])
                if not isinstance(strengths, list):
                    strengths = []
                    competitor["strengths"] = strengths
                
                for j, strength in enumerate(strengths):
                    str_cols = st.columns([5, 1])
                    with str_cols[0]:
                        str_key = f"edit_competitor_{i}_strength_{j}"
                        new_str = st.text_input(
                            f"Strength {j+1}",
                            value=strength,
                            key=str_key,
                            on_change=lambda comp_idx=i, str_idx=j, key=str_key: update_competitor_attribute(
                                comp_idx, "strengths", str_idx, st.session_state[key]
                            )
                        )
                    
                    with str_cols[1]:
                        if st.button("🗑️", key=f"delete_competitor_{i}_strength_{j}"):
                            # Remove this strength
                            if isinstance(competitor["strengths"], list) and j < len(competitor["strengths"]):
                                competitor["strengths"].pop(j)
                                st.rerun()
                
                # Add strength button
                if st.button("+ Add Strength", key=f"add_competitor_{i}_strength"):
                    if "strengths" not in competitor:
                        competitor["strengths"] = []
                    competitor["strengths"].append("New strength")
                    st.rerun()
                
                # Weaknesses
                st.markdown(f"#### Weaknesses for {competitor.get('name', 'Competitor')}")
                weaknesses = competitor.get("weaknesses", [])
                if not isinstance(weaknesses, list):
                    weaknesses = []
                    competitor["weaknesses"] = weaknesses
                
                for j, weakness in enumerate(weaknesses):
                    weak_cols = st.columns([5, 1])
                    with weak_cols[0]:
                        weak_key = f"edit_competitor_{i}_weakness_{j}"
                        new_weak = st.text_input(
                            f"Weakness {j+1}",
                            value=weakness,
                            key=weak_key,
                            on_change=lambda comp_idx=i, weak_idx=j, key=weak_key: update_competitor_attribute(
                                comp_idx, "weaknesses", weak_idx, st.session_state[key]
                            )
                        )
                    
                    with weak_cols[1]:
                        if st.button("🗑️", key=f"delete_competitor_{i}_weakness_{j}"):
                            # Remove this weakness
                            if isinstance(competitor["weaknesses"], list) and j < len(competitor["weaknesses"]):
                                competitor["weaknesses"].pop(j)
                                st.rerun()
                
                # Add weakness button
                if st.button("+ Add Weakness", key=f"add_competitor_{i}_weakness"):
                    if "weaknesses" not in competitor:
                        competitor["weaknesses"] = []
                    competitor["weaknesses"].append("New weakness")
                    st.rerun()
    else:
        # Check if we have real competitor data
        has_real_competitors = len(report_data["competitors"]) > 0 and report_data["competitors"][0]["name"] != "CompetitorA"
        
        if has_real_competitors:
            for competitor in report_data["competitors"]:
                # Create a more structured competitor display
                founded = f"Founded: {competitor['founded']}" if competitor.get('founded', "") else ""
                
                # Use Streamlit container for better rendering
                with st.container(border=True):
                    # Header with name and founded date
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"#### {competitor['name']}")
                    with col2:
                        if founded:
                            st.markdown(f"<div style='text-align: right; color: #6B7280; font-size: 0.875rem;'>{founded}</div>", unsafe_allow_html=True)
                    
                    # Description
                    st.markdown(competitor["description"])
                    
                    # Strengths section - using Streamlit's native bullet points
                    if competitor.get("strengths") and len(competitor["strengths"]) > 0:
                        valid_strengths = [str(s) for s in competitor["strengths"] if s and str(s).strip()]
                        if valid_strengths:
                            st.markdown("**Strengths:**")
                            for strength in valid_strengths:
                                st.markdown(f"• {strength}")
                    
                    # Weaknesses section - using Streamlit's native bullet points
                    if competitor.get("weaknesses") and len(competitor["weaknesses"]) > 0:
                        valid_weaknesses = [str(w) for w in competitor["weaknesses"] if w and str(w).strip()]
                        if valid_weaknesses:
                            st.markdown("**Weaknesses:**")
                            for weakness in valid_weaknesses:
                                st.markdown(f"• {weakness}")
                    
                    # Market share if available
                    if competitor.get("market_share") and competitor["market_share"].strip():
                        st.markdown(f"**Market Share:** {competitor['market_share']}")
        else:
            st.info("Detailed competitor information is not available for this company.")
    
    advantages_col, trends_col = st.columns(2)
    
    with advantages_col:
        st.markdown('<h3 class="subsection-title">Company Advantages</h3>', unsafe_allow_html=True)
        
        if st.session_state.editing_mode:
            # Editable market advantages
            editable_dict_array(
                "Market Advantages",
                "market_advantages",
                {"title": "Title", "description": "Description"},
                "advantage",
                {"title": "New Advantage", "description": "Description of market advantage"}
            )
        else:
            # Check if we have real market advantages data
            has_real_advantages = len(report_data["market_advantages"]) > 0 and report_data["market_advantages"][0]["title"] != "Market Position"
            
            if has_real_advantages:
                for advantage in report_data["market_advantages"]:
                    st.markdown(f"""
                    <div style="background-color: #EFF6FF; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.75rem;">
                        <h4 style="font-weight: 600; color: #1E40AF; margin-bottom: 0.25rem;">{advantage["title"]}</h4>
                        <p style="font-size: 0.875rem;">{advantage["description"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Detailed market advantage information is not available for this company.")
    
    with trends_col:
        st.markdown('<h3 class="subsection-title">Market Trends</h3>', unsafe_allow_html=True)
        
        if st.session_state.editing_mode:
            # Editable market trends
            editable_dict_array(
                "Market Trends",
                "market_trends",
                {"title": "Title", "description": "Description", "impact_level": "Impact (High/Medium/Low)", "timeline": "Timeline"},
                "trend",
                {"title": "New Trend", "description": "Description of market trend", "impact_level": "Medium", "timeline": "Current"}
            )
        else:
            # Check if we have real market trends data
            has_real_trends = len(report_data["market_trends"]) > 0 and report_data["market_trends"][0]["title"] != "Market Trend 1"
            
            if has_real_trends:
                for trend in report_data["market_trends"]:
                    # Add impact level if available
                    # Create a Streamlit container for each trend
                    with st.container(border=False):
                        # Create a cleaner card layout with Streamlit elements instead of HTML
                        with st.container(border=True, height=None):
                            # Title row with impact badge
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"#### {trend['title']}")
                            with col2:
                                if trend.get("impact_level") and trend["impact_level"]:
                                    impact_level = trend["impact_level"]
                                    # Use colored text for impact level
                                    if impact_level == "High":
                                        st.markdown(f"<div style='text-align: right;'><span style='background-color: #ECFDF5; color: #065F46; font-size: 0.8rem; padding: 0.1rem 0.5rem; border-radius: 9999px;'>{impact_level} Impact</span></div>", unsafe_allow_html=True)
                                    elif impact_level == "Medium":
                                        st.markdown(f"<div style='text-align: right;'><span style='background-color: #FFFBEB; color: #92400E; font-size: 0.8rem; padding: 0.1rem 0.5rem; border-radius: 9999px;'>{impact_level} Impact</span></div>", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"<div style='text-align: right;'><span style='background-color: #EFF6FF; color: #1E40AF; font-size: 0.8rem; padding: 0.1rem 0.5rem; border-radius: 9999px;'>{impact_level} Impact</span></div>", unsafe_allow_html=True)
                            
                            # Description
                            st.markdown(trend["description"] if trend.get("description") else "")
                            
                            # Timeline if available
                            if trend.get("timeline") and trend["timeline"]:
                                st.markdown(f"<div style='font-size: 0.8rem; color: #6B7280;'><strong>Timeline:</strong> {trend['timeline']}</div>", unsafe_allow_html=True)
            else:
                st.info("Detailed market trend information is not available for this company.")

# Helper function for updating competitor attributes
def update_competitor_attribute(competitor_idx, attr_name, item_idx, new_value):
    """Update a specific attribute in a competitor's array"""
    if (competitor_idx < len(st.session_state.document_data["competitors"]) and
        attr_name in st.session_state.document_data["competitors"][competitor_idx] and
        isinstance(st.session_state.document_data["competitors"][competitor_idx][attr_name], list) and
        item_idx < len(st.session_state.document_data["competitors"][competitor_idx][attr_name])):
        st.session_state.document_data["competitors"][competitor_idx][attr_name][item_idx] = new_value

def standardize_date_format(date_str):
    """
    Standardize various date formats to a consistent display format.
    
    Args:
        date_str (str): Input date string in various formats
        
    Returns:
        str: Standardized date string in "Month DD, YYYY" format or original if unparseable
    """
    if not date_str or date_str.lower() == "recent":
        return "Recent"
    
    # Try different date formats
    date_formats = [
        "%Y%m%d",            # 20250304
        "%B %d, %Y",         # March 18, 2025
        "%Y-%m-%d",          # 2025-03-18
        "%b %d, %Y",         # Mar 18, 2025
        "%d %B %Y",          # 18 March 2025
        "%m/%d/%Y",          # 03/18/2025
        "%Y"                 # 2025 (year only)
    ]
    
    for fmt in date_formats:
        try:
            # Parse the date using the current format
            date_obj = datetime.strptime(date_str, fmt)
            # Return in standard format
            return date_obj.strftime("%B %d, %Y")
        except ValueError:
            continue
    
    # If no format matches, return original
    return date_str

def render_editable_news_tab(report_data):
    """Render the News & Developments tab with editing capabilities"""
    st.markdown('<div class="tab-content" style="padding: 0.2rem; text-align: center;"><h2 class="section-title">News & Developments</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-title">Recent News</h3>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Editable news data
        editable_dict_array(
            "News Items",
            "recent_news",
            {"date": "Date", "title": "Title", "summary": "Summary", "url": "URL"},
            "news",
            {"date": "Recent", "title": "New Headline", "summary": "News summary", "url": ""}
        )
        
        # Strategic direction
        st.markdown('<h3 class="subsection-title">Strategic Direction</h3>', unsafe_allow_html=True)
        editable_dict_array(
            "Strategic Initiatives",
            "strategic_direction",
            {"title": "Title", "description": "Description"},
            "strategy",
            {"title": "New Strategy", "description": "Description of strategic direction"}
        )
    else:
        # Create a list to store processed news items
        processed_news = []
        
        # Check if we have actual news items
        if len(report_data["recent_news"]) == 0 or (len(report_data["recent_news"]) == 1 and report_data["recent_news"][0]["title"] == "Company News"):
            st.info("No recent news articles found for this company.")
        else:
            for news in report_data["recent_news"]:
                # Standardize date format
                std_date = standardize_date_format(news.get("date", "Recent"))
                
                # Use proper summary or placeholder
                summary = news.get("summary", "")
                if not summary or summary == "Recent company update":
                    # Try to extract summary from title if it looks like a complete sentence
                    if news.get("title", "") and len(news.get("title", "")) > 30:
                        summary = news.get("title", "")
                    else:
                        summary = "No detailed summary available."
                
                # Extract source if available
                source = ""
                if news.get("source"):
                    source = news["source"]
                elif news.get("url"):
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(news["url"]).netloc
                        source = domain
                    except:
                        source = news["url"]
                
                # Add processed news item
                processed_news.append({
                    "title": news.get("title", "Untitled"),
                    "date": std_date,
                    "std_date_obj": datetime.strptime(std_date, "%B %d, %Y") if std_date != "Recent" else datetime.now(),
                    "summary": summary,
                    "is_recent": std_date == "Recent",
                    "url": news.get("url", ""),
                    "source": source
                })
            
            # Sort news by date (recent items at the top, then by date descending)
            sorted_news = sorted(
                processed_news, 
                key=lambda x: (0 if x["is_recent"] else 1, x["std_date_obj"] if not x["is_recent"] else datetime.now()),
                reverse=True
            )
            
            # Display sorted news items
            for i, news in enumerate(sorted_news):
                # Create a container for each news item
                with st.container(border=True):
                    # Display title and date
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"#### {news['title']}")
                    with col2:
                        st.markdown(f"<div style='text-align: right; color: #6B7280;'>{news['date']}</div>", unsafe_allow_html=True)
                    
                    # Display summary
                    st.markdown(news["summary"])
                    
                    # Display URL using Streamlit's built-in components if available
                    if news["url"]:
                        url_display = news["source"] if news["source"] else news["url"]
                        st.markdown(f"**Source:** [{url_display}]({news['url']})")
        
        # Strategic direction
        st.markdown('<h3 class="subsection-title" style="margin-top: 2rem;">Strategic Direction</h3>', unsafe_allow_html=True)
        
        # Check if we have real strategic direction data
        has_real_strategy = len(report_data["strategic_direction"]) > 0 and report_data["strategic_direction"][0]["title"] != "Business Strategy"
        
        if has_real_strategy:
            for strategy in report_data["strategic_direction"]:
                st.markdown(f"""
                <div style="background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <h4 style="font-weight: 600; color: #1F2937; margin-bottom: 0.5rem;">{strategy["title"]}</h4>
                    <p style="font-size: 0.875rem;">{strategy["description"]}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Detailed strategic direction information is not available for this company.")

def render_editable_github_tab(report_data):
    """Render the GitHub Analytics tab with editing capabilities"""
    st.markdown('<div class="tab-content" style="padding: 0.2rem; text-align: center;"><h2 class="section-title">GitHub Analytics</h2></div>', unsafe_allow_html=True)
    
    # GitHub icon SVG path - defined separately to avoid backslash issues in f-strings
    github_icon = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" style="fill: #1F2937;"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.652.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>'
    
    # Check if GitHub stats are available
    if "github_stats" not in report_data or not report_data["github_stats"]:
        st.warning("GitHub statistics are not directly available for this company.")
        
        if st.session_state.editing_mode:
            # Add GitHub repository manually
            st.markdown("### Add GitHub Repository Manually")
            
            repo_col1, repo_col2 = st.columns(2)
            
            with repo_col1:
                repo_name = st.text_input("Repository Name (e.g., 'organization/repo')", key="add_github_repo_name")
            
            with repo_col2:
                if st.button("Fetch Repository Data"):
                    if repo_name:
                        with st.spinner(f"Fetching data for {repo_name}..."):
                            try:
                                # Try to get the github token from secrets
                                try:
                                    github_token = st.secrets["GITHUB_TOKEN"]
                                except:
                                    github_token = None
                                
                                # Fetch GitHub stats
                                github_stats = get_github_stats(repo_name, github_token)
                                
                                # Update document data
                                if "github_stats" not in st.session_state.document_data:
                                    st.session_state.document_data["github_stats"] = {}
                                
                                # Update the stats
                                st.session_state.document_data["github_stats"] = github_stats
                                
                                # Show success message
                                st.success(f"Successfully fetched GitHub data for {repo_name}.")
                                
                                # Rerun to show the updated data
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error fetching GitHub data: {str(e)}")
        
        # Search for related repositories
        company_name = report_data["company_data"]["name"]
        st.info(f"Searching for GitHub repositories related to {company_name}...")
        
        try:
            # Try to get the github token from secrets
            try:
                github_token = st.secrets["GITHUB_TOKEN"]
            except:
                github_token = None
            
            # Use the GitHub API to search for repositories by company name
            g = Github(github_token)
            related_repos = g.search_repositories(query=f"{company_name} in:name,description", sort="stars", order="desc")
            
            if related_repos.totalCount > 0:
                st.success(f"Found {min(related_repos.totalCount, 5)} related repositories")
                
                # Display the top 5 (or fewer) related repositories
                for i, repo in enumerate(related_repos[:5]):
                    # Create a smaller icon for the list
                    small_icon = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" style="fill: #1F2937;"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.652.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>'
                    
                    # Topic tags HTML - generating outside the f-string to avoid backslash issues
                    topics_html = ""
                    for topic in repo.get_topics()[:5]:
                        topics_html += f'<span style="background: #F3F4F6; border: 1px solid #E5E7EB; font-size: 0.75rem; padding: 0.1rem 0.5rem; border-radius: 9999px;">{topic}</span> '
                    
                    st.markdown(f"""
                    <div style='background: #F3F4F6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                        <div style='display: flex; align-items: center; gap: 0.5rem;'>
                            {small_icon}
                            <a href='{repo.html_url}' style='text-decoration: none; color: #1F2937; font-weight: 600;'>
                                {repo.full_name}
                            </a>
                            <span style='background: #EFF6FF; color: #1E40AF; font-size: 0.75rem; padding: 0.1rem 0.5rem; border-radius: 9999px;'>
                                ⭐ {repo.stargazers_count:,}
                            </span>
                        </div>
                        <p style='margin-top: 0.5rem; font-size: 0.875rem;'>{repo.description or "No description available"}</p>
                        <div style='margin-top: 0.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap;'>
                            {topics_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add button to use this repository in edit mode
                    if st.session_state.editing_mode:
                        use_repo_key = f"use_repo_{i}"
                        if st.button(f"Use This Repository", key=use_repo_key):
                            with st.spinner(f"Setting up repository {repo.full_name}..."):
                                # Update document data with this repository
                                if "github_stats" not in st.session_state.document_data:
                                    st.session_state.document_data["github_stats"] = {}
                                
                                # Add repo data
                                st.session_state.document_data["github_stats"] = {
                                    "name": repo.name,
                                    "stars": repo.stargazers_count,
                                    "forks": repo.forks_count,
                                    "open_issues": repo.open_issues_count,
                                    "subscribers": repo.subscribers_count,
                                    "html_url": repo.html_url,
                                    "full_name": repo.full_name
                                }
                                
                                # Show success message
                                st.success(f"Added {repo.full_name} to the report.")
                                
                                # Rerun to show the updated data
                                st.rerun()
                
                # Select the most relevant repository for detailed analysis
                primary_repo = related_repos[0]
                st.markdown(f"### Detailed Analysis of {primary_repo.name}")
                
                # Create column layout for statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("GitHub Stars", f"{primary_repo.stargazers_count:,}")
                with col2:
                    st.metric("Forks", f"{primary_repo.forks_count:,}")
                with col3:
                    st.metric("Open Issues", f"{primary_repo.open_issues_count:,}")
                
                # Display repository activity if token allows
                try:
                    # Get last 10 commits
                    st.markdown("### Recent Activity")
                    
                    # Display last 5 commits
                    st.markdown("#### Recent Commits")
                    commits = primary_repo.get_commits()[:5]
                    for commit in commits:
                        commit_date = commit.commit.author.date.strftime("%b %d, %Y")
                        commit_message = commit.commit.message.split('\n')[0][:60]
                        st.markdown(f"""
                        <div style='background: #F9FAFB; padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 0.25rem;'>
                            <div style='font-size: 0.875rem; display: flex; justify-content: space-between;'>
                                <span>{commit_message}</span>
                                <span style='color: #6B7280;'>{commit_date}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except:
                    st.info("Detailed activity data requires additional permissions")
            else:
                st.warning(f"No repositories found for {company_name}. Consider manual research.")
        except Exception as e:
            st.error(f"Error searching for repositories: {str(e)}")
            st.info("GitHub search requires API authentication. Please check your GitHub token.")
        
        return
    
    # If we have GitHub stats, proceed with standard display
    github_stats = report_data["github_stats"]
    
    # Enhanced repository discovery - try multiple search strategies
    github_url = github_stats.get("html_url", "")
    company_name = report_data["company_data"]["name"]

    if not github_url or github_stats.get("needs_search", False):
        # Attempt to find the repository via search
        try:
            # Try to get the github token from secrets
            try:
                github_token = st.secrets["GITHUB_TOKEN"]
            except:
                github_token = None
            
            g = Github(github_token)
            
            # Try several search strategies in order of specificity
            search_queries = [
                f"org:{company_name.lower().replace(' ', '')} stars:>10",  # Try by organization
                f"{company_name} {github_stats.get('name_component', github_stats.get('name', ''))} in:name",  # By company and repo name
                f"{company_name} in:name",  # Just by company name
                f"{github_stats.get('name_component', github_stats.get('name', ''))} in:name stars:>50"  # Just by repo name with filters
            ]
            
            # Try each search query until we find results
            for query in search_queries:
                search_results = g.search_repositories(query=query, sort="stars", order="desc")
                if search_results.totalCount > 0:
                    top_result = search_results[0]
                    github_stats["html_url"] = top_result.html_url
                    github_stats["full_name"] = top_result.full_name
                    github_url = top_result.html_url
                    
                    # Update document data in edit mode
                    if st.session_state.editing_mode:
                        st.session_state.document_data["github_stats"]["html_url"] = top_result.html_url
                        st.session_state.document_data["github_stats"]["full_name"] = top_result.full_name
                    
                    break
            
            # If we still don't have a valid URL, use the GitHub search page
            if not github_url:
                # Create a better search experience by combining company and repository names
                search_term = f"{company_name} {github_stats.get('name_component', github_stats.get('name', ''))}"
                github_url = f"https://github.com/search?q={search_term.replace(' ', '+')}&type=repositories"
                
                # Alert the user that we're showing search results
                st.info(f"No specific repository found. Showing GitHub search results for '{search_term}'")
        except Exception as e:
            # If search fails, provide a generic GitHub page with helpful message
            st.warning(f"Could not connect to GitHub API. Consider adding a GitHub token in secrets.")
            github_url = "https://github.com"
    
    # Editable GitHub fields in edit mode
    if st.session_state.editing_mode:
        st.markdown("### Edit GitHub Repository Information")
        
        repo_col1, repo_col2 = st.columns(2)
        
        with repo_col1:
            repo_name = editable_text_input("Repository Name", "github_stats.full_name", "github")
            stars = st.number_input(
                "Stars",
                value=int(github_stats.get("stars", 0)),
                key="edit_github_stars",
                on_change=lambda: update_simple_field("github_stats.stars", st.session_state["edit_github_stars"])
            )
            forks = st.number_input(
                "Forks",
                value=int(github_stats.get("forks", 0)),
                key="edit_github_forks",
                on_change=lambda: update_simple_field("github_stats.forks", st.session_state["edit_github_forks"])
            )
        
        with repo_col2:
            repo_url = editable_text_input("Repository URL", "github_stats.html_url", "github")
            issues = st.number_input(
                "Open Issues",
                value=int(github_stats.get("open_issues", 0)),
                key="edit_github_issues",
                on_change=lambda: update_simple_field("github_stats.open_issues", st.session_state["edit_github_issues"])
            )
            subscribers = st.number_input(
                "Subscribers",
                value=int(github_stats.get("subscribers", 0)),
                key="edit_github_subscribers",
                on_change=lambda: update_simple_field("github_stats.subscribers", st.session_state["edit_github_subscribers"])
            )
        
        # GitHub analysis text area
        st.markdown("### GitHub Analysis")
        analysis = editable_text_area(
            "Analysis",
            "github_analysis.importance",
            "github",
            height=150,
            placeholder="Analysis of the GitHub metrics importance for this company"
        )
    
    # Display repository link
    repo_name = github_stats.get('full_name', github_stats.get('name', 'GitHub Repository'))
    
    st.markdown(f"""
    <div style='background: #F3F4F6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
        <div style='display: flex; align-items: center; gap: 0.5rem;'>
            {github_icon}
            <a href='{github_url}' style='text-decoration: none; color: #1F2937; font-weight: 600;'>
                {repo_name}
            </a>
        </div>
        <div style='margin-top: 0.5rem;'>
            <a href='{github_url}' style='text-decoration: none;'>
                <button style='background: #1F2937; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer;'>
                    View on GitHub
                </button>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This tab provides real-time GitHub metrics for the repository, showing current statistics and historical trends.
    GitHub data serves as a key indicator of community engagement and project growth.
    """)
    
    # Display GitHub metrics UI
    display_github_metrics_ui(github_stats, compact=False)
    
    # Try to get additional repository data if possible
    try:
        # Try to get the github token from secrets
        try:
            github_token = st.secrets["GITHUB_TOKEN"]
        except:
            github_token = None
        
        g = Github(github_token)
        if "full_name" in github_stats:
            repo = g.get_repo(github_stats["full_name"])
            
            # Display repository topics
            topics = repo.get_topics()
            if topics:
                st.markdown("### Repository Topics")
                topics_html = ""
                for topic in topics:
                    topics_html += f'<span style="background: #EFF6FF; color: #1E40AF; font-size: 0.875rem; padding: 0.25rem 0.75rem; border-radius: 9999px; margin-right: 0.5rem;">{topic}</span>'
                st.markdown(f"<div style='margin: 0.5rem 0;'>{topics_html}</div>", unsafe_allow_html=True)
            
            # Display recent commits
            st.markdown("### Recent Activity")
            try:
                commits = repo.get_commits()[:5]
                for commit in commits:
                    commit_date = commit.commit.author.date.strftime("%b %d, %Y")
                    commit_message = commit.commit.message.split('\n')[0][:60]
                    st.markdown(f"""
                    <div style='background: #F9FAFB; padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 0.25rem;'>
                        <div style='font-size: 0.875rem; display: flex; justify-content: space-between;'>
                            <span>{commit_message}</span>
                            <span style='color: #6B7280;'>{commit_date}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except:
                st.info("Commit history requires additional permissions")
    except:
        pass
    
    # Add context about GitHub metrics
    st.markdown(f"""
    <div style="background-color: #EFF6FF; padding: 1rem; border-radius: 0.5rem; margin-top: 1.5rem;">
        <h4 style="font-weight: 600; color: #1E40AF; margin-bottom: 0.5rem;">Why GitHub Metrics Matter</h4>
        <p style="font-size: 0.875rem;">
            {report_data["github_analysis"]["importance"]}
        </p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------
# Main Editable Dashboard Function
# --------------------------------

def parselyfi_editable_company_report_page(report_data):
    """
    Main function to generate an editable company report page.
    
    Args:
        report_data (dict or str): The company report data or path to JSON file
    """
    # Check if report_data is a string (file path) and load it
    if isinstance(report_data, str):
        try:
            with open(report_data, 'r') as f:
                report_data = json.load(f)
            # Apply compatibility fixes for loaded data
            report_data = ensure_report_compatibility(report_data)
        except Exception as e:
            st.error(f"Error loading report file: {e}")
            st.stop()
    
    # Initialize document data in session state
    data = initialize_document_data(report_data)
    
    # IMPORTANT: All render functions should use session state data, NOT the original report_data
    # Render the editable report header
    render_editable_report_header(
        data, 
        data["report_metadata"]
    )
    
    # Create tabs
    tabs = create_navigation_tabs(data)
    
    # Unpack tabs
    tab_summary = tabs[0]
    tab_company = tabs[1]
    tab_products = tabs[2]
    tab_leadership = tabs[3]
    tab_market = tabs[4]
    tab_news = tabs[5]
    
    # GitHub tab is conditional
    tab_github = None
    if has_valid_github_info(data):
        tab_github = tabs[6]
    
    # Render each tab with editable content, passing session state data
    with tab_summary:
        render_editable_executive_summary_tab(data)
    
    with tab_company:
        render_editable_company_profile_tab(data)
    
    with tab_products:
        render_editable_products_tab(data)
    
    with tab_leadership:
        render_editable_leadership_tab(data)
    
    with tab_market:
        render_editable_market_tab(data)
    
    with tab_news:
        render_editable_news_tab(data)
    
    # GitHub tab is conditional
    if tab_github:
        with tab_github:
            render_editable_github_tab(data)
    
    # Render the report footer
    render_report_footer(data)

# --------------------------------
# List Available Reports Function
# --------------------------------

def list_available_reports():
    """
    List all available company reports in the output directory.
    
    Returns:
        list: List of (filename, company_name, timestamp) tuples for available reports
    """
    output_dir = "company_data_json"
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return []
    
    reports = []
    report_pattern = re.compile(r"(.+)_report_(\d{8}_\d{6})\.json")
    
    for filename in os.listdir(output_dir):
        if filename.endswith("_report_" + datetime.now().strftime("%Y%m%d") + "*.json"):
            # Prioritize today's reports
            match = report_pattern.match(filename)
            if match:
                company_name = match.group(1)
                timestamp = match.group(2)
                reports.insert(0, (os.path.join(output_dir, filename), company_name, timestamp))
        elif filename.endswith("_report_.json") or "_report_2" in filename:
            match = report_pattern.match(filename)
            if match:
                company_name = match.group(1)
                timestamp = match.group(2)
                reports.append((os.path.join(output_dir, filename), company_name, timestamp))
    
    # Sort by timestamp (newest first) after prioritizing today's reports
    return sorted(reports, key=lambda x: x[2], reverse=True)

# --------------------------------
# Dashboard Main Function
# --------------------------------

def main():
    st.title("Editable Business Intelligence Dashboard")
    
    # Get list of available reports
    available_reports = list_available_reports()
    
    if available_reports:
        # Add tabs for different ways to access reports
        tab1, tab2 = st.tabs(["Company Report", "Search Companies"])
        
        with tab1:
            # Create options for dropdown
            report_options = ["Select a company..."] + [f"{company} ({timestamp})" for _, company, timestamp in available_reports]
            
            # Track previous selection to detect changes
            previous_selection = st.session_state.get('previous_selection', '')
            
            # Add a dropdown for report selection
            selected_option = st.selectbox(
                "Choose a company report to view:",
                options=report_options,
                index=0,
                key=f"report_selector_{hash('selector') % 10000}"  # Add a unique key
            )
            
            # Check if selection has changed
            if selected_option != previous_selection:
                # Clear document data when selection changes
                if 'document_data' in st.session_state:
                    del st.session_state.document_data
                # Also clear editing mode when switching reports
                st.session_state.editing_mode = False
                
                # Update previous selection
                st.session_state.previous_selection = selected_option
                
                # Set refresh flag
                st.session_state.refresh_data = True
            
            if selected_option != "Select a company...":
                # Extract the selected index
                selected_index = report_options.index(selected_option) - 1
                selected_file = available_reports[selected_index][0]
                
                # Load the JSON file properly
                try:
                    with open(selected_file, 'r') as f:
                        selected_data = json.load(f)
                    # Apply compatibility fixes
                    selected_data = ensure_report_compatibility(selected_data)
                    
                    # Display the selected report
                    with st.spinner(f"Loading report for {available_reports[selected_index][1]}..."):
                        parselyfi_editable_company_report_page(selected_data)
                except Exception as e:
                    st.error(f"Error loading report: {str(e)}")
            else:
                st.info("Please select a company report, or view the demo data.")
                view_demo = st.button("View Demo Report")
                if view_demo:
                    # Create and display demo report
                    demo_data = create_default_report_data()
                    # Clear existing data
                    if 'document_data' in st.session_state:
                        del st.session_state.document_data
                    st.session_state.refresh_data = True
                    parselyfi_editable_company_report_page(demo_data)
        
        with tab2:
            # Add a search box
            search_term = st.text_input("Search for a company by name:")
            
            if search_term:
                # Filter reports by search term
                matching_reports = [(file, company, time) for file, company, time in available_reports 
                                   if search_term.lower() in company.lower()]
                
                if matching_reports:
                    st.success(f"Found {len(matching_reports)} matching reports")
                    
                    # Display as a table with a view button
                    for i, (file, company, timestamp) in enumerate(matching_reports):
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.write(f"**{company}**")
                        with col2:
                            formatted_time = f"{timestamp[4:6]}/{timestamp[6:8]}/{timestamp[:4]} {timestamp[9:11]}:{timestamp[11:13]}"
                            st.write(f"Generated: {formatted_time}")
                        with col3:
                            if st.button(f"View", key=f"view_{i}"):
                                # Store selection in session state and rerun
                                st.session_state.selected_file = file
                                st.session_state.selected_company = company
                                st.rerun()
                else:
                    st.warning("No matching companies found.")
    else:
        st.warning("No company reports found in the output directory.")
        st.info("Showing demo data for LlamaIndex instead.")
        
        # Create and display demo report
        demo_data = create_default_report_data()
        parselyfi_editable_company_report_page(demo_data)
    
    # Handle viewing a report when selected via search
    if 'selected_file' in st.session_state:
        selected_file = st.session_state.selected_file
        selected_company = st.session_state.selected_company
        
        # Clear the selection from session state
        del st.session_state.selected_file
        del st.session_state.selected_company
        
        # Show the selected report
        with st.spinner(f"Loading report for {selected_company}..."):
            parselyfi_editable_company_report_page(selected_file)

if __name__ == "__main__":
    main()