import streamlit as st
import pandas as pd
from datetime import datetime
import time
import base64
from io import BytesIO
import re
import json

st.set_page_config(
    page_title="Financial Content Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add this function to handle updates to the session state
def update_document_data(field_name):
    """Update a simple field in the document data"""
    key = f"edit_{field_name}"
    if key in st.session_state:
        st.session_state.document_data[field_name] = st.session_state[key]

# Add this function to apply custom CSS for editable content
def apply_editable_css():
    """Apply custom CSS styling for editable content"""
    st.markdown("""
    <style>
        /* New styles for editable content */
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
        
        /* Streamlit text input override for inline editing */
        .edit-mode div[data-testid="stTextInput"] {
            margin-bottom: 0 !important;
        }
        
        .edit-mode div[data-testid="stTextInput"] > div > div > input {
            border: 1px solid #3B82F6 !important;
            background-color: #EFF6FF !important;
        }
        
        /* Make text areas blend better */
        .edit-mode .stTextArea textarea {
            background-color: #F9FAFB;
            border: 1px solid #D1D5DB;
            border-radius: 0.375rem;
            min-height: 100px;
        }
        
        /* Add a faint edit icon on hover */
        .editable-container {
            position: relative;
        }
        
        .editable-container:hover::after {
            content: "✏️";
            position: absolute;
            right: 8px;
            top: 8px;
            font-size: 14px;
            opacity: 0.5;
        }
        
        /* Style for the editing placeholder */
        .editing-placeholder {
            color: #9CA3AF;
            font-style: italic;
        }
        
        /* Add button styling */
        .add-item-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.25rem 0.75rem;
            font-size: 0.75rem;
            color: #4B5563;
            background-color: #F9FAFB;
            border: 1px dashed #D1D5DB;
            border-radius: 0.375rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .add-item-button:hover {
            background-color: #F3F4F6;
            color: #1F2937;
            border-color: #9CA3AF;
        }
    </style>
    """, unsafe_allow_html=True)

# Add helper functions for updating arrays in the document data
def update_document_key_point(index):
    """Update a key point in the document data"""
    key = f"edit_keyPoint_{index}"
    if key in st.session_state:
        st.session_state.document_data["keyPoints"][index] = st.session_state[key]

def update_document_insight(index):
    """Update an insight in the document data"""
    key = f"edit_insight_{index}"
    if key in st.session_state:
        st.session_state.document_data["keyInsights"][index] = st.session_state[key]

def update_document_statement(index, field):
    """Update a field in a quoted statement"""
    key = f"edit_{field}_{index}"
    if key in st.session_state:
        st.session_state.document_data["quotedStatements"][index][field] = st.session_state[key]

# Add this function to initialize the document data
def initialize_document_data(content_type):
    """Initialize document data in session state"""
    if 'document_data' not in st.session_state:
        if content_type == 'video':
            st.session_state.document_data = video_data
        else:
            st.session_state.document_data = article_data
        
        # Initialize editing mode
        st.session_state.editing_mode = False

# Modify the configure_page_style function to include editable styles
def configure_page_style():
    """Configure page settings and inject custom CSS styling"""
    
    # Apply the editable CSS
    apply_editable_css()
    
    st.markdown("""
    <style>
        /* Main Container Styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Report Header */
        .report-header {
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
        
        /* Tags */
        .tag-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
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
        
        /* Status indicators */
        .status-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        .positive-status {
            background-color: #D1FAE5;
            color: #065F46;
        }
        .neutral-status {
            background-color: #EFF6FF;
            color: #1E40AF;
        }
        .negative-status {
            background-color: #FEE2E2;
            color: #B91C1C;
        }
        .warning-status {
            background-color: #FEF3C7;
            color: #92400E;
        }
        
        /* Report metadata */
        .metadata {
            text-align: right;
            font-size: 0.875rem;
            color: #6B7280;
        }
        .metadata-highlight {
            color: #3B82F6;
            font-weight: 600;
        }
        .high-confidence {
            color: #059669;
            font-weight: 600;
        }
        
        /* Cards */
        .card {
            background-color: white;
            border: 1px solid #E5E7EB;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
        
        /* Section Titles */
        .section-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1F2937;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #E5E7EB;
        }
        .subsection-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1F2937;
            margin: 1rem 0 0.5rem 0;
        }
        
        /* Insights and Statements */
        .key-insight {
            display: flex;
            align-items: flex-start;
            margin-bottom: 0.75rem;
            padding: 0.5rem;
            border-radius: 0.375rem;
            transition: background-color 0.2s;
        }
        .key-insight:hover {
            background-color: #F9FAFB;
        }
        .key-statement {
            background-color: #F8FAFC;
            padding: 1.25rem;
            border-radius: 0.5rem;
            margin-bottom: 1.25rem;
            border: 1px solid #E2E8F0;
        }
        .quote {
            font-style: italic;
            margin-bottom: 0.75rem;
            color: #334155;
            line-height: 1.6;
        }
        .speaker {
            font-weight: 600;
            color: #334155;
        }
        .timestamp {
            color: #3B82F6;
            font-weight: 500;
        }
        
        /* Data Tables */
        .dataframe {
            border-collapse: separate !important;
            border-spacing: 0;
            width: 100%;
            border: 1px solid #E5E7EB;
            border-radius: 0.5rem;
            overflow: hidden;
        }
        .dataframe thead {
            background-color: #F1F5F9;
        }
        .dataframe thead tr th {
            text-align: left;
            padding: 0.75rem;
            font-weight: 600;
            color: #1F2937;
            border-bottom: 1px solid #E5E7EB;
        }
        .dataframe tbody tr {
            border-bottom: 1px solid #E5E7EB;
        }
        .dataframe tbody tr:last-child {
            border-bottom: none;
        }
        .dataframe tbody tr:nth-child(even) {
            background-color: #F9FAFB;
        }
        .dataframe tbody tr:hover {
            background-color: #EFF6FF;
        }
        .dataframe tbody tr td {
            padding: 0.75rem;
            color: #374151;
            border: none;
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
        
        /* Tab content */
        .tab-content {
            padding: 1.5rem;
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Expander styling */
        .stExpander {
            border: 1px solid #E5E7EB;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            overflow: hidden;
        }
        details {
            background-color: white;
        }
        details summary {
            padding: 0.75rem;
            background-color: #F9FAFB;
            border-bottom: 1px solid #E5E7EB;
            font-weight: 500;
            color: #1F2937;
        }
        details summary:hover {
            background-color: #F3F4F6;
        }
        details[open] summary {
            border-bottom: 1px solid #E5E7EB;
        }
        
        /* Chat UI */
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #EFF6FF;
            border: 1px solid #BFDBFE;
            align-self: flex-end;
            margin-left: 2rem;
        }
        .ai-message {
            background-color: #F9FAFB;
            border: 1px solid #E5E7EB;
            align-self: flex-start;
            margin-right: 2rem;
        }
        
        /* Footer */
        .footer {
            margin-top: 2rem;
            padding-top: 1rem;
            text-align: center;
            font-size: 0.875rem;
            color: #6B7280;
            border-top: 1px solid #E5E7EB;
        }
    </style>
    """, unsafe_allow_html=True)

# Modify the render_report_header function to support editing
def render_report_header(title, report_metadata):
    """Render an enhanced report header section with report_metadata"""
    
    # Initialize document data if not already present
    initialize_document_data(st.session_state.content_type)
    
    # Add toggle for editing mode
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.session_state.editing_mode = st.toggle(
            "Enable Editing Mode", 
            value=st.session_state.get('editing_mode', False)
        )
        
        if st.session_state.editing_mode:
            st.info("📝 Editing mode is enabled. Click on any text to edit. Changes are saved automatically.")
            
            # Add export button
            if st.button("Export Edited Document"):
                export_data = st.session_state.document_data
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"{st.session_state.content_type}_analysis_edited.json",
                    mime="application/json"
                )
    
    # Inject CSS for styling
    st.markdown("""
    <style>
        .report-header-wrapper {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .company-title {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .report-subtitle {
            font-size: 16px;
            color: #666;
            margin-bottom: 10px;
        }
        .metadata {
            font-size: 14px;
            color: #555;
            text-align: right;
        }
        .metadata-row {
            margin-bottom: 4px;
            line-height: 1.4;
        }
        .metadata-highlight {
            color: #3B82F6;
            font-weight: 600;
        }
        .tag-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
            width: 100%;
            justify-content: flex-start;
        }
        .tag {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .green-tag { background-color: #e6f4ea; color: #137333; }
        .purple-tag { background-color: #f3e8fd; color: #8312c6; }
        .blue-tag { background-color: #e8f0fe; color: #1967d2; }
        .yellow-tag { background-color: #FFFBEB; color: #92400E; }
        .high-confidence { color: #137333; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # In edit mode, show editable fields
        new_title = st.text_input(
            "Document Title", 
            value=st.session_state.document_data["title"],
            key="edit_header_title",  # Changed from edit_title
            on_change=update_document_data,
            args=("title",)
        )
        st.session_state.document_data["title"] = new_title
        
        # Render the rest of the header with editable fields
        cols = st.columns(2)
        with cols[0]:
            if "channel" in st.session_state.document_data:
                new_channel = st.text_input(
                    "Channel/Source", 
                    value=st.session_state.document_data["channel"],
                    key="edit_header_channel",  # Changed from edit_channel
                    on_change=update_document_data,
                    args=("channel",)
                )
                st.session_state.document_data["channel"] = new_channel
            else:
                new_source = st.text_input(
                    "Source", 
                    value=st.session_state.document_data["source"],
                    key="edit_header_source",  # Changed from edit_source
                    on_change=update_document_data,
                    args=("source",)
                )
                st.session_state.document_data["source"] = new_source
            
            new_date = st.text_input(
                "Date", 
                value=st.session_state.document_data["date"],
                key="edit_header_date",  # Changed from edit_date
                on_change=update_document_data,
                args=("date",)
            )
            st.session_state.document_data["date"] = new_date
            
        with cols[1]:
            if "duration" in st.session_state.document_data:
                new_duration = st.text_input(
                    "Duration", 
                    value=st.session_state.document_data["duration"],
                    key="edit_header_duration",  # Changed from edit_duration
                    on_change=update_document_data,
                    args=("duration",)
                )
                st.session_state.document_data["duration"] = new_duration
            else:
                new_reading_time = st.text_input(
                    "Reading Time", 
                    value=st.session_state.document_data["readingTime"],
                    key="edit_header_readingTime",  # Changed from edit_readingTime
                    on_change=update_document_data,
                    args=("readingTime",)
                )
                st.session_state.document_data["readingTime"] = new_reading_time
            
            if "views" in st.session_state.document_data:
                new_views = st.text_input(
                    "Views", 
                    value=st.session_state.document_data["views"],
                    key="edit_header_views",  # Changed from edit_views
                    on_change=update_document_data,
                    args=("views",)
                )
                st.session_state.document_data["views"] = new_views
        
        # Summary editing
        new_summary = st.text_area(
            "Summary", 
            value=st.session_state.document_data["summary"],
            key="edit_header_summary",  # Changed from edit_summary
            on_change=update_document_data,
            args=("summary",)
        )
        st.session_state.document_data["summary"] = new_summary
    else:
        # Create the full header HTML with proper wrapper
        header_html = '<div class="report-header-wrapper">'
        
        # Title and subtitle section
        header_html += '<div style="display: flex; flex-direction: row; margin-bottom: 10px;">'
        
        # Left column (title and subtitle) - approximately 60% width
        header_html += '<div style="flex: 3;">'
        header_html += f'<div class="company-title">{title}</div>'
        header_html += f'<div class="report-subtitle">Financial Content Analysis</div>'
        header_html += '</div>'  # Close left column
        
        # Right column (metadata) - approximately 40% width
        header_html += '<div style="flex: 2;">'
        header_html += '<div class="metadata">'
        header_html += f'<div class="metadata-row">Report Generated: <span class="metadata-highlight">{report_metadata["date"]}</span></div>'
        header_html += f'<div class="metadata-row">Analysis Type: <span class="metadata-highlight">{report_metadata["type"]}</span></div>'
        header_html += f'<div class="metadata-row">Report ID: <span>{report_metadata["id"]}</span></div>'
        header_html += f'<div class="metadata-row">Source: <span>{report_metadata["source"]}</span></div>'
        header_html += f'<div class="metadata-row">Data Confidence: <span class="high-confidence">{report_metadata.get("confidence", "High")}</span></div>'
        header_html += '</div>'  # Close metadata div
        header_html += '</div>'  # Close right column
        
        header_html += '</div>'  # Close title and metadata row
        
        # Tags section - placed below the title/metadata and spread across full width
        header_html += '<div class="tag-container">'
        for tag in report_metadata['tags']:
            if "Fed" in tag or "Interest" in tag:
                tag_class = "blue-tag"
            elif "Economy" in tag or "Market" in tag:
                tag_class = "green-tag"
            elif "Analysis" in tag:
                tag_class = "purple-tag"
            else:
                tag_class = "yellow-tag"
            header_html += f'<span class="tag {tag_class}">{tag}</span>'
        header_html += '</div>'  # Close tag container
        
        header_html += '</div>'  # Close report-header-wrapper
        
        # Render the complete header HTML
        st.markdown(header_html, unsafe_allow_html=True)

# Modify the render_video_content_analysis function to support editing
def render_video_content_analysis(data):
    """Render content analysis for video data with editing support"""
    
    # Section title with edit toggle
    st.markdown('<div class="section-title">Source Information</div>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Editable thumbnail URL
        new_thumbnail = st.text_input(
            "Thumbnail URL", 
            value=data["thumbnail"],
            key="edit_video_thumbnail",  # Changed from edit_thumbnail
            on_change=update_document_data,
            args=("thumbnail",)
        )
        data["thumbnail"] = new_thumbnail
        
        if new_thumbnail:
            st.image(new_thumbnail, use_container_width=True)
    
    # Create a single container with border to visually group the info and thumbnail
    with st.container(border=True):
        # Create columns inside the container
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.session_state.editing_mode:
                # Editable video info
                st.markdown("### Video Information")
                
                new_channel = st.text_input(
                    "Channel", 
                    value=data["channel"],
                    key="edit_video_channel",  # Changed from edit_channel
                    on_change=update_document_data,
                    args=("channel",)
                )
                data["channel"] = new_channel
                
                new_duration = st.text_input(
                    "Duration", 
                    value=data["duration"],
                    key="edit_video_duration",  # Changed from edit_duration
                    on_change=update_document_data,
                    args=("duration",)
                )
                data["duration"] = new_duration
                
                new_views = st.text_input(
                    "Views", 
                    value=data["views"],
                    key="edit_video_views",  # Changed from edit_views
                    on_change=update_document_data,
                    args=("views",)
                )
                data["views"] = new_views
                
                new_date = st.text_input(
                    "Date", 
                    value=data["date"],
                    key="edit_video_date",  # Changed from edit_date
                    on_change=update_document_data,
                    args=("date",)
                )
                data["date"] = new_date
                
                new_url = st.text_input(
                    "URL", 
                    value=data["url"],
                    key="edit_video_url",  # Changed from edit_url
                    on_change=update_document_data,
                    args=("url",)
                )
                data["url"] = new_url
            else:
                # Video title
                st.markdown(f"""
                <h3 style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; color: #1F2937;">{data['title']}</h3>
                """, unsafe_allow_html=True)
                
                # Channel info with badge
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="background-color: #EF4444; color: white; width: 24px; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; margin-right: 8px;">
                        <span style="font-size: 10px; font-weight: 700;">YT</span>
                    </div>
                    <span style="font-weight: 500;">{data['channel']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Video metadata (duration, views, date)
                st.markdown(f"""
                <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 1rem;">
                    <div style="display: flex; align-items: center; color: #6B7280; font-size: 0.875rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12 6 12 12 16 14"></polyline>
                        </svg>
                        {data['duration']}
                    </div>
                    <div style="display: flex; align-items: center; color: #6B7280; font-size: 0.875rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                            <circle cx="12" cy="12" r="3"></circle>
                        </svg>
                        {data['views']} views
                    </div>
                    <div style="display: flex; align-items: center; color: #6B7280; font-size: 0.875rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                            <line x1="16" y1="2" x2="16" y2="6"></line>
                            <line x1="8" y1="2" x2="8" y2="6"></line>
                            <line x1="3" y1="10" x2="21" y2="10"></line>
                        </svg>
                        {data['date']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Watch link - using Streamlit's native components
                link_text = "Watch Video on YouTube"
                st.markdown(f'<a href="{data["url"]}" target="_blank" style="color: #3B82F6; text-decoration: none;">{link_text}</a>', unsafe_allow_html=True)
        
        with col2:
            if not st.session_state.editing_mode:
                # Instead of HTML for the thumbnail, use Streamlit's image with overlay
                # First add some spacing to align with the content
                st.write("")
                
                # Use a column for centering the image and creating proper proportions
                img_col = st.columns([0.1, 0.8, 0.1])[1]
                with img_col:
                    # Display the image
                    st.image(data['thumbnail'], use_container_width=True)
                    
                    # We'll have to use a simplified play button approach with Streamlit
                    st.markdown("""
                    <div style="text-align: center; margin-top: -50px; position: relative; z-index: 10;">
                        <div style="display: inline-block; background-color: rgba(0,0,0,0.7); width: 50px; height: 50px; border-radius: 50%; border: 2px solid white;">
                            <div style="margin-top: 12px; margin-left: 18px; width: 0; height: 0; border-top: 12px solid transparent; border-bottom: 12px solid transparent; border-left: 18px solid white;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Executive Summary
    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Editable summary
        new_summary = st.text_area(
            "Summary", 
            value=data["summary"],
            key="edit_video_summary",  # Changed from edit_summary
            on_change=update_document_data,
            args=("summary",)
        )
        data["summary"] = new_summary
    else:
        st.markdown(f'<div class="card">{data["summary"]}</div>', unsafe_allow_html=True)
    
    # Key Insights
    st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Editable insights
        st.markdown("### Key Insights")
        for i, insight in enumerate(data['keyInsights']):
            cols = st.columns([5, 1])
            with cols[0]:
                new_insight = st.text_input(
                    f"Insight {i+1}", 
                    value=insight,
                    key=f"edit_insight_{i}",
                    on_change=update_document_insight,
                    args=(i,)
                )
                data["keyInsights"][i] = new_insight
            with cols[1]:
                if st.button("🗑️", key=f"delete_insight_{i}"):
                    data["keyInsights"].pop(i)
                    st.rerun()
        
        if st.button("+ Add Insight", key="add_insight"):
            data["keyInsights"].append("New insight")
            st.rerun()
    else:
        # Create a card for insights with improved styling
        insights_cols = st.columns(2)
        for i, insight in enumerate(data['keyInsights']):
            with insights_cols[i % 2]:
                st.markdown(f"""
                <div class="card">
                    <div class="key-insight">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 10px; flex-shrink: 0; margin-top: 2px;">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        <span style="color: #374151;">{insight}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Key Statements with editing
    st.markdown('<div class="section-title">Key Statements with Timestamps</div>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Editable statements
        for i, statement in enumerate(data['quotedStatements']):
            st.markdown(f"### Statement {i+1}")
            
            cols = st.columns(2)
            with cols[0]:
                new_speaker = st.text_input(
                    "Speaker", 
                    value=statement['speaker'],
                    key=f"edit_speaker_{i}",
                    on_change=update_document_statement,
                    args=(i, "speaker")
                )
                data['quotedStatements'][i]['speaker'] = new_speaker
            
            with cols[1]:
                new_timestamp = st.text_input(
                    "Timestamp", 
                    value=statement['timestamp'],
                    key=f"edit_timestamp_{i}",
                    on_change=update_document_statement,
                    args=(i, "timestamp")
                )
                data['quotedStatements'][i]['timestamp'] = new_timestamp
            
            new_quote = st.text_area(
                "Quote", 
                value=statement['quote'],
                key=f"edit_quote_{i}",
                on_change=update_document_statement,
                args=(i, "quote")
            )
            data['quotedStatements'][i]['quote'] = new_quote
            
            if st.button("🗑️ Delete Statement", key=f"delete_statement_{i}"):
                data['quotedStatements'].pop(i)
                st.rerun()
            
            st.markdown("---")
        
        if st.button("+ Add Statement", key="add_statement"):
            data['quotedStatements'].append({
                "speaker": "New Speaker",
                "quote": "New quote text",
                "timestamp": "0:00"
            })
            st.rerun()
    else:
        # Create a grid for statements
        statement_cols = st.columns(2)
        for i, statement in enumerate(data['quotedStatements']):
            with statement_cols[i % 2]:
                st.markdown(f"""
                <div class="key-statement">
                    <div class="quote">"{statement['quote']}"</div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="speaker">— {statement['speaker']}</span>
                        <span class="timestamp">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                                <circle cx="12" cy="12" r="10"></circle>
                                <polyline points="12 6 12 12 16 14"></polyline>
                            </svg>
                            {statement['timestamp']}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Modify the render_article_content_analysis function
def render_article_content_analysis(data):
    """Render content analysis for article data with editing support"""
    
    # Source information section
    with st.container():
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="section-title">Source Information</div>', unsafe_allow_html=True)
            
            if st.session_state.editing_mode:
                # Editable article info
                st.markdown("### Article Information")
                
                cols = st.columns(2)
                with cols[0]:
                    new_title = st.text_input(
                        "Title", 
                        value=data["title"],
                        key="edit_article_title",  # Changed from edit_title
                        on_change=update_document_data,
                        args=("title",)
                    )
                    data["title"] = new_title
                    
                    new_source = st.text_input(
                        "Source", 
                        value=data["source"],
                        key="edit_article_source",  # Changed from edit_source
                        on_change=update_document_data,
                        args=("source",)
                    )
                    data["source"] = new_source
                    
                    new_author = st.text_input(
                        "Author", 
                        value=data["author"],
                        key="edit_article_author",  # Changed from edit_author
                        on_change=update_document_data,
                        args=("author",)
                    )
                    data["author"] = new_author
                    
                with cols[1]:
                    new_date = st.text_input(
                        "Date", 
                        value=data["date"],
                        key="edit_article_date",  # Changed from edit_date
                        on_change=update_document_data,
                        args=("date",)
                    )
                    data["date"] = new_date
                    
                    new_reading_time = st.text_input(
                        "Reading Time", 
                        value=data["readingTime"],
                        key="edit_article_readingTime",  # Changed from edit_readingTime
                        on_change=update_document_data,
                        args=("readingTime",)
                    )
                    data["readingTime"] = new_reading_time
                    
                    new_url = st.text_input(
                        "URL", 
                        value=data["url"],
                        key="edit_article_url",  # Changed from edit_url
                        on_change=update_document_data,
                        args=("url",)
                    )
                    data["url"] = new_url
                
                # Editable thumbnail
                new_thumbnail = st.text_input(
                    "Thumbnail URL", 
                    value=data["thumbnail"],
                    key="edit_article_thumbnail",  # Changed from edit_thumbnail
                    on_change=update_document_data,
                    args=("thumbnail",)
                )
                data["thumbnail"] = new_thumbnail
            else:
                # Source info as a card with improved styling
                st.markdown(f"""
                <div class="card">
                    <h3 style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; color: #1F2937;">{data['title']}</h3>
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <div style="background-color: #2563EB; color: white; width: 24px; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; margin-right: 8px;">
                            <span style="font-size: 10px; font-weight: 700;">FT</span>
                        </div>
                        <span style="font-weight: 500;">{data['source']}</span>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 1rem;">
                        <div style="display: flex; align-items: center; color: #6B7280; font-size: 0.875rem;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                                <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"></path>
                            </svg>
                            {data['author']}
                        </div>
                        <div style="display: flex; align-items: center; color: #6B7280; font-size: 0.875rem;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                                <circle cx="12" cy="12" r="10"></circle>
                                <polyline points="12 6 12 12 16 14"></polyline>
                            </svg>
                            {data['readingTime']}
                        </div>
                        <div style="display: flex; align-items: center; color: #6B7280; font-size: 0.875rem;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                                <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                                <line x1="16" y1="2" x2="16" y2="6"></line>
                                <line x1="8" y1="2" x2="8" y2="6"></line>
                                <line x1="3" y1="10" x2="21" y2="10"></line>
                            </svg>
                            {data['date']}
                        </div>
                    </div>
                    <a href="{data['url']}" target="_blank" style="color: #3B82F6; display: flex; align-items: center; font-size: 0.875rem; text-decoration: none; margin-bottom: 1rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
                        </svg>
                        Read Full Article
                    </a>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.editing_mode and data["thumbnail"]:
                st.image(data["thumbnail"], use_container_width=True)
            elif not st.session_state.editing_mode:
                # Display article image with styled container
                st.markdown(f"""
                <div style="margin-top: 3.5rem;">
                    <img src="{data['thumbnail']}" alt="Article Image" style="width: 100%; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                </div>
                """, unsafe_allow_html=True)
    
    # Executive Summary
    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Editable summary
        new_summary = st.text_area(
            "Summary", 
            value=data["summary"],
            key="edit_article_summary",  # Changed from edit_summary
            on_change=update_document_data,
            args=("summary",)
        )
        data["summary"] = new_summary
    else:
        st.markdown(f'<div class="card">{data["summary"]}</div>', unsafe_allow_html=True)
    
    # Key Points
    st.markdown('<div class="section-title">Key Points</div>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Editable key points
        for i, point in enumerate(data["keyPoints"]):
            cols = st.columns([5, 1])
            with cols[0]:
                new_point = st.text_input(
                    f"Point {i+1}", 
                    value=point,
                    key=f"edit_keyPoint_{i}",
                    on_change=update_document_key_point,
                    args=(i,)
                )
                data["keyPoints"][i] = new_point
            with cols[1]:
                if st.button("🗑️", key=f"delete_keyPoint_{i}"):
                    data["keyPoints"].pop(i)
                    st.rerun()
        
        if st.button("+ Add Key Point", key="add_key_point"):
            data["keyPoints"].append("New key point")
            st.rerun()
    else:
        # Create a card for key points with improved styling
        st.markdown('<div class="card">', unsafe_allow_html=True)
        points_cols = st.columns(2)
        for i, point in enumerate(data['keyPoints']):
            with points_cols[i % 2]:
                st.markdown(f"""
                <div class="key-insight">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 10px; flex-shrink: 0; margin-top: 2px;">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                    </svg>
                    <span style="color: #374151;">{point}</span>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Expert Opinions with editing
    if "expertOpinions" in data:
        st.markdown('<div class="section-title">Expert Opinions</div>', unsafe_allow_html=True)
        
        if st.session_state.editing_mode:
            # Editable expert opinions
            for i, opinion in enumerate(data["expertOpinions"]):
                st.markdown(f"### Expert {i+1}")
                
                cols = st.columns(2)
                with cols[0]:
                    new_name = st.text_input(
                        "Name", 
                        value=opinion['name'],
                        key=f"edit_expert_name_{i}",
                        on_change=lambda i=i, field='name': update_expert_opinion(i, field),
                    )
                    data["expertOpinions"][i]['name'] = new_name
                
                with cols[1]:
                    new_title = st.text_input(
                        "Title", 
                        value=opinion['title'],
                        key=f"edit_expert_title_{i}",
                        on_change=lambda i=i, field='title': update_expert_opinion(i, field),
                    )
                    data["expertOpinions"][i]['title'] = new_title
                
                new_quote = st.text_area(
                    "Quote", 
                    value=opinion['quote'],
                    key=f"edit_expert_quote_{i}",
                    on_change=lambda i=i, field='quote': update_expert_opinion(i, field),
                )
                data["expertOpinions"][i]['quote'] = new_quote
                
                if st.button("🗑️ Delete Expert", key=f"delete_expert_{i}"):
                    data["expertOpinions"].pop(i)
                    st.rerun()
                
                st.markdown("---")
            
            if st.button("+ Add Expert Opinion", key="add_expert"):
                data["expertOpinions"].append({
                    "name": "New Expert",
                    "title": "Title/Organization",
                    "quote": "Expert opinion quote"
                })
                st.rerun()
        else:
            # Create opinions in a grid layout with improved styling
            opinion_cols = st.columns(len(data['expertOpinions']))
            
            for i, opinion in enumerate(data['expertOpinions']):
                with opinion_cols[i]:
                    st.markdown(f"""
                    <div class="card" style="height: 100%;">
                        <div style="margin-bottom: 1rem;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#E5E7EB" stroke="#9CA3AF" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V20c0 1 0 1 1 1z"></path>
                                <path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 4v3c0 1 0 1 1 1z"></path>
                            </svg>
                        </div>
                        <div class="quote" style="margin-bottom: 1rem; min-height: 4rem;">{opinion['quote']}</div>
                        <div style="border-top: 1px solid #E5E7EB; padding-top: 1rem;">
                            <div class="speaker" style="font-size: 1rem; color: #1F2937;">{opinion['name']}</div>
                            <div style="color: #6B7280; font-size: 0.875rem;">{opinion['title']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Add helper function for expert opinion updates
def update_expert_opinion(index, field):
    """Update a field in an expert opinion"""
    key = f"edit_expert_{field}_{index}"
    if key in st.session_state:
        st.session_state.document_data["expertOpinions"][index][field] = st.session_state[key]

def create_excel_download_button(data, content_type):
    """Create a download button for exporting data to Excel"""
    output = BytesIO()
    
    try:
        # Create basic info dataframe
        if content_type == 'video':
            basic_info = {
                "Title": data['title'],
                "Channel": data['channel'],
                "Duration": data['duration'],
                "Views": data['views'],
                "Published Date": data['date'],
                "URL": data['url']
            }
        else:  # article
            basic_info = {
                "Title": data['title'],
                "Source": data['source'],
                "Author": data['author'],
                "Reading Time": data['readingTime'],
                "Published Date": data['date'],
                "URL": data['url']
            }
        
        basic_df = pd.DataFrame([basic_info])
        
        # Create Excel with pandas
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write basic info
            basic_df.to_excel(writer, sheet_name='Source Information', index=False)
            
            # Write executive summary
            pd.DataFrame({"Content": [data['summary']]}).to_excel(
                writer, sheet_name='Executive Summary', index=False)
            
            # Write key insights/points
            if content_type == 'video':
                pd.DataFrame({"Key Insights": data['keyInsights']}).to_excel(
                    writer, sheet_name='Key Insights', index=False)
                
                # Write key topics
                topics_df = pd.DataFrame([
                    {
                        "Topic": t["name"],
                        "Description": t["description"],
                        "Quote 1": t["quotes"][0]["text"] if len(t["quotes"]) > 0 else "",
                        "Quote 2": t["quotes"][1]["text"] if len(t["quotes"]) > 1 else ""
                    }
                    for t in data["keyTopics"]
                ])
                topics_df.to_excel(writer, sheet_name='Key Topics', index=False)
                
                # Write statements
                statements_df = pd.DataFrame([
                    {
                        "Speaker": s["speaker"],
                        "Quote": s["quote"],
                        "Timestamp": s["timestamp"]
                    }
                    for s in data["quotedStatements"]
                ])
                statements_df.to_excel(writer, sheet_name='Key Statements', index=False)
                
                # Write indicators
                indicators_df = pd.DataFrame(data["economicContext"]["indicators"])
                indicators_df.to_excel(writer, sheet_name='Financial Indicators', index=False)
            else:
                pd.DataFrame({"Key Points": data['keyPoints']}).to_excel(
                    writer, sheet_name='Key Points', index=False)
                
                # Write expert opinions
                opinions_df = pd.DataFrame([
                    {
                        "Expert": o["name"],
                        "Title": o["title"],
                        "Opinion": o["quote"]
                    }
                    for o in data["expertOpinions"]
                ])
                opinions_df.to_excel(writer, sheet_name='Expert Opinions', index=False)
            
            # Format columns in each worksheet
            workbook = writer.book
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                # Set column width based on content type
                if sheet_name in ["Source Information", "Executive Summary"]:
                    worksheet.set_column(0, 0, 20)
                    worksheet.set_column(1, 1, 60)
                else:
                    # Standard width for other sheets
                    worksheet.set_column(0, 0, 20)
                    worksheet.set_column(1, 5, 30)
        
        # Get Excel data and create filename
        excel_data = output.getvalue()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        title_slug = re.sub(r'[^a-zA-Z0-9]', '_', data['title'][:30])
        filename = f"{content_type.capitalize()}_{title_slug}_{current_time}.xlsx"
        
        return st.download_button(
            label=f"📊 Download {content_type.capitalize()} Analysis",
            data=excel_data,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None


def main():
    # Configure page style
    configure_page_style()
    
    # Sample video report JSON (in production, this would come from a file or API)
    video_report_json = """
    {
        "title": "Fed Chair Powell's Latest Interest Rate Decision - Economic Outlook 2025",
        "channel": "EconInsights",
        "date": "March 18, 2025",
        "duration": "18:35",
        "views": "156,847",
        "url": "https://www.youtube.com/watch?v=example1234",
        "thumbnail": "https://via.placeholder.com/640x360",
        "summary": "This video features a detailed analysis of Federal Reserve Chair Jerome Powell's latest press conference following the March 2025 FOMC meeting. The Fed signaled plans for three potential interest rate cuts in 2025, citing improving inflation trends and balanced economic growth. The analyst provides context around the decision, examines market reactions, and discusses implications for various sectors including housing, technology, and banking. The overall outlook is cautiously optimistic, with emphasis on the Fed's data-dependent approach going forward.",
        "keyInsights": [
            "Federal Reserve signals three potential rate cuts for 2025",
            "Inflation trending toward 2% target with core PCE at 2.4%",
            "Labor market showing balanced growth with unemployment steady at 4.1%",
            "Housing market expected to benefit from potential rate cuts",
            "Tech sector rallied following the announcement with 2.8% gains"
        ],
        "quotedStatements": [
            {
                "speaker": "Powell (via analyst)",
                "quote": "The committee sees the risks to achieving its employment and inflation goals as moving into better balance.",
                "timestamp": "3:45"
            },
            {
                "speaker": "Analyst",
                "quote": "This signals a significant pivot from the Fed's previous hawkish stance maintained throughout 2024.",
                "timestamp": "4:18"
            },
            {
                "speaker": "Powell (via analyst)",
                "quote": "We need to see continued good data before we can begin the process of reducing our policy rate.",
                "timestamp": "7:22"
            },
            {
                "speaker": "Analyst",
                "quote": "The housing market could see a meaningful recovery as mortgage rates begin to decline from their multi-decade highs.",
                "timestamp": "12:45"
            }
        ],
        "keyTopics": [
            {
                "name": "Interest Rate Outlook",
                "description": "Analysis of the Federal Reserve's projection for three rate cuts in 2025 and the conditions that would support this path.",
                "quotes": [
                    { "text": "The dot plot showed a median projection of three 25 basis point cuts for 2025, with the first potentially occurring in June.", "timestamp": "5:12" },
                    { "text": "There's still a significant dispersion in the dot plot, showing some committee members prefer only two cuts while others project four.", "timestamp": "6:05" }
                ]
            },
            {
                "name": "Inflation Trends",
                "description": "Current inflation data and the Fed's assessment of progress toward the 2% target.",
                "quotes": [
                    { "text": "Core PCE has declined from 2.8% at the end of 2024 to the current 2.4%, showing meaningful progress toward the 2% target.", "timestamp": "8:30" },
                    { "text": "Powell emphasized that services inflation remains somewhat elevated, requiring continued vigilance.", "timestamp": "9:15" }
                ]
            },
            {
                "name": "Labor Market Conditions",
                "description": "Analysis of current employment trends and how they influence the Fed's policy stance.",
                "quotes": [
                    { "text": "The unemployment rate has remained steady at 4.1% for three consecutive months, indicating a resilient but not overheating labor market.", "timestamp": "10:40" },
                    { "text": "Wage growth has moderated to 3.8% year-over-year, which is more consistent with the Fed's 2% inflation target.", "timestamp": "11:22" }
                ]
            },
            {
                "name": "Housing Market Impact",
                "description": "How potential rate cuts may affect mortgage rates and housing market activity.",
                "quotes": [
                    { "text": "The 30-year fixed mortgage rate could decline from its current 6.5% to potentially below 5.5% by year-end if the Fed delivers on all projected cuts.", "timestamp": "13:05" },
                    { "text": "Housing affordability could improve by approximately 15% with the anticipated rate reductions and ongoing wage growth.", "timestamp": "14:30" }
                ]
            }
        ],
        "transcriptHighlights": [
            { "timestamp": "0:45", "text": "Today we'll be analyzing the March 2025 FOMC meeting and Chair Powell's press conference from earlier this week." },
            { "timestamp": "2:15", "text": "The Federal Reserve kept rates unchanged at 5.25-5.50%, maintaining this level since July 2023." },
            { "timestamp": "3:45", "text": "Powell stated: 'The committee sees the risks to achieving its employment and inflation goals as moving into better balance.'" },
            { "timestamp": "5:12", "text": "The updated dot plot revealed a median projection of three 25 basis point cuts during 2025." },
            { "timestamp": "8:30", "text": "Core PCE inflation has declined to 2.4%, showing steady progress toward the Fed's 2% target." },
            { "timestamp": "10:40", "text": "The labor market has maintained a healthy 4.1% unemployment rate for three consecutive months." },
            { "timestamp": "13:05", "text": "Mortgage rates are projected to decline in response to the anticipated Fed cuts, potentially falling below 5.5% by year-end." },
            { "timestamp": "15:30", "text": "The tech sector has already responded positively to the Fed's signals, with the Nasdaq gaining 2.8% following the announcement." }
        ],
        "economicContext": {
            "events": [
                { "name": "Q4 2024 GDP Report", "date": "January 30, 2025", "description": "Showed 2.3% annualized growth, slightly above expectations." },
                { "name": "February 2025 CPI Report", "date": "March 12, 2025", "description": "Inflation at 2.6% year-over-year, continuing downward trend." },
                { "name": "March FOMC Meeting", "date": "March 15, 2025", "description": "Fed maintained current rates but signaled future cuts." }
            ],
            "policies": [
                "The Fed's dual mandate focus on both price stability and maximum employment",
                "Gradual transition from tightening to easing monetary policy",
                "Data-dependent approach with emphasis on incoming inflation and employment reports"
            ],
            "indicators": [
                { "name": "Federal Funds Rate", "current": "5.25-5.50%", "trend": "Stable, projected to decrease" },
                { "name": "Core PCE Inflation", "current": "2.4%", "trend": "Declining toward target" },
                { "name": "Unemployment Rate", "current": "4.1%", "trend": "Stable" },
                { "name": "30-Year Mortgage Rate", "current": "6.5%", "trend": "Projected to decline" },
                { "name": "10-Year Treasury Yield", "current": "3.95%", "trend": "Declining" }
            ]
        },
        "significance": "This video analyzes a pivotal moment in the Fed's monetary policy cycle, marking the potential end of a prolonged high-rate environment that has significantly impacted borrowing costs across the economy. The analysis is particularly valuable for investors, homebuyers, and businesses making capital expenditure decisions, as it provides detailed projections for interest rate movements that will influence financial markets throughout 2025. The timing is especially relevant as it comes amid improving inflation data that has given the Fed more flexibility after a restrictive policy stance throughout 2023-2024."
    }
    """
    
    # Initialize session state
    if 'content_type' not in st.session_state:
        st.session_state.content_type = 'video'
    
    if 'chat_query' not in st.session_state:
        st.session_state.chat_query = ""
    
    if 'editing_mode' not in st.session_state:
        st.session_state.editing_mode = False
    
    # Load document data if not already loaded
    if 'document_data' not in st.session_state:
        try:
            st.session_state.document_data = json.loads(video_report_json)
        except json.JSONDecodeError as e:
            st.error(f"Error parsing video report JSON: {str(e)}")
            st.session_state.document_data = {
                "title": "Error Loading Report",
                "channel": "",
                "date": "",
                "duration": "",
                "views": "",
                "url": "",
                "thumbnail": "",
                "summary": "Could not load the video report data.",
                "keyInsights": [],
                "quotedStatements": [],
                "keyTopics": [],
                "transcriptHighlights": [],
                "economicContext": {
                    "events": [],
                    "policies": [],
                    "indicators": []
                },
                "significance": ""
            }
    
    def change_content_type(new_type):
        st.session_state.content_type = new_type
        # Reset document_data when changing content type
        if 'document_data' in st.session_state:
            del st.session_state.document_data
        initialize_document_data(new_type)
    
    # Create sidebar with improved styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h1 style="font-size: 1.75rem; font-weight: 700; color: #1F2937; margin-bottom: 0;">ParselyFi</h1>
            <p style="color: #6B7280; font-size: 0.875rem;">Financial Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Content type selection
        st.markdown('<p style="font-size: 0.875rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Analysis Type</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            video_btn_style = "primary" if st.session_state.content_type == 'video' else "secondary"
            if st.button("📹 Video", use_container_width=True, type=video_btn_style):
                change_content_type('video')
        with col2:
            article_btn_style = "primary" if st.session_state.content_type == 'article' else "secondary"
            if st.button("📰 Article", use_container_width=True, type=article_btn_style):
                change_content_type('article')
        
        st.markdown('<div style="border-top: 1px solid #E5E7EB; margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
        
        # URL input field
        st.markdown('<p style="font-size: 0.875rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Analyze New Content</p>', unsafe_allow_html=True)
        
        url_input = st.text_input(
            "Enter URL to analyze:", 
            placeholder=f"{'YouTube URL...' if st.session_state.content_type == 'video' else 'News article URL...'}",
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Analyze Content", type="primary", use_container_width=True):
            if url_input:
                st.success(f"Analysis started for: {url_input}")
                with st.spinner("Analyzing content..."):
                    time.sleep(2)  # Simulate processing time
                st.success("Analysis complete!")
                st.rerun()
            else:
                st.error("Please enter a valid URL")
        
        st.markdown('<div style="border-top: 1px solid #E5E7EB; margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
        
        # Chat history section
        st.markdown('<p style="font-size: 0.875rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Conversation History</p>', unsafe_allow_html=True)
        
        if st.session_state.content_type == 'video':
            chat_data = video_chat_history
        else:
            chat_data = article_chat_history
            
        for message in chat_data:
            if message['type'] == 'user':
                st.markdown(f"""
                <div style="background-color: #EFF6FF; border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 0.75rem; border: 1px solid #BFDBFE;">
                    <p style="margin: 0; font-size: 0.875rem; color: #374151;">{message['content']}</p>
                    <p style="margin: 0; text-align: right; font-size: 0.75rem; color: #6B7280; margin-top: 0.25rem;">{message['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #F9FAFB; border-radius: 0.5rem; padding: 0.75rem; margin-bottom: 0.75rem; border: 1px solid #E5E7EB;">
                    <p style="margin: 0; font-size: 0.875rem; color: #374151;">{message['content']}</p>
                    <p style="margin: 0; text-align: right; font-size: 0.75rem; color: #6B7280; margin-top: 0.25rem;">{message['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        st.markdown('<p style="font-size: 0.875rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem; margin-top: 1rem;">Ask Follow-up Questions</p>', unsafe_allow_html=True)
        
        chat_query = st.text_input(
            "Type your question:", 
            key="sidebar_chat_input",
            placeholder="Ask about specific details...",
            label_visibility="collapsed"
        )
        
        chat_btn_col1, chat_btn_col2 = st.columns([3, 1])
        with chat_btn_col1:
            if st.button("📨 Send", use_container_width=True, type="primary"):
                if chat_query:
                    st.session_state.chat_query = chat_query
                    st.success("Question sent!")
                    time.sleep(1)
                    st.rerun()
        with chat_btn_col2:
            if st.button("🔄", help="Clear input"):
                st.session_state.chat_query = ""
                st.rerun()
    
    # Get current data based on content type
    current_data = st.session_state.document_data
    
    if st.session_state.content_type == 'video':
        report_metadata = {
            'date': datetime.now().strftime("%B %d, %Y"),
            'type': 'Video Analysis',
            'id': f"VID-{datetime.now().strftime('%Y%m%d')}-FED-01",
            'source': 'YouTube',
            'confidence': 'High',
            'tags': ["Fed Policy", "Interest Rates", "Economy", "Financial Markets", "Analysis"]
        }
    else:
        report_metadata = {
            'date': datetime.now().strftime("%B %d, %Y"),
            'type': 'Article Analysis',
            'id': f"ART-{datetime.now().strftime('%Y%m%d')}-FED-01",
            'source': 'Financial Times',
            'confidence': 'High',
            'tags': ["Fed Policy", "Rate Cuts", "Market Analysis", "Economy", "Financial News"]
        }
    
    # Render report header
    render_report_header(
        title=current_data['title'], 
        report_metadata=report_metadata
    )
    
    # Create tabs
    tabs = st.tabs([
        "📊 Content Analysis", 
        "💹 Financial Impacts", 
        "📈 Market Response", 
        "🔍 Follow-up Analysis"
    ])
    
    with tabs[0]:
        # Content Analysis Tab
        if st.session_state.content_type == 'video':
            render_video_content_analysis(current_data)
        else:
            render_article_content_analysis(current_data)
        
        # Download button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<div style='text-align: center; margin-top: 1.5rem;'>", unsafe_allow_html=True)
            create_excel_download_button(current_data, st.session_state.content_type)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[1]:
        # Financial Impacts Tab
        st.markdown('<div class="section-title">Financial Impacts</div>', unsafe_allow_html=True)
        
        # Market metrics in cards
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.markdown("""
            <div class="card">
                <div style="color: #6B7280; font-size: 0.875rem; margin-bottom: 0.5rem;">10-Year Treasury Yield</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1F2937; margin-bottom: 0.75rem;">3.95%</div>
                <div style="display: flex; align-items: center; color: #10B981; font-size: 0.875rem; font-weight: 500;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                        <polyline points="23 18 13.5 8.5 8.5 13.5 1 6"></polyline>
                        <polyline points="17 18 23 18 23 12"></polyline>
                    </svg>
                    -0.15%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            st.markdown("""
            <div class="card">
                <div style="color: #6B7280; font-size: 0.875rem; margin-bottom: 0.5rem;">S&P 500</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1F2937; margin-bottom: 0.75rem;">5,685.24</div>
                <div style="display: flex; align-items: center; color: #10B981; font-size: 0.875rem; font-weight: 500;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
                        <polyline points="17 6 23 6 23 12"></polyline>
                    </svg>
                    +1.8%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col3:
            st.markdown("""
            <div class="card">
                <div style="color: #6B7280; font-size: 0.875rem; margin-bottom: 0.5rem;">30-Year Mortgage Rate</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1F2937; margin-bottom: 0.75rem;">6.5%</div>
                <div style="display: flex; align-items: center; color: #10B981; font-size: 0.875rem; font-weight: 500;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                        <polyline points="23 18 13.5 8.5 8.5 13.5 1 6"></polyline>
                        <polyline points="17 18 23 18 23 12"></polyline>
                    </svg>
                    -0.2%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Sector impacts
        st.markdown('<div class="subsection-title">Sector-Specific Impacts</div>', unsafe_allow_html=True)
        
        sector_data = {
            'Sector': ['Technology', 'Financials', 'Real Estate', 'Utilities', 'Consumer Discretionary'],
            'Impact': ['Strong Positive', 'Positive', 'Strong Positive', 'Positive', 'Moderate Positive'],
            'Change (%)': ['+2.8%', '+1.5%', '+2.2%', '+1.9%', '+1.2%'],
            'Analysis': [
                'Tech stocks rallied on expectations of lower cost of capital benefiting growth companies.',
                'Banks may face near-term net interest margin pressure but benefit from increased lending activity.',
                'Housing and commercial real estate expected to benefit from lower borrowing costs.',
                'Yield-sensitive utilities gain attractiveness in a falling rate environment.',
                'Consumer spending projected to increase with potentially lower credit card and loan rates.'
            ]
        }
        
        sector_df = pd.DataFrame(sector_data)
        st.dataframe(sector_df, hide_index=True, use_container_width=True)
        
        # Interest rate projections chart
        st.markdown('<div class="subsection-title">Federal Funds Rate Projections</div>', unsafe_allow_html=True)
        
        chart_data = pd.DataFrame({
            'Date': ['Mar 2025', 'Jun 2025', 'Sep 2025', 'Dec 2025', 'Mar 2026'],
            'Projected Rate (%)': [5.50, 5.25, 5.00, 4.75, 4.50],
            'Previous Projection (%)': [5.50, 5.50, 5.25, 5.25, 5.00]
        })
        
        st.markdown('<div class="card" style="padding: 1rem;">', unsafe_allow_html=True)
        st.line_chart(
            chart_data, 
            x='Date',
            y=['Projected Rate (%)', 'Previous Projection (%)'],
            color=['#3B82F6', '#94A3B8']
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.caption("Source: Federal Reserve Summary of Economic Projections, March 2025")
        
    with tabs[2]:
        # Market Response Tab
        st.markdown('<div class="section-title">Market Response Analysis</div>', unsafe_allow_html=True)
        
        # Asset class performance
        st.markdown('<div class="subsection-title">Asset Class Performance Following Announcement</div>', unsafe_allow_html=True)
        
        asset_data = {
            'Asset': ['U.S. Stocks (S&P 500)', 'Tech Stocks (NASDAQ)', 'Small Caps (Russell 2000)', 
                      'U.S. Bonds (Agg.)', 'Gold', 'U.S. Dollar Index'],
            'Change (%)': ['+1.8%', '+2.4%', '+2.1%', '+0.8%', '+0.5%', '-0.7%'],
            'Analysis': [
                'Broad market rally driven by potential economic growth and lower borrowing costs.',
                'Growth stocks outperformed on expectations of improved valuations in lower-rate environment.',
                'Small caps showed strong gains as they typically benefit more from domestic economic improvement.',
                'Bonds rallied as lower future interest rates increase the value of existing bonds.',
                'Moderate gains as lower rates reduce opportunity cost of holding non-yielding assets.',
                'Dollar weakened as lower relative interest rates typically reduce currency attractiveness.'
            ]
        }
        
        asset_df = pd.DataFrame(asset_data)
        st.dataframe(asset_df, hide_index=True, use_container_width=True)
        
        # Market sentiment
        st.markdown('<div class="subsection-title">Market Sentiment Indicators</div>', unsafe_allow_html=True)
        
        sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
        
        with sentiment_col1:
            st.markdown("""
            <div class="card">
                <div style="color: #6B7280; font-size: 0.875rem; margin-bottom: 0.5rem;">VIX Volatility Index</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1F2937; margin-bottom: 0.75rem;">16.8</div>
                <div style="display: flex; align-items: center; color: #10B981; font-size: 0.875rem; font-weight: 500;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                        <polyline points="23 18 13.5 8.5 8.5 13.5 1 6"></polyline>
                        <polyline points="17 18 23 18 23 12"></polyline>
                    </svg>
                    -2.3 (Low Fear)
                </div>
                <p style="margin-top: 0.75rem; font-size: 0.75rem; color: #6B7280;">Market volatility index - lower values indicate less fear</p>
            </div>
            """, unsafe_allow_html=True)
            
        with sentiment_col2:
            st.markdown("""
            <div class="card">
                <div style="color: #6B7280; font-size: 0.875rem; margin-bottom: 0.5rem;">Fear & Greed Index</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1F2937; margin-bottom: 0.75rem;">72</div>
                <div style="background-color: #FEF3C7; color: #92400E; font-size: 0.75rem; font-weight: 500; display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; margin-bottom: 0.5rem;">
                    Greed
                </div>
                <div style="height: 8px; width: 100%; background-color: #E5E7EB; border-radius: 4px; margin-bottom: 0.5rem;">
                    <div style="height: 100%; width: 72%; background: linear-gradient(90deg, #EF4444 0%, #F59E0B 50%, #10B981 100%); border-radius: 4px;"></div>
                </div>
                <p style="margin-top: 0.75rem; font-size: 0.75rem; color: #6B7280;">Higher values indicate market greed (0-100 scale)</p>
            </div>
            """, unsafe_allow_html=True)
            
        with sentiment_col3:
            st.markdown("""
            <div class="card">
                <div style="color: #6B7280; font-size: 0.875rem; margin-bottom: 0.5rem;">Put/Call Ratio</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1F2937; margin-bottom: 0.75rem;">0.78</div>
                <div style="display: flex; align-items: center; color: #10B981; font-size: 0.875rem; font-weight: 500;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                        <polyline points="23 18 13.5 8.5 8.5 13.5 1 6"></polyline>
                        <polyline points="17 18 23 18 23 12"></polyline>
                    </svg>
                    -0.12 (Bullish)
                </div>
                <p style="margin-top: 0.75rem; font-size: 0.75rem; color: #6B7280;">Ratio of put to call options - lower values indicate bullish sentiment</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Expert commentary
        st.markdown('<div class="subsection-title">Expert Market Commentary</div>', unsafe_allow_html=True)
        
        expert_col1, expert_col2 = st.columns(2)
        
        with expert_col1:
            st.markdown("""
            <div class="card" style="height: 100%;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="background-color: #3B82F6; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 0.75rem; font-weight: 600;">MS</div>
                    <div>
                        <h3 style="font-weight: 600; margin: 0; color: #1F2937;">Morgan Stanley</h3>
                        <p style="margin: 0; color: #6B7280; font-size: 0.75rem;">Chief Investment Officer</p>
                    </div>
                </div>
                <p style="color: #374151; font-style: italic; border-left: 3px solid #3B82F6; padding-left: 1rem; margin-bottom: 0;">"The Fed's messaging represents a meaningful pivot that should extend the economic expansion and benefit risk assets."</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card" style="height: 100%; margin-top: 1rem;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="background-color: #8B5CF6; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 0.75rem; font-weight: 600;">JPM</div>
                    <div>
                        <h3 style="font-weight: 600; margin: 0; color: #1F2937;">JPMorgan</h3>
                        <p style="margin: 0; color: #6B7280; font-size: 0.75rem;">Chief Market Strategist</p>
                    </div>
                </div>
                <p style="color: #374151; font-style: italic; border-left: 3px solid #8B5CF6; padding-left: 1rem; margin-bottom: 0;">"The labor market resilience gives the Fed flexibility to cut rates gradually while maintaining inflation vigilance. This balanced approach is a positive for markets."</p>
            </div>
            """, unsafe_allow_html=True)
        
        with expert_col2:
            st.markdown("""
            <div class="card" style="height: 100%;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="background-color: #F59E0B; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 0.75rem; font-weight: 600;">GS</div>
                    <div>
                        <h3 style="font-weight: 600; margin: 0; color: #1F2937;">Goldman Sachs</h3>
                        <p style="margin: 0; color: #6B7280; font-size: 0.75rem;">Senior Economist</p>
                    </div>
                </div>
                <p style="color: #374151; font-style: italic; border-left: 3px solid #F59E0B; padding-left: 1rem; margin-bottom: 0;">"We now expect three 25bp cuts in 2025, likely beginning in June, which should support equity valuations particularly in the technology and consumer discretionary sectors."</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card" style="height: 100%; margin-top: 1rem;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="background-color: #10B981; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 0.75rem; font-weight: 600;">BLK</div>
                    <div>
                        <h3 style="font-weight: 600; margin: 0; color: #1F2937;">BlackRock</h3>
                        <p style="margin: 0; color: #6B7280; font-size: 0.75rem;">Head of Investment Strategy</p>
                    </div>
                </div>
                <p style="color: #374151; font-style: italic; border-left: 3px solid #10B981; padding-left: 1rem; margin-bottom: 0;">"This development supports our overweight position in U.S. equities and high-quality bonds. We're particularly positive on the tech and real estate sectors."</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[3]:
        # Follow-up Analysis Tab
        st.markdown('<div class="section-title">Ask Follow-up Questions</div>', unsafe_allow_html=True)
        
        # Sample questions
        st.markdown("""
        <div class="card" style="margin-bottom: 1.5rem;">
            <p style="margin: 0 0 0.75rem 0; font-weight: 500; color: #1F2937;">Suggested Questions</p>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem;">
        """, unsafe_allow_html=True)
        
        question_options = [
            {"text": "How might these rate cuts affect mortgage rates?", "icon": "home"},
            {"text": "What sectors typically perform best when rates fall?", "icon": "pie-chart"},
            {"text": "How does this compare to the 2019 rate cuts?", "icon": "trending-down"},
            {"text": "What are the implications for bond investors?", "icon": "briefcase"}
        ]
        
        for idx, q in enumerate(question_options):
            if st.button(
                f"{q['text']}",
                key=f"question_btn_{idx}",
                use_container_width=True
            ):
                st.session_state.chat_query = q['text']
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Custom question input
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 0.75rem;">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            <span style="font-size: 1.125rem; font-weight: 600; color: #1F2937;">Custom Analysis Question</span>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("follow_up_form"):
            user_question = st.text_area(
                "Enter your specific question about this content:",
                placeholder="I'd like to understand how these projected rate cuts might impact...",
                value=st.session_state.chat_query,
                height=100,
                label_visibility="collapsed"
            )
            
            col1, col2 = st.columns([4, 1])
            with col1:
                submitted = st.form_submit_button("📨 Submit Question", use_container_width=True)
            with col2:
                clear = st.form_submit_button("🔄 Clear", use_container_width=True)
                if clear:
                    st.session_state.chat_query = ""
                    st.rerun()
            
            if submitted and user_question:
                st.success("Question submitted! Processing analysis...")
                time.sleep(1)  # Simulate processing time
                
                # Display simulated response
                st.markdown("""
                <div style="margin-top: 1.5rem;">
                    <div class="section-title" style="margin-top: 1rem;">Analysis Response</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.container(border=True):
                    if "mortgage rates" in user_question.lower():
                        st.markdown("""
                        <h3 style="color: #1F2937; font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem;">Impact on Mortgage Rates</h3>
                        
                        <p style="color: #374151; margin-bottom: 1.5rem;">Based on the content analysis, the projected three 25-basis-point Fed rate cuts in 2025 are likely to have the following effects on mortgage rates:</p>
                        
                        <div style="margin-bottom: 1rem; display: flex; gap: 1rem; align-items: flex-start;">
                            <div style="background-color: #3B82F6; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 0.25rem;">1</div>
                            <div>
                                <p style="margin: 0; font-weight: 600; color: #1F2937;">Direct Correlation</p>
                                <p style="margin: 0; color: #374151;">The 30-year fixed mortgage rate is projected to decline from its current 6.5% to potentially below 5.5% by year-end if the Fed delivers all projected cuts.</p>
                            </div>
                        </div>
                        
                        <div style="margin-bottom: 1rem; display: flex; gap: 1rem; align-items: flex-start;">
                            <div style="background-color: #3B82F6; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 0.25rem;">2</div>
                            <div>
                                <p style="margin: 0; font-weight: 600; color: #1F2937;">Housing Affordability</p>
                                <p style="margin: 0; color: #374151;">The analysis indicates housing affordability could improve by approximately 15% with the anticipated rate reductions combined with ongoing wage growth.</p>
                            </div>
                        </div>
                        
                        <div style="margin-bottom: 1rem; display: flex; gap: 1rem; align-items: flex-start;">
                            <div style="background-color: #3B82F6; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 0.25rem;">3</div>
                            <div>
                                <p style="margin: 0; font-weight: 600; color: #1F2937;">Market Timing</p>
                                <p style="margin: 0; color: #374151;">The first rate cut is anticipated around June 2025, so prospective homebuyers might see meaningful mortgage rate reductions beginning in mid-2025.</p>
                            </div>
                        </div>
                        
                        <div style="margin-bottom: 1rem; display: flex; gap: 1rem; align-items: flex-start;">
                            <div style="background-color: #3B82F6; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 0.25rem;">4</div>
                            <div>
                                <p style="margin: 0; font-weight: 600; color: #1F2937;">Historical Context</p>
                                <p style="margin: 0; color: #374151;">Typically, mortgage rates don't fall in perfect parallel with Fed cuts, but there is a strong correlation, especially for adjustable-rate mortgages.</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif "sectors perform" in user_question.lower():
                        st.markdown("""
                        <h3 style="color: #1F2937; font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem;">Sectors That Typically Perform Best When Rates Fall</h3>
                        
                        <p style="color: #374151; margin-bottom: 1.5rem;">According to the analysis and historical patterns, these sectors typically perform best in a falling rate environment:</p>
                        
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                            <div class="blue-card" style="margin-bottom: 0; padding: 1rem; border-radius: 0.5rem;">
                                <h4 style="margin: 0 0 0.5rem 0; color: #1E40AF; font-weight: 600;">Technology</h4>
                                <p style="margin: 0; color: #374151; font-size: 0.875rem;">Tech stocks rallied following the announcement with 2.8% gains. Growth-oriented companies typically benefit from lower discount rates applied to future earnings.</p>
                            </div>
                            
                            <div class="green-card" style="margin-bottom: 0; padding: 1rem; border-radius: 0.5rem;">
                                <h4 style="margin: 0 0 0.5rem 0; color: #065F46; font-weight: 600;">Real Estate</h4>
                                <p style="margin: 0; color: #374151; font-size: 0.875rem;">Both residential and commercial real estate tend to perform well as financing costs decrease. The analysis projects meaningful recovery in the housing market.</p>
                            </div>
                            
                            <div class="purple-card" style="margin-bottom: 0; padding: 1rem; border-radius: 0.5rem;">
                                <h4 style="margin: 0 0 0.5rem 0; color: #5B21B6; font-weight: 600;">Utilities</h4>
                                <p style="margin: 0; color: #374151; font-size: 0.875rem;">The analysis shows utilities gaining attractiveness in a falling rate environment due to their typically higher dividend yields becoming more competitive.</p>
                            </div>
                            
                            <div class="yellow-card" style="margin-bottom: 0; padding: 1rem; border-radius: 0.5rem;">
                                <h4 style="margin: 0 0 0.5rem 0; color: #92400E; font-weight: 600;">Consumer Discretionary</h4>
                                <p style="margin: 0; color: #374151; font-size: 0.875rem;">Lower interest rates typically boost consumer spending through reduced credit costs and increased disposable income.</p>
                            </div>
                        </div>
                        
                        <p style="color: #374151;">The analysis suggests focusing on companies with strong growth prospects but relatively higher debt levels that would benefit from lower financing costs.</p>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <h3 style="color: #1F2937; font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem;">Analysis Based on Your Question</h3>
                        
                        <p style="color: #374151; margin-bottom: 1rem;">Based on the content and financial context provided in the analysis:</p>
                        
                        <div class="card" style="background-color: #F9FAFB; margin-bottom: 1.5rem;">
                            <p style="margin: 0 0 1rem 0; color: #374151;">The Federal Reserve's signaling of three potential rate cuts for 2025 represents a pivotal shift in monetary policy after maintaining high rates since July 2023. This change is supported by improving inflation data (core PCE at 2.4%) and a balanced labor market (unemployment steady at 4.1%).</p>
                            
                            <p style="margin: 0; font-weight: 600; color: #1F2937; margin-bottom: 0.5rem;">Market impacts include:</p>
                            <ul style="margin: 0 0 1rem 0; padding-left: 1.5rem; color: #374151;">
                                <li style="margin-bottom: 0.5rem;">Equity markets responded positively (S&P 500 +1.8%, NASDAQ +2.4%)</li>
                                <li style="margin-bottom: 0.5rem;">10-year Treasury yield fell to 3.95%</li>
                                <li>Financial sector may face some near-term margin pressure but benefit from increased lending</li>
                            </ul>
                            
                            <p style="margin: 0; color: #374151;">The timing aligns with economic data showing inflation trending toward the Fed's 2% target while maintaining good labor market conditions. This balanced approach aims to support continued economic expansion while ensuring inflation remains controlled.</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>ParselyFi Financial Content Analysis | Generated on {}</p>
        <p style="margin-top: 0.5rem;">Data sources include financial news, expert analysis, and market data</p>
        <p style="margin-top: 0.5rem; font-size: 0.75rem;">This report is for informational purposes only and not financial advice</p>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)

# Sample data for video analysis
video_data = {
    "title": "Fed Chair Powell's Latest Interest Rate Decision - Economic Outlook 2025",
    "channel": "EconInsights",
    "date": "March 18, 2025",
    "duration": "18:35",
    "views": "156,847",
    "url": "https://www.youtube.com/watch?v=example1234",
    "thumbnail": "https://via.placeholder.com/640x360",
    "summary": "This video features a detailed analysis of Federal Reserve Chair Jerome Powell's latest press conference following the March 2025 FOMC meeting. The Fed signaled plans for three potential interest rate cuts in 2025, citing improving inflation trends and balanced economic growth. The analyst provides context around the decision, examines market reactions, and discusses implications for various sectors including housing, technology, and banking. The overall outlook is cautiously optimistic, with emphasis on the Fed's data-dependent approach going forward.",
    "keyInsights": [
        "Federal Reserve signals three potential rate cuts for 2025",
        "Inflation trending toward 2% target with core PCE at 2.4%",
        "Labor market showing balanced growth with unemployment steady at 4.1%",
        "Housing market expected to benefit from potential rate cuts",
        "Tech sector rallied following the announcement with 2.8% gains"
    ],
    "quotedStatements": [
        {
            "speaker": "Powell (via analyst)",
            "quote": "The committee sees the risks to achieving its employment and inflation goals as moving into better balance.",
            "timestamp": "3:45"
        },
        {
            "speaker": "Analyst",
            "quote": "This signals a significant pivot from the Fed's previous hawkish stance maintained throughout 2024.",
            "timestamp": "4:18"
        },
        {
            "speaker": "Powell (via analyst)",
            "quote": "We need to see continued good data before we can begin the process of reducing our policy rate.",
            "timestamp": "7:22"
        },
        {
            "speaker": "Analyst",
            "quote": "The housing market could see a meaningful recovery as mortgage rates begin to decline from their multi-decade highs.",
            "timestamp": "12:45"
        }
    ],
    "keyTopics": [
        {
            "name": "Interest Rate Outlook",
            "description": "Analysis of the Federal Reserve's projection for three rate cuts in 2025 and the conditions that would support this path.",
            "quotes": [
                { "text": "The dot plot showed a median projection of three 25 basis point cuts for 2025, with the first potentially occurring in June.", "timestamp": "5:12" },
                { "text": "There's still a significant dispersion in the dot plot, showing some committee members prefer only two cuts while others project four.", "timestamp": "6:05" }
            ]
        },
        {
            "name": "Inflation Trends",
            "description": "Current inflation data and the Fed's assessment of progress toward the 2% target.",
            "quotes": [
                { "text": "Core PCE has declined from 2.8% at the end of 2024 to the current 2.4%, showing meaningful progress toward the 2% target.", "timestamp": "8:30" },
                { "text": "Powell emphasized that services inflation remains somewhat elevated, requiring continued vigilance.", "timestamp": "9:15" }
            ]
        },
        {
            "name": "Labor Market Conditions",
            "description": "Analysis of current employment trends and how they influence the Fed's policy stance.",
            "quotes": [
                { "text": "The unemployment rate has remained steady at 4.1% for three consecutive months, indicating a resilient but not overheating labor market.", "timestamp": "10:40" },
                { "text": "Wage growth has moderated to 3.8% year-over-year, which is more consistent with the Fed's 2% inflation target.", "timestamp": "11:22" }
            ]
        },
        {
            "name": "Housing Market Impact",
            "description": "How potential rate cuts may affect mortgage rates and housing market activity.",
            "quotes": [
                { "text": "The 30-year fixed mortgage rate could decline from its current 6.5% to potentially below 5.5% by year-end if the Fed delivers on all projected cuts.", "timestamp": "13:05" },
                { "text": "Housing affordability could improve by approximately 15% with the anticipated rate reductions and ongoing wage growth.", "timestamp": "14:30" }
            ]
        }
    ],
    "transcriptHighlights": [
        { "timestamp": "0:45", "text": "Today we'll be analyzing the March 2025 FOMC meeting and Chair Powell's press conference from earlier this week." },
        { "timestamp": "2:15", "text": "The Federal Reserve kept rates unchanged at 5.25-5.50%, maintaining this level since July 2023." },
        { "timestamp": "3:45", "text": "Powell stated: 'The committee sees the risks to achieving its employment and inflation goals as moving into better balance.'" },
        { "timestamp": "5:12", "text": "The updated dot plot revealed a median projection of three 25 basis point cuts during 2025." },
        { "timestamp": "8:30", "text": "Core PCE inflation has declined to 2.4%, showing steady progress toward the Fed's 2% target." },
        { "timestamp": "10:40", "text": "The labor market has maintained a healthy 4.1% unemployment rate for three consecutive months." },
        { "timestamp": "13:05", "text": "Mortgage rates are projected to decline in response to the anticipated Fed cuts, potentially falling below 5.5% by year-end." },
        { "timestamp": "15:30", "text": "The tech sector has already responded positively to the Fed's signals, with the Nasdaq gaining 2.8% following the announcement." }
    ],
    "economicContext": {
        "events": [
            { "name": "Q4 2024 GDP Report", "date": "January 30, 2025", "description": "Showed 2.3% annualized growth, slightly above expectations." },
            { "name": "February 2025 CPI Report", "date": "March 12, 2025", "description": "Inflation at 2.6% year-over-year, continuing downward trend." },
            { "name": "March FOMC Meeting", "date": "March 15, 2025", "description": "Fed maintained current rates but signaled future cuts." }
        ],
        "policies": [
            "The Fed's dual mandate focus on both price stability and maximum employment",
            "Gradual transition from tightening to easing monetary policy",
            "Data-dependent approach with emphasis on incoming inflation and employment reports"
        ],
        "indicators": [
            { "name": "Federal Funds Rate", "current": "5.25-5.50%", "trend": "Stable, projected to decrease" },
            { "name": "Core PCE Inflation", "current": "2.4%", "trend": "Declining toward target" },
            { "name": "Unemployment Rate", "current": "4.1%", "trend": "Stable" },
            { "name": "30-Year Mortgage Rate", "current": "6.5%", "trend": "Projected to decline" },
            { "name": "10-Year Treasury Yield", "current": "3.95%", "trend": "Declining" }
        ]
    },
    "significance": "This video analyzes a pivotal moment in the Fed's monetary policy cycle, marking the potential end of a prolonged high-rate environment that has significantly impacted borrowing costs across the economy. The analysis is particularly valuable for investors, homebuyers, and businesses making capital expenditure decisions, as it provides detailed projections for interest rate movements that will influence financial markets throughout 2025. The timing is especially relevant as it comes amid improving inflation data that has given the Fed more flexibility after a restrictive policy stance throughout 2023-2024."
}

# Sample data for article analysis
article_data = {
    "title": "Fed Signals Three Rate Cuts for 2025, Markets Rally",
    "source": "Financial Times",
    "author": "Sarah Johnson",
    "date": "March 19, 2025",
    "readingTime": "5 min read",
    "url": "https://www.financialtimes.com/markets/2025/03/19/fed-signals-three-rate-cuts-for-2025-markets-rally",
    "thumbnail": "https://via.placeholder.com/640x360",
    "summary": "This Financial Times article reports on the Federal Reserve's recent announcement indicating plans for three interest rate cuts during 2025. The Fed's decision is based on improving inflation data that shows a consistent decline toward the 2% target, alongside normalized economic conditions. Markets responded positively to the announcement, with the S&P 500 gaining 1.8% and the 10-year Treasury yield falling to 3.85%. The technology sector showed the strongest positive reaction with a 2.4% gain, while utilities and consumer discretionary sectors also performed well. Analysts generally view the announcement as positive for economic growth prospects while maintaining inflation control.",
    "keyPoints": [
        "Fed projects three 25-basis-point rate cuts for 2025",
        "Inflation data shows consistent decline toward 2% target",
        "S&P 500 gained 1.8% following the announcement",
        "10-year Treasury yield fell to 3.85%",
        "Consumer spending remains robust despite higher rate environment"
    ],
    "expertOpinions": [
        {
            "name": "Dr. Michael Chen",
            "title": "Chief Economist, Global Investment Bank",
            "quote": "The Fed's clear signaling helps reduce market uncertainty and should support continued economic expansion in 2025.",
        },
        {
            "name": "Amanda Rodriguez",
            "title": "Portfolio Manager, Capital Investments",
            "quote": "This gives growth stocks room to run, but we're still cautious about highly leveraged companies.",
        },
        {
            "name": "James Wilson",
            "title": "Senior Market Strategist, Atlantic Securities",
            "quote": "The pace of cuts is measured enough to avoid reigniting inflation while supporting economic momentum.",
        }
    ]
}

# Sample chat history data
video_chat_history = [
    {
        "type": "user",
        "content": "https://www.youtube.com/watch?v=example1234",
        "timestamp": "10:32 AM"
    },
    {
        "type": "ai",
        "content": "I've analyzed the video about Fed Chair Powell's latest interest rate decision. The Fed is signaling three potential rate cuts for 2025, with inflation trending toward the 2% target and a stable labor market. The analysis discusses implications for various sectors, particularly the housing market which could benefit from lower mortgage rates. Would you like me to provide more specific details about any aspect of this analysis?",
        "timestamp": "10:33 AM",
        "actions": [
            "Transcribed video content",
            "Extracted key financial metrics",
            "Performed sentiment analysis",
            "Identified key statements and timestamps",
            "Analyzed economic context and significance"
        ]
    }
]

article_chat_history = [
    {
        "type": "user",
        "content": "https://www.financialtimes.com/markets/2025/03/19/fed-signals-three-rate-cuts-for-2025-markets-rally",
        "timestamp": "11:45 AM"
    },
    {
        "type": "ai",
        "content": "I've analyzed the Financial Times article about the Federal Reserve's plan for three rate cuts in 2025. The Fed is responding to improving inflation data, which is moving closer to the 2% target. Markets reacted positively with the S&P 500 gaining 1.8% and the 10-year Treasury yield falling to 3.85%. The technology sector showed the strongest positive reaction with a 2.4% gain.",
        "timestamp": "11:46 AM",
        "actions": [
            "Extracted key points from article",
            "Analyzed market sector impacts",
            "Compiled expert opinions",
            "Performed sentiment analysis"
        ]
    }
]

if __name__ == "__main__":
    main()