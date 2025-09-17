import streamlit as st
import pandas as pd
from datetime import datetime
import re
import json

def render_document_report_page(document_report_json=None):
    """Render the document report page with inline editing capabilities"""

    # Configure page settings
    st.set_page_config(
        page_title="Financial Document Analysis",
        page_icon="📄",
        layout="wide"
    )

    # Apply custom CSS for styling including edit mode indicators
    apply_custom_css()
    
    # Initialize document data in session state if not already present
    if 'document_data' not in st.session_state:
        if not document_report_json:
            # Default document metadata if none provided
            document_data = {
                "file_name": "Q1_2025_Financial_Report.pdf",
                "page_count": 42,
                "title": "Market Trends and Economic Outlook",
                "title_summary": "Q1 2025 Financial Analysis and Investment Opportunities",
                "concept_theme_hashtags": ["#MarketOutlook", "#EconomicTrends", "#InterestRates", "#InvestmentStrategy", "#Q1Report"],
                "date_published": "March 15, 2025",
                "source": "Global Investment Advisors",
                "confidence": "High",
                "document_summary": "This comprehensive report analyzes current market conditions following the Federal Reserve's recent signals for three potential rate cuts in 2025. It examines how improved inflation data and stable employment statistics are influencing various market sectors including technology, real estate, and financials. The document provides detailed investment strategies tailored for the anticipated lower interest rate environment while highlighting potential risks and opportunities across asset classes.",
                "key_insights": [
                    "Rate cuts expected to positively impact growth stocks and interest-rate sensitive sectors",
                    "Housing market projected to see significant recovery as mortgage rates decline",
                    "Technology sector positioned for strong performance with 15-20% growth potential",
                    "Financial stocks face mixed outlook with potential margin pressure offset by loan volume",
                    "Global markets show varied responses with emerging markets potentially outperforming"
                ],
                "key_topics": [
                    {
                        "name": "Interest Rate Outlook",
                        "description": "Analysis of the Federal Reserve's projection for three rate cuts in 2025 and the conditions that would support this path.",
                        "relevance": "High",
                        "sentiment": "Positive",
                        "analysis": "The Fed's pivot suggests increasing confidence in controlled inflation with minimal economic disruption."
                    },
                    {
                        "name": "Sector Performance",
                        "description": "Detailed breakdown of how different market sectors are likely to respond to changing monetary policy.",
                        "relevance": "High",
                        "sentiment": "Mixed",
                        "analysis": "Technology and consumer discretionary sectors show strongest potential while financials face headwinds."
                    },
                    {
                        "name": "Investment Strategies",
                        "description": "Specific investment approaches recommended for the anticipated economic environment.",
                        "relevance": "High",
                        "sentiment": "Positive",
                        "analysis": "A balanced approach favoring growth-oriented equities with some defensive positioning is recommended."
                    },
                    {
                        "name": "Risk Assessment",
                        "description": "Evaluation of potential risks that could derail the projected market trajectory.",
                        "relevance": "Medium",
                        "sentiment": "Cautionary",
                        "analysis": "Inflation reacceleration remains the primary risk, with geopolitical tensions as a secondary concern."
                    }
                ],
                "quoted_statements": [
                    {
                        "speaker": "Chief Economist",
                        "quote": "The data increasingly supports a soft landing scenario with normalized inflation and sustained economic growth.",
                        "page": 8
                    },
                    {
                        "speaker": "Head of Investment Strategy",
                        "quote": "We recommend a balanced approach with growth-oriented equities and rate-sensitive assets like REITs.",
                        "page": 15
                    },
                    {
                        "speaker": "Market Analyst",
                        "quote": "Technology and consumer discretionary sectors stand to benefit most from the improving interest rate outlook.",
                        "page": 22
                    },
                    {
                        "speaker": "Risk Manager",
                        "quote": "Investors should remain vigilant about inflation reacceleration as the primary risk to our base case scenario.",
                        "page": 29
                    }
                ],
                "content_excerpt": """
    # Market Trends and Economic Outlook

    ## Executive Summary

    Recent Federal Reserve communications indicate a significant shift in monetary policy guidance, with projections now showing three potential 25 basis point rate cuts during 2025. This represents a meaningful pivot from the restrictive stance maintained throughout 2023-2024 and has important implications for market dynamics and investment strategies.

    The data increasingly supports a "soft landing" scenario where inflation continues to normalize toward the 2% target while economic growth remains resilient. Core PCE inflation has declined to 2.4%, showing steady progress, while the labor market maintains a healthy 4.1% unemployment rate with moderating wage pressures.

    Market reactions have been predominantly positive, with the S&P 500 gaining 1.8% following the announcements and the 10-year Treasury yield declining to 3.95%. Sector performance has shown notable dispersion, with technology stocks rallying 2.8% while financial stocks demonstrated more modest gains of 1.5% amid concerns about net interest margin compression.

    This report examines the implications of the changing interest rate environment across asset classes and provides specific investment recommendations tailored to the anticipated conditions. While our base case outlook is constructive, we highlight several risk factors that could alter the trajectory and suggest appropriate risk management strategies.

    ## Interest Rate Outlook

    The Federal Reserve's "dot plot" now shows a median projection of three 25 basis point cuts for 2025, with the first potentially occurring in June. This represents a notable shift from previous guidance and reflects increasing confidence that inflation is on a sustainable path toward the 2% target.

    Several factors support this evolving outlook:
    """
            }
        else:
            document_data = document_report_json
        
        # Save to session state
        st.session_state.document_data = document_data
        st.session_state.editing_mode = False
    
    # Toggle for editing mode
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.session_state.editing_mode = st.toggle("Enable Editing Mode", value=st.session_state.get('editing_mode', False))
        
        if st.session_state.editing_mode:
            st.info("📝 Editing mode is enabled. Click on any text to edit. Changes are saved automatically.")
            
            # Add an export button when in edit mode
            if st.button("Export Edited Document"):
                export_data = st.session_state.document_data
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name="edited_document_report.json",
                    mime="application/json"
                )

    # Render the document report header
    render_report_header(st.session_state.document_data)

    # Create main page tabs
    tabs = st.tabs([
        "📄 Document Overview",
        "🔍 Key Topics & Insights",
        "📝 Full Content"
    ])

    # Document Overview Tab
    with tabs[0]:
        render_document_overview(st.session_state.document_data)

    # Key Topics & Insights Tab
    with tabs[1]:
        render_key_topics_insights(st.session_state.document_data)

    # Full Content Tab
    with tabs[2]:
        render_full_content(st.session_state.document_data)

    # Render footer
    render_footer()


def apply_custom_css():
    """Apply custom CSS styling with additional styles for editable content"""
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
        .document-title {
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
        .red-tag {
            background-color: #FEF2F2;
            color: #B91C1C;
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

        /* Document metadata */
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
        .red-card {
            border-left: 4px solid #EF4444;
            background-color: #FEF2F2;
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

        /* Insights and Key Points */
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

        /* Footer */
        .footer {
            margin-top: 2rem;
            padding-top: 1rem;
            text-align: center;
            font-size: 0.875rem;
            color: #6B7280;
            border-top: 1px solid #E5E7EB;
        }

        /* Content display */
        .document-content {
            background-color: #F9FAFB;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid #E5E7EB;
            font-family: monospace;
            white-space: pre-wrap;
            line-height: 1.6;
            font-size: 0.9rem;
            color: #1F2937;
            overflow-x: auto;
        }
        
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
        
        /* Tailwind-like transition */
        * {
            transition-property: background-color, border-color, color, fill, stroke;
            transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
            transition-duration: 150ms;
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
        
        /* Streamlit components */
        div[data-testid="stVerticalBlock"] {
            gap: 0.5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

def render_report_header(document_data):
    """Render document report header with metadata and inline editing capability"""
    # Create container for editable title
    if st.session_state.editing_mode:
        # In edit mode, use text inputs
        new_title = st.text_input(
            "Document Title", 
            value=document_data["title"], 
            key="edit_title",
            on_change=update_document_data,
            args=("title",)
        )
        document_data["title"] = new_title
        
        new_summary = st.text_input(
            "Document Summary", 
            value=document_data["title_summary"], 
            key="edit_title_summary",
            on_change=update_document_data,
            args=("title_summary",)
        )
        document_data["title_summary"] = new_summary
        
        # Editable tags
        st.markdown('<div class="section-title" style="font-size: 1rem;">Document Tags</div>', unsafe_allow_html=True)
        tags_container = st.container()
        
        with tags_container:
            # Display current tags with delete button
            tag_cols = st.columns(5)
            for i, tag in enumerate(document_data["concept_theme_hashtags"]):
                with tag_cols[i % 5]:
                    new_tag = st.text_input(
                        f"Tag {i+1}", 
                        value=tag, 
                        key=f"edit_tag_{i}",
                        on_change=update_document_tag,
                        args=(i,)
                    )
            
            # Add new tag button
            if st.button("+ Add Tag", key="add_tag_btn"):
                document_data["concept_theme_hashtags"].append("#NewTag")
                st.session_state.document_data = document_data
                st.rerun()
                
        # Editable metadata
        meta_cols = st.columns(4)
        with meta_cols[0]:
            new_file = st.text_input(
                "File Name", 
                value=document_data["file_name"], 
                key="edit_file_name",
                on_change=update_document_data,
                args=("file_name",)
            )
            document_data["file_name"] = new_file
        
        with meta_cols[1]:
            new_source = st.text_input(
                "Source", 
                value=document_data["source"], 
                key="edit_source",
                on_change=update_document_data,
                args=("source",)
            )
            document_data["source"] = new_source
            
        with meta_cols[2]:
            new_date = st.text_input(
                "Date Published", 
                value=document_data["date_published"], 
                key="edit_date",
                on_change=update_document_data,
                args=("date_published",)
            )
            document_data["date_published"] = new_date
            
        with meta_cols[3]:
            confidence_options = ["High", "Medium", "Low"]
            new_confidence = st.selectbox(
                "Confidence", 
                options=confidence_options,
                index=confidence_options.index(document_data["confidence"]),
                key="edit_confidence",
                on_change=update_document_data,
                args=("confidence",)
            )
            document_data["confidence"] = new_confidence
    else:
        # In view mode, use HTML
        header_html = '<div class="report-header">'

        # Title and subtitle section
        header_html += '<div style="display: flex; flex-direction: row; margin-bottom: 10px;">'

        # Left column (title and subtitle) - approximately 60% width
        header_html += '<div style="flex: 3;">'
        header_html += f'<div class="document-title editable-text">{document_data["title"]}</div>'
        header_html += f'<div class="report-subtitle editable-text">{document_data["title_summary"]}</div>'
        header_html += '</div>'  # Close left column

        # Right column (metadata) - approximately 40% width
        header_html += '<div style="flex: 2;">'
        header_html += '<div class="metadata">'
        header_html += f'<div>File: <span class="metadata-highlight editable-text">{document_data["file_name"]}</span></div>'
        header_html += f'<div>Source: <span class="metadata-highlight editable-text">{document_data["source"]}</span></div>'
        header_html += f'<div>Published: <span class="editable-text">{document_data["date_published"]}</span></div>'
        header_html += f'<div>Confidence: <span class="high-confidence editable-text">{document_data["confidence"]}</span></div>'
        header_html += '</div>'  # Close metadata div
        header_html += '</div>'  # Close right column

        header_html += '</div>'  # Close title and metadata row

        # Tags section - placed below the title/metadata and spread across full width
        header_html += '<div class="tag-container">'
        tag_colors = ["blue-tag", "green-tag", "purple-tag", "yellow-tag", "red-tag"]
        for i, tag in enumerate(document_data["concept_theme_hashtags"]):
            tag_class = tag_colors[i % len(tag_colors)]
            header_html += f'<span class="tag {tag_class} editable-text">{tag}</span>'
        header_html += '</div>'  # Close tag container

        header_html += '</div>'  # Close report-header

        # Render the complete header HTML
        st.markdown(header_html, unsafe_allow_html=True)

def render_document_overview(document_data):
    """Render document overview with summary and insights with inline editing"""
    # Document Summary Section
    st.markdown('<div class="section-title">Document Summary</div>', unsafe_allow_html=True)
    
    if st.session_state.editing_mode:
        # Show editable text area for summary
        new_summary = st.text_area(
            "Edit document summary", 
            value=document_data["document_summary"],
            height=150,
            key="edit_doc_summary",
            on_change=update_document_data,
            args=("document_summary",)
        )
        document_data["document_summary"] = new_summary
    else:
        # Show read-only summary with editable styling
        st.markdown(f'<div class="card editable-text">{document_data["document_summary"]}</div>', unsafe_allow_html=True)

    # Key Insights Section
    st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)

    if st.session_state.editing_mode:
        # Generate a column for each insight with edit capabilities
        insights_container = st.container()
        with insights_container:
            for i, insight in enumerate(document_data["key_insights"]):
                cols = st.columns([5, 1])
                with cols[0]:
                    new_insight = st.text_input(
                        f"Insight {i+1}", 
                        value=insight,
                        key=f"edit_insight_{i}",
                        on_change=update_document_insight,
                        args=(i,)
                    )
                with cols[1]:
                    if st.button("🗑️", key=f"delete_insight_{i}"):
                        document_data["key_insights"].pop(i)
                        st.session_state.document_data = document_data
                        st.rerun()
            
            # Add new insight button
            if st.button("+ Add Insight", key="add_insight_btn"):
                document_data["key_insights"].append("New insight")
                st.session_state.document_data = document_data
                st.rerun()
    else:
        # Display insights in read-only mode with editable styling
        insights_cols = st.columns(2)
        for i, insight in enumerate(document_data["key_insights"]):
            with insights_cols[i % 2]:
                st.markdown(f"""
                <div class="card">
                    <div class="key-insight">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 10px; flex-shrink: 0; margin-top: 2px;">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        <span class="editable-text" style="color: #374151;">{insight}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Key Statements Section
    st.markdown('<div class="section-title">Key Statements</div>', unsafe_allow_html=True)

    if st.session_state.editing_mode:
        # Generate editable fields for each statement
        for i, statement in enumerate(document_data["quoted_statements"]):
            cols = st.columns([3, 2, 1, 1])
            with cols[0]:
                new_quote = st.text_area(
                    f"Quote {i+1}", 
                    value=statement["quote"],
                    key=f"edit_quote_{i}",
                    height=80,
                    on_change=update_document_statement,
                    args=(i, "quote")
                )
                statement["quote"] = new_quote
            
            with cols[1]:
                new_speaker = st.text_input(
                    "Speaker", 
                    value=statement["speaker"],
                    key=f"edit_speaker_{i}",
                    on_change=update_document_statement,
                    args=(i, "speaker")
                )
                statement["speaker"] = new_speaker
                
            with cols[2]:
                new_page = st.number_input(
                    "Page", 
                    value=int(statement["page"]),
                    key=f"edit_page_{i}",
                    min_value=1,
                    on_change=update_document_statement,
                    args=(i, "page")
                )
                statement["page"] = new_page
                
            with cols[3]:
                if st.button("🗑️", key=f"delete_statement_{i}"):
                    document_data["quoted_statements"].pop(i)
                    st.session_state.document_data = document_data
                    st.rerun()
        
        # Add new statement button
        if st.button("+ Add Statement", key="add_statement_btn"):
            document_data["quoted_statements"].append({
                "speaker": "New Speaker",
                "quote": "New quote text",
                "page": 1
            })
            st.session_state.document_data = document_data
            st.rerun()
    else:
        # Display statements in read-only mode with editable styling
        for statement in document_data["quoted_statements"]:
            st.markdown(f"""
            <div class="key-statement">
                <div class="quote editable-text">"{statement['quote']}"</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="speaker editable-text">— {statement['speaker']}</span>
                    <span class="timestamp editable-text">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 4px;">
                            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                            <line x1="16" y1="2" x2="16" y2="6"></line>
                            <line x1="8" y1="2" x2="8" y2="6"></line>
                            <line x1="3" y1="10" x2="21" y2="10"></line>
                        </svg>
                        Page {statement['page']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_key_topics_insights(document_data):
    """Render key topics and insights with detailed analysis and inline editing"""
    # Key Topics Section
    st.markdown('<div class="section-title">Key Topics</div>', unsafe_allow_html=True)

    if st.session_state.editing_mode:
        # Create editable fields for each topic
        for i, topic in enumerate(document_data["key_topics"]):
            with st.expander(f"{topic['name']} (Click to edit)", expanded=False):
                # Topic name
                new_name = st.text_input(
                    "Topic Name", 
                    value=topic["name"],
                    key=f"edit_topic_name_{i}",
                    on_change=update_document_topic,
                    args=(i, "name")
                )
                topic["name"] = new_name
                
                # Topic description
                new_description = st.text_area(
                    "Description", 
                    value=topic["description"],
                    key=f"edit_topic_desc_{i}",
                    height=100,
                    on_change=update_document_topic,
                    args=(i, "description")
                )
                topic["description"] = new_description
                
                # Topic metadata
                cols = st.columns(2)
                with cols[0]:
                    relevance_options = ["High", "Medium", "Low"]
                    new_relevance = st.selectbox(
                        "Relevance", 
                        options=relevance_options,
                        index=relevance_options.index(topic["relevance"]) if topic["relevance"] in relevance_options else 0,
                        key=f"edit_topic_relevance_{i}",
                        on_change=update_document_topic,
                        args=(i, "relevance")
                    )
                    topic["relevance"] = new_relevance
                
                with cols[1]:
                    sentiment_options = ["Positive", "Neutral", "Mixed", "Cautionary", "Negative"]
                    new_sentiment = st.selectbox(
                        "Sentiment", 
                        options=sentiment_options,
                        index=sentiment_options.index(topic["sentiment"]) if topic["sentiment"] in sentiment_options else 0,
                        key=f"edit_topic_sentiment_{i}",
                        on_change=update_document_topic,
                        args=(i, "sentiment")
                    )
                    topic["sentiment"] = new_sentiment
                
                # Topic analysis
                new_analysis = st.text_area(
                    "Analysis", 
                    value=topic.get("analysis", ""),
                    key=f"edit_topic_analysis_{i}",
                    height=150,
                    on_change=update_document_topic,
                    args=(i, "analysis")
                )
                topic["analysis"] = new_analysis
                
                # Delete topic button
                if st.button("Delete Topic", key=f"delete_topic_{i}"):
                    document_data["key_topics"].pop(i)
                    st.session_state.document_data = document_data
                    st.rerun()
        
        # Add new topic button
        if st.button("+ Add Topic", key="add_topic_btn"):
            document_data["key_topics"].append({
                "name": "New Topic",
                "description": "Enter description here",
                "relevance": "Medium",
                "sentiment": "Neutral",
                "analysis": "Enter analysis here"
            })
            st.session_state.document_data = document_data
            st.rerun()
    else:
        # Display topics in read-only mode with editable styling
        for topic in document_data["key_topics"]:
            with st.expander(topic["name"], expanded=False):
                # Determine status chip color based on sentiment
                status_class = "neutral-status"
                if topic["sentiment"].lower() == "positive":
                    status_class = "positive-status"
                elif topic["sentiment"].lower() == "cautionary" or topic["sentiment"].lower() == "warning":
                    status_class = "warning-status"
                elif topic["sentiment"].lower() == "negative":
                    status_class = "negative-status"

                # Topic header
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <span class="status-chip neutral-status editable-text">Relevance: {topic["relevance"]}</span>
                    <span class="status-chip {status_class} editable-text">{topic["sentiment"]}</span>
                </div>
                """, unsafe_allow_html=True)

                # Topic description
                st.markdown(f"<p class='editable-text' style='color: #374151; margin-bottom: 1rem;'>{topic['description']}</p>", unsafe_allow_html=True)

                # Analysis
                if "analysis" in topic:
                    st.markdown(f'<div class="blue-card card editable-text" style="margin-top: 1rem;"><h3 style="font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem;">Analysis</h3>{topic["analysis"]}</div>', unsafe_allow_html=True)

def render_full_content(document_data):
    """Renders the full document content with inline editing capability"""
    st.markdown('<div class="section-title">Document Content</div>', unsafe_allow_html=True)

    # Create functionality to highlight sections using pills (multi-select)
    highlight_options = [topic["name"] for topic in document_data["key_topics"]]
    highlight_selection = st.pills("Highlight sections related to:", highlight_options, selection_mode="multi")

    # Create a text search box
    search_term = st.text_input("Search within document:", placeholder="Enter search term...")

    # In edit mode, provide a full text area for editing the content
    if st.session_state.editing_mode:
        new_content = st.text_area(
            "Edit full content", 
            value=document_data["content_excerpt"],
            height=500,
            key="edit_full_content",
            on_change=update_document_data,
            args=("content_excerpt",)
        )
        document_data["content_excerpt"] = new_content
        
        # Show preview of formatted content
        st.markdown("### Preview (with formatting)")
        content = document_data["content_excerpt"]
        
        # Convert markdown headers to streamlit markdown format
        content = content.replace("# ", "### ").replace("## ", "#### ")
        
        # Display the formatted content
        st.markdown(content)
    else:
        content = document_data["content_excerpt"]

        # Apply highlights based on pills selection and search term
        if highlight_selection:
            for selected_topic in highlight_selection:
                if selected_topic == "Interest Rate Outlook":
                    content = content.replace(
                        "Federal Reserve's \"dot plot\" now shows a median projection of three 25 basis point cuts",
                        ":blue-background[Federal Reserve's \"dot plot\" now shows a median projection of three 25 basis point cuts]"
                    )
                elif selected_topic == "Sector Performance":
                    content = content.replace(
                        "Sector performance has shown notable dispersion",
                        ":blue-background[Sector performance has shown notable dispersion]"
                    )
                elif selected_topic == "Investment Strategies":
                    content = content.replace(
                        "provides specific investment recommendations",
                        ":blue-background[provides specific investment recommendations]"
                    )
                elif selected_topic == "Risk Assessment":
                    content = content.replace(
                        "we highlight several risk factors",
                        ":blue-background[we highlight several risk factors]"
                    )

        if search_term and search_term.strip():
            def highlight_search_term(match):
                return f":yellow-background[{match.group(0)}]"

            pattern = re.compile(re.escape(search_term), re.IGNORECASE)
            content = pattern.sub(highlight_search_term, content)

        # Convert markdown headers to streamlit markdown format
        content = content.replace("# ", "### ").replace("## ", "#### ")

        # Add page numbers using markdown
        content = content.replace(
            "### Market Trends",
            f":grey[Page 1]\n### Market Trends"
        )
        content = content.replace(
            "#### Executive Summary",
            f":grey[Page 1]\n#### Executive Summary"
        )
        content = content.replace(
            "#### Interest Rate Outlook",
            f":grey[Page 5]\n#### Interest Rate Outlook"
        )

        # Display the content using st.markdown with the editable-text class
        st.markdown(f"""
        <div class="document-content editable-text">
        {content}
        </div>
        """, unsafe_allow_html=True)

    # Create a download button for the document
    st.markdown("<div style='text-align: center; margin-top: 1.5rem;'>", unsafe_allow_html=True)

    # Create a download button
    st.download_button(
        label="📥 Download Full Document",
        data=document_data["content_excerpt"],
        file_name=document_data["file_name"].replace(".pdf", ".txt"),
        mime="text/plain"
    )

    st.markdown("</div>", unsafe_allow_html=True)

def render_footer():
    """Render a standardized footer"""
    st.markdown("""
    <div class="footer">
        <p>Financial Document Analysis | Generated on {}</p>
        <p style="margin-top: 0.5rem;">Source: Global Investment Advisors</p>
        <p style="margin-top: 0.5rem; font-size: 0.75rem;">This analysis is for informational purposes only and not financial advice</p>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)

# Helper functions for updating the document data in session state
def update_document_data(field_name):
    """Update a simple field in the document data"""
    key = f"edit_{field_name}"
    if key in st.session_state:
        st.session_state.document_data[field_name] = st.session_state[key]

def update_document_tag(index):
    """Update a tag in the concept_theme_hashtags array"""
    key = f"edit_tag_{index}"
    if key in st.session_state:
        st.session_state.document_data["concept_theme_hashtags"][index] = st.session_state[key]

def update_document_insight(index):
    """Update an insight in the key_insights array"""
    key = f"edit_insight_{index}"
    if key in st.session_state:
        st.session_state.document_data["key_insights"][index] = st.session_state[key]

def update_document_statement(index, field):
    """Update a field in a quoted statement"""
    key = f"edit_{field}_{index}"
    if key in st.session_state:
        st.session_state.document_data["quoted_statements"][index][field] = st.session_state[key]

def update_document_topic(index, field):
    """Update a field in a key topic"""
    key = f"edit_topic_{field}_{index}"
    if key in st.session_state:
        st.session_state.document_data["key_topics"][index][field] = st.session_state[key]

# Main function to render the document report page
if __name__ == "__main__":
    try:
        # Try to load from a file first
        with open('document_report.json', 'r') as f:
            document_report_json = json.load(f)
        render_document_report_page(document_report_json)
    except FileNotFoundError:
        # If file not found, use the default data
        render_document_report_page()