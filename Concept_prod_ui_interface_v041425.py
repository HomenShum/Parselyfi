import streamlit as st
from streamlit_anywidget import anywidget
from anywidget import AnyWidget
import traitlets
import time
from datetime import datetime
import json
# -----------------------------------------------------------------------------
# 1. Configuration and Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide", 
    page_title="ParselyFi Report", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CSS Styles - Updated for better visual hierarchy and readability
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* --- ParselyFi Theme & Global Styles --- */
    :root {
        --primary-color: #376f4e; /* Plant Green */
        --primary-light: rgba(55, 111, 78, 0.1);
        --secondary-color: #69380a; /* Coffee Brown */
        --background-color: #fcf9f8;
        --sidebar-bg: #f1e9dd;
        --text-color: #4d2907;
        --border-color: rgba(105, 56, 10, 0.15);
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.1);
        --border-radius: 6px;
        --accent-blue: #3a7ca5;
        --accent-red: #c73e1d;
        --accent-orange: #f58a07;
        --accent-yellow: #ffca28;
        --font-main: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    /* Base Resets */
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: var(--font-main);
        background-color: var(--background-color);
        color: var(--text-color);
        line-height: 1.6;
    }
    
    /* Streamlit Element Overrides */
    /* Make buttons consistent with our design */
    .stButton > button {
        background-color: #f1f1f1;
        border: 1px solid #ddd;
        border-radius: var(--border-radius);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #e0e0e0;
        border-color: #ccc;
    }
    /* Remove padding from containers */
    .block-container {
        padding-top: 0;
        padding-left: 0;
        padding-right: 0;
    }
    
    /* Layout Containers */
    .main-content-wrapper {
        padding: 1.5rem;
        background-color: var(--background-color);
        min-height: 90vh;
        position: relative;
    }
    .sidebar-wrapper {
        padding: 1.5rem;
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
        min-height: 90vh;
    }
    .right-sidebar-wrapper {
        padding: 1.5rem;
        background-color: var(--sidebar-bg);
        border-left: 1px solid var(--border-color);
        min-height: 90vh;
    }

    /* Sidebar Components */
    .sidebar-section { 
        margin-bottom: 1.8rem; 
        position: relative;
    }
    .sidebar-section h4 {
        margin-bottom: 0.8rem;
        color: var(--primary-color);
        font-size: 1.1rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    /* User Info */
    .user-info { 
        display: flex; 
        align-items: center; 
        gap: 10px; 
        padding: 12px; 
        background: rgba(255, 255, 255, 0.5); 
        border-radius: var(--border-radius);
        margin-bottom: 1rem;
        box-shadow: var(--shadow-sm);
    }
    .user-avatar { 
        width: 44px; 
        height: 44px; 
        background-color: var(--primary-color); 
        border-radius: 50%; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        color: white; 
        font-weight: bold;
        font-size: 1.1rem;
    }
    .user-details { font-size: 0.9rem; }
    .user-name { font-weight: 600; }

    /* Projects List */
    .projects-list { font-size: 0.9rem; }
    .project-item { 
        margin-bottom: 12px; 
        border: 1px solid var(--border-color); 
        border-radius: var(--border-radius); 
        overflow: hidden;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .project-item:hover {
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .project-item.active { 
        border-color: var(--primary-color);
        box-shadow: 0 2px 8px rgba(55, 111, 78, 0.15);
    }
    .project-header { 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        padding: 10px 12px; 
        background-color: rgba(255, 255, 255, 0.6);
        cursor: pointer;
        user-select: none;
    }
    .project-item.active .project-header { 
        background-color: var(--primary-light);
        font-weight: 500;
    }
    .project-toggle { 
        color: var(--secondary-color); 
        font-size: 0.8rem;
        transition: transform 0.2s;
    }
    .project-item.active .project-toggle {
        transform: rotate(90deg);
    }
    .project-files { 
        padding: 5px 0; 
        max-height: 250px; 
        overflow-y: auto;
    }
    .file-group { margin-bottom: 5px; }
    .file-group-header { 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        padding: 5px 12px; 
        font-size: 0.85rem; 
        color: var(--secondary-color); 
        background-color: rgba(105, 56, 10, 0.05);
        user-select: none;
    }
    .file-count { 
        background-color: rgba(105, 56, 10, 0.2); 
        border-radius: 10px; 
        padding: 1px 6px; 
        font-size: 0.75rem;
        min-width: 22px;
        text-align: center;
    }
    .file-list { list-style: none; padding: 0; margin: 0; }
    .file-item { 
        display: flex; 
        align-items: center; 
        padding: 6px 12px 6px 15px; 
        font-size: 0.85rem; 
        position: relative; 
        cursor: pointer; 
        border-left: 3px solid transparent;
        transition: background-color 0.1s;
    }
    .file-item:hover { background-color: rgba(255, 255, 255, 0.8); }
    .file-item.processed { border-left-color: #4caf50; }
    .file-item.processing { border-left-color: #ff9800; }
    .file-item.queued { border-left-color: #9e9e9e; }
    .file-icon { margin-right: 6px; font-size: 1rem; }
    .status-badge { 
        font-size: 0.7rem; 
        background-color: #ffe0b2; 
        padding: 1px 4px; 
        border-radius: 3px; 
        margin-left: 5px; 
        color: #e65100;
    }
    .file-item.queued .status-badge { 
        background-color: #e0e0e0; 
        color: #616161;
    }

    /* Chapter Navigation */
    .chapter-nav ul { list-style: none; padding-left: 10px; }
    .chapter-nav li { margin-bottom: 0.5rem; }
    .chapter-nav a { 
        text-decoration: none; 
        color: var(--secondary-color); 
        font-size: 0.95rem;
        display: block;
        padding: 3px 0;
        transition: color 0.1s, transform 0.1s;
    }
    .chapter-nav a:hover { 
        color: var(--primary-color);
        transform: translateX(2px);
    }
    .chapter-nav .active { 
        font-weight: 600; 
        color: var(--primary-color);
    }
    .chapter-nav ul ul { 
        padding-left: 15px; 
        font-size: 0.9em; 
        margin-top: 5px;
        margin-bottom: 8px;
    }

    /* Agent Chat */
    .agent-chat { position: relative; }
    .agent-chat-messages { 
        height: 180px; 
        overflow-y: auto; 
        border: 1px solid var(--border-color); 
        border-radius: var(--border-radius); 
        padding: 10px; 
        margin-bottom: 8px; 
        background-color: #fff; 
        font-size: 0.9rem;
        scroll-behavior: smooth;
    }
    .agent-message { 
        margin-bottom: 8px; 
        padding: 6px 10px; 
        border-radius: 12px; 
        max-width: 90%; 
        display: inline-block; 
        clear: both; 
        word-wrap: break-word;
        line-height: 1.4;
        transition: opacity 0.3s;
        animation: fade-in 0.3s ease-out;
    }
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message { 
        background-color: #e1f5fe; 
        float: right; 
        border-bottom-right-radius: 2px; 
        text-align: right;
    }
    .bot-message { 
        background-color: #f0f0f0; 
        float: left; 
        border-bottom-left-radius: 2px;
    }
    .agent-thinking {
        color: #888;
        text-align: center;
        padding: 5px;
        font-style: italic;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }

    /* Main Content */
    .main-header { 
        margin-bottom: 1.8rem; 
        padding-bottom: 1rem; 
        border-bottom: 1px solid var(--border-color);
    }
    .main-header h1 { 
        color: var(--primary-color); 
        font-size: 1.8rem; 
        margin-bottom: 0.4rem;
    }
    .main-header p {
        color: #666;
        max-width: 600px;
    }

    /* Edit Mode Toggle */
    .edit-mode-toggle {
        position: absolute;
        top: 10px;
        right: 20px;
        z-index: 100;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 5px 10px;
        border-radius: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        font-size: 0.9rem;
    }
    .edit-badge {
        background-color: #ef5350;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        animation: pulse-edit 2s infinite;
    }
    @keyframes pulse-edit {
        0%, 100% { opacity: 0.8; }
        50% { opacity: 1; }
    }

    /* Chapter Cards */
    .chapter { 
        background-color: #fff; 
        padding: 1.5rem; 
        margin-bottom: 1.8rem; 
        border-radius: var(--border-radius); 
        box-shadow: var(--shadow-sm); 
        border: 1px solid var(--border-color);
        position: relative;
        transition: box-shadow 0.2s;
    }
    .chapter:hover {
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    }
    .chapter h3 { 
        color: var(--primary-color); 
        margin-top: 0; 
        margin-bottom: 0.6rem;
        font-weight: 600;
    }
    .source-info { 
        font-size: 0.8rem; 
        color: #777; 
        margin-bottom: 1.2rem; 
        font-style: italic;
    }
    .summary { 
        line-height: 1.6;
        font-size: 1rem;
    }
    .summary p { margin-bottom: 1rem; }

    /* Interactive Elements */
    .keyword-expandable { 
        color: var(--primary-color); 
        text-decoration: underline dotted; 
        cursor: pointer; 
        font-weight: 500;
        padding: 0 1px;
        transition: background-color 0.1s, color 0.1s;
    }
    .keyword-expandable:hover { 
        background-color: var(--primary-light);
        text-decoration: underline;
    }

    /* Visual Placeholders */
    .visuals-placeholder { 
        border: 1px dashed var(--border-color); 
        padding: 25px; 
        text-align: center; 
        color: #999; 
        margin-top: 1.2rem; 
        font-size: 0.9rem; 
        background-color: #fafafa; 
        border-radius: var(--border-radius); 
        min-height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Details/Summary (Expanders) */
    .details-content { 
        background-color: #fafafa; 
        padding: 15px; 
        border-radius: 5px;
        margin-top: 5px;
    }
    .details-content ul { list-style: none; padding: 0; }
    .details-content li { margin-bottom: 10px; }
    .qna-item strong { color: var(--primary-color); }

    /* Entities and Relationships */
    .entity-nodes-container { 
        display: flex; 
        flex-wrap: wrap; 
        gap: 12px; 
        margin-top: 12px; 
        padding-bottom: 10px;
    }
    .entity-node { 
        border: 2px solid; 
        padding: 6px 12px; 
        border-radius: 6px; 
        text-align: center; 
        min-width: 120px; 
        box-shadow: var(--shadow-sm); 
        font-size: 0.85rem; 
        cursor: pointer; 
        transition: transform 0.15s ease-out, box-shadow 0.15s;
        background-color: white;
    }
    .entity-node:hover { 
        transform: translateY(-2px) scale(1.02); 
        box-shadow: 0 3px 8px rgba(0,0,0,0.12);
    }
    .entity-node .key { 
        font-weight: 600; 
        display: block; 
        margin-bottom: 3px;
    }
    .entity-node .type { 
        font-size: 0.8em; 
        color: #555; 
        font-style: italic;
    }
    .entity-node[data-type="System/Concept"] { 
        border-color: var(--primary-color); 
        background-color: rgba(55, 111, 78, 0.08);
    }
    .entity-node[data-type="Technology/Model"] { 
        border-color: var(--accent-blue); 
        background-color: rgba(58, 124, 165, 0.08);
    }
    .entity-node[data-type="Data/Resource"] { 
        border-color: var(--accent-orange); 
        background-color: rgba(245, 138, 7, 0.08);
    }
    .entity-node[data-type="Data Structure/Concept"] { 
        border-color: var(--accent-yellow); 
        background-color: rgba(255, 202, 40, 0.1);
    }
    .entity-node[data-type="Limitation/Concept"] { 
        border-color: var(--accent-red); 
        background-color: rgba(199, 62, 29, 0.08);
    }
    .entity-node[data-type="Proposed System/Framework"] { 
        border-color: darkgreen; 
        background-color: rgba(0, 100, 0, 0.08);
    }

    /* Relationships List */
    .relationships-list ul { list-style: none; padding: 0; }
    .relationship-item { 
        border: 1px solid #eee; 
        background-color: #fff; 
        padding: 12px; 
        margin-bottom: 12px; 
        border-radius: 5px; 
        font-size: 0.9rem; 
        line-height: 1.5;
        transition: box-shadow 0.2s;
    }
    .relationship-item:hover {
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
    }
    .relationship-item strong { 
        font-weight: bold; 
        color: var(--primary-color);
    }
    .rel-source, .rel-target { 
        background-color: rgba(105, 56, 10, 0.08); 
        padding: 3px 8px; 
        border-radius: 4px; 
        border: 1px solid var(--border-color); 
        font-weight: 500; 
        color: var(--secondary-color); 
        display: inline-block;
        margin: 2px 1px;
    }
    .rel-description { 
        display: block; 
        margin: 8px 0 5px; 
        color: #333;
    }
    .rel-keys { 
        font-size: 0.85em; 
        color: #666; 
        margin-top: 6px; 
        word-wrap: break-word;
    }
    .rel-keys code { 
        background-color: #f0f0f0; 
        padding: 2px 5px; 
        border-radius: 3px; 
        font-size: 0.95em;
        color: #333;
        margin-right: 1px;
    }

    /* Keyword Popup & Knowledge Graph Widgets */
    #keyword-popup-widget { 
        position: fixed; 
        z-index: 9999;
    }
    #keyword-popup-widget > div { 
        position: absolute;
        background-color: white;
        border: 1px solid #ccc;
        border-radius: var(--border-radius);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
        padding: 15px;
        width: 320px;
        font-size: 0.9rem;
        color: var(--text-color);
        transform-origin: top left;
        animation: popup-appear 0.2s ease-out;
    }
    @keyframes popup-appear {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    #keyword-popup-widget h5 { 
        margin: 0 0 10px 0; 
        font-size: 1.1rem; 
        color: var(--primary-color);
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 5px;
    }
    #keyword-popup-widget p { margin: 0 0 6px 0; }
    #keyword-popup-widget strong { color: var(--secondary-color); }
    #keyword-popup-widget .context-snippet { 
        font-style: italic; 
        color: #555; 
        border-left: 2px solid var(--primary-color); 
        padding-left: 10px;
        margin: 10px 0; 
        background: #fafafa;
        padding: 8px 10px;
        border-radius: 3px;
        font-size: 0.9rem;
    }
    #keyword-popup-widget .related-terms { 
        margin-top: 12px; 
        font-size: 0.85rem;
    }
    #keyword-popup-widget .related-terms span { 
        display: inline-block; 
        margin-right: 6px; 
        margin-bottom: 6px; 
        padding: 3px 8px; 
        background-color: rgba(55, 111, 78, 0.08); 
        border-radius: 12px; 
        cursor: pointer; 
        border: 1px solid transparent;
        transition: all 0.1s;
    }
    #keyword-popup-widget .related-terms span:hover { 
        background-color: rgba(55, 111, 78, 0.15); 
        border-color: var(--primary-color);
    }

    /* Knowledge Graph */
    #knowledge-graph-widget > div { 
        height: 340px;
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        background-color: #fdfdfd;
        overflow: hidden;
        position: relative;
    }
    #knowledge-graph-tooltip {
        position: absolute;
        background-color: rgba(0,0,0,0.7);
        color: white;
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        pointer-events: none;
        display: none;
        z-index: 1010;
        max-width: 200px;
    }

    /* Right Sidebar Components */
    .search-box { 
        position: relative; 
        margin-bottom: 1.2rem;
    }
    .glossary-list ul { 
        list-style: none; 
        padding: 0 8px; 
        font-size: 0.9rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .glossary-list li { 
        margin-bottom: 6px;
        transition: transform 0.1s;
    }
    .glossary-list li:hover {
        transform: translateX(2px);
    }
    .glossary-list a { 
        color: var(--secondary-color); 
        text-decoration: none; 
        cursor: pointer;
        display: block;
        padding: 2px 0;
    }
    .glossary-list a:hover { 
        color: var(--primary-color); 
        text-decoration: underline;
    }

    /* Loading Animation for Graph and Data */
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
        color: #888;
    }
    .loading-dots {
        display: inline-block;
    }
    .loading-dots::after {
        content: '.';
        animation: loading 1.2s infinite;
    }
    @keyframes loading {
        0% { content: '.'; }
        33% { content: '..'; }
        66% { content: '...'; }
    }

    /* Edit Mode Styles */
    .edit-mode-active .editable-content {
        border: 1px dashed #ddd;
        border-radius: 4px;
        padding: 8px;
        background-color: #fafcff;
        transition: all 0.2s;
    }
    .edit-mode-active .editable-content:hover,
    .edit-mode-active .editable-content:focus {
        border-color: var(--primary-color);
        background-color: #f0f8ff;
    }
    .edit-indicator {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: var(--primary-color);
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        animation: pulse-edit 2s infinite;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color);
        font-size: 0.85rem;
        color: #777;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Mock Data with Enhanced Structure
# -----------------------------------------------------------------------------

# Initialize default data if not in session state
if 'report_data' not in st.session_state:
    st.session_state.report_data = {
        "projectName": "LightRAG Analysis",
        "report_metadata": {
            "report_id": "PF-LRAG-2024-01",
            "report_date": datetime.now().strftime("%B %d, %Y"),
            "data_confidence": "High",
            "disclaimer": "Synthesized from provided examples. For demonstration purposes only."
        },
        "company_data": {
            "name": "LightRAG Project",
            "description": "Analysis of the LightRAG framework for Retrieval-Augmented Generation.",
            "tags": ["AI", "RAG", "Knowledge Graph", "Research Paper"]
        },
        "chapters": [
            {
                "id": "chapter1",
                "title": "1. Executive Summary",
                "source": "Automated Synthesis",
                "type": "Generated",
                "summary": "This report analyzes the proposed LightRAG framework, contrasting it with standard Retrieval-Augmented Generation (RAG) systems. LightRAG aims to overcome limitations inherent in traditional RAG, such as reliance on Flat Data Representations, by leveraging Graph Structures for enhanced text indexing and retrieval. This approach promises more contextually aware and accurate responses from Large Language Models (LLMs) by integrating relationship data from External Knowledge Sources more effectively.",
                "qna": [
                    {"q": "What is the core innovation of LightRAG?", "a": "Its use of graph structures during indexing and retrieval to improve context and accuracy over standard RAG systems.", "ref": "📄 p.2"}
                ],
                "entities": [
                    {"key": "LightRAG", "type": "Proposed System/Framework"},
                    {"key": "Retrieval-Augmented Generation (RAG)", "type": "System/Concept"},
                    {"key": "Large Language Models (LLMs)", "type": "Technology/Model"},
                    {"key": "Flat Data Representations", "type": "Limitation/Concept"},
                    {"key": "Graph Structures", "type": "Data Structure/Concept"},
                    {"key": "External Knowledge Sources", "type": "Data/Resource"},
                ],
                "relationships": [
                    {"source": "LightRAG", "target": "Graph Structures", "desc": "Incorporates", "keys": ["Incorporation", "Improvement"]},
                    {"source": "Graph Structures", "target": "Flat Data Representations", "desc": "Addresses/Solves", "keys": ["Addressing Challenge"]},
                ]
            },
            {
                "id": "chapter2",
                "title": "2. RAG Systems Overview",
                "source": "LightRAG_Paper.pdf",
                "type": "PDF",
                "pages": "1-3",
                "summary": "RAG systems primarily function by retrieving relevant text chunks from a vector database based on query similarity. These chunks are then fed into an LLM as context to generate a response.\n\n**Limitation:** Standard RAG often struggles with complex queries requiring multi-hop reasoning as it treats knowledge chunks independently, missing inherent relationships present in the External Knowledge Sources.\nThe reliance on Flat Data Representations (simple text embeddings) is a key bottleneck for deeper contextual understanding.",
                "qna": [
                    {"q": "What is the main drawback mentioned for existing RAG systems?", "a": "Their inability to effectively utilize relationships between knowledge chunks due to using flat data representations.", "ref": "📄 p.2"},
                    {"q": "What data structure do standard RAG systems typically rely on for retrieval?", "a": "Vector databases containing embeddings of text chunks (a flat representation).", "ref": "📄 p.1"}
                ],
                "entities": [
                    {"key": "Retrieval-Augmented Generation (RAG)", "type": "System/Concept"},
                    {"key": "Large Language Models (LLMs)", "type": "Technology/Model"},
                    {"key": "Vector Database", "type": "Data Structure/Concept"},
                    {"key": "Flat Data Representations", "type": "Limitation/Concept"},
                    {"key": "External Knowledge Sources", "type": "Data/Resource"},
                ],
                "relationships": [
                    {"source": "RAG", "target": "Vector Database", "desc": "Utilizes", "keys": ["Data Retrieval"]},
                    {"source": "RAG", "target": "Flat Data Representations", "desc": "Limited By", "keys": ["Limitation", "Reliance On"]},
                    {"source": "RAG", "target": "LLMs", "desc": "Enhances", "keys": ["Enhancement"]},
                    {"source": "RAG", "target": "External Knowledge Sources", "desc": "Integrates", "keys": ["Integration"]},
                ]
            },
            {
                "id": "chapter3",
                "title": "3. LightRAG Architecture",
                "source": "LightRAG_Paper.pdf",
                "type": "PDF",
                "pages": "4-7",
                "summary": "LightRAG introduces a novel architecture that combines traditional vector-based retrieval with knowledge graph structures. The system constructs a dynamic graph during the indexing phase, where entities become nodes and their relationships form edges. During query processing, LightRAG performs both local keyword matching and global context-aware traversal of the knowledge graph.\n\nThis hybrid approach enables more precise retrieval for complex, multi-hop questions that require understanding relationships between different pieces of information.",
                "qna": [
                    {"q": "How does LightRAG handle the indexing phase differently?", "a": "It constructs a knowledge graph alongside traditional vector embeddings, capturing entities as nodes and relationships as edges.", "ref": "📄 p.4"},
                    {"q": "What makes LightRAG better for multi-hop questions?", "a": "Its ability to perform graph traversal operations that follow relationship paths between entities, rather than just retrieving individual text chunks.", "ref": "📄 p.6"}
                ],
                "entities": [
                    {"key": "LightRAG", "type": "Proposed System/Framework"},
                    {"key": "Knowledge Graph", "type": "Data Structure/Concept"},
                    {"key": "Graph Traversal", "type": "System/Concept"},
                    {"key": "Multi-hop Questions", "type": "Concept"},
                    {"key": "Entity Extraction", "type": "Technology/Model"},
                ],
                "relationships": [
                    {"source": "LightRAG", "target": "Knowledge Graph", "desc": "Constructs", "keys": ["Construction", "Indexing"]},
                    {"source": "LightRAG", "target": "Graph Traversal", "desc": "Utilizes", "keys": ["Retrieval Method"]},
                    {"source": "LightRAG", "target": "Entity Extraction", "desc": "Performs", "keys": ["Processing Step"]},
                    {"source": "Graph Traversal", "target": "Multi-hop Questions", "desc": "Enables", "keys": ["Capability", "Benefit"]},
                ]
            }
        ],
        "keyword_data": {
            "LightRAG": {"definition": "A proposed RAG framework using graph structures for indexing/retrieval.", "source": "LightRAG_Paper.pdf (Abstract)", "context": "we propose LightRAG, which incorporates graph structures...", "related": ["RAG", "Graph Structures", "LLMs"] },
            "Retrieval-Augmented Generation (RAG)": {"definition": "Systems enhancing LLMs with external knowledge retrieval.", "source": "LightRAG_Paper.pdf (Abstract)", "context": "Retrieval-Augmented Generation (RAG) systems enhance large language models...", "related": ["LLMs", "LightRAG", "Flat Data Representations", "Vector Database"] },
            "Large Language Models (LLMs)": {"definition": "Foundation models trained on vast text data (e.g., GPT, Gemini).", "source": "LightRAG_Paper.pdf (Abstract)", "context": "...enhance large language models (LLMs) by integrating...", "related": ["RAG", "LightRAG"] },
            "Flat Data Representations": {"definition": "Storing data as independent chunks (e.g., text embeddings) without explicit relationships.", "source": "LightRAG_Paper.pdf (Abstract)", "context": "...reliance on flat data representations...", "related": ["RAG", "Graph Structures", "Vector Database"] },
            "Graph Structures": {"definition": "Data representation using nodes (entities) and edges (relationships).", "source": "LightRAG_Paper.pdf (Abstract)", "context": "...incorporates graph structures into text indexing...", "related": ["LightRAG", "Flat Data Representations", "Knowledge Graph"] },
            "External Knowledge Sources": {"definition": "Databases, documents, or websites used to supplement LLM knowledge.", "source": "LightRAG_Paper.pdf (Abstract)", "context": "...integrating external knowledge sources...", "related": ["RAG"] },
            "Vector Database": {"definition": "Database optimized for storing and querying high-dimensional vectors (embeddings).", "source": "LightRAG_Paper.pdf (Page 1)", "context": "...retrieving relevant text chunks from a vector database...", "related": ["RAG", "Flat Data Representations"] },
            "Knowledge Graph": {"definition": "A structured representation of knowledge with entities (nodes) connected by relationships (edges).", "source": "LightRAG_Paper.pdf (Page 4)", "context": "LightRAG constructs a knowledge graph during indexing...", "related": ["Graph Structures", "LightRAG"] },
            "Graph Traversal": {"definition": "The process of visiting and exploring nodes and edges in a graph structure.", "source": "LightRAG_Paper.pdf (Page 6)", "context": "...performs graph traversal operations to follow relationships...", "related": ["Knowledge Graph", "Multi-hop Questions"] },
            "Multi-hop Questions": {"definition": "Complex queries requiring information from multiple connected sources to answer.", "source": "LightRAG_Paper.pdf (Page 5)", "context": "...particularly effective for multi-hop questions...", "related": ["Graph Traversal", "Knowledge Graph"] },
            "Entity Extraction": {"definition": "The NLP process of identifying and classifying named entities in text.", "source": "LightRAG_Paper.pdf (Page 4)", "context": "...uses entity extraction to identify key elements...", "related": ["LightRAG", "Knowledge Graph"] },
        },
        "graph_data": {
            "nodes": [
                {"id": "LightRAG", "group": 6, "desc": "Graph-enhanced RAG framework"},
                {"id": "RAG", "group": 1, "desc": "Retrieval-Augmented Generation"},
                {"id": "LLMs", "group": 2, "desc": "Large Language Models"},
                {"id": "Flat Data", "group": 5, "desc": "Flat Data Representations"},
                {"id": "Graph Structures", "group": 4, "desc": "Node-edge data structures"},
                {"id": "External Knowledge", "group": 3, "desc": "External data sources"},
                {"id": "Vector Database", "group": 4, "desc": "Vector storage system"},
                {"id": "Knowledge Graph", "group": 4, "desc": "Entity relationship graph"},
                {"id": "Graph Traversal", "group": 1, "desc": "Navigation through graph"},
                {"id": "Multi-hop Questions", "group": 7, "desc": "Complex queries requiring connections"},
                {"id": "Entity Extraction", "group": 2, "desc": "Identifying entities in text"}
            ],
            "links": [
                {"source": "RAG", "target": "LLMs", "value": 1},
                {"source": "RAG", "target": "External Knowledge", "value": 1},
                {"source": "RAG", "target": "Flat Data", "value": 1},
                {"source": "RAG", "target": "Vector Database", "value": 1},
                {"source": "LightRAG", "target": "Graph Structures", "value": 1},
                {"source": "LightRAG", "target": "RAG", "value": 1},
                {"source": "Graph Structures", "target": "Flat Data", "value": 1},
                {"source": "LightRAG", "target": "Knowledge Graph", "value": 1},
                {"source": "LightRAG", "target": "Graph Traversal", "value": 1},
                {"source": "LightRAG", "target": "Entity Extraction", "value": 1},
                {"source": "Graph Traversal", "target": "Multi-hop Questions", "value": 1},
                {"source": "Knowledge Graph", "target": "Graph Structures", "value": 1}
            ]
        },
        "glossary_terms": [
            "LightRAG", "Retrieval-Augmented Generation (RAG)", "Large Language Models (LLMs)",
            "Graph Structures", "Flat Data Representations", "External Knowledge Sources", 
            "Vector Database", "Knowledge Graph", "Graph Traversal", "Multi-hop Questions", 
            "Entity Extraction"
        ],
        "projects": [
            {
                "name": "LightRAG Analysis", "active": True,
                "files": [
                    {"group": "PDF Documents", "count": 2, "items": [
                        {"name": "Research_Paper.pdf", "status": "processed", "icon": "📄"},
                        {"name": "Implementation_Guide.pdf", "status": "processed", "icon": "📄"}
                    ]},
                    {"group": "Code", "count": 3, "items": [
                        {"name": "lightrag_indexer.py", "status": "processed", "icon": "🐍"},
                        {"name": "graph_builder.py", "status": "processed", "icon": "🐍"},
                        {"name": "query_processor.py", "status": "processed", "icon": "🐍"}
                    ]},
                    {"group": "Data", "count": 1, "items": [
                        {"name": "benchmark_results.csv", "status": "processing", "icon": "📊"}
                    ]},
                ]
            },
            {
                "name": "Traditional RAG Comparison", "active": False, 
                "files": [
                    {"group": "PDF Documents", "count": 2, "items": [
                        {"name": "Baseline_RAG_Methods.pdf", "status": "processed", "icon": "📄"},
                        {"name": "Performance_Metrics.pdf", "status": "processed", "icon": "📄"}
                    ]}
                ]
            },
            {
                "name": "Competitor Analysis", "active": False, "files": []}
        ],
        "chat_history": [
            {"sender": "bot", "message": "Welcome! I can help you explore the LightRAG analysis report. Feel free to ask about specific concepts or request summaries of chapters."}
        ]
    }

# Initialize other session state variables
if 'editing_mode' not in st.session_state:
    st.session_state.editing_mode = False
if 'active_project' not in st.session_state:
    st.session_state.active_project = st.session_state.report_data['projects'][0]['name'] if st.session_state.report_data['projects'] else None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = st.session_state.report_data['chat_history'][:]
if 'chat_thinking' not in st.session_state:
    st.session_state.chat_thinking = False
if 'edit_changes' not in st.session_state:
    st.session_state.edit_changes = {}

# -----------------------------------------------------------------------------
# 4. Custom Widgets: Knowledge Graph and Keyword Popup
# -----------------------------------------------------------------------------

# Define the Knowledge Graph widget
class KnowledgeGraphWidget(AnyWidget):
    # Define traitlets (synchronized properties)
    nodes = traitlets.List([]).tag(sync=True)
    links = traitlets.List([]).tag(sync=True)
    clicked_node = traitlets.Dict({}).tag(sync=True)
    
    # JavaScript code for the widget
    _esm = """
    function render({ model, el }) {
        // Create container for the graph
        const container = document.createElement('div');
        container.id = 'knowledge-graph-container';
        container.style.width = '100%';
        container.style.height = '100%';
        container.style.position = 'relative';
        el.appendChild(container);
        
        // Create tooltip element
        const tooltip = document.createElement('div');
        tooltip.id = 'knowledge-graph-tooltip';
        container.appendChild(tooltip);
        
        // Load D3.js
        const script = document.createElement('script');
        script.src = 'https://d3js.org/d3.v7.min.js';
        script.onload = () => renderGraph(model, container, tooltip);
        document.head.appendChild(script);
        
        function renderGraph(model, container, tooltip) {
            // Get data from model
            const nodes = model.get('nodes');
            const links = model.get('links');
            
            if (!nodes || !nodes.length || !links || !links.length) {
                container.innerHTML = '<div class="loading-animation"><span>No graph data available</span></div>';
                return;
            }
            
            // Clear previous content
            container.innerHTML = '';
            container.appendChild(tooltip);
            
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            // Create SVG
            const svg = d3.select(container).append('svg')
                .attr('width', width)
                .attr('height', height)
                .attr('viewBox', [0, 0, width, height]);
                
            // Add a subtle background pattern
            svg.append('defs')
                .append('pattern')
                .attr('id', 'grid')
                .attr('width', 20)
                .attr('height', 20)
                .attr('patternUnits', 'userSpaceOnUse')
                .append('path')
                .attr('d', 'M 20 0 L 0 0 0 20')
                .attr('fill', 'none')
                .attr('stroke', '#f0f0f0')
                .attr('stroke-width', 1);
                
            svg.append('rect')
                .attr('width', width)
                .attr('height', height)
                .attr('fill', 'url(#grid)');
            
            // Force simulation
            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(90).strength(0.6))
                .force("charge", d3.forceManyBody().strength(-120))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("x", d3.forceX(width/2).strength(0.05))
                .force("y", d3.forceY(height/2).strength(0.05))
                .force("collision", d3.forceCollide().radius(30));
            
            // Create relationship map for later lookup
            const linkedByIndex = {};
            links.forEach(d => {
                const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
                const targetId = typeof d.target === 'object' ? d.target.id : d.target;
                linkedByIndex[`${sourceId},${targetId}`] = 1;
            });
            
            function isConnected(a, b) {
                const aId = typeof a === 'object' ? a.id : a;
                const bId = typeof b === 'object' ? b.id : b;
                return linkedByIndex[`${aId},${bId}`] || 
                       linkedByIndex[`${bId},${aId}`] || 
                       aId === bId;
            }
            
            // Draw links
            const link = svg.append("g")
                .attr("class", "links")
                .selectAll("line")
                .data(links)
                .join("line")
                .attr("class", "link")
                .attr("stroke-width", d => Math.sqrt(d.value) * 1.5)
                .attr("stroke-opacity", 0.6);
            
            // Draw nodes
            const node = svg.append("g")
                .attr("class", "nodes")
                .selectAll("g")
                .data(nodes)
                .join("g")
                .attr("class", "node")
                .attr("data-group", d => d.group)
                .call(drag(simulation));
            
            // Add circles to nodes
            node.append("circle")
                .attr("r", d => d.group === 6 ? 9 : 7)
                .attr("fill-opacity", 0.9);
            
            // Add labels to nodes
            node.append("text")
                .text(d => d.id.length > 15 ? d.id.substring(0,12) + '...' : d.id)
                .attr("x", 10)
                .attr("y", 3)
                .attr("font-size", d => d.group === 6 ? "10px" : "9px")
                .attr("font-weight", d => d.group === 6 ? "bold" : "normal");
            
            // Node interactions
            node.on('mouseover', (event, d) => {
                // Show tooltip
                const tooltipContent = d.desc ? `<strong>${d.id}</strong><br>${d.desc}` : `<strong>${d.id}</strong>`;
                tooltip.style.display = 'block';
                tooltip.innerHTML = tooltipContent;
                tooltip.style.left = (event.pageX + 10) + 'px';
                tooltip.style.top = (event.pageY - 25) + 'px';
                
                // Highlight connected nodes and links
                link.style('stroke-opacity', l => {
                    const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
                    const targetId = typeof l.target === 'object' ? l.target.id : l.target;
                    return (sourceId === d.id || targetId === d.id) ? 1 : 0.1;
                });
                
                node.style('opacity', n => isConnected(d, n) ? 1 : 0.2);
                d3.select(event.currentTarget).select('circle')
                  .transition().duration(200)
                  .attr('r', d.group === 6 ? 11 : 9);
            })
            .on('mouseout', () => {
                // Hide tooltip
                tooltip.style.display = 'none';
                
                // Reset highlights
                link.style('stroke-opacity', 0.6);
                node.style('opacity', 1);
                node.select('circle')
                  .transition().duration(200)
                  .attr('r', d => d.group === 6 ? 9 : 7);
            })
            .on('click', (event, d) => {
                // Send click info back to Python
                model.set('clicked_node', {id: d.id, group: d.group});
                model.save_changes();
                
                // Trigger keyword popup event
                window.dispatchEvent(new CustomEvent('showKeywordPopup', { 
                    detail: { 
                        keyword: d.id, 
                        event: {
                            target: event.currentTarget,
                            pageX: event.pageX,
                            pageY: event.pageY
                        }
                    } 
                }));
            });
            
            // Start simulation and update positions
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node.attr("transform", d => `translate(${d.x},${d.y})`);
            });
            
            // Add zoom capability
            const zoom = d3.zoom()
                .scaleExtent([0.5, 3])
                .on("zoom", (event) => {
                    svg.selectAll("g").attr("transform", event.transform);
                });
            
            svg.call(zoom);
            
            // Reset zoom button
            const resetButton = document.createElement('button');
            resetButton.textContent = "Reset View";
            resetButton.style.position = "absolute";
            resetButton.style.bottom = "10px";
            resetButton.style.right = "10px";
            resetButton.style.padding = "5px 10px";
            resetButton.style.fontSize = "0.8rem";
            resetButton.style.cursor = "pointer";
            resetButton.style.backgroundColor = "#f0f0f0";
            resetButton.style.border = "1px solid #ddd";
            resetButton.style.borderRadius = "4px";
            resetButton.onclick = () => {
                svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
            };
            
            container.appendChild(resetButton);
            
            // Drag functionality
            function drag(simulation) {
                function dragstarted(event, d) {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }
                
                function dragged(event, d) {
                    d.fx = event.x;
                    d.fy = event.y;
                }
                
                function dragended(event, d) {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }
                
                return d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended);
            }
        }
        
        // Handle resize
        const resizeObserver = new ResizeObserver(() => {
            renderGraph(model, container, tooltip);
        });
        
        resizeObserver.observe(container, { box: 'border-box' });
        
        // Handle model changes
        model.on('change:nodes change:links', () => {
            renderGraph(model, container, tooltip);
        });
    }
    
    export default { render };
    """
    
    # CSS for the widget
    _css = """
    #knowledge-graph-container {
        height: 340px;
        border: 1px solid rgba(105, 56, 10, 0.15);
        border-radius: 6px;
        background-color: #fdfdfd;
        overflow: hidden;
        position: relative;
    }
    
    #knowledge-graph-tooltip {
        position: absolute;
        background-color: rgba(0,0,0,0.7);
        color: white;
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        pointer-events: none;
        display: none;
        z-index: 1010;
        max-width: 200px;
    }
    
    .link {
        stroke: rgba(105, 56, 10, 0.15);
        stroke-opacity: 0.6;
    }
    
    .node circle {
        stroke: #69380a;
        stroke-width: 1.5px;
        cursor: pointer;
    }
    
    .node text {
        pointer-events: none;
        font-size: 9px;
        fill: #4d2907;
    }
    
    .node:hover circle {
        stroke-width: 3px;
    }
    
    /* Node Fill Colors */
    .node[data-group="1"] circle { fill: rgba(55, 111, 78, 0.1); }
    .node[data-group="2"] circle { fill: rgba(58, 124, 165, 0.1); }
    .node[data-group="3"] circle { fill: rgba(245, 138, 7, 0.1); }
    .node[data-group="4"] circle { fill: rgba(255, 202, 40, 0.15); }
    .node[data-group="5"] circle { fill: rgba(199, 62, 29, 0.1); }
    .node[data-group="6"] circle { fill: rgba(0, 100, 0, 0.08); }
    .node[data-group="7"] circle { fill: #eee; }
    """

# Define the Keyword Popup widget
class KeywordPopupWidget(AnyWidget):
    # Define traitlets
    keyword_data = traitlets.Dict({}).tag(sync=True)
    shown_keyword = traitlets.Unicode('').tag(sync=True)
    
    # JavaScript code for the widget
    _esm = """
    function render({ model, el }) {
        // Create popup container
        const popup = document.createElement('div');
        popup.id = 'keyword-popup-container';
        popup.innerHTML = `
            <h5 id="popup-title">Keyword</h5>
            <p><strong>Definition:</strong> <span id="popup-definition">...</span></p>
            <p><strong>Source:</strong> <span id="popup-source">...</span></p>
            <div class="context-snippet">
                <strong>Context:</strong> <span id="popup-context">...</span>
                <span id="popup-source-ref" class="source-reference" style="cursor: help;" title="">📄</span>
            </div>
            <div class="related-terms">
                <strong>Related:</strong> <div id="popup-related"></div>
            </div>
        `;
        el.appendChild(popup);
        
        // Get popup elements
        const title = popup.querySelector('#popup-title');
        const definition = popup.querySelector('#popup-definition');
        const source = popup.querySelector('#popup-source');
        const context = popup.querySelector('#popup-context');
        const sourceRef = popup.querySelector('#popup-source-ref');
        const related = popup.querySelector('#popup-related');
        let hideTimeout;
        
        // Hide popup on click outside
        document.addEventListener('click', function(event) {
            if (popup.style.display === 'block' &&
                !popup.contains(event.target) &&
                !event.target.closest('.keyword-expandable, .glossary-list a, .node')) {
                hidePopup();
            }
        });
        
        function showPopup(keyword, event) {
            // Clear any pending hide
            clearTimeout(hideTimeout);
            
            // Get keyword data
            const keywordData = model.get('keyword_data');
            const data = keywordData[keyword];
            
            if (!data) {
                console.warn("No data for keyword:", keyword);
                hidePopup();
                return;
            }
            
            // Set shown keyword in model
            model.set('shown_keyword', keyword);
            model.save_changes();
            
            // Populate content
            title.textContent = keyword;
            definition.textContent = data.definition || 'Not available';
            source.textContent = data.source || 'Unknown source';
            context.textContent = data.context || 'No context available';
            sourceRef.title = data.source ? `Source: ${data.source}` : '';
            sourceRef.style.display = data.source ? 'inline-block' : 'none';
            
            // Clear and populate related terms
            related.innerHTML = '';
            if (data.related && data.related.length > 0) {
                data.related.forEach(term => {
                    const span = document.createElement('span');
                    span.textContent = term;
                    span.setAttribute('data-keyword', term);
                    span.addEventListener('click', (e) => {
                        e.stopPropagation();
                        // Find trigger for positioning or use current event
                        const triggerEl = document.querySelector(`.keyword-expandable[data-keyword='${term}']`) || 
                                        document.querySelector(`.glossary-list a[data-keyword='${term}']`) || 
                                        span;
                        // Show popup for related term
                        showPopup(term, { 
                            target: triggerEl, 
                            pageX: e.pageX, 
                            pageY: e.pageY 
                        });
                    });
                    related.appendChild(span);
                });
            } else {
                const noRelated = document.createElement('span');
                noRelated.textContent = 'None found';
                noRelated.style.fontStyle = 'italic';
                noRelated.style.color = '#888';
                related.appendChild(noRelated);
            }
            
            // Position the popup smartly
            positionPopup(event);
            
            // Show with animation
            popup.style.display = 'block';
        }
        
        function positionPopup(event) {
            // Get dimensions and position
            const rect = event.target ? event.target.getBoundingClientRect() : { top: 0, right: 0, bottom: 0, left: 0 };
            const popupWidth = 320; // Fixed width from CSS
            const popupHeight = popup.offsetHeight || 200; // Estimated if not visible yet
            
            // Calculate available space
            const windowWidth = window.innerWidth;
            const windowHeight = window.innerHeight;
            const scrollY = window.scrollY || document.documentElement.scrollTop;
            const scrollX = window.scrollX || document.documentElement.scrollLeft;
            
            // Default position at pointer
            let left = event.pageX;
            let top = event.pageY + 15; // Below pointer
            
            // Check right edge
            if (left + popupWidth + 10 > windowWidth + scrollX) {
                left = windowWidth + scrollX - popupWidth - 10;
            }
            
            // Make sure left isn't negative
            if (left < scrollX + 10) {
                left = scrollX + 10;
            }
            
            // Check if popup would go below viewport
            if (top + popupHeight + 10 > windowHeight + scrollY) {
                // Position above if there's room
                if (rect.top - popupHeight - 10 > 0) {
                    top = rect.top + scrollY - popupHeight - 10;
                } else {
                    // Otherwise just make sure it's visible
                    top = windowHeight + scrollY - popupHeight - 10;
                }
            }
            
            // Set position
            popup.style.top = `${top}px`;
            popup.style.left = `${left}px`;
        }
        
        function hidePopup() {
            hideTimeout = setTimeout(() => {
                if (popup && popup.style.display === 'block' && !popup.matches(':hover')) {
                    popup.style.display = 'none';
                }
            }, 200);
        }
        
        // Keep popup shown when hovering over it
        popup.addEventListener('mouseenter', () => clearTimeout(hideTimeout));
        popup.addEventListener('mouseleave', hidePopup);
        
        // Listen for custom events from other components
        window.addEventListener('showKeywordPopup', (e) => {
            if (e.detail && e.detail.keyword && e.detail.event) {
                showPopup(e.detail.keyword, e.detail.event);
            }
        });
        
        // Add listeners to keyword elements
        function addKeywordListeners() {
            // Keywords in content
            document.querySelectorAll('.keyword-expandable').forEach(el => {
                const keyword = el.getAttribute('data-keyword') || el.textContent.trim();
                if (keyword && model.get('keyword_data')[keyword]) {
                    // Remove any existing listeners
                    el.removeEventListener('mouseenter', el._mouseenterHandler);
                    el.removeEventListener('mouseleave', el._mouseleaveHandler);
                    el.removeEventListener('click', el._clickHandler);
                    
                    // Add new listeners
                    el._mouseenterHandler = (e) => showPopup(keyword, e);
                    el._mouseleaveHandler = () => hidePopup();
                    el._clickHandler = (e) => {
                        e.preventDefault();
                        showPopup(keyword, e);
                    };
                    
                    el.addEventListener('mouseenter', el._mouseenterHandler);
                    el.addEventListener('mouseleave', el._mouseleaveHandler);
                    el.addEventListener('click', el._clickHandler);
                }
            });
            
            // Glossary links
            document.querySelectorAll('.glossary-list a').forEach(el => {
                const keyword = el.getAttribute('data-keyword') || el.textContent.trim();
                if (keyword && model.get('keyword_data')[keyword]) {
                    // Remove existing handlers
                    el.removeEventListener('click', el._clickHandler);
                    
                    // Add new click handler
                    el._clickHandler = (e) => {
                        e.preventDefault(); 
                        showPopup(keyword, e);
                    };
                    el.addEventListener('click', el._clickHandler);
                }
            });
            
            // Entity nodes
            document.querySelectorAll('.entity-node').forEach(el => {
                const keyword = el.getAttribute('data-keyword') || el.querySelector('.key')?.textContent?.trim();
                if (keyword && model.get('keyword_data')[keyword]) {
                    // Remove existing handlers
                    el.removeEventListener('mouseenter', el._mouseenterHandler);
                    el.removeEventListener('mouseleave', el._mouseleaveHandler);
                    el.removeEventListener('click', el._clickHandler);
                    
                    // Add new handlers
                    el._mouseenterHandler = (e) => showPopup(keyword, e);
                    el._mouseleaveHandler = () => hidePopup(); 
                    el._clickHandler = (e) => {
                        e.preventDefault();
                        showPopup(keyword, e);
                    };
                    
                    el.addEventListener('mouseenter', el._mouseenterHandler);
                    el.addEventListener('mouseleave', el._mouseleaveHandler);
                    el.addEventListener('click', el._clickHandler);
                }
            });
        }
        
        // Initial setup of listeners
        addKeywordListeners();
        
        // Watch for DOM changes to add listeners to new elements
        const observer = new MutationObserver(mutations => {
            let shouldUpdate = false;
            for (const mutation of mutations) {
                if (mutation.type === 'childList' && 
                    (mutation.addedNodes.length > 0 || mutation.removedNodes.length > 0)) {
                    shouldUpdate = true;
                    break;
                }
            }
            if (shouldUpdate) {
                addKeywordListeners();
            }
        });
        
        // Start observing
        observer.observe(document.body, { childList: true, subtree: true });
        
        // Listen for model changes
        model.on('change:keyword_data', () => {
            addKeywordListeners();
        });
    }
    
    export default { render };
    """
    
    # CSS for the widget
    _css = """
    #keyword-popup-container {
        position: fixed;
        z-index: 9999;
        background-color: white;
        border: 1px solid #ccc;
        border-radius: 6px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
        padding: 15px;
        width: 320px;
        font-size: 0.9rem;
        color: #4d2907;
        display: none;
        transform-origin: top left;
        animation: popup-appear 0.2s ease-out;
    }
    
    @keyframes popup-appear {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    #keyword-popup-container h5 {
        margin: 0 0 10px 0;
        font-size: 1.1rem;
        color: #376f4e;
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 5px;
    }
    
    #keyword-popup-container p {
        margin: 0 0 6px 0;
    }
    
    #keyword-popup-container strong {
        color: #69380a;
    }
    
    #keyword-popup-container .context-snippet {
        font-style: italic;
        color: #555;
        border-left: 2px solid #376f4e;
        margin: 10px 0;
        background: #fafafa;
        padding: 8px 10px;
        border-radius: 3px;
        font-size: 0.9rem;
    }
    
    #keyword-popup-container .related-terms {
        margin-top: 12px;
        font-size: 0.85rem;
    }
    
    #keyword-popup-container .related-terms span {
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
        padding: 3px 8px;
        background-color: rgba(55, 111, 78, 0.08);
        border-radius: 12px;
        cursor: pointer;
        border: 1px solid transparent;
        transition: all 0.1s;
    }
    
    #keyword-popup-container .related-terms span:hover {
        background-color: rgba(55, 111, 78, 0.15);
        border-color: #376f4e;
    }
    
    .source-reference {
        display: inline-block;
        background-color: #eee;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.8em;
        color: #555;
        margin-left: 5px;
        cursor: help;
        border: 1px solid #ddd;
    }
"""

# -----------------------------------------------------------------------------
# 5. Fragment Functions for UI Components
# -----------------------------------------------------------------------------

@st.fragment
def render_header(report_data):
    """Render the report header with company info and metadata."""
    # Apply edit mode class if needed
    editing_mode = st.session_state.editing_mode
    css_class = "edit-mode-active" if editing_mode else ""
    
    if editing_mode:
        # Editable fields with streamlit controls
        company_name = st.text_input(
            "Project Name", 
            value=report_data["company_data"]["name"],
            key="edit_header_name"
        )
        company_desc = st.text_input(
            "Project Description", 
            value=report_data["company_data"]["description"],
            key="edit_header_desc"
        )
        
        # Update session state
        if company_name != report_data["company_data"]["name"]:
            st.session_state.edit_changes["company_data.name"] = company_name
        if company_desc != report_data["company_data"]["description"]:
            st.session_state.edit_changes["company_data.description"] = company_desc
                
        # Process tags
        tags_str = st.text_input(
            "Tags (comma separated)", 
            value=", ".join(report_data["company_data"]["tags"]),
            key="edit_header_tags"
        )
        new_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
        if new_tags != report_data["company_data"]["tags"]:
            st.session_state.edit_changes["company_data.tags"] = new_tags
        
        # Build tags HTML
        tags_html = ""
        for tag in new_tags:
            tag_class = get_tag_class(tag)
            tags_html += f'<span class="tag {tag_class}">{tag}</span>'
        
        # Display the header
        st.markdown(f"""
        <div class="{css_class}">
            <div class="report-header-wrapper">
                <div class="company-title">{company_name}</div>
                <div class="report-subtitle">{company_desc}</div>
                <div class="tag-container" style="margin-top: 0.5rem;">
                    {tags_html}
                </div>
                <div class="report-metadata">
                    <div class="metadata-item">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                            <line x1="16" y1="2" x2="16" y2="6"></line>
                            <line x1="8" y1="2" x2="8" y2="6"></line>
                            <line x1="3" y1="10" x2="21" y2="10"></line>
                        </svg>
                        <span>Report Date: {report_data["report_metadata"]["report_date"]}</span>
                    </div>
                    <div class="metadata-item">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        <span>Confidence: {report_data["report_metadata"]["data_confidence"]}</span>
                    </div>
                    <div class="metadata-item">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                        </svg>
                        <span>Report ID: {report_data["report_metadata"]["report_id"]}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show disclaimer in a smaller note
        st.markdown(f"""
        <div class="disclaimer">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            <span>{report_data["report_metadata"]["disclaimer"]}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Non-editable display
        company_name = report_data["company_data"]["name"]
        company_desc = report_data["company_data"]["description"]
        
        # Build tags HTML
        tags_html = ""
        for tag in report_data["company_data"]["tags"]:
            tag_class = get_tag_class(tag)
            tags_html += f'<span class="tag {tag_class}">{tag}</span>'
        
        # Display the header
        st.markdown(f"""
        <div class="{css_class}">
            <div class="report-header-wrapper">
                <div class="company-title">{company_name}</div>
                <div class="report-subtitle">{company_desc}</div>
                <div class="tag-container" style="margin-top: 0.5rem;">
                    {tags_html}
                </div>
                <div class="report-metadata">
                    <div class="metadata-item">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                            <line x1="16" y1="2" x2="16" y2="6"></line>
                            <line x1="8" y1="2" x2="8" y2="6"></line>
                            <line x1="3" y1="10" x2="21" y2="10"></line>
                        </svg>
                        <span>Report Date: {report_data["report_metadata"]["report_date"]}</span>
                    </div>
                    <div class="metadata-item">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        <span>Confidence: {report_data["report_metadata"]["data_confidence"]}</span>
                    </div>
                    <div class="metadata-item">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                        </svg>
                        <span>Report ID: {report_data["report_metadata"]["report_id"]}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show disclaimer in a smaller note
        st.markdown(f"""
        <div class="disclaimer">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            <span>{report_data["report_metadata"]["disclaimer"]}</span>
        </div>
        """, unsafe_allow_html=True)


@st.fragment
def render_left_sidebar(report_data):
    """
    Render the left sidebar with project list, navigation, and chat.
    Groups related HTML elements and uses placeholders effectively.
    """
    # === 1. CREATE PLACEHOLDERS for interactive elements ===
    # Create these first to reserve spots in the Streamlit execution flow
    settings_btn_placeholder = st.empty()
    new_project_btn_placeholder = st.empty()
    # Use a dictionary for easier access if needed, or keep as list
    chapter_nav_placeholders = {chapter['id']: st.empty() for chapter in report_data['chapters']}
    chat_input_placeholder = st.empty()
    chat_send_btn_placeholder = st.empty()

    # === 2. BUILD DYNAMIC HTML CONTENT ===

    # --- Build Projects List HTML ---
    projects_list_html = ""
    for i, project in enumerate(report_data['projects']):
        is_active = project['name'] == st.session_state.active_project
        active_class = "active" if is_active else ""
        # Unique key for potential interaction (though click is not implemented here)
        project_key = f"project_{i}"

        # Basic project item structure
        projects_list_html += f'<div class="project-item {active_class}" id="{project_key}">'
        projects_list_html += f'<div class="project-header"><span class="project-name">{project["name"]}</span><span class="project-toggle">{"▼" if is_active else "▶"}</span></div>' # Simple toggle indicator

        # Add files if active
        if is_active and project.get('files'):
            projects_list_html += '<div class="project-files">'
            for file_group in project['files']:
                projects_list_html += f'<div class="file-group"><div class="file-group-header"><span class="file-group-name">{file_group["group"]}</span><span class="file-count">{file_group["count"]}</span></div><ul class="file-list">'
                for item in file_group['items']:
                    status_badge = ""
                    if item['status'] != 'processed':
                        label = "Processing..." if item['status'] == 'processing' else "Queued"
                        status_badge = f'<span class="status-badge {item["status"]}">{label}</span>' # Added status class to badge itself
                    projects_list_html += f'<li class="file-item {item["status"]}"><span class="file-icon">{item["icon"]}</span><span class="file-name">{item["name"]}</span>{status_badge}</li>'
                projects_list_html += '</ul></div>' # Close file-group
            projects_list_html += '</div>' # Close project-files

        projects_list_html += '</div>' # Close project-item
        # NOTE: Clicking on project headers to toggle active state would require JS or more complex Streamlit logic (e.g., buttons within markdown or separate components)

    # --- Build Chat History HTML ---
    chat_messages_html = ""
    for message in st.session_state.chat_history:
        css_class = "user-message" if message["sender"] == "user" else "bot-message"
        # Basic sanitization might be needed here in a real app if messages can contain HTML
        escaped_message = message['message'].replace('<', '<').replace('>', '>')
        chat_messages_html += f'<div class="agent-message {css_class}">{escaped_message}</div>'

    # Add thinking animation if active
    if st.session_state.chat_thinking:
        chat_messages_html += "<div class='agent-thinking'>Thinking<span class='loading-dots'></span></div>"


    # === 3. RENDER STATIC STRUCTURE & FILL PLACEHOLDERS ===

    with st.container(border=False): # Main container for the sidebar

        # --- User Info Section ---
        st.markdown("""
            <div class="sidebar-section user-section">
                <div class="user-info">
                    <div class="user-avatar">PF</div>
                    <div class="user-details">
                        <div class="user-name">ParselyFi Demo</div>
                        <div>Analyst View</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        with settings_btn_placeholder:
            # Add button logic if needed
            st.button("User Settings", key="user_settings_btn", use_container_width=True, type="secondary")

        # --- Projects Section ---
        st.markdown(f"""
            <div class="sidebar-section projects-section">
                <h4>Projects</h4>
                <div class="projects-list">
                    {projects_list_html}
                </div>
            </div>
        """, unsafe_allow_html=True)
        with new_project_btn_placeholder:
            if st.button("+ New Project", key="new_proj_btn", use_container_width=True):
                st.toast("New project creation coming soon!", icon="🏗️")
                # time.sleep(0.5) # Avoid sleep in main thread if possible

        # --- Chapter Navigation Section ---
        st.markdown("""
            <div class="sidebar-section chapter-nav-section">
                <h4>Report Outline</h4>
                <nav class="chapter-nav">
                    <ul>
        """, unsafe_allow_html=True)

        # Fill chapter button placeholders IN ORDER
        for chapter in report_data['chapters']:
            placeholder = chapter_nav_placeholders[chapter['id']]
            with placeholder:
                 # Wrap button in an li for semantic structure - CSS might need adjustment
                st.markdown("<li>", unsafe_allow_html=True)
                if st.button(
                    chapter["title"],
                    key=f"nav_{chapter['id']}",
                    use_container_width=True
                ):
                    # Scroll to chapter using JavaScript
                    components.html( # Use components.html for JS
                        f"""
                        <script>
                            const element = document.getElementById('{chapter["id"]}');
                            if (element) {{
                                element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                                // Optional: Add a brief highlight effect
                                element.style.transition = 'background-color 0.5s ease-in-out';
                                element.style.backgroundColor = 'rgba(0, 123, 255, 0.1)'; // Light blue highlight
                                setTimeout(() => {{ element.style.backgroundColor = ''; }}, 1000);
                            }} else {{
                                console.warn('Element with ID {chapter["id"]} not found for scrolling.');
                            }}
                        </script>
                        """,
                        height=0,
                        width=0,
                    )
                st.markdown("</li>", unsafe_allow_html=True)


        st.markdown("""
                    </ul>
                </nav>
            </div>
        """, unsafe_allow_html=True)

        # --- Agent Chat Section ---
        st.markdown(f"""
            <div class="sidebar-section agent-chat-section">
                <h4>Agent Chat</h4>
                <div class="agent-chat-container">
                    <div class="agent-chat-messages">
                        {chat_messages_html}
                    </div>
                </div>
                <!-- Placeholders for input/send will be filled below -->
            </div>
        """, unsafe_allow_html=True)

        # Fill Chat Input Placeholder (appears visually after the messages)
        with chat_input_placeholder:
            user_message = st.text_area(
                "chat_input_label", # Use a label for accessibility
                key="chat_input_area",
                height=68,
                max_chars=1000,
                label_visibility="collapsed",
                placeholder="Ask about report concepts or chapters..."
            )

        # Fill Chat Send Button Placeholder
        with chat_send_btn_placeholder:
            send_pressed = st.button(
                "Send",
                key="chat_send_btn",
                use_container_width=True,
                type="primary", # Make send button primary
                disabled=st.session_state.chat_thinking # Disable while thinking
            )

        # === 4. HANDLE CHAT LOGIC (Outside main rendering flow) ===
        # Trigger chat processing *after* rendering the button and getting its state
        if send_pressed and user_message and not st.session_state.chat_thinking:
            st.session_state.chat_history.append({"sender": "user", "message": user_message})
            st.session_state.chat_thinking = True
            st.rerun() # Rerun to show user message and thinking indicator

        # Handle bot response if in thinking state (can be triggered by rerun)
        if st.session_state.chat_thinking:
            # Simulate processing delay (replace with actual API call/logic)
            time.sleep(1.5)

            last_message = next((msg["message"] for msg in reversed(st.session_state.chat_history)
                               if msg["sender"] == "user"), None)

            response = "Sorry, I couldn't process that. Please try again." # Default
            if last_message:
                # Simple keyword-based logic (replace with actual NLP/RAG)
                last_message_lower = last_message.lower()
                if "lightrag" in last_message_lower:
                    response = "LightRAG enhances RAG with graph structures for better contextual retrieval and handling complex queries needing connected information."
                elif "graph" in last_message_lower:
                    response = "In LightRAG, graph nodes represent entities and edges represent relationships. This allows traversing connections for more comprehensive answers."
                elif "chapter" in last_message_lower:
                    chapter_num_match = next((i for i in range(1, 4) if str(i) in last_message_lower), None)
                    if chapter_num_match:
                        chapter_content = {
                            1: "the Executive Summary, overviewing LightRAG vs. traditional RAG.",
                            2: "RAG Systems Overview, explaining their function and limitations.",
                            3: "LightRAG Architecture, detailing its graph-based retrieval approach."
                        }
                        response = f"Chapter {chapter_num_match} covers {chapter_content[chapter_num_match]}"
                    else:
                        response = "Which chapter are you asking about (1, 2, or 3)?"
                else:
                    response = "I can help explain LightRAG concepts or chapters. Ask about 'graph structures' or 'Chapter 1', for example."

            # Add bot response to history
            st.session_state.chat_history.append({"sender": "bot", "message": response})
            st.session_state.chat_thinking = False
            # Clear the input field after processing
            st.session_state.chat_input_area = "" # Clear input by resetting its state key
            st.rerun() # Rerun to display bot response and clear thinking indicator


                    
@st.fragment
def render_main_content(report_data):
    """Render the main content area with chapters."""
    editing_mode = st.session_state.editing_mode
    css_class = "edit-mode-active" if editing_mode else ""

    # Create placeholders for chapters - match number of chapters
    chapter_placeholders = [st.empty() for _ in report_data['chapters']]

    # --- Render Opening HTML Structure ---
    html_open = f"""
    <div class="main-content-wrapper {css_class}">
        {f'<div class="edit-indicator">📝 Editing Mode Active</div>' if editing_mode else ''}

        <header class="main-header">
            <h1>{report_data.get("projectName", "Report Analysis")}</h1>
            <p>This interactive report analyzes the LightRAG framework and compares it with traditional RAG systems.</p>
        </header>

        <div class="chapters-container">
            <!-- Chapter content will be inserted sequentially below -->
    """
    st.markdown(html_open, unsafe_allow_html=True)
        
    # Render each chapter in its placeholder
    for chapter, placeholder in zip(report_data['chapters'], chapter_placeholders):
        with placeholder:
            render_chapter(chapter, editing_mode)

@st.fragment
def render_chapter(chapter, editing_mode):
    """Render a single chapter with its content."""
    # Create placeholders for interactive elements
    title_input_placeholder = st.empty()
    source_cols_placeholder = st.empty()
    summary_input_placeholder = st.empty()
    
    # Create placeholders for expandable sections
    qna_expander_placeholder = st.empty()
    entities_expander_placeholder = st.empty()
    relationships_expander_placeholder = st.empty()
    
    if editing_mode:
        # Editable title
        with title_input_placeholder:
            chapter_title = st.text_input(
                "Chapter Title",
                value=chapter["title"],
                key=f"edit_title_{chapter['id']}"
            )
            if chapter_title != chapter["title"]:
                st.session_state.edit_changes[f"chapters.{chapter['id']}.title"] = chapter_title
        
        # Source info fields
        with source_cols_placeholder:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                chapter_source = st.text_input(
                    "Source",
                    value=chapter["source"],
                    key=f"edit_source_{chapter['id']}"
                )
            with col2:
                chapter_type = st.text_input(
                    "Type",
                    value=chapter["type"],
                    key=f"edit_type_{chapter['id']}"
                )
            with col3:
                chapter_pages = st.text_input(
                    "Pages",
                    value=chapter.get("pages", ""),
                    key=f"edit_pages_{chapter['id']}"
                )
            
            # Update changes
            if chapter_source != chapter["source"]:
                st.session_state.edit_changes[f"chapters.{chapter['id']}.source"] = chapter_source
            if chapter_type != chapter["type"]:
                st.session_state.edit_changes[f"chapters.{chapter['id']}.type"] = chapter_type
            if chapter.get("pages", "") != chapter_pages:
                st.session_state.edit_changes[f"chapters.{chapter['id']}.pages"] = chapter_pages
        
        # Display formatted header with proper HTML structure
        st.markdown(f"""
            <article class="chapter" id="{chapter['id']}">
                <h3>{chapter_title}</h3>
                <p class="source-info">Source: {chapter_source} | Type: {chapter_type}
                {" | Pages: " + chapter_pages if chapter_pages else ""}</p>
            </article>
        """, unsafe_allow_html=True)
        
        # Editable summary
        with summary_input_placeholder:
            summary = st.text_area(
                "Summary",
                value=chapter["summary"],
                key=f"edit_summary_{chapter['id']}",
                height=200
            )
            if summary != chapter["summary"]:
                st.session_state.edit_changes[f"chapters.{chapter['id']}.summary"] = summary
        
        # Display with markdown formatting in a single container
        st.markdown(f"""
            <div class="summary">
                {enhance_markdown(summary)}
            </div>
        """, unsafe_allow_html=True)
    else:
        # Static display with complete container
        # Process summary to add interactive keyword spans
        enhanced_summary = process_keywords_in_text(chapter["summary"], st.session_state.report_data["keyword_data"])
        
        # Render the entire chapter with proper HTML structure
        st.markdown(f"""
            <article class="chapter" id="{chapter['id']}">
                <h3>{chapter["title"]}</h3>
                <p class="source-info">Source: {chapter["source"]} | Type: {chapter["type"]}
                {" | Pages: " + chapter["pages"] if chapter.get("pages") else ""}</p>
                
                <div class="summary">
                    {enhance_markdown(enhanced_summary)}
                </div>
            </article>
        """, unsafe_allow_html=True)
    
    # Visual placeholder for chapter 1 and 2
    if chapter['id'] in ['chapter1', 'chapter2']:
        display_visual_placeholder(chapter['id'])
    
    # Expandable sections
    with qna_expander_placeholder.expander("Q&A", expanded=False):
        render_chapter_qna(chapter, editing_mode)
    
    with entities_expander_placeholder.expander("Key Entities", expanded=False):
        render_chapter_entities(chapter, editing_mode)
    
    with relationships_expander_placeholder.expander("Key Relationships", expanded=False):
        render_chapter_relationships(chapter, editing_mode)

@st.fragment
def render_chapter_qna(chapter, editing_mode):
    """Render the Q&A section for a chapter."""
    if editing_mode:
        # Editable Q&A items
        st.markdown("### Edit Q&A Items")
        qna_items = chapter.get("qna", [])
        
        # Display each Q&A with edit controls
        for i, item in enumerate(qna_items):
            cols = st.columns([2, 3, 1])
            with cols[0]:
                question = st.text_input(
                    "Question",
                    value=item["q"],
                    key=f"edit_q_{chapter['id']}_{i}"
                )
            with cols[1]:
                answer = st.text_area(
                    "Answer",
                    value=item["a"],
                    key=f"edit_a_{chapter['id']}_{i}",
                    height=80
                )
            with cols[2]:
                reference = st.text_input(
                    "Reference",
                    value=item.get("ref", ""),
                    key=f"edit_ref_{chapter['id']}_{i}"
                )
            
            # Delete button
            if st.button("Remove Q&A", key=f"delete_qna_{chapter['id']}_{i}"):
                st.session_state.edit_changes[f"chapters.{chapter['id']}.qna.delete_{i}"] = True
                st.rerun()
            
            st.markdown("---")
        
        # Add new Q&A button
        if st.button("+ Add Q&A Item", key=f"add_qna_{chapter['id']}"):
            if "qna" not in chapter:
                st.session_state.edit_changes[f"chapters.{chapter['id']}.qna.new"] = [{"q": "New question", "a": "New answer", "ref": ""}]
            else:
                st.session_state.edit_changes[f"chapters.{chapter['id']}.qna.new"] = {"q": "New question", "a": "New answer", "ref": ""}
            st.rerun()
    else:
        # Static Q&A display
        if chapter.get("qna"):
            st.markdown('<div class="details-content">', unsafe_allow_html=True)
            for item in chapter["qna"]:
                ref_html = f'<span class="source-reference" title="Source: {item["ref"]}">{item["ref"]}</span>' if item.get("ref") else ''
                st.markdown(f'<div class="qna-item"><strong>Q:</strong> {item["q"]}<br><strong>A:</strong> {item["a"]} {ref_html}</div>', unsafe_allow_html=True)
                st.markdown("<hr style='margin: 10px 0; border: 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="details-content"><p>No Q&A items available for this chapter.</p></div>', unsafe_allow_html=True)

@st.fragment
def render_chapter_entities(chapter, editing_mode):
    """Render the entities section for a chapter."""
    if editing_mode:
        # Editable entities with standard Streamlit controls
        st.markdown("### Edit Entity Nodes")
        entities = chapter.get("entities", [])
        
        # Display each entity with edit controls
        for i, entity in enumerate(entities):
            cols = st.columns([3, 2])
            with cols[0]:
                key = st.text_input(
                    "Entity Name",
                    value=entity["key"],
                    key=f"edit_entity_key_{chapter['id']}_{i}"
                )
            with cols[1]:
                entity_type = st.selectbox(
                    "Entity Type",
                    options=[
                        "System/Concept", 
                        "Technology/Model", 
                        "Data/Resource",
                        "Data Structure/Concept",
                        "Limitation/Concept",
                        "Proposed System/Framework",
                        "Other"
                    ],
                    index=[
                        "System/Concept", 
                        "Technology/Model", 
                        "Data/Resource",
                        "Data Structure/Concept",
                        "Limitation/Concept",
                        "Proposed System/Framework",
                    ].index(entity["type"]) if entity["type"] in [
                        "System/Concept", 
                        "Technology/Model", 
                        "Data/Resource",
                        "Data Structure/Concept",
                        "Limitation/Concept",
                        "Proposed System/Framework",
                    ] else 6,
                    key=f"edit_entity_type_{chapter['id']}_{i}"
                )
            
            # Delete entity button
            if st.button("Remove Entity", key=f"delete_entity_{chapter['id']}_{i}"):
                st.session_state.edit_changes[f"chapters.{chapter['id']}.entities.delete_{i}"] = True
                st.rerun()
            
            st.markdown("---")
    else:
        # Static entity display in a single container
        if chapter.get("entities"):
            entities_html = ""
            for entity in chapter["entities"]:
                entities_html += f"""
                    <div class="entity-node" data-type="{entity['type']}" data-keyword="{entity['key']}">
                        <span class="key">{entity['key']}</span>
                        <span class="type">({entity['type']})</span>
                    </div>
                """
            
            st.markdown(f"""
                <div class="details-content">
                    <div class="entity-nodes-container">
                        {entities_html}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="details-content">
                    <p>No entities defined for this chapter.</p>
                </div>
            """, unsafe_allow_html=True)
                        
@st.fragment
def render_chapter_relationships(chapter, editing_mode):
    """Render the relationships section for a chapter."""
    if editing_mode:
        # Editable relationships
        st.markdown("### Edit Relationships")
        relationships = chapter.get("relationships", [])
        
        # Display each relationship with edit controls
        for i, relation in enumerate(relationships):
            cols = st.columns([2, 2, 2])
            with cols[0]:
                source = st.text_input(
                    "Source Entity",
                    value=relation["source"],
                    key=f"edit_rel_source_{chapter['id']}_{i}"
                )
            with cols[1]:
                target = st.text_input(
                    "Target Entity",
                    value=relation["target"],
                    key=f"edit_rel_target_{chapter['id']}_{i}"
                )
            with cols[2]:
                description = st.text_input(
                    "Relationship Type",
                    value=relation["desc"],
                    key=f"edit_rel_desc_{chapter['id']}_{i}"
                )
            
            # Delete relationship button
            if st.button("Remove Relationship", key=f"delete_rel_{chapter['id']}_{i}"):
                st.session_state.edit_changes[f"chapters.{chapter['id']}.relationships.delete_{i}"] = True
                st.rerun()
            
            st.markdown("---")
    else:
        # Static relationship display in a single container
        if chapter.get("relationships"):
            relationships_html = ""
            for relation in chapter["relationships"]:
                # Create the relationship display with clickable entities
                keys_html = ' '.join([f'<code>{k}</code>' for k in relation.get("keys", [])])
                
                relationships_html += f"""
                    <li class="relationship-item">
                        <span class="rel-source keyword-expandable" data-keyword="{relation['source']}">{relation['source']}</span> →
                        <span class="rel-target keyword-expandable" data-keyword="{relation['target']}">{relation['target']}</span>
                        <span class="rel-description"><strong>{relation['desc']}:</strong> 
                            Relationship between these concepts affects how information is processed and retrieved.
                        </span>
                        <div class="rel-keys">Keys: {keys_html}</div>
                    </li>
                """
            
            st.markdown(f"""
                <div class="details-content">
                    <ul class="relationships-list">
                        {relationships_html}
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="details-content">
                    <p>No relationships defined for this chapter.</p>
                </div>
            """, unsafe_allow_html=True)

@st.fragment
# --- Glossary Rendering Fragment (Keep as is, it handles its internal HTML well) ---
@st.fragment
def render_glossary_terms(terms, filter_text=""):
    """Render the glossary terms with proper containerization."""
    if filter_text:
        # Simple case-insensitive filtering
        terms = [term for term in terms if filter_text.lower() in term.lower()]

    # Group terms by type for better organization (can be pre-processed)
    term_types = {
        "Framework": ["LightRAG", "Retrieval-Augmented Generation (RAG)"],
        "Models": ["Large Language Models (LLMs)"],
        "Data Structures": ["Graph Structures", "Knowledge Graph", "Vector Database", "Flat Data Representations"],
        "Concepts": ["External Knowledge Sources", "Graph Traversal", "Multi-hop Questions", "Entity Extraction"]
    }

    # Build HTML for terms by category
    glossary_html_content = ""
    found_terms = False

    all_known_terms_in_data = set(terms) # Use set for faster lookup

    for category, category_terms in term_types.items():
        # Filter terms within the category that are actually present in the report data AND match the filter
        matching_terms_in_category = [term for term in category_terms if term in all_known_terms_in_data]

        if matching_terms_in_category:
            category_html = f"<h6 style='font-size: 0.85rem; margin-top: 10px; margin-bottom: 5px; color: #666;'>{category}</h6>"
            term_list_html = ""
            for term in matching_terms_in_category:
                 # Add data-keyword attribute for potential JS interaction
                term_list_html += f"""<li><a href="#" data-keyword="{term}" class="glossary-term-link">{term}</a></li>""" # Added class

            if term_list_html: # Only add category if it has terms
                glossary_html_content += category_html + term_list_html
                found_terms = True

    # Add remaining terms that didn't fit a category
    categorized_terms = set(t for cat_terms in term_types.values() for t in cat_terms)
    uncategorized_terms = [term for term in terms if term not in categorized_terms]
    if uncategorized_terms:
         glossary_html_content += f"<h6 style='font-size: 0.85rem; margin-top: 10px; margin-bottom: 5px; color: #666;'>Other</h6>"
         for term in uncategorized_terms:
              glossary_html_content += f"""<li><a href="#" data-keyword="{term}" class="glossary-term-link">{term}</a></li>"""
         found_terms = True

    if not found_terms and filter_text:
         glossary_html_content = "<p style='font-size: 0.8rem; color: #888;'>No matching terms found.</p>"
    elif not terms:
         glossary_html_content = "<p style='font-size: 0.8rem; color: #888;'>No glossary terms available.</p>"


    # Render the complete glossary structure
    st.markdown(f"""
    <div class="glossary-list">
        <ul style="list-style: none; padding-left: 5px; margin: 0;">
            {glossary_html_content}
        </ul>
    </div>
    <script>
    // Optional: Add JS to handle clicks on glossary terms if needed
    document.querySelectorAll('.glossary-term-link').forEach(link => {{
        link.addEventListener('click', function(event) {{
            event.preventDefault(); // Prevent default link behavior
            const keyword = this.getAttribute('data-keyword');
            // Dispatch custom event or call a Streamlit callback via JS->Python
            console.log('Glossary term clicked:', keyword);
            // Example: Trigger the same popup as search results
             window.dispatchEvent(new CustomEvent('showKeywordPopup', {{
                detail: {{
                    keyword: keyword,
                    event: {{ // Simulate event data for popup positioning
                        target: this,
                        pageX: event.pageX,
                        pageY: event.pageY
                    }}
                }}
            }}));
        }});
    }});
    </script>
    """, unsafe_allow_html=True)


# --- Right Sidebar Rendering Function ---
@st.fragment
def render_right_sidebar(report_data):
    """Render the right sidebar with knowledge graph and related info."""
    # Use Streamlit's container, no need for manual wrapper div
    with st.container(border=False):

        # --- Knowledge Graph ---
        # Use markdown for styled header, then widget
        st.markdown("<h5 class='sidebar-section-header'>Knowledge Graph</h5>", unsafe_allow_html=True)

        # Instantiate your custom widget
        kg_widget = KnowledgeGraphWidget()
        # Safely get data, providing defaults
        graph_data = report_data.get('graph_data', {})
        kg_widget.nodes = graph_data.get('nodes', [])
        kg_widget.links = graph_data.get('links', [])

        # Render knowledge graph widget using anywidget
        graph_result = anywidget(kg_widget, key="kg_widget")

        # Handle interaction results (optional)
        if graph_result and graph_result.get('type') == 'node_click':
            node_id = graph_result.get('id')
            st.toast(f"Exploring: {node_id}") # Example action
        st.markdown("---") # Add a visual separator

        # --- Search Digest ---
        st.markdown("<h5 class='sidebar-section-header'>Search Digest</h5>", unsafe_allow_html=True)
        search_query = st.text_input(
            "Search keywords or text...",
            key="search_input",
            placeholder="e.g., 'Graph Structures'"
        )

        # Process search results
        if search_query:
            results = []
            keyword_data = report_data.get('keyword_data', {})
            for keyword, data in keyword_data.items():
                # Improved search matching (keyword OR definition)
                if (search_query.lower() in keyword.lower() or
                    (data.get('definition') and search_query.lower() in data['definition'].lower())):
                    results.append({
                        "keyword": keyword,
                        "definition": data.get('definition', 'No definition available.'),
                        "source": data.get('source', 'Unknown source')
                    })

            # Display results
            if results:
                st.success(f"Found {len(results)} match(es) for '{search_query}'")
                # Limit displayed results if needed
                for result in results[:5]: # Show max 5 results for brevity
                    with st.container(border=True):
                        st.markdown(f"**{result['keyword']}**")
                        # Use st.caption for definition, truncate safely
                        definition_preview = result['definition'][:100] + ('...' if len(result['definition']) > 100 else '')
                        st.caption(definition_preview)
                        # Use st.text or st.caption for source
                        st.caption(f"Source: {result['source']}")

                        # Button to trigger keyword details (e.g., via JS popup)
                        # Ensure unique key for button
                        button_key = f"view_{result['keyword'].replace(' ','_').replace('(','').replace(')','')}" # Make key safer
                        if st.button("View Details", key=button_key, type="secondary", use_container_width=True):
                             # --- FIX START ---
                             # 1. Escape the keyword *outside* the f-string expression part
                             escaped_keyword = result['keyword'].replace('"', '\\"') # Creates JS-safe string

                             # 2. Use the simple variable inside the f-string
                             js_code = f"""
                                <script>
                                    console.log('Dispatching showKeywordPopup for: {escaped_keyword}');
                                    window.dispatchEvent(new CustomEvent('showKeywordPopup', {{
                                        detail: {{
                                            keyword: "{escaped_keyword}", // Use the prepared variable
                                            event: {{ // Fake event data for positioning
                                                target: document.body, // Or find a better anchor
                                                pageX: window.innerWidth * 0.75, // Position towards right
                                                pageY: window.innerHeight / 4
                                            }}
                                        }}
                                    }}));
                                </script>
                                """
                             # --- FIX END ---

                             st.components.v1.html(js_code, height=0) # Pass the constructed JS code

            else:
                st.info(f"No matching results found for '{search_query}'.")
        st.markdown("---")

        # --- Filters (Example - if needed) ---
        # st.markdown("<h5 class='sidebar-section-header'>Filter by Source</h5>", unsafe_allow_html=True)
        # source_filter = st.radio(
        #     "Select source type:", # Provide a visible label
        #     options=["All", "PDF", "Web", "Code"],
        #     horizontal=True,
        #     # label_visibility="collapsed", # Consider keeping label visible
        #     key="source_filter"
        # )
        # if source_filter != "All":
        #     st.info(f"Filtering logic for {source_filter} sources would go here.") # Placeholder
        # st.markdown("---")

        # --- Glossary / Entities ---
        st.markdown("<h5 class='sidebar-section-header'>Glossary / Entities</h5>", unsafe_allow_html=True)
        glossary_filter = st.text_input(
            "Filter terms:",
            key="glossary_filter",
            placeholder="Filter glossary terms..."
        )

        # Render glossary using the dedicated fragment
        render_glossary_terms(report_data.get('glossary_terms', []), glossary_filter)
        st.markdown("---")


        # --- Export Options ---
        st.markdown("<h5 class='sidebar-section-header'>Export Options</h5>", unsafe_allow_html=True)

        # Example: Export PDF button (logic external)
        if st.button("📄 Export Report as PDF", key="export_pdf", use_container_width=True):
            st.toast("PDF export functionality coming soon!", icon="📄")

        # Correct usage of st.download_button
        try:
            # Prepare data ONCE, before the button
            json_str = json.dumps(report_data, indent=2, default=str) # Use default=str for non-serializable types like datetime
            file_name = f"parselyfi_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

            st.download_button(
                label="💾 Download Data (JSON)", # Label for the button itself
                data=json_str,
                file_name=file_name,
                mime="application/json",
                key="download_json_btn", # Key for the download button widget
                use_container_width=True # Make button full width
            )
        except Exception as e:
            st.error(f"Error preparing JSON data: {e}")


        # Example: Audio Summary button
        if st.button("🔊 Generate Audio Summary", key="export_audio", use_container_width=True):
            st.toast("Audio summary generation coming soon!", icon="🔊")
        
@st.fragment
def render_footer(report_data):
    """Render the report footer with proper containerization."""
    disclaimer = report_data["report_metadata"]["disclaimer"]
    
    st.markdown(f"""
    <div class="footer">
        <p>ParselyFi Digest | Generated: {report_data["report_metadata"]["report_date"]}</p>
        <p style="margin-top: 0.5rem;">{disclaimer}</p>
        <p style="margin-top: 0.5rem;">{datetime.now().year} ParselyFi Demo</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Helper Functions
# -----------------------------------------------------------------------------

def get_tag_class(tag):
    """Return the appropriate CSS class for a tag based on content."""
    if "AI" in tag or "Model" in tag:
        return "purple-tag"
    elif "RAG" in tag or "Framework" in tag:
        return "blue-tag"
    elif "Series" in tag or "Data" in tag:
        return "green-tag"
    else:
        return "yellow-tag"

def get_confidence_color(confidence):
    """Return color for confidence level."""
    if confidence == "High":
        return "#059669"  # Green
    elif confidence == "Medium":
        return "#d97706"  # Amber
    else:
        return "#dc2626"  # Red

def process_keywords_in_text(text, keyword_data):
    """Process text to add keyword-expandable spans for known terms."""
    # Sort keywords by length (descending) to handle longer phrases first
    sorted_keywords = sorted(keyword_data.keys(), key=len, reverse=True)
    
    for keyword in sorted_keywords:
        # Use a simpler approach for the prototype - in production, use regex with word boundaries
        if keyword in text:
            text = text.replace(
                keyword, 
                f'<span class="keyword-expandable" data-keyword="{keyword}">{keyword}</span>'
            )
    
    return text

def enhance_markdown(text):
    """Enhance markdown text with additional formatting."""
    # Keep bold formatting
    text = text.replace("**", "<strong>", 1)
    if "**" in text:
        text = text.replace("**", "</strong>", 1)
        while "**" in text:
            text = text.replace("**", "<strong>", 1)
            if "**" in text:
                text = text.replace("**", "</strong>", 1)
    
    # Convert newlines to breaks
    text = text.replace('\n', '<br>')
    
    return text

def display_visual_placeholder(chapter_id):
    """Display visual placeholder for specific chapters with proper containerization."""
    if chapter_id == 'chapter1':
        # Bar chart comparison placeholder
        st.markdown("""
        <div class="visuals-placeholder">
            <svg width="400" height="200" xmlns="http://www.w3.org/2000/svg">
                <rect x="50" y="20" width="50" height="150" fill="#376f4e" />
                <rect x="150" y="60" width="50" height="110" fill="#3a7ca5" />
                <rect x="250" y="100" width="50" height="70" fill="#f58a07" />
                <text x="75" y="190" text-anchor="middle" font-size="12">LightRAG</text>
                <text x="175" y="190" text-anchor="middle" font-size="12">Standard RAG</text>
                <text x="275" y="190" text-anchor="middle" font-size="12">Baseline</text>
                <text x="200" y="15" text-anchor="middle" font-size="14" font-weight="bold">Performance Comparison</text>
            </svg>
        </div>
        """, unsafe_allow_html=True)
    elif chapter_id == 'chapter2':
        # RAG system diagram placeholder
        st.markdown("""
        <div class="visuals-placeholder">
            <svg width="500" height="200" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
                    </marker>
                </defs>
                <rect x="50" y="70" width="80" height="60" rx="5" fill="#e1f5fe" stroke="#3a7ca5" />
                <text x="90" y="105" text-anchor="middle" font-size="12">Query</text>
                <rect x="200" y="70" width="100" height="60" rx="5" fill="#e8f5e9" stroke="#376f4e" />
                <text x="250" y="105" text-anchor="middle" font-size="12">Vector DB</text>
                <rect x="370" y="70" width="80" height="60" rx="5" fill="#fff3e0" stroke="#f58a07" />
                <text x="410" y="105" text-anchor="middle" font-size="12">LLM</text>
                <line x1="130" y1="100" x2="200" y2="100" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)" />
                <line x1="300" y1="100" x2="370" y2="100" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)" />
                <text x="165" y="90" text-anchor="middle" font-size="10">Retrieval</text>
                <text x="335" y="90" text-anchor="middle" font-size="10">Augmentation</text>
                <text x="250" y="30" text-anchor="middle" font-size="14" font-weight="bold">Standard RAG Architecture</text>
            </svg>
        </div>
        """, unsafe_allow_html=True)

def apply_edit_changes():
    """Apply accumulated edit changes to the report data."""
    if not st.session_state.edit_changes:
        return
    
    changes = st.session_state.edit_changes
    report_data = st.session_state.report_data
    
    for path, value in changes.items():
        # Handle simple field updates
        if "." in path and not any(x in path for x in ["delete_", "new"]):
            parts = path.split(".")
            data = report_data
            
            # Navigate to the parent object
            for part in parts[:-1]:
                if part not in data:
                    data[part] = {}
                data = data[part]
            
            # Set the value
            data[parts[-1]] = value
        
        # Handle chapter entity/relationship additions
        elif ".new" in path:
            base_path = path.split(".new")[0]
            parts = base_path.split(".")
            data = report_data
            
            # Navigate to the parent object
            for part in parts[:-1]:
                if part not in data:
                    data[part] = {}
                data = data.get(part, {})
            
            # Get the array to modify
            array_name = parts[-1]
            if array_name not in data:
                data[array_name] = []
            
            # Add the new item
            if isinstance(data[array_name], list):
                data[array_name].append(value)
            else:
                data[array_name] = [value]
        
        # Handle chapter entity/relationship deletions
        elif "delete_" in path:
            base_path, delete_part = path.split(".delete_")
            delete_index = int(delete_part)
            parts = base_path.split(".")
            data = report_data
            
            # Navigate to the parent object
            for part in parts[:-1]:
                if part not in data:
                    data[part] = {}
                data = data.get(part, {})
            
            # Get the array to modify
            array_name = parts[-1]
            if array_name in data and isinstance(data[array_name], list) and delete_index < len(data[array_name]):
                # Remove the item
                data[array_name].pop(delete_index)
    
    # Clear the changes
    st.session_state.edit_changes = {}
    
    # Show success message
    st.success("Changes applied successfully!")

# -----------------------------------------------------------------------------
# 7. Main Application
# -----------------------------------------------------------------------------

def main():
    """Main application entry point."""
    # Render Keyword Popup Widget (must be rendered once)
    kw_widget = KeywordPopupWidget()
    popup_result = anywidget(kw_widget, key="kw_widget")
    
    # Handle any results returned from the popup
    if popup_result and popup_result.get('type') == 'node_click':
        # Node was clicked - could trigger actions based on this
        node_id = popup_result.get('id')
        st.toast(f"Exploring: {node_id}")
    
    # Main layout with three columns
    col_left, col_main, col_right = st.columns([2.6, 6, 2.8])
    
    with col_left:
        # Render left sidebar
        render_left_sidebar(st.session_state.report_data)
    
    with col_main:
        # Edit mode toggle
        edit_col1, edit_col2 = st.columns([6, 1])
        with edit_col1:
            st.write("")  # Empty space for alignment
        with edit_col2:
            edit_mode_toggle = st.toggle(
                "Edit Mode", 
                value=st.session_state.editing_mode,
                key="edit_mode_toggle"
            )
            
            # Update mode if changed
            if edit_mode_toggle != st.session_state.editing_mode:
                st.session_state.editing_mode = edit_mode_toggle
                # Reset changes when toggling edit mode off
                if not edit_mode_toggle:
                    st.session_state.edit_changes = {}
                st.rerun()
        
        # Save changes button when in edit mode
        if st.session_state.editing_mode and st.session_state.edit_changes:
            save_col1, save_col2 = st.columns([3, 2])
            with save_col1:
                st.info(f"{len(st.session_state.edit_changes)} pending changes")
            with save_col2:
                if st.button("💾 Save Changes", use_container_width=True):
                    apply_edit_changes()
                    st.rerun()
        
        # Render main content
        render_header(st.session_state.report_data)
        render_main_content(st.session_state.report_data)
    
    with col_right:
        # Render right sidebar
        render_right_sidebar(st.session_state.report_data)
    
    # Footer (outside columns)
    render_footer(st.session_state.report_data)

if __name__ == "__main__":
    main()