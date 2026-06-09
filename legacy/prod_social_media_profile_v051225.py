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
MAX_PILLS_ON_CARD = 7 # Increased to show more on the main card
MAX_ITEMS_PER_SECTION_ON_CARD = 2
MAX_DESC_LINES_ON_CARD = 3 # Slightly more description
MAX_PILLS_FOR_WALLET_PASS = 10
MAX_PILLS_FOR_SOCIAL_PNG = 10

# --- NEW: Data for Deep Popouts ---
# This dictionary will store the content for the hover popouts.
# Keys should match the skill/interest names from your profile_data.
DEEP_DIVE_INFO = {
    "Generative AI": {
        "detail_title": "Generative AI Focus",
        "detail_snippet": "Exploring cutting-edge models for content creation, problem-solving, and workflow automation.",
        "inner_tags": {
            "Deep Learning": "Neural networks with many layers, enabling complex pattern recognition.",
            "Creative Tech": "Using technology for artistic and creative expression and generation.",
            "LLMs": "Large Language Models are a core component of many Generative AI systems."
        }
    },
    "Large Language Models (LLMs)": { # Exact match to skill in JSON
        "detail_title": "LLM Expertise",
        "detail_snippet": "Building, fine-tuning, and deploying LLMs for diverse applications, including RAG and multi-agent systems.",
        "inner_tags": {
            "NLP": "Natural Language Processing: Enabling computers to understand human language.",
            "RAG": "Retrieval Augmented Generation: Enhancing LLMs with external knowledge.",
            "Transformers": "A type of neural network architecture, central to modern LLMs.",
            "Fine-tuning": "Adapting pre-trained LLMs to specific tasks or datasets."
        }
    },
    "FinTech": {
        "detail_title": "FinTech Innovation",
        "detail_snippet": "Developing AI-driven tools for financial research, risk management, and workflow automation.",
        "inner_tags": {
            "WealthTech": "Technology solutions for wealth management and investment.",
            "RegTech": "Technology to enhance regulatory processes and compliance.",
            "Algorithmic Trading": "Using computer programs to execute trades at high speeds."
        }
    },
    "HealthTech": {
        "detail_title": "HealthTech Solutions",
        "detail_snippet": "Applying AI for medical code matching, real-time transcription, and improving healthcare processes.",
        "inner_tags": {
            "Digital Health": "Use of digital tech for health, healthcare, and wellness.",
            "MedAI": "Artificial Intelligence applications specifically in medicine.",
            "HIPAA Compliance": "Ensuring patient data privacy and security in tech solutions."
        }
    },
    "Python (Programming Language)": {
        "detail_title": "Python Proficiency",
        "detail_snippet": "Extensive experience using Python for data science, machine learning, web development (Flask, FastAPI), and automation.",
        "inner_tags": {
            "Pandas (Software)": "Data manipulation and analysis library.",
            "Streamlit": "Building interactive web applications for data science.",
            "Flask/FastAPI": "Web frameworks for API development."
        }
    },
    "Data Analysis": {
        "detail_title": "Data Analysis Expertise",
        "detail_snippet": "Extracting insights from complex datasets, data quality checks, and visualization.",
        "inner_tags": {
            "Pandas (Software)": "Core library for data manipulation in Python.",
            "SQL": "Querying and managing relational databases.",
            "Data Visualization": "Tools like Matplotlib, Seaborn, Plotly."
        }
    },
    "Machine Learning": {
        "detail_title": "Machine Learning Application",
        "detail_snippet": "Developing and deploying ML models for classification, regression, and clustering tasks.",
        "inner_tags": {
            "Supervised Learning": "Training models on labeled data.",
            "Unsupervised Learning": "Finding patterns in unlabeled data.",
            "Model Evaluation": "Assessing the performance of ML models."
        }
    },
    "Cloud Computing": {
        "detail_title": "Cloud Platform Experience",
        "detail_snippet": "Deploying and managing applications on AWS, GCP, and Azure.",
        "inner_tags": {
            "AWS": "Amazon Web Services.",
            "GCP": "Google Cloud Platform.",
            "Azure": "Microsoft Azure."
        }
    },
    "Amazon Web Services (AWS)": {
        "detail_title": "AWS Expertise",
        "detail_snippet": "Utilizing services like EC2, S3, Lambda, SageMaker for scalable cloud solutions.",
        "inner_tags": {} # Can add more specific AWS services
    },
    "Google Cloud Platform (GCP)": {
        "detail_title": "GCP Expertise",
        "detail_snippet": "Leveraging Vertex AI, BigQuery, Cloud Run, and GKE for AI/ML and application deployment.",
        "inner_tags": {}
    },
    "Microsoft Azure": {
        "detail_title": "Azure Expertise",
        "detail_snippet": "Working with Azure Machine Learning, Azure Functions, and other PaaS/IaaS offerings.",
        "inner_tags": {}
    },
    "Docker": { # Assuming 'Docker' might be a skill name
        "detail_title": "Docker Containerization",
        "detail_snippet": "Using Docker to build, ship, and run applications in isolated containers for consistent environments.",
        "inner_tags": {
            "Microservices": "Architecting applications as a collection of small, independent services.",
            "CI/CD": "Continuous Integration and Continuous Deployment pipelines."
        }
    },
    "RAG (Retrieval Augmented Generation)": { # Skill from FinAdvizly experience
        "detail_title": "RAG Implementation",
        "detail_snippet": "Enhancing LLMs with external, up-to-date knowledge bases for more accurate and contextual responses.",
        "inner_tags": {
            "Vector Databases": "Storing and querying embeddings for semantic search.",
            "LLMs": "Core component, augmented by retrieved information."
        }
    },
     "Streamlit": {
        "detail_title": "Streamlit Development",
        "detail_snippet": "Rapidly building and deploying interactive web applications for data science and AI projects.",
        "inner_tags": {
            "Python (Programming Language)": "Streamlit is a Python library.",
            "Data Visualization": "Often used to display insights from data."
        }
    }
    # Add more entries as needed, ensure keys match skill names in your JSON data
}


# --- CSS Styling ---
CSS_STYLES = f"""
<style>
:root {{
    /* --- Coffee Card Theme Palette (existing from your app) --- */
    --cc-bg-main: #fff8f0;
    --cc-bg-exp: #fffaf5;
    --cc-bg-tag: #ffe8d6;
    --cc-bg-progress-track: #eee;
    --cc-bg-btn-delete-hover: #fdf7f7;
    --cc-bg-btn-default-hover: #fff0e0;

    --cc-accent-dark-brown: #6b4f4f;
    --cc-accent-theme-brown: #b36b00;
    --cc-accent-light-tan: #e6ccb2;

    --cc-text-name: #4a3f3f;
    --cc-text-title: #555;
    --cc-text-exp: #333;
    --cc-text-tag: var(--cc-accent-dark-brown);
    --cc-text-progress: #555;
    --cc-text-placeholder: #999;
    --cc-text-missing-summary: #666;
    --cc-text-general: #37352F;
    --cc-text-secondary: #6B6B6B;

    --cc-btn-delete-text: #d9534f;
    --cc-btn-delete-hover-text: #c9302c;
    --cc-btn-delete-hover-border: #ac2925;
    --cc-btn-default-text: var(--cc-accent-theme-brown);
    --cc-btn-default-hover-text: #8a5a00;
    --cc-btn-default-hover-border: #8a5a00;

    /* --- NEW: Variables for Popouts --- */
    --cc-bg-detail-popout: #fdf6ec; /* Specific for 2nd level popout */
    --cc-bg-deep-detail-popout: #faf0e6; /* Specific for 3rd level popout */
    --cc-text-header: var(--cc-accent-dark-brown);
    --cc-border-color: var(--cc-accent-dark-brown);
    --cc-shadow-color: rgba(0,0,0,0.1);
    --cc-link-color: #8c6d62;
}}

/* Card styling - Apply Coffee Card Theme to our generated card */
.coffee-card-generated {{
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
}}

.coffee-card-generated:hover {{
    transform: translateY(-3px);
    box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 12px;
}}

.coffee-card-generated::before {{
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
}}

.cc-card-content {{
    padding-top: 8px;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}}

.cc-header-content {{
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1rem;
}}

img.cc-avatar {{
    width: 80px;
    height: 80px;
    border-radius: 50%;
    border: 2px solid var(--cc-accent-light-tan);
    object-fit: cover;
    flex-shrink: 0;
    box-shadow: rgba(0, 0, 0, 0.05) 0px 2px 4px;
}}

.cc-header-text h1.cc-name {{
    color: var(--cc-text-name);
    margin: 0 0 0.2rem 0;
    font-size: 1.4em;
    font-weight: bold;
}}

.cc-header-text p.cc-title {{
    font-size: 1em;
    color: var(--cc-text-title);
    margin: 0.2rem 0 0.5rem 0;
    line-height: 1.4;
}}
.cc-header-text p.cc-tagline {{
    font-size: 0.9em;
    color: var(--cc-text-secondary);
    margin: 0.3rem 0 0.5rem 0;
    font-style: italic;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 3; /* Allow 3 lines for tagline */
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
    max-height: calc(1.3em * 3);
}}
.cc-header-text p.cc-location {{
    font-size: 0.85em;
    color: var(--cc-text-secondary);
    margin: 0.1rem 0 0.3rem 0;
    line-height: 1.3;
}}
.cc-header-text p.cc-profile-url a {{
    font-size: 0.8em;
    color: var(--cc-accent-theme-brown);
    text-decoration: none;
}}
.cc-header-text p.cc-profile-url a:hover {{
    text-decoration: underline;
}}

.cc-section {{
    margin-bottom: 0.8rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--cc-accent-light-tan);
}}
.cc-card-content > .cc-section:last-of-type {{
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}}

h5.cc-section-header {{
    color: var(--cc-accent-dark-brown);
    font-weight: bold;
    font-size: 0.9em;
    margin-bottom: 0.4rem;
    margin-top: 0;
    display: flex;
    align-items: center;
}}
.cc-icon {{ margin-right: 0.4rem; opacity: 0.9; font-size: 1em; }}

.cc-pill-container {{
    margin-top: 0.3rem;
    margin-bottom: 0.3rem;
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    min-height: auto;
    width: 100%;
}}

.cc-pill {{
    display: inline-block;
    padding: 0.25rem 0.7rem;
    margin: 0;
    border-radius: 10px;
    background: var(--cc-bg-tag);
    font-size: 0.75em;
    color: var(--cc-text-tag);
    font-weight: 500;
    line-height: 1.2;
    word-break: break-word;
    overflow-wrap: break-word;
    max-width: 100%;
    white-space: normal;
    border: 1px solid var(--cc-accent-light-tan);
    transition: background-color 0.2s ease, border-color 0.2s ease;
}}
.cc-pill:hover {{ /* General hover for non-interactive pills */
     background: #f7e1ce;
}}
.cc-pill-placeholder {{
    font-style: italic;
    color: var(--cc-text-placeholder);
    font-size: 0.75em;
    padding: 0.3rem 0;
}}

.cc-item-list {{
    margin-top: 0.4rem;
}}
.cc-item {{
    background: var(--cc-bg-exp);
    padding: 0.7rem;
    border-radius: 4px;
    border: 1px solid var(--cc-accent-light-tan);
    margin-bottom: 0.6rem;
    font-size: 0.85em;
}}
.cc-item:last-child {{
    margin-bottom: 0;
}}
.cc-item-header {{
    font-weight: bold;
    color: var(--cc-text-exp);
    font-size: 1em;
}}
.cc-item-subheader {{
    font-size: 0.9em;
    color: var(--cc-text-secondary);
    margin-bottom: 0.2rem;
}}
.cc-item-dates {{
    font-size: 0.8em;
    color: var(--cc-text-secondary);
    font-style: italic;
    margin-bottom: 0.3rem;
}}
.cc-item-description {{
    font-size: 0.95em;
    line-height: 1.4;
    color: var(--cc-text-exp);
    white-space: pre-wrap;
    max-height: calc(1.4em * {MAX_DESC_LINES_ON_CARD});
    overflow: hidden;
    position: relative;
}}

.cc-item-skills-header {{
    font-size: 0.85em;
    font-weight: bold;
    color: var(--cc-accent-dark-brown);
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
}}
.cc-item-skill-pill {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    margin-right: 3px;
    margin-bottom: 3px;
    border-radius: 8px;
    background: var(--cc-bg-tag);
    font-size: 0.7em;
    color: var(--cc-text-tag);
    border: 1px solid var(--cc-accent-light-tan);
}}
.cc-item-project-url a {{
    font-size: 0.85em;
    color: var(--cc-accent-theme-brown);
    text-decoration: none;
}}
.cc-item-project-url a:hover {{
    text-decoration: underline;
}}
.cc-list-placeholder {{
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
}}

.cc-progress-summary-container {{
    margin-top: auto;
    padding-top: 0.8rem;
    border-top: 1px solid var(--cc-accent-light-tan);
}}
.cc-missing-fields-summary {{ font-size: 0.8em; color: var(--cc-text-missing-summary); margin-bottom: 0.5rem; }}
.cc-missing-fields-summary h6 {{ margin-bottom: 4px; font-weight: bold; font-size: 0.9em; color: var(--cc-accent-dark-brown); }}
.cc-missing-fields-summary ul {{ margin: 0; padding-left: 18px; list-style-type: '☕ '; }}
.cc-missing-fields-summary li {{ margin-bottom: 2px; }}

.cc-progress-label {{ font-size: 0.8em; color: var(--cc-text-progress); margin-bottom: 4px; text-align: right; }}
.cc-progress-bar-bg {{ width: 100%; background-color: var(--cc-bg-progress-track); border-radius: 4px; height: 8px; overflow: hidden; margin-bottom: 8px; }}
.cc-progress-bar-fill {{ height: 100%; background-color: var(--cc-accent-theme-brown); border-radius: 4px; transition: width 0.5s ease-in-out; }}

/* --- NEW/ADAPTED STYLES FOR HOVER POPOUTS --- */
.tag-list {{ /* Added to .cc-pill-container */
    /* display: flex; flex-wrap: wrap; gap: 0.6rem; /* .cc-pill-container already handles this */
    /* padding: 0; list-style: none; margin-bottom: 0; */
}}

/* Styling for the interactive pills that trigger popouts */
.tag-list-item-interactive {{ /* Added to interactive .cc-pill spans */
    position: relative; /* Crucial for absolute positioning of child popouts */
    cursor: default;
}}
.tag-list-item-interactive:hover {{
    background-color: var(--cc-accent-light-tan) !important; /* Ensure override */
    border-color: var(--cc-accent-dark-brown) !important; /* Ensure override */
}}

/* Second-level popout (details for a skill/interest) */
.detail-popout {{
    position: absolute;
    bottom: calc(100% + 8px); /* Position above the parent pill */
    left: 50%;
    transform: translateX(-50%) translateY(5px) scale(0.95);
    transform-origin: bottom center;
    width: 280px; /* Slightly wider */
    max-width: 90vw; /* Prevent overflow on small screens */
    padding: 0.85rem; /* Slightly more padding */
    background: var(--cc-bg-detail-popout);
    border: 1px solid var(--cc-accent-dark-brown);
    border-radius: 6px;
    box-shadow: 0 -3px 10px rgba(0,0,0,0.12); /* Slightly stronger shadow */
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.2s ease, visibility 0.2s ease, transform 0.2s ease;
    z-index: 3000; /* High z-index for Streamlit */
    font-size: 0.8rem;
    line-height: 1.3;
    color: var(--cc-text-general);
    text-align: left;
}}

.tag-list-item-interactive:hover .detail-popout,
.tag-list-item-interactive:focus-within .detail-popout {{
    opacity: 1;
    visibility: visible;
    transform: translateX(-50%) translateY(0) scale(1);
}}

.detail-popout strong {{
    color: var(--cc-text-header);
    display: block;
    margin-bottom: 0.3rem; /* More space */
    font-weight: bold;
    font-size: 0.9rem; /* Slightly larger title */
}}
.detail-popout .context-snippet {{
    font-style: italic;
    color: var(--cc-text-secondary);
    margin-bottom: 0.4rem; /* More space */
    display: block;
    font-size: 0.78rem; /* Slightly smaller */
}}
.detail-popout .related-tags {{
    font-size: 0.75rem;
    margin-top: 0.5rem; /* More space */
}}

/* Styling for inner tags within a detail-popout (third-level trigger) */
.detail-popout .related-tags span.inner-tag {{
    position: relative;
    display: inline-block;
    background: var(--cc-bg-tag);
    padding: 0.25rem 0.5rem; /* Slightly more padding */
    border-radius: 4px; /* Slightly more rounded */
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    border: 1px solid var(--cc-accent-light-tan);
    cursor: help;
    transition: background-color 0.2s, border-color 0.2s;
    color: var(--cc-text-tag);
    font-size: 0.72rem; /* Slightly smaller */
}}
.detail-popout .related-tags span.inner-tag:hover {{
    background-color: var(--cc-accent-light-tan);
    border-color: var(--cc-accent-dark-brown);
}}

/* Third-level popout (deep details for an inner tag) */
.deep-detail-popout {{
    position: absolute;
    bottom: calc(100% + 5px);
    left: 50%;
    transform: translateX(-50%) translateY(3px) scale(0.9);
    transform-origin: bottom center;
    width: 220px; /* Slightly wider */
    max-width: 80vw; /* Prevent overflow */
    padding: 0.6rem; /* More padding */
    background: var(--cc-bg-deep-detail-popout);
    border: 1px solid var(--cc-accent-dark-brown);
    border-radius: 4px;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.1); /* Slightly stronger shadow */
    opacity: 0;
    visibility: hidden;
    font-size: 0.75rem;
    line-height: 1.25; /* Better readability */
    color: var(--cc-text-general);
    transition: opacity 0.15s ease, visibility 0.15s ease, transform 0.15s ease;
    z-index: 4000; /* Highest z-index */
    text-align: left;
}}

.detail-popout .related-tags span.inner-tag:hover .deep-detail-popout,
.detail-popout .related-tags span.inner-tag:focus-within .deep-detail-popout {{
    opacity: 1;
    visibility: visible;
    transform: translateX(-50%) translateY(0) scale(1);
}}
.deep-detail-popout p {{
    margin: 0;
    font-size: inherit;
    color: var(--cc-text-general);
}}


/* Streamlit Button Overrides */
div[data-testid="stForm"] .stButton button:not([kind="secondary"]),
.stButton button:not([kind="secondary"]) {{
    border: 1px solid var(--cc-btn-default-text) !important;
    color: var(--cc-btn-default-text) !important;
    background-color: transparent !important;
    border-radius: 4px !important;
    padding: 0.3rem 0.7rem !important;
    font-size: 0.9em !important;
}}
div[data-testid="stForm"] .stButton button:not([kind="secondary"]):hover,
.stButton button:not([kind="secondary"]):hover {{
    border-color: var(--cc-btn-default-hover-border) !important;
    color: var(--cc-btn-default-hover-text) !important;
    background-color: var(--cc-bg-btn-default-hover) !important;
}}
div[data-testid="stForm"] .stButton button[kind="secondary"],
.stButton button[kind="secondary"] {{
    border: 1px solid var(--cc-btn-delete-text) !important;
    color: var(--cc-btn-delete-text) !important;
    background-color: transparent !important;
    border-radius: 4px !important;
    padding: 0.3rem 0.7rem !important;
    font-size: 0.9em !important;
}}
div[data-testid="stForm"] .stButton button[kind="secondary"]:hover,
.stButton button[kind="secondary"]:hover {{
    border-color: var(--cc-btn-delete-hover-border) !important;
    color: var(--cc-btn-delete-hover-text) !important;
    background-color: var(--cc-bg-btn-delete-hover) !important;
}}

/* Style for popover trigger buttons in full details */
div[data-testid="stExpander"] div[data-testid="stButton"] > button {{
    font-size: 0.85em !important;
    padding: 0.25rem 0.6rem !important;
    margin: 3px 2px !important;
    border-radius: 12px !important;
    background-color: var(--cc-bg-tag) !important;
    color: var(--cc-text-tag) !important;
    border: 1px solid var(--cc-accent-light-tan) !important;
}}
div[data-testid="stExpander"] div[data-testid="stButton"] > button:hover {{
    background-color: #f7e1ce !important;
    border-color: var(--cc-accent-theme-brown) !important;
}}

/* Popover content styling */
div[data-testid="stPopover"] h3 {{
    color: var(--cc-accent-dark-brown);
    margin-top: 0;
    margin-bottom: 0.5rem;
    font-size: 1.1em;
}}
div[data-testid="stPopover"] p {{
    font-size: 0.9em;
    margin-bottom: 0.3rem;
}}
div[data-testid="stPopover"] strong {{
    color: var(--cc-text-exp);
}}
div[data-testid="stPopover"] hr {{
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}}

/* --- Apple Wallet Preview Styles --- */
.wallet-pass-preview-container {{
    display: flex;
    justify-content: center;
    margin-bottom: 15px;
}}
.wallet-pass-preview {{
    background-color: #2c2c2e; color: #fff; border-radius: 12px; padding: 15px;
    width: 100%; max-width: 340px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2); font-size: 14px;
}}
.wallet-pass-header {{
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #4a4a4e;
}}
.wallet-pass-header .logo {{ font-size: 1.6em; color: var(--cc-accent-theme-brown); margin-top: 2px; }}
.wallet-pass-header .pass-type-stack {{ display: flex; flex-direction: column; align-items: flex-end; text-align: right; }}
.wallet-pass-header .pass-type {{ font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.8px; color: #c7c7cc; }}
.wallet-pass-header .pass-location {{
    font-size: 0.65em; color: #8e8e93; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; max-width: 100px;
}}
.wallet-pass-body .name {{
    font-size: 1.2em; font-weight: 500; margin-bottom: 2px; color: #fff;
    line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.wallet-pass-body .title {{
    font-size: 0.85em; color: #e5e5ea; margin-bottom: 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.wallet-pass-body .summary {{
    font-size: 0.75em; color: #aeaeb2; line-height: 1.2;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 5px;
}}
.wallet-pass-body .key-skills-list {{
    font-size: 0.7em; color: #c7c7cc; margin-top: 4px; line-height: 1.3;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 90%;
}}
.wallet-pass-body .key-skills-list .skills-label {{ font-weight: 500; color: #e5e5ea; }}
.wallet-pass-qr-section {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #4a4a4e; text-align: center; }}
.wallet-pass-qr-section img {{
    background-color: white; padding: 3px; border-radius: 3px;
    max-width: 75px; display: block; margin: 0 auto;
}}
.wallet-pass-qr-section .qr-label {{ font-size: 0.7em; color: #8e8e93; margin-top: 5px; }}

/* --- Social PNG Preview Styles --- */
.social-png-preview-container {{ display: flex; justify-content: center; margin-bottom: 15px; }}
.social-png-preview {{
    border: 1px solid var(--cc-accent-light-tan); border-radius: 8px; padding: 18px;
    background-color: var(--cc-bg-exp); width: 100%; max-width: 380px; font-family: sans-serif;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; flex-direction: column;
}}
.social-png-header {{ display: flex; align-items: center; margin-bottom: 12px; gap: 12px; }}
.social-png-avatar {{
    width: 65px; height: 65px; border-radius: 50%; object-fit: cover;
    border: 2px solid var(--cc-accent-theme-brown); flex-shrink: 0;
}}
.social-png-text-info {{ flex-grow: 1; min-width: 0; }}
.social-png-text-info .name {{
    font-size: 1.25em; font-weight: bold; color: var(--cc-text-name); margin: 0 0 2px 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.social-png-text-info .title {{
    font-size: 0.8em; color: var(--cc-text-title); margin: 0; line-height: 1.3;
    white-space: normal; overflow: hidden; display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}}
.social-png-tagline {{
    font-size: 0.85em; color: var(--cc-text-secondary); margin-bottom: 8px; line-height: 1.4;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden; text-overflow: ellipsis;
}}
.social-png-location {{
    font-size: 0.75em; color: var(--cc-text-secondary); margin-bottom: 10px;
    display: flex; align-items: center;
}}
.social-png-location .icon {{ margin-right: 5px; opacity: 0.7; }}
.social-png-pills-section {{ margin-bottom: 10px; }}
.social-png-pills-section .pills-label {{
    font-size: 0.7em; color: var(--cc-accent-dark-brown); font-weight: bold;
    margin-bottom: 4px; text-transform: uppercase;
}}
.social-png-pills-container .cc-pill {{ font-size: 0.7em; padding: 0.2rem 0.6rem; margin: 2px; }}
.social-png-footer {{
    margin-top: auto; padding-top: 8px; border-top: 1px dashed var(--cc-accent-light-tan);
    font-size: 0.7em; color: var(--cc-text-placeholder); text-align: center;
}}
.social-png-footer .cta {{ font-weight: bold; color: var(--cc-accent-theme-brown); }}

/* Additional styling for dialog form elements if needed */
div[data-testid="stDialog"] .stTextArea textarea,
div[data-testid="stDialog"] .stTextInput input {{ font-size: 0.95em; }}
div[data-testid="stDialog"] h3, div[data-testid="stDialog"] h4 {{
    color: var(--cc-accent-dark-brown); margin-top: 0.8rem; margin-bottom: 0.3rem;
}}
div[data-testid="stDialog"] hr {{ margin-top: 0.8rem; margin-bottom: 0.8rem; }}

/* Styling for native card popover buttons to make them look like pills */
div[data-testid="stVerticalBlock"] div[data-testid="stPopover"] > button,
div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button {{
    background-color: var(--cc-bg-tag) !important;
    color: var(--cc-text-tag) !important;
    border: 1px solid var(--cc-accent-light-tan) !important;
    border-radius: 10px !important;
    padding: 0.25rem 0.7rem !important;
    font-size: 0.75em !important;
    margin: 2px 3px !important;
    line-height: 1.3 !important;
    display: inline-block !important;
    text-align: center !important;
}}
div[data-testid="stVerticalBlock"] div[data-testid="stPopover"] > button:hover,
div[data-testid="stHorizontalBlock"] div[data-testid="stPopover"] > button:hover {{
    background-color: #f7e1ce !important;
    border-color: var(--cc-accent-theme-brown) !important;
}}

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
        if profile_data.get(key) and (isinstance(profile_data[key], str) and profile_data[key].strip() or isinstance(profile_data[key], list) and profile_data[key]):
            completed_essential_count += 1
        else: missing_summary.append(display_name)

    has_edu_or_proj = bool(profile_data.get("education") or profile_data.get("projects"))
    total_essential_fields_adjusted = total_essential_fields + 1 # Education OR Projects counts as one
    if has_edu_or_proj: completed_essential_count +=1
    else: missing_summary.append("Education or Projects")

    percentage = int((completed_essential_count / total_essential_fields_adjusted) * 100) if total_essential_fields_adjusted > 0 else 0
    if not missing_summary and completed_essential_count == total_essential_fields_adjusted : percentage = 100

    return percentage, missing_summary

def make_initials_svg_avatar(name: str, size: int = 80, bg: str = "#6b4f4f", fg: str = "#fff8f0") -> str:
    display_name = name if name and name.strip() else "?"
    initials = "".join([w[0].upper() for w in display_name.split()][:2]) or "?"
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"><circle cx="{size/2}" cy="{size/2}" r="{size/2}" fill="{bg}"/><text x="50%" y="50%" fill="{fg}" font-size="{int(size/2.2)}" text-anchor="middle" dominant-baseline="central" font-family="sans-serif" font-weight="500">{initials}</text></svg>'
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"

def open_edit_dialog(profile_id_to_edit: str):
    st.session_state.editing_profile_id_dialog = profile_id_to_edit

def delete_profile(profile_id):
    # Find the profile and remove it
    st.session_state.profiles_new_structure = [
        p for p in st.session_state.profiles_new_structure if p.get("id") != profile_id
    ]
    st.toast(f"Profile {profile_id[:8]}... deleted.", icon="🗑️")
    # If the deleted profile was being edited, clear the edit state
    if st.session_state.editing_profile_id_dialog == profile_id:
        st.session_state.editing_profile_id_dialog = None
    st.rerun()


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

def generate_pills_html_for_card(items: List[str], placeholder: str) -> str:
    if not items: return f'<span class="cc-pill-placeholder">{html.escape(placeholder)}</span>'

    safe_items = [item for item in items if item and item.strip()]
    if not safe_items: return f'<span class="cc-pill-placeholder">{html.escape(placeholder)}</span>'

    pills_html_list = []
    for item_name_raw in safe_items:
        item_name = html.escape(item_name_raw) # Escape the item name once for display
        popout_html_content = ""
        # Check if this item has deep dive info defined (use raw name for lookup)
        if item_name_raw in DEEP_DIVE_INFO:
            info = DEEP_DIVE_INFO[item_name_raw]

            inner_tags_html = ""
            if info.get("inner_tags"):
                inner_tags_html += '<div class="related-tags">'
                for tag_name_raw, tag_desc_raw in info["inner_tags"].items():
                    tag_name_esc = html.escape(tag_name_raw)
                    tag_desc_esc = html.escape(tag_desc_raw)
                    inner_tags_html += f'<span class="inner-tag" tabindex="0">{tag_name_esc}' # Display escaped name
                    inner_tags_html += f'<div class="deep-detail-popout"><p>{tag_desc_esc}</p></div>' # Display escaped description
                    inner_tags_html += '</span>' # Close inner-tag
                inner_tags_html += '</div>' # Close related-tags

            detail_title_esc = html.escape(info.get("detail_title", item_name_raw))
            detail_snippet_esc = html.escape(info.get("detail_snippet", ""))

            popout_html_content = f"""
            <div class="detail-popout">
                <strong>{detail_title_esc}</strong>
                <span class="context-snippet">{detail_snippet_esc}</span>
                {inner_tags_html}
            </div>
            """
            # This pill is interactive
            pill_class = "cc-pill tag-list-item-interactive"
            pills_html_list.append(f'<span class="{pill_class}" tabindex="0">{item_name}{popout_html_content}</span>')
        else:
            # This pill is not interactive in the deep-hover sense
            pills_html_list.append(f'<span class="cc-pill">{item_name}</span>')

    # Add 'tag-list' class to the container for CSS targeting
    return f'<div class="cc-pill-container tag-list">{"".join(pills_html_list)}</div>'


# --- HTML CONCISE CARD RENDERING FUNCTION ---
def render_coffee_card_concise(profile_data: Dict, profile_id: str):
    if not profile_data or not isinstance(profile_data, dict):
        st.warning("Invalid Coffee Card data provided for concise card.")
        return

    completion_percentage, missing_fields = calculate_profile_completion_new(profile_data)

    name = html.escape(profile_data.get("name", "N/A"))
    title_text = html.escape(profile_data.get("title", "N/A"))
    tagline = html.escape(profile_data.get("taglineOrBriefSummary", ""))
    location = html.escape(profile_data.get("location", ""))
    avatar_url = profile_data.get("profilePictureUrlForCard")
    if not avatar_url or "example.com" in avatar_url: # Fallback if placeholder or missing
        avatar_url = make_initials_svg_avatar(name if name != 'N/A' else '??')

    profile_url = profile_data.get("primaryProfileUrlForCard", "")
    call_to_action = html.escape(profile_data.get("callToActionForCard", ""))

    # Use all skills/interests/hobbies for the card, let CSS handle wrapping
    skills_on_card: List[str] = profile_data.get("skills", [])[:MAX_PILLS_ON_CARD] # Still cap for display count, but generate_pills can handle more
    interests_on_card: List[str] = profile_data.get("interests", [])[:MAX_PILLS_ON_CARD]
    hobbies_on_card: List[str] = profile_data.get("hobbies", [])[:MAX_PILLS_ON_CARD]

    key_achievements_on_card: List[str] = profile_data.get("keyAchievementsOverall", [])[:MAX_ITEMS_PER_SECTION_ON_CARD]
    experiences_on_card: List[Dict] = profile_data.get("experiences", [])[:MAX_ITEMS_PER_SECTION_ON_CARD]
    education_on_card: List[Dict] = profile_data.get("education", [])[:MAX_ITEMS_PER_SECTION_ON_CARD]
    projects_on_card: List[Dict] = profile_data.get("projects", [])[:MAX_ITEMS_PER_SECTION_ON_CARD]


    summary_list_html = ""
    if missing_fields:
        for field in missing_fields: summary_list_html += f"<li>Update '{html.escape(field)}'</li>"
    else: summary_list_html = "<li>All essential fields complete! ✔️</li>"


    def generate_items_list_html_for_card(items_data: List[Dict], item_type: str) -> str:
        if not items_data: return f'<div class="cc-list-placeholder">No {item_type.lower()} on card.</div>'
        items_html_list = []
        for idx, item in enumerate(items_data):
            item_html = '<div class="cc-item">'
            desc_id = f"card-desc-{profile_id}-{item_type}-{idx}" # Not strictly needed if not using JS for expand
            if item_type == "Experiences":
                role = html.escape(item.get("role", "N/A"))
                company = html.escape(item.get("company", "N/A"))
                dates = html.escape(item.get("dates", ""))
                description_raw = item.get("description", "")
                # For the concise card, use briefSummaryForMiniCard if available, else truncate full description
                description_for_card = item.get("briefSummaryForMiniCard")
                if not description_for_card:
                    description_for_card = truncate_description(description_raw, MAX_DESC_LINES_ON_CARD)
                else:
                    description_for_card = truncate_description(description_for_card, MAX_DESC_LINES_ON_CARD) # Ensure brief summary is also truncated if too long

                skill_details_on_card = item.get("skillDetails", [])[:MAX_PILLS_ON_CARD] # Cap pills per experience item

                item_html += f'<div class="cc-item-header">{role} at {company}</div>'
                if dates: item_html += f'<div class="cc-item-dates">{dates}</div>'
                if description_for_card:
                    item_html += f'<div class="cc-item-description">{description_for_card.replace(chr(10), "<br>")}</div>'

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
                if description:
                    item_html += f'<div class="cc-item-description">{description.replace(chr(10), "<br>")}</div>'

            elif item_type == "Projects":
                project_name = html.escape(item.get("projectName", "N/A"))
                dates = html.escape(item.get("datesOrDuration", ""))
                description = truncate_description(item.get("description", ""), MAX_DESC_LINES_ON_CARD)
                skills_used_on_card = item.get("skillsUsed", [])[:MAX_PILLS_ON_CARD] # Cap pills per project item
                project_url_val = item.get("projectUrl", "")

                item_html += f'<div class="cc-item-header">{project_name}</div>'
                if dates: item_html += f'<div class="cc-item-dates">{dates}</div>'
                if description:
                    item_html += f'<div class="cc-item-description">{description.replace(chr(10), "<br>")}</div>'
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
    if tagline: card_html += f'<p class="cc-tagline">{html.escape(tagline)}</p>'
    if location: card_html += f'<p class="cc-location">📍 {html.escape(location)}</p>'
    if profile_url: card_html += f'<p class="cc-profile-url"><a href="{html.escape(profile_url)}" target="_blank" rel="noopener noreferrer">🔗 View Profile</a></p>'
    if call_to_action: card_html += f'<p class="cc-tagline" style="font-weight:bold; color: var(--cc-accent-theme-brown);">{call_to_action}</p>'
    card_html += "</div></div>" # Close cc-header-text and cc-header-content

    # Sections: Skills, Interests, Hobbies
    if skills_on_card or profile_data.get("skills") is not None:
        card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🛠️</span>Top Skills</h5>{generate_pills_html_for_card(skills_on_card, "No skills added.")}</div>'
    if interests_on_card or profile_data.get("interests") is not None:
        card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">💡</span>Interests</h5>{generate_pills_html_for_card(interests_on_card, "No interests added.")}</div>'
    if hobbies_on_card or profile_data.get("hobbies") is not None:
        card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🎨</span>Hobbies</h5>{generate_pills_html_for_card(hobbies_on_card, "No hobbies added.")}</div>'

    # Sections: Key Achievements, Experiences, Education, Projects
    if key_achievements_on_card or profile_data.get("keyAchievementsOverall") is not None:
        card_html += '<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🏆</span>Key Achievements</h5>'
        if key_achievements_on_card:
            card_html += '<div class="cc-item-list">'
            for ach_idx, ach in enumerate(key_achievements_on_card): card_html += f'<div class="cc-item" style="padding: 0.5rem; background: var(--cc-bg-main);"><p style="margin:0; font-size:0.9em;">{truncate_description(html.escape(ach), MAX_DESC_LINES_ON_CARD)}</p></div>'
            card_html += '</div>'
        else: card_html += '<div class="cc-list-placeholder">No achievements on card.</div>'
        card_html += '</div>'

    if experiences_on_card or profile_data.get("experiences") is not None:
        card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">💼</span>Recent Experience</h5>{generate_items_list_html_for_card(experiences_on_card, "Experiences")}</div>'
    if education_on_card or profile_data.get("education") is not None:
        card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🎓</span>Education</h5>{generate_items_list_html_for_card(education_on_card, "Education")}</div>'
    if projects_on_card or profile_data.get("projects") is not None:
        card_html += f'<div class="cc-section"><h5 class="cc-section-header"><span class="cc-icon">🚀</span>Featured Projects</h5>{generate_items_list_html_for_card(projects_on_card, "Projects")}</div>'

    card_html += '</div>' # Close cc-card-content
    card_html += f"""
        <div class="cc-progress-summary-container">
            <div class="cc-progress-label">{completion_percentage}% Complete</div>
            <div class="cc-progress-bar-bg"><div class="cc-progress-bar-fill" style="width: {completion_percentage}%;"></div></div>"""
    if missing_fields or completion_percentage < 100: card_html += f'<div class="cc-missing-fields-summary"><h6>Profile Checklist:</h6><ul>{summary_list_html}</ul></div>'
    else: card_html += '<div class="cc-missing-fields-summary"><h6>Profile Checklist:</h6><ul><li>All essential fields complete! ✔️</li></ul></div>'
    card_html += "</div></div>" # Close cc-progress-summary-container and coffee-card-generated

    st.markdown(card_html, unsafe_allow_html=True)
    action_cols = st.columns([0.75, 0.125, 0.125]) # Ratio for buttons
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
        avatar_url = profile_data.get("profilePictureUrlForCard")
        if not avatar_url or "example.com" in avatar_url: # Fallback if placeholder or missing
            avatar_url = make_initials_svg_avatar(name if name != 'N/A' else '??', size=100)
        profile_url = profile_data.get("primaryProfileUrlForCard", "")
        call_to_action = profile_data.get("callToActionForCard", "")

        header_cols = st.columns([1, 3])
        with header_cols[0]:
            st.image(avatar_url, width=100, caption="" if avatar_url and "pexels.com" in avatar_url else "Avatar") # Keep pexels check
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
            for idx, skill_name_raw in enumerate(skills_all):
                skill_name = html.escape(skill_name_raw)
                with skill_cols[idx % num_skill_cols]:
                    # Check for DEEP_DIVE_INFO for native card as well
                    popover_content_parts = []
                    popover_content_parts.append(f"### {skill_name}")

                    # Attempt to find detailed context from experiences/projects
                    experiences_with_skill = []
                    for exp in profile_data.get("experiences", []):
                        for sd in exp.get("skillDetails", []):
                            if sd.get("skillName") == skill_name_raw: # Use raw name for matching
                                context_snippet = sd.get("contextualSnippet", "")
                                exp_info = f"_{html.escape(exp.get('role', 'N/A'))} at {html.escape(exp.get('company', 'N/A'))}_"
                                if context_snippet:
                                    exp_info += f": {html.escape(context_snippet)}"
                                experiences_with_skill.append(exp_info)
                    projects_with_skill = []
                    for proj in profile_data.get("projects", []):
                        if skill_name_raw in proj.get("skillsUsed", []): # Use raw name for matching
                            projects_with_skill.append(html.escape(proj.get("projectName", "N/A")))

                    # Add deep dive info if available (similar to HTML card's detail_popout)
                    if skill_name_raw in DEEP_DIVE_INFO:
                        deep_info = DEEP_DIVE_INFO[skill_name_raw]
                        if deep_info.get("detail_snippet"):
                            popover_content_parts.append(f"_{html.escape(deep_info['detail_snippet'])}_")
                        if deep_info.get("inner_tags"):
                            popover_content_parts.append("**Related Concepts:**")
                            for inner_tag_name, inner_tag_desc in deep_info["inner_tags"].items():
                                popover_content_parts.append(f"- **{html.escape(inner_tag_name)}**: {html.escape(inner_tag_desc)}")
                        popover_content_parts.append("---") # Separator if deep dive info was added

                    if experiences_with_skill:
                        popover_content_parts.append("**Applied in Experiences:**\n" + "\n".join([f"- {e}" for e in experiences_with_skill]))
                    if projects_with_skill:
                        popover_content_parts.append("**Used in Projects:**\n" + "\n".join([f"- {p}" for p in projects_with_skill]))
                    if not experiences_with_skill and not projects_with_skill and not (skill_name_raw in DEEP_DIVE_INFO and deep_info.get("inner_tags")):
                        popover_content_parts.append("General skill proficiency.")

                    with st.popover(skill_name, use_container_width=True):
                        st.markdown("\n\n".join(popover_content_parts), unsafe_allow_html=True) # Allow markdown for formatting
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
                            skill_name_exp_raw = skill_info.get("skillName","")
                            skill_name_exp = html.escape(skill_name_exp_raw)
                            with exp_skill_cols[s_idx % num_exp_skill_cols]:
                                popover_content_exp = [f"### {skill_name_exp}"]
                                popover_content_exp.append(f"**Context:** {html.escape(skill_info.get('contextualSnippet', 'No specific context provided.'))}")
                                related_skills_exp = skill_info.get("relatedSkillsInThisExperience", [])
                                if related_skills_exp:
                                    popover_content_exp.append(f"**Related within this experience:** {', '.join(map(html.escape, related_skills_exp))}")

                                # Add deep dive info for this skill if available
                                if skill_name_exp_raw in DEEP_DIVE_INFO:
                                    deep_info_exp = DEEP_DIVE_INFO[skill_name_exp_raw]
                                    if deep_info_exp.get("detail_snippet"):
                                        popover_content_exp.append(f"_{html.escape(deep_info_exp['detail_snippet'])}_")
                                    if deep_info_exp.get("inner_tags"):
                                        popover_content_exp.append("**Broader Concepts:**")
                                        for inner_tag_name_exp, inner_tag_desc_exp in deep_info_exp["inner_tags"].items():
                                            popover_content_exp.append(f"- **{html.escape(inner_tag_name_exp)}**: {html.escape(inner_tag_desc_exp)}")

                                with st.popover(skill_name_exp, use_container_width=True):
                                     st.markdown("\n\n".join(popover_content_exp), unsafe_allow_html=True)
                st.markdown("---" if exp_idx < len(experiences_all) -1 else "")
        elif experiences_all is not None:
            st.markdown("**💼 Experience**")
            st.caption("No experience added.")
        if experiences_all is not None: st.markdown("---")


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
                    skills_used_proj_raw = proj.get("skillsUsed", [])
                    if skills_used_proj_raw:
                        st.markdown("**Skills/Tech Used:**")
                        num_proj_skill_cols = min(len(skills_used_proj_raw), 4)
                        proj_skill_cols = st.columns(num_proj_skill_cols)
                        for sk_idx, skill_name_proj_raw in enumerate(skills_used_proj_raw):
                            skill_name_proj = html.escape(skill_name_proj_raw)
                            with proj_skill_cols[sk_idx % num_proj_skill_cols]:
                                popover_content_proj = [f"### {skill_name_proj}"]
                                popover_content_proj.append("Used in this project.")
                                if skill_name_proj_raw in DEEP_DIVE_INFO:
                                    deep_info_proj = DEEP_DIVE_INFO[skill_name_proj_raw]
                                    if deep_info_proj.get("detail_snippet"):
                                        popover_content_proj.append(f"_{html.escape(deep_info_proj['detail_snippet'])}_")
                                    if deep_info_proj.get("inner_tags"):
                                        popover_content_proj.append("**Related Concepts:**")
                                        for inner_tag_name_p, inner_tag_desc_p in deep_info_proj["inner_tags"].items():
                                            popover_content_proj.append(f"- **{html.escape(inner_tag_name_p)}**: {html.escape(inner_tag_desc_p)}")
                                with st.popover(skill_name_proj, use_container_width=True):
                                    st.markdown("\n\n".join(popover_content_proj), unsafe_allow_html=True)

                st.markdown("---" if proj_idx < len(projects_all)-1 else "")
        elif projects_all is not None:
            st.markdown("**🚀 Projects**")
            st.caption("No projects added.")
        if projects_all is not None: st.markdown("---")


        # --- Interests and Hobbies (Full lists, simpler display but with popovers) ---
        interests_all = profile_data.get("interests", [])
        if interests_all:
            st.markdown(f"**💡 Interests**")
            num_interest_cols = min(len(interests_all), 5)
            interest_cols = st.columns(num_interest_cols)
            for idx, interest_raw in enumerate(interests_all):
                interest = html.escape(interest_raw)
                with interest_cols[idx % num_interest_cols]:
                    popover_content_interest = [f"### {interest}"]
                    if interest_raw in DEEP_DIVE_INFO: # Check if interest has deep dive
                        deep_info_int = DEEP_DIVE_INFO[interest_raw]
                        if deep_info_int.get("detail_snippet"): popover_content_interest.append(f"_{html.escape(deep_info_int['detail_snippet'])}_")
                        if deep_info_int.get("inner_tags"):
                            popover_content_interest.append("**Related Concepts:**")
                            for it_name, it_desc in deep_info_int["inner_tags"].items(): popover_content_interest.append(f"- **{html.escape(it_name)}**: {html.escape(it_desc)}")
                    else: popover_content_interest.append("General interest.")
                    with st.popover(interest, use_container_width=True): st.markdown("\n\n".join(popover_content_interest), unsafe_allow_html=True)
            st.markdown("---")
        elif interests_all is not None:
            st.markdown(f"**💡 Interests**")
            st.caption("No interests added.")
            st.markdown("---")

        hobbies_all = profile_data.get("hobbies", [])
        if hobbies_all:
            st.markdown(f"**🎨 Hobbies**")
            num_hobby_cols = min(len(hobbies_all), 5)
            hobby_cols = st.columns(num_hobby_cols)
            for idx, hobby_raw in enumerate(hobbies_all):
                hobby = html.escape(hobby_raw)
                with hobby_cols[idx % num_hobby_cols]:
                    popover_content_hobby = [f"### {hobby}"]
                    if hobby_raw in DEEP_DIVE_INFO: # Check if hobby has deep dive
                        deep_info_hob = DEEP_DIVE_INFO[hobby_raw]
                        if deep_info_hob.get("detail_snippet"): popover_content_hobby.append(f"_{html.escape(deep_info_hob['detail_snippet'])}_")
                        if deep_info_hob.get("inner_tags"):
                            popover_content_hobby.append("**Related Concepts:**")
                            for ht_name, ht_desc in deep_info_hob["inner_tags"].items(): popover_content_hobby.append(f"- **{html.escape(ht_name)}**: {html.escape(ht_desc)}")
                    else: popover_content_hobby.append("Personal hobby.")
                    with st.popover(hobby, use_container_width=True): st.markdown("\n\n".join(popover_content_hobby), unsafe_allow_html=True)

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


# --- PREVIEW RENDERING FUNCTIONS ---
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
    qr_data = profile_data.get("primaryProfileUrlForCard", f"https://example.com/profile/{profile_id}") # Fallback QR
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
        qr_img_html_embed = '<div class="qr-label" style="text-align:center;">QR (Error)</div>'

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
    avatar_url = profile_data.get("profilePictureUrlForCard")
    if not avatar_url or "example.com" in avatar_url:
        avatar_url = make_initials_svg_avatar(name if name != 'N/A' else '??', size=65)
    location = html.escape(profile_data.get("location", ""))
    call_to_action_short = html.escape(truncate_to_one_line(profile_data.get("callToActionForCard", "View Full Profile"), 30))
    skills_all = profile_data.get("skills", [])
    interests_all = profile_data.get("interests", [])
    pills_for_social = []
    pills_for_social.extend(skills_all[:MAX_PILLS_FOR_SOCIAL_PNG])
    remaining_slots = MAX_PILLS_FOR_SOCIAL_PNG - len(pills_for_social)
    if remaining_slots > 0 and interests_all:
        pills_for_social.extend(interests_all[:remaining_slots])

    if not pills_for_social and (skills_all or interests_all): # If source lists exist but still no pills (e.g. all empty strings)
        pills_html = '<span class="cc-pill-placeholder">Key areas...</span>'
    elif not pills_for_social: # If source lists are empty or None
         pills_html = '<span class="cc-pill-placeholder">No skills/interests</span>'
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

# --- FULL PROFILE DETAILS EXPANDER (Used with HTML card) ---
def render_full_profile_details_expander(profile_data: Dict, profile_id: str):
    if not profile_data: return

    expander_title = f"View Full Profile Details for {html.escape(profile_data.get('name', 'N/A'))}"
    with st.expander(expander_title, expanded=False):
        if profile_data.get("taglineOrBriefSummary"):
            st.markdown(f"**Tagline/Summary:** {html.escape(profile_data['taglineOrBriefSummary'])}")
        if profile_data.get("primaryProfileUrlForCard"):
            st.markdown(f"🔗 [View LinkedIn/Profile]({html.escape(profile_data['primaryProfileUrlForCard'])})")
        st.markdown("---")

        # Re-use the comprehensive card's logic for displaying sections, but without the main container
        # Skills
        skills_all = profile_data.get("skills", [])
        if skills_all:
            st.subheader("🛠️ Skills")
            num_skill_cols = min(len(skills_all), 4)
            skill_cols = st.columns(num_skill_cols)
            for idx, skill_name_raw in enumerate(skills_all):
                skill_name = html.escape(skill_name_raw)
                with skill_cols[idx % num_skill_cols]:
                    popover_content_parts = [f"### {skill_name}"]
                    experiences_with_skill, projects_with_skill = [], []
                    for exp in profile_data.get("experiences", []):
                        for sd in exp.get("skillDetails", []):
                            if sd.get("skillName") == skill_name_raw:
                                context = sd.get("contextualSnippet", "")
                                exp_info = f"_{html.escape(exp.get('role', 'N/A'))} at {html.escape(exp.get('company', 'N/A'))}_"
                                if context: exp_info += f": {html.escape(context)}"
                                experiences_with_skill.append(exp_info)
                    for proj in profile_data.get("projects", []):
                        if skill_name_raw in proj.get("skillsUsed", []): projects_with_skill.append(html.escape(proj.get("projectName", "N/A")))
                    if skill_name_raw in DEEP_DIVE_INFO:
                        deep_info = DEEP_DIVE_INFO[skill_name_raw]
                        if deep_info.get("detail_snippet"): popover_content_parts.append(f"_{html.escape(deep_info['detail_snippet'])}_")
                        if deep_info.get("inner_tags"):
                            popover_content_parts.append("**Related Concepts:**")
                            for it_name, it_desc in deep_info["inner_tags"].items(): popover_content_parts.append(f"- **{html.escape(it_name)}**: {html.escape(it_desc)}")
                        if experiences_with_skill or projects_with_skill: popover_content_parts.append("---")
                    if experiences_with_skill: popover_content_parts.append("**Applied in Experiences:**\n" + "\n".join([f"- {e}" for e in experiences_with_skill]))
                    if projects_with_skill: popover_content_parts.append("**Used in Projects:**\n" + "\n".join([f"- {p}" for p in projects_with_skill]))
                    if not experiences_with_skill and not projects_with_skill and not (skill_name_raw in DEEP_DIVE_INFO and DEEP_DIVE_INFO[skill_name_raw].get("inner_tags")):
                        popover_content_parts.append("General skill proficiency.")
                    with st.popover(skill_name, use_container_width=True): st.markdown("\n\n".join(popover_content_parts), unsafe_allow_html=True)
            st.markdown("---")

        # Key Achievements
        key_achievements_all = profile_data.get("keyAchievementsOverall", [])
        if key_achievements_all:
            st.subheader("🏆 Key Achievements")
            for ach in key_achievements_all: st.markdown(f"- {html.escape(ach)}")
            st.markdown("---")

        # Experiences
        experiences_all = profile_data.get("experiences", [])
        if experiences_all:
            st.subheader("💼 Experience")
            for exp_idx, exp in enumerate(experiences_all):
                st.markdown(f"#### {html.escape(exp.get('role', 'N/A'))} at {html.escape(exp.get('company', 'N/A'))}")
                st.caption(f"_{html.escape(exp.get('dates', ''))}_")
                if exp.get('description'): st.markdown(html.escape(exp.get('description')).replace("\n", "  \n"), unsafe_allow_html=True)
                skill_details = exp.get("skillDetails", [])
                if skill_details:
                    st.markdown("**Skills/Tools Used in this Role:**")
                    num_exp_skill_cols = min(len(skill_details), 4)
                    exp_skill_cols = st.columns(num_exp_skill_cols)
                    for s_idx, skill_info in enumerate(skill_details):
                        skill_name_exp_raw = skill_info.get("skillName","")
                        skill_name_exp = html.escape(skill_name_exp_raw)
                        with exp_skill_cols[s_idx % num_exp_skill_cols]:
                            popover_content_exp = [f"### {skill_name_exp}"]
                            popover_content_exp.append(f"**Context:** {html.escape(skill_info.get('contextualSnippet', 'No specific context.'))}")
                            related_skills_exp = skill_info.get("relatedSkillsInThisExperience", [])
                            if related_skills_exp: popover_content_exp.append(f"**Related (in exp):** {', '.join(map(html.escape, related_skills_exp))}")
                            if skill_name_exp_raw in DEEP_DIVE_INFO:
                                deep_info_exp = DEEP_DIVE_INFO[skill_name_exp_raw]
                                if deep_info_exp.get("detail_snippet"): popover_content_exp.append(f"_{html.escape(deep_info_exp['detail_snippet'])}_")
                                if deep_info_exp.get("inner_tags"):
                                    popover_content_exp.append("**Broader Concepts:**")
                                    for it_name, it_desc in deep_info_exp["inner_tags"].items(): popover_content_exp.append(f"- **{html.escape(it_name)}**: {html.escape(it_desc)}")
                            with st.popover(skill_name_exp, use_container_width=True): st.markdown("\n\n".join(popover_content_exp), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True) # Spacer after skill pills
                st.markdown("---" if exp_idx < len(experiences_all) -1 else "")
            if experiences_all: st.markdown("---") # Final divider for section

        # Education
        education_all = profile_data.get("education", [])
        if education_all:
            st.subheader("🎓 Education")
            for edu_idx, edu in enumerate(education_all):
                st.markdown(f"#### {html.escape(edu.get('degree', 'N/A'))} - _{html.escape(edu.get('fieldOfStudy', ''))}_")
                st.markdown(f"{html.escape(edu.get('institution', 'N/A'))} ({html.escape(edu.get('dates', ''))})")
                if edu.get('description'): st.markdown(html.escape(edu.get('description')).replace("\n", "  \n"), unsafe_allow_html=True)
                st.markdown("---" if edu_idx < len(education_all) -1 else "")
            if education_all: st.markdown("---")

        # Projects
        projects_all = profile_data.get("projects", [])
        if projects_all:
            st.subheader("🚀 Projects")
            for proj_idx, proj in enumerate(projects_all):
                st.markdown(f"#### {html.escape(proj.get('projectName', 'N/A'))}")
                st.caption(f"_{html.escape(proj.get('datesOrDuration', ''))}_")
                if proj.get('projectUrl'): st.markdown(f"🔗 [View Project]({html.escape(proj.get('projectUrl'))})")
                if proj.get('description'): st.markdown(html.escape(proj.get('description')).replace("\n", "  \n"), unsafe_allow_html=True)
                skills_used_proj_raw = proj.get("skillsUsed", [])
                if skills_used_proj_raw:
                    st.markdown("**Skills/Tech Used:**")
                    num_proj_skill_cols = min(len(skills_used_proj_raw), 4)
                    proj_skill_cols = st.columns(num_proj_skill_cols)
                    for sk_idx, skill_name_proj_raw in enumerate(skills_used_proj_raw):
                        skill_name_proj = html.escape(skill_name_proj_raw)
                        with proj_skill_cols[sk_idx % num_proj_skill_cols]:
                            popover_content_proj = [f"### {skill_name_proj}"]
                            popover_content_proj.append("Used in this project.")
                            if skill_name_proj_raw in DEEP_DIVE_INFO:
                                deep_info_proj = DEEP_DIVE_INFO[skill_name_proj_raw]
                                if deep_info_proj.get("detail_snippet"): popover_content_proj.append(f"_{html.escape(deep_info_proj['detail_snippet'])}_")
                                if deep_info_proj.get("inner_tags"):
                                    popover_content_proj.append("**Related Concepts:**")
                                    for it_name, it_desc in deep_info_proj["inner_tags"].items(): popover_content_proj.append(f"- **{html.escape(it_name)}**: {html.escape(it_desc)}")
                            with st.popover(skill_name_proj, use_container_width=True): st.markdown("\n\n".join(popover_content_proj), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("---" if proj_idx < len(projects_all)-1 else "")
            if projects_all: st.markdown("---")

        # Interests and Hobbies (with popovers)
        interests_all = profile_data.get("interests", [])
        if interests_all:
            st.subheader("💡 Interests")
            num_interest_cols_exp = min(len(interests_all), 4)
            interest_cols_exp = st.columns(num_interest_cols_exp)
            for idx, interest_raw in enumerate(interests_all):
                interest = html.escape(interest_raw)
                with interest_cols_exp[idx % num_interest_cols_exp]:
                    popover_content_interest_exp = [f"### {interest}"]
                    if interest_raw in DEEP_DIVE_INFO:
                        deep_info_int_exp = DEEP_DIVE_INFO[interest_raw]
                        if deep_info_int_exp.get("detail_snippet"): popover_content_interest_exp.append(f"_{html.escape(deep_info_int_exp['detail_snippet'])}_")
                        if deep_info_int_exp.get("inner_tags"):
                            popover_content_interest_exp.append("**Related Concepts:**")
                            for it_name, it_desc in deep_info_int_exp["inner_tags"].items(): popover_content_interest_exp.append(f"- **{html.escape(it_name)}**: {html.escape(it_desc)}")
                    else: popover_content_interest_exp.append("General interest.")
                    with st.popover(interest, use_container_width=True): st.markdown("\n\n".join(popover_content_interest_exp), unsafe_allow_html=True)
            st.markdown("---")

        hobbies_all = profile_data.get("hobbies", [])
        if hobbies_all:
            st.subheader("🎨 Hobbies")
            num_hobby_cols_exp = min(len(hobbies_all), 4)
            hobby_cols_exp = st.columns(num_hobby_cols_exp)
            for idx, hobby_raw in enumerate(hobbies_all):
                hobby = html.escape(hobby_raw)
                with hobby_cols_exp[idx % num_hobby_cols_exp]:
                    popover_content_hobby_exp = [f"### {hobby}"]
                    if hobby_raw in DEEP_DIVE_INFO:
                        deep_info_hob_exp = DEEP_DIVE_INFO[hobby_raw]
                        if deep_info_hob_exp.get("detail_snippet"): popover_content_hobby_exp.append(f"_{html.escape(deep_info_hob_exp['detail_snippet'])}_")
                        if deep_info_hob_exp.get("inner_tags"):
                            popover_content_hobby_exp.append("**Related Concepts:**")
                            for ht_name, ht_desc in deep_info_hob_exp["inner_tags"].items(): popover_content_hobby_exp.append(f"- **{html.escape(ht_name)}**: {html.escape(ht_desc)}")
                    else: popover_content_hobby_exp.append("Personal hobby.")
                    with st.popover(hobby, use_container_width=True): st.markdown("\n\n".join(popover_content_hobby_exp), unsafe_allow_html=True)
            st.markdown("---")


# --- Example Usage Data (init_session_state) ---
def init_session_state():
    if 'profiles_new_structure' not in st.session_state:
        # Using the provided JSON data for Homen Shum
        json_data = {
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
            "profilePictureUrlForCard": "https://example.com/homen_shum_linkedin_profile.jpg", # Will fallback to initials
            "taglineOrBriefSummary": "Versatile professional with expertise in Generative AI, Large Language Models, and data analytics, driving innovation in finance and healthcare through leadership in startup ventures and impactful projects at JPMorgan Chase.",
            "primaryProfileUrlForCard": "https://www.linkedin.com/in/homen-shum",
            "callToActionForCard": "Connect on LinkedIn",
            "location": "Fremont, California, United States",
            "interests": [
              "Generative AI", "Large Language Models (LLMs)", "FinTech", "HealthTech",
              "Data Science", "Machine Learning", "Cloud Computing", "Startups",
              "Investment", "Technology Trends"
            ],
            "hobbies": [
              "Exploring new AI technologies", "Hackathons", "Content Creation (technical blogs/videos - inferred)"
            ],
            "skills": [
              "Large Language Models (LLMs)", "Python (Programming Language)", "Generative AI", "Data Analysis",
              "Machine Learning", "Pandas (Software)", "Microsoft Azure", "Google Cloud Platform (GCP)",
              "Amazon Web Services (AWS)", "Automation", "Financial Analysis", "Startup Development",
              "Docker", "RAG (Retrieval Augmented Generation)", "Streamlit" # Added a few more for popout demo
            ],
            "experiences": [
              {
                "role": "Founder", "company": "FinAdvizly LLC", "dates": "Dec 2021 - Present (2 yrs 5 mos)",
                "isCurrentOrPrimary": True,
                "briefSummaryForMiniCard": "Leading FinAdvizly in developing GenAI-powered sales recommendation, workflow automation, and financial research tools.",
                "description": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS, and Render using Docker. Led a team of 5 people to win Top-25 placement at UC Berkeley AI Hackathon (Jun 2023) by implementing medical code matching with real-time transcription and RAG (Retrieval Augmented Generation) implementation for healthcare, subsequently becoming Technical Co-Founder of FinAdvizly's new GenAI product for healthcare. Currently (Nov 2023-Present), developing FinAdvizly's capabilities by integrating multi-agent architecture for comprehensive financial research, featuring cross-validation with web data and structured output processing for report generation.",
                "skillDetails": [
                  {"skillName": "Large Language Models (LLMs)", "contextualSnippet": "subsequently becoming Technical Co-Founder of FinAdvizly's new GenAI product for healthcare.", "relatedSkillsInThisExperience": ["Generative AI", "RAG", "GCP", "Azure", "AWS", "Docker", "Python (Programming Language)", "Multi-agent architecture"]},
                  {"skillName": "Generative AI", "contextualSnippet": "subsequently becoming Technical Co-Founder of FinAdvizly's new GenAI product for healthcare.", "relatedSkillsInThisExperience": ["Large Language Models (LLMs)", "RAG", "Python (Programming Language)"]},
                  {"skillName": "Google Cloud Platform (GCP)", "contextualSnippet": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS...", "relatedSkillsInThisExperience": ["Microsoft Azure", "Amazon Web Services (AWS)", "Docker", "Automation"]},
                  {"skillName": "Microsoft Azure", "contextualSnippet": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS...", "relatedSkillsInThisExperience": ["Google Cloud Platform (GCP)", "Amazon Web Services (AWS)", "Docker", "Automation"]},
                  {"skillName": "Amazon Web Services (AWS)", "contextualSnippet": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS...", "relatedSkillsInThisExperience": ["Google Cloud Platform (GCP)", "Microsoft Azure", "Docker", "Automation"]},
                  {"skillName": "Docker", "contextualSnippet": "Built and deployed sales recommendation and workflow automation applications across GCP, Azure, AWS, and Render using Docker.", "relatedSkillsInThisExperience": ["Google Cloud Platform (GCP)", "Microsoft Azure", "Amazon Web Services (AWS)"]},
                  {"skillName": "RAG (Retrieval Augmented Generation)", "contextualSnippet": "implementing medical code matching with real-time transcription and RAG (Retrieval Augmented Generation) implementation for healthcare.", "relatedSkillsInThisExperience": ["Large Language Models (LLMs)", "Generative AI", "Medical Code Matching", "Real-time Transcription"]},
                  {"skillName": "Multi-agent architecture", "contextualSnippet": "integrating multi-agent architecture for comprehensive financial research...", "relatedSkillsInThisExperience": ["Large Language Models (LLMs)", "Financial Research"]},
                ]
              },
              {
                "role": "Rotation 3: Healthcare and Life Science Banking Team", "company": "JPMorgan Chase & Co.", "dates": "May 2023 - Feb 2024 (10 mos)",
                "isCurrentOrPrimary": False, "briefSummaryForMiniCard": None,
                "description": "Initiated collaborations between internal teams such as CC / SF Risk and Fin/ML Team to streamline client onboarding / management / servicing. Key projects included: (Oct 2023) Automated classification system for JPM Healthcare Banking team using GPT and Embeddings, implementing structured outputs to populate an instruction matrix and DPT 3.5 turbo, reducing processing time for 2,000+ companies from two weeks to under 30 seconds. (Feb 2024) Designed real-time AI transcription application for JPM internal meetings and assumed LLM Application Pilot role within AI ML Technology team.",
                "skillDetails": [
                  {"skillName": "GPT", "contextualSnippet": "Automated classification system for JPM Healthcare Banking team using GPT and Embeddings...", "relatedSkillsInThisExperience": ["Embeddings", "DPT 3.5 turbo", "Large Language Models (LLMs)", "AI Transcription", "Automation"]},
                  {"skillName": "Large Language Models (LLMs)", "contextualSnippet": "assumed LLM Application Pilot role within AI ML Technology team.", "relatedSkillsInThisExperience": ["GPT", "AI Transcription", "Automation"]},
                  {"skillName": "Automation", "contextualSnippet": "Automated classification system for JPM Healthcare Banking team...", "relatedSkillsInThisExperience": ["GPT", "Large Language Models (LLMs)"]},
                  {"skillName": "Cloud Computing", "contextualSnippet": "Initiated collaborations to streamline client onboarding / management / servicing.", "relatedSkillsInThisExperience": ["Amazon Web Services (AWS)"]},
                ]
              }
            ],
            "education": [
              {"institution": "UC Santa Barbara", "degree": "Certificate", "fieldOfStudy": "Business Administration and Management, General", "dates": "2020 - 2021", "description": "Cooperated with professors on personal project, recommended by 2 professors and a graduate student."}
            ],
            "projects": [
              {"projectName": "Patent Screening Tool & Report Generation", "datesOrDuration": "Ongoing", "description": "A Streamlit application for patent screening and generating reports.", "skillsUsed": ["Streamlit", "Python (Programming Language)", "AI", "Patent Analysis"], "projectUrl": "https://homen-patent-screening.streamlit.app/"}
            ],
            "keyAchievementsOverall": [
              "Co-founded FinAdvizly LLC, developing Generative AI solutions for finance and healthcare.",
              "Achieved Top-25 placement in UC Berkeley AI Hackathon and 2nd place in Nation's Cybersecurity Challenge.",
              "Pioneered AI applications within JPMorgan Chase, including automated systems and LLM pilots."
            ]
          }
        }
        homen_profile_data = json_data["yourCoffeeCard"]
        homen_profile_data["id"] = str(uuid.uuid4()) # Add a unique ID

        # A simpler second profile for variety
        EXAMPLE_YOUR_COFFEE_CARD_JANE = {
            "id": str(uuid.uuid4()), "name": "Jane Doe", "title": "Senior UX Designer • Creative Solutions",
            "profilePictureUrlForCard": "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
            "taglineOrBriefSummary": "Crafting intuitive and accessible user experiences for innovative tech products. Passionate about user-centered design and ethical AI.",
            "primaryProfileUrlForCard": "https://example.com/janedoe", "location": "Remote", "callToActionForCard": "Let's collaborate!",
            "skills": ["Figma", "User Research", "Wireframing", "Prototyping", "Accessibility", "UI Design", "Interaction Design", "Design Systems", "A/B Testing", "Streamlit"],
            "keyAchievementsOverall": ["Led a team to win the 'Best UX' award at TechCrunch Disrupt 2022.", "Successfully launched 3 major products, achieving an average NPS of 75+."],
            "experiences": [
                {
                    "role": "Senior UX Designer", "company": "TechSolutions Inc.", "dates": "2020-Present", "briefSummaryForMiniCard": "Leading design for key products, focusing on user-centered principles.",
                    "description": "Lead designer for several key product lines, focusing on user-centered design principles and accessibility standards. Mentor junior designers.",
                    "skillDetails": [
                        {"skillName": "User Research", "contextualSnippet": "Conducted heuristic evaluations, usability testing.", "relatedSkillsInThisExperience":["A/B Testing"]},
                        {"skillName": "Figma", "contextualSnippet": "Created high-fidelity prototypes in Figma."}
                    ]
                }
            ],
            "education": [{"institution": "Design University", "degree": "MFA in Interaction Design", "fieldOfStudy": "Human-Computer Interaction", "dates": "2016-2018", "description":"Thesis on ethical AI interfaces."}],
            "interests": ["Minimalist Design", "Ethical AI", "Sustainable Tech", "Photography", "Data Visualization"],
            "hobbies": ["Pottery", "Urban Sketching", "Yoga", "Reading Psychology Books"],
            "projects": [
                {
                    "projectName": "Accessible Mobile Banking App", "datesOrDuration": "6 Months (2022)",
                    "description": "Led the redesign of a mobile banking app to meet WCAG 2.1 AAA standards.",
                    "skillsUsed": ["Accessibility", "Figma", "User Testing", "Mobile UI/UX"],
                }
            ]
        }
        st.session_state.profiles_new_structure = [
            homen_profile_data,
            EXAMPLE_YOUR_COFFEE_CARD_JANE
        ]
    if 'editing_profile_id_dialog' not in st.session_state:
        st.session_state.editing_profile_id_dialog = None


# --- EDIT DIALOG ---
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

    st.subheader(f"Editing: {html.escape(profile_data_for_dialog.get('name', 'N/A'))}")
    st.markdown(f"_Profile ID: {profile_id_to_edit}_"); st.markdown("---")

    with st.form(key=f"edit_form_{profile_id_to_edit}"):
        updated_data = {} # To store form inputs

        # Basic Information
        st.markdown("### 👤 Basic Information")
        updated_data["name"] = st.text_input("Name", value=profile_data_for_dialog.get("name", ""), key=f"edit_name_{profile_id_to_edit}")
        updated_data["title"] = st.text_input("Title", value=profile_data_for_dialog.get("title", ""), key=f"edit_title_{profile_id_to_edit}")
        updated_data["profilePictureUrlForCard"] = st.text_input("Profile Picture URL", value=profile_data_for_dialog.get("profilePictureUrlForCard", ""), key=f"edit_avatar_{profile_id_to_edit}")
        updated_data["taglineOrBriefSummary"] = st.text_area("Tagline/Summary", value=profile_data_for_dialog.get("taglineOrBriefSummary", ""), height=100, key=f"edit_tagline_{profile_id_to_edit}")
        updated_data["primaryProfileUrlForCard"] = st.text_input("Primary Profile URL (e.g., LinkedIn)", value=profile_data_for_dialog.get("primaryProfileUrlForCard", ""), key=f"edit_profileurl_{profile_id_to_edit}")
        updated_data["callToActionForCard"] = st.text_input("Call to Action (for card)", value=profile_data_for_dialog.get("callToActionForCard", ""), key=f"edit_cta_{profile_id_to_edit}")
        updated_data["location"] = st.text_input("Location", value=profile_data_for_dialog.get("location", ""), key=f"edit_location_{profile_id_to_edit}")

        # Skills, Interests, Hobbies, Achievements (as text areas)
        st.markdown("---"); st.markdown("### 🛠️ Skills")
        updated_data["skills_text_area"] = st.text_area("Skills (one per line)", value=join_list_to_text_area(profile_data_for_dialog.get("skills")), height=150, key=f"edit_skills_{profile_id_to_edit}")
        st.markdown("---"); st.markdown("### 💡 Interests")
        updated_data["interests_text_area"] = st.text_area("Interests (one per line)", value=join_list_to_text_area(profile_data_for_dialog.get("interests")), height=100, key=f"edit_interests_{profile_id_to_edit}")
        st.markdown("---"); st.markdown("### 🎨 Hobbies")
        updated_data["hobbies_text_area"] = st.text_area("Hobbies (one per line)", value=join_list_to_text_area(profile_data_for_dialog.get("hobbies")), height=100, key=f"edit_hobbies_{profile_id_to_edit}")
        st.markdown("---"); st.markdown("### 🏆 Key Achievements (Overall)")
        updated_data["achievements_text_area"] = st.text_area("Key Achievements (one per line)", value=join_list_to_text_area(profile_data_for_dialog.get("keyAchievementsOverall")), height=150, key=f"edit_achievements_{profile_id_to_edit}")

        # Experiences
        st.markdown("---"); st.markdown("### 💼 Experiences")
        experiences_data = profile_data_for_dialog.get("experiences", [])
        updated_experiences = []
        for i, exp in enumerate(experiences_data):
            st.markdown(f"#### Experience {i+1}")
            role = st.text_input(f"Role##exp{i}", value=exp.get("role", ""), key=f"edit_exp_{i}_role_{profile_id_to_edit}")
            company = st.text_input(f"Company##exp{i}", value=exp.get("company", ""), key=f"edit_exp_{i}_company_{profile_id_to_edit}")
            dates = st.text_input(f"Dates##exp{i}", value=exp.get("dates", ""), key=f"edit_exp_{i}_dates_{profile_id_to_edit}")
            is_current = st.checkbox(f"Current/Primary Role##exp{i}", value=exp.get("isCurrentOrPrimary", False), key=f"edit_exp_{i}_current_{profile_id_to_edit}")
            brief_summary_mini = st.text_input(f"Brief Summary for Mini Card##exp{i}", value=exp.get("briefSummaryForMiniCard", ""), key=f"edit_exp_{i}_brief_summary_{profile_id_to_edit}")
            description = st.text_area(f"Full Description##exp{i}", value=exp.get("description", ""), height=120, key=f"edit_exp_{i}_desc_{profile_id_to_edit}")

            st.markdown(f"##### Skill Details for Experience {i+1}")
            current_skill_details = exp.get("skillDetails", [])
            updated_skill_details_for_exp = []
            for sd_idx, sd in enumerate(current_skill_details):
                st.markdown(f"###### Skill Detail {sd_idx+1}")
                sd_name = st.text_input(f"Skill Name##exp{i}sd{sd_idx}", value=sd.get("skillName", ""), key=f"edit_exp_{i}_sd_{sd_idx}_name_{profile_id_to_edit}")
                sd_context = st.text_area(f"Contextual Snippet##exp{i}sd{sd_idx}", value=sd.get("contextualSnippet", ""), height=70, key=f"edit_exp_{i}_sd_{sd_idx}_context_{profile_id_to_edit}")
                sd_related_text = join_list_to_comma_separated(sd.get("relatedSkillsInThisExperience", []))
                sd_related_input = st.text_input(f"Related Skills (comma-separated)##exp{i}sd{sd_idx}", value=sd_related_text, key=f"edit_exp_{i}_sd_{sd_idx}_related_{profile_id_to_edit}")
                if sd_name: # Only add if skill name is provided
                    updated_skill_details_for_exp.append({
                        "skillName": sd_name,
                        "contextualSnippet": sd_context,
                        "relatedSkillsInThisExperience": parse_comma_separated_to_list(sd_related_input)
                    })
            # TODO: Add button to add new skill detail item for an experience
            updated_experiences.append({
                "role": role, "company": company, "dates": dates,
                "isCurrentOrPrimary": is_current, "briefSummaryForMiniCard": brief_summary_mini,
                "description": description, "skillDetails": updated_skill_details_for_exp
            })
            st.markdown("---")
        # TODO: Add button to add new experience item
        updated_data["experiences"] = updated_experiences

        # Education
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
            updated_education.append({
                "institution": institution, "degree": degree, "fieldOfStudy": fieldOfStudy,
                "dates": dates_edu, "description": description_edu
            })
            st.markdown("---")
        # TODO: Add button to add new education item
        updated_data["education"] = updated_education

        # Projects
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
            updated_projects.append({
                "projectName": projectName, "datesOrDuration": datesOrDuration,
                "projectUrl": projectUrl, "description": description_proj,
                "skillsUsed": parse_text_area_to_list(skillsUsed_input)
            })
            st.markdown("---")
        # TODO: Add button to add new project item
        updated_data["projects"] = updated_projects

        submitted = st.form_submit_button("Save Changes")
        if submitted:
            final_profile_update = {
                "id": profile_id_to_edit, # Keep original ID
                "name": updated_data["name"], "title": updated_data["title"],
                "profilePictureUrlForCard": updated_data["profilePictureUrlForCard"],
                "taglineOrBriefSummary": updated_data["taglineOrBriefSummary"],
                "primaryProfileUrlForCard": updated_data["primaryProfileUrlForCard"],
                "callToActionForCard": updated_data["callToActionForCard"],
                "location": updated_data["location"],
                "skills": parse_text_area_to_list(updated_data["skills_text_area"]),
                "interests": parse_text_area_to_list(updated_data["interests_text_area"]),
                "hobbies": parse_text_area_to_list(updated_data["hobbies_text_area"]),
                "keyAchievementsOverall": parse_text_area_to_list(updated_data["achievements_text_area"]),
                "experiences": updated_data["experiences"],
                "education": updated_data["education"],
                "projects": updated_data["projects"],
            }
            if profile_index != -1:
                st.session_state.profiles_new_structure[profile_index] = final_profile_update
                st.toast("Profile updated successfully!", icon="✔️")
            else:
                st.toast("Error: Profile not found for update.", icon="❌")
            st.session_state.editing_profile_id_dialog = None
            st.rerun()

    if st.button("Cancel", key=f"cancel_edit_dialog_{profile_id_to_edit}"):
        st.session_state.editing_profile_id_dialog = None
        st.rerun()


# --- MAIN APP ---
def main():
    st.set_page_config(layout="wide", page_title="Coffee Card Showcase")
    init_session_state()
    load_css()

    st.title("☕ Coffee Card Profiles Showcase")
    st.caption("Displaying profiles with HTML/CSS hover effects, comprehensive native cards, and shareable previews.")
    st.markdown("---")

    if not st.session_state.get('profiles_new_structure'):
        st.info("No profiles yet. The example profile should load on first run.")
        # Potentially add a button to explicitly load/reset example data if needed
    else:
        card_type_to_show = st.radio(
            "Select Card Display Method:",
            ("HTML/CSS Card (Interactive Hover ✨)", "Native Streamlit Card (Comprehensive 🎈)"),
            index=0, # Default to HTML Card with hover
            horizontal=True,
            key="card_display_type_selector"
        )
        st.markdown("---")

        num_columns = 1
        if card_type_to_show == "HTML/CSS Card (Interactive Hover ✨)":
            # Allow for 2 columns if screen width is large, otherwise 1 for better popout visibility
            # This is a simple heuristic, true responsive column count is harder in Streamlit
            num_columns = st.session_state.get("num_display_columns", 2)


        cols = st.columns(num_columns)
        current_col_idx = 0

        for profile_data_loop in st.session_state.profiles_new_structure:
            profile_id_loop = profile_data_loop.get("id", str(uuid.uuid4())) # Ensure ID
            target_col = cols[current_col_idx % num_columns]

            with target_col:
                st.header(f"{html.escape(profile_data_loop.get('name', 'Profile'))}", divider="rainbow")

                if card_type_to_show == "HTML/CSS Card (Interactive Hover ✨)":
                    # st.subheader("Method 1: HTML/CSS Card ✨ (Interactive Hover)")
                    render_coffee_card_concise(profile_data_loop, profile_id_loop)
                    render_full_profile_details_expander(profile_data_loop, profile_id_loop)
                    st.markdown("<br>", unsafe_allow_html=True)

                elif card_type_to_show == "Native Streamlit Card (Comprehensive 🎈)":
                    # st.subheader("Method 2: Native Streamlit Card 🎈 (Comprehensive)")
                    render_coffee_card_native_comprehensive(profile_data_loop, profile_id_loop)
                    st.markdown("<br>", unsafe_allow_html=True)

                # Previews (Common to both card display methods)
                st.markdown("<h5 style='text-align: center; color: var(--cc-accent-dark-brown); margin-top:15px; margin-bottom:5px;'>🔗 Shareable Previews</h5>", unsafe_allow_html=True)
                preview_cols = st.columns(2)
                with preview_cols[0]:
                    render_apple_wallet_preview(profile_data_loop, profile_id_loop)
                with preview_cols[1]:
                    render_social_png_preview(profile_data_loop, profile_id_loop)
                st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True) # Spacer

            current_col_idx += 1

        if st.session_state.editing_profile_id_dialog:
            profile_id_to_show_dialog_for = st.session_state.editing_profile_id_dialog
            edit_profile_modal(profile_id_to_show_dialog_for)

if __name__ == "__main__":
    main()