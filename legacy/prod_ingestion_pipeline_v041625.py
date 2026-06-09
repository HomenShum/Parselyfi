import streamlit as st
import os
import json
import asyncio
import base64
import pandas as pd
import pymupdf  # PyMuPDF
from typing import List, Dict, Optional, Union, Any, Set
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict
from google import genai
from google.genai import types
import docx
from pptx import Presentation
import csv
import openpyxl
import re
from collections import Counter, defaultdict
import time
import io
import datetime
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

###############################
# DATA MODELS
###############################

class VisualElement(BaseModel):
    """Model for visual elements like charts, graphs, etc."""
    type: str = Field(..., description="Type of visual element")
    description: str = Field(..., description="Description of the visual")
    data_summary: Optional[str] = Field(None, description="Summary of the data")
    page_numbers: List[int] = Field(default_factory=list, description="Pages where this appears")
    source_url: Optional[str] = Field(None, description="Source URL of the visual")
    alt_text: Optional[str] = Field(None, description="Alternative text for the visual")
    visual_id: Optional[str] = Field(None, description="Unique identifier for the visual")

class NumericalDataPoint(BaseModel):
    """Model for numerical data points extracted from text."""
    value: str = Field(..., description="The numerical value")
    description: str = Field(default="", description="What the number represents")
    context: str = Field(default="", description="Surrounding text context")
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")
    @field_validator('value', 'description', 'context', mode='before')
    def convert_to_string(cls, v): return str(v) if v is not None else ""

class TableData(BaseModel):
    """Model for tables extracted from documents."""
    table_content: str = Field(default="", description="Markdown formatted table content")
    title: Optional[str] = None
    summary: Optional[str] = None
    page_number: int = 0
    table_id: Optional[str] = Field(None, description="Unique identifier for the table")
    model_config = ConfigDict(arbitrary_types_allowed=True)
    @field_validator('table_content')
    def clean_table_content(cls, v): return str(v) if v is not None else ""

class FinancialMetric(BaseModel):
    """Model for financial metrics extracted from documents."""
    name: str = Field(..., description="Name of the financial metric")
    value: str = Field(..., description="Metric value")
    period: Optional[str] = Field(None, description="Reporting period")
    context: Optional[str] = Field(None, description="Surrounding text context")
    trend: Optional[str] = Field(None, description="Trend direction if available")

class FinancialStatement(BaseModel):
    """Model for financial statements extracted from documents."""
    type: str = Field(..., description="Type of statement")
    period: str = Field(..., description="Reporting period")
    metrics: List[FinancialMetric] = Field(default_factory=list, description="Key metrics")

class Subsection(BaseModel):
    """Model for subsections extracted from pages."""
    subsection_id: str = Field(..., description="Unique identifier for the subsection")
    order: int = Field(..., description="Order of the subsection within the page")
    title: str = Field(..., description="Title of the subsection")
    content: str = Field(..., description="Raw content of the subsection")
    is_cutoff: bool = Field(False, description="True if content appears to be cut off")
    referenced_visuals: List[str] = Field(default_factory=list, description="IDs of referenced visuals")
    referenced_tables: List[str] = Field(default_factory=list, description="IDs of referenced tables")
    page_number: int = Field(..., description="Page number where this subsection appears")

class EntityRelationship(BaseModel):
    """Model for entity relationships within documents."""
    source_entity: str = Field(..., description="Source entity name")
    target_entity: str = Field(..., description="Target entity name")
    relationship_description: str = Field(..., description="Description of the relationship")
    relationship_keywords: List[str] = Field(default_factory=list, description="Keywords describing the relationship")
    relationship_strength: float = Field(..., description="Strength of the relationship (0-10)")

class Chapter(BaseModel):
    """Model for chapters composed of subsections."""
    chapter_id: str = Field(..., description="Unique identifier for the chapter")
    title: str = Field(..., description="Title of the chapter")
    summary: str = Field(..., description="Summary of the chapter")
    subsections: List[Subsection] = Field(default_factory=list, description="List of subsections in this chapter")
    entity_relationships: List[EntityRelationship] = Field(default_factory=list, description="Entity relationships within the chapter")
    order: int = Field(..., description="Order of the chapter in the document")
    key_points: List[str] = Field(default_factory=list, description="Key takeaways from the chapter")
    explicit_references: List[str] = Field(default_factory=list, description="Referenced tables and visuals")
    questions_answers: List[dict] = Field(default_factory=list, description="Q&A pairs from chapter content")

class PageContent(BaseModel):
    """Model for content extracted from a single page."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    page_number: int = Field(..., description="Page number in document")
    text: str = Field("", description="Full text content")
    title: str = Field("Untitled", description="Brief title for this page")
    topics: List[str] = Field(default_factory=list, description="Key topics discussed")
    summary: str = Field("", description="Summary of key points")
    entities: List[str] = Field(default_factory=list, description="Important entities mentioned")
    has_tables: bool = Field(False, description="True if page contains tables")
    has_visuals: bool = Field(False, description="True if page contains visuals")
    has_numbers: bool = Field(False, description="True if page contains key numerical data")
    tables: List[Union[TableData, dict]] = Field(default_factory=list, description="Extracted tables")
    visuals: List[Union[VisualElement, dict]] = Field(default_factory=list, description="Extracted visual elements")
    numbers: List[Union[NumericalDataPoint, dict]] = Field(default_factory=list, description="Extracted numerical data")
    dates: List[str] = Field(default_factory=list, description="Important dates mentioned")
    financial_statements: List[Union[FinancialStatement, dict]] = Field(default_factory=list)
    key_metrics: List[Union[FinancialMetric, dict]] = Field(default_factory=list)
    financial_terms: List[str] = Field(default_factory=list)
    subsections: List[Union[Subsection, dict]] = Field(default_factory=list, description="Extracted subsections")

class DocumentSummary(BaseModel):
    """Document-level summary."""
    title: str = Field(..., description="Concise title/summary")
    themes: List[str] = Field(..., description="Concept/theme tags")
    questions: List[str] = Field(..., description="Hypothetical questions")
    summary: str = Field(..., description="Comprehensive summary")
    tables_summary: Optional[str] = Field(None, description="Summary of key tables")
    visuals_summary: Optional[str] = Field(None, description="Summary of key visuals")
    chapters: List[Chapter] = Field(default_factory=list, description="Extracted chapters")
    entity_relationships: List[EntityRelationship] = Field(default_factory=list, description="Document-level entity relationships")
    table_of_contents: List[dict] = Field(default_factory=list, description="Table of contents")
    chapter_relationship_graph: dict = Field(default_factory=dict, description="Relationship graph between chapters")

class KeyTopic(BaseModel):
    """Represents a key topic in the document."""
    name: str = Field(..., description="Short topic name (1-3 words)")
    description: str = Field(..., description="Brief description of the topic")
    relevance: str = Field(..., description="Relevance level (High/Medium/Low)")
    sentiment: str = Field(..., description="Sentiment analysis (Positive/Neutral/Mixed/Cautionary/Negative)")
    analysis: str = Field(..., description="Brief analysis of the topic")

class QuotedStatement(BaseModel):
    """Represents an important quoted statement in the document."""
    speaker: str = Field(..., description="Person or entity who made the statement")
    quote: str = Field(..., description="The quoted text")
    page: int = Field(..., description="Page number where the quote appears")

class DocumentReport(BaseModel):
    """UI-friendly document report structure for presentation."""
    file_name: str = Field(..., description="Original filename")
    page_count: int = Field(..., description="Number of pages in document")
    title: str = Field(..., description="Document title")
    title_summary: str = Field(..., description="Brief summary/subtitle")
    concept_theme_hashtags: List[str] = Field(default_factory=list, description="Theme tags as hashtags")
    date_published: Optional[str] = Field(None, description="Publication date if available")
    source: Optional[str] = Field(None, description="Document source")
    confidence: str = Field("Medium", description="Analysis confidence (High/Medium/Low)")
    document_summary: str = Field(..., description="Comprehensive summary")
    key_insights: List[str] = Field(default_factory=list, description="Key takeaways and insights")
    key_topics: List[KeyTopic] = Field(default_factory=list, description="Detailed topic analysis")
    quoted_statements: List[QuotedStatement] = Field(default_factory=list, description="Important quotes from document")
    content_excerpt: Optional[str] = Field(None, description="Highlighted document content")
    chapters: List[Chapter] = Field(default_factory=list, description="Document chapters")

class ProcessedDocument(BaseModel):
    """Fully processed document with all content."""
    filename: str = Field(..., description="Original filename")
    pages: List[PageContent] = Field(..., description="Processed pages")
    summary: Optional[DocumentSummary] = None

class ProjectOntology(BaseModel):
    """Project-wide ontology generated from multiple documents."""
    title: str = Field(..., description="Project title")
    overview: str = Field(..., description="Project overview")
    document_count: int = Field(..., description="Number of documents in the project")
    documents: List[str] = Field(..., description="List of document titles")
    global_themes: List[str] = Field(..., description="High-level project themes")
    entity_relationships: List[EntityRelationship] = Field(..., description="Project-wide entity relationships")
    key_concepts: List[str] = Field(..., description="Key concepts across all documents")

###############################
# API AND UTILITY FUNCTIONS
###############################

def convert_to_model(data, model_class):
    """Convert dictionary data to a Pydantic model with error handling."""
    if not isinstance(data, dict):
        print(f"Warning: Expected dict for {model_class.__name__}, got {type(data)}")
        return None
    
    try:
        # Extract only the fields that the model expects
        model_fields = model_class.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in model_fields}
        return model_class(**filtered_data)
    except Exception as e:
        print(f"Error converting to {model_class.__name__}: {str(e)}")
        return None

def get_gemini_api_key():
    """Get the Gemini API key from environment or secrets."""
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("🚫 Gemini API key not found. Please set the GEMINI_API_KEY environment variable or in Streamlit secrets.")
        st.stop()
    return api_key

async def retry_api_call(func, *args, max_retries=3, **kwargs):
    """Retry API call with exponential backoff and JSON validation."""
    for attempt in range(max_retries):
        try:
            response = await func(*args, **kwargs)
            
            # Validate JSON if applicable
            config = kwargs.get('config')
            if (config and hasattr(config, 'response_mime_type') and 
                config.response_mime_type == 'application/json' and 
                hasattr(response, 'candidates') and response.candidates):
                try:
                    json_text = response.candidates[0].content.parts[0].text
                    json.loads(clean_json_response(json_text))
                except json.JSONDecodeError as e:
                    if attempt < max_retries - 1:
                        print(f"Received malformed JSON on attempt {attempt+1}, retrying: {e}")
                        await asyncio.sleep(2 ** attempt)
                        continue
            
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"API call failed on attempt {attempt+1}, retrying: {e}")
            await asyncio.sleep(2 ** attempt)

def clean_json_response(json_text: str, extract_text_on_failure=True) -> str:
    """Clean Gemini JSON response with improved error handling and text extraction fallback."""
    if json_text is None:
        return "{}"
    
    # Ensure it's a string
    json_text = str(json_text)
    
    try:
        # Handle markdown code blocks
        if json_text.startswith("```"):
            blocks = json_text.split("```")
            if len(blocks) >= 3:
                json_text = blocks[1]
                if json_text.startswith("json"):
                    json_text = json_text[4:].strip()
            else:
                json_text = json_text.replace("```", "").strip()
        
        elif json_text.startswith("json"):
            json_text = json_text[4:].strip()
        
        # Remove any leading/trailing whitespace
        json_text = json_text.strip()
        
        # Attempt to parse the JSON as is
        try:
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            # Continue with fixes
            pass
        
        # Common JSON fixes
        # Fix missing quotes around property names
        json_text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_text)
        
        # Fix trailing commas in arrays/objects
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
        
        # Final attempt to parse with Python's JSON parser
        try:
            data = json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            # If we're extracting text on failure and can't fix the JSON
            if extract_text_on_failure:
                # Create fallback with minimal content
                fallback_json = {
                    "text": "Fallback extraction",
                    "title": "Fallback",
                    "topics": [],
                    "summary": "Response processing failed - returning raw text.",
                    "entities": [],
                    "has_tables": False,
                    "has_visuals": False,
                    "has_numbers": False,
                    "dates": [],
                    "tables": [],
                    "visuals": [],
                    "numbers": [],
                    "key_metrics": [],
                    "financial_terms": [],
                    "subsections": []
                }
                return json.dumps(fallback_json)
            else:
                return '{"text":"Error parsing JSON response","error":"Invalid JSON structure"}'
            
    except Exception as e:
        print(f"Error cleaning JSON response: {e}")
        
        if extract_text_on_failure:
            # Create a fallback with whatever raw content we have
            fallback_json = {
                "text": "Error occurred",
                "title": "Raw Text Fallback",
                "topics": ["raw text"],
                "summary": "Response processing failed - returning raw text.",
                "entities": [],
                "has_tables": False,
                "has_visuals": False,
                "has_numbers": False,
                "dates": [],
                "tables": [],
                "visuals": [],
                "numbers": [],
                "key_metrics": [],
                "financial_terms": [],
                "subsections": []
            }
            return json.dumps(fallback_json)
        else:
            return '{"text":"Error cleaning JSON response","error":"' + str(e).replace('"', '\\"') + '"}'

def run_async(func, *args, **kwargs):
    """Run an async function from Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(func(*args, **kwargs))
    finally:
        loop.close()

def safe_async_run(coro):
    """Run async function safely with appropriate error handling."""
    try:
        return run_async(lambda: coro)
    except Exception as e:
        print(f"Error in async operation: {str(e)}")
        st.error(f"Operation failed: {str(e)}")
        return None

def create_error_page(page_num: int, error_msg: str, validation_errors: Optional[List] = None) -> PageContent:
    """Creates a PageContent object representing an error state."""
    error_text = f"Error processing page {page_num}: {error_msg}"

    if validation_errors:
        try:
            error_details = "\nValidation Errors:\n" + "\n".join(
                f"- Field '{'.'.join(map(str, err.get('loc', ['unknown'])))}': {err.get('msg', 'No message')}"
                for err in validation_errors
            )
            error_text += error_details
        except Exception as format_error:
            error_text += f"\n(Could not format validation errors: {format_error})"

    # Initialize the PageContent object with standard fields
    error_page = PageContent(
        page_number=page_num,
        text=error_text,
        title=f"Page {page_num} (Error)",
        topics=["error"],
        summary=f"Failed to process page {page_num}.",
        entities=[],
        has_tables=False,
        has_visuals=False,
        has_numbers=False,
        tables=[],
        visuals=[],
        numbers=[],
        dates=[],
        financial_statements=[],
        key_metrics=[],
        financial_terms=[],
        subsections=[]
    )

    return error_page

###############################
# STEP 1.A INITIAL EXTRACTION
###############################

async def extract_page_content_from_memory(client, image_part, page_num, filename=None):
    """Extract content from a single page using the optimized prompt."""
    # Initialize page data with defaults
    page_data = {
        "page_number": page_num,
        "text": "",
        "title": f"Page {page_num}",
        "topics": [],
        "summary": "",
        "entities": [],
        "has_tables": False,
        "has_visuals": False,
        "has_numbers": False,
        "tables": [],
        "visuals": [],
        "numbers": [],
        "dates": [],
        "financial_statements": [],
        "key_metrics": [],
        "financial_terms": [],
        "subsections": []
    }

    # Combined prompt for initial extraction following Step 1.A
    combined_prompt = f"""
    Analyze page {page_num} from the input document. Extract the core content and structural elements. Return a JSON object with this exact structure:
    {{
      "text": "full text content extracted from the page or text chunk",
      "title": "brief title summarizing the page's main content or heading (2–5 words)",
      "has_tables": true/false,
      "has_visuals": true/false,
      "has_numbers": true/false,
      "tables": [
        {{
          "table_id": "page_{page_num}_table_1",
          "table_content": "markdown formatted table. Accurately extract all text content within the table.",
          "title": "optional table title, if explicitly present",
          "summary": "optional brief summary (1–2 sentences) of the table's purpose or key data shown",
          "page_number": {page_num}
        }}
        // ... more tables if present
      ],
      "visuals": [
        {{
          "visual_id": "page_{page_num}_visual_1",
          "type": "Type of visual (e.g., bar chart, line graph, diagram, flowchart, equation, image, code block)",
          "description": "detailed description of the visual: its purpose, elements shown, and overall message",
          "data_summary": "summary of key data/trends presented, or \\"N/A\\" if decorative",
          "page_numbers": [{page_num}],
          "source_url": "URL if available, otherwise null",
          "alt_text": "Alternative text if available, otherwise null"
        }}
        // ... more visuals if present
      ],
      "numbers": [
        {{
          "value": "string numeric value (e.g., '123.45', '1e-6', '50%', '3/4', 'formula (1)')",
          "description": "what the number represents, with units/context",
          "context": "surrounding text snippet providing context"
        }}
        // ... more numbers if present
      ]
    }}

    Rules:
    1. Include ALL fields; use [] for empty arrays and set has_… flags accordingly.
    2. Maintain exact field names & structure; use double quotes.
    3. Assign unique sequential IDs (page_{page_num}_table_X, page_{page_num}_visual_X).
    4. Tables must be valid markdown.
    5. CRITICAL: Return ONLY the JSON.
    """

    try:
        response = await retry_api_call(
            client.aio.models.generate_content,
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[image_part, types.Part.from_text(text=combined_prompt)]),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )

        if response.candidates:
            json_text = clean_json_response(response.candidates[0].content.parts[0].text)
            data = json.loads(json_text)
            
            # Update page data with extracted content
            page_data.update(data)
            
            # Ensure lists are properly initialized
            for key in ["tables", "visuals", "numbers", "dates", "financial_terms", "subsections"]:
                if key not in page_data:
                    page_data[key] = []
                elif page_data[key] is None:
                    page_data[key] = []

    except Exception as e:
        print(f"Error in initial extraction for page {page_num}: {str(e)}")
        if filename:
            print(f"File: {filename}")
        return PageContent(**create_error_page(page_num, str(e), None).model_dump())

    return PageContent(**page_data)

###############################
# STEP 1.B SECONDARY ANALYSIS
###############################

async def perform_secondary_analysis(client, page_data, page_num):
    """Perform secondary analysis on the extracted page content."""
    # Convert to dict if it's a PageContent object
    if hasattr(page_data, 'model_dump'):
        page_dict = page_data.model_dump()
    else:
        page_dict = page_data
    
    # Prompt for secondary analysis following Step 1.B
    prompt = f"""
    Analyze the provided JSON data from page {page_num}. Focus primarily on the "text" field, using "tables", "visuals", and "numbers" for context. Identify topics, summarize key points, extract named entities, dates, key terms, quantifiable data points, and logical subsections. Adapt to the apparent domain (scientific, technical, narrative, etc.). Return a new JSON object with exactly:

    {{
      "topics": ["topic1","topic2"],            // ≤5
      "summary": "3–5 sentence key-point summary",
      "entities": ["entity1","entity2"],
      "dates": ["date1","date2"],
      "key_terms": ["term1","term2"],
      "key_data_points": [
        {{
          "name": "metric/result name",
          "value": "string value",
          "unit": "unit if applicable",
          "period_or_condition": "time period or condition",
          "trend_or_comparison": "trend/comparison if mentioned",
          "context": "brief context"
        }}
        // … more data points
      ],
      "subsections": [
        {{
          "subsection_id": "page_{page_num}_section_1",
          "order": 1,
          "title": "subsection title (≤7 words)",
          "content": "raw text for this subsection",
          "is_cutoff": true/false,
          "referenced_visuals": ["page_{page_num}_visual_X"],
          "referenced_tables": ["page_{page_num}_table_Y"]
        }}
        // … more subsections
      ]
    }}

    Rules:
    1. Include ALL fields; use [] where empty.
    2. Exact field names & structure; double quotes.
    3. CRITICAL: Return ONLY the JSON.
    """
    
    # Convert page_data to JSON for the prompt
    page_json = json.dumps(page_dict)
    
    try:
        response = await retry_api_call(
            client.aio.models.generate_content,
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_text(text=page_json)
                ]),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        if response.candidates:
            json_text = clean_json_response(response.candidates[0].content.parts[0].text)
            data = json.loads(json_text)
            
            # Update the original page_data with the secondary analysis
            page_dict.update(data)
            
            # Convert key_data_points to the key_metrics format used in the existing code
            if "key_data_points" in data:
                page_dict["key_metrics"] = []
                for point in data["key_data_points"]:
                    metric = {
                        "name": point.get("name", ""),
                        "value": point.get("value", ""),
                        "period": point.get("period_or_condition", ""),
                        "trend": point.get("trend_or_comparison", ""),
                        "context": point.get("context", "")
                    }
                    page_dict["key_metrics"].append(metric)
            
            # Convert subsections to make sure they're properly formatted
            if "subsections" in data and data["subsections"]:
                formatted_subsections = []
                for sub in data["subsections"]:
                    # Ensure page_number is set
                    if "page_number" not in sub:
                        sub["page_number"] = page_num
                    formatted_subsections.append(sub)
                page_dict["subsections"] = formatted_subsections
                    
            return page_dict
            
    except Exception as e:
        print(f"Error in secondary analysis for page {page_num}: {str(e)}")
        return page_dict  # Return original data if analysis fails

###############################
# STEP 2 CHAPTER GROUPING
###############################

async def extract_chapters_from_subsections(client, pages):
    """Extract chapters from subsections using the optimized prompt."""
    # Extract all subsections from all pages
    all_subsections = []
    for page in pages:
        if hasattr(page, 'subsections') and page.subsections:
            # Add page_number field to each subsection if not already present
            for subsection in page.subsections:
                if not hasattr(subsection, 'page_number'):
                    subsection.page_number = page.page_number
            all_subsections.extend(page.subsections)
    
    if not all_subsections:
        # Create a default chapter if no subsections found
        default_chapter = Chapter(
            chapter_id="chapter_1",
            title="Document Content",
            summary="Complete document content without chapter structure.",
            subsections=[],
            entity_relationships=[],
            order=1,
            key_points=[],
            explicit_references=[],
            questions_answers=[]
        )
        
        # Create a subsection for each page
        for page in pages:
            default_subsection = Subsection(
                subsection_id=f"page_{page.page_number}_section_1",
                order=page.page_number,
                title=page.title,
                content=page.text,
                is_cutoff=False,
                referenced_visuals=[],
                referenced_tables=[],
                page_number=page.page_number
            )
            default_chapter.subsections.append(default_subsection)
        
        return [default_chapter]
    
    # Sort subsections by page number and order
    sorted_subsections = sorted(all_subsections, key=lambda s: (s.page_number, s.order))
    
    # Prepare subsections data for the prompt
    subsections_data = []
    for subsection in sorted_subsections:
        subsections_data.append({
            "subsection_id": subsection.subsection_id,
            "title": subsection.title,
            "page_number": subsection.page_number,
            "order": subsection.order,
            "content_preview": subsection.content[:100] + "..." if len(subsection.content) > 100 else subsection.content
        })
    
    # Create prompt for chapter extraction following Step 2
    chapter_prompt = f"""
    Analyze the following list of subsection titles and their IDs. Group these subsections into logical chapters (3–10 total). Assign each chapter a concise, descriptive title and maintain document order. Return ONLY a JSON array:

    [
      {{
        "chapter_id": "chapter_1_methodology",
        "title": "Algorithm and Methodology",
        "order": 1,
        "subsection_ids": ["page_1_section_1","page_1_section_2"]
      }},
      {{
        "chapter_id": "chapter_2_results",
        "title": "Experimental Results",
        "order": 2,
        "subsection_ids": ["page_3_section_1","page_3_section_2"]
      }}
    ]

    Subsections data:
    {json.dumps(subsections_data, indent=2)}
    """
    
    try:
        response = await retry_api_call(
            client.aio.models.generate_content,
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[types.Part.from_text(text=chapter_prompt)]),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        if response.candidates:
            json_text = clean_json_response(response.candidates[0].content.parts[0].text)
            chapters_data = json.loads(json_text)
            
            # Convert to Chapter objects
            chapters = []
            for chapter_data in chapters_data:
                # Get subsections for this chapter
                chapter_subsections = []
                for subsection_id in chapter_data.get("subsection_ids", []):
                    for subsection in sorted_subsections:
                        if subsection.subsection_id == subsection_id:
                            chapter_subsections.append(subsection)
                            break
                
                # Create the chapter
                chapter = Chapter(
                    chapter_id=chapter_data.get("chapter_id", f"chapter_{len(chapters)+1}"),
                    title=chapter_data.get("title", f"Chapter {len(chapters)+1}"),
                    summary="",  # Will be filled in later
                    subsections=chapter_subsections,
                    entity_relationships=[],
                    order=chapter_data.get("order", len(chapters)+1),
                    key_points=[],
                    explicit_references=[],
                    questions_answers=[]
                )
                chapters.append(chapter)
            
            # Sort chapters by order
            chapters = sorted(chapters, key=lambda c: c.order)
            
            # If no chapters were created, create a default chapter
            if not chapters:
                default_chapter = Chapter(
                    chapter_id="chapter_1",
                    title="Document Content",
                    summary="Complete document content without chapter structure.",
                    subsections=sorted_subsections,
                    entity_relationships=[],
                    order=1,
                    key_points=[],
                    explicit_references=[],
                    questions_answers=[]
                )
                chapters = [default_chapter]
            
            return chapters
            
    except Exception as e:
        print(f"Error extracting chapters: {str(e)}")
        
        # Create a default chapter on error
        default_chapter = Chapter(
            chapter_id="chapter_1",
            title="Document Content",
            summary="Error occurred during chapter extraction.",
            subsections=sorted_subsections,
            entity_relationships=[],
            order=1,
            key_points=[],
            explicit_references=[],
            questions_answers=[]
        )
        return [default_chapter]

###############################
# STEP 3.B+C ENTITY & CONCEPT ANALYSIS
###############################

async def analyze_entity_relationships(client, chapter):
    """Analyze entity relationships and concept model using the combined prompt."""
    # Collect content from all subsections
    subsection_texts = []
    for subsection in chapter.subsections:
        subsection_texts.append(f"--- {subsection.title} ---\n{subsection.content}")
    
    # Combine texts while keeping under token limit
    combined_text = "\n\n".join(subsection_texts)
    if len(combined_text) > 100000:
        combined_text = combined_text[:100000] + "..."
    
    # Create prompt for entity and concept analysis following Step 3.B+3.C
    entity_prompt = f"""
    Analyze the chapter text below. Return a JSON object with two sections:

    {{
      "entity_relationships": [
        {{
          "source_entity": "...",
          "target_entity": "...",
          "relationship_description": "...",
          "relationship_strength": 1–10,
          "relationship_keywords": ["…"]
        }}
        // … more
      ],
      "core_abstractions": [
        {{ "name": "...", "description": "30–60 word beginner‑friendly description" }}
        // … top 3–5
      ],
      "concept_relationships": [
        {{
          "source_abstraction": "...",
          "target_abstraction": "...",
          "relationship_description": "..."
        }}
        // …
      ],
      "concept_flow_diagram_data": {{
        "nodes": [ {{ "id":"…","label":"…" }}, … ],
        "edges": [ {{ "source":"…","target":"…","label":"…" }}, … ]
      }}
    }}

    Chapter: {chapter.title}
    
    Text:
    {combined_text}
    """
    
    try:
        response = await retry_api_call(
            client.aio.models.generate_content,
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[types.Part.from_text(text=entity_prompt)]),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        if response.candidates:
            json_text = clean_json_response(response.candidates[0].content.parts[0].text)
            data = json.loads(json_text)
            
            # Convert to EntityRelationship objects
            relationships = []
            for rel_data in data.get("entity_relationships", []):
                try:
                    relationship = EntityRelationship(
                        source_entity=rel_data.get("source_entity", "Unknown"),
                        target_entity=rel_data.get("target_entity", "Unknown"),
                        relationship_description=rel_data.get("relationship_description", ""),
                        relationship_keywords=rel_data.get("relationship_keywords", []),
                        relationship_strength=float(rel_data.get("relationship_strength", 5))
                    )
                    relationships.append(relationship)
                except Exception as rel_error:
                    print(f"Error creating relationship: {rel_error}")
            
            # Store concept data for later use
            concept_data = {
                "core_abstractions": data.get("core_abstractions", []),
                "concept_relationships": data.get("concept_relationships", []),
                "concept_flow_diagram_data": data.get("concept_flow_diagram_data", {"nodes": [], "edges": []})
            }
            
            return relationships, concept_data
            
    except Exception as e:
        print(f"Error analyzing entity relationships: {str(e)}")
        return [], {"core_abstractions": [], "concept_relationships": [], "concept_flow_diagram_data": {"nodes": [], "edges": []}}

###############################
# STEP 3.A+D CHAPTER SUMMARY & Q&A
###############################

async def generate_chapter_summary(client, chapter):
    """Generate chapter summary and Q&A using the combined prompt."""
    # Collect content from all subsections
    subsection_texts = []
    for subsection in chapter.subsections:
        subsection_texts.append(f"--- {subsection.title} ---\n{subsection.content}")
    
    # Combine texts while keeping under token limit
    combined_text = "\n\n".join(subsection_texts)
    if len(combined_text) > 100000:
        combined_text = combined_text[:100000] + "..."
    
    # Create prompt for summary generation following Step 3.A+3.D
    summary_prompt = f"""
    Given the chapter titled "{chapter.title}" with content:
    {combined_text}

    1. Generate a concise summary (3–6 sentences).
    2. List 3–5 key takeaways (callouts).
    3. Enumerate explicit references to tables/visuals by their IDs.
    4. Generate 3 Q&A pairs (questions the text answers + concise answers).

    Return ONLY a JSON object:
    {{
      "summary": "...",
      "key_points": ["…","…"],
      "explicit_references": ["page_1_table_1","page_3_visual_2"],
      "questions_answers": [
        {{ "question": "…", "answer": "…" }},
        …
      ]
    }}
    """
    
    try:
        response = await retry_api_call(
            client.aio.models.generate_content,
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[types.Part.from_text(text=summary_prompt)]),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        if response.candidates:
            json_text = clean_json_response(response.candidates[0].content.parts[0].text)
            data = json.loads(json_text)
            
            # Update chapter summary
            if "summary" in data:
                chapter.summary = data["summary"]
            
            # Store additional data in chapter object
            chapter.key_points = data.get("key_points", [])
            chapter.explicit_references = data.get("explicit_references", [])
            chapter.questions_answers = data.get("questions_answers", [])
            
            return chapter
            
    except Exception as e:
        print(f"Error generating chapter summary: {str(e)}")
        # No change to chapter on error
        return chapter

###############################
# STEP 4 FILE OVERVIEW
###############################

async def generate_document_summary(client, chapters, filename):
    """Generate file overview using the combined prompt."""
    # Prepare chapters data for the prompt
    chapters_data = []
    for chapter in chapters:
        chapters_data.append({
            "chapter_id": chapter.chapter_id,
            "title": chapter.title,
            "summary": chapter.summary
        })
    
    # Create prompt for file overview following Step 4.A+4.B+4.C
    summary_prompt = f"""
    Given these chapters:
    {json.dumps(chapters_data, indent=2)}

    1. Produce a Table of Contents:  [ {{ "order":1,"title":"…" }}, … ]
    2. Synthesize a 100–200 word file summary.
    3. Build a chapter relationship graph: 
       {{ "nodes":[{{"id":"…","label":"…"}},…], "edges":[{{"source":"…","target":"…","label":"…"}},…] }}

    Return ONLY a JSON object:
    {{
      "table_of_contents": […],
      "file_summary": "…",
      "chapter_relationship_graph": {{
        "nodes":[…],
        "edges":[…]
      }}
    }}
    """
    
    try:
        response = await retry_api_call(
            client.aio.models.generate_content,
            model="gemini-2.0-flash", 
            contents=[
                types.Content(parts=[types.Part.from_text(text=summary_prompt)]),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        if response.candidates:
            json_text = clean_json_response(response.candidates[0].content.parts[0].text)
            summary_data = json.loads(json_text)
            
            # Create document summary
            doc_summary = {
                "title": f"Summary: {filename}",
                "themes": get_themes_from_chapters(chapters),
                "questions": get_questions_from_chapters(chapters),
                "summary": summary_data.get("file_summary", "Generated summary unavailable"),
                "tables_summary": get_tables_summary(chapters),
                "visuals_summary": get_visuals_summary(chapters),
                "chapters": [chapter.model_dump() for chapter in chapters],
                "entity_relationships": get_all_entity_relationships(chapters),
                "table_of_contents": summary_data.get("table_of_contents", []),
                "chapter_relationship_graph": summary_data.get("chapter_relationship_graph", {"nodes":[], "edges":[]})
            }
            
            return doc_summary
            
    except Exception as e:
        print(f"Summary generation error: {str(e)}")
    
    # Fallback summary
    return {
        "title": f"Summary: {filename}",
        "themes": get_themes_from_chapters(chapters),
        "questions": ["What are the key points in this document?"],
        "summary": "Generated summary unavailable",
        "tables_summary": "",
        "visuals_summary": "",
        "chapters": [chapter.model_dump() for chapter in chapters],
        "entity_relationships": get_all_entity_relationships(chapters)
    }

def get_themes_from_chapters(chapters):
    """Extract themes from chapter summaries."""
    themes = set()
    for chapter in chapters:
        # Extract keywords from summaries
        if chapter.summary:
            words = re.findall(r'\b\w+\b', chapter.summary.lower())
            # Filter common words, keep only significant terms
            significant_terms = [w for w in words if len(w) > 4 and w not in ['about', 'these', 'those', 'their', 'there']]
            themes.update(significant_terms[:3])  # Add top 3 significant terms
    return list(themes)[:5]  # Return up to 5 themes

def get_questions_from_chapters(chapters):
    """Extract questions from chapter Q&A sections."""
    questions = []
    for chapter in chapters:
        if hasattr(chapter, 'questions_answers'):
            for qa in chapter.questions_answers:
                if isinstance(qa, dict) and 'question' in qa:
                    questions.append(qa['question'])
    if not questions:
        questions = ["What is the main topic of this document?", 
                    "What are the key findings or conclusions?", 
                    "How is the document structured?"]
    return questions[:3]  # Return up to 3 questions

def get_tables_summary(chapters):
    """Generate a summary of tables in the document."""
    table_count = 0
    for chapter in chapters:
        for subsection in chapter.subsections:
            table_count += len(subsection.referenced_tables)
    
    if table_count > 0:
        return f"The document contains {table_count} tables providing supporting data and evidence."
    return ""

def get_visuals_summary(chapters):
    """Generate a summary of visuals in the document."""
    visual_count = 0
    for chapter in chapters:
        for subsection in chapter.subsections:
            visual_count += len(subsection.referenced_visuals)
    
    if visual_count > 0:
        return f"The document includes {visual_count} visual elements such as charts, diagrams, or images."
    return ""

def get_all_entity_relationships(chapters):
    """Collect all entity relationships from chapters."""
    relationships = []
    for chapter in chapters:
        for rel in chapter.entity_relationships:
            relationships.append(rel.model_dump() if hasattr(rel, 'model_dump') else rel)
    return relationships

###############################
# STEP 5-6 FOLDER LEVEL PROCESSING
###############################

async def generate_folder_level_ontology(client, processed_documents):
    """Generate folder-level clustering and ontology."""
    # Prepare documents data for the prompt
    docs_data = []
    for doc in processed_documents:
        if isinstance(doc, dict) and "raw_extracted_content" in doc:
            raw_content = doc["raw_extracted_content"]
            filename = raw_content.get("filename", "Unknown")
            
            # Get summary if available
            summary = ""
            if "summary" in raw_content and raw_content["summary"]:
                summary = raw_content["summary"].get("summary", "")
            
            # Get table of contents if available
            toc = []
            if "summary" in raw_content and raw_content["summary"] and "chapters" in raw_content["summary"]:
                for chapter in raw_content["summary"]["chapters"]:
                    toc.append({
                        "order": chapter.get("order", 0),
                        "title": chapter.get("title", "Untitled Chapter")
                    })
            
            # Get chapter relationship graph if available
            graph = {}
            if "summary" in raw_content and raw_content["summary"] and "chapter_relationship_graph" in raw_content["summary"]:
                graph = raw_content["summary"]["chapter_relationship_graph"]
            
            docs_data.append({
                "filename": filename,
                "file_summary": summary,
                "table_of_contents": toc,
                "chapter_relationship_graph": graph
            })
    
    # Create prompt for folder-level ontology following Step 5
    ontology_prompt = f"""
    You have processed {len(docs_data)} documents in a folder. For each, you have:
    - filename
    - file_summary
    - table_of_contents (with chapter titles/orders)
    - file_overview.chapter_relationship_graph

    1. **Cluster Generation**  
       Group these documents into 3–7 thematic clusters. For each cluster, output:
       - cluster_id: "cluster_1_research_methods"
       - title: concise cluster name
       - file_names: [ "DocA.pdf", "DocB.pdf", … ]
    2. **Corpus Ontology**  
       From all chapter titles across files, extract 5–10 high‑level topics/concepts. For each:
       - name
       - description (30–60 words)
       - related_files: [ … ]
       - related_chapters: [ "DocA:Methodology", "DocC:Background", … ]
    3. **Corpus Relationship Graph**  
       Build a directed graph showing how clusters or concepts interlink. Provide:
       - nodes: [ {{ id, label, type: "cluster"/"concept"/"file" }} ]
       - edges: [ {{ source, target, label }} ]

    Return ONLY a JSON object:
    {{
      "document_clusters": [ … ],
      "corpus_ontology": [ … ],
      "corpus_relationship_graph": {{
        "nodes": [ … ],
        "edges": [ … ]
      }}
    }}

    Documents data:
    {json.dumps(docs_data, indent=2)}
    """
    
    try:
        response = await retry_api_call(
            client.aio.models.generate_content,
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[types.Part.from_text(text=ontology_prompt)]),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        if response.candidates:
            json_text = clean_json_response(response.candidates[0].content.parts[0].text)
            return json.loads(json_text)
        else:
            return {
                "document_clusters": [],
                "corpus_ontology": [],
                "corpus_relationship_graph": {"nodes": [], "edges": []}
            }
    except Exception as e:
        print(f"Error generating folder-level ontology: {str(e)}")
        return {
            "document_clusters": [],
            "corpus_ontology": [],
            "corpus_relationship_graph": {"nodes": [], "edges": []}
        }

async def update_folder_ontology(client, existing_ontology, new_document):
    """Update folder ontology when adding or removing a document."""
    # Extract document info
    filename = ""
    file_summary = ""
    toc = []
    graph = {}
    
    if isinstance(new_document, dict) and "raw_extracted_content" in new_document:
        raw_content = new_document["raw_extracted_content"]
        filename = raw_content.get("filename", "Unknown")
        
        # Get summary if available
        if "summary" in raw_content and raw_content["summary"]:
            file_summary = raw_content["summary"].get("summary", "")
        
        # Get table of contents if available
        if "summary" in raw_content and raw_content["summary"] and "chapters" in raw_content["summary"]:
            for chapter in raw_content["summary"]["chapters"]:
                toc.append({
                    "order": chapter.get("order", 0),
                    "title": chapter.get("title", "Untitled Chapter")
                })
        
        # Get chapter relationship graph if available
        if "summary" in raw_content and raw_content["summary"] and "chapter_relationship_graph" in raw_content["summary"]:
            graph = raw_content["summary"]["chapter_relationship_graph"]
    
    # Create prompt for folder update following Step 6
    update_prompt = f"""
    You have an existing folder summary:
    {json.dumps(existing_ontology, indent=2)}

    A new document arrives. You're given its:
    - filename: {filename}
    - file_summary: {file_summary}
    - table_of_contents: {json.dumps(toc, indent=2)}
    - chapter_relationship_graph: {json.dumps(graph, indent=2)}

    1. **Update Clusters**  
       Decide if the new file fits into an existing cluster or forms a new one. Output the updated `document_clusters`.
    2. **Update Ontology**  
       If new concepts appear, add or refine entries in `corpus_ontology`. Output the updated list.
    3. **Update Relationship Graph**  
       Insert or remove the node for the file and adjust edges accordingly. Output the updated graph.

    Return ONLY a JSON object with the same three top‑level keys:
    {{
      "document_clusters": [ … ],
      "corpus_ontology": [ … ],
      "corpus_relationship_graph": {{ "nodes":[…],"edges":[…] }}
    }}
    """
    
    try:
        response = await retry_api_call(
            client.aio.models.generate_content,
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[types.Part.from_text(text=update_prompt)]),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        if response.candidates:
            json_text = clean_json_response(response.candidates[0].content.parts[0].text)
            return json.loads(json_text)
        else:
            return existing_ontology
    except Exception as e:
        print(f"Error updating folder ontology: {str(e)}")
        return existing_ontology

###############################
# STEP 7-10 GOAL-DRIVEN GUIDE
###############################

async def construct_retrieval_plan(client, user_goal, folder_metadata):
    """Construct a retrieval plan for a specific user goal."""
    prompt = f"""
    You are given:
    - A user goal: "{user_goal}"
    - Folder metadata for N documents, including:
      • document_clusters (ids & titles)
      • corpus_ontology (concept names & descriptions)
      • file_summaries (filename → summary)
      • table_of_contents per file (chapter orders & titles)

    Based on this, draft a **Markdown To‑Do list** organized into logical phases (e.g. Research & Analysis, Design, Development, etc.) that covers everything needed to achieve the goal.  
    Use checkboxes (`- [ ]`) and nested subtasks.  
    Return ONLY the raw Markdown.

    Example header:
    # {user_goal} To‑Do List

    ## Research & Analysis
    - [ ] …

    ## Design
    - [ ] …

    … etc.
    """
    
    # Convert folder_metadata to a string for the prompt
    metadata_str = json.dumps(folder_metadata, indent=2)
    
    try:
        response = await retry_api_call(
            client.aio.models.generate_content,
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_text(text=metadata_str)
                ]),
            ]
        )
        
        if response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return "Error: No response generated for retrieval plan."
            
    except Exception as e:
        print(f"Error constructing retrieval plan: {str(e)}")
        return f"Error constructing retrieval plan: {str(e)}"

async def retrieve_snippets(client, confirmed_todo, folder_index):
    """Retrieve snippets for each task in the confirmed to-do list."""
    prompt = f"""
    You are given:
    - The confirmed To‑Do list in Markdown, with each task labeled uniquely (e.g. "1.1 Install project skeleton").
    - Folder‑level index containing all processed pages/subsections, each with:
      • page_number or segment_times
      • subsection_id & title
      • text content
      • file and chapter context

    For each task in the To‑Do list, retrieve up to 3 relevant content snippets (subsection text or transcript segment) that best help accomplish the task.  
    Return a JSON object mapping:
    {{
      "task_id": {{
        "task_text": "…",
        "snippets": [
          {{
            "source": "DocA.pdf, page_3_section_2",
            "content": "…snippet text…"
          }},
          …
        ]
      }},
      …
    }}
    """
    
    # Convert folder_index to a string for the prompt
    index_str = json.dumps(folder_index, indent=2)
    
    try:
        response = await retry_api_call(
            client.aio.models.generate_content,
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_text(text=confirmed_todo),
                    types.Part.from_text(text=index_str)
                ]),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        if response.candidates:
            json_text = clean_json_response(response.candidates[0].content.parts[0].text)
            return json.loads(json_text)
        else:
            return {"error": "No response generated for snippet retrieval."}
            
    except Exception as e:
        print(f"Error retrieving snippets: {str(e)}")
        return {"error": f"Error retrieving snippets: {str(e)}"}

async def assemble_guide(client, confirmed_todo, retrieval_results):
    """Assemble a complete guide from the to-do list and retrieved snippets."""
    prompt = f"""
    Using the confirmed Markdown To‑Do list and the retrieved snippets JSON, write a **complete Markdown guide** for the user goal.  
    Structure the guide exactly as the To‑Do list's categories and tasks, and for each task:
    1. Restate the task heading as a sub‑section (e.g. "### 1.1 Install project skeleton").
    2. Summarize the purpose in 1–2 sentences.
    3. Embed the retrieved snippets (in blockquotes) to illustrate the answer.
    4. If needed, add connective text to turn snippets into a coherent narrative.

    Return ONLY the raw Markdown of the full guide.
    """
    
    # Convert retrieval_results to a string for the prompt
    results_str = json.dumps(retrieval_results, indent=2)
    
    try:
        response = await retry_api_call(
            client.aio.models.generate_content,
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_text(text=confirmed_todo),
                    types.Part.from_text(text=results_str)
                ]),
            ]
        )
        
        if response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return "Error: No response generated for guide assembly."
            
    except Exception as e:
        print(f"Error assembling guide: {str(e)}")
        return f"Error assembling guide: {str(e)}"

async def build_folder_metadata(processed_documents):
    """Build folder metadata from processed documents for retrieval plan."""
    metadata = {
        "document_clusters": [],
        "corpus_ontology": [],
        "file_summaries": {},
        "table_of_contents": {}
    }
    
    # Extract metadata from processed documents
    for doc in processed_documents:
        if isinstance(doc, dict) and "raw_extracted_content" in doc:
            raw_content = doc["raw_extracted_content"]
            filename = raw_content.get("filename", "Unknown")
            
            # Extract summary
            if "summary" in raw_content and raw_content["summary"]:
                summary = raw_content["summary"]
                metadata["file_summaries"][filename] = summary.get("summary", "")
                
                # Extract TOC if available
                if "chapters" in summary:
                    toc = []
                    for chapter in summary["chapters"]:
                        toc.append({
                            "order": chapter.get("order", 0),
                            "title": chapter.get("title", "Untitled Chapter")
                        })
                    metadata["table_of_contents"][filename] = toc
    
    return metadata

async def build_folder_index(processed_documents):
    """Build folder index from processed documents for snippet retrieval."""
    index = []
    
    # Extract all subsections from all documents
    for doc in processed_documents:
        if isinstance(doc, dict) and "raw_extracted_content" in doc:
            raw_content = doc["raw_extracted_content"]
            filename = raw_content.get("filename", "Unknown")
            
            # Extract pages and their subsections
            for page in raw_content.get("pages", []):
                page_num = page.get("page_number", 0)
                
                for subsection in page.get("subsections", []):
                    index.append({
                        "file": filename,
                        "page_number": page_num,
                        "subsection_id": subsection.get("subsection_id", ""),
                        "title": subsection.get("title", ""),
                        "content": subsection.get("content", ""),
                        "chapter_title": find_chapter_title(raw_content.get("summary", {}).get("chapters", []), subsection.get("subsection_id", ""))
                    })
    
    return index

def find_chapter_title(chapters, subsection_id):
    """Find the chapter title for a given subsection ID."""
    for chapter in chapters:
        if isinstance(chapter, dict) and "subsections" in chapter:
            for subsection in chapter.get("subsections", []):
                if isinstance(subsection, dict) and subsection.get("subsection_id") == subsection_id:
                    return chapter.get("title", "Unknown Chapter")
                elif hasattr(subsection, "subsection_id") and subsection.subsection_id == subsection_id:
                    return chapter.get("title", "Unknown Chapter")
    return "Unknown Chapter"

async def generate_goal_driven_guide(client, user_goal, processed_documents):
    """Generate a complete goal-driven guide through all steps."""
    try:
        # Step 1: Build folder metadata from processed documents
        folder_metadata = await build_folder_metadata(processed_documents)
        
        # Step 2: Construct retrieval plan
        retrieval_plan = await construct_retrieval_plan(client, user_goal, folder_metadata)
        
        # Step 3: (This would be a UI interaction in the real app)
        # For this implementation, we'll assume the plan is confirmed
        confirmed_todo = retrieval_plan
        
        # Step 4: Build folder index for retrieval
        folder_index = await build_folder_index(processed_documents)
        
        # Step 5: Retrieve snippets
        snippets = await retrieve_snippets(client, confirmed_todo, folder_index)
        
        # Step 6: Assemble guide
        guide = await assemble_guide(client, confirmed_todo, snippets)
        
        return {
            "retrieval_plan": retrieval_plan,
            "snippets": snippets,
            "guide": guide
        }
        
    except Exception as e:
        print(f"Error generating goal-driven guide: {str(e)}")
        return {
            "error": f"Error generating goal-driven guide: {str(e)}",
            "retrieval_plan": None,
            "snippets": None,
            "guide": None
        }

###############################
# GRAPH INTEGRATION FUNCTIONS
###############################

async def integrate_entity_concept_graphs(client, entity_graph, concept_graph):
    """
    Given two JSON graphs—
      • entity_graph: { nodes: [...], edges: [...] } 
      • concept_graph: { nodes: [...], edges: [...] }
    Return a unified graph with:
      - all original nodes (annotated type="entity" or "concept")
      - all original edges
      - new cross‑graph edges linking entities to concepts they participate in or instantiate
    """

    prompt = f"""
You are a Graph Integrator. You get two JSON objects:

1) ENTITY_GRAPH:
{json.dumps(entity_graph, indent=2)}

2) CONCEPT_GRAPH:
{json.dumps(concept_graph, indent=2)}

Build a single JSON with:
{{
  "nodes": [
    {{ "id": "...", "type": "entity" }},         // from ENTITY_GRAPH.nodes
    {{ "id": "...", "type": "concept" }}         // from CONCEPT_GRAPH.nodes
  ],
  "links": [
    // All original edges:
    {{ "source":"…","target":"…","type":"ee" }}, // from ENTITY_GRAPH.edges
    {{ "source":"…","target":"…","type":"cc" }}, // from CONCEPT_GRAPH.edges

    // PLUS cross‑graph edges:
    // For each ENTITY node, link to any CONCEPT node it "enables", "triggers", or "instantiates".
    // For each CONCEPT node, link to any ENTITY it "applies_to" or "manifests_as".
    {{ "source":"EntityID","target":"ConceptID","type":"ec","description":"…" }},
    {{ "source":"ConceptID","target":"EntityID","type":"ce","description":"…" }}
  ]
}}

—Infer the best cross‑graph relationships by looking at node IDs, edge descriptions, and keywords.
—Return ONLY the JSON above, without any commentary.
"""

    # build the combined prompt
    contents = [
        types.Content(parts=[types.Part.from_text(text=prompt)])
    ]

    resp = await client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    raw = resp.candidates[0].content.parts[0].text
    return json.loads(raw)

async def integrate_chapter_relationships_with_subgraphs(client, chapter_graph, per_chapter_graphs):
    """
    Merge a top‑level chapter relationship graph with detailed per‑chapter graphs.

    Args:
      client:     Gemini API client
      chapter_graph: {
        "nodes": [ {"id":"chapter_1","label":"Intro"}, … ],
        "edges": [ {"source":"chapter_1","target":"chapter_2","label":"leads_to"}, … ]
      }
      per_chapter_graphs: {
        "chapter_1": { "nodes":[…],"edges":[…] },
        "chapter_2": { "nodes":[…],"edges":[…] },
        …
      }

    Returns:
      A single JSON with:
      - all original chapter nodes (type="chapter")
      - original chapter→chapter edges (type="cc")
      - all subgraph nodes (type="entity" or "concept")
      - all subgraph edges (type="ee"/"ec"/"cc" as originally)
      - new edges: chapter→subnode ("contains_entity"/"contains_concept")
    """
    prompt = f"""
You are a Graph Integrator.  You get TWO JSON objects:

1) CHAPTER_GRAPH:
{json.dumps(chapter_graph, indent=2)}

2) PER_CHAPTER_GRAPHS: a map from chapter_id to its own subgraph:
{json.dumps(per_chapter_graphs, indent=2)}

Produce a single JSON:
{{
  "nodes": [
    // each chapter from CHAPTER_GRAPH.nodes, plus each subnode from every per‑chapter graph
    {{ "id":"…", "type":"chapter" }},
    {{ "id":"…", "type":"entity" }},
    {{ "id":"…", "type":"concept" }}
  ],
  "edges": [
    // original chapter→chapter edges:
    {{ "source":"…","target":"…","type":"cc","label":"…" }},
    // all subgraph edges exactly as given, but annotate type:
    {{ "source":"…","target":"…","type":"ee" or "ec" or "cc","label":"…" }},
    // new containment edges:
    {{ "source":"<chapter_id>","target":"<entity_or_concept_id>","type":"contains_entity","description":"…"}},
    {{ "source":"<chapter_id>","target":"<entity_or_concept_id>","type":"contains_concept","description":"…"}}
  ]
}}

Rules:
- Preserve all original chapter_graph edges with type="cc".
- Preserve each per‑chapter graph's nodes and edges (annotate edges type by their origin: entity→entity = "ee", concept→concept = "cc", entity→concept = "ec").
- For every node in each per‑chapter graph, add one edge from that chapter's node → subnode:
    • if subnode.type=="entity": type="contains_entity"
    • if subnode.type=="concept": type="contains_concept"
    description can be simply "chapter contains subnode"
- Return ONLY the JSON.
"""
    contents = [ types.Content(parts=[ types.Part.from_text(prompt) ]) ]

    resp = await client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(resp.candidates[0].content.parts[0].text)

async def generate_unified_document_graph(client, document_result):
    """
    Generate a unified graph for the document that combines:
    1. Chapter-to-chapter relationships
    2. Entity relationships within each chapter
    3. Concept relationships within each chapter
    
    Returns a complete graph JSON for visualization.
    """
    try:
        # Extract raw content
        if isinstance(document_result, dict) and "raw_extracted_content" in document_result:
            raw_content = document_result["raw_extracted_content"]
        else:
            raw_content = document_result
            
        # Get chapters data
        summary = raw_content.get("summary", {})
        chapters = summary.get("chapters", [])
        
        if not chapters:
            return {"error": "No chapters found in the document"}
        
        # Create chapter relationship graph
        chapter_graph = {"nodes": [], "edges": []}
        
        # Create nodes for each chapter
        for chapter in chapters:
            chapter_id = chapter.get("chapter_id", "")
            if not chapter_id:
                continue
                
            title = chapter.get("title", "Untitled Chapter")
            chapter_graph["nodes"].append({
                "id": chapter_id,
                "label": title
            })
            
        # Create edges between chapters (sequential ordering)
        for i in range(len(chapter_graph["nodes"]) - 1):
            source_id = chapter_graph["nodes"][i]["id"]
            target_id = chapter_graph["nodes"][i + 1]["id"]
            chapter_graph["edges"].append({
                "source": source_id,
                "target": target_id,
                "label": "follows"
            })
            
        # Extract per-chapter graphs
        per_chapter_graphs = {}
        
        for chapter in chapters:
            chapter_id = chapter.get("chapter_id", "")
            if not chapter_id:
                continue
                
            # Extract entity relationships from this chapter
            entity_relationships = chapter.get("entity_relationships", [])
            
            # Build entity graph
            entity_graph = {"nodes": [], "edges": []}
            entity_nodes = set()
            
            for rel in entity_relationships:
                source = rel.get("source_entity", "") if isinstance(rel, dict) else getattr(rel, "source_entity", "")
                target = rel.get("target_entity", "") if isinstance(rel, dict) else getattr(rel, "target_entity", "")
                desc = rel.get("relationship_description", "") if isinstance(rel, dict) else getattr(rel, "relationship_description", "")
                strength = rel.get("relationship_strength", 5) if isinstance(rel, dict) else getattr(rel, "relationship_strength", 5)
                
                if source and target:
                    if source not in entity_nodes:
                        entity_graph["nodes"].append({"id": source, "type": "entity"})
                        entity_nodes.add(source)
                        
                    if target not in entity_nodes:
                        entity_graph["nodes"].append({"id": target, "type": "entity"})
                        entity_nodes.add(target)
                        
                    entity_graph["edges"].append({
                        "source": source,
                        "target": target,
                        "label": desc,
                        "strength": strength
                    })
            
            # Generate concept graph if available (you may need to add this to your extraction logic)
            # For now, let's create a placeholder with sample data
            concept_graph = {"nodes": [], "edges": []}
            
            # Try to extract concepts
            subsections = chapter.get("subsections", [])
            topics = set()
            
            for subsection in subsections:
                # Try to extract topic keywords
                title = subsection.get("title", "") if isinstance(subsection, dict) else getattr(subsection, "title", "")
                topics.add(title)
            
            # Create concept nodes from topics
            for topic in topics:
                if topic:
                    concept_graph["nodes"].append({
                        "id": topic,
                        "type": "concept"
                    })
            
            # Integrate entity and concept graphs
            try:
                integrated_graph = await integrate_entity_concept_graphs(client, entity_graph, concept_graph)
                per_chapter_graphs[chapter_id] = integrated_graph
            except Exception as e:
                print(f"Error integrating graphs for chapter {chapter_id}: {e}")
                # Use entity graph as fallback
                per_chapter_graphs[chapter_id] = entity_graph
        
        # Integrate chapter relationship graph with per-chapter graphs
        unified_graph = await integrate_chapter_relationships_with_subgraphs(client, chapter_graph, per_chapter_graphs)
        
        return unified_graph
        
    except Exception as e:
        print(f"Error generating unified document graph: {e}")
        return {"error": f"Error generating unified document graph: {e}"}

def display_unified_graph(unified_graph, title="Unified Document Graph"):
    """
    Display unified graph with D3.js visualization.
    Different node and edge types will have distinct styles.
    """
    if not unified_graph or "error" in unified_graph:
        st.error(f"Error generating graph: {unified_graph.get('error', 'Unknown error')}")
        return
        
    # Check if we have nodes and edges
    if "nodes" not in unified_graph or "edges" not in unified_graph:
        st.error("Invalid graph structure: missing nodes or edges")
        return
        
    nodes = unified_graph.get("nodes", [])
    edges = unified_graph.get("edges", [])
    
    if not nodes:
        st.warning("No nodes found in the graph")
        return
        
    # Create network graph with NetworkX
    G = nx.DiGraph()
    
    # Add nodes
    for node in nodes:
        node_id = node.get("id", "")
        if not node_id:
            continue
            
        node_type = node.get("type", "unknown")
        label = node.get("label", node_id)
        
        G.add_node(node_id, label=label, type=node_type)
    
    # Add edges
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        if not source or not target or source not in G or target not in G:
            continue
            
        edge_type = edge.get("type", "unknown")
        label = edge.get("label", "")
        
        G.add_edge(source, target, type=edge_type, label=label)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define node colors based on type
    node_colors = {
        "chapter": "lightblue",
        "entity": "lightgreen",
        "concept": "lightsalmon"
    }
    
    # Get position
    pos = nx.spring_layout(G, k=0.5, iterations=100)
    
    # Draw nodes by type
    for node_type, color in node_colors.items():
        node_list = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
        if node_list:
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=node_list, 
                                  node_color=color, 
                                  node_size=700 if node_type == "chapter" else 300, 
                                  alpha=0.8, 
                                  ax=ax)
    
    # Draw edges with different styles based on type
    edge_types = {
        "cc": {"color": "blue", "width": 2.0, "style": "solid"},
        "ee": {"color": "green", "width": 1.5, "style": "solid"},
        "ec": {"color": "red", "width": 1.5, "style": "dashed"},
        "ce": {"color": "orange", "width": 1.5, "style": "dashed"},
        "contains_entity": {"color": "purple", "width": 1.0, "style": "dotted"},
        "contains_concept": {"color": "brown", "width": 1.0, "style": "dotted"}
    }
    
    for edge_type, style in edge_types.items():
        edge_list = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == edge_type]
        if edge_list:
            nx.draw_networkx_edges(G, pos, 
                                   edgelist=edge_list, 
                                   edge_color=style["color"], 
                                   width=style["width"], 
                                   style=style["style"], 
                                   alpha=0.7, 
                                   arrowsize=15, 
                                   connectionstyle='arc3,rad=0.1',
                                   ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Add title
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    # Display the plot
    st.pyplot(fig)
    
    # Display edge information as a table
    edge_data = []
    for source, target, data in G.edges(data=True):
        edge_data.append({
            "Source": G.nodes[source].get("label", source),
            "Target": G.nodes[target].get("label", target),
            "Type": data.get("type", "unknown"),
            "Description": data.get("label", "")
        })
    
    if edge_data:
        st.markdown("### Relationship Details")
        st.dataframe(pd.DataFrame(edge_data))

###############################
# UI DISPLAY FUNCTIONS
###############################

def display_guide_generation_ui():
    """Display UI for goal-driven guide generation."""
    st.header("🎯 Goal-Driven Guide Generation")
    
    # Show this section only if documents have been processed
    if not st.session_state.get("processed_documents", []):
        st.info("Upload and process documents first to use the guide generation feature.")
        return
    
    # User goal input
    user_goal = st.text_input("Enter your goal or task:", 
                             placeholder="e.g., Implement a recommendation system for e-commerce")
    
    if st.button("Generate Guide") and user_goal:
        with st.spinner("Generating guide..."):
            # Get API client
            api_key = get_gemini_api_key()
            client = genai.Client(api_key=api_key)
            
            # Generate guide
            guide_data = safe_async_run(generate_goal_driven_guide(
                client,
                user_goal,
                st.session_state.processed_documents
            ))
            
            # Store in session state
            st.session_state.guide_data = guide_data
            
            # Show success message
            st.success("Guide generated successfully!")
            
    # Display guide if available
    if "guide_data" in st.session_state:
        guide_data = st.session_state.guide_data
        
        tabs = st.tabs(["Complete Guide", "Retrieval Plan", "Snippets"])
        
        with tabs[0]:
            st.markdown(guide_data.get("guide", "No guide available."))
            
        with tabs[1]:
            st.markdown(guide_data.get("retrieval_plan", "No retrieval plan available."))
            
        with tabs[2]:
            snippets = guide_data.get("snippets", {})
            if snippets and not isinstance(snippets, str):
                for task_id, task_data in snippets.items():
                    with st.expander(f"Task: {task_data.get('task_text', task_id)}"):
                        for i, snippet in enumerate(task_data.get("snippets", [])):
                            st.markdown(f"**Source:** {snippet.get('source', 'Unknown')}")
                            st.markdown(f"> {snippet.get('content', 'No content')}")
                            st.divider()
            else:
                st.info("No snippets available.")

def display_graph_visualization_ui():
    """Display UI for graph visualization of documents and chapters."""
    st.header("🔄 Document Graph Visualization")
    
    # Show this section only if documents have been processed
    if not st.session_state.get("processed_documents", []):
        st.info("Upload and process documents first to view document graphs.")
        return
    
    # Document selection
    doc_options = []
    for doc in st.session_state.processed_documents:
        if isinstance(doc, dict) and "raw_extracted_content" in doc:
            filename = doc["raw_extracted_content"].get("filename", "Unknown")
            doc_options.append(filename)
    
    if not doc_options:
        st.warning("No processed documents found with valid structure.")
        return
        
    selected_doc = st.selectbox("Select document to visualize:", options=doc_options)
    
    if st.button("Generate Unified Graph"):
        with st.spinner("Generating unified graph visualization..."):
            # Get API client
            api_key = get_gemini_api_key()
            client = genai.Client(api_key=api_key)
            
            # Find the selected document
            doc_to_visualize = None
            for doc in st.session_state.processed_documents:
                if isinstance(doc, dict) and "raw_extracted_content" in doc:
                    if doc["raw_extracted_content"].get("filename") == selected_doc:
                        doc_to_visualize = doc
                        break
            
            if not doc_to_visualize:
                st.error(f"Could not find document: {selected_doc}")
                return
                
            # Generate unified graph
            unified_graph = safe_async_run(generate_unified_document_graph(client, doc_to_visualize))
            
            # Store in session state
            st.session_state.unified_graph = unified_graph
            st.session_state.current_graph_doc = selected_doc
            
            st.success("Graph generated successfully!")
    
    # Display unified graph if available
    if "unified_graph" in st.session_state and "current_graph_doc" in st.session_state:
        if st.session_state.current_graph_doc == selected_doc:
            st.subheader(f"Unified Graph for: {selected_doc}")
            display_unified_graph(st.session_state.unified_graph)
        else:
            st.info("Click 'Generate Unified Graph' to view the graph for this document.")

def display_entity_relationship_network(relationships, title="Entity Relationship Network"):
    """Display entity relationships as a network graph."""
    if not relationships:
        st.info("No entity relationships to display.")
        return
    
    # Create a network graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for rel in relationships:
        if isinstance(rel, dict):
            source = rel.get('source_entity', 'Unknown')
            target = rel.get('target_entity', 'Unknown')
            strength = rel.get('relationship_strength', 5)
            description = rel.get('relationship_description', '')
        else:
            source = getattr(rel, 'source_entity', 'Unknown')
            target = getattr(rel, 'target_entity', 'Unknown')
            strength = getattr(rel, 'relationship_strength', 5)
            description = getattr(rel, 'relationship_description', '')
        
        # Add nodes if they don't exist
        if source not in G:
            G.add_node(source)
        if target not in G:
            G.add_node(target)
        
        # Add edge with attributes
        G.add_edge(source, target, weight=strength, description=description)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up the layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)
    
    # Draw edges with width based on relationship strength
    edge_width = [G[u][v]['weight']/2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.7, edge_color='gray', 
                          arrowsize=20, connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add title
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Display edge information as a table
    edge_data = []
    for source, target, data in G.edges(data=True):
        edge_data.append({
            "Source": source,
            "Target": target,
            "Strength": data['weight'],
            "Description": data['description']
        })
    
    if edge_data:
        st.markdown("### Relationship Details")
        df = pd.DataFrame(edge_data)
        st.dataframe(df)

def display_table(table):
    """Display table that could be either TableData model or dict."""
    if hasattr(table, 'model_dump'):
        table_data = table.model_dump()
    else:
        table_data = table
    
    # Display table title if available
    if table_data.get("title"):
        st.subheader(table_data["title"])
    
    # Display the table content
    if table_data.get("table_content"):
        try:
            df = pd.read_csv(
                io.StringIO(table_data["table_content"]),
                sep="|",
                engine="python",
                skipinitialspace=True
            )
            df = df.dropna(axis=1, how="all")
            st.dataframe(df)
        except Exception:
            st.markdown(f"```markdown\n{table_data['table_content']}\n```")

def display_visual_element(visual):
    """Display a visual element."""
    if isinstance(visual, dict):
        visual_type = visual.get('type', 'Unknown')
        description = visual.get('description', '')
        data_summary = visual.get('data_summary', '')
    else:
        visual_type = getattr(visual, 'type', 'Unknown')
        description = getattr(visual, 'description', '')
        data_summary = getattr(visual, 'data_summary', '')
        
    st.markdown(f"**{visual_type.capitalize()}**")
    st.markdown(f"*Description:* {description}")
    if data_summary:
        st.markdown(f"*Data Summary:* {data_summary}")

def display_chapter(chapter):
    """Display chapter details including subsections and entity relationships."""
    if isinstance(chapter, dict):
        title = chapter.get('title', 'Untitled Chapter')
        summary = chapter.get('summary', 'No summary available.')
        subsections = chapter.get('subsections', [])
        entity_relationships = chapter.get('entity_relationships', [])
        key_points = chapter.get('key_points', [])
        questions_answers = chapter.get('questions_answers', [])
    else:
        title = getattr(chapter, 'title', 'Untitled Chapter')
        summary = getattr(chapter, 'summary', 'No summary available.')
        subsections = getattr(chapter, 'subsections', [])
        entity_relationships = getattr(chapter, 'entity_relationships', [])
        key_points = getattr(chapter, 'key_points', [])
        questions_answers = getattr(chapter, 'questions_answers', [])
    
    st.markdown(f"## {title}")
    st.markdown(summary)
    
    # Display key points if available
    if key_points:
        st.markdown("### Key Points")
        for point in key_points:
            st.markdown(f"- {point}")
    
    # Display Q&A if available
    if questions_answers:
        with st.expander("Questions & Answers", expanded=False):
            for qa in questions_answers:
                if isinstance(qa, dict):
                    question = qa.get('question', '')
                    answer = qa.get('answer', '')
                    if question and answer:
                        st.markdown(f"**Q: {question}**")
                        st.markdown(f"A: {answer}")
                        st.markdown("---")
    
    # Display entity relationships if any
    if entity_relationships:
        with st.expander("Entity Relationships", expanded=False):
            display_entity_relationship_network(entity_relationships, f"Entity Relationships in {title}")
    
    # Display subsections
    if subsections:
        st.markdown("### Subsections")
        for subsection in subsections:
            if isinstance(subsection, dict):
                sub_title = subsection.get('title', 'Untitled Subsection')
                sub_content = subsection.get('content', 'No content available.')
                page_num = subsection.get('page_number', '?')
                ref_visuals = subsection.get('referenced_visuals', [])
                ref_tables = subsection.get('referenced_tables', [])
            else:
                sub_title = getattr(subsection, 'title', 'Untitled Subsection')
                sub_content = getattr(subsection, 'content', 'No content available.')
                page_num = getattr(subsection, 'page_number', '?')
                ref_visuals = getattr(subsection, 'referenced_visuals', [])
                ref_tables = getattr(subsection, 'referenced_tables', [])
            
            with st.expander(f"{sub_title} (Page {page_num})"):
                st.markdown(sub_content)
                
                # Display referenced visuals/tables if any
                if ref_visuals:
                    st.markdown("#### Referenced Visuals")
                    st.write(", ".join(ref_visuals))
                
                if ref_tables:
                    st.markdown("#### Referenced Tables")
                    st.write(", ".join(ref_tables))

def display_page_details(document):
    """Display page details, handling both model objects and dictionaries."""
    st.header("📑 Page Level Details")
    
    if hasattr(document, 'pages') and document.pages:
        # Create tab labels that handle both dict and model objects
        tab_labels = []
        for page in document.pages:
            if isinstance(page, dict):
                page_num = page.get('page_number', '?')
            else:
                page_num = getattr(page, 'page_number', '?')
            tab_labels.append(f"Page {page_num}")
        
        tabs = st.tabs(tab_labels)
        
        for i, page in enumerate(document.pages):
            # Convert to dict if it's a model
            if hasattr(page, 'model_dump'):
                page_dict = page.model_dump()
            else:
                page_dict = page
                
            with tabs[i]:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(page_dict.get('title', f'Page {page_dict.get("page_number", "?")}'))
                    st.markdown("#### Topics")
                    st.write(", ".join(page_dict.get('topics', [])))
                    
                    if page_dict.get('summary'):
                        st.markdown("#### Summary")
                        st.markdown(page_dict.get('summary', ''))
                
                with col2:
                    st.markdown("#### Entities")
                    st.write(", ".join(page_dict.get('entities', [])))
                    
                    st.markdown("#### Content Flags")
                    st.write(f"Tables: {'✅' if page_dict.get('has_tables', False) else '❌'}")
                    st.write(f"Visuals: {'✅' if page_dict.get('has_visuals', False) else '❌'}")
                    st.write(f"Numbers: {'✅' if page_dict.get('has_numbers', False) else '❌'}")
                
                # Display tables
                if page_dict.get('tables'):
                    st.markdown("#### Tables")
                    for table in page_dict.get('tables', []):
                        display_table(table)
                
                # Display visuals
                if page_dict.get('visuals'):
                    st.markdown("#### Visual Elements")
                    for visual in page_dict.get('visuals', []):
                        display_visual_element(visual)
                
                # Display numerical data
                if page_dict.get('numbers'):
                    st.markdown("#### Numerical Data")
                    for num in page_dict.get('numbers', []):
                        value = num.get('value', '') if isinstance(num, dict) else getattr(num, 'value', '')
                        description = num.get('description', '') if isinstance(num, dict) else getattr(num, 'description', '')
                        context = num.get('context', '') if isinstance(num, dict) else getattr(num, 'context', '')
                        
                        st.markdown(f"**{value}**: {description}")
                        if context:
                            st.markdown(f"*Context: {context}*")
                
                # Display subsections
                if page_dict.get('subsections'):
                    st.markdown("#### Subsections")
                    for subsection in page_dict.get('subsections', []):
                        if isinstance(subsection, dict):
                            sub_title = subsection.get('title', 'Untitled')
                            sub_content = subsection.get('content', 'No content')
                            sub_id = subsection.get('subsection_id', 'unknown')
                            is_cutoff = subsection.get('is_cutoff', False)
                        else:
                            sub_title = getattr(subsection, 'title', 'Untitled')
                            sub_content = getattr(subsection, 'content', 'No content')
                            sub_id = getattr(subsection, 'subsection_id', 'unknown')
                            is_cutoff = getattr(subsection, 'is_cutoff', False)
                        
                        with st.expander(f"{sub_title} {'(Cut off)' if is_cutoff else ''}"):
                            st.markdown(sub_content)
                
                # Display page text
                st.markdown("#### Page Content")
                st.text_area("Page Content", page_dict.get('text', ''), height=300, 
                             key=f"page_text_{page_dict.get('page_number', i)}", 
                             label_visibility="collapsed")  # This hides the label visually but keeps it for accessibility

def display_folder_level_ui():
    """Display UI for folder-level processing."""
    st.header("📁 Folder-Level Analysis")
    
    # Show this section only if documents have been processed
    if not st.session_state.get("processed_documents", []):
        st.info("Upload and process multiple documents first to use folder-level analysis.")
        return
    
    # Check if we already have a folder ontology
    has_ontology = "folder_ontology" in st.session_state and st.session_state.folder_ontology
    
    # Buttons for generating or updating
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Folder Ontology" if not has_ontology else "Regenerate Folder Ontology"):
            with st.spinner("Analyzing all documents..."):
                # Get API client
                api_key = get_gemini_api_key()
                client = genai.Client(api_key=api_key)
                
                # Generate folder ontology
                ontology = safe_async_run(generate_folder_level_ontology(
                    client,
                    st.session_state.processed_documents
                ))
                
                # Store in session state
                st.session_state.folder_ontology = ontology
                
                # Show success message
                st.success("Folder ontology generated successfully!")
    
    with col2:
        if has_ontology and len(st.session_state.processed_documents) > 0:
            # Select document to add/update
            doc_options = [doc.get("raw_extracted_content", {}).get("filename", "Unknown") 
                          for doc in st.session_state.processed_documents 
                          if isinstance(doc, dict)]
            
            selected_doc = st.selectbox(
                "Select document to update in ontology:",
                options=doc_options,
                key="update_doc_selector"
            )
            
            if st.button("Update with Selected Document"):
                with st.spinner("Updating folder ontology..."):
                    # Get API client
                    api_key = get_gemini_api_key()
                    client = genai.Client(api_key=api_key)
                    
                    # Find the selected document
                    update_doc = None
                    for doc in st.session_state.processed_documents:
                        if isinstance(doc, dict) and doc.get("raw_extracted_content", {}).get("filename", "") == selected_doc:
                            update_doc = doc
                            break
                    
                    if update_doc:
                        # Update folder ontology
                        updated_ontology = safe_async_run(update_folder_ontology(
                            client,
                            st.session_state.folder_ontology,
                            update_doc
                        ))
                        
                        # Store updated ontology
                        st.session_state.folder_ontology = updated_ontology
                        
                        # Show success message
                        st.success(f"Folder ontology updated with '{selected_doc}'!")
                    else:
                        st.error("Selected document not found.")
    
    # Display folder ontology if available
    if has_ontology:
        ontology = st.session_state.folder_ontology
        
        # Display document clusters
        st.subheader("Document Clusters")
        for cluster in ontology.get("document_clusters", []):
            with st.expander(f"Cluster: {cluster.get('title', 'Unnamed Cluster')}"):
                st.write(f"**ID:** {cluster.get('cluster_id', 'No ID')}")
                st.write("**Files:**")
                for file in cluster.get("file_names", []):
                    st.markdown(f"- {file}")
        
        # Display corpus ontology
        st.subheader("Corpus Ontology")
        for concept in ontology.get("corpus_ontology", []):
            with st.expander(f"Concept: {concept.get('name', 'Unnamed Concept')}"):
                st.write(concept.get("description", "No description"))
                st.write("**Related Files:**")
                for file in concept.get("related_files", []):
                    st.markdown(f"- {file}")
                st.write("**Related Chapters:**")
                for chapter in concept.get("related_chapters", []):
                    st.markdown(f"- {chapter}")
        
        # Display corpus relationship graph
        st.subheader("Corpus Relationship Graph")
        graph = ontology.get("corpus_relationship_graph", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        if nodes and edges:
            G = nx.DiGraph()
            
            # Add nodes
            for node in nodes:
                G.add_node(node.get("id", ""),
                          label=node.get("label", ""),
                          type=node.get("type", ""))
            
            # Add edges
            for edge in edges:
                G.add_edge(edge.get("source", ""),
                          edge.get("target", ""),
                          label=edge.get("label", ""))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Define node colors based on type
            node_colors = {
                "cluster": "lightblue",
                "concept": "lightgreen",
                "file": "lightcoral"
            }
            
            # Get positions
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes by type
            for node_type, color in node_colors.items():
                node_list = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
                if node_list:
                    nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=color, 
                                          node_size=700, alpha=0.8, ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, edge_color='gray', 
                                  arrowsize=20, connectionstyle='arc3,rad=0.1', ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
            
            # Add title
            plt.title("Corpus Relationship Graph", fontsize=16)
            plt.axis('off')
            
            # Display the plot
            st.pyplot(fig)
            
            # Display edge information as a table
            edge_data = []
            for edge in edges:
                edge_data.append({
                    "Source": edge.get("source", ""),
                    "Target": edge.get("target", ""),
                    "Relationship": edge.get("label", "")
                })
            
            if edge_data:
                st.markdown("### Relationship Details")
                st.dataframe(pd.DataFrame(edge_data))
        else:
            st.info("No relationship graph data available.")
    else:
        st.info("Generate a folder ontology to see relationships across documents.")

def render_unified_document_report(document_data):
    """All document data in one scrollable view with detailed page tabs"""
    # Convert to dict if it's a Pydantic model
    if hasattr(document_data, 'model_dump'):
        doc = document_data.model_dump()
    else:
        doc = document_data
    
    # Safely get raw_content
    raw_content = doc.get('raw_extracted_content', doc)
    
    # --- Header Section ---
    st.title(raw_content.get('filename', 'Document Report'))
    
    # Count tables/visuals (works with both dict and model)
    def count_attr(items, attr):
        return sum(
            1 for item in items 
            if (isinstance(item, dict) and item.get(attr)) 
            or (hasattr(item, attr) and getattr(item, attr))
        )
    
    st.caption(
        f"🔢 {len(raw_content.get('pages', []))} pages | " +
        f"📊 {count_attr(raw_content.get('pages', []), 'has_tables')} tables | " +
        f"📈 {count_attr(raw_content.get('pages', []), 'has_visuals')} visuals"
    )

    # Create tabs for document view modes
    tab1, tab2, tab3 = st.tabs(["Executive Summary", "Chapters", "Detailed Page View"])
    
    with tab1:
        # --- Summary Card ---
        with st.container(border=True):
            summary = raw_content.get('summary', {})
            if isinstance(summary, dict):
                summary_data = summary
            else:
                summary_data = summary.model_dump() if hasattr(summary, 'model_dump') else {}
            
            st.subheader("Executive Summary")
            
            # Display summary text
            if summary_data.get('summary'):
                st.markdown(summary_data.get('summary'))
            else:
                st.info("No summary available for this document.")
            
            # Display themes if available
            if summary_data.get('themes'):
                st.markdown("#### Key Themes")
                themes_html = " ".join([f"<span style='background-color:#e6f3ff; padding:5px; margin:2px; border-radius:5px'>#{tag}</span>" 
                                     for tag in summary_data.get('themes', [])])
                st.markdown(themes_html, unsafe_allow_html=True)
            
            # Metrics in columns
            if summary_data.get('key_metrics'):
                st.markdown("#### Key Metrics")
                cols = st.columns(3)
                metrics = summary_data['key_metrics']
                if not isinstance(metrics, list):
                    metrics = list(metrics) if hasattr(metrics, '__iter__') else []
                
                for i, metric in enumerate(metrics[:6]):
                    metric_data = metric if isinstance(metric, dict) else metric.model_dump()
                    cols[i%3].metric(
                        label=metric_data.get('name', 'Metric'),
                        value=metric_data.get('value', 'N/A'),
                        delta=metric_data.get('trend'),
                        help=metric_data.get('context')
                    )
            
            # Display entity relationships if any
            entity_relationships = summary_data.get('entity_relationships', [])
            if entity_relationships:
                st.markdown("#### Entity Relationships")
                with st.expander("View Entity Relationship Network", expanded=False):
                    display_entity_relationship_network(entity_relationships, "Document Entity Relationships")
                    
        # --- Simple Page Navigation (keep this for the summary view) ---
        st.divider()
        pages = raw_content.get('pages', [])
        page_options = []
        for p in pages:
            if isinstance(p, dict):
                title = p.get('title', '')
                num = p.get('page_number', len(page_options)+1)
            else:
                title = getattr(p, 'title', '')
                num = getattr(p, 'page_number', len(page_options)+1)
            page_options.append(f"Page {num} - {title}")
        
        selected_page = st.selectbox(
            "Navigate to page:",
            options=page_options,
            key="page_nav_summary"
        )
        page_idx = int(selected_page.split()[1]) - 1
        page_data = pages[page_idx]
        
        # --- Selected Page Content ---
        with st.container(border=True):
            # Text Content
            st.subheader(page_data.get('title', selected_page))
            st.text_area("Full Text", 
                        page_data.get('text', ''), 
                        height=200,
                        label_visibility="collapsed")

            # Tables
            tables = page_data.get('tables', [])
            if tables:
                st.subheader(f"Tables ({len(tables)})")
                for table in tables:
                    if hasattr(table, 'model_dump'):
                        table = table.model_dump()
                    display_table(table)
    
    with tab2:
        # Display chapters
        chapters = summary_data.get('chapters', [])
        if chapters:
            chapter_titles = [chapter.get('title', f"Chapter {i+1}") for i, chapter in enumerate(chapters)]
            chapter_tabs = st.tabs(chapter_titles)
            
            for i, (tab, chapter) in enumerate(zip(chapter_tabs, chapters)):
                with tab:
                    display_chapter(chapter)
        else:
            st.info("No chapter information available for this document.")
    
    with tab3:
        # Use existing detailed page view but adapt it for the new structure
        pages = raw_content.get('pages', [])
        
        # Convert all pages to PageContent objects or compatible dicts
        processed_pages = []
        for page in pages:
            if hasattr(page, 'model_dump'):
                processed_pages.append(page)  # Already a model
            else:
                # It's a dict, either use as is or convert to PageContent
                processed_pages.append(page)
        
        # Create a wrapper object compatible with display_page_details
        document_wrapper = type('ProcessedDocument', (), {'pages': processed_pages})
        
        # Call your existing function
        display_page_details(document_wrapper)

def display_sidebar_chat():
    """Manages the sidebar content: file upload, chat, and triggers processing."""
    st.sidebar.title("📝 Input & Chat")
    st.sidebar.markdown("Upload documents and chat about their content.")

    # Define supported file types
    supported_types = ["pdf", "xlsx", "xls", "docx", "pptx", "csv", "txt"]

    # Initialize session state keys if they don't exist
    if "processed_file_names" not in st.session_state:
        st.session_state.processed_file_names = set()  # Store strings only
    if "processing_active" not in st.session_state:
        st.session_state.processing_active = False
    if "last_uploaded_files" not in st.session_state:
        st.session_state.last_uploaded_files = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = []  # Can store both dicts and models
    if "selected_docs" not in st.session_state:
        st.session_state.selected_docs = []
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    if "selected_doc_to_view" not in st.session_state:
        st.session_state.selected_doc_to_view = None
    if "project_ontology" not in st.session_state:
        st.session_state.project_ontology = None
    if "guide_data" not in st.session_state:
        st.session_state.guide_data = None
    if "unified_graph" not in st.session_state:
        st.session_state.unified_graph = None
    if "current_graph_doc" not in st.session_state:
        st.session_state.current_graph_doc = None

    # --- File Upload Section ---
    st.sidebar.subheader("📁 Upload Documents")
    
    uploaded_files = st.sidebar.file_uploader(
        "Select files:",
        type=supported_types,
        accept_multiple_files=True,
        key="sidebar_file_uploader_main"
    )

    # Detect *new* files to process automatically
    new_files_to_process = []
    current_files = {f.name for f in uploaded_files} if uploaded_files else set()
    previous_files = {f.name for f in st.session_state.last_uploaded_files} if st.session_state.last_uploaded_files else set()
    
    # If there are new files and we're not already processing
    if uploaded_files and (current_files != previous_files) and not st.session_state.processing_active:
        st.session_state.last_uploaded_files = uploaded_files
        
        for file in uploaded_files:
            if file.name not in st.session_state.processed_file_names:
                new_files_to_process.append(file)
    
    # If we have new files to process, prepare and trigger processing directly
    if new_files_to_process and not st.session_state.processing_active:
        st.sidebar.info(f"Processing {len(new_files_to_process)} new files...")
        
        # Create the status elements directly in the sidebar
        status_container = st.sidebar.status("Starting document processing...", expanded=True)
        progress_bar = st.sidebar.progress(0)
        time_info = st.sidebar.empty()
        
        # Store in session state for access
        st.session_state.status_container = status_container
        st.session_state.progress_bar = progress_bar
        st.session_state.time_info = time_info
        
        # Prepare file data
        files_data = []
        for file in new_files_to_process:
            try:
                file_content = file.getvalue()
                file_ext = file.name.split('.')[-1].lower()
                files_data.append({
                    "name": file.name,
                    "content": file_content,
                    "type": file_ext
                })
            except Exception as e:
                st.sidebar.error(f"Error reading file {file.name}: {e}")

        if files_data:
            # Process documents directly
            try:
                st.session_state.processing_active = True
                
                # Call the main processing function
                processed_docs = run_async(
                    process_all_documents_async,
                    files_data
                )

                # Update session state with results
                new_processed_docs = [doc for doc in processed_docs if doc is not None]
                st.session_state.processed_documents.extend(new_processed_docs)

                # Update the set of processed file names
                for file_info in files_data:
                    st.session_state.processed_file_names.add(file_info["name"])

                # Set flag to show results
                st.session_state.show_results = True
                
                status_container.update(label="✅ Document processing finished!", state="complete")
                
            except Exception as e:
                st.sidebar.error(f"Error during document processing: {e}")
                status_container.update(label=f"❌ Processing Error: {e}", state="error")
            finally:
                # Ensure processing flag is reset
                st.session_state.processing_active = False
    
    # Display already processed files
    if st.session_state.processed_file_names:
        st.sidebar.markdown("---")
        st.sidebar.write("Processed files:")
        for filename in sorted(st.session_state.processed_file_names):
            st.sidebar.caption(f"✓ {filename}")
        
        # Add a button to view results
        if st.sidebar.button("View Document Reports"):
            st.session_state.show_results = True
            st.rerun()

    # --- Chat Section --- 
    st.sidebar.divider()
    st.sidebar.subheader("💬 Document Chat")

    # Document selection for chat context
    if st.session_state.processed_documents:
        doc_options = []
        for doc_result in st.session_state.processed_documents:
            filename = None
            
            if isinstance(doc_result, dict):
                raw_content = doc_result.get("raw_extracted_content", {})
                filename = raw_content.get("filename")
            elif hasattr(doc_result, 'filename'):
                filename = doc_result.filename
            elif hasattr(doc_result, 'raw_extracted_content'):
                filename = doc_result.raw_extracted_content.filename
                
            if filename:
                doc_options.append(filename)

        if doc_options:
            st.session_state.selected_docs = st.sidebar.multiselect(
                "Select documents for chat context:",
                options=sorted(list(set(doc_options))),
                default=st.session_state.selected_docs,
                key="doc_context_selector_sidebar"
            )
        else:
            st.sidebar.caption("No processed documents available to reference.")
    else:
        st.sidebar.caption("No documents processed yet. You can still chat, but I won't be able to reference any document content.")

    # Create a container with border in the sidebar for chat history
    with st.sidebar.container(border=True, height=400):
        st.subheader("Chat History")
        # Display chat history
        if not st.session_state.get("messages", []):
            st.info("No messages yet. Start a conversation below!")
        else:
            for message in st.session_state.messages:
                role = message.get("role", "")
                content = message.get("content", "")
                timestamp = message.get("timestamp", "")
                
                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(content)
                        if timestamp:
                            st.caption(f"Sent: {timestamp}")
                elif role == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(content)
                        if timestamp:
                            st.caption(f"Received: {timestamp}")
                elif role == "system":
                    with st.chat_message("system"):
                        st.markdown(f"*{content}*")
                        if timestamp:
                            st.caption(f"System: {timestamp}")

        # Chat input
        if prompt := st.sidebar.chat_input("Ask a question...", key="sidebar_chat_input_main"):
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_time})
            run_async(process_chat_message, prompt)
            st.rerun()

async def process_chat_message(message):
    """Process a chat message with clear user feedback."""
    # Make sure the message list exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Add a thinking indicator that will be updated with the real response
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({"role": "assistant", "content": "Thinking...", "timestamp": current_time})
    
    try:
        # Get API client
        api_key = get_gemini_api_key()
        client = genai.Client(api_key=api_key)
        
        # Build context from documents (if any)
        context = ""
        selected_docs = st.session_state.get("selected_docs", [])
        processed_docs = st.session_state.get("processed_documents", [])
        
        if selected_docs and processed_docs:
            # Extract content from selected documents for context
            context = "Using document context from: " + ", ".join(selected_docs) + "\n\n"
            
            for doc_name in selected_docs:
                # Find the document in processed_documents
                selected_doc = None
                for doc in processed_docs:
                    if isinstance(doc, dict) and "raw_extracted_content" in doc:
                        filename = doc["raw_extracted_content"].get("filename", "")
                        if filename == doc_name:
                            selected_doc = doc
                            break
                
                if selected_doc and "raw_extracted_content" in selected_doc:
                    # Extract summary and key pages
                    tech_data = selected_doc["raw_extracted_content"]
                    
                    # Add summary if available
                    if "summary" in tech_data and tech_data["summary"]:
                        summary = tech_data["summary"]
                        context += f"Document: {doc_name}\n"
                        context += f"Title: {summary.get('title', '')}\n"
                        context += f"Summary: {summary.get('summary', '')}\n\n"
                    
                    # Add content from up to 3 key pages
                    if "pages" in tech_data and tech_data["pages"]:
                        pages = tech_data["pages"]
                        # Sort pages by importance (tables, numbers, length)
                        sorted_pages = sorted(pages, 
                                            key=lambda p: (p.get("has_tables", False), 
                                                          p.get("has_numbers", False), 
                                                          len(p.get("text", ""))), 
                                            reverse=True)[:3]
                        
                        for page in sorted_pages:
                            context += f"Page {page.get('page_number', '')}: {page.get('text', '')[:500]}...\n\n"
        
        # Simple prompt for demonstration
        prompt = f"""
        You are a helpful assistant that analyzes documents. 
        
        Document context:
        {context}
        
        Please respond to: {message}
        """
        
        # Generate content - using the correct async pattern
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]
        )
        
        # Update the thinking message with the actual response
        if st.session_state.messages and st.session_state.messages[-1].get("role") == "assistant":
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if response and response.candidates:
                content = response.candidates[0].content.parts[0].text
                st.session_state.messages[-1]["content"] = content
                st.session_state.messages[-1]["timestamp"] = timestamp
            else:
                st.session_state.messages[-1]["content"] = "Sorry, I couldn't generate a response."
                st.session_state.messages[-1]["timestamp"] = timestamp
        
    except Exception as e:
        # Update thinking message with the error
        if st.session_state.messages and st.session_state.messages[-1].get("role") == "assistant":
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages[-1]["content"] = f"Sorry, an error occurred: {str(e)}"
            st.session_state.messages[-1]["timestamp"] = timestamp
    
    # Trigger UI update
    st.rerun()

def document_storage_component():
    """Component for storing documents in Qdrant."""
    if not hasattr(st.session_state, 'documents_with_summary') or not st.session_state.documents_with_summary:
        st.info("No documents with metadata to store. Please process documents for metadata first.")
        return
    
    # Initialize Qdrant
    safe_async_run(initialize_qdrant_vector_store())
    
    if st.button("Store Documents in Vector Database"):
        with st.spinner("Storing documents..."):
            success = safe_async_run(store_documents_in_qdrant(st.session_state.documents_with_summary))
            
            if success:
                st.success("Documents successfully stored in vector database!")
                
                # Retrieve unique metadata values for filtering
                metadata = safe_async_run(retrieve_document_metadata())
                
                # Store in session state
                st.session_state.unique_file_name = list(metadata.get('file_names', []))
                st.session_state.unique_title = list(metadata.get('titles', []))
                st.session_state.unique_concept_theme_hashtags = list(metadata.get('hashtags', []))
                
                # Retrieve all nodes for display
                st.session_state.vector_store_all_nodes = safe_async_run(get_nodes_with_filter())
            else:
                st.error("Failed to store documents in vector database.")

def document_search_component():
    """Component for searching documents using filters."""
    if not hasattr(st.session_state, 'qdrant_vector_store_initialized') or not st.session_state.qdrant_vector_store_initialized:
        st.info("Vector store not initialized. Please store documents first.")
        return
    
    # Initialize session state for search filters
    if "selected_file_names" not in st.session_state:
        st.session_state.selected_file_names = []
    if "selected_title" not in st.session_state:
        st.session_state.selected_title = []
    if "selected_concept_theme_hashtags" not in st.session_state:
        st.session_state.selected_concept_theme_hashtags = []
    
    st.subheader("Document Search")
    
    # Filter selection
    col1, col2 = st.columns(2)
    
    with col1:
        if hasattr(st.session_state, 'unique_file_name') and st.session_state.unique_file_name:
            st.session_state.selected_file_names = st.multiselect(
                "Select Document Sources:",
                options=st.session_state.unique_file_name,
                default=st.session_state.selected_file_names
            )
        else:
            st.info("No document sources available")
    
    with col2:
        if hasattr(st.session_state, 'unique_concept_theme_hashtags') and st.session_state.unique_concept_theme_hashtags:
            st.session_state.selected_concept_theme_hashtags = st.multiselect(
                "Select Topics:",
                options=st.session_state.unique_concept_theme_hashtags,
                default=st.session_state.selected_concept_theme_hashtags
            )
        else:
            st.info("No topics available")
    
    # Search box
    query = st.text_input("Search documents:", key="document_search_query")
    
    # Search button
    search_col1, search_col2 = st.columns([1, 2])
    with search_col1:
        filter_button = st.button("🔍 Filter Documents")
    with search_col2:
        search_button = st.button("🔎 Search Documents", disabled=not query)
    
    # Execute search or filter
    if filter_button:
        with st.spinner("Retrieving documents..."):
            nodes = safe_async_run(get_nodes_with_filter(
                file_names=st.session_state.selected_file_names,
                hashtags=st.session_state.selected_concept_theme_hashtags
            ))
            
            if nodes:
                st.success(f"Found {len(nodes)} documents")
                
                # Organize nodes by source
                nodes_by_source = {}
                for node in nodes:
                    file_name = node.metadata.get('file_name', 'Unknown')
                    if file_name not in nodes_by_source:
                        nodes_by_source[file_name] = []
                    nodes_by_source[file_name].append(node)
                
                # Display results
                for file_name, source_nodes in nodes_by_source.items():
                    with st.expander(f"Source: {file_name}", expanded=True):
                        for node in source_nodes:
                            st.markdown(f"**{node.metadata.get('chunk_metadata', {}).get('chunk_title', 'Unknown')}**")
                            st.markdown(node.text[:300] + "..." if len(node.text) > 300 else node.text)
                            st.markdown("---")
            else:
                st.info("No documents found matching the selected filters")
    
    elif search_button and query:
        with st.spinner(f"Searching for: {query}"):
            results = safe_async_run(search_documents(
                query_str=query,
                file_names=st.session_state.selected_file_names if st.session_state.selected_file_names else None,
                hashtags=st.session_state.selected_concept_theme_hashtags if st.session_state.selected_concept_theme_hashtags else None
            ))
            
            if results and results.get("success", False):
                st.success(f"Found {results.get('total_results', 0)} results in {results.get('time', 0):.2f}s using {results.get('mode', 'unknown')} search")
                
                # Display results
                nodes_by_source = results.get("nodes_by_source", {})
                if nodes_by_source:
                    for file_name, nodes in nodes_by_source.items():
                        with st.expander(f"Source: {file_name} ({len(nodes)} results)", expanded=True):
                            for node in nodes:
                                title = node.metadata.get('chunk_metadata', {}).get('chunk_title', 'Unknown Section')
                                st.markdown(f"**{title}**")
                                st.markdown(node.text[:300] + "..." if len(node.text) > 300 else node.text)
                                st.markdown("---")
                else:
                    st.info("No results found for your query.")
            else:
                st.error(f"Search failed: {results.get('error', 'Unknown error')}")

def retrieve_conversation_context(context_id):
    """Retrieve full conversation context by ID."""
    for item in st.session_state.conversation_store:
        if item["id"] == context_id:
            return {
                "user": item["user"],
                "assistant": item["assistant"],
                "summary": item["summary"],
                "timestamp": item["timestamp"]
            }
    return None

def prepare_document_for_qdrant(doc):
    """
    Prepare a processed document for storage in Qdrant.
    Converts document structure into TextNode objects with metadata.
    
    Args:
        doc: Processed document with raw_extracted_content
        
    Returns:
        List of TextNode objects
    """
    if not isinstance(doc, dict) or "raw_extracted_content" not in doc:
        return []
        
    text_nodes = []
    raw_content = doc["raw_extracted_content"]
    filename = raw_content.get("filename", "Unknown")
    
    # Process each page's subsections
    for page in raw_content.get("pages", []):
        page_num = page.get("page_number", 0)
        
        for subsection in page.get("subsections", []):
            subsection_id = subsection.get("subsection_id", f"page_{page_num}_section_unknown")
            title = subsection.get("title", "Untitled Section")
            content = subsection.get("content", "")
            
            if not content:
                continue  # Skip empty content
                
            # Find chapter for this subsection
            chapter_title = "Unknown Chapter"
            for chapter in raw_content.get("summary", {}).get("chapters", []):
                for chapter_subsection in chapter.get("subsections", []):
                    if isinstance(chapter_subsection, dict) and chapter_subsection.get("subsection_id") == subsection_id:
                        chapter_title = chapter.get("title", "Unknown Chapter")
                        break
            
            # Extract topics
            topics = page.get("topics", [])
            
            # Create node with metadata
            node = TextNode(
                text=content,
                id_=subsection_id,
                metadata={
                    "file_name": filename,
                    "page_number": page_num,
                    "subsection_id": subsection_id,
                    "chapter_title": chapter_title,
                    "chunk_metadata": {
                        "chunk_title": title,
                        "key_topics": topics
                    }
                }
            )
            
            text_nodes.append(node)
    
    return text_nodes

def main():
    """Main function to run the Streamlit application."""
    # Configure page
    st.set_page_config(
        page_title="Advanced Document Processor",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = []
    if "conversation_store" not in st.session_state:
        st.session_state.conversation_store = []
    if "processing_active" not in st.session_state:
        st.session_state.processing_active = False
    if "guide_data" not in st.session_state:
        st.session_state.guide_data = None
    if "unified_graph" not in st.session_state:
        st.session_state.unified_graph = None
    if "current_graph_doc" not in st.session_state:
        st.session_state.current_graph_doc = None
    
    # --- SIDEBAR ---
    display_sidebar_chat()
    
    # --- MAIN CONTENT ---
    st.title("📑 Advanced Document Processor")
    
    if st.session_state.get("processing_active"):
        with st.status("Processing documents..."):
            st.write("This may take a few minutes")
    elif st.session_state.processed_documents:
        # Add tabs for different views
        main_tabs = st.tabs([
            "Document Reports", 
            "Folder Analysis", 
            "Vector Search", 
            "Goal-Driven Guides",
            "Graph Visualization"
        ])
        
        with main_tabs[0]:
            # Document selection and rendering
            if len(st.session_state.processed_documents) > 1:
                # Document selection UI
                doc_options = []
                for d in st.session_state.processed_documents:
                    if isinstance(d, dict):
                        name = d.get("filename") or d.get("raw_extracted_content", {}).get("filename")
                    else:
                        name = getattr(d, "filename", None)
                    if name:
                        doc_options.append(name)
                
                selected_name = st.selectbox(
                    "Choose document:",
                    options=doc_options,
                    key="doc_selector"
                )
                
                # Find matching document
                doc = None
                for d in st.session_state.processed_documents:
                    current_name = None
                    if isinstance(d, dict):
                        current_name = d.get("filename") or d.get("raw_extracted_content", {}).get("filename")
                    else:
                        current_name = getattr(d, "filename", None)
                    
                    if current_name == selected_name:
                        doc = d
                        break
                
                if not doc:
                    doc = st.session_state.processed_documents[0]
            else:
                doc = st.session_state.processed_documents[0]
            
            # Render the selected document
            render_unified_document_report(doc)
        
        with main_tabs[1]:
            # Display folder-level UI
            display_folder_level_ui()
        
        with main_tabs[2]:
            # Display vector search UI
            st.header("🔍 Vector Search")
            
            # Add document to vector store
            with st.expander("Add Documents to Vector Store", expanded=True):
                # Check if we have documents ready for processing
                if st.session_state.processed_documents:
                    # Button to prepare documents for vector store
                    if "documents_with_summary" not in st.session_state and st.button("Prepare Documents for Vector Store"):
                        with st.spinner("Converting documents to vector format..."):
                            # Process each document
                            documents_by_file = {}
                            for doc in st.session_state.processed_documents:
                                if isinstance(doc, dict) and "raw_extracted_content" in doc:
                                    filename = doc["raw_extracted_content"].get("filename", "Unknown")
                                    nodes = prepare_document_for_qdrant(doc)
                                    if nodes:
                                        documents_by_file[filename] = nodes
                            
                            # Store in session state
                            st.session_state.documents_with_summary = documents_by_file
                            st.success(f"Prepared {len(documents_by_file)} documents for vector storage!")
                
                # Document storage component
                document_storage_component()
            
            # Document search UI
            with st.expander("Search Documents", expanded=True):
                document_search_component()
        
        with main_tabs[3]:
            # Display goal-driven guide generation UI
            display_guide_generation_ui()
            
        with main_tabs[4]:
            # Display graph visualization UI
            display_graph_visualization_ui()
            
    else:
        # Display welcome screen
        st.info("Upload documents in the sidebar to begin")
        
        # Project description
        st.markdown("""
        ## Advanced Document Processing Pipeline
        
        This application follows a complete hierarchical document processing pipeline:
        
        1. **Initial Extraction & Analysis** - Extract content from document pages
        2. **Chapter Organization** - Group subsections into logical chapters
        3. **Chapter Synthesis** - Generate rich content for each chapter
        4. **File-Level Overview** - Create document-wide summaries and relationships
        5. **Folder-Level Processing** - Organize multiple documents
        6. **Incremental Updates** - Add new files without reprocessing everything
        7. **Goal-Driven Guides** - Create task-oriented guides based on the processed content
        8. **Graph Visualization** - Visualize document relationships at multiple levels
        9. **Vector Search** - Store and search documents using vector embeddings
        
        ### Getting Started
        
        1. Upload one or more documents using the sidebar
        2. Wait for processing to complete
        3. Explore document content, folder-level connections, and generate guides
        4. Visualize document relationships with interactive graphs
        5. Add documents to the vector store for semantic search
        6. Use the chat interface to ask questions about your documents
        """)
    
    # Check for retrieved context
    if "current_context_id" in st.session_state:
        context = retrieve_conversation_context(st.session_state.current_context_id)
        if context:
            with st.expander("Retrieved Conversation Context", expanded=True):
                st.markdown(f"**Timestamp:** {context['timestamp']}")
                st.markdown(f"**Summary:** {context['summary']}")
                st.markdown("**User Query:**")
                st.markdown(f"> {context['user']}")
                st.markdown("**Assistant Response:**")
                st.markdown(f"{context['assistant']}")
            
            # Option to use as context for new question
            if st.button("Use as Context for New Question"):
                st.session_state.messages.append({
                    "role": "system", 
                    "content": f"Using context from previous conversation: {context['summary']}",
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                # Remove current_context_id to close the expander
                del st.session_state.current_context_id
                st.rerun()

if __name__ == "__main__":
    main()                    