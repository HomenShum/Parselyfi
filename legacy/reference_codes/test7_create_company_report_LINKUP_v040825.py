import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import logging
import io
import pandas as pd
import copy

import streamlit as st
from pydantic import BaseModel, Field
from threading import Thread

from google import genai
from google.genai import types

# Import LinkUp
from linkup import LinkupClient

# Import the EthicalWebScraper
from test8_util_ethical_scraper_v040525 import EthicalWebScraper, ScraperConfig
import asyncio
from urllib.parse import urlparse
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Gemini client
GEMINI_API_KEY = st.secrets.get("GOOGLE_AI_STUDIO", os.getenv("GOOGLE_AI_STUDIO"))

# Initialize LinkUp client
LINKUP_API_KEY = st.secrets.get("LINKUP_API_KEY", os.getenv("LINKUP_API_KEY"))
if LINKUP_API_KEY:
    linkup_client = LinkupClient(api_key=LINKUP_API_KEY)
    logger.info("LinkUp client initialized.")
else:
    linkup_client = None
    logger.warning("LinkUp API key not found. LinkUp features will be disabled.")

# Define LinkUp schema for company information
LINKUP_COMPANY_SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {
            "type": "string",
            "description": "Full legal name of the company."
        },
        "company_website": {
            "type": "string",
            "format": "uri",
            "description": "URL of the company's official website, if found."
        },
        "company_description": {
            "type": "string",
            "description": "Comprehensive description of the company's business, mission, and offerings."
        },
        "industry": {
            "type": "string",
            "description": "Primary industry or sector the company operates in."
        },
        "headquarters": {
            "type": "string",
            "description": "Location of the company's headquarters, including city and country."
        },
        "founded_year": {
            "type": "string",
            "description": "Year the company was founded."
        },
        "company_size": {
            "type": "string",
            "description": "Approximate number of employees or size category."
        },
        "business_model": {
            "type": "string",
            "description": "Description of how the company generates revenue (B2B, B2C, SaaS, etc.)."
        },
        "leadership_team": {
            "type": "array",
            "description": "Key leadership personnel at the company.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the leader."},
                    "role": {"type": "string", "description": "Position or title."},
                    "linkedin_url": {"type": "string", "description": "LinkedIn profile URL, if available."},
                    "is_founder": {"type": "boolean", "description": "Whether the person is a founder."},
                    "background": {"type": "string", "description": "Brief professional background."}
                },
                "required": ["name", "role"]
            }
        },
        "funding": {
            "type": "array",
            "description": "Funding rounds and investments, from most recent to oldest.",
            "items": {
                "type": "object",
                "properties": {
                    "round": {"type": "string", "description": "Funding round type (Seed, Series A, etc.)."},
                    "date": {"type": "string", "description": "Date of the funding round."},
                    "amount": {"type": "string", "description": "Amount raised."},
                    "investors": {"type": "array", "items": {"type": "string"}, "description": "List of investors."},
                    "lead_investor": {"type": "string", "description": "Lead investor of the round."}
                },
                "required": ["round", "date", "amount"]
            }
        },
        "products": {
            "type": "array",
            "description": "Main products or services offered by the company.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the product or service."},
                    "description": {"type": "string", "description": "Comprehensive description."},
                    "launch_date": {"type": "string", "description": "When the product was launched."},
                    "key_features": {"type": "array", "items": {"type": "string"}, "description": "Main features or capabilities."}
                },
                "required": ["name", "description"]
            }
        },
        "tech_stack": {
            "type": "array",
            "description": "Technologies and tools used by the company.",
            "items": {"type": "string"}
        },
        "competitors": {
            "type": "array",
            "description": "Main competitors in the market.",
            "items": {"type": "string"}
        },
        "recent_news": {
            "type": "array",
            "description": "Recent news or announcements about the company from the past 3 months.",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title of the news item."},
                    "date": {"type": "string", "description": "Date of the news."},
                    "summary": {"type": "string", "description": "Brief summary of the news."},
                    "url": {"type": "string", "description": "URL to the full news article."}
                },
                "required": ["title", "date", "summary"]
            }
        },
        "market_position": {
            "type": "object",
            "description": "Company's position in the market.",
            "properties": {
                "strengths": {"type": "array", "items": {"type": "string"}, "description": "Key competitive advantages."},
                "challenges": {"type": "array", "items": {"type": "string"}, "description": "Major challenges or weaknesses."},
                "market_share": {"type": "string", "description": "Estimated market share, if available."},
                "growth_trajectory": {"type": "string", "description": "Recent growth pattern (rapid, steady, declining)."}
            }
        },
        "sources": {
            "type": "array",
            "description": "Source URLs used to compile this company information",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title of the source page"},
                    "url": {"type": "string", "description": "URL of the source"},
                    "domain": {"type": "string", "description": "Domain of the source"}
                },
                "required": ["url"]
            }
        }        
    },
    "required": ["company_name", "company_description", "industry", "headquarters", "founded_year"]
}
LINKUP_COMPANY_SCHEMA_STR = json.dumps(LINKUP_COMPANY_SCHEMA)

# Output directory configuration
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "company_research_reports")

# ===================== MODELS =====================
class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    rank: int = 0
    is_official_site: bool = False
    is_news_article: bool = False

class SearchResultsList(BaseModel):
    query: str
    results: List[SearchResult] = []
    has_official_website: bool = False
    has_recent_news: bool = False

class CompanySearchTask(BaseModel):
    company_name: str = ""
    additional_context: str = ""
    status: str = "pending"  # pending, running, complete, error
    results: Dict[str, Any] = {}
    error: str = ""
    duration: float = 0.0

class TokenUsage(BaseModel):
    agent_name: str
    token_count: int
    timestamp: datetime = Field(default_factory=datetime.now)

# Create a session state variable to track token usage
if 'token_usage' not in st.session_state:
    st.session_state.token_usage = []

# Create a semaphore for LinkUp API rate limiting
sem = asyncio.Semaphore(5)  # Limit to 5 concurrent requests


# ===================== ASYNC TASK RUNNER =====================
def generate_task_key(*args):
    """Generate a unique hash-based key for a task."""
    import hashlib
    return hashlib.sha256("-".join(map(str, args)).encode()).hexdigest()

class AsyncTaskRunner:
    def __init__(self):
        self.loop = None
        self.thread = None
        self.initialized = False

    def initialize(self):
        if not self.initialized:
            self.loop = asyncio.new_event_loop()
            self.thread = Thread(target=self._run_event_loop_forever)
            self.thread.daemon = True  # Set daemon to True for clean exit
            self.thread.start()
            self.initialized = True

    def _run_event_loop_forever(self):
        """Thread target function to run the event loop forever."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_task(self, coro):
        """Run a coroutine in the worker thread and return its result."""
        if not self.initialized:
            self.initialize()
            
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def run_task_async(self, coro):
        """Run a coroutine in the worker thread without waiting for its result."""
        if not self.initialized:
            self.initialize()
            
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

@st.cache_resource(show_spinner=False)
def get_async_runner():
    runner = AsyncTaskRunner()
    runner.initialize()
    return runner
# ===================== HELPER FUNCTIONS =====================
async def retry_api_call(func, max_retries=3, initial_delay=1, *args, **kwargs):
    """
    Retry an API call with exponential backoff
    
    Args:
        func: The async function to call
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        The result of the function call
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {str(e)}")
            
            # Wait with exponential backoff
            await asyncio.sleep(delay)
            delay *= 2
    
    # If we get here, all retries failed
    logger.error(f"API call failed after {max_retries} attempts: {str(last_exception)}")
    raise last_exception

def estimate_remaining_time(completed_tasks, total_tasks, elapsed_time):
    """
    Estimate remaining time based on completed tasks and elapsed time
    
    Args:
        completed_tasks: Number of completed tasks
        total_tasks: Total number of tasks to complete
        elapsed_time: Time elapsed so far in seconds
        
    Returns:
        Estimated remaining time in seconds
    """
    if completed_tasks == 0:
        return 0
    
    avg_time_per_task = elapsed_time / completed_tasks
    remaining_tasks = total_tasks - completed_tasks
    return avg_time_per_task * remaining_tasks

def format_time(seconds):
    """Format time in seconds to a readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

async def count_tokens_wrapper(client, model_name, contents, agent_name="Unknown Agent"):
    """
    Count tokens for the given content and add to the session state tracking
    
    Args:
        client: Gemini API client
        model_name: Model name to use
        contents: Content to count tokens for
        agent_name: Name of the agent for tracking purposes
        
    Returns:
        Number of tokens
    """
    try:
        response = await retry_api_call(
            client.aio.models.count_tokens,
            model=model_name,
            contents=contents
        )
        
        token_count = response.total_tokens
        
        # Add to a global variable that we'll sync with session state later
        if not hasattr(count_tokens_wrapper, 'token_usage_buffer'):
            count_tokens_wrapper.token_usage_buffer = []
            
        count_tokens_wrapper.token_usage_buffer.append({
            "agent_name": agent_name,
            "token_count": token_count,
            "timestamp": datetime.now()
        })
        
        logger.info(f"Token count for {agent_name}: {token_count}")
        return token_count
    except Exception as e:
        logger.error(f"Error counting tokens for {agent_name}: {str(e)}")
        return 0

def sync_token_usage_to_session_state():
    """
    Sync collected token usage data to the session state
    Call this function from the main thread
    """
    if hasattr(count_tokens_wrapper, 'token_usage_buffer') and count_tokens_wrapper.token_usage_buffer:
        # Initialize session state if needed
        if 'token_usage' not in st.session_state:
            st.session_state.token_usage = []
            
        # Add each buffered token usage to session state
        for usage in count_tokens_wrapper.token_usage_buffer:
            st.session_state.token_usage.append(
                TokenUsage(
                    agent_name=usage["agent_name"],
                    token_count=usage["token_count"],
                    timestamp=usage["timestamp"]
                )
            )
        
        # Clear the buffer
        count_tokens_wrapper.token_usage_buffer = []
        return True
    return False


# ===================== CONTEXT EXTRACTION HELPERS =====================
def extract_context_category(context_text: str, keywords: List[str]) -> str:
    """
    Extract information related to specific context categories based on keywords
    
    Args:
        context_text: The additional context text
        keywords: List of keywords to match for this category
        
    Returns:
        Extracted context for this category, or empty string if not found
    """
    if not context_text:
        return ""
        
    # Convert to lowercase for case-insensitive matching
    context_lower = context_text.lower()
    
    # Try to extract information based on key-value pairs
    for keyword in keywords:
        # Look for patterns like "keyword: value" or "keyword - value"
        patterns = [
            rf'{keyword}\s*:\s*([^,;]+)',
            rf'{keyword}\s*-\s*([^,;]+)',
            rf'{keyword}[=]\s*([^,;]+)'
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, context_lower)
            if matches:
                return matches.group(1).strip()
    
    # Try to find sentences containing the keywords
    for keyword in keywords:
        if keyword in context_lower:
            # Find the sentence containing the keyword
            sentences = re.split(r'[.!?]\s+', context_text)
            for sentence in sentences:
                if keyword.lower() in sentence.lower():
                    # Return just the relevant part if possible
                    parts = sentence.split(",")
                    for part in parts:
                        if keyword.lower() in part.lower():
                            return part.strip()
                    return sentence.strip()
    
    return ""

def extract_disambiguated_company_info(company_name: str, additional_context: str) -> Dict[str, str]:
    """
    Extract disambiguating information about a company
    
    Args:
        company_name: Name of the company
        additional_context: Additional context about the company
        
    Returns:
        Dictionary with disambiguating information
    """
    info = {
        "industry": extract_context_category(additional_context, 
                                           ["industry", "sector", "field", "domain", "market"]),
        "location": extract_context_category(additional_context, 
                                           ["located", "location", "headquarters", "hq", "based in", "region", "city", "country"]),
        "founded": extract_context_category(additional_context, 
                                          ["founded", "established", "started", "creation", "year", "since"]),
        "founders": extract_context_category(additional_context, 
                                           ["founder", "creator", "started by", "founded by", "ceo", "owner"]),
        "product": extract_context_category(additional_context, 
                                          ["product", "service", "offering", "solution", "platform", "app", "software"]),
        "size": extract_context_category(additional_context, 
                                       ["employees", "team size", "company size", "headcount", "staff"]),
        "website": extract_context_category(additional_context, 
                                          ["website", "site", "url", "web", "domain", "online at"]),
    }
    
    # Clean up extracted values
    for key in info:
        # Remove irrelevant prefixes that might have been captured
        if info[key]:
            for prefix in [f"{key}:", f"{key} -", f"{key}="]:
                if info[key].lower().startswith(prefix.lower()):
                    info[key] = info[key][len(prefix):].strip()
    
    return info

def build_structured_prompt(company_name: str, additional_context: str) -> str:
    """
    Build a structured prompt following best practices
    
    Args:
        company_name: Name of the company
        additional_context: Additional context about the company
        
    Returns:
        Structured prompt for LinkUp
    """
    # Extract disambiguation information
    company_info = extract_disambiguated_company_info(company_name, additional_context)
    
    # Build the core prompt
    prompt = f'''
Research on company "{company_name}":

1. COMPANY OVERVIEW & IDENTITY
   - Full legal name and primary website
   - Comprehensive description of business focus and value proposition
   - Industry classification and market position
   - Founding date, headquarters location, and company size
   - Business model and revenue generation approach

2. LEADERSHIP & ORGANIZATION
   - CEO and key executive team members with roles
   - Founders' backgrounds and current positions
   - Board of directors composition if available
   - Company structure and key departments

3. FINANCIAL INSIGHTS
   - Total funding raised to date
   - Details of funding rounds with dates, amounts, and investors
   - Recent acquisitions or being acquired information
   - Revenue estimates or financial performance indicators if public

4. PRODUCT & TECHNOLOGY
   - Main products/services with detailed descriptions
   - Technology stack and infrastructure
   - Recent product launches or updates
   - Key differentiators in the marketplace

5. MARKET & COMPETITION
   - Primary target market and customer segments
   - Major competitors with comparative strengths
   - Market opportunities and challenges
   - Strategic partnerships or alliances

6. RECENT DEVELOPMENTS
   - News and press releases from the past 3 months
   - Recent growth metrics or milestones
   - Regulatory developments or challenges
   - Planned expansion or future direction
'''

    # Add disambiguation information if available
    if any(company_info.values()):
        prompt += "\n\nADDITIONAL COMPANY IDENTIFIERS:"
        
        if company_info["website"]:
            prompt += f"\n- Company website: {company_info['website']}"
        
        if company_info["industry"]:
            prompt += f"\n- Industry/sector: {company_info['industry']}"
            
        if company_info["location"]:
            prompt += f"\n- Headquarters: {company_info['location']}"
            
        if company_info["founded"]:
            prompt += f"\n- Founded: {company_info['founded']}"
            
        if company_info["founders"]:
            prompt += f"\n- Founder(s)/CEO: {company_info['founders']}"
            
        if company_info["product"]:
            prompt += f"\n- Primary product/service: {company_info['product']}"
            
        if company_info["size"]:
            prompt += f"\n- Company size: {company_info['size']}"
    
    # Add source preferences
    prompt += '''

SOURCE PREFERENCES:
- Primary: Official company website, SEC filings, earnings reports, press releases
- Secondary: Professional profiles (LinkedIn, Crunchbase), industry databases
- Tertiary: Recent news articles, interviews, and analyses from reputable sources
- Time frame for news: Focus on last 3 months for recent developments
'''

    return prompt

# ===================== LINKUP SEARCH FUNCTIONS =====================
async def company_search_with_linkup(company_name: str,
                                    additional_context: str = "",
                                    schema_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for company information using LinkUp, prioritizing sourcedAnswer for verifiability.

    Args:
        company_name: The name of the company to search for.
        additional_context: The prompt/query (can include context, instructions, schema).
        schema_str: The JSON schema string (optional, used for prompting).

    Returns:
        Dictionary containing:
        - success: Boolean status.
        - answer: The raw text answer from LinkUp.
        - sources: List of source objects (name, url, snippet) from LinkUp.
        - structured_data: Parsed JSON data if found in the answer, else None.
        - search_text_formatted: A formatted string combining answer/structured data.
        - error: Error message if applicable.
        - raw_response: The complete raw response from LinkUp.
    """
    if not linkup_client:
        logger.warning("LinkUp client not available.")
        return {
            "success": False, "error": "LinkUp client not initialized",
            "answer": "", "sources": [], "structured_data": None,
            "search_text_formatted": "",
            "raw_response": None
        }

    # Use the provided context/prompt directly
    query = additional_context

    logger.info(f"Querying LinkUp (sourcedAnswer) for: {company_name}")
    logger.debug(f"Using prompt: {query[:300]}...") # Log just a preview of the prompt

    try:
        # Always use sourcedAnswer for better traceability
        async with sem:
            response = await asyncio.to_thread(
                linkup_client.search,
                query=query,
                depth="standard",
                include_images=True,
                output_type="sourcedAnswer"
            )
            
        # Check if response exists
        if not response:
            logger.error(f"No response received from LinkUp for {company_name}")
            return {
                "success": False, "error": "No response from LinkUp",
                "answer": "", "sources": [], "structured_data": None,
                "search_text_formatted": f"Error: No response from LinkUp for {company_name}",
                "raw_response": None
            }

        # Store raw response
        raw_response = response
        print(f"Raw response type: {type(raw_response).__name__}")
        print(f"Raw response: {raw_response}")
        # Extract answer and sources using safe_get_attribute helper
        answer_text = safe_get_attribute(response, "answer", "")
        sources = safe_get_attribute(response, "sources", [])
        
        if not answer_text:
            logger.warning(f"No answer found in LinkUp response for {company_name}")
        
        if not sources:
            logger.warning(f"No sources found in LinkUp response for {company_name}")
        # Format the output text
        search_text_formatted = f"# Information for {company_name}\n\n"
        search_text_formatted += answer_text if answer_text else f"No information found for {company_name}."
        
        # Add source information to the formatted text
        if sources:
            search_text_formatted += "\n\n## Sources (from LinkUp)"
            for i, src in enumerate(sources):
                # Extract source fields using the safe_get_attribute helper
                src_name = safe_get_attribute(src, "name", f"Source {i+1}")
                src_url = safe_get_attribute(src, "url", "N/A")
                src_snippet = safe_get_attribute(src, "snippet", "")
                
                # Format the source information
                search_text_formatted += f"\n{i+1}. **{src_name}**"
                search_text_formatted += f"\n   - URL: {src_url}"
                if src_snippet:
                    search_text_formatted += f"\n   - Snippet: {src_snippet[:300]}..." if len(src_snippet) > 300 else f"\n   - Snippet: {src_snippet}"
                else:
                    search_text_formatted += "\n   - Snippet: N/A"
        elif not answer_text:
            search_text_formatted += "\nNo sources were returned."

        return {
            "success": True,
            "answer": answer_text,
            "sources": sources,
            "structured_data": None,  # No JSON parsing, so this is always None
            "search_text_formatted": search_text_formatted,
            "parsing_error": None,  # No JSON parsing, so this is always None
            "raw_response": raw_response
        }

    except Exception as e:
        logger.error(f"Error during LinkUp search for {company_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False, "error": str(e),
            "answer": "", "sources": [], "structured_data": None,
            "search_text_formatted": f"Error searching for {company_name}: {str(e)}",
            "raw_response": None
        }
        
def safe_get_attribute(obj, attribute_name, default=None):
    """
    Safely gets an attribute from an object, whether it's a dictionary or has attributes.
    
    Args:
        obj: The object to get the attribute from
        attribute_name: The name of the attribute to get
        default: The default value to return if the attribute doesn't exist
        
    Returns:
        The attribute value or default
    """
    if hasattr(obj, attribute_name):
        return getattr(obj, attribute_name)
    elif isinstance(obj, dict) and attribute_name in obj:
        return obj[attribute_name]
    else:
        return default

def extract_source_fields(source):
    """
    Extracts fields from a source object, handling both dictionary and attribute access.
    
    Args:
        source: The source object
        
    Returns:
        Dictionary with name, url, and snippet fields
    """
    return {
        "name": safe_get_attribute(source, "name", "Unnamed Source"),
        "url": safe_get_attribute(source, "url", ""),
        "snippet": safe_get_attribute(source, "snippet", "")
    }

def access_linkup_response(response):
    """
    Accesses a LinkupSourcedAnswer response, normalizing the data structure.
    
    Args:
        response: The LinkupSourcedAnswer response
        
    Returns:
        Dictionary with answer and sources
    """
    answer = safe_get_attribute(response, "answer", "")
    sources_raw = safe_get_attribute(response, "sources", [])
    
    # Normalize sources to always be a list of dictionaries
    sources = [extract_source_fields(source) for source in sources_raw]
    
    return {
        "answer": answer,
        "sources": sources
    }

# Example usage:
def process_linkup_data(response):
    """
    Process data from a LinkupClient response, handling any response type.
    
    Args:
        response: The response from LinkupClient
        
    Returns:
        Processed data with normalized structure
    """
    # Get normalized data
    data = access_linkup_response(response)
    
    # Now you can safely work with the data using dictionary access
    answer_text = data["answer"]
    sources = data["sources"]
    
    # Process sources
    formatted_sources = []
    for source in sources:
        url = source.get("url") if isinstance(source, dict) else (source.url if hasattr(source, "url") else None)
        name = source.get("name") if isinstance(source, dict) else (source.name if hasattr(source, "name") else "Source")
        snippet = source.get("snippet") if isinstance(source, dict) else (source.snippet if hasattr(source, "snippet") else "")
        
        formatted_source = f"**{name}**\n- URL: {url or 'N/A'}"
        if snippet:
            formatted_source += f"\n- Snippet: {snippet[:100]}..." if len(snippet) > 100 else f"\n- Snippet: {snippet}"
        formatted_sources.append(formatted_source)
    return {
        "answer": answer_text,
        "formatted_sources": formatted_sources
    }

def test_linkup_response_structure(response):
    """Tests and prints the structure of a LinkupSourcedAnswer response for debugging.
    
    Args:
        response: The response from LinkupClient
        
    Returns:
        A string with detailed information about the response structure
    """
    results = []
    
    # General object information
    results.append(f"Response type: {type(response).__name__}")
    results.append(f"Is dictionary? {isinstance(response, dict)}")
    results.append(f"Has __dict__ attribute? {hasattr(response, '__dict__')}")
    
    # Check for common attributes/keys
    for field in ["answer", "sources"]:
        # Check for attribute access
        has_attr = hasattr(response, field)
        results.append(f"Has '{field}' attribute? {has_attr}")
        
        # If it has the attribute, get the value type
        if has_attr:
            attr_value = getattr(response, field)
            results.append(f"  Type of {field}: {type(attr_value).__name__}")
            
            # If it's a sequence, check its length and first item
            if hasattr(attr_value, "__len__"):
                results.append(f"  Length of {field}: {len(attr_value)}")
                
                # For sources, check the first source structure
                if field == "sources" and len(attr_value) > 0:
                    first_item = attr_value[0]
                    results.append(f"  First {field[:-1]} type: {type(first_item).__name__}")
                    
                    # Check source fields
                    for source_field in ["name", "url", "snippet"]:
                        has_source_attr = hasattr(first_item, source_field)
                        results.append(f"    Has '{source_field}' attribute? {has_source_attr}")
                        
                        if isinstance(first_item, dict):
                            results.append(f"    Has '{source_field}' key? {source_field in first_item}")
    
    # Check if it has a dictionary interface but isn't a dict
    if not isinstance(response, dict) and hasattr(response, "__getitem__"):
        results.append("Object has __getitem__ method but is not a dictionary")
        try:
            # Try to access as a dictionary
            for field in ["answer", "sources"]:
                try:
                    results.append(f"Can access ['{field}'] via indexing? {field in response}")
                except Exception as e:
                    results.append(f"Error checking if '{field}' in response: {str(e)}")
        except Exception as e:
            results.append(f"Error testing dictionary-like access: {str(e)}")
    
    # Check for other interesting attributes
    if hasattr(response, "__dir__"):
        all_attrs = dir(response)
        public_attrs = [attr for attr in all_attrs if not attr.startswith("_")]
        results.append(f"Public attributes/methods: {', '.join(public_attrs)}")
    
    # If it has __dict__, show top-level keys
    if hasattr(response, "__dict__"):
        results.append(f"__dict__ keys: {', '.join(response.__dict__.keys())}")
    
    return "\n".join(results)

def format_linkup_results_as_search_text(structured_data: Dict[str, Any], company_name: str) -> str:
    """
    Format structured LinkUp results as search results text
    
    Args:
        structured_data: Structured data from LinkUp
        company_name: Name of the company
        
    Returns:
        Formatted search results text
    """
    lines = []
    
    # Company header
    lines.append(f"# {structured_data.get('company_name', company_name)}")
    lines.append("")
    
    # Section 1: Company Overview
    lines.append("## 1. Company Overview")
    
    # Website
    if website := structured_data.get('company_website'):
        lines.append(f"**Official Website:** {website}")
    
    # Company description
    if desc := structured_data.get('company_description'):
        lines.append(f"**Description:** {desc}")
    
    # Basic info
    basic_info = []
    if industry := structured_data.get('industry'):
        basic_info.append(f"Industry: {industry}")
    if hq := structured_data.get('headquarters'):
        basic_info.append(f"Headquarters: {hq}")
    if founded := structured_data.get('founded_year'):
        basic_info.append(f"Founded: {founded}")
    if size := structured_data.get('company_size'):
        basic_info.append(f"Company Size: {size}")
    if biz_model := structured_data.get('business_model'):
        basic_info.append(f"Business Model: {biz_model}")
    
    if basic_info:
        lines.append("**Company Details:**")
        for info in basic_info:
            lines.append(f"- {info}")
    lines.append("")
    
    # Section 2: Leadership Team
    if leaders := structured_data.get('leadership_team'):
        lines.append("## 2. Leadership Team")
        for leader in leaders:
            name = leader.get('name', '')
            role = leader.get('role', '')
            is_founder = leader.get('is_founder', False)
            linkedin = leader.get('linkedin_url', '')
            bg = leader.get('background', '')
            
            leader_title = f"**{name}**"
            if role:
                leader_title += f", {role}"
            if is_founder:
                leader_title += " (Founder)"
            
            lines.append(leader_title)
            
            if bg:
                lines.append(f"- Background: {bg}")
            if linkedin:
                lines.append(f"- LinkedIn: {linkedin}")
            
            lines.append("")
    
    # Section 3: Funding History
    if funding := structured_data.get('funding'):
        lines.append("## 3. Funding History")
        for round_data in funding:
            round_name = round_data.get('round', '')
            date = round_data.get('date', '')
            amount = round_data.get('amount', '')
            
            round_line = f"**{round_name}**"
            if date:
                round_line += f" ({date})"
            if amount:
                round_line += f": {amount}"
            lines.append(round_line)
            
            if lead := round_data.get('lead_investor'):
                lines.append(f"- Lead Investor: {lead}")
            
            if investors := round_data.get('investors'):
                if len(investors) == 1:
                    lines.append(f"- Investor: {investors[0]}")
                elif len(investors) > 1:
                    lines.append(f"- Investors: {', '.join(investors[:3])}" + 
                                (f" and {len(investors) - 3} more" if len(investors) > 3 else ""))
        
        lines.append("")
    
    # Section 4: Products & Technology
    products_section_added = False
    
    if products := structured_data.get('products'):
        products_section_added = True
        lines.append("## 4. Products & Technology")
        for product in products:
            name = product.get('name', '')
            desc = product.get('description', '')
            launch = product.get('launch_date', '')
            features = product.get('key_features', [])
            
            if name:
                product_line = f"**{name}**"
                if launch:
                    product_line += f" (Launched: {launch})"
                lines.append(product_line)
                
                if desc:
                    lines.append(f"- {desc}")
                
                if features:
                    lines.append("- Key Features:")
                    for feature in features:
                        lines.append(f"  • {feature}")
                
                lines.append("")
    
    if tech_stack := structured_data.get('tech_stack'):
        if not products_section_added:
            lines.append("## 4. Products & Technology")
            products_section_added = True
        
        lines.append("**Technology Stack:**")
        # Group technologies into batches of 5 for readability
        for i in range(0, len(tech_stack), 5):
            batch = tech_stack[i:i+5]
            lines.append(f"- {', '.join(batch)}")
        
        lines.append("")
    
    # Section 5: Market Position & Competitors
    market_section_added = False
    
    if market_pos := structured_data.get('market_position'):
        market_section_added = True
        lines.append("## 5. Market Position & Competitors")
        
        if strengths := market_pos.get('strengths'):
            lines.append("**Strengths:**")
            for strength in strengths:
                lines.append(f"- {strength}")
            lines.append("")
        
        if challenges := market_pos.get('challenges'):
            lines.append("**Challenges:**")
            for challenge in challenges:
                lines.append(f"- {challenge}")
            lines.append("")
        
        if market_share := market_pos.get('market_share'):
            lines.append(f"**Market Share:** {market_share}")
        
        if growth := market_pos.get('growth_trajectory'):
            lines.append(f"**Growth Trajectory:** {growth}")
        
        lines.append("")
    
    if competitors := structured_data.get('competitors'):
        if not market_section_added:
            lines.append("## 5. Market Position & Competitors")
            market_section_added = True
        
        lines.append("**Competitors:**")
        # List all competitors
        for competitor in competitors:
            lines.append(f"- {competitor}")
        
        lines.append("")
    
    # Section 6: Recent News & Developments
    if news := structured_data.get('recent_news'):
        lines.append("## 6. Recent News & Developments")
        
        for news_item in news:
            title = news_item.get('title', '')
            date = news_item.get('date', '')
            summary = news_item.get('summary', '')
            url = news_item.get('url', '')
            
            if title:
                news_title = f"**{title}**"
                if date:
                    news_title += f" ({date})"
                lines.append(news_title)
                
                if summary:
                    lines.append(f"- {summary}")
                if url:
                    lines.append(f"- Source: {url}")
                
                lines.append("")

    # Add a new "Sources" section at the end
    if sources := structured_data.get('sources', []):
        lines.append("\n## Sources")
        for idx, source in enumerate(sources):
            title = source.get('title', source.get('url', f'Source {idx+1}'))
            url = source.get('url', '')
            if url:
                lines.append(f"{idx+1}. [{title}]({url})")

    return "\n".join(lines)

def format_unstructured_linkup_results(unstructured_data: Dict[str, Any], company_name: str) -> str:
    """
    Format unstructured LinkUp results as search results text
    
    Args:
        unstructured_data: Unstructured data from LinkUp
        company_name: Name of the company
        
    Returns:
        Formatted search results text
    """
    lines = []
    
    # Company header
    lines.append(f"# Search Results for: {company_name}")
    lines.append("")
    
    # Extract important components from the response
    search_results = unstructured_data.get('search_results', [])
    content = unstructured_data.get('response', {}).get('content', '')
    
    # If we have content directly, use it first
    if content:
        lines.append("## Overview")
        lines.append(content)
        lines.append("")
    
    # Add search results as sections
    if search_results:
        lines.append("## Detailed Information")
        lines.append("")
        
        # Limit to top 5 results for readability
        for idx, result in enumerate(search_results[:5]):
            title = result.get('title', f'Result {idx+1}')
            url = result.get('url', '')
            snippet = result.get('snippet', '')
            
            lines.append(f"### {title}")
            if url:
                lines.append(f"**Source:** {url}")
            if snippet:
                lines.append(f"\n{snippet}\n")
    
    # If no results were found
    if not search_results and not content:
        lines.append(f"No detailed information found for {company_name}")

    # Add explicit sources section
    if search_results := unstructured_data.get('search_results', []):
        lines.append("\n## Sources")
        for idx, result in enumerate(search_results):
            title = result.get('title', f'Source {idx+1}')
            url = result.get('url', '')
            lines.append(f"**{idx+1}. {title}**")
            if url:
                lines.append(f"- URL: {url}")
            lines.append("")

    return "\n".join(lines)

async def entity_context_extraction_agent(client, model_name, text):
    """
    Agent 1: Extract entity names and additional context from text
    
    Args:
        client: Gemini API client
        model_name: Model name to use
        text: Text to extract entities and context from
        
    Returns:
        Dict with entity_names and additional_context
    """
    parsing_prompt = f"""
    Extract company names and additional context from the following information:
    
    {text}
    
    Return your response in JSON format with the following structure:
    {{
        "entity_names": ["company1", "company2", ...],  // Array of company names
        "additional_context": "Any additional context about the companies"  // String with additional context
    }}
    
    When extracting entity names:
    1. Focus on full, official company names when possible
    2. If industry is mentioned, include it in additional_context
    3. If location, founding date, or size is mentioned, include it in additional_context
    4. If product names are mentioned, include them in additional_context
    
    If no company names are mentioned, return an empty array for entity_names.
    """
    
    try:
        # Count tokens for this request
        await count_tokens_wrapper(
            client, 
            model_name, 
            [parsing_prompt], 
            "Entity Extraction Agent"
        )
        
        response = await retry_api_call(
            client.aio.models.generate_content,
            model=model_name,
            contents=[parsing_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        # Parse JSON response
        result = json.loads(response.candidates[0].content.parts[0].text)
        logger.info(f"Entity extraction successful: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in entity_context_extraction_agent: {str(e)}")
        # Return empty result on error
        return {"entity_names": [], "additional_context": ""}

async def company_search_agent(client, model_name, company_name):
    """
    Agent 2: Search for company information using LinkUp instead of Google Search
    
    Args:
        client: Gemini API client
        model_name: Model name to use
        company_name: Company name to search for
        
    Returns:
        Search results text
    """
    try:
        # Count tokens for this request (still using Gemini for token counting)
        await count_tokens_wrapper(
            client, 
            model_name, 
            [f"Search for company information: {company_name}"], 
            f"Company Search Agent ({company_name})"
        )
        
        # Extract additional context for better search
        company_context = extract_disambiguated_company_info(company_name, "")
        additional_context = ", ".join([f"{k}: {v}" for k, v in company_context.items() if v])
        
        # Use LinkUp for the actual search instead of Gemini's Google Search
        search_result = await company_search_with_linkup(company_name, additional_context)
        
        if not search_result["success"]:
            logger.warning(f"No search results for company: {company_name}")
            return ""
            
        search_text = search_result["search_results"]
        logger.info(f"Company search successful for: {company_name}")
        return search_text
    except Exception as e:
        logger.error(f"Error in company_search_agent for {company_name}: {str(e)}")
        return ""

async def result_verification_agent(client, model_name, company_name, additional_context, search_results):
    """
    Agent 3: Verify search results against additional context
    
    Args:
        client: Gemini API client
        model_name: Model name to use
        company_name: Company name
        additional_context: Additional context about the company
        search_results: Search results to verify
        
    Returns:
        Verification text
    """
    # Extract structured information from additional context for clearer verification
    context_info = extract_disambiguated_company_info(company_name, additional_context)
    
    # Build a structured verification prompt
    verification_prompt = f"""
    # Company Verification Task
    
    ## COMPANY TO VERIFY
    Company name: {company_name}
    
    ## KNOWN INFORMATION
    """
    
    # Add structured context information
    if any(context_info.values()):
        for label, field in [
            ("Industry/Sector", "industry"),
            ("Location/HQ", "location"),
            ("Founded", "founded"),
            ("Founder(s)/CEO", "founders"),
            ("Product/Service", "product"),
            ("Company Size", "size"),
            ("Website", "website")
        ]:
            if context_info[field]:
                verification_prompt += f"- {label}: {context_info[field]}\n"
    else:
        verification_prompt += "- Limited additional context available\n"
    
    # Add the search results and verification instructions
    verification_prompt += f"""
    ## SEARCH RESULTS TO VERIFY
    {search_results}
    
    ## VERIFICATION TASK
    1. Compare the search results with the known information
    2. Determine if the search results refer to the same company as described in the known information
    3. Identify any discrepancies or conflicting information
    4. Highlight which information is confirmed by multiple sources
    
    ## RESPONSE FORMAT
    
    VERIFICATION RESULT:
    [State whether search results match the expected company]
    
    CONFIRMED INFORMATION:
    [List information points that are confirmed by multiple sources]
    
    DISCREPANCIES OR CONFLICTS:
    [List any discrepancies or conflicting information]
    
    MISSING INFORMATION:
    [List any important information that is missing from the search results]
    
    RECOMMENDATION:
    [Suggest if additional research is needed and on which aspects]
    """
    
    try:
        # Count tokens for this request
        await count_tokens_wrapper(
            client, 
            model_name, 
            [verification_prompt], 
            f"Verification Agent ({company_name})"
        )
        
        response = await retry_api_call(
            client.aio.models.generate_content,
            model=model_name,
            contents=[verification_prompt]
        )
        
        if not response.candidates or not response.candidates[0].content.parts:
            logger.warning(f"No verification results for company: {company_name}")
            return ""
            
        verification_text = response.candidates[0].content.parts[0].text
        logger.info(f"Result verification successful for: {company_name}")
        return verification_text
    except Exception as e:
        logger.error(f"Error in result_verification_agent for {company_name}: {str(e)}")
        return ""


# ===================== GEMINI COMPANY SEARCH WITH LINKUP =====================
async def gemini_company_search_with_verification(
    user_input: str = "", 
    company_names: List[str] = None, 
    additional_context: str = "",
    status_container = None,
    max_results: int = 10,
    max_concurrent: int = 5,
    enable_web_scraping: bool = True  # New parameter
) -> Dict[str, Any]:
    """
    Generate a company search with verification using LinkUp's search capabilities.
    Parallelized to process multiple companies simultaneously.
    
    Args:
        user_input: The original user query
        company_names: List of company names (if already parsed)
        additional_context: Additional context about the companies
        status_container: Streamlit status container to update progress
        max_results: Maximum number of search results to return
        max_concurrent: Maximum number of concurrent API requests
        enable_web_scraping: Whether to enhance context by scraping URLs
        
    Returns:
        Dictionary with parsed entities, search results, and verified information
    """
    start_time = time.time()
    logger.info(f"Performing company search with verification for: {user_input}")
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        model_name = "gemini-2.0-flash"
        
        # Create a semaphore to limit concurrent API requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Step 1: Parse entity names and additional context if not provided
        if status_container:
            status_container.update(label="Step 1: Parsing company names and additional context...")
            
        if not company_names:
            # Use the entity parser agent with JSON response format
            extraction_result = await entity_context_extraction_agent(
                client, 
                model_name, 
                f"User input: {user_input}"
            )
            
            # Extract entity names from result
            company_names = extraction_result.get("entity_names", [])
            
            # Extract additional context if not provided
            if not additional_context and "additional_context" in extraction_result:
                additional_context = extraction_result.get("additional_context", "").strip()
        
        # If no company names were found, return error and stop execution
        if not company_names:
            if status_container:
                status_container.update(
                    label="No company names could be identified in your query. Please try again with clearer company information.",
                    state="error"
                )
            return {
                "success": False,
                "error": "No company names could be identified in your query",
                "parsed_entities": [],
                "search_results": [],
                "verified_results": []
            }
            
        step1_time = time.time() - start_time
        
        # Step 1.5: Enhance context with web scraping if enabled
        enhanced_context_data = None
        scraped_urls = []
        failed_urls = []
        
        if enable_web_scraping and additional_context:
            if status_container:
                status_container.update(label="Step 1.5: Enhancing context by scraping URLs...")
                
            step15_start_time = time.time()
            
            # Extract and scrape URLs from additional context
            enhanced_context_data = await enhance_context_with_web_scraping(
                additional_context, 
                max_concurrent=max(1, max_concurrent // 2)  # Use half of available concurrency
            )
            
            if enhanced_context_data:
                additional_context = enhanced_context_data.get("enhanced_context", additional_context)
                scraped_urls = enhanced_context_data.get("scraped_urls", [])
                failed_urls = enhanced_context_data.get("failed_urls", [])
                
                if status_container:
                    status_container.update(
                        label=f"Step 1.5: Enhanced context with {len(scraped_urls)} scraped URLs"
                    )
            
            step15_time = time.time() - step15_start_time
        else:
            step15_time = 0
            
        # Step 2: Search for all companies in parallel
        if status_container:
            status_container.update(label=f"Step 2: Searching for information on {len(company_names)} companies in parallel...")
        
        step2_start_time = time.time()
        
        # [Rest of the original function remains unchanged]
        # Search and verification steps continue as before...
        
        # Wrapper function for rate-limited API calls
        async def rate_limited_search(company_name):
            async with semaphore:
                logger.info(f"Starting search for: {company_name}")
                # Extract company-specific context
                company_specific_context = extract_context_for_company(additional_context, company_name)
                # Use LinkUp search instead of Google Search
                search_result = await company_search_with_linkup(company_name, company_specific_context)
                return {
                    "company_name": company_name,
                    "search_results": search_result["search_results"],
                    "structured_data": search_result.get("structured_data")
                }
        
        async def rate_limited_verification(company_name, search_results):
            async with semaphore:
                logger.info(f"Starting verification for: {company_name}")
                # Extract company-specific context
                company_specific_context = extract_context_for_company(additional_context, company_name)
                verification_text = await result_verification_agent(
                    client, model_name, company_name, company_specific_context, search_results
                )
                return {
                    "company_name": company_name,
                    "verification": verification_text
                }
        
        # Create search tasks for all companies
        search_tasks = []
        for company_name in company_names:
            task = asyncio.create_task(rate_limited_search(company_name))
            search_tasks.append(task)
        
        # Execute all search tasks and collect results
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process search results
        all_search_results = []
        for result in search_results:
            if isinstance(result, Exception):
                logger.error(f"Search task failed with exception: {str(result)}")
            elif result["search_results"]:  # Only add successful searches with content
                all_search_results.append(result)
            
            # Update progress if status container is available
            if status_container:
                completed = len(all_search_results)
                status_container.update(
                    label=f"Step 2: Completed {completed}/{len(company_names)} searches"
                )
        
        step2_time = time.time() - step2_start_time
        
        # Step 3: Verify results in parallel if additional context was provided
        verified_results = []
        
        if additional_context and all_search_results:
            if status_container:
                status_container.update(label=f"Step 3: Verifying results against additional context...")
            
            step3_start_time = time.time()
            
            # Create verification tasks for all search results
            verification_tasks = []
            for result in all_search_results:
                company_name = result["company_name"]
                search_results_text = result["search_results"]
                task = asyncio.create_task(rate_limited_verification(company_name, search_results_text))
                verification_tasks.append(task)
            
            # Execute all verification tasks and collect results
            verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
            
            # Process verification results
            for result in verification_results:
                if isinstance(result, Exception):
                    logger.error(f"Verification task failed with exception: {str(result)}")
                elif result["verification"]:  # Only add successful verifications with content
                    verified_results.append(result)
                
                # Update progress if status container is available
                if status_container:
                    completed = len(verified_results)
                    status_container.update(
                        label=f"Step 3: Completed {completed}/{len(all_search_results)} verifications"
                    )
            
            step3_time = time.time() - step3_start_time
        else:
            step3_time = 0
        
        total_time = time.time() - start_time
        
        if status_container:
            status_container.update(
                label=f"Search complete! Processed {len(company_names)} companies in {format_time(total_time)}",
                state="complete"
            )
        
        return {
            "success": True,
            "parsed_entities": company_names,
            "additional_context": additional_context,
            "search_results": all_search_results,
            "verified_results": verified_results if additional_context else [],
            "scraped_urls": scraped_urls,
            "failed_urls": failed_urls,
            "enhanced_context_data": enhanced_context_data,
            "timing": {
                "step1_time": step1_time,
                "step15_time": step15_time,
                "step2_time": step2_time,
                "step3_time": step3_time,
                "total_time": total_time
            }
        }
        
    except Exception as e:
        logger.error(f"Error in gemini_company_search_with_verification: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        if status_container:
            status_container.update(
                label=f"Error: {str(e)}",
                state="error"
            )
        
        return {
            "success": False,
            "error": str(e),
            "parsed_entities": company_names or [],
            "search_results": [],
            "verified_results": []
        }

def extract_context_for_company(general_context: str, company_name: str) -> str:
    """
    Extract company-specific context from general context
    
    Args:
        general_context: General context string that may contain info about multiple companies
        company_name: The specific company to extract context for
        
    Returns:
        Context specific to the company
    """
    if not general_context:
        return ""
    
    # Check if context contains company name - if not, return the whole context
    if company_name.lower() not in general_context.lower():
        return general_context
    
    # Split context by companies
    company_context = ""
    
    # Look for sentences or sections about this company
    sentences = re.split(r'[.!?]\s+', general_context)
    relevant_sentences = []
    
    for sentence in sentences:
        if company_name.lower() in sentence.lower():
            relevant_sentences.append(sentence)
        # Also include adjacent sentences for context
        elif relevant_sentences and len(relevant_sentences) < 5:
            relevant_sentences.append(sentence)
    
    if relevant_sentences:
        company_context = ". ".join(relevant_sentences)
        if not company_context.endswith("."):
            company_context += "."
    else:
        company_context = general_context
    
    return company_context


async def batch_process_companies(search_df, max_concurrent=5, enable_web_scraping=True):
    """
    Process multiple companies from a dataframe in parallel
    
    Args:
        search_df: DataFrame containing company information
        max_concurrent: Maximum number of concurrent requests
        enable_web_scraping: Whether to enhance context with web scraping
        
    Returns:
        List of results for each company
    """
    # Initialize results storage
    all_results = []
    client = genai.Client(api_key=GEMINI_API_KEY)
    model_name = "gemini-2.0-flash"
    
    # Create a semaphore to limit concurrent API requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_company(idx, row):
        """Process a single company with rate limiting"""
        async with semaphore:
            company_name = row['company_name']
            additional_context = row['additional_context']
            
            try:
                logger.info(f"Starting search for: {company_name}")
                
                # Run the search for this company
                response = await gemini_company_search_with_verification(
                    company_names=[company_name],
                    additional_context=additional_context,
                    status_container=None,  # Don't use status container in async context
                    max_concurrent=1,  # Each company search already has internal parallelism
                    enable_web_scraping=enable_web_scraping  # Pass the parameter
                )
                
                logger.info(f"Completed search for: {company_name}")
                return {
                    "idx": idx,
                    "company_name": company_name,
                    "additional_context": additional_context,
                    "result": response,
                    "success": response["success"]
                }
            except Exception as e:
                logger.error(f"Error processing company {company_name}: {str(e)}")
                return {
                    "idx": idx,
                    "company_name": company_name,
                    "additional_context": additional_context,
                    "result": {"success": False, "error": str(e)},
                    "success": False
                }

    
    # Create tasks for all companies
    start_time = time.time()
    tasks = []
    for idx, row in search_df.iterrows():
        task = asyncio.create_task(process_company(idx, row))
        tasks.append(task)
    
    # Execute all tasks and collect results
    # Using gather instead of as_completed to avoid the unhashable dict issue
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {str(result)}")
            # Add a generic error result
            all_results.append({
                "idx": -1,  # Unknown index
                "company_name": "Unknown",
                "additional_context": "",
                "result": {"success": False, "error": str(result)},
                "success": False
            })
        else:
            all_results.append(result)
    
    # Sort by original index to maintain order
    all_results = sorted([r for r in all_results if r["idx"] >= 0], key=lambda x: x["idx"])
    
    total_time = time.time() - start_time
    logger.info(f"All searches complete! Processed {len(search_df)} companies in {format_time(total_time)}")
    
    return {
        "results": all_results,
        "total_time": total_time
    }

def process_companies_with_progress(search_df, async_runner, max_concurrent=5, enable_web_scraping=True):
    """
    Process companies with progress updates in the Streamlit UI
    
    Args:
        search_df: DataFrame containing company information
        async_runner: AsyncTaskRunner instance
        max_concurrent: Maximum number of concurrent requests
        
    Returns:
        List of results for each company
    """
    # Create a status container for the overall process
    status = st.status(f"Preparing to search for {len(search_df)} companies...", expanded=True)
    progress_placeholder = st.empty()
    company_progress = st.empty()
    
    # Initialize and display company statuses
    company_statuses = {idx: {"status": "pending", "name": row['company_name']} 
                        for idx, row in search_df.iterrows()}
    
    # Display initial progress
    status_text = "\n".join([
        f"⏳ {cs['name']}" for idx, cs in company_statuses.items()
    ])
    progress_placeholder.text(status_text)
    
    # Start the batch processing task
    batch_task = async_runner.run_task_async(
        batch_process_companies(search_df, max_concurrent, enable_web_scraping)
    )
    
    # Set up progress tracking
    start_time = time.time()
    total_companies = len(search_df)
    
    # Poll the task until it's done
    completed = False
    while not completed:
        # Check if the task is done
        if batch_task.done():
            try:
                # Get the results
                result = batch_task.result()
                all_results = result["results"]
                total_time = result["total_time"]
                
                # Sync token usage to session state
                sync_token_usage_to_session_state()
                
                # Update all company statuses
                for company_result in all_results:
                    idx = company_result["idx"]
                    company_name = company_result["company_name"]
                    success = company_result["success"]
                    
                    company_statuses[idx] = {
                        "status": "complete" if success else "error",
                        "name": company_name
                    }
                
                # Final status update
                status.update(
                    label=f"All searches complete! Processed {total_companies} companies in {format_time(total_time)}",
                    state="complete",
                    expanded=False
                )
                
                # Final progress display
                status_text = "\n".join([
                    f"{'✅' if cs['status'] == 'complete' else '❌' if cs['status'] == 'error' else '⏳'} {cs['name']}"
                    for idx, cs in company_statuses.items()
                ])
                progress_placeholder.text(status_text)
                
                completed = True
                return all_results
                
            except Exception as e:
                st.error(f"Error in batch processing: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                status.update(
                    label=f"Error: {str(e)}",
                    state="error"
                )
                completed = True
                return []
        
        # If not done, update the progress based on what we know
        elapsed = time.time() - start_time
        status.update(label=f"Processing {total_companies} companies in parallel... (Elapsed: {format_time(elapsed)})")
        company_progress.text(f"Please wait, companies are being processed in parallel...")
        
        # Sync token usage to session state every polling iteration
        if sync_token_usage_to_session_state():
            logger.info("Synced token usage data to session state")
            
        # Short sleep to avoid UI freezing
        time.sleep(1)

# ===================== FILE PROCESSING FUNCTIONS =====================
async def parse_excel_file(file_buffer, async_runner=None):
    """
    Parse Excel file to extract company names and additional context
    
    Args:
        file_buffer: File buffer of the uploaded Excel file
        async_runner: AsyncTaskRunner instance for running async tasks
        
    Returns:
        DataFrame with company names and additional context
    """
    try:
        df = pd.read_excel(file_buffer)
        
        # Look for column names containing company name or similar
        company_cols = [col for col in df.columns if any(re.search(rf'\b{name}\b', col.lower()) for name in 
                                                     ['company', 'organization', 'entity', 'name'])]
        
        # Initialize Gemini client for entity extraction if needed
        client = genai.Client(api_key=GEMINI_API_KEY)
        model_name = "gemini-2.0-flash"
        
        # Initialize result DataFrame
        result_df = pd.DataFrame()
        
        if company_cols:
            # If we found company name columns, use the first one
            company_col = company_cols[0]
            result_df['company_name'] = df[company_col]
            
            # Use all remaining columns as context
            cols_to_combine = [col for col in df.columns if col != company_col]
            if cols_to_combine:
                # Create a more structured context by including column names
                context_data = []
                for _, row in df.iterrows():
                    row_context = ", ".join([f"{col}: {row[col]}" for col in cols_to_combine if pd.notna(row[col])])
                    context_data.append(row_context)
                result_df['additional_context'] = context_data
            else:
                result_df['additional_context'] = ''
        else:
            # If no company name column was found, process each row to extract company names
            company_names = []
            contexts = []
            progress = []  # We'll collect progress info here and return it
            
            for idx, row in df.iterrows():
                row_info = ", ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                
                progress_msg = f"Analyzing row {idx+1}/{len(df)}..."
                logger.info(progress_msg)
                progress.append(progress_msg)
                
                # Use the entity extraction agent to process the row
                if async_runner:
                    extraction_result = async_runner.run_task(
                        entity_context_extraction_agent(client, model_name, row_info)
                    )
                else:
                    # Fallback to synchronous execution if no async runner
                    loop = asyncio.new_event_loop()
                    extraction_result = loop.run_until_complete(
                        entity_context_extraction_agent(client, model_name, row_info)
                    )
                    loop.close()
                
                # Get the first company name or use a placeholder
                company_name = extraction_result.get("entity_names", ["Unknown Company"])[0] if extraction_result.get("entity_names") else f"Row {idx+1}"
                additional_context = extraction_result.get("additional_context", "")
                
                company_names.append(company_name)
                contexts.append(additional_context if additional_context else row_info)  # Use original row info if no context extracted
                
                result_msg = f"Row {idx+1}: Identified '{company_name}'"
                logger.info(result_msg)
                progress.append(result_msg)
            
            # If no company names were found in any row, return error
            if not company_names or all(name.startswith("Row ") for name in company_names):
                error_msg = "Could not identify any company names in the uploaded file. Please try a file with clearer company information."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Create the result dataframe
            result_df['company_name'] = company_names
            result_df['additional_context'] = contexts
            result_df['raw_data'] = [", ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]) for idx, row in df.iterrows()]
        
        # Add status column
        result_df['status'] = 'Pending'
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in parse_excel_file: {str(e)}")
        raise  # Propagate the error up to be handled by the caller

async def process_uploaded_file(uploaded_file, async_runner):
    """Process the uploaded file and return a dataframe ready for the data editor"""
    try:
        # Create file buffer
        if uploaded_file.name.endswith('.csv'):
            file_buffer = io.BytesIO(uploaded_file.read())
            file_buffer.seek(0)
        else:
            file_buffer = io.BytesIO(uploaded_file.read())
            file_buffer.seek(0)
        
        # Process the Excel file with our advanced parsing
        editor_df = await parse_excel_file(file_buffer, async_runner)
        return {"success": True, "data": editor_df, "error": None}
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "data": None, "error": str(e)}

# ===================== UTILITY FUNCTIONS =====================
def display_token_usage(inside_expander=False):
    """Display token usage information in Streamlit
    
    Args:
        inside_expander: Boolean indicating if function is called from inside an expander
    """
    if 'token_usage' not in st.session_state or not st.session_state.token_usage:
        st.info("No token usage data available yet.")
        return
    
    # Calculate totals by agent
    agent_totals = {}
    for usage in st.session_state.token_usage:
        if usage.agent_name not in agent_totals:
            agent_totals[usage.agent_name] = 0
        agent_totals[usage.agent_name] += usage.token_count
    
    # Calculate grand total
    grand_total = sum(usage.token_count for usage in st.session_state.token_usage)
    
    # Create a DataFrame for display
    data = []
    for agent, total in agent_totals.items():
        data.append({"Agent": agent, "Tokens Used": total})
    
    # Add grand total
    data.append({"Agent": "TOTAL", "Tokens Used": grand_total})
    
    df = pd.DataFrame(data)
    
    # Display as a table
    st.subheader("Token Usage Summary")
    st.dataframe(
        df,
        column_config={
            "Agent": st.column_config.TextColumn("Agent"),
            "Tokens Used": st.column_config.NumberColumn(
                "Tokens Used",
                format="%d",
                help="Number of tokens used by this agent"
            )
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Add detailed log based on whether we're already in an expander
    if inside_expander:
        # Just show the detailed log without wrapping in another expander
        st.subheader("Detailed Token Log")
        token_df = pd.DataFrame([
            {
                "Timestamp": usage.timestamp.strftime("%H:%M:%S"),
                "Agent": usage.agent_name,
                "Tokens": usage.token_count
            }
            for usage in st.session_state.token_usage
        ])
        
        st.dataframe(
            token_df,
            column_config={
                "Timestamp": st.column_config.TextColumn("Time"),
                "Agent": st.column_config.TextColumn("Agent"),
                "Tokens": st.column_config.NumberColumn("Tokens Used", format="%d")
            },
            hide_index=True
        )
    else:
        # Use an expander since we're not already in one
        with st.expander("View Detailed Token Log"):
            # Display token usage information in table format
            token_df = pd.DataFrame([
                {
                    "Timestamp": usage.timestamp.strftime("%H:%M:%S"),
                    "Agent": usage.agent_name,
                    "Tokens": usage.token_count
                }
                for usage in st.session_state.token_usage
            ])
            
            st.dataframe(
                token_df,
                column_config={
                    "Timestamp": st.column_config.TextColumn("Time"),
                    "Agent": st.column_config.TextColumn("Agent"),
                    "Tokens": st.column_config.NumberColumn("Tokens Used", format="%d")
                },
                hide_index=True
            )
    
    # Add an explanation about costs
    st.write(f"""
    **Cost Estimate:**
    - Total tokens used: {grand_total:,}
    - Input cost (@ $0.10/million tokens): ${(grand_total / 1000000) * 0.10:.6f}
    - Output cost (@ $0.40/million tokens): ${(grand_total / 1000000) * 0.40:.6f}
    - Estimated total cost: ${(grand_total / 1000000) * 0.50:.6f}
    """)
    
    # Note about LinkUp costs
    st.caption(
        "Note: This estimate does not include LinkUp API costs, which are charged separately based on your LinkUp subscription."
    )
    
    # Optionally, add a visualization
    if len(data) > 1:  # Only show chart if we have data beside the total
        st.subheader("Token Usage by Agent")
        chart_df = df[df["Agent"] != "TOTAL"]  # Remove total for better visualization
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("Agent"))

def save_to_search_history(company_names, search_results, verified_results=None, scraped_urls=None, failed_urls=None):
    """Save search results to history in session state with enhanced data"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a history entry with essential information
    history_entry = {
        "timestamp": timestamp,
        "companies": company_names,
        "search_results": search_results,
        "verified_results": verified_results if verified_results else [],
        "scraped_urls": scraped_urls if scraped_urls else [],
        "failed_urls": failed_urls if failed_urls else [],
        "id": str(hash(timestamp + str(company_names)))  # Create a unique ID
    }
    
    # Add to the beginning of the list (most recent first)
    st.session_state.search_history.insert(0, history_entry)
    
    # Limit history size (optional)
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]
        
def export_search_history():
    """Export search history as JSON"""
    if not st.session_state.search_history:
        st.warning("No search history to export.")
        return
    
    # Convert history to JSON
    history_json = json.dumps(st.session_state.search_history, default=str)
    
    # Create a download button
    st.download_button(
        label="Download Search History",
        data=history_json,
        file_name="linkup_search_history.json",
        mime="application/json"
    )

def import_search_history(uploaded_file):
    """Import search history from JSON file"""
    try:
        imported_history = json.loads(uploaded_file.getvalue().decode())
        
        # Validate the imported data (basic check)
        if not isinstance(imported_history, list):
            st.error("Invalid history file format.")
            return False
        
        # Merge with existing history (avoid duplicates)
        existing_ids = {entry.get("id") for entry in st.session_state.search_history}
        
        for entry in imported_history:
            if entry.get("id") not in existing_ids:
                st.session_state.search_history.append(entry)
                existing_ids.add(entry.get("id"))
        
        st.success(f"Successfully imported {len(imported_history)} search entries.")
        return True
    except Exception as e:
        st.error(f"Error importing history: {str(e)}")
        return False

async def enhance_context_with_web_scraping(additional_context: str, max_concurrent: int = 3) -> Dict[str, Any]:
    """
    Extract URLs from additional context, scrape their content, and add it to the context
    
    Args:
        additional_context: Original additional context string containing URLs
        max_concurrent: Maximum number of concurrent scraping tasks
        
    Returns:
        Dictionary with enhanced context and metadata
    """
    # Extract URLs from the context
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    urls = re.findall(url_pattern, additional_context)
    
    if not urls:
        return {
            "enhanced_context": additional_context,
            "scraped_urls": [],
            "failed_urls": []
        }
    
    # Initialize the scraper
    scraper_config = ScraperConfig(
        request_delay=2.0,
        user_agent="CompanyResearchAssistant/1.0",
        respect_robots=True,
        cache_enabled=True,
        max_concurrent=max_concurrent,
        max_redirect_depth=5
    )
    
    # Track scraped and failed URLs
    scraped_urls = []
    failed_urls = []
    enhanced_context = additional_context
    
    try:
        async with EthicalWebScraper(scraper_config) as scraper:
            # Create tasks for scraping each URL
            scrape_tasks = []
            
            for url in urls:
                # Skip notion.so URLs - they require JavaScript rendering
                if 'notion.so' in url:
                    failed_urls.append({
                        "url": url,
                        "reason": "Authentication required",
                        "domain": "notion.so"
                    })
                    continue
                    
                # Parse the domain
                domain = urlparse(url).netloc
                
                # Check if this domain has already been scraped
                if domain in [parsed_url["domain"] for parsed_url in scraped_urls + failed_urls]:
                    logger.info(f"Skipping already processed domain: {domain}")
                    continue
                    
                # Create a task for scraping
                scrape_tasks.append((url, scraper.scrape_page(url)))
            
            # Execute all scraping tasks concurrently with semaphore
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def scrape_with_semaphore(url, task):
                async with semaphore:
                    try:
                        result = await task
                        return url, result
                    except Exception as e:
                        logger.warning(f"Exception scraping {url}: {str(e)}")
                        return url, None
                    
            concurrent_tasks = [scrape_with_semaphore(url, task) for url, task in scrape_tasks]
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Failed to scrape: {str(result)}")
                    continue
                    
                url, content = result
                domain = urlparse(url).netloc
                
                if content is None:
                    logger.warning(f"No content scraped from {url}")
                    failed_urls.append({
                        "url": url,
                        "reason": "Failed to retrieve content",
                        "domain": domain
                    })
                else:
                    # Extract relevant content
                    text = content.get('text', '')
                    title = content.get('title', '')
                    
                    if text:
                        # Truncate long content (limit to ~1000 words)
                        words = text.split()
                        truncated = len(words) > 1000
                        if truncated:
                            text = ' '.join(words[:1000]) + ' [truncated due to length]'
                        
                        # Store scraped content info
                        scraped_urls.append({
                            "url": url,
                            "domain": domain,
                            "title": title,
                            "truncated": truncated,
                            "content_length": len(words)
                        })
                        
                        # Add content to enhanced context
                        content_marker = f"""
[Content from {url}]:
Title: {title}
{text}
[End of content]
"""
                        enhanced_context = enhanced_context.replace(url, content_marker)
                    else:
                        failed_urls.append({
                            "url": url,
                            "reason": "No text content found",
                            "domain": domain
                        })
            
            # Add a section about failed URLs
            if failed_urls:
                enhanced_context += "\n\n--- URLS THAT COULD NOT BE SCRAPED ---\n"
                for failed in failed_urls:
                    reason = failed["reason"]
                    url = failed["url"]
                    enhanced_context += f"\n• {url} - Reason: {reason}"
    
    except Exception as e:
        logger.error(f"Error during URL scraping: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # On error, return original context
        return {
            "enhanced_context": additional_context,
            "scraped_urls": scraped_urls,
            "failed_urls": failed_urls,
            "error": str(e)
        }
    
    return {
        "enhanced_context": enhanced_context,
        "scraped_urls": scraped_urls,
        "failed_urls": failed_urls
    }

# ============================ FIRST PASS ============================
async def first_pass_entity_selection(user_input: str, observability_log: List[Dict]) -> Dict[str, Any]:
    """
    First pass: Initial entity extraction and context gathering using sourcedAnswer.
    Logs activity to observability_log.
    
    Args:
        user_input: Raw user query text
        observability_log: List to store process logs
        
    Returns:
        Dictionary with entity information and selection status
    """
    start_time = time.time()
    log_entry = {"step": "Pass 1", "input_query": user_input}
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        model_name = "gemini-2.0-flash"

        # 1a. Gemini: Extract potential entities
        extraction_result = await entity_context_extraction_agent(
            client, model_name, f"User input: {user_input}"
        )
        potential_entities = extraction_result.get("entity_names", [])
        initial_context = extraction_result.get("additional_context", "")
        log_entry["gemini_entities"] = potential_entities
        log_entry["gemini_context"] = initial_context

        if not potential_entities:
            log_entry["status"] = "Failed - No entities identified"
            log_entry["duration"] = time.time() - start_time
            observability_log.append(log_entry)
            return {
                "success": False,
                "error": "No entities identified in input text",
                "entities": [],
                "selected_entity": None,
                "negative_entities": [],
                "additional_context": initial_context
            }

        # 1b. LinkUp: Basic search for each entity using sourcedAnswer
        entity_info = []
        linkup_tasks = []
        disambiguation_prompt = f"""
        Provide a brief description and key identifying information for the company: {{entity_name}}
        
        Focus on:
        1. What the company does (main business/products)
        2. When it was founded and where it's headquartered
        3. Industry sector
        4. Company size if available
        5. Official website URL
        
        Use reliable sources and verify the information relates to the correct company.
        """

        for entity in potential_entities:
            prompt = disambiguation_prompt.format(entity_name=entity)
            # Pass the prompt via additional_context argument
            linkup_tasks.append(company_search_with_linkup(entity, additional_context=prompt))

        search_results = await asyncio.gather(*linkup_tasks, return_exceptions=True)
        
        logger.info(f"Pass 1 LinkUp search results: {search_results}")

        log_entry["linkup_calls"] = []
        for i, result in enumerate(search_results):
            entity_name = potential_entities[i]
            linkup_log = {
                "entity": entity_name, 
                "prompt": disambiguation_prompt.format(entity_name=entity_name)[:200] + "..."
            }
            
            if isinstance(result, Exception):
                linkup_log["status"] = "Error"
                linkup_log["error"] = str(result)
                logger.error(f"Error in Pass 1 LinkUp search for {entity_name}: {result}")
                entity_info.append({"name": entity_name, "snippet": "Error fetching info", "sources": []})
            elif result.get("success", False):
                linkup_log["status"] = "Success"
                linkup_log["answer"] = result.get("answer", "")[:100] + "..." if result.get("answer") else "" # Log truncated answer
                linkup_log["num_sources"] = len(result.get("sources", []))
                # Use answer for snippet, sources for disambiguation
                entity_info.append({
                    "name": entity_name,
                    "snippet": result.get("answer", "No description available."),
                    "sources": result.get("sources", []),
                    "search_text_formatted": result.get("search_text_formatted", ""),
                    "raw_response": result.get("raw_response", None)
                })
            else:
                linkup_log["status"] = "Failed"
                linkup_log["error"] = result.get("error", "Unknown LinkUp error")
                entity_info.append({"name": entity_name, "snippet": f"Failed: {result.get('error')}", "sources": []})
            
            log_entry["linkup_calls"].append(linkup_log)

        log_entry["status"] = "Success"
        log_entry["duration"] = time.time() - start_time
        observability_log.append(log_entry)
        return {
            "success": True,
            "entities": entity_info, # Contains name, snippet (answer), sources
            "selected_entity": None,
            "negative_entities": [],
            "additional_context": initial_context
        }

    except Exception as e:
        logger.error(f"Error in first_pass_entity_selection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        log_entry["status"] = "Failed - Exception"
        log_entry["error"] = str(e)
        log_entry["duration"] = time.time() - start_time
        observability_log.append(log_entry)
        return {
            "success": False,
            "error": str(e),
            "entities": [],
            "selected_entity": None,
            "negative_entities": [],
            "additional_context": ""
        }

# ============================ SECOND PASS ============================
async def second_pass_structured_search(selected_entity: Dict[str, Any],
                                       negative_entities: List[Dict[str, Any]],
                                       additional_context: str,
                                       observability_log: List[Dict]) -> Dict[str, Any]:
    """
    Second pass: Detailed structured search using sourcedAnswer with schema.
    
    Args:
        selected_entity: The target entity information
        negative_entities: Entities to exclude from search
        additional_context: Additional context about the company
        observability_log: List to store process logs
        
    Returns:
        Dictionary with search results in multiple formats
    """
    start_time = time.time()
    log_entry = {"step": "Pass 2", "target_entity": selected_entity.get("name")}
    try:
        # Construct the structured prompt with context
        positive_context = f"**TARGET ENTITY:** {selected_entity['name']}\n"
        if selected_entity.get('sources'):
            source_urls = [safe_get_attribute(s, 'url', None) for s in selected_entity['sources'] 
              if safe_get_attribute(s, 'url', None)][:5]
            if source_urls:
                positive_context += "Key URLs (verify against these):\n" + "\n".join([f"- {url}" for url in source_urls]) + "\n"
        
        if additional_context:
            positive_context += f"\n**Additional Context:** {additional_context}"

        negative_context_str = "**ENTITIES TO IGNORE:**\n"
        if negative_entities:
            negative_context_str += "\n".join([f"- {safe_get_attribute(neg, 'name', 'Unknown')}" for neg in negative_entities])
        else:
            negative_context_str += "N/A"

        schema_json_string = json.dumps(LINKUP_COMPANY_SCHEMA, indent=2)
        structured_prompt_pass2 = f"""
        Perform comprehensive research analysis for the TARGET ENTITY, generating JSON output matching the schema. Ensure information pertains ONLY to the TARGET.

        {positive_context}

        {negative_context_str}

        **RESEARCH SCOPE & JSON SCHEMA:**
        - Adhere strictly to the JSON schema.
        - Fill fields based *only* on the TARGET ENTITY. Use info from its key URLs if possible.
        - Prioritize official sources, recent news (last 6 months).
        - Use JSON null/empty values (`[]`, `""`, `null`) if info for TARGET is not found. Do NOT use data from IGNORED entities.
        - Output the result as a JSON block: ```json ... ```

        **JSON Schema Definition:**
        ```json
        {schema_json_string}
        ```
        """
        log_entry["input_prompt"] = structured_prompt_pass2[:500] + "..." # Log truncated prompt

        # Perform the search using revised function
        search_result = await company_search_with_linkup(
            selected_entity['name'],
            additional_context=structured_prompt_pass2, # Pass the full prompt here
            schema_str=schema_json_string # Hint that schema is involved
        )

        # For debugging
        if search_result.get("raw_response"):
            test_linkup_response_structure(search_result["raw_response"])

        log_entry["linkup_output_type"] = "sourcedAnswer"
        log_entry["linkup_answer"] = search_result.get("answer", "")[:200] + "..." if search_result.get("answer") else ""
        log_entry["linkup_num_sources"] = len(search_result.get("sources", []))
        log_entry["linkup_parsing_error"] = search_result.get("parsing_error")
        log_entry["status"] = "Success" if search_result.get("success", False) else "Failed"
        log_entry["error"] = search_result.get("error")

        log_entry["duration"] = time.time() - start_time
        observability_log.append(log_entry)
        # Toward the end of the function, add this line:
        search_results = search_result.get("search_text_formatted", "")
        
        # Return the richer result dictionary
        return {
            "success": search_result.get("success", False),
            "answer": search_result.get("answer", ""),
            "sources": search_result.get("sources", []),
            "structured_data": search_result.get("structured_data"), # Parsed JSON
            "search_results": search_results,  # Add this key with the correct value
            "search_text_formatted": search_result.get("search_text_formatted", ""),
            "error": search_result.get("error"),
            "parsing_error": search_result.get("parsing_error"),
            "raw_response": search_result.get("raw_response")
        }

    except Exception as e:
        logger.error(f"Error in second_pass_structured_search: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        log_entry["status"] = "Failed - Exception"
        log_entry["error"] = str(e)
        log_entry["duration"] = time.time() - start_time
        observability_log.append(log_entry)
        return {
            "success": False,
            "error": str(e),
            "answer": "",
            "sources": [],
            "structured_data": None,
            "search_results": f"Error in detailed search: {str(e)}",  # Add this key
            "search_text_formatted": f"Error in detailed search: {str(e)}",
            "parsing_error": None,
            "raw_response": None
        }

# ============================ THIRD PASS ============================
async def analyze_missing_fields(client, model_name, structured_data):
    """
    Analyze structured data for missing fields and suggest queries.
    
    Args:
        client: Gemini API client
        model_name: Model name to use
        structured_data: Current structured data to analyze
        
    Returns:
        Dictionary with missing fields analysis
    """
    analysis_prompt = f"""
    Analyze the following company data and identify important missing fields:
    
    ```json
    {json.dumps(structured_data, indent=2, default=str)}
    ```
    
    For each missing or empty field that's important, provide:
    1. The field name (use exact dot notation for nested fields)
    2. Importance level (high, medium, low)
    3. A specific targeted query to find that information
    
    Return ONLY a JSON object with this structure:
    {{
        "missing_fields": [
            {{
                "field_name": "products",
                "importance": "high",
                "query_suggestion": "What are the main products and services of [company]?"
            }},
            ...more fields...
        ]
    }}
    
    Prioritize missing high-importance fields like company_description, headquarters, industry, 
    leadership_team, products, and recent_news if they're empty.
    """
    
    try:
        response = await retry_api_call(
            client.aio.models.generate_content,
            model=model_name,
            contents=[analysis_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        # Parse the response as JSON
        if response.candidates and response.candidates[0].content.parts:
            result = json.loads(response.candidates[0].content.parts[0].text)
            logger.info(f"Successfully analyzed missing fields, found {len(result.get('missing_fields', []))} important fields")
            return result
        else:
            logger.warning("Empty response from missing fields analysis")
            return {"missing_fields": []}
            
    except Exception as e:
        logger.error(f"Error analyzing missing fields: {str(e)}")
        return {"missing_fields": []}

async def third_pass_fill_missing_info(structured_data: Optional[Dict[str, Any]],
                                       selected_entity: Dict[str, Any],
                                       negative_entities: List[Dict[str, Any]],
                                       pass2_sources: List[Dict], 
                                       observability_log: List[Dict]) -> Dict[str, Any]:
    """
    Third pass: Fill missing information using targeted sourcedAnswer queries.
    Logs activity to observability_log.
    """
    start_time = time.time()
    log_entry = {"step": "Pass 3", "target_entity": selected_entity.get("name")}

    if not structured_data:
        log_entry["status"] = "Skipped - No data from Pass 2"
        log_entry["duration"] = 0
        observability_log.append(log_entry)
        return {"success": True, "structured_data": {}, "enhanced": False, "enhanced_fields": []} 

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        model_name = "gemini-2.0-flash"
        enhanced_data = copy.deepcopy(structured_data)
        enhanced_fields = []
        all_pass3_linkup_calls = []

        # 1. Gemini: Identify missing fields
        missing_fields_analysis = await analyze_missing_fields(client, model_name, structured_data)
        missing_fields = missing_fields_analysis.get("missing_fields", [])
        high_importance_fields = [f for f in missing_fields if f.get("importance") in ["high", "medium"]][:3] # Limit calls
        log_entry["gemini_missing_fields"] = high_importance_fields

        if not high_importance_fields:
            log_entry["status"] = "Success - No important fields missing"
            log_entry["duration"] = time.time() - start_time
            observability_log.append(log_entry)
            return {
                "success": True, 
                "structured_data": structured_data, 
                "enhanced": False,
                "enhanced_fields": []
            }

        # 2. LinkUp: Targeted searches for each missing field using sourcedAnswer
        linkup_tasks = []
        context_str = f"""
        **TARGET ENTITY (Focus ONLY on this one):**
        {selected_entity['name']} (Key URLs: {', '.join([s.get('url', 'N/A') for s in selected_entity.get('sources',[])[:3]])})

        **ENTITIES TO IGNORE:**
        {', '.join([neg.get('name', 'N/A') for neg in negative_entities]) if negative_entities else 'N/A'}
        """

        for field_info in high_importance_fields:
            field_path = field_info.get("field_name", "")
            query_suggestion = field_info.get("query_suggestion", "")
            # Ensure the query asks for source verification
            targeted_query = f"""
            Regarding *only* the TARGET ENTITY, find information for: '{query_suggestion}'?

            {context_str}

            Provide the answer based *only* on verifiable sources related to the TARGET ENTITY.
            State the answer clearly. Include supporting source snippets and URLs.
            If unknown, state 'Unknown based on available sources for the target'.
            """
            # Pass the full query via additional_context
            linkup_tasks.append((field_path, company_search_with_linkup(selected_entity.get('name', 'Unknown'), additional_context=targeted_query)))

        # Execute targeted searches
        targeted_results = await asyncio.gather(*[task[1] for task in linkup_tasks], return_exceptions=True)

        # 3. Analyze results and prepare for integration
        combined_pass3_evidence = "" # Collect text answers and sources for Gemini

        for i, result in enumerate(targeted_results):
            field_path = linkup_tasks[i][0]
            
            linkup_log = {"field": field_path}

            if isinstance(result, Exception):
                logger.error(f"Pass 3 LinkUp search for field {field_path} failed: {result}")
                linkup_log["status"] = "Error"
                linkup_log["error"] = str(result)
            elif result.get("success", False) and result.get("answer"):
                linkup_log["status"] = "Success"
                linkup_log["answer"] = result.get("answer", "")[:100] + "..." if result.get("answer") else ""
                linkup_log["num_sources"] = len(result.get("sources", []))

                # Add result to evidence for Gemini, clearly marking field and sources
                combined_pass3_evidence += f"\n\n--- Evidence for field: {field_path} ---\n"
                combined_pass3_evidence += f"Answer Found: {result.get('answer', '')}\n"
                if result.get('sources'):
                    combined_pass3_evidence += "Supporting Sources:\n" + "\n".join([
                        f"- {s.get('url', 'N/A')} ({s.get('snippet', '')[:80]}...)" 
                        for s in result.get('sources', [])
                    ])
                else:
                    combined_pass3_evidence += "No specific sources returned for this answer.\n"
            else:
                linkup_log["status"] = "Failed / No Answer"
                linkup_log["error"] = result.get("error", "Unknown error")
                linkup_log["answer"] = result.get("answer", "")
            all_pass3_linkup_calls.append(linkup_log)

        log_entry["pass3_linkup_calls"] = all_pass3_linkup_calls

        # If we gathered any evidence, ask Gemini to integrate it
        if combined_pass3_evidence:
            integration_prompt = f"""
            Existing company data (potentially incomplete):
            ```json
            {json.dumps(structured_data, indent=2, default=str)}
            ```

            New evidence found for specific fields:
            {combined_pass3_evidence}

            Based *only* on the new evidence provided above, identify updates for the existing JSON data.
            *   Only update fields if the evidence provides a clear, verifiable answer for the TARGET company.
            *   Verify the source URLs seem relevant to the target entity if possible.
            *   Format the updates strictly as a JSON object: {{ "field_path1": "new_value", "field_path2": [...] }}
            *   If evidence confirms a field is empty/null for the target, represent that appropriately (e.g., `[]`, `null`, `""`).
            *   If evidence is insufficient or contradictory for a field, do not include it in the update JSON.

            Return ONLY the JSON object containing the updates.
            """
            log_entry["gemini_integration_prompt"] = integration_prompt[:500] + "..."

            update_response = await retry_api_call(
                client.aio.models.generate_content,
                model=model_name, 
                contents=[integration_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            update_response_text = update_response.candidates[0].content.parts[0].text
            log_entry["gemini_integration_response"] = update_response_text
            try:
                updates = json.loads(update_response_text)
                log_entry["gemini_parsed_updates"] = updates
                if updates:
                    # Apply updates
                    for path, value in updates.items():
                        try:
                            apply_nested_update(enhanced_data, path, value)
                            enhanced_fields.append(path)
                            logger.info(f"Pass 3: Applied update for {path}")
                        except Exception as apply_err:
                            logger.error(f"Pass 3: Failed to apply update for {path}: {apply_err}")
                            log_entry["integration_errors"] = log_entry.get("integration_errors", []) + [f"Apply error for {path}: {apply_err}"]

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON updates from Gemini in Pass 3: {e}")
                log_entry["gemini_parsing_error"] = str(e)
        else:
            log_entry["gemini_integration_prompt"] = "No new evidence gathered, skipping integration."

        log_entry["enhanced_fields"] = enhanced_fields
        log_entry["status"] = "Success"
        log_entry["duration"] = time.time() - start_time
        observability_log.append(log_entry)

        return {
            "success": True,
            "structured_data": enhanced_data,
            "enhanced": len(enhanced_fields) > 0,
            "enhanced_fields": enhanced_fields
        }

    except Exception as e:
        logger.error(f"Error in third_pass_fill_missing_info: {str(e)}")
        log_entry["status"] = "Failed - Exception"
        log_entry["error"] = str(e)
        log_entry["duration"] = time.time() - start_time
        observability_log.append(log_entry)
        return { 
            "success": False, 
            "error": str(e), 
            "structured_data": structured_data, 
            "enhanced": False,
            "enhanced_fields": []
        }

def apply_nested_update(data_dict, path_str, value):
    """
    Applies an update to a nested dictionary using a dot-separated path.
    
    Args:
        data_dict: The dictionary to update
        path_str: Dot-separated path to the field (e.g. "headquarters.location")
        value: Value to set at the path
        
    Raises:
        KeyError: If an intermediate key doesn't exist
        IndexError: If an array index is out of bounds
    """
    keys = path_str.split('.')
    current = data_dict
    
    # Navigate through the nested structure except for the last key
    for i, key in enumerate(keys[:-1]):
        # Handle array indices
        if key.isdigit() and isinstance(current, list):
            key = int(key)
            # Extend list if needed
            while len(current) <= key:
                current.append({})
            current = current[key]
        # Handle array indices with bracket notation (e.g., "array[0]")
        elif '[' in key and key.endswith(']'):
            base_key, idx_str = key.split('[', 1)
            idx = int(idx_str.rstrip(']'))
            
            # Create the key if it doesn't exist
            if base_key not in current:
                current[base_key] = []
            
            # Ensure list is long enough
            while len(current[base_key]) <= idx:
                current[base_key].append({})
            
            current = current[base_key][idx]
        # Regular dict key
        else:
            if key not in current:
                current[key] = {}
            current = current[key]
    
    # Set the value at the final location
    final_key = keys[-1]
    
    # Handle array indices in the final key
    if final_key.isdigit() and isinstance(current, list):
        final_key = int(final_key)
        # Extend list if needed
        while len(current) <= final_key:
            current.append(None)
        current[final_key] = value
    # Handle array indices with bracket notation
    elif '[' in final_key and final_key.endswith(']'):
        base_key, idx_str = final_key.split('[', 1)
        idx = int(idx_str.rstrip(']'))
        
        # Create the key if it doesn't exist
        if base_key not in current:
            current[base_key] = []
        
        # Ensure list is long enough
        while len(current[base_key]) <= idx:
            current[base_key].append(None)
        
        current[base_key][idx] = value
    # Regular dict key
    else:
        current[final_key] = value

def first_pass_ui_with_data_editor():
    """
    Enhanced first pass UI implementation using st.data_editor for selection
    with proper handling of both dictionary and object attributes.
    """
    st.subheader("First Pass: Entity Selection")
    st.write("Please select the correct entity and mark any irrelevant entities:")

    # Add this right after the function declaration
    st.warning("Running with enhanced debugging")

    # Check first entity's sources directly
    if st.session_state.first_pass_result and st.session_state.first_pass_result["entities"]:
        first_entity = st.session_state.first_pass_result["entities"][0]
        
        # Direct attribute check
        has_sources_direct = hasattr(first_entity, "sources")
        st.write(f"First entity has 'sources' attribute: {has_sources_direct}")
        
        if has_sources_direct:
            sources = first_entity.sources
            st.write(f"Number of sources: {len(sources)}")
            
            # Display raw sources info
            st.write("Source preview:")
            for i, source in enumerate(sources[:2]):  # First 2 sources
                st.write(f"Source {i+1} type: {type(source).__name__}")
                
                # Try different access methods
                name_attr = getattr(source, "name", None) if hasattr(source, "name") else "No name attribute"
                url_attr = getattr(source, "url", None) if hasattr(source, "url") else "No url attribute"
                
                st.write(f"- Name (attr): {name_attr}")
                st.write(f"- URL (attr): {url_attr}")
                
                if url_attr:
                    st.write(f"- Domain: {urlparse(url_attr).netloc}")

    
    # Debug source access patterns (can be removed after verification)
    with st.expander("Debug Entity Data Structure"):
        st.write("This section helps diagnose source data issues")
        for i, entity in enumerate(st.session_state.first_pass_result["entities"]):
            st.write(f"**Entity {i+1}: {safe_get_attribute(entity, 'name', 'Unknown')}**")
            
            # Check sources existence
            has_sources_attr = hasattr(entity, "sources")
            has_sources_key = isinstance(entity, dict) and "sources" in entity
            
            st.write(f"- Has 'sources' attribute: {has_sources_attr}")
            st.write(f"- Has 'sources' key: {has_sources_key}")
            
            # Get sources using appropriate method
            sources = safe_get_attribute(entity, "sources", [])
            st.write(f"- Number of sources: {len(sources) if sources else 0}")
            
            # Check first source structure if available
            if sources and len(sources) > 0:
                first_source = sources[0]
                st.write(f"- First source type: {type(first_source).__name__}")
                
                # Test URL access
                url = safe_get_attribute(first_source, "url", None)
                st.write(f"- First source URL: {url if url else 'Not available'}")
    
    entity_options = st.session_state.first_pass_result["entities"]
    
    if not entity_options:
        st.warning("No entities were found. Please try again with more specific information.")
        if st.button("Start Over"):
            st.session_state.research_phase = 'input'
            st.rerun()
        st.stop()
    
    # Store the full entity data in session state for reference
    if 'entity_full_data' not in st.session_state:
        st.session_state.entity_full_data = entity_options
    
    # Create a DataFrame for the data editor
    data = []
    for i, entity in enumerate(entity_options):
        # Create snippet preview (truncated to avoid overly wide display)
        snippet_preview = safe_get_attribute(entity, "snippet", "No description available.")
        if len(snippet_preview) > 5000:
            snippet_preview = snippet_preview[:5000] + "..."
        
        # Get source URLs if available
        source_urls = []
        source_domains = []
        
        # Access sources consistently using safe_get_attribute
        sources = safe_get_attribute(entity, "sources", [])
        
        if sources:
            for source in sources[:3]:  # Limit to 3 sources
                url = safe_get_attribute(source, "url", None)
                if url:
                    source_urls.append(url)
                    domain = urlparse(url).netloc
                    source_domains.append(domain)
        
        # Format domains as a string
        domains_str = ", ".join(source_domains) if source_domains else "No sources"
        
        # Create row for the DataFrame - without hidden columns
        data.append({
            "Select Target": False,  # Default unselected
            "Mark Irrelevant": False,  # Default unselected
            "Company Name": safe_get_attribute(entity, "name", f"Entity {i+1}"),
            "Description": snippet_preview,
            "Sources": domains_str,
            "Index": i  # Make index visible instead of hidden
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Configure the data editor - without using 'hidden' parameter
    edited_df = st.data_editor(
        df,
        column_config={
            "Select Target": st.column_config.CheckboxColumn(
                "Select as Target",
                help="Check to select this as the target entity",
                default=False,
                width="small"
            ),
            "Mark Irrelevant": st.column_config.CheckboxColumn(
                "Mark as Irrelevant",
                help="Check to mark this entity as irrelevant",
                default=False,
                width="small"
            ),
            "Company Name": st.column_config.TextColumn(
                "Company Name",
                help="Name of the entity",
                width="small"
            ),
            "Description": st.column_config.TextColumn(
                "Description",
                help="Brief description of the entity",
                width="large",
            ),
            "Sources": st.column_config.TextColumn(
                "Source Domains",
                help="Source domains for this entity",
                width="medium",
            ),
            "Index": st.column_config.NumberColumn(
                "Index",
                help="Reference number",
                width="small"
            )
        },
        hide_index=True,
        use_container_width=True,
        key="entity_selection_editor",
        disabled=["Index", "Company Name", "Description", "Sources"]  # Make these columns read-only
    )
    
    # Add a preview section for the selected row
    st.divider()
    st.subheader("Entity Preview")
    
    # Create selection for which entity to preview
    preview_options = [f"{i+1}: {safe_get_attribute(entity, 'name', f'Entity {i+1}')}" 
                      for i, entity in enumerate(entity_options)]
    selected_preview = st.selectbox("Select entity to preview:", preview_options)
    
    # Get the index of the selected preview
    selected_preview_idx = int(selected_preview.split(":")[0]) - 1
    
    # Display the full preview for the selected entity
    preview_entity = entity_options[selected_preview_idx]
    
    st.write(f"### {safe_get_attribute(preview_entity, 'name', 'Unknown Entity')}")
    st.markdown(safe_get_attribute(preview_entity, 'snippet', 'No description available.'))
    
    # Display sources if available using safe attribute access
    sources = safe_get_attribute(preview_entity, "sources", [])
    if sources:
        st.write("#### Sources:")
        for source in sources[:5]:  # Limit to 5 sources
            url = safe_get_attribute(source, "url", None)
            if url:
                name = safe_get_attribute(source, "name", "Source")
                domain = urlparse(url).netloc
                st.write(f"- [{domain}]({url})")
                
                # Optionally show snippet for context
                snippet = safe_get_attribute(source, "snippet", "")
                if snippet and len(snippet) > 0:
                    with st.expander(f"Preview from {domain}"):
                        st.write(snippet[:300] + "..." if len(snippet) > 300 else snippet)
    else:
        st.info("No sources available for this entity.")
    
    # Process the edited DataFrame
    st.divider()
    st.subheader("Selection Summary")
    
    selected_entity = None
    negative_entities = []
    
    for _, row in edited_df.iterrows():
        idx = int(row["Index"])
        entity = entity_options[idx]
        
        # Check for selection as target
        if row["Select Target"]:
            selected_entity = entity
        
        # Check for marking as irrelevant
        if row["Mark Irrelevant"]:
            negative_entities.append(entity)
    
    # Validation and warning for conflicting selections
    if selected_entity is not None and selected_entity in negative_entities:
        st.warning("You've marked the same entity as both target and irrelevant. Please choose one option.")
        # Remove from negative if it's selected as target
        if selected_entity in negative_entities:
            negative_entities.remove(selected_entity)
    
    # Show current selection status
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Selected Target Entity:**")
        if selected_entity:
            st.success(f"**{safe_get_attribute(selected_entity, 'name', 'Unknown Entity')}**")
            # Save to session state immediately
            st.session_state.selected_entity = selected_entity
        else:
            st.info("No entity selected as target yet")
    
    with col2:
        st.write("**Marked as Irrelevant:**")
        if negative_entities:
            for neg_entity in negative_entities:
                st.warning(f"**{safe_get_attribute(neg_entity, 'name', 'Unknown Entity')}**")
            # Save to session state immediately
            st.session_state.negative_entities = negative_entities
        else:
            st.info("No entities marked as irrelevant")
    
    # Additional context input
    st.divider()
    additional_context = st.text_area(
        "Additional context about the selected entity (optional):",
        value=safe_get_attribute(st.session_state.first_pass_result, "additional_context", ""),
        placeholder="E.g., Founded in 2020, AI-based company in healthcare sector, etc."
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Continue button (only enabled if an entity is selected)
        continue_disabled = selected_entity is None
        if st.button("Continue to Second Pass", disabled=continue_disabled):
            # Save additional context
            st.session_state.additional_context = additional_context
            # Update phase
            st.session_state.research_phase = 'detailed_search'
            st.rerun()
        
        if continue_disabled:
            st.warning("Please select a target entity to continue")
    
    with col2:
        # Option to start over
        if st.button("Start Over"):
            st.session_state.research_phase = 'input'
            st.session_state.selected_entity = None
            st.session_state.negative_entities = []
            st.rerun()
# ============================ STREAMLIT UI INTEGRATION ============================
def initialize_session_state():
    """
    Initialize all required session state variables for the application
    This should be called at the beginning of the main function
    """
    # Research process phases
    if 'research_phase' not in st.session_state:
        st.session_state.research_phase = 'input'
    
    # First pass results
    if 'first_pass_result' not in st.session_state:
        st.session_state.first_pass_result = None
    
    # Entity selection
    if 'selected_entity' not in st.session_state:
        st.session_state.selected_entity = None
    
    if 'negative_entities' not in st.session_state:
        st.session_state.negative_entities = []
    
    if 'entity_full_data' not in st.session_state:
        st.session_state.entity_full_data = []
    
    # Second pass results
    if 'second_pass_result' not in st.session_state:
        st.session_state.second_pass_result = None
    
    # Third pass results
    if 'third_pass_result' not in st.session_state:
        st.session_state.third_pass_result = None
    
    # Additional context
    if 'additional_context' not in st.session_state:
        st.session_state.additional_context = ""
    
    # Viewing historical searches
    if 'viewing_history_id' not in st.session_state:
        st.session_state.viewing_history_id = None
    
    # Search history
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Structured data visibility toggling
    if 'structured_data_visibility' not in st.session_state:
        st.session_state.structured_data_visibility = {}
    
    # Token usage tracking
    if 'token_usage' not in st.session_state:
        st.session_state.token_usage = []
    
    # Observability log
    if 'observability_log' not in st.session_state:
        st.session_state.observability_log = []
def modified_save_to_search_history(company_names, search_results, verified_results=None, scraped_urls=None, failed_urls=None):
    """Modified version of save_to_search_history with consistent formatting"""
    # Ensure search_history is initialized
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Ensure all search_results have the expected fields
    normalized_results = []
    for result in search_results:
        company_name = safe_get_attribute(result, "company_name", "Unknown")
        
        # Create a normalized result structure
        normalized_result = {
            "company_name": company_name,
            "result": {}
        }
        
        # Get the actual results data, handling different formats
        result_obj = safe_get_attribute(result, "result", result)  # Some might have 'result' field, others direct
        
        # Ensure search_results is present
        search_results_text = safe_get_attribute(result_obj, "search_results", 
                              safe_get_attribute(result_obj, "search_text_formatted", 
                              "No search results available"))
        
        normalized_result["result"]["search_results"] = [{"search_results": search_results_text}]
        
        # Copy other useful fields
        for field in ["structured_data", "verified_results"]:
            if hasattr(result_obj, field) or (isinstance(result_obj, dict) and field in result_obj):
                normalized_result["result"][field] = safe_get_attribute(result_obj, field, None)
        
        normalized_results.append(normalized_result)
    
    # Create a history entry with the normalized data
    history_entry = {
        "timestamp": timestamp,
        "companies": company_names,
        "search_results": normalized_results,
        "verified_results": verified_results if verified_results else [],
        "scraped_urls": scraped_urls if scraped_urls else [],
        "failed_urls": failed_urls if failed_urls else [],
        "id": str(hash(timestamp + str(company_names)))  # Create a unique ID
    }
    
    # Add to the beginning of the list (most recent first)
    st.session_state.search_history.insert(0, history_entry)
    
    # Limit history size (optional)
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]
        
def main():
    st.set_page_config(page_title="Enhanced Company Research", page_icon="🔍", layout="wide")
    
    # Initialize session state variables
    initialize_session_state()
    
    # Sidebar with app options
    with st.sidebar:
        app_mode = st.radio(
            "Choose Research Mode",
            options=["Standard Search", "Three-Pass Research"],
            index=1  # Default to Three-Pass
        )
        
        # API usage metrics (common to both modes)
        st.subheader("API Usage Metrics")
        if st.button("Reset Token Counter"):
            st.session_state.token_usage = []
            st.success("Token counter reset successfully.")
        
        display_token_usage()
        
        # Add LinkUp API status
        st.subheader("LinkUp API Status")
        if linkup_client:
            st.success("✅ LinkUp API connected")
        else:
            st.error("❌ LinkUp API not configured - Check your API key")

        st.divider()
        
        st.subheader("Search History")
        
        # Export button
        if hasattr(st.session_state, 'search_history') and st.session_state.search_history:
            export_search_history()
        
        # Import section
        with st.expander("Import History"):
            uploaded_file = st.file_uploader("Upload history file", type=["json"])
            if uploaded_file is not None:
                if st.button("Import"):
                    import_search_history(uploaded_file)
    
    # Common UI elements and title
    st.title("Company Research Assistant")
    
    # Set research mode in session state
    st.session_state.research_mode = app_mode
    
    # Get the async runner
    async_runner = get_async_runner()
    
    # Check if we're viewing a historical search
    if hasattr(st.session_state, 'viewing_history_id') and st.session_state.viewing_history_id:
        display_historical_search_view()
        st.stop()
    
    # Display the appropriate UI based on the current phase if in Three-Pass mode
    if app_mode == "Three-Pass Research" and 'research_phase' in st.session_state:
        if st.session_state.research_phase == 'entity_selection':
            first_pass_ui_with_data_editor()
            st.stop()
        elif st.session_state.research_phase == 'detailed_search':
            display_second_pass_ui(async_runner)
            st.stop()
        elif st.session_state.research_phase == 'enhancement':
            display_third_pass_ui(async_runner)
            st.stop()
    
    # If we're not in a specific three-pass phase, or we're in standard mode,
    # show the common input UI with tabs
    integrated_search_ui(async_runner, app_mode)


def integrated_search_ui(async_runner, app_mode):
    """
    Integrated UI that supports both Standard and Three-Pass modes
    """
    # Set up tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Enter Manually", "Upload Spreadsheet", "Search History"])
    
    with tab1:
        # Manual input form
        with st.form("company_search_form"):
            user_input = st.text_area(
                "Enter company names or describe what you're looking for:", 
                placeholder="Example: Tell me about Ideaflow, a company founded in 2014 in the Bay Area"
            )
            
            additional_context = st.text_area(
                "Additional context (optional):", 
                placeholder="Example: Founded in 2014, located in Palo Alto, funded by 8VC"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Add slider for controlling concurrency
                max_concurrent = st.slider(
                    "Maximum concurrent requests:", 
                    min_value=1, 
                    max_value=10, 
                    value=5, 
                    help="Higher values may improve speed but could hit API rate limits"
                )
            
            with col2:
                # Add toggle for web scraping
                enable_scraping = st.checkbox(
                    "Enhanced context with web scraping", 
                    value=True,
                    help="Scrape content from URLs in the additional context to improve search quality"
                )
            
            submit_button = st.form_submit_button("Search")
        
        if submit_button and user_input:
            # Check if LinkUp is available
            if not linkup_client:
                st.error("LinkUp API is not configured. Please check your API key.")
                st.stop()
            
            # Create a status container to track progress
            status = st.status("Initializing search...", expanded=True)
            
            if app_mode == "Three-Pass Research":
                # Start the three-pass workflow
                with status:
                    status.update(label="First Pass: Finding possible companies...", state="running")
                    first_pass_result = async_runner.run_task(first_pass_entity_selection(user_input, st.session_state.observability_log))
                    
                    if not first_pass_result["success"]:
                        status.update(label=f"Error: {first_pass_result.get('error', 'Unknown error')}", state="error")
                        st.stop()
                    
                    # Save additional context
                    st.session_state.additional_context = additional_context
                    
                    # Save result to session state
                    st.session_state.first_pass_result = first_pass_result
                    st.session_state.research_phase = 'entity_selection'
                    
                    status.update(label="Found potential entities! Please select the correct one.", state="complete")
                    
                    # Force rerun to show the entity selection interface
                    st.rerun()
            else:
                # Run the standard search workflow
                response = async_runner.run_task(
                    gemini_company_search_with_verification(
                        user_input=user_input, 
                        additional_context=additional_context,
                        status_container=status,
                        max_concurrent=max_concurrent,
                        enable_web_scraping=enable_scraping
                    )
                )
                
                # Display standard search results
                display_standard_search_results(response)
    
    with tab2:
        # File upload section
        uploaded_file = st.file_uploader("Upload a spreadsheet with company information", type=["xlsx", "xls", "csv"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Add concurrency controls
            max_concurrent = st.slider(
                "Maximum concurrent requests:", 
                min_value=1, 
                max_value=10, 
                value=5, 
                help="Higher values may improve speed but could hit API rate limits"
            )
        
        with col2:
            # Add toggle for web scraping
            enable_batch_scraping = st.checkbox(
                "Enable web scraping for context enhancement", 
                value=True,
                help="Scrape content from URLs in the additional context to improve search quality"
            )
            
        if uploaded_file is not None:
            # Check if LinkUp is available
            if not linkup_client:
                st.error("LinkUp API is not configured. Please check your API key.")
                st.stop()
                
            # Show a spinner while processing
            with st.spinner("Processing uploaded file..."):
                # Process the uploaded file
                result = async_runner.run_task(
                    process_uploaded_file(uploaded_file, async_runner)
                )
            
            # Process and display the file upload results
            display_file_upload_results(result, async_runner, max_concurrent, enable_batch_scraping)
    
    with tab3:
        display_search_history()

def display_second_pass_ui(async_runner):
    """Display the second pass (detailed search) UI for Three-Pass mode"""
    st.subheader("Second Pass: Detailed Company Information")
    
    # Show a summary of selections - use safe_get_attribute for name access
    entity_name = safe_get_attribute(st.session_state.selected_entity, 'name', 'Unknown Entity')
    st.write(f"Researching: **{entity_name}**")
    
    if not st.session_state.second_pass_result:
        with st.status("Running detailed search...") as status:
            second_pass_result = async_runner.run_task(
                second_pass_structured_search(
                    st.session_state.selected_entity, 
                    st.session_state.negative_entities,
                    st.session_state.additional_context,
                    st.session_state.observability_log
                )
            )
            
            if not second_pass_result["success"]:
                status.update(label=f"Error: {second_pass_result.get('error', 'Unknown error')}", state="error")
                st.stop()
            
            # Save result to session state
            st.session_state.second_pass_result = second_pass_result
            status.update(label="Information gathered successfully!", state="complete")
    
    # Display the search results - use get() for fallback to handle missing keys
    search_results = st.session_state.second_pass_result.get("search_results", 
                    st.session_state.second_pass_result.get("search_text_formatted", 
                    "No search results available."))
    
    st.markdown(search_results)
    
    # Display structured data in an expander
    structured_data = st.session_state.second_pass_result.get("structured_data")
    if structured_data:
        with st.expander("Show structured data (JSON)"):
            st.json(structured_data)
    
    # Controls for third pass
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Enhance with Third Pass"):
            st.session_state.research_phase = 'enhancement'
            st.rerun()
    
    with col2:
        if st.button("Start Over"):
            # Reset session state
            st.session_state.research_phase = 'input'
            st.session_state.selected_entity = None
            st.session_state.negative_entities = []
            st.session_state.second_pass_result = None
            st.rerun()

def display_third_pass_ui(async_runner):
    """Display the third pass (enhancement) UI for Three-Pass mode"""
    st.subheader("Third Pass: Enhanced Company Information")
    
    # Show a summary of selections
    st.write(f"Enhancing information for: **{st.session_state.selected_entity['name']}**")
    
    if not st.session_state.third_pass_result:
        with st.status("Finding missing information...") as status:
            third_pass_result = async_runner.run_task(
                third_pass_fill_missing_info(
                    st.session_state.second_pass_result.get("structured_data", {}),
                    st.session_state.selected_entity,
                    st.session_state.negative_entities,
                    st.session_state.second_pass_result.get("sources", []),
                    st.session_state.observability_log
                )
            )
            
            if not third_pass_result["success"]:
                status.update(label=f"Error: {third_pass_result.get('error', 'Unknown error')}", state="error")
                st.stop()
            
            # Save result to session state
            st.session_state.third_pass_result = third_pass_result
            
            if third_pass_result["enhanced"]:
                status.update(label=f"Enhanced {len(third_pass_result['enhanced_fields'])} fields with additional information!", state="complete")
            else:
                status.update(label="No significant enhancements needed or found.", state="complete")
    
    # Display enhanced data
    enhanced_data = st.session_state.third_pass_result["structured_data"]
    
    if st.session_state.third_pass_result["enhanced"]:
        st.success(f"Successfully enhanced {len(st.session_state.third_pass_result.get('enhanced_fields', []))} fields")
        
        # List the enhanced fields
        st.write("**Enhanced fields:**")
        for field in st.session_state.third_pass_result.get('enhanced_fields', []):
            st.write(f"- `{field}`")
    else:
        st.info("No significant enhancements were needed or found.")
    
    # Format the enhanced data as search results text
    enhanced_text = format_linkup_results_as_search_text(enhanced_data, st.session_state.selected_entity["name"])
    st.markdown(enhanced_text)
    
    # Show the structured data with highlights for enhanced fields
    with st.expander("Show enhanced structured data (JSON)"):
        st.json(enhanced_data)
    
    # Save to search history - using the modified version
    modified_save_to_search_history(
        [st.session_state.selected_entity["name"]],
        [{"company_name": st.session_state.selected_entity["name"], "result": {"search_results": [enhanced_text]}}],
        None
    )
    
    # Option to start over
    if st.button("Start New Research"):
        # Reset session state
        st.session_state.research_phase = 'input'
        st.session_state.selected_entity = None
        st.session_state.negative_entities = []
        st.session_state.first_pass_result = None
        st.session_state.second_pass_result = None
        st.session_state.third_pass_result = None
        st.rerun()


def display_standard_search_results(response):
    """Display results for the standard search workflow"""
    if not response["success"]:
        st.error(f"Error: {response.get('error', 'An unknown error occurred')}")
    else:
        # Save to search history for manual searches too
        companies_searched = response["parsed_entities"]
        save_to_search_history(
            companies_searched,
            [{"company_name": company, "result": {"search_results": results}} 
             for company, results in zip(companies_searched, response["search_results"])],
            response["verified_results"]
        )
        
        # Display timing and token usage information
        timing = response.get("timing", {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Search completed in {format_time(timing.get('total_time', 0))}**
            - Step 1 (Parsing): {format_time(timing.get('step1_time', 0))}
            - Step 2 (Searching): {format_time(timing.get('step2_time', 0))}
            - Step 3 (Verification): {format_time(timing.get('step3_time', 0))}
            """)
        
        with col2:
            # Show token usage summary
            if 'token_usage' in st.session_state and st.session_state.token_usage:
                total_tokens = sum(usage.token_count for usage in st.session_state.token_usage)
                input_cost = (total_tokens / 1000000) * 0.10
                output_cost = (total_tokens / 1000000) * 0.40
                total_cost = input_cost + output_cost
                
                st.info(f"""
                **Token Usage Summary:**
                - Total tokens: {total_tokens:,}
                - Est. cost: ${total_cost:.6f}
                """)
        
        # Detailed token usage in an expander
        with st.expander("View Detailed Token Usage", expanded=False):
            display_token_usage(inside_expander=True)

        # Display web scraping information
        if "scraped_urls" in response and (response["scraped_urls"] or response["failed_urls"]):
            with st.expander("📄 Web Content Enhancement Details", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Successfully Scraped URLs")
                    if response["scraped_urls"]:
                        for url_data in response["scraped_urls"]:
                            st.markdown(f"""
                            **{url_data['domain']}**  
                            Title: {url_data['title']}  
                            Content: {url_data['content_length']} words
                            {'(truncated)' if url_data['truncated'] else ''}
                            """)
                    else:
                        st.info("No URLs were successfully scraped.")
                
                with col2:
                    st.subheader("URLs That Couldn't Be Scraped")
                    if response["failed_urls"]:
                        for url_data in response["failed_urls"]:
                            st.markdown(f"""
                            **{url_data['domain']}**  
                            Reason: {url_data['reason']}  
                            URL: {url_data['url']}
                            """)
                            
                        # Show info message for common scenarios
                        if any("notion.so" in failed["domain"] for failed in response["failed_urls"]):
                            st.info("📝 **Note:** Notion pages typically require authentication and cannot be scraped automatically.")
                    else:
                        st.info("No failed URL scraping attempts.")
        
        # Display parsed entities
        st.subheader("Companies Identified")
        for company in response["parsed_entities"]:
            st.write(f"- {company}")
        
        # Display search results for each company
        st.subheader("Search Results")
        for idx, result in enumerate(response["search_results"]):
            company_name = result["company_name"]
            st.markdown(f"### {company_name}")
            
            # Display the search results in a cleaner format
            search_text = result["search_results"]
            st.markdown(search_text)
            
            # If there's structured data, show it with session state
            if structured_data := result.get("structured_data"):
                # Create a unique key for this data visibility toggle
                visibility_key = f"manual_data_{company_name}_{idx}"
                
                # Initialize this key in session state if it doesn't exist
                if visibility_key not in st.session_state.structured_data_visibility:
                    st.session_state.structured_data_visibility[visibility_key] = False
                
                # Display toggle checkbox that updates session state
                show_data = st.checkbox(
                    f"Show structured data for {company_name}", 
                    key=visibility_key,
                    value=st.session_state.structured_data_visibility[visibility_key]
                )
                
                # Update session state when checkbox changes
                st.session_state.structured_data_visibility[visibility_key] = show_data
                
                # Display data if toggle is on
                if show_data:
                    st.json(structured_data)
        
        # Display verified results if available
        if response["verified_results"]:
            st.subheader("Verified Results")
            for verified in response["verified_results"]:
                company_name = verified["company_name"]
                verification = verified["verification"]
                
                st.markdown(f"### {company_name}")
                st.markdown(verification)


def display_file_upload_results(result, async_runner, max_concurrent, enable_batch_scraping):
    """Display results for file upload workflow"""
    # Check if processing was successful
    if not result["success"]:
        st.error(f"Error processing uploaded file: {result['error']}")
    elif result["data"] is not None:
        editor_df = result["data"]
        
        # Show information about the parsed data
        num_companies = len(editor_df)
        st.success(f"Successfully processed file and identified {num_companies} potential companies.")
        
        # Display the data editor
        st.subheader("Review and Edit Company Information")
        st.write("You can edit the company names and additional context before searching.")
        
        # Determine which columns to display in the editor
        columns_to_display = ["company_name", "additional_context", "status"]
        
        # If "raw_data" column exists and contains data, offer to display it
        has_raw_data = "raw_data" in editor_df.columns and not editor_df["raw_data"].isna().all()
        
        # Create column configuration
        column_config = {
            "company_name": st.column_config.TextColumn(
                "Company Name",
                help="Name of the company to research"
            ),
            "additional_context": st.column_config.TextColumn(
                "Additional Context",
                help="Any additional information about the company (e.g., industry, founding year, location)"
            ),
            "status": st.column_config.SelectboxColumn(
                "Status",
                help="Current status of this search",
                options=["Pending", "Complete", "Skip"],
                required=True
            )
        }
        
        # Add raw_data column to config if present
        if has_raw_data:
            show_raw_data = st.checkbox("Show original row data from spreadsheet", value=False)
            if show_raw_data:
                columns_to_display.append("raw_data")
                column_config["raw_data"] = st.column_config.TextColumn(
                    "Original Data",
                    help="Original data from the spreadsheet row",
                    width="large"
                )
        
        # Create the data editor with the appropriate columns
        edited_df = st.data_editor(
            editor_df[columns_to_display],
            column_config=column_config,
            use_container_width=True,
            num_rows="dynamic"
        )
        
        # Add a button to start the search
        if st.button("Search for All Companies") and not edited_df.empty:
            # Filter to only include rows with status "Pending"
            search_df = edited_df[edited_df['status'] == 'Pending']
            
            if search_df.empty:
                st.warning("No companies with 'Pending' status to search for.")
            else:
                # Use our new function that handles progress updates in the main thread
                all_results = process_companies_with_progress(
                    search_df, 
                    async_runner,
                    max_concurrent=max_concurrent,
                    enable_web_scraping=enable_batch_scraping
                )
                
                if all_results:
                    # Get list of company names
                    companies_searched = [result["company_name"] for result in all_results if result["success"]]
                    # Save to history
                    save_to_search_history(
                        companies_searched,
                        all_results,
                        None  # You can add verified results if available
                    )

                    # Display token usage summary in the main area
                    st.subheader("Resource Usage Summary")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Companies Processed", len(search_df))
                        successful_results = [r for r in all_results if r["success"]]
                        avg_time_per_company = 0
                        if successful_results:
                            avg_time_per_company = sum(
                                r["result"].get("timing", {}).get("total_time", 0) 
                                for r in successful_results
                            ) / len(successful_results)
                        st.metric("Average Processing Time", format_time(avg_time_per_company))
                    
                    with col2:
                        if 'token_usage' in st.session_state and st.session_state.token_usage:
                            total_tokens = sum(usage.token_count for usage in st.session_state.token_usage)
                            input_cost = (total_tokens / 1000000) * 0.10
                            output_cost = (total_tokens / 1000000) * 0.40
                            total_cost = input_cost + output_cost
                            
                            st.metric("Total API Tokens", f"{total_tokens:,}")
                            st.metric("Estimated Cost", f"${total_cost:.6f}")
                    
                    # Detailed token usage
                    with st.expander("View Detailed Token Usage", expanded=False):
                        display_token_usage(inside_expander=True)
                    
                    # Display results
                    st.subheader("Search Results")
                    
                    # Group results by success/failure
                    successful_results = [r for r in all_results if r["success"]]
                    failed_results = [r for r in all_results if not r["success"]]
                    
                    # Show stats
                    st.write(f"Successfully searched: {len(successful_results)}/{len(all_results)} companies")
                    
                    if failed_results:
                        with st.expander(f"Failed Searches ({len(failed_results)})", expanded=False):
                            for result in failed_results:
                                company_name = result["company_name"]
                                error = result["result"].get("error", "Unknown error")
                                st.error(f"{company_name}: {error}")
                    
                    # Display results for each successful company
                    for company_result in successful_results:
                        company_name = company_result["company_name"]
                        response = company_result["result"]
                        
                        with st.expander(f"Results for {company_name}"):
                            # Display search results
                            for result_idx, result in enumerate(response["search_results"]):
                                search_text = result["search_results"]
                                st.markdown(search_text)
                                
                                # If there's structured data, show it with a toggle controlled by session state
                                if structured_data := result.get("structured_data"):
                                    # Create a unique key for this data visibility toggle
                                    visibility_key = f"data_{company_name}_{result_idx}"
                                    
                                    # Initialize in session state if needed
                                    if visibility_key not in st.session_state.structured_data_visibility:
                                        st.session_state.structured_data_visibility[visibility_key] = False
                                    
                                    # Display toggle checkbox that updates session state
                                    show_data = st.checkbox(
                                        "Show structured data", 
                                        key=f"checkbox_{visibility_key}",
                                        value=st.session_state.structured_data_visibility[visibility_key]
                                    )
                                    
                                    # Update session state when checkbox changes
                                    st.session_state.structured_data_visibility[visibility_key] = show_data
                                    
                                    # Display data if toggle is on
                                    if show_data:
                                        st.json(structured_data)
                                        st.divider()
                            
                            # Display verified results if available
                            if response["verified_results"]:
                                st.subheader("Verified Results")
                                for verified in response["verified_results"]:
                                    verification = verified["verification"]
                                    st.markdown(verification)
    else:
        st.error("Could not process the uploaded file. Please try another file.")


def display_search_history():
    """Display the search history tab content without nested expanders"""
    st.header("Previous Search Results")
    
    if not hasattr(st.session_state, 'search_history') or not st.session_state.search_history:
        st.info("No search history available. Run a search to see results here.")
    else:
        for idx, entry in enumerate(st.session_state.search_history):
            timestamp = safe_get_attribute(entry, "timestamp", "Unknown date")
            companies = ", ".join(safe_get_attribute(entry, "companies", ["Unknown"]))
            
            with st.expander(f"{companies} - {timestamp}"):
                # Display companies and timestamp
                st.markdown(f"**Companies:** {companies}")
                st.markdown(f"**Time:** {timestamp}")
                
                # Preview some of the results - WITHOUT using another expander
                search_results = safe_get_attribute(entry, "search_results", [])
                if search_results:
                    st.markdown("### Results Preview")
                    # Use a divider instead of an expander
                    st.divider()
                    
                    for result in search_results[:2]:  # Show just first 2 for preview
                        company_name = safe_get_attribute(result, "company_name", "Unknown")
                        st.markdown(f"#### {company_name}")
                        
                        # Try different fields that might contain the results text
                        result_text = None
                        result_obj = safe_get_attribute(result, "result", {})
                        
                        # Check different possible locations for the text
                        if isinstance(result_obj, dict):
                            search_result_items = safe_get_attribute(result_obj, "search_results", [])
                            if search_result_items and len(search_result_items) > 0:
                                first_item = search_result_items[0]
                                result_text = safe_get_attribute(first_item, "search_results", 
                                             safe_get_attribute(first_item, "search_text_formatted", None))
                        
                        # If not found, check direct properties
                        if not result_text:
                            result_text = safe_get_attribute(result, "search_results", 
                                         safe_get_attribute(result, "search_text_formatted", 
                                         "No results available"))
                        
                        # Show a truncated version of the text
                        st.markdown(result_text[:500] + "..." if len(result_text) > 500 else result_text)
                        st.divider()
                
                # Action buttons in columns to save space
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show a view button for full details
                    if st.button("View Full Details", key=f"view_{safe_get_attribute(entry, 'id', idx)}"):
                        st.session_state.viewing_history_id = safe_get_attribute(entry, "id", idx)
                        st.rerun()
                
                with col2:
                    # Add an option to remove this entry
                    if st.button("Remove from History", key=f"remove_{safe_get_attribute(entry, 'id', idx)}"):
                        st.session_state.search_history.pop(idx)
                        st.rerun()

def display_historical_search_view():
    """Display a historical search from the history"""
    # Find the history entry
    history_entry = next((entry for entry in st.session_state.search_history 
                         if entry["id"] == st.session_state.viewing_history_id), None)
    
    if history_entry:
        st.success(f"Viewing historical search from {history_entry['timestamp']}")
        
        # Display a button to exit history view
        if st.button("← Return to Current Search"):
            del st.session_state.viewing_history_id
            st.rerun()
        
        # Display the historical results (same way you'd display current results)
        st.subheader(f"Search Results for: {', '.join(history_entry['companies'])}")
        
        for result in history_entry["search_results"]:
            company_name = result["company_name"]
            
            with st.expander(f"Results for {company_name}", expanded=True):
                # Display stored search results
                for result_idx, search_result in enumerate(result.get("result", {}).get("search_results", [])):
                    st.markdown(search_result)
                    
                    # If there's structured data, show it
                    if structured_data := result.get("result", {}).get("structured_data", {}):
                        # Create a unique key for session state
                        visibility_key = f"history_{history_entry['id']}_{company_name}_{result_idx}"
                        
                        # Initialize in session state if needed
                        if visibility_key not in st.session_state.structured_data_visibility:
                            st.session_state.structured_data_visibility[visibility_key] = False
                        
                        # Checkbox with session state
                        show_data = st.checkbox(
                            "Show structured data", 
                            key=visibility_key,
                            value=st.session_state.structured_data_visibility[visibility_key]
                        )
                        
                        # Update session state
                        st.session_state.structured_data_visibility[visibility_key] = show_data
                        
                        if show_data:
                            st.json(structured_data)
                            st.divider()
                
                # Display verified results if available
                verified_results = result.get("result", {}).get("verified_results", [])
                if verified_results:
                    st.subheader("Verified Results")
                    for verified in verified_results:
                        verification = verified["verification"]
                        st.markdown(verification)

        if "scraped_urls" in history_entry and (history_entry["scraped_urls"] or history_entry["failed_urls"]):
            with st.expander("📄 Web Content Enhancement Details", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Successfully Scraped URLs")
                    if history_entry["scraped_urls"]:
                        for url_data in history_entry["scraped_urls"]:
                            st.markdown(f"""
                            **{url_data['domain']}**  
                            Title: {url_data['title']}  
                            Content: {url_data['content_length']} words
                            {'(truncated)' if url_data['truncated'] else ''}
                            """)
                    else:
                        st.info("No URLs were successfully scraped.")
                
                with col2:
                    st.subheader("URLs That Couldn't Be Scraped")
                    if history_entry["failed_urls"]:
                        for url_data in history_entry["failed_urls"]:
                            st.markdown(f"""
                            **{url_data['domain']}**  
                            Reason: {url_data['reason']}  
                            URL: {url_data['url']}
                            """)
                    else:
                        st.info("No failed URL scraping attempts.")


def initialize_session_state():
    """Initialize all session state variables needed for both workflows"""
    # Common variables for both workflows
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if 'structured_data_visibility' not in st.session_state:
        st.session_state.structured_data_visibility = {}
    
    if 'token_usage' not in st.session_state:
        st.session_state.token_usage = []
    
    if 'observability_log' not in st.session_state:
        st.session_state.observability_log = []
    
    # Three-pass specific variables
    if 'research_phase' not in st.session_state:
        st.session_state.research_phase = 'input'
    
    if 'first_pass_result' not in st.session_state:
        st.session_state.first_pass_result = None
    
    if 'selected_entity' not in st.session_state:
        st.session_state.selected_entity = None
    
    if 'negative_entities' not in st.session_state:
        st.session_state.negative_entities = []
    
    if 'second_pass_result' not in st.session_state:
        st.session_state.second_pass_result = None
    
    if 'third_pass_result' not in st.session_state:
        st.session_state.third_pass_result = None
    
    if 'additional_context' not in st.session_state:
        st.session_state.additional_context = ""


if __name__ == "__main__":
    main()