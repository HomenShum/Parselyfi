import os
import sys
import logging
import asyncio
from typing import List, Optional, Dict, Any, Literal # Import Literal
from pydantic import BaseModel, Field
from datetime import datetime
from tavily import AsyncTavilyClient
from pydantic_ai import Agent, RunContext, UnexpectedModelBehavior, capture_run_messages
from pydantic_ai.result import RunResult
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
import spacy
import traceback
import streamlit as st
import pandas as pd  # Import pandas
from cleanco import basename # Import cleanco
import re

# Initialize Tavily client (replace with your API key)
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

# Logger setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
GEMINI_API_KEY = st.secrets["GOOGLE_AI_STUDIO"]

openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# models
openrouter_model_llama = OpenAIModel("meta-llama/llama-3.3-70b-instruct", openai_client=openrouter_client)
openrouter_model_deepseekv3 = OpenAIModel('deepseek/deepseek-chat', openai_client=openrouter_client)
openrouter_model_gemini = OpenAIModel("google/gemini-2.0-flash-001", openai_client=openrouter_client)
gemini_2o_model = GeminiModel('gemini-2.0-flash', api_key=GEMINI_API_KEY)

openrouter_model = gemini_2o_model
logger.info(f"Using model: {openrouter_model}")

# Semaphore to limit concurrent tasks
sem = asyncio.Semaphore(1000)

# Load spaCy model for NLP tasks
nlp = spacy.load("en_core_web_lg")


### Updated Class Structures ###

class CompanyNameOutput(BaseModel):
    """Output for company name extraction."""
    company_name: Optional[str] = Field(None, description="Extracted name of the company from the text.")

# Updated CompanyFeaturesOutput model

class CompanyFeaturesOutput(BaseModel):
    """Output for company feature extraction - focusing on general overview."""
    company_overview_summary: Optional[str] = Field(None, description="Brief descriptive summary of the company, its purpose, and activities.")
    industry_overview: Optional[List[str]] = Field(None, description="General industry or sector overview the company operates within.")
    product_service_overview: Optional[List[str]] = Field(None, description="Overview of the main products or services offered by the company.")
    mission_vision_statement: Optional[str] = Field(None, description="Company's stated mission or vision for its goals and impact.")
    target_audience_customers: Optional[List[str]] = Field(None, description="Target customers or audience for the company's offerings.")
    technology_platform_overview: Optional[List[str]] = Field(None, description="Overview of the core technology platform or scientific domains used by the company.")
    geographic_focus: Optional[List[str]] = Field(None, description="Geographic areas of operation or focus (e.g., HQ location, regions served).")
    organization_type: Optional[List[str]] = Field(None, description="Type of organization (e.g., startup, public, private).")

class CompanyDataOutput(BaseModel):
    """Simplified output for company data extraction - Table format."""
    company_name: Optional[str] = Field(None, description="Name of the company.")
    company_url: Optional[List[str]] = Field(None, description="List of unique company website URLs.") # Modified to List[str] and unique
    product_name: Optional[List[str]] = Field(None, description="List of unique product names.") # Modified to unique
    product_type: Optional[str] = Field(None, description="Type of product/service.")
    scientific_domain: Optional[str] = Field(None, description="Scientific domain.")
    organization_type: Optional[str] = Field(None, description="Type of organization.")
    hq_locations: Optional[List[str]] = Field(None, description="List of unique HQ locations.") # Modified to unique
    description_abstract: Optional[str] = Field(None, description="Brief company description - summarized from all sources.") # Summarized description
    total_funding: Optional[str] = Field(None, description="Aggregated total funding amount (e.g., 'USD 100 Million').") # Aggregated funding
    employee_count: Optional[str] = Field(None, description="Employee count range or estimate.")
    relevant_segments: Optional[List[str]] = Field(None, description="List of unique relevant market segments.") # Modified to unique
    investor_name: Optional[List[str]] = Field(None, description="List of unique investor names.") # Modified to unique
    competitors: Optional[List[str]] = Field(None, description="List of unique competitor company names.") # Modified to unique

class SentenceVerificationResult(BaseModel):
    """Represents the verification results for a single field in CompanyDataOutput."""
    field_name: str = Field(..., description="The name of the field being verified.")
    extracted_value: Any = Field(..., description="The extracted value for the field.")
    verification_status: str = Field(..., description="The verification status of the extracted value (e.g., 'verified', 'not found', 'partially verified', 'contradicts source').")
    reason: Optional[str] = Field(None, description="Optional reason for the verification status if not 'verified'.")


class VerificationResult(BaseModel):
    """Represents the verification results for the CompanyDataOutput fields."""
    field_verifications: List[SentenceVerificationResult] = Field(..., description="A list of verification results for each field in CompanyDataOutput.")

# Modified SingleResultSelectionOutput to remove match_strength and simplify to exact match
class SingleResultSelectionOutput(BaseModel):
    """Represents the output of the search result selection agent for ONE result - simplified for table inclusion."""
    include_in_table: bool = Field(..., description="True if this result should be included in the aggregated table, False otherwise.")
    reason: Optional[str] = Field(None, description="Reasoning for including or excluding this search result from the table.")

company_name_agent = Agent(
    model=openrouter_model,
    result_type=CompanyNameOutput,
    system_prompt="""Extract the primary company name from the given text (title, URL, content).
Focus on the title and initial sentences. Return a CompanyNameOutput with the extracted name.
If no clear company name is found, set company_name to None."""
)

company_features_agent = Agent(
    model=openrouter_model,
    result_type=CompanyFeaturesOutput,
    model_settings=dict(parallel_tool_calls=False),
    system_prompt="""Extract general company features from provided context:
- Analyze company name, data, and context text
- Focus on broad understanding of company's purpose and characteristics
- Extract information for: company overview, industry, products/services, mission/vision, target audience, technology, geographic focus, and organization type
- Provide general overviews and summaries for each category
- Output a CompanyFeaturesOutput object with extracted information"""
)

company_data_agent = Agent(
    model=openrouter_model,
    result_type=CompanyDataOutput,
    model_settings=dict(parallel_tool_calls=False),
    system_prompt="""Extract key company info for database table.
Input: extracted_company_name, search_query_company_name, text (title, URL, content).
Task: Analyze text, extract details for PRIMARY COMPANY. Use CompanyDataOutput model.
Fields: company_name, company_url, product_name, product_type, scientific_domain,
        organization_type, hq_locations, description_abstract, total_funding,
        employee_count, relevant_segments, investor_name, competitors.
Focus on accuracy and relevance for primary company. Leave empty if not found."""
)

aggregated_company_data_agent = Agent(
    model=openrouter_model,
    result_type=CompanyDataOutput,
    model_settings=dict(parallel_tool_calls=False),
    system_prompt="""Summarize and refine aggregated company data from a CompanyDataOutput object:
1. Create a concise description_abstract.
2. Aggregate total_funding and employee_count.
3. Ensure unique values in list fields.
4. Review and refine all fields for accuracy and conciseness.
Output a refined CompanyDataOutput object."""
)

verification_agent = Agent(
    model=openrouter_model,
    result_type=VerificationResult,
    model_settings=dict(parallel_tool_calls=False),
    system_prompt="""Verify extracted company data against original content.
Input: Extracted Company Data (JSON), Raw Content (text)
Task: Verify each field in Extracted Company Data against Raw Content
Output: VerificationResult with SentenceVerificationResult for each field
Criteria: Verified, Not Found, Partially Verified, Contradicts Source
Process: Analyze content, determine status, provide reason if not verified"""
)


search_result_selection_agent = Agent(
    model=openrouter_model,
    result_type=SingleResultSelectionOutput,
    model_settings=dict(parallel_tool_calls=False),
    system_prompt="""You are an expert AI analyst. Determine if this search result should be included in a company information table related to a company search query.

CONTEXT:
User searched for a company. The cleaned search query company name is: {search_query_company_name_cleaned}.
Exact Match Results Found: You have already identified some search results as exact matches for the company name. These are provided as exact_match_results_metadata.  (Use this to understand the primary company better).
Extracted Content: For the searched company, content was extracted from a user-provided URL. This 'extracted_content' is considered highly reliable and should be used as a primary reference. {extracted_content_available_message} Extracted content is provided as 'extracted_content'.
You are now evaluating a non-exact match search result.

INPUT:
search_query_company_name_cleaned: Cleaned, lowercased search query company name.
exact_match_results_metadata: List of metadata for search results already identified as exact matches.
extracted_content: Content extracted from user-provided URL (if available, otherwise None).
search_result: A single processed non-exact match search result (with metadata, company_data, extracted_company_name, errors).

INCLUSION CRITERIA (for Table):
1. Relevance to Searched Company (Primary):
Does this non-exact match result provide relevant and useful information related to the company the user searched for ({search_query_company_name_cleaned})?
Consider if it offers:
Complementary Information:  Details not found in exact match results or extracted_content (if available).
Confirmation/Verification: Information that aligns with or verifies details from extracted_content or exact match results.
Related Entities: Information about subsidiaries, parent companies, key personnel, or closely associated organizations.
Broader Context: Market analysis, industry trends, etc., directly relevant to the searched company.
It DOES NOT need to be an exact name match.  Relevance is key.

2. Consistency with Extracted Content (If Available, High Priority):
If extracted_content is available, prioritize search results that are consistent with it.  Favor results that confirm or add details to the extracted_content.  Reject results that contradict the extracted_content unless there's a strong reason to believe the extracted_content is outdated or incorrect.

3. Avoid Redundancy (Secondary):
If the information in this non-exact match result is already comprehensively covered in the exact_match_results_metadata or extracted_content, it might be less valuable to include.  Prioritize new and unique information.

4. Avoid Errors: Reject results with errors (error field present).

OUTPUT: SingleResultSelectionOutput object with:
include_in_table: True if this result should be included in the aggregated table because it provides relevant and useful information about the searched company (considering context, extracted_content, and avoiding redundancy). False otherwise.
reason: Justify your decision, focusing on Relevance (Criterion 1), Consistency with Extracted Content (Criterion 2), redundancy (Criterion 3), and any errors. Explain why it is or is not valuable to include in the table.

Determine if this non-exact match result is valuable to include in the company information table. Focus on relevance, usefulness and consistency with extracted content, not just name matching.""",
)


### Core Functions ###
# Renamed function to process_data_for_table
async def process_data_for_table(content_text: str, extracted_company_name: str) -> CompanyDataOutput:
    """
    Processes the input content text to extract company information for database table.

    Args:
        content_text: The input text content.
        extracted_company_name: The name of the company extracted by company_name_agent.
        search_query_company_name: The original company name used in the search query.

    Returns:
        A CompanyDataOutput object containing extracted company data.
    """
    try:
        with capture_run_messages() as messages:
            company_data_output = await company_data_agent.run(
                user_prompt=f"Extracted company name: {extracted_company_name}, text: {content_text}",
            )
            return company_data_output.data

    except UnexpectedModelBehavior as e:
        logger.error(f"UnexpectedModelBehavior in process_data_for_table: {e}")
        logger.error(f"Cause: {repr(e.__cause__)}")
        logger.error(f"Messages: {messages}")
        raise e
    except Exception as e:
        logger.error(f"Error in process_data_for_table: {e}")
        raise e


async def verification_agent_run(company_data: CompanyDataOutput, raw_content: str) -> VerificationResult:
    """
    Verifies the accuracy of extracted CompanyDataOutput against the original raw content.

    Args:
        company_data: The CompanyDataOutput object to verify.
        raw_content: The original raw content for comparison.

    Returns:
        A VerificationResult object containing verification results for each field.
    """
    try:
        with capture_run_messages() as messages:
            verification_agent_result = await verification_agent.run(
                user_prompt=f"Extracted Company Data:\n{company_data.model_dump_json()}\n\nRaw Content:\n{raw_content}"
            )
            return verification_agent_result.data

    except UnexpectedModelBehavior as e:
        logger.error(f"UnexpectedModelBehavior in verification_agent_run: {e}")
        logger.error(f"Cause: {repr(e.__cause__)}")
        logger.error(f"Messages: {messages}")
        raise e
    except Exception as e:
        logger.error(f"Error in verification_agent_run: {e}")
        raise e



async def process_content(content_text: str, title: str, url: str, extracted_content_gold_standard: Optional[str] = None) -> Dict[str, Any]: # Added extracted_content_gold_standard
    """
    Processes the input content text to extract company information.

    Args:
        content_text: The input text content to process (raw content if available).
        title: The title of the search result.
        url: The URL of the search result.
        extracted_company_name: The original company name used in the search query.
        extracted_content_gold_standard: Gold standard content extracted from user-provided URL (optional).

    Returns:
        A dictionary containing:
            - extracted_company_name: The extracted company name from the text.
            - company_data: A CompanyDataOutput object with extracted company data.
            - verification_results: Verification results (currently placeholder).
    """
    try:
        # Step 1: Extract Company Name using company_name_agent
        logger.info("Extracting company name...")
        company_name_extraction_output = await company_name_agent.run(
            user_prompt=f"Text:\nTitle: {title}\nURL: {url}\nContent: {content_text}"
        )
        extracted_company_name_raw = company_name_extraction_output.data.company_name # Keep the raw extracted name
        extracted_company_name = basename(extracted_company_name_raw).lower() if extracted_company_name_raw else None # Clean and lowercase

        logger.info(f"Extracted company name (cleaned): {extracted_company_name}, (raw): {extracted_company_name_raw}") # Log both

        # Step 2: Extract Company Data for Table
        logger.info("Processing company data extraction for table...")
        # Construct a more informative prompt by including title, URL, and search query company name
        prompt_text = f"Extracted Company Name: {extracted_company_name}\nTitle: {title}\nURL: {url}\n\nContent:\n{content_text}\n\nExtract company information for the PRIMARY company, which is likely '{extracted_company_name}' (if available, otherwise use search query name), discussed in the above text, title, and URL, keeping in mind the search query company name."
        company_data_output = await process_data_for_table(prompt_text, extracted_company_name) # Pass extracted company name and search query name
        logger.info(f"Extracted company data: {company_data_output.model_dump()}")

        # Verification - You can adapt or remove verification for this table-filling task as needed.
        verification_results = {} # Placeholder - adapt or remove verification

        output = {
            "extracted_company_name_raw": extracted_company_name_raw, # Store raw name
            "extracted_company_name": extracted_company_name, # Store the cleaned, lowercase name
            "company_data": company_data_output.model_dump(),
            "verification_results": verification_results # Keep verification results as empty for now - removed .model_dump() here
        }

        return output

    except Exception as e:
        logger.error(f"Error in process_content: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e

# Test case with mock data
TEST_SEARCH_RESULT = {
    'query': 'Company Name: 20n Bio',
    'follow_up_questions': None,
    'answer': None,
    'images': [],
    'results': [
        {
            'title': '20n Bio, a peptide discovery company',
            'url': 'https://www.20n.bio/',
            'content': '20n Bio, a peptide discovery company "twenty-n bio") is a VC-backed startup dedicated to discovering cyclic peptides for therapeutic use. Based in the Greater Philadelphia region, 20n Bio\'s mission is to address unmet medical needs by leveraging high-throughput screening and chemical optimization to develop peptide drugs across various therapeutic areas. 20n Bio has developed a proprietary peptide discovery platform capable of screening libraries containing trillions of peptides at once, offering unparalleled potential to identify the most potent candidates. In addition to internal pipeline development, 20n Bio is also dedicated to partnering with pharmaceutical and biotechnology companies, providing customized solutions to help bring novel peptide therapeutics to market efficiently. Jan 31, 2024: 20n Bio Appoints Linghang Zhuang, Ph.D., as President, to lead the discovery of cyclic peptide drugs.',
            'score': 0.8757978,
            'raw_content': '20n Bio, a peptide discovery company\nSearch this site\nSkip to main content\nSkip to navigation\n\xa0\nA Peptide Discovery Company\n20n Bio (pr. "twenty-n bio") is a VC-backed startup dedicated to discovering cyclic peptides for therapeutic use. Based in the Greater Philadelphia region, 20n Bio\'s mission is to address unmet medical needs by leveraging high-throughput screening and chemical optimization to develop peptide drugs across various therapeutic areas.\nOur Focus\nPeptides offer the best of both small molecules and antibodies, demonstrating significant therapeutic potential across a wide range of modalities, including oral peptides, radioligand therapy, peptide-drug conjugates, etc. Their versatility enables them to target diverse biological pathways, making them suitable for treating conditions from metabolic disorders to complex cancers. Peptides also generally exhibit high specificity and affinity for their targets, resulting in fewer off-target effects and improved safety compared to traditional small molecules.\nWhile most FDA-approved peptide drugs are natural derivatives, advancements in de novo peptide discovery are addressing challenges such as the vast sequence space and the optimization of peptides for stability and bioavailability. Emerging technologies are making it increasingly feasible to identify peptides with desired therapeutic activity.\n20n Bio has developed a proprietary peptide discovery platform capable of screening libraries containing trillions of peptides at once, offering unparalleled potential to identify the most potent candidates. By utilizing high-throughput screening, the platform efficiently navigates the immense peptide sequence space to quickly identify peptides with optimal binding and therapeutic properties. This ability to discover high-affinity peptides de novo opens new avenues for developing innovative treatments to address unmet medical needs.\nIn addition to internal pipeline development, 20n Bio is also dedicated to partnering with pharmaceutical and biotechnology companies, providing customized solutions to help bring novel peptide therapeutics to market efficiently. As the demand for targeted and effective therapies continues to grow, 20n Bio is at the forefront of peptide innovation, driving advancements that are shaping the future of modern medicine.\nLeadership Team\nWe are a group of peptide believers.\nWe encourage enthusiastic scientists to join us to fulfill our mission.\n\nXiaole Chen\nVP, Biology\n\nMingfu Zhu\nFounder & CEO\n\nLinghang Zhuang\nPresident\n\nCompany News\n=========================================================\nJan 31, 2024: 20n Bio Appoints Linghang Zhuang, Ph.D., as President, to lead the discovery of cyclic peptide drugs.\nDec 17, 2021: 20n Bio 20n Bio Raises $3.3 Million in Series Seed Funding.\nContact\n115 Great Valley Pkwy\nMalvern, PA, 19355\ninfo at 20n.bio\n\n© 2024 20n Bio, Limited. All rights reserved.\nGoogle Sites\nReport abuse\nPage details\nPage updated\nGoogle Sites\nReport abuse'
        },
        {
            'title': '20N - Crunchbase Company Profile & Funding',
            'url': 'https://www.crunchbase.com/organization/20n',
            'content': '20N is a biotechnology company. The company provides peptide therapeutics for unmet medical needs. 20N is based out of Malvern, Pennsylvania, United States.',
            'score': 0.85,
            'raw_content': '20N Overview\n20N '
        }
    ],
    'response_time': 1.9
}


async def search_company_summary(company_name: str, company_urls: Optional[List[str]] = None, use_test_data: bool = False, NUMBER_OF_SEARCH_RESULTS: int = 5) -> tuple[Dict[str, List[Dict[str, Any]]], str, List[Dict[str, Any]], List[RunResult[SingleResultSelectionOutput]], List[Dict[str, Any]], Optional[Dict[str, Any]]]: # RETURN non_exact_match_results_metadata and extracted_content_ gold_standard
    """
    Searches for information about a company using Tavily, processes the search results,
    classifies sentences, and returns structured information grouped by entity name.
    """
    # Initialize or get session state - INITIALIZATION MOVED TO render_search_interface FUNCTION
    if 'search_results_state' not in st.session_state: # Keep this check here just in case for re-runs within the function.
        st.session_state.search_results_state = {
            'grouped_results_dict': {},
            'cleaned_search_query_name': None,
            'results_with_metadata': [],
            'non_exact_match_results_metadata': [],
            'selection_results': [],
            'extracted_content_gold_standard': None # Initialize gold standard content
        }
    search_results_state = st.session_state.search_results_state # Get state *after* initialization


    grouped_results_dict = search_results_state['grouped_results_dict']
    cleaned_search_query_name = search_results_state['cleaned_search_query_name']
    results_with_metadata = search_results_state['results_with_metadata']
    non_exact_match_results_metadata = search_results_state['non_exact_match_results_metadata']
    selection_results = search_results_state['selection_results']
    extracted_content_gold_standard_data = search_results_state['extracted_content_gold_standard'] # Get gold standard data


    exact_match_results_metadata = [] # Initialize outside try block
    extracted_content_gold_standard_raw = None # Initialize raw extracted content
    all_extracted_content_list = [] # List to hold extracted content from all URLs


    async with sem:
        try:
            # --- [NEW: Extract content from URLs if provided] ---
            if company_urls and isinstance(company_urls, list) and company_urls: # Check if company_urls is a non-empty list
                logger.info(f"Extracting content from provided URLs: {company_urls}")
                all_extract_results = []
                for url in company_urls: # Iterate through the list of URLs
                    try:
                        extract_results = await tavily_client.extract(urls=url) # Extract content for each URL
                        if extract_results and extract_results.get('results'):
                            all_extract_results.extend(extract_results['results']) # Collect results from all URLs
                            logger.info(f"Extracted content from URL successfully: {url}")
                        else:
                            logger.warning(f"No content extracted from URL: {url}")
                    except Exception as extract_error:
                        logger.error(f"Error during Tavily extract for URL {url}: {extract_error}")

                if all_extract_results: # If we have any successful extract results
                    combined_content = ""
                    for result in all_extract_results: # Concatenate content from all extracted results
                        combined_content += result.get('content', "") + "\n\n"

                    extracted_content_gold_standard_raw = combined_content # Combined content becomes raw gold standard
                    extracted_content_gold_standard_data = { # Structure the combined extracted content data - using the first URL as representative URL (you might want to change this)
                        'url': company_urls[0] if company_urls else None, # Or handle URL representation differently if needed
                        'content': combined_content,
                        'raw_content': combined_content, # Raw and processed content are the same in this concat scenario
                        'title': "Combined Content from Provided URLs", # Representative title
                    }
                    st.session_state.search_results_state['extracted_content_gold_standard'] = extracted_content_gold_standard_data # Store in session state
                    logger.info(f"Combined extracted content from URLs successfully.")
                else:
                    logger.warning("No content extracted from any of the provided URLs.")
                    extracted_content_gold_standard_data = None # Clear if no extraction success
                    st.session_state.search_results_state['extracted_content_gold_standard'] = None # Update session state on failure

            else:
                logger.info("No company URLs provided, skipping Tavily extract.")
                extracted_content_gold_standard_data = None # Ensure it's None if no URLs provided
                st.session_state.search_results_state['extracted_content_gold_standard'] = None # Update session state

            # Use test data or perform search (same as before)
            if use_test_data:
                search_results = TEST_SEARCH_RESULT
                logger.info("Using test search results")
            else:
                search_results = await tavily_client.search(
                    query=f"{company_name} company information", # Modified query here
                    max_results=NUMBER_OF_SEARCH_RESULTS,
                    search_depth="advanced",
                    include_raw_content=True
                )

            logger.info(f"Search Results: {search_results}")

            # Store the search query name for later use
            company_name_extraction_output = await company_name_agent.run(user_prompt=f"Extract the company name from the following text: {company_name}")

            cleaned_search_query_name = basename(company_name_extraction_output.data.company_name).lower() # Clean and store the search query name

            # Process each search result in parallel
            processing_tasks = [
                process_content(
                    content_text=result['raw_content'] if result.get('raw_content') else result['content'],
                    title=result['title'], # Pass title
                    url=result['url'],     # Pass url
                    extracted_content_gold_standard=extracted_content_gold_standard_raw # Pass gold standard content
                )
                for result in search_results['results']
            ]
            processed_results = await asyncio.gather(*processing_tasks, return_exceptions=True)

            logger.info(f"Processed Results: {processed_results}")

            results_with_metadata = []


            for index, processed_result in enumerate(processed_results):
                result_dict = {"search_result_metadata": {
                    "title": search_results['results'][index]['title'],
                    "url": search_results['results'][index]['url'],
                    "content": search_results['results'][index]['content'],
                    "raw_content": search_results['results'][index]['raw_content'],
                    "search_result_index": index
                },
                                 "extracted_company_name": None if isinstance(processed_result, Exception) else processed_result.get('extracted_company_name'), # Add extracted company name
                                 "company_data": None if isinstance(processed_result, Exception) else processed_result.get('company_data')}

                if isinstance(processed_result, Exception):
                    result_dict["error"] = str(processed_result)
                elif isinstance(processed_result, dict):
                    result_dict.update(processed_result)
                else:
                    result_dict["error"] = "Unexpected result type"
                results_with_metadata.append(result_dict)

            # First if the extracted company name matches the cleaned_search_query_name, it's an exact match
            exact_match_results_metadata = []
            non_exact_match_results_metadata = []

            for result_dict in results_with_metadata:
                if result_dict.get("extracted_company_name") == cleaned_search_query_name:
                    exact_match_results_metadata.append(result_dict)
                else:
                    non_exact_match_results_metadata.append(result_dict)

            # --- [INSERT NEW FEATURE EXTRACTION HERE] ---
            # Pass content_text as context for feature extraction
            context_for_features = ""
            if exact_match_results_metadata:
                context_for_features = "\n\n".join([res['search_result_metadata']['content'] for res in exact_match_results_metadata if res.get('search_result_metadata') and res['search_result_metadata'].get('content')]) # Concatenate content from exact matches
            if not context_for_features and results_with_metadata:
                context_for_features = results_with_metadata[0]['search_result_metadata']['content'] if results_with_metadata[0].get('search_result_metadata') and results_with_metadata[0]['search_result_metadata'].get('content') else "" # Fallback to first result if no exact matches

            company_features_output = await company_features_agent.run(
                user_prompt=f"Company Name: {cleaned_search_query_name}\nExact Match Company Data List: {[res['company_data'] for res in exact_match_results_metadata if res.get('company_data')]}\nContext Text:\n{context_for_features}" # Added context_text to prompt
            )
            company_features = company_features_output.data
            logger.info(f"Extracted Company Features: {company_features.model_dump_json(indent=2)}")

            # Run result selection agent INDIVIDUALLY for each result in non_exact_match_results_metadata using asyncio.gather
            selection_tasks = []
            for result_dict in non_exact_match_results_metadata:
                extracted_content_for_agent = extracted_content_gold_standard_data.get('content') if extracted_content_gold_standard_data else None # Get content to pass to agent
                extracted_content_available_message = "Extracted content IS available." if extracted_content_for_agent else "Extracted content is NOT available." # Message for prompt
                selection_tasks.append(
                    search_result_selection_agent.run(
                        user_prompt=f"Search Query Company Name Cleaned: {cleaned_search_query_name}\nExact Match Results Metadata: {exact_match_results_metadata}\nExtracted Content: {extracted_content_for_agent}\nExtracted Content Available Message: {extracted_content_available_message}\nSearch Result: {result_dict}",
                        # context={"extracted_content": extracted_content_for_agent, "extracted_content_available_message": extracted_content_available_message} # Pass as context too, for safety # REMOVE CONTEXT ARGUMENT
                    )
                )

            selection_results = await asyncio.gather(*selection_tasks)


            selected_indices = []
            for index, selection_result in enumerate(selection_results):
                if isinstance(selection_result, RunResult) and selection_result.data: # Check for RunResult and data
                    if selection_result.data.include_in_table: # Use include_in_table
                        selected_indices.append(index)
                        non_exact_match_results_metadata[index]["selection_result"] = selection_result.data.model_dump() # Store selection result
                        logger.info(f"Selection Agent for non-exact index {index}: {selection_result.data.model_dump_json(indent=2)}")
                    else:
                        non_exact_match_results_metadata[index]["selection_result"] = selection_result.data.model_dump() # Store selection result even if not included, for debugging
                        logger.info(f"Selection Agent for non-exact index {index}: {selection_result.data.model_dump_json(indent=2)}")


            # Grouped by entity name (only exact matches for now)
            grouped_results_dict = {} # Reset grouped results to only include selected ones.
            for result_dict in exact_match_results_metadata: # Use exact_match_results_metadata
                entity_name_raw = result_dict.get("company_data", {}).get("company_name", result_dict.get("extracted_company_name", "Unknown Entity")) # Prioritize company_data.company_name, fallback to extracted_company_name
                entity_name_cleaned = basename(entity_name_raw) if entity_name_raw and entity_name_raw != "Unknown Entity" else "Unknown Entity"
                entity_name = entity_name_cleaned.lower()

                if entity_name not in grouped_results_dict:
                    grouped_results_dict[entity_name] = []
                grouped_results_dict[entity_name].append(result_dict)


            logger.info(f"Final Grouped Results (Exact Matches Only): {grouped_results_dict}")
            logger.info(f"Non-Exact Match Results (with Selection Info): {non_exact_match_results_metadata}") # Log non-exact matches

        except Exception as e:
            logger.error(f"General error in search_company_summary: {e}")
            grouped_results_dict["General Error"] = [{"error": f"General error: {e}"}]
            non_exact_match_results_metadata = [] # Ensure it's still an empty list in case of error
            selection_results = [] # Ensure it's still an empty list in case of error
            company_features = None # Ensure company_features is None in error case
            extracted_content_gold_standard_data = None # Ensure gold standard data is None in error case


        # Store all variables in session state
        st.session_state.search_results_state['grouped_results_dict'] = grouped_results_dict
        st.session_state.search_results_state['cleaned_search_query_name'] = cleaned_search_query_name
        st.session_state.search_results_state['results_with_metadata'] = results_with_metadata
        st.session_state.search_results_state['non_exact_match_results_metadata'] = non_exact_match_results_metadata # Store non-exact matches
        st.session_state.search_results_state['selection_results'] = selection_results
        st.session_state.search_results_all_companies[company_name]['company_features'] = company_features # Store company features in session state
        st.session_state.search_results_state['extracted_content_gold_standard'] = extracted_content_gold_standard_data # Store gold standard data in session state

        return grouped_results_dict, cleaned_search_query_name, results_with_metadata, selection_results, non_exact_match_results_metadata, extracted_content_gold_standard_data # RETURN it here, including gold standard data


### Streamlit UI Components - Top level functions (Display Functions moved to top level) ###
async def display_interactive_results(cleaned_search_query_name: str, selection_results: List[RunResult[SingleResultSelectionOutput]]):
    """Displays search results with table and checkboxes for interactive selection."""
    results_with_metadata = st.session_state.search_results_all_companies[st.session_state.selected_company].get("results_with_metadata", []) # Access from the correct company in session state
    non_exact_match_results_metadata = st.session_state.search_results_all_companies[st.session_state.selected_company].get("non_exact_match_results_metadata", []) # Access from the correct company in session state
    extracted_content_gold_standard_data = st.session_state.search_results_all_companies[st.session_state.selected_company].get("extracted_content_gold_standard") # Access gold standard data

    # ADD THESE LINES to retrieve from search_results_all_companies using cleaned_search_query_name:
    if st.session_state.selected_company and st.session_state.search_results_all_companies.get(st.session_state.selected_company): # Use selected_company
        company_features = st.session_state.search_results_all_companies[st.session_state.selected_company].get("company_features", None) # Use selected_company
        cleaned_search_query_name = st.session_state.search_results_all_companies[st.session_state.selected_company].get("cleaned_search_query_name") # Use selected_company, get cleaned name again to be sure
    else:
        company_features = None
        cleaned_search_query_name = "Unknown Company"


    interactive_results_container = st.container()
    with interactive_results_container:
        if not results_with_metadata:
            st.info("No search results to display interactively.")
            return

        # --- [MODIFIED: Integrate Extracted Company Name into Features Table] ---
        st.header("Company Overview and Features")
        if company_features:
            st.write("This section provides an overview and key features of the company based on aggregated search results.")

            features_data = {
                "Feature": [],
                "Value": []
            }

            # Get unique exact match names (as before)
            exact_match_names = cleaned_search_query_name

            # Add Extracted Company Name as the FIRST row
            features_data["Feature"].append("Company Name Searched")
            features_data["Value"].append(exact_match_names)

            feature_mapping = {
                "Company Overview": company_features.company_overview_summary,
                "Industry": company_features.industry_overview,
                "Products/Services": company_features.product_service_overview,
                "Technology Platform": company_features.technology_platform_overview,
                "Mission/Vision Statement": company_features.mission_vision_statement,
                "Target Audience/Customers": company_features.target_audience_customers,
                "Geographic Focus": company_features.geographic_focus,
                "Organization Type": company_features.organization_type
            }

            for feature, value in feature_mapping.items():
                if value:
                    features_data["Feature"].append(feature)
                    features_data["Value"].append(", ".join(value) if isinstance(value, list) else value)

            if features_data["Feature"]:
                st.table(pd.DataFrame(features_data))
            else:
                st.info("No specific features extracted from the search results.")
        else:
            st.info("No company features extracted from the search results.")

        # --- [NEW: Display table of individual source summaries] ---
        st.header(f"Due Diligence Results for: {cleaned_search_query_name.title()}")
        if results_with_metadata:
            source_summary_data = []
            for result_dict in results_with_metadata:
                company_data = result_dict.get('company_data')
                if company_data:
                    source_summary_data.append({
                        "Source Title": result_dict['search_result_metadata']['title'],
                        "Source URL": result_dict['search_result_metadata']['url'],
                        "Source Content": result_dict['search_result_metadata']['content'],
                        "Source Raw Content": result_dict['search_result_metadata']['raw_content'],
                        "Company Name": company_data.get("company_name", "N/A"),
                        "Company URL": ", ".join(company_data.get("company_url", []) or []),
                        "Product Name": ", ".join(company_data.get("product_name", []) or []),
                        "Product Type": company_data.get("product_type", "N/A"),
                        "Scientific Domain": company_data.get("scientific_domain", "N/A"),
                        "Organization Type": company_data.get("organization_type", "N/A"),
                        "HQ Locations": ", ".join(company_data.get("hq_locations", []) or []),
                        "Description": company_data.get("description_abstract", "N/A"),
                        "Total Funding": company_data.get("total_funding", "N/A"),
                        "Employee Count": company_data.get("employee_count", "N/A"),
                        "Relevant Segments": ", ".join(company_data.get("relevant_segments", []) or []),
                        "Investors": ", ".join(company_data.get("investor_name", []) or []),
                        "Competitors": ", ".join(company_data.get("competitors", []) or [])
                    })
            if source_summary_data:
                df_source_summaries = pd.DataFrame(source_summary_data)
                st.dataframe(df_source_summaries)
            else:
                st.info("No company data extracted from search results to display in table.")
        else:
            st.info("No search results available to summarize.")
        # --- [END: Table display] ---

        # --- [NEW: Display Extracted Content - Gold Standard] ---
        gold_standard_expander = st.expander(f"Extracted Content from Provided URL (Gold Standard): {cleaned_search_query_name.title()}", expanded=False)
        if extracted_content_gold_standard_data:
            with gold_standard_expander:
                st.info("This is the content extracted from the user-provided URL. It is considered the most reliable source.")
                st.write(f"Source URL: {extracted_content_gold_standard_data['url']}")
                st.write(f"Title: {extracted_content_gold_standard_data['title']}")
                st.write("Content:")
                st.write(extracted_content_gold_standard_data['content'])
                st.markdown("---")
        elif st.session_state.search_results_all_companies[st.session_state.selected_company].get('company_url'): # Check if URL was provided but no extract data
            with gold_standard_expander:
                st.warning("No content could be extracted from the provided URL. There might be an issue accessing the URL or with the extraction process.")
        # --- [END: Extracted Content Display] ---


        exact_matches_container = st.container()
        with exact_matches_container:
            st.header(f"Exact Match Results Checkboxes (for aggregation): {cleaned_search_query_name.title()}")
            selected_company_data_list_exact = []
            exact_match_count = 0

            for index, result_dict in enumerate(results_with_metadata):
                default_checkbox_value = False
                is_exact_match = False

                if result_dict.get("extracted_company_name") == cleaned_search_query_name:
                    is_exact_match = True
                    default_checkbox_value = True

                if is_exact_match:
                    exact_match_count += 1
                    # More unique key: include cleaned company name and result index
                    checkbox_key = f"exact_match_checkbox_{cleaned_search_query_name}_{index}" # CORRECT KEY
                    is_selected = st.checkbox(f"Select Source: {result_dict['search_result_metadata']['title']} (Exact Match)", key=checkbox_key, value=default_checkbox_value, label_visibility="visible") # Added label_visibility
                    if is_selected:
                        if result_dict.get('company_data'):
                            selected_company_data_list_exact.append(result_dict['company_data'])

        non_exact_matches_container = st.container()
        with non_exact_matches_container:
            st.header(f"Potentially Relevant Non-Exact Match Results Checkboxes (for aggregation): {cleaned_search_query_name.title()}")
            selected_company_data_list_non_exact = []
            non_exact_match_count = 0
            for index_non_exact, result_dict in enumerate(non_exact_match_results_metadata):
                # More unique key: include cleaned company name and non-exact result index
                checkbox_key = f"non_exact_result_checkbox_{cleaned_search_query_name}_{index_non_exact}" # CORRECT KEY
                default_checkbox_value = False
                reason = None

                if selection_results and index_non_exact < len(selection_results):
                    selection_result = selection_results[index_non_exact]
                    if isinstance(selection_result, RunResult) and selection_result.data:
                        if selection_result.data.include_in_table:
                            default_checkbox_value = True
                            non_exact_match_count += 1
                        reason = selection_result.data.reason

                is_selected = st.checkbox(f"Select Source: {result_dict['search_result_metadata']['title']} (Potentially Relevant)", key=checkbox_key, value=default_checkbox_value, label_visibility="visible") # Added label_visibility
                if is_selected:
                    if result_dict.get('company_data'):
                        selected_company_data_list_non_exact.append(result_dict['company_data'])
                if reason:
                    st.write(f"    Selection Reason: *{reason}*")
                else:
                    st.write("    Selection Reason: *(Selection agent reason not available)*")

            if non_exact_match_count == 0 and non_exact_match_results_metadata:
                st.info("No non-exact match results were selected as relevant to include in the table.")

        raw_output_expander = st.expander(f"Non-Exact Match Results - Raw Output (for debug): {cleaned_search_query_name.title()}", expanded=False)
        if non_exact_match_results_metadata:
            with raw_output_expander:
                st.info("These are non-exact match results and their raw extracted data. Check 'Potentially Relevant Non-Exact Match Results' section above for interactive selection.")
                for result_dict in non_exact_match_results_metadata:
                    st.subheader(f"Source: {result_dict['search_result_metadata']['title']}")
                    st.write(f"URL: {result_dict['search_result_metadata']['url']}")
                    if 'company_data' in result_dict and result_dict['company_data']:
                        st.write("Extracted Company Data:")
                        st.write(result_dict['company_data'])
                    else:
                        st.warning("No company data extracted from this result.")
                    if 'selection_result' in result_dict:
                        st.write("Selection Agent Output:")
                        st.write(result_dict['selection_result'])
                    st.markdown("---")

        button_container = st.container()
        with button_container:
            # Modified button key to include company name for uniqueness
            if st.button("Generate Aggregated Table from Selected Sources", key=f"generate_table_button_{cleaned_search_query_name}"):
                st.session_state.last_selected_indices = [result['search_result_metadata']['search_result_index'] for index, result in enumerate(results_with_metadata) if st.session_state.get(f"exact_match_checkbox_{cleaned_search_query_name}_{index}")] # CORRECT KEY
                st.session_state.last_selected_non_exact_indices = [index_non_exact for index_non_exact, result_dict in enumerate(non_exact_match_results_metadata) if st.session_state.get(f"non_exact_result_checkbox_{cleaned_search_query_name}_{index_non_exact}")] # CORRECT KEY

                selected_company_data_list_combined = selected_company_data_list_exact + selected_company_data_list_non_exact
                selected_metadata_exact = [result_dict['search_result_metadata'] for index, result_dict in enumerate(results_with_metadata) if st.session_state.get(f"exact_match_checkbox_{cleaned_search_query_name}_{index}")] # CORRECT KEY
                selected_metadata_non_exact = [result_dict['search_result_metadata'] for index_non_exact, result_dict in enumerate(non_exact_match_results_metadata) if st.session_state.get(f"non_exact_result_checkbox_{cleaned_search_query_name}_{index_non_exact}")] # CORRECT KEY
                selected_metadata_combined = selected_metadata_exact + selected_metadata_non_exact

                await display_table_results(selected_company_data_list_combined, cleaned_search_query_name, selected_metadata_combined) # Removed asyncio.run

def display_non_exact_match_results(cleaned_search_query_name: str, non_exact_match_results_metadata: List[Dict[str, Any]]):
    """Displays non-exact match results in a separate section."""
    if non_exact_match_results_metadata:
        non_exact_raw_output_expander = st.expander(f"Non-Exact Match Results - Raw Output (for debug): {cleaned_search_query_name.title()}", expanded=False)
        with non_exact_raw_output_expander:
            st.info("These are non-exact match results and their raw extracted data. Check 'Potentially Relevant Non-Exact Match Results' section above for interactive selection.")
            for result_dict in non_exact_match_results_metadata:
                st.subheader(f"Source: {result_dict['search_result_metadata']['title']}")
                st.write(f"URL: {result_dict['search_result_metadata']['url']}")
                if 'company_data' in result_dict and result_dict['company_data']:
                    st.write("Extracted Company Data:")
                    st.write(result_dict['company_data'])
                else:
                    st.warning("No company data extracted from this result.")
                if 'selection_result' in result_dict:
                    st.write("Selection Agent Output:")
                    st.write(result_dict['selection_result'])
                st.markdown("---")


async def get_aggregated_company_data(company_data_for_table: List[Dict[str, Any]], from_exact_match: bool = True) -> tuple[CompanyDataOutput, bool]: # Modified to return tuple with flag
    """Aggregates a list of CompanyDataOutput dictionaries into a single CompanyDataOutput object."""
    aggregated_data = {}
    fields_to_aggregate = [
        "company_name", "company_url", "product_name", "product_type",
        "scientific_domain", "organization_type", "hq_locations",
        "description_abstract", "total_funding", "employee_count",
        "relevant_segments", "investor_name", "competitors"
    ]
    for field in fields_to_aggregate:
        aggregated_data[field] = None

    for company_data in company_data_for_table:
        for field in fields_to_aggregate:
            value = company_data.get(field)
            if value:
                if field in ["product_name", "hq_locations", "relevant_segments", "investor_name", "competitors", "company_url"]:
                    if aggregated_data[field] is None:
                        aggregated_data[field] = []
                    if isinstance(value, list):
                        aggregated_data[field].extend([item for item in value if item not in aggregated_data[field]])
                    else:
                        if value not in aggregated_data[field]:
                            aggregated_data[field].append(value)
                else:
                    if aggregated_data[field] is None:
                        aggregated_data[field] = value

    # Convert lists to sets to ensure uniqueness and then back to lists
    for field in ["company_url", "product_name", "hq_locations", "relevant_segments", "investor_name", "competitors"]:
        if aggregated_data[field] is not None:
            aggregated_data[field] = list(set(aggregated_data[field]))

    return CompanyDataOutput(**aggregated_data), from_exact_match # Return tuple with flag


def highlight_non_exact_match_row(row):
    '''
    Highlights the row in yellow if 'Non-Exact Match' is True, otherwise no highlight.
    '''
    is_non_exact = row['Data Source'] == 'Potentially Relevant (Non-Exact Match)'
    return ['background-color: yellow' if is_non_exact else '' for _ in row.index]


async def display_table_results(company_data_for_table: List[Dict[str, Any]], cleaned_search_query_name: str, search_result_metadata_list):
    """Displays the aggregated company data in a table format (single row) and source list, now using aggregated agent."""
    table_container = st.container()
    with table_container:
        if cleaned_search_query_name and cleaned_search_query_name != "unknown entity":
            entity_header_name = cleaned_search_query_name.title()
        else:
            entity_header_name = "Company Information"

        st.header(f"Entity: {entity_header_name} (Aggregated & Summarized from Selected Sources)") # Updated header
        st.warning("AI-Generated Company Data: Please verify critical information with official sources.")

        if company_data_for_table:
            aggregated_company_data_object, from_exact_match = await get_aggregated_company_data(company_data_for_table) # Aggregate data first, get flag

            with st.spinner("Summarizing and refining aggregated data..."): # Spinner for aggregated agent
                try:
                    aggregated_agent_output = await aggregated_company_data_agent.run(
                        user_prompt=f"Aggregated Company Data:\n{aggregated_company_data_object.model_dump_json()}",
                    )
                    final_company_data = aggregated_agent_output.data # Get the final refined data
                except Exception as e:
                    st.error(f"Error during aggregated data summarization: {e}")
                    final_company_data = aggregated_company_data_object # Fallback to non-summarized data in case of error

            df_aggregated_final = pd.DataFrame([final_company_data.model_dump()]) # Use final_company_data from aggregated agent
            df_aggregated_final.columns = [col.replace('_', ' ').title() for col in df_aggregated_final.columns]

            # Add 'Data Source' column to indicate exact or non-exact match source
            data_source_label = 'Exact Match' if from_exact_match else 'Potentially Relevant (Non-Exact Match)'
            df_aggregated_final['Data Source'] = data_source_label

            # Apply styling only if from_non_exact_match is True
            if not from_exact_match:
                styled_df = df_aggregated_final.style.apply(highlight_non_exact_match_row, axis=1)
                st.dataframe(styled_df) # Display styled dataframe
            else:
                st.dataframe(df_aggregated_final) # Display regular dataframe if exact match

            st.info("Rows highlighted in yellow indicate data aggregated from potentially relevant (non-exact match) sources.  Manual review on the 'Results' tab is recommended.") # Info message for highlighting

            # Replace expander with simple instruction to check results tab
            st.markdown("For detailed results and manual review, please check the **Results** tab.")


            st.markdown("**Sources Used for Aggregation (Selected Results):**")
            for metadata in search_result_metadata_list:
                st.write(f"- [{metadata['title']}]({metadata['url']})")
        else:
            st.info("No company data to display in table format (from selected results).")

# Modified function to handle list of company names and data
def display_exact_match_aggregated_table(company_names, aggregated_data_list, from_exact_match_list): # Added from_exact_match_list
    """Displays a single aggregated table for exact match results of all companies on the main tab."""
    exact_match_table_container = st.container()
    with exact_match_table_container:
        st.subheader(f"Aggregated Company Information Table for: {', '.join([name.title() for name in company_names])}") # Fixed duplicate subheader issue - removed one instance
        if aggregated_data_list:
            # Create a DataFrame from the list of aggregated data
            df_aggregated_all_companies_raw = pd.DataFrame([data.model_dump() for data in aggregated_data_list])
            df_aggregated_all_companies_raw.columns = [col.replace('_', ' ').title() for col in df_aggregated_all_companies_raw.columns]

            # Add 'Data Source' column
            data_source_labels = ['Exact Match' if is_exact else 'Potentially Relevant (Non-Exact Match)' for is_exact in from_exact_match_list]
            df_aggregated_all_companies_raw['Data Source'] = data_source_labels

            # Apply styling based on 'from_exact_match_list'
            def highlight_all_companies_row(row):
                is_non_exact_all = row['Data Source'] == 'Potentially Relevant (Non-Exact Match)'
                return ['background-color: yellow' if is_non_exact_all else '' for _ in row.index]

            styled_all_companies_df = df_aggregated_all_companies_raw.style.apply(highlight_all_companies_row, axis=1)
            st.dataframe(styled_all_companies_df)

            st.info("Rows highlighted in yellow indicate data aggregated from potentially relevant (non-exact match) sources. Manual review on the 'Results' tab is recommended.") # Info message for highlighting

            # Replace expander with simple instruction to check results tab
            st.markdown("For detailed results and manual review, please check the **Results** tab.")


        else:
            st.warning(f"No exact match results found for {', '.join(company_names)}. Please review results on the 'Results' tab for manual selection and aggregation.")

def display_results(grouped_results: Dict[str, List[Dict[str, Any]]], cleaned_search_query_name: str):
    """Displays raw grouped results for debugging in an expander."""
    raw_results_expander = st.expander(f"Raw Results (for debug): {cleaned_search_query_name.title() if cleaned_search_query_name else 'Search Results'}", expanded=False)
    with raw_results_expander:
        if cleaned_search_query_name and cleaned_search_query_name != "unknown entity":
            entity_header_name = cleaned_search_query_name.title()
        else:
            entity_header_name = "Company Information"

        for entity_name, entity_results_list in grouped_results.items():
            st.header(f"Entity: {entity_header_name} - Raw Results (for debug)")
            for result in entity_results_list:
                if 'error' in result:
                    st.error(f"Error processing result: {result['error']}")
                    continue

                company_data_dict = result.get('company_data', {})

                if company_data_dict:
                    st.subheader(f"Source: {result['search_result_metadata']['title']}")
                    st.write(company_data_dict)
                    st.write(f"URL: {result['search_result_metadata']['url']}")
                st.markdown("---")


### Streamlit UI Components - Tab Functions ###
def render_input_tab():
    """Renders the input tab for company name entry."""
    st.header("Company Search Input")

    input_text_display = st.container()
    company_entries_dict = {} # Use a dict to aggregate by company name, URLs will be a list
    with input_text_display:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Text Area Input (Company Names or URLs)") # Updated title
            input_text = st.text_area("Enter company names or website URLs (one per line or comma-separated):", placeholder="e.g., Apple, Google, microsoft.com or\nApple\nGoogle\nmicrosoft.com", label_visibility="visible") # Updated placeholder hint, added label_visibility
            input_lines = [item.strip() for item in input_text.splitlines() if item.strip()]
            input_comma_separated = [item.strip() for line in input_text.splitlines() for item in line.split(',') if item.strip()]
            all_inputs = list(dict.fromkeys(input_lines + input_comma_separated)) # Unique inputs

            for item in all_inputs:
                if re.match(r'^(http|https)://', item): # Check if it's a URL
                    url = item
                    company_name = None # Company name might be extracted later
                    if "company_from_url" not in company_entries_dict:
                        company_entries_dict["company_from_url"] = [] # Placeholder key for URL-only entries
                    company_entries_dict["company_from_url"].append({'name': company_name, 'urls': [url]}) # Store URL entry under placeholder key
                else: # Treat as company name
                    name = basename(item).lower()
                    if name not in company_entries_dict:
                        company_entries_dict[name] = [] # Initialize with empty URL list

        with col2:
            st.subheader("Excel Upload (Company Name and URL)")
            uploaded_file = st.file_uploader("Upload Excel file:", type=["xlsx", "xls"], label_visibility="visible") # added label_visibility
            excel_input_entries = []
            if uploaded_file:
                try:
                    df = pd.read_excel(uploaded_file)
                    if "Company Name" in df.columns or "Company URL" in df.columns: # Check for at least one of the columns
                        df['Company Name'] = df['Company Name'].fillna(None) # Handle missing names
                        df['Company URL'] = df['Company URL'].fillna(None) # Handle missing URLs

                        excel_input_entries = df[["Company Name", "Company URL"]].to_dict('records') # Keep all rows, even with missing names or URLs

                        # Clean company names and basenames, aggregate URLs
                        for entry in excel_input_entries:
                            company_name_raw = entry['Company Name']
                            url = entry['Company URL']

                            if pd.isna(company_name_raw) and pd.notna(url): # Only URL provided, no name
                                url_cleaned = url.strip()
                                if "company_from_url" not in company_entries_dict:
                                    company_entries_dict["company_from_url"] = []
                                if url_cleaned:
                                    company_entries_dict["company_from_url"].append({'name': None, 'urls': [url_cleaned]}) # Store URL entry with None name
                            elif pd.notna(company_name_raw): # Company name is provided
                                cleaned_name = basename(company_name_raw.strip()).lower()
                                url_cleaned = url.strip() if pd.notna(url) else None

                                if cleaned_name not in company_entries_dict:
                                    company_entries_dict[cleaned_name] = []
                                if url_cleaned and url_cleaned not in company_entries_dict[cleaned_name]: # Only add unique URLs
                                    company_entries_dict[cleaned_name].append(url_cleaned)
                    else:
                        st.error("Neither 'Company Name' nor 'Company URL' column found in Excel file. Please include at least one of these columns.")

                except Exception as e:
                    st.error(f"Error reading Excel: {e}")


        with col3:
            st.subheader("Data Editor (Company Name and URLs)") # Changed title
            data_editor_df = st.data_editor(pd.DataFrame({"Company Name": [""] * 3, "Company URL": [""] * 3}), num_rows="dynamic")

            for index, row in data_editor_df.iterrows():
                company_name_raw = row["Company Name"]
                company_url = row["Company URL"]

                if pd.isna(company_name_raw) and pd.notna(company_url): # URL but no name
                    url_cleaned = company_url.strip()
                    if "company_from_url" not in company_entries_dict:
                        company_entries_dict["company_from_url"] = []
                    if url_cleaned:
                        company_entries_dict["company_from_url"].append({'name': None, 'urls': [url_cleaned]}) # Store URL entry with None name

                elif pd.notna(company_name_raw) and company_name_raw.strip(): # Company name provided
                    cleaned_name = basename(company_name_raw.strip()).lower()
                    url_to_add = company_url.strip() if pd.notna(company_url) and company_url.strip() else None

                    if cleaned_name not in company_entries_dict:
                        company_entries_dict[cleaned_name] = []
                    if url_to_add and url_to_add not in company_entries_dict[cleaned_name]: # Only add unique URLs
                        company_entries_dict[cleaned_name].append(url_to_add)


        # Convert dict to list of dicts for easier processing later, URLs are now lists
        company_entries = []
        for name, entries in company_entries_dict.items():
            if name == "company_from_url": # Handle URL-only entries
                company_entries.extend(entries) # Extend directly if it's URL-only placeholder
            else: # Handle name-based entries
                company_entries.append({'name': name, 'urls': [url for url in entries if url] if entries else None}) # Create entry for each name

        input_text_display.markdown(f"**Companies to search:**")
        if company_entries:
            display_data = []
            for entry in company_entries:
                display_name = entry['name'].title() if entry['name'] else "Company from URL" # Display name or placeholder
                display_urls = ", ".join(entry['urls']) if entry['urls'] else 'Not provided'

                # Conditionally display "Company from URL" only if URLs are present and no name is given
                company_display_name = display_name if entry['name'] else ("Company from URL" if entry['urls'] else "")
                if company_display_name: # Only add to display if there's a name or URL-based company
                    display_data.append({'Company Name': company_display_name, 'Company URLs': display_urls})

            if display_data: # Only display dataframe if there's data to show
                df_companies_to_search = pd.DataFrame(display_data)
                st.dataframe(df_companies_to_search, hide_index=True)
            else:
                st.info("No companies or URLs entered yet.") # Keep info message even if no display data
        else:
            st.info("No companies or URLs entered yet.")


    use_test_data = st.checkbox("Use test data for search (first company only)", value=False, label_visibility="visible") # added label_visibility
    NUMBER_OF_SEARCH_RESULTS = st.number_input("Number of search results to retrieve per topic", value=5, step=1, label_visibility="visible") # added label_visibility

    all_companies_aggregated_table_container = st.container() # Container for all companies aggregated table

    if st.button("Search Companies", key="search_companies_button"):
        if not company_entries:
            st.warning("Please enter at least one company name or URL.")
        else:
            st.session_state.search_results_all_companies = {}
            st.session_state.exact_match_aggregated_data = {}
            st.session_state.has_exact_matches = {}
            # **Use session state lists instead of local variables:**
            st.session_state.aggregated_data_for_table_all_companies = []
            st.session_state.company_names_processed_for_table = []
            st.session_state.from_exact_match_all_companies = []
            aggregated_data_for_table_all_companies = [] # List to store aggregated data for all companies
            company_names_processed_for_table = [] # List to store company names with exact matches for table display
            from_exact_match_all_companies = [] # List to store from_exact_match flag for each company

            # Extract company names for display in the aggregated table header
            company_names_for_display = [entry['name'] if entry['name'] else "Company from URL Input" for entry in company_entries] # Updated display name

            with st.spinner(f"Searching for information about {', '.join([name.title() if name != 'company_from_url' else 'Company from URL Input' for name in company_names_for_display])}..."): # Updated spinner display
                async def search_company(company_entry, index):
                    company_name = company_entry['name']
                    company_urls = company_entry['urls'] # Now a list of URLs (or None)
                    if not company_name and company_urls: # If no name but URLs, use "Company from URL" as name placeholder for session state etc.
                        company_name = "company_from_url" # Use consistent placeholder key

                    try:
                        use_test = use_test_data if index == 0 else False
                        # Initialize ...
                        st.session_state.search_results_all_companies[company_name] = { # Use placeholder name if no name given
                            'results_with_metadata': [],
                            'non_exact_match_results_metadata': [],
                            'selection_results': [],
                            'grouped_results_dict': {},
                            'cleaned_search_query_name': None,
                            'company_features': None,
                            'company_urls': company_urls, # Store list of URLs
                            'extracted_content_gold_standard': None
                        }
                        results = await search_company_summary(company_name, company_urls=company_urls, use_test_data=use_test, NUMBER_OF_SEARCH_RESULTS=NUMBER_OF_SEARCH_RESULTS) # Pass company_urls list
                        if results[2]:  # Check if results_with_metadata is not empty
                            return company_name, results
                        else:
                            return company_name, None
                    except Exception as e:
                        logger.error(f"Search error for {company_name}: {str(e)}", exc_info=True)
                        return company_name, str(e)

                search_tasks = [search_company(entry, idx) for idx, entry in enumerate(company_entries)]

                async def run_searches():
                    return await asyncio.gather(*search_tasks)

                search_results = asyncio.run(run_searches())

                for company_name, result in search_results:
                    if isinstance(result, str):
                        st.error(f"An error occurred during search for {company_name}: {result}")
                    elif result is None:
                        st.info(f"No relevant company data found in search results for {company_name}.")
                    else:
                        grouped_results, cleaned_search_query_name, results_with_metadata, selection_results, non_exact_match_results_metadata, extracted_content_gold_standard_data = result
                        st.session_state.search_results_all_companies[company_name]['results_with_metadata'] = results_with_metadata # Update existing dict
                        st.session_state.search_results_all_companies[company_name]['non_exact_match_results_metadata'] = non_exact_match_results_metadata
                        st.session_state.search_results_all_companies[company_name]['selection_results'] = selection_results
                        st.session_state.search_results_all_companies[company_name]['grouped_results_dict'] = grouped_results
                        st.session_state.search_results_all_companies[company_name]['cleaned_search_query_name'] = cleaned_search_query_name
                        st.session_state.search_results_all_companies[company_name]['extracted_content_gold_standard'] = extracted_content_gold_standard_data

                        # Extract exact match data and aggregate for main tab display
                        exact_match_results_metadata = [res for res in results_with_metadata if res.get("extracted_company_name") == cleaned_search_query_name]
                        exact_match_company_data_list = [res['company_data'] for res in exact_match_results_metadata if res.get('company_data')]

                        aggregated_exact_match_data = None # Initialize
                        from_exact_match_flag = True # Default to True, might be changed below

                        try:
                            if exact_match_company_data_list:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                aggregated_exact_match_data_obj, from_exact_match_flag = loop.run_until_complete(get_aggregated_company_data(exact_match_company_data_list)) # Get flag
                                aggregated_exact_match_data = aggregated_exact_match_data_obj
                                loop.close()
                            else:
                                # If no exact matches, try to aggregate from selected non-exact matches
                                non_exact_match_selected_data = []
                                for res_non_exact in non_exact_match_results_metadata:
                                    if res_non_exact.get('selection_result') and res_non_exact['selection_result'].get('include_in_table'):
                                        if res_non_exact.get('company_data'):
                                            non_exact_match_selected_data.append(res_non_exact['company_data'])
                                if non_exact_match_selected_data:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    aggregated_non_exact_data_obj, from_exact_match_flag = loop.run_until_complete(get_aggregated_company_data(non_exact_match_selected_data, from_exact_match=False)) # Set flag to False
                                    aggregated_exact_match_data = aggregated_non_exact_data_obj # Use non-exact aggregated data
                                    from_exact_match_flag = False # Ensure flag is False
                                    loop.close()
                                else:
                                    aggregated_exact_match_data = None
                                    from_exact_match_flag = True # Still default to True if nothing is aggregated

                            st.session_state.exact_match_aggregated_data[company_name] = aggregated_exact_match_data
                            if aggregated_exact_match_data:
                                # **Append to session state lists:**
                                st.session_state.aggregated_data_for_table_all_companies.append(aggregated_exact_match_data)
                                st.session_state.company_names_processed_for_table.append(company_name)
                                st.session_state.from_exact_match_all_companies.append(from_exact_match_flag)
                                # print(f"DEBUG: Added {company_name} to table data") # DEBUG PRINT
                            else:
                                st.info(f"No aggregated data generated for {company_name}.")
                                # print(f"DEBUG: No aggregated data for {company_name}") # DEBUG PRINT

                        except Exception as e:
                            logger.error(f"Error during aggregated data summarization for {company_name}: {str(e)}")
                            st.session_state.exact_match_aggregated_data[company_name] = None
                        st.session_state.has_exact_matches[company_name] = bool(exact_match_company_data_list)


    # --- MOVED TABLE DISPLAY OUTSIDE THE BUTTON BLOCK ---
    all_companies_aggregated_table_container = st.container()
    with all_companies_aggregated_table_container:
        if st.session_state.company_names_processed_for_table and st.session_state.aggregated_data_for_table_all_companies:
            # --- [NEW] Instruction above the table ---
            st.markdown("**:warning: Review Highlighted Rows:** Rows in yellow indicate data from potentially relevant (non-exact match) sources. **Please manually verify these rows.**")

            display_exact_match_aggregated_table(
                st.session_state.company_names_processed_for_table,
                st.session_state.aggregated_data_for_table_all_companies,
                st.session_state.from_exact_match_all_companies
            )
            # --- [MODIFIED] Info message below table ---
            st.info("For detailed review of each source and to verify highlighted information, please switch to the **Results** tab and select the company from the dropdown. On the 'Results' tab, you can examine the original sources and selection reasons in detail.")

        elif company_entries and not st.session_state.company_names_processed_for_table:
            st.info("No aggregated company information to display yet. Please click 'Search Companies' to generate the table after entering company names/URLs.")
        elif not company_entries:
            st.info("Enter company names or URLs to begin the search and display aggregated information here.")
    # --- END MOVED TABLE DISPLAY ---

    # --- [MODIFIED SECTION - ENCLOSED IN EXPANDER] Review Non-Exact Match Sources ---
    non_exact_review_expander = st.expander("Review Potentially Relevant Sources (Non-Exact Matches)", expanded=False) # Enclose in expander
    with non_exact_review_expander: # Use the expander as the container
        if st.session_state.company_names_processed_for_table and st.session_state.aggregated_data_for_table_all_companies:
            st.write("Below are the sources that contributed to the highlighted yellow rows in the table above. Please review these to verify the information.")

            for company_index, company_name in enumerate(st.session_state.company_names_processed_for_table):
                from_exact_match_flag = st.session_state.from_exact_match_all_companies[company_index]
                aggregated_data = st.session_state.aggregated_data_for_table_all_companies[company_index]

                if not from_exact_match_flag: # Only show review for non-exact match data
                    st.markdown(f"#### {company_name.title() if company_name != 'company_from_url' else 'Company from URL Input'}") # Company Name Header

                    # Get the search result metadata list (assuming you have access to it here - if not, you might need to pass it to render_input_tab or store in session state)
                    # For now, let's assume you can retrieve it from session state based on company_name:
                    search_result_metadata_list = st.session_state.search_results_all_companies[company_name].get('results_with_metadata', []) # Adjust key if needed

                    non_exact_match_sources = []
                    if search_result_metadata_list:
                        # Filter for non-exact match sources based on your logic (e.g., check 'Data Source' in company_data, or selection_result)
                        # For simplicity, let's assume we can check if the 'Data Source' in the aggregated data was 'Potentially Relevant...'
                        #  This is a simplification and might need adjustment based on your actual data structure.
                        # Corrected condition: Check from_exact_match_flag directly
                        if not from_exact_match_flag:
                            # If the WHOLE aggregated data is flagged as non-exact, we'll show ALL sources used for it.  More refined approach needed if you want source-level tracking.
                            # For now, show ALL metadata for this company as potentially contributing to the non-exact data.
                            non_exact_match_sources = search_result_metadata_list #  Again, this might need refinement.
                    if non_exact_match_sources:
                        for source_metadata in non_exact_match_sources:
                            with st.container():
                                st.subheader(f"Source: {source_metadata['search_result_metadata']['title']}")
                                st.write(f"**URL:** {source_metadata['search_result_metadata']['url']}")
                                company_data_snippet = source_metadata.get('company_data', {})
                                if company_data_snippet:
                                    st.write("**Extracted Company Name:**", company_data_snippet.get('company_name', 'N/A'))
                                    st.write("**Product Type:**", company_data_snippet.get('product_type', 'N/A'))
                                    st.write("**Description:**", company_data_snippet.get('description_abstract', 'N/A'))
                                else:
                                    st.write("No company data extracted from this source.")
                                st.markdown("---")
                    else:
                        st.info("No potentially relevant sources to review for this company (or sources not found).")
        else:
            st.info("No aggregated data available yet to review non-exact match sources.") # Info when no data

    # --- [END] Review Non-Exact Match Sources ---


def render_results_tab():
    """Renders the results tab to display company search results."""
    st.header("Company Search Results")
    if 'search_results_all_companies' in st.session_state and st.session_state.search_results_all_companies:
        all_company_results = st.session_state.search_results_all_companies
        company_names_list = list(all_company_results.keys())

        if 'selected_company' not in st.session_state:
            st.session_state.selected_company = company_names_list[0] if company_names_list else None

        if company_names_list:
            st.subheader("Select Company to View Results:")
            st.session_state.selected_company = st.selectbox("", company_names_list, index=company_names_list.index(st.session_state.selected_company) if st.session_state.selected_company in company_names_list else 0, key="company_selectbox", label_visibility="visible") # Added key to selectbox, added label_visibility

        selected_company_name = st.session_state.selected_company

        if selected_company_name:
            results_data = all_company_results[selected_company_name]
            st.subheader(f"Results for: {selected_company_name.title() if selected_company_name != 'company_from_url' else 'Company from URL Input'}") # Updated display name in results tab
            results_with_metadata = results_data['results_with_metadata']
            non_exact_match_results_metadata = results_data['non_exact_match_results_metadata']
            selection_results = results_data['selection_results']
            grouped_results = results_data['grouped_results_dict']
            cleaned_search_query_name = results_data['cleaned_search_query_name']

            if results_with_metadata:
                asyncio.run(display_interactive_results(cleaned_search_query_name, selection_results))
                display_results(grouped_results, cleaned_search_query_name)
            else:
                st.info(f"No relevant company data found in search results for {selected_company_name.title() if selected_company_name != 'company_from_url' else 'Company from URL Input'}.") # Updated info message
        elif company_names_list:
            st.info("Please select a company from the dropdown to view results.")
    else:
        st.info("No company search results available yet. Please use the Input tab to search for companies.")


def main():
    st.set_page_config(
        page_title="Company Search and Analysis",
        page_icon="🔍",
        layout="wide"
    )

    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Main"

    # Initialize lists in session state
    if 'company_names_processed_for_table' not in st.session_state:
        st.session_state.company_names_processed_for_table = []
    if 'aggregated_data_for_table_all_companies' not in st.session_state:
        st.session_state.aggregated_data_for_table_all_companies = []
    if 'from_exact_match_all_companies' not in st.session_state:
        st.session_state.from_exact_match_all_companies = []


    tab1, tab2 = st.tabs(["Main", "Results"])

    with tab1:
        render_input_tab()

    with tab2:
        render_results_tab()


if __name__ == "__main__":
    main()