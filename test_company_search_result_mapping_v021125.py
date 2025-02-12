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
You are now evaluating a non-exact match search result.

INPUT:
search_query_company_name_cleaned: Cleaned, lowercased search query company name.
exact_match_results_metadata: List of metadata for search results already identified as exact matches.
search_result: A single processed non-exact match search result (with metadata, company_data, extracted_company_name, errors).

INCLUSION CRITERIA (for Table):
1. Relevance to Searched Company (Primary):
Does this non-exact match result provide relevant and useful information related to the company the user searched for ({search_query_company_name_cleaned})?
Consider if it offers:
Complementary Information:  Details not found in exact match results.
Related Entities: Information about subsidiaries, parent companies, key personnel, or closely associated organizations.
Broader Context: Market analysis, industry trends, etc., directly relevant to the searched company.
It DOES NOT need to be an exact name match.  Relevance is key.

2. Avoid Redundancy (Secondary):
If the information in this non-exact match result is already comprehensively covered in the exact_match_results_metadata, it might be less valuable to include.  Prioritize new and unique information.

3. Avoid Errors: Reject results with errors (error field present).

OUTPUT: SingleResultSelectionOutput object with:
include_in_table: True if this result should be included in the aggregated table because it provides relevant and useful information about the searched company (considering context and avoiding redundancy). False otherwise.
reason: Justify your decision, focusing on Relevance (Criterion 1), redundancy (Criterion 2), and any errors. Explain why it is or is not valuable to include in the table.

Determine if this non-exact match result is valuable to include in the company information table. Focus on relevance and usefulness, not just name matching.""",
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



async def process_content(content_text: str, title: str, url: str) -> Dict[str, Any]:
    """
    Processes the input content text to extract company information.

    Args:
        content_text: The input text content to process (raw content if available).
        title: The title of the search result.
        url: The URL of the search result.
        search_query_company_name: The original company name used in the search query.

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


async def search_company_summary(company_name: str, use_test_data: bool = False, NUMBER_OF_SEARCH_RESULTS: int = 5) -> tuple[Dict[str, List[Dict[str, Any]]], str, List[Dict[str, Any]], List[RunResult[SingleResultSelectionOutput]], List[Dict[str, Any]]]: # RETURN non_exact_match_results_metadata
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
            'selection_results': []
        }
    search_results_state = st.session_state.search_results_state # Get state *after* initialization


    grouped_results_dict = search_results_state['grouped_results_dict']
    cleaned_search_query_name = search_results_state['cleaned_search_query_name']
    results_with_metadata = search_results_state['results_with_metadata']
    non_exact_match_results_metadata = search_results_state['non_exact_match_results_metadata']
    selection_results = search_results_state['selection_results']


    exact_match_results_metadata = [] # Initialize outside try block


    async with sem:
        try:
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
                selection_tasks.append(
                    search_result_selection_agent.run(
                        user_prompt=f"Search Query Company Name Cleaned: {cleaned_search_query_name}\nExact Match Results Metadata: {exact_match_results_metadata}\nSearch Result: {result_dict}"
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


        # Store all variables in session state
        st.session_state.search_results_state['grouped_results_dict'] = grouped_results_dict
        st.session_state.search_results_state['cleaned_search_query_name'] = cleaned_search_query_name
        st.session_state.search_results_state['results_with_metadata'] = results_with_metadata
        st.session_state.search_results_state['non_exact_match_results_metadata'] = non_exact_match_results_metadata # Store non-exact matches
        st.session_state.search_results_state['selection_results'] = selection_results
        st.session_state.search_results_all_companies[company_name]['company_features'] = company_features # Store company features in session state
        return grouped_results_dict, cleaned_search_query_name, results_with_metadata, selection_results, non_exact_match_results_metadata # RETURN it here


### Streamlit UI Components - Top level functions (Display Functions moved to top level) ###
async def display_interactive_results(cleaned_search_query_name: str, selection_results: List[RunResult[SingleResultSelectionOutput]]):
    """Displays search results with table and checkboxes for interactive selection."""
    results_with_metadata = st.session_state.search_results_all_companies[st.session_state.selected_company].get("results_with_metadata", []) # Access from the correct company in session state
    non_exact_match_results_metadata = st.session_state.search_results_all_companies[st.session_state.selected_company].get("non_exact_match_results_metadata", []) # Access from the correct company in session state

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
                    is_selected = st.checkbox(f"Select Source: {result_dict['search_result_metadata']['title']} (Exact Match)", key=checkbox_key, value=default_checkbox_value)
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

                is_selected = st.checkbox(f"Select Source: {result_dict['search_result_metadata']['title']} (Potentially Relevant)", key=checkbox_key, value=default_checkbox_value)
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
            if st.button("Generate Aggregated Table from Selected Sources", key=f"generate_table_button_{cleaned_search_query_name}"): # CORRECT KEY
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


async def get_aggregated_company_data(company_data_for_table: List[Dict[str, Any]]) -> CompanyDataOutput:
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

    return CompanyDataOutput(**aggregated_data)


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
            aggregated_company_data_object = await get_aggregated_company_data(company_data_for_table) # Aggregate data first

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
            st.dataframe(df_aggregated_final)

            st.markdown("**Sources Used for Aggregation (Selected Results):**")
            for metadata in search_result_metadata_list:
                st.write(f"- [{metadata['title']}]({metadata['url']})")
        else:
            st.info("No company data to display in table format (from selected results).")

# Modified function to handle list of company names and data
def display_exact_match_aggregated_table(company_names, aggregated_data_list):
    """Displays a single aggregated table for exact match results of all companies on the main tab."""
    exact_match_table_container = st.container()
    with exact_match_table_container:
        st.subheader(f"Exact Match Aggregated Result Table for: {', '.join([name.title() for name in company_names])}")
        if aggregated_data_list:
            # Create a DataFrame from the list of aggregated data
            df_aggregated_all_companies = pd.DataFrame([data.model_dump() for data in aggregated_data_list])
            df_aggregated_all_companies.columns = [col.replace('_', ' ').title() for col in df_aggregated_all_companies.columns]
            st.dataframe(df_aggregated_all_companies)
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
    company_names = []
    with input_text_display:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Text Area Input")
            company_names_text = st.text_area("Enter company names (one per line or comma-separated):", placeholder="e.g., Apple, Google, Microsoft or\nApple\nGoogle\nMicrosoft") # Added placeholder hint
            text_input_names_lines = [name.strip() for name in company_names_text.splitlines() if name.strip()]
            text_input_names_comma = [name.strip() for line in company_names_text.splitlines() for name in line.split(',') if name.strip()]
            text_input_names = list(set(text_input_names_lines + text_input_names_comma))

        with col2:
            st.subheader("Excel Upload")
            uploaded_file = st.file_uploader("Upload Excel file:", type=["xlsx", "xls"])
            excel_input_names = []
            if uploaded_file:
                try:
                    df = pd.read_excel(uploaded_file)
                    if "Company Name" in df.columns:
                        excel_input_names = df["Company Name"].dropna().tolist()
                    else:
                        st.error("'Company Name' column not found.")
                except Exception as e:
                    st.error(f"Error reading Excel: {e}")

        with col3:
            st.subheader("Data Editor")
            data_editor_df = st.data_editor(pd.DataFrame({"Company Name": [""] * 3}), num_rows="dynamic")
            data_editor_names = data_editor_df["Company Name"].dropna().tolist()

        company_names = list(set(text_input_names + excel_input_names + data_editor_names))
        company_names = [name for name in company_names if name.strip()]
        input_text_display.markdown(f"**Companies to search:**")
        for name in company_names:
            input_text_display.markdown(f"- {name}")
    use_test_data = st.checkbox("Use test data for search (first company only)", value=False)
    NUMBER_OF_SEARCH_RESULTS = st.number_input("Number of search results to retrieve per topic", value=5, step=1)

    all_companies_aggregated_table_container = st.container() # Container for all companies aggregated table

    if st.button("Search Companies", key="search_companies_button"):
        if not company_names:
            st.warning("Please enter at least one company name.")
        else:
            st.session_state.search_results_all_companies = {}
            st.session_state.exact_match_aggregated_data = {} # Initialize storage for exact match aggregated data
            st.session_state.has_exact_matches = {} # Initialize flag for exact matches
            aggregated_data_for_table_all_companies = [] # List to store aggregated data for all companies
            company_names_processed_for_table = [] # List to store company names with exact matches for table display

            with st.spinner(f"Searching for information about {', '.join(company_names)}..."):
                async def search_company(company_name, index):
                    try:
                        use_test = use_test_data if index == 0 else False
                        # Initialize company_features before calling search_company_summary
                        st.session_state.search_results_all_companies[company_name] = {
                            'results_with_metadata': [],
                            'non_exact_match_results_metadata': [],
                            'selection_results': [],
                            'grouped_results_dict': {},
                            'cleaned_search_query_name': None,
                            'company_features': None
                        }
                        results = await search_company_summary(company_name, use_test_data=use_test, NUMBER_OF_SEARCH_RESULTS=NUMBER_OF_SEARCH_RESULTS)
                        if results[2]:  # Check if results_with_metadata is not empty
                            return company_name, results
                        else:
                            return company_name, None
                    except Exception as e:
                        logger.error(f"Search error for {company_name}: {str(e)}", exc_info=True)
                        return company_name, str(e)

                search_tasks = [search_company(name, idx) for idx, name in enumerate(company_names)]

                async def run_searches():
                    return await asyncio.gather(*search_tasks)

                search_results = asyncio.run(run_searches())

                for company_name, result in search_results:
                    if isinstance(result, str):
                        st.error(f"An error occurred during search for {company_name}: {result}")
                    elif result is None:
                        st.info(f"No relevant company data found in search results for {company_name}.")
                    else:
                        grouped_results, cleaned_search_query_name, results_with_metadata, selection_results, non_exact_match_results_metadata = result
                        # st.session_state.search_results_all_companies[company_name] = { # REMOVE THIS LINE - initialized before calling search_company_summary
                        #     'results_with_metadata': results_with_metadata,
                        #     'non_exact_match_results_metadata': non_exact_match_results_metadata,
                        #     'selection_results': selection_results,
                        #     'grouped_results_dict': grouped_results,
                        #     'cleaned_search_query_name': cleaned_search_query_name,
                        #     'company_features': None # Initialize company_features here
                        # }
                        st.session_state.search_results_all_companies[company_name]['results_with_metadata'] = results_with_metadata # Update existing dict
                        st.session_state.search_results_all_companies[company_name]['non_exact_match_results_metadata'] = non_exact_match_results_metadata
                        st.session_state.search_results_all_companies[company_name]['selection_results'] = selection_results
                        st.session_state.search_results_all_companies[company_name]['grouped_results_dict'] = grouped_results
                        st.session_state.search_results_all_companies[company_name]['cleaned_search_query_name'] = cleaned_search_query_name


                        # Extract exact match data and aggregate for main tab display
                        exact_match_results_metadata = [res for res in results_with_metadata if res.get("extracted_company_name") == cleaned_search_query_name]
                        exact_match_company_data_list = [res['company_data'] for res in exact_match_results_metadata if res.get('company_data')]
                        try:
                            if exact_match_company_data_list:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                aggregated_exact_match_data = loop.run_until_complete(get_aggregated_company_data(exact_match_company_data_list))
                                loop.close()
                            else:
                                aggregated_exact_match_data = None
                            st.session_state.exact_match_aggregated_data[company_name] = aggregated_exact_match_data
                            if aggregated_exact_match_data:
                                aggregated_data_for_table_all_companies.append(aggregated_exact_match_data)
                                company_names_processed_for_table.append(company_name)
                            else:
                                st.info(f"No aggregated exact match data generated for {company_name}.") # Debug info if no aggregated data

                        except Exception as e:
                            logger.error(f"Error during aggregated data summarization for {company_name}: {str(e)}")
                            st.session_state.exact_match_aggregated_data[company_name] = None
                        st.session_state.has_exact_matches[company_name] = bool(exact_match_company_data_list)

            st.write("company_names_processed_for_table:", company_names_processed_for_table) # Debug print
            st.write("aggregated_data_for_table_all_companies:", aggregated_data_for_table_all_companies) # Debug print

            with all_companies_aggregated_table_container:
                st.subheader(f"Aggregated Company Information Table for: {', '.join([name.title() for name in company_names])}")
                if aggregated_data_for_table_all_companies:
                    df_aggregated_all_companies = pd.DataFrame([data.model_dump() for data in aggregated_data_for_table_all_companies])
                    df_aggregated_all_companies.columns = [col.replace('_', ' ').title() for col in df_aggregated_all_companies.columns]
                    st.dataframe(df_aggregated_all_companies)
                else:
                    st.warning(f"No aggregated data found for {', '.join(company_names)}.")




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
            st.session_state.selected_company = st.selectbox("", company_names_list, index=company_names_list.index(st.session_state.selected_company) if st.session_state.selected_company in company_names_list else 0, key="company_selectbox") # Added key to selectbox

        selected_company_name = st.session_state.selected_company

        if selected_company_name:
            results_data = all_company_results[selected_company_name]
            st.subheader(f"Results for: {selected_company_name}")
            results_with_metadata = results_data['results_with_metadata']
            non_exact_match_results_metadata = results_data['non_exact_match_results_metadata']
            selection_results = results_data['selection_results']
            grouped_results = results_data['grouped_results_dict']
            cleaned_search_query_name = results_data['cleaned_search_query_name']

            if results_with_metadata:
                asyncio.run(display_interactive_results(cleaned_search_query_name, selection_results))
                display_results(grouped_results, cleaned_search_query_name)
            else:
                st.info(f"No relevant company data found in search results for {selected_company_name}.")
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

    tab1, tab2 = st.tabs(["Main", "Results"])

    with tab1:
        render_input_tab()

    with tab2:
        render_results_tab()

if __name__ == "__main__":
    main()