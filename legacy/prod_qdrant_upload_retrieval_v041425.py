# -*- coding: utf-8 -*-
import logging
import asyncio
import streamlit as st
from typing import List, Dict, Any, Optional, Union, Tuple, Coroutine
from pathlib import Path
from tempfile import NamedTemporaryFile
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http import exceptions as qdrant_exceptions
from pydantic import BaseModel, Field
from enum import Enum
import os
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import time
# Removed nest_asyncio import
import hashlib
import pandas as pd
import httpx
import io
import threading
from concurrent.futures import Future, TimeoutError
import uuid
# --- PDF Handling ---
try:
    import PyPDF2
except ImportError:
    st.warning("PyPDF2 not installed. PDF text extraction will be basic. Install with: pip install pypdf2")
    PyPDF2 = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s", # Added threadName
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
if "username" not in st.session_state:
    st.session_state.username = "guest"
USERNAME = st.session_state.username
UNIFIED_COLLECTION_NAME = f"parsely_google_gemini_{USERNAME}"
QDRANT_URL = st.secrets.get("qdrant_url", "http://localhost:6333")
QDRANT_API_KEY = st.secrets.get("qdrant_api_key", None)
GEMINI_API_KEY = st.secrets.get("GOOGLE_AI_STUDIO")

if not GEMINI_API_KEY:
    st.error("GOOGLE_AI_STUDIO secret not found. Please set it in your Streamlit secrets.")
    st.stop()

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
EMBEDDING_DIM = 768
logger.info(f"Using Embedding Model: {EMBEDDING_MODEL_NAME} with Dimension: {EMBEDDING_DIM}")

# --- Set Environment Variable for Pydantic-AI GeminiModel Auth ---
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
logger.info("Set GOOGLE_API_KEY environment variable for pydantic-ai.")

# --- Google GenAI Client Initialization ---
# (Ensure this is thread-safe - google-genai client generally is)
@st.cache_resource
def configure_genai_client():
    """Configures and returns the Google GenAI client."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    except Exception as e:
        st.error(f"Failed to configure Google GenAI client: {e}")
        logger.error(f"Google GenAI Configuration Error: {e}", exc_info=True)
        return False

genai_configured = configure_genai_client()
if not genai_configured:
    st.stop()

# --- Pydantic AI Model Initialization ---
try:
    gemini_model = GeminiModel('gemini-2.0-flash')
    logger.info(f"Using QA model: {gemini_model.model_name}")
except Exception as e:
    st.error(f"Failed to initialize Gemini QA model: {e}")
    logger.error(f"Gemini QA Model Error: {e}", exc_info=True)
    st.stop()


# --- Global Async Task Manager ---
class AsyncTaskManager:
    """
    Manages a dedicated background thread with an asyncio event loop
    to run coroutines submitted from other threads (like Streamlit's main thread).
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(AsyncTaskManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initializes the manager, starts the background thread and loop."""
        if hasattr(self, '_initialized') and self._initialized:
            return

        with self._lock:
            if hasattr(self, '_initialized') and self._initialized:
                return

            logger.info("Initializing AsyncTaskManager...")
            self._loop: Optional[asyncio.AbstractEventLoop] = None
            self._thread: Optional[threading.Thread] = None
            self._loop_ready = threading.Event()
            self._start_background_loop()
            self._initialized = True
            logger.info("AsyncTaskManager initialized.")

    def _run_loop(self):
        """Target function for the background thread."""
        try:
            logger.info(f"Background asyncio thread started (TID: {threading.get_ident()}).")
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            logger.info(f"Event loop {id(self._loop)} created and set for background thread.")
            self._loop_ready.set()
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Exception in background asyncio loop: {e}", exc_info=True)
        finally:
            if self._loop and self._loop.is_running():
                logger.info("Closing background event loop...")
                try:
                    # Gracefully shutdown async generators
                    self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                    # Close pending tasks (optional, but can help)
                    tasks = asyncio.all_tasks(self._loop)
                    for task in tasks:
                        task.cancel()
                    self._loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                except Exception as shutdown_err:
                     logger.error(f"Error during loop shutdown: {shutdown_err}")
                finally:
                    self._loop.close()
            logger.info(f"Background asyncio thread finished (TID: {threading.get_ident()}).")
            self._loop = None

    def _start_background_loop(self):
        """Starts the background thread if not already running."""
        if self._thread is None or not self._thread.is_alive():
            logger.info("Starting background thread for asyncio loop...")
            self._loop_ready.clear()
            self._thread = threading.Thread(target=self._run_loop, name="AsyncTaskManagerThread", daemon=True)
            self._thread.start()
            if not self._loop_ready.wait(timeout=10):
                 logger.error("Background asyncio loop failed to start within timeout!")
                 raise RuntimeError("AsyncTaskManager background loop failed to initialize.")
            else:
                 logger.info(f"Background loop {id(self._loop)} reported ready.")
        else:
             logger.info("Background thread already running.")

    def submit_task(self, coro: Coroutine) -> Future:
        """Submits a coroutine to be executed on the background event loop."""
        if not self._loop or not self._loop.is_running():
             logger.error("AsyncTaskManager: Background loop is not running. Cannot submit task.")
             raise RuntimeError("AsyncTaskManager background loop is not operational.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run_sync(self, coro: Coroutine, timeout: Optional[float] = 180.0) -> Any:
        """Runs a coroutine on the background loop and blocks until it completes."""
        if not self._initialized or not self._loop or not self._thread or not self._thread.is_alive():
            logger.warning("AsyncTaskManager is not fully initialized or background thread died. Attempting to restart.")
            try:
                # Reset state and attempt restart (use with caution)
                self._initialized = False
                self.__init__() # Re-run initialization logic
            except Exception as e:
                 logger.error(f"Failed to restart AsyncTaskManager: {e}")
                 raise RuntimeError("AsyncTaskManager failed to recover.") from e

        future = self.submit_task(coro)
        try:
            result = future.result(timeout=timeout)
            return result
        except TimeoutError:
            logger.error(f"Operation timed out after {timeout} seconds for coroutine: {coro.__name__}")
            future.cancel()
            raise
        except Exception as e:
            # Log the exception originating from the coroutine
            logger.error(f"Exception occurred in awaited coroutine ({coro.__name__}): {e}", exc_info=True)
            # Check if it's a cancelled error from our side
            if isinstance(e, asyncio.CancelledError):
                raise TimeoutError(f"Task {coro.__name__} was cancelled (likely due to timeout).") from e
            raise # Re-raise the original exception

    def stop(self):
        """Stops the background event loop and waits for the thread to join."""
        # This might be called by Streamlit's resource cleanup implicitly
        logger.info("Attempting to stop AsyncTaskManager...")
        if hasattr(self, '_loop') and self._loop and self._loop.is_running():
            logger.info(f"Requesting stop for background loop {id(self._loop)}...")
            self._loop.call_soon_threadsafe(self._loop.stop)
        else:
            logger.info("Background loop already stopped or not initialized.")

        if hasattr(self, '_thread') and self._thread and self._thread.is_alive():
            logger.info(f"Waiting for background thread (TID: {self._thread.ident}) to join...")
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("Background thread did not join cleanly.")
            else:
                logger.info("Background thread joined.")
        self._thread = None
        if hasattr(self, '_initialized'):
            self._initialized = False
        # Clear the singleton instance only if we are sure it's safe
        # Be cautious with singleton modification if multiple sessions might exist
        # AsyncTaskManager._instance = None
        logger.info("AsyncTaskManager stop sequence completed.")

@st.cache_resource(ttl=3600) # Cache for 1 hour
def get_async_manager():
    """Gets the singleton instance of the AsyncTaskManager."""
    logger.info("Accessing AsyncTaskManager instance via st.cache_resource.")
    # __init__ handles the singleton logic and initialization
    manager = AsyncTaskManager()
    return manager


# --- Core Functions (remain mostly the same, but are now async) ---

# --- Google GenAI Embedding Function ---
async def generate_embedding_google(
    texts: Union[str, List[str]],
    task_type: str,
    model_name: str = EMBEDDING_MODEL_NAME,
    title: Optional[str] = None
    ) -> List[List[float]]:
    """Generates embeddings using the Google Generative AI SDK. (Async)"""
    if not texts: return []
    input_texts = [texts] if isinstance(texts, str) else texts
    if not input_texts: return []

    try:
        embed_params = {
            "model": model_name,
            "content": input_texts,
            "task_type": task_type,
        }
        if task_type == "RETRIEVAL_DOCUMENT" and title:
            embed_params["title"] = title

        # Use the async SDK call directly
        result = await genai.embed_content_async(**embed_params)

        embeddings_list = []
        if 'embedding' in result and isinstance(result['embedding'], list):
            if result['embedding'] and isinstance(result['embedding'][0], list):
                 embeddings_list = result['embedding']
            elif result['embedding'] and isinstance(result['embedding'][0], dict) and 'values' in result['embedding'][0]:
                 embeddings_list = [e['values'] for e in result['embedding']]
        elif 'embeddings' in result and isinstance(result['embeddings'], list):
             if result['embeddings'] and isinstance(result['embeddings'][0], dict) and 'values' in result['embeddings'][0]:
                 embeddings_list = [e['values'] for e in result['embeddings']]

        if not embeddings_list:
            logger.error(f"Embedding generation failed. Could not extract embeddings. Response: {result}")
            # Avoid showing streamlit error directly from async func if possible
            # Let the caller handle UI feedback based on the return value
            return [] # Indicate failure

        if len(embeddings_list) != len(input_texts):
             logger.warning(f"Mismatch text count ({len(input_texts)}) vs embeddings count ({len(embeddings_list)})")
             return [] # Indicate failure

        return embeddings_list

    except google_exceptions.GoogleAPIError as api_err:
        logger.error(f"Google API error during embedding: {api_err}", exc_info=True)
        # Propagate a generic error message or the API error itself
        raise RuntimeError(f"Embedding failed: Google API Error - {api_err.message}") from api_err
    except Exception as e:
        logger.error(f"Error generating Google embeddings: {e}", exc_info=True)
        raise RuntimeError(f"Embedding generation failed: {e}") from e

# --- Qdrant Client Initialization ---
async def initialize_qdrant_client():
    """Initialize Async Qdrant client and ensure collection exists. (Async)"""
    if st.session_state.get("qdrant_initialized", False):
        # Add a quick check if client exists; re-init if needed
        if "async_qdrant_client" not in st.session_state or st.session_state.async_qdrant_client is None:
             st.session_state.qdrant_initialized = False # Force re-init
        else:
            # Optional: Quick health check
            try:
                aclient: AsyncQdrantClient = st.session_state.async_qdrant_client
                logger.info("Qdrant connection already initialized and healthy.")
                return # Already good
            except Exception as health_err:
                logger.warning(f"Qdrant health check failed: {health_err}. Re-initializing.")
                st.session_state.qdrant_initialized = False
                st.session_state.async_qdrant_client = None
    # Continue with initialization if needed
    if st.session_state.get("qdrant_initialized", False): return

    logger.info(f"Initializing Qdrant client for URL: {QDRANT_URL}")
    try:
        aclient = AsyncQdrantClient(
            url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0
        )

        collection_exists = False
        try:
            await aclient.get_collection(collection_name=UNIFIED_COLLECTION_NAME)
            logger.info(f"Qdrant collection '{UNIFIED_COLLECTION_NAME}' already exists.")
            collection_exists = True
        except (qdrant_exceptions.UnexpectedResponse, ValueError) as e:
             if "Not found" in str(e) or "404" in str(e) or isinstance(e, ValueError):
                logger.info(f"Collection '{UNIFIED_COLLECTION_NAME}' not found. Will create.")
                collection_exists = False
             else: raise

        if not collection_exists:
            logger.info(f"Creating collection '{UNIFIED_COLLECTION_NAME}' with dim {EMBEDDING_DIM}...") # EMBEDDING_DIM is 768
            await aclient.create_collection(
                collection_name=UNIFIED_COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE), # Will use 768 here
            )
            logger.info(f"Created Qdrant collection '{UNIFIED_COLLECTION_NAME}'.")

        # Store client and status in session state AFTER successful init
        st.session_state.async_qdrant_client = aclient
        st.session_state.qdrant_initialized = True
        logger.info("Async Qdrant client initialized successfully.")

    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, qdrant_exceptions.ResponseHandlingException) as conn_err:
        logger.error(f"Qdrant Connection Error: {conn_err}", exc_info=True)
        st.session_state.qdrant_initialized = False
        st.session_state.async_qdrant_client = None
        raise RuntimeError(f"Failed to connect to Qdrant: {conn_err}") from conn_err # Propagate error
    except Exception as e:
        logger.error(f"Qdrant Initialization Error: {e}", exc_info=True)
        st.session_state.qdrant_initialized = False
        st.session_state.async_qdrant_client = None
        raise RuntimeError(f"Unexpected error during Qdrant initialization: {e}") from e

# --- Text Extraction and Chunking (Synchronous) ---
def extract_text_from_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[str]:
    """Extracts text content from uploaded file, handling basic types and PDF."""
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    logger.info(f"Attempting text extraction for: {file_name} (Type: {file_type})")
    try:
        content_bytes = uploaded_file.getvalue()
        if not content_bytes: return None
        if file_type == "application/pdf" and PyPDF2:
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content_bytes))
                text_content = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
                return text_content.strip() if text_content else None
            except Exception as pdf_err:
                logger.error(f"Error extracting text from PDF {file_name} with PyPDF2: {pdf_err}", exc_info=True)
                # Fall through to basic decoding
        try: return content_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            try: return content_bytes.decode("latin-1").strip()
            except UnicodeDecodeError:
                logger.warning(f"Used lossy decoding for file {file_name}")
                return content_bytes.decode("utf-8", errors="ignore").strip()
    except Exception as extract_err:
        logger.error(f"Error reading/decoding file {file_name}: {extract_err}", exc_info=True)
        return None

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """Simple text chunking."""
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text) or chunk_size <= chunk_overlap: break # Prevent infinite loop
    return [chunk for chunk in chunks if chunk.strip()]

# --- File Upload Handling ---
async def handle_single_file_upload(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, aclient: AsyncQdrantClient) -> Tuple[str, int, Optional[str]]:
    """Handles processing and upload of a single file. (Async)"""
    file_name = uploaded_file.name
    points_to_upload = []
    error_message = None
    points_count = 0

    try:
        # Synchronous parts can run directly
        text_content = extract_text_from_file(uploaded_file)
        if not text_content: return file_name, 0, "Text extraction failed or file empty"
        chunks = chunk_text(text_content)
        if not chunks: return file_name, 0, "No text chunks generated"
        logger.info(f"Extracted and chunked {file_name} into {len(chunks)} chunk(s)")

        # Asynchronous part: Embedding generation
        try:
            embeddings = await generate_embedding_google(
                texts=chunks, task_type="RETRIEVAL_DOCUMENT", title=file_name
            )
        except Exception as embed_err:
            # Catch errors propagated from generate_embedding_google
            logger.error(f"Embedding failed for {file_name}: {embed_err}", exc_info=True)
            return file_name, 0, f"Embedding generation failed: {embed_err}"

        if not embeddings: # Should not happen if exceptions are caught, but check
            return file_name, 0, "Embedding generation returned empty list"

        logger.info(f"Generated {len(embeddings)} embeddings for {file_name} using Google.")

        # Prepare points (synchronous)
        for i, chunk in enumerate(chunks):
            # Use UUID v4 for a unique ID
            point_id = str(uuid.uuid4())
            payload = {"text": chunk, "file_name": file_name, "chunk_index": i}
            if len(embeddings[i]) != EMBEDDING_DIM:
                error_message = f"Embedding dimension mismatch (expected {EMBEDDING_DIM})"
                break
            points_to_upload.append(PointStruct(id=point_id, vector=embeddings[i], payload=payload))
        if error_message: return file_name, 0, error_message

        # Asynchronous part: Upload to Qdrant
        if points_to_upload:
            batch_size = 100; total_uploaded = 0
            for j in range(0, len(points_to_upload), batch_size):
                batch = points_to_upload[j:j + batch_size]
                try:
                    await aclient.upsert(collection_name=UNIFIED_COLLECTION_NAME, points=batch, wait=True)
                    total_uploaded += len(batch)
                except (httpx.HTTPStatusError, qdrant_exceptions.ResponseHandlingException, httpx.ReadTimeout) as upload_err:
                    error_message = f"Qdrant upload failed: {upload_err}"; break
                except Exception as general_upload_err:
                     error_message = f"Unexpected upload error: {general_upload_err}"; break
            points_count = total_uploaded
            if not error_message: logger.info(f"Successfully uploaded {points_count} points for {file_name}")
        else:
            logger.warning(f"No points generated to upload for file {file_name}")

    except Exception as e:
        logger.error(f"Unhandled error processing file {file_name}: {e}", exc_info=True)
        # Capture the error to return it
        error_message = f"General processing error: {e}"

    # Return error message if any occurred
    return file_name, points_count, error_message


# --- Process and Upload Files ---
async def process_and_upload_files(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    aclient: AsyncQdrantClient # <-- ADD client as argument
    ) -> Dict[str, Any]:
    """
    Processes uploaded files and uploads to Qdrant asynchronously.
    Returns a dictionary with results and errors. (Async)
    """
    # NOTE: No st.* calls allowed in this function!

    if not uploaded_files:
        # Return structure indicating nothing to do
        return {"processed_files": 0, "processed_points": 0, "errors": ["No files selected."]}

    processed_files_count = 0
    processed_points_count = 0
    processing_errors = [] # Collect error strings
    total_files = len(uploaded_files)

    logger.info(f"Starting processing of {total_files} files in background task.")

    # Pass the client down to handle_single_file_upload
    tasks = [handle_single_file_upload(uf, aclient) for uf in uploaded_files] # Pass 'aclient'
    results = await asyncio.gather(*tasks, return_exceptions=True) # Capture results and exceptions

    # --- Process results ---
    logger.info(f"Finished processing {total_files} files in background task. Collating results...")
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Handle exceptions raised by handle_single_file_upload or asyncio itself
            file_name_approx = uploaded_files[i].name # Get filename for context
            error_msg = f"File '{file_name_approx}': Unexpected error - {result}"
            processing_errors.append(error_msg)
            logger.error(f"Error processing file {file_name_approx} in gather: {result}", exc_info=result)
        else:
            # Unpack successful result from handle_single_file_upload
            file_name, points_count, error = result
            if error:
                # Append error reported by the handler
                processing_errors.append(f"File '{file_name}': {error}")
                logger.warning(f"Error reported for file '{file_name}': {error}")
            else:
                # Count successful processing
                processed_files_count += 1
                processed_points_count += points_count
                logger.debug(f"Successfully processed file '{file_name}', added {points_count} points.")

    logger.info(f"Background task summary: Processed {processed_files_count}/{total_files} files successfully, added {processed_points_count} points. Errors: {len(processing_errors)}")

    # Return the summary dictionary
    return {
        "processed_files": processed_files_count,
        "processed_points": processed_points_count,
        "errors": processing_errors,
        "total_files_attempted": total_files
    }

# --- Financial QA Agent Definition & Models ---
class FinancialQueryInput(BaseModel):
    question: str = Field(..., description="The financial question to answer.")
    context: str = Field(..., description="Context retrieved from documents relevant to the question.")

class FinancialAnswer(BaseModel):
    answer: str = Field(..., description="The direct answer based *only* on the provided context.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0).")
    supporting_quote: Optional[str] = Field(None, description="A direct quote supporting the answer.")
    caveats: Optional[str] = Field(None, description="Limitations or ambiguities.")
    relevant_sources: List[int] = Field([], description="List of source numbers (e.g., [1, 3]) from the context that were most relevant.")

# Initialize agent (assuming gemini_model is ready)
try:
    SYSTEM_PROMPT = """You are a meticulous financial analyst AI assistant...""" # (Your full prompt)
    financial_qa_agent = Agent(
        model=gemini_model,
        deps_type=FinancialQueryInput,
        result_type=FinancialAnswer,
        system_prompt=SYSTEM_PROMPT
    )
    logger.info("Financial QA agent initialized.")
except NameError:
     financial_qa_agent = None # Should have been caught earlier
     logger.error("Cannot initialize agent, Gemini model missing.")
except Exception as e:
     financial_qa_agent = None
     logger.error(f"Failed to initialize Financial QA Agent: {e}", exc_info=True)
     # Let main function handle this potential None value

# --- Financial QA Function ---
async def perform_financial_qa(
    query_str: str,
    aclient: AsyncQdrantClient # <-- ADD client as argument
    ): # Removed return type hint as it updates session_state
    """Performs Q&A using the initialized agent and Qdrant client. (Async)"""
    # Checks performed in main UI logic before calling run_sync
    if not financial_qa_agent:
         raise RuntimeError("Financial QA Agent is not available.")

    st.session_state.last_answer = None # Reset previous answer in UI state

    # 1. Generate Query Embedding
    try:
        query_embedding_list = await generate_embedding_google(
            texts=query_str, task_type="RETRIEVAL_QUERY"
        )
    except Exception as embed_err:
         logger.error(f"QA Query embedding failed: {embed_err}", exc_info=True)
         raise RuntimeError(f"Failed to generate query embedding: {embed_err}") from embed_err

    if not query_embedding_list:
        raise RuntimeError("Query embedding generation returned no result.")

    query_embedding = query_embedding_list[0]
    if len(query_embedding) != EMBEDDING_DIM:
        raise RuntimeError(f"Query embedding dimension mismatch ({len(query_embedding)} vs {EMBEDDING_DIM}).")
    logger.info("Query embedding generated successfully.")

    # 2. Search Qdrant - Use the passed 'aclient'
    try:
        search_limit = 5
        logger.debug(f"Performing Qdrant search with client {id(aclient)}") # Log client ID
        search_result = await aclient.search( # Use the 'aclient' argument
            collection_name=UNIFIED_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=search_limit
        )
        logger.info(f"Qdrant search returned {len(search_result)} results.")
    except Exception as q_err:
         logger.error(f"Qdrant search failed: {q_err}", exc_info=True)
         raise RuntimeError(f"Failed to search relevant documents: {q_err}") from q_err

    if not search_result:
        # Okay to update session state here for feedback
        st.session_state.last_answer = "No relevant documents found."
        st.session_state.last_context = []
        logger.info("No relevant documents found for QA query.")
        return # Not an error, just no results

    # 3. Assemble Context
    context_parts = []
    sources_metadata = []
    for i, hit in enumerate(search_result):
        payload = hit.payload or {}
        file_name = payload.get("file_name", "Unknown")
        chunk_idx = payload.get("chunk_index", "N/A")
        text = payload.get("text", "")
        score = hit.score
        context_parts.append(f"Source {i+1} (File: {file_name}, Chunk: {chunk_idx}, Score: {score:.4f}):\n{text}\n---")
        sources_metadata.append({"id": i+1, "file": file_name, "chunk": chunk_idx, "score": score, "text": text})

    full_context = "\n".join(context_parts)
    st.session_state.last_context = sources_metadata # Save for UI display

    if not full_context.strip():
        raise RuntimeError("Retrieved context is empty after assembling.")

    # 4. Run QA Agent
    logger.info("Running Financial QA Agent...")
    agent_input_data = FinancialQueryInput(question=query_str, context=full_context)
    try:
        # Pydantic-AI v0.5.0+ directly returns the result_type instance
        agent_response = await financial_qa_agent.run(agent_input_data)

        if not agent_response or not isinstance(agent_response, FinancialAnswer):
            logger.error(f"Invalid agent response type: {type(agent_response)}, value: {agent_response}")
            raise RuntimeError("QA agent returned an invalid response structure.")

        st.session_state.last_answer = agent_response # Store the successful Pydantic object
        logger.info("Financial QA Agent finished successfully.")

    except Exception as agent_err:
        logger.error(f"Financial QA agent failed: {agent_err}", exc_info=True)
        raise RuntimeError(f"Failed to generate answer: {agent_err}") from agent_err


# --- Streamlit UI ---
def main():
    # st.set_page_config(
    #     page_title="Parsely QA (Google Embeddings)",
    #     page_icon="🧩",
    #     layout="wide",
    # )
    st.title("🧩 Parsely QA - Google Embeddings & Gemini (Async Manager)")
    st.markdown("""
    Upload documents and ask financial questions. Uses a background thread for async operations.
    """)

    # Get the manager instance (created once via @st.cache_resource)
    try:
        async_task_manager = get_async_manager()
    except Exception as manager_err: # Catch potential init errors from manager itself
        st.error(f"Fatal Error: Could not initialize the background task manager: {manager_err}")
        logger.error(f"AsyncTaskManager retrieval/init failed: {manager_err}", exc_info=True)
        st.stop() # Critical failure

    # --- Initialization executed synchronously via Manager ---
    init_status_placeholder = st.empty() # Placeholder for status messages
    if not st.session_state.get("qdrant_initialized", False):
        with init_status_placeholder, st.spinner("🔄 Connecting to Vector Database..."):
            try:
                # Run the async function SYNCHRONOUSLY using the manager
                async_task_manager.run_sync(initialize_qdrant_client(), timeout=30.0)
                # Check the result stored in session_state by initialize_qdrant_client
                init_status_placeholder.success("✅ Vector Database connected.")
                time.sleep(1) # Keep success message visible briefly
                init_status_placeholder.empty() # Clear message
            except TimeoutError:
                 init_status_placeholder.error("⏳ Connection to Vector Database timed out.")
                 st.stop()
            except Exception as init_err:
                 logger.error(f"Initialization via AsyncTaskManager failed: {init_err}", exc_info=True)
                 init_status_placeholder.error(f"❌ Vector DB connection failed: {init_err}")
                 st.stop() # Stop if DB connection fails critically

    # --- File Upload Section ---
    with st.expander("1. Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Select text-based files (.txt, .md, .pdf, etc.)",
            accept_multiple_files=True,
            type=['txt', 'md', 'csv', 'json', 'pdf'],
            key="file_uploader"
        )
        
        
        # Initialize async client separately
        async_qdrant_client = AsyncQdrantClient(
            url=st.secrets["qdrant_url"], 
            api_key=st.secrets['qdrant_api_key'],
            timeout=30  # Set a longer timeout (30 seconds)
        )

        # Store client and status in session state AFTER successful init
        st.session_state.async_qdrant_client = async_qdrant_client
        st.session_state.qdrant_initialized = True
        logger.info("Async Qdrant client initialized successfully.")

        upload_pressed = st.button("Process and Upload Files to Vector DB", key="upload_button", type="primary")
        if upload_pressed and uploaded_files:
            # --- FIX: Check if client exists in session state before upload ---
            if "async_qdrant_client" not in st.session_state or st.session_state.async_qdrant_client is None:
                st.error("Qdrant client not initialized or initialization failed. Cannot upload.")
                # Optionally add a button to retry initialization?
                
            else:
                # --- FIX: Retrieve the INITIALIZED client from session state ---
                aclient_for_upload = st.session_state.async_qdrant_client
                st.info(f"Using Qdrant client instance ID: {id(aclient_for_upload)} for upload.") # Optional: Log client ID for debugging

                with st.spinner(f"Processing {len(uploaded_files)} file(s)... This may take a while."):
                    try:
                        # Run the async processing function SYNCHRONOUSLY via manager
                        # It now returns a dictionary of results
                        upload_result = async_task_manager.run_sync(
                            process_and_upload_files(uploaded_files, aclient_for_upload),
                            timeout=600.0 # Long timeout for upload
                        )

                        # --- Process results returned from the background task ---
                        files_processed = upload_result.get("processed_files", 0)
                        points_added = upload_result.get("processed_points", 0)
                        errors = upload_result.get("errors", [])
                        total_attempted = upload_result.get("total_files_attempted", len(uploaded_files))

                        # Display results using st calls HERE in the main thread
                        if files_processed > 0:
                            st.success(f"Successfully processed {files_processed}/{total_attempted} file(s), uploaded {points_added} data points/chunks.")
                        if errors:
                            st.error("Errors occurred during processing:")
                            # Limit displayed errors if there are too many
                            max_errors_to_show = 10
                            for i, err in enumerate(errors):
                                if i < max_errors_to_show:
                                    st.error(f"- {err}")
                                elif i == max_errors_to_show:
                                    st.error(f"- ... ({len(errors) - max_errors_to_show} more errors)")
                                    break
                        elif files_processed == 0 and not errors:
                            st.warning("No new data points were uploaded. Files might have been empty or failed extraction/embedding.")


                    except TimeoutError:
                        st.error("⏳ File processing timed out. Please try again with fewer or smaller files.")
                    except Exception as upload_err:
                        # Catch errors raised by process_and_upload_files itself (like Qdrant client check)
                        # or errors from run_sync / manager
                        st.error(f"❌ An unexpected error occurred during the upload process: {upload_err}")
                        logger.error(f"File processing error caught in main thread: {upload_err}", exc_info=True)

        elif upload_pressed and not uploaded_files:
             st.warning("Please select files to upload first.")

    # --- QA Section ---
    st.header("2. Ask a Question")
    with st.form(key="qa_form"):
        query = st.text_input("Enter your financial question:", key="qa_query", placeholder="e.g., What were total revenues in Q4?")
        answer_pressed = st.form_submit_button("Get Answer", type="primary")

    if answer_pressed and query:
        # Check prerequisites before running async task
        if "async_qdrant_client" not in st.session_state or st.session_state.async_qdrant_client is None:
            st.error("Qdrant client not initialized. Cannot perform search.")
        elif not financial_qa_agent:
             st.error("Financial QA Agent not initialized. Cannot generate answer.")
        else:
            with st.spinner("Searching relevant documents and generating answer..."):
                try:
                    # --- FIX: Retrieve client from session state ---
                    if "async_qdrant_client" in st.session_state and st.session_state.async_qdrant_client:
                        aclient_for_qa = st.session_state.async_qdrant_client
                        # --- FIX: Pass client to the async function ---
                        async_task_manager.run_sync(
                            perform_financial_qa(query, aclient_for_qa), # Pass client
                            timeout=120.0
                        )
                    else:
                        st.error("Qdrant client not found in session state. Cannot perform QA.")

                except TimeoutError:
                    st.error("⏳ Answer generation timed out. Please try again or simplify your query.")
                except Exception as qa_err:
                    # Display the error from the QA process
                    st.error(f"❌ An error occurred while getting the answer: {qa_err}")
                    logger.error(f"QA error via run_sync: {qa_err}", exc_info=True)
                    # Reset potentially partial results
                    st.session_state.last_answer = None
                    st.session_state.last_context = None


    elif answer_pressed and not query:
        st.warning("Please enter a question in the text box.")

    # --- Display results ---
    st.divider()
    if "last_answer" in st.session_state and st.session_state.last_answer:
        st.subheader("Answer")
        answer_data = st.session_state.last_answer

        if isinstance(answer_data, FinancialAnswer):
            # Display structured answer
            st.markdown(f"**Answer:** {answer_data.answer}")
            st.metric(label="Confidence", value=f"{answer_data.confidence_score:.2f}")
            if answer_data.caveats: st.info(f"**Caveats:** {answer_data.caveats}")
            if answer_data.supporting_quote:
                st.markdown(f"**Supporting Quote:**")
                st.markdown(f"> {answer_data.supporting_quote}", help="Verbatim quote from source context.")

            # Display context used
            if "last_context" in st.session_state and st.session_state.last_context:
                 with st.expander("Show Context Used for Answer"):
                     st.caption(f"Retrieved {len(st.session_state.last_context)} document chunks. Relevant sources identified by AI: {answer_data.relevant_sources or 'None'}")
                     for source in st.session_state.last_context:
                          highlight_style = "border-left: 4px solid #FF4B4B; padding-left: 8px; margin-bottom: 8px;" if source['id'] in answer_data.relevant_sources else "padding-left: 8px; margin-bottom: 8px;"
                          with st.container():
                             st.markdown(f"<div style='{highlight_style}'><b>Source {source['id']}:</b> File: <i>{source['file']}</i> (Chunk: {source['chunk']}, Score: {source['score']:.4f})</div>", unsafe_allow_html=True)
                             st.text(source['text'][:500] + "..." if len(source['text']) > 500 else source['text'])
                             st.markdown("---")
        elif isinstance(answer_data, str):
             # Display simple string feedback (e.g., "No documents found")
             st.info(answer_data)
        else:
             st.warning("Could not display answer - unexpected format.")


    # --- Sidebar Info ---
    st.sidebar.title("Info")
    st.sidebar.markdown(f"**User:** `{USERNAME}`")
    st.sidebar.markdown(f"**Vector Collection:** `{UNIFIED_COLLECTION_NAME}`")
    st.sidebar.markdown(f"**Embedding Model:**", help=f"Dim: {EMBEDDING_DIM}")
    st.sidebar.code(EMBEDDING_MODEL_NAME, language=None)
    if 'gemini_model' in globals() and gemini_model:
        st.sidebar.markdown(f"**QA Model:**")
        st.sidebar.code(gemini_model.model_name, language=None)
    else: st.sidebar.markdown("**QA Model:** `Not Initialized`")
    qdrant_status = '🟢 Connected' if st.session_state.get('qdrant_initialized') else '🔴 Disconnected'
    st.sidebar.markdown(f"**Qdrant Status:** {qdrant_status}")
    genai_status = '🟢 Configured' if genai_configured else '🔴 Config Failed'
    st.sidebar.markdown(f"**Google GenAI Status:** {genai_status}")
    manager = AsyncTaskManager._instance # Get instance without re-creating
    if manager and manager._thread and manager._thread.is_alive():
         st.sidebar.markdown(f"**Async Manager:** 🟢 Running")
    else:
         st.sidebar.markdown(f"**Async Manager:** 🔴 Stopped / Error")
    st.sidebar.divider()
    st.sidebar.caption("Reload page if issues persist.")


# --- Entry Point ---
if __name__ == "__main__":
    # Initialize session state keys if they don't exist
    keys_to_init = {
        'qdrant_initialized': False,
        'async_qdrant_client': None,
        'last_answer': None,
        'last_context': None,
        # '_embedding_error_shown': False # No longer needed with new structure
    }
    for key, default_value in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Run the main Streamlit app function
    main()

    # Note: Cleanup of the AsyncTaskManager is implicitly handled by
    # Streamlit's @st.cache_resource lifecycle and the thread being a daemon.
    # Explicit cleanup using atexit is generally not recommended with Streamlit.