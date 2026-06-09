import streamlit as st
import asyncio
import time
import logging
import os
# Removed unused imports like pandas, uuid, defaultdict

# --- Qdrant/Data Specific Imports ---
from datasets import load_dataset
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    VectorParams, HnswConfigDiff, SparseVectorParams, SparseIndexParams,
    OptimizersConfigDiff, BinaryQuantization, QuantizationSearchParams,
    SearchParams, NamedVector, NamedSparseVector, Distance, # Added Distance
    Prefetch, Fusion, FusionQuery, SparseVector, PointStruct,
    CollectionStatus # Added for HNSW status check
)
# Removed Ranx imports

# --- Configuration (Adapted from the new example) ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s'
)

# --- Qdrant App Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "scifact_bge_binary_fg" # Descriptive name
EMB_MODEL_NAME = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
EMB_DIM = 768 # Dimension for bge-base-en-v1.5
BATCH_SIZE = 32
INGEST_CONCURRENCY = 5 # For gather within foreground task
HNSW_EF_SEARCH = 64
OVERSAMPLING = 4.0 # For binary quantized search
DENSE_M_PROD = 16 # HNSW M value for Phase 2
SEARCH_PREFETCH_LIMIT = 50 # Keep reasonable prefetch limit

# --- Helper to get/create loop ---
# Still needed for run_until_complete
def get_or_create_eventloop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        logging.info("No running event loop found, creating a new one for this thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# --- Client factory ---
@st.cache_resource
def get_client():
    logging.info(f"Initializing Qdrant client for {QDRANT_URL}")
    # Use a longer timeout potentially needed for HNSW update/optimization
    return AsyncQdrantClient(
        url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False, timeout=900
    )

# --- Embedding models ---
@st.cache_resource
def load_models(): # Renamed function for clarity
    logging.info(f"Loading embedding models ({EMB_MODEL_NAME}, {SPARSE_MODEL_NAME})...")
    start = time.time()
    dense_model = TextEmbedding(EMB_MODEL_NAME)
    sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)
    logging.info(f"Embedding models loaded in {time.time() - start:.2f} seconds.")
    return dense_model, sparse_model

# --- Async Helper Functions (Adapted for Two-Phase) ---

async def create_collection_phase1(_client):
    """Phase 1a: Creates the collection with HNSW disabled (m=0) and binary quantization."""
    logging.info(f"Phase 1a: Attempting to create collection '{COLLECTION_NAME}' with HNSW disabled...")
    try:
        # Use recreate_collection for simplicity (handles existing collection)
        await _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                EMB_MODEL_NAME: VectorParams(
                    size=EMB_DIM,
                    distance=Distance.COSINE, # Explicitly use COSINE
                    on_disk=True, # Store original vectors on disk
                    hnsw_config=HnswConfigDiff(m=0) # Disable HNSW for initial ingest
                )
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(
                    index=SparseIndexParams(on_disk=True) # Sparse index on disk
                )
            },
            quantization_config=BinaryQuantization( # Enable binary quantization
                binary=models.BinaryQuantizationConfig(always_ram=True) # Keep quantized in RAM
            ),
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20000) # Adjust if needed
        )
        logging.info(f"Collection '{COLLECTION_NAME}' created/recreated successfully (HNSW disabled).")
        return True
    except Exception as e:
        logging.error(f"Failed to create collection '{COLLECTION_NAME}': {e}", exc_info=True)
        st.error(f"Failed to create collection: {e}")
        return False

async def enable_hnsw_phase2(_client, m_val=DENSE_M_PROD):
    """Phase 2: Enables HNSW index after ingestion and waits for optimization."""
    logging.info(f"Phase 2: Attempting to enable HNSW (m={m_val}) for collection '{COLLECTION_NAME}'...")
    try:
        # Update HNSW config
        update_op = await _client.update_collection(
            collection_name=COLLECTION_NAME,
            hnsw_config=HnswConfigDiff(m=m_val) # Set desired M value
        )
        if not update_op:
            logging.warning("Update collection operation returned False.")
            # Optionally raise an error or return False

        logging.info("HNSW update request sent. Waiting for optimizer status to become green...")
        # Wait until the optimizer status is green, indicating indexing is complete
        while True:
            collection_info = await _client.get_collection(collection_name=COLLECTION_NAME)
            current_status = collection_info.status
            logging.debug(f"Current collection status: {current_status}")
            if current_status == CollectionStatus.GREEN:
                logging.info("Collection status is GREEN. HNSW index is ready.")
                break
            elif current_status == CollectionStatus.YELLOW:
                logging.info("Collection status is YELLOW (optimizing)... waiting 5s.")
                await asyncio.sleep(5)
            elif current_status == CollectionStatus.RED:
                 logging.error("Collection status turned RED. Optimization failed.")
                 st.error("Collection optimization failed (Status: RED). Check Qdrant logs.")
                 return False # Indicate failure
            else: # Grey or unexpected status
                 logging.warning(f"Unexpected collection status: {current_status}. Waiting 5s.")
                 await asyncio.sleep(5)
        return True # Indicate success
    except Exception as e:
        logging.error(f"Failed to enable HNSW or wait for optimization: {e}", exc_info=True)
        st.error(f"Failed to enable HNSW: {e}")
        return False

# upload_batch remains async, used internally by the foreground ingest task
async def upload_batch(_client, batch, batch_num, total_batches, _dense_model, _sparse_model):
    logging.debug(f"Processing batch {batch_num}/{total_batches}")
    texts = batch["text"]
    try:
        # Ensure models are loaded
        if not all([_dense_model, _sparse_model]):
            raise ValueError("Required embedding models not loaded")

        # Generate embeddings for the batch
        dense_embs = list(_dense_model.embed(texts, batch_size=len(texts)))
        sparse_embs = list(_sparse_model.embed(texts, batch_size=len(texts)))

        if not dense_embs or not sparse_embs:
             logging.warning(f"Batch {batch_num}: Embedding generation returned empty lists.")
             return

        points = []
        for i, doc_id in enumerate(batch["_id"]):
             # Ensure index is valid
            if i >= len(dense_embs) or i >= len(sparse_embs):
                logging.warning(f"Batch {batch_num}: Index {i} out of range. Skipping doc_id {doc_id}.")
                continue

            try: point_id = int(doc_id)
            except ValueError: point_id = str(doc_id) # Use string ID if not integer

            # Prepare sparse vector data safely
            sparse_indices = sparse_embs[i].indices.tolist() if hasattr(sparse_embs[i], 'indices') else []
            sparse_values = sparse_embs[i].values.tolist() if hasattr(sparse_embs[i], 'values') else []

            # Create PointStruct using configured model names
            points.append(PointStruct(
                id=point_id,
                vector={
                    EMB_MODEL_NAME: dense_embs[i].tolist(), # Use constant for key
                    "bm25": SparseVector(indices=sparse_indices, values=sparse_values)
                },
                payload={"title": batch["title"][i], "text": batch["text"][i]}
            ))

        if points:
            await _client.upsert(collection_name=COLLECTION_NAME, points=points, wait=False) # wait=False for async bulk
            logging.debug(f"Upsert command for batch {batch_num} sent ({len(points)} points).")
        else:
             logging.warning(f"Batch {batch_num}: No points generated for upsert.")

    except Exception as e:
        # Log errors, especially during embedding or upsert
        logging.error(f"Error processing/upserting batch {batch_num}: {e}", exc_info=True)
        # Avoid st calls here

# --- Foreground Task for Bulk Ingestion ---
async def run_bulk_ingest_foreground(_client, _dense_model, _sparse_model):
    """Phase 1b: Loads data and uploads it in batches (foreground blocking)."""
    start_ingest_time = time.perf_counter()
    logging.info("Phase 1b: Starting FOREGROUND bulk ingest process...")

    ingest_successful = False
    total_docs = 0
    ingest_time = 0.0
    status_placeholder = st.empty() # For potential temporary messages (less effective in fg)

    try:
        logging.info("Loading SciFact corpus dataset...")
        # status_placeholder.info("Loading dataset...") # Won't show
        ds = load_dataset("BeIR/scifact", "corpus", split="corpus", trust_remote_code=True)
        total_docs = len(ds)
        logging.info(f"Dataset loaded. Total documents: {total_docs}. Starting upload...")
        # status_placeholder.info(f"Loaded {total_docs} docs. Starting upload...")

        processed_count = 0
        processed_batches = 0
        total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
        upload_tasks = []

        # Create all batch upload tasks
        for i in range(0, total_docs, BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            batch_data_dict = ds[i : i + BATCH_SIZE]
            upload_tasks.append(
                upload_batch(_client, batch_data_dict, batch_num, total_batches, _dense_model, _sparse_model)
            )

        # Process tasks concurrently using asyncio.gather within this foreground task
        logging.info(f"Dispatching {len(upload_tasks)} upload tasks in groups of {INGEST_CONCURRENCY}...")
        for i in range(0, len(upload_tasks), INGEST_CONCURRENCY):
            group = upload_tasks[i : i + INGEST_CONCURRENCY]
            if not group: break

            current_batch_start_num = i + 1
            current_batch_end_num = min(i + INGEST_CONCURRENCY, len(upload_tasks))
            progress_text_log = f"Processing task group {current_batch_start_num}-{current_batch_end_num}..."
            logging.info(progress_text_log)
            # status_placeholder.info(progress_text_log) # Won't show

            await asyncio.gather(*group) # Run concurrent uploads for the group

            # Update progress logging after group completion
            processed_batches += len(group)
            processed_count = min(total_docs, processed_batches * BATCH_SIZE)
            progress = processed_count / total_docs if total_docs > 0 else 0
            progress_text_log = f"Completed ~{processed_count}/{total_docs} documents ({processed_batches}/{total_batches} batches, {progress:.1%})"
            logging.info(progress_text_log)
            # status_placeholder.info(progress_text_log) # Won't show

            await asyncio.sleep(0.01) # Small yield

        # Wait briefly for last upserts to likely register
        logging.info("Waiting a few seconds for final upserts...")
        # status_placeholder.info("Waiting for final upserts...") # Won't show
        await asyncio.sleep(5)

        # Verify count (optional but good practice)
        final_count_res = await _client.count(collection_name=COLLECTION_NAME, exact=False)
        final_count = final_count_res.count
        logging.info(f"Final approximate count in Qdrant: {final_count}")

        # Decide success based on whether any documents were counted
        # (Could also compare final_count to total_docs if exactness is needed)
        if final_count > 0:
            ingest_successful = True
            logging.info("Bulk ingest completed successfully.")
            # status_placeholder.success("Ingest checks complete.")
        else:
            logging.warning(f"Ingest finished, but Qdrant count is {final_count}.")
            # status_placeholder.warning(f"Ingest finished, but Qdrant count is {final_count}.")
            ingest_successful = False # Treat 0 count as failure for clarity

    except Exception as e:
        error_msg = f"Bulk ingest failed: {type(e).__name__}"
        logging.error("Bulk ingest failed.", exc_info=True)
        st.error(f"{error_msg} - See console logs for details.") # Show error after block
        ingest_successful = False

    finally:
        end_ingest_time = time.perf_counter()
        ingest_time = end_ingest_time - start_ingest_time
        logging.info(f"Bulk ingest processing ended after {ingest_time:.2f}s.")
        status_placeholder.empty() # Clear any temporary message

    # Return results for state update
    return ingest_successful, ingest_time, ingest_successful # has_data is true if successful


# --- Search Function (Adapted for Binary Quantization) ---
async def do_query(_client, query_text, method, top_k, _dense_model, _sparse_model):
    """Performs search using appropriate parameters for binary quantization."""
    logging.info(f"Performing '{method}' search for query: '{query_text[:50]}...'")
    start_time = time.time()
    response = None

    try:
        # Calculate embeddings once
        dense_q = next(_dense_model.embed(query_text))
        sparse_q = next(_sparse_model.embed(query_text))

        if method == "Dense":
            response = await _client.search(
                collection_name=COLLECTION_NAME,
                query_vector=NamedVector(name=EMB_MODEL_NAME, vector=dense_q.tolist()), # Use constant
                limit=top_k,
                with_payload=True,
                search_params=SearchParams(
                    hnsw_ef=HNSW_EF_SEARCH, # Standard HNSW ef
                    # Parameters specific to binary quantization search:
                    quantization=QuantizationSearchParams(
                        ignore=False, # MUST be False to use quantization
                        rescore=True, # MUST be True to rescore with original vectors
                        oversampling=OVERSAMPLING # Fetch more candidates for rescoring
                    )
                )
            )
        elif method == "Sparse":
            # Sparse search doesn't use quantization parameters
            response = await _client.search(
                collection_name=COLLECTION_NAME,
                query_vector=NamedSparseVector(
                    name="bm25", # Use constant if defined, or hardcode
                    vector=SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist())
                ),
                limit=top_k,
                with_payload=True
                # No specific search_params needed here for BM25 typically
            )
        elif method == "RRF":
            # Prefetch needs to account for oversampling for the dense part
            dense_limit = int(top_k * OVERSAMPLING)
            sparse_limit = top_k # Sparse typically doesn't need oversampling for RRF

            prefetch_list = [
                Prefetch(
                    query=dense_q.tolist(),
                    using=EMB_MODEL_NAME, # Use constant
                    limit=dense_limit # Use calculated oversampled limit
                ),
                Prefetch(
                    query=SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist()),
                    using="bm25",
                    limit=sparse_limit # Use standard k for sparse
                )
            ]
            # Query points combines results, respecting the final 'limit'
            # Qdrant handles the RRF fusion based on prefetched results
            response = await _client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=prefetch_list,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k, # Final number of results after RRF
                with_payload=True,
                 # When using prefetch with dense vectors that use quantization,
                 # specify the quantization params here as well for the prefetch step.
                search_params=SearchParams(
                     quantization=QuantizationSearchParams(
                        ignore=False,
                        rescore=True, # Rescoring happens *before* fusion here
                        oversampling=OVERSAMPLING # Necessary for the dense prefetch step
                    )
                )
            )
        else:
            logging.error(f"Unknown search method: {method}"); st.error(f"Unknown search method: {method}"); return None, 0.0

        elapsed_time = time.time() - start_time
        logging.info(f"Search completed in {elapsed_time:.4f} seconds.")
        # Ensure response format is handled correctly (search returns list, query_points returns object)
        points_to_return = response if isinstance(response, list) else (response.points if hasattr(response, 'points') else [])
        return points_to_return, elapsed_time # Return list of points and time

    except Exception as e:
        logging.error(f"Search query failed for method {method}: {e}", exc_info=True)
        st.error(f"Search failed: {e}")
        return None, time.time() - start_time

# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(layout="wide", page_title="Qdrant 2-Phase Binary FG")
    st.title("🔍 Qdrant Search (2-Phase Ingest, Binary Quant, FG)")
    st.write(f"Using **{EMB_MODEL_NAME}** (Quantized) + **BM25**. Indexing is foreground.")

    # --- Initialize client and models ---
    client = get_client()
    dense_model, sparse_model = load_models()

    # --- Initialize Session State ---
    # Track the state of the ingestion phases
    if "collection_created" not in st.session_state: st.session_state.collection_created = False # Phase 1a done
    if "indexing_complete" not in st.session_state: st.session_state.indexing_complete = False # Phase 1b done
    if "hnsw_enabled" not in st.session_state: st.session_state.hnsw_enabled = False       # Phase 2 done
    # Keep search state
    if "last_query" not in st.session_state: st.session_state.last_query = ""
    if "last_method" not in st.session_state: st.session_state.last_method = "RRF"
    if "last_top_k" not in st.session_state: st.session_state.last_top_k = 10
    if "last_search_results" not in st.session_state: st.session_state.last_search_results = None
    if "last_search_time" not in st.session_state: st.session_state.last_search_time = None

    # --- Sidebar ---
    with st.sidebar:
        st.header("Ingestion Pipeline")

        # --- Phase 1a Button ---
        if st.button("Phase 1a: Create Collection (m=0)", help="Creates/recreates the collection with HNSW disabled for bulk ingest."):
            with st.spinner("Creating collection..."):
                current_loop = get_or_create_eventloop()
                success = current_loop.run_until_complete(create_collection_phase1(client))
                if success:
                    st.session_state.collection_created = True
                    st.session_state.indexing_complete = False # Reset subsequent steps
                    st.session_state.hnsw_enabled = False    # Reset subsequent steps
                    st.success("Phase 1a Complete: Collection created (HNSW disabled).")
                else:
                    st.session_state.collection_created = False # Mark as failed
                st.rerun()

        # --- Phase 1b Button ---
        # Disable if collection not created or if indexing already done (needs Phase 1a reset)
        phase1b_disabled = not st.session_state.collection_created or st.session_state.indexing_complete
        if st.button("Phase 1b: Bulk Ingest Data (FG)", disabled=phase1b_disabled, help="Uploads SciFact corpus data. Blocks UI."):
            if not st.session_state.collection_created:
                 st.warning("Collection must be created first (Phase 1a).")
            else:
                with st.spinner("Ingesting data... This will block the UI. See console for progress."):
                    current_loop = get_or_create_eventloop()
                    try:
                        success, time_taken, has_data = current_loop.run_until_complete(
                            run_bulk_ingest_foreground(client, dense_model, sparse_model)
                        )
                        st.session_state.indexing_complete = success
                        st.session_state.collection_has_data = has_data # Store if data likely exists
                        st.session_state.hnsw_enabled = False # HNSW still needs enabling

                        if success:
                            st.success(f"Phase 1b Complete: Bulk ingest finished in {time_taken:.2f}s.")
                        else:
                            st.error("Phase 1b Failed: Bulk ingest encountered errors.")
                    except Exception as e:
                        logging.error(f"Exception during foreground ingest trigger: {e}", exc_info=True)
                        st.error(f"Ingest run failed: {e}")
                        st.session_state.indexing_complete = False
                        st.session_state.collection_has_data = False
                st.rerun()

        # --- Phase 2 Button ---
        # Disable if indexing not done or HNSW already enabled
        phase2_disabled = not st.session_state.indexing_complete or st.session_state.hnsw_enabled
        if st.button(f"Phase 2: Enable HNSW (m={DENSE_M_PROD})", disabled=phase2_disabled, help="Builds the HNSW graph for fast search."):
            if not st.session_state.indexing_complete:
                st.warning("Data must be ingested first (Phase 1b).")
            else:
                 with st.spinner(f"Enabling HNSW (m={DENSE_M_PROD}) and optimizing... This may take time."):
                    current_loop = get_or_create_eventloop()
                    try:
                        success = current_loop.run_until_complete(enable_hnsw_phase2(client))
                        st.session_state.hnsw_enabled = success
                        if success:
                            st.success("Phase 2 Complete: HNSW enabled and optimized.")
                        else:
                            st.error("Phase 2 Failed: Could not enable HNSW or optimization failed.")
                    except Exception as e:
                         logging.error(f"Exception during HNSW enabling trigger: {e}", exc_info=True)
                         st.error(f"HNSW enabling run failed: {e}")
                         st.session_state.hnsw_enabled = False # Mark as failed
                 st.rerun()

        # --- Status Display ---
        st.divider()
        st.header("Current Status")
        if st.session_state.collection_created:
            st.success("✅ Phase 1a: Collection Created")
        else:
            st.warning("⚪ Phase 1a: Collection Not Created")

        if st.session_state.indexing_complete:
            st.success("✅ Phase 1b: Data Ingested")
        elif st.session_state.collection_created:
            st.warning("⚪ Phase 1b: Data Not Ingested")
        else:
             st.caption("⚪ Phase 1b: (Requires Collection)")

        if st.session_state.hnsw_enabled:
            st.success(f"✅ Phase 2: HNSW Enabled (m={DENSE_M_PROD})")
        elif st.session_state.indexing_complete:
            st.warning("⚪ Phase 2: HNSW Not Enabled")
        else:
             st.caption("⚪ Phase 2: (Requires Ingest)")

    # --- Main Area for Search ---
    st.header("🔎 Search the Collection")
    query_text = st.text_input("Enter search query:", value=st.session_state.last_query, placeholder="e.g., What is the effect of remdesivir?")
    method_options = ["Dense", "Sparse", "RRF"]
    try: method_index = method_options.index(st.session_state.last_method)
    except ValueError: method_index = method_options.index("RRF") # Default to RRF
    method = st.selectbox("Select search method:", method_options, index=method_index)
    top_k = st.slider("Top K results", 1, 50, value=st.session_state.last_top_k)
    status_placeholder = st.empty(); results_placeholder = st.container()

    # --- Search Button ---
    # Search should only be enabled AFTER HNSW is built (Phase 2 complete)
    can_search = st.session_state.collection_created and st.session_state.indexing_complete and st.session_state.hnsw_enabled
    search_tooltip = "Search the collection (requires all ingestion phases to be complete)." if not can_search else "Run the search query."
    if st.button("Search", type="primary", disabled=not can_search, help=search_tooltip):
        if not query_text:
            st.warning("Enter a search query.")
        else:
            st.session_state.last_search_results = None; st.session_state.last_search_time = None
            status_placeholder.empty(); results_placeholder.empty()
            points = None; elapsed_time = 0.0
            with st.spinner(f"Searching with {method} (Binary Quant Aware)..."):
                try:
                    current_loop = get_or_create_eventloop()
                    # Call the updated do_query function
                    points, elapsed_time = current_loop.run_until_complete(
                        do_query(client, query_text, method, top_k, dense_model, sparse_model)
                    )
                    st.session_state.last_query = query_text
                    st.session_state.last_method = method
                    st.session_state.last_top_k = top_k
                    st.session_state.last_search_time = elapsed_time
                    st.session_state.last_search_results = points # Already a list or None
                except Exception as e:
                    logging.error(f"Error running do_query via button: {e}", exc_info=True)
                    status_placeholder.error(f"Search execution failed: {e}")
                    st.session_state.last_search_results = None
            st.rerun()

    # Display Search Results (handles list directly)
    if st.session_state.last_search_results is not None:
        result_count = len(st.session_state.last_search_results)
        status_placeholder.success(f"Displaying top {result_count} results for '{st.session_state.last_query}' ({st.session_state.last_method}). Query time: **{st.session_state.last_search_time:.3f} seconds**.")
        with results_placeholder:
            points_to_display = st.session_state.last_search_results
            if not points_to_display:
                st.info("No results found.")
            else:
                for i, p in enumerate(points_to_display):
                    payload = p.payload if hasattr(p, 'payload') else {}
                    title = payload.get('title', 'N/A')
                    score = p.score if hasattr(p, 'score') else 'N/A'
                    point_id = p.id if hasattr(p, 'id') else 'N/A'
                    score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
                    with st.expander(f"**{i+1}. {title}** (Score: {score_str}, ID: {point_id})", expanded=(i<3)):
                        text = payload.get('text', 'N/A')
                        st.markdown(text)

    elif not can_search:
         # Guide user if search is disabled
         if not st.session_state.collection_created:
             status_placeholder.warning("Please complete Phase 1a: Create Collection.")
         elif not st.session_state.indexing_complete:
              status_placeholder.warning("Please complete Phase 1b: Bulk Ingest Data.")
         elif not st.session_state.hnsw_enabled:
              status_placeholder.warning("Please complete Phase 2: Enable HNSW.")

    st.divider()
    st.caption(f"Model: {EMB_MODEL_NAME} (Binary Quant) + BM25 | Ingestion: Foreground, 2-Phase | Qdrant URL: {QDRANT_URL}")

if __name__ == "__main__":
    main()