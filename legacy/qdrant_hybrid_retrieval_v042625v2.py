import streamlit as st
import asyncio
import time
import logging
import os
# Removed unused imports

# --- Qdrant/Data Specific Imports ---
from datasets import load_dataset
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    VectorParams, HnswConfigDiff, SparseVectorParams, SparseIndexParams,
    OptimizersConfigDiff, BinaryQuantization, QuantizationSearchParams,
    SearchParams, NamedVector, NamedSparseVector, Distance,
    Prefetch, Fusion, FusionQuery, SparseVector, PointStruct,
    CollectionStatus
)
# Removed Ranx imports

# --- Configuration (Adapted from the new example) ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s'
)

# --- Qdrant App Configuration ---
QDRANT_URL = st.secrets["qdrant_url"] # Or your Qdrant URL
QDRANT_API_KEY = st.secrets["qdrant_api_key"] # Optional: For Qdrant Cloud
COLLECTION_NAME = "scifact_bge_binary_fg_ui" # Descriptive name
EMB_MODEL_NAME = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
EMB_DIM = 768
BATCH_SIZE = 32
INGEST_CONCURRENCY = 5 # How many batches to process per UI update cycle
HNSW_EF_SEARCH = 64
OVERSAMPLING = 4.0
DENSE_M_PROD = 16
SEARCH_PREFETCH_LIMIT = 50

# --- Helper to get/create loop ---
def get_or_create_eventloop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        logging.info("No running event loop found, creating a new one.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# --- Client factory ---
@st.cache_resource
def get_client():
    logging.info(f"Initializing Qdrant client for {QDRANT_URL}")
    return AsyncQdrantClient(
        url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False, timeout=900
    )

# --- Embedding models ---
@st.cache_resource
def load_models():
    logging.info(f"Loading embedding models ({EMB_MODEL_NAME}, {SPARSE_MODEL_NAME})...")
    start = time.time()
    dense_model = TextEmbedding(EMB_MODEL_NAME)
    sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)
    logging.info(f"Embedding models loaded in {time.time() - start:.2f} seconds.")
    return dense_model, sparse_model

# --- Async Helper Functions (Phase 1a, Phase 2, upload_batch - unchanged) ---

async def create_collection_phase1(_client):
    logging.info(f"Phase 1a: Creating collection '{COLLECTION_NAME}' (m=0)...")
    try:
        await _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={ EMB_MODEL_NAME: VectorParams(size=EMB_DIM, distance=Distance.COSINE, on_disk=True, hnsw_config=HnswConfigDiff(m=0)) },
            sparse_vectors_config={ "bm25": SparseVectorParams(index=SparseIndexParams(on_disk=True)) },
            quantization_config=BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True)),
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20000)
        )
        logging.info(f"Collection '{COLLECTION_NAME}' created/recreated (HNSW disabled).")
        return True
    except Exception as e:
        logging.error(f"Failed to create collection '{COLLECTION_NAME}': {e}", exc_info=True)
        st.error(f"Failed to create collection: {e}")
        return False

async def enable_hnsw_phase2(_client, m_val=DENSE_M_PROD):
    logging.info(f"Phase 2: Enabling HNSW (m={m_val}) for '{COLLECTION_NAME}'...")
    try:
        update_op = await _client.update_collection(collection_name=COLLECTION_NAME, hnsw_config=HnswConfigDiff(m=m_val))
        if not update_op: logging.warning("Update collection op returned False.")
        logging.info("HNSW update sent. Waiting for optimizer status green...")
        while True:
            collection_info = await _client.get_collection(collection_name=COLLECTION_NAME)
            current_status = collection_info.status
            logging.debug(f"Current collection status: {current_status}")
            if current_status == CollectionStatus.GREEN: logging.info("Status GREEN. HNSW ready."); break
            elif current_status == CollectionStatus.YELLOW: logging.info("Status YELLOW (optimizing)... waiting 5s."); await asyncio.sleep(5)
            elif current_status == CollectionStatus.RED: logging.error("Status RED. Optimization failed."); st.error("Opt failed (Status: RED)."); return False
            else: logging.warning(f"Unexpected status: {current_status}. Waiting 5s."); await asyncio.sleep(5)
        return True
    except Exception as e:
        logging.error(f"Failed to enable HNSW: {e}", exc_info=True)
        st.error(f"Failed to enable HNSW: {e}")
        return False

async def upload_batch(_client, batch, batch_num, total_batches, _dense_model, _sparse_model):
    # This function remains the core async uploader for a single batch
    logging.debug(f"Processing batch {batch_num}/{total_batches}")
    texts = batch["text"]
    try:
        if not all([_dense_model, _sparse_model]): raise ValueError("Models not loaded")
        dense_embs = list(_dense_model.embed(texts, batch_size=len(texts)))
        sparse_embs = list(_sparse_model.embed(texts, batch_size=len(texts)))
        if not dense_embs or not sparse_embs: logging.warning(f"Batch {batch_num}: Embeddings empty."); return 0 # Return 0 points processed

        points = []
        processed_in_batch = 0
        for i, doc_id in enumerate(batch["_id"]):
            if i >= len(dense_embs) or i >= len(sparse_embs): logging.warning(f"Batch {batch_num}: Idx {i} out of range. Skip {doc_id}."); continue
            try: point_id = int(doc_id)
            except ValueError: point_id = str(doc_id)
            sparse_indices = sparse_embs[i].indices.tolist() if hasattr(sparse_embs[i], 'indices') else []
            sparse_values = sparse_embs[i].values.tolist() if hasattr(sparse_embs[i], 'values') else []
            points.append(PointStruct(
                id=point_id,
                vector={ EMB_MODEL_NAME: dense_embs[i].tolist(), "bm25": SparseVector(indices=sparse_indices, values=sparse_values) },
                payload={"title": batch["title"][i], "text": batch["text"][i]}
            ))
            processed_in_batch += 1

        if points:
            await _client.upsert(collection_name=COLLECTION_NAME, points=points, wait=False)
            logging.debug(f"Upsert for batch {batch_num} sent ({len(points)} points).")
            return processed_in_batch # Return number successfully prepared for upsert
        else:
            logging.warning(f"Batch {batch_num}: No points generated.")
            return 0
    except Exception as e:
        logging.error(f"Error processing/upserting batch {batch_num}: {e}", exc_info=True)
        return 0 # Indicate failure for this batch by returning 0

# --- Foreground Task for Bulk Ingestion (Modified for UI Progress) ---
# This function is NO LONGER async itself, it orchestrates async calls.
def run_bulk_ingest_foreground_with_ui(_client, _dense_model, _sparse_model):
    """Phase 1b: Loads data and uploads it in groups, updating UI progress."""
    start_ingest_time = time.perf_counter()
    logging.info("Phase 1b: Starting FOREGROUND bulk ingest process with UI updates...")

    # --- UI Elements ---
    # These must be created *before* the loop that might block
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    status_placeholder.info("Initiating ingest process...")

    # --- Get Event Loop ---
    # We need the loop multiple times inside the synchronous function
    current_loop = get_or_create_eventloop()

    overall_successful = True # Track if any group failed
    ingest_time = 0.0
    has_data = False # Assume no data initially

    try:
        logging.info("Loading SciFact corpus dataset...")
        status_placeholder.info("Loading dataset...")
        # Load the dataset SYNCHRONOUSLY before the loop
        ds = load_dataset("BeIR/scifact", "corpus", split="corpus", trust_remote_code=True)
        total_docs = len(ds)
        logging.info(f"Dataset loaded. Total documents: {total_docs}.")

        if total_docs == 0:
             status_placeholder.warning("Dataset is empty. Nothing to ingest.")
             return False, 0.0, False # Failed, 0 time, no data

        total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
        num_groups = (total_batches + INGEST_CONCURRENCY - 1) // INGEST_CONCURRENCY

        processed_docs_count = 0
        processed_batches_count = 0

        # --- Loop through groups, calling async helper for each ---
        status_placeholder.info(f"Starting upload of {total_docs} docs in {total_batches} batches ({num_groups} groups)...")
        time.sleep(0.1) # Use time.sleep in a synchronous function
        
        for group_idx in range(num_groups):
            group_start_batch_num = group_idx * INGEST_CONCURRENCY + 1
            group_end_batch_num = min((group_idx + 1) * INGEST_CONCURRENCY, total_batches)

            # Calculate document indices for the slice
            doc_start_index = group_idx * INGEST_CONCURRENCY * BATCH_SIZE
            doc_end_index = min((group_idx + 1) * INGEST_CONCURRENCY * BATCH_SIZE, total_docs)

            # Get the data slice for the current group
            # Slicing datasets returns a dictionary of lists/arrays
            group_data_slice = ds[doc_start_index : doc_end_index]
            docs_in_group = len(group_data_slice['text'])
            batches_in_group = (docs_in_group + BATCH_SIZE - 1) // BATCH_SIZE

            if docs_in_group == 0:
                logging.warning(f"Group {group_idx+1}/{num_groups}: Calculated slice is empty, skipping.")
                continue

            log_text = f"Processing group {group_idx+1}/{num_groups} (Batches {group_start_batch_num}-{group_end_batch_num}, ~{docs_in_group} docs)..."
            logging.info(log_text)
            status_placeholder.info(log_text + " This step blocks UI.") # Indicate blocking part

            # --- Run the async processing for THIS GROUP ONLY ---
            # This call BLOCKS the Streamlit UI for the duration of this group's processing
            group_success = current_loop.run_until_complete(
                process_batch_group(
                    _client, group_data_slice, group_start_batch_num, total_batches,
                    _dense_model, _sparse_model
                )
            )
            # --- Control returns to Streamlit here ---

            if not group_success:
                error_msg = f"Group {group_idx+1} failed. Aborting ingest."
                logging.error(error_msg)
                status_placeholder.error(error_msg)
                overall_successful = False
                break # Stop processing further groups

            # --- Update Progress ---
            processed_docs_count += docs_in_group
            processed_batches_count += batches_in_group
            # Ensure counts don't exceed totals due to rounding/slicing logic
            processed_docs_count = min(processed_docs_count, total_docs)
            processed_batches_count = min(processed_batches_count, total_batches)

            progress_percentage = processed_docs_count / total_docs if total_docs > 0 else 0.0
            progress_bar.progress(progress_percentage)
            status_placeholder.info(f"Completed ~{processed_docs_count}/{total_docs} documents ({progress_percentage:.1%}). Processed group {group_idx+1}/{num_groups}.")
            # Optional small sleep to help ensure UI renders update, might not be needed
            # await asyncio.sleep(0.05) # Cannot await in sync function, use time.sleep
            time.sleep(0.05)

        # --- After the loop ---
        if overall_successful:
            logging.info("All groups processed. Waiting a few seconds for final upserts...")
            status_placeholder.info("Waiting for final upserts...")
            # Use run_until_complete for the async sleep
            current_loop.run_until_complete(asyncio.sleep(5))

            # Verify count
            final_count_res = current_loop.run_until_complete(
                _client.count(collection_name=COLLECTION_NAME, exact=False)
            )
            final_count = final_count_res.count
            logging.info(f"Final approximate count in Qdrant: {final_count}")

            if final_count > 0:
                has_data = True
                logging.info("Bulk ingest completed successfully.")
                status_placeholder.success(f"Ingest complete. Final count: ~{final_count}")
            else:
                logging.warning(f"Ingest finished, but Qdrant count is {final_count}.")
                status_placeholder.warning(f"Ingest finished, but Qdrant count is {final_count}. Check logs.")
                overall_successful = False # Treat 0 count as failure
        else:
             # Already logged the error during the group failure
             pass # Status placeholder should show the error

    except Exception as e:
        error_msg = f"Bulk ingest failed during setup or coordination: {type(e).__name__}"
        logging.error(error_msg, exc_info=True)
        status_placeholder.error(f"{error_msg} - See console logs.")
        overall_successful = False
        has_data = False # Unlikely to have data if setup failed

    finally:
        end_ingest_time = time.perf_counter()
        ingest_time = end_ingest_time - start_ingest_time
        logging.info(f"Bulk ingest coordination ended after {ingest_time:.2f}s. Success: {overall_successful}")
        # Clear progress bar only on success? Or always? Let's clear it after a delay or leave it.
        # Clear status message after a delay maybe?
        # For now, let the final status message persist. You could add:
        # time.sleep(5)
        # status_placeholder.empty()
        # progress_bar.empty()

    # Return results for state update
    return overall_successful, ingest_time, has_data

# --- Helper for processing a group of batches ---
async def process_batch_group(_client, group_data_slice, group_start_batch_num, total_batches, _dense_model, _sparse_model):
    """Processes a specific group of batches concurrently."""
    tasks = []
    num_batches_in_group = (len(group_data_slice['text']) + BATCH_SIZE - 1) // BATCH_SIZE
    logging.debug(f"Group {group_start_batch_num}: Processing {num_batches_in_group} batches within this group.")

    for i in range(0, len(group_data_slice['text']), BATCH_SIZE):
        batch_num = group_start_batch_num + (i // BATCH_SIZE)
        # Create a batch dictionary compatible with upload_batch
        batch_dict = {
            'text': group_data_slice['text'][i : i + BATCH_SIZE],
            '_id': group_data_slice['_id'][i : i + BATCH_SIZE],
            'title': group_data_slice['title'][i : i + BATCH_SIZE]
            # Add other fields if needed by upload_batch payload
        }
        if not batch_dict['text']: continue # Skip empty batches at the end

        tasks.append(
            upload_batch(_client, batch_dict, batch_num, total_batches, _dense_model, _sparse_model)
        )

    if not tasks:
        logging.warning(f"Group {group_start_batch_num}: No tasks generated for this data slice.")
        return True # Nothing to do, considered successful for this group

    try:
        await asyncio.gather(*tasks)
        logging.debug(f"Group {group_start_batch_num}: asyncio.gather completed.")
        return True # Group processed successfully
    except Exception as e:
        logging.error(f"Error processing group starting at batch {group_start_batch_num}: {e}", exc_info=True)
        return False # Indicate failure for this group

# --- Search Function (do_query - unchanged from previous binary quant version) ---
async def do_query(_client, query_text, method, top_k, _dense_model, _sparse_model):
    logging.info(f"Performing '{method}' search for query: '{query_text[:50]}...'")
    start_time = time.time()
    response = None
    try:
        dense_q = next(_dense_model.embed(query_text))
        sparse_q = next(_sparse_model.embed(query_text))
        if method == "Dense":
            response = await _client.search( collection_name=COLLECTION_NAME, query_vector=NamedVector(name=EMB_MODEL_NAME, vector=dense_q.tolist()), limit=top_k, with_payload=True,
                search_params=SearchParams( hnsw_ef=HNSW_EF_SEARCH, quantization=QuantizationSearchParams(ignore=False, rescore=True, oversampling=OVERSAMPLING) ) )
        elif method == "Sparse":
            response = await _client.search( collection_name=COLLECTION_NAME, query_vector=NamedSparseVector( name="bm25", vector=SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist()) ), limit=top_k, with_payload=True )
        elif method == "RRF":
            dense_limit = int(top_k * OVERSAMPLING); sparse_limit = top_k
            prefetch_list = [ Prefetch( query=dense_q.tolist(), using=EMB_MODEL_NAME, limit=dense_limit ), Prefetch( query=SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist()), using="bm25", limit=sparse_limit ) ]
            response = await _client.query_points( collection_name=COLLECTION_NAME, prefetch=prefetch_list, query=FusionQuery(fusion=Fusion.RRF), limit=top_k, with_payload=True,
                 search_params=SearchParams( quantization=QuantizationSearchParams(ignore=False, rescore=True, oversampling=OVERSAMPLING ) ) )
        else: logging.error(f"Unknown method: {method}"); st.error(f"Unknown method: {method}"); return None, 0.0
        elapsed_time = time.time() - start_time
        logging.info(f"Search completed in {elapsed_time:.4f} seconds.")
        points_to_return = response if isinstance(response, list) else (response.points if hasattr(response, 'points') else [])
        return points_to_return, elapsed_time
    except Exception as e: logging.error(f"Search failed {method}: {e}", exc_info=True); st.error(f"Search failed: {e}"); return None, time.time() - start_time


# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(layout="wide", page_title="Qdrant 2-Phase FG UI Prog")
    st.title("🔍 Qdrant Search (2-Phase Ingest, FG w/ UI Progress)")
    st.write(f"Using **{EMB_MODEL_NAME}** (Quantized) + **BM25**. Indexing (Phase 1b) is foreground with UI progress updates.")

    # --- Initialize client and models ---
    client = get_client()
    dense_model, sparse_model = load_models()

    # --- Initialize Session State ---
    if "collection_created" not in st.session_state: st.session_state.collection_created = False
    if "indexing_complete" not in st.session_state: st.session_state.indexing_complete = False
    if "hnsw_enabled" not in st.session_state: st.session_state.hnsw_enabled = False
    if "collection_has_data" not in st.session_state: st.session_state.collection_has_data = False # Track if ingest likely added data
    # Search state remains the same
    if "last_query" not in st.session_state: st.session_state.last_query = ""
    if "last_method" not in st.session_state: st.session_state.last_method = "RRF"
    if "last_top_k" not in st.session_state: st.session_state.last_top_k = 10
    if "last_search_results" not in st.session_state: st.session_state.last_search_results = None
    if "last_search_time" not in st.session_state: st.session_state.last_search_time = None
    # Add state for chunked progress
    if "ingest_progress" not in st.session_state: st.session_state.ingest_progress = 0.0
    if "ingest_processed_count" not in st.session_state: st.session_state.ingest_processed_count = 0
    if "ingest_total_docs" not in st.session_state: st.session_state.ingest_total_docs = 0


    # --- Sidebar ---
    with st.sidebar:
        st.header("Ingestion Pipeline")

        # --- Phase 1a Button ---
        if st.button("Phase 1a: Create Collection (m=0)", help="Creates/recreates the collection with HNSW disabled."):
             # Reset all subsequent phase states when recreating
            with st.spinner("Creating collection..."):
                current_loop = get_or_create_eventloop()
                success = current_loop.run_until_complete(create_collection_phase1(client))
                st.session_state.collection_created = success
                st.session_state.indexing_complete = False
                st.session_state.hnsw_enabled = False
                st.session_state.collection_has_data = False
                # Reset progress tracking too
                st.session_state.ingest_progress = 0.0
                st.session_state.ingest_processed_count = 0
                st.session_state.ingest_total_docs = 0
                if success: st.success("Phase 1a Complete.")
            st.rerun() # Rerun to update UI state

        # --- Phase 1b Button ---
        phase1b_disabled = not st.session_state.collection_created or st.session_state.indexing_complete
        if st.button("Phase 1b: Bulk Ingest Data (FG)", disabled=phase1b_disabled, help="Uploads SciFact corpus data. Blocks UI per group, shows progress."):
            if not st.session_state.collection_created:
                 st.warning("Collection must be created first (Phase 1a).")
            else:
                # No need for st.spinner here as the function updates UI itself
                try:
                    # Call the *synchronous* orchestrator function directly
                    success, time_taken, has_data = run_bulk_ingest_foreground_with_ui(
                        client, dense_model, sparse_model
                    )
                    st.session_state.indexing_complete = success
                    st.session_state.collection_has_data = has_data # Store if data likely exists
                    st.session_state.hnsw_enabled = False # HNSW still needs enabling

                    # Final status reporting based on the return values
                    if success:
                        st.success(f"Phase 1b Complete: Bulk ingest finished in {time_taken:.2f}s.")
                    else:
                        st.error("Phase 1b Failed: Bulk ingest encountered errors during processing.")
                except Exception as e:
                    logging.error(f"Exception during foreground ingest trigger: {e}", exc_info=True)
                    st.error(f"Ingest run failed unexpectedly: {e}")
                    st.session_state.indexing_complete = False
                    st.session_state.collection_has_data = False
                st.rerun() # Rerun to update the status indicators sidebar correctly

        # --- Phase 2 Button (Unchanged) ---
        phase2_disabled = not st.session_state.indexing_complete or st.session_state.hnsw_enabled
        if st.button(f"Phase 2: Enable HNSW (m={DENSE_M_PROD})", disabled=phase2_disabled, help="Builds the HNSW graph."):
            if not st.session_state.indexing_complete: st.warning("Data must be ingested first (Phase 1b).")
            else:
                 with st.spinner(f"Enabling HNSW (m={DENSE_M_PROD}) and optimizing..."):
                    current_loop = get_or_create_eventloop()
                    try:
                        success = current_loop.run_until_complete(enable_hnsw_phase2(client))
                        st.session_state.hnsw_enabled = success
                        if success: st.success("Phase 2 Complete.")
                        else: st.error("Phase 2 Failed.")
                    except Exception as e: logging.error(f"Exception HNSW trigger: {e}", exc_info=True); st.error(f"HNSW enabling run failed: {e}"); st.session_state.hnsw_enabled = False
                 st.rerun()

        # --- Status Display (Unchanged) ---
        st.divider(); st.header("Current Status")
        if st.session_state.collection_created: st.success("✅ Phase 1a: Collection Created")
        else: st.warning("⚪ Phase 1a: Collection Not Created")
        if st.session_state.indexing_complete: st.success("✅ Phase 1b: Data Ingested")
        elif st.session_state.collection_created: st.warning("⚪ Phase 1b: Data Not Ingested")
        else: st.caption("⚪ Phase 1b: (Requires Collection)")
        if st.session_state.hnsw_enabled: st.success(f"✅ Phase 2: HNSW Enabled (m={DENSE_M_PROD})")
        elif st.session_state.indexing_complete: st.warning("⚪ Phase 2: HNSW Not Enabled")
        else: st.caption("⚪ Phase 2: (Requires Ingest)")

    # --- Main Area for Search (Unchanged) ---
    st.header("🔎 Search the Collection")
    query_text = st.text_input("Enter search query:", value=st.session_state.last_query, placeholder="e.g., What is the effect of remdesivir?")
    method_options = ["Dense", "Sparse", "RRF"]; method_idx = method_options.index(st.session_state.last_method) if st.session_state.last_method in method_options else 2
    method = st.selectbox("Select search method:", method_options, index=method_idx)
    top_k = st.slider("Top K results", 1, 50, value=st.session_state.last_top_k)
    status_placeholder = st.empty(); results_placeholder = st.container()

    can_search = st.session_state.collection_created and st.session_state.indexing_complete and st.session_state.hnsw_enabled
    search_tooltip = "Search (requires all phases complete)." if not can_search else "Run search."
    if st.button("Search", type="primary", disabled=not can_search, help=search_tooltip):
        if not query_text: st.warning("Enter a search query.")
        else:
            st.session_state.last_search_results = None; st.session_state.last_search_time = None
            status_placeholder.empty(); results_placeholder.empty()
            points = None; elapsed_time = 0.0
            with st.spinner(f"Searching with {method} (Binary Quant Aware)..."):
                try:
                    current_loop = get_or_create_eventloop()
                    points, elapsed_time = current_loop.run_until_complete( do_query(client, query_text, method, top_k, dense_model, sparse_model) )
                    st.session_state.last_query = query_text; st.session_state.last_method = method; st.session_state.last_top_k = top_k
                    st.session_state.last_search_time = elapsed_time; st.session_state.last_search_results = points
                except Exception as e: logging.error(f"Error do_query button: {e}", exc_info=True); status_placeholder.error(f"Search failed: {e}"); st.session_state.last_search_results = None
            st.rerun()

    if st.session_state.last_search_results is not None:
        result_count = len(st.session_state.last_search_results)
        status_placeholder.success(f"Top {result_count} results ({st.session_state.last_method}). Time: **{st.session_state.last_search_time:.3f}s**.")
        with results_placeholder:
            points_to_display = st.session_state.last_search_results
            if not points_to_display: st.info("No results found.")
            else:
                for i, p in enumerate(points_to_display):
                    payload=p.payload or {}; title=payload.get('title','N/A'); score=p.score; point_id=p.id
                    score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
                    with st.expander(f"**{i+1}. {title}** (Score: {score_str}, ID: {point_id})", expanded=(i<3)):
                        st.markdown(payload.get('text','N/A'))
    elif not can_search:
         if not st.session_state.collection_created: status_placeholder.warning("Complete Phase 1a: Create Collection.")
         elif not st.session_state.indexing_complete: status_placeholder.warning("Complete Phase 1b: Bulk Ingest Data.")
         elif not st.session_state.hnsw_enabled: status_placeholder.warning("Complete Phase 2: Enable HNSW.")

    st.divider()
    st.caption(f"Model: {EMB_MODEL_NAME} (Binary Quant) + BM25 | Ingestion: Foreground, 2-Phase w/ UI Prog | Qdrant: {QDRANT_URL}")

if __name__ == "__main__":
    main()