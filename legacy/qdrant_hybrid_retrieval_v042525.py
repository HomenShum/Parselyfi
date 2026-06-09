import streamlit as st
import asyncio
# import nest_asyncio # Not needed here
import time
# import uuid # No longer needed for task IDs
import logging
import pandas as pd # Keep for displaying search results if needed later, though not strictly required now
# from collections import defaultdict # No longer needed
# from itertools import islice # No longer needed
# from concurrent.futures import ThreadPoolExecutor # Not used

# --- Qdrant/Data Specific Imports ---
from datasets import load_dataset
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    SparseVector, Prefetch, FusionQuery, Fusion, NamedVector, NamedSparseVector
)
# Removed Ranx imports: Qrels, Run, evaluate, compare

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s'
)

# --- Qdrant App Configuration ---
QDRANT_URL = st.secrets.get("qdrant_url", "http://localhost:6333")
QDRANT_API_KEY = st.secrets.get("qdrant_api_key", None)
COLLECTION_NAME = "scifact_simple" # Foreground indexing specific name
INGEST_CONCURRENCY = 5 # Still relevant for asyncio.gather within indexing
BATCH_SIZE = 32
HNSW_EF_SEARCH = 64
# BENCHMARK_TOP_K removed
SEARCH_PREFETCH_LIMIT = 50 # Kept for RRF search

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
    return AsyncQdrantClient(
        url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False, timeout=600
    )

# --- Embedding models ---
@st.cache_resource
def load_embedding_models():
    logging.info("Loading embedding models (Dense, Sparse)...")
    start = time.time()
    dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    logging.info(f"Embedding models loaded in {time.time() - start:.2f} seconds.")
    return dense_model, sparse_model

# --- Async Helper Functions (Qdrant Related) ---

async def recreate_collection(_client):
    # Modified to remove benchmark state resets
    logging.info(f"Attempting to recreate collection '{COLLECTION_NAME}'...")
    st.warning(f"Recreating collection '{COLLECTION_NAME}' (Dense/Sparse only)...")
    delete_success = False
    try:
        logging.info(f"Checking existence and attempting delete for {COLLECTION_NAME}...")
        exists = await _client.collection_exists(collection_name=COLLECTION_NAME)
        if exists:
            delete_op = await _client.delete_collection(collection_name=COLLECTION_NAME, timeout=120)
            if delete_op:
                logging.info(f"Delete request for collection '{COLLECTION_NAME}' successful. Waiting...")
                delete_success = True
                await asyncio.sleep(5)
            else:
                logging.warning(f"Delete operation for collection '{COLLECTION_NAME}' returned False.")
        else:
            logging.info(f"Collection '{COLLECTION_NAME}' does not exist, skipping delete.")
            delete_success = True

    except Exception as e:
        logging.error(f"Error during pre-delete/wait phase for collection '{COLLECTION_NAME}': {e}", exc_info=True)
        st.warning(f"Could not ensure old collection deletion: {e}. Attempting creation anyway.")
        await asyncio.sleep(2)

    try:
        logging.info(f"Attempting to create collection '{COLLECTION_NAME}'...")
        await _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"all-MiniLM-L6-v2": models.VectorParams(size=384, distance=models.Distance.COSINE, hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200), on_disk=True)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))},
            optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000), timeout=120
        )
        logging.info(f"Collection '{COLLECTION_NAME}' created successfully.")
        # Reset relevant session state on successful creation
        st.session_state.collection_created = True
        st.session_state.indexing_complete = False # New collection needs indexing
        st.session_state.collection_has_data = False # New collection is empty
        # No benchmark state to reset
        st.success(f"Collection '{COLLECTION_NAME}' created successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to create collection '{COLLECTION_NAME}': {e}", exc_info=True); st.error(f"Failed to create collection: {e}")
        st.session_state.collection_created = False
        st.session_state.collection_has_data = False
        st.session_state.indexing_complete = False
        return False

# upload_batch remains the same, used by the indexing function
async def upload_batch(_client, batch, batch_num, total_batches, _dense_model, _sparse_model):
    logging.debug(f"Processing batch {batch_num}/{total_batches}")
    texts = batch["text"]
    try:
        if not all([_dense_model, _sparse_model]): raise ValueError("Required embedding models not loaded")
        dense_embs_iter = _dense_model.embed(texts, batch_size=len(texts))
        sparse_embs_iter = _sparse_model.embed(texts, batch_size=len(texts))
        dense_embs = list(dense_embs_iter)
        sparse_embs = list(sparse_embs_iter)

        if not dense_embs or not sparse_embs:
             logging.warning(f"Batch {batch_num}: Embedding generation returned empty lists.")
             return

        points = []
        for i, doc_id in enumerate(batch["_id"]):
            if i >= len(dense_embs) or i >= len(sparse_embs):
                logging.warning(f"Batch {batch_num}: Index {i} out of range for embeddings (len dense: {len(dense_embs)}, len sparse: {len(sparse_embs)}). Skipping doc_id {doc_id}.")
                continue

            try: point_id = int(doc_id)
            except ValueError: point_id = str(doc_id)

            sparse_indices = sparse_embs[i].indices.tolist() if hasattr(sparse_embs[i], 'indices') else []
            sparse_values = sparse_embs[i].values.tolist() if hasattr(sparse_embs[i], 'values') else []

            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "all-MiniLM-L6-v2": dense_embs[i].tolist(),
                    "bm25": models.SparseVector(indices=sparse_indices, values=sparse_values)
                },
                payload={"title": batch["title"][i], "text": batch["text"][i]}
            ))

        if points:
            await _client.upsert(collection_name=COLLECTION_NAME, points=points, wait=False)
            logging.debug(f"Upsert command for batch {batch_num} sent ({len(points)} points).")
        else:
             logging.warning(f"Batch {batch_num}: No points generated for upsert.")

    except Exception as e:
        logging.error(f"Error processing/upserting batch {batch_num}: {e}", exc_info=True)
        # Cannot use st.toast safely from foreground blocking task. Log only.

# --- MODIFIED: Foreground Task for Indexing ---
async def run_indexing_foreground(_client, _dense_model, _sparse_model):
    start_index_time = time.perf_counter()
    logging.info("Starting FOREGROUND indexing process...")
    # Use logging for progress; st elements won't update until the end

    indexing_successful = False
    final_count = 0
    total_docs = 0
    indexing_time = 0.0
    status_placeholder = st.empty() # Placeholder for temporary messages if needed

    try:
        logging.info("Loading dataset...")
        # status_placeholder.info("Loading dataset...") # Won't show during block
        ds = load_dataset("BeIR/scifact", "corpus", split="corpus", trust_remote_code=True)
        total_docs = len(ds)
        logging.info(f"Dataset loaded. Total docs: {total_docs}. Starting indexing...")
        # status_placeholder.info(f"Loaded {total_docs} docs. Starting indexing...")

        processed_count = 0
        processed_batches = 0
        total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
        upload_tasks = []

        # Create Batch Upload Tasks
        for i in range(0, total_docs, BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            batch_data_dict = ds[i : i + BATCH_SIZE]
            upload_tasks.append(
                upload_batch(_client, batch_data_dict, batch_num, total_batches, _dense_model, _sparse_model)
            )

        # Process Tasks Concurrently (within the foreground task)
        for i in range(0, len(upload_tasks), INGEST_CONCURRENCY):
            group = upload_tasks[i : i + INGEST_CONCURRENCY]
            if not group: break

            current_batch_start_num = (i // INGEST_CONCURRENCY) * INGEST_CONCURRENCY + 1
            current_batch_end_num = current_batch_start_num + len(group) - 1
            progress_text_log = f"Dispatching batches {current_batch_start_num}-{current_batch_end_num} of {total_batches}..."
            logging.info(progress_text_log)
            # status_placeholder.info(progress_text_log) # Won't show

            await asyncio.gather(*group) # This is where concurrent uploads happen

            processed_batches += len(group)
            processed_count = min(total_docs, processed_batches * BATCH_SIZE)
            progress = processed_count / total_docs if total_docs > 0 else 0
            progress_text_log = f"Processed ~{processed_count}/{total_docs} docs ({processed_batches}/{total_batches} batches, {progress:.1%})"
            logging.info(progress_text_log)
            # status_placeholder.info(progress_text_log) # Won't show

            await asyncio.sleep(0.01) # Small yield within the loop

        # Final Steps
        logging.info("Waiting for final upserts...")
        # status_placeholder.info("Waiting for final upserts...") # Won't show
        await asyncio.sleep(5)

        final_count_res = await _client.count(collection_name=COLLECTION_NAME, exact=False)
        final_count = final_count_res.count
        logging.info(f"Final approximate count in Qdrant: {final_count}")

        if final_count > 0:
            indexing_successful = True
            logging.info("Indexing completed successfully.")
            # Can optionally set a final status message here if placeholder is used
            # status_placeholder.success("Indexing checks complete.")
        else:
            logging.warning(f"Indexing finished, but Qdrant count is {final_count}.")
            # status_placeholder.warning(f"Indexing finished, but Qdrant count is {final_count}.")
            indexing_successful = False

    except Exception as e:
        error_msg = f"Indexing failed: {type(e).__name__}"
        logging.error("Indexing failed.", exc_info=True)
        st.error(f"{error_msg} - See console logs for details.") # Show error after block
        indexing_successful = False

    finally:
        end_index_time = time.perf_counter()
        indexing_time = end_index_time - start_index_time
        logging.info(f"Indexing processing ended after {indexing_time:.2f}s.")
        status_placeholder.empty() # Clear any temporary message

    # Return results for state update
    has_data = final_count > 0 if indexing_successful else False
    return indexing_successful, indexing_time, has_data


# --- Benchmark Function REMOVED ---

# do_query remains the same for interactive search
async def do_query(_client, query_text, method, top_k, _dense_model, _sparse_model):
    logging.info(f"Performing '{method}' search for query: '{query_text[:50]}...'")
    start_time = time.time()
    response = None
    try:
        dense_q = next(_dense_model.embed(query_text))
        sparse_q = next(_sparse_model.embed(query_text))

        if method == "Dense":
            response = await _client.search(collection_name=COLLECTION_NAME, query_vector=NamedVector(name="all-MiniLM-L6-v2", vector=dense_q.tolist()), limit=top_k, search_params=models.SearchParams(hnsw_ef=HNSW_EF_SEARCH), with_payload=True)
        elif method == "Sparse":
            response = await _client.search(collection_name=COLLECTION_NAME, query_vector=NamedSparseVector(name="bm25", vector=models.SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist())), limit=top_k, with_payload=True)
        elif method == "RRF":
            prefetch_list = [ Prefetch(query=dense_q.tolist(), using="all-MiniLM-L6-v2", limit=SEARCH_PREFETCH_LIMIT), Prefetch(query=models.SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist()), using="bm25", limit=SEARCH_PREFETCH_LIMIT) ]
            response = await _client.query_points(collection_name=COLLECTION_NAME, prefetch=prefetch_list, query=FusionQuery(fusion=Fusion.RRF), limit=top_k, with_payload=True)
        else:
            logging.error(f"Unknown search method: {method}"); st.error(f"Unknown search method: {method}"); return None, 0.0

        elapsed_time = time.time() - start_time
        logging.info(f"Search completed in {elapsed_time:.4f} seconds.")
        return response, elapsed_time

    except Exception as e:
        logging.error(f"Search query failed for method {method}: {e}", exc_info=True)
        st.error(f"Search failed: {e}")
        return None, time.time() - start_time

# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(layout="wide", page_title="Qdrant Search FG Index")
    st.title("🔍 Qdrant Search Demo (Foreground Indexing)")
    st.write("Indexing now runs as a blocking foreground task. Benchmarking removed.")

    # --- Initialize client and models ---
    client = get_client()
    dense_model, sparse_model = load_embedding_models()

    # --- Initialize Session State (Simplified) ---
    if "collection_created" not in st.session_state: st.session_state.collection_created = False
    if "collection_has_data" not in st.session_state: st.session_state.collection_has_data = False
    if "initial_check_done" not in st.session_state: st.session_state.initial_check_done = False
    if "indexing_complete" not in st.session_state: st.session_state.indexing_complete = False
    if "indexing_time" not in st.session_state: st.session_state.indexing_time = None
    # Benchmark state removed
    if "last_query" not in st.session_state: st.session_state.last_query = ""
    if "last_method" not in st.session_state: st.session_state.last_method = "RRF"
    if "last_top_k" not in st.session_state: st.session_state.last_top_k = 10
    if "last_search_results" not in st.session_state: st.session_state.last_search_results = None
    if "last_search_time" not in st.session_state: st.session_state.last_search_time = None
    # Background task state ('tasks') removed

    # --- Initial Check (Remains the same) ---
    if not st.session_state.initial_check_done:
        try:
            current_loop = get_or_create_eventloop()
            logging.info("Running initial check for collection existence and data...")
            exists = current_loop.run_until_complete(client.collection_exists(collection_name=COLLECTION_NAME))
            has_data = False
            if exists:
                st.session_state.collection_created = True
                logging.info(f"Collection '{COLLECTION_NAME}' found.")
                try:
                    count_res = current_loop.run_until_complete(client.count(collection_name=COLLECTION_NAME, exact=True))
                    if count_res.count > 0: has_data = True; logging.info("Collection has data (exact count).")
                    else: logging.info("Collection exists but is empty (exact count).")
                except Exception:
                    try:
                        count_res = current_loop.run_until_complete(client.count(collection_name=COLLECTION_NAME, exact=False))
                        if count_res.count > 0: has_data = True; logging.info("Collection has data (approx count).")
                        else: logging.info("Collection exists but is empty (approx count).")
                    except Exception as count_err_inner:
                         logging.warning(f"Could not get collection count: {count_err_inner}")
            else:
                st.session_state.collection_created = False
                logging.info(f"Collection '{COLLECTION_NAME}' not found.")

            st.session_state.collection_has_data = has_data
            st.session_state.indexing_complete = has_data # Set initial indexing state based on data presence
            st.session_state.initial_check_done = True
            logging.info(f"Initial check complete. State: created={st.session_state.collection_created}, has_data={st.session_state.collection_has_data}")
            st.rerun()
        except Exception as q_err:
            logging.error(f"Initial Qdrant check failed: {q_err}", exc_info=True)
            st.warning(f"Could not verify collection status on Qdrant: {q_err}")
            st.session_state.collection_created = False
            st.session_state.collection_has_data = False
            st.session_state.indexing_complete = False
            st.session_state.initial_check_done = True

    # --- Sidebar ---
    with st.sidebar:
        st.header("Collection Management")

        # Display status
        if st.session_state.collection_created:
            st.success(f"Collection '{COLLECTION_NAME}' exists.")
            if st.session_state.collection_has_data: st.info("Collection has data.")
            else: st.warning("Collection appears empty.")
        else:
            st.info(f"Collection '{COLLECTION_NAME}' needs creation.")

        # Recreate Button
        if st.button("Recreate Collection"):
             with st.spinner("Recreating collection..."):
                 current_loop = get_or_create_eventloop()
                 success = current_loop.run_until_complete(recreate_collection(client))
                 # No need to reset initial_check_done, state is known after recreate
                 st.session_state.initial_check_done = True # We know the state now
                 if success:
                     st.session_state.collection_created = True
                     st.session_state.collection_has_data = False
                     st.session_state.indexing_complete = False
                 else:
                     st.session_state.collection_created = False # Failed
                 st.rerun()

        st.divider()
        st.header("Data Operations")

        # --- MODIFIED: Indexing Button (Foreground) ---
        can_index = st.session_state.collection_created
        if st.button("Index Corpus (Foreground)", disabled=not can_index):
            if not can_index:
                st.warning("Create collection first.")
            else:
                # Reset relevant states before starting
                st.session_state.indexing_complete = False
                st.session_state.collection_has_data = False
                st.session_state.indexing_time = None
                # No benchmark state to reset

                with st.spinner("Indexing corpus... This will block the UI. See console for progress."):
                    try:
                        current_loop = get_or_create_eventloop()
                        # Call the foreground indexing function
                        success, time_taken, has_data = current_loop.run_until_complete(
                            run_indexing_foreground(client, dense_model, sparse_model)
                        )
                        # Update state AFTER the blocking call finishes
                        st.session_state.indexing_complete = success
                        st.session_state.indexing_time = time_taken
                        st.session_state.collection_has_data = has_data

                        # Display result message after spinner
                        if success:
                            st.success(f"Foreground indexing completed in {time_taken:.2f}s.")
                        else:
                            st.error("Foreground indexing failed. Check console logs.")

                    except Exception as e:
                        # Catch errors during the run_until_complete call itself
                        logging.error(f"Exception during foreground indexing trigger: {e}", exc_info=True)
                        st.error(f"Indexing run failed: {e}")
                        st.session_state.indexing_complete = False
                        st.session_state.indexing_time = None
                        st.session_state.collection_has_data = False # Assume data state is bad
                st.rerun() # Rerun to reflect updated state in UI elements

        # --- Display Indexing Status ---
        st.divider()
        st.header("Operation Status")
        if st.session_state.indexing_complete:
             st.success(f"Indexing completed in {st.session_state.indexing_time:.2f}s." if st.session_state.indexing_time else "Indexing completed.")
        elif st.session_state.collection_created and not st.session_state.collection_has_data:
             st.info("Collection exists but needs indexing.")
        else:
            # Covers case where collection doesn't exist or initial check failed
             st.caption("Indexing not completed.")

        # Benchmark Status section removed

    # --- Main Area for Search (Remains the same) ---
    st.header("🔎 Search the Collection")
    query_text = st.text_input("Enter search query:", value=st.session_state.last_query)
    method_options = ["Dense", "Sparse", "RRF"]
    try: method_index = method_options.index(st.session_state.last_method)
    except ValueError: method_index = method_options.index("RRF")
    method = st.selectbox("Select search method:", method_options, index=method_index)
    top_k = st.slider("Top K results", 1, 50, value=st.session_state.last_top_k)
    status_placeholder = st.empty(); results_placeholder = st.container()

    can_search = st.session_state.collection_created and st.session_state.collection_has_data
    if st.button("Search", type="primary", disabled=not can_search):
        if not can_search: st.warning("Collection must exist and contain indexed data to search.")
        elif not query_text: st.warning("Enter a search query.")
        else:
            st.session_state.last_search_results = None; st.session_state.last_search_time = None
            status_placeholder.empty(); results_placeholder.empty()
            resp = None; elapsed_time = 0.0
            with st.spinner(f"Searching with {method}..."):
                try:
                    current_loop = get_or_create_eventloop()
                    resp, elapsed_time = current_loop.run_until_complete(
                        do_query(client, query_text, method, top_k, dense_model, sparse_model)
                    )
                    st.session_state.last_query = query_text
                    st.session_state.last_method = method
                    st.session_state.last_top_k = top_k
                    st.session_state.last_search_time = elapsed_time
                    points_to_store = []
                    if resp is not None: points_to_store = resp if isinstance(resp, list) else (resp.points if hasattr(resp, 'points') else [])
                    st.session_state.last_search_results = points_to_store
                except Exception as e:
                    logging.error(f"Error running do_query via button: {e}", exc_info=True)
                    status_placeholder.error(f"Search execution failed: {e}")
                    st.session_state.last_search_results = None
            st.rerun()

    # Display Search Results
    if st.session_state.last_search_results is not None:
        status_placeholder.success(f"Displaying results for '{st.session_state.last_query}' ({st.session_state.last_method}, Top {st.session_state.last_top_k}). Query time: **{st.session_state.last_search_time:.3f} seconds**.")
        with results_placeholder:
            points_to_display = st.session_state.last_search_results
            if not points_to_display: st.info("No results found.")
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
    elif st.session_state.last_query and can_search: status_placeholder.info("Ready to search. Enter a query and click Search.")
    elif not can_search and st.session_state.collection_created: status_placeholder.warning(f"Collection '{COLLECTION_NAME}' exists but appears empty. Please Index Corpus.")
    elif not st.session_state.collection_created: status_placeholder.warning(f"Collection '{COLLECTION_NAME}' does not exist. Please Create Collection.")


    st.divider()
    st.caption("Demo using Qdrant, FastEmbed, Streamlit (Foreground Indexing), Datasets.") # Updated caption

if __name__ == "__main__":
    main()