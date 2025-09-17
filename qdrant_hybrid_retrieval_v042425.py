import streamlit as st
import asyncio
# import nest_asyncio # Not typically needed with Streamlit's threading model unless you have specific nested loop issues. Removed for clarity.
import time
import uuid # Keep for task IDs
import logging
import pandas as pd
from collections import defaultdict
# from itertools import islice # No longer explicitly used
# from concurrent.futures import ThreadPoolExecutor # Not used

# --- Qdrant/Data Specific Imports ---
from datasets import load_dataset
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    SparseVector, Prefetch, FusionQuery, Fusion, NamedVector, NamedSparseVector
)
from ranx import Qrels, Run, evaluate, compare

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s'
)

# --- Qdrant App Configuration ---
QDRANT_URL = st.secrets.get("qdrant_url", "http://localhost:6333")
QDRANT_API_KEY = st.secrets.get("qdrant_api_key", None)
COLLECTION_NAME = "scifact_simple_bg" # Keep separate name for background task demo
INGEST_CONCURRENCY = 5
BATCH_SIZE = 32
HNSW_EF_SEARCH = 64
BENCHMARK_TOP_K = 10
SEARCH_PREFETCH_LIMIT = 50

# --- Helper to get/create loop ---
# Ensures we have a running loop in the current thread context
# (Important for background tasks potentially running in different threads)
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
    # (Keep the existing robust recreate_collection logic)
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
        st.session_state.indexing_complete = False
        st.session_state.collection_has_data = False
        st.session_state.tasks = {} # Clear background tasks state
        st.success(f"Collection '{COLLECTION_NAME}' created successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to create collection '{COLLECTION_NAME}': {e}", exc_info=True); st.error(f"Failed to create collection: {e}")
        st.session_state.collection_created = False
        st.session_state.collection_has_data = False
        st.session_state.indexing_complete = False
        return False

async def upload_batch(_client, batch, batch_num, total_batches, _dense_model, _sparse_model):
    # (Keep the existing upload_batch logic)
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
        # Cannot use st.toast safely from background thread. Log only.

# --- MODIFIED: Background Task for Indexing with Detailed Progress ---
async def run_indexing_background(task_id: str, _client, _dense_model, _sparse_model):
    start_index_time = time.perf_counter()
    logging.info(f"Task {task_id}: Starting indexing process...")
    # Initial state update for UI
    st.session_state[f'task_status_{task_id}'] = "Running"
    st.session_state[f'task_progress_{task_id}'] = 0.0
    st.session_state[f'task_progress_text_{task_id}'] = "Loading dataset..."

    indexing_successful = False
    final_count = 0
    total_docs = 0
    try:
        # --- Load Dataset ---
        ds = load_dataset("BeIR/scifact", "corpus", split="corpus", trust_remote_code=True)
        total_docs = len(ds)
        logging.info(f"Task {task_id}: Dataset loaded. Total docs: {total_docs}.")
        # Update progress state
        st.session_state[f'task_progress_{task_id}'] = 0.0 # Still 0% done with processing
        st.session_state[f'task_progress_text_{task_id}'] = f"Loaded {total_docs} docs. Starting indexing..."

        processed_count = 0
        processed_batches = 0
        total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
        upload_tasks = []

        # --- Create Batch Upload Tasks ---
        for i in range(0, total_docs, BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            batch_data_dict = ds[i : i + BATCH_SIZE]
            upload_tasks.append(
                upload_batch(_client, batch_data_dict, batch_num, total_batches, _dense_model, _sparse_model)
            )

        # --- Process Tasks Concurrently ---
        for i in range(0, len(upload_tasks), INGEST_CONCURRENCY):
            current_asyncio_task = asyncio.current_task()
            if current_asyncio_task and current_asyncio_task.cancelled():
                logging.info(f"Task {task_id}: Cancellation detected before processing batch group starting at {i}.")
                raise asyncio.CancelledError()

            group = upload_tasks[i : i + INGEST_CONCURRENCY]
            if not group: break

            current_batch_start_num = (i // INGEST_CONCURRENCY) * INGEST_CONCURRENCY + 1
            current_batch_end_num = current_batch_start_num + len(group) - 1
            # Update status before await
            st.session_state[f'task_progress_text_{task_id}'] = f"Dispatching batches {current_batch_start_num}-{current_batch_end_num}/{total_batches}..."
            logging.info(f"Task {task_id}: {st.session_state[f'task_progress_text_{task_id}']}")

            await asyncio.gather(*group) # Run the concurrent uploads

            processed_batches += len(group)
            processed_count = min(total_docs, processed_batches * BATCH_SIZE)
            progress = processed_count / total_docs if total_docs > 0 else 0
            progress_text = f"Processed ~{processed_count}/{total_docs} docs ({processed_batches}/{total_batches} batches)"

            # Update session state for progress bar
            st.session_state[f'task_progress_{task_id}'] = progress
            st.session_state[f'task_progress_text_{task_id}'] = progress_text
            logging.info(f"Task {task_id}: {progress_text} ({progress:.2%})")

            await asyncio.sleep(0.05) # Yield control briefly

        # --- Final Steps ---
        st.session_state[f'task_progress_text_{task_id}'] = "Waiting for final upserts..."
        logging.info(f"Task {task_id}: Waiting for final upserts...")
        await asyncio.sleep(5)

        current_asyncio_task = asyncio.current_task()
        if current_asyncio_task and current_asyncio_task.cancelled():
             logging.info(f"Task {task_id}: Cancellation detected before final count.")
             raise asyncio.CancelledError()

        final_count_res = await _client.count(collection_name=COLLECTION_NAME, exact=False)
        final_count = final_count_res.count
        logging.info(f"Task {task_id}: Final approximate count in Qdrant: {final_count}")

        if final_count > 0:
            indexing_successful = True
            st.session_state[f'task_status_{task_id}'] = "Completed"
            st.session_state[f'task_result_{task_id}'] = f"Indexing complete. Processed {total_docs} docs. Final Qdrant count ~{final_count}."
            st.session_state[f'task_progress_{task_id}'] = 1.0 # Ensure progress hits 100%
            st.session_state[f'task_progress_text_{task_id}'] = "Complete" # Set final text
            logging.info(f"Task {task_id}: Indexing completed successfully.")
        else:
            st.session_state[f'task_status_{task_id}'] = "Failed"
            st.session_state[f'task_result_{task_id}'] = f"Indexing finished, but Qdrant count is {final_count}. Processed {total_docs} docs. Check logs."
            st.session_state[f'task_progress_{task_id}'] = 1.0 # Show 100% even on failure if all docs processed
            st.session_state[f'task_progress_text_{task_id}'] = f"Finished (Count: {final_count})"
            logging.warning(f"Task {task_id}: Indexing finished, but Qdrant count is {final_count}.")
            indexing_successful = False

    except asyncio.CancelledError:
        logging.info(f"Task {task_id}: Indexing task cancelled.")
        st.session_state[f'task_status_{task_id}'] = "Cancelled"
        st.session_state[f'task_result_{task_id}'] = "Indexing was cancelled."
        # Keep last progress value, or set text explicitly
        st.session_state[f'task_progress_text_{task_id}'] = "Cancelled"

    except Exception as e:
        error_msg = f"Indexing failed: {type(e).__name__}"
        logging.error(f"Task {task_id}: Indexing failed.", exc_info=True)
        st.session_state[f'task_status_{task_id}'] = "Failed"
        st.session_state[f'task_result_{task_id}'] = f"{error_msg} - See logs."
        st.session_state[f'task_progress_text_{task_id}'] = f"Failed: {error_msg}" # Update text on error

    finally:
        end_index_time = time.perf_counter()
        indexing_time = end_index_time - start_index_time
        logging.info(f"Task {task_id}: Indexing processing ended after {indexing_time:.2f}s.")

        if indexing_successful:
            st.session_state.indexing_complete = True
            st.session_state.collection_has_data = True
            st.session_state.indexing_time = indexing_time
        # Don't reset flags on failure/cancel, partial data might exist

        task_info = st.session_state.tasks.get(task_id)
        if task_info:
            task_info['is_running'] = False


# --- MODIFIED: Background Task for Benchmarking with Detailed Progress ---
async def run_benchmark_background(task_id: str, _client, _dense_model, _sparse_model):
    start_bench_time = time.perf_counter()
    logging.info(f"Task {task_id}: Starting benchmark...")
    # Initial state
    st.session_state[f'task_status_{task_id}'] = "Running"
    st.session_state[f'task_progress_{task_id}'] = 0.0
    st.session_state[f'task_progress_text_{task_id}'] = "Loading queries/QRELS..."

    results = {}
    method_total_times = {}
    benchmark_report_df = None
    total_queries = 0
    qrels = None
    benchmark_time = 0.0
    success = False

    # Define progress stages (approximate allocation)
    PROGRESS_LOAD = 0.05
    PROGRESS_EMBED = 0.25 # Total for embedding
    PROGRESS_SEARCH = 0.60 # Total for all searches
    PROGRESS_COMPARE = 0.10 # Total for comparison

    try:
        # --- Load Data ---
        queries_ds = load_dataset("BeIR/scifact", "queries", split="queries", trust_remote_code=True)
        qrels_ds = load_dataset("BeIR/scifact-qrels", split="train", trust_remote_code=True)
        qrels_dict = defaultdict(dict)
        for entry in qrels_ds: qrels_dict[str(entry["query-id"])][str(entry["corpus-id"])] = int(entry["score"])
        qrels = Qrels(qrels_dict, name="scifact")
        logging.info(f"Task {task_id}: Loaded {len(queries_ds)} queries, QRELS for {len(qrels_dict)} IDs.")
        if not qrels_dict: raise ValueError("QRELS dictionary is empty.")
        queries = [{"query-id": str(q["_id"]), "text": q["text"]} for q in queries_ds]
        total_queries = len(queries)
        if not queries: raise ValueError("No queries loaded.")
        # Update progress after loading
        st.session_state[f'task_progress_{task_id}'] = PROGRESS_LOAD
        st.session_state[f'task_progress_text_{task_id}'] = f"Loaded {total_queries} queries/QRELS."


        # --- Precompute Embeddings ---
        st.session_state[f'task_progress_text_{task_id}'] = "Precomputing query embeddings..."
        dense_qv, sparse_qv = [], []
        query_texts = [q["text"] for q in queries]
        embed_batch_size = 128
        for i in range(0, total_queries, embed_batch_size):
            current_asyncio_task = asyncio.current_task();
            if current_asyncio_task and current_asyncio_task.cancelled(): raise asyncio.CancelledError()

            batch_texts = query_texts[i : i+embed_batch_size]
            dense_qv.extend(list(_dense_model.embed(batch_texts)))
            sparse_qv.extend(list(_sparse_model.embed(batch_texts)))

            # Calculate progress within the embedding stage
            embed_progress = min(1.0, (i + len(batch_texts)) / total_queries)
            # Update overall progress state
            st.session_state[f'task_progress_{task_id}'] = PROGRESS_LOAD + (embed_progress * PROGRESS_EMBED)
            st.session_state[f'task_progress_text_{task_id}'] = f"Embedding queries... {min(i+len(batch_texts), total_queries)}/{total_queries}"
            await asyncio.sleep(0.01)

        logging.info(f"Task {task_id}: Query embeddings computed.")
        # Ensure progress reflects embedding completion before search starts
        st.session_state[f'task_progress_{task_id}'] = PROGRESS_LOAD + PROGRESS_EMBED
        st.session_state[f'task_progress_text_{task_id}'] = "Embeddings computed. Starting searches..."


        # --- Run Searches ---
        search_methods = ["Dense", "Sparse", "RRF"]
        num_methods = len(search_methods)
        progress_per_method = PROGRESS_SEARCH / num_methods

        for idx, method_name in enumerate(search_methods):
            current_asyncio_task = asyncio.current_task()
            if current_asyncio_task and current_asyncio_task.cancelled(): raise asyncio.CancelledError()

            method_key = method_name.lower().replace(" ", "_")
            logging.info(f"Task {task_id}: Performing {method_name} Search (top {BENCHMARK_TOP_K})...")
            # Update progress text for the current search method
            current_progress_base = PROGRESS_LOAD + PROGRESS_EMBED + (idx * progress_per_method)
            st.session_state[f'task_progress_id_{task_id}'] = current_progress_base # Show start of this method's progress
            st.session_state[f'task_progress_text_{task_id}'] = f"Running {method_name} search ({idx+1}/{num_methods})..."

            search_start_time = time.time()
            run_dict = {}
            tasks = []
            results_list = []

            try:
                # Prepare search tasks (same as before)
                if method_name == "Dense":
                    for i in range(total_queries): tasks.append(_client.search(collection_name=COLLECTION_NAME, query_vector=NamedVector(name="all-MiniLM-L6-v2", vector=dense_qv[i].tolist()), limit=BENCHMARK_TOP_K, search_params=models.SearchParams(hnsw_ef=HNSW_EF_SEARCH), with_payload=False))
                elif method_name == "Sparse":
                    for i in range(total_queries):
                        sparse_q = sparse_qv[i]
                        tasks.append(_client.search(collection_name=COLLECTION_NAME, query_vector=NamedSparseVector(name="bm25", vector=models.SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist())), limit=BENCHMARK_TOP_K, with_payload=False))
                elif method_name == "RRF":
                    for i in range(total_queries):
                        sparse_q = sparse_qv[i]
                        prefetch = [ models.Prefetch(query=dense_qv[i].tolist(), using="all-MiniLM-L6-v2", limit=SEARCH_PREFETCH_LIMIT), models.Prefetch(query=models.SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist()), using="bm25", limit=SEARCH_PREFETCH_LIMIT) ]
                        tasks.append(_client.query_points(collection_name=COLLECTION_NAME, prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=BENCHMARK_TOP_K, with_payload=False))

                # Execute search tasks
                results_list = await asyncio.gather(*tasks)

                # Process results (same as before)
                if method_name in ["Dense", "Sparse"]: run_dict = { queries[i]["query-id"]: {str(p.id): p.score for p in batch_result} for i, batch_result in enumerate(results_list) }
                elif method_name == "RRF": run_dict = { queries[i]["query-id"]: {str(p.id): p.score for p in resp.points} for i, resp in enumerate(results_list) }

                search_time = time.time() - search_start_time
                method_total_times[method_key] = search_time

                if run_dict:
                    processed_run_dict = {qid: {doc_id: float(score) for doc_id, score in hits.items()} for qid, hits in run_dict.items()}
                    results[method_key] = Run(processed_run_dict, name=method_key)
                    logging.info(f"Task {task_id}: {method_name} search complete in {search_time:.2f}s.")
                else:
                    logging.warning(f"Task {task_id}: No results generated for {method_name} search.")

            except Exception as search_err:
                 logging.error(f"Task {task_id}: Error during {method_name} search", exc_info=True)

            # Update overall progress after completing this method
            st.session_state[f'task_progress_{task_id}'] = current_progress_base + progress_per_method
            st.session_state[f'task_progress_text_{task_id}'] = f"Completed {method_name} search..."
            await asyncio.sleep(0.01)

        # --- Compare Results ---
        st.session_state[f'task_progress_text_{task_id}'] = "Comparing results..."
        logging.info(f"Task {task_id}: Comparing results...")
        active_runs = [run for run_name, run in results.items() if run is not None and len(run.run) > 0]

        if not active_runs:
            raise ValueError("No successful benchmark runs to compare.")

        metrics_to_evaluate = [f"precision@{BENCHMARK_TOP_K}", f"recall@{BENCHMARK_TOP_K}", f"mrr@{BENCHMARK_TOP_K}", f"ndcg@{BENCHMARK_TOP_K}"]
        benchmark_report = compare(qrels=qrels, runs=active_runs, metrics=metrics_to_evaluate, make_comparable=True)

        report_df = benchmark_report.to_dataframe()
        if total_queries > 0:
            avg_method_times = {name: total_time / total_queries for name, total_time in method_total_times.items() if name in results}
            report_df['Avg Time (s)'] = report_df.index.map(avg_method_times).fillna(0)
            report_df['Avg Time (s)'] = report_df['Avg Time (s)'].map('{:.4f}'.format)
        else:
            report_df['Avg Time (s)'] = 'N/A'

        benchmark_report_df = report_df # Store result DataFrame
        success = True

        st.session_state[f'task_status_{task_id}'] = "Completed"
        st.session_state[f'task_result_{task_id}'] = benchmark_report_df # Store DF in result
        st.session_state[f'task_progress_{task_id}'] = 1.0 # Final progress
        st.session_state[f'task_progress_text_{task_id}'] = "Benchmark complete."
        logging.info(f"Task {task_id}: Benchmark finished successfully.")

    except asyncio.CancelledError:
        logging.info(f"Task {task_id}: Benchmark task cancelled.")
        st.session_state[f'task_status_{task_id}'] = "Cancelled"
        st.session_state[f'task_result_{task_id}'] = "Benchmark was cancelled."
        st.session_state[f'task_progress_text_{task_id}'] = "Cancelled" # Update text

    except Exception as e:
        error_msg = f"Benchmark failed: {type(e).__name__}"
        logging.error(f"Task {task_id}: Benchmark failed.", exc_info=True)
        st.session_state[f'task_status_{task_id}'] = "Failed"
        st.session_state[f'task_result_{task_id}'] = f"{error_msg} - See logs."
        st.session_state[f'task_progress_text_{task_id}'] = f"Failed: {error_msg}" # Update text
        success = False # Ensure success flag is false

    finally:
        end_bench_time = time.perf_counter()
        benchmark_time = end_bench_time - start_bench_time
        logging.info(f"Task {task_id}: Benchmark processing ended after {benchmark_time:.2f}s.")

        if success: # Update global state only on full success
            st.session_state.benchmark_report_df = benchmark_report_df
            st.session_state.benchmark_time = benchmark_time

        task_info = st.session_state.tasks.get(task_id)
        if task_info:
            task_info['is_running'] = False


async def do_query(_client, query_text, method, top_k, _dense_model, _sparse_model):
    # (Keep the existing do_query logic)
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
    st.set_page_config(layout="wide", page_title="Qdrant Hybrid Search BG Progress") # New Title
    st.title("🔍 Hybrid Qdrant Search (Background Tasks w/ Progress)") # New Title
    st.write("Indexing and Benchmarking run as non-blocking background tasks with progress bars.")

    # --- Initialize client and models ---
    client = get_client()
    dense_model, sparse_model = load_embedding_models()

    # --- Initialize Session State (Ensure all keys used are present) ---
    if "collection_created" not in st.session_state: st.session_state.collection_created = False
    if "collection_has_data" not in st.session_state: st.session_state.collection_has_data = False
    if "initial_check_done" not in st.session_state: st.session_state.initial_check_done = False
    if "indexing_complete" not in st.session_state: st.session_state.indexing_complete = False
    if "indexing_time" not in st.session_state: st.session_state.indexing_time = None
    if "benchmark_report_df" not in st.session_state: st.session_state.benchmark_report_df = None
    if "benchmark_time" not in st.session_state: st.session_state.benchmark_time = None
    if "last_query" not in st.session_state: st.session_state.last_query = ""
    if "last_method" not in st.session_state: st.session_state.last_method = "RRF"
    if "last_top_k" not in st.session_state: st.session_state.last_top_k = 10
    if "last_search_results" not in st.session_state: st.session_state.last_search_results = None
    if "last_search_time" not in st.session_state: st.session_state.last_search_time = None
    if 'tasks' not in st.session_state: st.session_state.tasks = {}

    # --- Initial Check (Keep existing robust check) ---
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
                    # Prefer exact count for initial check if possible
                    count_res = current_loop.run_until_complete(client.count(collection_name=COLLECTION_NAME, exact=True))
                    if count_res.count > 0: has_data = True; logging.info("Collection has data (exact count).")
                    else: logging.info("Collection exists but is empty (exact count).")
                except Exception: # Fallback to approximate on error/timeout
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
            # Assume indexing complete if data is found, otherwise needs indexing
            st.session_state.indexing_complete = has_data
            st.session_state.initial_check_done = True
            logging.info(f"Initial check complete. State: created={st.session_state.collection_created}, has_data={st.session_state.collection_has_data}")
            st.rerun()
        except Exception as q_err:
            logging.error(f"Initial Qdrant check failed: {q_err}", exc_info=True)
            st.warning(f"Could not verify collection status on Qdrant: {q_err}")
            st.session_state.collection_created = False
            st.session_state.collection_has_data = False
            st.session_state.indexing_complete = False
            st.session_state.initial_check_done = True # Prevent check loop


    # --- Sidebar ---
    with st.sidebar:
        st.header("Collection Management")
        # Status display (same as before)
        if st.session_state.collection_created:
            st.success(f"Collection '{COLLECTION_NAME}' exists.")
            if st.session_state.collection_has_data: st.info("Collection has data.")
            else: st.warning("Collection appears empty.")
        else:
            st.info(f"Collection '{COLLECTION_NAME}' needs creation.")

        # Action Buttons (same as before)
        if st.button("Recreate Collection"):
             with st.spinner("Recreating collection..."):
                 current_loop = get_or_create_eventloop()
                 success = current_loop.run_until_complete(recreate_collection(client))
                 st.session_state.initial_check_done = False # Re-check state after recreation
                 st.rerun()

        st.divider()
        st.header("Background Tasks")

        # Indexing Button (logic to start task is same)
        can_index = st.session_state.collection_created
        is_indexing_running = any(info.get('type') == 'indexing' and info.get('is_running', False) for info in st.session_state.tasks.values())
        if st.button("Index Corpus (Background)", disabled=not can_index or is_indexing_running):
            if not can_index: st.warning("Create collection first.")
            elif is_indexing_running: st.warning("An indexing task is already running.")
            else:
                task_id = f"idx-{uuid.uuid4().hex[:6]}"
                logging.info(f"Button clicked: Preparing indexing task {task_id}")
                current_loop = get_or_create_eventloop()
                try:
                    coro = run_indexing_background(task_id, client, dense_model, sparse_model)
                    task = current_loop.create_task(coro)
                    st.session_state.tasks[task_id] = {'task_obj': task, 'start_time': time.time(), 'is_running': True, 'type': 'indexing'}
                    # Initialize state for progress bar display
                    st.session_state[f'task_status_{task_id}'] = "Scheduled"
                    st.session_state[f'task_result_{task_id}'] = None
                    st.session_state[f'task_progress_{task_id}'] = 0.0
                    st.session_state[f'task_progress_text_{task_id}'] = "Waiting..."
                    logging.info(f"Indexing task {task_id} created and scheduled.")
                    st.rerun()
                except Exception as e:
                    logging.error(f"Failed to create/schedule indexing task {task_id}", exc_info=True)
                    st.error(f"Failed to start indexing task: {e}")

        # Benchmark Button (logic to start task is same)
        can_run_benchmark = st.session_state.collection_created and st.session_state.collection_has_data
        is_benchmark_running = any(info.get('type') == 'benchmark' and info.get('is_running', False) for info in st.session_state.tasks.values())
        if st.button("Run Benchmark (Background)", disabled=not can_run_benchmark or is_benchmark_running):
            if not can_run_benchmark: st.warning("Collection must exist and contain data.")
            elif is_benchmark_running: st.warning("A benchmark task is already running.")
            else:
                task_id = f"bench-{uuid.uuid4().hex[:6]}"
                logging.info(f"Button clicked: Preparing benchmark task {task_id}")
                current_loop = get_or_create_eventloop()
                try:
                    coro = run_benchmark_background(task_id, client, dense_model, sparse_model)
                    task = current_loop.create_task(coro)
                    st.session_state.tasks[task_id] = {'task_obj': task, 'start_time': time.time(), 'is_running': True, 'type': 'benchmark'}
                    # Initialize state for progress bar display
                    st.session_state[f'task_status_{task_id}'] = "Scheduled"
                    st.session_state[f'task_result_{task_id}'] = None
                    st.session_state[f'task_progress_{task_id}'] = 0.0
                    st.session_state[f'task_progress_text_{task_id}'] = "Waiting..."
                    logging.info(f"Benchmark task {task_id} created and scheduled.")
                    st.rerun()
                except Exception as e:
                    logging.error(f"Failed to create/schedule benchmark task {task_id}", exc_info=True)
                    st.error(f"Failed to start benchmark task: {e}")

        st.divider()
        st.header("Task Status")
        # --- MODIFIED: Display Background Task Statuses with Progress Bar ---
        if not st.session_state.tasks:
            st.caption("No background tasks running.")
        else:
            task_ids = list(st.session_state.tasks.keys()) # Avoid modification during iteration issues
            for task_id in task_ids:
                task_info = st.session_state.tasks.get(task_id)
                if not task_info: continue # Should not happen, but safety check

                # Retrieve state, providing defaults
                status = st.session_state.get(f'task_status_{task_id}', "Unknown")
                result = st.session_state.get(f'task_result_{task_id}', None)
                progress = st.session_state.get(f'task_progress_{task_id}', 0.0) # Default to 0.0
                progress_text = st.session_state.get(f'task_progress_text_{task_id}', "") # Default to empty string
                task_type = task_info.get('type', 'Unknown').capitalize()
                start_time = task_info.get('start_time')
                elapsed = f" ({(time.time() - start_time):.1f}s)" if start_time and status in ["Scheduled", "Running"] else ""

                # Use an expander for each task
                with st.expander(f"Task: {task_id} ({task_type}) - Status: {status}{elapsed}", expanded=(status in ["Running", "Scheduled"])):
                    col_status, col_control = st.columns([4, 1]) # Give more space for status/progress

                    with col_status:
                        # Display progress bar only when running
                        if status == "Running":
                            # Ensure progress is clamped between 0.0 and 1.0
                            clamped_progress = max(0.0, min(1.0, progress))
                            st.progress(clamped_progress, text=progress_text)
                        # Display text statuses for other states
                        elif status == "Scheduled":
                            st.info(f"⏳ {progress_text or 'Waiting to run...'}")
                        elif status == "Completed":
                            st.success(f"✅ Completed.")
                            # Show result details (like benchmark DataFrame)
                            if isinstance(result, pd.DataFrame):
                                st.dataframe(result)
                            elif result:
                                st.write(result)
                        elif status == "Failed":
                            st.error(f"❌ Failed: {result or 'Check logs.'}")
                            # Optionally show the last progress text if available
                            if progress_text not in ["", f"Failed: {result or 'Check logs.'}"]:
                                st.caption(f"Last status: {progress_text}")
                        elif status == "Cancelled":
                            st.warning(f"⏹️ Cancelled: {result or ''}")
                            if progress_text not in ["", "Cancelled"]:
                                st.caption(f"Last status: {progress_text}")
                        else:
                            st.warning(f"❓ Unknown status: {status}")

                    with col_control:
                        task_obj = task_info.get('task_obj')
                        # Cancel Button (only if running/scheduled and task exists)
                        if status in ["Scheduled", "Running"] and task_obj and not task_obj.done():
                            if st.button("Cancel", key=f"cancel_{task_id}", help="Request cancellation of the task."):
                                logging.info(f"Cancel button clicked for task {task_id}")
                                if task_obj.cancel():
                                    logging.info(f"Cancellation request sent for task {task_id}.")
                                    # Update status immediately for responsiveness (task itself handles actual cancellation)
                                    st.session_state[f'task_status_{task_id}'] = "Cancelling"
                                    st.session_state[f'task_progress_text_{task_id}'] = "Cancellation requested..."
                                else:
                                    logging.warning(f"Failed to send cancellation request for task {task_id}.")
                                st.rerun()
                        # Clear Button (only for terminal states)
                        elif status in ["Completed", "Failed", "Cancelled"]:
                             if st.button("Clear", key=f"clear_{task_id}", help="Remove this task from the list."):
                                 logging.info(f"Clearing task {task_id} from state.")
                                 # Clean up all associated state keys
                                 keys_to_delete = [
                                     f'task_status_{task_id}', f'task_result_{task_id}',
                                     f'task_progress_{task_id}', f'task_progress_text_{task_id}'
                                 ]
                                 for key in keys_to_delete:
                                     if key in st.session_state: del st.session_state[key]
                                 if task_id in st.session_state.tasks: del st.session_state.tasks[task_id]
                                 # Clear global state if a completed benchmark is cleared
                                 if task_type == 'Benchmark' and status == 'Completed':
                                     st.session_state.benchmark_report_df = None
                                     st.session_state.benchmark_time = None
                                 st.rerun()


    # --- Main Area for Search (Keep existing search logic) ---
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

    # --- Display Search Results (Keep existing logic) ---
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
    st.caption("Demo using Qdrant, FastEmbed, Streamlit Background Tasks w/ Progress, Datasets, Ranx.")

if __name__ == "__main__":
    main()