import streamlit as st
import asyncio
import nest_asyncio
from datasets import load_dataset
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
    NamedVector, # Added for clarity in search
    NamedSparseVector # Added for clarity in search
)
import time
from ranx import Qrels, Run, evaluate, compare
from collections import defaultdict
from itertools import islice
import logging # Added for better logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Apply nest_asyncio to allow running asyncio event loop within another loop (like Streamlit's)
nest_asyncio.apply()
# Get the current event loop
try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# --- Configuration ---
# Best practice: Load sensitive info from secrets
QDRANT_URL = st.secrets.get("qdrant_url", "http://localhost:6333") # Provide a default for local dev if needed
QDRANT_API_KEY = st.secrets.get("qdrant_api_key", None)
COLLECTION_NAME = "scifact"
INGEST_CONCURRENCY = 5       # number of batches to upload in parallel
BATCH_SIZE = 32
HNSW_EF_SEARCH = 64      # trade‐off between speed & recall
BENCHMARK_TOP_K = 10     # K for benchmark evaluation
SEARCH_PREFETCH_LIMIT = 50 # How many results to fetch per vector type for RRF (increased a bit)

# --- Client factory ---
@st.cache_resource # Cache the client for efficiency across reruns
def get_client():
    logging.info(f"Initializing Qdrant client for {QDRANT_URL}")
    return AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=600 # Generous timeout for potentially long operations
    )
# client instance is created when get_client() is first called by Streamlit

# --- Embedding models ---
# Use st.cache_resource to load models only once
@st.cache_resource
def load_embedding_models():
    logging.info("Loading embedding models...")
    start = time.time()
    dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    # Ensure you have sentence_transformers[sparse] and potentially other dependencies installed
    # pip install -U sentence-transformers fastembed qdrant-client[sparse] datasets ranx protobuf nest_asyncio
    late_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    logging.info(f"Embedding models loaded in {time.time() - start:.2f} seconds.")
    return dense_model, sparse_model, late_model
# model instances are created when load_embedding_models() is first called

# --- Async Helper Functions ---

async def recreate_collection(_client): # Pass client explicitly
    logging.info(f"Attempting to recreate collection '{COLLECTION_NAME}'...")
    st.warning(f"Recreating collection '{COLLECTION_NAME}'...") # Show status in UI
    try:
        await _client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "all-MiniLM-L6-v2": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
                    on_disk=True
                ),
                "colbertv2.0": models.VectorParams(
                    size=128,
                    distance=models.Distance.MAXSIM,
                    multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                    on_disk=True
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True)
                )
            },
            optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
            timeout=120 # Increase timeout for potentially long operation
        )
        logging.info(f"Collection '{COLLECTION_NAME}' recreated successfully.")
        st.success(f"Collection '{COLLECTION_NAME}' recreated successfully.")
    except Exception as e:
        logging.error(f"Failed to recreate collection: {e}", exc_info=True)
        st.error(f"Failed to recreate collection: {e}")

async def upload_batch(_client, batch, batch_num, total_batches, _dense_model, _sparse_model, _late_model): # Pass client and models
    logging.debug(f"Processing batch {batch_num}/{total_batches}")
    texts = batch["text"]
    try:
        # Ensure models are loaded if called outside Streamlit context (less likely here)
        if not all([_dense_model, _sparse_model, _late_model]):
             raise ValueError("Embedding models not loaded")

        dense_embs = list(_dense_model.embed(texts, batch_size=len(texts))) # Embed batch at once
        sparse_embs= list(_sparse_model.embed(texts, batch_size=len(texts)))
        late_embs  = list(_late_model.embed(texts, batch_size=len(texts)))

        points = []
        for i, doc_id in enumerate(batch["_id"]):
            try:
                point_id = int(doc_id)
            except ValueError:
                point_id = str(doc_id)

            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "all-MiniLM-L6-v2": dense_embs[i].tolist(),
                    "bm25": models.SparseVector(
                        indices=sparse_embs[i].indices.tolist(),
                        values=sparse_embs[i].values.tolist()
                    ),
                    "colbertv2.0": late_embs[i].tolist(),
                },
                payload={"title": batch["title"][i], "text": batch["text"][i]}
            ))

        await _client.upsert(collection_name=COLLECTION_NAME, points=points, wait=False)
        logging.debug(f"Upsert command for batch {batch_num} sent.")
    except Exception as e:
        logging.error(f"Error processing or upserting batch {batch_num}: {e}", exc_info=True)
        st.toast(f"Error in batch {batch_num}: {e}", icon="⚠️") # Non-blocking notification


async def run_indexing(_client, _dense_model, _sparse_model, _late_model): # Pass client and models
    logging.info("Starting indexing process...")
    st.info("Loading SciFact corpus dataset...")
    try:
        ds = load_dataset("BeIR/scifact", "corpus", split="corpus", trust_remote_code=True) # Added trust_remote_code
        total = len(ds)
        st.info(f"Dataset loaded. Total documents: {total}. Starting indexing...")
        progress_bar = st.progress(0, text="Initializing...")
        processed_count = 0
        processed_batches = 0

        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        uploads = []
        for i in range(0, total, BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            batch_data = ds[i : i + BATCH_SIZE]
            batch_dict = {key: batch_data[key] for key in batch_data.column_names}
            uploads.append(upload_batch(_client, batch_dict, batch_num, total_batches, _dense_model, _sparse_model, _late_model))

        task_status = st.empty()
        for i in range(0, len(uploads), INGEST_CONCURRENCY):
            group = uploads[i : i + INGEST_CONCURRENCY]
            if not group:
                break
            current_batch_start = (i // INGEST_CONCURRENCY) * INGEST_CONCURRENCY + 1
            current_batch_end = current_batch_start + len(group) -1
            task_status.info(f"Dispatching batches {current_batch_start}-{current_batch_end} of {total_batches}...")
            await asyncio.gather(*group)

            processed_batches += len(group)
            processed_count = min(total, processed_batches * BATCH_SIZE) # Update count based on batches processed
            progress = processed_count / total
            progress_text = f"Processed ~{processed_count}/{total} documents ({processed_batches}/{total_batches} batches)"
            progress_bar.progress(progress, text=progress_text)
            logging.info(progress_text)

        task_status.info("Waiting for final upserts to potentially settle...")
        await asyncio.sleep(5) # Simple wait as wait=False was used

        progress_bar.progress(1.0, text=f"Indexing complete. Processed {total} documents.")
        st.success("Indexing complete.")
        logging.info("Indexing process finished.")
    except Exception as e:
        logging.error(f"Indexing failed: {e}", exc_info=True)
        st.error(f"Indexing failed: {e}")


async def run_benchmark(_client, _dense_model, _sparse_model, _late_model): # Pass client and models
    logging.info("Starting benchmark...")
    st.info("Running benchmark...")
    results = {}
    active_runs = []
    try:
        st.write("Loading queries and QRELS...")
        queries_ds = load_dataset("BeIR/scifact", "queries", split="queries", trust_remote_code=True)
        qrels_ds = load_dataset("BeIR/scifact-qrels", split="train", trust_remote_code=True)

        qrels_dict = defaultdict(dict)
        for entry in qrels_ds:
            qid = entry["query-id"]
            did = entry["corpus-id"]
            qrels_dict[str(qid)][str(did)] = int(entry["score"])
        qrels = Qrels(qrels_dict, name="scifact")
        st.write(f"Loaded {len(queries_ds)} queries and QRELS for {len(qrels_dict)} query IDs.")
        if not qrels_dict:
             st.warning("QRELS dictionary is empty. Cannot perform benchmark.")
             return

        queries = [{"query-id": str(q["_id"]), "text": q["text"]} for q in queries_ds]
        if not queries:
             st.warning("No queries loaded. Cannot perform benchmark.")
             return

        st.write("Precomputing query embeddings...")
        progress_bar_emb = st.progress(0, text="Embedding queries...")
        total_queries = len(queries)
        dense_qv, sparse_qv, late_qv = [], [], []
        query_texts = [q["text"] for q in queries]
        embed_batch_size = 128
        for i in range(0, total_queries, embed_batch_size):
            batch_texts = query_texts[i : i+embed_batch_size]
            dense_qv.extend(list(_dense_model.embed(batch_texts)))
            sparse_qv.extend(list(_sparse_model.embed(batch_texts)))
            late_qv.extend(list(_late_model.embed(batch_texts))) # late_qv will contain list of multi-vector embeddings
            progress = min(1.0, (i + len(batch_texts)) / total_queries)
            progress_bar_emb.progress(progress, text=f"Embedding queries... {min(i+len(batch_texts), total_queries)}/{total_queries}")
        progress_bar_emb.progress(1.0, text="Query embeddings computed.")
        st.write("Embeddings computed.")

        search_methods = ["Dense", "Sparse", "Late", "RRF", "Full RRF"]
        progress_bar_search = st.progress(0, text="Starting searches...")

        for idx, method_name in enumerate(search_methods):
            st.write(f"Performing {method_name} Search (top {BENCHMARK_TOP_K})...")
            search_start_time = time.time()
            run_dict = {}
            # Use asyncio.gather for batching individual searches/queries for consistency
            tasks = []
            results_list = [] # To store results in order

            try:
                if method_name == "Dense":
                    for i in range(total_queries):
                        tasks.append(_client.search( # Use search (newer than query_points for simple vectors)
                            collection_name=COLLECTION_NAME,
                            query_vector=NamedVector(name="all-MiniLM-L6-v2", vector=dense_qv[i].tolist()),
                            limit=BENCHMARK_TOP_K,
                            search_params=models.SearchParams(hnsw_ef=HNSW_EF_SEARCH),
                            with_payload=False
                        ))
                    results_list = await asyncio.gather(*tasks)
                    run_dict = { queries[i]["query-id"]: {str(p.id): p.score for p in batch_result} for i, batch_result in enumerate(results_list) }

                elif method_name == "Sparse":
                    for i in range(total_queries):
                         sparse_q = sparse_qv[i]
                         tasks.append(_client.search(
                            collection_name=COLLECTION_NAME,
                            query_vector=NamedSparseVector(name="bm25", vector=models.SparseVector(
                                indices=sparse_q.indices.tolist(),
                                values=sparse_q.values.tolist()
                            )),
                            limit=BENCHMARK_TOP_K,
                            with_payload=False
                         ))
                    results_list = await asyncio.gather(*tasks)
                    run_dict = { queries[i]["query-id"]: {str(p.id): p.score for p in batch_result} for i, batch_result in enumerate(results_list) }

                elif method_name == "Late":
                    # --- FIX: Use query_points for ColBERT/Late Interaction ---
                    for i in range(total_queries):
                         # Pass the raw multi-vector embedding from late_qv
                         # Ensure late_qv[i] is in a format query_points accepts (e.g., list of lists, or ndarray)
                         multi_vector_query = late_qv[i]
                         if hasattr(multi_vector_query, 'tolist'): # Convert if it's a numpy array
                             multi_vector_query = multi_vector_query.tolist()

                         tasks.append(_client.query_points(
                            collection_name=COLLECTION_NAME,
                            query=multi_vector_query, # Pass the multi-vector directly
                            using="colbertv2.0",      # Specify the multi-vector config name
                            limit=BENCHMARK_TOP_K,
                            with_payload=False
                         ))
                    results_list = await asyncio.gather(*tasks) # results_list will contain QueryResponse objects
                    run_dict = { queries[i]["query-id"]: {str(p.id): p.score for p in resp.points} for i, resp in enumerate(results_list) }

                elif method_name == "RRF":
                    for i in range(total_queries):
                         sparse_q = sparse_qv[i]
                         prefetch = [
                             models.Prefetch(query=dense_qv[i].tolist(), using="all-MiniLM-L6-v2", limit=SEARCH_PREFETCH_LIMIT),
                             models.Prefetch(query=models.SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist()), using="bm25", limit=SEARCH_PREFETCH_LIMIT),
                         ]
                         tasks.append(_client.query_points(
                            collection_name=COLLECTION_NAME,
                            prefetch=prefetch,
                            query=models.FusionQuery(fusion=models.Fusion.RRF),
                            limit=BENCHMARK_TOP_K,
                            with_payload=False
                        ))
                    results_list = await asyncio.gather(*tasks) # results_list will contain QueryResponse objects
                    run_dict = { queries[i]["query-id"]: {str(p.id): p.score for p in resp.points} for i, resp in enumerate(results_list) }

                elif method_name == "Full RRF":
                     for i in range(total_queries):
                         sparse_q = sparse_qv[i]
                         late_q = late_qv[i]
                         if hasattr(late_q, 'tolist'):
                             late_q = late_q.tolist()
                         prefetch = [
                             models.Prefetch(query=dense_qv[i].tolist(), using="all-MiniLM-L6-v2", limit=SEARCH_PREFETCH_LIMIT),
                             models.Prefetch(query=models.SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist()), using="bm25", limit=SEARCH_PREFETCH_LIMIT),
                             models.Prefetch(query=late_q, using="colbertv2.0", limit=SEARCH_PREFETCH_LIMIT), # Pass raw multi-vector here too
                         ]
                         tasks.append(_client.query_points(
                             collection_name=COLLECTION_NAME,
                             prefetch=prefetch,
                             query=models.FusionQuery(fusion=models.Fusion.RRF),
                             limit=BENCHMARK_TOP_K,
                             with_payload=False
                         ))
                     results_list = await asyncio.gather(*tasks) # results_list will contain QueryResponse objects
                     run_dict = { queries[i]["query-id"]: {str(p.id): p.score for p in resp.points} for i, resp in enumerate(results_list) }

                # If run_dict was successfully populated
                if run_dict:
                    results[method_name.lower().replace(" ", "_")] = Run(run_dict, name=method_name.lower().replace(" ", "_"))
                    search_time = time.time() - search_start_time
                    st.write(f"{method_name} search complete in {search_time:.2f}s.")
                else:
                     st.warning(f"No results generated for {method_name} search.")

            except Exception as search_err:
                st.error(f"Error during {method_name} search: {search_err}")
                logging.error(f"Error during {method_name} search", exc_info=True)


            progress_bar_search.progress((idx + 1) / len(search_methods), text=f"Completed {method_name} search...")
        # --- End of loop through search_methods ---

        st.write("Comparing results...")
        active_runs = [run for run_name, run in results.items() if run is not None and len(run.run) > 0] # Filter results dict

        if not active_runs:
            st.warning("No benchmark runs were successfully completed or yielded results.")
            return

        metrics_to_evaluate = [f"precision@{BENCHMARK_TOP_K}", f"recall@{BENCHMARK_TOP_K}", f"mrr@{BENCHMARK_TOP_K}", f"ndcg@{BENCHMARK_TOP_K}"]
        report = compare(
            qrels=qrels,
            runs=active_runs,
            metrics=metrics_to_evaluate,
            make_comparable=True
        )
        st.success("Benchmark complete.")
        st.dataframe(report)
        logging.info("Benchmark finished.")

    except Exception as e:
        logging.error(f"Benchmark failed: {e}", exc_info=True)
        st.error(f"Benchmark failed: {e}")
        if active_runs and qrels:
            try:
                st.info("Displaying results for successfully completed runs:")
                metrics_to_evaluate = [f"precision@{BENCHMARK_TOP_K}", f"recall@{BENCHMARK_TOP_K}", f"mrr@{BENCHMARK_TOP_K}", f"ndcg@{BENCHMARK_TOP_K}"]
                report = compare(qrels=qrels, runs=active_runs, metrics=metrics_to_evaluate, make_comparable=True)
                st.dataframe(report)
            except Exception as report_err:
                st.error(f"Failed to generate partial report: {report_err}")

async def do_query(_client, query_text, method, top_k, _dense_model, _sparse_model, _late_model): # Pass client and models
    """Performs the actual search based on the selected method."""
    logging.info(f"Performing '{method}' search for query: '{query_text[:50]}...'")
    start_time = time.time()

    try:
        if method == "Dense":
            qvec = next(_dense_model.embed(query_text))
            response = await _client.search(
                collection_name=COLLECTION_NAME,
                query_vector=NamedVector(name="all-MiniLM-L6-v2", vector=qvec.tolist()),
                limit=top_k,
                search_params=models.SearchParams(hnsw_ef=HNSW_EF_SEARCH),
                with_payload=True
            )
        elif method == "Sparse":
            sparse_q = next(_sparse_model.embed(query_text))
            response = await _client.search(
                collection_name=COLLECTION_NAME,
                query_vector=NamedSparseVector(name="bm25", vector=models.SparseVector(
                    indices=sparse_q.indices.tolist(),
                    values=sparse_q.values.tolist()
                )),
                limit=top_k,
                with_payload=True
            )
        elif method == "Late":
            late_q = next(_late_model.embed(query_text))
            response = await _client.search(
                collection_name=COLLECTION_NAME,
                query_vector=NamedVector(name="colbertv2.0", vector=late_q.tolist()),
                limit=top_k,
                with_payload=True
            )
        elif method == "RRF":
            dense_q = next(_dense_model.embed(query_text))
            sparse_q = next(_sparse_model.embed(query_text))
            prefetch_list = [
                Prefetch(query=dense_q.tolist(), using="all-MiniLM-L6-v2", limit=SEARCH_PREFETCH_LIMIT),
                Prefetch(query=models.SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist()), using="bm25", limit=SEARCH_PREFETCH_LIMIT),
            ]
            response = await _client.query_points( # Changed to query_points for fusion
                collection_name=COLLECTION_NAME,
                prefetch=prefetch_list,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True
            )
        elif method == "Full RRF":
            dense_q = next(_dense_model.embed(query_text))
            sparse_q = next(_sparse_model.embed(query_text))
            late_q = next(_late_model.embed(query_text))
            prefetch_list = [
                Prefetch(query=dense_q.tolist(), using="all-MiniLM-L6-v2", limit=SEARCH_PREFETCH_LIMIT),
                Prefetch(query=models.SparseVector(indices=sparse_q.indices.tolist(), values=sparse_q.values.tolist()), using="bm25", limit=SEARCH_PREFETCH_LIMIT),
                Prefetch(query=late_q.tolist(), using="colbertv2.0", limit=SEARCH_PREFETCH_LIMIT),
            ]
            response = await _client.query_points( # Changed to query_points for fusion
                collection_name=COLLECTION_NAME,
                prefetch=prefetch_list,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True
            )
        else:
            logging.error(f"Unknown search method: {method}")
            st.error(f"Unknown search method: {method}")
            return None

        end_time = time.time()
        logging.info(f"Search completed in {end_time - start_time:.4f} seconds.")
        return response

    except Exception as e:
        logging.error(f"Search query failed for method {method}: {e}", exc_info=True)
        st.error(f"Search failed: {e}")
        return None


# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(layout="wide", page_title="Qdrant Hybrid Search")
    st.title("🔍 Hybrid Qdrant Async Search Demo")

    # Initialize client and models using caching
    # These will only compute on the first run or if the cache is cleared
    client = get_client()
    dense_model, sparse_model, late_model = load_embedding_models()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Indexing & Benchmarking")
        if st.button("(Re)Create Collection"):
            with st.spinner("Recreating collection... Please wait."):
                # Pass the client instance to the async function
                loop.run_until_complete(recreate_collection(client))

        if st.button("Index Corpus to Qdrant"):
            with st.spinner("Starting indexing process... This may take a while."):
                start_index_time = time.perf_counter()
                # Pass client and models to the async function
                loop.run_until_complete(run_indexing(client, dense_model, sparse_model, late_model))
                end_index_time = time.perf_counter()
                st.info(f"Indexing took {end_index_time - start_index_time:.2f} seconds.")

        st.divider()
        st.header("Benchmark")
        st.caption(f"Evaluates retrieval performance using the SciFact 'test' split and top {BENCHMARK_TOP_K} results.")
        if st.button("Run Benchmark"):
            with st.spinner("Running benchmark... This involves embedding all test queries and running multiple search types. Might take time."):
                 start_bench_time = time.perf_counter()
                 # Pass client and models
                 loop.run_until_complete(run_benchmark(client, dense_model, sparse_model, late_model))
                 end_bench_time = time.perf_counter()
                 st.info(f"Benchmarking took {end_bench_time - start_bench_time:.2f} seconds.")

    # --- Main Area for Search ---
    st.header("🔎 Search the Collection")

    query_text = st.text_input("Enter search query:", "What is the impact of COVID-19 on the environment?")
    method = st.selectbox("Select search method:", ["Dense", "Sparse", "Late", "RRF", "Full RRF"], index=4) # Default to Full RRF
    top_k = st.slider("Top K results", 1, 50, 10) # Increased max K

    results_placeholder = st.container()
    status_placeholder = st.empty()

    if st.button("Search", type="primary"):
        if not query_text:
            st.warning("Please enter a search query.")
        else:
            start_time = time.perf_counter()
            resp = None
            error_occured = False
            with st.spinner(f"Searching with {method} method..."):
                try:
                    # Pass client and models to the async query function
                    resp = loop.run_until_complete(do_query(client, query_text, method, top_k, dense_model, sparse_model, late_model))
                except Exception as e:
                    # Catch errors during the async call execution itself (less likely if handled inside do_query)
                    logging.error(f"Error running do_query: {e}", exc_info=True)
                    status_placeholder.error(f"Search execution failed: {e}")
                    error_occured = True

            elapsed = time.perf_counter() - start_time

            # Clear previous results/status *before* showing new ones
            results_placeholder.empty()
            status_placeholder.empty()

            if not error_occured and resp is not None:
                 # Determine the list of points based on response type
                 points_to_display = []
                 if isinstance(resp, list): # Result from client.search(...) -> List[ScoredPoint]
                      points_to_display = resp
                 elif hasattr(resp, 'points') and isinstance(resp.points, list): # Result from client.query_points(...) -> QueryResponse
                      points_to_display = resp.points
                 else:
                      logging.warning(f"Received unexpected response format from search: {type(resp)}")
                      status_placeholder.warning("Received unexpected response format from search.")

                 status_placeholder.success(f"Query completed in **{elapsed:.3f} seconds**. Displaying top {len(points_to_display)} results.")

                 with results_placeholder:
                    if not points_to_display:
                         st.info("No results found.")
                    else:
                        for i, p in enumerate(points_to_display):
                            with st.expander(f"**{i+1}. {p.payload.get('title', 'No Title')}** (Score: {p.score:.4f}, ID: {p.id})", expanded=(i<3)): # Expand first few
                                text = p.payload.get('text', 'No Text')
                                st.markdown(text)
                                # st.markdown("---") # Expander provides separation
            elif not error_occured and resp is None:
                 # This case means do_query returned None likely due to an internal error caught there
                 status_placeholder.warning("Search executed but encountered an issue. Check logs for details.")
            # If error_occurred is True, the error message is already shown by the except block

    # Footer or other info
    st.divider()
    st.caption("Demo using Qdrant, FastEmbed, Streamlit, Datasets, Ranx. Ensure Qdrant server is running.")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()