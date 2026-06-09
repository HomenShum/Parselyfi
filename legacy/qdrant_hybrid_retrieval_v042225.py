import streamlit as st
import asyncio
import nest_asyncio
from datasets import load_dataset
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client import AsyncQdrantClient, models
from ranx import Qrels, Run, evaluate, compare
from collections import defaultdict

# Allow nested event loop in Streamlit
nest_asyncio.apply()
loop = asyncio.get_event_loop()

# Configuration
QDRANT_URL = st.secrets["qdrant_url"]
QDRANT_API_KEY = st.secrets["qdrant_api_key"]
COLLECTION_NAME = "scifact"

# Initialize Async Qdrant client
def get_client():
    return AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=600
    )

client = get_client()

# Embedding models
dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding("Qdrant/bm25")
late_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

# Async helper wrappers
async def recreate_collection():
    await client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "all-MiniLM-L6-v2": models.VectorParams(size=384, distance=models.Distance.COSINE),
            "colbertv2.0": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
    )

async def upload_batch(batch):
    texts = batch["text"]
    dense_embs = list(dense_model.passage_embed(texts))
    sparse_embs = list(sparse_model.passage_embed(texts))
    late_embs = list(late_model.passage_embed(texts))
    points = []
    for i, doc_id in enumerate(batch["_id"]):
        points.append(
            models.PointStruct(
                id=int(doc_id),
                vector={
                    "all-MiniLM-L6-v2": dense_embs[i].tolist(),
                    "bm25": sparse_embs[i].as_object(),
                    "colbertv2.0": late_embs[i].tolist()
                },
                payload={"title": batch["title"][i], "text": batch["text"][i]}
            )
        )
    # Use upsert for Async client
    await client.upsert(collection_name=COLLECTION_NAME, points=points)

async def run_indexing():
    dataset = load_dataset("BeIR/scifact", "corpus", split="corpus")
    total = len(dataset)
    batch_size = 32
    for start in range(0, total, batch_size):
        batch = dataset[start:start+batch_size]
        await upload_batch(batch)

# Streamlit UI
st.title("Hybrid Qdrant Async Search Demo")

# Sidebar controls
st.sidebar.header("Indexing & Benchmarking")
if st.sidebar.button("(Re)Create Collection"):
    loop.run_until_complete(recreate_collection())
    st.sidebar.success("Collection recreated")

if st.sidebar.button("Index Corpus to Qdrant"):
    loop.run_until_complete(run_indexing())
    st.sidebar.success("Indexing complete")

if st.sidebar.button("Run Benchmark"):
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels_raw = load_dataset("BeIR/scifact-qrels", split="train")
    qrels_dict = defaultdict(dict)
    for entry in qrels_raw:
        qid = str(entry["query-id"])
        did = str(entry["corpus-id"])
        qrels_dict[qid][did] = entry["score"]
    qrels = Qrels(qrels_dict, name="scifact")
    # Precompute query embeddings
    dense_qv = [next(dense_model.query_embed(q["text"])) for q in queries]
    sparse_qv = [next(sparse_model.query_embed(q["text"])) for q in queries]
    late_qv = [next(late_model.query_embed(q["text"])) for q in queries]
    # Evaluate pipelines
    runs = []
    async def evaluate_run():
        dense_run = Run({str(q["_id"]): {str(p.id): p.score for p in (await client.query_points(COLLECTION_NAME, query=vec, using="all-MiniLM-L6-v2", limit=10)).points} for q, vec in zip(queries, dense_qv)}, name="dense")
        runs.append(dense_run)
        # Additional runs (sparse, late, RRF, full RRF) should be implemented similarly
    loop.run_until_complete(evaluate_run())
    df = compare(qrels=qrels, runs=runs, metrics=["precision@10", "recall@10", "mrr@10", "ndcg@10"])
    st.sidebar.write(df)

# Search interface
st.header("Search the Collection")
query_text = st.text_input("Enter search query:", "What is the impact of COVID-19 on the environment?")
method = st.selectbox("Select search method:", ["Dense", "Sparse", "Late", "RRF", "Full RRF"] )
top_k = st.slider("Top K results", 1, 20, 10)
if st.button("Search"):
    async def do_query():
        if method == "Dense":
            qvec = next(dense_model.query_embed(query_text))
            resp = await client.query_points(COLLECTION_NAME, query=qvec, using="all-MiniLM-L6-v2", limit=top_k, with_payload=True)
        # Add other methods accordingly...
        return resp
    resp = loop.run_until_complete(do_query())
    st.write(f"Showing top {top_k} results using {method}:")
    for point in resp.points:
        st.markdown(f"**ID**: {point.id} | **Score**: {point.score:.4f}")
        title = point.payload.get("title", "")[:100]
        text = point.payload.get("text", "")[:300]
        st.markdown(f"*{title}*")
        st.write(text + "...")
        st.markdown("---")
