"""
rag_query.py — LogLens AI · RAG Query Pipeline
================================================
Retrieves the most relevant log chunks from ChromaDB,
builds a prompt, and sends it to a hosted vLLM endpoint.

Works in two modes:
  1. LOCAL mode  — uses keyword analysis (no GPU needed, for demos)
  2. RUNPOD mode — calls your live vLLM endpoint on RunPod

Usage:
    # Local demo (no RunPod needed)
    python3 rag_query.py --query "Why did the database fail?"

    # With RunPod endpoint
    python3 rag_query.py --query "Why did the database fail?" \\
        --endpoint https://YOUR_POD_ID-8000.proxy.runpod.net/v1 \\
        --model ShivanshiNigam/loglens-mini-deepseek

Install dependencies:
    pip3 install chromadb sentence-transformers openai
"""

import os
import argparse

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Query log database using RAG + LLM")
parser.add_argument("--query",    type=str, required=True,  help="Your question about the logs")
parser.add_argument("--db-path",  type=str, default="./log_vector_db", help="Path to ChromaDB store")
parser.add_argument("--top-k",    type=int, default=5,      help="Number of log chunks to retrieve")
parser.add_argument("--endpoint", type=str, default=None,   help="vLLM RunPod endpoint URL")
parser.add_argument("--model",    type=str, default="ShivanshiNigam/loglens-mini-deepseek", help="Model name")
args = parser.parse_args()


# ── Step 1: Retrieve relevant logs from ChromaDB ──────────────────────────────
def retrieve_logs(query: str, db_path: str, top_k: int) -> list[str]:
    """Search ChromaDB for log chunks most relevant to the query."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("❌ Run: pip3 install chromadb sentence-transformers")
        raise

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client     = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="system_logs",
        embedding_function=embedding_fn
    )

    if collection.count() == 0:
        print("⚠️  Vector DB is empty. Run ingest_logs.py first.")
        return []

    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
    chunks  = results["documents"][0]    # list of top-k matching text chunks
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources


# ── Step 2: Build the RAG prompt ──────────────────────────────────────────────
def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    """
    Combine the retrieved log context + user question into a clean prompt.
    Small models (like ours) perform best with structured, short prompts.
    """
    context = "\n---\n".join(retrieved_chunks)
    return f"""You are a log analysis expert.
Analyze the following log snippets and answer the question.
Be specific — cite the exact error, timestamp, or service name from the logs.

=== Relevant Log Snippets ===
{context}

=== Question ===
{query}

=== Analysis ==="""


# ── Step 3A: Query vLLM endpoint (RunPod) ────────────────────────────────────
def query_vllm(prompt: str, endpoint: str, model: str) -> str:
    """Call the vLLM OpenAI-compatible API on RunPod."""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ Run: pip3 install openai")
        raise

    client = OpenAI(base_url=endpoint, api_key="loglens-key")
    response = client.chat.completions.create(
        model       = model,
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = 256,
        temperature = 0.2,   # Low temperature = factual, deterministic responses
    )
    return response.choices[0].message.content


# ── Step 3B: Local keyword analysis (fallback, no RunPod) ────────────────────
def local_keyword_analysis(query: str, chunks: list[str], sources: list[str]) -> str:
    """
    Pattern-matching analysis used when no RunPod endpoint is provided.
    Scans retrieved chunks for known error patterns and suggests a fix.
    """
    combined = "\n".join(chunks).upper()

    # Root cause detection
    if "MISSING INDEX" in combined or "FULL_TABLE_SCAN" in combined:
        rc   = "Missing database index causing full table scan and connection pool exhaustion."
        fix  = "CREATE INDEX on the slow query's WHERE clause column."
    elif "OOMKILLED" in combined or "OUT OF MEMORY" in combined:
        rc   = "Memory leak — service RSS growing unbounded, hitting container limit."
        fix  = "Set memory limit in Kubernetes. Fix buffer/chart objects not freed after export."
    elif "DEADLOCK" in combined:
        rc   = "Deadlock — two transactions acquiring locks in inconsistent order."
        fix  = "Standardise lock order: always lock inventory → then orders."
    elif "CIRCUIT BREAKER" in combined or "EXHAUSTED" in combined:
        rc   = "Upstream dependency overwhelmed, circuit breaker tripped."
        fix  = "Isolate analytics to a read replica. Add connection pool monitoring at 70%."
    elif "TIMEOUT" in combined or "REFUSED" in combined:
        rc   = "Service timeout — downstream dependency not responding."
        fix  = "Check downstream health. Increase timeout threshold or add retry logic."
    elif "SSL" in combined or "CERTIFICATE" in combined:
        rc   = "SSL certificate handshake failure."
        fix  = "Renew or verify the SSL certificate. Check client IP allowlist."
    else:
        rc   = "Anomaly detected in the retrieved log chunks."
        fix  = "Review the highlighted log lines for unusual patterns."

    return f"""Root Cause:
  {rc}

Recommended Fix:
  {fix}

Retrieved from: {', '.join(set(sources))}
(Local analysis mode — connect RunPod for full LLM response)"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n🔍 Query: \"{args.query}\"")
    print(f"📂 Searching vector DB at: {args.db_path}")

    if not os.path.exists(args.db_path):
        print("❌ Vector DB not found. Run first: python3 ingest_logs.py")
        return

    # Retrieve
    chunks, sources = retrieve_logs(args.query, args.db_path, args.top_k)
    if not chunks:
        return

    print(f"\n📋 Retrieved {len(chunks)} relevant log chunks:")
    for i, (chunk, src) in enumerate(zip(chunks, sources), 1):
        first_line = chunk.split('\n')[0][:80]
        print(f"   {i}. [{src}] {first_line}…")

    # Build prompt
    prompt = build_prompt(args.query, chunks)

    # Generate answer
    print("\n🤖 Generating analysis…\n")
    print("─" * 60)

    if args.endpoint:
        # Real LLM via RunPod vLLM
        answer = query_vllm(prompt, args.endpoint, args.model)
    else:
        # Local keyword analysis
        print("ℹ️  No --endpoint provided. Using local keyword analysis.")
        print("   For full LLM: add --endpoint https://YOUR_POD-8000.proxy.runpod.net/v1\n")
        answer = local_keyword_analysis(args.query, chunks, sources)

    print(answer)
    print("─" * 60)
    print(f"\n💡 Source logs: {', '.join(set(sources))}")


if __name__ == "__main__":
    main()
