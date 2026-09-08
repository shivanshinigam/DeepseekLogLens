"""
ingest_logs.py — LogLens AI · RAG Data Ingestion Pipeline
==========================================================
Reads raw log files, splits them into small chunks, generates
vector embeddings, and stores them in a local ChromaDB database.

After running this, use rag_query.py to ask questions about the logs.

Usage:
    python3 ingest_logs.py                        # ingests all logs/ files
    python3 ingest_logs.py --file logs/acmecorp_payment.log
    python3 ingest_logs.py --chunk-size 5         # 5 lines per chunk

Install dependencies first:
    pip3 install chromadb sentence-transformers tqdm
"""

import os
import uuid
import argparse
import glob
from tqdm import tqdm

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Ingest log files into ChromaDB vector store")
parser.add_argument("--file",       type=str, default=None,    help="Single log file to ingest (default: all logs/ files)")
parser.add_argument("--logs-dir",   type=str, default="logs",  help="Directory of log files to ingest")
parser.add_argument("--db-path",    type=str, default="./log_vector_db", help="Path to ChromaDB store")
parser.add_argument("--chunk-size", type=int, default=5,       help="Lines per chunk (default: 5)")
parser.add_argument("--reset",      action="store_true",        help="Delete existing DB and start fresh")
args = parser.parse_args()


def setup_vector_db(db_path: str, reset: bool = False):
    """Initialize ChromaDB with the all-MiniLM-L6-v2 embedding model."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("❌ Missing dependency. Run: pip3 install chromadb sentence-transformers tqdm")
        raise

    print("🧮 Loading embedding model (all-MiniLM-L6-v2)…")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    if reset and os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
        print(f"🗑  Deleted existing database at {db_path}")

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="system_logs",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def chunk_log_file(file_path: str, chunk_size: int = 5):
    """
    Split a log file into overlapping chunks.
    Each chunk = chunk_size lines. Adjacent chunks share 1 line (overlap)
    so context is not lost at chunk boundaries.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]

    chunks = []
    step = max(1, chunk_size - 1)          # 1-line overlap between chunks
    for start in range(0, len(lines), step):
        chunk_lines = lines[start : start + chunk_size]
        chunks.append("\n".join(chunk_lines))

    return chunks


def ingest_file(collection, file_path: str, chunk_size: int = 5, batch_size: int = 32):
    """Chunk one log file and upsert into the vector DB."""
    print(f"\n📖 Reading: {file_path}")
    chunks = chunk_log_file(file_path, chunk_size)
    filename = os.path.basename(file_path)
    print(f"   → {len(chunks)} chunks of {chunk_size} lines each")

    for i in tqdm(range(0, len(chunks), batch_size), desc=f"   Embedding {filename}"):
        batch_docs  = chunks[i : i + batch_size]
        batch_ids   = [str(uuid.uuid4()) for _ in batch_docs]
        batch_meta  = [{"source": filename, "chunk_index": i + j} for j, _ in enumerate(batch_docs)]

        collection.add(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_meta
        )


def main():
    collection = setup_vector_db(args.db_path, reset=args.reset)

    # Find files to ingest
    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(args.logs_dir, "*.log")))
        if not files:
            print(f"⚠️  No .log files found in {args.logs_dir}/")
            print("   Generate them first: python3 generate_logs.py && python3 generate_3_log_files.py")
            return

    print(f"\n📂 Found {len(files)} log file(s) to ingest:")
    for f in files:
        print(f"   • {f}")

    before = collection.count()

    for file_path in files:
        ingest_file(collection, file_path, chunk_size=args.chunk_size)

    after = collection.count()

    print(f"""
════════════════════════════════════════════════════
  ✅ Ingestion complete!
  Files ingested  : {len(files)}
  Chunks added    : {after - before}
  Total in DB     : {after}
  DB path         : {args.db_path}

  Next step → python3 rag_query.py
════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
