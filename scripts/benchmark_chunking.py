"""
benchmark_chunking.py — compare processing time of chunking strategies.

Compares:
  - recursive               : the current RecursiveCharacterTextSplitter
  - semantic + hierarchical : markdown-header split, then semantic split per section

It pulls a document's cleaned markdown from MinIO (the most recent upload, or a
given document_id), runs both strategies on the SAME text, and reports time +
chunk stats.

Usage (from project root, venv active):
    python scripts/benchmark_chunking.py                      # latest doc, full text
    python scripts/benchmark_chunking.py <document_id>        # specific document
    python scripts/benchmark_chunking.py --chars 50000        # limit input (quick test)
    python scripts/benchmark_chunking.py <document_id> --chars 50000

Tip: semantic chunking embeds every sentence on CPU, so it is MUCH slower than
recursive. Start with a small --chars (e.g. 50000) to confirm it works before
running on the full document.
"""
import os
import sys
import time
import tempfile

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.config import settings
from backend.db.postgre import SessionLocal, Document
from backend.db.minio_client import download_file
from backend.rag.ingestion import chunk, semantic_hierarchical_chunk, _get_embeddings


def parse_args(argv):
    document_id = None
    max_chars = None
    i = 0
    while i < len(argv):
        if argv[i] == "--chars" and i + 1 < len(argv):
            max_chars = int(argv[i + 1])
            i += 2
        else:
            document_id = argv[i]
            i += 1
    return document_id, max_chars


def load_markdown(document_id):
    db = SessionLocal()
    try:
        if document_id:
            doc = db.get(Document, document_id)
            if doc is None:
                sys.exit(f"No document found with id {document_id}")
        else:
            doc = db.query(Document).order_by(Document.created_at.desc()).first()
            if doc is None:
                sys.exit("No documents in the database yet. Upload one first.")
        if doc.status.value != "ready":
            sys.exit(f"Document status is '{doc.status.value}', not 'ready'.")
        md_key = doc.minio_key + ".md"
        tmp = tempfile.mktemp(suffix=".md")
        download_file(tmp, md_key, settings.minio_bucket_markdown)
        return doc, open(tmp, encoding="utf-8").read()
    finally:
        db.close()


def stats(chunks):
    n = len(chunks)
    sizes = [len(c["text"]) for c in chunks]
    avg = sum(sizes) / n if n else 0
    return n, avg


def main():
    document_id, max_chars = parse_args(sys.argv[1:])
    doc, text = load_markdown(document_id)
    if max_chars:
        text = text[:max_chars]

    print(f"Document : {doc.filename} ({doc.id})")
    print(f"Input    : {len(text):,} characters"
          + (f"  (limited to --chars {max_chars})" if max_chars else "  (full)"))
    print("=" * 70)

    # Warm up the embedding model so its one-time load isn't counted in the timing.
    t = time.perf_counter()
    _get_embeddings()
    load_time = time.perf_counter() - t
    print(f"Embedding model load (one-time): {load_time:.1f}s\n")

    # --- recursive ---
    t = time.perf_counter()
    rec = chunk(text, "bench", "bench")
    rec_time = time.perf_counter() - t

    # --- semantic + hierarchical ---
    t = time.perf_counter()
    sem = semantic_hierarchical_chunk(text, "bench", "bench")
    sem_time = time.perf_counter() - t

    rec_n, rec_avg = stats(rec)
    sem_n, sem_avg = stats(sem)

    print(f"{'Strategy':<28}{'Time (s)':>12}{'Chunks':>10}{'Avg chars':>12}")
    print("-" * 70)
    print(f"{'recursive':<28}{rec_time:>12.3f}{rec_n:>10}{rec_avg:>12.0f}")
    print(f"{'semantic + hierarchical':<28}{sem_time:>12.3f}{sem_n:>10}{sem_avg:>12.0f}")
    print("-" * 70)
    if rec_time > 0:
        print(f"\nsemantic+hierarchical is {sem_time / rec_time:.1f}x slower than recursive"
              f" (excluding the {load_time:.1f}s model load).")


if __name__ == "__main__":
    main()
