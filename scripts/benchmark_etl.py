"""
benchmark_etl.py — compare the FULL ETL pipeline for two chunking strategies.

For the same document, runs:
    chunk -> embed (BGE-M3 dense+sparse) -> upsert to Qdrant
once with RECURSIVE chunks and once with SEMANTIC+HIERARCHICAL chunks.
Times each stage and prints a comparison table.

Safety: writes into a SEPARATE Qdrant collection (`<prod>_benchmark`) so the
real `documents` collection is never touched. The bench collection is deleted
at the end unless --keep is passed.

Usage (from project root, venv active, Docker up with Postgres+MinIO+Qdrant):
    python scripts/benchmark_etl.py                       # latest ready doc, full text
    python scripts/benchmark_etl.py <document_id>         # specific document
    python scripts/benchmark_etl.py --chars 50000         # quick test on first 50k chars
    python scripts/benchmark_etl.py --keep                # don't delete bench collection
"""
import os
import sys
import time
import uuid
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.config import settings
from backend.db.postgre import SessionLocal, Document
from backend.db.minio_client import download_file
from backend.rag.ingestion import chunk, semantic_hierarchical_chunk, _get_embeddings

# Importing embedder triggers eager load of BGE-M3 (~2GB). Announce it.
print("Loading BGE-M3 model (~2GB, slow on first run)...", flush=True)
_t0 = time.time()
from backend.rag.embedder import embed_chunks
print(f"  BGE-M3 loaded in {time.time() - _t0:.1f}s\n", flush=True)

from backend.db.qdrant_client import get_qdrant_client
from qdrant_client.models import (
    PointStruct, SparseVector, VectorParams, Distance, SparseVectorParams,
)


BENCH_COLLECTION = f"{settings.qdrant_collection}_benchmark"


# ---------- args + loading ----------

def parse_args(argv):
    document_id = None
    max_chars = None
    keep = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--chars" and i + 1 < len(argv):
            max_chars = int(argv[i + 1]); i += 2
        elif a == "--keep":
            keep = True; i += 1
        else:
            document_id = a; i += 1
    return document_id, max_chars, keep


def load_text(document_id):
    db = SessionLocal()
    try:
        if document_id:
            doc = db.get(Document, document_id)
            if doc is None:
                sys.exit(f"No document found with id {document_id}")
        else:
            doc = db.query(Document).order_by(Document.created_at.desc()).first()
            if doc is None:
                sys.exit("No documents in the database. Upload one first.")
        if doc.status.value != "ready":
            sys.exit(f"Document status is '{doc.status.value}', not 'ready'.")
        tmp = tempfile.mktemp(suffix=".md")
        download_file(tmp, doc.minio_key + ".md", settings.minio_bucket_markdown)
        return doc.filename, open(tmp, encoding="utf-8").read()
    finally:
        db.close()


# ---------- bench-collection helpers ----------

def setup_bench_collection():
    """(Re)create a clean benchmark collection separate from production."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if BENCH_COLLECTION in existing:
        client.delete_collection(BENCH_COLLECTION)
    client.create_collection(
        collection_name=BENCH_COLLECTION,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    return client


def cleanup_bench_collection():
    try:
        get_qdrant_client().delete_collection(BENCH_COLLECTION)
    except Exception:
        pass


def upsert_to_bench(client, chunks):
    points = [
        PointStruct(
            id=c["chunk_id"],
            vector={
                "dense": c["dense_vector"],
                "sparse": SparseVector(
                    indices=c["sparse_vector"]["indices"],
                    values=c["sparse_vector"]["values"],
                ),
            },
            payload={
                "text": c["text"],
                "document_id": c["document_id"],
                "course_id": c["course_id"],
                "chunk_index": c["chunk_index"],
                "section": c.get("section", ""),
                "covers_concepts": c.get("covers_concepts", []),
            },
        )
        for c in chunks
    ]
    client.upsert(collection_name=BENCH_COLLECTION, points=points)


# ---------- one run of a strategy ----------

def run_strategy(name, chunker, text, doc_id, client):
    print(f"\n--- {name} ---", flush=True)
    times = {}

    t = time.perf_counter()
    chunks = chunker(text, doc_id, "bench")
    times["chunk"] = time.perf_counter() - t
    print(f"  chunk  : {len(chunks)} chunks in {times['chunk']:.2f}s")

    t = time.perf_counter()
    chunks = embed_chunks(chunks)
    times["embed"] = time.perf_counter() - t
    print(f"  embed  : {times['embed']:.2f}s")

    t = time.perf_counter()
    upsert_to_bench(client, chunks)
    times["upsert"] = time.perf_counter() - t
    print(f"  upsert : {times['upsert']:.2f}s")

    times["total"] = times["chunk"] + times["embed"] + times["upsert"]
    sizes = [len(c["text"]) for c in chunks]
    return {
        "name": name,
        "chunks": len(chunks),
        "avg_chars": sum(sizes) / len(sizes) if sizes else 0,
        "times": times,
    }


# ---------- pretty print comparison ----------

def print_table(rec, sem):
    WIDTH = 72
    print(f"\n{'=' * WIDTH}")
    print("  FULL ETL BENCHMARK — Recursive vs Semantic+Hierarchical")
    print("=" * WIDTH)
    print(f"\n{'Stage':<12}{'Recursive':>15}{'Sem+Hier':>15}{'Ratio (S/R)':>15}")
    print("-" * WIDTH)
    for stage in ("chunk", "embed", "upsert"):
        r = rec["times"][stage]
        s = sem["times"][stage]
        ratio = (s / r) if r > 0 else float("inf")
        print(f"{stage:<12}{r:>13.2f}s {s:>13.2f}s {ratio:>13.2f}x")
    print("-" * WIDTH)
    r, s = rec["times"]["total"], sem["times"]["total"]
    ratio = (s / r) if r > 0 else float("inf")
    print(f"{'TOTAL':<12}{r:>13.2f}s {s:>13.2f}s {ratio:>13.2f}x")
    print("-" * WIDTH)
    print(f"{'chunks':<12}{rec['chunks']:>14}  {sem['chunks']:>14}")
    print(f"{'avg chars':<12}{rec['avg_chars']:>14.0f}  {sem['avg_chars']:>14.0f}")
    print("=" * WIDTH)


# ---------- main ----------

def main():
    document_id, max_chars, keep = parse_args(sys.argv[1:])
    label, text = load_text(document_id)
    if max_chars:
        text = text[:max_chars]

    print(f"Document          : {label}")
    print(f"Input             : {len(text):,} chars"
          + (f"  (limited to --chars {max_chars})" if max_chars else "  (full)"))
    print(f"Bench collection  : {BENCH_COLLECTION}"
          + ("  (kept after run)" if keep else "  (deleted after run)"))

    # warm up sentence-transformers (used only by semantic chunker)
    print("\nWarming up sentence-transformers (semantic chunker)...", flush=True)
    t = time.perf_counter()
    _get_embeddings()
    print(f"  loaded in {time.perf_counter() - t:.2f}s")

    client = setup_bench_collection()

    try:
        # unique doc_ids so the two runs don't collide in Qdrant
        rec_doc = f"bench-rec-{uuid.uuid4().hex[:8]}"
        sem_doc = f"bench-sem-{uuid.uuid4().hex[:8]}"

        rec = run_strategy("RECURSIVE", chunk, text, rec_doc, client)
        sem = run_strategy("SEMANTIC+HIERARCHICAL",
                           semantic_hierarchical_chunk, text, sem_doc, client)

        print_table(rec, sem)
    finally:
        if not keep:
            cleanup_bench_collection()
            print(f"\nCleaned up bench collection '{BENCH_COLLECTION}'.")
        else:
            print(f"\n[--keep] Bench collection '{BENCH_COLLECTION}' preserved.")


if __name__ == "__main__":
    main()
