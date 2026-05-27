"""
show_semantic_chunks.py — pretty-print semantic + hierarchical chunks.

Runs semantic_hierarchical_chunk() on a document's markdown and displays the
chunks in a colorful, boxed, easy-to-read format.

Two ways to provide the text:
  - from MinIO (needs Docker up):   python scripts/show_semantic_chunks.py [document_id]
  - from a local markdown file:     python scripts/show_semantic_chunks.py --file path/to.md

Options:
  --chars N   limit input characters (semantic chunking is slow; good for testing)
  --max N     only display the first N chunks (default 20; use 0 for all)
  --out PATH  write the JSON output to this path
              (default: <document_id_or_filename>_semantic_chunks.json)
  --no-json   skip writing the JSON file (display only)

By default the script ALSO writes a JSON file containing all chunks + metadata
(filename, document id, strategy, embedding model, stats, generated_at).

Examples:
  python scripts/show_semantic_chunks.py --chars 30000 --max 10
  python scripts/show_semantic_chunks.py --file mydoc.md
  python scripts/show_semantic_chunks.py --out my_chunks.json
  python scripts/show_semantic_chunks.py --no-json --max 5
"""
import json
import os
import sys
import shutil
import tempfile
import textwrap
from datetime import datetime

# Force UTF-8 output so box-drawing chars / emoji don't crash a cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

from backend.rag.ingestion import semantic_hierarchical_chunk

WIDTH = min(shutil.get_terminal_size((90, 25)).columns, 90)

# palette
C_BORDER = Fore.BLUE + Style.DIM
C_IDX = Fore.YELLOW + Style.BRIGHT
C_SIZE = Fore.GREEN
C_SECT = Fore.CYAN + Style.BRIGHT
C_TITLE = Fore.MAGENTA + Style.BRIGHT
C_TEXT = Fore.WHITE
R = Style.RESET_ALL


def parse_args(argv):
    file_path = None
    document_id = None
    max_chars = None
    max_chunks = 20
    out_path = None
    no_json = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--file" and i + 1 < len(argv):
            file_path = argv[i + 1]; i += 2
        elif a == "--chars" and i + 1 < len(argv):
            max_chars = int(argv[i + 1]); i += 2
        elif a == "--max" and i + 1 < len(argv):
            max_chunks = int(argv[i + 1]); i += 2
        elif a == "--out" and i + 1 < len(argv):
            out_path = argv[i + 1]; i += 2
        elif a == "--no-json":
            no_json = True; i += 1
        else:
            document_id = a; i += 1
    return file_path, document_id, max_chars, max_chunks, out_path, no_json


def load_text(file_path, document_id):
    """Return (label, text, doc_id, source). doc_id is None for local files."""
    if file_path:
        with open(file_path, encoding="utf-8") as f:
            return os.path.basename(file_path), f.read(), None, "file"
    # MinIO path (needs Docker)
    from backend.core.config import settings
    from backend.db.postgre import SessionLocal, Document
    from backend.db.minio_client import download_file
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
        return doc.filename, open(tmp, encoding="utf-8").read(), doc.id, "minio"
    finally:
        db.close()


def banner(label, total, shown):
    line = "═" * (WIDTH - 2)
    print(f"{C_BORDER}╔{line}╗{R}")
    title = f"  ✂  SEMANTIC + HIERARCHICAL CHUNKS  ·  {label}"
    print(f"{C_BORDER}║{R}{C_TITLE}{title.ljust(WIDTH - 2)}{R}{C_BORDER}║{R}")
    sub = f"  {total} chunks total · showing {shown}"
    print(f"{C_BORDER}║{R}{C_SIZE}{sub.ljust(WIDTH - 2)}{R}{C_BORDER}║{R}")
    print(f"{C_BORDER}╚{line}╝{R}\n")


def size_bar(n, max_n, width=24):
    filled = int(round((n / max_n) * width)) if max_n else 0
    return "█" * filled + "░" * (width - filled)


def show_chunk(c, max_size):
    idx = f" #{c['chunk_index']:04d} "
    size = f" {len(c['text'])} chars "
    section = c.get("section") or "(no header)"

    # top border with index (left) and size (right)
    dash = WIDTH - 2 - len(idx) - len(size)
    print(f"{C_BORDER}╭{R}{C_IDX}{idx}{R}{C_BORDER}{'─' * dash}{R}{C_SIZE}{size}{R}{C_BORDER}╮{R}")

    # section breadcrumb + size bar
    bar = size_bar(len(c["text"]), max_size)
    print(f"  {C_SECT}▸ {section}{R}")
    print(f"  {C_BORDER}{bar}{R}")
    print(f"{C_BORDER}{'─' * WIDTH}{R}")

    # wrapped text body
    body = " ".join(c["text"].split())          # collapse whitespace for tidy display
    for line in textwrap.wrap(body, WIDTH - 4)[:8]:   # cap at 8 lines per chunk preview
        print(f"  {C_TEXT}{line}{R}")
    if len(body) > WIDTH * 8:
        print(f"  {Style.DIM}…(truncated){R}")
    print()


def main():
    file_path, document_id, max_chars, max_chunks, out_path, no_json = parse_args(sys.argv[1:])
    label, text, doc_id, source = load_text(file_path, document_id)
    if max_chars:
        text = text[:max_chars]

    print(f"\n{Style.DIM}Chunking {len(text):,} chars (semantic step embeds every sentence — please wait)…{R}")
    chunks = semantic_hierarchical_chunk(text, doc_id or "demo", "demo")
    if not chunks:
        sys.exit("No chunks produced.")

    shown = chunks if max_chunks == 0 else chunks[:max_chunks]
    max_size = max(len(c["text"]) for c in chunks)

    banner(label, len(chunks), len(shown))
    for c in shown:
        show_chunk(c, max_size)

    # footer summary
    sizes = [len(c["text"]) for c in chunks]
    avg = sum(sizes) / len(sizes)
    print(f"{C_BORDER}{'─' * WIDTH}{R}")
    print(f"{C_TITLE}Summary:{R} {len(chunks)} chunks · "
          f"avg {avg:.0f} chars · min {min(sizes)} · max {max(sizes)}")
    if max_chunks and len(chunks) > max_chunks:
        print(f"{Style.DIM}(showing first {max_chunks}; use --max 0 to show all){R}")

    # ---- write JSON with metadata ----
    if no_json:
        return
    if not out_path:
        basename = doc_id or os.path.splitext(label)[0]
        out_path = f"{basename}_semantic_chunks.json"

    payload = {
        "metadata": {
            "filename": label,
            "document_id": doc_id,
            "source": source,
            "strategy": "semantic_hierarchical",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "header_levels": ["#", "##", "###"],
            "input_chars": len(text),
            "input_truncated": max_chars is not None,
            "total_chunks": len(chunks),
            "avg_chunk_chars": round(avg, 1),
            "min_chunk_chars": min(sizes),
            "max_chunk_chars": max(sizes),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        # add per-chunk char_count for convenience; keep original fields too
        "chunks": [{**c, "char_count": len(c["text"])} for c in chunks],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"{C_SIZE}✓ Wrote {len(chunks)} chunks + metadata to:{R} {out_path}")


if __name__ == "__main__":
    main()
