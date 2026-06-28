# =============================================================================
# generate_dataset.py
# AI-Powered Educational Support System — Phase 2
# Synthetic QA Dataset Generator
#
# What this does:
#   1. Connects to Google Drive and recursively walks all subfolders
#   2. Downloads every supported document (PDF, PPTX, DOCX, XLSX)
#   3. Converts to Markdown using MarkItDown (Microsoft) — preserves
#      headings, tables, and structure that raw text extraction loses
#   4. Splits into chunks using LangChain RecursiveCharacterTextSplitter
#   5. Calls NVIDIA NIM (Llama 3.1 8B) with ASYNC parallelism —
#      up to MAX_CONCURRENT requests in flight at once
#   6. Saves everything to qa_dataset.csv and qa_dataset.json
#      with resume support (skips already-processed chunks)
#
# Install:
#   pip install httpx tqdm "markitdown[pdf,pptx,docx,xlsx]" langchain-text-splitters \
#               pandas google-api-python-client google-auth python-dotenv
#
# LLM Provider (default: NVIDIA NIM — change PROVIDER to switch):
#   nim        → build.nvidia.com   — free 1000 credits/day, accepts Gmail ← DEFAULT
#   openrouter → openrouter.ai      — free tier, no credit card
#   groq       → console.groq.com   — requires org/uni email
#   together   → api.together.xyz   — free $25 credit
#
# Setup:
#   1. Get your NVIDIA NIM API key:
#      - Go to build.nvidia.com → sign in → Get API Key → copy it
#      - Add to .env: NIM_API_KEY=nvapi-xxxxxxxxxxxx
#   2. Google Drive service account:
#      - console.cloud.google.com → enable Drive API
#      - Create service account → download JSON → save as service_account.json
#      - Share the Drive folder with the service account email
# =============================================================================

import os
import io
import json
import asyncio
import tempfile
import time

import re
import pandas as pd
import httpx
from dotenv import load_dotenv
from markitdown import MarkItDown
from tqdm import tqdm

# Load .env from project root
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(_env_path)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

# =============================================================================
# CONFIGURATION
# =============================================================================

# Google Drive root folder ID
ROOT_FOLDER_ID = "18HpNDB-WEXv88ZVr4iXlSZJm0oI4H4sv"

# Path to your Google service account JSON key
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "..", "service_account.json")

# Output directory
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_CSV  = os.path.join(DATA_DIR, "qa_dataset.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "qa_dataset.json")

# Chunking
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# ---------------------------------------------------------------------------
# LLM Provider — set DATASET_LLM_PROVIDER env var to switch
# Options: "nim" | "openrouter" | "groq" | "together" | "ollama"
# ---------------------------------------------------------------------------
PROVIDER = os.getenv('DATASET_LLM_PROVIDER', 'nim')

MODELS = {
    "nim":        "meta/llama-3.1-8b-instruct",
    "openrouter": "meta-llama/llama-3.1-8b-instruct:free",
    "groq":       "llama-3.1-8b-instant",
    "together":   "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "ollama":     os.getenv("DATASET_LLM_MODEL", "qwen3:14b"),
}

BASE_URLS = {
    "nim":        "https://integrate.api.nvidia.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq":       "https://api.groq.com/openai/v1",
    "together":   "https://api.together.xyz/v1",
    "ollama":     os.getenv("DATASET_LLM_BASE_URL", "http://localhost:11434/v1"),
}

API_KEY_ENV = {
    "nim":        "NIM_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq":       "GROQ_API_KEY",
    "together":   "TOGETHER_API_KEY",
    "ollama":     None,
}

QA_PER_CHUNK = 1          # QA pairs per chunk — 1 is efficient for large docs

# ---------------------------------------------------------------------------
# Async concurrency settings
# MAX_CONCURRENT = how many API requests run in parallel
# NIM free tier handles ~50 concurrent requests comfortably
# Lower this if you see 429 rate-limit errors
# ---------------------------------------------------------------------------
MAX_CONCURRENT = 1

# Resume support: skip chunks already in output CSV
RESUME = True

# Subject filter — limit to specific folders for testing
# Set to [] to process all subjects
SUBJECT_FILTER = ["Business"]  # ← change to [] for full run

# Supported MIME types -> file extensions
SUPPORTED_MIME_TYPES = {
    "application/pdf":
        ".pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        ".pptx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        ".xlsx",
    "application/vnd.ms-powerpoint":
        ".ppt",
    "application/msword":
        ".doc",
}

# =============================================================================


# =============================================================================
# GOOGLE DRIVE
# =============================================================================

def get_drive_service():
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(
            f"'{SERVICE_ACCOUNT_FILE}' not found.\n"
            "Download it from Google Cloud Console → IAM → Service Accounts → Keys.\n"
            "Share the Drive folder with the service account email."
        )
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)


def list_all_documents(service, folder_id: str, path: str = "") -> list[dict]:
    results = []
    page_token = None
    while True:
        kwargs = dict(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageSize=100,
        )
        if page_token:
            kwargs["pageToken"] = page_token
        response = service.files().list(**kwargs).execute()
        for item in response.get("files", []):
            mime = item["mimeType"]
            if mime == "application/vnd.google-apps.folder":
                sub_path = f"{path}/{item['name']}" if path else item["name"]
                print(f"  [DIR]  {sub_path}/")
                results.extend(list_all_documents(service, item["id"], sub_path))
            elif mime in SUPPORTED_MIME_TYPES:
                results.append({
                    "id":          item["id"],
                    "name":        item["name"],
                    "mime_type":   mime,
                    "extension":   SUPPORTED_MIME_TYPES[mime],
                    "folder_path": path if path else "root",
                })
                print(f"  [FILE] {item['name']}")
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return results


def download_file_bytes(service, file_id: str) -> bytes:
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


# =============================================================================
# DOCUMENT CONVERSION
# =============================================================================

_markitdown = MarkItDown()


def convert_to_markdown(file_bytes: bytes, extension: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        if extension.lower() == ".pdf":
            return _parse_pdf_docling(tmp_path)
        else:
            result = _markitdown.convert(tmp_path)
            return result.text_content or ""
    finally:
        os.unlink(tmp_path)

_docling_converter: DocumentConverter | None = None

def _get_docling_converter() -> DocumentConverter:
    global _docling_converter
    if _docling_converter is None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_formula_enrichment = True
        device_str = os.getenv('DOCLING_DEVICE', 'cpu')
        device = AcceleratorDevice.CUDA if device_str == 'cuda' else AcceleratorDevice.CPU
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=device,
        )
        _docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    return _docling_converter

def _parse_pdf_docling(file_path: str) -> str:
    converter = _get_docling_converter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()
# =============================================================================
# CHUNKING
# =============================================================================

def chunk_markdown(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    docs = splitter.create_documents([text])
    return [
        doc.page_content
        for doc in docs
        if len(doc.page_content.strip()) > 150
    ]


# =============================================================================
# ASYNC QA GENERATION
# =============================================================================

SYSTEM_PROMPT = """You are an educational assessment assistant building a question-answer dataset.

Given a passage of text, generate exactly {n} question-answer pairs.

STRICT RULES:
1. Each answer MUST be closely paraphrased from the passage. Do NOT use external knowledge.
2. Return {{"pairs": []}} if the passage is a table of contents, index, bibliography, cover page, or has no educational content.
3. Return {{"pairs": []}} if the passage is an incomplete sentence or starts mid-thought.
4. Questions must be fully answerable from the passage alone.
5. NEVER start questions with "What happens when", "What is", or single-word-answer questions.
   Use: How, Why, Explain, What causes, How does X differ from Y, What are the implications of.
6. Every answer must be at least 2 full sentences. If you cannot write a 2-sentence answer, return {{"pairs": []}}.
7. Never generate questions about page numbers, authors, publishers, URLs, or document structure.
8. Difficulty must reflect genuine cognitive demand — easy = requires reading carefully, medium = requires connecting 2+ ideas, hard = requires applying the concept to a new situation.

Return ONLY valid JSON. No markdown, no backticks, no explanation.

Format:
{{
  "pairs": [
    {{"question": "...", "answer": "...", "difficulty": "easy"}},
    {{"question": "...", "answer": "...", "difficulty": "medium"}}
  ]
}}"""

def _get_api_key() -> str | None:
    key_name = API_KEY_ENV[PROVIDER]
    if key_name is None:
        return None
    api_key = os.environ.get(key_name)
    if not api_key:
        raise EnvironmentError(
            f"{key_name} is not set.\n"
            f"For NVIDIA NIM: go to build.nvidia.com → Get API Key\n"
            f"Then add to .env: {key_name}=your_key_here"
        )
    return api_key


async def generate_qa_async(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    chunk: str,
    chunk_id: str,
    subject: str,
    source_file: str,
) -> list[dict]:
    """
    Fire a single async HTTP request to the LLM API.
    The semaphore limits how many run concurrently (MAX_CONCURRENT).
    Retries once on 429 (rate limit) with a short backoff.
    Returns a list of row dicts ready to save, or [] on failure.
    """
    url = f"{BASE_URLS[PROVIDER]}/chat/completions"
    api_key = _get_api_key()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": MODELS[PROVIDER],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(n=QA_PER_CHUNK)},
            {"role": "user",   "content": f"Passage:\n\n{chunk}"},
        ],
        "temperature": 0.2,
        "max_tokens":  800,
    }

    async with semaphore:
        for attempt in range(6):  # up to 6 attempts per chunk
            try:
                timeout = 120.0 if PROVIDER == "ollama" else 30.0
                response = await client.post(url, headers=headers, json=payload, timeout=timeout)

                if response.status_code == 429:
                    wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s, 32s backoff
                    print(f"  [429 rate limit — waiting {wait}s] {chunk_id}", flush=True)
                    await asyncio.sleep(wait)
                    continue

                if response.status_code != 200:
                    print(f"  [HTTP {response.status_code}] {chunk_id}", flush=True)
                    return []

                data = response.json()
                raw = data["choices"][0]["message"]["content"].strip()

                # Strip <think>...</think> blocks (Qwen3 and other reasoning models)
                raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

                # Strip markdown fences if model adds them
                if raw.startswith("```"):
                    parts = raw.split("```")
                    raw = parts[1] if len(parts) > 1 else raw
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                pairs = json.loads(raw).get("pairs", [])
                rows = []
                for p in pairs:
                    if p.get("question", "").strip() and p.get("answer", "").strip():
                        rows.append({
                            "subject":     subject,
                            "source_file": source_file,
                            "chunk_id":    chunk_id,
                            "context":     chunk,
                            "question":    p["question"],
                            "answer":      p["answer"],
                            "difficulty":  p.get("difficulty", "medium"),
                        })
                return rows

            except json.JSONDecodeError:
                print(f"  [JSON error] {chunk_id}", flush=True)
                return []
            except httpx.TimeoutException:
                print(f"  [Timeout attempt {attempt+1}] {chunk_id}", flush=True)
                await asyncio.sleep(1)
            except Exception as e:
                print(f"  [Error: {e}] {chunk_id}", flush=True)
                return []

    return []


# =============================================================================
# OUTPUT
# =============================================================================

def load_done_chunks() -> set[str]:
    if RESUME and os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        return set(df["chunk_id"].unique())
    return set()


def save_rows(new_rows: list[dict], append: bool = False):
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    new_df = new_df[new_df["question"].str.strip() != ""]
    new_df = new_df[new_df["answer"].str.strip() != ""]
    if append and os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        df = pd.concat([existing, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=["chunk_id", "question"])
    else:
        df = new_df.reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    df.to_json(OUTPUT_JSON, orient="records", indent=2)


# =============================================================================
# ASYNC PROCESSING CORE
# =============================================================================

async def process_document_async(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    doc_info: dict,
    done_chunks: set[str],
    service,
) -> list[dict]:
    """Download, convert, chunk, and generate QA for one document — fully async."""
    subject  = doc_info["folder_path"]
    filename = doc_info["name"]
    ext      = doc_info["extension"]

    print(f"\n{'-' * 65}")
    print(f"[{subject}] {filename}")

    # Download (sync — Drive SDK is not async, but download is fast)
    try:
        file_bytes = download_file_bytes(service, doc_info["id"])
    except Exception as e:
        print(f"  Download failed: {e} — skipping")
        return []

    # Convert to Markdown
    try:
        markdown = convert_to_markdown(file_bytes, ext)
    except Exception as e:
        print(f"  Conversion failed: {e} — skipping")
        return []

    if not markdown.strip():
        print("  No text extracted (possibly scanned image PDF) — skipping")
        return []

    # Chunk
    chunks = chunk_markdown(markdown)
    print(f"  {len(chunks)} chunks — firing {MAX_CONCURRENT} concurrent requests")

    if not chunks:
        return []

    # Build tasks only for chunks not already done
    tasks = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{subject}__{filename}__chunk_{i:04d}"
        if chunk_id in done_chunks:
            continue
        tasks.append(
            generate_qa_async(client, semaphore, chunk, chunk_id, subject, filename)
        )

    if not tasks:
        print("  All chunks already processed — skipping")
        return []

    # Run all tasks concurrently, bounded by semaphore — with live progress bar
    all_rows = []
    with tqdm(total=len(tasks), desc=f"  {filename[:40]}", unit="chunk", ncols=80) as pbar:
        async def tracked(coro):
            result = await coro
            pbar.update(1)
            pbar.set_postfix(pairs=len(all_rows) + len(result))
            return result

        results = await asyncio.gather(*[tracked(t) for t in tasks])

    all_rows = [row for result in results for row in result]
    print(f"  ✓ {len(all_rows)} QA pairs generated from {len(tasks)} chunks")
    return all_rows


# =============================================================================
# MAIN
# =============================================================================

async def run_async():
    print("=" * 65)
    print("  Synthetic QA Dataset Generator  [ASYNC MODE]")
    print(f"  Drive folder    : {ROOT_FOLDER_ID}")
    print(f"  Provider        : {PROVIDER}")
    print(f"  Model           : {MODELS[PROVIDER]}")
    print(f"  Chunk size      : {CHUNK_SIZE} chars / {CHUNK_OVERLAP} overlap")
    print(f"  QA per chunk    : {QA_PER_CHUNK}")
    print(f"  Max concurrent  : {MAX_CONCURRENT} requests")
    print(f"  Converter       : MarkItDown (PDF, PPTX, DOCX, XLSX)")
    print(f"  Resume          : {RESUME}")
    print("=" * 65)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Validate API key early
    _get_api_key()

    # Resume: load already-processed chunk IDs
    done_chunks = load_done_chunks()
    if done_chunks:
        print(f"\nResuming — {len(done_chunks)} chunk(s) already processed\n")

    # Discover documents
    print("\nScanning Google Drive folder tree...\n")
    service = get_drive_service()
    documents = list_all_documents(service, ROOT_FOLDER_ID)

    if SUBJECT_FILTER:
        documents = [d for d in documents if d["folder_path"] in SUBJECT_FILTER]
        print(f"\nSubject filter active: {SUBJECT_FILTER}")

    if not documents:
        print("\nNo supported documents found.")
        print("Check ROOT_FOLDER_ID and that the folder is shared with the service account.")
        return

    print(f"\nFound {len(documents)} document(s)\n")

    # Shared async HTTP client + semaphore for the whole run
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    total_pairs = 0

    async with httpx.AsyncClient() as client:
        for doc_info in documents:
            t0 = time.time()
            new_rows = await process_document_async(
                client, semaphore, doc_info, done_chunks, service
            )
            elapsed = time.time() - t0

            if new_rows:
                save_rows(new_rows, append=bool(done_chunks))
                # Update done_chunks so resume works mid-run
                for row in new_rows:
                    done_chunks.add(row["chunk_id"])
                total_pairs += len(new_rows)
                print(f"  Saved in {elapsed:.1f}s")

    # Final summary
    print(f"\n{'=' * 65}")
    print(f"Complete")
    print(f"  Total new QA pairs : {total_pairs}")

    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        print(f"  Total in dataset   : {len(df)}")
        print(f"\nBreakdown by subject:")
        print(df.groupby("subject")["question"].count().to_string())
        print(f"\nBreakdown by difficulty:")
        print(df["difficulty"].value_counts().to_string())
        print(f"\nOutput files:")
        print(f"  {OUTPUT_CSV}")
        print(f"  {OUTPUT_JSON}")


def run():
    asyncio.run(run_async())


if __name__ == "__main__":
    run()