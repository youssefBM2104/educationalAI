# Session Report — 2026-06-11
## GPU Portability, Ollama Provider, and Code Audit

---

## 1. What Was Done

This session made the ETL pipeline and dataset generator fully portable between a
local CPU machine and a remote GPU server (vast.ai), entirely via environment
variables, with no code changes between environments.

Six categories of changes were applied, followed by two code audits.

---

## 2. GPU Portability Changes

### 2.1 BGE-M3 embedding device — `backend/etl/embedder.py`

**Before:** Used `torch.cuda.is_available()` auto-detection.

**After:** Reads `EMBEDDING_DEVICE` env var (default `cpu`).

```python
device = os.getenv('EMBEDDING_DEVICE', 'cpu')
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=device)
```

`torch` import was removed (was only used for device detection).

---

### 2.2 Docling PDF accelerator — `backend/etl/ingestion.py` and `scripts/generate_dataset.py`

**Before:** No accelerator options set (CPU default, implicit).

**After:** Reads `DOCLING_DEVICE` env var and sets `AcceleratorOptions` explicitly.

```python
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

device_str = os.getenv('DOCLING_DEVICE', 'cpu')
device = AcceleratorDevice.CUDA if device_str == 'cuda' else AcceleratorDevice.CPU
pipeline_options.accelerator_options = AcceleratorOptions(num_threads=4, device=device)
```

**Note on import path:** The correct import for the installed version of docling is
`docling.datamodel.accelerator_options`, NOT `docling.pipeline.options` (which does
not exist in this version).

---

### 2.3 Ollama provider — `scripts/generate_dataset.py`

Added `"ollama"` as a fifth LLM provider option alongside `nim`, `openrouter`,
`groq`, `together`.

`PROVIDER` is now driven by env var (default `nim`):
```python
PROVIDER = os.getenv('DATASET_LLM_PROVIDER', 'nim')
```

New dict entries:
```python
"ollama": os.getenv("DATASET_LLM_MODEL", "qwen3:14b")        # MODELS
"ollama": os.getenv("DATASET_LLM_BASE_URL", "http://localhost:11434/v1")  # BASE_URLS
"ollama": None                                                 # API_KEY_ENV
```

`_get_api_key()` returns `None` for Ollama (no auth header needed).
`generate_qa_async()` only adds `Authorization` header when key is non-None.

---

### 2.4 `docker-compose.gpu.yml` (new file, project root)

Docker Compose override for GPU environments. **Not committed** (added to `.gitignore`).
Applied at runtime on the GPU server only:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Adds NVIDIA GPU reservations to `ollama` and `celery_worker` services.

---

### 2.5 `.env.example` (new file, project root, committed)

Comprehensive template listing every environment variable across all files.
Groups: Infrastructure, LLM Providers, Ollama, GPU/Device, Dataset generation.
Includes the three new variables from this session:
- `EMBEDDING_DEVICE`
- `DOCLING_DEVICE`
- `DATASET_LLM_PROVIDER`
- `DATASET_LLM_MODEL`
- `DATASET_LLM_BASE_URL`

---

### 2.6 `.gitignore` additions

Added:
```
docker-compose.gpu.yml
.env.vast
.env.local
```

---

## 3. Audit — `scripts/generate_dataset.py`

### 3.1 Bugs fixed

**Qwen3 thinking tokens (🔴 critical — was silently dropping all output)**

`qwen3:14b` outputs `<think>…</think>` blocks before the JSON response by default.
The existing fence-stripping code handled backticks but not `<think>` blocks.
`json.loads()` would throw `JSONDecodeError` on every single chunk, silently
dropping all generated QA pairs.

Fix applied: strip `<think>` blocks before fence-stripping:
```python
raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
```

**30-second timeout too short for Ollama (🔴 critical)**

`qwen3:14b` at `max_tokens=800` generates ~32 seconds on a modest GPU. The first
request also loads the model into VRAM. A 30s timeout causes constant timeouts and
wasted retry attempts.

Fix applied: provider-specific timeout:
```python
timeout = 120.0 if PROVIDER == "ollama" else 30.0
```

**Unused `import fitz` removed**

`fitz` (PyMuPDF) was imported but never used. Replaced with `import re` (which was
already needed for the `<think>` stripping fix).

### 3.2 Configuration flags to set before full run

| Variable | Current value | Correct value for full run |
|---|---|---|
| `SUBJECT_FILTER` (line 128) | `["Business"]` | `[]` |
| `MAX_CONCURRENT` (line 121) | `1` | `1` (correct for Ollama — it queues requests serially) |

---

## 4. Audit — `backend/etl/ingestion.py`

### 4.1 Bugs fixed

**Image description mismatch on VLM failure (🔴 bug)**

When `_describe_image` failed both retry attempts, it returned `"<!-- image -->"` —
the same string as the docling placeholder. The `markdown.replace("<!-- image -->", description, 1)`
call became a no-op, leaving the first placeholder in place. The next picture item
then replaced that same slot with its own description, shifting all subsequent image
descriptions by one slot.

Fix: return a distinct string on failure:
```python
return "<!-- image-failed -->"   # was: "<!-- image -->"
```

**Slow silent failure when Ollama unreachable (🟡 performance)**

With `VLM_ENRICHMENT_ENABLED=True` and Ollama down, each image cost 2 attempts × 3s
sleep = 6 seconds. A PDF with 20 figures stalled for 2 minutes per document.

Fix: single 5-second reachability probe before the picture loop. If Ollama is down,
the whole document skips enrichment with one log warning:

```python
def _ollama_reachable() -> bool:
    try:
        httpx.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5.0).raise_for_status()
        return True
    except Exception:
        return False
```

**`MarkItDown()` instantiated on every non-PDF parse call (⚪ minor)**

Was creating a new `MarkItDown()` instance on every call to `parse()`. Fixed to a
module-level singleton `_markitdown = MarkItDown()`, consistent with how
`generate_dataset.py` already handled it.

### 4.2 Remaining known limitations

- `_sanitize_formula_output` regex only catches LaTeX commands with `{...}` args —
  bare repeated commands like `\alpha \alpha \alpha` are not caught.
- `semantic_hierarchical_chunk` docstring mentions "pymupdf4llm" — stale reference
  to the old PDF converter (now replaced by Docling).

---

## 5. vast.ai GPU Setup

### Hardware (rented instance)

| Spec | Value |
|---|---|
| GPU | RTX A4000 |
| VRAM | 16 GB |
| CUDA | 13.0 |
| Disk | 20 GB (consider upgrading to 80–100 GB) |
| Current image | `vastai/vllm_v0.21.0-cuda-13.0/jupyter` |

### VRAM budget

| Component | VRAM |
|---|---|
| `qwen3:14b` Q4_K_M (Ollama) | ~8.5 GB |
| BGE-M3 embedding | ~2.2 GB |
| Docling layout/table/formula models | ~1.5 GB |
| **Total** | **~12.2 GB** — safe on 16 GB |

### Issue: current image has no Ollama

The rented instance runs `vastai/vllm_v0.21.0-cuda-13.0/jupyter`. Two options:

**Option A — Redeploy with Ollama template (recommended)**

Search vast.ai for an Ollama template. Pick 80–100 GB disk. Then:
```bash
ollama pull qwen3:14b          # ~8.5 GB download
```
Set env vars:
```
DATASET_LLM_PROVIDER=ollama
DATASET_LLM_BASE_URL=http://localhost:11434/v1
DATASET_LLM_MODEL=qwen3:14b
EMBEDDING_DEVICE=cuda
DOCLING_DEVICE=cuda
```

**Option B — Use vLLM (current image)**

vLLM exposes an OpenAI-compatible `/v1` endpoint. Use AWQ quantization to fit in 16 GB:
```
DATASET_LLM_PROVIDER=ollama        # reuses the same code path
DATASET_LLM_BASE_URL=http://localhost:8000/v1
DATASET_LLM_MODEL=Qwen/Qwen3-14B-AWQ
```
Note: plain fp16 `qwen3:14b` requires ~28 GB VRAM — does NOT fit on the A4000.

### `qwen3:14b` thinking mode

Qwen3 models output `<think>…</think>` blocks before the JSON response when thinking
is enabled. The `generate_dataset.py` script now strips these blocks before JSON
parsing (fix applied in this session), so no special Ollama flags are needed.

---

## 6. Current Environment Variables (full reference)

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_USER` | `edu_user` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `changeme` | PostgreSQL password |
| `POSTGRES_DB` | `edu_db` | PostgreSQL database name |
| `POSTGRES_URL` | — | Full SQLAlchemy DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO S3 endpoint |
| `MINIO_ROOT_USER` | — | MinIO admin user |
| `MINIO_ROOT_PASSWORD` | — | MinIO admin password |
| `MINIO_BUCKET_ORIGINALS` | `originals` | Raw files bucket |
| `MINIO_BUCKET_MARKDOWN` | `markdown` | Converted MD bucket |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST endpoint |
| `QDRANT_COLLECTION` | `edu_collection` | Qdrant collection name |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | Neo4j password |
| `NIM_API_KEY` | — | NVIDIA NIM API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `GROQ_API_KEY` | — | Groq API key |
| `TOGETHER_API_KEY` | — | Together AI API key |
| `ANTHROPIC_API_KEY` | — | Claude API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama API host |
| `VLM_MODEL` | `moondream` | VLM for image description |
| `VLM_ENRICHMENT_ENABLED` | `true` | Enable/disable VLM enrichment |
| `EMBEDDING_DEVICE` | `cpu` | BGE-M3 device: `cpu` or `cuda` |
| `DOCLING_DEVICE` | `cpu` | Docling accelerator: `cpu` or `cuda` |
| `DATASET_LLM_PROVIDER` | `nim` | Dataset gen provider: `nim`, `openrouter`, `groq`, `together`, `ollama` |
| `DATASET_LLM_MODEL` | `qwen3:14b` | Ollama model (only when provider=ollama) |
| `DATASET_LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama base URL (only when provider=ollama) |

---

## 7. Files Changed in This Session

| File | Change type | Summary |
|---|---|---|
| `backend/etl/embedder.py` | Modified | Replace torch auto-detection with `EMBEDDING_DEVICE` env var |
| `backend/etl/ingestion.py` | Modified | Add Docling CUDA accelerator; fix image mismatch bug; fast-fail Ollama probe; MarkItDown singleton |
| `scripts/generate_dataset.py` | Modified | Add Ollama provider; `PROVIDER` from env; Docling CUDA accelerator; strip `<think>` tokens; fix timeout; remove unused `fitz` |
| `docker-compose.gpu.yml` | Created | GPU overlay for vast.ai — not committed |
| `.env.example` | Created | Full env var template — committed |
| `.gitignore` | Modified | Add `docker-compose.gpu.yml`, `.env.vast`, `.env.local` |



---

## 8. Known Gaps / Next Steps

1. **Run the dataset generator** — full run with `SUBJECT_FILTER=[]` on GPU server.
   Expected output: `data/qa_dataset.csv` and `data/qa_dataset.json`.

2. **Phase 3** — Index the generated chunks into Qdrant (dense + sparse vectors via
   BGE-M3) and Neo4j (KG triples via LLMGraphTransformer + NIM).

3. **Docling `images_scale=2.0`** — monitor VRAM usage during ingestion on GPU;
   reduce to `1.0` if OOM on large PDFs with many figures.

4. **VLM model upgrade** — `moondream` is small and fast but low quality. Switch to
   `llava:7b` or `llava:13b` on GPU once the pipeline is validated end-to-end.

5. **`_sanitize_formula_output` regex** — only catches `\cmd{arg}` repetition, not
   bare command repetition. Extend pattern if hallucinations appear in output.
