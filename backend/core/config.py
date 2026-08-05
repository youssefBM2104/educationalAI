import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent  # → backend/


class Settings(BaseSettings):
    # PostgreSQL
    postgres_user: str = "edu_user"
    postgres_password: str = "changeme"
    postgres_db: str = "edu_db"
    postgres_url: str = "postgresql://edu_user:changeme@localhost:5432/edu_db"
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_root_user: str = ""
    minio_root_password: str = ""
    minio_bucket_originals: str = "originals"
    minio_bucket_markdown: str = "markdown"
    minio_bucket_images: str = "images"
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "edu_collection"
    # Neo4j
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    # NVIDIA NIM
    nim_api_key: str = ""
    # OpenAI — GPT-5 family (per-agent model plan wired in core/models.py)
    openai_api_key: str = ""
    # Google Gemini — held-out judge for offline evaluation (different provider than the
    # pipeline, so the judge never grades output from its own model family)
    google_api_key: str = ""
    # LangSmith — tracing for the LangGraph pipelines. Read from .env here, then exported to
    # os.environ below so the LangChain callback layer (which reads the raw env, not this
    # settings object) picks them up. Works with any provider, NVIDIA NIM included.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "educational-ai"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    # Ollama / VLM
    ollama_host: str = "http://localhost:11434"
    vlm_model: str = "llava:7b"
    vlm_enrichment_enabled: bool = True
    # GPU device selection
    embedding_device: str = "cpu"
    docling_device: str = "cpu"
    # --- Base model (downloaded from HuggingFace the first time) -----------------
    small_ai_base_model: str   = "Qwen/Qwen3-1.7B"
    
    # --- Where the fine-tuned model is saved after training/train.py -------------

    small_ai_output_dir: str = str(_BASE_DIR / "trained_model" / "edu-qwen-v1")
    # --- Existing Q&A dataset (already have this, just point to it) ---------
    small_ai_dataset_path: str = str(_BASE_DIR / "data" / "qa_dataset.json")

    small_ai_lora_r:        int   = 16
    small_ai_lora_alpha:    int   = 32
    small_ai_lora_dropout:  float = 0.05
    small_ai_epochs:        int   = 3
    small_ai_batch_size:    int   = 2
    small_ai_learning_rate: float = 2e-4
    small_ai_max_seq_length: int  = 512
    small_ai_min_interactions: int = 10
    small_ai_user_data_dir: str = str(_BASE_DIR / "user_data")


    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# Export LangSmith config to the real environment so LangChain's callback tracer sees it.
# setdefault -> a value already set in the shell wins over .env, and turning it off is just
# LANGSMITH_TRACING=false in .env.
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)