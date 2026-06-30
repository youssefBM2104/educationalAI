from pydantic_settings import BaseSettings, SettingsConfigDict

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
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "edu_collection"
    # Neo4j
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    # NVIDIA NIM
    nim_api_key: str = ""
    # Ollama / VLM
    ollama_host: str = "http://ollama:11434"
    vlm_model: str = "llava:7b"
    vlm_enrichment_enabled: bool = False
    # GPU device selection
    embedding_device: str = "cpu"
    docling_device: str = "cpu"
    # --- Base model (downloaded from HuggingFace the first time) -----------------
    small_ai_base_model   = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # --- Where the fine-tuned model is saved after training/train.py -------------
    small_ai_output_dir   = "./trained_model/edu-qwen-v1"
    
    # --- Your existing Q&A dataset (already have this, just point to it) ---------
    small_ai_dataset_path = "../data/qa_dataset.json"
    
    # --- LoRA hyperparameters (tuned for 8GB RAM) ---
    small_ai_lora_r          = 16
    small_ai_lora_alpha      = 32
    small_ai_lora_dropout    = 0.05
    
    # --- Training hyperparameters ------------------------------------------------
    small_ai_epochs          = 3
    small_ai_batch_size      = 2
    small_ai_learning_rate   = 2e-4
    small_ai_max_seq_length  = 512
    
    # --- Continuous learning (runs on user machine) ------------------------------
    small_ai_min_interactions = 10   # minimum interactions before personalizing
    small_ai_user_data_dir    = "./user_data"   # local, never uploaded to server

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()