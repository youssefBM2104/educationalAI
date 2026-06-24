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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()