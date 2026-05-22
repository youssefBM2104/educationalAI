from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # PostgreSQL
    postgres_user: str = "edu_user"
    postgres_password: str = "changeme"
    postgres_db: str = "edu_db"
    postgres_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_root_user: str
    minio_root_password: str
    minio_bucket_originals: str = "originals"
    minio_bucket_markdown: str = "markdown"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # Neo4j
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # NVIDIA
    nim_api_key: str
    

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()