import asyncio
import tempfile
import os
import logging

from celery import Celery

from backend.core.config import settings
from backend.db.postgre import SessionLocal, Document, DocumentStatus
from backend.db.minio_client import download_file, upload_file
from backend.etl.ingestion import parse_and_semantic_hierarchical_chunk
from backend.db.kg_client import get_kg


from backend.db.qdrant_client import upsert_chunks

logger = logging.getLogger(__name__)

celery_app = Celery(
    "ingestion",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

@celery_app.task(bind=True)
def process_document(self, document_id: str, minio_key: str, course_id: str):
    from backend.etl.embedder import embed_chunks
    logger.info("[%s] Starting ingestion — document_id=%s minio_key=%s course_id=%s",
                self.request.id, document_id, minio_key, course_id)
    db = SessionLocal()
    doc = None
    try:
        doc = db.get(Document, document_id)
        doc.status = DocumentStatus.processing
        db.commit()
        logger.info("[%s] Status → processing", self.request.id)

        suffix = os.path.splitext(minio_key)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        logger.debug("[%s] Downloading from MinIO bucket=%s key=%s → %s",
                     self.request.id, settings.minio_bucket_originals, minio_key, tmp_path)
        download_file(tmp_path, minio_key, settings.minio_bucket_originals)

        logger.info("[%s] Parsing and chunking %s", self.request.id, tmp_path)
        chunks = parse_and_semantic_hierarchical_chunk(tmp_path, document_id, course_id)
        logger.info("[%s] Produced %d chunks", self.request.id, len(chunks))

        logger.info("[%s] Building knowledge graph (async)", self.request.id)
        kg = get_kg()
        # Concurrent extraction — fans the per-chunk LLM calls out in parallel instead of
        # sequential-with-throttle. The Celery worker runs this task synchronously, so drive the
        # coroutine to completion with asyncio.run.
        asyncio.run(kg.abuild_from_dicts(chunks))
        logger.info("[%s] Knowledge graph built", self.request.id)

        logger.info("[%s] Embedding chunks", self.request.id)
        chunks = embed_chunks(chunks)
        logger.info("[%s] Upserting %d chunks into Qdrant", self.request.id, len(chunks))
        upsert_chunks(chunks)

        md_path = tmp_path + ".md"
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"Markdown file not generated at {md_path}")
        logger.debug("[%s] Uploading markdown to MinIO bucket=%s key=%s",
                     self.request.id, settings.minio_bucket_markdown, minio_key + ".md")
        upload_file(tmp_path + ".md", minio_key + ".md", settings.minio_bucket_markdown)

        doc.status = DocumentStatus.ready
        db.commit()
        logger.info("[%s] Status → ready | chunks=%d", self.request.id, len(chunks))

        return {"document_id": document_id, "chunks_count": len(chunks)}

    except Exception as exc:
        logger.exception("[%s] Ingestion failed for document_id=%s: %s",
                         self.request.id, document_id, exc)
        if doc:
            doc.status = DocumentStatus.failed
            doc.error_msg = str(exc)
            db.commit()
        raise

    finally:
        db.close()
        if "tmp_path" in locals():
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if os.path.exists(tmp_path + ".md"):
                os.unlink(tmp_path + ".md")
            logger.debug("[%s] Cleaned up temp files", self.request.id)
