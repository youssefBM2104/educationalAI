import tempfile
import os
import asyncio

from celery import Celery
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from backend.core.config import settings
from backend.db.postgre import SessionLocal, Document, DocumentStatus
from backend.db.minio_client import download_file, upload_file
from backend.rag.ingestion import parse_and_chunk
from backend.KG.kg_builder import KGBuilder
from backend.db.kg_client import get_kg



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

@celery_app.task
def process_document(document_id: str, minio_key: str, course_id: str):
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        doc.status = DocumentStatus.processing
        db.commit()

        suffix = os.path.splitext(minio_key)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        download_file(tmp_path, minio_key, settings.minio_bucket_originals)

        chunks = parse_and_chunk(tmp_path, document_id, course_id)

        # Knowledge Graph construction 
        kg = get_kg()
        
        kg.build_from_dicts(chunks)

        md_path = tmp_path + ".md"
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"Markdown file not generated at {md_path}")
        upload_file(tmp_path +".md", minio_key+".md", settings.minio_bucket_markdown)

        doc.status = DocumentStatus.ready
        db.commit()


        return {"document_id": document_id, "chunks_count": len(chunks)}

    except Exception as exc:
        doc = db.get(Document, document_id)
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
