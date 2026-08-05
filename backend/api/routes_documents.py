import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.db.postgre import Document, DocumentStatus, get_db
from backend.db.minio_client import  sha256_of_file, upload_file
from backend.tasks.ingestion_tasks import process_document

import tempfile
import os

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", status_code=202)
async def upload_document(
    course_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # Save upload to a temp file so we can hash it and pass it to MinIO
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        file_hash = sha256_of_file(tmp_path)

        # Dedup: reject if this exact file is already in the DB
        existing = db.query(Document).filter(Document.sha256_hash == file_hash).first()
        if existing:
            return JSONResponse(
                status_code=200,
                content={"message": "Already indexed.", "document_id": existing.id, "status": existing.status.value},
            )

        document_id = str(uuid.uuid4())
        minio_key = f"{document_id}{suffix}"

        upload_file(tmp_path, minio_key, settings.minio_bucket_originals)

        doc = Document(
            id=document_id,
            filename=file.filename,
            course_id=course_id,
            status=DocumentStatus.pending,
            sha256_hash=file_hash,
            minio_key=minio_key,
        )
        db.add(doc)
        db.commit()

        process_document.delay(document_id, minio_key, course_id)

        return JSONResponse(
            status_code=202,
            content={"document_id": document_id, "status": DocumentStatus.pending.value},
        )

    finally:
        os.unlink(tmp_path)


@router.get("/")
def list_documents(course_id: str = "default", db: Session = Depends(get_db)):
    docs = (
        db.query(Document)
        .filter(Document.course_id == course_id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return [
        {
            "document_id": d.id,
            "filename": d.filename,
            "status": d.status,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "error_msg": d.error_msg,
        }
        for d in docs
    ]


@router.get("/{document_id}/status")
def get_document_status(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "error_msg": doc.error_msg,
    }
