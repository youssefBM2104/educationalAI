import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from neo4j import GraphDatabase
from qdrant_client.models import Filter, FieldCondition, MatchValue

from backend.core.config import settings
from backend.db.postgre import Document, DocumentStatus, Image, get_db
from backend.db.minio_client import sha256_of_file, upload_file, get_minio_client
from backend.db.qdrant_client import get_qdrant_client
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
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        file_hash = sha256_of_file(tmp_path)

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


@router.get("/{document_id}/extraction")
def get_extraction(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if doc.status != DocumentStatus.ready:
        raise HTTPException(status_code=409, detail=f"Document not ready (status: {doc.status}).")

    # ── Chunks from Qdrant ────────────────────────────────────────────────────
    qdrant = get_qdrant_client()
    doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
    chunks = []
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=doc_filter,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            chunks.append({
                "chunk_id": str(p.id),
                "text": p.payload.get("text", ""),
                "page_number": p.payload.get("chunk_index"),
            })
        if offset is None:
            break

    # ── Images from Postgres + MinIO presigned URLs ───────────────────────────
    image_rows = db.query(Image).filter(Image.document_id == document_id).all()
    images = []
    for img in image_rows:
        images.append({
            "image_id": img.image_id,
            "url": f"/documents/{document_id}/images/{img.image_id}",
            "vlm_description": img.vlm_description,
            "page_number": img.page_number,
        })

    # ── KG from Neo4j ─────────────────────────────────────────────────────────
    kg: list = []
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        with driver.session() as session:
            # Concept nodes directly mentioned by this document's chunks
            nodes_result = session.run(
                """
                MATCH (d:Document)-[:MENTIONS]->(n)
                WHERE d.document_id = $doc_id
                RETURN DISTINCT n.id AS label, labels(n) AS types
                LIMIT 50
                """,
                doc_id=document_id,
            )
            primary_labels = {row["label"] for row in nodes_result}

            # Edges between those concept nodes
            edges_result = session.run(
                """
                MATCH (d:Document)-[:MENTIONS]->(a)
                WHERE d.document_id = $doc_id
                MATCH (a)-[r]->(b)
                WHERE b.id IN $labels
                RETURN DISTINCT a.id AS from_label, type(r) AS predicate, b.id AS to_label
                LIMIT 100
                """,
                doc_id=document_id,
                labels=list(primary_labels),
            )
            edges = [{"from": row["from_label"], "predicate": row["predicate"], "to": row["to_label"]}
                     for row in edges_result]
        driver.close()

        # Build flat KgElement list: node → edge → node chains
        seen = set()
        for edge in edges:
            if edge["from"] not in seen:
                kg.append({"label": edge["from"], "type": "primary"})
                seen.add(edge["from"])
            kg.append({"predicate": edge["predicate"]})
            if edge["to"] not in seen:
                kg.append({"label": edge["to"], "type": "secondary"})
                seen.add(edge["to"])
        # Append any isolated nodes not involved in edges
        for label in primary_labels:
            if label not in seen:
                kg.append({"label": label, "type": "primary"})
    except Exception:
        pass  # KG is best-effort; chunks and images still return

    return {"chunks": chunks, "images": images, "kg": kg}


@router.get("/{document_id}/images/{image_id}")
def get_image(document_id: str, image_id: str, db: Session = Depends(get_db)):
    img = db.query(Image).filter(
        Image.document_id == document_id,
        Image.image_id == image_id,
    ).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found.")
    minio = get_minio_client()
    response = minio.get_object(settings.minio_bucket_images, img.minio_path)
    return StreamingResponse(response, media_type="image/png")


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
