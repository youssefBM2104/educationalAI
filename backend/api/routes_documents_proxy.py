"""
Document routes for the local backend — thin proxy to the ETL backend.

The local stack owns RAG and sync; the ETL stack owns document processing.
All upload / list / status calls are forwarded to the ETL backend so that
Celery, MinIO, and the ETL databases stay as the single source of truth for
document state. The local databases (Qdrant, Neo4j) are populated later via
POST /sync/import.
"""

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from backend.core.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])

_HTTP_TIMEOUT = 60.0


def _etl_url(path: str) -> str:
    return f"{settings.shared_db_url.rstrip('/')}{path}"


@router.post("/upload", status_code=202)
async def upload_document(
    course_id: str = Form(...),
    file: UploadFile = File(...),
):
    content = await file.read()
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        try:
            resp = await client.post(
                _etl_url("/documents/upload"),
                data={"course_id": course_id},
                files={"file": (file.filename, content, file.content_type)},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"ETL backend unreachable: {exc}")

    return JSONResponse(status_code=resp.status_code, content=resp.json())


@router.get("/")
async def list_documents(course_id: str = "default"):
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        try:
            resp = await client.get(
                _etl_url("/documents/"),
                params={"course_id": course_id},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"ETL backend unreachable: {exc}")

    return JSONResponse(status_code=resp.status_code, content=resp.json())


@router.get("/{document_id}/status")
async def get_document_status(document_id: str):
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        try:
            resp = await client.get(_etl_url(f"/documents/{document_id}/status"))
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"ETL backend unreachable: {exc}")

    return JSONResponse(status_code=resp.status_code, content=resp.json())
