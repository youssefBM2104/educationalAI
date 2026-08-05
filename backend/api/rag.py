import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.rag.retriever import retrieve, retrieve_with_kg
from backend.rag.reranker import rerank_chunks
from backend.db.postgre import SessionLocal, Image
from backend.db.minio_client import get_presigned_image_url, get_minio_client
from backend.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------


class ImageRef(BaseModel):
    image_id: str
    page_number: int | None
    bbox: dict | None
    image_base64: str
    mime_type: str
    vlm_description: str | None

class ChunkResult(BaseModel):
    chunk_id: str | None
    score: float | None
    text: str | None
    document_id: str | None
    course_id: str | None
    chunk_index: int | None
    covers_concepts: list | None
    source: str
    images: list[ImageRef] = []

import base64
def _attach_images(chunks: list[dict]) -> list[dict]:
    chunk_ids = [c["chunk_id"] for c in chunks if c.get("chunk_id")]
    if not chunk_ids:
        for c in chunks:
            c["images"] = []
        return chunks

    db = SessionLocal()
    try:
        rows = db.query(Image).filter(Image.chunk_id.in_(chunk_ids)).all()
    finally:
        db.close()

    client = get_minio_client()
    by_chunk: dict[str, list[dict]] = {}
    for row in rows:
        obj = client.get_object(settings.minio_bucket_images, row.minio_path)
        img_bytes = obj.read()
        obj.close()

        compressed = _compress_image(img_bytes)

        by_chunk.setdefault(row.chunk_id, []).append({
            "image_id": row.image_id,
            "page_number": row.page_number,
            "bbox": {
                "left": row.bbox_left, "top": row.bbox_top,
                "right": row.bbox_right, "bottom": row.bbox_bottom,
            },
            "image_base64": base64.b64encode(compressed).decode(),
            "mime_type": "image/jpeg",
            "vlm_description": row.vlm_description,
        })

    for c in chunks:
        c["images"] = by_chunk.get(c.get("chunk_id"), [])
    return chunks


from PIL import Image as PILImage
from io import BytesIO
import base64

def _compress_image(img_bytes: bytes, max_dimension: int = 1200, quality: int = 85) -> bytes:
    img = PILImage.open(BytesIO(img_bytes))
    img.thumbnail((max_dimension, max_dimension), PILImage.LANCZOS)
    buffer = BytesIO()
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")  # JPEG doesn't support alpha/palette
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()

class KGContext(BaseModel):
    concepts: list[str]
    relations: list[dict]


# ---------------------------------------------------------------------------
# POST /rag/query — pure vector retrieval + optional reranking
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    course_id: str | None = Field(
        default=None,
        description="Restrict retrieval to a specific course. Omit to search all courses.",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = Field(
        default=True,
        description=(
            "Rerank retrieved chunks using BGE cross-encoder. "
            "Produces more precise relevance scores at the cost of extra latency. "
            "Set to False for faster responses during development."
        ),
    )


class QueryResponse(BaseModel):
    query: str
    course_id: str | None
    reranked: bool
    results: list[ChunkResult]




@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Hybrid dense+sparse retrieval with RRF fusion.
    Optionally reranked by BGE cross-encoder.
    """
    fetch_k = request.top_k * 3 if request.rerank else request.top_k

    try:
        chunks = retrieve(
            query=request.query,
            top_k=fetch_k,
            course_id=request.course_id,
        )
    except Exception as e:
        logger.error("Retrieval failed for query=%r: %s", request.query, e)
        raise HTTPException(status_code=500, detail="Retrieval failed. Check server logs.")

    # Tag source before reranking
    for chunk in chunks:
        chunk["source"] = "vector"

    if request.rerank and chunks:
        try:
            chunks = rerank_chunks(request.query, chunks, top_k=request.top_k)
        except Exception as e:
            logger.error("Reranking failed: %s", e)
            raise HTTPException(status_code=500, detail="Reranking failed. Check server logs.")

    chunks = _attach_images(chunks)

    return QueryResponse(
        query=request.query,
        course_id=request.course_id,
        reranked=request.rerank,
        results=[ChunkResult(**c) for c in chunks],
    )
# ---------------------------------------------------------------------------
# POST /rag/query-with-kg — vector + KG expansion + optional reranking
# ---------------------------------------------------------------------------

class KGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    course_id: str | None = Field(default=None)
    top_k: int = Field(default=5, ge=1, le=20)
    kg_hops: int = Field(
        default=1,
        ge=1,
        le=3,
        description=(
            "Neo4j traversal depth. "
            "1 = direct concept neighbors (high precision). "
            "2 = neighbors of neighbors (broader context). "
            "3+ not recommended on large graphs."
        ),
    )
    kg_extra_chunks: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max number of KG-sourced chunks to append before reranking.",
    )
    rerank: bool = Field(
        default=True,
        description=(
            "Rerank the merged (vector + KG) chunk list using BGE cross-encoder. "
            "When True, returns a single flat list sorted by cross-encoder score. "
            "When False, vector chunks come first (by RRF score), "
            "KG chunks appended at the end with score=null."
        ),
    )


class KGQueryResponse(BaseModel):
    query: str
    course_id: str | None
    reranked: bool
    chunks: list[ChunkResult]   # merged vector + KG, use chunk.source to distinguish
    kg_context: KGContext


@router.post("/query-with-kg", response_model=KGQueryResponse)
def query_rag_with_kg(request: KGQueryRequest):
    """
    KG-augmented retrieval: vector search + knowledge graph expansion.
    Optionally reranked by BGE cross-encoder over the merged chunk list.
    """
    fetch_k = request.top_k * 3 if request.rerank else request.top_k

    try:
        result = retrieve_with_kg(
            query=request.query,
            top_k=fetch_k,
            course_id=request.course_id,
            kg_hops=request.kg_hops,
            kg_extra_chunks=request.kg_extra_chunks,
        )
    except Exception as e:
        logger.error("KG retrieval failed for query=%r: %s", request.query, e)
        raise HTTPException(status_code=500, detail="KG retrieval failed. Check server logs.")

    for chunk in result["base_chunks"]:
        chunk["source"] = "vector"
    for chunk in result["kg_chunks"]:
        chunk["source"] = "kg"

    merged = result["base_chunks"] + result["kg_chunks"]

    if request.rerank and merged:
        try:
            merged = rerank_chunks(request.query, merged, top_k=request.top_k)
        except Exception as e:
            logger.error("Reranking failed: %s", e)
            raise HTTPException(status_code=500, detail="Reranking failed. Check server logs.")

    merged = _attach_images(merged)

    return KGQueryResponse(
        query=request.query,
        course_id=request.course_id,
        reranked=request.rerank,
        chunks=[ChunkResult(**c) for c in merged],
        kg_context=KGContext(**result["kg_context"]),
    )