import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.rag.retriever import retrieve, retrieve_with_kg, resolve_retrieval_query
from backend.rag.reranker import rerank_chunks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------

class ChunkResult(BaseModel):
    score: float | None
    text: str | None
    document_id: str | None
    course_id: str | None
    chunk_index: int | None
    covers_concepts: list | None
    source: str  # "vector" | "kg"


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
    retrieval_query: str | None = Field(
        default=None,
        description="Explicit topic query to embed for retrieval; when set, skips the rewrite.",
    )
    rewrite_query: bool = Field(
        default=True,
        description=(
            "Rewrite the request down to its topic before embedding (strips instruction wording "
            "like 'create a 5-question MCQ exam about …' that pollutes retrieval relevance)."
        ),
    )
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
    # Resolve the topic query once so retrieval AND the rerank use the same clean signal.
    eff_query = resolve_retrieval_query(request.query, request.retrieval_query, request.rewrite_query)
    try:
        chunks = retrieve(
            query=eff_query,
            top_k=request.top_k,
            course_id=request.course_id,
            rewrite_query=False,
        )
    except Exception as e:
        logger.error("Retrieval failed for query=%r: %s", request.query, e)
        raise HTTPException(status_code=500, detail="Retrieval failed. Check server logs.")

    # Tag source before reranking
    for chunk in chunks:
        chunk["source"] = "vector"

    if request.rerank and chunks:
        try:
            chunks = rerank_chunks(eff_query, chunks, top_k=request.top_k)
        except Exception as e:
            logger.error("Reranking failed: %s", e)
            raise HTTPException(status_code=500, detail="Reranking failed. Check server logs.")

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
        description="Max number of KG-sourced chunks to append after the cross-encoder relevance gate.",
    )
    kg_score_threshold: float = Field(
        default=0.0,
        description=(
            "Minimum cross-encoder score for a KG-expanded chunk to be kept. KG chunks are always "
            "scored by the cross-encoder (query × chunk) and only the relevant ones are appended; "
            "0.0 ≈ the relevant/irrelevant boundary for the ms-marco cross-encoder."
        ),
    )
    retrieval_query: str | None = Field(
        default=None,
        description="Explicit topic query to embed for retrieval; when set, skips the rewrite.",
    )
    rewrite_query: bool = Field(
        default=True,
        description=(
            "Rewrite the request down to its topic before embedding (strips instruction wording "
            "like 'create a 5-question MCQ exam about …' that pollutes retrieval relevance)."
        ),
    )
    rerank: bool = Field(
        default=True,
        description=(
            "Rerank the merged (vector + KG) chunk list using BGE cross-encoder. "
            "When True, returns a single flat list sorted by cross-encoder score. "
            "When False, vector chunks come first (by RRF score), then the relevance-gated "
            "KG chunks (each already carrying its cross-encoder score)."
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

    Response always returns a flat `chunks` list. Use chunk.source to
    distinguish vector-retrieved ("vector") from KG-expanded ("kg") chunks.
    """
    try:
        result = retrieve_with_kg(
            query=request.query,
            top_k=request.top_k,
            course_id=request.course_id,
            kg_hops=request.kg_hops,
            kg_extra_chunks=request.kg_extra_chunks,
            kg_score_threshold=request.kg_score_threshold,
            retrieval_query=request.retrieval_query,
            rewrite_query=request.rewrite_query,
        )
    except Exception as e:
        logger.error("KG retrieval failed for query=%r: %s", request.query, e)
        raise HTTPException(status_code=500, detail="KG retrieval failed. Check server logs.")

    # Tag sources before merging
    for chunk in result["base_chunks"]:
        chunk["source"] = "vector"
    for chunk in result["kg_chunks"]:
        chunk["source"] = "kg"

    # Merge — vector chunks first, then KG chunks
    merged = result["base_chunks"] + result["kg_chunks"]

    if request.rerank and merged:
        try:
            # Rerank over the full merged set — cross-encoder decides final order. Use the resolved
            # topic query (not the raw instruction) so the final order matches retrieval.
            # top_k not applied here: reranker scores all candidates, consumer decides how many.
            merged = rerank_chunks(result.get("retrieval_query") or request.query, merged)
        except Exception as e:
            logger.error("Reranking failed: %s", e)
            raise HTTPException(status_code=500, detail="Reranking failed. Check server logs.")

    return KGQueryResponse(
        query=request.query,
        course_id=request.course_id,
        reranked=request.rerank,
        chunks=[ChunkResult(**c) for c in merged],
        kg_context=KGContext(**result["kg_context"]),
    )