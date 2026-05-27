import logging

from backend.core.config import settings
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
     VectorParams, Distance,
     SparseVectorParams,
     PointStruct,
     SparseVector
)

_client = None

logger = logging.getLogger(__name__)

def get_qdrant_client():
    global _client
    if _client :
        return _client

    client = QdrantClient(
        settings.qdrant_url,
    )

    existing = client.get_collections()
    names = [c.name for c in existing.collections]

    if settings.qdrant_collection not in names:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            }
        )
        logger.info(f"Created Qdrant collection: {settings.qdrant_collection}")
    else:
        logger.info(f"Using existing Qdrant collection: {settings.qdrant_collection}")


    _client = client
    return _client

_UPSERT_BATCH_SIZE = 100

def upsert_chunks(chunks):
    client = get_qdrant_client()
    points = [
        PointStruct(
            id=chunk["chunk_id"],
            vector={
                "dense": chunk["dense_vector"],
                "sparse": SparseVector(
                    indices=chunk["sparse_vector"]["indices"],
                    values=chunk["sparse_vector"]["values"],
                ),
            },
            payload={
                "text": chunk["text"],
                "document_id": chunk["document_id"],
                "course_id": chunk["course_id"],
                "chunk_index": chunk["chunk_index"],
                "covers_concepts": chunk["covers_concepts"],
            }
        )
        for chunk in chunks
    ]
    try:
        for i in range(0, len(points), _UPSERT_BATCH_SIZE):
            client.upsert(
                collection_name=settings.qdrant_collection,
                points=points[i : i + _UPSERT_BATCH_SIZE],
            )
    except UnexpectedResponse as e:
        logger.error(
            "Qdrant upsert failed — status=%s reason=%r body=%r",
            e.status_code, e.reason_phrase, e.content,
        )
        raise
