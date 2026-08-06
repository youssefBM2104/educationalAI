"""
POST /sync/import — Pull a course's indexed content from the Shared DB and
upsert it into this local instance's Qdrant and Neo4j.

Idempotent: Qdrant upserts by the same deterministic chunk UUID already used
by the ETL pipeline (uuid.uuid5), so re-syncing produces no duplicates.
Neo4j uses MERGE on node id for the same guarantee.

⚠  No auth yet — flagged as a known gap for production hardening.
"""

import logging
import re
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from backend.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sync", tags=["sync"])

_UPSERT_BATCH = 100
_HTTP_TIMEOUT = 120.0
_EXPORT_PAGE_SIZE = 500

# Only allow Neo4j labels/relationship types that are safe identifiers
_SAFE_ID = re.compile(r'^[A-Za-z0-9_]+$')


class ImportRequest(BaseModel):
    course_id: str = Field(..., min_length=1)
    shared_db_url: Optional[str] = Field(
        default=None,
        description=(
            "Base URL of the Shared DB export service. "
            "Falls back to the SHARED_DB_URL env var if omitted."
        ),
    )


@router.post("/import")
def import_from_shared(request: ImportRequest):
    """
    Pull all indexed content for *course_id* from the Shared DB and upsert it
    into the local Qdrant and Neo4j.  Loops through all export pages until
    `has_more` is false.

    Returns a summary of what was synced.
    """
    base_url = (request.shared_db_url or settings.shared_db_url).rstrip("/")
    export_url = f"{base_url}/shared/export"

    qdrant = _get_local_qdrant()
    neo4j_driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    total_points = 0
    total_nodes  = 0
    total_rels   = 0
    total_images = 0
    offset_id: Optional[str] = None
    first_page = True

    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT) as http:
            while True:
                params: dict = {"course_id": request.course_id, "limit": _EXPORT_PAGE_SIZE}
                if offset_id:
                    params["offset_id"] = offset_id

                try:
                    resp = http.get(export_url, params=params)
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Shared DB returned {exc.response.status_code}: {exc.response.text[:200]}",
                    )
                except httpx.HTTPError as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Shared DB unreachable at {export_url}: {exc}",
                    )

                page = resp.json()

                # ── Qdrant upsert ─────────────────────────────────────────────
                points: list[PointStruct] = []
                for p in page.get("qdrant_points", []):
                    dense = p.get("dense_vector") or []
                    sparse_raw = p.get("sparse_vector")

                    vector: dict = {}
                    if dense:
                        vector["dense"] = dense
                    if sparse_raw:
                        vector["sparse"] = SparseVector(
                            indices=sparse_raw["indices"],
                            values=sparse_raw["values"],
                        )
                    if not vector:
                        logger.warning("Skipping point %s — no usable vectors.", p.get("id"))
                        continue

                    points.append(PointStruct(
                        id=p["id"],
                        vector=vector,
                        payload=p.get("payload", {}),
                    ))

                _upsert_batched(qdrant, points)
                total_points += len(points)

                # ── Neo4j + image refs — first page only ──────────────────────
                if first_page:
                    with neo4j_driver.session() as session:
                        total_nodes += _upsert_nodes(session, page.get("neo4j_nodes", []))
                        total_rels  += _upsert_rels(session,  page.get("neo4j_relationships", []))
                    total_images = len(page.get("image_refs", []))
                    first_page = False

                if not page.get("has_more"):
                    break
                offset_id = page.get("next_offset_id")

    finally:
        neo4j_driver.close()

    logger.info(
        "Sync complete — course=%s points=%d nodes=%d rels=%d images_refs=%d",
        request.course_id, total_points, total_nodes, total_rels, total_images,
    )

    return {
        "course_id":                request.course_id,
        "shared_db_url":            base_url,
        "qdrant_points_upserted":   total_points,
        "neo4j_nodes_upserted":     total_nodes,
        "neo4j_relationships_upserted": total_rels,
        "image_refs_received":      total_images,
        "note": (
            "Image binary data was not transferred — only paths. "
            "Binary sync is a planned follow-up task."
        ),
    }


# ── Qdrant helpers ─────────────────────────────────────────────────────────────

def _get_local_qdrant() -> QdrantClient:
    client = QdrantClient(url=settings.qdrant_url)
    existing_names = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing_names:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": VectorParams(size=1024, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        logger.info("Created local Qdrant collection: %s", settings.qdrant_collection)
    return client


def _upsert_batched(qdrant: QdrantClient, points: list[PointStruct]) -> None:
    try:
        for i in range(0, len(points), _UPSERT_BATCH):
            qdrant.upsert(
                collection_name=settings.qdrant_collection,
                points=points[i : i + _UPSERT_BATCH],
            )
    except UnexpectedResponse as exc:
        logger.error("Qdrant upsert failed — %s %r", exc.status_code, exc.content)
        raise HTTPException(status_code=500, detail="Local Qdrant upsert failed.")


# ── Neo4j helpers ──────────────────────────────────────────────────────────────

def _safe_identifier(value: str, kind: str) -> str:
    """Validate Neo4j label / relationship type to prevent Cypher injection."""
    if _SAFE_ID.match(value):
        return value
    raise HTTPException(
        status_code=422,
        detail=f"Unsafe Neo4j {kind} received from Shared DB: {value!r}",
    )


def _upsert_nodes(session, nodes: list[dict]) -> int:
    count = 0
    for node in nodes:
        labels = node.get("labels") or []
        props  = node.get("props") or {}
        node_id = props.get("id")
        if not node_id or not labels:
            continue
        label = _safe_identifier(labels[0], "label")
        # MERGE on (label, id) — fully idempotent
        session.run(
            f"MERGE (n:`{label}` {{id: $id}}) SET n += $props",
            id=node_id,
            props=props,
        )
        count += 1
    return count


def _upsert_rels(session, relationships: list[dict]) -> int:
    count = 0
    for rel in relationships:
        from_id  = rel.get("from_id")
        to_id    = rel.get("to_id")
        rel_type = rel.get("rel_type")
        if not (from_id and to_id and rel_type):
            continue
        _safe_identifier(rel_type, "relationship type")
        # MERGE relationship — idempotent; nodes must already exist from _upsert_nodes
        session.run(
            f"MATCH (a {{id: $from_id}}), (b {{id: $to_id}}) "
            f"MERGE (a)-[:`{rel_type}`]->(b)",
            from_id=from_id,
            to_id=to_id,
        )
        count += 1
    return count
