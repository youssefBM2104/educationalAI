"""
GET /shared/export — Export endpoint for the "Shared DB" stand-in.

Reads from whichever Qdrant/Neo4j/Postgres instance this process is configured
against (i.e. the ETL stack's DBs when run as part of docker-compose.yml).
Returns everything a local instance needs to reconstruct one course's content.

Pagination: Qdrant is cursor-paginated (pass next_offset_id back each call).
Neo4j and image refs are returned in full on the first page only (offset_id=null)
because they are small compared to the vector payload.

⚠  No auth yet — flagged as a known gap for production hardening.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.db.postgre import Document, Image, get_db
from backend.db.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/shared", tags=["shared-db"])

_DEFAULT_PAGE_SIZE = 200
_MAX_PAGE_SIZE = 1000


def _neo4j_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


@router.get("/export")
def export_course(
    course_id: str = Query(..., description="Scope the export to this course."),
    limit: int = Query(default=_DEFAULT_PAGE_SIZE, ge=1, le=_MAX_PAGE_SIZE),
    offset_id: Optional[str] = Query(
        default=None,
        description="Cursor returned by the previous page. Omit to start from the beginning.",
    ),
    db: Session = Depends(get_db),
):
    """
    Export all indexed content for *course_id* — Qdrant points (with vectors),
    Neo4j concept nodes and relationships, and MinIO image paths.

    Clients should paginate by passing the returned `next_offset_id` on each
    subsequent call until `has_more` is false.  Neo4j and image refs are only
    included on the first page (offset_id=null); Qdrant points are paginated.
    """
    # ── Qdrant: scroll with vectors ───────────────────────────────────────────
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    qdrant = get_qdrant_client()
    course_filter = Filter(
        must=[FieldCondition(key="course_id", match=MatchValue(value=course_id))]
    )

    try:
        points_raw, next_offset = qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=course_filter,
            limit=limit,
            offset=offset_id,         # None → start; UUID string → resume
            with_payload=True,
            with_vectors=True,
        )
    except Exception as exc:
        logger.error("Qdrant scroll failed for course_id=%s: %s", course_id, exc)
        raise HTTPException(status_code=500, detail="Qdrant read error.")

    qdrant_points = []
    for p in points_raw:
        vectors = p.vector if isinstance(p.vector, dict) else {}
        dense = vectors.get("dense") or []

        sparse_raw = vectors.get("sparse")
        sparse = None
        if sparse_raw is not None:
            sparse = {
                "indices": list(sparse_raw.indices),
                "values":  [float(v) for v in sparse_raw.values],
            }

        qdrant_points.append({
            "id":            str(p.id),
            "dense_vector":  dense,
            "sparse_vector": sparse,
            "payload":       dict(p.payload or {}),
        })

    has_more = next_offset is not None
    next_offset_str = str(next_offset) if next_offset is not None else None

    # ── Neo4j + image refs — first page only ─────────────────────────────────
    neo4j_nodes: list[dict] = []
    neo4j_relationships: list[dict] = []
    image_refs: list[dict] = []

    if offset_id is None:
        # Neo4j: fetch concept nodes reachable from documents in this course
        try:
            driver = _neo4j_driver()
            with driver.session() as session:
                node_rows = session.run(
                    """
                    MATCH (d:Document)-[:MENTIONS]->(c)
                    WHERE d.course_id = $course_id
                      AND NOT c:Document
                    RETURN DISTINCT labels(c) AS labels, properties(c) AS props
                    """,
                    course_id=course_id,
                )
                for row in node_rows:
                    neo4j_nodes.append({
                        "labels": list(row["labels"]),
                        "props":  dict(row["props"]),
                    })

                concept_ids = [
                    n["props"]["id"]
                    for n in neo4j_nodes
                    if n["props"].get("id")
                ]
                if concept_ids:
                    rel_rows = session.run(
                        """
                        MATCH (a)-[r]->(b)
                        WHERE a.id IN $ids AND b.id IN $ids
                          AND NOT a:Document AND NOT b:Document
                        RETURN DISTINCT a.id AS from_id, type(r) AS rel_type, b.id AS to_id
                        """,
                        ids=concept_ids,
                    )
                    for row in rel_rows:
                        neo4j_relationships.append({
                            "from_id":  row["from_id"],
                            "rel_type": row["rel_type"],
                            "to_id":    row["to_id"],
                        })
            driver.close()
        except Exception as exc:
            # Non-fatal: Qdrant data is still valid even if Neo4j is unavailable.
            logger.warning("Neo4j export failed for course_id=%s (skipped): %s", course_id, exc)

        # Postgres → image refs
        doc_ids = [
            row.id
            for row in db.query(Document.id)
            .filter(Document.course_id == course_id)
            .all()
        ]
        if doc_ids:
            images = db.query(Image).filter(Image.document_id.in_(doc_ids)).all()
            image_refs = [
                {
                    "image_id":       img.image_id,
                    "document_id":    img.document_id,
                    "chunk_id":       img.chunk_id,
                    "page_number":    img.page_number,
                    "minio_path":     img.minio_path,
                    "vlm_description": img.vlm_description,
                }
                for img in images
            ]

    return {
        "course_id":           course_id,
        "qdrant_points":       qdrant_points,
        "neo4j_nodes":         neo4j_nodes,
        "neo4j_relationships": neo4j_relationships,
        "image_refs":          image_refs,
        "next_offset_id":      next_offset_str,
        "has_more":            has_more,
    }
