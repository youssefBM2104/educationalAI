import logging

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Prefetch,
    SparseVector,
)

from backend.etl.embedder import model
from backend.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clients — initialized once at module load
# ---------------------------------------------------------------------------

qdrant = QdrantClient(url=settings.qdrant_url)

neo4j_driver = GraphDatabase.driver(
    settings.neo4j_uri,
    auth=(settings.neo4j_user, settings.neo4j_password),
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_sparse_vector(lexical_weights: dict) -> dict:
    """
    Convert BGE-M3 lexical weights to Qdrant sparse vector format.

    BGE-M3 returns lexical weights keyed by token string or token ID.
    Qdrant expects {indices: list[int], values: list[float]}.
    Duplicate token IDs (from subword tokenization) are summed.
    """
    id_to_weight: dict[int, float] = {}
    for token_string, weight in lexical_weights.items():
        token_id = (
            token_string
            if isinstance(token_string, int)
            else model.tokenizer.convert_tokens_to_ids(token_string)
        )
        if isinstance(token_id, int) and token_id >= 0:
            id_to_weight[token_id] = id_to_weight.get(token_id, 0.0) + float(weight)
    return {
        "indices": list(id_to_weight.keys()),
        "values": list(id_to_weight.values()),
    }


def _get_kg_subgraph(concept_ids: list[str], hops: int = 1) -> dict:
    """
    Traverse the Neo4j knowledge graph from a set of seed concept IDs and
    return the subgraph (nodes + edges) reachable within `hops` edges.

    Traversal is undirected — follows relationships in both directions.

    Returns:
        {
            "concepts":  list[str]   — all concept IDs in the subgraph
                                       (seeds + discovered neighbors)
            "relations": list[dict]  — edges: {"from", "type", "to"}
        }

    Node labels:   Concept, Formula, Theorem, Example, Method, Definition
    Relationships: PREREQUISITE, EXTENDS, DEFINES, APPLIES_TO, ILLUSTRATES, PART_OF
    """
    # NOTE: Neo4j does not support parameters as variable-length path bounds.
    # [*1..$hops] with $hops as a Cypher parameter is silently ignored in most
    # Neo4j versions. The bound must be a literal integer embedded in the query.
    # hops is validated as ge=1, le=3 by the Pydantic model — safe to interpolate.
    # NOTE: Neo4j does not support parameters as variable-length path bounds.
    # [*1..$hops] with $hops as a Cypher parameter is silently ignored in most
    # Neo4j versions. The bound must be a literal integer embedded in the query.
    # hops is validated as ge=1, le=3 by the Pydantic model — safe to interpolate.
    #
    # NOTE: A WHERE clause cannot follow UNWIND directly in all Neo4j versions.
    # A WITH clause is required to re-scope the variable before filtering.
    cypher = f"""
        MATCH path = (start)-[*1..{hops}]-(related)
        WHERE start.id IN $concept_ids
          AND NOT related:Document
        UNWIND relationships(path) AS rel
        WITH rel
        WHERE NOT startNode(rel):Document AND NOT endNode(rel):Document
        RETURN DISTINCT
            startNode(rel).id AS from_id,
            type(rel)         AS rel_type,
            endNode(rel).id   AS to_id
    """
    relations = []
    all_concept_ids = set(concept_ids)

    with neo4j_driver.session() as session:
        result = session.run(cypher, concept_ids=concept_ids)
        for record in result:
            from_id  = record["from_id"]
            rel_type = record["rel_type"]
            to_id    = record["to_id"]
            if from_id and to_id:
                relations.append({"from": from_id, "type": rel_type, "to": to_id})
                all_concept_ids.add(from_id)
                all_concept_ids.add(to_id)

    return {
        "concepts": list(all_concept_ids),
        "relations": relations,
    }


def _scroll_by_concepts(
    concept_ids: list[str],
    collection_name: str,
    course_id: str | None,
    limit: int,
) -> list[dict]:
    """
    Retrieve chunks from Qdrant whose covers_concepts list contains
    at least one of the given concept IDs.

    Uses scroll (no vector search) — we are filtering by metadata,
    not by semantic similarity.
    """
    must_conditions = [
        FieldCondition(
            key="covers_concepts[].id",
            match=MatchAny(any=concept_ids),
        )
    ]
    if course_id:
        must_conditions.append(
            FieldCondition(key="course_id", match=MatchValue(value=course_id))
        )

    points, _ = qdrant.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(must=must_conditions),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return [
        {
            "score": None,  # no relevance score — selected via KG traversal
            "text": p.payload.get("text"),
            "document_id": p.payload.get("document_id"),
            "course_id": p.payload.get("course_id"),
            "chunk_index": p.payload.get("chunk_index"),
            "covers_concepts": p.payload.get("covers_concepts"),
        }
        for p in points
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_query(text: str) -> dict:
    """
    Embed a user query using BGE-M3.

    Differences from chunk embedding:
    - batch_size=1   (single query at a time)
    - max_length=512 (queries are short — no need for the 8192 chunk window)

    Returns:
        {
            "dense":  list[float]           (1024-dim cosine vector)
            "sparse": {"indices", "values"} (SPLADE-style lexical weights)
        }
    """
    output = model.encode(
        [text],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return {
        "dense": output["dense_vecs"][0].tolist(),
        "sparse": _build_sparse_vector(output["lexical_weights"][0]),
    }


def retrieve(
    query: str,
    collection_name: str = settings.qdrant_collection,
    top_k: int = 5,
    course_id: str | None = None,
) -> list[dict]:
    """
    Hybrid retrieval over local Qdrant using dense + sparse vectors with RRF fusion.

    Strategy:
        Two prefetch searches run in parallel inside Qdrant:
          - Dense:  cosine similarity over the 1024-dim BGE-M3 vector (semantic)
          - Sparse: dot product over SPLADE-style lexical weights (keyword)
        Each prefetch retrieves 2*top_k candidates.
        Qdrant merges both lists via Reciprocal Rank Fusion (RRF) and
        returns the final top_k results.

    Args:
        query:           Raw user query string (not pre-embedded).
        collection_name: Qdrant collection to search.
        top_k:           Number of chunks to return after fusion.
        course_id:       Optional — restrict to a specific course.

    Returns:
        List of dicts ordered by RRF score descending.
    """
    query_vectors = embed_query(query)

    query_filter = (
        Filter(
            must=[FieldCondition(key="course_id", match=MatchValue(value=course_id))]
        )
        if course_id
        else None
    )

    results = qdrant.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_vectors["dense"],
                using="dense",
                limit=top_k * 2,
            ),
            Prefetch(
                query=SparseVector(
                    indices=query_vectors["sparse"]["indices"],
                    values=query_vectors["sparse"]["values"],
                ),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
    )

    return [
        {
            "score": point.score,
            "text": point.payload.get("text"),
            "document_id": point.payload.get("document_id"),
            "course_id": point.payload.get("course_id"),
            "chunk_index": point.payload.get("chunk_index"),
            "covers_concepts": point.payload.get("covers_concepts"),
        }
        for point in results.points
    ]


def retrieve_with_kg(
    query: str,
    collection_name: str = settings.qdrant_collection,
    top_k: int = 5,
    course_id: str | None = None,
    kg_hops: int = 1,
    kg_extra_chunks: int = 3,
) -> dict:
    """
    KG-augmented retrieval: vector search + knowledge graph expansion.

    Pipeline:
        1. Run base hybrid retrieval (dense + sparse + RRF) → top_k chunks
        2. Collect all concept IDs from the retrieved chunks' covers_concepts
        3. Traverse Neo4j kg_hops away from those concepts → subgraph
           (related concept IDs + the edges between them)
        4. Scroll Qdrant for chunks covering the discovered related concepts
        5. Merge base and KG chunks, deduplicate by (document_id, chunk_index)

    Args:
        query:            Raw user query string.
        collection_name:  Qdrant collection to search.
        top_k:            Number of chunks from base vector retrieval.
        course_id:        Optional course filter applied to both retrieval steps.
        kg_hops:          Traversal depth in Neo4j (1 = direct neighbors only).
                          Keep at 1 for precision; 2 for broader context.
        kg_extra_chunks:  Max number of KG-sourced chunks to append.

    Returns:
        {
            "base_chunks": list[dict],  # top_k from vector retrieval, with RRF score
            "kg_chunks":   list[dict],  # up to kg_extra_chunks from KG expansion, score=None
            "kg_context": {
                "concepts":  list[str],   # all concept IDs in the traversed subgraph
                "relations": list[dict],  # edges: {"from", "type", "to"}
            }
        }
    """
    # Step 1 — base vector retrieval
    base_chunks = retrieve(query, collection_name, top_k, course_id)

    # Step 2 — collect seed concept IDs from retrieved chunks
    seed_concept_ids = [
        concept["id"]
        for chunk in base_chunks
        if chunk.get("covers_concepts")
        for concept in chunk["covers_concepts"]
        if concept.get("id")
    ]

    if not seed_concept_ids:
        logger.info("No covers_concepts in retrieved chunks — skipping KG expansion.")
        return {
            "base_chunks": base_chunks,
            "kg_chunks": [],
            "kg_context": {"concepts": [], "relations": []},
        }

    # Step 3 — traverse Neo4j, get subgraph (concepts + relations)
    try:
        subgraph = _get_kg_subgraph(seed_concept_ids, hops=kg_hops)
    except Exception as e:
        logger.error("Neo4j traversal failed: %s", e)
        return {
            "base_chunks": base_chunks,
            "kg_chunks": [],
            "kg_context": {"concepts": seed_concept_ids, "relations": []},
        }

    # Concepts discovered by the traversal, excluding seeds (used for Qdrant scroll)
    related_concept_ids = [
        cid for cid in subgraph["concepts"] if cid not in set(seed_concept_ids)
    ]

    if not related_concept_ids:
        logger.info("KG traversal returned no new concepts.")
        return {
            "base_chunks": base_chunks,
            "kg_chunks": [],
            "kg_context": subgraph,
        }

    # Step 4 — fetch KG-sourced chunks from Qdrant
    try:
        kg_candidates = _scroll_by_concepts(
            related_concept_ids,
            collection_name,
            course_id,
            limit=kg_extra_chunks * 3,  # over-fetch then trim after dedup
        )
    except Exception as e:
        logger.error("Qdrant scroll by concepts failed: %s", e)
        return {
            "base_chunks": base_chunks,
            "kg_chunks": [],
            "kg_context": subgraph,
        }

    # Step 5 — deduplicate KG chunks against base chunks
    seen = {(c["document_id"], c["chunk_index"]) for c in base_chunks}
    kg_chunks = []
    for chunk in kg_candidates:
        key = (chunk["document_id"], chunk["chunk_index"])
        if key not in seen:
            seen.add(key)
            kg_chunks.append(chunk)
        if len(kg_chunks) >= kg_extra_chunks:
            break

    return {
        "base_chunks": base_chunks,
        "kg_chunks": kg_chunks,
        "kg_context": subgraph,
    }