import logging

from langchain_core.messages import SystemMessage, HumanMessage
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

    Node labels:   Concept, Formula, Theorem, Method, Example, Quantity
    Relationships: PREREQUISITE, PART_OF, EXTENDS, ILLUSTRATES, APPLIES_TO,
                   CAUSES, INCREASES, REDUCES, PREVENTS,
                   CONTRASTS_WITH, TRADES_OFF_AGAINST, IS_INSTANCE_OF, COMMON_MISCONCEPTION_OF
    (Traversal is relation-type agnostic — any new ontology edge is picked up automatically.)
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
# Query rewrite for retrieval
# ---------------------------------------------------------------------------
# A request like "Create a 5-question MCQ exam about X" is embedded verbatim during retrieval, and
# the instruction words ("create", "exam", "5-question", "MCQ") pollute both the dense and sparse
# query vectors — the cross-encoder score against on-topic chunks collapses (measured: +6.75 with
# the topic alone vs +0.33 with the full instruction). We rewrite the request down to its topic
# before embedding. The ORIGINAL request is left untouched for downstream generators (the exam agent
# still needs "5-question MCQ" to know the count and type).

_REWRITE_PROMPT = (
    "You convert a user's request into a short search query for a retrieval system.\n"
    "Keep ONLY the subject matter / topic. Strip task instructions (create, generate, make),\n"
    "output-type words (exam, quiz, MCQ, essay, slides, mindmap, summary), question counts and\n"
    "formatting. Output ONLY the topic phrase — no quotes, no extra words."
)


def _rewrite_query_for_retrieval(query: str) -> str:
    from backend.core.models import MODELS
    try:
        resp = MODELS["gpt-5-nano"].invoke([
            SystemMessage(content=_REWRITE_PROMPT),
            HumanMessage(content=query),
        ])
        topic = (resp.content or "").strip().strip('"').strip()
        if topic and topic.lower() != query.lower():
            logger.info("Query rewrite: %r -> %r", query, topic)
        return topic or query
    except Exception as e:
        logger.warning("Query rewrite failed (%s) — using raw query", e)
        return query


def resolve_retrieval_query(
    query: str, retrieval_query: str | None = None, rewrite_query: bool = True
) -> str:
    """The string actually embedded for retrieval: an explicit `retrieval_query` wins; otherwise the
    request is rewritten to its topic (unless `rewrite_query` is False, which embeds it verbatim)."""
    if retrieval_query:
        return retrieval_query
    if rewrite_query:
        return _rewrite_query_for_retrieval(query)
    return query


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
    retrieval_query: str | None = None,
    rewrite_query: bool = True,
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
    eff_query = resolve_retrieval_query(query, retrieval_query, rewrite_query)
    query_vectors = embed_query(eff_query)

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
    kg_score_threshold: float = 0.0,
    retrieval_query: str | None = None,
    rewrite_query: bool = True,
) -> dict:
    """
    KG-augmented retrieval: vector search + knowledge graph expansion.

    Pipeline:
        1. Run base hybrid retrieval (dense + sparse + RRF) → top_k chunks
        2. Collect all concept IDs from the retrieved chunks' covers_concepts
        3. Traverse Neo4j kg_hops away from those concepts → subgraph
           (related concept IDs + the edges between them)
        4. Scroll Qdrant for chunks covering the discovered related concepts
        5. Deduplicate KG chunks against base chunks by (document_id, chunk_index)
        6. Score the KG chunks with the cross-encoder (query × chunk) and keep only the
           genuinely relevant ones (score ≥ kg_score_threshold), highest first, capped at
           kg_extra_chunks.

    Why step 6: KG expansion widens *recall* by concept adjacency, but adjacency ≠ relevance —
    it pulls in chunks about loosely-related concepts. Previously the kg_extra_chunks slots were
    filled in arbitrary Qdrant scroll order, so a highly relevant KG chunk could be dropped while
    an off-topic one was kept. The cross-encoder is the precise relevance signal we already use
    for reranking, so we apply it here to *select* which KG chunks are worth adding.

    Args:
        query:              Raw user query string.
        collection_name:    Qdrant collection to search.
        top_k:              Number of chunks from base vector retrieval.
        course_id:          Optional course filter applied to both retrieval steps.
        kg_hops:            Traversal depth in Neo4j (1 = direct neighbors only).
                            Keep at 1 for precision; 2 for broader context.
        kg_extra_chunks:    Max number of KG-sourced chunks to append after the relevance gate.
        kg_score_threshold: Minimum cross-encoder score for a KG chunk to be kept. The
                            ms-marco cross-encoder emits raw logits where 0.0 ≈ the
                            relevant/irrelevant boundary (relevant pairs score well above 0,
                            off-topic ones strongly negative), so 0.0 is a sensible default gate.
        retrieval_query:    Explicit topic query to embed. If given, used as-is (no rewrite).
        rewrite_query:      When no retrieval_query is given, rewrite `query` down to its topic
                            before embedding (strips "create a 5-question MCQ exam about …"). The
                            resolved query is returned as `retrieval_query` for the caller's rerank.

    Returns:
        {
            "base_chunks": list[dict],  # top_k from vector retrieval, with RRF score
            "retrieval_query": str,     # the topic query actually embedded (for the caller's rerank)
            "kg_chunks":   list[dict],  # relevance-gated KG chunks, with cross-encoder score
            "kg_context": {
                "concepts":  list[str],   # all concept IDs in the traversed subgraph
                "relations": list[dict],  # edges: {"from", "type", "to"}
            }
        }
    """
    # Resolve the retrieval query once (topic-only), then reuse it for base retrieval, the KG
    # relevance gate, and the returned value so the caller's final rerank uses it too.
    eff_query = resolve_retrieval_query(query, retrieval_query, rewrite_query)

    # Step 1 — base vector retrieval (eff_query already resolved; do not rewrite again)
    base_chunks = retrieve(eff_query, collection_name, top_k, course_id, rewrite_query=False)

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
            "retrieval_query": eff_query,
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
            "retrieval_query": eff_query,
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
            "retrieval_query": eff_query,
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
            "retrieval_query": eff_query,
            "kg_chunks": [],
            "kg_context": subgraph,
        }

    # Step 5 — deduplicate KG candidates against base chunks (keep ALL; do not truncate yet,
    # so the relevance gate below chooses from the full candidate pool rather than scroll order).
    seen = {(c["document_id"], c["chunk_index"]) for c in base_chunks}
    deduped = []
    for chunk in kg_candidates:
        key = (chunk["document_id"], chunk["chunk_index"])
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)

    # Step 6 — cross-encoder relevance gate. Score every (query, KG chunk) pair, keep only those
    # at or above the threshold, highest first, capped at kg_extra_chunks. This replaces the
    # None placeholder score with the authoritative cross-encoder relevance signal. Imported
    # locally so a plain retrieve() call never pays the cross-encoder model load.
    if deduped:
        from backend.rag.reranker import rerank_chunks
        ranked = rerank_chunks(eff_query, deduped)
        kg_chunks = [c for c in ranked if c["score"] >= kg_score_threshold][:kg_extra_chunks]
    else:
        kg_chunks = []

    logger.info("KG expansion: %d candidates -> %d kept (threshold=%.2f)",
                len(deduped), len(kg_chunks), kg_score_threshold)

    return {
        "base_chunks": base_chunks,
        "retrieval_query": eff_query,
        "kg_chunks": kg_chunks,
        "kg_context": subgraph,
    }