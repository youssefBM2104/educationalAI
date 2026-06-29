import logging

from backend.agents.state import AgentState

logger = logging.getLogger(__name__)


# --- Shared retrieve node (used by the top-level graph, before routing) ---

def retrieve_node(state: AgentState) -> dict:
    # Lazy import: pulls in Qdrant / Neo4j / embedding model only when run
    from backend.rag.retriever import retrieve_with_kg
    from backend.rag.reranker import rerank_chunks

    result = retrieve_with_kg(query=state["query"], top_k=5, course_id=state.get("course_id"))
    for c in result["base_chunks"]:
        c["source"] = "vector"
    for c in result["kg_chunks"]:
        c["source"] = "kg"

    merged = result["base_chunks"] + result["kg_chunks"]
    if merged:
        merged = rerank_chunks(state["query"], merged)

    logger.info("Retrieve: %d chunks, %d relations",
                len(merged), len(result["kg_context"].get("relations", [])))
    return {"rag_chunks": merged, "kg_context": result["kg_context"]}
