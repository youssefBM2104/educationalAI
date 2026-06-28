import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# cross-encoder/ms-marco-MiniLM-L-6-v2 — BERT-based cross-encoder (~80MB).
# BGE reranker models (bge-reranker-base, bge-reranker-v2-m3) were attempted
# but both raise "XLMRobertaTokenizer has no attribute prepare_for_model" due
# to a transformers version incompatibility — all XLMRoberta-based models are
# affected regardless of FlagEmbedding version.
# CrossEncoder from sentence-transformers (already a FlagEmbedding dependency)
# uses BertTokenizer which is unaffected. No new dependency required.
# Upgrade path: once transformers>=4.44 is pinned, swap back to
# FlagAutoReranker.from_finetuned("BAAI/bge-reranker-v2-m3") for multilingual support.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
logger.info("CrossEncoder reranker loaded")


def rerank_chunks(query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
    """
    Rerank a list of retrieved chunks using a BGE cross-encoder.

    Unlike the bi-encoder used during retrieval (BGE-M3), the cross-encoder
    reads the query and each chunk together in a single forward pass, producing
    a more precise relevance score at the cost of higher latency.

    Args:
        query:   The raw user query string.
        chunks:  List of chunk dicts as returned by retrieve() or retrieve_with_kg().
                 Each must have a "text" field.
        top_k:   If set, return only the top_k chunks after reranking.
                 If None, return all chunks reranked.

    Returns:
        The same list of chunk dicts with "score" replaced by the cross-encoder
        relevance score, sorted descending. Original retrieval scores are
        discarded — the reranker score is the authoritative relevance signal.
    """
    if not chunks:
        return chunks

    pairs = [[query, chunk["text"] or ""] for chunk in chunks]

    # normalize=True maps raw logits to [0, 1] — easier to interpret and
    # consistent across different queries and corpus sizes.
    # CrossEncoder.predict() always returns a numpy array, no single-float edge case
    raw_scores = reranker.predict(pairs)

    reranked = []
    for chunk, score in zip(chunks, raw_scores):
        reranked.append({**chunk, "score": float(score)})

    reranked.sort(key=lambda c: c["score"], reverse=True)

    if top_k is not None:
        reranked = reranked[:top_k]

    return reranked