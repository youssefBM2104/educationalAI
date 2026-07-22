import logging
import random

from backend.agents.state import ExamState

logger = logging.getLogger(__name__)

MAX_DISTRACTORS = 6


# --- Helpers ---

def _chunk_key(chunk: dict) -> tuple:
    return (chunk.get("document_id"), chunk.get("chunk_index"))


def _concept_ids(chunk: dict) -> set[str]:
    out = set()
    for c in chunk.get("covers_concepts", []) or []:
        cid = c.get("id") if isinstance(c, dict) else c
        if cid:
            out.add(str(cid).lower())
    return out


def _distractor_concepts(generated_question: dict, kg_path: list[str]) -> set[str]:
    path = {c.lower() for c in kg_path}
    concepts = set()
    for d in generated_question.get("distractor_paths", []) or []:
        for c in d.get("concept_path", []) or []:
            concepts.add(str(c).lower())
    return concepts - path


# --- Node ---

def build_chunk_pool(state: ExamState) -> dict:
    bundle = state.get("chunk_bundle", []) or []

    if state["question_type"] == "essay":
        return {"chunk_pool": list(bundle)}

    rag_chunks = state.get("rag_chunks", []) or []
    bundle_keys = {_chunk_key(c) for c in bundle}
    wrong_concepts = _distractor_concepts(state.get("generated_question", {}), state.get("kg_path", []))

    distractors = [
        c for c in rag_chunks
        if _chunk_key(c) not in bundle_keys and (_concept_ids(c) & wrong_concepts)
    ]

    if not distractors:
        distractors = [c for c in rag_chunks if _chunk_key(c) not in bundle_keys]

    if len(distractors) > MAX_DISTRACTORS:
        distractors = random.sample(distractors, MAX_DISTRACTORS)

    pool = list(bundle) + distractors
    random.shuffle(pool)

    logger.info(
        "Chunk pool: %d real + %d distractor = %d (shuffled, unlabeled)",
        len(bundle), len(distractors), len(pool),
    )
    return {"chunk_pool": pool}
