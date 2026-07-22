import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import LearningMaterialsState

logger = logging.getLogger(__name__)

llm = MODELS["llama31"]

# How many ideas to extract per concept depending on the requested detail level
IDEA_LIMITS: dict[str, int] = {
    "short":  2,
    "medium": 4,
    "long":   7,
}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class Idea(BaseModel):
    rank: Literal["essential", "detail"] = Field(
        description=(
            "'essential' — must appear in any summary regardless of length. "
            "'detail' — included only in medium or long summaries."
        )
    )
    idea: str = Field(
        description="One concise sentence capturing this idea, grounded in the source chunks"
    )
    source_chunk_id: str = Field(
        description="source chunk label as 'document_id#chunk_index'"
    )


class ConceptIdeas(BaseModel):
    concept: str = Field(description="KG concept id this group of ideas belongs to")
    ideas: list[Idea] = Field(description="Extracted ideas for this concept, ranked by importance")


class ExtractedIdeas(BaseModel):
    topic: str = Field(description="The overall topic derived from the query")
    concept_groups: list[ConceptIdeas] = Field(
        description="One group per major KG concept, ordered simple to complex"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_chunks(chunks: list | None) -> str:
    if not chunks:
        return "(no chunks available)"
    return "\n\n".join(
        f"[{c.get('document_id')}#{c.get('chunk_index')}] {c.get('text', '')}" for c in chunks
    )


def _format_kg_node_ids(kg: dict | None) -> str:
    kg = kg or {}
    concepts = kg.get("concepts")
    if not concepts:
        concepts = set()
        for r in kg.get("relations", []):
            for key in ("from", "to"):
                if r.get(key):
                    concepts.add(str(r[key]))
    if not concepts:
        return "(none)"
    return ", ".join(sorted(str(c) for c in concepts))


def _build_prompt(state: LearningMaterialsState, idea_limit: int) -> str:
    return f"""# Role

You are the **Extractor Agent** in a multi-agent summary generation pipeline.
Your output will be passed to a Writer Agent that will turn it into fluent prose.
You are NOT writing the summary — you are extracting and structuring the raw material for it.

# Task

Read all the source chunks below and extract the key ideas, grouped by concept.

## Rules

- Group ideas by the KG concept they belong to — one `ConceptIdeas` entry per major concept.
- Order the concept groups from simple to complex (same logic as a lecture plan).
- For each concept, extract at most {idea_limit} ideas.
- Rank each idea as "essential" or "detail":
    - "essential" → must appear in any summary regardless of length
    - "detail"    → included only in medium or long summaries
- Each idea must be a single concise sentence grounded in the source chunks.
- Always fill `source_chunk_id` with the [document_id#chunk_index] label of the source chunk.
- Do NOT invent ideas absent from the chunks.
- Do NOT write prose or transitions — that is the Writer's job.

# Topic

{state["query"]}

# Major KG concepts available

_Use only these as concept group labels._

{_format_kg_node_ids(state.get("kg_context"))}

# Source chunks

{_format_chunks(state.get("rag_chunks"))}
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def extractor_agent(state: LearningMaterialsState) -> dict:
    detail_level = state.get("detail_level", "medium")
    idea_limit   = IDEA_LIMITS.get(detail_level, IDEA_LIMITS["medium"])

    structured_llm = llm.with_structured_output(ExtractedIdeas)

    result: ExtractedIdeas = structured_llm.invoke([
        SystemMessage(content=_build_prompt(state, idea_limit)),
        HumanMessage(content="Extract the key ideas now."),
    ])

    extracted = result.model_dump()

    total_ideas = sum(len(g["ideas"]) for g in extracted["concept_groups"])
    essential   = sum(
        1 for g in extracted["concept_groups"]
        for i in g["ideas"] if i["rank"] == "essential"
    )

    logger.info(
        "Extractor: topic=%r detail_level=%s groups=%d total_ideas=%d essential=%d",
        extracted["topic"],
        detail_level,
        len(extracted["concept_groups"]),
        total_ideas,
        essential,
    )

    return {"extracted_ideas": extracted}