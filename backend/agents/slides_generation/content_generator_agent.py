import logging

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import LectureState

logger = logging.getLogger(__name__)

llm = MODELS["gemma"]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class SectionContent(BaseModel):
    order: int = Field(description="Matches the section order from the lecture plan")
    section_title: str
    explanation: str = Field(
        description="2 to 4 paragraphs explaining the section, grounded in the source chunks"
    )
    key_points: list[str] = Field(
        description="3 to 5 bullets, each under 15 words, used directly as slide bullets"
    )
    example: str = Field(
        description="One concrete real-world or textbook example anchoring the concept"
    )
    speaker_notes: str = Field(
        description=(
            "Lecturer-facing notes: what to emphasise verbally, "
            "potential student questions, timing hints. Never displayed on the slide."
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_chunks(chunks: list | None) -> str:
    if not chunks:
        return "(no chunks available)"
    return "\n\n".join(
        f"[{c.get('chunk_id', '?')}] {c.get('text', '')}" for c in chunks
    )


def _filter_chunks_for_section(
    chunks: list | None,
    key_concepts: list[str],
) -> list:
    """
    Keep chunks whose text contains at least one key concept (case-insensitive).
    Falls back to all chunks if nothing matches, to avoid an empty context.
    """
    if not chunks:
        return []
    lower_concepts = [c.lower() for c in key_concepts]
    filtered = [
        c for c in chunks
        if any(concept in c.get("text", "").lower() for concept in lower_concepts)
    ]
    return filtered if filtered else chunks


def _build_prompt(section: dict, relevant_chunks: list) -> str:
    return f"""# Role

You are the **Content Generator** in a multi-agent educational content generation pipeline.

# Task

Write the detailed educational content for ONE section of a lecture.
Your output will be forwarded to the Slide Builder, which will format it into slides.

## Rules

- `explanation` must be 2 to 4 paragraphs, grounded STRICTLY in the source chunks below.
- `key_points` must be 3 to 5 bullets, each under 15 words — they will appear verbatim on slides.
- `example` must be concrete and directly related to the section topic.
- `speaker_notes` are for the lecturer only — NEVER displayed on the slide.
- Do NOT invent facts absent from the source chunks.

# Section to develop

Title: {section["title"]}
Key concepts: {", ".join(section["key_concepts"])}
Learning objective: {section["learning_objective"]}

# Relevant source chunks

{_format_chunks(relevant_chunks)}
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def content_generator_agent(state: LectureState) -> dict:
    plan = state["lecture_plan"]
    chunks = state.get("rag_chunks") or []

    structured_llm = llm.with_structured_output(SectionContent)

    lecture_content: list[dict] = []

    for section in plan["sections"]:
        relevant_chunks = _filter_chunks_for_section(chunks, section["key_concepts"])

        result: SectionContent = structured_llm.invoke([
            SystemMessage(content=_build_prompt(section, relevant_chunks)),
            HumanMessage(content="Generate the section content now."),
        ])

        content = result.model_dump()
        # Ensure order and title are consistent with the plan
        content["order"] = section["order"]
        content["section_title"] = section["title"]

        lecture_content.append(content)

        logger.info(
            "ContentGenerator: section=%d/%d title=%r key_points=%d",
            section["order"],
            len(plan["sections"]),
            section["title"],
            len(result.key_points),
        )

    return {"lecture_content": lecture_content}