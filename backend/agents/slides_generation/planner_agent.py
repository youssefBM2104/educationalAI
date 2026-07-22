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

class Section(BaseModel):
    order: int = Field(description="Position of this section in the lecture, starting at 1")
    title: str = Field(description="Concise section title")
    key_concepts: list[str] = Field(description="Main concepts covered in this section")
    kg_nodes_used: list[str] = Field(
        description="Concept ids from kg_context that map to this section"
    )
    learning_objective: str = Field(
        description="One sentence starting with 'Understand…' or 'Learn…'"
    )


class LecturePlan(BaseModel):
    course_title: str = Field(description="Concise title for the full lecture")
    estimated_slides: int = Field(
        description="Total number of slides recommended (title slide included)"
    )
    sections: list[Section] = Field(
        description="Ordered list of sections — simple to complex"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_kg(kg: dict | None) -> str:
    relations = (kg or {}).get("relations", [])
    if not relations:
        return "(no relations available)"
    return "\n".join(
        f"{r.get('from')} --[{r.get('type')}]--> {r.get('to')}"
        for r in relations
    )


def _format_chunks(chunks: list | None) -> str:
    if not chunks:
        return "(no chunks available)"
    return "\n\n".join(
        f"[{c.get('document_id')}#{c.get('chunk_index')}] {c.get('text', '')}" for c in chunks
    )


def _format_kg_node_ids(kg: dict | None) -> str:
    """Return a flat deduplicated list of concept ids visible in the KG."""
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


def _build_prompt(state: LectureState) -> str:
    return f"""# Role

You are the **Lecture Planner** in a multi-agent educational content generation pipeline.

# Task

Design a clear, pedagogically ordered course plan for a lecture on the given topic.
The plan will be handed to a Content Generator that will develop each section in detail.

## Rules

- Sections MUST follow a logical order: simple concepts first, complex ones last.
- Each section MUST map at least one concept id from the available KG nodes.
- `learning_objective` must be a single sentence starting with "Understand…" or "Learn…".
- `estimated_slides` = number of content slides + 1 title slide + 1 summary slide.
- Do NOT invent concepts absent from the chunks or the KG.
- Aim for 3 to 7 sections depending on topic breadth.

# Topic / Learning objective

{state["query"]}

# Available knowledge graph nodes

_Only use concept ids from this list in `kg_nodes_used`._

{_format_kg_node_ids(state.get("kg_context"))}

# Knowledge graph relations

{_format_kg(state.get("kg_context"))}

# Source chunks

_Ground every concept in these chunks._

{_format_chunks(state.get("rag_chunks"))}
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def planner_agent(state: LectureState) -> dict:
    structured_llm = llm.with_structured_output(LecturePlan)

    result: LecturePlan = structured_llm.invoke([
        SystemMessage(content=_build_prompt(state)),
        HumanMessage(content="Generate the lecture plan now."),
    ])

    plan = result.model_dump()

    logger.info(
        "Planner: title=%r sections=%d estimated_slides=%d",
        plan["course_title"],
        len(plan["sections"]),
        plan["estimated_slides"],
    )

    return {"lecture_plan": plan}