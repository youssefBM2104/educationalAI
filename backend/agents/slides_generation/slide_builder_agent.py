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

class Slide(BaseModel):
    slide_number: int
    type: str = Field(description="One of: title | content | summary")
    title: str = Field(description="Slide title, under 8 words")
    bullets: list[str] = Field(
        description=(
            "List of bullet points. Empty list for title slide. "
            "Each bullet under 12 words."
        )
    )
    visual_hint: str = Field(
        description=(
            "Optional suggestion for a diagram or image the lecturer could add. "
            "Empty string if not applicable. Never generated — text description only."
        )
    )
    speaker_notes: str = Field(
        description="Carried over from Content Generator. Never displayed on the slide."
    )


class SlideDeck(BaseModel):
    slides: list[Slide] = Field(
        description="Full ordered slide deck, title slide first, summary slide last"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_lecture_content(lecture_content: list) -> str:
    blocks = []
    for section in lecture_content:
        key_points = "\n".join(f"  - {p}" for p in section.get("key_points", []))
        blocks.append(
            f"## Section {section['order']}: {section['section_title']}\n"
            f"Key points:\n{key_points}\n"
            f"Example: {section.get('example', '')}\n"
            f"Speaker notes: {section.get('speaker_notes', '')}"
        )
    return "\n\n".join(blocks)


def _build_prompt(state: LectureState) -> str:
    course_title = state["lecture_plan"]["course_title"]
    content_text = _format_lecture_content(state["lecture_content"])

    return f"""# Role

You are the **Slide Builder** in a multi-agent educational content generation pipeline.

# Task

Format the provided lecture content into a complete, clean slide deck.

## Rules

- Slide 1 MUST be a title slide (type: "title") with the course title and an empty bullets list.
- One content slide (type: "content") per section — do not merge or split sections.
- The last slide MUST be a summary slide (type: "summary") collecting the most important
  key_point from each section (one bullet per section, max 12 words each).
- Bullets must be copied or lightly paraphrased from `key_points` — under 12 words each.
- `visual_hint` is optional: suggest a diagram or image the lecturer could add.
  Leave it as an empty string if nothing relevant comes to mind.
- `speaker_notes` must be carried over verbatim from the section's speaker notes.
- `slide_number` must be sequential starting at 1.
- Do NOT invent content not present in the provided sections.

# Course title

{course_title}

# Lecture content

{content_text}
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def slide_builder_agent(state: LectureState) -> dict:
    structured_llm = llm.with_structured_output(SlideDeck)

    result: SlideDeck = structured_llm.invoke([
        SystemMessage(content=_build_prompt(state)),
        HumanMessage(content="Build the slide deck now."),
    ])

    deck = result.model_dump()

    logger.info(
        "SlideBuilder: total_slides=%d (title + %d content + summary)",
        len(deck["slides"]),
        len(deck["slides"]) - 2,
    )

    return {"lecture_slides": deck}