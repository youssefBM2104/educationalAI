import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import LearningMaterialsState

logger = logging.getLogger(__name__)

llm = MODELS["llama31"]

# Target word counts per detail level — communicated to the LLM via the prompt
TARGET_WORDS: dict[str, str] = {
    "short":  "around 150 words — one short paragraph per concept, essential ideas only",
    "medium": "around 400 words — one developed paragraph per concept",
    "long":   "around 800 words — multiple paragraphs per concept with full explanations",
}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class SummarySection(BaseModel):
    concept: str = Field(description="The KG concept this section covers")
    text: str = Field(
        description="Fluent prose for this concept, written from the extracted ideas"
    )


class Summary(BaseModel):
    title: str = Field(description="Summary title, derived from the topic")
    introduction: str = Field(
        description="One opening paragraph framing the topic and its importance"
    )
    sections: list[SummarySection] = Field(
        description="One section per concept group, in the same order as the extracted ideas"
    )
    conclusion: str = Field(
        description="One closing paragraph restating the most important takeaways"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_extracted_ideas(extracted: dict, detail_level: str) -> str:
    """
    Format the ExtractedIdeas dict into a readable block for the Writer prompt.
    For short summaries, only essential ideas are passed to keep the context lean.
    """
    lines = [f"Topic: {extracted.get('topic', '')}\n"]

    for group in extracted.get("concept_groups", []):
        lines.append(f"## Concept: {group['concept']}")
        for idea in group.get("ideas", []):
            if detail_level == "short" and idea["rank"] != "essential":
                continue
            tag = "[essential]" if idea["rank"] == "essential" else "[detail]  "
            lines.append(f"  {tag} {idea['idea']}")
        lines.append("")

    return "\n".join(lines)


def _build_prompt(state: LearningMaterialsState) -> str:
    detail_level   = state.get("detail_level", "medium")
    target         = TARGET_WORDS.get(detail_level, TARGET_WORDS["medium"])
    extracted_text = _format_extracted_ideas(state["extracted_ideas"], detail_level)

    return f"""# Role

You are the **Writer Agent** in a multi-agent summary generation pipeline.
You receive a structured list of key ideas extracted from a course — NOT the raw source chunks.
Your job is to turn these ideas into a clear, fluent, well-structured summary.

# Task

Write a complete summary of the topic at the requested detail level.

## Rules

- Target length: {target}.
- Write one section per concept group, in the same order as the input.
- Each section must cover the ideas provided — do NOT add facts not present in the idea list.
- Write fluent prose with smooth transitions between sentences and sections.
- The introduction must frame the topic without repeating the title verbatim.
- The conclusion must restate the 2 or 3 most important points in a fresh way.
- Do NOT use bullet points — this is flowing prose, not a list.
- Do NOT invent examples or analogies absent from the idea list.

# Detail level

{detail_level.upper()} — {target}

# Extracted ideas (your only source material)

{extracted_text}
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def writer_agent(state: LearningMaterialsState) -> dict:
    detail_level = state.get("detail_level", "medium")

    structured_llm = llm.with_structured_output(Summary)

    result: Summary = structured_llm.invoke([
        SystemMessage(content=_build_prompt(state)),
        HumanMessage(content="Write the summary now."),
    ])

    summary = result.model_dump()

    total_words = (
        len(summary["introduction"].split())
        + sum(len(s["text"].split()) for s in summary["sections"])
        + len(summary["conclusion"].split())
    )

    logger.info(
        "Writer: title=%r detail_level=%s sections=%d approx_words=%d",
        summary["title"],
        detail_level,
        len(summary["sections"]),
        total_words,
    )

    return {"summary": summary}