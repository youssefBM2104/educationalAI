import logging

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import LectureState
from backend.agents.slides_generation._common import DEFAULT_TEMPLATE, format_chunks_for_prompt

logger = logging.getLogger(__name__)

llm = MODELS["gpt-5-mini"]
CONTENT_THRESHOLD = 0.8


class HeadingVerdict(BaseModel):
    heading_id: str
    grounded: bool = Field(description="content_points trace to the source chunks, no invented facts")
    issue: str = Field(default="", description="what is ungrounded, if not grounded")


class ImageNote(BaseModel):
    heading_id: str
    image_id: str
    issue: str = Field(description="a strongly-relevant image that never made it onto a slide")


class VerifyResult(BaseModel):
    heading_verdicts: list[HeadingVerdict]
    image_notes: list[ImageNote] = Field(default_factory=list)


def _prompt(state: LectureState) -> str:
    headings = (state.get("composer_output") or {}).get("headings", [])
    slides = (state.get("lecture_slides") or {}).get("slides", [])
    return f"""# Role
You verify a generated slide deck. Judge ONLY: (1) does each heading's content trace to the
source chunks (grounding)? (2) advisory: any strongly-relevant image left off every slide?

# Source chunks
{format_chunks_for_prompt(state.get('rag_chunks'))}

# Headings (content to check)
{[{k: h.get(k) for k in ('heading_id', 'title', 'source_concepts', 'content_points', 'image_refs')} for h in headings]}

# Slides built (to see which images were placed)
{[{k: s.get(k) for k in ('slide_id', 'heading_id', 'image_refs')} for s in slides]}

Return a grounding verdict per heading, and image_notes for relevant images not placed anywhere.
"""


def _template_notes(state: LectureState) -> list[dict]:
    """Deterministic, advisory-only: flag bullets that run over the template's word limit."""
    template = state.get("slide_template") or DEFAULT_TEMPLATE
    max_w = template.get("max_words_per_bullet", 16)
    notes = []
    for s in (state.get("lecture_slides") or {}).get("slides", []):
        for b in s.get("bullets") or []:
            if len(str(b).split()) > max_w:
                notes.append({"type": "template", "slide_id": s.get("slide_id"),
                              "issue": f"bullet exceeds {max_w} words: '{b[:40]}...'"})
                break
    return notes


def verification_agent(state: LectureState) -> dict:
    composer = state.get("composer_output") or {}
    headings = composer.get("headings", [])
    coverage = composer.get("coverage_check", {})

    result = llm.with_structured_output(VerifyResult, method="function_calling").invoke([
        SystemMessage(content=_prompt(state)),
        HumanMessage(content="Verify the deck now."),
    ])

    issues: list[dict] = []

    # (1) Grounding — content, gating
    grounded = 0
    for v in result.heading_verdicts:
        if v.grounded:
            grounded += 1
        else:
            issues.append({"type": "content", "heading_id": v.heading_id, "issue": v.issue or "ungrounded"})
    content_score = grounded / len(headings) if headings else 0.0

    # (2) Coverage gap — content, gating, deterministic. Orphaned concept -> heading_id=null.
    missing = set(coverage.get("all_concepts", [])) - set(coverage.get("concepts_represented", []))
    for concept in sorted(missing):
        issues.append({"type": "content", "heading_id": None,
                       "issue": f"concept not represented in any heading: {concept}"})

    # (3) Template density — advisory only
    issues.extend(_template_notes(state))

    # (4) Image relevance — advisory only
    for n in result.image_notes:
        issues.append({"type": "image", "heading_id": n.heading_id, "image_id": n.image_id, "issue": n.issue})

    passed = content_score >= CONTENT_THRESHOLD and not missing
    content_ids = [i.get("heading_id") for i in issues if i["type"] == "content"]

    logger.info("Verification: passed=%s content_score=%.2f coverage_gap=%d",
                passed, content_score, len(missing))
    return {
        "verification_passed": passed,
        "verification_feedback": {"content_score": content_score, "issues": issues},
        "retry_scope": {"heading_ids": content_ids},
    }
