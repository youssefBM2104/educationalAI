import logging

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import LectureState
from backend.agents.slide_generation._common import (
    all_concepts, available_images, format_chunks_for_prompt, format_relations,
)

logger = logging.getLogger(__name__)

llm = MODELS["gpt-5"]


class Heading(BaseModel):
    heading_id: str = Field(description="Stable id, e.g. 'h1', 'h2'")
    title: str
    source_concepts: list[str] = Field(description="Concept names (verbatim) this heading covers")
    content_points: list[str] = Field(description="Full teaching points, grounded in the chunks")
    image_refs: list[str] = Field(
        default_factory=list,
        description="image_id of genuinely illustrative figures only; [] if none. Never a logo/screenshot.",
    )


class ComposerResult(BaseModel):
    headings: list[Heading]


def _coverage_check(chunks: list | None, headings: list[dict]) -> dict:
    every = all_concepts(chunks)
    represented = {c for h in headings for c in (h.get("source_concepts") or [])}
    return {
        "all_concepts": every,
        "concepts_represented": [c for c in every if c in represented],
    }


def _base_prompt(state: LectureState) -> str:
    return f"""# Role
You are the **Content Composer** for a lecture-slide pipeline. Organize the retrieved material
into a clean heading structure with fully-written teaching content, and pick illustrative images.

# Knowledge-graph relations
_PART_OF / DEFINES suggest the same heading; PREREQUISITE suggests ordering._

{format_relations(state.get('kg_context'))}

# Source chunks (text + image descriptions — no raw images)

{format_chunks_for_prompt(state.get('rag_chunks'))}

# Available images (reference by image_id)

{available_images(state.get('rag_chunks')) or '(none)'}

# Rules
- Cluster concepts into headings using the relations; order by PREREQUISITE.
- Write real teaching content per heading, grounded strictly in the chunks. Do not invent facts.
- Attach an image_id ONLY if it genuinely illustrates the heading (judge from its description).
  Reject logos, screenshots, off-topic captures. A heading with no fitting image gets image_refs=[].
- NEVER replace a figure with descriptive bullets — reference it by image_id so it can be shown.
- Every concept that appears in the chunks must belong to some heading.

# Request
{state.get('query', '')}
"""


def _retry_prompt(state: LectureState) -> str:
    scope = (state.get("retry_scope") or {}).get("heading_ids", [])
    fb = state.get("verification_feedback") or {}
    return _base_prompt(state) + f"""
# Previous attempt failed verification — fix ONLY these headings
Regenerate content/image_refs for headings: {scope}
Keep their heading_id stable. A feedback issue with heading_id=null means CREATE a new heading
for an orphaned concept.

Issues to fix:
{[i for i in fb.get('issues', []) if i.get('type') == 'content']}
"""


def composer_agent(state: LectureState) -> dict:
    is_retry = bool((state.get("retry_scope") or {}).get("heading_ids") or
                    (state.get("verification_feedback") and state.get("composer_output")))
    prompt = _retry_prompt(state) if is_retry else _base_prompt(state)

    result = llm.with_structured_output(ComposerResult, method="function_calling").invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Produce the heading structure now."),
    ])
    new_headings = [h.model_dump() for h in result.headings]

    if is_retry and state.get("composer_output"):
        # Merge: replace only the regenerated headings, carry the rest through unchanged.
        scope = set((state.get("retry_scope") or {}).get("heading_ids", []))
        changed_ids = {h["heading_id"] for h in new_headings}
        kept = [h for h in state["composer_output"]["headings"]
                if h["heading_id"] not in scope and h["heading_id"] not in changed_ids]
        headings = kept + new_headings
    else:
        headings = new_headings

    composer_output = {"headings": headings, "coverage_check": _coverage_check(state.get("rag_chunks"), headings)}
    logger.info("Composer: %d headings (retry=%s)", len(headings), is_retry)
    return {"composer_output": composer_output}
