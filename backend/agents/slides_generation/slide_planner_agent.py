import logging

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import LectureState
from backend.agents.slides_generation._common import DEFAULT_TEMPLATE

logger = logging.getLogger(__name__)

llm = MODELS["gpt-5-mini"]   # layout mapping is mechanical; the heavy reasoning is in the Composer


class Slide(BaseModel):
    slide_id: str = Field(description="Stable id, e.g. 's1'")
    heading_id: str = Field(description="The heading this slide was built from")
    layout: str = Field(description="The template layout id whose purpose fits this content")
    title: str
    bullets: list[str] = Field(
        default_factory=list, description="Only for bullet layouts; [] for stat/card/image layouts"
    )
    slots: dict = Field(
        default_factory=dict,
        description="Layout-specific content for non-bullet layouts, keyed by the layout's slot "
                    "names, e.g. {'cards': [{'label','text'}, ...]} or {'stat': '87%', 'explanation': '...'}",
    )
    image_refs: list[str] = Field(
        default_factory=list, description="image_id(s) carried from the heading; [] if none"
    )


class PlannerResult(BaseModel):
    slides: list[Slide]


def _prompt(headings: list[dict], template: dict) -> str:
    return f"""# Role
You are the **Slide Planner**. Lay composed headings out onto slides by choosing, for each, the
template layout whose PURPOSE fits the content — then fill that layout's slots.

# Template — layout library
Each layout lists its purpose and slots. Match the content SHAPE to the purpose:
three parallel items -> three_card; a single number -> stat; a figure that is the point ->
image_caption; a concept with a supporting figure -> two_column_image; otherwise title_bullets.

{template}

# Headings to lay out
{[{k: h.get(k) for k in ('heading_id', 'title', 'content_points', 'image_refs')} for h in headings]}

# Rules
- Set `layout` to the chosen layout id. Fill `bullets` for bullet layouts; put non-bullet content
  in `slots` keyed by that layout's slot names (e.g. slots.cards, slots.stat, slots.caption).
- Respect max_words_per_bullet. If a heading holds more than one slide's worth, SPLIT it across
  several slides sharing the SAME heading_id — never truncate content to fit.
- Carry the heading's image_refs onto the slide(s). A slide shows an image only if its heading has
  a real image_id; no placeholder visuals, never re-describe an image as text.
- Give every slide a unique slide_id.
"""


def _plan(headings: list[dict], template: dict) -> list[dict]:
    result = llm.with_structured_output(PlannerResult, method="function_calling").invoke([
        SystemMessage(content=_prompt(headings, template)),
        HumanMessage(content="Lay out the slides now."),
    ])
    return [s.model_dump() for s in result.slides]


def slide_planner_agent(state: LectureState) -> dict:
    template = state.get("slide_template") or DEFAULT_TEMPLATE
    headings = (state.get("composer_output") or {}).get("headings", [])
    scope = set((state.get("retry_scope") or {}).get("heading_ids", []))

    if scope and state.get("lecture_slides"):
        # Scoped retry: rebuild slides only for the flagged headings, keep the rest.
        to_rebuild = [h for h in headings if h["heading_id"] in scope]
        rebuilt = _plan(to_rebuild, template) if to_rebuild else []
        kept = [s for s in state["lecture_slides"]["slides"] if s["heading_id"] not in scope]
        slides = kept + rebuilt
    else:
        slides = _plan(headings, template)

    logger.info("Slide Planner: %d slides (scoped=%s)", len(slides), bool(scope))
    return {"lecture_slides": {"slides": slides}}
