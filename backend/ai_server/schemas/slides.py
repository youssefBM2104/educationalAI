"""
Request schema for the slides-generation endpoint.

Same idea as exam.py: `query` is the free-text topic/learning objective
that the planner_agent turns into a lecture plan. `output_format` picks
between a PowerPoint deck (.pptx) or a printable handout (.pdf), matching
the two builders already implemented in agents/slides_generation/export.py.
"""

from typing import Literal

from pydantic import BaseModel


class SlidesRequest(BaseModel):
    user_id: str
    course_id: str
    query: str
    output_format: Literal["pptx", "pdf"] = "pptx"
