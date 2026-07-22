"""
Business logic for slide-deck (lecture) generation.

Same translator role as exam_service.py: turns a SlidesRequest into the
state dict expected by the existing planner -> content_generator ->
slide_builder -> export pipeline (agents/slides_generation/slides_graph.py).

Note: we only return the output file path, not the raw file bytes. The
Celery result backend serializes task results as JSON, which cannot
carry raw binary safely. If the caller needs the actual file, add a
dedicated download endpoint that reads `lecture_output_path` from disk.
"""

import logging

from backend.agents.retrieve import retrieve_node
from backend.agents.slides_generation.slides_graph import get_slides_graph
from backend.ai_server.schemas.slides import SlidesRequest

logger = logging.getLogger(__name__)


def run_slides_pipeline(request: SlidesRequest) -> dict:
    """Run the full slide-generation pipeline and return the export metadata."""
    state = {
        "user_id": request.user_id,
        "course_id": request.course_id,
        "query": request.query,
        "output_format": request.output_format,
    }

    logger.info("Slides pipeline: retrieving context for course_id=%s", request.course_id)
    state.update(retrieve_node(state))

    logger.info("Slides pipeline: invoking slides_graph")
    result = get_slides_graph().invoke(state)

    return {
        "lecture_output_path": result["lecture_output_path"],
        "course_title": result["lecture_plan"]["course_title"],
    }
