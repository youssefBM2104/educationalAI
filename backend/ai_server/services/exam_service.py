"""
Business logic for exam generation.

This is the "translator" between the HTTP world (a plain ExamRequest)
and the LangGraph world (a state dict). It does NOT contain any agent
logic itself — it just wires together two pieces that already exist:

1. retrieve_node: fetches the relevant course chunks + KG context for
   the query (normally run by the top-level graph in agents/graph.py,
   but exam_graph does not run it on its own, so we call it explicitly).
2. get_exam_graph(): the compiled intake -> plan -> generator -> solver
   -> judge -> collect loop that produces the validated question set.
"""

import logging

from backend.agents.retrieve import retrieve_node
from backend.agents.exam_generation.exam_graph import get_exam_graph
from backend.ai_server.schemas.exam import ExamRequest

logger = logging.getLogger(__name__)


def run_exam_pipeline(request: ExamRequest) -> list[dict]:
    """Run the full exam-generation pipeline and return the final exam set."""
    state = {
        "user_id": request.user_id,
        "course_id": request.course_id,
        "query": request.query,
    }

    logger.info("Exam pipeline: retrieving context for course_id=%s", request.course_id)
    state.update(retrieve_node(state))

    logger.info("Exam pipeline: invoking exam_graph")
    result = get_exam_graph().invoke(state)

    return result["exam_set"]
