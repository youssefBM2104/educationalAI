import logging

from langgraph.graph import StateGraph, START, END

from backend.agents.state import LectureState
from backend.agents.slide_generation.composer_agent import composer_agent
from backend.agents.slide_generation.slide_planner_agent import slide_planner_agent
from backend.agents.slide_generation.verification_agent import verification_agent
from backend.agents.slide_generation.export import export_agent

logger = logging.getLogger(__name__)

_app = None


def init_node(state: LectureState) -> dict:
    return {
        "attempt": 0,
        "max_attempts": state.get("max_attempts", 3),
        "best_attempt": None,
    }


def retry_control_node(state: LectureState) -> dict:
    """Update best attempt and advance the counter before looping back to the Composer."""
    fb = state.get("verification_feedback") or {}
    score = fb.get("content_score", 0.0)
    best = state.get("best_attempt")
    update = {"attempt": state.get("attempt", 0) + 1}
    if best is None or score > best.get("score", -1.0):
        update["best_attempt"] = {
            "composer_output": state.get("composer_output"),
            "lecture_slides": state.get("lecture_slides"),
            "score": score,
        }
    logger.info("Retry control: attempt=%d score=%.2f", update["attempt"], score)
    return update


def route_after_verify(state: LectureState) -> str:
    return "export" if state.get("verification_passed") else "retry"


def route_after_retry(state: LectureState) -> str:
    if state.get("attempt", 0) >= state.get("max_attempts", 3):
        return "give_up"
    return "again"


def get_slide_graph():
    global _app
    if _app is not None:
        return _app

    g = StateGraph(LectureState)

    g.add_node("init", init_node)
    g.add_node("composer", composer_agent)
    g.add_node("planner", slide_planner_agent)
    g.add_node("verification", verification_agent)
    g.add_node("retry_control", retry_control_node)
    g.add_node("export", export_agent)

    g.add_edge(START, "init")
    g.add_edge("init", "composer")
    g.add_edge("composer", "planner")
    g.add_edge("planner", "verification")

    g.add_conditional_edges("verification", route_after_verify, {
        "export": "export",
        "retry": "retry_control",
    })
    g.add_conditional_edges("retry_control", route_after_retry, {
        "again": "composer",      # scoped retry — composer reads retry_scope
        "give_up": "export",      # export best attempt, labelled unverified
    })
    g.add_edge("export", END)

    _app = g.compile()
    return _app
