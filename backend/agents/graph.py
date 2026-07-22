import logging

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from backend.agents.state import MainState
from backend.agents.retrieve import retrieve_node
from backend.agents.orchestrator_agent import orchestrator_agent, route_intent
from backend.agents.exam_generation.exam_graph import get_exam_graph
from backend.agents.slides_generation.slides_graph import get_slides_graph
from backend.agents.learning_materials.mindmaps_generation.mindmap_graph import get_mindmap_graph
from backend.agents.learning_materials.summaries_generation.summaries_graph import get_summaries_graph
from backend.agents.tutoring.tutoring_graph import get_tutoring_graph

logger = logging.getLogger(__name__)

_app = None


# --- Tutoring entry (first turn only) ---

def tutoring_entry(state: MainState, config: RunnableConfig) -> dict:
    """First turn of a tutoring session. The top graph dispatches here once;
    later turns bypass this graph and call get_tutoring_graph() directly with the
    same thread_id, so retrieve/orchestrator do not re-run mid-session."""
    return get_tutoring_graph().invoke(state, config)


# --- Top-level dispatcher graph ---

def get_graph():
    global _app
    if _app is not None:
        return _app

    g = StateGraph(MainState)

    g.add_node("retrieve", retrieve_node)
    g.add_node("orchestrator", orchestrator_agent)
    g.add_node("exam", get_exam_graph())
    g.add_node("slides", get_slides_graph())
    g.add_node("mindmap", get_mindmap_graph())
    g.add_node("summary", get_summaries_graph())
    g.add_node("tutoring", tutoring_entry)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "orchestrator")
    g.add_conditional_edges("orchestrator", route_intent, {
        "exam": "exam",
        "slides": "slides",
        "mindmap": "mindmap",
        "summary": "summary",
        "tutoring": "tutoring",
    })
    for node in ("exam", "slides", "mindmap", "summary", "tutoring"):
        g.add_edge(node, END)

    _app = g.compile()
    return _app
