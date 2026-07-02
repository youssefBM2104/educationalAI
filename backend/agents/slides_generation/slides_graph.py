import logging

from langgraph.graph import StateGraph, START, END

from backend.agents.state import LectureState
from backend.agents.slides_generation.planner_agent import planner_agent
from backend.agents.slides_generation.content_generator_agent import content_generator_agent
from backend.agents.slides_generation.slide_builder_agent import slide_builder_agent
from backend.agents.slides_generation.export import export

logger = logging.getLogger(__name__)

_app = None


# --- Slides Graph ---

def get_slides_graph():
    global _app
    if _app is not None:
        return _app

    g = StateGraph(LectureState)

    g.add_node("planner", planner_agent)
    g.add_node("content_generator", content_generator_agent)
    g.add_node("slide_builder", slide_builder_agent)
    g.add_node("export", export)

    g.add_edge(START, "planner")
    g.add_edge("planner", "content_generator")
    g.add_edge("content_generator", "slide_builder")
    g.add_edge("slide_builder", "export")
    g.add_edge("export", END)

    _app = g.compile()
    return _app
