import logging

from langgraph.graph import StateGraph, START, END

from backend.agents.state import LearningMaterialsState
from backend.agents.learning_materials.summaries_generation.extractor_agent import extractor_agent
from backend.agents.learning_materials.summaries_generation.writer_agent import writer_agent

logger = logging.getLogger(__name__)

_app = None


# --- Graph (subgraph: assumes rag_chunks + kg_context already in state) ---

def get_summaries_graph():
    global _app
    if _app is not None:
        return _app

    g = StateGraph(LearningMaterialsState)

    g.add_node("extractor", extractor_agent)
    g.add_node("writer", writer_agent)

    g.add_edge(START, "extractor")
    g.add_edge("extractor", "writer")
    g.add_edge("writer", END)

    _app = g.compile()
    return _app
