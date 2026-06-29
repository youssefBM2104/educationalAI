import logging

from langgraph.graph import StateGraph, START, END

from backend.agents.state import LearningMaterialsState
from backend.agents.learning_materials.mindmaps_generation.mindmap_agent import mindmap_agent
from backend.agents.learning_materials.mindmaps_generation.mindmap_export import mindmap_export

logger = logging.getLogger(__name__)

_app = None


# --- Graph (subgraph: assumes rag_chunks + kg_context already in state) ---

def get_mindmap_graph():
    global _app
    if _app is not None:
        return _app

    g = StateGraph(LearningMaterialsState)

    g.add_node("mindmap", mindmap_agent)
    g.add_node("export", mindmap_export)

    g.add_edge(START, "mindmap")
    g.add_edge("mindmap", "export")
    g.add_edge("export", END)

    _app = g.compile()
    return _app
