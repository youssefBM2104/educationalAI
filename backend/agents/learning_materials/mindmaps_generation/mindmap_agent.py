import logging

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import LearningMaterialsState

logger = logging.getLogger(__name__)

llm = MODELS["gemma"]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class MindmapNode(BaseModel):
    label: str = Field(description="Short concept label, under 6 words")
    description: str = Field(
        description="One sentence explaining this concept, grounded in the source chunks"
    )
    children: list["MindmapNode"] = Field(
        default_factory=list,
        description="Child concepts directly related to this node in the KG",
    )

MindmapNode.model_rebuild()  # required for self-referencing Pydantic models


class MindmapTree(BaseModel):
    root: MindmapNode = Field(
        description="The central concept of the mindmap, derived from the query and the KG root"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_kg(triples: list | None) -> str:
    if not triples:
        return "(no relations available)"
    return "\n".join(
        f"{t.get('source')} --[{t.get('relation')}]--> {t.get('target')}"
        for t in triples
    )


def _format_chunks(chunks: list | None) -> str:
    if not chunks:
        return "(no chunks available)"
    return "\n\n".join(
        f"[{c.get('chunk_id', '?')}] {c.get('text', '')}" for c in chunks
    )


def _format_kg_node_ids(triples: list | None) -> str:
    if not triples:
        return "(none)"
    ids: set[str] = set()
    for t in triples:
        for key in ("source", "target"):
            val = t.get(key)
            if val:
                ids.add(str(val))
    return ", ".join(sorted(ids))


def _build_prompt(state: LearningMaterialsState) -> str:
    return f"""# Role

You are the **Mindmap Agent** in a multi-agent educational content generation pipeline.

# Task

Build a hierarchical mindmap tree for the given topic.
The tree will be rendered as a Mermaid mindmap or an interactive markmap file.

## Rules

- The root node must be the central concept of the topic, derived from the query.
- Main branches (root's direct children) must correspond to top-level KG concepts
  directly connected to the root — do NOT invent branches absent from the KG.
- Each branch may have sub-nodes following KG relations one level deeper.
- Keep labels short: under 6 words per node.
- Each description must be one sentence grounded strictly in the source chunks.
- Aim for 3 to 6 main branches, each with 2 to 4 sub-nodes.
- Do NOT repeat the same concept at multiple levels of the tree.

# Topic

{state["query"]}

# Available KG concept ids

_Use only these as node labels — do not invent concepts._

{_format_kg_node_ids(state.get("kg_context"))}

# Knowledge graph relations

_Use these to determine parent-child relationships in the tree._

{_format_kg(state.get("kg_context"))}

# Source chunks

_Ground every node description in these chunks._

{_format_chunks(state.get("rag_chunks"))}
"""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def mindmap_agent(state: LearningMaterialsState) -> dict:
    structured_llm = llm.with_structured_output(MindmapTree)

    result: MindmapTree = structured_llm.invoke([
        SystemMessage(content=_build_prompt(state)),
        HumanMessage(content="Build the mindmap tree now."),
    ])

    tree = result.model_dump()

    def _count_nodes(node: dict) -> int:
        return 1 + sum(_count_nodes(c) for c in node.get("children", []))

    total_nodes = _count_nodes(tree["root"])

    logger.info(
        "MindmapAgent: root=%r branches=%d total_nodes=%d",
        tree["root"]["label"],
        len(tree["root"]["children"]),
        total_nodes,
    )

    return {"mindmap_tree": tree}