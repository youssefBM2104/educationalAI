"""
Standalone test for the mindmap-generation pipeline.

Like the exam test, the mindmap graph expects `rag_chunks` + `kg_context` already in state
(retrieve is factored out). This loads the saved RAG-service fixture, maps it into
LearningMaterialsState, and invokes the compiled mindmap graph (mindmap -> export).

Run (from repo root, inside .venv):
    python -m backend.tests.test_mindmap_generation
    python -m backend.tests.test_mindmap_generation "Mindmap of thread synchronization" markmap
"""
import json
import sys
import logging
from pathlib import Path

from backend.agents.learning_materials.mindmaps_generation.mindmap_graph import get_mindmap_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

FIXTURE = Path(__file__).parent / "fixtures" / "rag_passive_waiting.json"
DEFAULT_QUERY = "Create a mindmap about thread synchronization"


def load_state(query: str, fmt: str) -> dict:
    rag = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {
        "user_id": "test-user",
        "course_id": rag.get("course_id", "test"),
        "query": query,
        "mindmap_format": fmt,          # "mermaid" | "markmap"
        "rag_chunks": rag["chunks"],
        "kg_context": rag["kg_context"],
    }


def _print_tree(node: dict, depth: int = 0):
    print("  " * depth + f"- {node.get('label', '?')}: {node.get('description', '')}")
    for child in node.get("children", []) or []:
        _print_tree(child, depth + 1)


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    fmt = sys.argv[2] if len(sys.argv) > 2 else "mermaid"
    state = load_state(query, fmt)

    print(f"\n=== QUERY ===\n{query}  (format={fmt})\n")
    result = get_mindmap_graph().invoke(state)

    tree = result.get("mindmap_tree") or {}
    print("\n=== MINDMAP TREE ===")
    root = tree.get("root")
    if root:
        _print_tree(root)
    else:
        print("(no tree)")

    print(f"\n=== OUTPUT ===\npath: {result.get('mindmap_output_path')}")
    raw = result.get("mindmap_output_bytes") or b""
    print(f"bytes: {len(raw)}")


if __name__ == "__main__":
    main()
