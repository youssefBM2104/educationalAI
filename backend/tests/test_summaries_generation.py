"""
Standalone test for the summaries-generation pipeline.

The summaries graph expects `rag_chunks` + `kg_context` already in state (retrieve is factored
out). This loads the saved RAG-service fixture, maps it into LearningMaterialsState, and invokes
the compiled summaries graph (extractor -> writer).

Run (from repo root, inside .venv):
    python -m backend.tests.test_summaries_generation
    python -m backend.tests.test_summaries_generation "Summarize thread synchronization" short
"""
import json
import sys
import logging
from pathlib import Path

from backend.agents.learning_materials.summaries_generation.summaries_graph import get_summaries_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

FIXTURE = Path(__file__).parent / "fixtures" / "rag_passive_waiting.json"
DEFAULT_QUERY = "Summarize thread synchronization"


def load_state(query: str, detail: str) -> dict:
    rag = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {
        "user_id": "test-user",
        "course_id": rag.get("course_id", "test"),
        "query": query,
        "detail_level": detail,          # "short" | "medium" | "long"
        "rag_chunks": rag["chunks"],
        "kg_context": rag["kg_context"],
    }


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    detail = sys.argv[2] if len(sys.argv) > 2 else "medium"
    state = load_state(query, detail)

    print(f"\n=== QUERY ===\n{query}  (detail_level={detail})\n")
    result = get_summaries_graph().invoke(state)

    ideas = result.get("extracted_ideas") or {}
    groups = ideas.get("concept_groups", []) or []
    total = sum(len(g.get("ideas", [])) for g in groups)
    print(f"=== EXTRACTED IDEAS (topic={ideas.get('topic', '?')}, "
          f"{len(groups)} concepts, {total} ideas) ===")
    for g in groups:
        print(f"  [{g.get('concept')}]")
        for idea in g.get("ideas", []) or []:
            print(f"    - ({idea.get('rank')}) {idea.get('idea')}")

    summary = result.get("summary") or {}
    print(f"\n=== SUMMARY: {summary.get('title', '(no title)')} ===\n")
    print(summary.get("introduction", ""))
    for sec in summary.get("sections", []) or []:
        print(f"\n## {sec.get('concept', '')}\n{sec.get('text', '')}")
    print(f"\n{summary.get('conclusion', '')}")


if __name__ == "__main__":
    main()
