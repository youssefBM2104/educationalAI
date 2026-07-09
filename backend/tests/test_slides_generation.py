"""
Standalone test for the slides-generation pipeline.

The slides graph expects `rag_chunks` + `kg_context` already in state (retrieve is factored
out). This loads the saved RAG-service fixture, maps it into LectureState, and invokes the
compiled slides graph (planner -> content_generator -> slide_builder -> export).

Run (from repo root, inside .venv):
    python -m backend.tests.test_slides_generation
    python -m backend.tests.test_slides_generation "Lecture on deadlock" pdf
"""
import json
import sys
import logging
from pathlib import Path

from backend.agents.slides_generation.slides_graph import get_slides_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

FIXTURE = Path(__file__).parent / "fixtures" / "rag_passive_waiting.json"
DEFAULT_QUERY = "Create lecture slides about thread synchronization"


def load_state(query: str, fmt: str) -> dict:
    rag = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {
        "user_id": "test-user",
        "course_id": rag.get("course_id", "test"),
        "query": query,
        "output_format": fmt,           # "pptx" | "pdf"
        "rag_chunks": rag["chunks"],
        "kg_context": rag["kg_context"],
    }


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    fmt = sys.argv[2] if len(sys.argv) > 2 else "pptx"
    state = load_state(query, fmt)

    print(f"\n=== QUERY ===\n{query}  (format={fmt})\n")
    result = get_slides_graph().invoke(state)

    plan = result.get("lecture_plan") or {}
    print(f"=== PLAN: {plan.get('course_title', '(no title)')} "
          f"(~{plan.get('estimated_slides', '?')} slides) ===")
    for s in plan.get("sections", []) or []:
        print(f"  - {s if isinstance(s, str) else s.get('section_title', s)}")

    deck = result.get("lecture_slides") or {}
    slides = deck.get("slides", []) or []
    print(f"\n=== SLIDES ({len(slides)}) ===")
    for sl in slides:
        print(f"\n[{sl.get('slide_number', '?')}] ({sl.get('type', '')}) {sl.get('title', '')}")
        for b in sl.get("bullets", []) or []:
            print(f"    • {b}")
        if sl.get("visual_hint"):
            print(f"    (visual: {sl.get('visual_hint')})")

    print(f"\n=== OUTPUT ===\npath: {result.get('lecture_output_path')}")
    raw = result.get("lecture_output_bytes") or b""
    print(f"bytes: {len(raw)}")


if __name__ == "__main__":
    main()
