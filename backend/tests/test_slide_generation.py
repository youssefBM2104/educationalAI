"""
Standalone test for the slide-generation pipeline (new design).

Loads a saved RAG-service fixture into LectureState and runs the compiled slide graph end-to-end:
init -> composer -> planner -> verification -> (retry loop) -> export.

Run:
    python -m backend.tests.test_slide_generation
    python -m backend.tests.test_slide_generation "deadlock detection" pdf
    python -m backend.tests.test_slide_generation "thread sync" pptx path/to/template.pptx
"""
import json
import sys
import logging
from pathlib import Path

from backend.agents.slide_generation.slides_graph import get_slide_graph
from backend.eval.usage import UsageTracker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

FIXTURE = Path(__file__).parent / "fixtures" / "inflation.json"


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "Create lecture slides about inflation"
    fmt = sys.argv[2] if len(sys.argv) > 2 else "pptx"
    template = sys.argv[3] if len(sys.argv) > 3 else None   # path to a .pptx visual template

    rag = json.loads(FIXTURE.read_text(encoding="utf-8"))
    state = {
        "user_id": "test-user",
        "course_id": rag.get("course_id") or "test",
        "query": query,
        "rag_chunks": rag["chunks"],
        "kg_context": rag["kg_context"],
        "output_format": fmt,
        "max_attempts": 2,
    }
    if template:
        state["slide_template_pptx"] = template
        print(f"(using visual template: {template})")

    print(f"\n=== QUERY ===\n{query}  (format={fmt})\n")
    app = get_slide_graph()
    with UsageTracker() as tracker:
        result = app.invoke(state, config={"callbacks": [tracker]})

    composer = result.get("composer_output") or {}
    print(f"\n=== HEADINGS ({len(composer.get('headings', []))}) ===")
    for h in composer.get("headings", []):
        print(f"  [{h['heading_id']}] {h['title']}  (imgs: {h.get('image_refs')})")

    deck = result.get("lecture_slides") or {}
    print(f"\n=== SLIDES ({len(deck.get('slides', []))}) ===")
    for s in deck.get("slides", []):
        print(f"  [{s['slide_id']}<-{s['heading_id']}] {s['title']}  ({s.get('layout')})")
        for b in s.get("bullets", []):
            print(f"      • {b}")

    fb = result.get("verification_feedback") or {}
    print(f"\n=== VERIFICATION ===")
    print(f"  passed={result.get('verification_passed')}  content_score={fb.get('content_score')}")
    for i in fb.get("issues", []):
        print(f"  [{i['type']}] {i.get('heading_id') or i.get('slide_id') or ''}: {i['issue']}")

    print(f"\n=== EXPORT ===")
    print(f"  status={result.get('export_status')}")
    print(f"  file={result.get('lecture_output_path')}")

    tracker.report()


if __name__ == "__main__":
    main()
