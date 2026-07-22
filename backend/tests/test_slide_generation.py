"""
Standalone test for the slide-generation pipeline.

The lecture graph expects `rag_chunks` and `kg_context` already in state
(the RAG retrieval step is handled by the top-level orchestrator graph).
This script loads the saved RAG fixture, maps it into LectureState,
and runs the full pipeline:

    planner → content_generator → slide_builder → export

Two output files will be written under  outputs/lectures/ :
    <course_title>.pptx   (default)
    <course_title>.pdf    (if --format pdf is passed)

Run:
    python -m backend.tests.test_slide_generation
    python -m backend.tests.test_slide_generation "Explain passive waiting and mutexes" --format pdf
    python -m backend.tests.test_slide_generation "Explain passive waiting and mutexes" --format pptx
"""

import json
import sys
import logging
import argparse
from pathlib import Path

from backend.agents.slides_generation.slides_graph import get_slides_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixture path
# ---------------------------------------------------------------------------
FIXTURE = Path(__file__).parent / "fixtures" / "rag_passive_waiting.json"

DEFAULT_QUERY = (
    "Generate a lecture on passive waiting, mutexes, and deadlocks "
    "in thread synchronization"
)


# ---------------------------------------------------------------------------
# State loader
# ---------------------------------------------------------------------------

def load_state(query: str, output_format: str) -> dict:
    if not FIXTURE.exists():
        raise FileNotFoundError(
            f"Fixture not found at {FIXTURE}.\n"
            "Make sure rag_passive_waiting.json is in backend/tests/fixtures/"
        )

    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))

    return {
        "user_id":       "test-user",
        "course_id":     raw.get("course_id", "test"),
        "query":         query,
        # RAG retrieval normally maps raw["chunks"] → state["rag_chunks"]
        "rag_chunks":    raw["chunks"],
        # Convert fixture's relation format to the triples format agents expect
        "kg_context":    raw["kg_context"],
        "output_format": output_format,
    }


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def _print_plan(plan: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  LECTURE PLAN")
    print(f"{'='*60}")
    print(f"  Title            : {plan.get('course_title')}")
    print(f"  Estimated slides : {plan.get('estimated_slides')}")
    print(f"  Sections         : {len(plan.get('sections', []))}")
    for s in plan.get("sections", []):
        print(f"\n  [{s['order']}] {s['title']}")
        print(f"       objective : {s.get('learning_objective')}")
        print(f"       concepts  : {', '.join(s.get('key_concepts', []))}")


def _print_content(content: list) -> None:
    print(f"\n{'='*60}")
    print(f"  CONTENT GENERATOR OUTPUT")
    print(f"{'='*60}")
    for section in content:
        print(f"\n  Section {section['order']} — {section['section_title']}")
        print(f"  Key points:")
        for kp in section.get("key_points", []):
            print(f"    • {kp}")
        print(f"  Example : {section.get('example', '')[:120]}...")
        speaker = section.get("speaker_notes", "")
        if speaker:
            print(f"  Speaker notes : {speaker[:100]}...")


def _print_slides(deck: dict) -> None:
    
    slides = deck.get("slides", [])
    print(f"\n{'='*60}")
    print(f"  SLIDE DECK  ({len(slides)} slides)")
    print(f"{'='*60}")
    for slide in slides:
        stype = slide.get("type", "content").upper()
        print(f"\n  [{slide['slide_number']}] [{stype}] {slide['title']}")
        for b in slide.get("bullets", []):
            print(f"    • {b}")
        hint = slide.get("visual_hint", "")
        if hint:
            print(f"    💡 {hint}")


def _print_export(result: dict) -> None:
    path  = result.get("lecture_output_path", "")
    size  = len(result.get("lecture_output_bytes", b""))
    print(f"\n{'='*60}")
    print(f"  EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  File path : {path}")
    print(f"  File size : {size:,} bytes")
    if path:
        abs_path = Path(path).resolve()
        print(f"  Open with : {abs_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test the slide generation pipeline")
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Topic for the lecture (default: passive waiting & mutexes)",
    )
    parser.add_argument(
        "--format",
        choices=["pptx", "pdf"],
        default="pptx",
        help="Output format (default: pptx)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  QUERY  : {args.query}")
    print(f"  FORMAT : {args.format}")
    print(f"{'='*60}\n")

    # --- Load state ---
    state = load_state(args.query, args.format)
    logger.info("State loaded — %d chunks, %d KG triples",
                len(state["rag_chunks"]), len(state["kg_context"]))

    # --- Run graph ---
    app    = get_slides_graph()
    result = app.invoke(state)

    # --- Print results at each stage ---
    if result.get("lecture_plan"):
        _print_plan(result["lecture_plan"])

    if result.get("lecture_content"):
        _print_content(result["lecture_content"])

    if result.get("lecture_slides"):
        _print_slides(result["lecture_slides"])

    if result.get("lecture_output_path"):
        _print_export(result)
    else:
        print("\n⚠️  No output file found in result — export may have failed.")
        logger.error("Missing lecture_output_path in final state: %s", list(result.keys()))


if __name__ == "__main__":
    main()