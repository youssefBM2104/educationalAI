"""
Standalone test for the exam-generation pipeline.

The exam graph starts at `intake` (the `retrieve` node was factored out into the top-level
graph), so it expects `rag_chunks` + `kg_context` already in state. This script loads a saved
RAG-service output fixture, maps it into ExamState, and invokes the compiled exam graph
end-to-end (intake -> plan -> generator -> chunk_pool -> solver -> judge -> collect).

Run:
    python -m backend.tests.test_exam_generation
    python -m backend.tests.test_exam_generation "Create 3 essay questions about deadlock"
"""
import json
import sys
import logging
from pathlib import Path

from backend.agents.exam_generation.exam_graph import get_exam_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

FIXTURE = Path(__file__).parent / "fixtures" / "rag_passive_waiting.json"
DEFAULT_QUERY = "Create 5 multiple-choice questions about thread synchronization"


def load_state(query: str) -> dict:
    rag = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {
        "user_id": "test-user",
        "course_id": rag.get("course_id", "test"),
        "query": query,
        # retrieve_node normally maps RAG's `chunks` -> state `rag_chunks`; we do it here.
        "rag_chunks": rag["chunks"],
        "kg_context": rag["kg_context"],
    }


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    state = load_state(query)

    print(f"\n=== QUERY ===\n{query}\n")
    app = get_exam_graph()
    result = app.invoke(state)

    exam_set = result.get("exam_set") or []
    print(f"\n=== EXAM SET ({len(exam_set)} questions, "
          f"target={result.get('num_questions_target')}, "
          f"type={result.get('question_type')}) ===\n")

    for i, q in enumerate(exam_set, 1):
        print(f"--- Q{i} [{q.get('difficulty')}] ---")
        print(q.get("question"))
        if result.get("question_type") == "mcq":
            for key, choice in (q.get("choices") or {}).items():
                print(f"  {key}. {choice}")
            print(f"  correct: {q.get('correct_answer')}")
            print(f"  explanation: {q.get('explanation')}")
        else:
            print(f"  model answer: {q.get('model_answer')}")
            print(f"  marking scheme: {q.get('marking_scheme')}")
        print(f"  kg_path: {q.get('kg_path')}\n")


if __name__ == "__main__":
    main()
