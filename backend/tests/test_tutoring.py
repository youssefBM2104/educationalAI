"""
Standalone test for the tutoring pipeline (multi-turn Socratic scaffolding).

Unlike the one-shot pipelines, the tutoring graph is TURN-BASED: each `invoke` handles exactly
one student turn, then ends. State persists across turns via MemorySaver + thread_id, so we
re-invoke the same compiled graph with the same config to continue the session.

Flow per turn:
    turn 1        : {query, rag_chunks}  -> tutor asks a sub-question (or answers directly)
    turn 2..N     : {student_answer}     -> evaluate -> scaffold/climb -> next question
    final turn    : phase == "done"      -> full answer in final_output

Run (from repo root, inside .venv):
    # interactive: you type the student's answers
    python -m backend.tests.test_tutoring

    # scripted: answers passed as extra args (handy for a quick smoke test)
    python -m backend.tests.test_tutoring "Why does passive waiting avoid busy CPU?" "I dont know" "It blocks the thread"
"""
import json
import sys
import logging
from pathlib import Path

from backend.agents.tutoring.tutoring_graph import get_tutoring_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

FIXTURE = Path(__file__).parent / "fixtures" / "rag_passive_waiting.json"
DEFAULT_QUERY = "Why does passive waiting avoid wasting CPU compared to active waiting?"
THREAD_ID = "tutoring-test-session"
MAX_TURNS = 12          # safety cap: the scaffold loop should converge well before this


def initial_state(query: str) -> dict:
    rag = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {
        "user_id": "test-student",
        "course_id": rag.get("course_id", "test"),
        "query": query,
        # tutoring grounds only in the chunks (no KG, per the instruction spec)
        "rag_chunks": rag["chunks"],
    }


def _debug(state: dict) -> str:
    """One-line view of where the scaffold controller currently is."""
    return (
        f"bloom_level={state.get('bloom_level')} "
        f"phase={state.get('phase')} "
        f"current_level={state.get('current_level')} "
        f"eval={state.get('eval_result')} "
        f"fail_streak={state.get('fail_streak')}"
    )


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    scripted = sys.argv[2:]          # optional canned student answers
    answers = iter(scripted)

    app = get_tutoring_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    print(f"\n=== STUDENT QUESTION ===\n{query}\n")

    # --- Turn 1: classify + first sub-question (no student_answer yet) ---
    state = app.invoke(initial_state(query), config)
    turn = 1
    print(f"--- turn {turn} | {_debug(state)} ---")
    print(f"TUTOR: {state.get('tutor_message')}\n")

    # --- Turns 2..N: feed the student's answer back into the same thread ---
    while state.get("phase") != "done" and turn < MAX_TURNS:
        if scripted:
            try:
                answer = next(answers)
            except StopIteration:
                print("(scripted answers exhausted — stopping)")
                break
            print(f"STUDENT: {answer}")
        else:
            answer = input("STUDENT: ").strip()
            if not answer:
                print("(empty answer — stopping)")
                break

        state = app.invoke({"student_answer": answer}, config)
        turn += 1
        print(f"\n--- turn {turn} | {_debug(state)} ---")
        print(f"TUTOR: {state.get('tutor_message')}\n")

    if state.get("phase") == "done":
        print("=== SESSION COMPLETE ===")
        print(state.get("final_output"))
    else:
        print(f"=== STOPPED (phase={state.get('phase')}, turns={turn}) ===")


if __name__ == "__main__":
    main()
