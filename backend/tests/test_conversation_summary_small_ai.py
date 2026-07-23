"""
Standalone test for the conversation-summary task, handled locally by the
fine-tuned small AI (MODEL["small_ai"] in local_ai/local_model.py).

Calls local_ai.tasks.conversation_summary.summarize_conversation(...) directly
— no agents/ pipeline, no FastAPI app involved.

Run from the repo root, same convention as the other scripts in this folder:

    python -m backend.tests.test_conversation_summary_small_ai

A short fixed tutoring transcript is used by default. Pass a JSON file path
as the only arg to summarize a different transcript instead
(list of {"role": "user"|"assistant", "content": str} objects).
"""
import sys
import json
import logging

from backend.local_ai.tasks.conversation_summary import summarize_conversation

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DEFAULT_TRANSCRIPT = [
    {"role": "user", "content": "What is the difference between a mutex and a semaphore?"},
    {"role": "assistant", "content": "A mutex allows only one thread to access a resource "
                                      "at a time and must be unlocked by the thread that "
                                      "locked it. A semaphore maintains a count and can allow "
                                      "several threads through; any thread can signal it."},
    {"role": "user", "content": "So a mutex is basically a semaphore with a max count of 1?"},
    {"role": "assistant", "content": "That's a common way to see it, yes — a binary semaphore "
                                      "is similar, but a true mutex also enforces ownership: "
                                      "only the locking thread can unlock it."},
    {"role": "user", "content": "What about deadlocks then, how do those happen with mutexes?"},
    {"role": "assistant", "content": "A deadlock happens when two or more threads each hold a "
                                      "lock the other needs, and neither can proceed. For "
                                      "example, thread 1 locks A then waits for B, while thread "
                                      "2 locks B then waits for A."},
]


def main():
    if len(sys.argv) > 1:
        transcript = json.loads(open(sys.argv[1], encoding="utf-8").read())
    else:
        transcript = DEFAULT_TRANSCRIPT

    print("=== TRANSCRIPT ===")
    for m in transcript:
        role = "Student" if m["role"] == "user" else "Tutor"
        print(f"{role}: {m['content']}")

    result = summarize_conversation(messages=transcript)

    print("\n=== RECAP ===")
    print(result.get("recap", ""))

    print("\n=== TOPICS COVERED ===")
    for t in result.get("topics_covered", []):
        print(f"  - {t}")

    print("\n=== OPEN QUESTIONS ===")
    for q in result.get("open_questions", []):
        print(f"  - {q}")

    if not result.get("topics_covered") and not result.get("open_questions"):
        print("\n(NOTE: empty topics/open_questions usually means the model didn't follow the "
              "RECAP:/TOPICS COVERED:/OPEN QUESTIONS: format exactly — check result['raw'] below)")
        print("\n=== RAW MODEL OUTPUT ===")
        print(result.get("raw", ""))


if __name__ == "__main__":
    main()
