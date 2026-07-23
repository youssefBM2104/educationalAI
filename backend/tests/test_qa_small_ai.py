"""
Standalone test for the QA task, handled locally by the fine-tuned small AI
(MODEL["small_ai"] in local_ai/local_model.py).

Does NOT go through the agents/ pipeline or the FastAPI app — it calls
local_ai.tasks.qa.answer_question(...) directly, so it also doubles as a
minimal smoke test that the fine-tuned adapter loads correctly on this
machine. Watch the log line printed by local_model.py at import time:
  - "Loading fine-tuned adapter from ..." -> using edu-qwen-v1, what we want to test
  - "Fine-tuned adapter not found ... Falling back to raw Qwen2.5 via Ollama"
    -> something is wrong with the adapter path / settings.small_ai_output_dir

Run from the repo root, same convention as the other scripts in this folder:

    python -m backend.tests.test_qa_small_ai
    python -m backend.tests.test_qa_small_ai "What is a race condition?"

A short fixed list of educational questions is used by default (with and
without a grounding context) so you can eyeball a handful of results in one
run instead of just one.
"""
import sys
import logging

from backend.local_ai.tasks.qa import answer_question

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# (question, context) pairs — context="" means "no grounding, general knowledge"
DEFAULT_CASES = [
    ("What is the difference between a mutex and a semaphore?", ""),
    ("What is passive waiting?", ""),
    (
        "According to this excerpt, what happens when a thread exits a critical section?",
        "Exit of the CS: Makes the condition to enter CS for some (other or same) thread "
        "again true. If passive waiting: wake up one thread that is waiting/sleeping.",
    ),
    ("What is the capital of France?", ""),  # off-topic control: model should still answer sanely
]


def main():
    if len(sys.argv) > 1:
        cases = [(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")]
    else:
        cases = DEFAULT_CASES

    for question, context in cases:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        if context:
            print(f"CONTEXT : {context}")
        result = answer_question(question=question, context=context)
        print(f"\nANSWER (has_context={result['has_context']}):\n{result['answer']}")


if __name__ == "__main__":
    main()
