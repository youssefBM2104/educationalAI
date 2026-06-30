import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import TutoringState

logger = logging.getLogger(__name__)

llm = MODELS["gpt_oss"]

ANSWER_PROMPT = """
## Role
You give the **full, correct answer** to the original question.

## Rules
- Ground the answer in the **context**.
- Be clear and complete.
""".strip()

MISTAKE_PROMPT = """
## Role
The student made a specific mistake while deriving the answer.

## Task
- Write a short, targeted correction.
- **Name the slip** and show the right step.
- Be brief.
""".strip()

TEACH_PROMPT = """
## Role
You briefly explain the answer to a sub-question so the student can move on.

## Rules
- Keep it short and clear.
""".strip()


def _context(state: TutoringState) -> str:
    chunks = state.get("rag_chunks") or []
    return "\n\n".join(c.get("text", "") for c in chunks) or "(no context)"


def _answer(state: TutoringState) -> str:
    res = llm.invoke([
        SystemMessage(content=ANSWER_PROMPT),
        HumanMessage(content=f"Original question: {state['query']}\n\nContext:\n{_context(state)}"),
    ])
    return res.content.strip()


def _mistake_fix(state: TutoringState) -> str:
    diag = state.get("diagnosis") or {}
    res = llm.invoke([
        SystemMessage(content=MISTAKE_PROMPT),
        HumanMessage(content=(
            f"Student attempt: {state.get('student_answer')}\n"
            f"Specific error: {diag.get('specific_error')}"
        )),
    ])
    return res.content.strip()


def teach_explanation(state: TutoringState) -> str:
    res = llm.invoke([
        SystemMessage(content=TEACH_PROMPT),
        HumanMessage(content=(
            f"Sub-question: {state.get('sub_question')}\n"
            f"Expected answer: {state.get('expected_answer')}"
        )),
    ])
    return res.content.strip()


def answer_node(state: TutoringState) -> dict:
    diag = state.get("diagnosis") or {}
    prefix = ""
    if diag.get("status") == "has_mistake":
        prefix = _mistake_fix(state) + "\n\n"

    final = prefix + _answer(state)
    logger.info("Answer emitted (mistake_fix=%s)", bool(prefix))
    return {"final_output": final, "tutor_message": final, "phase": "done"}
