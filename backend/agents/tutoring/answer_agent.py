import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import TutoringState

logger = logging.getLogger(__name__)

llm = MODELS["gemma"]

ANSWER_PROMPT = """
## Role
You give the **full, correct answer** to the original question.

## Rules
- Ground the answer in the **context**.
- Be clear and complete.
""".strip()

MISTAKE_PROMPT = """
## Role
The student reasoned about the question but made a **specific mistake**.

## Task
- Write a short, targeted correction of THAT mistake, grounded in the context.
- **Name the slip** and show the right step.
- Stay on the topic of the question — never give generic study advice.
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
            # Static-first for prefix caching: query + context are constant across the session.
            f"Original question: {state['query']}\n\n"
            f"Context:\n{_context(state)}\n\n"
            f"---\n"
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
    # Only correct a real slip. On "no_attempt" (or "correct") there is nothing to fix —
    # forcing a correction there makes the model invent generic, off-topic advice.
    if diag.get("status") == "has_mistake" and diag.get("specific_error"):
        prefix = _mistake_fix(state) + "\n\n"

    final = prefix + _answer(state)
    logger.info("Answer emitted (status=%s mistake_fix=%s)", diag.get("status"), bool(prefix))
    return {"final_output": final, "tutor_message": final, "phase": "done"}
