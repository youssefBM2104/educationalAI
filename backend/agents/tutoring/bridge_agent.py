import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import TutoringState

logger = logging.getLogger(__name__)

llm = MODELS["gpt_oss"]

BRIDGE_PROMPT = """
## Role
You write a single **bridge question**.

## Task
- Ask the student to connect their sub-answer back to the **original question**.
- Example: *"From your answer above, can you now derive the answer to the original question?"*
- Adapt the wording to the domain.

## Output
Output **only the question** — no preamble.
""".strip()


def bridge_generator(state: TutoringState) -> dict:
    res = llm.invoke([
        SystemMessage(content=BRIDGE_PROMPT),
        HumanMessage(content=(
            f"Original question: {state['query']}\n"
            f"Sub-question: {state.get('sub_question')}\n"
            f"Student's sub-answer: {state.get('student_answer')}"
        )),
    ])
    bridge_q = res.content.strip()
    logger.info("Bridge generated")
    return {"bridge_question": bridge_q, "phase": "await_bridge", "tutor_message": bridge_q}
