import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import TutoringState

logger = logging.getLogger(__name__)

llm = MODELS["nemotron"]

DIAG_PROMPT = """
## Role
You diagnose the student's attempt to derive the full answer to the original question.
Use the **context as ground truth**.

## Diagnosis
- **correct** — the derivation reaches the right conclusion.
- **has_mistake** — otherwise; pinpoint the **specific error**
  (e.g. *"sign flipped in step 3"*, *"confused chain rule with product rule"*).

## Output
- `status`: `correct` or `has_mistake`
- `specific_error`: where the student slipped, or `null` if correct
""".strip()


class Diagnosis(BaseModel):
    status: Literal["correct", "has_mistake"]
    specific_error: Optional[str] = Field(default=None, description="Where the student slipped, or null if correct")


def _context(state: TutoringState) -> str:
    chunks = state.get("rag_chunks") or []
    return "\n\n".join(c.get("text", "") for c in chunks) or "(no context)"


def diagnoser(state: TutoringState) -> dict:
    res = llm.with_structured_output(Diagnosis).invoke([
        SystemMessage(content=DIAG_PROMPT),
        HumanMessage(content=(
            f"Original question: {state['query']}\n"
            f"Student's derivation attempt: {state.get('student_answer')}\n\n"
            f"Context:\n{_context(state)}"
        )),
    ])
    logger.info("Diagnoser: %s", res.status)
    return {"diagnosis": {"status": res.status, "specific_error": res.specific_error}}
