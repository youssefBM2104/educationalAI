import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import TutoringState

logger = logging.getLogger(__name__)

llm = MODELS["gemma"]

DIAG_PROMPT = """
## Role
You diagnose the student's attempt to derive the full answer to the original question.
Use the **context as ground truth**.

## Diagnosis
- **no_attempt** — the student did not actually try (e.g. *"I don't know"*, *"no idea"*, blank,
  or a reply with no reasoning about the question). There is no slip to point at.
- **correct** — the derivation reaches the right conclusion.
- **has_mistake** — the student DID reason but went wrong; pinpoint the **specific error**
  (e.g. *"sign flipped in step 3"*, *"confused chain rule with product rule"*).

## Output
- `status`: `no_attempt`, `correct`, or `has_mistake`
- `specific_error`: where the student slipped — only for `has_mistake`, otherwise `null`
""".strip()


class Diagnosis(BaseModel):
    status: Literal["no_attempt", "correct", "has_mistake"]
    specific_error: Optional[str] = Field(default=None, description="Where the student slipped; null unless has_mistake")


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
