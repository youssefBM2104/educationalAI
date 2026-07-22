import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import TutoringState

logger = logging.getLogger(__name__)

llm = MODELS["qwen"]

EVAL_PROMPT = """
## Role
You grade a student's answer to a sub-question.

## Grading
- **pass** — correct and equivalent in meaning to the expected answer (different wording is fine).
- **fail** — wrong, contradicts, or misses the key point.

## Output
- `result`: `pass` or `fail`
- `notes`: a short note on what was right or wrong
""".strip()


class EvalResult(BaseModel):
    result: Literal["pass", "fail"]
    notes: str = Field(description="Short note on what was right or wrong")


def response_evaluator(state: TutoringState) -> dict:
    res = llm.with_structured_output(EvalResult).invoke([
        SystemMessage(content=EVAL_PROMPT),
        HumanMessage(content=(
            f"Sub-question: {state.get('sub_question')}\n"
            f"Expected answer: {state.get('expected_answer')}\n"
            f"Student answer: {state.get('student_answer')}"
        )),
    ])
    logger.info("Evaluator: %s", res.result)
    return {"eval_result": res.result, "eval_notes": res.notes}
