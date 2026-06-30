import logging

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import TutoringState

logger = logging.getLogger(__name__)

llm = MODELS["gpt_oss"]

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyze", 5: "Evaluate", 6: "Create"}


class SubQA(BaseModel):
    sub_question: str = Field(description="A scaffolding sub-question at the target Bloom level")
    expected_answer: str = Field(description="The correct answer to the sub-question")


def _context(state: TutoringState) -> str:
    chunks = state.get("rag_chunks") or []
    return "\n\n".join(c.get("text", "") for c in chunks) or "(no context)"


def generate_subquestion(state: TutoringState, level: int) -> tuple[str, str]:
    level = max(1, min(level, 6))
    prompt = f"""## Role
You generate a scaffolding sub-question for **Socratic tutoring**.

## Task
- Target Bloom level: **{level} ({BLOOM_NAMES.get(level)})**.
- The sub-question must be **answerable from the context**.
- It should lead the student one step toward the original question.
- Also provide the **expected correct answer**.

## Original question
{state['query']}

## Context
{_context(state)}
"""
    res = llm.with_structured_output(SubQA).invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate the sub-question now."),
    ])
    logger.info("QAGenerator: level=%d", level)
    return res.sub_question, res.expected_answer
