import logging

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import TutoringState

logger = logging.getLogger(__name__)

llm = MODELS["gemma"]

BLOOM_NAMES = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyze", 5: "Evaluate", 6: "Create"}

# What a question at each level must DEMAND of the student — without this the model
# produces the same question at every level.
BLOOM_GUIDANCE = {
    1: "Ask the student to RECALL a specific fact, term or definition stated almost verbatim in the context.",
    2: "Ask the student to EXPLAIN the idea in their own words, or say what it means / why it holds. "
       "Not a verbatim recall.",
    3: "Ask the student to APPLY the idea to a concrete situation or example that is NOT spelled out in the context.",
    4: "Ask the student to ANALYZE: compare or contrast two things, or break the idea into parts and relate them "
       "(e.g. contrast the two mechanisms described).",
    5: "Ask the student to EVALUATE: judge or justify which option is better, and on what criterion.",
    6: "Ask the student to CREATE: design, propose, or combine the ideas into something new.",
}


class SubQA(BaseModel):
    sub_question: str = Field(description="A scaffolding sub-question at the target Bloom level")
    expected_answer: str = Field(description="The correct answer to the sub-question")


def _context(state: TutoringState) -> str:
    chunks = state.get("rag_chunks") or []
    return "\n\n".join(c.get("text", "") for c in chunks) or "(no context)"


def _asked_block(state: TutoringState) -> str:
    asked = state.get("asked_questions") or []
    if not asked:
        return "_(none yet)_"
    return "\n".join(f"- {q}" for q in asked)


def generate_subquestion(state: TutoringState, level: int) -> tuple[str, str]:
    level = max(1, min(level, 6))
    prompt = f"""## Role
You generate a scaffolding sub-question for **Socratic tutoring**.

## Task
- Target Bloom level: **{level} ({BLOOM_NAMES.get(level)})**.
- {BLOOM_GUIDANCE[level]}
- The sub-question must be **answerable from the context**.
- It should lead the student one step toward the original question.
- Also provide the **expected correct answer**.

## Already asked — do NOT repeat or reword these
The student has already seen the questions below. Ask about a **different angle or sub-idea**;
never restate one of them.

{_asked_block(state)}

## Original question
{state['query']}

## Context
{_context(state)}
"""
    res = llm.with_structured_output(SubQA).invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate the sub-question now."),
    ])
    logger.info("QAGenerator: level=%d asked_so_far=%d", level, len(state.get("asked_questions") or []))
    return res.sub_question, res.expected_answer
