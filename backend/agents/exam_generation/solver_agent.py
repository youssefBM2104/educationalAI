import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import ExamState

logger = logging.getLogger(__name__)

llm = MODELS["llama31"]


# --- Output schema ---

class MCQSolution(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning chain using only the passages")
    answer: Literal["A", "B", "C", "D"]


class EssaySolution(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning chain using only the passages")
    answer: str = Field(description="Free-form answer")


# --- Helpers ---

def _format_question(generated_question: dict, question_type: str) -> str:
    lines = [generated_question.get("question", "")]
    if question_type == "mcq":
        for key, val in (generated_question.get("choices") or {}).items():
            lines.append(f"{key}. {val}")
    return "\n".join(lines)


def _format_pool(pool: list | None) -> str:
    if not pool:
        return "(no passages available)"
    return "\n\n".join(f"[{i + 1}] {c.get('text', '')}" for i, c in enumerate(pool))


def _build_prompt(state: ExamState, question_type: str) -> str:
    if question_type == "mcq":
        answer_rule = "Choose exactly ONE option: A, B, C, or D."
    else:
        answer_rule = "Write a free-form answer."

    return f"""# Role

You are a student answering an exam question. Use ONLY the passages below — you have NO outside
knowledge and NO access to any knowledge graph or answer key.

# Question

{_format_question(state["generated_question"], question_type)}

# Passages

{_format_pool(state.get("chunk_pool"))}

# Instructions

- Reason step by step, using only the passages.
- {answer_rule}
- If the passages are insufficient, say so, then give your best attempt.
"""


# --- Node ---

def solver_agent(state: ExamState) -> dict:
    question_type = state["question_type"]
    schema = MCQSolution if question_type == "mcq" else EssaySolution
    structured_llm = llm.with_structured_output(schema)

    result = structured_llm.invoke([
        SystemMessage(content=_build_prompt(state, question_type)),
        HumanMessage(content="Answer the question now."),
    ])

    logger.info("Solver: type=%s answer=%s", question_type, result.answer)
    return {
        "solver_answer": result.answer,
        "solver_reasoning": result.reasoning,
    }
