import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import ExamState

logger = logging.getLogger(__name__)

llm = MODELS["gpt-5-nano"]

INTAKE_PROMPT = """
You parse a teacher's exam request into structured parameters.

Extract:
- question_type: "mcq" if they ask for multiple-choice / a quiz; "essay" if open-ended / written answers.
- num_questions: how many questions are requested. If not stated, use 5.
""".strip()


class ExamSpec(BaseModel):
    question_type: Literal["mcq", "essay"]
    num_questions: int = Field(default=5, description="Number of questions requested; 5 if unspecified")


def intake_agent(state: ExamState) -> dict:
    spec = llm.with_structured_output(ExamSpec, method="function_calling").invoke([
        SystemMessage(content=INTAKE_PROMPT),
        HumanMessage(content=state["query"]),
    ])
    logger.info("Intake: question_type=%s num_questions=%d", spec.question_type, spec.num_questions)
    return {
        "question_type": spec.question_type,
        "num_questions_target": spec.num_questions,
    }
