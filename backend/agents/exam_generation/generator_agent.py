import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import ExamState

logger = logging.getLogger(__name__)

llm = MODELS["gemma"]

# Difficulty is the number of concepts (chunks) on the correct path
PATH_LENGTH_RANGES = {
    "easy":   (1, 2),
    "medium": (3, 4),
    "hard":   (5, 7),
}
PATH_HARD_LIMIT = 8


# --- Output schema ---

class Distractor(BaseModel):
    choice_key: Literal["A", "B", "C", "D"]
    concept_path: list[str] = Field(
        description="Near-miss path backing this wrong option, e.g. [a, b, c, e]"
    )


class MCQQuestion(BaseModel):
    kg_path: list[str] = Field(description="Ordered concept ids on the correct path, e.g. [a, b, c, d]")
    question: str
    choices: dict[str, str] = Field(description='Exactly four options keyed "A","B","C","D"')
    correct_option: Literal["A", "B", "C", "D"]
    explanation: str = Field(description="Why the correct option is right, grounded in the chunks")
    distractor_paths: list[Distractor] = Field(
        description="One near-miss path per wrong option, used to build distractor chunks"
    )


class EssayQuestion(BaseModel):
    kg_path: list[str] = Field(description="Ordered concept ids on the correct path")
    question: str
    model_answer: str = Field(description="Generator's ground-truth answer")
    marking_scheme: str = Field(
        description=(
            "10-point rubric: Path coverage (4) + Content accuracy (4) "
            "+ Completeness & clarity (2), itemized per node/relation of the path"
        )
    )


# --- Helpers ---

def _format_kg(kg: dict | None) -> str:
    relations = (kg or {}).get("relations", [])
    if not relations:
        return "(no relations available)"
    return "\n".join(
        f"{r.get('from')} --[{r.get('type')}]--> {r.get('to')}"
        for r in relations
    )


def _format_chunks(chunks: list | None) -> str:
    if not chunks:
        return "(no chunks available)"
    return "\n\n".join(
        f"[{c.get('document_id')}#{c.get('chunk_index')}] {c.get('text', '')}" for c in chunks
    )


def _concept_ids(chunk: dict) -> set[str]:
    out = set()
    for c in chunk.get("covers_concepts", []) or []:
        cid = c.get("id") if isinstance(c, dict) else c
        if cid:
            out.add(str(cid).lower())
    return out


def _select_chunk_bundle(chunks: list | None, kg_path: list[str]) -> list:
    path = {c.lower() for c in kg_path}
    return [c for c in (chunks or []) if _concept_ids(c) & path]


def _distractor_rule(difficulty: str) -> str:
    if difficulty == "easy":
        return (
            "The path is short, so distractors may be any plausible-but-wrong concepts from the KG."
        )
    return (
        "Build each wrong option as a NEAR-MISS path: share the longest possible prefix with the "
        "correct path, then diverge into a KG neighbor that is NOT on the path "
        "(e.g. correct a→b→c→d → distractor a→b→c→e or a→b→m→d)."
    )


def _type_rules(question_type: str, difficulty: str) -> str:
    if question_type == "mcq":
        return (
            "Produce a **multiple-choice** question with exactly 4 options (A–D): one correct, "
            "three plausible-but-wrong distractors clearly incorrect per the KG.\n\n"
            f"- {_distractor_rule(difficulty)}\n"
            "- For each wrong option, also return its near-miss `concept_path` in `distractor_paths`."
        )
    return (
        "Produce an **open-ended essay** question with a model answer and a 10-point marking "
        "scheme — Path coverage (4) + Content accuracy (4) + Completeness & clarity (2) — "
        "itemized against each node/relation of the chosen path."
    )


def _build_prompt(state: ExamState, question_type: str, difficulty: str, length: tuple[int, int]) -> str:
    min_len, max_len = length

    feedback = state.get("judge_feedback")
    feedback_block = ""
    if feedback:
        feedback_block = (
            "\n## Previous attempt failed\n"
            "Fix these issues specifically:\n\n"
            f"{feedback}\n"
        )

    prior = state.get("exam_questions", []) or []
    prior_block = "\n".join(f"- {q.get('question', '')}" for q in prior) or "_(none yet)_"

    return f"""# Role

You are the **Generator** in a multi-agent exam-generation pipeline.

# Task

Pick a **connected path of {min_len}–{max_len} concepts (chunks)** through the knowledge graph below
(difficulty: **{difficulty}**; never exceed {PATH_HARD_LIMIT} concepts). The question MUST require
traversing the whole path — it cannot be answerable by a shortcut.

## Question type

{_type_rules(question_type, difficulty)}

## Rules

- Ground the question and the correct answer **strictly** in the source chunks. Do not invent facts.
- The correct answer must be derivable from the chunks (directly or via multi-hop combination).
- Target a specific path of concepts, not a generic question.
{feedback_block}
# Knowledge graph relations

_Choose your path from these._

{_format_kg(state.get('kg_context'))}

# Source chunks

_Ground everything in these._

{_format_chunks(state.get('rag_chunks'))}

# Already generated

_Pick a DIFFERENT path/aspect — do not repeat:_

{prior_block}
"""


# --- Node ---

def generator_agent(state: ExamState) -> dict:
    question_type = state["question_type"]
    difficulty = state["difficulty"]
    iteration = state.get("iteration", 0)
    length = PATH_LENGTH_RANGES[difficulty]

    schema = MCQQuestion if question_type == "mcq" else EssayQuestion
    structured_llm = llm.with_structured_output(schema)

    result = structured_llm.invoke([
        SystemMessage(content=_build_prompt(state, question_type, difficulty, length)),
        HumanMessage(content="Generate the question now."),
    ])

    actual_len = len(result.kg_path)
    min_len, max_len = length
    if not (min_len <= actual_len <= max_len):
        logger.warning(
            "Path length %d outside %s range %s — difficulty not met",
            actual_len, difficulty, length,
        )

    chunk_bundle = _select_chunk_bundle(state.get("rag_chunks"), result.kg_path)
    generated = result.model_dump()

    update = {
        "generated_question": generated,
        "kg_path": result.kg_path,
        "chunk_bundle": chunk_bundle,
        "iteration": iteration + 1,
        "exam_questions": (state.get("exam_questions", []) or []) + [generated],
    }

    if question_type == "mcq":
        update["correct_answer"] = result.correct_option
    else:
        update["correct_answer"] = result.model_answer
        update["marking_scheme"] = result.marking_scheme

    logger.info(
        "Generator: type=%s difficulty=%s length=%d path=%s round=%d",
        question_type, difficulty, actual_len, result.kg_path, iteration + 1,
    )
    return update
