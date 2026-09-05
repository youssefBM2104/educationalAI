import logging
import random
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import ExamState

logger = logging.getLogger(__name__)

llm = MODELS["gpt-5"]

# Difficulty is the number of concepts (chunks) on the correct path
PATH_LENGTH_RANGES = {
    "easy":   (1, 2),
    "medium": (3, 4),
    "hard":   (5, 7),
}
PATH_HARD_LIMIT = 8


# --- Output schema ---

PATH_RULE = (
    "Ordered concept names, copied verbatim from the knowledge graph. Relation types are NOT "
    "concepts: given `A --[PART_OF]--> B --[EXTENDS]--> C`, the path is exactly ['A', 'B', 'C']. "
    "Each consecutive pair must be joined by a relation that appears in the graph, and no concept "
    "may appear more than once."
)


class Distractor(BaseModel):
    choice_key: Literal["A", "B", "C", "D"]
    concept_path: list[str] = Field(
        description=f"Near-miss path backing this wrong option. {PATH_RULE}"
    )


class MCQQuestion(BaseModel):
    kg_path: list[str] = Field(description=f"The correct path. {PATH_RULE}")
    question: str
    choices: dict[str, str] = Field(description='Exactly four options keyed "A","B","C","D"')
    correct_option: Literal["A", "B", "C", "D"]
    explanation: str = Field(description="Why the correct option is right, grounded in the chunks")
    distractor_paths: list[Distractor] = Field(
        description="One near-miss path per wrong option, used to build distractor chunks"
    )


class EssayQuestion(BaseModel):
    kg_path: list[str] = Field(description=f"The correct path. {PATH_RULE}")
    question: str
    model_answer: str = Field(description="Generator's ground-truth answer")
    marking_scheme: str = Field(
        description=(
            "10-point rubric: Path coverage (4) + Content accuracy (4) "
            "+ Completeness & clarity (2), itemized per node/relation of the path"
        )
    )


# --- Helpers ---

def _covered_concepts(chunks: list | None) -> set[str]:
    """All concept ids that actually appear in a retrieved chunk (lowercased), across all chunks."""
    covered: set[str] = set()
    for c in chunks or []:
        covered |= _concept_ids(c)
    return covered


def _format_kg(kg: dict | None, chunks: list | None = None) -> str:
    """Render the KG edges the Generator may build a path from. Soft-grounded filter: when `chunks`
    is given, keep an edge if AT LEAST ONE endpoint is covered by a retrieved chunk. This keeps the
    graph rich — a covered concept can branch to its neighbours (good for near-miss distractors) and
    two covered regions can bridge through a single hop — while dropping pure expansion↔expansion
    edges that lead nowhere grounded. A path may thus touch an ungrounded concept, but the Judge
    enforces that the ANSWER still traces to chunk text, so a question that leans on an ungrounded
    concept's facts is regenerated. (Hard 'both endpoints covered' guaranteed grounding but starved
    the graph — sparse paths caused redundancy and weak distractors.)"""
    relations = (kg or {}).get("relations", [])
    if chunks is not None:
        covered = _covered_concepts(chunks)
        relations = [
            r for r in relations
            if str(r.get("from", "")).lower() in covered or str(r.get("to", "")).lower() in covered
        ]
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


def _shuffle_mcq_options(result: MCQQuestion) -> MCQQuestion:
    """Remap A/B/C/D labels to a random permutation so the correct answer
    isn't always in the same slot. The LLM has no reliable control over
    where it places the correct option (it tends to default to "A"), so we
    fix the position programmatically after generation instead of relying
    on prompt instructions.
    """
    labels = ["A", "B", "C", "D"]
    shuffled = labels[:]
    random.shuffle(shuffled)
    mapping = dict(zip(labels, shuffled))  # old_label -> new_label

    remapped = {mapping[old]: text for old, text in result.choices.items()}

    new_choices = {k: remapped[k] for k in labels}
    new_correct = mapping[result.correct_option]
    new_distractors = [
        Distractor(choice_key=mapping[d.choice_key], concept_path=d.concept_path)
        for d in result.distractor_paths
    ]

    return result.model_copy(update={
        "choices": new_choices,
        "correct_option": new_correct,
        "distractor_paths": new_distractors,
    })


def _distractor_rule(difficulty: str) -> str:
    if difficulty == "easy":
        return (
            "The path is short, so distractors may be any plausible-but-wrong concepts from the KG."
        )
    return (
        "Each wrong option is a NEAR-MISS. Internally, pick its concept by branching off the correct "
        "path one step early (share the longest prefix, then diverge to a KG neighbour NOT on the "
        "path). But WRITE the option as a natural statement about that wrong concept — never show "
        "the branch as an arrow chain."
    )


def _type_rules(question_type: str, difficulty: str) -> str:
    if question_type == "mcq":
        return (
            "Produce a **multiple-choice** question with exactly 4 options (A–D): one correct, "
            "three plausible-but-wrong distractors clearly incorrect per the KG.\n\n"
            f"- {_distractor_rule(difficulty)}\n"
            "- Record each wrong option's near-miss branch in `distractor_paths` — this field is "
            "INTERNAL to the pipeline and never shown to the student, so the option text itself "
            "must still read as natural prose."
        )
    return (
        "Produce an **open-ended essay** question with a model answer and a 10-point marking "
        "scheme — Path coverage (4) + Content accuracy (4) + Completeness & clarity (2) — "
        "itemized against each node/relation of the chosen path.\n\n"
        "- The question must be answerable ONLY by explaining the specific relations along the path "
        "in order — not by a general summary of the topic. Anchor it in a concrete scenario so a "
        "broad 'describe everything about X' answer cannot score full marks."
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

    # Ordered static-first for prefix caching. The knowledge graph and source chunks are identical
    # for every question generated from one document, so they lead the prompt as a long cacheable
    # prefix. The per-attempt dynamic parts — judge feedback and the already-generated list — go at
    # the very end, so a regenerate or the next question reuses the whole KG+chunks+rules prefix.
    return f"""# Role

You are the **Generator** in a multi-agent exam-generation pipeline.

# Knowledge graph relations

_Choose your path from these._

{_format_kg(state.get('kg_context'), state.get('rag_chunks'))}

# Source chunks

_Ground everything in these._

{_format_chunks(state.get('rag_chunks'))}

# Task

Pick a **connected path of {min_len}–{max_len} concepts** through the knowledge graph above
(difficulty: **{difficulty}**; never exceed {PATH_HARD_LIMIT} concepts). The question MUST require
traversing every **edge** on the path — the specific relations linking the concepts — not merely
naming the concepts. It cannot be answerable by a shortcut.

## Path format

The graph is written as `from --[TYPE]--> to`. A path lists **only the concepts**, never the
relation types:

> Graph: `mutex --[PART_OF]--> passive waiting solutions --[EXTENDS]--> Semaphores`
> Path: `["mutex", "passive waiting solutions", "Semaphores"]`  ← 3 concepts, 2 relations

- Copy concept names **verbatim** from the graph.
- `PART_OF`, `CAUSES`, `EXTENDS`, ... are **relations, not concepts** — never put them in the path.
- Each consecutive pair must be joined by a relation that actually appears in the graph.
- `kg_path` must contain **exactly {min_len}–{max_len} entries** — this is what sets the difficulty.

## Build the question on the CONNECTIONS, not the concepts

The value of the path is its **edges** — the specific relations linking one concept to the next,
usually across *different* chunks. A question that merely name-drops the concepts (e.g. "Explain how
deadlocks occur and what causes them") is REJECTED: it is answerable from general knowledge or from a
single chunk, and wastes the graph.

Instead:
- For each consecutive pair `A --[REL]--> B` on your path, make the student reason about **that
  specific relationship** — why A leads to B, how B enables C — chained from one end to the other.
- The answer must require **combining facts from at least two different chunks** that the path's
  edges connect. If the whole answer sits inside one chunk, the path is too shallow — choose one
  whose concepts are spread across chunks.
- Prefer a **concrete scenario** over "explain the topic": give a specific situation, then ask the
  student to trace its consequence through the chain of relations.

## Question type

{_type_rules(question_type, difficulty)}

## Surface form — what the student actually reads

The path is your INTERNAL scaffolding for *choosing* what to ask; it must NOT appear on the surface.

- The `question` and every option in `choices` must be **natural, self-contained sentences**,
  grounded in the chunks — the kind a teacher writes.
- NEVER render a concept chain as text: no `A → B → C`, no arrows, no lists of concept names as an answer.
- NEVER use the words "concept path", "path", "graph", "node", or "traverse" in the visible
  question or options.
- A student who has never seen the knowledge graph must be able to read and answer it.
- `kg_path` and `distractor_paths` are separate internal fields — the scaffolding lives there,
  never in the question/choices text.

## Rules

- Ground the question and the correct answer **strictly** in the source chunks above. Do not invent facts.
- The correct answer must require a **multi-hop combination** of the chunks — never a single-chunk lookup.
- Reject broad "explain the topic" questions: the question must hinge on the specific relations along
  the path and force the student to connect facts that live in different chunks.

# Already generated

_Pick a DIFFERENT path/aspect — do not repeat:_

{prior_block}
{feedback_block}"""


# --- Node ---

def generator_agent(state: ExamState) -> dict:
    question_type = state["question_type"]
    difficulty = state["difficulty"]
    iteration = state.get("iteration", 0)
    length = PATH_LENGTH_RANGES[difficulty]

    schema = MCQQuestion if question_type == "mcq" else EssayQuestion
    # function_calling (tool-calling) instead of OpenAI's strict json_schema: the strict mode
    # rejects `choices: dict[str, str]` (open-ended object). Tool-calling handles it, and it is
    # also how the NVIDIA path already ran.
    structured_llm = llm.with_structured_output(schema, method="function_calling")

    result = structured_llm.invoke([
        SystemMessage(content=_build_prompt(state, question_type, difficulty, length)),
        HumanMessage(content="Generate the question now."),
    ])

    if question_type == "mcq":
        result = _shuffle_mcq_options(result)

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