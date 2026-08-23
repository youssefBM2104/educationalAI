import logging

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage

from backend.core.models import MODELS
from backend.agents.state import ExamState

logger = logging.getLogger(__name__)

llm = MODELS["gpt-5.1"]


# --- Output schema ---

class MCQJudgement(BaseModel):
    relevance_pass: bool = Field(description="Question + correct answer are relevant and on-topic, no invented facts")
    answer_grounding_pass: bool = Field(description="Correct answer derivable from the real chunks")
    distractor_quality_pass: bool = Field(description="Wrong options plausible but clearly incorrect per KG")
    path_coverage_pass: bool = Field(description="Solver's reasoning traverses the path in order")
    solver_correct: bool = Field(description="Solver's answer matches the correct answer")
    feedback: str = Field(description="Actionable note for the Generator on what to fix")


class EssayJudgement(BaseModel):
    relevance_pass: bool = Field(description="Question + model answer are relevant and on-topic, no invented facts")
    answer_grounding_pass: bool = Field(description="Model answer derivable from the real chunks")
    path_coverage_pass: bool = Field(description="Solver's reasoning traverses the path in order")
    solver_correct: bool = Field(description="Solver's answer matches the model answer")
    feedback: str = Field(description="Actionable note for the Generator on what to fix")


# --- Helpers ---

def _format_kg(kg: dict | None) -> str:
    relations = (kg or {}).get("relations", [])
    if not relations:
        return "(no relations available)"
    return "\n".join(
        f"{r.get('from')} --[{r.get('type')}]--> {r.get('to')}" for r in relations
    )


def _format_chunks(chunks: list | None) -> str:
    if not chunks:
        return "(no chunks available)"
    return "\n\n".join(f"[{c.get('document_id')}#{c.get('chunk_index')}] {c.get('text', '')}" for c in chunks)


def _format_question(gq: dict, question_type: str) -> str:
    lines = [f"Question: {gq.get('question', '')}"]
    if question_type == "mcq":
        for key, val in (gq.get("choices") or {}).items():
            lines.append(f"{key}. {val}")
        lines.append(f"Correct option: {gq.get('correct_option')}")
        lines.append(f"Explanation: {gq.get('explanation', '')}")
    else:
        lines.append(f"Model answer: {gq.get('model_answer', '')}")
    return "\n".join(lines)


def _build_prompt(state: ExamState, question_type: str) -> str:
    distractor_criterion = (
        "- **Distractor quality**: are the wrong options plausible but clearly incorrect per the KG?\n"
        if question_type == "mcq" else ""
    )
    # Static-first for prefix caching: Role, Criteria and the full knowledge graph are identical
    # for every question judged in one run, so they lead as a cacheable prefix. The per-question
    # parts (the question itself, its path, the real chunks, the solver's attempt) go at the end.
    return f"""# Role

You are the **Judge** in a multi-agent exam-generation pipeline, with FULL knowledge-graph access.
Evaluate the **quality of the QUESTION** (not the student's performance), using the generator's
correct answer as the reference.

# Criteria

- **Relevance**: are the question + correct answer relevant and on-topic for the source chunks (no invented facts)?
- **Answer grounding**: is the correct answer derivable from the SOURCE CHUNKS (directly or via multi-hop)?
  A claim that relies on a concept **absent from the source chunks is NOT grounded** — even if the
  knowledge graph connects it. The KG shows *structure* (which concepts relate), it is **not evidence**;
  verify every fact against the chunk TEXT, not against the graph. If the answer needs a fact about a
  concept that no chunk states, fail grounding.
{distractor_criterion}- **Path coverage**: does the SOLVER's reasoning traverse the reasoning path in order (not a shortcut)?
- Also report **solver_correct**: does the solver's answer match the correct answer?

# Knowledge graph (full)

{_format_kg(state.get("kg_context"))}

# Question

{_format_question(state["generated_question"], question_type)}

# Reasoning path (correct)

{state.get("kg_path")}

# Source chunks (real)

{_format_chunks(state.get("chunk_bundle"))}

# Solver output (simulated student)

Answer: {state.get("solver_answer")}
Reasoning: {state.get("solver_reasoning")}
"""


def _decide(question_type: str, ev, solver_correct: bool) -> tuple[bool, str]:
    # Generator-target failures → regenerate (question broken)
    if not ev.relevance_pass or not ev.answer_grounding_pass:
        return False, "regenerate: question broken (grounding/relevance)"
    if question_type == "mcq" and not ev.distractor_quality_pass:
        return False, "regenerate: question broken (distractor quality)"
    # Solver-target: path coverage fails but solver still got it → too easy
    if not ev.path_coverage_pass and solver_correct:
        return False, "regenerate: question too easy (solver shortcut)"
    # Otherwise keep; if solver could not solve, flag difficulty calibration
    if not solver_correct:
        return True, "keep: calibrate difficulty up (solver could not solve)"
    return True, "ok"


# --- Node ---

def judge_agent(state: ExamState) -> dict:
    question_type = state["question_type"]
    schema = MCQJudgement if question_type == "mcq" else EssayJudgement
    structured_llm = llm.with_structured_output(schema, method="function_calling")

    ev = structured_llm.invoke([SystemMessage(content=_build_prompt(state, question_type))])

    # MCQ correctness is deterministic; essay relies on the judge's assessment
    if question_type == "mcq":
        solver_correct = (
            str(state.get("solver_answer", "")).strip().upper()
            == str(state.get("correct_answer", "")).strip().upper()
        )
    else:
        solver_correct = ev.solver_correct

    passed, diagnosis = _decide(question_type, ev, solver_correct)

    feedback = {
        "relevance_pass": ev.relevance_pass,
        "answer_grounding_pass": ev.answer_grounding_pass,
        "path_coverage_pass": ev.path_coverage_pass,
        "solver_correct": solver_correct,
        "diagnosis": diagnosis,
        "notes": ev.feedback,
    }
    if question_type == "mcq":
        feedback["distractor_quality_pass"] = ev.distractor_quality_pass

    logger.info("Judge: passed=%s diagnosis=%s", passed, diagnosis)
    return {"judge_passed": passed, "judge_feedback": feedback}


# --- Routing (conditional edge after judge) ---

def route_after_judge(state: ExamState) -> str:
    if state["judge_passed"]:
        return "accept"
    if state.get("iteration", 0) >= state.get("max_iterations", 4):
        return "give_up"
    return "regenerate"
