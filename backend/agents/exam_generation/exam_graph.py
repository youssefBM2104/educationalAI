import logging

from langgraph.graph import StateGraph, START, END

from backend.agents.state import ExamState
from backend.agents.exam_generation.intake_agent import intake_agent
from backend.agents.exam_generation.generator_agent import generator_agent
from backend.agents.exam_generation.chunk_pool import build_chunk_pool
from backend.agents.exam_generation.solver_agent import solver_agent
from backend.agents.exam_generation.judge_agent import judge_agent, route_after_judge

logger = logging.getLogger(__name__)

_app = None


# --- helpers ---

def plan_difficulties(n: int) -> list[str]:
    n_easy = round(n * 0.4)
    n_medium = round(n * 0.4)
    n_hard = n - n_easy - n_medium
    return ["easy"] * n_easy + ["medium"] * n_medium + ["hard"] * n_hard


# --- Nodes ---

def plan_node(state: ExamState) -> dict:
    n = state.get("num_questions_target", 5)
    return {
        "difficulty_plan": plan_difficulties(n),
        "current_index": 0,
        "exam_set": [],
        "exam_questions": [],
        "iteration": 0,
    }


def set_difficulty_node(state: ExamState) -> dict:
    plan = state.get("difficulty_plan") or []
    idx = state.get("current_index", 0)
    difficulty = plan[idx] if idx < len(plan) else "medium"
    return {"difficulty": difficulty}


def collect_node(state: ExamState) -> dict:
    update = {"current_index": state.get("current_index", 0) + 1, "iteration": 0}

    if state.get("judge_passed"):
        gq = state.get("generated_question", {})
        item = {
            "question": gq.get("question"),
            "difficulty": state.get("difficulty"),
            "kg_path": state.get("kg_path"),
            "correct_answer": state.get("correct_answer"),
        }
        if state["question_type"] == "mcq":
            item["choices"] = gq.get("choices")
            item["explanation"] = gq.get("explanation")
        else:
            item["model_answer"] = gq.get("model_answer")
            item["marking_scheme"] = state.get("marking_scheme")
        update["exam_set"] = (state.get("exam_set") or []) + [item]

    logger.info("Collect: passed=%s -> next index=%d",
                state.get("judge_passed"), update["current_index"])
    return update


def route_after_collect(state: ExamState) -> str:
    if state.get("current_index", 0) < state.get("num_questions_target", 1):
        return "next"
    return "done"


# --- Exam Graph ---

def get_exam_graph():
    global _app
    if _app is not None:
        return _app

    g = StateGraph(ExamState)

    g.add_node("intake", intake_agent)
    g.add_node("plan", plan_node)
    g.add_node("set_difficulty", set_difficulty_node)
    g.add_node("generator", generator_agent)
    g.add_node("chunk_pool", build_chunk_pool)
    g.add_node("solver", solver_agent)
    g.add_node("judge", judge_agent)
    g.add_node("collect", collect_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "plan")
    g.add_edge("plan", "set_difficulty")
    g.add_edge("set_difficulty", "generator")
    g.add_edge("generator", "chunk_pool")
    g.add_edge("chunk_pool", "solver")
    g.add_edge("solver", "judge")

    g.add_conditional_edges("judge", route_after_judge, {
        "accept": "collect",
        "give_up": "collect",
        "regenerate": "generator",
    })
    g.add_conditional_edges("collect", route_after_collect, {
        "next": "set_difficulty",
        "done": END,
    })

    _app = g.compile()
    return _app
