"""
Offline evaluation of exam generation: deterministic structural checks, then G-Eval rubrics
judged by Gemini (a held-out provider).

Generation is separated from evaluation on purpose: generating an exam is slow and flaky, so
the first run saves the exam to disk and every later eval re-scores that saved file for free.

Run (from repo root, inside .venv):
    # generate a fresh exam from the fixture, save it, then evaluate
    python -m backend.eval.run_exam_eval

    # re-evaluate an exam that was already generated (no LLM generation, no cost)
    python -m backend.eval.run_exam_eval --from outputs/eval/exam_latest.json

    # generate with a different request
    python -m backend.eval.run_exam_eval --query "Create 3 essay questions about deadlock"
"""
import argparse
import json
import logging
from pathlib import Path

from backend.eval.deterministic import check_exam_set
from backend.eval.judge import judge_name, set_judge
from backend.eval.metrics_exam import metrics_for

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

FIXTURE = Path("backend/tests/fixtures/rag_passive_waiting.json")
SAVE_PATH = Path("outputs/eval/exam_latest.json")
DEFAULT_QUERY = "Create 3 multiple choice questions about thread synchronization"


# --- Generation (only when we don't already have an exam to score) ---

def generate_exam(query: str) -> dict:
    from backend.agents.exam_generation.exam_graph import get_exam_graph

    rag = json.loads(FIXTURE.read_text(encoding="utf-8"))
    state = {
        "user_id": "eval",
        "course_id": rag.get("course_id", "test"),
        "query": query,
        "rag_chunks": rag["chunks"],
        "kg_context": rag["kg_context"],
    }
    print(f"Generating exam: {query!r} ...")
    result = get_exam_graph().invoke(state)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVE_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved exam -> {SAVE_PATH}\n")
    return result


# --- Turning one question into what the judge is allowed to see ---

def build_case(q: dict, result: dict) -> dict:
    chunks = result.get("rag_chunks") or []
    relations = (result.get("kg_context") or {}).get("relations", [])
    options = "\n".join(f"{k}. {v}" for k, v in (q.get("choices") or {}).items()) or "(essay)"

    return {
        "query": result.get("query", ""),
        "difficulty": q.get("difficulty", ""),
        "question": q.get("question", ""),
        "options": options,
        "correct_answer": q.get("correct_answer", ""),
        "explanation": q.get("explanation") or q.get("model_answer", ""),
        "chunks": "\n\n".join(c.get("text", "") for c in chunks),
        "kg_relations": "\n".join(
            f"{r.get('from')} --[{r.get('type')}]--> {r.get('to')}" for r in relations
        ),
    }


# --- Report ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_file", help="score a saved exam JSON instead of generating")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument(
        "--judge", default="gemini", choices=["gemini", "llama31"],
        help="gemini = held-out judge (report these numbers, 20 calls/day free); "
             "llama31 = same model as the pipeline's own judge_agent, so it is biased in the "
             "pipeline's favour — use it to iterate on rubrics, never to report results",
    )
    args = ap.parse_args()
    set_judge(args.judge)

    if args.from_file:
        result = json.loads(Path(args.from_file).read_text(encoding="utf-8"))
        print(f"Scoring saved exam: {args.from_file}\n")
    else:
        result = generate_exam(args.query)

    exam_set = result.get("exam_set") or []
    question_type = result.get("question_type", "mcq")
    if not exam_set:
        print("No questions in exam_set — nothing to evaluate.")
        return

    # --- 1. Deterministic (free, exact) ---
    print("=" * 72)
    print("STRUCTURAL CHECKS")
    print("=" * 72)
    det = check_exam_set(result)
    print(f"questions: {len(exam_set)}  target: {result.get('num_questions_target')}  "
          f"difficulty mix: {det['difficulty_mix']}")
    for issue in det["set_issues"]:
        print(f"  [SET]  {issue}")
    for i, issues in det["per_question"].items():
        for issue in issues:
            print(f"  [Q{i}]   {issue}")
    if det["total_issues"] == 0:
        print("  all structural checks passed")
    print(f"\ntotal structural issues: {det['total_issues']}")

    # --- 2. G-Eval (LLM judge, held-out provider) ---
    metrics = metrics_for(question_type)
    print("\n" + "=" * 72)
    print(f"G-EVAL  (judge: {judge_name()})")
    print("=" * 72)

    totals: dict[str, list[float]] = {m.name: [] for m in metrics}

    for i, q in enumerate(exam_set, 1):
        case = build_case(q, result)
        print(f"\n--- Q{i} [{q.get('difficulty')}] {q.get('question', '')[:70]}...")
        for metric in metrics:
            res = metric.measure(case)
            totals[metric.name].append(res.score)
            flag = "PASS" if res.passed else "FAIL"
            print(f"  {metric.name:<28} {res.score:.2f}  [{flag}]  {res.reason[:90]}")

    # --- 3. Summary ---
    print("\n" + "=" * 72)
    print("SUMMARY (mean score per metric, 0-1)")
    print("=" * 72)
    for name, scores in totals.items():
        mean = sum(scores) / len(scores)
        bar = "#" * int(round(mean * 20))
        print(f"  {name:<28} {mean:.2f}  {bar}")
    overall = sum(sum(s) for s in totals.values()) / sum(len(s) for s in totals.values())
    print(f"\n  {'OVERALL':<28} {overall:.2f}")
    print(f"  structural issues            {det['total_issues']}")


if __name__ == "__main__":
    main()
