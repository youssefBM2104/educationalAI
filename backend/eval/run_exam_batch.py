"""
Benchmark exam generation across a spread of teacher requests, then report aggregate stats.

A single query tells you almost nothing — one bad exam could be luck. This runs the pipeline over
several requests that vary in question type, count and topic, and reports the mean/spread per
metric so a weakness has to show up consistently before we act on it.

Generation and scoring are separate phases: exams are written to disk first, so re-scoring (a new
rubric, a different judge) never pays the generation cost again.

Run (from repo root, inside .venv):
    python -m backend.eval.run_exam_batch                 # generate all, then score
    python -m backend.eval.run_exam_batch --score-only    # re-score what is already on disk
    python -m backend.eval.run_exam_batch --judge gemini  # held-out judge (20 calls/day free)
"""
import argparse
import json
import logging
import statistics
import traceback
from pathlib import Path

from backend.eval.deterministic import check_exam_set
from backend.eval.judge import judge_name, set_judge
from backend.eval.metrics_exam import metrics_for
from backend.eval.run_exam_eval import build_case

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

FIXTURE = Path("backend/tests/fixtures/rag_passive_waiting.json")
OUT_DIR = Path("outputs/eval/batch")

# Varied on purpose: question type, count, and how directly the topic is named in the source.
QUERIES = {
    "mcq_sync":     "Create 3 multiple-choice questions about thread synchronization",
    "mcq_deadlock": "Create 3 multiple-choice questions about deadlock and how it is detected",
    "mcq_locking":  "Write a 3-question quiz comparing mutex locks and semaphores",
    "mcq_rust":     "Create 3 multiple-choice questions about shared-state concurrency in Rust",
    "essay_avoid":  "Create 2 essay questions about strategies for avoiding deadlocks",
    "essay_wait":   "Create 2 essay questions about passive waiting versus active waiting",
}


def generate(name: str, query: str) -> dict | None:
    from backend.agents.exam_generation.exam_graph import get_exam_graph

    rag = json.loads(FIXTURE.read_text(encoding="utf-8"))
    state = {
        "user_id": "eval",
        "course_id": rag.get("course_id", "test"),
        "query": query,
        "rag_chunks": rag["chunks"],
        "kg_context": rag["kg_context"],
    }
    try:
        result = get_exam_graph().invoke(state)
    except Exception:
        print(f"  [{name}] GENERATION FAILED")
        traceback.print_exc(limit=1)
        return None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  [{name}] {len(result.get('exam_set') or [])} questions -> {OUT_DIR / f'{name}.json'}")
    return result


def score(name: str, result: dict) -> dict | None:
    exam_set = result.get("exam_set") or []
    if not exam_set:
        return None

    det = check_exam_set(result)
    metrics = metrics_for(result.get("question_type", "mcq"))

    per_metric: dict[str, list[float]] = {m.name: [] for m in metrics}
    for q in exam_set:
        case = build_case(q, result)
        for metric in metrics:
            per_metric[metric.name].append(metric.measure(case).score)

    means = {k: statistics.mean(v) for k, v in per_metric.items()}
    print(f"  [{name}] " + "  ".join(f"{k.split()[0][:4]}={v:.2f}" for k, v in means.items())
          + f"  struct_issues={det['total_issues']}")
    return {
        "asked": result.get("num_questions_target"),
        "produced": len(exam_set),
        "question_type": result.get("question_type"),
        "difficulty_mix": det["difficulty_mix"],
        "structural_issues": det["total_issues"],
        "issue_list": det["set_issues"] + [f"Q{i}: {m}" for i, ms in det["per_question"].items() for m in ms],
        "scores": per_metric,
        "means": means,
    }


def report(stats: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("PER-REQUEST")
    print("=" * 78)
    print(f"{'request':<14}{'type':<7}{'asked':>6}{'made':>6}{'struct':>8}   metric means")
    for name, s in stats.items():
        means = "  ".join(f"{k.split()[0][:4].lower()}={v:.2f}" for k, v in s["means"].items())
        print(f"{name:<14}{s['question_type']:<7}{s['asked']:>6}{s['produced']:>6}"
              f"{s['structural_issues']:>8}   {means}")

    print("\n" + "=" * 78)
    print("AGGREGATE  (per-question scores pooled across all requests)")
    print("=" * 78)
    pooled: dict[str, list[float]] = {}
    for s in stats.values():
        for k, v in s["scores"].items():
            pooled.setdefault(k, []).extend(v)

    for name, scores in sorted(pooled.items()):
        mean = statistics.mean(scores)
        sd = statistics.pstdev(scores)
        below = sum(1 for x in scores if x < 0.7)
        bar = "#" * int(round(mean * 20))
        print(f"  {name:<28} {mean:.2f} +/-{sd:.2f}  n={len(scores):<3} "
              f"below_threshold={below:<3} {bar}")

    all_scores = [x for v in pooled.values() for x in v]
    total_q = sum(s["produced"] for s in stats.values())
    total_asked = sum(s["asked"] for s in stats.values())
    total_issues = sum(s["structural_issues"] for s in stats.values())
    print(f"\n  {'OVERALL':<28} {statistics.mean(all_scores):.2f}")
    print(f"  questions produced           {total_q}/{total_asked}")
    print(f"  structural issues            {total_issues}")

    issues = [i for s in stats.values() for i in s["issue_list"]]
    if issues:
        print("\n  structural issues seen:")
        for i in issues:
            print(f"    - {i}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true", help="score exams already on disk")
    ap.add_argument("--judge", default="llama31", choices=["gemini", "llama31"])
    args = ap.parse_args()
    set_judge(args.judge)

    exams: dict[str, dict] = {}
    if args.score_only:
        for name in QUERIES:
            path = OUT_DIR / f"{name}.json"
            if path.exists():
                exams[name] = json.loads(path.read_text(encoding="utf-8"))
        print(f"Loaded {len(exams)} saved exams from {OUT_DIR}\n")
    else:
        print(f"GENERATING {len(QUERIES)} exams\n" + "=" * 78)
        for name, query in QUERIES.items():
            print(f"  [{name}] {query}")
            result = generate(name, query)
            if result:
                exams[name] = result

    if not exams:
        print("No exams to score.")
        return

    print("\n" + "=" * 78)
    print(f"SCORING  (judge: {judge_name()})")
    print("=" * 78)
    stats = {}
    for name, result in exams.items():
        s = score(name, result)
        if s:
            stats[name] = s

    if stats:
        report(stats)
        (OUT_DIR / "summary.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nsaved -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
