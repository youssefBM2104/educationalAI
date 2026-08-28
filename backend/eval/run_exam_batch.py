"""
Benchmark exam generation across a spread of teacher requests, then report aggregate stats.

A single query tells you almost nothing — one bad exam could be luck. This runs the pipeline over
several requests that vary in document, question type and topic, and reports the mean/spread per
metric so a weakness has to show up consistently before we act on it.

Two source documents are exercised (a thread-synchronization lecture and an inflation lecture) and
MCQ vs essay are scored and aggregated SEPARATELY — they use different rubrics and mixing them hides
where the weakness is. ~3 questions per exam.

Generation and scoring are separate phases: exams are written to disk first (both a full .json and a
readable .md), so re-scoring (a new rubric, a different judge) never pays the generation cost again.

Run (from repo root, inside .venv):
    python -m backend.eval.run_exam_batch                 # generate all, then score
    python -m backend.eval.run_exam_batch --score-only    # re-score what is already on disk
    python -m backend.eval.run_exam_batch --judge gemini  # held-out judge (20 calls/day free)
    python -m backend.eval.run_exam_batch --only thread_mcq_sync infl_essay_effects
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

OUT_DIR = Path("outputs/eval/batch")

# The two source documents (RAG-service payloads captured as fixtures).
FIXTURES = {
    "thread":    Path("backend/tests/fixtures/rag_passive_waiting.json"),
    "inflation": Path("backend/tests/fixtures/rag_inflation.json"),
}

# Each request: (document, query). Varied on purpose across document, type and topic.
# ~3 questions per exam; MCQ and essay are aggregated separately in the report.
QUERIES = {
    # --- thread-synchronization document ---
    "thread_mcq_sync":     ("thread", "Create 3 multiple-choice questions about thread synchronization"),
    "thread_mcq_deadlock": ("thread", "Create 3 multiple-choice questions about deadlock and how it is detected"),
    "thread_essay_wait":   ("thread", "Create 3 essay questions about passive waiting versus active waiting"),
    # --- inflation document ---
    "infl_mcq_calc":       ("inflation", "Create 3 multiple-choice questions about how the inflation rate is calculated"),
    "infl_mcq_index":      ("inflation", "Create 3 multiple-choice questions about price indices and their categories"),
    "infl_essay_effects":  ("inflation", "Create 3 essay questions about the causes and effects of inflation"),
}


def _load_fixture(doc: str) -> dict:
    return json.loads(FIXTURES[doc].read_text(encoding="utf-8"))


def _dump_readable(name: str, doc: str, result: dict) -> None:
    """Human-readable dump of the generated exam, so the questions can be reviewed by eye."""
    lines = [
        f"# {name}",
        f"- document: {doc}",
        f"- query: {result.get('query')}",
        f"- type: {result.get('question_type')}   target: {result.get('num_questions_target')}"
        f"   produced: {len(result.get('exam_set') or [])}",
        "",
    ]
    for i, q in enumerate(result.get("exam_set") or [], 1):
        lines.append(f"## Q{i}  [{q.get('difficulty', '?')}]")
        lines.append(q.get("question", ""))
        for k, v in (q.get("choices") or {}).items():
            lines.append(f"- {k}. {v}")
        if q.get("correct_answer"):
            lines.append(f"\n**Answer:** {q.get('correct_answer')}")
        expl = q.get("explanation") or q.get("model_answer")
        if expl:
            lines.append(f"**Explanation / model answer:** {expl}")
        lines.append("")
    (OUT_DIR / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def generate(name: str, doc: str, query: str) -> dict | None:
    from backend.agents.exam_generation.exam_graph import get_exam_graph

    rag = _load_fixture(doc)
    state = {
        "user_id": "eval",
        "course_id": rag.get("course_id") or doc,
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

    result["_doc"] = doc            # remember which document this exam came from
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _dump_readable(name, doc, result)
    print(f"  [{name}] {len(result.get('exam_set') or [])} questions -> {OUT_DIR / f'{name}.json'} (+ .md)")
    return result


def score(name: str, result: dict) -> dict | None:
    exam_set = result.get("exam_set") or []
    if not exam_set:
        return None

    det = check_exam_set(result)
    q_type = result.get("question_type", "mcq")
    metrics = metrics_for(q_type)

    per_metric: dict[str, list[float]] = {m.name: [] for m in metrics}
    for q in exam_set:
        case = build_case(q, result)
        for metric in metrics:
            per_metric[metric.name].append(metric.measure(case).score)

    means = {k: statistics.mean(v) for k, v in per_metric.items()}
    print(f"  [{name}] " + "  ".join(f"{k.split()[0][:4]}={v:.2f}" for k, v in means.items())
          + f"  struct_issues={det['total_issues']}")
    return {
        "doc": result.get("_doc", "?"),
        "asked": result.get("num_questions_target"),
        "produced": len(exam_set),
        "question_type": q_type,
        "difficulty_mix": det["difficulty_mix"],
        "structural_issues": det["total_issues"],
        "issue_list": det["set_issues"] + [f"Q{i}: {m}" for i, ms in det["per_question"].items() for m in ms],
        "scores": per_metric,
        "means": means,
    }


def _aggregate(title: str, group: dict[str, dict]) -> None:
    """Pool per-question scores across every request of ONE question type and print the summary."""
    if not group:
        return
    print("\n" + "=" * 78)
    print(f"AGGREGATE — {title}  (per-question scores pooled across these requests)")
    print("=" * 78)
    pooled: dict[str, list[float]] = {}
    for s in group.values():
        for k, v in s["scores"].items():
            pooled.setdefault(k, []).extend(v)

    for metric_name, scores in sorted(pooled.items()):
        mean = statistics.mean(scores)
        sd = statistics.pstdev(scores)
        below = sum(1 for x in scores if x < 0.7)
        bar = "#" * int(round(mean * 20))
        print(f"  {metric_name:<28} {mean:.2f} +/-{sd:.2f}  n={len(scores):<3} "
              f"below_0.70={below:<3} {bar}")

    all_scores = [x for v in pooled.values() for x in v]
    total_q = sum(s["produced"] for s in group.values())
    total_asked = sum(s["asked"] or 0 for s in group.values())
    total_issues = sum(s["structural_issues"] for s in group.values())
    print(f"\n  {'OVERALL ' + title:<28} {statistics.mean(all_scores):.2f}")
    print(f"  questions produced           {total_q}/{total_asked}")
    print(f"  structural issues            {total_issues}")


def report(stats: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("PER-REQUEST")
    print("=" * 78)
    print(f"{'request':<20}{'doc':<11}{'type':<7}{'asked':>6}{'made':>6}{'struct':>8}   metric means")
    for name, s in stats.items():
        means = "  ".join(f"{k.split()[0][:4].lower()}={v:.2f}" for k, v in s["means"].items())
        print(f"{name:<20}{s['doc']:<11}{s['question_type']:<7}{s['asked'] or 0:>6}{s['produced']:>6}"
              f"{s['structural_issues']:>8}   {means}")

    # MCQ and essay use different rubrics -> aggregate them separately.
    mcq = {n: s for n, s in stats.items() if s["question_type"] == "mcq"}
    essay = {n: s for n, s in stats.items() if s["question_type"] == "essay"}
    _aggregate("MCQ (trắc nghiệm)", mcq)
    _aggregate("ESSAY (tự luận)", essay)

    issues = [f"[{n}] {i}" for n, s in stats.items() for i in s["issue_list"]]
    if issues:
        print("\n" + "=" * 78)
        print("STRUCTURAL ISSUES SEEN")
        print("=" * 78)
        for i in issues:
            print(f"  - {i}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true", help="score exams already on disk")
    ap.add_argument("--judge", default="llama31", choices=["gemini", "llama31", "gpt4o"])
    ap.add_argument("--only", nargs="*", help="run only these request names")
    args = ap.parse_args()
    set_judge(args.judge)

    selected = {k: v for k, v in QUERIES.items() if not args.only or k in args.only}

    exams: dict[str, dict] = {}
    if args.score_only:
        for name in selected:
            path = OUT_DIR / f"{name}.json"
            if path.exists():
                exams[name] = json.loads(path.read_text(encoding="utf-8"))
        print(f"Loaded {len(exams)} saved exams from {OUT_DIR}\n")
    else:
        print(f"GENERATING {len(selected)} exams\n" + "=" * 78)
        for name, (doc, query) in selected.items():
            print(f"  [{name}] ({doc}) {query}")
            result = generate(name, doc, query)
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
