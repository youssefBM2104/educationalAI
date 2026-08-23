"""
RAGAS evaluation of the mindmap- and summary-generation pipelines, judged by gpt-4o.

Metrics (judged by gpt-4o), per pipeline:
  - Mindmap : Faithfulness (map grounded in source — precision)
              + Context Recall (source content covered by the map — recall; reference=source,
                context=the map, so it measures how much of the source the map captured)
  - Summary : Faithfulness (grounded)
              + Answer Relevancy (addresses the request — uses OpenAI embeddings)

Both pipelines are generation-over-context, so the source chunks are the RAGAS `context`, the
flattened mindmap / assembled summary is the `response`. Generation and scoring are separated:
outputs are written to disk first, so re-scoring never pays the generation cost again.

Run (from repo root, inside .venv):
    python -m backend.eval.run_learning_eval                 # generate all, then score
    python -m backend.eval.run_learning_eval --score-only    # re-score saved outputs
    python -m backend.eval.run_learning_eval --doc thread    # one document only
"""
# --- ragas import shim: ragas 0.4.3 hard-imports Google-Vertex classes that were dropped from the
#     sunset langchain-community 0.4.x. We never use them; stub them so `import ragas` succeeds. ---
import sys as _sys
import types as _types
_vm = _types.ModuleType("langchain_community.chat_models.vertexai")
_vm.ChatVertexAI = type("ChatVertexAI", (), {})
_sys.modules.setdefault("langchain_community.chat_models.vertexai", _vm)
import langchain_community.llms as _lcl          # noqa: E402
if not hasattr(_lcl, "VertexAI"):
    _lcl.VertexAI = type("VertexAI", (), {})

import argparse                                   # noqa: E402
import asyncio                                    # noqa: E402
import json                                       # noqa: E402
import logging                                    # noqa: E402
import statistics                                 # noqa: E402
from pathlib import Path                          # noqa: E402

from ragas import SingleTurnSample                # noqa: E402
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall   # noqa: E402
from ragas.llms import LangchainLLMWrapper        # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper     # noqa: E402
from langchain_openai import OpenAIEmbeddings     # noqa: E402

from backend.core.models import MODELS            # noqa: E402
from backend.core.config import settings          # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

FIXTURES = {
    "thread":    Path("backend/tests/fixtures/rag_passive_waiting.json"),
    "inflation": Path("backend/tests/fixtures/rag_inflation.json"),
}
OUT_DIR = Path("outputs/eval/learning")

# Requests ALIGNED to each fixture's own retrieval query, so the pipeline works on the topic the
# retrieved chunks actually cover (a mismatched query starves the generator and tanks the scores).
#   thread    retrieve query: "entry and exit of critical section active passive waiting"
#   inflation retrieve query: "shopping basket bread haircut car price increase over time"
# Two near-paraphrases per pipeline: on-topic AND a consistency check (similar queries should score
# similarly).
QUERIES = {
    "thread": {
        "mindmap": [
            "Create a mindmap about entry and exit of a critical section with active and passive waiting",
            "Create a mindmap of how a thread enters and exits a critical section (active vs passive waiting)",
        ],
        "summary": [
            "Summarize entry and exit of a critical section with active and passive waiting",
            "Summarize how a thread blocks on entry and how active and passive waiting differ",
        ],
    },
    "inflation": {
        "mindmap": [
            "Create a mindmap about the shopping basket and how prices increase over time",
            "Create a mindmap of how prices of a shopping basket (bread, haircut, car) rise over time",
        ],
        "summary": [
            "Summarize how a shopping basket tracks price increases over time",
            "Summarize how prices of everyday items in a shopping basket rise over time",
        ],
    },
}


# --- Generation -------------------------------------------------------------------------

def _state(rag: dict, query: str, **extra) -> dict:
    return {"user_id": "eval", "course_id": rag.get("course_id") or "test", "query": query,
            "rag_chunks": rag["chunks"], "kg_context": rag["kg_context"], **extra}


def _items() -> list[dict]:
    """Flatten QUERIES into individual eval items: one per (doc, pipeline, query)."""
    items = []
    for doc, kinds in QUERIES.items():
        for kind, queries in kinds.items():
            for i, query in enumerate(queries):
                items.append({"name": f"{doc}_{kind}_{i}", "doc": doc, "kind": kind, "query": query})
    return items


def generate(item: dict) -> dict:
    from backend.agents.learning_materials.mindmaps_generation.mindmap_graph import get_mindmap_graph
    from backend.agents.learning_materials.summaries_generation.summaries_graph import get_summaries_graph

    rag = json.loads(FIXTURES[item["doc"]].read_text(encoding="utf-8"))
    if item["kind"] == "mindmap":
        res = get_mindmap_graph().invoke(_state(rag, item["query"], mindmap_format="mermaid"))
        text = _mindmap_text(res)
    else:
        res = get_summaries_graph().invoke(_state(rag, item["query"], detail_level="medium"))
        text = _summary_text(res)

    out = {**item, "text": text,
           "contexts": [c.get("text", "") for c in rag["chunks"] if c.get("text")]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{item['name']}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [{item['name']}] {item['query'][:55]}  -> {len(text)} chars")
    return out


def _mindmap_text(result: dict) -> str:
    tree = (result.get("mindmap_tree") or {})
    lines: list[str] = []

    def walk(node, depth=0):
        if not node:
            return
        line = f"{'  ' * depth}- {node.get('label', '')}: {node.get('description', '')}".rstrip(": ")
        lines.append(line)
        for child in node.get("children") or []:
            walk(child, depth + 1)

    walk(tree.get("root"))
    return "\n".join(lines)


def _summary_text(result: dict) -> str:
    s = result.get("summary") or {}
    parts = [s.get("title", ""), s.get("introduction", "")]
    for sec in s.get("sections") or []:
        parts.append(f"{sec.get('concept', '')}: {sec.get('text', '')}".strip(": "))
    parts.append(s.get("conclusion", ""))
    return "\n\n".join(p for p in parts if p)


# --- Scoring ----------------------------------------------------------------------------

def _score(metric, sample: SingleTurnSample) -> float:
    return asyncio.run(metric.single_turn_ascore(sample))


def score(out: dict, faith, relev, ctxrec) -> dict:
    ctx = out["contexts"]
    q = out["query"]
    if out["kind"] == "mindmap":
        # Faithfulness: is the map grounded in the source? (precision)
        grounded = SingleTurnSample(user_input=q, response=out["text"], retrieved_contexts=ctx)
        # Context Recall: how much of the SOURCE does the map cover? (recall) — reference=source,
        # context=the map, so recall measures source-covered-by-map (role-reversed for coverage).
        coverage = SingleTurnSample(
            user_input=q, reference="\n\n".join(ctx), retrieved_contexts=[out["text"]])
        scores = {"faithfulness": _score(faith, grounded),
                  "context_recall": _score(ctxrec, coverage)}
    else:  # summary
        sample = SingleTurnSample(user_input=q, response=out["text"], retrieved_contexts=ctx)
        scores = {"faithfulness": _score(faith, sample),
                  "answer_relevancy": _score(relev, sample)}
    print(f"  [{out['name']}] " + "  ".join(f"{k[:5]}={v:.2f}" for k, v in scores.items()))
    return scores


def report(rows: list[dict]) -> None:
    print("\n" + "=" * 82)
    print("PER-REQUEST")
    print("=" * 82)
    print(f"{'name':<20}{'doc':<11}{'kind':<9}   scores")
    for r in rows:
        parts = "  ".join(f"{k}={v:.2f}" for k, v in r["scores"].items())
        print(f"{r['name']:<20}{r['doc']:<11}{r['kind']:<9}   {parts}")

    print("\n" + "=" * 82)
    print("AGGREGATE (mean per pipeline/metric)")
    print("=" * 82)

    def _mean(kind, key):
        vals = [r["scores"][key] for r in rows if r["kind"] == kind and key in r["scores"]]
        return statistics.mean(vals) if vals else float("nan")

    lines = [
        ("mindmap  — Faithfulness",   _mean("mindmap", "faithfulness")),
        ("mindmap  — Context Recall", _mean("mindmap", "context_recall")),
        ("summary  — Faithfulness",   _mean("summary", "faithfulness")),
        ("summary  — Answer Relevancy", _mean("summary", "answer_relevancy")),
    ]
    for label, mean in lines:
        if mean != mean:                               # skip pipelines not run this pass (nan)
            continue
        print(f"  {label:<30} {mean:.2f}  {'#' * int(round(mean * 20))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true", help="score outputs already on disk")
    ap.add_argument("--doc", choices=list(FIXTURES), help="evaluate a single document")
    ap.add_argument("--only", nargs="*", help="run only these item names, e.g. thread_mindmap_0")
    args = ap.parse_args()

    items = [it for it in _items()
             if (not args.doc or it["doc"] == args.doc)
             and (not args.only or it["name"] in args.only)]

    outs: list[dict] = []
    if args.score_only:
        for it in items:
            path = OUT_DIR / f"{it['name']}.json"
            if path.exists():
                outs.append(json.loads(path.read_text(encoding="utf-8")))
        print(f"Loaded {len(outs)} saved outputs from {OUT_DIR}\n")
    else:
        print(f"GENERATING {len(items)} request(s)\n" + "=" * 82)
        for it in items:
            outs.append(generate(it))

    if not outs:
        print("Nothing to score.")
        return

    judge = LangchainLLMWrapper(MODELS["gpt-4o"])
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.openai_api_key))
    faith = Faithfulness(llm=judge)
    relev = ResponseRelevancy(llm=judge, embeddings=embeddings)   # summary: Answer Relevancy (embeddings)
    ctxrec = LLMContextRecall(llm=judge)                          # mindmap: Context Recall (coverage)

    print("\n" + "=" * 82)
    print("SCORING with RAGAS  (judge: gpt-4o)")
    print("=" * 82)
    for out in outs:
        out["scores"] = score(out, faith, relev, ctxrec)

    report(outs)
    (OUT_DIR / "ragas_scores.json").write_text(
        json.dumps([{k: r[k] for k in ("name", "doc", "kind", "query", "scores")} for r in outs],
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {OUT_DIR / 'ragas_scores.json'}")


if __name__ == "__main__":
    main()
