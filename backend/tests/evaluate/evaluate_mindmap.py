"""
Evaluation for the mindmap pipeline (mindmap_agent -> mindmap_export).

Does two things:
1. RAGAS (faithfulness / answer_relevancy) on each node, using the rag_chunks
   that were fed to the agent -> catches hallucinated descriptions.
2. Custom structural checks (RAGAS doesn't cover this):
   - number of main branches (3 to 6, per the prompt in mindmap_agent.py)
   - no concept repeated at multiple levels of the tree
   - labels actually come from the KG (not invented)

Saves each run to a unique, timestamped JSON file (no overwriting, unlike
mindmap_export.py which names files after the query alone).

The RAGAS judge LLM is the project's own model (MODELS["llama31"]) wrapped in
LangchainLLMWrapper, so no external API key is required.

Temperature is varied per run by monkeypatching the module-level `llm` used
inside mindmap_agent.py, so we don't need to touch the production agent code.

Run (from repo root, inside the venv):
    python -m backend.tests.evaluate.evaluate_mindmap
    python -m backend.tests.evaluate.evaluate_mindmap --temperatures 0.0 0.7 1.2
"""
import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy

from backend.core.models import MODELS
from FlagEmbedding import BGEM3FlagModel
from backend.agents.learning_materials.mindmaps_generation.mindmap_graph import get_mindmap_graph
from backend.agents.learning_materials.mindmaps_generation import mindmap_agent as mindmap_agent_module

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
EVAL_OUTPUT_DIR = Path(__file__).parent / "eval_outputs" / "mindmaps"

# ---------------------------------------------------------------------------
# RAGAS judge setup - reuse the project's own model instead of an external API.
# Judge temperature is pinned to 0 for consistent scoring, independent of
# whatever temperature is being tested on the agent itself.
# ---------------------------------------------------------------------------
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from backend.core.config import settings

# IMPORTANT: do NOT use MODELS["llama31"].bind(temperature=0.0) here.
# .bind() wraps the model in a RunnableBinding, which has no "temperature"
# pydantic field. RAGAS reads/writes llm.temperature directly internally to
# vary sampling across statements, and crashes with:
#   ValueError("_ChatModelBinding" object has no field "temperature")
# A real ChatNVIDIA instance (not a binding) keeps that field, so RAGAS can
# use it.
JUDGE_LLM = LangchainLLMWrapper(
    ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        api_key=settings.nim_api_key,
        temperature=0.0,
        max_completion_tokens=4096,
    )
)

# answer_relevancy needs an embeddings model to compare the generated
# question against the original one. Wire this up if the project exposes one
# in MODELS (e.g. MODELS["embeddings"]); otherwise fall back to faithfulness
# only and log a warning once at import time.
_EMBEDDINGS_MODEL = model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=settings.embedding_device)
if _EMBEDDINGS_MODEL is not None:
    logger.info("BGE-M3 model loaded")
    from ragas.embeddings import LangchainEmbeddingsWrapper
    JUDGE_EMBEDDINGS = LangchainEmbeddingsWrapper(_EMBEDDINGS_MODEL)
    METRICS = [faithfulness, answer_relevancy]
else:
    JUDGE_EMBEDDINGS = None
    METRICS = [faithfulness]
    logger.warning(
        "No 'embeddings' model found in MODELS - skipping answer_relevancy, "
        "running faithfulness only. Add an embeddings model to MODELS to enable it."
    )


# ---------------------------------------------------------------------------
# Test cases: (fixture, query, list of temperatures to try on the agent LLM)
# Add rows here to cover more fixtures / phrasings / temperature ranges.
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    fixture: str
    query: str
    temperatures: list[float] = field(default_factory=lambda: [0.7])


DEFAULT_CASES = [
    TestCase("rag_passive_waiting.json", "Create a mindmap about thread synchronization", temperatures=[0.0, 0.7, 1.2]),
    TestCase("rag_passive_waiting.json", "Mindmap of thread synchronization", temperatures=[0.7]),
]


# ---------------------------------------------------------------------------
# State loading, same shape as test_mindmap_generation.py
# ---------------------------------------------------------------------------
def load_state(fixture_name: str, query: str, fmt: str = "mermaid") -> dict:
    rag = json.loads((FIXTURE_DIR / fixture_name).read_text(encoding="utf-8"))
    return {
        "user_id": "eval-user",
        "course_id": rag.get("course_id", "test"),
        "query": query,
        "mindmap_format": fmt,
        "rag_chunks": rag["chunks"],
        "kg_context": rag["kg_context"],
    }


def _format_chunks_text(chunks: list) -> str:
    return "\n\n".join(
        f"[{c.get('document_id')}#{c.get('chunk_index')}] {c.get('text', '')}" for c in chunks
    )


def _kg_concepts(kg: dict | None) -> set[str]:
    kg = kg or {}
    concepts = kg.get("concepts")
    if concepts:
        return {str(c) for c in concepts}
    concepts = set()
    for r in kg.get("relations", []):
        for key in ("from", "to"):
            if r.get(key):
                concepts.add(str(r[key]))
    return concepts


# ---------------------------------------------------------------------------
# Step 1 - Flatten the tree into rows RAGAS can score
# ---------------------------------------------------------------------------
def flatten_tree(node: dict, contexts: list[str], parent_label: str | None = None, rows: list | None = None) -> list[dict]:
    rows = rows if rows is not None else []
    question = f"Explain '{node['label']}'"
    if parent_label:
        question += f" in relation to '{parent_label}'"
    rows.append({
        "question": question,
        "answer": node.get("description", ""),
        "contexts": contexts,
        "node_label": node["label"],
    })
    for child in node.get("children", []):
        flatten_tree(child, contexts, node["label"], rows)
    return rows


# ---------------------------------------------------------------------------
# Step 2 - RAGAS, judged by the project's own llama31 model
# ---------------------------------------------------------------------------
def run_ragas(rows: list[dict]):
    ds = Dataset.from_list([
        {"question": r["question"], "answer": r["answer"], "contexts": r["contexts"]}
        for r in rows
    ])
    eval_kwargs = {"llm": JUDGE_LLM}
    if JUDGE_EMBEDDINGS is not None:
        eval_kwargs["embeddings"] = JUDGE_EMBEDDINGS

    result = evaluate(ds, metrics=METRICS, **eval_kwargs)
    df = result.to_pandas()
    df["node_label"] = [r["node_label"] for r in rows]
    return df


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace so 'Mutex.' == 'mutex' == ' Mutex '."""
    import re
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _label_matches_kg(label: str, kg_concepts: set[str]) -> bool:
    """
    Tolerant match: exact (normalized) match, or containment either way
    (handles the agent rephrasing 'Typical Sequence' as
    'Typical Sequence in the Use of Mutex Lock'). This intentionally
    under-flags borderline rewordings; use STRICT_KG_MATCH=True below if you
    want the harsh exact-match version for a stricter audit pass.
    """
    norm_label = _normalize(label)
    norm_concepts = {_normalize(c) for c in kg_concepts}
    if norm_label in norm_concepts:
        return True
    return any(
        norm_label in c or c in norm_label
        for c in norm_concepts
        if c  # skip empty strings
    )


STRICT_KG_MATCH = False  # flip to True to fall back to exact string matching


# ---------------------------------------------------------------------------
# Step 3 - Structural checks (RAGAS doesn't do this)
# ---------------------------------------------------------------------------
def check_structure(tree: dict, kg_concepts: set[str]) -> list[str]:
    issues = []
    root = tree["root"]

    n_branches = len(root.get("children", []))
    if not (3 <= n_branches <= 6):
        issues.append(f"branches={n_branches}, outside the expected [3,6] range")

    seen: set[str] = set()

    def walk(node: dict, depth: int):
        label = node["label"]
        if label in seen:
            issues.append(f"concept repeated at multiple levels: '{label}'")
        seen.add(label)
        if depth > 0 and kg_concepts:
            matched = (label in kg_concepts) if STRICT_KG_MATCH else _label_matches_kg(label, kg_concepts)
            if not matched:
                issues.append(f"concept not found in KG (possible hallucination): '{label}'")
        for child in node.get("children", []):
            walk(child, depth + 1)

    walk(root, 0)
    return issues


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_one(case: TestCase, temperature: float, run_idx: int) -> dict:
    # Swap the agent's LLM for a version with a different temperature, without
    # touching mindmap_agent.py itself. mindmap_agent() reads the module-level
    # `llm` name at call time, so patching the module attribute is enough.
    mindmap_agent_module.llm = MODELS["llama31"].bind(temperature=temperature)

    state = load_state(case.fixture, case.query)
    result = get_mindmap_graph().invoke(state)
    tree = result.get("mindmap_tree") or {}

    contexts = [_format_chunks_text(state["rag_chunks"])]
    rows = flatten_tree(tree["root"], contexts)
    ragas_df = run_ragas(rows)

    kg_concepts = _kg_concepts(state["kg_context"])
    logger.info("  KG concepts available: %s", sorted(kg_concepts))
    structure_issues = check_structure(tree, kg_concepts)

    # Unique file per run: no overwriting, unlike mindmap_export.py
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_query = case.query.replace(" ", "_")[:40]
    out_path = EVAL_OUTPUT_DIR / f"{safe_query}_temp{temperature}_{ts}_run{run_idx}.json"
    out_path.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "case": case,
        "temperature": temperature,
        "run_idx": run_idx,
        "tree_path": str(out_path),
        "ragas_df": ragas_df,
        "structure_issues": structure_issues,
        "n_nodes": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=None,
        help="Override the temperatures list for every test case (e.g. --temperatures 0.0 0.7 1.2)",
    )
    args = parser.parse_args()

    all_results = []
    for case in DEFAULT_CASES:
        temps = args.temperatures or case.temperatures
        for run_idx, temp in enumerate(temps):
            logger.info("=== Case: %r (fixture=%s) temperature=%.2f run %d/%d ===",
                        case.query, case.fixture, temp, run_idx + 1, len(temps))
            res = run_one(case, temp, run_idx)
            all_results.append(res)

            mean_faith = res["ragas_df"]["faithfulness"].mean()
            mean_rel = (
                res["ragas_df"]["answer_relevancy"].mean()
                if "answer_relevancy" in res["ragas_df"]
                else float("nan")
            )
            logger.info(
                "  nodes=%d  faithfulness_mean=%.3f  answer_relevancy_mean=%.3f  structure_issues=%d",
                res["n_nodes"], mean_faith, mean_rel, len(res["structure_issues"]),
            )
            for issue in res["structure_issues"]:
                logger.warning("  STRUCTURE ISSUE: %s", issue)

            worst = res["ragas_df"].sort_values("faithfulness").head(3)
            for _, row in worst.iterrows():
                if row["faithfulness"] < 0.7:
                    logger.warning(
                        "  LOW FAITHFULNESS node=%r score=%.2f",
                        row["node_label"], row["faithfulness"],
                    )

    # Global summary
    print("\n=== SUMMARY ===")
    for res in all_results:
        rel = (
            res["ragas_df"]["answer_relevancy"].mean()
            if "answer_relevancy" in res["ragas_df"]
            else float("nan")
        )
        print(
            f"{res['case'].query!r}  temp={res['temperature']}  run={res['run_idx']}  "
            f"faithfulness={res['ragas_df']['faithfulness'].mean():.3f}  "
            f"relevancy={rel:.3f}  "
            f"issues={len(res['structure_issues'])}  "
            f"saved={res['tree_path']}"
        )


if __name__ == "__main__":
    main()