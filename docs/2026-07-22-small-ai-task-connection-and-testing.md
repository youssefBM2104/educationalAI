# Session Report — 2026-07-22
## Connecting the fine-tuned small AI (edu-qwen-v1) to QA, conversation summary, and learning materials

Branch: `feat/use_small_ai`

---

## 1. What Was Done

Goal: connect the fine-tuned local model (Qwen3-1.7B + LoRA adapter `edu-qwen-v1`) to the
three tasks it's meant to handle — QA, conversation summary, and learning materials
(mindmap + summary) — then test it on each and record how it behaves.

### 1.1 Bug fixes / wiring

- **`backend/local_ai/tasks/q&a.py` → renamed `qa.py`** — `&` is not valid in a Python
  module name; every import of this file (`task_router.py`, `api/routes.py`) was already
  written as `local_ai.tasks.qa`, so the task crashed on import before this fix.
- **Import convention standardized on `backend.xxx`** — `local_model.py` already used
  `from backend.core.config import settings`, but `model_manager.py`, `task_router.py`,
  `api/routes.py`, `tasks/qa.py`, `tasks/conversation_summary.py` all used bare
  `from local_ai.xxx import ...`. Neither run convention (repo root vs `backend/` as cwd)
  could satisfy both at once. Fixed all of them to use `backend.local_ai.xxx`, matching
  `agents/` and the rest of the project — run everything from the repo root.
- **`local_model.py` wrapped in `ChatHuggingFace`** — was returning a bare
  `HuggingFacePipeline` (plain text-completion LLM, no `.with_structured_output()`).
  The three learning_materials agents all call `.with_structured_output(...)`, so this
  is required for them to work with the small model at all.
- **`mindmap_agent.py`, `extractor_agent.py`, `writer_agent.py`** — switched
  `llm = MODELS["llama31"]` → `llm = MODEL["small_ai"]` (llama31 line kept commented for
  easy rollback).
- **Missing `adapter_config.json` + tokenizer files** — `trained_model/edu-qwen-v1/` only
  had `adapter_model.safetensors`, `Modelfile`, `chat_template.jinja` (looks like it was
  exported for `ollama create`, not for direct `transformers`/`peft` loading). Root cause:
  `.gitignore` had a blanket `*.json` rule that silently excluded everything `train.py`
  writes via `tokenizer.save_pretrained()` / `trainer.save_model()`.
  - Tokenizer: now loaded from the base model (`Qwen/Qwen3-1.7B`) instead of
    `adapter_path` — the LoRA only targets attention projections (q/v/k/o_proj), never
    touches the vocab/embeddings, so this is identical to the original tokenizer.
  - `adapter_config.json`: reconstructed by hand from the exact LoRA hyperparameters
    hard-coded in `train.py` / `core/config.py` (r=16, alpha=32, dropout=0.05,
    target_modules=q/v/k/o_proj, task_type=CAUSAL_LM, bias=none). No `.env` override
    exists for these values, so this should match what training actually used.
    **If the original training output is ever recovered (e.g. from the vast.ai instance
    that ran it), replace this reconstructed file with the real one.**
  - `.gitignore` fixed with `!backend/trained_model/**/*.json` so this can't happen again.
- **`return_full_text=False`** added to the HF `pipeline(...)` call — without it, the
  pipeline echoes the whole prompt (system prompt + chat template tokens) back as part
  of the "answer". Found during the first QA test run.

### 1.2 New test scripts (`backend/tests/`, run from repo root)

- `test_qa_small_ai.py` — `python -m backend.tests.test_qa_small_ai`
- `test_conversation_summary_small_ai.py` — `python -m backend.tests.test_conversation_summary_small_ai`
- (learning materials already had `test_mindmap_generation.py` / `test_summaries_generation.py` —
  now exercise the small model automatically since the agents were rewired)

All run locally on CPU (tested on Tim's PC, no GPU, no vast.ai) — `device_map="auto"`
falls back to CPU automatically. First run downloads `Qwen/Qwen3-1.7B` (~3-4GB) from
Hugging Face if not already cached.

---

## 2. Test Results

### 2.1 QA (`test_qa_small_ai.py`)

Tested on Tim's PC (CPU). First run (before the `return_full_text` fix) showed the same
pattern below but with the whole prompt echoed back — re-run after the fix confirms
clean output and the same behavior.

| Question | Context given? | Result | Notes |
|---|---|---|---|
| What is the difference between a mutex and a semaphore? | No | **Good** | "A mutex is a software lock that prevents multiple threads from accessing a resource simultaneously... A semaphore... is a signaling mechanism... through a count that tracks available resources." Correct, matches the OS/concurrency domain the dataset covers. |
| What is passive waiting? | No | **Hallucinated (consistently)** | Run 1: "waiting in a room without doing anything but sitting there." Run 2 (different sample, same temperature=0.4): "the act of waiting for a response to a question without actively seeking it." Both wrong, and wrong in *different* ways — not a one-off fluke. Real definition (from the course fixtures): a thread relinquishes the CPU / becomes non-schedulable while waiting, as opposed to active waiting which busy-loops on a condition. |
| According to this excerpt, what happens when a thread exits a critical section? (context provided, excerpt about CS exit) | Yes | **Good** | "When a thread exits a critical section, it must recheck the condition to enter CS... If passive waiting is used, the system will wake up one thread that is waiting or sleeping." Correctly grounded in the provided context, and notably gets "passive waiting" *right* here — when it's handed to it in the context, vs. hallucinating when asked to recall it from parametric memory. |
| What is the capital of France? (off-topic control) | No | **Good** | Correct, general knowledge preserved — no obvious catastrophic forgetting. |

**Conclusion:** the small model is reliable when a grounding context is provided
(RAG-style) — confirmed twice on the same context-grounded question, including a case
where it got "passive waiting" right *only* when the context gave it the definition
verbatim, but hallucinated the same concept confidently (with different wrong wording
each run) when asked to recall it from training alone. This strongly suggests the "no
context" QA path can't be trusted as-is for this model size, and/or the training dataset
doesn't cover "passive waiting" specifically well enough, even though the domain
(thread synchronization) is clearly represented (mutex/semaphore answers are solid).
Worth cross-referencing `qa_dataset.json` to check actual coverage of this concept
before concluding it's a capacity limit rather than a data gap.

### 2.2 Conversation summary (`test_conversation_summary_small_ai.py`)

Tested on Tim's PC (CPU), default transcript (mutex/semaphore/deadlock tutoring exchange).

**RECAP: Good.** "The discussion clarified distinctions between mutexes and semaphores,
noting that a mutex acts as a semaphore with a maximum count of 1 while enforcing
ownership through unlocking. It also explained how deadlocks occur when threads hold
mutually required locks and cannot proceed due to circular dependencies." — accurate,
coherent, correctly captures the whole exchange.

**TOPICS COVERED / OPEN QUESTIONS: Empty.** The model stopped generating right after the
RECAP section instead of continuing with the rest of the requested
`RECAP: / TOPICS COVERED: / OPEN QUESTIONS:` template (see prompt in
`tasks/conversation_summary.py`). Raw output confirms it — generation just ends after
the recap paragraph.

**Interpretation:** not a comprehension problem (the recap itself is accurate) — it's an
instruction-following problem: the small model doesn't reliably complete a multi-section
format it's told to follow, it tends to stop after the first section. Same underlying
pattern as the QA hallucination-without-context issue — the model is fine on raw content
generation, less reliable on following precise structural instructions. This is a bigger
concern for learning materials (mindmap/summary), which use `with_structured_output`
under a schema, since that fully depends on strict format-following.

### 2.3 Learning materials — mindmap (`test_mindmap_generation.py`)

First attempt crashed with `NotImplementedError: Pydantic schema is not supported for
function calling` — confirmed as an open upstream bug in `langchain-huggingface`
(langchain-ai/langchain#32197: Pydantic support for `with_structured_output` is simply
not implemented under the default `method="function_calling"`). Worked around by passing
`method="json_schema"` explicitly in all three learning_materials agents.

After the workaround: no more crash on the *schema binding*, but **the model still
doesn't produce JSON** — it answers in plain prose instead:

> "The root node is 'Thread Synchronization', with main branches 'Deadlock' and
> 'Locking'. 'Deadlock' has sub-nodes 'Deadlock Situation' and 'Cycle in the waiting for
> resources graph', while 'Locking' includes 'mutex' and 'Semaphores'."

Content-wise this is actually **correct** — those are exactly the right concepts from the
KG fixture. But `PydanticOutputParser` fails on it (`JSONDecodeError: Expecting value`),
so the pipeline crashes with `OutputParserException` and no mindmap is produced.

**Root cause:** `ChatHuggingFace` + `method="json_schema"` has no real constrained/guided
decoding (unlike vLLM's `guided_json` or the `outlines` library) — it just stuffs the
JSON schema into the prompt as an instruction and hopes the model complies. A 1.7B model
doesn't reliably comply with that instruction.

### 2.4 Learning materials — summary (`test_summaries_generation.py`)

Same failure mode, confirmed on the extractor step (writer never got reached):

> "The passage discusses thread synchronization strategies, including entry/exit of
> critical sections, locks like mutexes, deadlock prevention through resource ordering,
> and the role of operating systems in managing CPU allocation. It also highlights how
> modern programming languages like Rust enforce ownership rules to avoid data races and
> deadlocks, while emphasizing the importance of proper resource management in concurrent
> programs."

Again, good content, zero JSON. Same `OutputParserException`. This confirms the pattern
from 2.3 is systemic, not a one-off on the mindmap prompt specifically — both
learning_materials pipelines fail identically.

---

## 3. Overall Conclusion

A clear pattern across all three tasks: **the small model's raw content quality is
consistently good — it's format/instruction-following that breaks down as soon as the
task requires a strict structure.**

| Task | Free-text content | Strict format compliance |
|---|---|---|
| QA (no context) | Inconsistent — good on well-covered concepts (mutex/semaphore), hallucinates confidently on others (passive waiting) | N/A (free text) |
| QA (with context) | Good, consistently | N/A (free text) |
| Conversation summary | Good (RECAP accurate) | **Fails** — stops after first section, ignores rest of template |
| Mindmap (structured) | Good (correct concepts identified) | **Fails completely** — plain prose instead of JSON, pipeline crashes |
| Summary/extractor (structured) | Good (correct concepts identified) | **Fails completely** — same as mindmap |

**Practical implication for this branch:** as wired today (LangChain `with_structured_output`
on top of a raw HF `transformers` pipeline), the small model is usable for **QA with a
provided context** and, with the caveat about the missing sections, usable for a
**loose/free-text conversation recap** (the RECAP field alone works — TOPICS COVERED /
OPEN QUESTIONS would need the code to tolerate a missing section rather than require it).
**Learning materials (mindmap + summary) cannot run on this model in its current form** —
the pipeline crashes 100% of the time in these tests, not intermittently.

To make learning materials actually work with the small model would need either:
- Real constrained/guided decoding (e.g. serving via vLLM with `guided_json`, or the
  `outlines` library) instead of `langchain-huggingface`'s prompt-based JSON mode, or
- A much more forceful few-shot JSON prompt + a repair/retry loop, or
- Accepting llama31 stays on learning materials and only routing QA (+ maybe a
  simplified conversation summary) to the small model.

This last option is a real decision Tim needs to make, not something to default into —
flagged as open below.

## 4. Open Questions / Risks

- **Decision needed:** does the small model stay wired into learning_materials given it
  cannot produce usable output there today, or does it get reverted to llama31 for those
  two agents (uncomment the `#llm = MODELS["llama31"]` line, revert the
  `method="json_schema"` additions)? QA and conversation summary are the two tasks that
  actually showed a working path.
- `adapter_config.json` is a reconstruction, not the original file — flagged in §1.1.
  If the real training output is ever recovered, replace it.
- Conversation summary: the parser (`tasks/conversation_summary.py`) currently produces
  empty `topics_covered`/`open_questions` lists rather than failing outright, which is
  silently misleading (looks "successful" but is incomplete) — worth deciding whether to
  surface this as a warning/error to the caller instead.
- QA hallucination pattern needs more samples before concluding it's a dataset coverage
  gap vs. a model capacity limit — worth cross-referencing `qa_dataset.json` for
  "passive waiting" coverage specifically.
