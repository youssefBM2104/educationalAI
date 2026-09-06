"""
G-Eval rubrics for exam generation.

Two families of rubric:
  1. Project rubrics — each targets a failure mode we actually observed in generated exams
     (naturalness/scaffolding leak, groundedness, distractor quality, difficulty), not a
     generic "is it good?" score.
  2. QGEval rubrics — the question-quality dimensions from QGEval (Fu et al., EMNLP 2024,
     arXiv:2406.05707), ported onto our MCQ setting as G-Eval rubrics. QGEval itself is a
     META-evaluation benchmark (it ranks automatic metrics against human scores on SQuAD/
     HotpotQA short-answer QG) and ships no drop-in scorer for our data, so we reuse its
     peer-reviewed DIMENSION definitions via our own LLM judge. Of its 7 dimensions we add
     the five not already covered here — fluency, clarity, conciseness, answerability,
     answer_consistency; the remaining two (relevance, consistency) are already covered by
     groundedness / question_naturalness, so they are not duplicated.

`fields` controls what the judge is allowed to see — withholding the knowledge graph from the
naturalness judge is deliberate: a question that is only solvable by staring at the graph
should score badly, and the judge can only detect that if it, like a student, cannot see the
graph either.
"""
from backend.eval.judge import GEval

# Observed failure: "What is the sequence of concepts that leads to X?" with options that are
# literal KG paths (A -> B -> C). A student never sees the graph, so the question is unusable.
# Refinement: penalise the *pipeline's* concept-graph scaffolding leaking to the surface — NOT
# ordinary subject terms that merely contain the word "graph"/"path" (e.g. a "waiting-for-
# resources graph" is real OS content and must pass).
question_naturalness = GEval(
    name="Question Naturalness",
    criteria=(
        "The question must read like a real exam question a student answers from subject "
        "knowledge. It must NOT expose the pipeline's internal concept-graph SCAFFOLDING — "
        "options that are arrow chains of concept names, or wording that tells the student to "
        "follow a 'concept path' / 'sequence of concepts' they were never shown. "
        "IMPORTANT: subject-matter terms that merely contain the words 'graph', 'path', 'node' "
        "or 'tree' (e.g. 'waiting-for-resources graph', 'call graph', 'shortest-path algorithm') "
        "are legitimate course content — do NOT penalise them."
    ),
    evaluation_steps=[
        "Are any answer options literal arrow chains of concept names (e.g. 'A -> B -> C') "
        "instead of real, self-contained statements? If so, penalise heavily.",
        "Does the wording refer to the SCAFFOLDING itself — 'the concept path', 'the sequence of "
        "concepts', 'traverse the graph of concepts' — i.e. a structure the student never saw? Penalise.",
        "Distinguish that from legitimate SUBJECT terminology: a 'waiting-for-resources graph' or a "
        "'shortest-path problem' is real course content and must NOT be penalised.",
        "Could a student who studied the material, but never saw the pipeline's concept graph, read "
        "and answer this question? If yes, it is natural.",
    ],
    fields=["query", "question", "options"],   # deliberately NOT shown the kg_path / KG
)

# Observed risk: the generator inventing facts that are not in the retrieved chunks.
groundedness = GEval(
    name="Groundedness",
    criteria=(
        "Every claim in the question, the correct answer and the explanation must be supported "
        "by the source chunks. No invented facts, no outside knowledge."
    ),
    evaluation_steps=[
        "List the factual claims made by the question, the correct answer and the explanation.",
        "For each claim, find the supporting sentence in the source chunks.",
        "Penalise any claim that cannot be traced back to the chunks.",
        "A claim that merely combines two chunks (multi-hop) is still grounded — do not penalise it.",
    ],
    fields=["question", "options", "correct_answer", "explanation", "chunks"],
)

# The whole point of the near-miss design: distractors must be tempting, not obviously silly.
distractor_quality = GEval(
    name="Distractor Quality",
    criteria=(
        "The three wrong options must be plausible enough to tempt a student who has only "
        "partially understood the material, yet clearly wrong according to the source material. "
        "Exactly one option may be correct."
    ),
    evaluation_steps=[
        "Check that exactly one option is correct — penalise severely if two options are defensible.",
        "For each wrong option, judge whether a partially-prepared student could plausibly pick it.",
        "Penalise distractors that are obviously absurd, off-topic, or trivially eliminated.",
        "Reward distractors that are near-misses: closely related concepts that diverge on one step.",
    ],
    fields=["question", "options", "correct_answer", "chunks", "kg_relations"],
)

# Observed failure: an "easy" question generated from a 7-concept path, and a "hard" one that
# is really a single-chunk lookup. Path length alone does not make a question hard.
difficulty_appropriateness = GEval(
    name="Difficulty Appropriateness",
    criteria=(
        "The cognitive demand of the question must match its declared difficulty label. "
        "'easy' = recall or a single-step lookup. 'medium' = combining a few facts. "
        "'hard' = genuine multi-step reasoning across several concepts."
    ),
    evaluation_steps=[
        "Read the declared difficulty label.",
        "Estimate how many reasoning steps a student needs, ignoring how long the kg_path is.",
        "Penalise a 'hard' question that is answerable by looking up one sentence.",
        "Penalise an 'easy' question that in fact requires chaining several concepts.",
    ],
    fields=["difficulty", "question", "options", "correct_answer", "chunks"],
)


# --- QGEval dimensions (Fu et al., EMNLP 2024, arXiv:2406.05707) --------------------------
# Ported from QGEval's human-judged dimensions. Fluency/clarity/conciseness judge the LANGUAGE
# of the stem + options only (content correctness is judged by groundedness/distractor above).
# Answerability/answer_consistency judge that the question is answerable from the chunks and that
# the keyed option is genuinely its answer — QGEval reports these two are where QG models fail most.

fluency = GEval(
    name="Fluency",
    criteria=(
        "The question and every option must be grammatically correct and read fluently, as "
        "written by a native speaker. Judge the LANGUAGE only, not whether the content is correct."
    ),
    evaluation_steps=[
        "Read the question stem and each option for grammar errors, awkward phrasing, or typos.",
        "Penalise broken grammar, missing words, or wording no fluent writer would produce.",
        "Do NOT penalise a fluent-but-factually-wrong option — correctness is judged elsewhere.",
    ],
    fields=["question", "options"],
)

clarity = GEval(
    name="Clarity",
    criteria=(
        "The question must express a single, unambiguous meaning — a student should know exactly "
        "what is being asked without guessing. No vague referents, no double-barrelled asks."
    ),
    evaluation_steps=[
        "Decide whether the question has one clear interpretation.",
        "Penalise ambiguity: unclear pronouns/referents, several things asked at once, or a stem "
        "that could be read multiple ways.",
        "A clear question lets the student identify the single intended task.",
    ],
    fields=["question", "options"],
)

conciseness = GEval(
    name="Conciseness",
    criteria=(
        "The question must be as short as it can be while staying complete — no redundant, "
        "repeated, or irrelevant words. Context genuinely needed to answer is NOT redundancy."
    ),
    evaluation_steps=[
        "Check for padding: repeated information, filler clauses, context already implied.",
        "Penalise verbosity that does not change what is being asked.",
        "Do NOT penalise scenario detail that is actually required to answer the question.",
    ],
    fields=["question", "options"],
)

answerability = GEval(
    name="Answerability",
    criteria=(
        "The question must be answerable from the source chunks, and exactly one option must be "
        "the answer the chunks support. A question with no supported option, or several equally "
        "supported, fails."
    ),
    evaluation_steps=[
        "Try to answer the question using ONLY the source chunks.",
        "Check that exactly one option is supported by the chunks as the answer.",
        "Penalise if the chunks do not settle the answer, or if more than one option is defensible.",
    ],
    fields=["question", "options", "correct_answer", "chunks"],
)

answer_consistency = GEval(
    name="Answer Consistency",
    criteria=(
        "The keyed correct option must genuinely be the correct answer to the question, consistent "
        "with the chunks and the explanation — no key/answer mismatch."
    ),
    evaluation_steps=[
        "Answer the question yourself from the chunks, then find which option that matches.",
        "Penalise a mismatch between the question's real answer and the option marked correct.",
        "Check the explanation actually justifies the keyed option, not a different one.",
    ],
    fields=["question", "options", "correct_answer", "explanation", "chunks"],
)


# Fluency/clarity/conciseness are language-quality checks that apply to essays too; answerability
# and answer_consistency are MCQ-shaped (they reason over the keyed option), so essays keep the
# groundedness check for the same "is the model answer derivable" concern.
_QGEVAL_STEM = [fluency, clarity, conciseness]

MCQ_METRICS = [
    question_naturalness, groundedness, distractor_quality, difficulty_appropriateness,
    *_QGEVAL_STEM, answerability, answer_consistency,
]
ESSAY_METRICS = [
    question_naturalness, groundedness, difficulty_appropriateness, *_QGEVAL_STEM,
]


def metrics_for(question_type: str) -> list[GEval]:
    return MCQ_METRICS if question_type == "mcq" else ESSAY_METRICS
