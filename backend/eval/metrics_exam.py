"""
G-Eval rubrics for exam generation.

Each rubric targets a failure mode we actually observed in generated exams, not a generic
"is it good?" score. `fields` controls what the judge is allowed to see — withholding the
knowledge graph from the naturalness judge is deliberate: a question that is only solvable
by staring at the graph should score badly, and the judge can only detect that if it, like
a student, cannot see the graph either.
"""
from backend.eval.judge import GEval

# Observed failure: "What is the sequence of concepts that leads to X?" with options that are
# literal KG paths (A -> B -> C). A student never sees the graph, so the question is unusable.
question_naturalness = GEval(
    name="Question Naturalness",
    criteria=(
        "The question must read like a real exam question a student answers from subject "
        "knowledge. It must NOT expose the internal knowledge-graph structure, and must not "
        "be answerable only by inspecting a concept graph the student cannot see."
    ),
    evaluation_steps=[
        "Does the question mention a 'path', a 'sequence of concepts', or the knowledge graph itself?",
        "Are the answer options literal chains of concepts (A -> B -> C) rather than real answers?",
        "Could a student who has studied the material, but has never seen a concept graph, answer it?",
        "Penalise heavily any question that is really a graph-traversal puzzle rather than a subject question.",
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


MCQ_METRICS = [question_naturalness, groundedness, distractor_quality, difficulty_appropriateness]
ESSAY_METRICS = [question_naturalness, groundedness, difficulty_appropriateness]


def metrics_for(question_type: str) -> list[GEval]:
    return MCQ_METRICS if question_type == "mcq" else ESSAY_METRICS
