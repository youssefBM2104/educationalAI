from typing import TypedDict, NotRequired, Literal

class AgentState (TypedDict):
    user_id: str
    course_id: str
    query: str
    intent: NotRequired[str]
    rag_chunks: NotRequired[list]
    kg_context: NotRequired[list]
    final_output: NotRequired[str]      

class ExamState(AgentState):

    # --- Exam request ---
    question_type: NotRequired[Literal["mcq", "essay"]]
    difficulty: NotRequired[Literal["easy", "medium", "hard"]]
    num_questions_target: NotRequired[int]

    # --- Generator output ---
    kg_path: NotRequired[list]
    generated_question: NotRequired[dict]
    correct_answer: NotRequired[str]
    chunk_bundle: NotRequired[list]
    marking_scheme: NotRequired[str]

    # --- Solver input / output ---
    chunk_pool: NotRequired[list]
    solver_answer: NotRequired[str]
    solver_reasoning: NotRequired[str]

    # --- Judge output ---
    judge_passed: NotRequired[bool]
    judge_feedback: NotRequired[dict]

    # --- Loop control ---
    iteration: NotRequired[int]
    max_iterations: NotRequired[int]

    # --- Exam set ---
    difficulty_plan: NotRequired[list] 
    current_index: NotRequired[int]
    exam_questions: NotRequired[list] # passed + failed questions
    exam_set: NotRequired[list] # final exam set