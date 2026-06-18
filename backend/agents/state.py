from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class AgentState (TypedDict):
    user_id: str
    query: str
    intent: str
    rag_chunks: list
    #kg_nodes : list           # Use the cover concpets of the chunks
    final_output: str

    # --- Exam ---

    generated_question: str    # None for Tutoring
    grading_rubric : str
    correct_answer : str
    exam_questions: list
    num_questions_target : int

    structured_reasoning : str
    solver_answer: str         # None for Lecture
            
    judge_score: float         # None for Tutoring
    decision : str

    iteration: int             

    # --- Lecture ---
    slide_outline: list        # None for Exam

    # --- Tutoring ---
    student_profile: dict      # None for Lecture
    learning_gaps: list        