from typing import TypedDict, NotRequired, Literal

class AgentState (TypedDict):
    user_id: str
    course_id: str
    query: str
    intent: NotRequired[str]
    rag_chunks: NotRequired[list]
    kg_context: NotRequired[dict]
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

class LectureState(AgentState):
 
    # --- Lecture request ---
    output_format: NotRequired[Literal["pptx", "pdf"]]
 
    # --- Planner output ---
    lecture_plan: NotRequired[dict]     # { course_title, estimated_slides, sections[] }
 
    # --- Content Generator output ---
    lecture_content: NotRequired[list]  # list of { order, section_title, explanation,
                                        #            key_points, example, speaker_notes }
 
    # --- Slide Builder output ---
    lecture_slides: NotRequired[dict]   # { slides: [ { slide_number, type, title,
                                        #               bullets, visual_hint, speaker_notes } ] }
 
    # --- Export output ---
    lecture_output_path: NotRequired[str]   # path to the generated .pptx or .pdf file
    lecture_output_bytes: NotRequired[bytes]  # raw bytes for API response if needed

class LearningMaterialsState(AgentState):
 
    # --- Request ---
    mindmap_format: NotRequired[Literal["mermaid", "markmap"]]       # mindmap only
    detail_level: NotRequired[Literal["short", "medium", "long"]]   # summary only
 
    # --- Mindmap pipeline ---
    mindmap_tree: NotRequired[dict]          # nested MindmapNode tree from MindmapAgent
    mindmap_output_path: NotRequired[str]    # path to .md or .html on disk
    mindmap_output_bytes: NotRequired[bytes]
 
    # --- Summary pipeline ---
    extracted_ideas: NotRequired[dict]       # ExtractedIdeas dict from ExtractorAgent
    summary: NotRequired[dict]              # Summary dict from WriterAgent

class TutoringState(AgentState):

    # --- Classification ---
    bloom_level: NotRequired[int]            # N (1..6) from BloomClassifier

    # --- Socratic loop position (persists across turns) ---
    phase: NotRequired[Literal["await_sub_answer", "await_bridge", "done"]]
    current_level: NotRequired[int]          # k, during scaffold descent / climb back
    sub_question: NotRequired[str]
    expected_answer: NotRequired[str]
    asked_questions: NotRequired[list]       # every sub-question already asked -> never repeat
    fail_streak: NotRequired[dict]           # {level: consecutive fails} -> safety cap = 3
    next_step: NotRequired[str]              # controller hint: "ask" | "bridge"
    teach_note: NotRequired[str]             # direct explanation to prepend (safety cap)

    # --- Bridge ---
    bridge_question: NotRequired[str]

    # --- Per-turn student input ---
    student_answer: NotRequired[str]

    # --- Evaluation / diagnosis ---
    eval_result: NotRequired[str]            # "pass" | "fail"
    eval_notes: NotRequired[str]
    diagnosis: NotRequired[dict]             # {status, specific_error}

    # --- Output shown to the student this turn ---
    tutor_message: NotRequired[str]

class MainState(ExamState, LectureState, LearningMaterialsState, TutoringState):
    pass