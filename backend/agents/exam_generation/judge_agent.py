from backend.core.models import MODELS
from langchain_core.messages import SystemMessage
import json

llm = MODELS["nemotron"]

JUDGE_PROMPT = """
You are an expert evaluator agent in a multi-agent exam system.
Your role is to assess both the quality of a generated exam question AND the quality of the student's answer.

You will receive:
- The generated question
- The correct answer and grading rubric (produced by the Generator)
- The student's final answer and their step-by-step reasoning (produced by the Solver)
- The original context chunks the question was based on

Evaluate the following criteria:

1. Question clarity       : is the question unambiguous, precise, and well-formulated ?
2. Question relevance     : is the question strictly grounded in the provided context ? No invented facts ?
3. Answer correctness     : does the student's final answer match the correct answer and satisfy the rubric ?
4. Reasoning quality      : is the student's reasoning logical, coherent, and consistent with their final answer ?
5. Overall difficulty     : is the difficulty level moderate to hard and appropriate for an exam ?

Scoring rules:
- Each score is between 0.0 and 1.0
- overall_score is the weighted average: clarity(0.2) + relevance(0.25) + correctness(0.3) + reasoning(0.15) + difficulty(0.1)
- decision must be "retry" if overall_score is below 0.75, otherwise "done"
- If decision is "retry", feedback must clearly explain what the Generator should fix

Output a valid JSON and nothing else:
{{
    "question_clarity_score": <0.0 to 1.0>,
    "question_relevance_score": <0.0 to 1.0>,
    "answer_correctness_score": <0.0 to 1.0>,
    "reasoning_quality_score": <0.0 to 1.0>,
    "difficulty_score": <0.0 to 1.0>,
    "overall_score": <0.0 to 1.0>,
    "feedback": "<actionable explanation for the Generator if retry, or confirmation if done>",
    "decision": "<retry or done>"
}}

--- QUESTION ---
{generated_question}

--- CORRECT ANSWER & GRADING RUBRIC ---
Correct answer: {correct_answer}
Rubric: {grading_rubric}

--- STUDENT ANSWER ---
Final answer: {solver_answer}
Step-by-step reasoning: {structured_reasoning}

--- CONTEXT ---
{rag_chunks}
"""

def judge_node(state):
    messages = [
        SystemMessage(content=JUDGE_PROMPT.format(
            generated_question  = state["generated_question"],
            grading_rubric      = state["grading_rubric"],
            correct_answer      = state["correct_answer"],
            solver_answer       = state["solver_answer"],
            structured_reasoning= state["structured_reasoning"],
            rag_chunks          = state["rag_chunks"]
        ))
    ]

    response = llm.invoke(messages)

    content = response.content.strip().removeprefix("```json").removesuffix("```").strip()
    parsed = json.loads(content)

    # If decision is "done", we add the question in the list of questions
    if parsed["decision"] == "done":
        current_questions = state.get("exam_questions", [])
        updated_questions = current_questions + [{
            "question"      : state["generated_question"],
            "correct_answer": state["correct_answer"],
            "grading_rubric": state["grading_rubric"],
            "score"         : parsed["overall_score"]
        }]
        return {
            "judge_score"    : parsed["overall_score"],
            "judge_feedback" : parsed["feedback"],
            "decision"       : "done",
            "exam_questions" : updated_questions,
            "iteration"      : 0  # reset for the next question
        }
    else:
        return {
            "judge_score"   : parsed["overall_score"],
            "judge_feedback": parsed["feedback"],
            "decision"      : "retry"
        }


def should_retry(state):
    # "retry" to much time
    if state.get("iteration", 0) >= 4:
        return "next_question"
    
    if state["decision"] == "retry":
        return "generator"
    else:
        validated    = len(state.get("exam_questions", []))
        target       = state.get("num_questions_target", 5)
        
        if validated < target:
            return "generator"  # we continue to create questions
        else:
            return END