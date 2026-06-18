from backend.core.models import MODELS
from langchain_core.messages import SystemMessage, HumanMessage

llm = MODELS["glm"]

ROUTING_PROMPT = """
You are an expert educational content creator in a multi-agent exam generation system.
You will receive a topic or learning objective, along with relevant context chunks retrieved from a knowledge base.

Your job is to generate a high-quality exam question based on the provided context.

You must produce:
- A clear and unambiguous question (MCQ or essay)
- For MCQ: 4 answer choices (A, B, C, D) with exactly one correct answer
- A detailed grading rubric
- The correct answer with a justification

Rules:
- Base your question strictly on the provided context, do not invent facts
- The question must target a specific concept, not be overly generic
- Difficulty should be moderate to hard
- Output must be structured and consistent

You must output a valid JSON and nothing else, with this exact structure:
{{
    "question": "<the full question text>",
    "choices": {                          
        "A": "<choice A>",               
        "B": "<choice B>",               
        "C": "<choice C>",               
        "D": "<choice D>"                
    },                                   
    "correct_answer": "<A, B, C or D>",  
    "justification": "<why this answer is correct>",
    "grading_rubric": "<criteria to evaluate a student answer>"
}}

For essay questions, omit the choices field.
Do not add any text outside the JSON.

Already generated questions (DO NOT repeat these topics or questions):
{exam_questions}

Generate a NEW question on a DIFFERENT aspect of the topic.

Context:
{rag_chunks}

Topic / Learning objective:
{query}
"""

def generator_agent(state):
    messages = [
        SystemMessage(content=ROUTING_PROMPT.format(
            query=state["query"],
            rag_chunks=state["rag_chunks"],
            exam_questions=state["exam_questions"]
        )),
        HumanMessage(content="Generate the question now.")
    ]
    
    response = llm.invoke(messages)
    
    parsed = json.loads(response.content)
    
    return {
        "generated_question" : parsed["question"],
        "grading_rubric" : parsed["grading_rubric"],
        "correct_answer" : parsed["correct_answer"],
        "iteration" : state.get("iteration", 0) + 1
    }