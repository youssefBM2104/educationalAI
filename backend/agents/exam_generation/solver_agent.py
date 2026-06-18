from backend.core.models import MODELS
from langchain_core.messages import SystemMessage, HumanMessage

llm = MODELS["gpt_oss"]

ROUTING_PROMPT = """
You are a student simulation agent in a multi-agent exam system.
Your role is to answer the given exam question as a student would, using ONLY the provided context.

You must:
- Read the question carefully
- Use only the information available in the context below
- Produce a structured reasoning trace explaining your thought process step by step
- Give a final answer

Rules:
- Do NOT use any external knowledge beyond the provided context
- If the context is insufficient to answer, explicitly state it
- Your reasoning must be honest and reflect what a student can infer from the context

You must output a valid JSON and nothing else, with this exact structure:
{{
    "structured_reasoning": "<the structured reasoning trace explaining your thought process step by step>",                                
    "solver_answer": "<the final answer>",  
}}

Do not add any text outside the JSON.

Question:
{generated_question}

Context:
{rag_chunks}
"""

def solver_agent(state):
    messages = [
        SystemMessage(content=ROUTING_PROMPT.format(
            generated_question=state["generated_question"],
            rag_chunks=state["rag_chunks"]
        )),
        HumanMessage(content="Answer the question now.")
    ]
    
    response = llm.invoke(messages)
    
    parsed = json.loads(response.content)

    
    return {
        "solver_answer": parsed["solver_answer"],
        "structured_reasoning": parsed["structured_reasoning"]
        }