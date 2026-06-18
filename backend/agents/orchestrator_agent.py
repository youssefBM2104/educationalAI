from backend.core.models import MODELS
from langchain_core.messages import SystemMessage, HumanMessage

llm = MODELS["qwen"]

ROUTING_PROMPT = """
You are an orchestrator in an educational AI system.
Your job is to classify the user's intent into exactly one of these categories:

- "exam"     : the user wants to generate questions, take a quiz, or be graded
- "lecture"  : the user wants a course, a summary, slides, or educational content
- "tutoring" : the user wants explanations, help understanding, or a learning path

Respond with ONLY one word: exam or lecture or tutoring.
No explanation, no punctuation, just the word.
"""

def orchestrator_agent(state):
    messages = [
        SystemMessage(content=ROUTING_PROMPT),
        HumanMessage(content=state["query"])
    ]
    
    response = llm.invoke(messages)
    
    # cleaning
    intent = response.content.strip().lower()
    
    return {"intent": intent}



#fucntion for routing 
def route_intent(state):
    intent = state["intent"]
    
    if intent == "exam":
        return "generator"
    elif intent == "lecture":
        return "lecture"
    else:
        return "tutoring"