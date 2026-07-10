import logging
from enum import Enum
from local_ai.tasks.qa import answer_question
from local_ai.tasks.conversation_summary import summarize_conversation

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# NOTE: learning_materials tasks (mindmap, learning_summary) are NOT routed
# here. They are called directly in api/routes.py (async) because they go
# through the agents pipeline in agents/.
# ─────────────────────────────────────────────────────────────────────────────


class TaskType(str, Enum):
    # ── Handled locally by the small LLM ─────────────────────────────────────
    QA                   = "qa"                    # answer a simple question
    CONVERSATION_SUMMARY = "conversation_summary"  # recap a chat history

    # ── Handled by agents pipeline (async, see routes.py) ────────────────────
    MINDMAP          = "mindmap"          # → agents/mindmap pipeline
    LEARNING_SUMMARY = "learning_summary" # → agents/summary pipeline

    # ── Complex tasks → AI-Server (task 39, not yet available) ───────────────
    RAG_QUERY     = "rag_query"
    DEEP_ANALYSIS = "deep_analysis"
    MULTI_DOC     = "multi_doc"


LOCAL_TASKS = {
    TaskType.QA,
    TaskType.CONVERSATION_SUMMARY,
}

AGENT_TASKS = {
    TaskType.MINDMAP,
    TaskType.LEARNING_SUMMARY,
}

COMPLEX_TASKS = {
    TaskType.RAG_QUERY,
    TaskType.DEEP_ANALYSIS,
    TaskType.MULTI_DOC,
}


class TaskRouter:
    """
    Routes tasks to:
      - The small local LLM (Ollama / qwen2.5:1.5b) for qa and conversation_summary
      - The agents pipeline for mindmap and learning_summary
      - The AI-Server for complex tasks (coming soon, task 39)
    """

    def __init__(self, ai_server_url: str = None):
        self.ai_server_url = ai_server_url

    def run(self, task_type: str, payload: dict) -> dict:
        """
        Synchronous router — handles LOCAL_TASKS only.
        AGENT_TASKS are handled directly in routes.py (async).

        Args:
            task_type: One of the TaskType values.
            payload:   Dict containing all task inputs (varies by task).

        Returns:
            dict with 'source', 'task', and 'result' keys.
        """
        try:
            task = TaskType(task_type)
        except ValueError:
            return {
                "error": (
                    f"Unknown task type: '{task_type}'. "
                    f"Valid values: {[t.value for t in TaskType]}"
                )
            }

        if task in LOCAL_TASKS:
            logger.info(f"Local task '{task}' → small LLM")
            return self._run_local(task, payload)

        if task in AGENT_TASKS:
            return {
                "error": (
                    f"Task '{task}' is an agent pipeline task. "
                    "Use the dedicated async endpoints: "
                    "POST /local-ai/mindmap or POST /local-ai/learning-summary"
                )
            }

        if task in COMPLEX_TASKS:
            logger.info(f"Complex task '{task}' → AI-Server")
            return self._run_remote(task, payload)

        return {"error": f"Unhandled task: '{task}'"}

    # ── Local LLM dispatch ────────────────────────────────────────────────────

    def _run_local(self, task: TaskType, payload: dict) -> dict:
        try:
            if task == TaskType.QA:
                question = payload.get("question", "")
                context  = payload.get("context", "")
                result   = answer_question(question, context)

            elif task == TaskType.CONVERSATION_SUMMARY:
                messages = payload.get("messages", [])
                result   = summarize_conversation(messages)

            else:
                result = {"error": "Local task not implemented"}

            return {"source": "local", "task": task, "result": result}

        except RuntimeError as e:
            return {"source": "local", "error": str(e)}

    # ── AI-Server dispatch (task 39) ──────────────────────────────────────────

    def _run_remote(self, task: TaskType, payload: dict) -> dict:
        if not self.ai_server_url:
            return {
                "source": "remote",
                "error": (
                    "AI-Server is not configured yet. "
                    "Complex tasks are not available at this time."
                ),
            }
        # TODO: implement HTTP call to the AI-Server once task 39 is ready
        # import httpx
        # r = await client.post(f"{self.ai_server_url}/generate",
        #                       json={"task": task, "payload": payload})
        # return {"source": "remote", "task": task, "result": r.json()}
        return {"source": "remote", "error": "Not implemented yet"}


# Singleton — plug in the AI-Server URL here once task 39 is ready
task_router = TaskRouter(ai_server_url=None)