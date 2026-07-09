"""
local_ai/model_manager.py

Thin wrapper around local_model.py.
Handles availability checks (Ollama running, model downloaded)
and exposes a single generate() method used by tasks/qa.py
and tasks/conversation_summary.py.

The actual model loading logic (fine-tuned adapter vs Ollama fallback)
lives in local_model.py — this file does not duplicate it.
"""

import logging
import requests
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from local_ai.local_model import MODEL

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages inference calls to the small local AI model.
    Uses the fine-tuned adapter if available (via local_model.py),
    otherwise falls back to raw Qwen2.5 via Ollama.
    """

    def __init__(self):
        # MODEL["small_ai"] is already loaded at import time in local_model.py
        self._llm = MODEL["small_ai"]

    def check_ollama_running(self) -> bool:
        """
        Check if Ollama is running on this machine.
        Only relevant when the fine-tuned adapter is not yet available
        and the system falls back to Ollama.
        """
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def check_model_ready(self) -> bool:
        """
        Check whether the model (fine-tuned adapter or Ollama fallback) is ready.
        Returns True if the LLM object was successfully loaded.
        """
        return self._llm is not None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a response using the local model.

        Uses LangChain messages so it works identically whether
        local_model.py loaded the HuggingFace adapter or ChatOllama.

        Args:
            prompt:        The user message / question.
            system_prompt: Optional system instruction.
            temperature:   Sampling temperature (ignored by HuggingFacePipeline,
                           set at pipeline creation in local_model.py).
            max_tokens:    Max tokens to generate.

        Returns:
            The model's response as a plain string.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.check_model_ready():
            raise RuntimeError(
                "The local model is not loaded. "
                "Either run 'python -m training.train' to generate the fine-tuned adapter, "
                "or install Ollama at https://ollama.com as a fallback."
            )

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self._llm.invoke(messages)
            # Both HuggingFacePipeline and ChatOllama return an AIMessage
            if hasattr(response, "content"):
                return response.content.strip()
            # Safety fallback if response is a plain string
            return str(response).strip()
        except Exception as e:
            logger.error(f"Local model inference error: {e}")
            raise RuntimeError(f"Local model error: {e}")


# Singleton — imported by tasks/qa.py and tasks/conversation_summary.py
model_manager = ModelManager()  