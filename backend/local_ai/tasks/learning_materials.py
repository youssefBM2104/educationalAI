import httpx
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# This file does NOT call the local LLM.
# It bridges the local_ai router to your EXISTING agents pipeline
# (the two folders in agents/ with 4 Python files).
#
# The agents pipeline already handles:
#   - RAG retrieval (Qdrant + Neo4j)
#   - Mindmap Agent → Export
#   - Extractor Agent → Writer Agent → Summary
#
# We simply call those agents via HTTP (internal FastAPI routes).
# This keeps all the heavy LLM work on the AI-Server side, while the
# local_ai router acts as the entry point for the user's device.
# ─────────────────────────────────────────────────────────────────────────────

DetailLevel = Literal["short", "medium", "long"]


async def get_mindmap(
    query: str,
    internal_api_url: str = "http://localhost:8000",
) -> dict:
    """
    Request a mindmap from the existing Mindmap Agent pipeline.

    Flow (already implemented in agents/):
      query → RAG retrieval → Mindmap Agent → Export (Mermaid / HTML)

    Args:
        query:            The topic the user wants to map.
        internal_api_url: Base URL of your FastAPI backend.

    Returns:
        dict with 'mermaid' (str) and 'html' (str) from the Export step.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{internal_api_url}/agents/learning-materials/mindmap",
                json={"query": query},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Mindmap agent error: {e.response.status_code} — {e.response.text}")
            raise RuntimeError(f"Mindmap agent returned an error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Could not reach internal API: {e}")
            raise RuntimeError("Could not reach the internal agents API.")


async def get_summary(
    query: str,
    detail_level: DetailLevel = "medium",
    course_title: str = "",
    internal_api_url: str = "http://localhost:8000",
) -> dict:
    """
    Request a summary from the existing Extractor → Writer pipeline.

    Flow (already implemented in agents/):
      query → RAG retrieval → Extractor Agent → Writer Agent → Summary

    Args:
        query:            The topic the user wants summarized.
        detail_level:     'short' (~150 words) | 'medium' (~400) | 'long' (~800).
        course_title:     Passed to the Writer Agent as the summary heading.
        internal_api_url: Base URL of your FastAPI backend.

    Returns:
        dict with 'summary' (str) and 'detail_level' (str).
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{internal_api_url}/agents/learning-materials/summary",
                json={
                    "query": query,
                    "detail_level": detail_level,
                    "course_title": course_title,
                },
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Summary agent error: {e.response.status_code} — {e.response.text}")
            raise RuntimeError(f"Summary agent returned an error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Could not reach internal API: {e}")
            raise RuntimeError("Could not reach the internal agents API.")
            