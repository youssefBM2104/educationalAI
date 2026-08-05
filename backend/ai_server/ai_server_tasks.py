"""
Celery tasks for the AI-Server's long-running pipelines.

Exam and slides generation can take anywhere from tens of seconds to a
few minutes (multiple agents, possibly several LLM calls in a loop), so
we never run them synchronously inside an HTTP request. Instead, the
route hands the job off here, returns a job_id immediately, and the
caller polls a /jobs/{job_id} endpoint until it's done.

This is a separate Celery app from backend/tasks/ingestion_tasks.py,
and it should run on its own queue ("ai_server") so a slow exam
generation never blocks document ingestion, and vice versa.

Tutoring is intentionally NOT here — it's a fast, synchronous,
multi-turn conversation, handled directly in routes/tutoring.py.
"""

import logging

from celery import Celery

from backend.core.config import settings
from backend.ai_server.schemas.exam import ExamRequest
from backend.ai_server.schemas.slides import SlidesRequest
from backend.ai_server.services.exam_service import run_exam_pipeline
from backend.ai_server.services.slides_service import run_slides_pipeline

logger = logging.getLogger(__name__)

celery_app = Celery(
    "ai_server",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue="ai_server",
)


@celery_app.task(name="ai_server.generate_exam")
def generate_exam_task(payload: dict) -> list[dict]:
    logger.info("Celery task: generate_exam started")
    request = ExamRequest(**payload)
    return run_exam_pipeline(request)


@celery_app.task(name="ai_server.generate_slides")
def generate_slides_task(payload: dict) -> dict:
    logger.info("Celery task: generate_slides started")
    request = SlidesRequest(**payload)
    return run_slides_pipeline(request)
