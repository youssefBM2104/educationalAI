"""
Entry point for the AI-Server FastAPI app.

This is a separate process from backend/main.py (the main platform API).
It only mounts routers here — no business logic lives in this file.
Run locally with:

    uvicorn backend.ai_server.main:app --host 0.0.0.0 --port 8001 --reload

The exam/slides job workers must be started separately:

    celery -A backend.ai_server.ai_server_tasks worker --loglevel=info -Q ai_server
"""

from fastapi import FastAPI

from backend.ai_server.routes import exam, slides, tutoring

app = FastAPI(title="AI Server", version="0.1.0")

app.include_router(exam.router, prefix="/v1")
app.include_router(slides.router, prefix="/v1")
app.include_router(tutoring.router, prefix="/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
