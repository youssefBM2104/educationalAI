from fastapi import FastAPI

from backend.db.postgre import init_db
from backend.api.routes_documents import router as documents_router
from backend.api.rag import router as rag_router

app = FastAPI()


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(documents_router)
app.include_router(rag_router)

@app.get("/health")
def health():
    return {"status": "ok"}
