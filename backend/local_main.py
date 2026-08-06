from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes_documents_proxy import router as documents_router
from backend.api.routes_sync import router as sync_router
from backend.api.rag import router as rag_router

app = FastAPI(title="Local Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(sync_router)
app.include_router(rag_router)


@app.get("/health")
def health():
    return {"status": "ok"}
