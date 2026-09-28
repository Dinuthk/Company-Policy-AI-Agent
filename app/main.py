from fastapi import FastAPI

from app.routes.agent import router as agent_router
from app.routes.documents import router as documents_router
from app.routes.rag import router as rag_router


app = FastAPI(
    title="Company Policy AI Agent",
    description="Agentic RAG API for company policy documents",
    version="2.0.0"
)


app.include_router(documents_router)
app.include_router(rag_router)
app.include_router(agent_router)


@app.get("/")
def home():

    return {
        "message": "Company Policy AI Agent is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }
