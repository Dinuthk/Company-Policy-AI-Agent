from fastapi import APIRouter, HTTPException

from app.models.schemas import AskRequest, SearchRequest
from app.rag.retrieval import answer_policy_question, retrieve_policy_chunks


router = APIRouter(
    tags=["rag"]
)


def _require_question(question: str) -> str:

    question = question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    return question


# --------------------------------------------------
# Semantic Search
# --------------------------------------------------

@router.post("/search")
def search_policy(
    request: SearchRequest
):

    question = _require_question(request.question)

    return {
        "question": question,
        "matches": retrieve_policy_chunks(question, request.top_k)
    }


# --------------------------------------------------
# Single-shot RAG answer
# --------------------------------------------------

@router.post("/ask")
def ask_policy(
    request: AskRequest
):

    question = _require_question(request.question)

    return answer_policy_question(question, request.top_k)
