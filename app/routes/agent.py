from fastapi import APIRouter, HTTPException
from google.genai import errors

from app.models.schemas import AgentChatRequest
from app.agent.service import run_agent
from app.agent.memory import clear_session


router = APIRouter(
    prefix="/agent",
    tags=["agent"]
)


@router.post("/chat")
def agent_chat(
    request: AgentChatRequest
):

    message = request.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    try:

        return run_agent(
            message=message,
            session_id=request.session_id
        )

    except errors.APIError as exc:

        # 429 = quota / rate limit, 503 = model overloaded;
        # anything else is reported as a bad upstream response
        raise HTTPException(
            status_code=exc.code if exc.code in (429, 503) else 502,
            detail=f"LLM service error: {exc.message}"
        )


@router.delete("/sessions/{session_id}")
def reset_session(
    session_id: str
):

    return {
        "session_id": session_id,
        "cleared": clear_session(session_id)
    }
