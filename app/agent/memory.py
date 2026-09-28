from google.genai import types


# --------------------------------------------------
# Conversation memory
#
#   session_id  ->  full Gemini message history
#
# Gemini does not keep conversation state on the
# server (no previous_response_id), so we store the
# history here and send it back on every turn.
#
# In-memory only: restarting FastAPI clears all
# sessions. Use Redis / PostgreSQL for production.
# --------------------------------------------------

session_store: dict[str, list[types.Content]] = {}


def get_history(
    session_id: str
) -> list[types.Content]:

    # Return a copy so a failed request never
    # leaves half a turn in the stored history
    return list(
        session_store.get(session_id, [])
    )


def save_history(
    session_id: str,
    history: list[types.Content]
):

    session_store[session_id] = history


def clear_session(
    session_id: str
) -> bool:

    return session_store.pop(
        session_id,
        None
    ) is not None
