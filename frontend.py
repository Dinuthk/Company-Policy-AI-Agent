import os

import requests
import streamlit as st


# =========================================================
# Configuration
# =========================================================

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
)


st.set_page_config(
    page_title="Company Policy AI Agent",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_session_id" not in st.session_state:
    st.session_state.agent_session_id = None

if "documents" not in st.session_state:
    st.session_state.documents = None


# =========================================================
# Helper Functions
# =========================================================

def error_detail(error: requests.HTTPError) -> str:

    # FastAPI puts the reason in {"detail": ...}
    try:
        return error.response.json()["detail"]
    except (ValueError, KeyError, TypeError):
        return str(error)


def check_api():

    try:

        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


def upload_pdf(file):

    files = {
        "file": (
            file.name,
            file.getvalue(),
            "application/pdf"
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/upload",
        files=files,
        timeout=300
    )

    response.raise_for_status()

    return response.json()


def get_documents():

    response = requests.get(
        f"{API_BASE_URL}/documents",
        timeout=10
    )

    response.raise_for_status()

    return response.json()["documents"]


def refresh_documents():

    try:
        st.session_state.documents = get_documents()
    except requests.RequestException:
        st.session_state.documents = None


def send_message(message):

    payload = {
        "message": message
    }

    if st.session_state.agent_session_id:
        payload["session_id"] = st.session_state.agent_session_id

    response = requests.post(
        f"{API_BASE_URL}/agent/chat",
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    return response.json()


def reset_agent():

    session_id = st.session_state.agent_session_id

    if session_id:

        try:

            requests.delete(
                f"{API_BASE_URL}/agent/sessions/{session_id}",
                timeout=10
            )

        except requests.RequestException:

            pass

    st.session_state.agent_session_id = None
    st.session_state.messages = []


def show_sources(sources):

    if not sources:
        return

    with st.expander("Sources"):

        for source in sources:

            st.write(
                f"📄 {source['document']} — Page {source['page']}"
            )


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.title("Company Policy AI")

    st.caption("Agentic RAG Knowledge Assistant")

    # -----------------------------------------------------
    # API Status
    # -----------------------------------------------------

    api_online = check_api()

    if api_online:
        st.success("API connected")
    else:
        st.error("FastAPI server is not running")

    st.divider()

    # -----------------------------------------------------
    # Upload PDFs
    # -----------------------------------------------------

    st.subheader("Upload Policies")

    uploaded_files = st.file_uploader(
        "Select PDF policy documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", width="stretch"):

        if not uploaded_files:

            st.warning("Please select at least one PDF.")

        else:

            progress = st.progress(0)

            for index, file in enumerate(uploaded_files):

                try:

                    result = upload_pdf(file)

                    st.success(
                        f"{result['filename']} → "
                        f"{result['chunks_created']} chunks"
                    )

                except requests.HTTPError as error:

                    st.error(f"{file.name}: {error_detail(error)}")

                except requests.RequestException as error:

                    st.error(f"{file.name}: {error}")

                progress.progress((index + 1) / len(uploaded_files))

            refresh_documents()

    st.divider()

    # -----------------------------------------------------
    # Existing Documents
    # -----------------------------------------------------

    st.subheader("Knowledge Base")

    # Load once on first run, then on demand
    if api_online and st.session_state.documents is None:
        refresh_documents()

    if st.button("Refresh Policies", width="stretch"):
        refresh_documents()

    documents = st.session_state.documents or []

    if documents:

        st.write(f"**{len(documents)} documents**")

        for document in documents:
            st.caption(f"📄 {document}")

    else:

        st.caption("No documents loaded yet.")

    st.divider()

    # -----------------------------------------------------
    # New Conversation
    # -----------------------------------------------------

    if st.button("New Conversation", width="stretch"):

        reset_agent()

        st.rerun()


# =========================================================
# Main UI
# =========================================================

st.title("🤖 Company Policy AI Agent")

st.write(
    "Ask questions about leave, remote work, "
    "security, expenses, workplace conduct, "
    "training, and other uploaded company policies."
)


# =========================================================
# Example Questions
# =========================================================

with st.expander("Example questions"):

    st.markdown(
        """
- How many annual leave days do employees get?
- Can unused leave be carried forward?
- Can I work remotely from another country?
- What should I do if my laptop is stolen?
- What is the maximum hotel reimbursement?
- What policies do you have access to?
- I carried forward 3 leave days and used 8. How many days do I have left?
"""
    )


# =========================================================
# Display Chat History
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        show_sources(message.get("sources"))


# =========================================================
# Chat Input
# =========================================================

prompt = st.chat_input("Ask a company policy question...")

if prompt:

    # -----------------------------------------------------
    # Show user message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # -----------------------------------------------------
    # Call Agent API
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Checking company policies..."):

            try:

                result = send_message(prompt)

                # Save backend conversation ID
                st.session_state.agent_session_id = result["session_id"]

                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                show_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            except requests.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI. "
                    "Make sure the API server is running."
                )

            except requests.HTTPError as error:

                st.error(f"API error: {error_detail(error)}")

            except requests.RequestException as error:

                st.error(f"Request failed: {error}")
