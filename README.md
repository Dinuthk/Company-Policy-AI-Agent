# Company-Policy-AI-Agent

Agentic RAG API for company policy documents (FastAPI + ChromaDB + Gemini).

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Needs `GEMINI_API_KEY` in `.env`. Open http://127.0.0.1:8000/docs.

### Chat UI (Streamlit)

In a second terminal, with the API running:

```bash
streamlit run frontend.py
```

Opens http://localhost:8501. Set `API_BASE_URL` if the API is not on `http://127.0.0.1:8000`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/upload` | Upload a PDF, chunk, embed and store it in ChromaDB |
| GET | `/documents` | Policy documents in the knowledge base |
| GET | `/stats` | Number of chunks in the vector store |
| POST | `/search` | Semantic search over policy chunks |
| POST | `/ask` | Single-shot RAG answer |
| POST | `/agent/chat` | Agent with tools + conversation memory |
| DELETE | `/agent/sessions/{session_id}` | Clear a conversation |

### Conversation memory

The first `/agent/chat` call returns a `session_id`. Send it back to continue the conversation:

```json
{ "session_id": "76c8549c-...", "message": "Can I carry them forward?" }
```

Sessions are kept in memory, so restarting the server clears them.

## Structure

```text
app/
├── main.py            FastAPI app, router registration
├── config.py          paths, model names, Gemini client
├── models/schemas.py  request models
├── rag/               vector store, PDF ingestion, retrieval + /ask logic
├── agent/             tools, conversation memory, agent loop
└── routes/            documents, rag, agent routers
```
