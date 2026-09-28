import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# Paths (relative to the project root, not the cwd)
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

UPLOAD_FOLDER = BASE_DIR / "documents"
CHROMA_FOLDER = BASE_DIR / "chroma_db"


# --------------------------------------------------
# Models
# --------------------------------------------------

GEMINI_MODEL = "gemini-3.8-flash"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "company_policies"


# --------------------------------------------------
# LLM client
# --------------------------------------------------

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
