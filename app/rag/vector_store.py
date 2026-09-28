import chromadb

from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_FOLDER,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    UPLOAD_FOLDER
)


UPLOAD_FOLDER.mkdir(exist_ok=True)
CHROMA_FOLDER.mkdir(exist_ok=True)


# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_FOLDER)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)
