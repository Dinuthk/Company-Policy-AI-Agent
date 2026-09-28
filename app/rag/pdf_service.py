import hashlib

from pypdf import PdfReader

from app.rag.vector_store import (
    collection,
    embedding_model
)


# --------------------------------------------------
# Chunking
# --------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 700,
    overlap: int = 100
):

    chunks = []

    text = text.strip()

    if not text:
        return chunks

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# PDF extraction
# --------------------------------------------------

def extract_pdf_pages(file_path: str):

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages


# --------------------------------------------------
# Ingestion: pages -> chunks -> embeddings -> Chroma
# --------------------------------------------------

def ingest_pages(
    filename: str,
    pages: list[dict]
):

    documents = []
    metadatas = []
    ids = []

    for page_data in pages:

        page_number = page_data["page"]

        chunks = chunk_text(page_data["text"])

        for chunk_index, chunk in enumerate(chunks):

            documents.append(chunk)

            metadatas.append(
                {
                    "source": filename,
                    "page": page_number,
                    "chunk": chunk_index
                }
            )

            # deterministic unique ID
            raw_id = f"{filename}-{page_number}-{chunk_index}"

            ids.append(
                hashlib.sha1(raw_id.encode()).hexdigest()
            )

    embeddings = embedding_model.encode_document(
        documents,
        normalize_embeddings=True
    ).tolist()

    # Remove old version of same PDF
    collection.delete(
        where={
            "source": filename
        }
    )

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return {
        "chunks_created": len(documents),
        "embedding_dimension": len(embeddings[0]),
        "total_vectors_in_database": collection.count()
    }
