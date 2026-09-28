import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.agent.tools import list_available_policies
from app.config import COLLECTION_NAME, UPLOAD_FOLDER
from app.rag.pdf_service import extract_pdf_pages, ingest_pages
from app.rag.vector_store import collection


router = APIRouter(
    tags=["documents"]
)


# --------------------------------------------------
# Upload + RAG ingestion
# --------------------------------------------------

@router.post("/upload")
def upload_document(
    file: UploadFile = File(...)
):

    # Strip any directory parts from the client filename
    filename = Path(file.filename).name

    if not filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    file_path = UPLOAD_FOLDER / filename

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    pages = extract_pdf_pages(str(file_path))

    if not pages:

        raise HTTPException(
            status_code=400,
            detail="No readable text found in PDF"
        )

    stats = ingest_pages(filename, pages)

    return {
        "message": "Document processed successfully",
        "filename": filename,
        "pages": len(pages),
        **stats
    }


# --------------------------------------------------
# Database stats
# --------------------------------------------------

@router.get("/stats")
def database_stats():

    return {
        "collection": COLLECTION_NAME,
        "total_chunks": collection.count()
    }


# --------------------------------------------------
# Documents in the knowledge base
# --------------------------------------------------

@router.get("/documents")
def list_documents():

    return {
        "documents": list_available_policies()
    }
