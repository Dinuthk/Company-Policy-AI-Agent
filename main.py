from fastapi import FastAPI, UploadFile, File
import os
import shutil

app = FastAPI(
    title="Company Policy AI Agent"
)

UPLOAD_FOLDER = "documents"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Company Policy AI Agent API is running"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    text = extract_pdf_text(file_path)

    return {
        "message": "Document uploaded successfully",
        "filename": file.filename,
        "characters_extracted": len(text)
    }