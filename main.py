"""
LexiRAG FastAPI backend.
Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import uuid
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

def download_chromadb_if_needed():
    """Download ChromaDB from Hugging Face Hub if not present locally."""
    import os
    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    sqlite_path = os.path.join(chroma_path, "chroma.sqlite3")
    
    if not os.path.exists(sqlite_path):
        print("ChromaDB not found. Downloading from Hugging Face Hub...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Srijan-17/lexirag-chromadb",
                repo_type="dataset",
                local_dir=chroma_path,
                token=os.getenv("HF_TOKEN"),
            )
            print("ChromaDB downloaded successfully.")
        except Exception as e:
            print(f"Failed to download ChromaDB: {e}")
    else:
        print("ChromaDB found locally — skipping download.")

from app.rag.engine import engine
from app.rag.ingest import load_pdf_smart, chunk_documents, store_in_chroma
from app.summarizer.summarizer import summarize_legal_text, simplify_legal_text
from app.translation.translator import translate

load_dotenv()

app = FastAPI(
    title="LexiRAG API",
    description="AI-powered Indian legal assistant using RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic models ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list
    session_id: str

class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "hi"

class SummarizeRequest(BaseModel):
    text: str
    mode: str = "summarize"  # "summarize" or "simplify"

class ClearSessionRequest(BaseModel):
    session_id: str


# ─── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    download_chromadb_if_needed()
    engine.initialize()

# ─── Health ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    db_status = engine.check_db_status()
    return {
        "status": "ok",
        "service": "LexiRAG",
        "database": db_status,
    }


# ─── Chat ────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a legal question. Maintains conversation history per session_id.
    
    Example:
        POST /chat
        {"question": "What is Article 21?", "session_id": "user123"}
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        result = engine.query(
            question=request.question.strip(),
            session_id=request.session_id or "default",
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/chat/clear")
async def clear_session(request: ClearSessionRequest):
    """Clear conversation history for a session."""
    engine.clear_session(request.session_id)
    return {"message": f"Session '{request.session_id}' cleared."}


# ─── Upload ──────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    category: str = Form(default="legal"),
):
    """
    Upload a PDF and ingest it into the vector database.
    The PDF becomes immediately searchable after upload.
    
    Args:
        file: PDF file
        category: 'constitution', 'judgment', 'act', or 'legal'
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = load_pdf_smart(tmp_path)
        if not docs:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

        # Override category metadata
        for doc in docs:
            doc.metadata["category"] = category
            doc.metadata["source_file"] = file.filename

        chunks = chunk_documents(docs)
        store_in_chroma(chunks)

        return {
            "message": "PDF ingested successfully.",
            "filename": file.filename,
            "pages_loaded": len(docs),
            "chunks_stored": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ─── Summarize ───────────────────────────────────────────────────────────────

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    """
    Summarize or simplify legal text.
    
    mode='summarize': structured legal summary with citations
    mode='simplify': plain-language explanation for citizens
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        if request.mode == "simplify":
            result = simplify_legal_text(request.text)
        else:
            result = summarize_legal_text(request.text)
        return {"result": result, "mode": request.mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/summarize/pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    """Upload a PDF and get a summary directly without storing it in the DB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = load_pdf_smart(tmp_path)
        full_text = " ".join(d.page_content for d in docs)
        if not full_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text.")
        summary = summarize_legal_text(full_text)
        return {"summary": summary, "filename": file.filename, "pages": len(docs)}
    finally:
        os.unlink(tmp_path)


# ─── Translate ───────────────────────────────────────────────────────────────

@app.post("/translate")
async def translate_text(request: TranslateRequest):
    """
    Translate legal text between languages.
    Supported: en→hi, hi→en
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        translated = translate(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
        )
        return {
            "original": request.text,
            "translated": translated,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


# ─── DB info ─────────────────────────────────────────────────────────────────

@app.get("/db/status")
async def db_status():
    """Get vector database statistics."""
    return engine.check_db_status()