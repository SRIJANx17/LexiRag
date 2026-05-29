"""
PDF ingestion pipeline.
Usage:
    python -m app.rag.ingest --source data/constitution
    python -m app.rag.ingest --file data/constitution/Constitution_Of_India.pdf
"""
import os
import argparse
from pathlib import Path
from typing import List

from langchain.schema import Document
import langchain_community.document_loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import pdfplumber
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))


def load_pdf_pypdf(path: str) -> List[Document]:
    """Load a digital PDF using PyPDF."""
    try:
        loader = langchain_community.document_loaders.PyPDFLoader(path)
        docs = loader.load()
        # Tag each doc with source metadata
        for doc in docs:
            doc.metadata["source_file"] = Path(path).name
            doc.metadata["category"] = _infer_category(path)
        return docs
    except Exception as e:
        print(f"  PyPDF failed for {path}: {e}")
        return []


def load_pdf_pdfplumber(path: str) -> List[Document]:
    """Load PDF with pdfplumber — better for tables and structured layouts."""
    docs = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    docs.append(Document(
                        page_content=text.strip(),
                        metadata={
                            "source": path,
                            "source_file": Path(path).name,
                            "page": i + 1,
                            "category": _infer_category(path),
                        }
                    ))
    except Exception as e:
        print(f"  pdfplumber failed for {path}: {e}")
    return docs


def load_pdf_ocr(path: str) -> List[Document]:
    """
    Fallback OCR for scanned PDFs using easyocr.
    Only imported if needed — avoids slow startup.
    """
    try:
        import easyocr
        import fitz  # PyMuPDF — install: pip install pymupdf
        import numpy as np
        from PIL import Image
        import io

        reader = easyocr.Reader(["en"])
        docs = []
        pdf_doc = fitz.open(path)

        for i, page in enumerate(pdf_doc):
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img_array = np.array(img)
            results = reader.readtext(img_array, detail=0)
            text = " ".join(results)
            if text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source": path,
                        "source_file": Path(path).name,
                        "page": i + 1,
                        "category": _infer_category(path),
                        "ocr": True,
                    }
                ))
        pdf_doc.close()
        return docs
    except ImportError:
        print("  OCR skipped: install pymupdf and easyocr for scanned PDF support.")
        return []
    except Exception as e:
        print(f"  OCR failed for {path}: {e}")
        return []


def _infer_category(path: str) -> str:
    """Infer document category from path."""
    path_lower = path.lower()
    if "constitution" in path_lower:
        return "constitution"
    elif "judgment" in path_lower or "case" in path_lower:
        return "judgment"
    elif "act" in path_lower or "ipc" in path_lower or "crpc" in path_lower:
        return "act"
    return "legal"


def load_pdf_smart(path: str) -> List[Document]:
    """
    Smart loader: tries PyPDF first, falls back to pdfplumber,
    then OCR if the page has no extractable text.
    """
    print(f"  Loading: {Path(path).name}")
    docs = load_pdf_pypdf(path)

    # Check if text extraction yielded meaningful content
    total_chars = sum(len(d.page_content) for d in docs)
    if total_chars < 100:
        print("    PyPDF yielded little text, trying pdfplumber...")
        docs = load_pdf_pdfplumber(path)
        total_chars = sum(len(d.page_content) for d in docs)

    if total_chars < 100:
        print("    pdfplumber yielded little text, trying OCR...")
        docs = load_pdf_ocr(path)

    print(f"    Loaded {len(docs)} pages, {total_chars} chars total.")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Chunked into {len(chunks)} pieces (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def get_embeddings():
    """Return the embedding model (cached after first call)."""
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def store_in_chroma(chunks: List[Document], collection_name: str = "lexirag") -> Chroma:
    """Store embedded chunks in ChromaDB, appending to existing collection."""
    embeddings = get_embeddings()
    print(f"  Storing {len(chunks)} chunks in ChromaDB at {CHROMA_DB_PATH}...")

    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    vector_db.add_documents(chunks)
    print(f"  Done. Total docs in collection: {vector_db._collection.count()}")
    return vector_db


def ingest_file(path: str):
    """Ingest a single PDF file."""
    docs = load_pdf_smart(path)
    if not docs:
        print(f"  No content extracted from {path}. Skipping.")
        return
    chunks = chunk_documents(docs)
    store_in_chroma(chunks)


def ingest_directory(directory: str):
    """Ingest all PDFs in a directory recursively."""
    pdf_files = list(Path(directory).rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return
    print(f"\nFound {len(pdf_files)} PDF(s) in {directory}")
    for pdf_path in pdf_files:
        ingest_file(str(pdf_path))
    print("\nIngestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LexiRAG PDF ingestion pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single PDF file")
    group.add_argument("--source", help="Directory containing PDF files")
    args = parser.parse_args()

    if args.file:
        ingest_file(args.file)
    elif args.source:
        ingest_directory(args.source)