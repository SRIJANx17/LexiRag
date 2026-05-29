# LexiRAG — AI-Powered Indian Legal Assistant

RAG-based legal chatbot for the Indian Constitution, IPC, judgments, and legal acts.
Runs fully offline on your laptop using open-source models.

---

## Prerequisites

- Python 3.10+
- 16GB RAM minimum (32GB recommended)
- [Ollama](https://ollama.com) installed

---

## Quick Start (local, no Docker)

### 1. Install Ollama and pull the model
```bash
# Download Ollama from https://ollama.com and install it
ollama pull mistral
# Keep Ollama running in the background
```

### 2. Set up the Python environment
```bash
git clone <your-repo-url> lexirag
cd lexirag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add legal PDFs
Download the Indian Constitution PDF from https://india.gov.in and place it in:
```
data/constitution/constitution.pdf
```
Other useful sources:
- IPC/CrPC: https://indiacode.nic.in
- Supreme Court judgments: https://main.sci.gov.in

### 4. Ingest PDFs into the vector database
```bash
python -m app.rag.ingest --source data/constitution
python -m app.rag.ingest --source data/judgments
python -m app.rag.ingest --source data/acts
```

### 5. Start the backend API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
API docs available at: http://localhost:8000/docs

### 6. Start the frontend (in a new terminal)
```bash
source venv/bin/activate
streamlit run frontend/app.py
```
Open: http://localhost:8501

---

## Docker (one-command setup)

```bash
docker-compose up --build

# In a separate terminal, pull the model into the Ollama container:
docker exec -it lexirag-ollama-1 ollama pull mistral

# Ingest PDFs (after placing them in data/):
docker exec -it lexirag-api-1 python -m app.rag.ingest --source data/constitution
```

---

## Project structure

```
lexirag/
├── main.py                     # FastAPI app entrypoint
├── requirements.txt
├── .env                        # Configuration
├── Dockerfile
├── docker-compose.yml
├── app/
│   ├── rag/
│   │   ├── engine.py           # Core RAG engine (LLM + retrieval + memory)
│   │   └── ingest.py           # PDF ingestion pipeline
│   ├── summarizer/
│   │   └── summarizer.py       # Summarization + simplification
│   ├── translation/
│   │   └── translator.py       # EN↔HI translation
│   └── utils/
│       └── helpers.py          # Text utilities
├── frontend/
│   └── app.py                  # Streamlit UI
├── data/
│   ├── constitution/           # Place Constitution PDFs here
│   ├── judgments/              # Place judgment PDFs here
│   └── acts/                   # Place IPC/CrPC PDFs here
└── chroma_db/                  # Auto-created vector database
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /chat | Ask a legal question |
| POST | /chat/clear | Clear conversation history |
| POST | /upload | Upload and ingest a PDF |
| POST | /summarize | Summarize legal text |
| POST | /summarize/pdf | Summarize an uploaded PDF |
| POST | /translate | Translate EN↔HI |
| GET | /db/status | Vector DB stats |

Full interactive docs: http://localhost:8000/docs

---

## Deployment to Hugging Face Spaces

1. Create a Space with "Docker" SDK
2. Push this repository
3. Add secret: `OLLAMA_BASE_URL` pointing to an external Ollama instance
   (or use a smaller model like `phi3` for free-tier hardware)

---

## Troubleshooting

**"No module named app"** — run commands from the project root (`lexirag/`), not inside `app/`.

**Slow first response** — the first query loads the embedding model and warms up Mistral. Subsequent queries are faster.

**"ChromaDB has 0 chunks"** — you need to run the ingestion step first. See Step 4 above.

**Out of memory** — switch to a smaller model: set `OLLAMA_MODEL=phi3` in `.env` and run `ollama pull phi3`.