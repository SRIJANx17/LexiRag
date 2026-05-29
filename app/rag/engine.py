"""
LexiRAG core engine.
Supports both Groq API (fast, free, recommended) and local Ollama.
Set USE_GROQ=true in .env to use Groq.
"""
import os
from typing import Dict, Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

LEGAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are LexiRAG, an expert AI legal assistant specializing in Indian law,
the Indian Constitution, IPC, CrPC, and Supreme Court judgments.

Use ONLY the context provided below to answer the question. If the context does not
contain sufficient information, say so clearly — do not fabricate legal information.

Always:
- Cite the specific Article, Section, or document name when referencing legal text
- Explain legal terminology in simple, plain language
- Structure your answer clearly with the legal provision first, then explanation

Context:
{context}

Question: {question}

Answer (include citations where possible):"""
)


class LexiRAGEngine:
    def __init__(self):
        self._embeddings = None
        self._vector_db = None
        self._llm = None
        self._sessions: Dict[str, ConversationalRetrievalChain] = {}
        self._initialized = False

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def _get_vector_db(self):
        if self._vector_db is None:
            self._vector_db = Chroma(
                collection_name="lexirag",
                embedding_function=self._get_embeddings(),
                persist_directory=CHROMA_DB_PATH,
            )
        return self._vector_db

    def _get_llm(self):
        if self._llm is None:
            if USE_GROQ:
                try:
                    from langchain_groq import ChatGroq
                    self._llm = ChatGroq(
                        api_key=GROQ_API_KEY,
                        model=GROQ_MODEL,
                        temperature=0.1,
                        max_tokens=1024,
                    )
                    print(f"LLM: Groq API ({GROQ_MODEL})")
                except ImportError:
                    raise RuntimeError(
                        "langchain-groq not installed. Run: pip install langchain-groq"
                    )
            else:
                from langchain_community.llms import Ollama
                self._llm = Ollama(
                    base_url=OLLAMA_BASE_URL,
                    model=OLLAMA_MODEL,
                    temperature=0.1,
                    num_predict=1024,
                )
                print(f"LLM: Local Ollama ({OLLAMA_MODEL})")
        return self._llm

    def _get_or_create_chain(self, session_id: str) -> ConversationalRetrievalChain:
        if session_id not in self._sessions:
            memory = ConversationBufferWindowMemory(
                k=5,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
            retriever = self._get_vector_db().as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20},
            )
            chain = ConversationalRetrievalChain.from_llm(
                llm=self._get_llm(),
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": LEGAL_PROMPT},
                verbose=False,
            )
            self._sessions[session_id] = chain
        return self._sessions[session_id]

    def initialize(self):
        print("Initializing LexiRAG engine...")
        self._get_embeddings()
        self._get_vector_db()
        self._get_llm()
        self._initialized = True
        count = self._vector_db._collection.count()
        print(f"Engine ready. ChromaDB has {count} document chunks.")

    def query(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()
        chain = self._get_or_create_chain(session_id)
        result = chain.invoke({"question": question})
        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            meta = doc.metadata
            source_label = f"{meta.get('source_file', 'Unknown')}(page {meta.get('page', '?')})"
            if source_label not in seen:
                seen.add(source_label)
                sources.append({
                    "file": meta.get("source_file", "Unknown"),
                    "page": meta.get("page", "?"),
                    "category": meta.get("category", "legal"),
                    "snippet": doc.page_content[:200] + "...",
                })
        return {
            "answer": result["answer"],
            "sources": sources,
            "session_id": session_id,
        }

    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def check_db_status(self) -> Dict[str, Any]:
        try:
            db = self._get_vector_db()
            count = db._collection.count()
            return {"status": "ok", "document_chunks": count}
        except Exception as e:
            return {"status": "error", "error": str(e)}


engine = LexiRAGEngine()