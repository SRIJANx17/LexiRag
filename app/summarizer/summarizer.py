"""
Legal document summarizer using Groq API.
Falls back to Ollama if USE_GROQ is false.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

SUMMARIZE_PROMPT = """You are a legal document expert. Summarize the following legal text
in clear, simple language that any Indian citizen can understand.

Structure your summary as:
1. What this document/judgment is about (2-3 sentences)
2. Key legal points or provisions (bullet points)
3. What it means for ordinary citizens (2-3 sentences)

Legal text:
{text}

Summary:"""

SIMPLIFY_PROMPT = """You are a legal language simplifier. Take the following complex legal
text and explain it in very simple Hindi/English that a common person with no legal
background can understand. Use simple words, avoid jargon.

Legal text:
{text}

Simple explanation:"""


def _call_groq(prompt: str, max_tokens: int = 1024) -> str:
    """Call Groq API directly."""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        raise RuntimeError(f"Groq API error: {e}")


def _call_ollama(prompt: str, max_tokens: int = 1024) -> str:
    """Call local Ollama API."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama API error: {e}")


def _call_llm(prompt: str, max_tokens: int = 1024) -> str:
    """Route to Groq or Ollama based on config."""
    if USE_GROQ:
        return _call_groq(prompt, max_tokens)
    return _call_ollama(prompt, max_tokens)


def summarize_legal_text(text: str) -> str:
    """
    Summarize a legal document or judgment.
    Handles long texts by chunking.
    """
    MAX_INPUT = 3000
    if len(text) > MAX_INPUT:
        chunks = [text[i:i+MAX_INPUT] for i in range(0, min(len(text), 9000), MAX_INPUT)]
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  Summarizing chunk {i+1}/{len(chunks)}...")
            prompt = SUMMARIZE_PROMPT.format(text=chunk)
            summary = _call_llm(prompt, max_tokens=512)
            chunk_summaries.append(summary)
        combined = "\n\n".join(chunk_summaries)
        final_prompt = f"""Combine these partial summaries of a legal document into one
coherent summary:

{combined}

Final consolidated summary:"""
        return _call_llm(final_prompt, max_tokens=1024)
    else:
        prompt = SUMMARIZE_PROMPT.format(text=text)
        return _call_llm(prompt, max_tokens=1024)


def simplify_legal_text(text: str) -> str:
    """Convert complex legal language into plain, simple language."""
    text = text[:3000]
    prompt = SIMPLIFY_PROMPT.format(text=text)
    return _call_llm(prompt, max_tokens=800)