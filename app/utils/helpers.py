"""Utility functions for LexiRAG."""
import re


def clean_legal_text(text: str) -> str:
    """Clean common noise from PDF-extracted legal text."""
    # Remove page numbers like "Page 1 of 20" or lone numbers
    text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Collapse multiple whitespace/newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars, ending at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    return (truncated[:last_space] if last_space > 0 else truncated) + "..."