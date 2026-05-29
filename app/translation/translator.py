"""
Translation module.
Uses Helsinki-NLP opus-mt models — much lighter than IndicTrans2,
works on CPU, no GPU required. Supports English ↔ Hindi and other languages.

To add IndicTrans2 for higher quality Hindi/regional translation,
see the commented section at the bottom.
"""
import os
from typing import Optional
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_LANGUAGES = {
    "hindi": {"code": "hi", "model_en_to": "Helsinki-NLP/opus-mt-en-hi",  "model_to_en": "Helsinki-NLP/opus-mt-hi-en"},
    "bengali": {"code": "bn", "model_en_to": "Helsinki-NLP/opus-mt-en-NORD", "model_to_en": None},
    "marathi": {"code": "mr", "model_en_to": None, "model_to_en": None},
}


@lru_cache(maxsize=4)
def _get_pipeline(model_name: str):
    """Load a translation pipeline (cached — loads once per model)."""
    from transformers import pipeline
    print(f"Loading translation model: {model_name}")
    return pipeline(
        "translation",
        model=model_name,
        device=-1,  # CPU
    )


def translate_en_to_hi(text: str) -> str:
    """Translate English text to Hindi."""
    model = SUPPORTED_LANGUAGES["hindi"]["model_en_to"]
    pipe = _get_pipeline(model)
    # Chunk long texts (model has a 512 token limit)
    MAX_CHARS = 400
    if len(text) <= MAX_CHARS:
        result = pipe(text, max_length=600)
        return result[0]["translation_text"]
    
    # Chunk by sentences for cleaner results
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < MAX_CHARS:
            current += " " + sent
        else:
            if current:
                chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    
    translated_chunks = []
    for chunk in chunks:
        result = pipe(chunk, max_length=600)
        translated_chunks.append(result[0]["translation_text"])
    return " ".join(translated_chunks)


def translate_hi_to_en(text: str) -> str:
    """Translate Hindi text to English."""
    model = SUPPORTED_LANGUAGES["hindi"]["model_to_en"]
    pipe = _get_pipeline(model)
    result = pipe(text[:400], max_length=600)
    return result[0]["translation_text"]


def translate(text: str, source_lang: str = "en", target_lang: str = "hi") -> str:
    """
    General translate function.
    
    Args:
        text: Input text
        source_lang: 'en' or 'hi'
        target_lang: 'en' or 'hi'
    """
    if source_lang == target_lang:
        return text
    
    if source_lang == "en" and target_lang == "hi":
        return translate_en_to_hi(text)
    elif source_lang == "hi" and target_lang == "en":
        return translate_hi_to_en(text)
    else:
        raise ValueError(f"Translation pair {source_lang}→{target_lang} not yet supported. "
                        "Currently supported: en→hi, hi→en")


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: IndicTrans2 integration (higher quality but needs GPU or more RAM)
# Uncomment below and install: pip install indic-nlp-library
#
# from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
# from transformers import AutoModelForSeq2SeqLM
# import torch
#
# def translate_indictrans2(text, src_lang="eng_Latn", tgt_lang="hin_Deva"):
#     tokenizer = IndicTransTokenizer(direction="en-indic")
#     ip = IndicProcessor(inference=True)
#     model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B")
#     batch = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
#     inputs = tokenizer(batch, src=True, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         generated = model.generate(**inputs, num_beams=5, max_length=256)
#     decoded = tokenizer.batch_decode(generated, src=False)
#     return ip.postprocess_batch(decoded, lang=tgt_lang)[0]
# ─────────────────────────────────────────────────────────────────────────────