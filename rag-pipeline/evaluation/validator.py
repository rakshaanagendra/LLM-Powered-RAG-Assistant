##POST GENERATION QUALITY CONTROL
# This module provides a Validator class to evaluate the quality of generated responses
# against reference answers using various metrics.
# Also called as evaluation layer, guardrail layer or faithfullness layer in RAG systems.
# Checks if generated answer is correct/grounded in retrieved chunks, if it is factually correct, if it is relevant to the query, etc.

import re


def _extract_terms(text):
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if len(t) >= 3}


def _sentence_split(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _sentence_supported(sentence, chunks, min_overlap=2):
    sentence_terms = _extract_terms(sentence)

    for chunk in chunks:
        chunk_terms = _extract_terms(chunk.get("text", ""))
        overlap = len(sentence_terms.intersection(chunk_terms))

        if overlap >= min_overlap:
            return True

    return False


def validate_answer(answer, chunks):
    """
    Returns:
        (is_valid: bool, reason: str)
    """

    sentences = _sentence_split(answer)

    if not sentences:
        return False, "Empty answer"

    for sentence in sentences:
        if not _sentence_supported(sentence, chunks):
            return False, f"Unsupported sentence: {sentence}"

    return True, "All sentences grounded"