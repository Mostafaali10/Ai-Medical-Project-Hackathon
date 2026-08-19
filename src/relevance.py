"""
Lexical relevance gate.

Embedding similarity alone is not enough to reject an off-topic question:
a short, unrelated query (e.g. "what is the weather in Cairo") can still
score ~0.5 cosine similarity against a narrow medical corpus purely from
shared general-English sentence structure — well within striking distance
of a genuinely relevant clinical question (~0.6-0.7 in this corpus).

This module adds a second, independent check: does the question and the
candidate chunk actually share real clinical vocabulary? A chunk that
clears the embedding threshold but shares zero meaningful terms with the
question is almost certainly a false positive, not weak-but-real evidence.

This is deliberately simple (stopword-filtered, stemmed token overlap)
rather than a second model call — it's a cheap guardrail, not a relevance
judge.
"""

import re
from typing import List, Tuple

from langchain_core.documents import Document

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "if", "then", "so", "as", "with",
    "about", "what", "when", "where", "which", "who", "whom", "why", "how",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "i", "you", "he", "she", "we", "they", "my", "your", "his", "her",
    "our", "their", "not", "no", "yes", "there", "here", "than", "too",
    "very", "just", "also", "please", "tell", "me", "today", "now",
}

_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")


def _stem(word: str) -> str:
    """
    Crude prefix-stemming: words of 4+ letters are truncated to their first
    4 characters. This is not a real stemmer (no linguistic rules) — it's a
    cheap way to make "colon"/"colonoscopy", "test"/"tests", and
    "recommend"/"recommended" count as the same keyword for overlap
    purposes, without pulling in an NLP dependency for a guardrail check.
    """
    return word if len(word) <= 3 else word[:4]


def _keywords(text: str) -> set:
    """Lowercased, stopword-filtered, stemmed tokens of length >= 3."""
    tokens = _TOKEN_RE.findall(text.lower())
    return {_stem(t) for t in tokens if t not in _STOPWORDS}


def lexical_overlap(question: str, chunk_text: str) -> float:
    """
    Fraction of the question's meaningful keywords that literally appear
    in the chunk text. Returns 0.0 for an empty/degenerate question.
    """
    q_words = _keywords(question)
    if not q_words:
        return 0.0
    c_words = _keywords(chunk_text)
    shared = q_words & c_words
    return len(shared) / len(q_words)


def best_lexical_match(
    question: str, retrieved: List[Tuple[Document, float]]
) -> Tuple[int, float]:
    """
    Returns (index_into_retrieved, overlap_score) for the chunk among the
    retrieved candidates with the highest lexical overlap against the
    question. If retrieved is empty, returns (-1, 0.0).
    """
    if not retrieved:
        return -1, 0.0
    scores = [lexical_overlap(question, doc.page_content) for doc, _ in retrieved]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx, scores[best_idx]


def is_topically_relevant(
    question: str,
    retrieved: List[Tuple[Document, float]],
    min_overlap: float = 0.2,
) -> bool:
    """
    True if at least one retrieved chunk shares a meaningful fraction of the
    question's keywords. min_overlap=0.2 means: at least 20% of the
    question's non-stopword keywords must literally appear in some
    retrieved chunk. A generic clinical question ("What are the ABCDE
    warning signs of melanoma?") easily clears this; "what is the weather
    in Cairo" will not, because none of its content words appear anywhere
    in a cancer-screening guideline.
    """
    _, best_score = best_lexical_match(question, retrieved)
    return best_score >= min_overlap