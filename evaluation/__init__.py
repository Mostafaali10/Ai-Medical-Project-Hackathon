"""
Clinical RAG Shared Evaluation Benchmark & Metrics Package.
"""

from evaluation.evaluation_set import (
    EVAL_SET,
    get_in_scope_eval_set,
    get_out_of_scope_eval_set
)
from evaluation.metrics import (
    evaluate_chunk_relevance,
    is_chunk_relevant,
    calculate_precision_at_k,
    calculate_hit_rate_at_k,
    calculate_mrr,
    evaluate_retriever
)

__all__ = [
    "EVAL_SET",
    "get_in_scope_eval_set",
    "get_out_of_scope_eval_set",
    "evaluate_chunk_relevance",
    "is_chunk_relevant",
    "calculate_precision_at_k",
    "calculate_hit_rate_at_k",
    "calculate_mrr",
    "evaluate_retriever",
]
