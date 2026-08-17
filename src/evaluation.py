"""
Core Evaluation Interface Module (src/evaluation.py).

Provides backward-compatible re-exports from the canonical evaluation package:
- evaluation.evaluation_set (EVAL_SET, get_in_scope_eval_set, get_out_of_scope_eval_set)
- evaluation.metrics (evaluate_chunk_relevance, is_chunk_relevant, calculate_precision_at_k, calculate_hit_rate_at_k, calculate_mrr, evaluate_retriever)
"""

import json
from pathlib import Path
from typing import List, Dict, Any

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


def load_evaluation_dataset(dataset_path: Path = None) -> List[Dict[str, Any]]:
    """
    Loads evaluation dataset.
    If dataset_path is provided and exists, loads JSON. Otherwise returns canonical EVAL_SET.
    """
    if dataset_path and Path(dataset_path).exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return EVAL_SET


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
    "load_evaluation_dataset",
]
