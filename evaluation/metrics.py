"""
Canonical Clinical RAG Evaluation Metrics & Relevance Engine.

Implements standard information retrieval evaluation metrics:
- Precision@3, Precision@5
- Hit Rate@3, Hit Rate@5
- Mean Reciprocal Rank (MRR)

Relevance Criterion:
A retrieved chunk is relevant if:
1. Query is in-scope (expected_doc is not None).
2. The chunk's document_id matches expected_doc.
3. The chunk's page_content contains at least one ground-truth keyword.
4. (Optional) The chunk's page_number matches expected_pages.
"""

from typing import List, Dict, Any, Union
from langchain_core.documents import Document
from evaluation.evaluation_set import EVAL_SET


def evaluate_chunk_relevance(doc: Document, query_meta: Dict[str, Any]) -> bool:
    """
    Determines whether a single retrieved chunk contains the ground-truth evidence.
    
    A chunk is relevant if:
    1. Query is in-scope (expected_doc is not None).
    2. Chunk document_id matches expected_doc (or expected_document_id).
    3. Chunk content contains at least one required keyword/phrase.
    4. Chunk page_number is in expected_pages (if specified).
    """
    if query_meta.get("is_out_of_scope", False):
        return False

    expected_doc = query_meta.get("expected_doc") or query_meta.get("expected_document_id")
    if expected_doc is None:
        return False

    chunk_meta = doc.metadata or {}
    doc_id = chunk_meta.get("document_id")
    page_num = chunk_meta.get("page_number")
    content = doc.page_content.lower()

    # Document ID check
    if doc_id != expected_doc:
        return False

    # Page number check (if specified in query metadata)
    expected_pages = query_meta.get("expected_pages", [])
    if expected_pages and page_num and page_num not in expected_pages:
        return False

    # Keywords / evidence check
    keywords = query_meta.get("keywords") or query_meta.get("relevant_keywords", [])
    if not keywords:
        return True

    return any(kw.lower() in content for kw in keywords)


# Alias for backward compatibility
is_chunk_relevant = evaluate_chunk_relevance


def calculate_precision_at_k(
    retrieved_chunks: List[Document],
    query_meta: Dict[str, Any],
    k: int = 3
) -> float:
    """Calculates Precision@K = (Number of relevant chunks in top K) / K."""
    if k <= 0:
        return 0.0
    top_k = retrieved_chunks[:k]
    relevant_count = sum(1 for doc in top_k if evaluate_chunk_relevance(doc, query_meta))
    return round(relevant_count / float(k), 4)


def calculate_hit_rate_at_k(
    retrieved_chunks: List[Document],
    query_meta: Dict[str, Any],
    k: int = 3
) -> float:
    """Calculates Hit Rate@K = 1.0 if at least one relevant chunk in top K else 0.0."""
    top_k = retrieved_chunks[:k]
    return 1.0 if any(evaluate_chunk_relevance(doc, query_meta) for doc in top_k) else 0.0


def calculate_mrr(
    retrieved_chunks: List[Document],
    query_meta: Dict[str, Any]
) -> float:
    """Calculates Reciprocal Rank = 1 / (rank of first relevant chunk) or 0.0."""
    for rank, doc in enumerate(retrieved_chunks, start=1):
        if evaluate_chunk_relevance(doc, query_meta):
            return round(1.0 / float(rank), 4)
    return 0.0


def evaluate_retriever(
    vectorstore: Any,
    eval_dataset: List[Dict[str, Any]] = None,
    k_max: int = 5
) -> Dict[str, Any]:
    """
    Evaluates a vectorstore/retriever against the evaluation dataset.
    Computes Precision@3, Precision@5, Hit Rate@3, Hit Rate@5, and MRR.
    """
    if eval_dataset is None:
        eval_dataset = EVAL_SET

    results = []
    in_scope_results = []

    for item in eval_dataset:
        query = item["question"]
        is_out = item.get("is_out_of_scope", False)

        # Retrieve top k_max chunks with similarity scores
        retrieved_items = vectorstore.similarity_search_with_relevance_scores(query, k=k_max)

        docs = [doc for doc, _ in retrieved_items]
        relevance_flags = []
        retrieved_details = []

        for rank, (doc, score) in enumerate(retrieved_items, start=1):
            rel = evaluate_chunk_relevance(doc, item)
            relevance_flags.append(rel)
            retrieved_details.append({
                "rank": rank,
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                "document_id": doc.metadata.get("document_id", "N/A"),
                "page_number": doc.metadata.get("page_number", 0),
                "similarity_score": round(float(score), 4),
                "is_relevant": rel,
                "content_preview": doc.page_content[:120].replace("\n", " ") + "..."
            })

        p_at_3 = calculate_precision_at_k(docs, item, k=3)
        p_at_5 = calculate_precision_at_k(docs, item, k=5)
        hit_at_3 = calculate_hit_rate_at_k(docs, item, k=3)
        hit_at_5 = calculate_hit_rate_at_k(docs, item, k=5)
        mrr = calculate_mrr(docs, item)

        q_res = {
            "id": item["id"],
            "question": query,
            "category": item.get("category", "general"),
            "is_out_of_scope": is_out,
            "precision_at_3": p_at_3,
            "precision_at_5": p_at_5,
            "hit_rate_at_3": hit_at_3,
            "hit_rate_at_5": hit_at_5,
            "mrr": mrr,
            "retrieved_chunks": retrieved_details
        }
        results.append(q_res)
        if not is_out:
            in_scope_results.append(q_res)

    def compute_aggregates(res_list: List[Dict[str, Any]]) -> Dict[str, float]:
        if not res_list:
            return {
                "precision_at_3": 0.0,
                "precision_at_5": 0.0,
                "hit_rate_at_3": 0.0,
                "hit_rate_at_5": 0.0,
                "mrr": 0.0
            }
        n = len(res_list)
        return {
            "precision_at_3": round(sum(r["precision_at_3"] for r in res_list) / n, 4),
            "precision_at_5": round(sum(r["precision_at_5"] for r in res_list) / n, 4),
            "hit_rate_at_3": round(sum(r["hit_rate_at_3"] for r in res_list) / n, 4),
            "hit_rate_at_5": round(sum(r["hit_rate_at_5"] for r in res_list) / n, 4),
            "mrr": round(sum(r["mrr"] for r in res_list) / n, 4)
        }

    return {
        "overall_all_questions": compute_aggregates(results),
        "overall_in_scope_guideline_queries": compute_aggregates(in_scope_results),
        "total_questions": len(results),
        "in_scope_questions": len(in_scope_results),
        "out_of_scope_questions": len(results) - len(in_scope_results),
        "per_question_results": results
    }
