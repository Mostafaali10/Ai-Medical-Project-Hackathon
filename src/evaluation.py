import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


def load_evaluation_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Loads the labeled evaluation benchmark dataset from JSON."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_chunk_relevant(doc: Document, query_meta: Dict[str, Any]) -> bool:
    """
    Determines if a retrieved chunk contains the ground-truth evidence.
    
    A chunk is relevant if:
    1. The query is in-scope.
    2. The chunk's document_id matches the expected document_id.
    3. The chunk's page_number is in the expected pages.
    4. At least one of the essential clinical keywords/phrases is present in the chunk content.
    """
    if query_meta.get("is_out_of_scope", False):
        # Negative control queries have no valid guideline evidence
        return False

    chunk_meta = doc.metadata
    doc_id = chunk_meta.get("document_id")
    page_num = chunk_meta.get("page_number")
    content = doc.page_content.lower()

    # Document ID check
    expected_doc_id = query_meta.get("expected_document_id")
    if expected_doc_id and doc_id != expected_doc_id:
        return False

    # Page number check
    expected_pages = query_meta.get("expected_pages", [])
    if expected_pages and page_num not in expected_pages:
        return False

    # Semantic keyword / evidence check
    keywords = query_meta.get("relevant_keywords", [])
    if not keywords:
        return True

    # Check if any keyword/phrase is present in chunk text
    return any(kw.lower() in content for kw in keywords)


def evaluate_retriever(
    vectorstore: Chroma,
    eval_dataset: List[Dict[str, Any]],
    k_max: int = 5
) -> Dict[str, Any]:
    """
    Evaluates a vectorstore against the benchmark dataset for Precision@3, Precision@5,
    Hit Rate@3, Hit Rate@5, and MRR.
    """
    results = []
    in_scope_results = []

    for item in eval_dataset:
        query = item["question"]
        is_out = item.get("is_out_of_scope", False)

        # Retrieve top k_max chunks with similarity scores
        retrieved_items = vectorstore.similarity_search_with_relevance_scores(query, k=k_max)

        relevance_flags = []
        retrieved_details = []

        for rank, (doc, score) in enumerate(retrieved_items, start=1):
            rel = is_chunk_relevant(doc, item)
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

        # Precision@3 and Precision@5
        rel_at_3 = sum(relevance_flags[:3])
        rel_at_5 = sum(relevance_flags[:5])
        p_at_3 = rel_at_3 / 3.0
        p_at_5 = rel_at_5 / 5.0

        # Hit Rate@3 and Hit Rate@5
        hit_at_3 = 1.0 if rel_at_3 > 0 else 0.0
        hit_at_5 = 1.0 if rel_at_5 > 0 else 0.0

        # MRR (Reciprocal Rank of first relevant chunk)
        mrr = 0.0
        for rank, rel in enumerate(relevance_flags, start=1):
            if rel:
                mrr = 1.0 / rank
                break

        q_res = {
            "id": item["id"],
            "question": query,
            "category": item.get("category", "general"),
            "is_out_of_scope": is_out,
            "precision_at_3": round(p_at_3, 4),
            "precision_at_5": round(p_at_5, 4),
            "hit_rate_at_3": hit_at_3,
            "hit_rate_at_5": hit_at_5,
            "mrr": round(mrr, 4),
            "retrieved_chunks": retrieved_details
        }
        results.append(q_res)
        if not is_out:
            in_scope_results.append(q_res)

    def compute_aggregates(res_list: List[Dict[str, Any]]) -> Dict[str, float]:
        if not res_list:
            return {"precision_at_3": 0.0, "precision_at_5": 0.0, "hit_rate_at_3": 0.0, "hit_rate_at_5": 0.0, "mrr": 0.0}
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
