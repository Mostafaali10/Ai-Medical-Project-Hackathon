"""
Full Clinical RAG Pipeline — Day 2 (Retrieval Optimization) + Day 3 (Grounded
Generation & Citation), assembled per Day3_Conceptual_Done.pptx Module 4 /
Slide 24 (query -> retrieve -> assemble grounded prompt -> generate &
structure -> cite & return).

For any user question, `ask()` returns:
    {
        "question": ...,
        "answer": { status, recommendation, supporting_evidence, citations,
                    confidence, missing_information, safety_note },
        "schema_valid": bool,
        "citation_report": { citations_used, invented_citations,
                              claims_missing_citation, valid },
        "chunks_used": [
            {
                "rank": 1,
                "document_id": "...",
                "document_name": "...",
                "section": "...",
                "page_number": 4,
                "chunk_id": "...",
                "similarity_score": 0.83,
                "text": "<full chunk text>",
            },
            ...
        ],
    }

"chunks_used" always reflects every chunk retrieval actually returned for
this question (Day 2's evidence panel, Slide 33) — even on a refusal, so a
reviewer can see what WAS searched, not just what was used.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from langchain_core.documents import Document
from dotenv import load_dotenv

# Ensure root .env is loaded
load_dotenv(BASE_DIR / ".env")

from src.config import VECTORSEARCH_K
from src.loader import load_clinical_documents
from src.chunking import chunk_documents
from src.indexing import create_vectorstore
from src.generation import generate_structured_answer

# Default confidence threshold for the "insufficient_evidence" refusal path.
# Illustrative, per Day3 notebook Checkpoint 3 — calibrate against real
# Precision@K data (Day 2 / Day 4), don't treat this number as final.
DEFAULT_CONFIDENCE_THRESHOLD = 0.3


class ClinicalRAGPipeline:
    """Builds the index once, then answers any number of questions against it."""

    def __init__(
        self,
        k: int = VECTORSEARCH_K,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        use_llm: bool = True,
    ):
        self.k = k
        self.confidence_threshold = confidence_threshold

        raw_docs = load_clinical_documents()
        chunks = chunk_documents(raw_docs)
        self.vectorstore = create_vectorstore(chunks)

        self.llm = None
        if use_llm:
            self.llm = self._try_build_llm()

    @staticmethod
    def _try_build_llm():
        """
        Returns a live LLM client if GROQ_API_KEY is set in environment (or .env), else None.
        When None, generation.py falls back to simulation mode.
        """
        import os
        import traceback
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
        
        from src.config import get_groq_api_key
        api_key = get_groq_api_key()
        
        if not api_key:
            print("[PIPELINE] No GROQ_API_KEY set in environment — running generation in SIMULATION MODE.")
            return None
        
        try:
            from src.llm import get_llm_client
            client = get_llm_client()
            print("[PIPELINE] Live LLM client attached to ClinicalRAGPipeline successfully.")
            return client
        except Exception as e:
            print(f"[PIPELINE ERROR] Could not initialize live LLM client: {e}")
            traceback.print_exc()
            print("[PIPELINE] Falling back to SIMULATION MODE due to initialization error above.")
            return None

    def retrieve(self, question: str) -> List[Tuple[Document, float]]:
        """
        Day 2's retrieval step. Uses the vectorstore directly (not a plain
        retriever) so similarity scores travel with every chunk — required
        both for the confidence-threshold refusal check and for the
        evidence panel shown alongside the answer.
        """
        return self.vectorstore.similarity_search_with_relevance_scores(question, k=self.k)

    def ask(self, question: str) -> Dict[str, Any]:
        retrieved = self.retrieve(question)

        result = generate_structured_answer(
            question=question,
            retrieved=retrieved,
            llm=self.llm,
            confidence_threshold=self.confidence_threshold,
        )

        chunks_used = [
            {
                "rank": rank,
                "document_id": doc.metadata.get("document_id", "N/A"),
                "document_name": doc.metadata.get("document_name", "N/A"),
                "section": doc.metadata.get("section", "General"),
                "page_number": doc.metadata.get("page_number", "N/A"),
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                "similarity_score": round(float(score), 4),
                "text": doc.page_content,
            }
            for rank, (doc, score) in enumerate(retrieved, start=1)
        ]

        return {
            "question": question,
            "answer": result["answer"],
            "schema_valid": result["schema_valid"],
            "citation_report": result["citation_report"],
            "chunks_used": chunks_used,
        }


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------

def render_answer(result: Dict[str, Any]) -> str:
    a = result["answer"]
    lines = []
    lines.append("=" * 90)
    lines.append(f"QUESTION: {result['question']}")
    lines.append("=" * 90)
    lines.append(f"STATUS: {a['status']}    CONFIDENCE: {a['confidence']}    SCHEMA VALID: {result['schema_valid']}")
    lines.append("-" * 90)
    lines.append("RECOMMENDATION:")
    lines.append(f"  {a['recommendation']}")

    if a.get("supporting_evidence"):
        lines.append("\nSUPPORTING EVIDENCE:")
        for i, item in enumerate(a["supporting_evidence"], start=1):
            lines.append(f"  {i}. {item['claim']}")
            for cite in item.get("citations", []):
                lines.append(f"       cited: {cite}")

    if a.get("missing_information"):
        lines.append("\nMISSING INFORMATION:")
        for m in a["missing_information"]:
            lines.append(f"  - {m}")

    cr = result["citation_report"]
    lines.append(f"\nCITATION VALIDATION: valid={cr['valid']}  invented={cr['invented_citations']}  "
                  f"claims_missing_citation={len(cr['claims_missing_citation'])}")

    lines.append(f"\nSAFETY NOTE: {a['safety_note']}")

    lines.append("\n" + "-" * 90)
    lines.append(f"CHUNKS RETRIEVED FOR THIS QUESTION ({len(result['chunks_used'])}):")
    for c in result["chunks_used"]:
        lines.append(
            f"  Rank {c['rank']} | score {c['similarity_score']:.4f} | "
            f"{c['document_name']} | Page {c['page_number']} | "
            f"Section: {c['section']} | {c['chunk_id']}"
        )
        preview = c["text"][:160].replace("\n", " ")
        lines.append(f"      \"{preview}...\"")
    lines.append("=" * 90)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Full Clinical RAG pipeline (Day 2 retrieval + Day 3 grounded generation)")
    parser.add_argument("-q", "--query", type=str, help="Single clinical question. If omitted, starts interactive mode.")
    parser.add_argument("-k", type=int, default=VECTORSEARCH_K, help="Number of chunks to retrieve.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Confidence threshold for insufficient-evidence refusal.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of the formatted report.")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    pipeline = ClinicalRAGPipeline(k=args.k, confidence_threshold=args.threshold)

    def handle(question: str):
        result = pipeline.ask(question)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(render_answer(result))

    if args.query:
        handle(args.query)
        return

    print("=" * 90)
    print("FULL CLINICAL RAG PIPELINE — interactive mode (type 'exit' or 'quit' to stop)")
    print("=" * 90)
    while True:
        try:
            user_input = input("\nEnter your clinical question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting. Goodbye!")
                break
            handle(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    main()
