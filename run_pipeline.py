"""
Full Clinical RAG Pipeline — root-level entry point.

Combines everything built across Day 1 (ingestion), Day 2 (retrieval
optimization: tuned Top-K/chunking, scored + metadata-tagged retrieval), and
Day 3 (grounded generation: strict system prompt, structured JSON answer,
citation validation, insufficient-evidence + safety refusals).

For any question, prints:
  - the structured answer (status, recommendation, supporting evidence,
    citations, confidence, missing information, safety note)
  - whether the answer validates against schema/response_schema.json
  - a citation-validation report (invented / missing citations)
  - every retrieved chunk with its full metadata (document, page, section,
    chunk ID, similarity score) and a text preview

Usage:
    python run_pipeline.py
    python run_pipeline.py --query "What are the recommended colorectal cancer screening strategies?"
    python run_pipeline.py --query "Do I have lung cancer?"              # safety refusal
    python run_pipeline.py --query "What is the treatment for diabetes?" # insufficient evidence
    python run_pipeline.py --query "..." --json                          # raw JSON output
"""

from src.pipeline import main

if __name__ == "__main__":
    main()
