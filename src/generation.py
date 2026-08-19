"""
Day 3 — Grounded Generation & Citation.

Implements the four Module 1-4 deliverables from Day3_Conceptual_Done.pptx /
Day3.pptx on top of the already-tuned Day 2 retriever:

  Module 1  Grounded prompt design      -> DAY3_SYSTEM_PROMPT
  Module 2  Citation format + structure -> build_grounded_context, RESPONSE_SCHEMA
  Module 3  Refusal behavior            -> src/safety.py + src/relevance.py + confidence threshold below
  Module 4  Full pipeline assembly      -> generate_structured_answer / src/pipeline.py

Citation format matches the one already established in src/rag.py so both
layers agree on what a citation looks like:
    [Document Name | Page X | Chunk ID]
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jsonschema import validate, ValidationError
from langchain_core.documents import Document

from src.config import BASE_DIR
from src.safety import (
    is_patient_specific_request,
    build_refusal_answer,
)
from src.relevance import is_topically_relevant, best_lexical_match, lexical_overlap

SCHEMA_PATH = BASE_DIR / "schema" / "response_schema.json"

with open(SCHEMA_PATH, "r", encoding="utf-8") as _f:
    RESPONSE_SCHEMA: Dict[str, Any] = json.load(_f)


# ---------------------------------------------------------------------------
# Module 1 — The grounding prompt (a safety contract, not a style suggestion)
# ---------------------------------------------------------------------------

DAY3_SYSTEM_PROMPT = """You are an evidence-grounded clinical decision-support assistant.

SAFETY AND GROUNDING RULES:
1. Use ONLY the retrieved evidence supplied in the user message. Never use outside
   medical knowledge, memory, or general training data.
2. Do not invent missing facts, thresholds, diagnoses, treatments, or numbers.
3. Do not provide a patient-specific diagnosis, prescription, dosage, or treatment
   selection under any circumstances.
4. Every claim in "supporting_evidence" must include one or more citations, and every
   citation MUST be copied EXACTLY, character-for-character, from the bracketed
   citation tag shown before each evidence block below
   (format: [Document Name | Page X | Chunk ID]). Never alter, abbreviate, or
   fabricate a citation.
5. If the evidence is missing, weak, unrelated to the question's actual topic, or
   otherwise insufficient to answer the question reliably, set status to
   "insufficient_evidence", leave supporting_evidence and citations empty, and set
   confidence to "Insufficient Evidence". A chunk that is topically off — e.g. it
   comes from the indexed guideline but doesn't address what's being asked — must
   NOT be used just because it was retrieved.
6. If the request asks for a personal diagnosis, dosage, or treatment choice, set
   status to "safety_refusal" instead of answering from evidence.
7. "confidence" describes the QUALITY of the retrieved evidence, not your own
   certainty. Use "High" only when the evidence is strong, complete, and directly
   on-topic; "Medium" when relevant but partial; "Low" when weak or tangential.
8. Return VALID JSON ONLY — no Markdown code fences, no text before or after the
   JSON object.

Return exactly this structure:
{
  "status": "answered | insufficient_evidence | safety_refusal",
  "recommendation": "short, direct, evidence-grounded answer or refusal message",
  "supporting_evidence": [
    {"claim": "one supported claim, in plain language", "citations": ["exact citation tag(s)"]}
  ],
  "citations": [
    {"document": "...", "section": "...", "page": 0, "chunk_id": "..."}
  ],
  "confidence": "High | Medium | Low | Insufficient Evidence",
  "missing_information": ["what is missing, if anything"],
  "safety_note": "Educational information only; not a diagnosis or medical advice."
}"""


# ---------------------------------------------------------------------------
# Module 2 — Citation-ready context + structured output
# ---------------------------------------------------------------------------

def _citation_tag(doc: Document) -> str:
    meta = doc.metadata
    doc_name = meta.get("document_name", "Unknown Document")
    page_num = meta.get("page_number", "N/A")
    chunk_id = meta.get("chunk_id", "N/A")
    return f"[{doc_name} | Page {page_num} | {chunk_id}]"


def build_grounded_context(
    retrieved: List[Tuple[Document, float]]
) -> Tuple[str, Dict[str, Document]]:
    """
    Formats retrieved (doc, score) pairs into a citation-labeled context block,
    and returns the allow-list mapping citation tag -> source Document so the
    model's output can later be checked for invented citations.
    """
    blocks = []
    allowed: Dict[str, Document] = {}
    for doc, score in retrieved:
        tag = _citation_tag(doc)
        allowed[tag] = doc
        section = doc.metadata.get("section", "General")
        blocks.append(
            f"{tag} (section: {section}, similarity: {score:.4f})\n{doc.page_content}"
        )
    return "\n\n".join(blocks), allowed


def build_prompt(question: str, context: str) -> str:
    return (
        f"{DAY3_SYSTEM_PROMPT}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        f"Clinical question: {question}\n\n"
        f"Respond with the JSON object described above, nothing else."
    )


def _strip_markdown_fences(text: str) -> str:
    """Rule 8 exists because models default to ```json fences — strip them defensively."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        elif "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _simulated_answer(question: str, retrieved: List[Tuple[Document, float]]) -> Dict[str, Any]:
    """
    Deterministic, schema-valid placeholder used when no GROQ_API_KEY is set,
    so the citation-validation / refusal / pipeline logic downstream is still
    fully testable without a live LLM call.

    Unlike a naive "always use rank 1" placeholder, this picks the chunk(s)
    with the strongest lexical overlap against the question among the
    retrieved candidates, since embedding rank alone can put a tangentially
    related chunk above a more directly on-topic one. This is still not a
    real answer — it's a transparent stand-in — but it's a more honest
    approximation of what "the most relevant retrieved evidence" means.
    """
    scored = [
        (doc, score, lexical_overlap(question, doc.page_content))
        for doc, score in retrieved
    ]
    # Rank by lexical overlap first (does this chunk actually mention the
    # question's terms), embedding score as tiebreaker.
    scored.sort(key=lambda x: (x[2], x[1]), reverse=True)

    top_n = scored[:2]
    supporting_evidence = []
    citations = []
    for doc, score, overlap in top_n:
        tag = _citation_tag(doc)
        excerpt = doc.page_content[:280].replace("\n", " ").strip()
        supporting_evidence.append({"claim": excerpt, "citations": [tag]})
        citations.append({
            "document": doc.metadata.get("document_name", "Unknown Document"),
            "section": doc.metadata.get("section", "General"),
            "page": doc.metadata.get("page_number", "N/A"),
            "chunk_id": doc.metadata.get("chunk_id", "N/A"),
        })

    best_overlap = top_n[0][2] if top_n else 0.0
    confidence = "Medium" if best_overlap >= 0.35 else "Low"

    return {
        "status": "answered",
        "recommendation": (
            f"[SIMULATION MODE — no GROQ_API_KEY set. This is the most lexically "
            f"and semantically relevant retrieved passage, not a generated answer.] "
            f"{top_n[0][0].page_content[:280].replace(chr(10), ' ').strip()}"
        ),
        "supporting_evidence": supporting_evidence,
        "citations": citations,
        "confidence": confidence,
        "missing_information": [
            "Running in simulation mode: set GROQ_API_KEY for a real generated "
            "answer instead of the raw retrieved passage shown above."
        ],
        "safety_note": "Educational information only; not a diagnosis or medical advice.",
    }


# ---------------------------------------------------------------------------
# Citation validation (Slide 23: compare model citations against allow-list)
# ---------------------------------------------------------------------------

def validate_citations(answer: Dict[str, Any], allowed: Dict[str, Document]) -> Dict[str, Any]:
    """
    Checks every citation the model used in supporting_evidence against the
    allow-list built from retrieved chunks. Returns a validation report and
    mutates a defensive copy of the answer: if any citation is invented, the
    answer's confidence is downgraded and the problem is recorded in
    missing_information rather than silently trusted.
    """
    used_tags = set()
    for item in answer.get("supporting_evidence", []):
        used_tags.update(item.get("citations", []))

    invented = sorted(t for t in used_tags if t not in allowed)
    missing_citation_claims = [
        item["claim"] for item in answer.get("supporting_evidence", [])
        if not item.get("citations")
    ]

    report = {
        "citations_used": sorted(used_tags),
        "invented_citations": invented,
        "claims_missing_citation": missing_citation_claims,
        "valid": not invented and not missing_citation_claims,
    }

    if invented or missing_citation_claims:
        answer = dict(answer)
        answer["confidence"] = "Low"
        notes = list(answer.get("missing_information", []))
        if invented:
            notes.append(
                f"citation validation failed: {len(invented)} citation(s) were not "
                f"found in the retrieved evidence (invented citation)."
            )
        if missing_citation_claims:
            notes.append(
                f"citation validation failed: {len(missing_citation_claims)} claim(s) "
                f"had no citation attached (coverage failure)."
            )
        answer["missing_information"] = notes

    return {"answer": answer, "report": report}


# ---------------------------------------------------------------------------
# Module 4 — Full generation call (retrieval -> prompt -> LLM -> validation)
# ---------------------------------------------------------------------------

def generate_structured_answer(
    question: str,
    retrieved: List[Tuple[Document, float]],
    llm=None,
    confidence_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Runs the Day 3 grounded-generation flow for a single question against
    already-retrieved (doc, score) pairs (i.e. Day 2's output).

    Refusal order matches Day3_Conceptual_Done Slide 28 (the decision flow):
      1. Patient-specific?              -> safety_refusal   (checked first, ignores evidence)
      2. No/weak retrieved evidence?    -> insufficient_evidence (score threshold)
      2b. Topically unrelated evidence? -> insufficient_evidence (lexical relevance gate)
      3. Otherwise generate, then validate citations before returning.

    Returns a dict with keys: "answer" (schema-shaped dict), "schema_valid"
    (bool), "citation_report" (dict), and "prompt_used" (str, for debugging).
    """
    # 1. Safety refusal — fires regardless of evidence quality.
    if is_patient_specific_request(question):
        answer = build_refusal_answer("safety_refusal")
        return {
            "answer": answer,
            "schema_valid": _is_schema_valid(answer),
            "citation_report": {"citations_used": [], "invented_citations": [], "claims_missing_citation": [], "valid": True},
            "prompt_used": None,
        }

    # 2. Insufficient evidence — no chunks, or top score below threshold.
    top_score = retrieved[0][1] if retrieved else -999
    if not retrieved or top_score < confidence_threshold:
        reason = (
            "No chunks were retrieved for this question."
            if not retrieved
            else f"Top retrieval score ({top_score:.4f}) is below the confidence "
                 f"threshold ({confidence_threshold})."
        )
        answer = build_refusal_answer("insufficient_evidence", reason=reason)
        return {
            "answer": answer,
            "schema_valid": _is_schema_valid(answer),
            "citation_report": {"citations_used": [], "invented_citations": [], "claims_missing_citation": [], "valid": True},
            "prompt_used": None,
        }

    # 2b. Topical relevance gate — embedding similarity alone can't reliably
    # separate "genuinely on-topic" from "coincidentally similar phrasing"
    # for a short, unrelated query (see src/relevance.py). Require that at
    # least one retrieved chunk shares real clinical vocabulary with the
    # question before treating the evidence as usable.
    if not is_topically_relevant(question, retrieved, min_overlap=0.2):
        _, best_overlap = best_lexical_match(question, retrieved)
        reason = (
            f"Retrieved chunks passed the similarity score threshold (top score "
            f"{top_score:.4f}) but share no meaningful clinical vocabulary with the "
            f"question (best keyword overlap: {best_overlap:.0%}). This question "
            f"appears to be outside the scope of the indexed guidelines."
        )
        answer = build_refusal_answer("insufficient_evidence", reason=reason)
        return {
            "answer": answer,
            "schema_valid": _is_schema_valid(answer),
            "citation_report": {"citations_used": [], "invented_citations": [], "claims_missing_citation": [], "valid": True},
            "prompt_used": None,
        }

    # 3. Generate.
    context, allowed = build_grounded_context(retrieved)
    prompt = build_prompt(question, context)

    if llm is not None:
        raw = llm.invoke(prompt)
        raw_text = raw.content if hasattr(raw, "content") else str(raw)
        try:
            answer = json.loads(_strip_markdown_fences(raw_text))
        except json.JSONDecodeError as e:
            answer = build_refusal_answer(
                "insufficient_evidence",
                reason=f"Model output was not valid JSON and could not be parsed: {e}",
            )
    else:
        answer = _simulated_answer(question, retrieved)

    # 4. Validate citations against the retrieval allow-list.
    validated = validate_citations(answer, allowed)
    answer = validated["answer"]

    return {
        "answer": answer,
        "schema_valid": _is_schema_valid(answer),
        "citation_report": validated["report"],
        "prompt_used": prompt,
    }


def _is_schema_valid(answer: Dict[str, Any]) -> bool:
    try:
        validate(instance=answer, schema=RESPONSE_SCHEMA)
        return True
    except ValidationError:
        return False