"""
Day 3 — Refusal trigger detection.

Per Day3_Conceptual_Done.pptx (Module 3 / Slides 18, 25):
Three situations demand a refusal, and they are handled at two different
stages of the pipeline:

  1. Patient-specific request (diagnosis / dosage / personalized treatment)
     -> "safety_refusal". This is checked BEFORE retrieval even runs, because
        it fires "regardless of evidence quality" (Slide 25).
  2. No relevant chunks retrieved / scores below threshold
     -> "insufficient_evidence". Checked AFTER retrieval, in generation.py.
  3. Partial context only (touches the topic but doesn't answer it)
     -> also surfaces as "insufficient_evidence" or a low-confidence answer;
        this one is a generation-time judgment call, not a hard trigger.

This module only implements trigger #1 (the hard, pre-retrieval gate) plus
a lightweight out-of-scope heuristic used for documentation/testing.
"""

import re
from typing import Optional

# Patient-specific: asking the system to diagnose *this* patient.
_DIAGNOSIS_PATTERNS = [
    r"\bdo i have\b",
    r"\bdoes this mean i have\b",
    r"\bam i (at risk|going to|likely to)\b",
    r"\bdiagnose (me|my|him|her|them|this patient)\b",
    r"\bcan you diagnose\b",
    r"\bis this cancer\b",
    r"\bwhat('?s| is) wrong with me\b",
    r"\bwhat diagnosis\b",
    r"\b(my|this) patient\b",
    r"\bdiagnos\w*\b",
]

# Patient-specific: asking for a personal dosage / prescription.
_DOSAGE_PATTERNS = [
    r"\bwhat dose\b",
    r"\bwhich dose\b",
    r"\bhow much .*(should i|do i) take\b",
    r"\bprescrib\w*\b",
    r"\bdosage\b.*\b(for me|i should|should i)\b",
]

# Patient-specific: asking the system to choose a personalized treatment.
_TREATMENT_SELECTION_PATTERNS = [
    r"\bwhich treatment should i\b",
    r"\bwhat treatment should i\b",
    r"\bshould i (get|undergo|choose|start|stop)\b",
    r"\bwhat should i do (about|for) my\b",
    r"\brecommend .* for (me|my case|my situation)\b",
    r"\bin my case\b",
    r"\bfor my (case|situation|condition)\b",
]

_ALL_PATIENT_SPECIFIC_PATTERNS = (
    _DIAGNOSIS_PATTERNS + _DOSAGE_PATTERNS + _TREATMENT_SELECTION_PATTERNS
)
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _ALL_PATIENT_SPECIFIC_PATTERNS]


def is_patient_specific_request(question: str) -> bool:
    """
    Returns True if the question asks for a personal diagnosis, dosage, or
    treatment selection rather than general guideline information.

    This check runs BEFORE retrieval (Slide 25: "fires regardless of
    evidence quality") — a patient-specific question is refused even if the
    retriever would have found perfectly relevant, on-topic evidence.
    """
    if not question or not question.strip():
        return False
    return any(pattern.search(question) for pattern in _COMPILED_PATTERNS)


SAFETY_REFUSAL_MESSAGE = (
    "I cannot provide a patient-specific diagnosis, prescription, dosage, or "
    "treatment selection. Please consult a qualified clinician. I can share "
    "general educational information from the indexed guidelines if that "
    "would help instead."
)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The retrieved guideline does not provide sufficient evidence to answer "
    "this question reliably. This source doesn't appear to cover this topic "
    "in enough depth — try rephrasing, or consult a clinician directly."
)


def build_refusal_answer(status: str, reason: Optional[str] = None) -> dict:
    """
    Builds a schema-valid refusal answer for either refusal status.
    status must be "insufficient_evidence" or "safety_refusal".
    """
    if status not in ("insufficient_evidence", "safety_refusal"):
        raise ValueError(f"build_refusal_answer only handles refusal statuses, got: {status}")

    message = SAFETY_REFUSAL_MESSAGE if status == "safety_refusal" else INSUFFICIENT_EVIDENCE_MESSAGE
    missing = [reason] if reason else (
        ["A qualified clinician must assess the individual case."]
        if status == "safety_refusal"
        else ["No retrieved chunk reached the minimum evidence quality needed to answer this question."]
    )

    return {
        "status": status,
        "recommendation": message,
        "supporting_evidence": [],
        "citations": [],
        "confidence": "Insufficient Evidence",
        "missing_information": missing,
        "safety_note": "Educational information only; not a diagnosis or medical advice.",
    }
