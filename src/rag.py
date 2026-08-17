from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


SYSTEM_PROMPT = """You are a clinical decision support assistant.
Your answers must strictly adhere to the provided clinical context.

Rules:
1. Use ONLY the supplied evidence. Never introduce outside medical knowledge.
2. If the context lacks sufficient information, state: "The provided guideline evidence is insufficient to answer this question."
3. Include structured citations after every claim: [Document Name | Page X | Chunk ID].
4. Never make a diagnosis, prescribe medication, or choose personalized patient treatments.
5. Always conclude your response with the disclaimer:
"DISCLAIMER: For educational and clinical decision-support use only. This system does not replace the judgment of a qualified healthcare professional."
"""

CLINICAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved document chunks into context string with metadata citations."""
    formatted = []
    for doc in docs:
        meta = doc.metadata
        doc_name = meta.get("document_name", "Unknown Document")
        page_num = meta.get("page_number", "N/A")
        chunk_id = meta.get("chunk_id", "N/A")
        citation = f"[{doc_name} | Page {page_num} | {chunk_id}]"
        formatted.append(f"SOURCE {citation}:\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(retriever, llm):
    """Builds an LCEL RAG chain from a retriever and LLM instance."""
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | CLINICAL_PROMPT
        | llm
        | StrOutputParser()
    )


def ask_clinical_rag(query: str, retriever, llm) -> Dict[str, Any]:
    """
    Executes a RAG query and returns both the generated answer and retrieved source documents.
    """
    # Retrieve relevant docs for explicit reference
    docs = retriever.invoke(query)
    chain = build_rag_chain(retriever, llm)
    answer = chain.invoke(query)

    return {
        "query": query,
        "answer": answer,
        "source_documents": docs
    }
