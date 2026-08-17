from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: List[str] = CHUNK_SEPARATORS
) -> List[Document]:
    """
    Splits input documents into chunks and assigns deterministic chunk_ids for auditability.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )

    chunks = text_splitter.split_documents(documents)

    # Assign deterministic chunk_id metadata
    for idx, chunk in enumerate(chunks, start=1):
        doc_id = chunk.metadata.get("document_id", "DOC")
        page_num = chunk.metadata.get("page_number", 0)
        chunk.metadata["chunk_id"] = f"{doc_id}-P{page_num}-CH{idx:04d}"

    print(f"[CHUNKING] Split {len(documents)} document pages into {len(chunks)} chunks.")
    return chunks
