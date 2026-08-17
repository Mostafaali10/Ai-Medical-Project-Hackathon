from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import EMBEDDING_MODEL_NAME, COLLECTION_NAME, VECTORSEARCH_K


def get_embedding_function(model_name: str = EMBEDDING_MODEL_NAME) -> FastEmbedEmbeddings:
    """Initializes and returns the FastEmbed embeddings model."""
    return FastEmbedEmbeddings(model_name=model_name)


def create_vectorstore(
    chunks: List[Document],
    collection_name: str = COLLECTION_NAME,
    embedding_model: FastEmbedEmbeddings = None
) -> Chroma:
    """
    Creates an in-memory Chroma vectorstore indexed with document chunks.
    A fresh instance is built each time this function is called.
    """
    if embedding_model is None:
        embedding_model = get_embedding_function()

    print(f"[INDEXING] Creating fresh Chroma vectorstore '{collection_name}' with {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("[INDEXING] Vectorstore indexed and ready for retrieval.")
    return vectorstore


def get_retriever(vectorstore: Chroma, k: int = VECTORSEARCH_K):
    """Returns a retriever object configured for similarity search."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def test_retrieval(vectorstore: Chroma, query: str, k: int = VECTORSEARCH_K) -> List[Tuple[Document, float]]:
    """Helper function to run similarity search with scores and log debug info."""
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    print(f"\n[RETRIEVAL DEBUG] Query: '{query}'")
    for rank, (doc, score) in enumerate(results, start=1):
        m = doc.metadata
        print(f"  Rank {rank} | Sim Score: {score:.4f} | Page: {m.get('page_number')} | Chunk ID: {m.get('chunk_id')}")
    return results
