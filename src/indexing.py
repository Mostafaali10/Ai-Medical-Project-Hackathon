import hashlib
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import (
    FastEmbedEmbeddings,
)
from langchain_community.vectorstores import Chroma

from src.config import (
    BASE_DIR,
    EMBEDDING_MODEL_NAME,
    COLLECTION_NAME,
    VECTORSEARCH_K,
)


# ============================================================
# Persistent Chroma directory
# ============================================================

VECTORSTORE_DIR = (
    BASE_DIR
    / "storage"
    / "chroma"
)


# ============================================================
# Embeddings
# ============================================================

def get_embedding_function(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> FastEmbedEmbeddings:

    print(
        f"[INDEXING] Loading embedding model: "
        f"{model_name}"
    )

    return FastEmbedEmbeddings(
        model_name=model_name
    )


# ============================================================
# Calculate index fingerprint
# ============================================================

def _calculate_index_fingerprint(
    chunks: List[Document],
) -> str:

    hasher = hashlib.sha256()

    hasher.update(
        EMBEDDING_MODEL_NAME.encode()
    )

    hasher.update(
        COLLECTION_NAME.encode()
    )

    for chunk in chunks:

        hasher.update(
            chunk.metadata.get(
                "chunk_id",
                "",
            ).encode()
        )

        hasher.update(
            chunk.page_content.encode(
                "utf-8",
                errors="ignore",
            )
        )

    return hasher.hexdigest()


# ============================================================
# Persistent vectorstore
# ============================================================

def create_vectorstore(
    chunks: List[Document],
    collection_name: str = COLLECTION_NAME,
    embedding_model: FastEmbedEmbeddings = None,
) -> Chroma:

    if not chunks:

        raise ValueError(
            "Cannot create a vectorstore "
            "from zero chunks."
        )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if embedding_model is None:

        embedding_model = (
            get_embedding_function()
        )

    fingerprint = (
        _calculate_index_fingerprint(
            chunks
        )
    )

    fingerprint_file = (
        VECTORSTORE_DIR
        / "index_fingerprint.txt"
    )

    chroma_db_file = (
        VECTORSTORE_DIR
        / "chroma.sqlite3"
    )

    # ========================================================
    # Try loading existing index
    # ========================================================

    if (
        chroma_db_file.exists()
        and fingerprint_file.exists()
    ):

        saved_fingerprint = (
            fingerprint_file
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        if saved_fingerprint == fingerprint:

            try:

                print(
                    f"[INDEXING] Loading existing "
                    f"Chroma vectorstore "
                    f"'{collection_name}'..."
                )

                vectorstore = Chroma(
                    collection_name=(
                        collection_name
                    ),
                    embedding_function=(
                        embedding_model
                    ),
                    persist_directory=str(
                        VECTORSTORE_DIR
                    ),
                )

                count = (
                    vectorstore
                    ._collection
                    .count()
                )

                if count > 0:

                    print(
                        f"[INDEXING] Loaded existing "
                        f"vectorstore with "
                        f"{count} chunks."
                    )

                    return vectorstore

            except Exception as exc:

                print(
                    "[INDEXING] Could not load "
                    f"existing index: {exc}"
                )

        else:

            print(
                "[INDEXING] Source/chunk "
                "configuration changed."
            )

            print(
                "[INDEXING] Rebuilding "
                "vectorstore..."
            )

    # ========================================================
    # Remove old collection if necessary
    # ========================================================

    try:

        if chroma_db_file.exists():

            old_store = Chroma(
                collection_name=(
                    collection_name
                ),
                embedding_function=(
                    embedding_model
                ),
                persist_directory=str(
                    VECTORSTORE_DIR
                ),
            )

            old_store.delete_collection()

            print(
                "[INDEXING] Removed old "
                "vectorstore collection."
            )

    except Exception:
        # If there is no existing collection,
        # this is harmless.
        pass

    # ========================================================
    # Create new persistent index
    # ========================================================

    print(
        f"[INDEXING] Creating persistent "
        f"Chroma vectorstore "
        f"'{collection_name}' "
        f"with {len(chunks)} chunks..."
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=str(
            VECTORSTORE_DIR
        ),
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

    fingerprint_file.write_text(
        fingerprint,
        encoding="utf-8",
    )

    print(
        "[INDEXING] Vectorstore indexed "
        "and saved to disk."
    )

    print(
        f"[INDEXING] Location: "
        f"{VECTORSTORE_DIR}"
    )

    return vectorstore


# ============================================================
# Retriever
# ============================================================

def get_retriever(
    vectorstore: Chroma,
    k: int = VECTORSEARCH_K,
):

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k
        },
    )


# ============================================================
# Retrieval debugging
# ============================================================

def test_retrieval(
    vectorstore: Chroma,
    query: str,
    k: int = VECTORSEARCH_K,
) -> List[Tuple[Document, float]]:

    results = (
        vectorstore
        .similarity_search_with_relevance_scores(
            query,
            k=k,
        )
    )

    print(
        f"\n[RETRIEVAL DEBUG] "
        f"Query: '{query}'"
    )

    for rank, (
        doc,
        score,
    ) in enumerate(
        results,
        start=1,
    ):

        metadata = doc.metadata

        print(
            f"  Rank {rank} | "
            f"Sim Score: {score:.4f} | "
            f"Page: "
            f"{metadata.get('page_number')} | "
            f"Chunk ID: "
            f"{metadata.get('chunk_id')}"
        )

    return results