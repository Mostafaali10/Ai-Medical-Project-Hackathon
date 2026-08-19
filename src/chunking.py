import hashlib
import pickle
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    BASE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
)


# ============================================================
# Chunk cache
# ============================================================

CACHE_DIR = BASE_DIR / "storage"
CHUNK_CACHE_FILE = CACHE_DIR / "chunks.pkl"
CHUNK_MANIFEST_FILE = CACHE_DIR / "chunks_manifest.txt"


# ============================================================
# Heading detection
# ============================================================

_HEADING_RE = re.compile(
    r"^\s*((?:[0-9]+(?:\.[0-9]+)*\.?\s+)?"
    r"[A-Z][A-Za-z0-9,'()/\-:& ]{2,80})\s*$"
)


def _infer_section(
    page_text: str,
    chunk_text: str,
) -> Optional[str]:

    idx = (
        page_text.find(chunk_text[:60])
        if chunk_text
        else -1
    )

    search_region = (
        page_text[:idx]
        if idx > 0
        else page_text
    )

    candidates = []

    for line in search_region.split("\n"):

        line = line.strip()

        if not line:
            continue

        if len(line) > 80:
            continue

        match = _HEADING_RE.match(line)

        if match and not line.endswith("."):
            candidates.append(line)

    return (
        candidates[-1]
        if candidates
        else None
    )


# ============================================================
# Cache fingerprint
# ============================================================

def _calculate_fingerprint(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
) -> str:

    hasher = hashlib.sha256()

    # Include chunking configuration
    hasher.update(
        str(chunk_size).encode()
    )

    hasher.update(
        str(chunk_overlap).encode()
    )

    hasher.update(
        repr(separators).encode()
    )

    # Include the actual document content
    for doc in documents:

        hasher.update(
            str(
                doc.metadata.get(
                    "document_id",
                    "",
                )
            ).encode()
        )

        hasher.update(
            str(
                doc.metadata.get(
                    "page_number",
                    "",
                )
            ).encode()
        )

        hasher.update(
            doc.page_content.encode(
                "utf-8",
                errors="ignore",
            )
        )

    return hasher.hexdigest()


# ============================================================
# Load cached chunks
# ============================================================

def _load_cached_chunks(
    fingerprint: str,
) -> Optional[List[Document]]:

    if not CHUNK_CACHE_FILE.exists():
        return None

    if not CHUNK_MANIFEST_FILE.exists():
        return None

    try:

        saved_fingerprint = (
            CHUNK_MANIFEST_FILE
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        if saved_fingerprint != fingerprint:

            print(
                "[CHUNKING] Cached chunks are "
                "outdated. Rebuilding..."
            )

            return None

        with open(
            CHUNK_CACHE_FILE,
            "rb",
        ) as f:

            chunks = pickle.load(f)

        if not isinstance(chunks, list):
            return None

        print(
            f"[CHUNKING] Loaded "
            f"{len(chunks)} chunks from cache."
        )

        return chunks

    except Exception as exc:

        print(
            f"[CHUNKING] Failed to load cache: "
            f"{exc}"
        )

        return None


# ============================================================
# Save chunks
# ============================================================

def _save_cached_chunks(
    chunks: List[Document],
    fingerprint: str,
) -> None:

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CHUNK_CACHE_FILE,
        "wb",
    ) as f:

        pickle.dump(
            chunks,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    CHUNK_MANIFEST_FILE.write_text(
        fingerprint,
        encoding="utf-8",
    )

    print(
        f"[CHUNKING] Saved "
        f"{len(chunks)} chunks to cache."
    )


# ============================================================
# Main chunking function
# ============================================================

def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: List[str] = CHUNK_SEPARATORS,
) -> List[Document]:

    if not documents:

        raise ValueError(
            "Cannot chunk an empty document list."
        )

    # --------------------------------------------------------
    # Calculate fingerprint
    # --------------------------------------------------------

    fingerprint = _calculate_fingerprint(
        documents,
        chunk_size,
        chunk_overlap,
        separators,
    )

    # --------------------------------------------------------
    # Try loading cache
    # --------------------------------------------------------

    cached_chunks = _load_cached_chunks(
        fingerprint
    )

    if cached_chunks is not None:
        return cached_chunks

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    print(
        "[CHUNKING] Creating chunks..."
    )

    # Keep complete page text for section detection
    page_text_lookup = {
        (
            doc.metadata.get("document_id"),
            doc.metadata.get("page_number"),
        ): doc.page_content
        for doc in documents
    }

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
    )

    chunks = text_splitter.split_documents(
        documents
    )

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    for idx, chunk in enumerate(
        chunks,
        start=1,
    ):

        doc_id = chunk.metadata.get(
            "document_id",
            "DOC",
        )

        page_num = chunk.metadata.get(
            "page_number",
            0,
        )

        chunk.metadata["chunk_id"] = (
            f"{doc_id}"
            f"-P{page_num}"
            f"-CH{idx:04d}"
        )

        full_page_text = (
            page_text_lookup.get(
                (
                    doc_id,
                    page_num,
                ),
                chunk.page_content,
            )
        )

        section = _infer_section(
            full_page_text,
            chunk.page_content,
        )

        chunk.metadata["section"] = (
            section or "General"
        )

    print(
        f"[CHUNKING] Split "
        f"{len(documents)} document pages "
        f"into {len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    _save_cached_chunks(
        chunks,
        fingerprint,
    )

    return chunks