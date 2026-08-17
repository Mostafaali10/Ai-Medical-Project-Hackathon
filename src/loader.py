import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from src.config import DATA_DIR, DOCUMENT_METADATA_MAP


def load_clinical_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """
    Loads all PDF documents from the data directory and attaches canonical clinical metadata.
    Fails loudly if any PDF fails to load, produces 0 pages, or if no PDFs are found.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in directory: {data_dir}")

    raw_documents: List[Document] = []
    failed_files = []

    for pdf_path in pdf_files:
        filename_lower = pdf_path.name.lower()
        
        # Match against metadata mapping
        matched_meta = None
        for key, meta in DOCUMENT_METADATA_MAP.items():
            if key in filename_lower:
                matched_meta = meta
                break

        if matched_meta:
            doc_id = matched_meta["document_id"]
            doc_name = matched_meta["document_name"]
        else:
            doc_id = f"DOC-{pdf_path.stem[:8].upper()}"
            doc_name = pdf_path.name

        print(f"[LOADER] Loading PDF: {pdf_path.name} (Doc ID: {doc_id})")
        
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            if not pages:
                raise ValueError(f"PDF {pdf_path.name} loaded 0 pages (empty or corrupted).")

            for page in pages:
                page.metadata.update({
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "page_number": page.metadata.get("page", 0) + 1
                })

            raw_documents.extend(pages)
            print(f"[LOADER] Successfully loaded {len(pages)} pages from {pdf_path.name}")

        except Exception as e:
            failed_files.append((pdf_path.name, str(e)))

    if failed_files:
        failure_summary = "\n".join([f"  - {fname}: {err}" for fname, err in failed_files])
        raise RuntimeError(f"Failed to load one or more PDF documents:\n{failure_summary}")

    if not raw_documents:
        raise RuntimeError("No document pages were successfully loaded.")

    print(f"[LOADER] Total documents loaded across all files: {len(raw_documents)} pages.")
    return raw_documents
