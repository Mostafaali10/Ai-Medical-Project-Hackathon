from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from src.config import DATA_DIR, DOCUMENT_METADATA_MAP


def _get_document_metadata(pdf_path: Path) -> dict:
    """
    Returns canonical metadata for a known PDF.

    The filename is matched against DOCUMENT_METADATA_MAP.
    Unknown PDFs are still loaded, but receive generic metadata.
    """

    filename_lower = pdf_path.stem.lower()

    for key, metadata in DOCUMENT_METADATA_MAP.items():
        if key.lower() in filename_lower:
            return metadata.copy()

    # Fallback for an unknown PDF.
    return {
        "document_id": f"DOC-{pdf_path.stem[:12].upper()}",
        "document_name": pdf_path.stem,
        "source": "Unknown",
        "document_type": "Unknown",
    }


def load_clinical_documents(
    data_dir: Path = DATA_DIR
) -> List[Document]:
    """
    Load all PDFs from the data directory.

    Each page receives:
        - document_id
        - document_name
        - source
        - document_type
        - page_number
        - source_file
    """

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}"
        )

    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in directory: {data_dir}"
        )

    raw_documents: List[Document] = []
    failed_files = []

    for pdf_path in pdf_files:

        metadata = _get_document_metadata(pdf_path)

        print(
            f"[LOADER] Loading PDF: {pdf_path.name}"
        )

        print(
            f"[LOADER] Document: {metadata['document_name']}"
        )

        print(
            f"[LOADER] Source: {metadata['source']}"
        )

        try:

            loader = PyPDFLoader(str(pdf_path))

            pages = loader.load()

            if not pages:
                raise ValueError(
                    f"PDF {pdf_path.name} loaded 0 pages."
                )

            for page in pages:

                original_page = page.metadata.get(
                    "page",
                    0
                )

                page.metadata.update(
                    {
                        "document_id": metadata["document_id"],
                        "document_name": metadata["document_name"],
                        "source": metadata["source"],
                        "document_type": metadata["document_type"],
                        "source_file": pdf_path.name,
                        "page_number": original_page + 1,
                    }
                )

            raw_documents.extend(pages)

            print(
                f"[LOADER] Successfully loaded "
                f"{len(pages)} pages."
            )

        except Exception as exc:

            failed_files.append(
                (
                    pdf_path.name,
                    str(exc)
                )
            )

    if failed_files:

        failure_summary = "\n".join(
            f"  - {filename}: {error}"
            for filename, error in failed_files
        )

        raise RuntimeError(
            "Failed to load one or more PDF documents:\n"
            + failure_summary
        )

    if not raw_documents:
        raise RuntimeError(
            "No document pages were successfully loaded."
        )

    print(
        f"[LOADER] Total pages loaded: "
        f"{len(raw_documents)}"
    )

    return raw_documents