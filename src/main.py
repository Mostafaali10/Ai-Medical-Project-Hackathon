import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Ensure root directory is on sys.path and utf-8 output encoding
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.loader import load_clinical_documents
from src.chunking import chunk_documents
from src.indexing import create_vectorstore, get_retriever
from src.llm import get_llm_client
from src.rag import ask_clinical_rag


def parse_args():
    parser = argparse.ArgumentParser(description="Clinical RAG System CLI")
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Single clinical query to evaluate. If omitted, starts interactive mode."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load environment variables
    load_dotenv()

    print("=" * 80)
    print("CLINICAL RAG SYSTEM INITIALIZATION")
    print("=" * 80)

    # 1. Load PDF documents
    raw_docs = load_clinical_documents()

    # 2. Chunk documents
    chunks = chunk_documents(raw_docs)

    # 3. Create Chroma vectorstore index and retriever
    vectorstore = create_vectorstore(chunks)
    retriever = get_retriever(vectorstore)

    # 4. Initialize LLM client
    llm = get_llm_client()

    if args.query:
        print("\n" + "=" * 80)
        print(f"QUERY: {args.query}")
        print("=" * 80)
        result = ask_clinical_rag(args.query, retriever, llm)
        print("\nRESPONSE:\n")
        print(result["answer"])
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("INTERACTIVE CLINICAL RAG CLI (type 'exit' or 'quit' to stop)")
        print("=" * 80)
        while True:
            try:
                user_input = input("\nEnter your clinical question: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Exiting Clinical RAG system. Goodbye!")
                    break

                result = ask_clinical_rag(user_input, retriever, llm)
                print("\nRESPONSE:\n")
                print(result["answer"])
                print("-" * 80)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting Clinical RAG system. Goodbye!")
                break


if __name__ == "__main__":
    main()
