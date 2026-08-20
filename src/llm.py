import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.config import (
    BASE_DIR,
    LLM_MODEL,
    TEMPERATURE,
)

logger = logging.getLogger("clinical_rag_llm")

# Load local .env if present (no-op in containers without .env)
load_dotenv(BASE_DIR / ".env")


def _clean_env_val(val: str | None) -> str | None:
    if not val:
        return None
    v = val.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1].strip()
    return v if v else None


def get_llm_client(
    model_name: str = None,
    temperature: float = TEMPERATURE,
    use_groq_sdk: bool = True,
):
    """
    Initialize and return the live Groq LLM client.
    Reads GROQ_API_KEY and GROQ_MODEL from environment variables (Railway/container or local .env).
    """
    raw_api_key = os.environ.get("GROQ_API_KEY")
    api_key = _clean_env_val(raw_api_key)

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set or is empty."
        )

    resolved_model = _clean_env_val(model_name) or _clean_env_val(os.environ.get("GROQ_MODEL")) or LLM_MODEL or "openai/gpt-oss-120b"

    print(
        f"[LLM] Initializing Groq client "
        f"(model: {resolved_model}, key_length: {len(api_key)})"
    )

    if use_groq_sdk:
        llm = ChatGroq(
            model=resolved_model,
            temperature=temperature,
            groq_api_key=api_key,
        )

        print(
            "[LLM] Groq client initialized successfully."
        )
        return llm

    # Optional OpenAI-compatible Groq endpoint
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=resolved_model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    print(
        "[LLM] Groq OpenAI-compatible client initialized successfully."
    )
    return llm