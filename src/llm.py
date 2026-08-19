import os

from langchain_groq import ChatGroq

from src.config import (
    LLM_MODEL,
    TEMPERATURE,
)


def get_llm_client(
    model_name: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    use_groq_sdk: bool = True,
):
    """
    Initialize and return the live Groq LLM client.
    """

    api_key = os.environ.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY environment variable "
            "is not set."
        )

    print(
        f"[LLM] Initializing Groq client "
        f"(model: {model_name})"
    )

    if use_groq_sdk:

        llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=api_key,
        )

        print(
            "[LLM] Groq client initialized "
            "successfully."
        )

        return llm

    # Optional OpenAI-compatible Groq endpoint
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    print(
        "[LLM] Groq OpenAI-compatible "
        "client initialized successfully."
    )

    return llm