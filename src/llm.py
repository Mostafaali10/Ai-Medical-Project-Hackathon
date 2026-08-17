import os
from langchain_groq import ChatGroq
from src.config import LLM_MODEL, TEMPERATURE


def get_llm_client(
    model_name: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    use_groq_sdk: bool = True
):
    """
    Initializes a Groq-backed LLM client.
    Supports either ChatGroq or ChatOpenAI configured with Groq base URL.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )

    if use_groq_sdk:
        print(f"[LLM] Initializing ChatGroq client (model: {model_name})")
        return ChatGroq(
            model_name=model_name,
            temperature=temperature,
            groq_api_key=api_key
        )
    else:
        from langchain_openai import ChatOpenAI
        print(f"[LLM] Initializing Groq-backed ChatOpenAI client (model: {model_name})")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
