import os
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Ensure root .env is loaded
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_cors_origins() -> List[str]:
    """
    Constructs the list of allowed CORS origins from local dev defaults
    plus any production domains provided via environment variables.
    """
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    # Check for CORS_ORIGINS, FRONTEND_URL, or ALLOWED_ORIGINS env vars
    env_origins = os.getenv("CORS_ORIGINS") or os.getenv("FRONTEND_URL") or os.getenv("ALLOWED_ORIGINS")
    if env_origins:
        for o in env_origins.split(","):
            clean_origin = o.strip().rstrip("/")
            if clean_origin and clean_origin not in origins:
                origins.append(clean_origin)
    return origins


class Settings:
    PROJECT_NAME: str = "Clinical Decision Support RAG API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "FastAPI backend exposing the evidence-grounded Clinical RAG system for lung cancer screening and treatment guidelines."
    
    # Server configuration (Render supplies $PORT, default host to 0.0.0.0 for containers)
    HOST: str = os.getenv("HOST", os.getenv("API_HOST", "0.0.0.0"))
    PORT: int = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    
    # CORS Origins
    CORS_ORIGINS: List[str] = _get_cors_origins()
    
    # Pipeline defaults
    DEFAULT_K: int = 5
    DEFAULT_THRESHOLD: float = 0.30


settings = Settings()