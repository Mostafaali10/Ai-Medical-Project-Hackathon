import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.config import settings
from backend.app.routes import health, rag, documents
from src.pipeline import ClinicalRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("clinical_rag_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager:
    Initializes the singleton ClinicalRAGPipeline exactly ONCE on server startup
    and stores it in app.state.pipeline. Reuses the persistent Chroma vector store.
    """
    logger.info("Initializing ClinicalRAGPipeline on startup...")
    try:
        pipeline = ClinicalRAGPipeline(
            k=settings.DEFAULT_K,
            confidence_threshold=settings.DEFAULT_THRESHOLD,
            use_llm=True
        )
        app.state.pipeline = pipeline
        logger.info("ClinicalRAGPipeline initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize ClinicalRAGPipeline: {e}", exc_info=True)
        app.state.pipeline = None
        raise RuntimeError(f"Server startup failed due to pipeline initialization error: {e}") from e

    yield

    logger.info("Shutting down Clinical RAG backend.")


# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware for React frontend development and Vercel production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health.router)
app.include_router(rag.router)
app.include_router(documents.router)


from fastapi.encoders import jsonable_encoder

# Global Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler for request validation errors returning clear 422 JSON."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Invalid request parameters.",
            "details": jsonable_encoder(exc.errors())
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Fallback handler to prevent leaking internal stack traces or secrets."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
