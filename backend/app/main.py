import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# App imports
from app.api.routes import router
from app.services.langsmith_config import init_langsmith

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_app.main")

# Load environment
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="Basic RAG API",
    description="Production-ready FastAPI backend for document chunking, indexing, and retrieved LLM generation.",
    version="1.0.0"
)

# CORS Setup
# React Vite runs on port 5173 by default. We allow all origins or restrict to local ports.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach API endpoints
app.include_router(router)

@app.on_event("startup")
def startup_event():
    """
    Perform key validation and configuration reporting on backend startup.
    """
    logger.info("===============================================")
    logger.info("Initializing Basic RAG System Backend Services...")
    logger.info("===============================================")

    # 1. Check LangSmith tracing
    tracing_active = init_langsmith()
    if tracing_active:
        logger.info("LangSmith Observability is ACTIVE. System runs will be traced.")
    else:
        logger.warning("LangSmith Tracing is INACTIVE. Check your environment settings.")

    # 2. Check Embedding Configuration
    embedding_provider = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "huggingface").lower()
    logger.info(f"Default Embedding Provider: {embedding_provider}")
    
    # 3. Check LLM Key configurations and warn if empty
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    
    logger.info("Active LLM Credentials Status:")
    if groq_configured:
        logger.info(" - GROQ_API_KEY: AVAILABLE")
    else:
        logger.critical(
            "CRITICAL: GROQ_API_KEY is not configured in your .env file! "
            "Please provide GROQ_API_KEY for the generation stage to work."
        )

    # 4. Check/create indexing storage
    storage_dir = "storage"
    uploads_dir = "uploads"
    os.makedirs(storage_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)
    logger.info(f"Storage directory mapping: {os.path.abspath(storage_dir)}")
    logger.info(f"Uploads directory mapping: {os.path.abspath(uploads_dir)}")
    logger.info("Backend Startup sequence complete. API ready for connections.")
    logger.info("===============================================")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Basic RAG Application API.",
        "documentation": "/docs",
        "health_check": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
