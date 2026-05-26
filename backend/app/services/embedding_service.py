import os
import logging
from typing import Union
from dotenv import load_dotenv

# Import LangChain embedding wrappers
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("rag_app.embeddings")

class EmbeddingService:
    def __init__(self):
        load_dotenv()
        self._embeddings_cache = {}

    def get_embeddings(self, provider: str = None) -> HuggingFaceEmbeddings:
        """
        Dynamically load and cache embedding model.
        Supports: 'huggingface' (default).
        """
        if not provider:
            provider = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "huggingface").lower()
        
        provider = provider.strip().lower()

        if provider in self._embeddings_cache:
            return self._embeddings_cache[provider]

        logger.info(f"Initializing embedding provider: {provider}")

        try:
            if provider == "huggingface":
                model_name = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                logger.info(f"Loading local HuggingFace embeddings: {model_name} (This may take a moment on first load...)")
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'}
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}. Supported: ['huggingface']")

            # Cache instance
            self._embeddings_cache[provider] = embeddings
            logger.info(f"Successfully initialized embedding provider: {provider}")
            return embeddings

        except Exception as e:
            logger.error(f"Error initializing embedding provider '{provider}': {str(e)}")
            raise e

    def validate_api_keys(self) -> dict:
        """
        Quick check on active environment keys for reporting startup warnings.
        """
        return {
            "huggingface": True # HuggingFace is local, requires no keys
        }

