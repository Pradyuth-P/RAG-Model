import os
import logging
from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger("rag_app.reranker")

class RerankService:
    def __init__(self):
        self._model = None
        # Default to BAAI/bge-reranker-large, but support base/mini models via env overrides for CPU limits
        self.model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            logger.info(f"Loading reranker model: {self.model_name} (This may take a moment on first load...)")
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Reranker model successfully loaded.")
        return self._model

    def rerank(self, query: str, documents: List[Tuple[Document, float]], top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Reranks a list of retrieved documents for a query using a CrossEncoder.
        Returns a list of Tuple[Document, float] sorted by cross-encoder relevance score.
        """
        if not documents:
            return []

        try:
            model = self._get_model()
            # Prepare pairs of (query, document_text)
            pairs = [[query, doc.page_content] for doc, _ in documents]
            
            # Predict scores
            scores = model.predict(pairs)
            
            # Pair documents with new rerank scores
            reranked_pairs = []
            for idx, score in enumerate(scores):
                doc, _ = documents[idx]
                reranked_pairs.append((doc, float(score)))

            # Sort by score descending
            reranked_pairs.sort(key=lambda x: x[1], reverse=True)
            return reranked_pairs[:top_k]

        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}. Falling back to original retrieval ranking.")
            # Fall back to original retriever scores
            return documents[:top_k]
