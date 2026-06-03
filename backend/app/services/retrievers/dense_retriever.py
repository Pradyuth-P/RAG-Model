from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from app.services.vector_service import FAISSVectorStore

class DenseRetriever:
    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store

    def retrieve(
        self, 
        query: str, 
        provider: str, 
        embeddings: Embeddings, 
        session_id: str, 
        top_k: int = 5, 
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top_k similar chunks from FAISS dense vector store.
        """
        return self.vector_store.retrieve(
            query=query,
            provider=provider,
            embeddings=embeddings,
            session_id=session_id,
            top_k=top_k,
            score_threshold=score_threshold
        )
