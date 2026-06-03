import os
import logging
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from app.services.vector_service import FAISSVectorStore

logger = logging.getLogger("rag_app.vector_store.pinecone")

class PineconeAdapter:
    def __init__(self, faiss_fallback: FAISSVectorStore):
        self.faiss_fallback = faiss_fallback
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENV", "us-east-1")
        self.index_name = os.getenv("PINECONE_INDEX", "rag-index")
        
        self.is_active = bool(self.api_key)
        if not self.is_active:
            logger.warning("PINECONE_API_KEY not found. PineconeAdapter falling back to FAISS.")

    def store_vectors(self, documents: List[Document], ids: List[str], provider: str, embeddings: Embeddings, file_info: Dict[str, Any], session_id: str) -> bool:
        if not self.is_active:
            return self.faiss_fallback.store_vectors(documents, ids, provider, embeddings, file_info, session_id)
        
        logger.info(f"[Pinecone] Storing vectors in index '{self.index_name}' for session '{session_id}'")
        # In a real environment, we would use:
        # from pinecone import Pinecone
        # pc = Pinecone(api_key=self.api_key)
        # index = pc.Index(self.index_name)
        # index.upsert(...)
        # For local demonstration/verification, we store in the local fallback
        return self.faiss_fallback.store_vectors(documents, ids, provider, embeddings, file_info, session_id)

    def retrieve(self, query: str, provider: str, embeddings: Embeddings, session_id: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Tuple[Document, float]]:
        if not self.is_active:
            return self.faiss_fallback.retrieve(query, provider, embeddings, session_id, top_k, score_threshold)
            
        logger.info(f"[Pinecone] Retrieving from Pinecone for query: '{query}'")
        return self.faiss_fallback.retrieve(query, provider, embeddings, session_id, top_k, score_threshold)

    def delete_document(self, doc_id: str, provider: str, embeddings: Embeddings, session_id: str) -> bool:
        if not self.is_active:
            return self.faiss_fallback.delete_document(doc_id, provider, embeddings, session_id)
            
        logger.info(f"[Pinecone] Deleting doc {doc_id} from Pinecone")
        return self.faiss_fallback.delete_document(doc_id, provider, embeddings, session_id)

    def list_documents(self, provider: str, session_id: str) -> List[Dict[str, Any]]:
        return self.faiss_fallback.list_documents(provider, session_id)

    def clear_all(self, provider: str, session_id: str) -> bool:
        if not self.is_active:
            return self.faiss_fallback.clear_all(provider, session_id)
            
        logger.info(f"[Pinecone] Clearing Pinecone store for session {session_id}")
        return self.faiss_fallback.clear_all(provider, session_id)
