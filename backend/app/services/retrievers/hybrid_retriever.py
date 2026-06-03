from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.services.retrievers.dense_retriever import DenseRetriever
from app.services.retrievers.sparse_retriever import SparseRetriever

class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

    def retrieve(
        self,
        query: str,
        provider: str,
        embeddings: Embeddings,
        session_id: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        rrf_k: int = 60
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top_k similar chunks by combining FAISS Dense and BM25 Sparse retrievals via RRF.
        Returns a list of Tuple[Document, float] where float is the merged RRF score.
        """
        # Fetch candidates from both models
        candidate_count = top_k * 2
        
        dense_results = self.dense_retriever.retrieve(
            query=query,
            provider=provider,
            embeddings=embeddings,
            session_id=session_id,
            top_k=candidate_count,
            score_threshold=score_threshold
        )
        
        sparse_results = self.sparse_retriever.retrieve(
            query=query,
            provider=provider,
            embeddings=embeddings,
            session_id=session_id,
            top_k=candidate_count
        )

        # Apply Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # 1. RRF from Dense Retriever
        for rank, (doc, _) in enumerate(dense_results, start=1):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content)
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        # 2. RRF from Sparse Retriever
        for rank, (doc, _) in enumerate(sparse_results, start=1):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content)
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        # 3. Sort by aggregated RRF score descending
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. Filter top_k
        final_results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            final_results.append((doc_map[chunk_id], score))

        return final_results
