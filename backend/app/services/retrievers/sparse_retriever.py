import re
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi
from app.services.vector_service import FAISSVectorStore

class SparseRetriever:
    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store

    def _tokenize(self, text: str) -> List[str]:
        # Basic alphanumeric tokenizer that downcases text
        text = text.lower()
        return re.findall(r'\w+', text)

    def retrieve(
        self, 
        query: str, 
        provider: str, 
        embeddings: Embeddings, 
        session_id: str, 
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top_k similar chunks using BM25 sparse keyword matching.
        Returns a list of Tuple[Document, float] where float is the BM25 relevance score.
        """
        # 1. Load active FAISS index for the session
        db = self.vector_store._load_faiss_index(provider, embeddings, session_id)
        if not db:
            return []

        # 2. Extract all documents from FAISS docstore
        all_docs = list(db.docstore._dict.values())
        if not all_docs:
            return []

        # 3. Tokenize corpus and fit BM25
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # 4. Tokenize query and compute BM25 scores
        tokenized_query = self._tokenize(query)
        doc_scores = bm25.get_scores(tokenized_query)
        
        # 5. Pair documents with scores, sort descending, and return top_k
        doc_pairs = list(zip(all_docs, doc_scores))
        doc_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_pairs[:top_k]
