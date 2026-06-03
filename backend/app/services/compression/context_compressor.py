import logging
from typing import List, Tuple
from langchain_core.documents import Document

logger = logging.getLogger("rag_app.compression")

class ContextCompressor:
    def __init__(self, max_tokens: int = 3000, redundancy_threshold: float = 0.7):
        self.max_tokens = max_tokens
        # Estimate ~4 characters per English token on average
        self.max_chars = max_tokens * 4
        self.redundancy_threshold = redundancy_threshold

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1.intersection(words2)) / len(words1.union(words2))

    def compress(self, documents: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Removes redundant document chunks and truncates the list to fit within token/character boundaries.
        """
        if not documents:
            return []

        compressed_docs = []
        current_chars = 0

        for doc, score in documents:
            content = doc.page_content
            
            # 1. Redundancy check
            is_redundant = False
            for accepted_doc, _ in compressed_docs:
                similarity = self._jaccard_similarity(content, accepted_doc.page_content)
                if similarity >= self.redundancy_threshold:
                    logger.info(
                        f"Omitting redundant chunk from {doc.metadata.get('source')} "
                        f"(overlap similarity: {similarity:.2f})"
                    )
                    is_redundant = True
                    break
            
            if is_redundant:
                continue

            # 2. Token/Character budget limit
            doc_len = len(content)
            if current_chars + doc_len > self.max_chars:
                logger.warning(
                    f"Context threshold reached ({current_chars} + {doc_len} > {self.max_chars} chars). "
                    "Skipping remaining chunks."
                )
                break

            compressed_docs.append((doc, score))
            current_chars += doc_len

        return compressed_docs
