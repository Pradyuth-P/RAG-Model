import os
import json
import logging
import shutil
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

logger = logging.getLogger("rag_app.vector_store")

class BaseVectorStore(ABC):
    @abstractmethod
    def store_vectors(self, documents: List[Document], ids: List[str], provider: str, session_id: str) -> bool:
        pass

    @abstractmethod
    def retrieve(self, query: str, provider: str, session_id: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Tuple[Document, float]]:
        pass

    @abstractmethod
    def delete_document(self, doc_id: str, provider: str, session_id: str) -> bool:
        pass

    @abstractmethod
    def list_documents(self, provider: str, session_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear_all(self, provider: str, session_id: str) -> bool:
        pass


class FAISSVectorStore(BaseVectorStore):
    def __init__(self, storage_dir: str = "storage", uploads_dir: str = "uploads"):
        self.storage_dir = storage_dir
        self.uploads_dir = uploads_dir
        self.manifest_path = os.path.join(storage_dir, "documents.json")
        
        # Ensure directories exist
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.uploads_dir, exist_ok=True)
        
        # Load or create manifest
        self._init_manifest()

    def _init_manifest(self):
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w") as f:
                json.dump({"documents": []}, f, indent=2)

    def _load_manifest(self) -> Dict[str, Any]:
        try:
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading manifest file: {str(e)}")
            return {"documents": []}

    def _save_manifest(self, manifest: Dict[str, Any]):
        try:
            with open(self.manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving manifest file: {str(e)}")

    def _get_index_path(self, provider: str, session_id: str) -> str:
        """
        Get provider-specific path for FAISS index to prevent dimension collision.
        Organized by session_id to isolate index files.
        """
        safe_session = "".join(c for c in session_id if c.isalnum() or c in "._-")
        return os.path.join(self.storage_dir, f"faiss_{safe_session}_{provider}")

    def _load_faiss_index(self, provider: str, embeddings: Embeddings, session_id: str) -> Optional[FAISS]:
        index_path = self._get_index_path(provider, session_id)
        if not os.path.exists(os.path.join(index_path, "index.faiss")):
            return None
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Failed to load FAISS index for {provider} in session {session_id}: {str(e)}")
            return None

    def store_vectors(
        self, 
        documents: List[Document], 
        ids: List[str], 
        provider: str, 
        embeddings: Embeddings, 
        file_info: Dict[str, Any], 
        session_id: str
    ) -> bool:
        """
        Store documents in session-specific FAISS index and update global manifest.
        """
        try:
            index_path = self._get_index_path(provider, session_id)
            db = self._load_faiss_index(provider, embeddings, session_id)

            if db is None:
                # First time index initialization
                logger.info(f"Creating new FAISS index for provider {provider} (session: {session_id}) at {index_path}")
                db = FAISS.from_documents(documents, embeddings, ids=ids)
            else:
                logger.info(f"Adding {len(documents)} docs to existing FAISS index for {provider} (session: {session_id})")
                db.add_documents(documents, ids=ids)

            # Persist FAISS index
            db.save_local(index_path)

            # Update Manifest
            manifest = self._load_manifest()
            manifest["documents"].append({
                "id": file_info["id"],
                "filename": file_info["filename"],
                "file_path": file_info["file_path"],
                "uploaded_at": file_info.get("uploaded_at", datetime.utcnow().isoformat()),
                "file_size": file_info["file_size"],
                "chunk_count": len(documents),
                "chunk_ids": ids,
                "embedding_provider": provider,
                "session_id": session_id
            })
            self._save_manifest(manifest)
            return True

        except Exception as e:
            logger.error(f"Error storing vectors in FAISS: {str(e)}")
            return False

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
        Query session-specific FAISS index for similar chunks.
        """
        db = self._load_faiss_index(provider, embeddings, session_id)
        if db is None:
            logger.warning(f"No FAISS index found for provider {provider} (session: {session_id}). Returning empty results.")
            return []

        try:
            results = db.similarity_search_with_score(query, k=top_k)
            
            if score_threshold > 0.0:
                filtered_results = []
                for doc, score in results:
                    similarity = 1.0 - (score / 2.0)
                    if similarity >= score_threshold:
                        filtered_results.append((doc, score))
                return filtered_results
            
            return results

        except Exception as e:
            logger.error(f"Error querying FAISS for session {session_id}: {str(e)}")
            return []

    def delete_document(self, doc_id: str, provider: str, embeddings: Embeddings, session_id: str) -> bool:
        """
        Delete document from session FAISS index and remove file from disk.
        """
        manifest = self._load_manifest()
        doc_to_delete = None
        for doc in manifest["documents"]:
            if doc["id"] == doc_id and doc["embedding_provider"] == provider and doc.get("session_id") == session_id:
                doc_to_delete = doc
                break

        if not doc_to_delete:
            logger.error(f"Document with ID {doc_id} not found in manifest for provider {provider} (session: {session_id}).")
            return False

        try:
            # 1. Delete from FAISS index
            db = self._load_faiss_index(provider, embeddings, session_id)
            if db is not None:
                db.delete(ids=doc_to_delete["chunk_ids"])
                index_path = self._get_index_path(provider, session_id)
                db.save_local(index_path)
                logger.info(f"Deleted chunks for document {doc_id} from FAISS index.")
            
            # 2. Delete source file from uploads directory
            file_path = doc_to_delete["file_path"]
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed file from disk: {file_path}")

            # 3. Update manifest
            manifest["documents"].remove(doc_to_delete)
            self._save_manifest(manifest)
            return True

        except Exception as e:
            logger.error(f"Error deleting document {doc_id} for session {session_id}: {str(e)}")
            return False

    def list_documents(self, provider: str, session_id: str) -> List[Dict[str, Any]]:
        """
        List all documents registered under the active embedding provider and session.
        """
        manifest = self._load_manifest()
        return [
            doc for doc in manifest["documents"] 
            if doc["embedding_provider"] == provider and doc.get("session_id") == session_id
        ]

    def clear_all(self, provider: str, session_id: str) -> bool:
        """
        Completely delete the vector store and manifest for a given provider and session.
        """
        try:
            # Clear FAISS dir
            index_path = self._get_index_path(provider, session_id)
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
                logger.info(f"Deleted index directory {index_path}")

            # Remove associated documents from manifest and uploads
            manifest = self._load_manifest()
            remaining_docs = []
            for doc in manifest["documents"]:
                if doc["embedding_provider"] == provider and doc.get("session_id") == session_id:
                    file_path = doc["file_path"]
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Deleted source file: {file_path}")
                else:
                    remaining_docs.append(doc)
            
            manifest["documents"] = remaining_docs
            self._save_manifest(manifest)
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store for session {session_id}: {str(e)}")
            return False
