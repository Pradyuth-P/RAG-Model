import os
import logging
from typing import List
from langchain_core.documents import Document

# Import loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

logger = logging.getLogger("rag_app.loaders")

class DocumentLoader:
    @staticmethod
    def load_file(file_path: str) -> List[Document]:
        """
        Loads document content and metadata using appropriate LangChain loader
        based on file extension.
        Supports PDF, DOCX, TXT, and MD.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        logger.info(f"Loading document: {file_path} with extension {ext}")

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif ext == ".docx":
                # Docx2txtLoader works well for standard Word files
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            elif ext in [".txt", ".md"]:
                # Crucial to specify UTF-8 to prevent CP1252 conversion errors on Windows
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            else:
                raise ValueError(f"Unsupported file format: {ext}. System supports PDF, DOCX, TXT, and MD.")

            # Clean content a bit and inject file metadata
            filename = os.path.basename(file_path)
            for i, doc in enumerate(docs):
                doc.metadata["source"] = filename
                # If page is not set (e.g. for docx or txt), default to page 1
                if "page" not in doc.metadata:
                    doc.metadata["page"] = i + 1

            logger.info(f"Loaded {len(docs)} document pages/sections from {filename}")
            return docs

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise e
