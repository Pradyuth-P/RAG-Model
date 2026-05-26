import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, AsyncGenerator

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

# Service imports
from app.loaders.document_loader import DocumentLoader
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import FAISSVectorStore
from app.services.llm_service import LLMService

logger = logging.getLogger("rag_app.rag_service")

class RAGService:
    def __init__(self, embedding_service: EmbeddingService, vector_store: FAISSVectorStore, llm_service: LLMService):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service

    @traceable(name="Document Ingestion Pipeline", run_type="chain")
    def ingest_document(self, file_path: str, session_id: str, chunk_size: int = 1000, chunk_overlap: int = 200, provider: str = "huggingface") -> Dict[str, Any]:
        """
        Runs the full ingestion pipeline: Load file -> Chunk content -> Generate IDs & Metadata -> Embed & Store.
        """
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        doc_id = str(uuid.uuid4())
        uploaded_at = datetime.utcnow().isoformat()

        try:
            logger.info(f"Starting ingestion for {filename} (id={doc_id}, session={session_id}) using provider={provider}")
            
            # 1. Load document
            pages = DocumentLoader.load_file(file_path)
            if not pages:
                raise ValueError("Loaded document is empty.")

            # 2. Chunk document
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_documents(pages)
            logger.info(f"Split {len(pages)} pages into {len(chunks)} chunks for {filename}")

            # 3. Enrich metadata and generate vector IDs
            enriched_chunks = []
            chunk_ids = []
            timestamp = datetime.utcnow().isoformat()
            
            for idx, chunk in enumerate(chunks):
                c_id = f"{doc_id}_{idx}"
                chunk_ids.append(c_id)
                
                # Copy existing metadata and add new metadata fields
                meta = chunk.metadata.copy()
                meta.update({
                    "doc_id": doc_id,
                    "chunk_id": c_id,
                    "source": filename,
                    "timestamp": timestamp,
                    "page": meta.get("page", 1)
                })
                
                enriched_chunks.append(
                    Document(page_content=chunk.page_content, metadata=meta)
                )

            # 4. Generate embeddings and store in vector store
            embeddings = self.embedding_service.get_embeddings(provider)
            
            file_info = {
                "id": doc_id,
                "filename": filename,
                "file_path": file_path,
                "uploaded_at": uploaded_at,
                "file_size": file_size
            }
            
            success = self.vector_store.store_vectors(
                documents=enriched_chunks,
                ids=chunk_ids,
                provider=provider,
                embeddings=embeddings,
                file_info=file_info,
                session_id=session_id
            )
            
            if not success:
                raise Exception("Failed to store vectors in FAISS index.")

            logger.info(f"Ingestion completed successfully for {filename}.")
            return {
                "id": doc_id,
                "filename": filename,
                "chunk_count": len(enriched_chunks),
                "uploaded_at": uploaded_at,
                "file_size": file_size,
                "embedding_provider": provider
            }

        except Exception as e:
            logger.error(f"Ingestion failed for {filename}: {str(e)}")
            # Cleanup source file if ingestion failed
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as clean_err:
                    logger.error(f"Could not remove temporary file {file_path}: {str(clean_err)}")
            raise e

    @traceable(name="Vector Retrieval", run_type="retriever")
    def retrieve_context(self, query: str, provider: str, session_id: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """
        Retrieves similar text chunks from the vector store for a given query.
        """
        logger.info(f"Retrieving context for query: '{query}' with provider {provider}, session_id={session_id}, top_k={top_k}")
        embeddings = self.embedding_service.get_embeddings(provider)
        return self.vector_store.retrieve(
            query=query,
            provider=provider,
            embeddings=embeddings,
            session_id=session_id,
            top_k=top_k,
            score_threshold=score_threshold
        )

    @traceable(name="RAG Generation Pipeline", run_type="chain")
    async def generate_response_stream(
        self, 
        query: str, 
        embedding_provider: str,
        llm_provider: str,
        session_id: str,
        llm_model: str = None,
        temperature: float = 0.3,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming generator that retrieves context, constructs prompt, and yields tokens
        along with source chunks.
        Yields JSON objects with structure:
        - {"type": "sources", "content": [...]} (sent first)
        - {"type": "token", "content": "..."} (sent in increments)
        - {"type": "done"} (sent at end)
        """
        # 1. Retrieve chunks
        retrieved_docs_with_scores = self.retrieve_context(
            query=query, 
            provider=embedding_provider, 
            session_id=session_id,
            top_k=top_k, 
            score_threshold=score_threshold
        )
        
        # Format sources to send to client
        sources = []
        context_parts = []
        
        for idx, (doc, score) in enumerate(retrieved_docs_with_scores):
            source_name = doc.metadata.get("source", "Unknown")
            page_num = doc.metadata.get("page", 1)
            
            # Normalized score (mapping L2 score to ~0-1 range where 1 is identical)
            similarity = round(max(0.0, 1.0 - (score / 2.0)), 4)
            
            sources.append({
                "source": source_name,
                "page": page_num,
                "content": doc.page_content,
                "score": similarity
            })
            
            # Format context chunk
            context_parts.append(
                f"[Source: {source_name}, Page: {page_num}]\n{doc.page_content}"
            )
        
        # Yield source chunks first
        yield {"type": "sources", "content": sources}

        # 2. Build prompt context
        context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Setup system prompt and human messages
        system_prompt = (
            "You are an AI assistant answering only from retrieved context.\n"
            "Rules:\n"
            "Do not hallucinate.\n"
            "If information is unavailable say: 'I could not find relevant information.'\n"
            "Use retrieved chunks only.\n"
            "Mention source documents."
        )
        
        user_prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question:\n{query}\n\n"
            f"Answer:"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        # 3. Call LLM Service to get ChatModel and stream responses
        try:
            llm = self.llm_service.get_llm(
                provider=llm_provider,
                model=llm_model,
                temperature=temperature
            )

            logger.info("Triggering LLM generation stream...")
            
            # Stream from LangChain model using standard astream interface
            # Note: We wrap it in a child run if desired, but LangChain's astream automatically
            # hooks into LangSmith if environment variables are set.
            async for chunk in llm.astream(messages):
                token = chunk.content
                if token:
                    yield {"type": "token", "content": token}
            
            logger.info("LLM generation stream completed.")
            yield {"type": "done"}

        except Exception as e:
            logger.error(f"Error during LLM generation: {str(e)}")
            yield {"type": "error", "content": f"LLM Error: {str(e)}"}
            yield {"type": "done"}
