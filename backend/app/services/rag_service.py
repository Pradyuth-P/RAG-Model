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
        
        # Initialize retrievers for Hybrid Retrieval (Phase 1)
        from app.services.retrievers.dense_retriever import DenseRetriever
        from app.services.retrievers.sparse_retriever import SparseRetriever
        from app.services.retrievers.hybrid_retriever import HybridRetriever
        
        self.dense_retriever = DenseRetriever(vector_store)
        self.sparse_retriever = SparseRetriever(vector_store)
        self.hybrid_retriever = HybridRetriever(self.dense_retriever, self.sparse_retriever)

        # Initialize Reranker Service (Phase 2)
        from app.services.reranker.rerank_service import RerankService
        self.rerank_service = RerankService()

        # Initialize Context Compressor (Phase 4)
        from app.services.compression.context_compressor import ContextCompressor
        self.context_compressor = ContextCompressor()

        # Initialize Memory Services (Phase 6)
        from app.services.memory.session_memory import RedisSessionMemory
        from app.services.memory.summary_memory import SummaryMemory
        self.session_memory = RedisSessionMemory()
        self.summary_memory = SummaryMemory(self.session_memory, self.llm_service)

        # Initialize Evaluation Services (Phase 8)
        from app.services.evaluation.ragas_eval import RagasEvaluator
        from app.services.evaluation.hallucination_detector import HallucinationDetector
        self.ragas_evaluator = RagasEvaluator(self.llm_service)
        self.hallucination_detector = HallucinationDetector(self.llm_service)

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

            # 2. Chunk document (Phase 5 Parent-Child Retrieval)
            # We treat the user-specified chunk_size and chunk_overlap as Parent bounds.
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            parents = parent_splitter.split_documents(pages)
            logger.info(f"Split {len(pages)} pages into {len(parents)} parent chunks for {filename}")

            # Split parent chunks into smaller child chunks (e.g. 1/4 of parent size, minimum 200 chars)
            child_size = max(200, chunk_size // 4)
            child_overlap = max(30, chunk_overlap // 4)
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=child_size,
                chunk_overlap=child_overlap,
                length_function=len
            )

            enriched_chunks = []
            chunk_ids = []
            timestamp = datetime.utcnow().isoformat()
            
            child_idx = 0
            for parent_idx, parent_doc in enumerate(parents):
                parent_id = f"{doc_id}_p_{parent_idx}"
                parent_content = parent_doc.page_content
                
                # Split this parent doc into smaller children
                children = child_splitter.split_documents([parent_doc])
                
                for child in children:
                    c_id = f"{doc_id}_{child_idx}"
                    chunk_ids.append(c_id)
                    
                    meta = child.metadata.copy()
                    meta.update({
                        "doc_id": doc_id,
                        "chunk_id": c_id,
                        "source": filename,
                        "timestamp": timestamp,
                        "page": meta.get("page", 1),
                        "parent_id": parent_id,
                        "parent_content": parent_content  # Reference to parent text
                    })
                    
                    enriched_chunks.append(
                        Document(page_content=child.page_content, metadata=meta)
                    )
                    child_idx += 1

            logger.info(f"Generated {len(enriched_chunks)} child chunks from {len(parents)} parents for {filename}")

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
    def retrieve_context(
        self, 
        query: str, 
        provider: str, 
        session_id: str, 
        top_k: int = 5, 
        score_threshold: float = 0.0,
        llm_provider: str = "groq",
        llm_model: str = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves similar text chunks from the vector store for a given query, performs
        multi-query retrieval using LLM variations, and reranks them.
        """
        import re
        logger.info(f"Retrieving context for query: '{query}' with provider {provider}, session_id={session_id}, top_k={top_k}")
        embeddings = self.embedding_service.get_embeddings(provider)
        
        # 1. Generate alternative queries using LLM (Phase 3 Multi-Query)
        queries = [query]
        try:
            llm = self.llm_service.get_llm(provider=llm_provider, model=llm_model, temperature=0.1)
            prompt = (
                "You are an AI assistant helping to formulate alternative search queries for retrieval.\n"
                "Generate 3 alternative search queries related to the following query.\n"
                "Return only the queries, one per line, with no extra text or numbering.\n"
                f"Query: {query}"
            )
            response = llm.invoke(prompt)
            # Parse responses
            lines = [line.strip() for line in response.content.split("\n") if line.strip()]
            for line in lines:
                # Remove leading numbers, dashes, bullets if any are hallucinated
                cleaned = re.sub(r'^\d+\.\s*|^-\s*|^\*\s*', '', line).strip()
                if cleaned and cleaned not in queries:
                    queries.append(cleaned)
            logger.info(f"Multi-query generated alternative queries: {queries}")
        except Exception as e:
            logger.warning(f"Failed to generate alternative queries: {str(e)}. Falling back to single query retrieval.")

        # 2. Retrieve candidates for each query using the hybrid retriever and merge via RRF
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        rrf_k = 60
        candidate_count = top_k * 3

        for q in queries:
            candidates = self.hybrid_retriever.retrieve(
                query=q,
                provider=provider,
                embeddings=embeddings,
                session_id=session_id,
                top_k=candidate_count,
                score_threshold=score_threshold
            )
            for rank, (doc, _) in enumerate(candidates, start=1):
                chunk_id = doc.metadata.get("chunk_id", doc.page_content)
                doc_map[chunk_id] = doc
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        # Sort all chunks by RRF score descending
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        merged_candidates = []
        seen_parent_ids = set()
        
        for chunk_id, score in sorted_chunks[:candidate_count]:
            child_doc = doc_map[chunk_id]
            parent_id = child_doc.metadata.get("parent_id")
            
            # Phase 5: If this chunk has a parent reference, retrieve the parent content instead
            if "parent_content" in child_doc.metadata:
                if parent_id:
                    if parent_id in seen_parent_ids:
                        continue  # Skip redundant parent hits
                    seen_parent_ids.add(parent_id)
                
                parent_doc = Document(
                    page_content=child_doc.metadata["parent_content"],
                    metadata=child_doc.metadata.copy()
                )
                merged_candidates.append((parent_doc, score))
            else:
                merged_candidates.append((child_doc, score))
        
        # 3. Rerank retrieved candidate chunks against original query (retrieve top_k * 2 to give compressor candidates)
        reranked = self.rerank_service.rerank(
            query=query,
            documents=merged_candidates,
            top_k=top_k * 2
        )
        
        # 4. Compress the context to remove redundancies and fit token boundaries
        compressed = self.context_compressor.compress(reranked)
        
        return compressed[:top_k]

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
        # 1. Run retrieve, evaluate, and citation agent nodes (Phase 7 Agentic Layer)
        state = {
            "query": query,
            "embedding_provider": embedding_provider,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "session_id": session_id,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "temperature": temperature,
            "retrieved_docs": [],
            "generation": "",
            "sources": [],
            "evaluation_result": "relevant",
            "attempts": 0
        }

        try:
            logger.info("Agent Coordinator: Executing state machine...")
            attempts = 0
            while attempts < 2:
                # Run retrieve node
                from app.services.agents.retrieval_agent import retrieve_node
                ret_result = retrieve_node(state, self)
                state["retrieved_docs"] = ret_result["retrieved_docs"]
                state["attempts"] = ret_result["attempts"]
                
                # Run evaluate node
                from app.services.agents.evaluator_agent import evaluate_node
                eval_result = evaluate_node(state, self)
                state["evaluation_result"] = eval_result["evaluation_result"]
                
                if state["evaluation_result"] in ["relevant", "fallback"]:
                    break
                    
                # Reformulate query (evaluate node loops back to retrieve)
                attempts += 1
                try:
                    llm = self.llm_service.get_llm(provider=llm_provider, model=llm_model, temperature=0.1)
                    prompt = (
                        "You are an AI assistant helping to reformulate a search query.\n"
                        "The previous query failed to find relevant document chunks in the database.\n"
                        f"Previous Query: {state['query']}\n\n"
                        "Formulate a search query that might find relevant context. Return only the query, no extra text."
                    )
                    res = llm.invoke(prompt)
                    state["query"] = res.content.strip()
                    logger.info(f"Agent Coordinator: Reformulated query to: '{state['query']}'")
                except Exception as reform_err:
                    logger.warning(f"Failed to reformulate query: {str(reform_err)}")
                    break
        except Exception as graph_err:
            logger.error(f"Coordinator execution failed: {str(graph_err)}. Falling back to standard flow.")
            state["retrieved_docs"] = self.retrieve_context(
                query=query, provider=embedding_provider, session_id=session_id,
                top_k=top_k, score_threshold=score_threshold,
                llm_provider=llm_provider, llm_model=llm_model
            )

        # Run Citation Agent Node to format sources list
        from app.services.agents.citation_agent import citation_node
        citation_result = citation_node(state)
        sources = citation_result["sources"]
        
        # Yield source chunks first
        yield {"type": "sources", "content": sources}

        # Build prompt context parts from final retrieved chunks
        context_parts = []
        for doc, score in state["retrieved_docs"]:
            source_name = doc.metadata.get("source", "Unknown")
            page_num = doc.metadata.get("page", 1)
            context_parts.append(f"[Source: {source_name}, Page: {page_num}]\n{doc.page_content}")

        # 2. Build prompt context
        context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Fetch conversation summary if available (Phase 6 Memory)
        summary = self.summary_memory.get_summary(session_id)

        # Setup system prompt and human messages
        system_prompt = (
            "You are an AI assistant answering only from retrieved context.\n"
            "Rules:\n"
            "Do not hallucinate.\n"
            "If information is unavailable say: 'I could not find relevant information.'\n"
            "Use retrieved chunks only.\n"
            "Mention source documents."
        )
        if summary:
            system_prompt += f"\n\nHere is a summary of the previous conversation so far:\n{summary}"
        
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
            tokens_list = []
            async for chunk in llm.astream(messages):
                token = chunk.content
                if token:
                    tokens_list.append(token)
                    yield {"type": "token", "content": token}
            
            logger.info("LLM generation stream completed.")
            
            # Run grounding and RAG metrics evaluation (Phase 8 Evaluation)
            full_response = "".join(tokens_list)
            try:
                context_docs = [doc for doc, _ in state["retrieved_docs"]]
                
                hallucination_res = self.hallucination_detector.detect(
                    query=query, context_docs=context_docs, response=full_response, llm_provider=llm_provider
                )
                
                ragas_res = self.ragas_evaluator.evaluate_rag(
                    query=query, context_docs=context_docs, response=full_response, llm_provider=llm_provider
                )
                
                logger.info(
                    f"RAG Evaluation Results: "
                    f"Hallucinated={hallucination_res['is_hallucinated']} (Score: {hallucination_res['score']}) | "
                    f"Faithfulness={ragas_res['faithfulness']} | "
                    f"Context Recall={ragas_res['context_recall']}"
                )
                # Keep these metrics in state for monitoring integration
                state["evaluation_metrics"] = {
                    "hallucination": hallucination_res,
                    "ragas": ragas_res
                }
                
                # LangSmith monitoring extension (Phase 11)
                try:
                    from langsmith.run_helpers import get_current_run_tree
                    run_tree = get_current_run_tree()
                    if run_tree:
                        run_tree.add_tags(["advanced-rag", f"session-{session_id}"])
                        run_tree.metadata.update({
                            "session_id": session_id,
                            "reranker_model": os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                            "hallucination_detected": hallucination_res.get("is_hallucinated"),
                            "hallucination_score": hallucination_res.get("score"),
                            "faithfulness_score": ragas_res.get("faithfulness"),
                            "context_recall_score": ragas_res.get("context_recall"),
                        })
                        logger.info("LangSmith monitoring metrics injected successfully.")
                except Exception as monitor_err:
                    logger.warning(f"Could not inject LangSmith monitoring metadata: {str(monitor_err)}")
            except Exception as eval_err:
                logger.error(f"Failed to execute inline RAG evaluation: {str(eval_err)}")

            yield {"type": "done"}

        except Exception as e:
            logger.error(f"Error during LLM generation: {str(e)}")
            yield {"type": "error", "content": f"LLM Error: {str(e)}"}
            yield {"type": "done"}
