import logging
from typing import Dict, Any
logger = logging.getLogger("rag_app.agents.retrieval")

def retrieve_node(state: Dict[str, Any], rag_service) -> Dict[str, Any]:
    """
    LangGraph node that calls the RAG service retrieve_context pipeline.
    """
    query = state["query"]
    provider = state["embedding_provider"]
    session_id = state["session_id"]
    top_k = state["top_k"]
    score_threshold = state["score_threshold"]
    llm_provider = state["llm_provider"]
    llm_model = state["llm_model"]

    logger.info(f"Retrieval Agent: Fetching context for query: '{query}'")
    
    # Calls Phase 5 Parent-Child, Phase 4 Compression, Phase 3 Multi-Query, Phase 2 Rerank, Phase 1 Hybrid under the hood
    retrieved = rag_service.retrieve_context(
        query=query,
        provider=provider,
        session_id=session_id,
        top_k=top_k,
        score_threshold=score_threshold,
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    
    logger.info(f"Retrieval Agent: Fetched {len(retrieved)} chunks.")
    
    return {
        "retrieved_docs": retrieved,
        "attempts": state.get("attempts", 0) + 1
    }
