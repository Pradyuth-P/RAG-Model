import logging
from typing import Dict, Any

logger = logging.getLogger("rag_app.agents.citation")

def citation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates retrieved sources and formats the list of citations for the user response.
    """
    retrieved = state.get("retrieved_docs", [])
    logger.info("Citation Agent: Formatting source list...")
    
    sources = []
    for doc, score in retrieved:
        source_name = doc.metadata.get("source", "Unknown")
        page_num = doc.metadata.get("page", 1)
        
        # Normalize score for consistent frontend UI rendering (L2 distance to cosine-like similarity map)
        similarity = round(max(0.0, 1.0 - (score / 2.0)), 4)
        
        sources.append({
            "source": source_name,
            "page": page_num,
            "content": doc.page_content,
            "score": similarity
        })
        
    return {"sources": sources}
