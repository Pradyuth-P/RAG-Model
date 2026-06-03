import logging
from typing import Dict, Any

logger = logging.getLogger("rag_app.agents.evaluator")

def evaluate_node(state: Dict[str, Any], rag_service) -> Dict[str, Any]:
    """
    Evaluates the relevance of the retrieved documents for the user query.
    If irrelevant and attempts < 2, sets evaluation_result to "irrelevant" to trigger query reformulation.
    """
    retrieved = state.get("retrieved_docs", [])
    query = state["query"]
    llm_provider = state["llm_provider"]
    llm_model = state["llm_model"]
    attempts = state.get("attempts", 0)

    # Heuristic 1: If empty results
    if not retrieved:
        logger.info("Evaluator Agent: No documents retrieved.")
        return {
            "evaluation_result": "irrelevant" if attempts < 2 else "fallback",
            "attempts": attempts
        }

    # Heuristic 2: LLM relevance check
    try:
        llm = rag_service.llm_service.get_llm(provider=llm_provider, model=llm_model, temperature=0.0)
        
        top_doc, _ = retrieved[0]
        prompt = (
            "You are an evaluator assessing the relevance of retrieved context to a user query.\n"
            f"Query: {query}\n"
            f"Context: {top_doc.page_content[:400]}\n\n"
            "Does the context contain relevant facts to help answer the query? Answer strictly with either 'YES' or 'NO'."
        )
        response = llm.invoke(prompt)
        verdict = response.content.strip().upper()
        logger.info(f"Evaluator Agent: Relevance check response: '{verdict}'")
        
        if "YES" in verdict:
            return {
                "evaluation_result": "relevant",
                "attempts": attempts
            }
        else:
            logger.info("Evaluator Agent: Context deemed irrelevant by LLM evaluation.")
            return {
                "evaluation_result": "irrelevant" if attempts < 2 else "fallback",
                "attempts": attempts
            }
            
    except Exception as e:
        logger.warning(f"Evaluator Agent: Evaluation execution error: {str(e)}. Defaulting to relevant.")
        return {
            "evaluation_result": "relevant",
            "attempts": attempts
        }
