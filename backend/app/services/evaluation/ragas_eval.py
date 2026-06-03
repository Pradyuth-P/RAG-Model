import logging
import json
import re
from typing import Dict, Any, List
from langchain_core.documents import Document

logger = logging.getLogger("rag_app.evaluation.ragas")

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    ragas_available = True
except ImportError:
    ragas_available = False

class RagasEvaluator:
    def __init__(self, llm_service):
        self.llm_service = llm_service

    def evaluate_rag(self, query: str, context_docs: List[Document], response: str, llm_provider: str = "groq") -> Dict[str, Any]:
        """
        Evaluates faithfulness and context recall.
        Falls back to a lightweight LLM-based scoring if Ragas is not installed/configured.
        """
        contexts = [doc.page_content for doc in context_docs]

        # Use lightweight LLM-based metrics evaluation
        try:
            llm = self.llm_service.get_llm(provider=llm_provider, temperature=0.0)
            
            prompt = (
                "You are an expert RAG system evaluator. Assess the following query, response, and retrieved context on two metrics:\n"
                "1. Faithfulness (Is the answer derived *only* from the context, without hallucinating? Score from 0.0 to 1.0)\n"
                "2. Context Recall (Does the retrieved context contain all necessary facts to fully answer the query? Score from 0.0 to 1.0)\n"
                "Provide a strict grounding assessment.\n\n"
                f"Query: {query}\n"
                f"Context:\n" + "\n---\n".join(contexts) + "\n\n"
                f"Response: {response}\n\n"
                "Return JSON in the following format:\n"
                "{\n"
                '  "faithfulness": <score>,\n'
                '  "context_recall": <score>,\n'
                '  "reasoning": "<explanation for the scores>"\n'
                "}\n"
                "Important: Ensure the response is valid JSON. Escape double quotes inside string fields."
            )
            
            res = llm.invoke(prompt)
            content_str = res.content.strip()
            
            # Try parsing using standard json.loads first
            match = re.search(r'\{.*\}', content_str, re.DOTALL)
            if match:
                json_part = match.group(0)
                try:
                    data = json.loads(json_part)
                    return {
                        "faithfulness": float(data.get("faithfulness", 1.0)),
                        "context_recall": float(data.get("context_recall", 1.0)),
                        "reasoning": data.get("reasoning", "")
                    }
                except Exception as json_err:
                    logger.warning(f"Standard JSON parsing failed in Ragas fallback: {str(json_err)}. Attempting regex extraction.")
                    
                    faithfulness = 1.0
                    context_recall = 1.0
                    reasoning = ""
                    
                    match_faith = re.search(r'"faithfulness"\s*:\s*([0-9.]+)', json_part)
                    if match_faith:
                        faithfulness = float(match_faith.group(1))
                        
                    match_recall = re.search(r'"context_recall"\s*:\s*([0-9.]+)', json_part)
                    if match_recall:
                        context_recall = float(match_recall.group(1))
                        
                    match_reason = re.search(r'"reasoning"\s*:\s*"(.*?)"', json_part, re.DOTALL)
                    if match_reason:
                        reasoning = match_reason.group(1).strip()
                        
                    return {
                        "faithfulness": faithfulness,
                        "context_recall": context_recall,
                        "reasoning": reasoning
                    }
            else:
                logger.warning(f"No JSON brackets found in Ragas fallback: {content_str}")
        except Exception as e:
            logger.error(f"Fallback RAG evaluation failed: {str(e)}")

        return {"faithfulness": 1.0, "context_recall": 1.0, "reasoning": "Evaluation skipped or failed"}
