import logging
import json
import re
from typing import Dict, Any, List
from langchain_core.documents import Document

logger = logging.getLogger("rag_app.evaluation.hallucination")

class HallucinationDetector:
    def __init__(self, llm_service):
        self.llm_service = llm_service

    def detect(self, query: str, context_docs: List[Document], response: str, llm_provider: str = "groq") -> Dict[str, Any]:
        """
        Runs an NLI check using the LLM to verify if the response is fully grounded in the context.
        Returns a dict:
        - "is_hallucinated": bool
        - "reason": str
        - "score": float (1.0 = fully grounded, 0.0 = completely hallucinated)
        """
        if not context_docs or not response:
            return {"is_hallucinated": False, "reason": "Empty context or response", "score": 1.0}

        # Format retrieved chunks
        context_str = "\n\n".join([f"- {doc.page_content}" for doc in context_docs])
        
        prompt = (
            "You are a strict hallucination evaluator. Analyze the given response and determine if it contains statements "
            "not supported by the retrieved context. All facts in the response must be directly grounded in the context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Response:\n{response}\n\n"
            "Return JSON in the following format:\n"
            "{\n"
            '  "is_hallucinated": <true|false>,\n'
            '  "score": <0.0 to 1.0 representing proportion of grounded statements>,\n'
            '  "reason": "<explanation of unsupported claims if any>"\n'
            "}\n"
            "Important: Ensure the response is valid JSON. Escape double quotes inside string fields."
        )
        
        try:
            llm = self.llm_service.get_llm(provider=llm_provider, temperature=0.0)
            res = llm.invoke(prompt)
            content_str = res.content.strip()
            
            # Try parsing using standard json.loads first
            match = re.search(r'\{.*\}', content_str, re.DOTALL)
            if match:
                json_part = match.group(0)
                try:
                    data = json.loads(json_part)
                    return {
                        "is_hallucinated": bool(data.get("is_hallucinated", False)),
                        "reason": data.get("reason", ""),
                        "score": float(data.get("score", 1.0))
                    }
                except Exception as json_err:
                    logger.warning(f"Standard JSON parsing failed: {str(json_err)}. Attempting regex extraction.")
                    
                    # Regex fallback parser
                    is_hallucinated = False
                    score = 1.0
                    reason = ""
                    
                    match_hal = re.search(r'"is_hallucinated"\s*:\s*(true|false)', json_part, re.IGNORECASE)
                    if match_hal:
                        is_hallucinated = match_hal.group(1).lower() == "true"
                        
                    match_score = re.search(r'"score"\s*:\s*([0-9.]+)', json_part)
                    if match_score:
                        score = float(match_score.group(1))
                        
                    match_reason = re.search(r'"reason"\s*:\s*"(.*?)"', json_part, re.DOTALL)
                    if match_reason:
                        reason = match_reason.group(1).strip()
                        
                    return {
                        "is_hallucinated": is_hallucinated,
                        "reason": reason,
                        "score": score
                    }
            else:
                logger.warning(f"No JSON brackets found in response: {content_str}")
        except Exception as e:
            logger.error(f"Hallucination detection failed: {str(e)}")
            
        return {"is_hallucinated": False, "reason": "Evaluation execution failed, defaulted to safe", "score": 1.0}
